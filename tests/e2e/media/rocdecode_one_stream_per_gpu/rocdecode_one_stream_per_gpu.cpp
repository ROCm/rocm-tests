#include "flow_log.hpp"
#include "result_table.hpp"

#include "rocdecode/rocdecode.h"
#include "rocdecode/rocdecode_version.h"
#include "rocdecode/rocparser.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <signal.h>
#include <sstream>
#include <string>
#include <thread>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr int kStreamRepeats = 1;

enum class Layout { None, Spx, Dpx, Cpx, Other };

const char* layout_name(Layout layout) {
    switch (layout) {
        case Layout::Spx:
            return "spx";
        case Layout::Dpx:
            return "dpx";
        case Layout::Cpx:
            return "cpx";
        case Layout::Other:
            return "other";
        case Layout::None:
            return "none";
    }
    return "none";
}

#if defined(TEST_MODE_SPX)
constexpr Layout kRequiredLayout = Layout::Spx;
constexpr const char* kProgramName = "rocdecode_spx_one_stream_per_card";
constexpr const char* kDecodeName = "spx_one_stream_per_card";
#elif defined(TEST_MODE_DPX)
constexpr Layout kRequiredLayout = Layout::Dpx;
constexpr const char* kProgramName = "rocdecode_dpx_one_stream_per_half";
constexpr const char* kDecodeName = "dpx_one_stream_per_half";
#elif defined(TEST_MODE_CPX)
constexpr Layout kRequiredLayout = Layout::Cpx;
constexpr const char* kProgramName = "rocdecode_cpx_one_stream_per_slice";
constexpr const char* kDecodeName = "cpx_one_stream_per_slice";
#else
#error "Build rocdecode_spx_one_stream_per_card, rocdecode_dpx_one_stream_per_half, or rocdecode_cpx_one_stream_per_slice"
#endif

unsigned slices_for(Layout layout) {
    if (layout == Layout::Spx) {
        return 1;
    }
    if (layout == Layout::Dpx) {
        return 2;
    }
    if (layout == Layout::Cpx) {
        return 8;
    }
    return 0;
}

std::string expected_topology(Layout layout) {
    const unsigned slices = slices_for(layout);
    std::ostringstream out;
    out << slices << (slices == 1 ? " slice on every card, partition 0" : " slices on every card, partition 0");
    if (slices > 1) {
        out << "-" << (slices - 1);
    }
    return out.str();
}

std::string roc_status_name(rocDecStatus status) {
    switch (status) {
        case ROCDEC_SUCCESS:
            return "ROCDEC_SUCCESS";
        case ROCDEC_DEVICE_INVALID:
            return "ROCDEC_DEVICE_INVALID";
        case ROCDEC_CONTEXT_INVALID:
            return "ROCDEC_CONTEXT_INVALID";
        case ROCDEC_RUNTIME_ERROR:
            return "ROCDEC_RUNTIME_ERROR";
        case ROCDEC_OUTOF_MEMORY:
            return "ROCDEC_OUTOF_MEMORY";
        case ROCDEC_INVALID_PARAMETER:
            return "ROCDEC_INVALID_PARAMETER";
        case ROCDEC_NOT_IMPLEMENTED:
            return "ROCDEC_NOT_IMPLEMENTED";
        case ROCDEC_NOT_INITIALIZED:
            return "ROCDEC_NOT_INITIALIZED";
        case ROCDEC_NOT_SUPPORTED:
            return "ROCDEC_NOT_SUPPORTED";
        default:
            return "rocDecStatus " + std::to_string(static_cast<int>(status));
    }
}

std::string hip_status_name(hipError_t status) {
    const char* text = hipGetErrorString(status);
    return text != nullptr ? text : "hipError";
}

struct PciAddress {
    unsigned domain = 0;
    unsigned bus = 0;
    unsigned device = 0;
    unsigned function = 0;
    bool valid = false;
};

PciAddress parse_pci(const std::string& text) {
    PciAddress address;
    unsigned domain = 0;
    unsigned bus = 0;
    unsigned device = 0;
    unsigned function = 0;
    if (std::sscanf(text.c_str(), "%x:%x:%x.%x", &domain, &bus, &device, &function) == 4) {
        address.domain = domain;
        address.bus = bus;
        address.device = device;
        address.function = function;
        address.valid = true;
    }
    return address;
}

std::string card_key(const PciAddress& address) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(4) << address.domain << ":" << std::setw(2) << address.bus << ":"
        << std::setw(2) << address.device;
    return out.str();
}

std::string bdf_text(const PciAddress& address) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(4) << address.domain << ":" << std::setw(2) << address.bus << ":"
        << std::setw(2) << address.device << "." << address.function;
    return out.str();
}

struct NodeActivity {
    uint64_t ns = 0;
    int fds = 0;
};

struct Slice {
    int hip_device = -1;
    PciAddress pci;
    int render_minor = -1;
    int displayed_frames = 0;
    std::string error;
    std::string unique_id;
    std::map<int, NodeActivity> engine_by_minor;
    std::map<int, uint64_t> share_by_minor;
    std::set<std::string> video_engines;
    std::set<std::string> other_engines;
    bool scheduled = false;
};

std::string g_vcn_fw = "unknown";
std::string g_market_name;
std::string g_gfx_target;
std::string g_stream_text = "H.264";
std::map<int, std::vector<int>> g_vcn_by_gpu;

struct Card {
    std::string key;
    std::string serial;
    std::string compute_partition;
    std::string memory_partition;
    std::string vcn_fw;
    std::vector<Slice> slices;
    std::chrono::system_clock::time_point play_start{};
    std::chrono::system_clock::time_point play_end{};
    bool played = false;
};

enum class Emit { Text, Json, Csv };

Emit g_emit = Emit::Text;
bool g_color = false;

std::string read_text_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        return {};
    }
    std::string line;
    std::getline(input, line);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' ')) {
        line.pop_back();
    }
    return line;
}

std::string drm_unique_id(int minor) {
    if (minor < 0) {
        return "absent";
    }
    const std::string text = read_text_file("/sys/class/drm/renderD" + std::to_string(minor) + "/device/unique_id");
    return text.empty() ? "absent" : text;
}

std::string short_card_key(const std::string& key) {
    const std::string prefix = "0000:";
    if (key.compare(0, prefix.size(), prefix) == 0) {
        return key.substr(prefix.size());
    }
    return key;
}

std::string colorize(const std::string& text, const char* code) {
    if (!g_color) {
        return text;
    }
    return std::string("\033[") + code + "m" + text + "\033[0m";
}

std::string verdict_word(int code) {
    if (code == 0) {
        return colorize("PASS", "32");
    }
    if (code == 2) {
        return colorize("SKIP", "33");
    }
    return colorize("FAIL", "31");
}

std::string pad_word(const std::string& plain, const std::string& shown, int width) {
    std::string out = shown;
    if (static_cast<int>(plain.size()) < width) {
        out.append(static_cast<size_t>(width - static_cast<int>(plain.size())), ' ');
    }
    return out;
}

std::string column(const std::string& text, int width) {
    if (static_cast<int>(text.size()) >= width) {
        return text + "  ";
    }
    return text + std::string(static_cast<size_t>(width - static_cast<int>(text.size())), ' ');
}

std::string gpu_text(int device) { return "GPU " + std::to_string(device); }

std::string partition_text(unsigned function) { return "P" + std::to_string(function); }

std::string node_text(int minor) { return "D" + std::to_string(minor); }

std::string short_bdf(const PciAddress& address) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(2) << address.bus << ":" << std::setw(2) << address.device << "."
        << address.function;
    return out.str();
}

std::string format_engine_time(uint64_t ns) {
    if (ns == 0) {
        return "0";
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(1);
    const double microseconds = static_cast<double>(ns) / 1000.0;
    if (microseconds >= 1000.0) {
        out << (microseconds / 1000.0) << " ms";
    } else {
        out << microseconds << " us";
    }
    return out.str();
}

std::string timestamp_now() {
    const std::time_t now = std::time(nullptr);
    std::tm local {};
    localtime_r(&now, &local);
    char buffer[64] = {};
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M %Z", &local) == 0) {
        return {};
    }
    return buffer;
}

std::string clock_text(std::chrono::system_clock::time_point point) {
    const std::time_t seconds = std::chrono::system_clock::to_time_t(point);
    std::tm local {};
    localtime_r(&seconds, &local);
    char buffer[32] = {};
    std::strftime(buffer, sizeof(buffer), "%H:%M:%S", &local);
    const auto millis =
        std::chrono::duration_cast<std::chrono::milliseconds>(point.time_since_epoch()) % std::chrono::milliseconds(1000);
    std::ostringstream out;
    out << buffer << "." << std::setw(3) << std::setfill('0') << millis.count();
    return out.str();
}

std::string hostname_text() {
    char name[256] = {};
    if (gethostname(name, sizeof(name) - 1) != 0) {
        return "unknown";
    }
    const std::string host = name;
    const size_t dot = host.find('.');
    return dot == std::string::npos ? host : host.substr(0, dot);
}

std::string capture_output(const std::vector<std::string>& args) {
    if (args.empty()) {
        return {};
    }
    int fds[2] = {-1, -1};
    if (pipe(fds) != 0) {
        return {};
    }
    const pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return {};
    }
    if (pid == 0) {
        dup2(fds[1], STDOUT_FILENO);
        close(fds[0]);
        close(fds[1]);
        std::vector<char*> argv;
        argv.reserve(args.size() + 1);
        for (const std::string& arg : args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        argv.push_back(nullptr);
        if (args[0].find('/') == std::string::npos) {
            execvp(argv[0], argv.data());
        } else {
            execv(argv[0], argv.data());
        }
        _exit(127);
    }
    close(fds[1]);
    std::string output;
    char buffer[4096];
    ssize_t count = 0;
    while ((count = read(fds[0], buffer, sizeof(buffer))) > 0) {
        output.append(buffer, static_cast<size_t>(count));
    }
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    return output;
}

std::string first_output_line(const std::string& text) {
    size_t start = 0;
    while (start < text.size() && (text[start] == ' ' || text[start] == '\n' || text[start] == '\r')) {
        start += 1;
    }
    size_t end = text.find('\n', start);
    if (end == std::string::npos) {
        end = text.size();
    }
    while (end > start && (text[end - 1] == '\r' || text[end - 1] == ' ')) {
        end -= 1;
    }
    return text.substr(start, end - start);
}

std::string rocm_version_text() {
    if (const char* env = std::getenv("ROCM_PATH")) {
        const std::string from_env = read_text_file(std::string(env) + "/.info/version");
        if (!from_env.empty()) {
            return from_env;
        }
    }
    const std::string direct = read_text_file("/opt/rocm/.info/version");
    if (!direct.empty()) {
        return direct;
    }
    // Packaged ROCm points /opt/rocm/lib at core-<version>/lib, and .info lives next to that prefix.
    char resolved[4096];
    if (realpath("/opt/rocm/lib", resolved) != nullptr) {
        std::string prefix(resolved);
        const size_t slash = prefix.find_last_of('/');
        if (slash != std::string::npos && slash > 0) {
            const std::string from_prefix = read_text_file(prefix.substr(0, slash) + "/.info/version");
            if (!from_prefix.empty()) {
                return from_prefix;
            }
        }
    }
    for (const char* binary : {"/opt/rocm/bin/hipconfig", "hipconfig"}) {
        const std::string line = first_output_line(capture_output({binary, "--version"}));
        if (!line.empty() && line.find("not found") == std::string::npos) {
            const std::string marker = "HIP version:";
            const size_t position = line.find(marker);
            if (position != std::string::npos) {
                const std::string value = first_output_line(line.substr(position + marker.size()));
                if (!value.empty()) {
                    return value;
                }
            }
            return line;
        }
    }
    return "unknown";
}

std::string amdsmi_version_text() {
    std::ifstream input("/opt/rocm/lib/cmake/amd_smi/amd_smi-config-version.cmake");
    std::string line;
    while (std::getline(input, line)) {
        const std::string marker = "set(PACKAGE_VERSION \"";
        const size_t position = line.find(marker);
        if (position == std::string::npos) {
            continue;
        }
        const size_t start = position + marker.size();
        const size_t end = line.find('"', start);
        if (end == std::string::npos) {
            break;
        }
        std::string version = line.substr(start, end - start);
        const std::string suffix = ".99999";
        if (version.size() > suffix.size() && version.compare(version.size() - suffix.size(), suffix.size(), suffix) == 0) {
            version.erase(version.size() - suffix.size());
        }
        return version;
    }
    return "unknown";
}

std::string rocdecode_version_text() {
    return std::to_string(ROCDECODE_VERSION_MAJOR) + "." + std::to_string(ROCDECODE_VERSION_MINOR) + "." +
           std::to_string(ROCDECODE_VERSION_PATCH);
}

std::string amdgpu_version_text() {
    const std::string version = read_text_file("/sys/module/amdgpu/version");
    return version.empty() ? "unknown" : version;
}

std::string kernel_version_text() {
    utsname name {};
    if (uname(&name) != 0) {
        return "unknown";
    }
    return name.release;
}

std::string json_escape(const std::string& text) {
    std::string out;
    for (char character : text) {
        if (character == '\\' || character == '"') {
            out.push_back('\\');
            out.push_back(character);
        } else if (character == '\n') {
            out += "\\n";
        } else {
            out.push_back(character);
        }
    }
    return out;
}

std::string csv_field(const std::string& text) {
    if (text.find_first_of(",\"\n") == std::string::npos) {
        return text;
    }
    std::string out = "\"";
    for (char character : text) {
        if (character == '"') {
            out += "\"\"";
        } else {
            out.push_back(character);
        }
    }
    out.push_back('"');
    return out;
}

bool read_file(const std::string& path, std::vector<uint8_t>& bytes, std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "stream file was not opened";
        return false;
    }
    input.seekg(0, std::ios::end);
    const std::streamoff length = input.tellg();
    if (length <= 0) {
        error = "stream file is empty";
        return false;
    }
    input.seekg(0, std::ios::beg);
    bytes.resize(static_cast<size_t>(length));
    input.read(reinterpret_cast<char*>(bytes.data()), length);
    if (!input) {
        error = "stream file was not read";
        return false;
    }
    return true;
}

uint64_t parse_counter(const std::string& line) {
    const size_t split = line.find(':');
    if (split == std::string::npos) {
        return 0;
    }
    std::istringstream in(line.substr(split + 1));
    uint64_t value = 0;
    in >> value;
    return value;
}

std::string engine_key(const std::string& line) {
    const size_t split = line.find(':');
    return split == std::string::npos ? line : line.substr(0, split);
}

bool video_engine_key(const std::string& key) {
    return key.find("enc") != std::string::npos || key.find("dec") != std::string::npos ||
           key.find("vcn") != std::string::npos || key.find("video") != std::string::npos ||
           key.find("jpeg") != std::string::npos || key.find("vpe") != std::string::npos;
}

struct FdEngine {
    int fd = -1;
    int minor = -1;
    uint64_t ns = 0;
    std::set<std::string> video_engines;
    std::set<std::string> other_engines;
};

std::vector<FdEngine> sample_video_fds() {
    std::vector<FdEngine> rows;
    DIR* directory = opendir("/proc/self/fd");
    if (directory == nullptr) {
        return rows;
    }
    while (dirent* entry = readdir(directory)) {
        if (entry->d_name[0] == '.') {
            continue;
        }
        const std::string fd_path = std::string("/proc/self/fd/") + entry->d_name;
        char target[256];
        const ssize_t length = readlink(fd_path.c_str(), target, sizeof(target) - 1);
        if (length <= 0) {
            continue;
        }
        target[length] = '\0';
        const std::string node(target);
        const std::string marker = "/dev/dri/renderD";
        const size_t position = node.rfind(marker);
        if (position == std::string::npos) {
            continue;
        }
        FdEngine row;
        row.fd = std::atoi(entry->d_name);
        row.minor = std::atoi(node.c_str() + position + marker.size());
        std::ifstream info(std::string("/proc/self/fdinfo/") + entry->d_name);
        std::string line;
        while (std::getline(info, line)) {
            if (line.rfind("drm-engine-", 0) != 0) {
                continue;
            }
            const std::string key = engine_key(line);
            if (video_engine_key(key)) {
                row.video_engines.insert(key);
                row.ns += parse_counter(line);
            } else {
                row.other_engines.insert(key);
            }
        }
        rows.push_back(row);
    }
    closedir(directory);
    return rows;
}

std::map<int, uint64_t> sum_engine_time(const std::vector<FdEngine>& rows) {
    std::map<int, uint64_t> totals;
    for (const FdEngine& row : rows) {
        totals[row.minor] += row.ns;
    }
    return totals;
}

std::map<int, uint64_t> engine_time_delta(const std::map<int, uint64_t>& before, const std::map<int, uint64_t>& after) {
    std::set<int> minors;
    for (const auto& entry : before) {
        minors.insert(entry.first);
    }
    for (const auto& entry : after) {
        minors.insert(entry.first);
    }
    std::map<int, uint64_t> delta;
    for (int minor : minors) {
        const auto end = after.find(minor);
        const auto start = before.find(minor);
        const uint64_t later = end == after.end() ? 0 : end->second;
        const uint64_t earlier = start == before.end() ? 0 : start->second;
        if (later > earlier) {
            delta[minor] = later - earlier;
        }
    }
    return delta;
}

void add_session_activity(Slice& slice, const std::vector<FdEngine>& before, const std::vector<FdEngine>& after) {
    std::map<int, FdEngine> prior;
    for (const FdEngine& row : before) {
        prior[row.fd] = row;
    }
    for (const FdEngine& row : after) {
        const auto found = prior.find(row.fd);
        uint64_t delta = 0;
        const bool opened_here = found == prior.end() || found->second.minor != row.minor;
        if (opened_here) {
            delta = row.ns;
        } else if (row.ns > found->second.ns) {
            delta = row.ns - found->second.ns;
        }
        if (opened_here || delta > 0 || (slice.render_minor >= 0 && row.minor == slice.render_minor)) {
            slice.video_engines.insert(row.video_engines.begin(), row.video_engines.end());
            slice.other_engines.insert(row.other_engines.begin(), row.other_engines.end());
        }
        if (delta == 0) {
            continue;
        }
        NodeActivity& activity = slice.engine_by_minor[row.minor];
        activity.ns += delta;
        activity.fds += 1;
    }
}

int location_id_for(const PciAddress& address) {
    // KFD publishes this without the PCI domain, so every card at the same bus and function shares it.
    return static_cast<int>(((address.bus & 0xffu) << 8) | ((address.device & 0x1fu) << 3) | (address.function & 0x7u));
}

std::string symlink_basename(const std::string& path) {
    char target[512];
    const ssize_t length = readlink(path.c_str(), target, sizeof(target) - 1);
    if (length <= 0) {
        return {};
    }
    target[length] = '\0';
    const std::string link(target);
    const size_t slash = link.find_last_of('/');
    return slash == std::string::npos ? link : link.substr(slash + 1);
}

std::string pci_slot_of_render(int minor) {
    const std::string root = "/sys/class/drm/renderD" + std::to_string(minor);
    std::ifstream uevent(root + "/device/uevent");
    std::string line;
    while (std::getline(uevent, line)) {
        const std::string key = "PCI_SLOT_NAME=";
        if (line.compare(0, key.size(), key) == 0) {
            return line.substr(key.size());
        }
    }
    return symlink_basename(root + "/device");
}

std::map<std::string, int> render_minor_by_bdf() {
    std::map<std::string, int> minor_for_bdf;
    DIR* directory = opendir("/sys/class/drm");
    if (directory == nullptr) {
        return minor_for_bdf;
    }
    while (dirent* entry = readdir(directory)) {
        const std::string name(entry->d_name);
        const std::string prefix = "renderD";
        if (name.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        const int minor = std::atoi(name.c_str() + prefix.size());
        if (minor < 128) {
            continue;
        }
        const PciAddress pci = parse_pci(pci_slot_of_render(minor));
        if (!pci.valid) {
            continue;
        }
        const std::string bdf = bdf_text(pci);
        const auto existing = minor_for_bdf.find(bdf);
        if (existing == minor_for_bdf.end() || minor < existing->second) {
            minor_for_bdf[bdf] = minor;
        }
    }
    closedir(directory);
    return minor_for_bdf;
}

std::map<int, int> unique_location_minors() {
    std::map<int, int> minor_for_location;
    std::map<int, int> seen;
    const std::string root = "/sys/class/kfd/kfd/topology/nodes";
    DIR* directory = opendir(root.c_str());
    if (directory == nullptr) {
        return minor_for_location;
    }
    while (dirent* entry = readdir(directory)) {
        if (entry->d_name[0] == '.') {
            continue;
        }
        std::ifstream properties(root + "/" + entry->d_name + "/properties");
        int minor = -1;
        int location = -1;
        std::string key;
        std::string value;
        while (properties >> key >> value) {
            if (key == "drm_render_minor") {
                minor = std::atoi(value.c_str());
            } else if (key == "location_id") {
                location = std::atoi(value.c_str());
            }
        }
        if (minor < 128 || location < 0) {
            continue;
        }
        seen[location] += 1;
        minor_for_location[location] = minor;
    }
    closedir(directory);
    for (const auto& entry : seen) {
        if (entry.second != 1) {
            minor_for_location.erase(entry.first);
        }
    }
    return minor_for_location;
}

class StreamDecoder {
 public:
    explicit StreamDecoder(int hip_device) : hip_device_(hip_device) {}

    ~StreamDecoder() { close(); }

    StreamDecoder(const StreamDecoder&) = delete;
    StreamDecoder& operator=(const StreamDecoder&) = delete;

    bool play(const std::vector<uint8_t>& stream, int& displayed, std::string& error) {
        displayed = 0;
        error.clear();
        const hipError_t selected = hipSetDevice(hip_device_);
        if (selected != hipSuccess) {
            error = hip_status_name(selected);
            return false;
        }
        RocdecParserParams params{};
        params.codec_type = rocDecVideoCodec_AVC;
        params.max_num_decode_surfaces = 8;
        params.max_display_delay = 1;
        params.user_data = this;
        params.pfn_sequence_callback = &StreamDecoder::sequence_callback;
        params.pfn_decode_picture = &StreamDecoder::decode_callback;
        params.pfn_display_picture = &StreamDecoder::display_callback;
        const rocDecStatus created = rocDecCreateVideoParser(&parser_, &params);
        if (created != ROCDEC_SUCCESS) {
            error = roc_status_name(created);
            return false;
        }
        std::vector<std::pair<size_t, size_t>> units;
        size_t cursor = 0;
        while (cursor + 3 < stream.size()) {
            size_t start = std::string::npos;
            size_t prefix = 0;
            for (size_t index = cursor; index + 3 < stream.size(); ++index) {
                if (stream[index] == 0 && stream[index + 1] == 0 && stream[index + 2] == 1) {
                    start = index;
                    prefix = 3;
                    break;
                }
                if (index + 4 < stream.size() && stream[index] == 0 && stream[index + 1] == 0 && stream[index + 2] == 0 &&
                    stream[index + 3] == 1) {
                    start = index;
                    prefix = 4;
                    break;
                }
            }
            if (start == std::string::npos) {
                break;
            }
            size_t next = stream.size();
            for (size_t index = start + prefix; index + 3 < stream.size(); ++index) {
                const bool three = stream[index] == 0 && stream[index + 1] == 0 && stream[index + 2] == 1;
                const bool four = index + 4 < stream.size() && stream[index] == 0 && stream[index + 1] == 0 &&
                                  stream[index + 2] == 0 && stream[index + 3] == 1;
                if (three || four) {
                    next = index;
                    break;
                }
            }
            units.emplace_back(start, next - start);
            cursor = next;
        }
        if (units.empty()) {
            error = "stream has no H.264 start codes";
            return false;
        }
        for (int repeat = 0; repeat < kStreamRepeats; ++repeat) {
            for (const auto& unit : units) {
                RocdecSourceDataPacket packet{};
                packet.flags = ROCDEC_PKT_TIMESTAMP;
                packet.pts = static_cast<RocdecTimeStamp>(repeat * static_cast<int>(units.size()));
                packet.payload = stream.data() + unit.first;
                packet.payload_size = static_cast<uint32_t>(unit.second);
                const rocDecStatus parsed = rocDecParseVideoData(parser_, &packet);
                if (parsed != ROCDEC_SUCCESS) {
                    error = callback_error_.empty() ? roc_status_name(parsed) : callback_error_;
                    return false;
                }
            }
        }
        RocdecSourceDataPacket end{};
        end.flags = ROCDEC_PKT_ENDOFSTREAM | ROCDEC_PKT_NOTIFY_EOS;
        const rocDecStatus flushed = rocDecParseVideoData(parser_, &end);
        if (flushed != ROCDEC_SUCCESS) {
            error = callback_error_.empty() ? roc_status_name(flushed) : callback_error_;
            return false;
        }
        const hipError_t synced = hipDeviceSynchronize();
        if (synced != hipSuccess) {
            error = hip_status_name(synced);
            return false;
        }
        displayed = displayed_;
        if (!callback_error_.empty()) {
            error = callback_error_;
            return false;
        }
        return displayed > 0;
    }

 private:
    void close() {
        if (parser_ != nullptr) {
            rocDecDestroyVideoParser(parser_);
            parser_ = nullptr;
        }
        if (decoder_ != nullptr) {
            rocDecDestroyDecoder(decoder_);
            decoder_ = nullptr;
        }
    }

    static int ROCDECAPI sequence_callback(void* user, RocdecVideoFormat* format) {
        return static_cast<StreamDecoder*>(user)->on_sequence(format);
    }

    static int ROCDECAPI decode_callback(void* user, RocdecPicParams* picture) {
        return static_cast<StreamDecoder*>(user)->on_decode(picture);
    }

    static int ROCDECAPI display_callback(void* user, RocdecParserDispInfo*) {
        static_cast<StreamDecoder*>(user)->displayed_ += 1;
        return 1;
    }

    int on_sequence(RocdecVideoFormat* format) {
        if (format == nullptr || format->bit_depth_luma_minus8 != 0 || format->chroma_format != rocDecVideoChromaFormat_420) {
            callback_error_ = "stream is not 8-bit 4:2:0 H.264";
            if (format != nullptr) {
                callback_error_ += " bit_depth_minus8=" + std::to_string(format->bit_depth_luma_minus8) +
                                   " chroma=" + std::to_string(static_cast<int>(format->chroma_format)) +
                                   " " + std::to_string(format->coded_width) + "x" + std::to_string(format->coded_height);
            }
            return 0;
        }
        const int surfaces = std::max<int>(format->min_num_decode_surfaces, 8);
        if (decoder_ != nullptr) {
            return surfaces;
        }
        const hipError_t selected = hipSetDevice(hip_device_);
        if (selected != hipSuccess) {
            callback_error_ = hip_status_name(selected);
            return 0;
        }
        int width = static_cast<int>(format->display_area.right - format->display_area.left);
        int height = static_cast<int>(format->display_area.bottom - format->display_area.top);
        if (width <= 0 || height <= 0) {
            width = static_cast<int>(format->coded_width);
            height = static_cast<int>(format->coded_height);
        }
        width &= ~1;
        height &= ~1;
        RocDecoderCreateInfo info{};
        g_stream_text = "H.264 " + std::to_string(format->coded_width) + "x" + std::to_string(format->coded_height);
        info.device_id = static_cast<uint8_t>(hip_device_);
        info.width = format->coded_width;
        info.height = format->coded_height;
        info.max_width = format->coded_width;
        info.max_height = format->coded_height;
        info.num_decode_surfaces = static_cast<uint32_t>(surfaces);
        info.codec_type = rocDecVideoCodec_AVC;
        info.chroma_format = format->chroma_format;
        info.bit_depth_minus_8 = format->bit_depth_luma_minus8;
        info.output_format = rocDecVideoSurfaceFormat_NV12;
        info.target_width = static_cast<uint32_t>(width);
        info.target_height = static_cast<uint32_t>(height);
        info.num_output_surfaces = 2;
        info.display_rect.left = static_cast<int16_t>(format->display_area.left);
        info.display_rect.top = static_cast<int16_t>(format->display_area.top);
        info.display_rect.right = static_cast<int16_t>(format->display_area.left + width);
        info.display_rect.bottom = static_cast<int16_t>(format->display_area.top + height);
        info.target_rect.right = static_cast<int16_t>(width);
        info.target_rect.bottom = static_cast<int16_t>(height);
        const rocDecStatus status = rocDecCreateDecoder(&decoder_, &info);
        if (status != ROCDEC_SUCCESS) {
            callback_error_ = roc_status_name(status);
            return 0;
        }
        return surfaces;
    }

    int on_decode(RocdecPicParams* picture) {
        if (decoder_ == nullptr || picture == nullptr) {
            callback_error_ = "decoder was not created";
            return 0;
        }
        const rocDecStatus status = rocDecDecodeFrame(decoder_, picture);
        if (status != ROCDEC_SUCCESS) {
            callback_error_ = roc_status_name(status);
            return 0;
        }
        return 1;
    }

    int hip_device_;
    RocdecVideoParser parser_ = nullptr;
    rocDecDecoderHandle decoder_ = nullptr;
    int displayed_ = 0;
    std::string callback_error_;
};

bool discover(std::vector<Card>& cards, std::string& error) {
    int count = 0;
    const hipError_t status = hipGetDeviceCount(&count);
    if (status != hipSuccess) {
        error = hip_status_name(status);
        return false;
    }
    if (count <= 0) {
        error = "no GPU";
        return false;
    }
    const std::map<std::string, int> minors_by_bdf = render_minor_by_bdf();
    const std::map<int, int> minors_by_location = unique_location_minors();
    std::map<std::string, Card> grouped;
    for (int device = 0; device < count; ++device) {
        char bus[64] = {};
        const hipError_t pci_status = hipDeviceGetPCIBusId(bus, static_cast<int>(sizeof(bus)), device);
        if (pci_status != hipSuccess) {
            error = hip_status_name(pci_status);
            return false;
        }
        Slice slice;
        slice.hip_device = device;
        slice.pci = parse_pci(bus);
        if (!slice.pci.valid) {
            error = std::string("unparsed PCI address ") + bus;
            return false;
        }
        const auto by_bdf = minors_by_bdf.find(bdf_text(slice.pci));
        if (by_bdf != minors_by_bdf.end()) {
            slice.render_minor = by_bdf->second;
        } else {
            const auto by_location = minors_by_location.find(location_id_for(slice.pci));
            if (by_location != minors_by_location.end()) {
                slice.render_minor = by_location->second;
            }
        }
        slice.unique_id = drm_unique_id(slice.render_minor);
        Card& card = grouped[card_key(slice.pci)];
        card.key = card_key(slice.pci);
        card.slices.push_back(slice);
    }
    cards.clear();
    for (auto& entry : grouped) {
        std::sort(entry.second.slices.begin(), entry.second.slices.end(),
                  [](const Slice& left, const Slice& right) { return left.pci.function < right.pci.function; });
        cards.push_back(std::move(entry.second));
    }
    std::sort(cards.begin(), cards.end(), [](const Card& left, const Card& right) { return left.key < right.key; });
    for (Card& card : cards) {
        PciAddress physical = card.slices.front().pci;
        physical.function = 0;
        const std::string device = "/sys/bus/pci/devices/" + bdf_text(physical);
        card.serial = read_text_file(device + "/serial_number");
        card.compute_partition = read_text_file(device + "/current_compute_partition");
        card.memory_partition = read_text_file(device + "/current_memory_partition");
        const std::string firmware = read_text_file(device + "/fw_version/vcn_fw_version");
        if (!firmware.empty()) {
            card.vcn_fw = firmware;
        }
    }
    std::set<std::string> firmware_values;
    for (const Card& card : cards) {
        if (!card.vcn_fw.empty()) {
            firmware_values.insert(card.vcn_fw);
        }
    }
    if (firmware_values.empty()) {
        g_vcn_fw = "unknown";
    } else if (firmware_values.size() == 1) {
        g_vcn_fw = *firmware_values.begin();
    } else {
        std::ostringstream firmware;
        bool first = true;
        for (const Card& card : cards) {
            if (card.vcn_fw.empty()) {
                continue;
            }
            if (!first) {
                firmware << " ";
            }
            first = false;
            firmware << short_card_key(card.key) << "=" << card.vcn_fw;
        }
        g_vcn_fw = firmware.str();
    }
    return !cards.empty();
}

Layout detect_layout(const std::vector<Card>& cards) {
    if (cards.empty()) {
        return Layout::None;
    }
    int lowest = static_cast<int>(cards.front().slices.size());
    int highest = lowest;
    for (const Card& card : cards) {
        const int count = static_cast<int>(card.slices.size());
        lowest = std::min(lowest, count);
        highest = std::max(highest, count);
    }
    if (highest == 1 && lowest == 1) {
        return Layout::Spx;
    }
    if (highest == 2) {
        return Layout::Dpx;
    }
    if (highest >= 5 || (lowest != highest && highest > 2)) {
        return Layout::Cpx;
    }
    return Layout::Other;
}

std::string gpu_span(const Card& card) {
    int low = card.slices.front().hip_device;
    int high = low;
    for (const Slice& slice : card.slices) {
        low = std::min(low, slice.hip_device);
        high = std::max(high, slice.hip_device);
    }
    std::ostringstream out;
    out << "GPU " << low;
    if (high != low) {
        out << "-" << high;
    }
    return out.str();
}

std::string partition_span(const Card& card) {
    unsigned low = card.slices.front().pci.function;
    unsigned high = low;
    for (const Slice& slice : card.slices) {
        low = std::min(low, slice.pci.function);
        high = std::max(high, slice.pci.function);
    }
    std::ostringstream out;
    out << "partition " << low;
    if (high != low) {
        out << "-" << high;
    }
    return out.str();
}

std::string layout_detail(const std::vector<Card>& cards) {
    std::ostringstream out;
    out << cards.size() << " cards";
    for (const Card& card : cards) {
        out << ", " << card.key << " " << card.slices.size() << (card.slices.size() == 1 ? " slice, " : " slices, ")
            << gpu_span(card) << ", " << partition_span(card);
    }
    return out.str();
}

std::string member_list(const std::vector<const Slice*>& group) {
    std::ostringstream out;
    for (size_t index = 0; index < group.size(); ++index) {
        if (index != 0) {
            out << " and ";
        }
        const Slice& slice = *group[index];
        out << "partition " << slice.pci.function << " GPU " << slice.hip_device;
    }
    return out.str();
}

std::string frame_outcome(const Slice& slice) {
    std::ostringstream out;
    out << "GPU " << slice.hip_device << " partition " << slice.pci.function << " " << bdf_text(slice.pci) << " ";
    if (slice.render_minor < 0) {
        out << "render node not found";
    } else if (!slice.error.empty()) {
        out << "failed (" << slice.error << ")";
    } else if (slice.displayed_frames < 1) {
        out << "displayed 0 frames";
    } else {
        out << "displayed " << slice.displayed_frames << " frames";
    }
    return out.str();
}

bool touch_device(int device, std::string& error) {
    const hipError_t selected = hipSetDevice(device);
    if (selected != hipSuccess) {
        error = hip_status_name(selected);
        return false;
    }
    void* buffer = nullptr;
    const hipError_t allocated = hipMalloc(&buffer, 4096);
    if (allocated != hipSuccess) {
        error = hip_status_name(allocated);
        return false;
    }
    const hipError_t freed = hipFree(buffer);
    if (freed != hipSuccess) {
        error = hip_status_name(freed);
        return false;
    }
    const hipError_t synced = hipDeviceSynchronize();
    if (synced != hipSuccess) {
        error = hip_status_name(synced);
        return false;
    }
    return true;
}

int pair_of(unsigned function) { return static_cast<int>(function / 2); }

bool group_contains(const std::vector<const Slice*>& group, const Slice* slice) {
    return std::find(group.begin(), group.end(), slice) != group.end();
}

struct LayoutRow {
    std::string card;
    std::string serial;
    std::string slices;
    std::string gpus;
    std::string partitions;
    bool pass = false;
    std::string note;
};

struct LayoutReport {
    bool ran = false;
    int code = 2;
    std::string expected;
    std::string skip;
    int gpus = 0;
    int expected_gpus = 0;
    int pass_cards = 0;
    int cards = 0;
    std::vector<LayoutRow> rows;
    std::vector<std::string> missing;
};

struct DecoderRow {
    int index = 0;
    std::string pair;
    std::string gpus;
    std::string own_nodes;
    uint64_t own_ns = 0;
    uint64_t pair_ns = 0;
    uint64_t other_ns = 0;
    std::string uuid;
    std::string vcn;
    std::string ran_on;
    std::string frames;
    bool pass = false;
    std::string reason;
};

struct CardDecode {
    std::string card;
    int active = 0;
    int decoders = 0;
    int sessions = 0;
    std::string active_node;
    std::string active_gpu;
    std::string active_partition;
    bool pass = false;
    std::string window;
    std::string uuid_note;
    std::vector<DecoderRow> rows;
    std::vector<std::string> verbose;
};

struct DecodeReport {
    bool ran = false;
    int code = 2;
    std::string skip;
    std::string rule;
    std::string sample_window;
    bool split_by_count = false;
    std::vector<CardDecode> cards;
    int active_nodes = 0;
    int gpus = 0;
    int decoders_used = 0;
    int decoders_expected = 0;
    std::string per_card_ratio;
};

LayoutReport g_layout;
DecodeReport g_decode;
std::string g_compute;
std::string g_memory;
Layout g_detected = Layout::None;
int g_visible_gpus = 0;
int g_cards = 0;
std::vector<int> g_xcd_order;
std::string g_decode_label;
bool g_touch_all = false;
bool g_keep_decoder = false;
bool g_parallel = false;
bool g_allow_partial = false;

struct PlayStep {
    std::string card;
    int function = 0;
};

std::vector<PlayStep> g_sequence;

void remember_host(const std::vector<Card>& cards) {
    g_cards = static_cast<int>(cards.size());
    g_visible_gpus = 0;
    g_detected = detect_layout(cards);
    for (const Card& card : cards) {
        g_visible_gpus += static_cast<int>(card.slices.size());
        if (g_compute.empty()) {
            g_compute = card.compute_partition;
        }
        if (g_memory.empty()) {
            g_memory = card.memory_partition;
        }
    }
}

std::string host_mode_text() {
    if (!g_compute.empty()) {
        std::string text = g_compute;
        if (!g_memory.empty()) {
            text += " / " + g_memory;
        }
        return text;
    }
    return layout_name(g_detected);
}

std::string join_partitions(const std::vector<unsigned>& functions) {
    if (functions.empty()) {
        return "-";
    }
    bool contiguous = true;
    for (size_t index = 1; index < functions.size(); ++index) {
        if (functions[index] != functions[index - 1] + 1) {
            contiguous = false;
        }
    }
    if (functions.size() == 1) {
        return partition_text(functions.front());
    }
    if (contiguous) {
        return partition_text(functions.front()) + "-" + partition_text(functions.back());
    }
    std::ostringstream out;
    for (size_t index = 0; index < functions.size(); ++index) {
        if (index != 0) {
            out << ", ";
        }
        out << partition_text(functions[index]);
    }
    return out.str();
}

std::string gpu_id_span(const Card& card) {
    int low = card.slices.front().hip_device;
    int high = low;
    for (const Slice& slice : card.slices) {
        low = std::min(low, slice.hip_device);
        high = std::max(high, slice.hip_device);
    }
    if (low == high) {
        return std::to_string(low);
    }
    return std::to_string(low) + "-" + std::to_string(high);
}

const Slice* slice_for_minor(const std::vector<Card>& cards, int minor) {
    for (const Card& card : cards) {
        for (const Slice& slice : card.slices) {
            if (slice.render_minor == minor) {
                return &slice;
            }
        }
    }
    return nullptr;
}

std::string node_with_owner(const std::vector<Card>& cards, int minor, bool full_name) {
    std::ostringstream out;
    out << (full_name ? "render" : "") << node_text(minor);
    const Slice* slice = slice_for_minor(cards, minor);
    if (slice != nullptr) {
        out << " (" << gpu_text(slice->hip_device) << ", " << partition_text(slice->pci.function) << ")";
    }
    return out.str();
}

std::string where_sessions_ran(const std::vector<Card>& cards, const std::set<int>& minors) {
    if (minors.empty()) {
        return "-";
    }
    std::ostringstream out;
    bool first = true;
    for (int minor : minors) {
        if (!first) {
            out << ", ";
        }
        first = false;
        out << node_text(minor);
        const Slice* slice = slice_for_minor(cards, minor);
        if (slice != nullptr) {
            out << " (" << partition_text(slice->pci.function) << ")";
        }
    }
    return out.str();
}

std::string mode_skip_text(Layout detected) {
    return std::string("host mode is ") + layout_name(detected) + "; this program runs for " + layout_name(kRequiredLayout);
}

int discover_or_note(std::vector<Card>& cards, std::string& skip, int& code) {
    std::string error;
    if (discover(cards, error)) {
        remember_host(cards);
        return 0;
    }
    if (error == "no GPU") {
        code = 2;
        skip = "no GPU";
    } else {
        code = 1;
        skip = error;
    }
    return code;
}

void fill_layout(const std::vector<Card>& cards) {
    const unsigned required_slices = slices_for(kRequiredLayout);
    std::vector<unsigned> expected_functions;
    for (unsigned function = 0; function < required_slices; ++function) {
        expected_functions.push_back(function);
    }
    g_layout.expected = std::to_string(required_slices) + (required_slices == 1 ? " slice per card, " : " slices per card, ") +
                        join_partitions(expected_functions);
    g_layout.cards = static_cast<int>(cards.size());
    g_layout.expected_gpus = g_layout.cards * static_cast<int>(required_slices);
    g_layout.gpus = g_visible_gpus;
    bool every = true;
    int expected_first_gpu = 0;
    for (const Card& card : cards) {
        std::vector<unsigned> functions;
        for (const Slice& slice : card.slices) {
            functions.push_back(slice.pci.function);
        }
        std::vector<unsigned> missing;
        for (unsigned function : expected_functions) {
            if (std::find(functions.begin(), functions.end(), function) == functions.end()) {
                missing.push_back(function);
            }
        }
        std::vector<unsigned> extra;
        for (unsigned function : functions) {
            if (function >= required_slices) {
                extra.push_back(function);
            }
        }
        LayoutRow row;
        row.card = card.key;
        row.serial = card.serial.empty() ? "-" : card.serial;
        row.slices = std::to_string(card.slices.size()) + "/" + std::to_string(required_slices);
        row.gpus = gpu_id_span(card);
        row.partitions = join_partitions(functions);
        row.pass = missing.empty() && extra.empty() && card.slices.size() == required_slices;
        if (!missing.empty()) {
            row.note = "missing " + join_partitions(missing);
            g_layout.missing.push_back(card.key + " " + row.note);
        }
        if (!extra.empty()) {
            if (!row.note.empty()) {
                row.note += "; ";
            }
            row.note += "extra " + join_partitions(extra);
            if (missing.empty()) {
                g_layout.missing.push_back(card.key + " " + row.note);
            }
        }
        int actual_first_gpu = card.slices.front().hip_device;
        for (const Slice& slice : card.slices) {
            actual_first_gpu = std::min(actual_first_gpu, slice.hip_device);
        }
        const int shift = actual_first_gpu - expected_first_gpu;
        if (shift != 0) {
            if (!row.note.empty()) {
                row.note += "; ";
            }
            row.note += "GPU ids shifted " + std::to_string(shift);
        }
        expected_first_gpu += static_cast<int>(required_slices);
        if (row.pass) {
            g_layout.pass_cards += 1;
        } else {
            every = false;
        }
        g_layout.rows.push_back(row);
    }
    g_layout.code = every ? 0 : 1;
}

std::string fd_count_text(int fds) { return std::to_string(fds) + (fds == 1 ? " fd" : " fds"); }

std::string own_time_reason(const std::vector<const Slice*>& group) {
    std::set<std::string> video;
    std::set<std::string> other;
    for (const Slice* slice : group) {
        video.insert(slice->video_engines.begin(), slice->video_engines.end());
        other.insert(slice->other_engines.begin(), slice->other_engines.end());
    }
    if (!video.empty()) {
        return "no own-node time";
    }
    if (other.empty()) {
        return "no own-node time; fdinfo has no drm-engine counter";
    }
    std::ostringstream out;
    out << "no own-node time; fdinfo engines ";
    int shown = 0;
    for (const std::string& name : other) {
        if (shown == 4) {
            out << ", ...";
            break;
        }
        if (shown != 0) {
            out << ", ";
        }
        out << name;
        shown += 1;
    }
    out << " are not video";
    return out.str();
}

struct CountedNode {
    int minor = -1;
    uint64_t ns = 0;
    int fds = 0;
};

// A render node opened for an earlier card stays open, and its engine counter can still move during the next
// card's first session. That tail is not this session's decode. Keep it only when the session has no time on its own card.
std::vector<CountedNode> session_nodes(const Slice& slice, const std::set<int>& card_minors) {
    const auto on_card = [&](int minor) { return card_minors.empty() || card_minors.count(minor) != 0; };
    bool has_on_card = false;
    for (const auto& activity : slice.engine_by_minor) {
        if (activity.second.ns > 0 && on_card(activity.first)) {
            has_on_card = true;
            break;
        }
    }
    std::vector<CountedNode> nodes;
    for (const auto& activity : slice.engine_by_minor) {
        if (activity.second.ns == 0) {
            continue;
        }
        if (has_on_card && !on_card(activity.first)) {
            continue;
        }
        CountedNode node;
        node.minor = activity.first;
        node.ns = activity.second.ns;
        node.fds = activity.second.fds;
        nodes.push_back(node);
    }
    return nodes;
}

bool xcd_selected(unsigned function) {
    if (g_xcd_order.empty()) {
        return true;
    }
    return std::find(g_xcd_order.begin(), g_xcd_order.end(), static_cast<int>(function)) != g_xcd_order.end();
}

bool run_slice(const Slice& slice) {
    if (!g_sequence.empty()) {
        return slice.scheduled;
    }
    return xcd_selected(slice.pci.function);
}

std::string xcd_selection_text() {
    if (g_xcd_order.empty()) {
        return "all XCDs";
    }
    std::ostringstream out;
    if (g_xcd_order.size() == 1) {
        out << "XCD " << g_xcd_order.front() << " only";
    } else {
        out << "XCD order ";
        for (size_t index = 0; index < g_xcd_order.size(); ++index) {
            if (index != 0) {
                out << ",";
            }
            out << g_xcd_order[index];
        }
    }
    if (g_touch_all) {
        out << " | touch all partitions";
    }
    if (g_keep_decoder) {
        out << " | decoder kept open";
    }
    if (g_parallel) {
        out << " | parallel";
    }
    if (!g_sequence.empty()) {
        out << " | sequence";
        for (const PlayStep& step : g_sequence) {
            out << " " << step.card << ":" << step.function;
        }
    }
    return out.str();
}

void fill_decode(const std::vector<Card>& cards, const std::map<int, uint64_t>& node_delta, Layout required) {
    if (g_xcd_order.size() == 1) {
        g_decode.rule = "XCD " + std::to_string(g_xcd_order.front()) +
                        " engine time only on that XCD's render node; at least 1 frame on that XCD";
    } else if (g_xcd_order.size() > 1) {
        g_decode.rule = "play order " + xcd_selection_text() +
                        "; each selected XCD is scored on its own row";
    } else if (required == Layout::Cpx) {
        g_decode.rule =
            "decoder N engine time on its own pair's render node, only from that pair; all 4 decoders active on every "
            "card; at least 1 frame per GPU";
    } else if (required == Layout::Dpx) {
        g_decode.rule = "each half's engine time only on that half's render node; both halves active; at least 1 frame per GPU";
    } else {
        g_decode.rule = "each card's engine time only on that card's render node; at least 1 frame per GPU";
    }
    int selected_gpus = 0;
    for (const Card& card : cards) {
        for (const Slice& slice : card.slices) {
            if (run_slice(slice)) {
                selected_gpus += 1;
            }
        }
    }
    g_decode.gpus = selected_gpus;
    std::set<int> active_nodes;
    bool every_card = true;
    std::set<std::string> ratios;
    for (const Card& card : cards) {
        if (!g_sequence.empty()) {
            bool any_step = false;
            for (const Slice& slice : card.slices) {
                if (slice.scheduled) {
                    any_step = true;
                }
            }
            if (!any_step) {
                continue;
            }
        }
        CardDecode block;
        block.card = card.key;
        block.pass = true;
        block.sessions = 0;
        for (const Slice& slice : card.slices) {
            if (run_slice(slice)) {
                block.sessions += 1;
            }
        }
        if (card.played) {
            const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(card.play_end - card.play_start).count();
            block.window = clock_text(card.play_start) + "-" + clock_text(card.play_end) + " (" + std::to_string(millis) + " ms)";
        }
        std::vector<std::pair<int, std::vector<const Slice*>>> groups;
        auto push_slice = [&](const Slice& slice, int decoder) {
            for (auto& group : groups) {
                if (group.first == decoder) {
                    group.second.push_back(&slice);
                    return;
                }
            }
            groups.push_back({decoder, {&slice}});
        };
        if (!g_sequence.empty()) {
            for (const Slice& slice : card.slices) {
                if (slice.scheduled) {
                    push_slice(slice, static_cast<int>(slice.pci.function));
                }
            }
        } else if (g_xcd_order.empty()) {
            for (const Slice& slice : card.slices) {
                const int decoder = required == Layout::Cpx ? pair_of(slice.pci.function) : static_cast<int>(slice.pci.function);
                push_slice(slice, decoder);
            }
        } else {
            for (int xcd : g_xcd_order) {
                for (const Slice& slice : card.slices) {
                    if (static_cast<int>(slice.pci.function) == xcd) {
                        push_slice(slice, xcd);
                    }
                }
            }
        }
        std::set<int> card_minors;
        for (const Slice& slice : card.slices) {
            if (slice.render_minor >= 0) {
                card_minors.insert(slice.render_minor);
            }
        }
        std::map<int, int> hits;
        std::map<int, uint64_t> session_ns;
        for (const Slice& slice : card.slices) {
            if (!run_slice(slice)) {
                continue;
            }
            const std::vector<CountedNode> nodes = session_nodes(slice, card_minors);
            for (const CountedNode& node : nodes) {
                hits[node.minor] += 1;
                session_ns[node.minor] += node.ns;
            }
            if (verbose_output()) {
                std::ostringstream line;
                line << gpu_text(slice.hip_device) << " " << partition_text(slice.pci.function) << " " << short_bdf(slice.pci);
                if (nodes.empty()) {
                    line << "  no video-engine descriptor";
                }
                for (const CountedNode& node : nodes) {
                    line << "  " << node_text(node.minor) << "  " << fd_count_text(node.fds) << "  session sample "
                         << format_engine_time(node.ns);
                }
                if (!slice.error.empty()) {
                    line << "  " << slice.error;
                }
                block.verbose.push_back(line.str());
            }
        }
        int best_minor = -1;
        uint64_t best_ns = 0;
        int best_hits = -1;
        for (const auto& hit : hits) {
            const uint64_t ns = session_ns[hit.first];
            if (best_minor < 0 || ns > best_ns || (ns == best_ns && hit.second > best_hits)) {
                best_minor = hit.first;
                best_ns = ns;
                best_hits = hit.second;
            }
        }
        if (best_minor >= 0 && best_ns > 0) {
            active_nodes.insert(best_minor);
            block.active_node = "render" + node_text(best_minor);
            const Slice* owner = slice_for_minor(cards, best_minor);
            if (owner != nullptr) {
                block.active_gpu = std::to_string(owner->hip_device);
                block.active_partition = partition_text(owner->pci.function);
            }
        }
        for (const auto& entry : groups) {
            const std::vector<const Slice*>& group = entry.second;
            std::set<int> own_minors;
            std::vector<unsigned> present;
            std::vector<std::string> decode_errors;
            bool nodes_ok = true;
            bool frames_ok = true;
            for (const Slice* slice : group) {
                present.push_back(slice->pci.function);
                if (slice->render_minor >= 0) {
                    own_minors.insert(slice->render_minor);
                } else {
                    nodes_ok = false;
                }
                if (!slice->error.empty()) {
                    frames_ok = false;
                    decode_errors.push_back("GPU " + std::to_string(slice->hip_device) + " " + slice->error);
                } else if (slice->displayed_frames < 1) {
                    frames_ok = false;
                }
            }
            std::vector<unsigned> missing;
            if (required == Layout::Cpx && g_xcd_order.empty()) {
                const unsigned left = static_cast<unsigned>(entry.first * 2);
                const unsigned right = left + 1;
                if (std::find(present.begin(), present.end(), left) == present.end()) {
                    missing.push_back(left);
                }
                if (std::find(present.begin(), present.end(), right) == present.end()) {
                    missing.push_back(right);
                }
            }
            std::set<unsigned> foreign;
            for (const Slice& slice : card.slices) {
                if (group_contains(group, &slice)) {
                    continue;
                }
                for (int minor : own_minors) {
                    const auto found = slice.engine_by_minor.find(minor);
                    if (found != slice.engine_by_minor.end() && found->second.ns > 0) {
                        foreign.insert(slice.pci.function);
                    }
                }
            }
            std::set<int> outside;
            std::set<int> ran;
            for (const Slice* slice : group) {
                for (const CountedNode& node : session_nodes(*slice, card_minors)) {
                    ran.insert(node.minor);
                    if (own_minors.count(node.minor) == 0) {
                        outside.insert(node.minor);
                    }
                }
            }
            uint64_t own_ns = 0;
            for (int minor : own_minors) {
                const auto found = node_delta.find(minor);
                if (found != node_delta.end()) {
                    own_ns += found->second;
                }
            }
            uint64_t pair_ns = 0;
            uint64_t other_ns = 0;
            for (const Slice& slice : card.slices) {
                uint64_t share = 0;
                for (int minor : own_minors) {
                    const auto found = slice.share_by_minor.find(minor);
                    if (found != slice.share_by_minor.end()) {
                        share += found->second;
                    }
                }
                if (group_contains(group, &slice)) {
                    pair_ns += share;
                } else {
                    other_ns += share;
                }
            }
            if (pair_ns + other_ns == 0 && own_ns > 0) {
                int pair_hits = 0;
                int other_hits = 0;
                for (const Slice& slice : card.slices) {
                    bool hit = false;
                    for (int minor : own_minors) {
                        const auto found = slice.engine_by_minor.find(minor);
                        if (found != slice.engine_by_minor.end() && found->second.ns > 0) {
                            hit = true;
                        }
                    }
                    if (!hit) {
                        continue;
                    }
                    if (group_contains(group, &slice)) {
                        pair_hits += 1;
                    } else {
                        other_hits += 1;
                    }
                }
                const int hits = pair_hits + other_hits;
                if (hits > 0) {
                    pair_ns = own_ns * static_cast<uint64_t>(pair_hits) / static_cast<uint64_t>(hits);
                    other_ns = own_ns - pair_ns;
                    g_decode.split_by_count = true;
                }
            }
            const bool pass = nodes_ok && frames_ok && own_ns > 0 && foreign.empty() && outside.empty();
            if (own_ns > 0) {
                block.active += 1;
            }
            DecoderRow row;
            row.index = entry.first;
            row.pair = join_partitions(present);
            int glow = group.front()->hip_device;
            int ghigh = glow;
            for (const Slice* slice : group) {
                glow = std::min(glow, slice->hip_device);
                ghigh = std::max(ghigh, slice->hip_device);
            }
            row.gpus = glow == ghigh ? std::to_string(glow) : std::to_string(glow) + "-" + std::to_string(ghigh);
            std::ostringstream nodes;
            bool first_node = true;
            if (own_minors.empty()) {
                nodes << "-";
            }
            for (int minor : own_minors) {
                if (!first_node) {
                    nodes << "/";
                }
                first_node = false;
                nodes << node_text(minor);
            }
            row.own_nodes = nodes.str();
            row.own_ns = own_ns;
            row.pair_ns = pair_ns;
            row.other_ns = other_ns;
            std::ostringstream uuid;
            for (size_t index = 0; index < group.size(); ++index) {
                if (index != 0) {
                    uuid << "/";
                }
                const std::string& id = group[index]->unique_id;
                uuid << (id.empty() ? "absent" : id);
            }
            row.uuid = uuid.str().empty() ? "absent" : uuid.str();
            row.vcn = "n/a";
            for (const Slice& slice : card.slices) {
                const auto engines = g_vcn_by_gpu.find(slice.hip_device);
                if (engines == g_vcn_by_gpu.end() || engines->second.empty()) {
                    continue;
                }
                const int vcn_index = required == Layout::Cpx && !g_xcd_order.empty()
                                          ? pair_of(static_cast<unsigned>(entry.first))
                                          : entry.first;
                if (vcn_index >= 0 && static_cast<size_t>(vcn_index) < engines->second.size()) {
                    row.vcn = std::to_string(engines->second[static_cast<size_t>(vcn_index)]) + "%";
                }
                break;
            }
            row.ran_on = where_sessions_ran(cards, ran.empty() ? outside : ran);
            std::ostringstream frames;
            for (size_t index = 0; index < group.size(); ++index) {
                if (index != 0) {
                    frames << ",";
                }
                const Slice* slice = group[index];
                frames << slice->displayed_frames;
            }
            row.frames = frames.str();
            row.pass = pass;
            std::vector<std::string> reasons;
            if (!missing.empty()) {
                reasons.push_back("missing " + join_partitions(missing));
            }
            if (!nodes_ok) {
                reasons.push_back("render node not found");
            }
            if (!decode_errors.empty()) {
                std::ostringstream errors;
                for (size_t index = 0; index < decode_errors.size(); ++index) {
                    if (index != 0) {
                        errors << "; ";
                    }
                    errors << decode_errors[index];
                }
                reasons.push_back(errors.str());
            } else if (!frames_ok) {
                reasons.push_back("displayed 0 frames");
            }
            if (own_ns == 0) {
                reasons.push_back(own_time_reason(group));
            } else if (!foreign.empty()) {
                reasons.push_back("node also carries " + join_partitions(std::vector<unsigned>(foreign.begin(), foreign.end())));
            }
            for (size_t index = 0; index < reasons.size(); ++index) {
                if (index != 0) {
                    row.reason += "; ";
                }
                row.reason += reasons[index];
            }
            if (!row.pass) {
                block.pass = false;
            }
            block.rows.push_back(row);
        }
        block.decoders = static_cast<int>(block.rows.size());
        std::map<std::string, int> published;
        int absent_ids = 0;
        for (const Slice& slice : card.slices) {
            if (slice.unique_id.empty() || slice.unique_id == "absent") {
                absent_ids += 1;
            } else {
                published[slice.unique_id] += 1;
            }
        }
        const int slice_count = static_cast<int>(card.slices.size());
        if (published.size() == 1 && absent_ids == 0) {
            block.uuid_note = "same UUID " + published.begin()->first + " on " + std::to_string(slice_count) + " partitions";
        } else if (published.size() == 1) {
            block.uuid_note = "unique_id " + published.begin()->first + " on " + std::to_string(published.begin()->second) + "/" +
                              std::to_string(slice_count) + " partitions";
        } else if (published.size() > 1) {
            block.uuid_note = "unique_id differs across partitions";
        } else {
            block.uuid_note = "unique_id absent";
        }
        if (block.rows.empty()) {
            block.pass = false;
        }
        ratios.insert(std::to_string(block.active) + "/" + std::to_string(block.decoders));
        if (!block.pass) {
            every_card = false;
        }
        g_decode.cards.push_back(block);
    }
    g_decode.active_nodes = static_cast<int>(active_nodes.size());
    g_decode.per_card_ratio = ratios.size() == 1 ? *ratios.begin() : "mixed";
    g_decode.decoders_used = 0;
    for (const CardDecode& block : g_decode.cards) {
        g_decode.decoders_used += block.active;
    }
    if (!g_sequence.empty()) {
        g_decode.decoders_expected = static_cast<int>(g_sequence.size());
    } else {
        const int groups = g_xcd_order.empty() ? (required == Layout::Cpx ? 4 : (required == Layout::Dpx ? 2 : 1))
                                               : static_cast<int>(g_xcd_order.size());
        g_decode.decoders_expected = static_cast<int>(cards.size()) * groups;
    }
    g_decode.code = every_card && !g_decode.cards.empty() ? 0 : 1;
}


int run_partition_layout() {
    g_layout.ran = true;
    std::vector<Card> cards;
    const int discovered = discover_or_note(cards, g_layout.skip, g_layout.code);
    if (discovered != 0) {
        return g_layout.code;
    }
    if (g_detected != kRequiredLayout) {
        g_layout.code = 2;
        g_layout.skip = mode_skip_text(g_detected);
        return 2;
    }
    fill_layout(cards);
    return g_layout.code;
}

std::string value_after(const std::string& line, const char* key) {
    const size_t position = line.find(key);
    if (position == std::string::npos) {
        return {};
    }
    size_t start = position + std::char_traits<char>::length(key);
    while (start < line.size() && line[start] == ' ') {
        start += 1;
    }
    size_t end = line.size();
    while (end > start && (line[end - 1] == ' ' || line[end - 1] == '\r')) {
        end -= 1;
    }
    return line.substr(start, end - start);
}

std::string join_unique(const std::set<std::string>& values) {
    std::ostringstream out;
    bool first = true;
    for (const std::string& value : values) {
        if (!first) {
            out << ", ";
        }
        first = false;
        out << value;
    }
    return out.str();
}

std::string amd_smi_bin();

void ensure_gpu_identity() {
    if (!g_market_name.empty() || !g_gfx_target.empty()) {
        return;
    }
    const std::string output = capture_output({amd_smi_bin(), "static", "--asic"});
    std::set<std::string> markets;
    std::set<std::string> targets;
    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        const std::string market = value_after(line, "MARKET_NAME:");
        if (!market.empty() && market != "N/A") {
            markets.insert(market);
        }
        const std::string target = value_after(line, "TARGET_GRAPHICS_VERSION:");
        if (!target.empty() && target != "N/A") {
            targets.insert(target);
        }
    }
    g_market_name = markets.empty() ? "unknown" : join_unique(markets);
    g_gfx_target = targets.empty() ? "unknown" : join_unique(targets);
}

std::string gpu_identity_text() {
    ensure_gpu_identity();
    if (g_market_name == "unknown" && g_gfx_target == "unknown") {
        return "GPU unknown";
    }
    if (g_gfx_target == "unknown") {
        return g_market_name;
    }
    if (g_market_name == "unknown") {
        return g_gfx_target;
    }
    return g_market_name + " " + g_gfx_target;
}

std::string amd_smi_bin() {
    if (const char* env = std::getenv("ROCM_PATH")) {
        const std::string path = std::string(env) + "/bin/amd-smi";
        if (access(path.c_str(), X_OK) == 0) {
            return path;
        }
    }
    if (access("/opt/rocm/bin/amd-smi", X_OK) == 0) {
        return "/opt/rocm/bin/amd-smi";
    }
    return "amd-smi";
}

struct VcnPipe {
    int fd = -1;
    pid_t pid = -1;
};

VcnPipe start_vcn_sample() {
    VcnPipe sample;
    int fds[2] = {-1, -1};
    if (pipe(fds) != 0) {
        return sample;
    }
    const pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return sample;
    }
    if (pid == 0) {
        dup2(fds[1], STDOUT_FILENO);
        close(fds[0]);
        close(fds[1]);
        const std::string bin = amd_smi_bin();
        if (bin.find('/') == std::string::npos) {
            execlp(bin.c_str(), bin.c_str(), "metric", "-u", "--json", static_cast<char*>(nullptr));
        } else {
            execl(bin.c_str(), bin.c_str(), "metric", "-u", "--json", static_cast<char*>(nullptr));
        }
        _exit(127);
    }
    close(fds[1]);
    sample.fd = fds[0];
    sample.pid = pid;
    return sample;
}

std::string finish_vcn_sample(VcnPipe& sample) {
    std::string output;
    if (sample.fd >= 0) {
        char buffer[4096];
        ssize_t count = 0;
        while ((count = read(sample.fd, buffer, sizeof(buffer))) > 0) {
            output.append(buffer, buffer + count);
        }
        close(sample.fd);
        sample.fd = -1;
    }
    if (sample.pid > 0) {
        int status = 0;
        waitpid(sample.pid, &status, 0);
        sample.pid = -1;
    }
    return output;
}

std::map<int, std::vector<int>> parse_vcn_busy(const std::string& json) {
    std::map<int, std::vector<int>> engines;
    size_t cursor = 0;
    while (cursor < json.size()) {
        const size_t gpu_key = json.find("\"gpu\"", cursor);
        if (gpu_key == std::string::npos) {
            break;
        }
        const size_t colon = json.find(':', gpu_key);
        if (colon == std::string::npos) {
            break;
        }
        const int gpu = std::atoi(json.c_str() + colon + 1);
        const size_t next = json.find("\"gpu\"", colon + 1);
        const std::string chunk = json.substr(colon, next == std::string::npos ? std::string::npos : next - colon);
        const size_t busy = chunk.find("\"vcn_busy\"");
        std::vector<int> values;
        if (busy != std::string::npos && chunk.find("\"usage\"") != std::string::npos &&
            chunk.find("\"usage\": \"N/A\"") == std::string::npos && chunk.find("\"usage\":\"N/A\"") == std::string::npos) {
            size_t scan = busy;
            while (values.size() < 8) {
                const size_t value_key = chunk.find("\"value\"", scan);
                if (value_key == std::string::npos) {
                    break;
                }
                const size_t value_colon = chunk.find(':', value_key);
                if (value_colon == std::string::npos) {
                    break;
                }
                values.push_back(std::atoi(chunk.c_str() + value_colon + 1));
                scan = value_key + 7;
            }
        }
        engines[gpu] = values;
        cursor = next == std::string::npos ? json.size() : next;
    }
    return engines;
}

std::string missing_partition_note() {
    std::ostringstream out;
    bool first = true;
    for (const std::string& item : g_layout.missing) {
        const std::string marker = " missing ";
        const size_t position = item.find(marker);
        if (position == std::string::npos) {
            continue;
        }
        if (!first) {
            out << "; ";
        }
        first = false;
        out << item.substr(position + marker.size()) << " missing on " << short_card_key(item.substr(0, position));
    }
    return out.str();
}

int run_placement(Layout required) {
    g_decode.ran = true;
    std::vector<Card> cards;
    const int discovered = discover_or_note(cards, g_decode.skip, g_decode.code);
    if (discovered != 0) {
        return g_decode.code;
    }
    if (g_detected != required) {
        g_decode.code = 2;
        if (!g_allow_partial) {
            g_decode.skip = mode_skip_text(g_detected);
            return 2;
        }
    }
    std::string error;
    std::vector<uint8_t> stream;
    if (!read_file(SAMPLE_STREAM_PATH, stream, error)) {
        g_decode.code = 1;
        g_decode.skip = error;
        return 1;
    }
    if (!g_sequence.empty()) {
        for (const PlayStep& step : g_sequence) {
            for (Card& card : cards) {
                if (card.key.find(step.card) == std::string::npos) {
                    continue;
                }
                for (Slice& slice : card.slices) {
                    if (static_cast<int>(slice.pci.function) == step.function) {
                        slice.scheduled = true;
                    }
                }
            }
        }
    }
    int selected_slices = 0;
    for (const Card& card : cards) {
        for (const Slice& slice : card.slices) {
            if (run_slice(slice)) {
                selected_slices += 1;
            }
        }
    }
    if (selected_slices == 0) {
        g_decode.code = 1;
        g_decode.skip = "requested XCD is not present";
        return 1;
    }
    for (Card& card : cards) {
        for (Slice& slice : card.slices) {
            if (!g_touch_all && !run_slice(slice)) {
                continue;
            }
            touch_device(slice.hip_device, slice.error);
        }
    }
    const std::map<int, uint64_t> before = sum_engine_time(sample_video_fds());
    const auto sample_start = std::chrono::system_clock::now();
    VcnPipe vcn_sample = start_vcn_sample();
    std::vector<std::unique_ptr<StreamDecoder>> held_decoders;
    std::mutex held_mu;
    auto play_one = [&](Slice& slice) {
        if (!slice.error.empty()) {
            return;
        }
        if (slice.hip_device > 255) {
            slice.error = "HIP device index does not fit in the decoder device field";
            return;
        }
        const std::vector<FdEngine> open_before = sample_video_fds();
        const std::map<int, uint64_t> share_before = sum_engine_time(open_before);
        auto decoder = std::make_unique<StreamDecoder>(slice.hip_device);
        if (!decoder->play(stream, slice.displayed_frames, slice.error) && slice.error.empty()) {
            slice.error = "no frame was displayed";
        }
        add_session_activity(slice, open_before, sample_video_fds());
        slice.share_by_minor = engine_time_delta(share_before, sum_engine_time(sample_video_fds()));
        if (g_keep_decoder) {
            std::lock_guard<std::mutex> lock(held_mu);
            held_decoders.push_back(std::move(decoder));
        }
    };
    if (!g_sequence.empty()) {
        const auto started = std::chrono::system_clock::now();
        for (Card& card : cards) {
            card.play_start = started;
            card.played = true;
        }
        for (const PlayStep& step : g_sequence) {
            for (Card& card : cards) {
                if (card.key.find(step.card) == std::string::npos) {
                    continue;
                }
                for (Slice& slice : card.slices) {
                    if (static_cast<int>(slice.pci.function) == step.function) {
                        play_one(slice);
                    }
                }
            }
        }
        const auto ended = std::chrono::system_clock::now();
        for (Card& card : cards) {
            card.play_end = ended;
        }
    } else {
        for (Card& card : cards) {
            card.play_start = std::chrono::system_clock::now();
            card.played = true;
            std::vector<Slice*> play_order;
            if (g_xcd_order.empty()) {
                for (Slice& slice : card.slices) {
                    play_order.push_back(&slice);
                }
            } else {
                for (int xcd : g_xcd_order) {
                    for (Slice& slice : card.slices) {
                        if (static_cast<int>(slice.pci.function) == xcd) {
                            play_order.push_back(&slice);
                        }
                    }
                }
            }
            if (g_parallel && play_order.size() > 1) {
                std::vector<std::thread> threads;
                for (Slice* slice_ptr : play_order) {
                    threads.emplace_back([&, slice_ptr] { play_one(*slice_ptr); });
                }
                for (std::thread& thread : threads) {
                    thread.join();
                }
            } else {
                for (Slice* slice_ptr : play_order) {
                    play_one(*slice_ptr);
                }
            }
            card.play_end = std::chrono::system_clock::now();
        }
    }
    held_decoders.clear();
    g_vcn_by_gpu = parse_vcn_busy(finish_vcn_sample(vcn_sample));
    const std::map<int, uint64_t> after = sum_engine_time(sample_video_fds());
    const auto sample_end = std::chrono::system_clock::now();
    const auto sample_ms = std::chrono::duration_cast<std::chrono::milliseconds>(sample_end - sample_start).count();
    g_decode.sample_window = clock_text(sample_start) + "-" + clock_text(sample_end) + " (" + std::to_string(sample_ms) +
                             " ms), one snapshot before all streams and one after";
    fill_decode(cards, engine_time_delta(before, after), required);
    if (g_decode.split_by_count) {
        g_decode.sample_window += kRequiredLayout == Layout::Dpx
                                      ? "; half time is this half's share of that node, other time is the rest, split by session count"
                                      : "; pair time is this pair's share of that node, other time is the rest, split by session count";
    } else {
        g_decode.sample_window += kRequiredLayout == Layout::Dpx
                                      ? "; half time and other time are per-session counter deltas on the half's nodes"
                                      : "; pair time and other time are per-session counter deltas on the pair's nodes";
    }
    return g_decode.code;
}

int run_decode() { return run_placement(kRequiredLayout); }

std::string plain_verdict(int code) {
    if (code == 0) {
        return "PASS";
    }
    if (code == 2) {
        return "SKIP";
    }
    return "FAIL";
}

void print_header() {
    std::cout << kProgramName << "\n";
    std::cout << "host " << hostname_text() << " | " << timestamp_now() << " | ROCm " << rocm_version_text() << " | amd-smi "
              << amdsmi_version_text() << "\n";
    std::cout << "rocDecode " << rocdecode_version_text() << " | " << gpu_identity_text() << " | amdgpu " << amdgpu_version_text()
              << " | VCN fw " << g_vcn_fw << " | kernel " << kernel_version_text() << " | stream " << g_stream_text << "\n";
    std::ostringstream line;
    const bool skipped = (g_layout.ran && g_layout.code == 2) || (g_decode.ran && g_decode.code == 2);
    if (skipped) {
        line << "requested " << layout_name(kRequiredLayout) << " | host " << host_mode_text();
    } else {
        line << "mode " << host_mode_text();
    }
    line << " | " << g_cards << (g_cards == 1 ? " card" : " cards") << " | ";
    if (g_layout.ran && g_layout.code != 2 && g_layout.expected_gpus > 0) {
        line << g_visible_gpus << " of " << g_layout.expected_gpus << " GPUs visible";
    } else {
        line << g_visible_gpus << (g_visible_gpus == 1 ? " GPU visible" : " GPUs visible");
    }
    line << " | " << xcd_selection_text() << " | 1 stream per GPU, 2 frames";
    std::cout << line.str() << "\n";
}

void print_layout_text() {
    if (!g_layout.ran) {
        return;
    }
    std::cout << "\n=== [" << verdict_word(g_layout.code) << "] partition_layout ===";
    if (!g_layout.expected.empty() && g_layout.code != 2) {
        std::cout << "  expected: " << g_layout.expected;
    }
    std::cout << "\n";
    if (g_layout.code == 2 || g_layout.rows.empty()) {
        std::cout << g_layout.skip << "\n";
        return;
    }
    std::cout << column("card", 12) << column("serial", 14) << column("slices", 8) << column("GPUs", 8)
              << column("partitions", 12) << "result\n";
    for (const LayoutRow& row : g_layout.rows) {
        const std::string word = plain_verdict(row.pass ? 0 : 1);
        std::cout << column(row.card, 12) << column(row.serial, 14) << column(row.slices, 8) << column(row.gpus, 8)
                  << column(row.partitions, 12) << pad_word(word, verdict_word(row.pass ? 0 : 1), 6);
        if (!row.note.empty()) {
            std::cout << row.note;
        }
        std::cout << "\n";
    }
    std::cout << "-> " << g_layout.pass_cards << "/" << g_layout.cards << " cards PASS, " << g_layout.gpus << "/"
              << g_layout.expected_gpus << " GPUs\n";
}

const char* decode_label() { return g_decode_label.empty() ? kDecodeName : g_decode_label.c_str(); }

void print_decode_text() {
    if (!g_decode.ran) {
        return;
    }
    std::cout << "\n=== [" << verdict_word(g_decode.code) << "] " << decode_label() << " ===\n";
    if (g_decode.code == 2 || g_decode.cards.empty()) {
        std::cout << g_decode.skip << "\n";
        return;
    }
    const bool halves = kRequiredLayout == Layout::Dpx;
    const char* group_word = halves ? "halves" : "decoders";
    std::cout << "expected: " << g_decode.rule << "\n";
    std::cout << "engine sample " << g_decode.sample_window << "\n";
    std::cout << "VCN_BUSY is informational; one amd-smi sample over the window, not the engine-time result\n";
    for (const CardDecode& card : g_decode.cards) {
        std::cout << "\ncard " << card.card << "  ->  " << card.active << "/" << card.decoders << " " << group_word << " active";
        if (!card.active_node.empty()) {
            std::cout << ", " << card.sessions << (card.sessions == 1 ? " session on " : " sessions on ")
                      << card.active_node;
            if (!card.active_gpu.empty()) {
                std::cout << " (" << gpu_text(std::atoi(card.active_gpu.c_str())) << ", " << card.active_partition << ")";
            }
        }
        std::cout << "  " << verdict_word(card.pass ? 0 : 1);
        if (!card.uuid_note.empty()) {
            std::cout << "  " << card.uuid_note;
        }
        std::cout << "\n";
        std::cout << column("dec", 5) << column(halves ? "half" : "pair", 8) << column("GPUs", 8)
                  << column("own nodes", 14) << column(halves ? "half time" : "pair time", 12) << column("other time", 12)
                  << column("uuid", 25) << column("VCN_BUSY", 10)
                  << column("sessions ran on", 22) << column("frames", 8) << "result\n";
        for (const DecoderRow& row : card.rows) {
            const std::string word = plain_verdict(row.pass ? 0 : 1);
            std::cout << column(std::to_string(row.index), 5) << column(row.pair, 8) << column(row.gpus, 8)
                      << column(row.own_nodes, 14) << column(format_engine_time(row.pair_ns), 12)
                      << column(format_engine_time(row.other_ns), 12) << column(row.uuid, 25) << column(row.vcn, 10)
                      << column(row.ran_on, 22) << column(row.frames, 8) << pad_word(word, verdict_word(row.pass ? 0 : 1), 6);
            if (!row.pass && !row.reason.empty()) {
                std::cout << row.reason;
            }
            std::cout << "\n";
        }
        if (verbose_output() && !card.window.empty()) {
            std::cout << "sampling window " << card.window << "\n";
            for (const std::string& line : card.verbose) {
                std::cout << "  " << line << "\n";
            }
        }
    }
}

void print_summary_text(int passed, int failed, int skipped) {
    std::cout << "\n=== SUMMARY ===\n";
    if (g_layout.ran) {
        std::cout << column("partition_layout", 26) << pad_word(plain_verdict(g_layout.code), verdict_word(g_layout.code), 6);
        if (g_layout.code == 2) {
            std::cout << g_layout.skip;
        } else if (!g_layout.missing.empty()) {
            std::cout << g_layout.missing.front() << " -> " << g_layout.gpus << "/" << g_layout.expected_gpus << " GPUs";
        } else {
            std::cout << g_layout.gpus << "/" << g_layout.expected_gpus << " GPUs";
        }
        std::cout << "\n";
    }
    if (g_decode.ran) {
        std::cout << column(decode_label(), 26) << pad_word(plain_verdict(g_decode.code), verdict_word(g_decode.code), 6);
        if (g_decode.code == 2) {
            std::cout << g_decode.skip << "\n";
        } else {
            const bool halves = kRequiredLayout == Layout::Dpx;
            std::cout << (halves ? "halves active: " : "decoders used: ") << g_decode.decoders_used << " of "
                      << g_decode.decoders_expected;
            const std::string missing = missing_partition_note();
            if (!missing.empty()) {
                std::cout << " (" << missing << ")";
            }
            std::cout << "; " << g_decode.active_nodes << " active render nodes for " << g_decode.gpus << " GPUs; "
                      << g_decode.per_card_ratio << (halves ? " halves per card\n" : " decoders per card\n");
            std::cout << column("card", 12) << column("active node", 14) << column("GPU", 6) << column("partition", 12)
                      << column("sessions", 10) << (halves ? "halves active\n" : "decoders used\n");
            for (const CardDecode& card : g_decode.cards) {
                std::cout << column(card.card, 12) << column(card.active_node.empty() ? "-" : card.active_node, 14)
                          << column(card.active_gpu.empty() ? "-" : card.active_gpu, 6)
                          << column(card.active_partition.empty() ? "-" : card.active_partition, 12)
                          << column(std::to_string(card.sessions), 10) << card.active << "/" << card.decoders;
                if (card.decoders > 0 && card.active == 1 && card.active_partition.size() > 1 &&
                    card.active_partition != "P0") {
                    std::cout << "  node is " << card.active_partition;
                }
                std::cout << "\n";
            }
        }
    }
    const int code = failed != 0 ? 1 : (passed == 0 && skipped != 0 ? 2 : 0);
    std::cout << "result: " << verdict_word(code) << " (" << passed << " passed, " << failed << " failed, " << skipped
              << " skipped)\n";
}

void print_json() {
    std::ostringstream out;
    out << "{\n";
    out << "  \"program\": \"" << json_escape(kProgramName) << "\",\n";
    out << "  \"host\": \"" << json_escape(hostname_text()) << "\",\n";
    out << "  \"time\": \"" << json_escape(timestamp_now()) << "\",\n";
    out << "  \"rocm\": \"" << json_escape(rocm_version_text()) << "\",\n";
    out << "  \"amd_smi\": \"" << json_escape(amdsmi_version_text()) << "\",\n";
    ensure_gpu_identity();
    out << "  \"rocdecode\": \"" << json_escape(rocdecode_version_text()) << "\",\n";
    out << "  \"market_name\": \"" << json_escape(g_market_name) << "\",\n";
    out << "  \"gfx\": \"" << json_escape(g_gfx_target) << "\",\n";
    out << "  \"amdgpu\": \"" << json_escape(amdgpu_version_text()) << "\",\n";
    out << "  \"vcn_fw\": \"" << json_escape(g_vcn_fw) << "\",\n";
    out << "  \"kernel\": \"" << json_escape(kernel_version_text()) << "\",\n";
    out << "  \"stream\": \"" << json_escape(g_stream_text) << "\",\n";
    out << "  \"compute_partition\": \"" << json_escape(g_compute) << "\",\n";
    out << "  \"memory_partition\": \"" << json_escape(g_memory) << "\",\n";
    out << "  \"cards\": " << g_cards << ",\n";
    out << "  \"gpus_visible\": " << g_visible_gpus << ",\n";
    out << "  \"xcd\": \"" << json_escape(xcd_selection_text()) << "\",\n";
    out << "  \"gpus_expected\": " << g_layout.expected_gpus << ",\n";
    out << "  \"partition_layout\": {\n";
    out << "    \"result\": \"" << plain_verdict(g_layout.code) << "\",\n";
    out << "    \"expected\": \"" << json_escape(g_layout.expected) << "\",\n";
    out << "    \"skip\": \"" << json_escape(g_layout.skip) << "\",\n";
    out << "    \"rows\": [\n";
    for (size_t index = 0; index < g_layout.rows.size(); ++index) {
        const LayoutRow& row = g_layout.rows[index];
        out << "      {\"card\": \"" << json_escape(row.card) << "\", \"serial\": \"" << json_escape(row.serial)
            << "\", \"slices\": \"" << json_escape(row.slices) << "\", \"gpus\": \"" << json_escape(row.gpus)
            << "\", \"partitions\": \"" << json_escape(row.partitions) << "\", \"result\": \""
            << (row.pass ? "PASS" : "FAIL") << "\", \"note\": \"" << json_escape(row.note) << "\"}";
        out << (index + 1 == g_layout.rows.size() ? "\n" : ",\n");
    }
    out << "    ]\n  },\n";
    out << "  \"decode\": {\n";
    out << "    \"name\": \"" << json_escape(decode_label()) << "\",\n";
    out << "    \"result\": \"" << plain_verdict(g_decode.code) << "\",\n";
    out << "    \"rule\": \"" << json_escape(g_decode.rule) << "\",\n";
    out << "    \"sample_window\": \"" << json_escape(g_decode.sample_window) << "\",\n";
    out << "    \"skip\": \"" << json_escape(g_decode.skip) << "\",\n";
    out << "    \"active_nodes\": " << g_decode.active_nodes << ",\n";
    out << "    \"decoders_used\": " << g_decode.decoders_used << ",\n";
    out << "    \"decoders_expected\": " << g_decode.decoders_expected << ",\n";
    out << "    \"missing\": \"" << json_escape(missing_partition_note()) << "\",\n";
    out << "    \"cards\": [\n";
    for (size_t card_index = 0; card_index < g_decode.cards.size(); ++card_index) {
        const CardDecode& card = g_decode.cards[card_index];
        out << "      {\"card\": \"" << json_escape(card.card) << "\", \"active_node\": \"" << json_escape(card.active_node)
            << "\", \"gpu\": \"" << json_escape(card.active_gpu) << "\", \"partition\": \""
            << json_escape(card.active_partition) << "\", \"sessions\": " << card.sessions << ", \"active_decoders\": "
            << card.active << ", \"decoders\": " << card.decoders << ", \"window\": \"" << json_escape(card.window)
            << "\", \"rows\": [\n";
        for (size_t row_index = 0; row_index < card.rows.size(); ++row_index) {
            const DecoderRow& row = card.rows[row_index];
            out << "        {\"decoder\": " << row.index << ", \"pair\": \"" << json_escape(row.pair) << "\", \"gpus\": \""
                << json_escape(row.gpus) << "\", \"own_nodes\": \"" << json_escape(row.own_nodes) << "\", \"pair_ns\": "
                << row.pair_ns << ", \"other_ns\": " << row.other_ns << ", \"uuid\": \"" << json_escape(row.uuid)
                << "\", \"vcn_busy\": \"" << json_escape(row.vcn) << "\", \"ran_on\": \"" << json_escape(row.ran_on)
                << "\", \"frames\": \"" << json_escape(row.frames) << "\", \"result\": \"" << (row.pass ? "PASS" : "FAIL")
                << "\", \"reason\": \"" << json_escape(row.reason) << "\"}";
            out << (row_index + 1 == card.rows.size() ? "\n" : ",\n");
        }
        out << "      ]}";
        out << (card_index + 1 == g_decode.cards.size() ? "\n" : ",\n");
    }
    out << "    ]\n  }\n}\n";
    std::cout << out.str();
}

void print_csv() {
    std::cout << "section,card,serial,slices,gpus,partitions,result,note\n";
    for (const LayoutRow& row : g_layout.rows) {
        std::cout << "layout," << csv_field(row.card) << "," << csv_field(row.serial) << "," << csv_field(row.slices) << ","
                  << csv_field(row.gpus) << "," << csv_field(row.partitions) << "," << (row.pass ? "PASS" : "FAIL") << ","
                  << csv_field(row.note) << "\n";
    }
    std::cout << "section,card,decoder,pair,gpus,own_nodes,pair_ns,other_ns,uuid,vcn_busy,ran_on,frames,result,reason\n";
    for (const CardDecode& card : g_decode.cards) {
        for (const DecoderRow& row : card.rows) {
            std::cout << "decode," << csv_field(card.card) << "," << row.index << "," << csv_field(row.pair) << ","
                      << csv_field(row.gpus) << "," << csv_field(row.own_nodes) << "," << row.pair_ns << "," << row.other_ns
                      << "," << csv_field(row.uuid) << "," << csv_field(row.vcn) << "," << csv_field(row.ran_on) << ","
                      << csv_field(row.frames) << "," << (row.pass ? "PASS" : "FAIL") << "," << csv_field(row.reason) << "\n";
        }
    }
    std::cout << "section,card,active_node,gpu,partition,sessions,decoders_used\n";
    for (const CardDecode& card : g_decode.cards) {
        std::cout << "summary," << csv_field(card.card) << "," << csv_field(card.active_node) << ","
                  << csv_field(card.active_gpu) << "," << csv_field(card.active_partition) << "," << card.sessions << ","
                  << card.active << "/" << card.decoders << "\n";
    }
}

struct SmiRow {
    std::string card;
    int gpu = -1;
    int gfx_activity = -1;
    std::vector<int> busy;
    bool pass = false;
    std::string reason;
};

struct SmiReport {
    bool ran = false;
    int code = 2;
    std::string skip;
    int xcd = 0;
    std::string running_usage;
    std::vector<SmiRow> rows;
};

SmiReport g_smi;

std::string burn_binary_path() {
    char self[4096] = {};
    const ssize_t length = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (length <= 0) {
        return {};
    }
    std::string path(self, static_cast<size_t>(length));
    const size_t slash = path.find_last_of('/');
    if (slash == std::string::npos) {
        return "xcd_burn";
    }
    return path.substr(0, slash) + "/xcd_burn";
}

pid_t spawn_burn(const std::string& binary, int seconds, const std::vector<int>& devices) {
    const pid_t pid = fork();
    if (pid != 0) {
        if (pid > 0) {
            setpgid(pid, pid);
        }
        return pid;
    }
    setpgid(0, 0);
    const int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
        dup2(devnull, STDOUT_FILENO);
        dup2(devnull, STDERR_FILENO);
        close(devnull);
    }
    std::vector<std::string> args;
    args.push_back(binary);
    args.push_back(std::to_string(seconds));
    for (int device : devices) {
        args.push_back(std::to_string(device));
    }
    std::vector<char*> argv;
    for (std::string& arg : args) {
        argv.push_back(arg.data());
    }
    argv.push_back(nullptr);
    execv(argv[0], argv.data());
    _exit(127);
}

std::vector<int> parse_gfx_busy(const std::string& json, int& gfx_activity) {
    gfx_activity = -1;
    const size_t activity = json.find("\"gfx_activity\"");
    if (activity != std::string::npos) {
        const size_t value = json.find("\"value\"", activity);
        if (value != std::string::npos) {
            const size_t colon = json.find(':', value);
            if (colon != std::string::npos) {
                gfx_activity = std::atoi(json.c_str() + colon + 1);
            }
        }
    }
    std::vector<int> slots;
    const size_t busy = json.find("\"gfx_busy_inst\"");
    if (busy == std::string::npos) {
        return slots;
    }
    size_t end = json.find("\"jpeg_busy\"", busy);
    if (end == std::string::npos) {
        end = json.size();
    }
    size_t scan = busy;
    while (slots.size() < 8) {
        const size_t value = json.find("\"value\"", scan);
        if (value == std::string::npos || value >= end) {
            break;
        }
        const size_t colon = json.find(':', value);
        if (colon == std::string::npos || colon >= end) {
            break;
        }
        slots.push_back(std::atoi(json.c_str() + colon + 1));
        scan = value + 7;
    }
    return slots;
}

bool usage_is_na(const std::string& json) {
    return json.find("\"usage\": \"N/A\"") != std::string::npos || json.find("\"usage\":\"N/A\"") != std::string::npos;
}

std::string busy_list(const std::vector<int>& slots) {
    std::ostringstream out;
    out << "[";
    for (size_t index = 0; index < slots.size(); ++index) {
        if (index != 0) {
            out << " ";
        }
        out << slots[index];
    }
    out << "]";
    return out.str();
}

int run_smi_one_xcd() {
    g_smi.ran = true;
    std::vector<Card> cards;
    const int discovered = discover_or_note(cards, g_smi.skip, g_smi.code);
    if (discovered != 0) {
        return g_smi.code;
    }
    if (g_detected != Layout::Cpx || kRequiredLayout != Layout::Cpx) {
        g_smi.code = 2;
        g_smi.skip = "smi_one_xcd checks CPX partitions; host is " + host_mode_text();
        return 2;
    }
    g_smi.xcd = g_xcd_order.empty() ? 0 : g_xcd_order.front();
    const std::string binary = burn_binary_path();
    if (access(binary.c_str(), X_OK) != 0) {
        g_smi.code = 1;
        g_smi.skip = "missing helper " + binary;
        return 1;
    }
    std::vector<int> devices;
    struct Target {
        std::string card;
        int primary = -1;
        int selected = -1;
    };
    std::vector<Target> targets;
    for (const Card& card : cards) {
        Target target;
        target.card = card.key;
        for (const Slice& slice : card.slices) {
            if (slice.pci.function == 0) {
                target.primary = slice.hip_device;
            }
            if (static_cast<int>(slice.pci.function) == g_smi.xcd) {
                target.selected = slice.hip_device;
            }
        }
        if (target.primary < 0 || target.selected < 0) {
            SmiRow row;
            row.card = target.card;
            row.reason = "XCD " + std::to_string(g_smi.xcd) + " or P0 is not present";
            g_smi.rows.push_back(row);
            continue;
        }
        devices.push_back(target.selected);
        targets.push_back(target);
    }
    if (devices.empty()) {
        g_smi.code = 1;
        g_smi.skip = "XCD " + std::to_string(g_smi.xcd) + " is not present";
        return 1;
    }
    const pid_t burn = spawn_burn(binary, 40, devices);
    if (burn <= 0) {
        g_smi.code = 1;
        g_smi.skip = "could not start " + binary;
        return 1;
    }
    sleep(1);
    if (!targets.empty() && targets.front().selected != targets.front().primary) {
        const std::string selected_json =
            capture_output({amd_smi_bin(), "metric", "-u", "-g", std::to_string(targets.front().selected), "--json"});
        g_smi.running_usage = usage_is_na(selected_json) ? "N/A" : "reported";
    } else {
        g_smi.running_usage = "same GPU as P0";
    }
    bool every_card = !g_smi.rows.empty() ? false : !targets.empty();
    for (const SmiRow& row : g_smi.rows) {
        if (!row.pass) {
            every_card = false;
        }
    }
    for (const Target& target : targets) {
        const std::string json =
            capture_output({amd_smi_bin(), "metric", "-u", "-g", std::to_string(target.primary), "--json"});
        SmiRow row;
        row.card = target.card;
        row.gpu = target.primary;
        row.busy = parse_gfx_busy(json, row.gfx_activity);
        if (row.busy.size() != 8) {
            row.reason = "amd-smi did not return 8 gfx_busy_inst values on P0";
        } else if (g_smi.xcd < 0 || g_smi.xcd >= static_cast<int>(row.busy.size())) {
            row.reason = "XCD index is outside gfx_busy_inst";
        } else if (row.busy[static_cast<size_t>(g_smi.xcd)] <= 0) {
            row.reason = "selected XCD showed no gfx_busy_inst activity";
        } else {
            std::vector<unsigned> extra;
            for (size_t index = 0; index < row.busy.size(); ++index) {
                if (static_cast<int>(index) == g_smi.xcd) {
                    continue;
                }
                if (row.busy[index] > 0) {
                    extra.push_back(static_cast<unsigned>(index));
                }
            }
            if (!extra.empty()) {
                row.reason = "activity also on " + join_partitions(extra);
            } else {
                row.pass = true;
            }
        }
        if (!row.pass) {
            every_card = false;
        }
        g_smi.rows.push_back(row);
    }
    kill(-burn, SIGTERM);
    int status = 0;
    waitpid(burn, &status, 0);
    g_smi.code = every_card ? 0 : 1;
    return g_smi.code;
}

struct Subtest {
    const char* name;
    int (*run)();
};

const Subtest kSubtests[] = {
    {"partition_layout", run_partition_layout},
    {kDecodeName, run_decode},
};

const Subtest kOptionalSubtests[] = {
    {"smi_one_xcd", run_smi_one_xcd},
};

#if defined(TEST_MODE_CPX)
struct XcdCase {
    const char* suffix;
    const char* order;
};

const XcdCase kXcdCases[] = {
    {"xcd_all", "all"},
    {"xcd_one", "one"},
    {"xcd_0", "0"},
    {"xcd_1", "1"},
    {"xcd_2", "2"},
    {"xcd_3", "3"},
    {"xcd_4", "4"},
    {"xcd_5", "5"},
    {"xcd_6", "6"},
    {"xcd_7", "7"},
    {"xcd_0_1", "0,1"},
    {"xcd_0_2", "0,2"},
    {"xcd_0_3", "0,3"},
    {"xcd_3_0", "3,0"},
    {"xcd_7_6_5_4_3_2_1_0", "7,6,5,4,3,2,1,0"},
};

int g_xcd_case = -1;

std::string xcd_subtest_name(const XcdCase& item) { return std::string(kProgramName) + "_" + item.suffix; }

void apply_xcd_order(const std::string& order) {
    g_xcd_order.clear();
    if (order == "all") {
        return;
    }
    if (order == "one") {
        g_xcd_order.push_back(0);
        return;
    }
    std::stringstream items(order);
    std::string item;
    while (std::getline(items, item, ',')) {
        g_xcd_order.push_back(item[0] - '0');
    }
}

int run_named_xcd() {
    apply_xcd_order(kXcdCases[g_xcd_case].order);
    g_decode_label = xcd_subtest_name(kXcdCases[g_xcd_case]);
    return run_decode();
}
#endif

void print_smi_text() {
    if (!g_smi.ran) {
        return;
    }
    std::cout << "\n=== [" << verdict_word(g_smi.code) << "] smi_one_xcd ===\n";
    if (g_smi.code == 2 || g_smi.rows.empty()) {
        std::cout << g_smi.skip << "\n";
        return;
    }
    std::cout << "workload: HIP loop on XCD " << g_smi.xcd
              << " of every card. amd-smi metric -u gfx_busy_inst is read from that card's P0.\n";
    std::cout << "expected: only XCD " << g_smi.xcd << " is non-zero. The other seven XCDs stay at 0.\n";
    for (const SmiRow& row : g_smi.rows) {
        std::cout << "card " << row.card << "  P0 GPU " << row.gpu << "  gfx_activity " << row.gfx_activity << "%  busy "
                  << busy_list(row.busy) << "  " << verdict_word(row.pass ? 0 : 1);
        if (!row.pass && !row.reason.empty()) {
            std::cout << "  " << row.reason;
        }
        std::cout << "\n";
    }
    if (g_smi.running_usage == "N/A") {
        std::cout << "amd-smi metric -u on the running XCD itself returns usage N/A. The per-XCD values above come from P0.\n";
    }
    std::cout << "P0 gfx_activity is the card total. A non-zero value there does not mean XCD 0 is busy.\n";
    int passed_cards = 0;
    for (const SmiRow& row : g_smi.rows) {
        if (row.pass) {
            passed_cards += 1;
        }
    }
    std::cout << column("smi_one_xcd", 26) << pad_word(plain_verdict(g_smi.code), verdict_word(g_smi.code), 6) << passed_cards
              << "/" << g_smi.rows.size() << " cards, XCD " << g_smi.xcd << " only\n";
}

void print_usage() {
    std::cout << "usage: " << kProgramName
              << " [--xcd all|one|<list>] [--touch-all] [--keep-decoder] [--parallel]\n"
              << "       [--sequence a,b] [--allow-partial] [--verbose] [--json | --csv]\n"
              << "       [partition_layout | " << kDecodeName << " | smi_one_xcd]\n";
    std::cout << "  --xcd all       decode on every XCD, P0 through P7 (default)\n";
    std::cout << "  --xcd one       decode only XCD 0 on every card\n";
    std::cout << "  --xcd <list>    comma-separated play order, for example 7,6,5,4,3,2,1,0 or 3,0\n";
    std::cout << "  --touch-all     open every partition with HIP before the selected sessions\n";
    std::cout << "  --keep-decoder  leave each decoder open until every selected session has been created\n";
    std::cout << "  --parallel      create the selected sessions on one card at the same time\n";
    std::cout << "  --sequence a,b  play only these card:XCD steps, for example 0c:0,15:3\n";
    std::cout << "  --allow-partial keep decoding when the visible device set is not a full CPX layout\n";
    std::cout << "  smi_one_xcd     one XCD stays busy; amd-smi gfx_busy_inst must be zero on the other XCDs\n";
#if defined(TEST_MODE_CPX)
    std::cout << "  CPX XCD subtests. Each name runs only that decode:\n";
    for (const XcdCase& item : kXcdCases) {
        std::cout << "  " << xcd_subtest_name(item) << "    --xcd " << item.order << "\n";
    }
#endif
}

}  // namespace

int main(int argc, char** argv) {
    const Subtest* selected = nullptr;
    bool saw_json = false;
    bool saw_csv = false;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
        if (arg == "--verbose" || arg == "-v") {
            verbose_output() = true;
            continue;
        }
        if (arg == "--json") {
            saw_json = true;
            continue;
        }
        if (arg == "--csv") {
            saw_csv = true;
            continue;
        }
        if (arg == "--xcd" || arg.compare(0, 6, "--xcd=") == 0) {
            std::string value;
            if (arg == "--xcd") {
                if (index + 1 >= argc) {
                    std::cout << "missing value for --xcd\n";
                    print_usage();
                    return 1;
                }
                value = argv[++index];
            } else {
                value = arg.substr(6);
            }
            g_xcd_order.clear();
            if (value == "all") {
                continue;
            }
            if (value == "one") {
                g_xcd_order.push_back(0);
                continue;
            }
            std::stringstream items(value);
            std::string item;
            bool parsed = !value.empty();
            while (parsed && std::getline(items, item, ',')) {
                if (item.size() != 1 || item[0] < '0' || item[0] > '7') {
                    parsed = false;
                    break;
                }
                g_xcd_order.push_back(item[0] - '0');
            }
            if (!parsed || g_xcd_order.empty()) {
                g_xcd_order.clear();
                std::cout << "xcd must be all, one, an index 0-7, or a comma-separated order\n";
                print_usage();
                return 1;
            }
            continue;
        }
        if (arg == "--touch-all") {
            g_touch_all = true;
            continue;
        }
        if (arg == "--allow-partial") {
            g_allow_partial = true;
            continue;
        }
        if (arg == "--keep-decoder") {
            g_keep_decoder = true;
            continue;
        }
        if (arg == "--parallel") {
            g_parallel = true;
            continue;
        }
        if (arg == "--sequence") {
            if (index + 1 >= argc) {
                std::cout << "missing value for --sequence\n";
                print_usage();
                return 1;
            }
            std::stringstream items(argv[++index]);
            std::string item;
            bool parsed = true;
            while (parsed && std::getline(items, item, ',')) {
                const size_t colon = item.rfind(':');
                if (colon == std::string::npos || colon + 1 >= item.size() || item[colon + 1] < '0' || item[colon + 1] > '7' ||
                    item.size() != colon + 2) {
                    parsed = false;
                    break;
                }
                PlayStep step;
                step.card = item.substr(0, colon);
                step.function = item[colon + 1] - '0';
                if (step.card.empty()) {
                    parsed = false;
                    break;
                }
                g_sequence.push_back(step);
            }
            if (!parsed || g_sequence.empty()) {
                g_sequence.clear();
                std::cout << "sequence must look like 0c:0,15:3\n";
                print_usage();
                return 1;
            }
            continue;
        }
        if (selected != nullptr) {
            print_usage();
            return 1;
        }
        for (const Subtest& subtest : kSubtests) {
            if (arg == subtest.name) {
                selected = &subtest;
                break;
            }
        }
        if (selected == nullptr) {
            for (const Subtest& subtest : kOptionalSubtests) {
                if (arg == subtest.name) {
                    selected = &subtest;
                    break;
                }
            }
        }
#if defined(TEST_MODE_CPX)
        if (selected == nullptr && g_xcd_case < 0) {
            for (size_t case_index = 0; case_index < sizeof(kXcdCases) / sizeof(kXcdCases[0]); ++case_index) {
                if (arg == xcd_subtest_name(kXcdCases[case_index])) {
                    g_xcd_case = static_cast<int>(case_index);
                    break;
                }
            }
        }
#endif
        bool named_xcd = false;
#if defined(TEST_MODE_CPX)
        named_xcd = g_xcd_case >= 0;
#endif
        if (selected == nullptr && !named_xcd) {
            std::cout << "unknown argument: " << arg << "\n";
            print_usage();
            return 1;
        }
    }
    if (saw_json && saw_csv) {
        std::cout << "pass only one of --json or --csv\n";
        return 1;
    }
    g_emit = saw_json ? Emit::Json : (saw_csv ? Emit::Csv : Emit::Text);
    g_color = g_emit == Emit::Text && isatty(STDOUT_FILENO) == 1;

    int passed = 0;
    int failed = 0;
    int skipped = 0;
    const Subtest* subtests = kSubtests;
    size_t subtest_count = sizeof(kSubtests) / sizeof(kSubtests[0]);
    bool optional = false;
    for (const Subtest& subtest : kOptionalSubtests) {
        if (selected == &subtest) {
            optional = true;
            break;
        }
    }
    if (optional) {
        subtests = kOptionalSubtests;
        subtest_count = sizeof(kOptionalSubtests) / sizeof(kOptionalSubtests[0]);
    }
#if defined(TEST_MODE_CPX)
    if (g_xcd_case >= 0) {
        const int status = run_named_xcd();
        if (status == 0) {
            ++passed;
        } else if (status == 2) {
            ++skipped;
        } else {
            ++failed;
        }
    } else
#endif
    for (size_t subtest_index = 0; subtest_index < subtest_count; ++subtest_index) {
        const Subtest& subtest = subtests[subtest_index];
        if (selected != nullptr && &subtest != selected) {
            continue;
        }
        const int status = subtest.run();
        if (status == 0) {
            ++passed;
        } else if (status == 2) {
            ++skipped;
        } else {
            ++failed;
        }
    }
    if (g_emit == Emit::Json) {
        print_json();
    } else if (g_emit == Emit::Csv) {
        print_csv();
    } else {
        print_header();
        print_layout_text();
        print_decode_text();
        print_smi_text();
        print_summary_text(passed, failed, skipped);
    }
    if (failed != 0) {
        return 1;
    }
    if (passed == 0 && skipped != 0) {
        return 2;
    }
    return 0;
}
