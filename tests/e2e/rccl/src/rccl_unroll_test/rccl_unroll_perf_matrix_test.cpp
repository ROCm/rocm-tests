/*
   Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
   SPDX-License-Identifier: MIT
*/

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

int g_passed = 0;
int g_failed = 0;
int g_skipped = 0;

#define CHECK(cond, msg)                                          \
    do {                                                          \
        if (!(cond)) {                                            \
            fprintf(stderr, "  FAIL: %s\n        %s:%d\n",       \
                    msg, __FILE__, __LINE__);                     \
            ++g_failed;                                           \
            return;                                               \
        }                                                         \
    } while (0)

constexpr double kOptimalityTolerance = 0.10; // 10% tolerance
constexpr double kUnroll4PerfTolerance = 0.20; // 20% tolerance for UNROLL=4 target

bool verboseEnabled() {
    const char* v = getenv("RCCL_TEST_VERBOSE");
    return v && std::string(v) == "1";
}

// Single-quote for sh -c / popen; escape embedded single quotes.
std::string shellQuote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'')
            out += "'\\''";
        else
            out += c;
    }
    out += '\'';
    return out;
}

struct PerfSample {
    long long sizeBytes;
    double    timeUs;
};

std::string getAllReducePerfPath() {
    const char* env = getenv("ALL_REDUCE_PERF_PATH");
    if (env) return env;
    const char* candidates[] = {
        "/opt/rocm/bin/all_reduce_perf",
        "/opt/rocm-7.12.0/bin/all_reduce_perf",
    };
    for (auto c : candidates) {
        if (access(c, X_OK) == 0) return c;
    }
    return "";
}

std::string getGpuArch() {
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, 0) != hipSuccess)
        return "";
    std::string arch = props.gcnArchName;
    auto colon = arch.find(':');
    if (colon != std::string::npos)
        arch = arch.substr(0, colon);
    return arch;
}

int getGpuCUCount() {
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, 0) != hipSuccess)
        return -1;
    return props.multiProcessorCount;
}

int expectedUnrollForArch(const std::string& arch, int cuCount) {
    if (arch == "gfx950")  return 1;
    if (arch == "gfx908")  return 2;
    if (arch == "gfx942" && cuCount > 80) return 2;
    return 4;
}

// Run a command and capture combined stdout+stderr (same pattern as
// rccl_dual_kernel_build_test.cpp: `env ... 2>&1` via popen).
int runCommand(const std::string& cmd, std::string& out) {
    out.clear();
    FILE* fp = popen((cmd + " 2>&1").c_str(), "r");
    if (!fp) return -1;
    char buf[4096];
    while (fgets(buf, sizeof(buf), fp))
        out += buf;
    int st = pclose(fp);
    int rc = WEXITSTATUS(st);
    if (verboseEnabled() && !out.empty())
        fwrite(out.data(), 1, out.size(), stderr);
    return rc;
}

// Parse all_reduce_perf output and extract {size, time_us} pairs.
// Parse `tok` as a complete base-10 integer. Returns false unless the entire
// token is consumed, so partial matches like "6e99bc48428f" are rejected.
bool parseWholeLongLong(const std::string& tok, long long& value) {
    if (tok.empty()) return false;
    errno = 0;
    char* end = nullptr;
    value = std::strtoll(tok.c_str(), &end, 10);
    return errno == 0 && end == tok.c_str() + tok.size();
}

// Parse `tok` as a complete floating-point value (accepts scientific notation).
// Returns false unless the entire token is consumed.
bool parseWholeDouble(const std::string& tok, double& value) {
    if (tok.empty()) return false;
    errno = 0;
    char* end = nullptr;
    value = std::strtod(tok.c_str(), &end);
    return errno == 0 && end == tok.c_str() + tok.size();
}

// Data lines look like:
//   "        8             2     float     sum      -1    27.08    0.00 ..."
// Fields: size(0)  count(1)  type(2)  redop(3)  root(4)  time(5)  ...
std::vector<PerfSample> parseAllReduceTimes(const std::string& output) {
    std::vector<PerfSample> samples;
    std::istringstream iss(output);
    std::string line;

    while (std::getline(iss, line)) {
        // Skip blank and comment/header lines.
        size_t first = line.find_first_not_of(' ');
        if (first == std::string::npos) continue;
        if (line[first] == '#') continue;

        // Tokenize.
        std::istringstream ls(line);
        std::string tok;
        std::vector<std::string> fields;
        while (ls >> tok) fields.push_back(tok);
        if (fields.size() < 6) continue;

        // Genuine rccl-tests data rows have an all-numeric byte size (field 0)
        // and a numeric time (field 5). NCCL debug lines can also begin with a
        // digit when the host/container name is hex (e.g. "6e99bc48428f:..."),
        // so validate the whole token instead of trusting the first character —
        // this avoids a std::stod/std::stoll exception aborting the process.
        PerfSample s;
        if (!parseWholeLongLong(fields[0], s.sizeBytes)) continue;
        if (!parseWholeDouble(fields[5], s.timeUs)) continue;
        samples.push_back(s);
    }
    return samples;
}

// Compute the geometric mean of times (robust to outliers at different scales).
double geometricMean(const std::vector<PerfSample>& samples) {
    if (samples.empty()) return 0.0;
    double logSum = 0.0;
    for (auto& s : samples) logSum += std::log(s.timeUs);
    return std::exp(logSum / samples.size());
}

// Human-readable size string.
std::string sizeStr(long long bytes) {
    if (bytes >= 1024LL * 1024 * 1024) {
        char buf[32]; snprintf(buf, sizeof(buf), "%lldG", bytes / (1024LL * 1024 * 1024));
        return buf;
    }
    if (bytes >= 1024 * 1024) {
        char buf[32]; snprintf(buf, sizeof(buf), "%lldM", bytes / (1024 * 1024));
        return buf;
    }
    if (bytes >= 1024) {
        char buf[32]; snprintf(buf, sizeof(buf), "%lldK", bytes / 1024);
        return buf;
    }
    char buf[32]; snprintf(buf, sizeof(buf), "%lldB", bytes);
    return buf;
}

struct UnrollRun {
    int                    envVal;      // RCCL_UNROLL_FACTOR index
    int                    unrollLabel; // actual unroll factor (1, 2, or 4)
    std::string            label;
    std::vector<PerfSample> samples;
    double                 geoMean;
};

constexpr int kBestOfN = 3; // runs per config — keep lowest geo-mean

// Collect benchmark data for all 3 forced unroll runs.
// Each config is run kBestOfN times and the best (lowest geo-mean)
// is kept, so transient noise can only inflate times, never deflate.
// HIP_VISIBLE_DEVICES pins to GPU 0 to avoid multi-GPU contention.
// Returns false if any run fails.
bool collectPerfData(const std::string& perfBin,
                     std::vector<UnrollRun>& runs) {
    // Warm-up on the pinned GPU.
    {
        printf("  Warm-up (GPU 0, 8B–128MB)...\n");
        fflush(stdout);
        std::string dummy;
        std::string warmupCmd =
            "env HIP_VISIBLE_DEVICES=0 " + shellQuote(perfBin) +
            " -b 8 -e 128M -g 1 -f 2";
        runCommand(warmupCmd, dummy);
    }

    struct Config {
        int envVal;       // RCCL_UNROLL_FACTOR value
        int unrollLabel;  // actual unroll factor
        const char* label;
    };
    Config configs[] = {
        { 0, 1, "unroll=1"},
        { 1, 2, "unroll=2"},
        { 2, 4, "unroll=4"},
    };

    for (auto& cfg : configs) {
        UnrollRun best;
        best.envVal = cfg.envVal;
        best.unrollLabel = cfg.unrollLabel;
        best.label = cfg.label;
        best.geoMean = 1e30;

        printf("  Running: %s (best of %d)...\n", cfg.label, kBestOfN);
        fflush(stdout);

        for (int attempt = 0; attempt < kBestOfN; ++attempt) {
            std::string cmd =
                "env HIP_VISIBLE_DEVICES=0 RCCL_UNROLL_FACTOR=" +
                std::to_string(cfg.envVal) + " " +
                shellQuote(perfBin) + " -b 8 -e 128M -g 1 -f 2 -n 20";

            std::string output;
            int rc = runCommand(cmd, output);
            if (rc != 0) {
                fprintf(stderr, "  ERROR: %s attempt %d failed (exit=%d)\n",
                        cfg.label, attempt + 1, rc);
                return false;
            }

            bool hasOobError =
                output.find("Out of bounds values : 0 OK") == std::string::npos
                && output.find("Out of bounds values") != std::string::npos;
            if (hasOobError) {
                fprintf(stderr, "  ERROR: %s attempt %d had out-of-bounds errors\n",
                        cfg.label, attempt + 1);
                return false;
            }

            auto samples = parseAllReduceTimes(output);
            if (samples.empty()) {
                fprintf(stderr, "  ERROR: %s attempt %d produced no results\n",
                        cfg.label, attempt + 1);
                return false;
            }

            double gm = geometricMean(samples);
            printf("    attempt %d: geo-mean %.2f us\n", attempt + 1, gm);

            if (gm < best.geoMean) {
                best.samples = std::move(samples);
                best.geoMean = gm;
            }
        }

        printf("  Best:    %s — %zu sizes, geo-mean time: %.2f us\n",
               best.label.c_str(), best.samples.size(), best.geoMean);

        runs.push_back(std::move(best));
    }
    return true;
}

bool getSharedPerfData(const std::string& perfBin,
                       std::vector<UnrollRun>& runs) {
    static std::vector<UnrollRun> cache;
    static bool populated = false;
    static bool ok = false;
    if (!populated) {
        populated = true;
        ok = collectPerfData(perfBin, cache);
    }
    runs = cache;
    return ok;
}

// ---------------------------------------------------------------
// Test 1: Validate that the default heuristic-selected unroll
//         delivers performance within 10% of the best forced
//         unroll across the full message-size range.
// ---------------------------------------------------------------
void test_DefaultUnrollIsOptimal() {
    printf("[RUN ] DefaultUnrollIsOptimal\n");

    std::string perfBin = getAllReducePerfPath();
    if (perfBin.empty()) {
        printf("[SKIP] DefaultUnrollIsOptimal — all_reduce_perf not found\n");
        ++g_skipped;
        return;
    }

    std::string arch = getGpuArch();
    int cuCount = getGpuCUCount();
    if (arch.empty()) {
        printf("[SKIP] DefaultUnrollIsOptimal — no GPU detected\n");
        ++g_skipped;
        return;
    }

    int expectedUnroll = expectedUnrollForArch(arch, cuCount);
    printf("  GPU: %s (%d CUs) — heuristic should select unroll=%d\n",
           arch.c_str(), cuCount, expectedUnroll);

    std::vector<UnrollRun> runs;
    CHECK(getSharedPerfData(perfBin, runs),
          "Failed to collect performance data for one or more unroll configurations");
    CHECK(runs.size() == 3, "Expected 3 runs (unroll=1, 2, 4)");

    // Identify which forced unroll was fastest overall.
    double bestGeoMean = runs[0].geoMean;
    int bestIdx = 0;
    for (size_t i = 1; i < runs.size(); ++i) {
        if (runs[i].geoMean < bestGeoMean) {
            bestGeoMean = runs[i].geoMean;
            bestIdx = i;
        }
    }

    // Find the run that matches the heuristic's selection.
    int heuristicIdx = -1;
    for (size_t i = 0; i < runs.size(); ++i) {
        if (runs[i].unrollLabel == expectedUnroll) {
            heuristicIdx = i;
            break;
        }
    }
    CHECK(heuristicIdx >= 0, "Heuristic-selected unroll not found in benchmark runs");

    double heuristicGeoMean = runs[heuristicIdx].geoMean;
    double ratio = heuristicGeoMean / bestGeoMean;

    printf("\n  === Performance Summary (geometric mean time, lower is better) ===\n");
    printf("  %-12s  %10s  %10s\n", "Config", "GeoMean(us)", "vs Best");
    printf("  %-12s  %10s  %10s\n", "------", "-----------", "-------");
    for (size_t i = 0; i < runs.size(); ++i) {
        double vsRatio = runs[i].geoMean / bestGeoMean;
        std::string marker;
        if (static_cast<int>(i) == heuristicIdx)
            marker += " <-- heuristic";
        if (static_cast<int>(i) == bestIdx)
            marker += " <-- best";
        printf("  %-12s  %10.2f  %9.2fx%s\n",
               runs[i].label.c_str(), runs[i].geoMean, vsRatio, marker.c_str());
    }

    printf("\n  Heuristic selects: unroll=%d (%.2f us)\n",
           expectedUnroll, heuristicGeoMean);
    printf("  Best measured:     unroll=%d (%.2f us)\n",
           runs[bestIdx].unrollLabel, bestGeoMean);
    printf("  Ratio:             %.2fx (tolerance: %.0f%%)\n",
           ratio, kOptimalityTolerance * 100);

    CHECK(ratio <= 1.0 + kOptimalityTolerance,
          (std::string("Heuristic-selected unroll=") +
           std::to_string(expectedUnroll) + " is " +
           std::to_string((int)((ratio - 1.0) * 100)) +
           "% slower than best (unroll=" +
           std::to_string(runs[bestIdx].unrollLabel) +
           ") — exceeds " +
           std::to_string((int)(kOptimalityTolerance * 100)) +
           "% tolerance").c_str());

    printf("[PASS] DefaultUnrollIsOptimal\n");
    ++g_passed;
}

// ---------------------------------------------------------------
// Test 2: Print a per-size performance matrix and identify which
//         unroll "wins" at each message size.
//         Informational — always passes.
// ---------------------------------------------------------------
void test_UnrollSensitivityProfile() {
    printf("[RUN ] UnrollSensitivityProfile\n");

    std::string perfBin = getAllReducePerfPath();
    if (perfBin.empty()) {
        printf("[SKIP] UnrollSensitivityProfile — all_reduce_perf not found\n");
        ++g_skipped;
        return;
    }

    std::string arch = getGpuArch();
    if (arch.empty()) {
        printf("[SKIP] UnrollSensitivityProfile — no GPU detected\n");
        ++g_skipped;
        return;
    }

    std::vector<UnrollRun> runs;
    CHECK(getSharedPerfData(perfBin, runs),
          "Failed to collect performance data");

    // All runs should have the same number of samples.
    size_t nSizes = runs[0].samples.size();
    for (size_t i = 1; i < runs.size(); ++i) {
        if (runs[i].samples.size() != nSizes) {
            printf("  WARN: run '%s' has %zu sizes vs %zu — table may be ragged\n",
                   runs[i].label.c_str(), runs[i].samples.size(), nSizes);
            nSizes = std::min(nSizes, runs[i].samples.size());
        }
    }

    // Print the matrix header.
    printf("\n  === Per-Size Performance Matrix (time in us, lower is better) ===\n");
    printf("  %8s", "Size");
    for (auto& r : runs)
        printf("  %10s", r.label.c_str());
    printf("  %10s\n", "Winner");

    printf("  %8s", "----");
    for (size_t i = 0; i < runs.size(); ++i)
        printf("  %10s", "----------");
    printf("  %10s\n", "------");

    int winCounts[3] = {};

    for (size_t row = 0; row < nSizes; ++row) {
        long long sz = runs[0].samples[row].sizeBytes;
        printf("  %8s", sizeStr(sz).c_str());

        double bestTime = runs[0].samples[row].timeUs;
        int bestIdx = 0;
        for (size_t c = 1; c < runs.size(); ++c) {
            double t = runs[c].samples[row].timeUs;
            if (t < bestTime) {
                bestTime = t;
                bestIdx = c;
            }
        }

        for (size_t c = 0; c < runs.size(); ++c) {
            double t = runs[c].samples[row].timeUs;
            if (static_cast<int>(c) == bestIdx)
                printf("  %9.2f*", t);
            else
                printf("  %10.2f", t);
        }
        printf("  %10s\n", runs[bestIdx].label.c_str());
        winCounts[bestIdx]++;
    }

    printf("\n  === Win Count Summary ===\n");
    for (size_t i = 0; i < runs.size(); ++i)
        printf("  %-12s won at %d / %zu sizes\n",
               runs[i].label.c_str(), winCounts[i], nSizes);

    printf("[PASS] UnrollSensitivityProfile\n");
    ++g_passed;
}

// ---------------------------------------------------------------
// Test 3: Performance increases with higher unroll (up to a
//         point). For architectures where expected unroll > 1,
//         the optimal unroll must outperform unroll=1.
// ---------------------------------------------------------------
// void test_UnrollScalingBenefit() {
//     printf("[RUN ] UnrollScalingBenefit\n");

//     std::string perfBin = getAllReducePerfPath();
//     if (perfBin.empty()) {
//         printf("[SKIP] UnrollScalingBenefit — all_reduce_perf not found\n");
//         ++g_skipped;
//         return;
//     }

//     std::string arch = getGpuArch();
//     int cuCount = getGpuCUCount();
//     if (arch.empty()) {
//         printf("[SKIP] UnrollScalingBenefit — no GPU detected\n");
//         ++g_skipped;
//         return;
//     }

//     int expected = expectedUnrollForArch(arch, cuCount);

//     std::vector<UnrollRun> runs;
//     CHECK(getSharedPerfData(perfBin, runs),
//           "Failed to collect performance data");
//     CHECK(runs.size() == 3, "Expected 3 runs");

//     double t1 = runs[0].geoMean;

//     double bestGeoMean = runs[0].geoMean;
//     int    bestUnroll   = runs[0].unrollLabel;
//     for (size_t i = 1; i < runs.size(); ++i) {
//         if (runs[i].geoMean < bestGeoMean) {
//             bestGeoMean = runs[i].geoMean;
//             bestUnroll  = runs[i].unrollLabel;
//         }
//     }

//     printf("  unroll=1: %.2f us, unroll=2: %.2f us, unroll=4: %.2f us\n",
//            runs[0].geoMean, runs[1].geoMean, runs[2].geoMean);
//     printf("  Best: unroll=%d (%.2f us)\n", bestUnroll, bestGeoMean);

//     if (expected > 1) {
//         double ratio = bestGeoMean / t1;
//         printf("  Scaling: best/unroll=1 = %.2fx (< 1.0 means improvement)\n", ratio);
//         CHECK(ratio < 1.0,
//               "No performance benefit from higher unroll — expected "
//               "improvement for this GPU architecture");
//     } else {
//         double ratio = t1 / bestGeoMean;
//         printf("  Architecture %s: unroll=1 expected optimal, vs best: %.2fx\n",
//                arch.c_str(), ratio);
//         CHECK(ratio <= 1.0 + kOptimalityTolerance,
//               "unroll=1 is expected optimal but underperformed significantly");
//     }

//     printf("[PASS] UnrollScalingBenefit\n");
//     ++g_passed;
// }

// ---------------------------------------------------------------
// Test 4: UNROLL=4 can run the full message range (8B–128MB)
//         without failures, confirming no scratch reclaim issues.
// ---------------------------------------------------------------
void test_Unroll4NoScratchReclaim() {
    printf("[RUN ] Unroll4NoScratchReclaim\n");

    std::string perfBin = getAllReducePerfPath();
    if (perfBin.empty()) {
        printf("[SKIP] Unroll4NoScratchReclaim — all_reduce_perf not found\n");
        ++g_skipped;
        return;
    }

    std::string arch = getGpuArch();
    if (arch.empty()) {
        printf("[SKIP] Unroll4NoScratchReclaim — no GPU detected\n");
        ++g_skipped;
        return;
    }

    std::vector<UnrollRun> runs;
    CHECK(getSharedPerfData(perfBin, runs),
          "UNROLL=4 run failed — possible scratch reclaim issue");
    CHECK(runs.size() == 3, "Expected 3 runs");
    CHECK(!runs[2].samples.empty(),
          "UNROLL=4 produced no results — possible scratch reclaim issue");

    printf("  UNROLL=4 completed %zu message sizes (8B–128MB) without errors\n",
           runs[2].samples.size());
    printf("  Geo-mean latency: %.2f us\n", runs[2].geoMean);
    printf("[PASS] Unroll4NoScratchReclaim\n");
    ++g_passed;
}

// ---------------------------------------------------------------
// Test 5: UNROLL=4 achieves target performance — its geo-mean
//         latency must be within 20% of the best unroll factor.
// ---------------------------------------------------------------
void test_Unroll4TargetPerformance() {
    printf("[RUN ] Unroll4TargetPerformance\n");

    std::string perfBin = getAllReducePerfPath();
    if (perfBin.empty()) {
        printf("[SKIP] Unroll4TargetPerformance — all_reduce_perf not found\n");
        ++g_skipped;
        return;
    }

    std::string arch = getGpuArch();
    if (arch.empty()) {
        printf("[SKIP] Unroll4TargetPerformance — no GPU detected\n");
        ++g_skipped;
        return;
    }

    std::vector<UnrollRun> runs;
    CHECK(getSharedPerfData(perfBin, runs),
          "Failed to collect performance data");
    CHECK(runs.size() == 3, "Expected 3 runs");

    double bestGeoMean = runs[0].geoMean;
    int    bestUnroll  = runs[0].unrollLabel;
    for (size_t i = 1; i < runs.size(); ++i) {
        if (runs[i].geoMean < bestGeoMean) {
            bestGeoMean = runs[i].geoMean;
            bestUnroll  = runs[i].unrollLabel;
        }
    }

    double unroll4GeoMean = runs[2].geoMean;
    double ratio = unroll4GeoMean / bestGeoMean;

    printf("  UNROLL=4: %.2f us\n", unroll4GeoMean);
    printf("  Best:     %.2f us (unroll=%d)\n", bestGeoMean, bestUnroll);
    printf("  Ratio:    %.2fx (tolerance: %.0f%%)\n",
           ratio, kUnroll4PerfTolerance * 100);

    CHECK(ratio <= 1.0 + kUnroll4PerfTolerance,
          (std::string("UNROLL=4 is ") +
           std::to_string((int)((ratio - 1.0) * 100)) +
           "% slower than best (unroll=" +
           std::to_string(bestUnroll) +
           ") — exceeds " +
           std::to_string((int)(kUnroll4PerfTolerance * 100)) +
           "% tolerance").c_str());

    printf("[PASS] Unroll4TargetPerformance\n");
    ++g_passed;
}

} // namespace

struct TestEntry {
    const char* name;
    void (*func)();
    const char* description;
};

const TestEntry kTests[] = {
    {"DefaultUnrollIsOptimal",   test_DefaultUnrollIsOptimal,   "Heuristic within 10% of best unroll"},
    {"UnrollSensitivityProfile", test_UnrollSensitivityProfile, "Per-size perf matrix (informational)"},
    // {"UnrollScalingBenefit",     test_UnrollScalingBenefit,     "Higher unroll improves perf"},
    {"Unroll4NoScratchReclaim",  test_Unroll4NoScratchReclaim,  "Unroll=4 full range without errors"},
    {"Unroll4TargetPerformance", test_Unroll4TargetPerformance, "Unroll=4 within 20% of best"},
};
constexpr int kNumTests = sizeof(kTests) / sizeof(kTests[0]);

void printUsage(const char* prog) {
    printf("Usage: %s [OPTIONS]\n\n", prog);
    printf("Run RCCL unroll-factor performance sub-tests.\n");
    printf("With no flags, all sub-tests are executed.\n\n");
    printf("Options:\n");
    for (int i = 0; i < kNumTests; ++i)
        printf("  --%-28s  %s\n", kTests[i].name, kTests[i].description);
    printf("  --%-28s  Run all sub-tests (default)\n", "all");
    printf("  --%-28s  List available sub-tests\n", "list");
    printf("  --%-28s  Show this help message\n", "help");
}

void printTestList() {
    printf("Available sub-tests:\n");
    for (int i = 0; i < kNumTests; ++i)
        printf("  %-28s  %s\n", kTests[i].name, kTests[i].description);
}

int main(int argc, char* argv[]) {
    bool runTest[kNumTests] = {};
    bool anySelected = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        }
        if (arg == "--list") {
            printTestList();
            return EXIT_SUCCESS;
        }
        if (arg == "--all") {
            for (int j = 0; j < kNumTests; ++j) runTest[j] = true;
            anySelected = true;
            continue;
        }
        if (arg.substr(0, 2) != "--") {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
        std::string testName = arg.substr(2);
        bool found = false;
        for (int j = 0; j < kNumTests; ++j) {
            if (testName == kTests[j].name) {
                runTest[j] = true;
                anySelected = true;
                found = true;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "Unknown test: %s\n", testName.c_str());
            printTestList();
            return EXIT_FAILURE;
        }
    }

    if (!anySelected)
        for (int j = 0; j < kNumTests; ++j) runTest[j] = true;

    printf("=== Unroll Factor Performance Matrix ===\n");
    printf("=== all_reduce_perf: %s\n",
           getAllReducePerfPath().empty() ? "(not found)" : getAllReducePerfPath().c_str());

    std::string arch = getGpuArch();
    int cuCount = getGpuCUCount();
    if (!arch.empty())
        printf("=== GPU: %s (%d CUs)\n", arch.c_str(), cuCount);
    printf("\n");

    for (int i = 0; i < kNumTests; ++i)
        if (runTest[i]) kTests[i].func();

    printf("\n=== Results: %d passed, %d failed, %d skipped ===\n",
           g_passed, g_failed, g_skipped);
    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
