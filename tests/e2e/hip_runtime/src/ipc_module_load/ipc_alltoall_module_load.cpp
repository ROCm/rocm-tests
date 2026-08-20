/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
**
** Regression test for distributed-style jobs that share GPU buffers across
** ranks via HIP IPC and then load code objects with hipModuleLoad.
*/
#include <hip/hip_runtime.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <regex>
#include <string>
#include <vector>

#define HIP_OK_R(rank, x)                                                    \
    do {                                                                     \
        hipError_t _e = (x);                                                 \
        if (_e != hipSuccess) {                                              \
            fprintf(stderr, "[rank %d] HIP error %d at %s:%d (%s)\n", rank,  \
                    _e, __FILE__, __LINE__, hipGetErrorString(_e));          \
            std::_Exit(10 + rank);                                           \
        }                                                                    \
    } while (0)

namespace {

constexpr size_t IPC_BYTES               = 64ull << 20;
constexpr int    DUP_IMPORTS_PER_PEER    = 2;
constexpr int    LOAD_CYCLES             = 50;
constexpr int    LOAD_LATENCY_BUDGET_MS  = 5000;
constexpr int    BARRIER_TIMEOUT_SECS    = 60;
constexpr int    BARRIER_POLL_USECS      = 20000;
constexpr const char* CODE_OBJECT        = "noop.hsaco";

// Env-var protocol for the re-exec'd child. Children use fork+exec (not
// bare fork) so each rank gets a pristine HIP runtime.
constexpr const char* ENV_MODE        = "IPC_AA_MODE";        // "child"
constexpr const char* ENV_RANK        = "IPC_AA_RANK";
constexpr const char* ENV_N_RANKS     = "IPC_AA_N_RANKS";
constexpr const char* ENV_N_GPUS      = "IPC_AA_N_GPUS";
constexpr const char* ENV_BARRIER_DIR = "IPC_AA_BARRIER_DIR";

bool wait_for_file(const std::string& path, int timeout_secs) {
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(timeout_secs);
    while (access(path.c_str(), R_OK) != 0) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        usleep(BARRIER_POLL_USECS);
    }
    return true;
}

// Diagnostic-only debugfs helper. Returns -1 if debugfs is not readable.
int count_bos_of_size(int drm_minor, size_t bytes) {
    char path[128];
    std::snprintf(path, sizeof(path),
                  "/sys/kernel/debug/dri/%d/amdgpu_gem_info", drm_minor);
    std::ifstream f(path);
    if (!f) return -1;
    std::string line;
    int hits = 0;
    std::regex re(R"((\d+)\s+(\d+)kB)");
    while (std::getline(f, line)) {
        std::smatch m;
        if (std::regex_search(line, m, re)) {
            size_t sz = std::stoull(m[2]) * 1024ull;
            if (sz == bytes) ++hits;
        }
    }
    return hits;
}

// Publish a handle atomically: write to <path>.tmp, then rename into
// place. Readers using access(path, R_OK) as a barrier will only ever
// see the fully-written file, never a partial mid-write zero-length
// snapshot. Returns true on success.
bool publish_handle_atomic(const std::string& path,
                           const hipIpcMemHandle_t& h, int rank) {
    std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[rank %d] publish open(%s) failed: %s\n",
                rank, tmp.c_str(), std::strerror(errno));
        return false;
    }
    bool wrote = std::fwrite(&h, sizeof(h), 1, f) == 1;
    int  flushed = std::fflush(f);
    int  fd = fileno(f);
    int  synced = (fd >= 0) ? fsync(fd) : -1;
    std::fclose(f);
    if (!wrote || flushed != 0 || synced != 0) {
        fprintf(stderr,
                "[rank %d] publish write/flush failed (wrote=%d flush=%d "
                "sync=%d): %s\n",
                rank, wrote, flushed, synced, std::strerror(errno));
        unlink(tmp.c_str());
        return false;
    }
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        fprintf(stderr, "[rank %d] publish rename failed: %s\n",
                rank, std::strerror(errno));
        unlink(tmp.c_str());
        return false;
    }
    return true;
}

[[noreturn]] void rank_main(int rank, int n_ranks, int n_gpus,
                            const std::string& barrier_dir) {
    // If the parent dies (watchdog SIGKILL, OOM, manual kill), the
    // kernel SIGTERMs this rank instead of leaving it spinning at
    // barrier-2 forever and littering /tmp.
    (void)prctl(PR_SET_PDEATHSIG, SIGTERM);

    int gpu = rank % n_gpus;
    HIP_OK_R(rank, hipSetDevice(gpu));

    void* my_bo = nullptr;
    HIP_OK_R(rank, hipMalloc(&my_bo, IPC_BYTES));
    HIP_OK_R(rank, hipMemset(my_bo, 0xA0 + (rank & 0x1F), IPC_BYTES));

    hipIpcMemHandle_t my_handle{};
    HIP_OK_R(rank, hipIpcGetMemHandle(&my_handle, my_bo));

    std::string my_handle_path =
        barrier_dir + "/handle_" + std::to_string(rank);
    if (!publish_handle_atomic(my_handle_path, my_handle, rank)) {
        std::_Exit(20 + rank);
    }
    fprintf(stderr, "RANK %d PUBLISH ok\n", rank);

    for (int peer = 0; peer < n_ranks; ++peer) {
        std::string path = barrier_dir + "/handle_" + std::to_string(peer);
        if (!wait_for_file(path, BARRIER_TIMEOUT_SECS)) {
            fprintf(stderr,
                    "[rank %d] barrier-1 timeout waiting for rank %d\n",
                    rank, peer);
            std::_Exit(30 + rank);
        }
    }

    std::vector<int> peer_order;
    peer_order.reserve(n_ranks - 1);
    for (int p = 0; p < n_ranks; ++p) {
        if (p != rank) peer_order.push_back(p);
    }
    std::mt19937 rng(0xC0FFEE ^ static_cast<unsigned>(rank));
    std::shuffle(peer_order.begin(), peer_order.end(), rng);

    {
        std::string s = "RANK " + std::to_string(rank) + " IMPORT_ORDER";
        for (int p : peer_order) s += " " + std::to_string(p);
        fprintf(stderr, "%s\n", s.c_str());
    }

    std::vector<std::vector<void*>> remote_views(n_ranks);
    int imports_done = 0;
    int peers_skipped = 0;

    for (int peer : peer_order) {
        std::string path = barrier_dir + "/handle_" + std::to_string(peer);
        hipIpcMemHandle_t rh{};
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f || std::fread(&rh, sizeof(rh), 1, f) != 1) {
            fprintf(stderr, "[rank %d] read peer %d handle failed: %s\n",
                    rank, peer, std::strerror(errno));
            if (f) std::fclose(f);
            std::_Exit(40 + rank);
        }
        std::fclose(f);

        int peer_gpu = peer % n_gpus;
        if (peer_gpu != gpu) {
            int can = 0;
            HIP_OK_R(rank, hipDeviceCanAccessPeer(&can, gpu, peer_gpu));
            if (!can) {
                ++peers_skipped;
                continue;
            }
            hipError_t pe = hipDeviceEnablePeerAccess(peer_gpu, 0);
            if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled) {
                fprintf(stderr,
                        "[rank %d] enable peer %d->%d failed: %s\n", rank,
                        gpu, peer_gpu, hipGetErrorString(pe));
                std::_Exit(50 + rank);
            }
        }

        for (int k = 0; k < DUP_IMPORTS_PER_PEER; ++k) {
            void* v = nullptr;
            HIP_OK_R(rank,
                     hipIpcOpenMemHandle(&v, rh,
                                         hipIpcMemLazyEnablePeerAccess));
            remote_views[peer].push_back(v);
            ++imports_done;
        }
    }

    for (int peer = 0; peer < n_ranks; ++peer) {
        for (void* v : remote_views[peer]) {
            HIP_OK_R(rank, hipMemsetAsync(v, 0xCD, IPC_BYTES, 0));
        }
    }
    HIP_OK_R(rank, hipDeviceSynchronize());

    fprintf(stderr,
            "RANK %d IMPORT done=%d skipped_peers=%d (dup=%d per peer)\n",
            rank, imports_done, peers_skipped, DUP_IMPORTS_PER_PEER);

    {
        std::string done = barrier_dir + "/done_" + std::to_string(rank);
        FILE* f = std::fopen(done.c_str(), "w");
        if (f) std::fclose(f);
    }
    for (int peer = 0; peer < n_ranks; ++peer) {
        std::string path = barrier_dir + "/done_" + std::to_string(peer);
        if (!wait_for_file(path, BARRIER_TIMEOUT_SECS)) {
            fprintf(stderr,
                    "[rank %d] barrier-2 timeout waiting for rank %d\n",
                    rank, peer);
            std::_Exit(60 + rank);
        }
    }

    long long worst_ms = 0;
    bool hung = false;
    int last_cycle = 0;
    for (int i = 0; i < LOAD_CYCLES; ++i) {
        last_cycle = i;
        hipModule_t mod = nullptr;
        auto t0 = std::chrono::steady_clock::now();
        hipError_t e = hipModuleLoad(&mod, CODE_OBJECT);
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
        if (dt > worst_ms) worst_ms = dt;
        if (e != hipSuccess) {
            fprintf(stderr, "[rank %d] hipModuleLoad cycle %d err %d (%s)\n",
                    rank, i, e, hipGetErrorString(e));
            std::_Exit(70 + rank);
        }
        if (dt > LOAD_LATENCY_BUDGET_MS) {
            hung = true;
            // Don't unload here; the module may be in a weird state and
            // the verdict is already decided. _Exit below will reap the
            // process. The module BO leak is harmless.
            break;
        }
        // Soft unload: the verdict for this cycle is already "in
        // budget"; if unload hiccups (peer already gone, etc.), that
        // must not flip a real PASS to FAIL.
        (void)hipModuleUnload(mod);
    }

    fprintf(stderr, "RANK %d %s worst=%lldms cycles=%d\n", rank,
            hung ? "HANG" : "PASS", worst_ms,
            hung ? (last_cycle + 1) : LOAD_CYCLES);

    // Best-effort cleanup. The verdict has been decided. A teardown
    // hiccup must not flip a real PASS into an other_fail at the
    // parent.
    for (int peer = 0; peer < n_ranks; ++peer) {
        for (void* v : remote_views[peer]) {
            (void)hipIpcCloseMemHandle(v);
        }
    }
    (void)hipFree(my_bo);
    std::_Exit(hung ? (80 + rank) : 0);
}

int run_child() {
    const char* rank_s     = std::getenv(ENV_RANK);
    const char* n_ranks_s  = std::getenv(ENV_N_RANKS);
    const char* n_gpus_s   = std::getenv(ENV_N_GPUS);
    const char* barrier_s  = std::getenv(ENV_BARRIER_DIR);
    if (!rank_s || !n_ranks_s || !n_gpus_s || !barrier_s) {
        fprintf(stderr,
                "[pid %d] child mode missing required env "
                "(rank=%s n_ranks=%s n_gpus=%s barrier_dir=%s)\n",
                getpid(),
                rank_s     ? rank_s     : "(null)",
                n_ranks_s  ? n_ranks_s  : "(null)",
                n_gpus_s   ? n_gpus_s   : "(null)",
                barrier_s  ? barrier_s  : "(null)");
        return 1;
    }
    int rank    = std::atoi(rank_s);
    int n_ranks = std::atoi(n_ranks_s);
    int n_gpus  = std::atoi(n_gpus_s);
    rank_main(rank, n_ranks, n_gpus, std::string(barrier_s));
    // unreachable
}

int run_parent(const char* self_path) {
    int n_gpus = 0;
    if (hipGetDeviceCount(&n_gpus) != hipSuccess || n_gpus < 2) {
        fprintf(stderr,
                "this test requires at least 2 GPUs (got %d) -- skipping\n",
                n_gpus);
        return 77;
    }
    int n_ranks = n_gpus;

    char tmpl[] = "/tmp/ipc_alltoall_module_load-XXXXXX";
    if (mkdtemp(tmpl) == nullptr) {
        perror("mkdtemp");
        return 1;
    }
    std::string barrier_dir(tmpl);

    fprintf(stderr,
            "RUN n_ranks=%d n_gpus=%d dup_imports_per_peer=%d "
            "load_cycles=%d barrier=%s self_path=%s\n",
            n_ranks, n_gpus, DUP_IMPORTS_PER_PEER, LOAD_CYCLES,
            barrier_dir.c_str(), self_path);

    int bos_before = count_bos_of_size(0, IPC_BYTES);

    std::vector<pid_t> pids;
    pids.reserve(n_ranks);
    for (int r = 0; r < n_ranks; ++r) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return 1;
        }
        if (pid == 0) {
            // Auto-cleanup: survives exec, ensures children don't
            // outlive the parent on watchdog kill / OOM / crash.
            (void)prctl(PR_SET_PDEATHSIG, SIGTERM);

            char rank_buf[16], nr_buf[16], ng_buf[16];
            std::snprintf(rank_buf, sizeof(rank_buf), "%d", r);
            std::snprintf(nr_buf,   sizeof(nr_buf),   "%d", n_ranks);
            std::snprintf(ng_buf,   sizeof(ng_buf),   "%d", n_gpus);
            setenv(ENV_MODE,        "child",              1);
            setenv(ENV_RANK,        rank_buf,             1);
            setenv(ENV_N_RANKS,     nr_buf,               1);
            setenv(ENV_N_GPUS,      ng_buf,               1);
            setenv(ENV_BARRIER_DIR, barrier_dir.c_str(),  1);

            execlp(self_path, self_path, static_cast<char*>(nullptr));
            // exec failed -- can't barrier, can't IPC. Exit with a code
            // the parent will surface as other_fail (outside the HANG
            // range), so the operator sees the run as FAIL.
            fprintf(stderr,
                    "[rank %d] execlp(%s) failed: %s\n",
                    r, self_path, std::strerror(errno));
            std::_Exit(90 + r);
        }
        pids.push_back(pid);
    }

    int n_pass = 0;
    int n_hang = 0;
    int n_other = 0;
    for (pid_t pid : pids) {
        int st = 0;
        waitpid(pid, &st, 0);
        int code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
        if (code == 0) ++n_pass;
        else if (code >= 80 && code < 80 + n_ranks) ++n_hang;
        else ++n_other;
    }

    int bos_after = count_bos_of_size(0, IPC_BYTES);

    for (int r = 0; r < n_ranks; ++r) {
        unlink((barrier_dir + "/handle_" + std::to_string(r)).c_str());
        unlink((barrier_dir + "/handle_" + std::to_string(r) + ".tmp")
                   .c_str());
        unlink((barrier_dir + "/done_" + std::to_string(r)).c_str());
    }
    rmdir(barrier_dir.c_str());

    // Diagnostic only -- not pass/fail. The fix tolerates duplicates
    // rather than deduping them, so any non-trivial delta is consistent
    // with the imports having landed; we just print what we saw.
    if (bos_before < 0 || bos_after < 0) {
        fprintf(stderr,
                "DIAG  %-24s SKIP: debugfs amdgpu_gem_info not readable "
                "(run as root or chmod 755 /sys/kernel/debug/dri/0 for "
                "this diagnostic)\n",
                "bo_count_observed");
    } else {
        fprintf(stderr,
                "DIAG  %-24s bos_before=%d bos_after=%d delta=%d "
                "(n_ranks=%d, dups_per_peer=%d; informational only)\n",
                "bo_count_observed",
                bos_before, bos_after, bos_after - bos_before,
                n_ranks, DUP_IMPORTS_PER_PEER);
    }

    fprintf(stderr, "RESULT %s pass=%d hang=%d other_fail=%d total=%d\n",
            (n_hang == 0 && n_other == 0) ? "PASS" : "FAIL",
            n_pass, n_hang, n_other, n_ranks);
    return (n_hang == 0 && n_other == 0) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (const char* m = std::getenv(ENV_MODE);
        m != nullptr && std::strcmp(m, "child") == 0) {
        return run_child();
    }
    const char* self_path =
        (argc > 0 && argv[0] != nullptr) ? argv[0]
                                         : "./ipc_alltoall_module_load";
    return run_parent(self_path);
}
