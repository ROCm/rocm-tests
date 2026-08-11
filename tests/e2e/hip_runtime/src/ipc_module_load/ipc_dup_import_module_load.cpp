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
*/

// Regression scenario for repeated HIP IPC handle imports in a consumer process
// followed by hipModuleLoad while IPC mappings remain live.
//
// Design
//
//   The process under stress is a CONSUMER, not the exporter. A consumer
//   repeatedly opens one remote handle while subsequent module-load work is
//   in progress. The test keeps those imports live so the runtime and driver
//   exercise the relevant IPC/module-load interaction under pressure.
//
//   This scenario reproduces the precondition deterministically and
//   exercises the post-fix restore path:
//
//     * Parent exports one IPC handle and spawns n_gpus * 2 consumers
//       via fork+exec. Each consumer is the same binary re-exec'd with
//       IPC_DUP_* env vars, so it enters hipSetDevice /
//       hipIpcOpenMemHandle with a fresh HIP runtime (the parent's
//       runtime state is not inherited).
//     * One consumer is the "stress consumer": it imports the handle
//       CONSUMER_DUP times on GPU 0, runs an SDMA pump, and drives
//       LOAD_CYCLES of hipModuleLoad/Unload with a per-cycle latency watchdog.
//     * The remaining "pressure consumers" each import CONSUMER_DUP
//       times on their assigned GPU and hold the mappings live, keeping
//       cross-process / cross-device IPC bookkeeping loaded during the
//       run.
//
// Stages and pass/fail oracle
//
//   STAGE consumer_imports     PASS iff every hipIpcOpenMemHandle in
//                              every consumer returned hipSuccess. This
//                              is the precondition: duplicates must
//                              have been created for the module_load
//                              stage to be meaningful.
//   STAGE module_load_under_load
//                              PASS iff the stress consumer completed
//                              LOAD_CYCLES hipModuleLoad/Unload cycles
//                              with no single cycle exceeding the
//                              LOAD_LATENCY_BUDGET_MS watchdog, under
//                              SDMA pressure, with repeated IPC imports live.
//
//   DIAGNOSTIC (not pass/fail) debugfs count before and after consumer
//                              imports. Printed for visibility; SKIP if
//                              debugfs is not readable. No specific value is
//                              required for PASS.
//
//   RESULT                     FAIL if any stage failed or the stress
//                              consumer exited non-zero. Otherwise PASS.

#include <hip/hip_runtime.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#include <stdarg.h>

#define HIP_OK(x)                                                             \
    do {                                                                      \
        hipError_t _e = (x);                                                  \
        if (_e != hipSuccess) {                                               \
            fprintf(stderr, "[pid %d] HIP error %d at %s:%d (%s)\n", getpid(),\
                    _e, __FILE__, __LINE__, hipGetErrorString(_e));           \
            std::_Exit(1);                                                    \
        }                                                                    \
    } while (0)

namespace {

constexpr size_t      IPC_BYTES               = 64ull << 20;
constexpr int         CONSUMER_DUP            = 16;
constexpr int         CONSUMERS_PER_GPU       = 2;
constexpr int         LOAD_CYCLES             = 200;
constexpr int         LOAD_LATENCY_BUDGET_MS  = 5000;
constexpr int         STRESS_CONSUMER_RANK    = 0;
constexpr size_t      PUMP_BUF_BYTES          = 64ull << 20;
constexpr const char* CODE_OBJECT             = "noop.hsaco";

// Ready-pipe byte codes from consumer to parent.
constexpr unsigned char READY_OK   = 0x01;
constexpr unsigned char READY_SKIP = 0x02;  // gpu unreachable for peer
constexpr unsigned char READY_FAIL = 0x03;

bool stage_failed = false;

void stage_fail(const char* name, const char* fmt, ...) {
    va_list ap;
    fprintf(stderr, "STAGE %-24s FAIL: ", name);
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    stage_failed = true;
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

void log_hip_err(int rank, const char* call, hipError_t e) {
    fprintf(stderr, "[pid %d rank=%d] %s failed: %d (%s)\n",
            getpid(), rank, call, e, hipGetErrorString(e));
}

// Open the IPC handle CONSUMER_DUP times on the current device. Returns false (and prints
// diagnostics) on any non-success import or touch; never calls _Exit so
// the caller can still signal the parent before terminating.
bool do_dup_imports(int rank, const hipIpcMemHandle_t& h,
                    std::vector<void*>& dups) {
    dups.assign(CONSUMER_DUP, nullptr);
    for (int i = 0; i < CONSUMER_DUP; ++i) {
        hipError_t e = hipIpcOpenMemHandle(&dups[i], h,
                                           hipIpcMemLazyEnablePeerAccess);
        if (e != hipSuccess) {
            fprintf(stderr,
                    "[pid %d rank=%d] hipIpcOpenMemHandle iter=%d failed: "
                    "%d (%s)\n",
                    getpid(), rank, i, e, hipGetErrorString(e));
            return false;
        }
    }
    // Touch each mapping briefly so the import actually lands in the GPU
    // page tables; a pure cold import sometimes leaves the BO unmapped
    // until first access.
    for (auto p : dups) {
        if (hipError_t e = hipMemsetAsync(p, 0xCD, IPC_BYTES, 0);
            e != hipSuccess) {
            log_hip_err(rank, "hipMemsetAsync on imported buf", e);
            return false;
        }
    }
    if (hipError_t e = hipDeviceSynchronize(); e != hipSuccess) {
        log_hip_err(rank, "hipDeviceSynchronize after dup imports", e);
        return false;
    }
    return true;
}

enum class SetupResult { Ok, SkipPeer, Fail };

// All pre-ready-signal HIP work for a consumer. Soft-returns; never calls
// _Exit, so the caller is the single place that owns the ready-signal
// contract and the eventual exit code.
SetupResult setup_consumer(int rank, int gpu, const hipIpcMemHandle_t& h,
                           std::vector<void*>& dups) {
    if (hipError_t e = hipSetDevice(gpu); e != hipSuccess) {
        log_hip_err(rank, "hipSetDevice", e);
        return SetupResult::Fail;
    }
    // Warm the device: hipSetDevice alone does not always bring the KFD
    // per-process resources fully online, and some ROCm versions then
    // reject hipIpcOpenMemHandle with hipErrorInvalidValue. A throwaway
    // allocation forces the runtime to register this process with the
    // driver before the import.
    void* warm = nullptr;
    if (hipError_t e = hipMalloc(&warm, 4096); e != hipSuccess) {
        log_hip_err(rank, "hipMalloc(warm)", e);
        return SetupResult::Fail;
    }
    (void)hipFree(warm);
    if (gpu != 0) {
        int can = 0;
        if (hipError_t e = hipDeviceCanAccessPeer(&can, gpu, 0);
            e != hipSuccess) {
            log_hip_err(rank, "hipDeviceCanAccessPeer(gpu,0)", e);
            return SetupResult::Fail;
        }
        if (!can) return SetupResult::SkipPeer;
        hipError_t pe = hipDeviceEnablePeerAccess(0, 0);
        if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled) {
            log_hip_err(rank, "hipDeviceEnablePeerAccess(0,0)", pe);
            return SetupResult::Fail;
        }
    }
    if (!do_dup_imports(rank, h, dups)) return SetupResult::Fail;
    return SetupResult::Ok;
}

void send_ready(int fd, unsigned char code) {
    (void)write(fd, &code, 1);
}

// Read exactly len bytes from fd. Pipes may return short reads; callers
// may also see EINTR. Returns false (and sets *got) on EOF or error.
bool read_full(int fd, void* buf, size_t len, size_t* got) {
    char* p = static_cast<char*>(buf);
    size_t off = 0;
    while (off < len) {
        ssize_t n = read(fd, p + off, len - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (got) *got = off;
            return false;
        }
        if (n == 0) {
            if (got) *got = off;
            return false;
        }
        off += static_cast<size_t>(n);
    }
    if (got) *got = off;
    return true;
}

// Pressure consumer: import the handle CONSUMER_DUP times, signal ready,
// hold the mappings live until the parent SIGTERMs us.
[[noreturn]] void pressure_consumer_main(int rank, int gpu,
                                         int handle_fd, int ready_fd) {
    hipIpcMemHandle_t h{};
    size_t got = 0;
    if (!read_full(handle_fd, &h, sizeof(h), &got)) {
        fprintf(stderr,
                "[pid %d rank=%d] failed to read handle from parent "
                "(got=%zu/%zu): %s\n",
                getpid(), rank, got, sizeof(h),
                (got == 0 && errno == 0) ? "EOF" : std::strerror(errno));
        send_ready(ready_fd, READY_FAIL);
        std::_Exit(2);
    }
    close(handle_fd);

    std::vector<void*> dups;
    switch (setup_consumer(rank, gpu, h, dups)) {
        case SetupResult::Ok:
            send_ready(ready_fd, READY_OK);
            break;
        case SetupResult::SkipPeer:
            send_ready(ready_fd, READY_SKIP);
            close(ready_fd);
            std::_Exit(0);
        case SetupResult::Fail:
            send_ready(ready_fd, READY_FAIL);
            std::_Exit(4);
    }
    close(ready_fd);

    // Hold the imported mappings live for the parent's stress consumer.
    // Default SIGTERM disposition terminates the process while pause() is
    // blocked, so any code after pause() is unreachable in practice; the
    // _Exit below exists only to satisfy [[noreturn]] for hypothetical
    // future signal handlers.
    pause();
    std::_Exit(0);
}

// Stress consumer: same dup-import setup, then drive an SDMA pump and
// run LOAD_CYCLES of hipModuleLoad/Unload with a per-cycle latency
// watchdog while repeated IPC imports remain live. Exits 0 on full success,
// non-zero otherwise.
[[noreturn]] void stress_consumer_main(int rank, int gpu,
                                       int handle_fd, int ready_fd) {
    hipIpcMemHandle_t h{};
    size_t got = 0;
    if (!read_full(handle_fd, &h, sizeof(h), &got)) {
        fprintf(stderr,
                "[pid %d rank=%d] failed to read handle from parent "
                "(got=%zu/%zu): %s\n",
                getpid(), rank, got, sizeof(h),
                (got == 0 && errno == 0) ? "EOF" : std::strerror(errno));
        send_ready(ready_fd, READY_FAIL);
        std::_Exit(2);
    }
    close(handle_fd);

    std::vector<void*> dups;
    switch (setup_consumer(rank, gpu, h, dups)) {
        case SetupResult::Ok:
            send_ready(ready_fd, READY_OK);
            break;
        case SetupResult::SkipPeer:
            // Stress consumer pinned to gpu 0 should never hit this; if it
            // ever does, surface as a real failure -- without the stress
            // consumer landing the test has nothing to measure.
            fprintf(stderr,
                    "[pid %d rank=%d] stress consumer unexpectedly hit "
                    "SkipPeer; aborting\n",
                    getpid(), rank);
            send_ready(ready_fd, READY_FAIL);
            std::_Exit(5);
        case SetupResult::Fail:
            send_ready(ready_fd, READY_FAIL);
            std::_Exit(4);
    }
    close(ready_fd);

    std::atomic<bool> stop_pump{false};
    hipStream_t bg{};
    HIP_OK(hipStreamCreate(&bg));
    void *ping = nullptr, *pong = nullptr;
    HIP_OK(hipMalloc(&ping, PUMP_BUF_BYTES));
    HIP_OK(hipMalloc(&pong, PUMP_BUF_BYTES));
    std::thread pump([&] {
        // HIP per-thread current device defaults to 0 but the lambda runs
        // in a fresh thread; pin it explicitly so ping/pong (allocated on
        // device 0 by the main thread above) match the device the
        // memcpy ends up scheduled on.
        if (hipError_t e = hipSetDevice(gpu); e != hipSuccess) {
            log_hip_err(rank, "pump: hipSetDevice", e);
            return;
        }
        while (!stop_pump.load(std::memory_order_relaxed)) {
            (void)hipMemcpyAsync(pong, ping, PUMP_BUF_BYTES,
                                 hipMemcpyDeviceToDevice, bg);
        }
    });

    long long worst_load_ms = 0;
    bool load_hang = false;
    int last_cycle = 0;
    for (int i = 0; i < LOAD_CYCLES; ++i) {
        last_cycle = i;
        hipModule_t mod = nullptr;
        auto t0 = std::chrono::steady_clock::now();
        HIP_OK(hipModuleLoad(&mod, CODE_OBJECT));
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
        if (dt > worst_load_ms) worst_load_ms = dt;
        if (dt > LOAD_LATENCY_BUDGET_MS) {
            load_hang = true;
            fprintf(stderr,
                    "[pid %d rank=%d] STAGE %-24s FAIL: cycle %d took %lld "
                    "ms (> %d ms budget)\n",
                    getpid(), rank, "module_load_under_load", i,
                    static_cast<long long>(dt), LOAD_LATENCY_BUDGET_MS);
            // Don't unload here; the module may be in a weird state. The
            // module BO leak is harmless since we're about to _Exit.
            break;
        }
        HIP_OK(hipModuleUnload(mod));
    }

    stop_pump.store(true);
    pump.join();

    if (!load_hang) {
        fprintf(stderr,
                "[pid %d rank=%d] STAGE %-24s PASS (cycles=%d, worst=%lld "
                "ms, imports_landed=%d)\n",
                getpid(), rank, "module_load_under_load", last_cycle + 1,
                worst_load_ms, CONSUMER_DUP);
    }

    // Best-effort cleanup. The verdict has already been decided by
    // load_hang above; a hiccup here must not flip a real PASS into a
    // false-positive FAIL, so use soft (void) calls instead of HIP_OK.
    (void)hipFree(ping);
    (void)hipFree(pong);
    (void)hipStreamDestroy(bg);
    for (auto p : dups) {
        if (p) (void)hipIpcCloseMemHandle(p);
    }
    std::_Exit(load_hang ? 1 : 0);
}

// Internal env-var protocol used to dispatch into a re-exec'd child.
// We use fork+exec (not bare fork) so that each consumer enters
// hipSetDevice / hipIpcOpenMemHandle with a pristine HIP runtime state,
// not the parent's already-initialized runtime carried across fork.
// Same-process or fork-inherited HIP runtime state causes
// hipIpcOpenMemHandle on the consumer side to fail with
// hipErrorInvalidValue on ROCm, which would defeat the precondition
// this scenario is built on.
constexpr const char* ENV_MODE      = "IPC_DUP_MODE";       // "child"
constexpr const char* ENV_RANK      = "IPC_DUP_RANK";
constexpr const char* ENV_GPU       = "IPC_DUP_GPU";
constexpr const char* ENV_ROLE      = "IPC_DUP_ROLE";       // stress|pressure
constexpr const char* ENV_HANDLE_FD = "IPC_DUP_HANDLE_FD";
constexpr const char* ENV_READY_FD  = "IPC_DUP_READY_FD";

int run_child() {
    const char* rank_s = std::getenv(ENV_RANK);
    const char* gpu_s  = std::getenv(ENV_GPU);
    const char* role_s = std::getenv(ENV_ROLE);
    const char* hfd_s  = std::getenv(ENV_HANDLE_FD);
    const char* rfd_s  = std::getenv(ENV_READY_FD);
    if (!rank_s || !gpu_s || !role_s || !hfd_s || !rfd_s) {
        fprintf(stderr,
                "[pid %d] child mode missing required env "
                "(rank=%s gpu=%s role=%s handle_fd=%s ready_fd=%s)\n",
                getpid(),
                rank_s ? rank_s : "(null)", gpu_s ? gpu_s : "(null)",
                role_s ? role_s : "(null)", hfd_s ? hfd_s : "(null)",
                rfd_s ? rfd_s : "(null)");
        return 1;
    }
    int rank      = std::atoi(rank_s);
    int gpu       = std::atoi(gpu_s);
    int handle_fd = std::atoi(hfd_s);
    int ready_fd  = std::atoi(rfd_s);

    if (std::strcmp(role_s, "stress") == 0) {
        stress_consumer_main(rank, gpu, handle_fd, ready_fd);
    } else if (std::strcmp(role_s, "pressure") == 0) {
        pressure_consumer_main(rank, gpu, handle_fd, ready_fd);
    }
    fprintf(stderr, "[pid %d] child mode unknown role '%s'\n",
            getpid(), role_s);
    send_ready(ready_fd, READY_FAIL);
    return 1;
}

int run_parent(const char* self_path) {
    int n_gpus = 0;
    HIP_OK(hipGetDeviceCount(&n_gpus));
    if (n_gpus < 1) {
        fprintf(stderr, "no GPUs visible\n");
        return 77;
    }

    HIP_OK(hipSetDevice(0));

    void* producer_bo = nullptr;
    HIP_OK(hipMalloc(&producer_bo, IPC_BYTES));
    HIP_OK(hipMemset(producer_bo, 0xAB, IPC_BYTES));

    hipIpcMemHandle_t handle{};
    HIP_OK(hipIpcGetMemHandle(&handle, producer_bo));

    int bos_before = count_bos_of_size(0, IPC_BYTES);

    int handle_pipefd[2];
    int ready_pipefd[2];
    if (pipe(handle_pipefd) < 0 || pipe(ready_pipefd) < 0) {
        perror("pipe");
        return 1;
    }

    const int n_consumers = n_gpus * CONSUMERS_PER_GPU;
    std::vector<pid_t> consumer_pids;
    consumer_pids.reserve(n_consumers);
    pid_t stress_pid = -1;

    for (int c = 0; c < n_consumers; ++c) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return 1;
        }
        if (pid == 0) {
            close(handle_pipefd[1]);
            close(ready_pipefd[0]);
            // Auto-cleanup: if the parent dies for any reason (early
            // bailout, kill -9, crash), the kernel sends SIGTERM to this
            // child instead of leaving it orphaned at pause() forever
            // holding IPC mappings. Survives exec. Linux-specific,
            // which matches the rest of this scenario's debugfs
            // dependencies.
            (void)prctl(PR_SET_PDEATHSIG, SIGTERM);

            int gpu = c % n_gpus;
            char rank_buf[16], gpu_buf[16], hfd_buf[16], rfd_buf[16];
            std::snprintf(rank_buf, sizeof(rank_buf), "%d", c);
            std::snprintf(gpu_buf,  sizeof(gpu_buf),  "%d", gpu);
            std::snprintf(hfd_buf,  sizeof(hfd_buf),  "%d", handle_pipefd[0]);
            std::snprintf(rfd_buf,  sizeof(rfd_buf),  "%d", ready_pipefd[1]);
            setenv(ENV_MODE,      "child",   1);
            setenv(ENV_RANK,      rank_buf,  1);
            setenv(ENV_GPU,       gpu_buf,   1);
            setenv(ENV_ROLE,
                   c == STRESS_CONSUMER_RANK ? "stress" : "pressure", 1);
            setenv(ENV_HANDLE_FD, hfd_buf,   1);
            setenv(ENV_READY_FD,  rfd_buf,   1);

            // Pipe fds do not have O_CLOEXEC set, so they remain open
            // across exec with the same numeric values we just wrote
            // into the env vars above. The new process image starts
            // fresh with no inherited HIP runtime state.
            execlp(self_path, self_path, static_cast<char*>(nullptr));

            // exec failed; signal parent so its ready-pipe read does
            // not deadlock, then exit hard.
            fprintf(stderr,
                    "[pid %d rank=%d] execlp(%s) failed: %s\n",
                    getpid(), c, self_path, std::strerror(errno));
            send_ready(ready_pipefd[1], READY_FAIL);
            std::_Exit(99);
        }
        if (c == STRESS_CONSUMER_RANK) stress_pid = pid;
        consumer_pids.push_back(pid);
    }
    close(handle_pipefd[0]);
    close(ready_pipefd[1]);

    for (int c = 0; c < n_consumers; ++c) {
        if (write(handle_pipefd[1], &handle, sizeof(handle)) !=
            static_cast<ssize_t>(sizeof(handle))) {
            perror("write handle");
            return 1;
        }
    }
    close(handle_pipefd[1]);

    int n_ok = 0, n_skip = 0, n_fail = 0, n_unknown = 0;
    for (int c = 0; c < n_consumers; ++c) {
        unsigned char code = 0;
        ssize_t r = read(ready_pipefd[0], &code, 1);
        if (r != 1) {
            ++n_fail;
            continue;
        }
        switch (code) {
            case READY_OK:   ++n_ok;   break;
            case READY_SKIP: ++n_skip; break;
            case READY_FAIL: ++n_fail; break;
            default:         ++n_unknown; break;
        }
    }
    close(ready_pipefd[0]);

    int bos_after = count_bos_of_size(0, IPC_BYTES);

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
                "(n_consumers=%d, dups_per_consumer=%d, expected_delta is "
                "driver-dependent; this line is informational only)\n",
                "bo_count_observed",
                bos_before, bos_after, bos_after - bos_before,
                n_consumers, CONSUMER_DUP);
    }

    if (n_fail > 0 || n_unknown > 0) {
        stage_fail("consumer_imports",
                   "n_fail=%d n_unknown=%d n_ok=%d n_skip=%d (of %d "
                   "consumers); see per-pid diagnostic above",
                   n_fail, n_unknown, n_ok, n_skip, n_consumers);
    } else if (n_ok == 0) {
        stage_fail("consumer_imports",
                   "no consumer reported a successful import "
                   "(n_skip=%d, n_consumers=%d)",
                   n_skip, n_consumers);
    } else {
        fprintf(stderr,
                "STAGE %-24s PASS (n_ok=%d n_skip=%d n_consumers=%d)\n",
                "consumer_imports", n_ok, n_skip, n_consumers);
    }

    int stress_status = 0;
    bool stress_ran = false;
    if (stress_pid > 0) {
        if (waitpid(stress_pid, &stress_status, 0) == stress_pid) {
            stress_ran = true;
        }
    }

    if (!stress_ran) {
        stage_fail("module_load_under_load",
                   "stress consumer (pid=%d) could not be reaped", stress_pid);
    } else if (WIFEXITED(stress_status) && WEXITSTATUS(stress_status) == 0) {
        // Stress consumer printed its own STAGE module_load_under_load
        // PASS line; nothing more to do here.
    } else {
        stage_fail("module_load_under_load",
                   "stress consumer exit_status=0x%x (exited=%d, code=%d, "
                   "signaled=%d, signal=%d)",
                   stress_status,
                   WIFEXITED(stress_status), WEXITSTATUS(stress_status),
                   WIFSIGNALED(stress_status), WTERMSIG(stress_status));
    }

    for (pid_t pid : consumer_pids) {
        if (pid == stress_pid) continue;
        kill(pid, SIGTERM);
    }
    for (pid_t pid : consumer_pids) {
        if (pid == stress_pid) continue;
        waitpid(pid, nullptr, 0);
    }

    // Soft free: the verdict has already been decided. A cleanup hiccup
    // must not flip a real PASS into a confusing FAIL exit code.
    if (hipError_t e = hipFree(producer_bo); e != hipSuccess) {
        fprintf(stderr,
                "[pid %d] note: hipFree(producer_bo) returned %d (%s) during "
                "teardown; verdict unchanged\n",
                getpid(), e, hipGetErrorString(e));
    }

    fprintf(stderr,
            "RESULT %s (n_gpus=%d, n_consumers=%d, consumer_dup=%d, "
            "load_cycles=%d)\n",
            stage_failed ? "FAIL" : "PASS",
            n_gpus, n_consumers, CONSUMER_DUP, LOAD_CYCLES);
    return stage_failed ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (const char* m = std::getenv(ENV_MODE);
        m != nullptr && std::strcmp(m, "child") == 0) {
        return run_child();
    }
    const char* self_path =
        (argc > 0 && argv[0] != nullptr) ? argv[0]
                                         : "./ipc_dup_import_module_load";
    return run_parent(self_path);
}
