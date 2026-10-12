// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// buggy_workload.cpp -- Intentionally faulty device workloads for partition isolation testing.
//
// Each scenario is selected with --only so the orchestrator can run one process
// per scenario on the designated logical GPU:
//
//   oob-write   : write 256 MiB past a 16-float device allocation.
//   oob-read    : read 256 MiB past a 16-float device allocation.
//   null-deref  : null pointer dereference inside a GPU kernel.
//
// Each scenario must produce a non-success hipDeviceSynchronize result
// (the fault must be observable) for the test to be valid.
//
// Device selection: HIP_VISIBLE_DEVICES restricts the visible set to one device;
// --device N selects HIP device N within that visible set.  When launched from
// the orchestrator, HIP_VISIBLE_DEVICES=<idx> pins to one slot and --device 0
// addresses that single visible device.  Passing a real device index via
// --device without HIP_VISIBLE_DEVICES is also supported for standalone use.
//
// Exit codes:
//   0  — strict fault check passed; stdout emits [BUGGY] SUITE_MARKER success scenario=<name>
//   1  — HIP/host API error during setup
//   10 — strict fault check failed (sync did not report an error — test invalid)
//
// Usage:
//   HIP_VISIBLE_DEVICES=<N> ./buggy_workload --only oob-write|oob-read|null-deref [--device N]

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Signal handling for GPU fault delivery
//
// On some AMD GPU architectures the ROCm runtime delivers GPU memory faults
// as SIGABRT to the owning process rather than returning a non-success code
// from hipDeviceSynchronize().  Without a handler the process terminates with
// exit 134 (128 + SIGABRT) before it can print the success sentinel and exit
// cleanly.
//
// The SIGABRT is raised by ROCr's async-events thread (VMFaultHandler ->
// std::abort()), not by the thread blocked in hipDeviceSynchronize().  The
// handler therefore must not siglongjmp() (the jmp_buf belongs to another
// thread) and must not call back into HIP (the runtime is mid fault-handling).
// It emits the verdict with async-signal-safe write() and terminates the
// whole process with _exit(0).
// ---------------------------------------------------------------------------
static volatile sig_atomic_t g_in_sync = 0;  // set only while inside the guarded sync
static volatile sig_atomic_t g_scenario = 0;

struct FaultMsg {
    const char *text;
    size_t len;
};
#define FAULT_MSG(s) {s, sizeof(s) - 1}
static const FaultMsg k_fault_stderr[] = {
    FAULT_MSG("    fault signal caught during oob-write - treating as fault observed\n"),
    FAULT_MSG("    fault signal caught during oob-read - treating as fault observed\n"),
    FAULT_MSG("    fault signal caught during null-deref - treating as fault observed\n"),
};
static const FaultMsg k_fault_marker[] = {
    FAULT_MSG("[BUGGY] SUITE_MARKER success scenario=oob-write\n"),
    FAULT_MSG("[BUGGY] SUITE_MARKER success scenario=oob-read\n"),
    FAULT_MSG("[BUGGY] SUITE_MARKER success scenario=null-deref\n"),
};
#undef FAULT_MSG

static void fault_signal_handler(int /*sig*/)
{
    if (!g_in_sync)
        return;  // unexpected signal outside the guarded region — let default handler run
    const sig_atomic_t i = g_scenario;
    (void)!write(STDERR_FILENO, k_fault_stderr[i].text, k_fault_stderr[i].len);
    (void)!write(STDOUT_FILENO, k_fault_marker[i].text, k_fault_marker[i].len);
    _exit(0);
}

static void hip_check(hipError_t e, const char *what)
{
    if (e != hipSuccess) {
        fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        exit(1);
    }
}

__global__ void k_oob_write(float *p, size_t /*n*/, size_t bad_index)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i != 0)
        return;
    p[bad_index] = 3.14159f;
}

__global__ void k_oob_read(const float *p, size_t /*n*/, size_t bad_index, float *out)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i != 0)
        return;
    *out = p[bad_index];
}

__global__ void k_null_deref()
{
    float *const p = nullptr;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        p[0] = 1.f;
}

static constexpr size_t oob_far_past_elements(size_t logical_floats)
{
    constexpr size_t span_bytes = 256ULL * 1024 * 1024;
    return logical_floats + span_bytes / sizeof(float);
}

// The sync after a fault MUST return a non-success code.
// Returns true when the fault was observed; false when the GPU did not fault
// (the test is then invalid — caller should exit 10).
static bool check_fault_sync(hipError_t s, const char *stage)
{
    fprintf(stderr, "    sync result after %s: %s\n", stage, hipGetErrorString(s));
    fflush(stderr);
    if (s == hipSuccess) {
        fprintf(stderr,
                "    [BUGGY] UNEXPECTED: %s sync returned success — "
                "fault was not observed; test is invalid.\n",
                stage);
        fflush(stderr);
        return false;
    }
    return true;
}

static bool run_oob_write()
{
    fprintf(stderr, ">>> scenario: oob-write (256 MiB past logical alloc)\n");
    fflush(stderr);
    const size_t n = 16;
    const size_t bad_index = oob_far_past_elements(n);
    float *d = nullptr;
    hip_check(hipMalloc(&d, n * sizeof(float)), "hipMalloc");
    k_oob_write<<<1, 1>>>(d, n, bad_index);

    // Guard: on gfx94x the runtime may deliver SIGABRT instead of returning
    // a non-success error from hipDeviceSynchronize(); the handler then
    // terminates the process and this function does not return.
    g_scenario = 0;
    g_in_sync = 1;
    hipError_t s = hipDeviceSynchronize();
    g_in_sync = 0;
    bool ok = check_fault_sync(s, "oob-write");
    // hipDeviceReset() releases all device allocations; call it before hipFree()
    // to avoid abort() on a poisoned HIP context (observed on gfx94x).
    (void)hipDeviceReset();
    return ok;
}

static bool run_oob_read()
{
    fprintf(stderr, ">>> scenario: oob-read\n");
    fflush(stderr);
    const size_t n = 16;
    const size_t bad_index = oob_far_past_elements(n);
    float *d = nullptr;
    float *d_out = nullptr;
    hip_check(hipMalloc(&d, n * sizeof(float)), "hipMalloc buf");
    hip_check(hipMalloc(&d_out, sizeof(float)), "hipMalloc out");
    k_oob_read<<<1, 1>>>(d, n, bad_index, d_out);

    g_scenario = 1;
    g_in_sync = 1;
    hipError_t s = hipDeviceSynchronize();
    g_in_sync = 0;
    bool ok = check_fault_sync(s, "oob-read");
    (void)hipDeviceReset();
    return ok;
}

static bool run_null_deref()
{
    fprintf(stderr, ">>> scenario: null-deref\n");
    fflush(stderr);
    k_null_deref<<<1, 1>>>();

    g_scenario = 2;
    g_in_sync = 1;
    hipError_t s = hipDeviceSynchronize();
    g_in_sync = 0;
    bool ok = check_fault_sync(s, "null-deref");
    (void)hipDeviceReset();
    return ok;
}

enum class OnlyMode { OobWrite, OobRead, NullDeref };

static const char *only_mode_label(OnlyMode m)
{
    switch (m) {
    case OnlyMode::OobWrite:  return "oob-write";
    case OnlyMode::OobRead:   return "oob-read";
    case OnlyMode::NullDeref: return "null-deref";
    }
    return "unknown";
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "Usage: HIP_VISIBLE_DEVICES=<N> %s --only oob-write|oob-read|null-deref [--device N]\n"
            "\n"
            "Device selection: HIP_VISIBLE_DEVICES restricts the visible set;\n"
            "  --device N (default 0) selects HIP device N within that visible set.\n"
            "  When the orchestrator pins via HIP_VISIBLE_DEVICES=<idx>, pass --device 0.\n"
            "\n"
            "Exit 0: fault observed + SUITE_MARKER emitted.\n"
            "Exit 1: HIP/host setup error.\n"
            "Exit 10: fault not observed — test invalid.\n",
            argv0);
}

int main(int argc, char **argv)
{
    bool only_set = false;
    OnlyMode only = OnlyMode::OobWrite;
    int device = 0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--only") && i + 1 < argc) {
            const char *name = argv[++i];
            if (!strcmp(name, "oob-write"))
                only = OnlyMode::OobWrite;
            else if (!strcmp(name, "oob-read"))
                only = OnlyMode::OobRead;
            else if (!strcmp(name, "null-deref"))
                only = OnlyMode::NullDeref;
            else {
                fprintf(stderr, "Unknown --only scenario: %s\n", name);
                usage(argv[0]);
                return 1;
            }
            only_set = true;
        } else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            device = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!only_set) {
        fprintf(stderr, "ERROR: --only is required.\n");
        usage(argv[0]);
        return 1;
    }

    // Install signal handlers before touching the GPU.
    // runtime may deliver GPU memory faults as SIGABRT rather than returning a
    // non-success code from hipDeviceSynchronize().
    signal(SIGABRT, fault_signal_handler);
    signal(SIGBUS,  fault_signal_handler);

    hip_check(hipSetDevice(device), "hipSetDevice");
    hipDeviceProp_t prop{};
    hip_check(hipGetDeviceProperties(&prop, device), "hipGetDeviceProperties");
    fprintf(stderr, "[BUGGY] hip_device=%d name=%s scenario=%s\n",
            device, prop.name, only_mode_label(only));
    printf("buggy_workload pid=%d --only %s hip_device=%d\n",
           static_cast<int>(getpid()), only_mode_label(only), device);
    fflush(stdout);

    bool strict_ok = false;
    switch (only) {
    case OnlyMode::OobWrite:  strict_ok = run_oob_write();  break;
    case OnlyMode::OobRead:   strict_ok = run_oob_read();   break;
    case OnlyMode::NullDeref: strict_ok = run_null_deref(); break;
    }

    if (!strict_ok) {
        fprintf(stderr,
                "\n[BUGGY] VALIDATION FAILURE (exit 10): fault was not observed for %s.\n"
                "        Do not interpret NORMAL pass as meaningful — buggy did not fault as required.\n",
                only_mode_label(only));
        fflush(stderr);
        return 10;
    }

    fprintf(stderr, "\n[BUGGY] %s: strict fault check OK (exit 0).\n", only_mode_label(only));
    fflush(stderr);
    printf("[BUGGY] SUITE_MARKER success scenario=%s pid=%d\n",
           only_mode_label(only), static_cast<int>(getpid()));
    fflush(stdout);
    // Use _exit(0) rather than return/exit so that C++ destructors and HIP
    // runtime atexit handlers do not run after the device has been reset.
    // stdout/stderr are explicitly flushed above so no output is lost.
    _exit(0);
}
