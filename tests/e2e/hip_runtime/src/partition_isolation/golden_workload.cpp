// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// golden_workload.cpp -- Benign SAXPY loop for partition isolation testing.
//
// Runs a continuous hipDeviceSynchronize-checked SAXPY kernel for a wall-clock
// duration supplied via the GOLDEN_SECONDS environment variable (or a built-in
// default).  The process reports its device name at startup and a final batch
// count on exit.  Designed to run concurrently with buggy_workload on a
// disjoint logical GPU partition.
//
// Device selection: HIP_VISIBLE_DEVICES restricts the visible set to one device;
// --device N selects HIP device N within that visible set.  When launched from
// the orchestrator, HIP_VISIBLE_DEVICES=<idx> pins to one slot and --device 0
// addresses that single visible device.  Passing a real device index via
// --device without HIP_VISIBLE_DEVICES is also supported for standalone use.
//
// Usage:
//   HIP_VISIBLE_DEVICES=<N> GOLDEN_SECONDS=<sec> ./golden_workload [--device N]
//
// Exit 0 on success; exit 1 on any HIP error.

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

static constexpr float kBuggySuiteEstSec = 55.f;
static constexpr float kGoldenVsBuggyFactor = 4.f;

static void hip_check(hipError_t e, const char *what)
{
    if (e != hipSuccess) {
        fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        exit(1);
    }
}

__global__ void k_saxpy(size_t n, float a, const float *x, float *y)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

static float resolve_wall_seconds()
{
    const char *e = std::getenv("GOLDEN_SECONDS");
    if (e != nullptr && e[0] != '\0') {
        float v = static_cast<float>(atof(e));
        if (v > 0.f)
            return v;
        fprintf(stderr, "GOLDEN_SECONDS invalid; using default.\n");
    }
    return kGoldenVsBuggyFactor * kBuggySuiteEstSec;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "Usage: HIP_VISIBLE_DEVICES=<N> GOLDEN_SECONDS=<sec> %s [--device N]\n"
            "\n"
            "Wall duration:\n"
            "  GOLDEN_SECONDS env (seconds) — default: %.0f s (= %.0f x %.0f s buggy estimate).\n"
            "\n"
            "Device selection: HIP_VISIBLE_DEVICES restricts the visible set;\n"
            "  --device N (default 0) selects HIP device N within that visible set.\n"
            "  When the orchestrator pins via HIP_VISIBLE_DEVICES=<idx>, pass --device 0.\n",
            argv0,
            static_cast<double>(kGoldenVsBuggyFactor * kBuggySuiteEstSec),
            static_cast<double>(kGoldenVsBuggyFactor),
            static_cast<double>(kBuggySuiteEstSec));
}

int main(int argc, char **argv)
{
    int device = 0;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) {
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

    const float seconds = resolve_wall_seconds();
    printf("golden_workload start pid=%d hip_device=%d wall_sec=%.3f\n",
           static_cast<int>(getpid()), device, seconds);
    fflush(stdout);

    hip_check(hipSetDevice(device), "hipSetDevice");
    hipDeviceProp_t prop{};
    hip_check(hipGetDeviceProperties(&prop, device), "hipGetDeviceProperties");
    fprintf(stderr, "[NORMAL] hip_device=%d name=%s duration=%.3fs\n",
            device, prop.name, seconds);
    fflush(stderr);

    constexpr size_t n = 32U * 1024U * 1024U;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);

    float *dx = nullptr;
    float *dy = nullptr;
    hip_check(hipMalloc(&dx, n * sizeof(float)), "hipMalloc x");
    hip_check(hipMalloc(&dy, n * sizeof(float)), "hipMalloc y");
    hip_check(hipMemset(dx, 0, n * sizeof(float)), "hipMemset x");
    hip_check(hipMemset(dy, 1, n * sizeof(float)), "hipMemset y");

    const auto t0 = std::chrono::steady_clock::now();
    long long batches = 0;
    for (;;) {
        k_saxpy<<<blocks, threads>>>(n, 1.00001f, dx, dy);
        hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize");
        ++batches;
        const float el =
            std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();
        if (el >= seconds)
            break;
    }

    fprintf(stderr, "golden_workload hip_device=%d: finished %lld batches OK\n",
            device, static_cast<long long>(batches));
    printf("GOLDEN_OK batches=%lld hip_device=%d\n",
           static_cast<long long>(batches), device);
    fflush(stdout);
    (void)hipFree(dx);
    (void)hipFree(dy);
    return 0;
}
