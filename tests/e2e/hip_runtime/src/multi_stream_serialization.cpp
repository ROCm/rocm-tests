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

 * @brief HIP stress for AMD_SERIALIZE_KERNEL=1: many streams, deep launch queues, and cross-stream events.
 *
 * Single process calls setenv(AMD_SERIALIZE_KERNEL,"1") then runs independent-stream and
 * cross-stream (hipEventRecord / hipStreamWaitEvent) legs with fingerprint checks.
 *
 * Usage:
 *   ./multi_stream_serialization
 */

#include <hip/hip_runtime.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(stmt)                                                                 \
    do {                                                                                 \
        hipError_t _err = (stmt);                                                        \
        if (_err != hipSuccess) {                                                        \
            fprintf(stderr, "%s:%d HIP error: %s (%d)\n", __FILE__, __LINE__,            \
                    hipGetErrorString(_err), static_cast<int>(_err));                    \
            return false;                                                                \
        }                                                                                \
    } while (0)

#define HIP_CHECK_MAIN(stmt)                                                            \
    do {                                                                                 \
        hipError_t _err = (stmt);                                                        \
        if (_err != hipSuccess) {                                                        \
            fprintf(stderr, "%s:%d HIP error: %s (%d)\n", __FILE__, __LINE__,            \
                    hipGetErrorString(_err), static_cast<int>(_err));                    \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

namespace {

constexpr int kMaxStreams = 16;
// Launch covers all elements; kernel uses grid-stride so large buffers increase real GPU work.
constexpr int kThreadsPerBlock = 256;

// Fingerprint encodes stream index and kernel index so we can detect cross-stream corruption.
__global__ void write_fingerprint(int* data, int n, int stream_tag, int kernel_idx) {
    const int fp = stream_tag * 100000 + kernel_idx;
    const int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += stride) {
        data[idx] = fp;
    }
}

static void launch_write_fingerprint(hipStream_t stream, int* d_data, int n_elem, int stream_tag,
                                     int kernel_idx) {
    const int blocks = (n_elem + kThreadsPerBlock - 1) / kThreadsPerBlock;
    write_fingerprint<<<dim3(blocks), dim3(kThreadsPerBlock), 0, stream>>>(d_data, n_elem,
                                                                          stream_tag, kernel_idx);
}

static int getenv_int(const char* name, int default_val) {
    const char* s = getenv(name);
    if (!s || !*s) {
        return default_val;
    }
    return atoi(s);
}

// Many streams, no cross-stream events (broad fan-out).
bool run_independent_streams(int num_streams, int kernels_per_stream, int n_elem, int rounds) {
    printf("--- Independent streams (%d streams x %d kernels x %d rounds) ---\n",
           num_streams, kernels_per_stream, rounds);

    std::vector<hipStream_t> streams(static_cast<size_t>(num_streams));
    std::vector<int*> d_data(static_cast<size_t>(num_streams));
    std::vector<std::vector<int>> h_host(static_cast<size_t>(num_streams));

    for (int s = 0; s < num_streams; ++s) {
        HIP_CHECK(hipStreamCreateWithFlags(&streams[static_cast<size_t>(s)], hipStreamNonBlocking));
        HIP_CHECK(hipMalloc(&d_data[static_cast<size_t>(s)], sizeof(int) * n_elem));
        h_host[static_cast<size_t>(s)].resize(static_cast<size_t>(n_elem));
    }

    for (int r = 0; r < rounds; ++r) {
        for (int s = 0; s < num_streams; ++s) {
            const int tag = s + 1;
            for (int k = 0; k < kernels_per_stream; ++k) {
                launch_write_fingerprint(streams[static_cast<size_t>(s)],
                                         d_data[static_cast<size_t>(s)], n_elem, tag, k);
                HIP_CHECK(hipGetLastError());
            }
        }

        for (int s = 0; s < num_streams; ++s) {
            HIP_CHECK(hipStreamSynchronize(streams[static_cast<size_t>(s)]));
            const int expect = (s + 1) * 100000 + (kernels_per_stream - 1);
            HIP_CHECK(hipMemcpy(h_host[static_cast<size_t>(s)].data(), d_data[static_cast<size_t>(s)],
                                sizeof(int) * n_elem, hipMemcpyDeviceToHost));
            for (int i = 0; i < n_elem; ++i) {
                if (h_host[static_cast<size_t>(s)][static_cast<size_t>(i)] != expect) {
                    fprintf(stderr,
                            "FAIL independent: round %d stream %d idx %d got %d expected %d\n", r, s,
                            i, h_host[static_cast<size_t>(s)][static_cast<size_t>(i)], expect);
                    for (int t = 0; t < num_streams; ++t) {
                        hipFree(d_data[static_cast<size_t>(t)]);
                        hipStreamDestroy(streams[static_cast<size_t>(t)]);
                    }
                    return false;
                }
            }
        }
    }

    for (int s = 0; s < num_streams; ++s) {
        HIP_CHECK(hipFree(d_data[static_cast<size_t>(s)]));
        HIP_CHECK(hipStreamDestroy(streams[static_cast<size_t>(s)]));
    }

    printf("PASS: independent streams\n");
    return true;
}

// Linear chain: stream s waits for event on stream s-1 before launching kernels.
bool run_chain_dependencies(int num_streams, int kernels_per_stream, int n_elem, int rounds) {
    if (num_streams < 2) {
        printf("--- Cross-stream event chain: skipped (need >= 2 streams) ---\n");
        printf("PASS: cross-stream event chain (skipped)\n");
        return true;
    }

    printf("--- Cross-stream event chain (%d streams x %d kernels x %d rounds) ---\n",
           num_streams, kernels_per_stream, rounds);

    std::vector<hipStream_t> streams(static_cast<size_t>(num_streams));
    std::vector<hipEvent_t> events(static_cast<size_t>(num_streams));
    std::vector<int*> d_data(static_cast<size_t>(num_streams));
    std::vector<std::vector<int>> h_host(static_cast<size_t>(num_streams));

    for (int s = 0; s < num_streams; ++s) {
        HIP_CHECK(hipStreamCreateWithFlags(&streams[static_cast<size_t>(s)], hipStreamNonBlocking));
        HIP_CHECK(hipEventCreateWithFlags(&events[static_cast<size_t>(s)], hipEventDisableTiming));
        HIP_CHECK(hipMalloc(&d_data[static_cast<size_t>(s)], sizeof(int) * n_elem));
        h_host[static_cast<size_t>(s)].resize(static_cast<size_t>(n_elem));
    }

    for (int r = 0; r < rounds; ++r) {
        for (int s = 0; s < num_streams; ++s) {
            if (s > 0) {
                HIP_CHECK(hipStreamWaitEvent(streams[static_cast<size_t>(s)],
                                             events[static_cast<size_t>(s - 1)], 0));
            }
            const int tag = s + 1;
            for (int k = 0; k < kernels_per_stream; ++k) {
                launch_write_fingerprint(streams[static_cast<size_t>(s)],
                                         d_data[static_cast<size_t>(s)], n_elem, tag, k);
                HIP_CHECK(hipGetLastError());
            }
            HIP_CHECK(hipEventRecord(events[static_cast<size_t>(s)], streams[static_cast<size_t>(s)]));
        }

        HIP_CHECK(hipEventSynchronize(events[static_cast<size_t>(num_streams - 1)]));

        for (int s = 0; s < num_streams; ++s) {
            HIP_CHECK(hipStreamSynchronize(streams[static_cast<size_t>(s)]));
            const int expect = (s + 1) * 100000 + (kernels_per_stream - 1);
            HIP_CHECK(hipMemcpy(h_host[static_cast<size_t>(s)].data(), d_data[static_cast<size_t>(s)],
                                sizeof(int) * n_elem, hipMemcpyDeviceToHost));
            for (int i = 0; i < n_elem; ++i) {
                if (h_host[static_cast<size_t>(s)][static_cast<size_t>(i)] != expect) {
                    fprintf(stderr,
                            "FAIL chain: round %d stream %d idx %d got %d expected %d\n", r, s, i,
                            h_host[static_cast<size_t>(s)][static_cast<size_t>(i)], expect);
                    for (int t = 0; t < num_streams; ++t) {
                        hipEventDestroy(events[static_cast<size_t>(t)]);
                        hipFree(d_data[static_cast<size_t>(t)]);
                        hipStreamDestroy(streams[static_cast<size_t>(t)]);
                    }
                    return false;
                }
            }
        }
    }

    for (int s = 0; s < num_streams; ++s) {
        HIP_CHECK(hipEventDestroy(events[static_cast<size_t>(s)]));
        HIP_CHECK(hipFree(d_data[static_cast<size_t>(s)]));
        HIP_CHECK(hipStreamDestroy(streams[static_cast<size_t>(s)]));
    }

    printf("PASS: cross-stream event chain\n");
    return true;
}

}  // namespace

int main(void) {
    if (setenv("AMD_SERIALIZE_KERNEL", "1", 1) != 0) {
        fprintf(stderr,
                "ERROR: could not set AMD_SERIALIZE_KERNEL=1 for this process: %s\n"
                "       This test refuses to run without kernel serialization enabled. Exiting.\n",
                std::strerror(errno));
        return 2;
    }
    const char* ser_chk = getenv("AMD_SERIALIZE_KERNEL");
    if (!ser_chk || strcmp(ser_chk, "1") != 0) {
        fprintf(stderr,
                "ERROR: AMD_SERIALIZE_KERNEL must be 1 after setenv (got: %s). Exiting.\n",
                ser_chk ? ser_chk : "(unset)");
        return 2;
    }

    // Defaults (~KS_ROUNDS=900 with 40 kernels, 262144 elements, 4 streams) target **~120 s** wall time
    // on typical datacenter GPUs with AMD_SERIALIZE_KERNEL=1 — not exact; tune KS_ROUNDS for your tier.
    const int num_streams = getenv_int("KS_STREAMS", 4);
    const int kernels_per_stream = getenv_int("KS_KERNELS", 40);
    const int n_elem = getenv_int("KS_ELEMENTS", 262144);
    const int rounds = getenv_int("KS_ROUNDS", 900);

    if (num_streams < 1 || num_streams > kMaxStreams) {
        fprintf(stderr, "KS_STREAMS must be 1..%d\n", kMaxStreams);
        return 2;
    }
    if (kernels_per_stream < 1 || n_elem < 1 || rounds < 1) {
        fprintf(stderr, "Invalid KS_KERNELS / KS_ELEMENTS / KS_ROUNDS\n");
        return 2;
    }

    HIP_CHECK_MAIN(hipSetDevice(0));
    int dev = -1;
    HIP_CHECK_MAIN(hipGetDevice(&dev));
    hipDeviceProp_t prop{};
    HIP_CHECK_MAIN(hipGetDeviceProperties(&prop, dev));
    printf("device %d: %s\n", dev, prop.name);
    printf("streams=%d kernels/stream=%d elements=%d rounds=%d\n", num_streams, kernels_per_stream,
           n_elem, rounds);

    const bool ok_indep = run_independent_streams(num_streams, kernels_per_stream, n_elem, rounds);
    const bool ok_chain = run_chain_dependencies(num_streams, kernels_per_stream, n_elem, rounds);

    const char* chain_summary = "FAIL";
    if (num_streams < 2) {
        chain_summary = "SKIP";
    } else if (ok_chain) {
        chain_summary = "PASS";
    }

    printf("\n========== Summary (subtests) ==========\n");
    printf("  independent_streams:       %s\n", ok_indep ? "PASS" : "FAIL");
    printf("  cross_stream_event_chain:  %s\n", chain_summary);
    printf("========================================\n");

    if (ok_indep && ok_chain) {
        printf("OVERALL: PASSED\n");
        return 0;
    }
    printf("OVERALL: FAILED\n");
    return 1;
}
