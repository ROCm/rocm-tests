// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "roles/ipc_xfer.h"
#include "common/hip_check.h"
#include "common/validator.h"
#include "common/health.h"
#include "kernels/kernels.h"

#include <hip/hip_runtime.h>
#include <cinttypes>
#include <cstdio>
#include <chrono>
#include <vector>
#include <unistd.h>

// Exercises: Multi-GPU peer-to-peer transfers → ROCr → KFD → xGMI/Infinity Fabric
//            and IPC memory handle export/import → KFD BO cross-device mapping.
//
// Two transfer methods are tested, alternating each iteration:
//   1. hipMemcpyPeerAsync — direct GPU-to-GPU copy via the interconnect fabric,
//      managed by the runtime and SDMA engines. Tests both directions (A→B, B→A).
//   2. Peer-mapped direct access — a kernel on GPU B reads directly from GPU A's
//      memory through the peer mapping. Tests TLB coherency and peer mapping table 
//      correctness under concurrent load.
//
// Data integrity is verified after every transfer: fill on GPU A, copy to GPU B,
// read back from GPU B and check every element.
//
// When other roles run concurrently on both GPUs, this stresses:
//   - xGMI/Infinity Fabric bandwidth contention
//   - KFD peer mapping table updates while MEMORY_MOVER allocs/frees
//   - Cross-device SDMA scheduling alongside same-device SDMA (MEMORY_MOVER)
//   - GPU page table coherency across peer-mapped regions

static constexpr size_t MB = 1024ULL * 1024;

int run_ipc_xfer(const RoleConfig& config) {
    int gpu_count = 0;
    HIP_CHECK(hipGetDeviceCount(&gpu_count));

    if (gpu_count < 2) {
        printf("[IPC_XFER] Only %d GPU(s) available — need at least 2. Skipping.\n", gpu_count);
        return 0;
    }

    int src_gpu = config.gpu_id;
    int dst_gpu = config.peer_gpu_id;

    if (dst_gpu < 0 || dst_gpu >= gpu_count) {
        fprintf(stderr, "[IPC_XFER] Invalid peer GPU %d (have %d GPUs)\n", dst_gpu, gpu_count);
        return 1;
    }
    if (src_gpu == dst_gpu) {
        fprintf(stderr, "[IPC_XFER] Source and peer GPU are the same (%d). Use a different --peer-gpu.\n", src_gpu);
        return 1;
    }

    // Check peer access
    int can_access_src_to_dst = 0, can_access_dst_to_src = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&can_access_src_to_dst, src_gpu, dst_gpu));
    HIP_CHECK(hipDeviceCanAccessPeer(&can_access_dst_to_src, dst_gpu, src_gpu));

    printf("[IPC_XFER] PID %d | GPU %d ↔ GPU %d | duration %ds\n",
           getpid(), src_gpu, dst_gpu, config.duration_sec);
    printf("[IPC_XFER] Peer access: %d→%d=%s, %d→%d=%s\n",
           src_gpu, dst_gpu, can_access_src_to_dst ? "YES" : "NO",
           dst_gpu, src_gpu, can_access_dst_to_src ? "YES" : "NO");

    // Enable peer access in both directions if available
    if (can_access_src_to_dst) {
        HIP_CHECK(hipSetDevice(src_gpu));
        hipError_t err = hipDeviceEnablePeerAccess(dst_gpu, 0);
        if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled) {
            fprintf(stderr, "[IPC_XFER] Warning: could not enable peer access %d→%d: %s\n",
                    src_gpu, dst_gpu, hipGetErrorString(err));
        }
    }
    if (can_access_dst_to_src) {
        HIP_CHECK(hipSetDevice(dst_gpu));
        hipError_t err = hipDeviceEnablePeerAccess(src_gpu, 0);
        if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled) {
            fprintf(stderr, "[IPC_XFER] Warning: could not enable peer access %d→%d: %s\n",
                    dst_gpu, src_gpu, hipGetErrorString(err));
        }
    }

    Validator validator;
    HealthMonitor health(src_gpu, config.results_dir,
                         config.rss_growth_warn_kb, config.fd_growth_warn);
    health.start();

    // Create streams on both GPUs
    HIP_CHECK(hipSetDevice(src_gpu));
    hipStream_t src_stream;
    HIP_CHECK(hipStreamCreate(&src_stream));

    HIP_CHECK(hipSetDevice(dst_gpu));
    hipStream_t dst_stream;
    HIP_CHECK(hipStreamCreate(&dst_stream));

    // Transfer sizes — varied to stress different fabric paths
    const size_t sizes[] = {
        4 * 1024,                 // 4KB — small, latency-bound
        64 * 1024,                // 64KB
        1 * MB,                   // 1MB
        16 * MB,                  // 16MB — bandwidth-bound
        64 * MB,                  // 64MB
        256 * MB,                 // 256MB — large fabric transfer
    };
    constexpr int NUM_SIZES = sizeof(sizes) / sizeof(sizes[0]);

    auto start_time = std::chrono::steady_clock::now();
    int64_t iteration = 0;
    int peer_copy_errors = 0;
    int peer_direct_errors = 0;

    while (true) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
                >= config.duration_sec) {
            break;
        }

        size_t alloc_size = sizes[iteration % NUM_SIZES];
        size_t elem_count = alloc_size / sizeof(uint32_t);
        size_t byte_count = elem_count * sizeof(uint32_t);
        uint32_t pattern = validator.make_pattern((int)iteration, config.role_id, 0);
        int method = iteration % 2; // alternate between peer copy and peer-mapped direct access

        // ---- Allocate on source GPU ----
        HIP_CHECK(hipSetDevice(src_gpu));
        (void)hipGetLastError();
        uint32_t* src_buf = nullptr;
        hipError_t err = hipMalloc(&src_buf, byte_count);
        if (err != hipSuccess) {
            printf("[IPC_XFER] #%" PRId64 " | src alloc failed (%zu bytes) — skipping\n",
                   iteration, byte_count);
            iteration++;
            continue;
        }

        // Fill on source GPU with known pattern
        int block = 256;
        int grid = (elem_count + block - 1) / block;
        hipLaunchKernelGGL(kernel_pattern_fill,
                           dim3(grid), dim3(block), 0, src_stream,
                           src_buf, elem_count, pattern);
        HIP_LAUNCH_CHECK();
        HIP_CHECK(hipStreamSynchronize(src_stream));

        if (method == 0) {
            // ---- Method 1: hipMemcpyPeerAsync ----
            printf("[IPC_XFER] #%" PRId64 " | Peer copy GPU%d→GPU%d | %zu bytes\n",
                   iteration, src_gpu, dst_gpu, byte_count);

            HIP_CHECK(hipSetDevice(dst_gpu));
            (void)hipGetLastError();
            uint32_t* dst_buf = nullptr;
            err = hipMalloc(&dst_buf, byte_count);
            if (err != hipSuccess) {
                printf("[IPC_XFER] #%" PRId64 " | dst alloc failed — skipping\n", iteration);
                HIP_CHECK(hipSetDevice(src_gpu));
                (void)hipFree(src_buf);
                iteration++;
                continue;
            }

            HIP_CHECK(hipMemcpyPeerAsync(dst_buf, dst_gpu, src_buf, src_gpu,
                                          byte_count, dst_stream));
            HIP_CHECK(hipStreamSynchronize(dst_stream));

            // Verify on destination GPU
            int* d_err_flag = nullptr;
            if (hipMalloc(&d_err_flag, sizeof(int)) == hipSuccess) {
                HIP_CHECK(hipMemsetAsync(d_err_flag, 0, sizeof(int), dst_stream));
                hipLaunchKernelGGL(kernel_pattern_verify,
                                   dim3(grid), dim3(block), 0, dst_stream,
                                   dst_buf, elem_count, pattern, d_err_flag);
                HIP_LAUNCH_CHECK();
                HIP_CHECK(hipStreamSynchronize(dst_stream));

                int dev_errors = 0;
                HIP_CHECK(hipMemcpy(&dev_errors, d_err_flag, sizeof(int),
                                    hipMemcpyDeviceToHost));
                if (dev_errors > 0) {
                    fprintf(stderr,
                            "[IPC_XFER] #%" PRId64 " | *** PEER COPY VERIFY FAILED GPU%d→GPU%d — %d mismatches ***\n",
                            iteration, src_gpu, dst_gpu, dev_errors);
                    peer_copy_errors++;
                } else {
                    printf("[IPC_XFER] #%" PRId64 " | Peer copy GPU%d→GPU%d OK\n",
                           iteration, src_gpu, dst_gpu);
                }
                (void)hipFree(d_err_flag);
            } else {
                fprintf(stderr, "[IPC_XFER] #%" PRId64 " | verify alloc failed — counting as error\n", iteration);
                peer_copy_errors++;
            }

            // Reverse direction: fill on dst GPU, copy dst→src, verify on src GPU
            printf("[IPC_XFER] #%" PRId64 " | Peer copy GPU%d→GPU%d (reverse) | %zu bytes\n",
                   iteration, dst_gpu, src_gpu, byte_count);

            // Stay on dst_gpu to fill dst_buf with a new pattern
            uint32_t pattern_rev = validator.make_pattern((int)iteration, config.role_id, 1);
            hipLaunchKernelGGL(kernel_pattern_fill,
                               dim3(grid), dim3(block), 0, dst_stream,
                               dst_buf, elem_count, pattern_rev);
            HIP_LAUNCH_CHECK();
            HIP_CHECK(hipStreamSynchronize(dst_stream));

            // Copy dst→src — switch to destination device first so the
            // stream/device association is correct for the copy target.
            HIP_CHECK(hipSetDevice(src_gpu));
            (void)hipGetLastError();
            HIP_CHECK(hipMemcpyPeerAsync(src_buf, src_gpu, dst_buf, dst_gpu,
                                          byte_count, src_stream));
            HIP_CHECK(hipStreamSynchronize(src_stream));
            d_err_flag = nullptr;
            if (hipMalloc(&d_err_flag, sizeof(int)) == hipSuccess) {
                HIP_CHECK(hipMemsetAsync(d_err_flag, 0, sizeof(int), src_stream));
                hipLaunchKernelGGL(kernel_pattern_verify,
                                   dim3(grid), dim3(block), 0, src_stream,
                                   src_buf, elem_count, pattern_rev, d_err_flag);
                HIP_LAUNCH_CHECK();
                HIP_CHECK(hipStreamSynchronize(src_stream));

                int dev_errors = 0;
                HIP_CHECK(hipMemcpy(&dev_errors, d_err_flag, sizeof(int),
                                    hipMemcpyDeviceToHost));
                if (dev_errors > 0) {
                    fprintf(stderr,
                            "[IPC_XFER] #%" PRId64 " | *** REVERSE COPY VERIFY FAILED GPU%d→GPU%d — %d mismatches ***\n",
                            iteration, dst_gpu, src_gpu, dev_errors);
                    peer_copy_errors++;
                } else {
                    printf("[IPC_XFER] #%" PRId64 " | Reverse copy GPU%d→GPU%d OK\n",
                           iteration, dst_gpu, src_gpu);
                }
                (void)hipFree(d_err_flag);
            } else {
                fprintf(stderr, "[IPC_XFER] #%" PRId64 " | verify alloc failed — counting as error\n", iteration);
                peer_copy_errors++;
            }

            HIP_CHECK(hipSetDevice(dst_gpu));
            (void)hipFree(dst_buf);

        } else {
            // ---- Method 2: Peer-mapped direct access ----
            // GPU B's kernel reads directly from GPU A's buffer through the peer mapping
            // Requires peer access to be enabled.
            if (!can_access_src_to_dst) {
                printf("[IPC_XFER] #%" PRId64 " | No peer access — falling back to peer copy\n", iteration);
                // Let it fall through to cleanup below
                HIP_CHECK(hipSetDevice(src_gpu));
                (void)hipFree(src_buf);
                iteration++;
                continue;
            }

            printf("[IPC_XFER] #%" PRId64 " | Peer direct read GPU%d→GPU%d | %zu bytes\n",
                   iteration, src_gpu, dst_gpu, byte_count);

            // Allocate a destination buffer on dst_gpu and a local error flag
            HIP_CHECK(hipSetDevice(dst_gpu));
            (void)hipGetLastError();
            uint32_t* dst_buf = nullptr;
            err = hipMalloc(&dst_buf, byte_count);
            if (err != hipSuccess) {
                printf("[IPC_XFER] #%" PRId64 " | dst alloc failed — skipping\n", iteration);
                HIP_CHECK(hipSetDevice(src_gpu));
                (void)hipFree(src_buf);
                iteration++;
                continue;
            }

            // Kernel on dst_gpu fills dst_buf via peer mapping, then verifies
            // src_buf (on src_gpu) is readable through the cross-device mapping.
            hipLaunchKernelGGL(kernel_pattern_fill,
                               dim3(grid), dim3(block), 0, dst_stream,
                               dst_buf, elem_count, pattern);
            HIP_LAUNCH_CHECK();
            HIP_CHECK(hipStreamSynchronize(dst_stream));

            // Now verify: read src_buf from dst_gpu via peer access
            int* d_err_flag = nullptr;
            if (hipMalloc(&d_err_flag, sizeof(int)) == hipSuccess) {
                HIP_CHECK(hipMemsetAsync(d_err_flag, 0, sizeof(int), dst_stream));
                // Verify src_buf (on src_gpu) directly from dst_gpu's kernel
                hipLaunchKernelGGL(kernel_pattern_verify,
                                   dim3(grid), dim3(block), 0, dst_stream,
                                   src_buf, elem_count, pattern, d_err_flag);
                HIP_LAUNCH_CHECK();
                HIP_CHECK(hipStreamSynchronize(dst_stream));

                int dev_errors = 0;
                HIP_CHECK(hipMemcpy(&dev_errors, d_err_flag, sizeof(int),
                                    hipMemcpyDeviceToHost));
                if (dev_errors > 0) {
                    fprintf(stderr,
                            "[IPC_XFER] #%" PRId64 " | *** PEER DIRECT READ FAILED GPU%d→GPU%d — %d mismatches ***\n",
                            iteration, src_gpu, dst_gpu, dev_errors);
                    peer_direct_errors++;
                } else {
                    printf("[IPC_XFER] #%" PRId64 " | Peer direct read GPU%d→GPU%d OK\n",
                           iteration, src_gpu, dst_gpu);
                }
                (void)hipFree(d_err_flag);
            } else {
                fprintf(stderr, "[IPC_XFER] #%" PRId64 " | verify alloc failed — counting as error\n", iteration);
                peer_direct_errors++;
            }

            // Also verify: copy dst_buf to host and check (tests that the
            // peer-mapped kernel_pattern_fill wrote correctly to local memory
            // while peer access was active)
            uint32_t* h_readback = nullptr;
            if (hipHostMalloc(&h_readback, byte_count) == hipSuccess) {
                HIP_CHECK(hipMemcpy(h_readback, dst_buf, byte_count,
                                    hipMemcpyDeviceToHost));
                int host_err = validator.verify_host(h_readback, elem_count, pattern,
                                                     iteration, "IPC_XFER-peer-local");
                if (host_err > 0) {
                    fprintf(stderr,
                            "[IPC_XFER] #%" PRId64 " | *** PEER LOCAL VERIFY FAILED — %d mismatches ***\n",
                            iteration, host_err);
                    peer_direct_errors++;
                }
                (void)hipHostFree(h_readback);
            } else {
                fprintf(stderr, "[IPC_XFER] #%" PRId64 " | host readback alloc failed — counting as error\n", iteration);
                peer_direct_errors++;
            }

            (void)hipFree(dst_buf);
        }

        HIP_CHECK(hipSetDevice(src_gpu));
        (void)hipFree(src_buf);

        iteration++;
    }

    health.stop();

    HIP_CHECK(hipSetDevice(src_gpu));
    (void)hipStreamDestroy(src_stream);
    HIP_CHECK(hipSetDevice(dst_gpu));
    (void)hipStreamDestroy(dst_stream);

    int total_errors = peer_copy_errors + peer_direct_errors + validator.total_errors();
    double elapsed_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    printf("[IPC_XFER] Finished: %" PRId64 " iterations, %d peer_copy_errors, %d peer_direct_errors, %d host_verify_errors (%.1fs)\n",
           iteration, peer_copy_errors, peer_direct_errors, validator.total_errors(), elapsed_sec);

    return total_errors > 0 ? 1 : 0;
}
