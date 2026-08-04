// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "roles/compute.h"
#include "common/hip_check.h"
#include "common/validator.h"
#include "common/health.h"
#include "kernels/kernels.h"

#include <cinttypes>
#include <cstdio>
#include <chrono>
#include <vector>
#include <unistd.h>

// Exercises: HIP runtime → ROCr → KFD → HW compute path.
//
// Runs a continuous loop of kernel launches with varied grid sizes, stream
// counts, and synchronization patterns. Every iteration fills a buffer with a
// unique pattern on-device and reads it back to verify correctness.
//
// When other roles run concurrently, this stresses:
//   - HSA queue scheduling under multi-process contention
//   - Code object dispatch while COMPILER loads/unloads modules
//   - GPU page table coherency while MEMORY_MOVER alloc/frees
//   - Profiling interception while PROFILER wraps dispatches

int run_compute(const RoleConfig& config) {
    int hip_exit_code = 0;

    HIP_CHECK(hipSetDevice(config.gpu_id));

    printf("[COMPUTE] PID %d | GPU %d | duration %ds\n",
           getpid(), config.gpu_id, config.duration_sec);

    Validator validator;
    HealthMonitor health(config.gpu_id, config.results_dir,
                         config.rss_growth_warn_kb, config.fd_growth_warn);
    health.start();

    constexpr int NUM_STREAMS = 4;
    hipStream_t streams[NUM_STREAMS] = {};
    constexpr size_t BUF_ELEMS = 256 * 1024; // 1MB per buffer
    constexpr size_t BUF_BYTES = BUF_ELEMS * sizeof(uint32_t);
    std::vector<uint32_t*> d_bufs(NUM_STREAMS, nullptr);
    std::vector<uint32_t*> h_bufs(NUM_STREAMS, nullptr);
    int* d_error_flag = nullptr;
    auto start_time = std::chrono::steady_clock::now();
    int64_t iteration = 0;
    int device_verify_errors = 0;

    for (int i = 0; i < NUM_STREAMS; i++) {
        HIP_CHECK_OR(hipStreamCreate(&streams[i]), cleanup);
    }
    for (int i = 0; i < NUM_STREAMS; i++) {
        HIP_CHECK_OR(hipMalloc(&d_bufs[i], BUF_BYTES), cleanup);
        HIP_CHECK_OR(hipHostMalloc(&h_bufs[i], BUF_BYTES), cleanup);
    }
    HIP_CHECK_OR(hipMalloc(&d_error_flag, sizeof(int)), cleanup);

    while (true) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
                >= config.duration_sec) {
            break;
        }

        int stream_idx = iteration % NUM_STREAMS;
        hipStream_t stream = streams[stream_idx];

        uint32_t pattern = validator.make_pattern((int)iteration, config.role_id, stream_idx);

        // Vary grid sizes to exercise different dispatch paths
        int block_size;
        int grid_size;
        if (iteration % 3 == 0) {
            block_size = 64;
            grid_size = (BUF_ELEMS + block_size - 1) / block_size;
        } else if (iteration % 3 == 1) {
            block_size = 256;
            grid_size = (BUF_ELEMS + block_size - 1) / block_size;
        } else {
            block_size = 1024;
            grid_size = (BUF_ELEMS + block_size - 1) / block_size;
        }

        bool verbose = (iteration % 10000 == 0);

        if (verbose) {
            printf("[COMPUTE] #%" PRId64 " | stream=%d block=%d grid=%d\n",
                   iteration, stream_idx, block_size, grid_size);
        }

        hipLaunchKernelGGL(kernel_pattern_fill,
                           dim3(grid_size), dim3(block_size), 0, stream,
                           d_bufs[stream_idx], BUF_ELEMS, pattern);
        HIP_LAUNCH_CHECK_OR(cleanup);

        HIP_CHECK_OR(hipMemsetAsync(d_error_flag, 0, sizeof(int), stream), cleanup);
        hipLaunchKernelGGL(kernel_pattern_verify,
                           dim3(grid_size), dim3(block_size), 0, stream,
                           d_bufs[stream_idx], BUF_ELEMS, pattern, d_error_flag);
        HIP_LAUNCH_CHECK_OR(cleanup);

        HIP_CHECK_OR(hipMemcpyAsync(h_bufs[stream_idx], d_bufs[stream_idx],
                                    BUF_BYTES, hipMemcpyDeviceToHost, stream),
                     cleanup);
        HIP_CHECK_OR(hipStreamSynchronize(stream), cleanup);

        int device_errors = 0;
        HIP_CHECK_OR(hipMemcpy(&device_errors, d_error_flag, sizeof(int),
                               hipMemcpyDeviceToHost),
                     cleanup);
        if (device_errors > 0) {
            device_verify_errors += device_errors;
            fprintf(stderr, "[COMPUTE] #%" PRId64 " | *** DEVICE VERIFY FAILED — %d mismatches ***\n",
                    iteration, device_errors);
        }

        int host_mismatches = validator.verify_host(h_bufs[stream_idx], BUF_ELEMS,
                                                    pattern, iteration, "COMPUTE");
        if (verbose || device_errors > 0 || host_mismatches > 0) {
            printf("[COMPUTE] #%" PRId64 " | device_err=%d host_err=%d %s\n",
                   iteration, device_errors, host_mismatches,
                   (device_errors == 0 && host_mismatches == 0) ? "OK" : "FAIL");
        }

        if (iteration % 1000 == 0 && iteration > 0) {
            hipLaunchKernelGGL(kernel_shared_mem_test,
                               dim3(grid_size), dim3(block_size),
                               block_size * sizeof(uint32_t), stream,
                               d_bufs[stream_idx], BUF_ELEMS, pattern);
            HIP_LAUNCH_CHECK_OR(cleanup);
            HIP_CHECK_OR(hipMemcpyAsync(h_bufs[stream_idx], d_bufs[stream_idx],
                                        BUF_BYTES, hipMemcpyDeviceToHost, stream),
                         cleanup);
            HIP_CHECK_OR(hipStreamSynchronize(stream), cleanup);
            int shmem_err = validator.verify_host(h_bufs[stream_idx], BUF_ELEMS, pattern,
                                                   iteration, "COMPUTE-shmem");
            if (verbose || shmem_err > 0) {
                printf("[COMPUTE] #%" PRId64 " | Shared-memory test %s\n",
                       iteration, shmem_err == 0 ? "OK" : "FAIL");
            }
        }

        if (iteration % 5000 == 0 && iteration > 0) {
            HIP_CHECK_OR(hipDeviceSynchronize(), cleanup);
        }

        iteration++;
    }

cleanup:
    health.stop();

    if (d_error_flag) (void)hipFree(d_error_flag);
    for (int i = 0; i < NUM_STREAMS; i++) {
        if (d_bufs[i]) (void)hipFree(d_bufs[i]);
        if (h_bufs[i]) (void)hipHostFree(h_bufs[i]);
        if (streams[i]) (void)hipStreamDestroy(streams[i]);
    }

    double elapsed_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    printf("[COMPUTE] Finished: %" PRId64 " iterations, %d checks, %d errors (%.1fs)\n",
           iteration, validator.total_checks(), validator.total_errors(), elapsed_sec);

    if (hip_exit_code != 0)
        return 1;
    return (validator.total_errors() + device_verify_errors) > 0 ? 1 : 0;
}
