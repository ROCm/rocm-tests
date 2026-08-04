// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "roles/profiler.h"
#include "common/hip_check.h"
#include "common/health.h"
#include "kernels/kernels.h"

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <unistd.h>

// Exercises: HIP event-based timing → ROCr signal infrastructure → KFD →
//            GPU timestamp counters.
//
// Continuously runs kernels bracketed by hipEventRecord / hipEventElapsedTime,
// detecting timing anomalies that indicate GPU scheduling stalls. Periodically
// recycles the event pool to churn HSA signal allocation.
//
// When other roles run concurrently, this stresses:
//   - HSA signal allocation pool (events consume signals)
//   - GPU timestamp counter coherency across concurrent dispatches
//   - GPU command processor scheduling under multi-process contention

int run_profiler(const RoleConfig& config) {
    HIP_CHECK(hipSetDevice(config.gpu_id));

    printf("[PROFILER] PID %d | GPU %d | duration %ds\n",
           getpid(), config.gpu_id, config.duration_sec);

    HealthMonitor health(config.gpu_id, config.results_dir,
                         config.rss_growth_warn_kb, config.fd_growth_warn);
    health.start();

    hipStream_t stream = nullptr;
    constexpr size_t BUF_ELEMS = 128 * 1024;
    constexpr size_t BUF_BYTES = BUF_ELEMS * sizeof(float);
    float* d_buf = nullptr;
    constexpr int EVENT_POOL_SIZE = 64;
    hipEvent_t events_start[EVENT_POOL_SIZE] = {};
    hipEvent_t events_stop[EVENT_POOL_SIZE] = {};
    int events_created = 0;
    FILE* csv = nullptr;
    int hip_exit_code = 0;
    float baseline_ms = 1e9f;
    constexpr int BASELINE_RUNS = 5;
    constexpr int MAX_BASELINE_ATTEMPTS = 3;

    auto start_time = std::chrono::steady_clock::now();
    int64_t iteration = 0;
    int timing_anomalies = 0;
    int severe_anomalies = 0;
    int invalid_times = 0;

    HIP_CHECK_OR(hipStreamCreate(&stream), profiler_cleanup);
    HIP_CHECK_OR(hipMalloc(&d_buf, BUF_BYTES), profiler_cleanup);

    for (int i = 0; i < EVENT_POOL_SIZE; i++) {
        HIP_CHECK_OR(hipEventCreate(&events_start[i]), profiler_cleanup);
        HIP_CHECK_OR(hipEventCreate(&events_stop[i]), profiler_cleanup);
        events_created = i + 1;
    }

    {
        std::string csv_path = config.results_dir + "/profiler_gpu" +
                               std::to_string(config.gpu_id) +
                               "_pid" + std::to_string(getpid()) + ".csv";
        csv = fopen(csv_path.c_str(), "w");
        if (csv) {
            fprintf(csv, "iteration,kernel_type,grid,block,fma_iters,kernel_ms\n");
        }
    }

    // Baseline: run the kernel multiple times and take the minimum.
    // The first run warms up the GPU cache/scheduler. Subsequent runs
    // may overlap with other roles starting up, so the minimum gives
    // the best estimate of uncontested execution time.
    for (int attempt = 0; attempt < MAX_BASELINE_ATTEMPTS; attempt++) {
        baseline_ms = 1e9f;
        HIP_CHECK_OR(hipDeviceSynchronize(), profiler_cleanup);
        for (int b = 0; b < BASELINE_RUNS; b++) {
            hipExtLaunchKernelGGL(kernel_compute_burn,
                                  dim3(512), dim3(256), 0, stream,
                                  events_start[0], events_stop[0], 0,
                                  d_buf, BUF_ELEMS, 100);
            HIP_LAUNCH_CHECK_OR(profiler_cleanup);
            HIP_CHECK_OR(hipStreamSynchronize(stream), profiler_cleanup);
            float ms = 0.0f;
            HIP_CHECK_OR(hipEventElapsedTime(&ms, events_start[0], events_stop[0]), profiler_cleanup);
            printf("[PROFILER] Baseline run %d/%d: %.3f ms\n", b + 1, BASELINE_RUNS, ms);
            if (ms > 0.0f && ms < baseline_ms) baseline_ms = ms;
        }
        printf("[PROFILER] Baseline attempt %d/%d (min of %d runs): %.3f ms\n",
               attempt + 1, MAX_BASELINE_ATTEMPTS, BASELINE_RUNS, baseline_ms);

        if (baseline_ms > 0.0f && baseline_ms < 1e9f) break;

        fprintf(stderr, "[PROFILER] Baseline attempt %d/%d returned no valid times — retrying in 2s\n",
                attempt + 1, MAX_BASELINE_ATTEMPTS);
        sleep(2);
    }

    if (baseline_ms >= 1e9f || baseline_ms <= 0.0f) {
        fprintf(stderr, "[PROFILER] *** FATAL: no valid baseline after %d attempts — GPU timing infrastructure broken ***\n",
                MAX_BASELINE_ATTEMPTS);
        hip_exit_code = 1;
        goto profiler_cleanup;
    }

    // Signal that baseline is done — scripts can wait for this file
    // before launching other roles, guaranteeing an isolated measurement.
    {
        std::string signal = config.results_dir + "/profiler_baseline_done";
        FILE* f = fopen(signal.c_str(), "w");
        if (f) {
            fprintf(f, "%.4f\n", baseline_ms);
            fflush(f);
            fsync(fileno(f));
            fclose(f);
        }
    }

    while (true) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
                >= config.duration_sec) {
            break;
        }

        int event_idx = iteration % EVENT_POOL_SIZE;

        // Vary kernel configurations
        int fma_iters;
        int grid_size, block_size;
        const char* kernel_type;

        switch (iteration % 4) {
            case 0: // Small fast kernel
                grid_size = 128; block_size = 64; fma_iters = 10;
                kernel_type = "small";
                break;
            case 1: // Medium kernel
                grid_size = 512; block_size = 256; fma_iters = 100;
                kernel_type = "medium";
                break;
            case 2: // Large kernel
                grid_size = 1024; block_size = 256; fma_iters = 1000;
                kernel_type = "large";
                break;
            default: // Same as baseline for comparison
                grid_size = 512; block_size = 256; fma_iters = 100;
                kernel_type = "baseline";
                break;
        }

        hipExtLaunchKernelGGL(kernel_compute_burn,
                              dim3(grid_size), dim3(block_size), 0, stream,
                              events_start[event_idx], events_stop[event_idx], 0,
                              d_buf, BUF_ELEMS, fma_iters);
        HIP_LAUNCH_CHECK_OR(profiler_cleanup);
        HIP_CHECK_OR(hipStreamSynchronize(stream), profiler_cleanup);

        float kernel_ms = 0.0f;
        HIP_CHECK_OR(hipEventElapsedTime(&kernel_ms, events_start[event_idx],
                                         events_stop[event_idx]), profiler_cleanup);

        bool anomaly = false;
        if (strcmp(kernel_type, "baseline") == 0 && baseline_ms > 0) {
            float ratio = kernel_ms / baseline_ms;
            if (ratio > config.anomaly_ratio) {
                timing_anomalies++;
                if (ratio > config.severe_ratio) severe_anomalies++;
                anomaly = true;
                if (timing_anomalies <= 100) {
                    fprintf(stderr, "[PROFILER] #%" PRId64 " | *** TIMING ANOMALY: expected ~%.3fms got %.3fms (%.1fx) ***\n",
                            iteration, baseline_ms, kernel_ms, ratio);
                }
            }
        }

        if (kernel_ms <= 0.0f) {
            invalid_times++;
            anomaly = true;
            if (invalid_times <= 100) {
                fprintf(stderr, "[PROFILER] #%" PRId64 " | *** INVALID TIME: %.4fms (timing infrastructure bug?) ***\n",
                        iteration, kernel_ms);
            }
        }

        // Cap CSV at 10,000 anomaly rows — enough evidence to diagnose,
        // prevents multi-GB file growth under sustained contention.
        if (anomaly && csv && timing_anomalies <= 10000) {
            fprintf(csv, "%" PRId64 ",%s,%d,%d,%d,%.4f\n",
                    iteration, kernel_type, grid_size, block_size,
                    fma_iters, kernel_ms);
        }

        // Adaptive progress interval: every 10k up to 1M, then every 100k.
        // Prevents 4 GB stdout logs on long runs (100M+ iterations).
        int progress_interval = (iteration < 1000000) ? 10000 : 100000;
        if (iteration % progress_interval == 0) {
            printf("[PROFILER] #%" PRId64 " | %s kernel = %.3fms | anomalies so far: %d (%d severe)\n",
                   iteration, kernel_type, kernel_ms, timing_anomalies, severe_anomalies);
            if (csv) fflush(csv);
        }

        if (iteration % 1000 == 0 && iteration > 0) {
            for (int i = 0; i < EVENT_POOL_SIZE; i++) {
                HIP_CHECK_OR(hipEventDestroy(events_start[i]), profiler_cleanup);
                HIP_CHECK_OR(hipEventDestroy(events_stop[i]), profiler_cleanup);
                HIP_CHECK_OR(hipEventCreate(&events_start[i]), profiler_cleanup);
                HIP_CHECK_OR(hipEventCreate(&events_stop[i]), profiler_cleanup);
            }
        }

        iteration++;
    }

    health.stop();

profiler_cleanup:
    if (csv) fclose(csv);

    for (int i = 0; i < events_created; i++) {
        (void)hipEventDestroy(events_start[i]);
        (void)hipEventDestroy(events_stop[i]);
    }
    if (d_buf) (void)hipFree(d_buf);
    if (stream) (void)hipStreamDestroy(stream);

    float anomaly_pct = (iteration > 0) ? (100.0f * timing_anomalies / iteration) : 0;
    double elapsed_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    float severe_pct = (iteration > 0) ? (100.0f * severe_anomalies / iteration) : 0;
    printf("[PROFILER] Finished: %" PRId64 " iterations, %d anomalies (>%.0fx), %d severe (>%.0fx), %d invalid, %.4f%% anomaly rate (%.1fs)\n",
           iteration, timing_anomalies, config.anomaly_ratio, severe_anomalies, config.severe_ratio,
           invalid_times, anomaly_pct, elapsed_sec);

    bool fail = anomaly_pct > config.anomaly_pct_limit
             || severe_pct > config.severe_pct_limit
             || invalid_times > 0;
    printf("[PROFILER] Verdict: %s (anomaly=%.4f%%/%.3f%%, severe=%.6f%%/%.4f%%, invalid=%d)\n",
           fail ? "FAIL" : "PASS", anomaly_pct, config.anomaly_pct_limit,
           severe_pct, config.severe_pct_limit, invalid_times);
    return (fail || hip_exit_code != 0) ? 1 : 0;
}
