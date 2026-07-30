// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "roles/library.h"
#include "common/hip_check.h"
#include "common/health.h"

#include <hip/hip_runtime.h>
#include <cinttypes>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <unistd.h>

#ifdef HAS_HIPBLASLT
#include <hipblaslt/hipblaslt.h>

#define HIPBLASLT_CHECK(call)                                                  \
    do {                                                                        \
        hipblasStatus_t stat = (call);                                          \
        if (stat != HIPBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "[HIPBLASLT ERROR] %s:%d — %s returned %d\n",      \
                    __FILE__, __LINE__, #call, stat);                            \
            total_errors++;                                                     \
            goto iter_cleanup;                                                  \
        }                                                                       \
    } while (0)
#endif

// Exercises: hipBLASLt → algorithm heuristic search → Tensile/kernel dispatch
//            → HIP → ROCr → KFD → HW.
//
// Uses the descriptor-based hipBLASLt API (same API that PyTorch/JAX use) for
// GEMM operations. Each iteration creates and destroys all descriptors, layouts,
// and preference objects — churning the descriptor lifecycle to detect leaks.
// Cycles through multiple heuristic-selected algorithms across iterations.
// Verifies results against a simple CPU reference.
//
// When other roles run concurrently, this stresses:
//   - hipBLASLt workspace allocation under VRAM pressure from MEMORY_MOVER
//   - Heuristic/algorithm selection while GPU resources are contested
//   - Code object cache coherency while COMPILER loads/unloads modules
//   - HW queue scheduling fairness with COMPUTE's kernel launches

// CPU reference: column-major C = alpha * A * B + beta * C
// A is M×K (col-major: M rows, K cols, stride M between columns)
// B is K×N (col-major: K rows, N cols, stride K between columns)
// C is M×N (col-major: M rows, N cols, stride M between columns)
static void cpu_sgemm_colmajor(int M, int N, int K, float alpha,
                               const float* A, int lda,
                               const float* B, int ldb,
                               float beta, float* C, int ldc) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[k * lda + i] * B[j * ldb + k];
            }
            C[j * ldc + i] = alpha * sum + beta * C[j * ldc + i];
        }
    }
}

static constexpr size_t MB = 1024ULL * 1024;

int run_library(const RoleConfig& config) {
#ifndef HAS_HIPBLASLT
    printf("[LIBRARY] hipBLASLt not available — skipping\n");
    return 0;
#else
    HIP_CHECK(hipSetDevice(config.gpu_id));

    printf("[LIBRARY] PID %d | GPU %d | duration %ds | using hipBLASLt\n",
           getpid(), config.gpu_id, config.duration_sec);

    HealthMonitor health(config.gpu_id, config.results_dir,
                         config.rss_growth_warn_kb, config.fd_growth_warn);
    health.start();

    hipblasLtHandle_t handle = nullptr;
    hipStream_t stream = nullptr;

    {
        hipblasStatus_t stat = hipblasLtCreate(&handle);
        if (stat != HIPBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[HIPBLASLT ERROR] hipblasLtCreate failed: %d\n", stat);
            return 1;
        }
    }
    HIP_CHECK(hipStreamCreate(&stream));

    struct GemmSize { int M, N, K; };
    const std::vector<GemmSize> sizes = {
        {1, 1, 1},
        {1, 256, 256},
        {256, 1, 256},
        {7, 13, 17},
        {64, 64, 64},
        {128, 128, 128},
        {255, 255, 255},
        {256, 256, 256},
        {333, 444, 555},
        {512, 512, 512},
        {1024, 1024, 1024},
        {1023, 1025, 1024},
    };

    auto start_time = std::chrono::steady_clock::now();
    int64_t iteration = 0;
    int total_errors = 0;

    while (true) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
                >= config.duration_sec) {
            break;
        }

        const auto& sz = sizes[iteration % sizes.size()];
        int M = sz.M, N = sz.N, K = sz.K;
        float alpha = 1.0f, beta = 0.0f;

        printf("[LIBRARY] #%" PRId64 " | SGEMM M=%d N=%d K=%d | Computing CPU reference...\n",
               iteration, M, N, K);

        // Column-major storage: A(M×K) ld=M, B(K×N) ld=K, C/D(M×N) ld=M
        int64_t lda = M;
        int64_t ldb = K;
        int64_t ldc = M;
        int64_t ldd = M;

        size_t size_A = (size_t)lda * K;
        size_t size_B = (size_t)ldb * N;
        size_t size_C = (size_t)ldc * N;

        std::vector<float> h_A(size_A), h_B(size_B), h_C(size_C, 0.0f), h_D(size_C);
        std::vector<float> h_C_ref(size_C, 0.0f);

        for (size_t i = 0; i < size_A; i++)
            h_A[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.1f;
        for (size_t i = 0; i < size_B; i++)
            h_B[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.1f;

        cpu_sgemm_colmajor(M, N, K, alpha,
                           h_A.data(), lda, h_B.data(), ldb,
                           beta, h_C_ref.data(), ldc);

        // All per-iteration variables declared before any goto, per C++ rules.
        float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_D = nullptr;
        void* d_workspace = nullptr;
        hipblasLtMatmulDesc_t matmulDesc = nullptr;
        hipblasLtMatrixLayout_t layoutA = nullptr, layoutB = nullptr;
        hipblasLtMatrixLayout_t layoutC = nullptr, layoutD = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;
        size_t max_workspace = 32 * MB;
        const int max_algos = 4;
        hipblasLtMatmulHeuristicResult_t heurResults[4];
        int returnedAlgoCount = 0;
        hipblasStatus_t heur_status = HIPBLAS_STATUS_NOT_INITIALIZED;
        int algo_idx = 0;
        size_t ws_size = 0;
        hipblasStatus_t matmul_stat = HIPBLAS_STATUS_NOT_INITIALIZED;

        printf("[LIBRARY] #%" PRId64 " | Uploading matrices to GPU...\n", iteration);
        {
            (void)hipGetLastError();
            hipError_t e;
            e = hipMalloc(&d_A, size_A * sizeof(float));
            if (e != hipSuccess) { fprintf(stderr, "[LIBRARY] #%" PRId64 " | hipMalloc d_A failed: %s\n", iteration, hipGetErrorString(e)); total_errors++; goto iter_cleanup; }
            e = hipMalloc(&d_B, size_B * sizeof(float));
            if (e != hipSuccess) { fprintf(stderr, "[LIBRARY] #%" PRId64 " | hipMalloc d_B failed: %s\n", iteration, hipGetErrorString(e)); total_errors++; goto iter_cleanup; }
            e = hipMalloc(&d_C, size_C * sizeof(float));
            if (e != hipSuccess) { fprintf(stderr, "[LIBRARY] #%" PRId64 " | hipMalloc d_C failed: %s\n", iteration, hipGetErrorString(e)); total_errors++; goto iter_cleanup; }
            e = hipMalloc(&d_D, size_C * sizeof(float));
            if (e != hipSuccess) { fprintf(stderr, "[LIBRARY] #%" PRId64 " | hipMalloc d_D failed: %s\n", iteration, hipGetErrorString(e)); total_errors++; goto iter_cleanup; }

            e = hipMemcpy(d_A, h_A.data(), size_A * sizeof(float), hipMemcpyHostToDevice);
            if (e != hipSuccess) { fprintf(stderr, "[LIBRARY] #%" PRId64 " | hipMemcpy d_A failed: %s\n", iteration, hipGetErrorString(e)); total_errors++; goto iter_cleanup; }
            e = hipMemcpy(d_B, h_B.data(), size_B * sizeof(float), hipMemcpyHostToDevice);
            if (e != hipSuccess) { fprintf(stderr, "[LIBRARY] #%" PRId64 " | hipMemcpy d_B failed: %s\n", iteration, hipGetErrorString(e)); total_errors++; goto iter_cleanup; }
            e = hipMemcpy(d_C, h_C.data(), size_C * sizeof(float), hipMemcpyHostToDevice);
            if (e != hipSuccess) { fprintf(stderr, "[LIBRARY] #%" PRId64 " | hipMemcpy d_C failed: %s\n", iteration, hipGetErrorString(e)); total_errors++; goto iter_cleanup; }
        }

        // --- hipBLASLt descriptor-based workflow ---
        // Create and destroy all descriptors every iteration to churn the
        // descriptor lifecycle and test for leaks in hipBLASLt internals.

        HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&matmulDesc,
                        HIPBLAS_COMPUTE_32F, HIP_R_32F));

        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_32F,
                        M, K, lda));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_32F,
                        K, N, ldb));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_32F,
                        M, N, ldc));
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&layoutD, HIP_R_32F,
                        M, N, ldd));

        HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));

        HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref,
                        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                        &max_workspace, sizeof(max_workspace)));

        printf("[LIBRARY] #%" PRId64 " | Querying hipBLASLt heuristics...\n", iteration);
        heur_status = hipblasLtMatmulAlgoGetHeuristic(
            handle, matmulDesc, layoutA, layoutB, layoutC, layoutD,
            pref, max_algos, heurResults, &returnedAlgoCount);

        if (heur_status != HIPBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
            fprintf(stderr, "[LIBRARY] #%" PRId64 " | *** HEURISTIC FAILED for M=%d N=%d K=%d "
                    "(status=%d, algos=%d) ***\n",
                    iteration, M, N, K, heur_status, returnedAlgoCount);
            total_errors++;
            goto iter_cleanup;
        }

        algo_idx = iteration % returnedAlgoCount;
        ws_size = heurResults[algo_idx].workspaceSize;

        if (ws_size > 0) {
            if (hipMalloc(&d_workspace, ws_size) != hipSuccess) {
                printf("[LIBRARY] #%" PRId64 " | Workspace alloc failed (%zu bytes) — using algo with no workspace\n",
                       iteration, ws_size);
                d_workspace = nullptr;
                ws_size = 0;
                for (int a = 0; a < returnedAlgoCount; a++) {
                    if (heurResults[a].workspaceSize == 0) {
                        algo_idx = a;
                        break;
                    }
                }
            }
        }

        if (ws_size == 0 && heurResults[algo_idx].workspaceSize > 0) {
            printf("[LIBRARY] #%" PRId64 " | No algorithm found with zero workspace — skipping\n", iteration);
            goto iter_cleanup;
        }

        printf("[LIBRARY] #%" PRId64 " | Running hipblasLtMatmul (algo %d/%d, ws=%zu bytes)...\n",
               iteration, algo_idx + 1, returnedAlgoCount, ws_size);

        matmul_stat = hipblasLtMatmul(handle, matmulDesc,
                        &alpha,
                        d_A, layoutA,
                        d_B, layoutB,
                        &beta,
                        d_C, layoutC,
                        d_D, layoutD,
                        &heurResults[algo_idx].algo,
                        d_workspace, ws_size,
                        stream);
        if (matmul_stat != HIPBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[LIBRARY] #%" PRId64 " | hipblasLtMatmul failed: %d\n", iteration, matmul_stat);
            total_errors++;
        }

        {
            hipError_t sync_err = hipStreamSynchronize(stream);
            if (sync_err != hipSuccess) {
                fprintf(stderr, "[LIBRARY] #%" PRId64 " | hipStreamSynchronize failed: %s\n",
                        iteration, hipGetErrorString(sync_err));
                total_errors++;
                goto iter_cleanup;
            }
        }

        if (matmul_stat == HIPBLAS_STATUS_SUCCESS) {
            hipError_t cpy_err = hipMemcpy(h_D.data(), d_D, size_C * sizeof(float),
                                           hipMemcpyDeviceToHost);
            if (cpy_err != hipSuccess) {
                fprintf(stderr, "[LIBRARY] #%" PRId64 " | hipMemcpy D2H failed: %s\n",
                        iteration, hipGetErrorString(cpy_err));
                total_errors++;
                goto iter_cleanup;
            }

            printf("[LIBRARY] #%" PRId64 " | Downloading result and comparing vs CPU...\n", iteration);
            float max_diff = 0.0f;
            int mismatches = 0;
            int inf_nan_count = 0;
            float tolerance = 1e-3f * K;
            for (size_t i = 0; i < size_C; i++) {
                float gpu_val = h_D[i];
                float cpu_val = h_C_ref[i];
                if (std::isinf(cpu_val) || std::isnan(cpu_val) ||
                    std::isinf(gpu_val) || std::isnan(gpu_val)) {
                    if (inf_nan_count < 3) {
                        fprintf(stderr, "[LIBRARY] #%" PRId64 " | *** INF/NAN idx=%zu gpu=%.6f cpu=%.6f ***\n",
                                iteration, i, gpu_val, cpu_val);
                    }
                    inf_nan_count++;
                    mismatches++;
                    continue;
                }
                float diff = std::fabs(gpu_val - cpu_val);
                if (diff > max_diff) max_diff = diff;
                if (diff > tolerance) {
                    if (mismatches < 3) {
                        fprintf(stderr, "[LIBRARY] #%" PRId64 " | *** MISMATCH idx=%zu gpu=%.6f cpu=%.6f diff=%.6f ***\n",
                                iteration, i, gpu_val, cpu_val, diff);
                    }
                    mismatches++;
                }
            }

            if (mismatches > 0) {
                fprintf(stderr, "[LIBRARY] #%" PRId64 " | *** GEMM FAILED M=%d N=%d K=%d — %d/%zu wrong (%d inf/nan), max_diff=%.6f ***\n",
                        iteration, M, N, K, mismatches, size_C, inf_nan_count, max_diff);
                total_errors++;
            }

            printf("[LIBRARY] #%" PRId64 " | SGEMM(%d,%d,%d) algo=%d max_diff=%.6f %s%s\n",
                   iteration, M, N, K, algo_idx, max_diff,
                   (mismatches == 0) ? "OK" : "FAIL",
                   (inf_nan_count > 0) ? " (inf/nan detected)" : "");
        }

    iter_cleanup:
        if (pref) hipblasLtMatmulPreferenceDestroy(pref);
        if (layoutD) hipblasLtMatrixLayoutDestroy(layoutD);
        if (layoutC) hipblasLtMatrixLayoutDestroy(layoutC);
        if (layoutB) hipblasLtMatrixLayoutDestroy(layoutB);
        if (layoutA) hipblasLtMatrixLayoutDestroy(layoutA);
        if (matmulDesc) hipblasLtMatmulDescDestroy(matmulDesc);

        if (d_workspace) (void)hipFree(d_workspace);
        if (d_A) (void)hipFree(d_A);
        if (d_B) (void)hipFree(d_B);
        if (d_C) (void)hipFree(d_C);
        if (d_D) (void)hipFree(d_D);

        iteration++;
    }

    health.stop();
    (void)hipStreamDestroy(stream);
    hipblasLtDestroy(handle);

    double elapsed_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    printf("[LIBRARY] Finished: %" PRId64 " iterations, %d errors (%.1fs)\n",
           iteration, total_errors, elapsed_sec);

    return total_errors > 0 ? 1 : 0;
#endif
}
