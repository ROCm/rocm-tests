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
// SPDX-License-Identifier: MIT
/**
 * Small-Sliding Contact Solver via rocBLAS strided-batched operations.
 *
 * Real penalty contact Newton-Raphson step:
 *
 *   1. Build NxN SPD contact stiffness per node:
 *      K_i = k_n * (n*n^T) + k_t * (I - n*n^T)
 *      Normal direction uses 3D geometry, padded to N for N>3.
 *   2. Contact force:       f_i = K_i * g_i              (sgemv)
 *   3. Cholesky solve:      K_i * dd_i = f_i             (strsv x2)
 *      3a. Forward:   L_i * y_i  = f_i
 *      3b. Backward:  L_i^T * dd_i = y_i
 *   4. After one Newton step: dd_i == g_i, so gap -> 0
 *
 * rocBLAS calls (all n=N, batch_count=BATCH_COUNT):
 *   rocblas_sgemv_strided_batched      (contact force)
 *   rocblas_strsv_strided_batched x2   (Cholesky forward + backward)
 *
 * The big-batch heuristic triggers when N < 128 AND BATCH_COUNT > 16*N.
 */

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Error-check macros
// ---------------------------------------------------------------------------
#define HIP_CHECK(stat)                                                      \
    do {                                                                     \
        hipError_t err = (stat);                                             \
        if (err != hipSuccess) {                                             \
            fprintf(stderr, "[HIP ERROR] %s:%d  %s\n",                      \
                    __FILE__, __LINE__, hipGetErrorString(err));              \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define ROCBLAS_CHECK(stat)                                                  \
    do {                                                                     \
        rocblas_status err = (stat);                                         \
        if (err != rocblas_status_success) {                                 \
            fprintf(stderr, "[rocBLAS ERROR] %s:%d  status %d\n",            \
                    __FILE__, __LINE__, (int)err);                           \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Problem parameters (N and BATCH_COUNT overridable via command line)
//
// Default: N=3, BATCH_COUNT=100000
// Usage:   ./small_sliding_contact [N] [BATCH_COUNT]
// ---------------------------------------------------------------------------
static int   N           = 3;
static int   LDA         = 3;
static int   BATCH_COUNT = 100000;
static float K_N         = 1e4f;     // normal penalty stiffness
static float K_T         = 1e3f;     // tangential stiffness (friction)
constexpr int INCX       = 1;

constexpr float TOL = 1e-4f;

// ---------------------------------------------------------------------------
// Vec3 helpers (host only — used for 3D geometry generation)
// ---------------------------------------------------------------------------
struct Vec3 { float x, y, z; };

static inline Vec3  v_sub(Vec3 a, Vec3 b)    { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
static inline Vec3  v_add(Vec3 a, Vec3 b)    { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
static inline Vec3  v_scale(Vec3 a, float s) { return {a.x*s, a.y*s, a.z*s}; }
static inline float v_dot(Vec3 a, Vec3 b)    { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline Vec3  v_cross(Vec3 a, Vec3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
static inline Vec3 v_normalize(Vec3 a) {
    float l = sqrtf(v_dot(a, a));
    return (l > 1e-12f) ? v_scale(a, 1.0f / l) : Vec3{0, 0, 0};
}
static inline float clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

// ---------------------------------------------------------------------------
// CPU: project point onto triangle (barycentric)
// ---------------------------------------------------------------------------
static Vec3 project_cpu(Vec3 p, Vec3 v0, Vec3 v1, Vec3 v2)
{
    Vec3 e0 = v_sub(v1, v0);
    Vec3 e1 = v_sub(v2, v0);
    Vec3 ep = v_sub(p, v0);

    float d00 = v_dot(e0, e0), d01 = v_dot(e0, e1), d11 = v_dot(e1, e1);
    float d20 = v_dot(ep, e0), d21 = v_dot(ep, e1);

    float denom = d00 * d11 - d01 * d01;
    float inv   = (fabsf(denom) > 1e-12f) ? 1.0f / denom : 0.0f;

    float u = (d11 * d20 - d01 * d21) * inv;
    float v = (d00 * d21 - d01 * d20) * inv;
    u = clampf(u, 0.0f, 1.0f);
    v = clampf(v, 0.0f, 1.0f - u);

    return v_add(v0, v_add(v_scale(e0, u), v_scale(e1, v)));
}

// ---------------------------------------------------------------------------
// CPU: forward-substitution  L * x = b  (Lower, NoTrans, NonUnit, in-place)
// ---------------------------------------------------------------------------
static void cpu_trsv_lower(const float* L, float* x, int n, int lda)
{
    for (int i = 0; i < n; ++i) {
        float sum = x[i];
        for (int j = 0; j < i; ++j)
            sum -= L[j * lda + i] * x[j];
        x[i] = sum / L[i * lda + i];
    }
}

// ---------------------------------------------------------------------------
// CPU: back-substitution  L^T * x = b  (Lower, Trans, NonUnit, in-place)
// ---------------------------------------------------------------------------
static void cpu_trsv_lower_trans(const float* L, float* x, int n, int lda)
{
    for (int i = n - 1; i >= 0; --i) {
        float sum = x[i];
        for (int j = i + 1; j < n; ++j)
            sum -= L[i * lda + j] * x[j];
        x[i] = sum / L[i * lda + i];
    }
}

// ---------------------------------------------------------------------------
// CPU: matrix-vector product  y = A * x  (col-major)
// ---------------------------------------------------------------------------
static void cpu_gemv(const float* A, const float* x, float* y, int n, int lda)
{
    for (int r = 0; r < n; ++r) {
        float s = 0.0f;
        for (int c = 0; c < n; ++c)
            s += A[c * lda + r] * x[c];
        y[r] = s;
    }
}

// ---------------------------------------------------------------------------
// Host: NxN Cholesky  K = L * L^T  (in-place, col-major)
// ---------------------------------------------------------------------------
static void choleskyN(float* A, int n, int lda)
{
    for (int j = 0; j < n; ++j) {
        float sum = A[j * lda + j];
        for (int k = 0; k < j; ++k)
            sum -= A[k * lda + j] * A[k * lda + j];
        A[j * lda + j] = sqrtf(sum);

        for (int i = j + 1; i < n; ++i) {
            sum = A[j * lda + i];
            for (int k = 0; k < j; ++k)
                sum -= A[k * lda + i] * A[k * lda + j];
            A[j * lda + i] = sum / A[j * lda + j];
        }

        // zero upper triangle in this column
        for (int i = 0; i < j; ++i)
            A[j * lda + i] = 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Overflow-safe size_t arithmetic for memory pre-flight
// ---------------------------------------------------------------------------
static bool safe_mul(size_t a, size_t b, size_t& out)
{
    if (a != 0 && b > SIZE_MAX / a) { out = 0; return false; }
    out = a * b;
    return true;
}

static bool safe_add(size_t a, size_t b, size_t& out)
{
    if (a > SIZE_MAX - b) { out = 0; return false; }
    out = a + b;
    return true;
}

// Usage string for validation failure
static const char* const USAGE =
    "Usage: %s [N] [BATCH_COUNT]\n"
    "  N           matrix dimension (default 3), must be 1..11264\n"
    "  BATCH_COUNT batch size (default 100000), must be 1..100000000\n";

// ===================================================================
int main(int argc, char* argv[])
{
    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) BATCH_COUNT = atoi(argv[2]);
    LDA = N;

    constexpr int N_MAX = 11264;
    constexpr int BATCH_COUNT_MAX = 100000000;
    if (N < 1 || N > N_MAX || BATCH_COUNT < 1 || BATCH_COUNT > BATCH_COUNT_MAX) {
        fprintf(stderr, USAGE, argc > 0 ? argv[0] : "small_sliding_contact");
        fprintf(stderr, "Error: N=%d (must be 1..%d), BATCH_COUNT=%d (must be 1..%d)\n",
                N, N_MAX, BATCH_COUNT, BATCH_COUNT_MAX);
        return EXIT_FAILURE;
    }

    const rocblas_stride STRIDE_A = static_cast<rocblas_stride>(LDA) * N;
    const rocblas_stride STRIDE_X = N;

    // ------------------------------------------------------------------
    // Memory pre-flight: overflow-safe estimate vs. available GPU memory
    //
    //   Host:   h_K, h_L  (2 × N²×B)  +  h_gap, h_f_ref, h_dd_ref, h_dd_gpu (4 × N×B)
    //   Device: d_K, d_L  (2 × N²×B)  +  d_gap, d_work (2 × N×B)
    // ------------------------------------------------------------------
    {
        const size_t sN  = static_cast<size_t>(N);
        const size_t sB  = static_cast<size_t>(BATCH_COUNT);
        const size_t esz = sizeof(float);

        size_t n_sq, mat_pool, vec_pool;
        size_t two_mat, four_vec, two_vec;
        size_t host_elems, dev_elems, host_bytes, dev_bytes;
        bool ok = true;

        ok = ok && safe_mul(sN, sN, n_sq);
        ok = ok && safe_mul(n_sq, sB, mat_pool);
        ok = ok && safe_mul(sN, sB, vec_pool);
        ok = ok && safe_mul(mat_pool, 2, two_mat);
        ok = ok && safe_mul(vec_pool, 4, four_vec);
        ok = ok && safe_mul(vec_pool, 2, two_vec);
        ok = ok && safe_add(two_mat, four_vec, host_elems);
        ok = ok && safe_add(two_mat, two_vec,  dev_elems);
        ok = ok && safe_mul(host_elems, esz, host_bytes);
        ok = ok && safe_mul(dev_elems,  esz, dev_bytes);

        if (!ok) {
            fprintf(stderr,
                    "Error: Buffer size overflows for N=%d, BATCH_COUNT=%d.\n"
                    "Reduce N or BATCH_COUNT.\n", N, BATCH_COUNT);
            return EXIT_FAILURE;
        }

        size_t gpu_free = 0, gpu_total = 0;
        HIP_CHECK(hipMemGetInfo(&gpu_free, &gpu_total));

        const double GiB = 1024.0 * 1024.0 * 1024.0;
        printf("Memory estimate:  host %.2f GiB,  device %.2f GiB\n",
               host_bytes / GiB, dev_bytes / GiB);
        printf("GPU memory:       free %.2f GiB / %.2f GiB total\n\n",
               gpu_free / GiB, gpu_total / GiB);

        if (dev_bytes > gpu_free) {
            fprintf(stderr,
                    "Error: Device memory required (%.2f GiB) exceeds available GPU memory "
                    "(%.2f GiB free / %.2f GiB total).\n"
                    "Reduce N or BATCH_COUNT.\n",
                    dev_bytes / GiB, gpu_free / GiB, gpu_total / GiB);
            return EXIT_FAILURE;
        }
    }

    printf("=== Small-Sliding Contact Solver (rocBLAS batched) ===\n");
    printf("N = %d, batch_count = %d  (threshold 16*N = %d)\n",
           N, BATCH_COUNT, 16 * N);
    printf("k_n = %.0f, k_t = %.0f\n", K_N, K_T);
    if (BATCH_COUNT > 16 * N && N < 128)
        printf("Expected kernel path: BIG BATCH\n\n");
    else
        printf("Expected kernel path: REGULAR\n\n");

    // ------------------------------------------------------------------
    // 1.  Build contact geometry and prepare all batched arrays
    //
    //     For each contact pair b:
    //       - 3D geometry -> normal n (padded to N-dim)
    //       - K_i = k_t*I + (k_n - k_t)*n*n^T    (NxN SPD)
    //       - Cholesky K_i -> L_i                  (NxN lower tri)
    //       - gap vector from projection (padded to N-dim)
    // ------------------------------------------------------------------
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pos(-10.0f, 10.0f);
    std::uniform_real_distribution<float> off(-0.5f, 0.5f);

    std::vector<float> h_K(STRIDE_A * BATCH_COUNT, 0.0f);
    std::vector<float> h_L(STRIDE_A * BATCH_COUNT, 0.0f);
    std::vector<float> h_gap(STRIDE_X * BATCH_COUNT, 0.0f);

    // Reused buffer for N-dim normal (avoids BATCH_COUNT heap allocations)
    std::vector<float> nn(N, 0.0f);

    int penetrating = 0;
    for (int b = 0; b < BATCH_COUNT; ++b) {
        float bx = pos(rng), by = pos(rng), bz = pos(rng);
        Vec3 v0 = {bx, by, bz};
        Vec3 v1 = {bx + 1.0f + off(rng), by + off(rng),          bz + off(rng)};
        Vec3 v2 = {bx + off(rng),          by + 1.0f + off(rng), bz + off(rng)};

        Vec3 e0 = v_sub(v1, v0), e1 = v_sub(v2, v0);
        Vec3 n3 = v_normalize(v_cross(e0, e1));

        Vec3  centroid = v_scale(v_add(v_add(v0, v1), v2), 1.0f / 3.0f);
        float pen      = off(rng) * 0.2f;
        Vec3  slave    = v_add(centroid, v_scale(n3, pen));

        Vec3  proj    = project_cpu(slave, v0, v1, v2);
        Vec3  gap3    = v_sub(slave, proj);
        float gap_n   = v_dot(gap3, n3);
        if (gap_n < 0.0f) ++penetrating;

        // N-dimensional normal: first 3 from geometry, rest zero (overwrite buffer)
        nn[0] = n3.x;
        if (N > 1) nn[1] = n3.y;
        if (N > 2) nn[2] = n3.z;
        for (int d = 3; d < N; ++d) nn[d] = 0.0f;

        // gap vector: first 3 from projection, rest small random
        float* gi = &h_gap[b * STRIDE_X];
        gi[0] = gap3.x;
        if (N > 1) gi[1] = gap3.y;
        if (N > 2) gi[2] = gap3.z;
        for (int d = 3; d < N; ++d)
            gi[d] = off(rng) * 0.01f;

        // K = k_t * I + (k_n - k_t) * n*n^T  (col-major NxN)
        float dk  = K_N - K_T;
        float* Ki = &h_K[b * STRIDE_A];
        for (int c = 0; c < N; ++c)
            for (int r = 0; r < N; ++r)
                Ki[c * LDA + r] = K_T * (r == c ? 1.0f : 0.0f)
                                + dk * nn[r] * nn[c];

        // L = cholesky(K)
        float* Li = &h_L[b * STRIDE_A];
        for (int k = 0; k < N * N; ++k)
            Li[k] = Ki[k];
        choleskyN(Li, N, LDA);
    }

    printf("%d / %d contacts penetrating\n", penetrating, BATCH_COUNT);

    if (N <= 8 && BATCH_COUNT <= 8) {
        printf("\nK[0] (row view):\n");
        for (int r = 0; r < N; ++r) {
            printf("  ");
            for (int c = 0; c < N; ++c)
                printf("%10.2f", h_K[c * LDA + r]);
            printf("\n");
        }
        printf("L[0] (row view):\n");
        for (int r = 0; r < N; ++r) {
            printf("  ");
            for (int c = 0; c < N; ++c)
                printf("%10.4f", h_L[c * LDA + r]);
            printf("\n");
        }
    }
    printf("gap[0..%d] = [", std::min(N, 4) - 1);
    for (int k = 0; k < std::min(N, 4); ++k)
        printf("%.6f%s", h_gap[k], k < std::min(N, 4) - 1 ? ", " : "");
    printf(" ...]\n\n");

    // ------------------------------------------------------------------
    // 2.  CPU reference: full Newton step
    //       f  = K * gap         (contact force)
    //       dd = K^{-1} * f      (Cholesky solve)
    //     Since K^{-1}*K = I, dd should exactly equal gap.
    // ------------------------------------------------------------------
    std::vector<float> h_f_ref(STRIDE_X * BATCH_COUNT);
    std::vector<float> h_dd_ref(STRIDE_X * BATCH_COUNT);

    for (int b = 0; b < BATCH_COUNT; ++b) {
        const float* Ki = &h_K[b * STRIDE_A];
        const float* Li = &h_L[b * STRIDE_A];
        const float* gi = &h_gap[b * STRIDE_X];

        cpu_gemv(Ki, gi, &h_f_ref[b * STRIDE_X], N, LDA);

        for (int k = 0; k < N; ++k)
            h_dd_ref[b * STRIDE_X + k] = h_f_ref[b * STRIDE_X + k];
        cpu_trsv_lower(Li,  &h_dd_ref[b * STRIDE_X], N, LDA);
        cpu_trsv_lower_trans(Li, &h_dd_ref[b * STRIDE_X], N, LDA);
    }

    printf("CPU ref  force[0..%d] = [", std::min(N, 4) - 1);
    for (int k = 0; k < std::min(N, 4); ++k)
        printf("%.4f%s", h_f_ref[k], k < std::min(N, 4) - 1 ? ", " : "");
    printf(" ...]\n");

    printf("CPU ref  dd[0..%d]    = [", std::min(N, 4) - 1);
    for (int k = 0; k < std::min(N, 4); ++k)
        printf("%.6f%s", h_dd_ref[k], k < std::min(N, 4) - 1 ? ", " : "");
    printf(" ...]\n");

    float max_identity_err = 0.0f;
    for (int b = 0; b < BATCH_COUNT; ++b)
        for (int k = 0; k < N; ++k)
            max_identity_err = fmaxf(max_identity_err,
                fabsf(h_dd_ref[b * STRIDE_X + k] - h_gap[b * STRIDE_X + k]));
    printf("CPU sanity ||dd - gap||_inf = %e  (should be ~0)\n\n",
           max_identity_err);

    // ------------------------------------------------------------------
    // 3.  Allocate device memory & copy
    // ------------------------------------------------------------------
    float* d_K    = nullptr;
    float* d_L    = nullptr;
    float* d_gap  = nullptr;
    float* d_work = nullptr;

    HIP_CHECK(hipMalloc(&d_K,    sizeof(float) * STRIDE_A * BATCH_COUNT));
    HIP_CHECK(hipMalloc(&d_L,    sizeof(float) * STRIDE_A * BATCH_COUNT));
    HIP_CHECK(hipMalloc(&d_gap,  sizeof(float) * STRIDE_X * BATCH_COUNT));
    HIP_CHECK(hipMalloc(&d_work, sizeof(float) * STRIDE_X * BATCH_COUNT));

    HIP_CHECK(hipMemcpy(d_K,   h_K.data(),
                         sizeof(float) * STRIDE_A * BATCH_COUNT,
                         hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_L,   h_L.data(),
                         sizeof(float) * STRIDE_A * BATCH_COUNT,
                         hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_gap, h_gap.data(),
                         sizeof(float) * STRIDE_X * BATCH_COUNT,
                         hipMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // 4.  rocBLAS: one Newton iteration  (with hipEvent timing)
    //
    //     4A  sgemv:  f  = K * gap          (contact force)
    //     4B  strsv:  L * y  = f            (Cholesky forward)
    //     4C  strsv:  L^T * dd = y          (Cholesky backward)
    //     Result: dd = K^{-1} * K * gap = gap
    // ------------------------------------------------------------------
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    hipEvent_t ev_start_sgemv, ev_stop_sgemv;
    hipEvent_t ev_start_trsv_fwd, ev_stop_trsv_fwd;
    hipEvent_t ev_start_trsv_bwd, ev_stop_trsv_bwd;
    hipEvent_t ev_pipeline_start, ev_pipeline_stop;
    HIP_CHECK(hipEventCreate(&ev_start_sgemv));
    HIP_CHECK(hipEventCreate(&ev_stop_sgemv));
    HIP_CHECK(hipEventCreate(&ev_start_trsv_fwd));
    HIP_CHECK(hipEventCreate(&ev_stop_trsv_fwd));
    HIP_CHECK(hipEventCreate(&ev_start_trsv_bwd));
    HIP_CHECK(hipEventCreate(&ev_stop_trsv_bwd));
    HIP_CHECK(hipEventCreate(&ev_pipeline_start));
    HIP_CHECK(hipEventCreate(&ev_pipeline_stop));

    float alpha = 1.0f, beta = 0.0f;

    // 4A: sgemv  f = K * gap
    printf("Calling rocblas_sgemv_strided_batched (n=%d, batch=%d) ...\n",
           N, BATCH_COUNT);
    HIP_CHECK(hipEventRecord(ev_pipeline_start));
    HIP_CHECK(hipEventRecord(ev_start_sgemv));
    ROCBLAS_CHECK(rocblas_sgemv_strided_batched(
        handle,
        rocblas_operation_none,
        N, N,
        &alpha,
        d_K,   LDA, STRIDE_A,
        d_gap, INCX, STRIDE_X,
        &beta,
        d_work, INCX, STRIDE_X,
        BATCH_COUNT));
    HIP_CHECK(hipEventRecord(ev_stop_sgemv));

    // 4B: strsv forward  L * y = f
    printf("Calling rocblas_strsv_strided_batched [forward] (n=%d, batch=%d) ...\n",
           N, BATCH_COUNT);
    HIP_CHECK(hipEventRecord(ev_start_trsv_fwd));
    ROCBLAS_CHECK(rocblas_strsv_strided_batched(
        handle,
        rocblas_fill_lower,
        rocblas_operation_none,
        rocblas_diagonal_non_unit,
        N,
        d_L, LDA, STRIDE_A,
        d_work, INCX, STRIDE_X,
        BATCH_COUNT));
    HIP_CHECK(hipEventRecord(ev_stop_trsv_fwd));

    // 4C: strsv backward  L^T * dd = y
    printf("Calling rocblas_strsv_strided_batched [backward] (n=%d, batch=%d) ...\n",
           N, BATCH_COUNT);
    HIP_CHECK(hipEventRecord(ev_start_trsv_bwd));
    ROCBLAS_CHECK(rocblas_strsv_strided_batched(
        handle,
        rocblas_fill_lower,
        rocblas_operation_transpose,
        rocblas_diagonal_non_unit,
        N,
        d_L, LDA, STRIDE_A,
        d_work, INCX, STRIDE_X,
        BATCH_COUNT));
    HIP_CHECK(hipEventRecord(ev_stop_trsv_bwd));
    HIP_CHECK(hipEventRecord(ev_pipeline_stop));

    HIP_CHECK(hipEventSynchronize(ev_pipeline_stop));

    float ms_sgemv = 0, ms_trsv_fwd = 0, ms_trsv_bwd = 0, ms_total = 0;
    HIP_CHECK(hipEventElapsedTime(&ms_sgemv,    ev_start_sgemv,    ev_stop_sgemv));
    HIP_CHECK(hipEventElapsedTime(&ms_trsv_fwd, ev_start_trsv_fwd, ev_stop_trsv_fwd));
    HIP_CHECK(hipEventElapsedTime(&ms_trsv_bwd, ev_start_trsv_bwd, ev_stop_trsv_bwd));
    HIP_CHECK(hipEventElapsedTime(&ms_total,    ev_pipeline_start, ev_pipeline_stop));

    printf("All kernels returned successfully.\n\n");

    HIP_CHECK(hipEventDestroy(ev_start_sgemv));
    HIP_CHECK(hipEventDestroy(ev_stop_sgemv));
    HIP_CHECK(hipEventDestroy(ev_start_trsv_fwd));
    HIP_CHECK(hipEventDestroy(ev_stop_trsv_fwd));
    HIP_CHECK(hipEventDestroy(ev_start_trsv_bwd));
    HIP_CHECK(hipEventDestroy(ev_stop_trsv_bwd));
    HIP_CHECK(hipEventDestroy(ev_pipeline_start));
    HIP_CHECK(hipEventDestroy(ev_pipeline_stop));

    // ------------------------------------------------------------------
    // 5.  Copy result back & verify
    // ------------------------------------------------------------------
    std::vector<float> h_dd_gpu(STRIDE_X * BATCH_COUNT);
    HIP_CHECK(hipMemcpy(h_dd_gpu.data(), d_work,
                         sizeof(float) * STRIDE_X * BATCH_COUNT,
                         hipMemcpyDeviceToHost));

    printf("GPU dd[0..%d] = [", std::min(N, 4) - 1);
    for (int k = 0; k < std::min(N, 4); ++k)
        printf("%.6f%s", h_dd_gpu[k], k < std::min(N, 4) - 1 ? ", " : "");
    printf(" ...]\n");

    int errors = 0;
    for (int b = 0; b < BATCH_COUNT; ++b) {
        for (int k = 0; k < N; ++k) {
            float gpu_val = h_dd_gpu[b * STRIDE_X + k];
            float ref_val = h_dd_ref[b * STRIDE_X + k];
            if (std::fabs(gpu_val - ref_val) > TOL) {
                if (errors < 10)
                    printf("MISMATCH  batch %d  elem %d : gpu=%.8f  cpu=%.8f\n",
                           b, k, gpu_val, ref_val);
                ++errors;
            }
        }
    }

    float max_physics_err = 0.0f;
    for (int b = 0; b < BATCH_COUNT; ++b)
        for (int k = 0; k < N; ++k)
            max_physics_err = fmaxf(max_physics_err,
                fabsf(h_dd_gpu[b * STRIDE_X + k] - h_gap[b * STRIDE_X + k]));

    printf("Physics check ||dd - gap||_inf = %e\n", max_physics_err);

    printf("\n--- PERFORMANCE ---\n");
    printf("  %-30s %10.4f ms\n", "sgemv (contact force)",     ms_sgemv);
    printf("  %-30s %10.4f ms\n", "strsv forward  (L*y=f)",    ms_trsv_fwd);
    printf("  %-30s %10.4f ms\n", "strsv backward (L^T*dd=y)", ms_trsv_bwd);
    printf("  %-30s %10.4f ms\n", "---------- TOTAL PIPELINE", ms_total);
    double throughput = (double)BATCH_COUNT / (ms_total * 1e-3);
    printf("  Throughput: %.2e batches/sec\n", throughput);

    printf("\n--- RESULT ---\n");
    if (errors == 0) {
        printf("PASSED: All %d batches match CPU reference (tol=%.1e)\n",
               BATCH_COUNT, TOL);
    } else {
        printf("FAILED: %d element mismatches out of %d total\n",
               errors, BATCH_COUNT * N);
    }

    // ------------------------------------------------------------------
    // 6.  Cleanup
    // ------------------------------------------------------------------
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipFree(d_K));
    HIP_CHECK(hipFree(d_L));
    HIP_CHECK(hipFree(d_gap));
    HIP_CHECK(hipFree(d_work));

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
