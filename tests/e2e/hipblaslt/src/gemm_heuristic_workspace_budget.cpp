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
 * @file gemm_heuristic_algo_workspace_mem_budget_tflops_consistency.cpp
 * @brief Dynamic Memory Pressure Testing for hipBLASLt GEMM Operations
 *
 * Purpose:
 * --------
 * This demonstration tests how hipBLASLt algorithm selection adapts to
 * constrained GPU memory environments. In production ML systems, available
 * GPU memory fluctuates due to:
 *   - Concurrent workloads on shared GPUs
 *   - Dynamic batching in inference servers (vLLM, TensorRT-LLM)
 *   - Memory fragmentation over time
 *   - Multi-tenant GPU sharing
 *
 * Testing Methodology:
 * -------------------
 * We execute the same GEMM operation with progressively decreasing workspace
 * memory budgets to observe:
 *   1. Algorithm adaptation (does it switch to memory-efficient variants?)
 *   2. Graceful degradation (does it handle constraints without crashing?)
 *   3. Workspace compliance (does workspace usage respect budget limits?)
 *   4. Zero-workspace fallback (can it operate without extra memory?)
 *   5. Failure modes (does it fail gracefully at extreme constraints?)
 *
 * Real-World Applications:
 * -----------------------
 *   - LLM serving with bursty traffic patterns
 *   - Training with dynamic batch sizes
 *   - GPU memory oversubscription scenarios
 *   - Quality of Service (QoS) guarantees in multi-tenant systems
 *
 * Success Criteria:
 * ----------------
 *   - High memory (>=32 MB): Should always succeed
 *   - Medium memory (8-32 MB): Majority success expected
 *   - Low memory (1-8 MB): Partial success acceptable
 *   - Zero memory: Graceful failure or zero-workspace algorithm
 *
 */

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ============================================================================
// ERROR CHECKING MACROS
// ============================================================================

/**
 * @brief Error checking macro for HIP runtime API calls.
 * Prints file, line, and error string on failure, then exits.
 */
#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t err_ = (expr);                                                  \
    if (err_ != hipSuccess) {                                                  \
      std::cerr << "HIP error: " << hipGetErrorString(err_) << " at "         \
                << __FILE__ << ":" << __LINE__ << "\n";                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

/**
 * @brief Error checking macro for hipBLASLt API calls (setup/teardown).
 * Use for calls where failure is NOT expected (descriptor creation, etc.).
 * Do NOT use for heuristic queries or matmul execution where failure is
 * a valid outcome.
 */
#define HIPBLASLT_CHECK(expr)                                                  \
  do {                                                                         \
    hipblasStatus_t st_ = (expr);                                              \
    if (st_ != HIPBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "hipBLASLt error: " << static_cast<int>(st_) << " at "     \
                << __FILE__ << ":" << __LINE__ << "\n";                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ============================================================================
// CONFIGURATION & DATA STRUCTURES
// ============================================================================

/**
 * @struct GEMMRequest
 * @brief Encapsulates all parameters for a GEMM operation request
 */
struct GEMMRequest {
  std::string operation_name; ///< Descriptive name for logging
  int M, N, K;                ///< Matrix dimensions (M×K) × (K×N) = (M×N)
  hipblasOperation_t transA;  ///< Transpose operation for matrix A
  hipblasOperation_t transB;  ///< Transpose operation for matrix B
  hipDataType input_type;     ///< Input matrix data type
  hipDataType output_type;    ///< Output matrix data type
  hipblasComputeType_t compute_type; ///< Computation precision
  hipblasLtEpilogue_t epilogue;      ///< Fused epilogue operation
  size_t max_workspace_bytes; ///< Maximum workspace memory budget
};

/**
 * @struct AlgorithmChoice
 * @brief Contains the selected algorithm and its characteristics
 */
struct AlgorithmChoice {
  hipblasLtMatmulAlgo_t algo; ///< The selected algorithm handle
  size_t workspace_size;      ///< Workspace memory required by this algorithm
  float estimated_time_ms;    ///< Estimated execution time (if available)
  int algo_index; ///< Index in the heuristic results (-1 if none found)
  bool is_cached; ///< Whether this choice was retrieved from cache
  std::string algo_info; ///< Human-readable algorithm identifier
};

/**
 * @struct GEMMDescriptors
 * @brief Pre-created descriptors for efficient repeated GEMM execution.
 *
 * Created once before the budget sweep, reused across all warmup + timed
 * iterations across all budgets. Avoids the overhead of creating/destroying
 * 7 descriptors on each of the 110 calls per budget.
 */
struct GEMMDescriptors {
  hipblasLtMatmulDesc_t matmul_desc;
  hipblasLtMatrixLayout_t A_desc, B_desc, C_desc, D_desc;
  bool valid = false;
};

/**
 * @struct TierMetrics
 * @brief Performance metrics for a single algorithm tier
 *        (e.g., workspace-using vs zero-workspace).
 *
 * Consistency is measured WITHIN a tier, not across tiers,
 * because different tiers use different kernel strategies.
 */
struct TierMetrics {
  std::string label;
  std::vector<double> tflops;
  std::vector<size_t> budgets_mb;
  size_t workspace_size_bytes = 0; ///< Representative workspace size for label

  double mean() const {
    if (tflops.empty())
      return 0.0;
    double sum = std::accumulate(tflops.begin(), tflops.end(), 0.0);
    return sum / tflops.size();
  }

  double cv_percent() const {
    if (tflops.size() < 2)
      return 0.0;
    double m = mean();
    if (m == 0.0)
      return 0.0;
    double var = 0.0;
    for (double t : tflops)
      var += (t - m) * (t - m);
    return std::sqrt(var / tflops.size()) / m * 100.0;
  }

  double min_val() const {
    return tflops.empty()
               ? 0.0
               : *std::min_element(tflops.begin(), tflops.end());
  }

  double max_val() const {
    return tflops.empty()
               ? 0.0
               : *std::max_element(tflops.begin(), tflops.end());
  }
};

/**
 * @struct ValidationMetrics
 * @brief Aggregates metrics for validation and reporting
 */
struct ValidationMetrics {
  int total_tests = 0;
  int success_count = 0;
  int algo_found_count = 0; ///< Budgets where heuristic returned an algorithm
  int algo_changes = 0;     ///< Number of workspace tier transitions detected
  bool found_zero_workspace = false;
  bool high_memory_success = false;
  bool graceful_at_zero = false; ///< Did budget=0 fail gracefully or succeed?
  std::vector<int> successful_budgets_mb;

  // Per-tier performance tracking
  TierMetrics workspace_tier{"Workspace"};
  TierMetrics zero_workspace_tier{"Zero-Workspace"};

  // Overall (all tiers combined, informational only)
  std::vector<double> all_tflops;
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Compute element size in bytes for a given hipDataType.
 *
 * Used to correctly size GPU allocations regardless of input/output precision.
 * Without this, allocating C/D buffers with sizeof(__half) when output_type
 * is FP32 would cause a buffer overflow on the GPU.
 */
static size_t element_size_for_type(hipDataType dtype) {
  switch (dtype) {
  case HIP_R_16F:
    return sizeof(__half);
  case HIP_R_32F:
    return sizeof(float);
  default:
    std::cerr << "Unsupported hipDataType: " << static_cast<int>(dtype) << "\n";
    exit(1);
  }
}

/**
 * @brief Build a human-readable algorithm label.
 *
 * Shows the heuristic rank and workspace size. These are the two
 * meaningful identifiers. The raw bytes of the opaque hipblasLtMatmulAlgo_t
 * struct contain workspace-budget metadata that changes with each query,
 * so hashing them produces different values even for the same underlying
 * kernel. We avoid that pitfall by labelling with observable properties.
 */
static std::string
get_algorithm_info(const hipblasLtMatmulAlgo_t & /*algo*/, int algo_index,
                   size_t workspace_size) {
  char buffer[64];
  if (workspace_size > 0) {
    double ws_mb = workspace_size / (1024.0 * 1024.0);
    snprintf(buffer, sizeof(buffer), "#%d ws:%.0fMB", algo_index, ws_mb);
  } else {
    snprintf(buffer, sizeof(buffer), "#%d ws:0", algo_index);
  }
  return std::string(buffer);
}

/**
 * @brief Print a tier's consistency analysis.
 *
 * Factored out so the same logic is used for display (Metric 3) and
 * verdict calculation, eliminating the previous copy-paste duplication.
 *
 * @return true if CV is within the acceptable threshold
 */
static bool print_tier_consistency(const TierMetrics &tier,
                                   double threshold_cv = 3.0) {
  if (tier.tflops.empty()) {
    std::cout << "   (no data)\n";
    return true; // vacuously true
  }

  double ws_mb = tier.workspace_size_bytes / (1024.0 * 1024.0);
  if (tier.workspace_size_bytes > 0)
    std::cout << "   Tier: " << tier.label << " (" << std::fixed
              << std::setprecision(0) << ws_mb << " MB)\n";
  else
    std::cout << "   Tier: " << tier.label << "\n";

  std::cout << "     Budgets tested: ";
  for (size_t i = 0; i < tier.budgets_mb.size(); i++) {
    if (i > 0)
      std::cout << ", ";
    std::cout << tier.budgets_mb[i];
  }
  std::cout << " MB\n";

  std::cout << "     Mean: " << std::fixed << std::setprecision(1)
            << tier.mean() << " TFLOPS\n";
  std::cout << "     Range: " << tier.min_val() << " - " << tier.max_val()
            << " TFLOPS\n";

  double cv = tier.cv_percent();
  std::cout << "     CV: " << std::setprecision(2) << cv << "%";

  if (tier.tflops.size() < 2) {
    std::cout << "  (single sample, consistency N/A)\n";
    return true;
  }

  bool pass = (cv <= threshold_cv);
  std::cout << (pass ? "  ← PASS" : "  ← FAIL") << " (threshold ≤"
            << std::setprecision(0) << threshold_cv << "%)\n";
  return pass;
}

// ============================================================================
// SIMPLIFIED GEMM ENGINE
// ============================================================================

/**
 * @class SmartGEMMEngine
 * @brief Manages hipBLASLt handle, descriptor lifecycle, and algorithm
 *        selection.
 *
 * Provides two execution paths:
 * - create_descriptors() + execute_gemm_fast() + destroy_descriptors()
 *   for tight benchmark loops (no per-call overhead)
 * - select_algorithm() for heuristic queries
 */
class SmartGEMMEngine {
private:
  hipblasLtHandle_t handle_;

public:
  SmartGEMMEngine() {
    HIPBLASLT_CHECK(hipblasLtCreate(&handle_));
    std::cout << " GEMM Engine initialized\n";
  }

  ~SmartGEMMEngine() { hipblasLtDestroy(handle_); }

  /**
   * @brief Create reusable descriptors for a given GEMM request.
   *
   * These descriptors are independent of workspace budget and algorithm
   * choice, so they can be created once and reused across the entire
   * budget sweep (all warmup + timed iterations for all budgets).
   */
  GEMMDescriptors create_descriptors(const GEMMRequest &req) {
    GEMMDescriptors d;

    HIPBLASLT_CHECK(
        hipblasLtMatmulDescCreate(&d.matmul_desc, req.compute_type, HIP_R_32F));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        d.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &req.transA,
        sizeof(req.transA)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        d.matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &req.transB,
        sizeof(req.transB)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        d.matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &req.epilogue,
        sizeof(req.epilogue)));

    int row_A = (req.transA == HIPBLAS_OP_N) ? req.M : req.K;
    int col_A = (req.transA == HIPBLAS_OP_N) ? req.K : req.M;
    int row_B = (req.transB == HIPBLAS_OP_N) ? req.K : req.N;
    int col_B = (req.transB == HIPBLAS_OP_N) ? req.N : req.K;

    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&d.A_desc, req.input_type,
                                                row_A, col_A, row_A));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&d.B_desc, req.input_type,
                                                row_B, col_B, row_B));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&d.C_desc, req.output_type,
                                                req.M, req.N, req.M));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&d.D_desc, req.output_type,
                                                req.M, req.N, req.M));
    d.valid = true;
    return d;
  }

  /**
   * @brief Destroy previously created descriptors.
   */
  void destroy_descriptors(GEMMDescriptors &d) {
    if (!d.valid)
      return;
    hipblasLtMatrixLayoutDestroy(d.A_desc);
    hipblasLtMatrixLayoutDestroy(d.B_desc);
    hipblasLtMatrixLayoutDestroy(d.C_desc);
    hipblasLtMatrixLayoutDestroy(d.D_desc);
    hipblasLtMatmulDescDestroy(d.matmul_desc);
    d.valid = false;
  }

  /**
   * @brief Fast GEMM execution using pre-created descriptors.
   *
   * No descriptor creation/destruction overhead per call. This is the
   * hot path used inside the 10-warmup + 100-timed iteration loops.
   */
  bool execute_gemm_fast(const GEMMDescriptors &desc,
                         const AlgorithmChoice &choice, void *d_A, void *d_B,
                         void *d_C, void *d_D, void *workspace,
                         float alpha = 1.0f, float beta = 0.0f) {
    if (choice.algo_index < 0 || !desc.valid)
      return false;

    hipblasStatus_t status = hipblasLtMatmul(
        handle_, desc.matmul_desc, &alpha, d_A, desc.A_desc, d_B, desc.B_desc,
        &beta, d_C, desc.C_desc, d_D, desc.D_desc, &choice.algo, workspace,
        choice.workspace_size, 0);

    return (status == HIPBLAS_STATUS_SUCCESS);
  }

  /**
   * @brief Select the best algorithm for a given GEMM request.
   *
   * Queries hipBLASLt heuristics with the workspace budget from the request.
   * Creates temporary descriptors for the query only (lightweight).
   *
   * Always picks the first valid algorithm (heuristics are sorted by speed).
   */
  AlgorithmChoice select_algorithm(const GEMMRequest &req) {
    AlgorithmChoice choice;
    choice.is_cached = false;
    choice.algo_index = -1;
    choice.workspace_size = 0;
    choice.estimated_time_ms = -1.0f;

    // Temporary descriptors for heuristic query
    hipblasLtMatmulDesc_t matmul_desc;
    HIPBLASLT_CHECK(
        hipblasLtMatmulDescCreate(&matmul_desc, req.compute_type, HIP_R_32F));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &req.transA,
        sizeof(req.transA)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &req.transB,
        sizeof(req.transB)));
    HIPBLASLT_CHECK(hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &req.epilogue,
        sizeof(req.epilogue)));

    int row_A = (req.transA == HIPBLAS_OP_N) ? req.M : req.K;
    int col_A = (req.transA == HIPBLAS_OP_N) ? req.K : req.M;
    int row_B = (req.transB == HIPBLAS_OP_N) ? req.K : req.N;
    int col_B = (req.transB == HIPBLAS_OP_N) ? req.N : req.K;

    hipblasLtMatrixLayout_t A_desc, B_desc, C_desc, D_desc;
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&A_desc, req.input_type, row_A,
                                                col_A, row_A));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&B_desc, req.input_type, row_B,
                                                col_B, row_B));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&C_desc, req.output_type, req.M,
                                                req.N, req.M));
    HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&D_desc, req.output_type, req.M,
                                                req.N, req.M));

    hipblasLtMatmulPreference_t pref;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    size_t max_workspace = req.max_workspace_bytes;
    HIPBLASLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace,
        sizeof(max_workspace)));

    std::vector<hipblasLtMatmulHeuristicResult_t> results(128);
    int count = 0;

    // This call may legitimately return no results — do NOT use HIPBLASLT_CHECK
    hipblasStatus_t status = hipblasLtMatmulAlgoGetHeuristic(
        handle_, matmul_desc, A_desc, B_desc, C_desc, D_desc, pref, 128,
        results.data(), &count);

    if (status == HIPBLAS_STATUS_SUCCESS && count > 0) {
      int selected_idx = -1;

      for (int i = 0; i < count; i++) {
        if (results[i].state == HIPBLAS_STATUS_SUCCESS) {
          selected_idx = i;
          break;
        }
      }

      if (selected_idx >= 0) {
        choice.algo = results[selected_idx].algo;
        choice.workspace_size = results[selected_idx].workspaceSize;
        choice.algo_index = selected_idx;
        choice.algo_info =
            get_algorithm_info(results[selected_idx].algo, selected_idx,
                               results[selected_idx].workspaceSize);
      }
    }

    // Cleanup query descriptors
    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(A_desc);
    hipblasLtMatrixLayoutDestroy(B_desc);
    hipblasLtMatrixLayoutDestroy(C_desc);
    hipblasLtMatrixLayoutDestroy(D_desc);
    hipblasLtMatmulDescDestroy(matmul_desc);

    return choice;
  }
};


void demo_dynamic_memory_pressure(SmartGEMMEngine &engine, int M, int N,
                                  int K) {
  std::cout << "\n╔════════════════════════════════════════════════════════════"
               "═══════╗\n";
  std::cout << "║  Sample GEMM algorithm selection under constrained memory "
               "budget  ║\n";
  std::cout << "╚══════════════════════════════════════════════════════════════"
               "═════╝\n\n";

  // Define memory budgets (in MB, from abundant down to zero)
  // 0 MB tests the graceful-failure / zero-workspace fallback path
  std::vector<size_t> memory_available_mb = {512, 256, 128, 64, 32,
                                             16,  8,   4,   2,  1, 0};

  // Try multiple GEMM configurations to find workspace-using algorithms
  std::vector<GEMMRequest> test_configs = {
      {"double_transpose", M, N, K, HIPBLAS_OP_T, HIPBLAS_OP_T, HIP_R_16F,
       HIP_R_16F, HIPBLAS_COMPUTE_32F, HIPBLASLT_EPILOGUE_GELU, 0},
      {"mixed_precision", M, N, K, HIPBLAS_OP_N, HIPBLAS_OP_T, HIP_R_16F,
       HIP_R_32F, HIPBLAS_COMPUTE_32F, HIPBLASLT_EPILOGUE_GELU, 0},
      {"user_config_transpose_gelu", M, N, K, HIPBLAS_OP_N, HIPBLAS_OP_T,
       HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F, HIPBLASLT_EPILOGUE_GELU, 0},
      {"transpose_relu", M, N, K, HIPBLAS_OP_N, HIPBLAS_OP_T, HIP_R_16F,
       HIP_R_16F, HIPBLAS_COMPUTE_32F, HIPBLASLT_EPILOGUE_RELU, 0},
  };

  // Find a configuration that uses workspace
  GEMMRequest fixed_req;
  bool found_workspace_config = false;

  for (auto &config : test_configs) {
    config.max_workspace_bytes = 256 * 1024 * 1024; // Test with 256 MB
    auto test_choice = engine.select_algorithm(config);

    if (test_choice.algo_index >= 0 && test_choice.workspace_size > 0 &&
        !found_workspace_config) {
      double ws_mb = test_choice.workspace_size / (1024.0 * 1024.0);
      std::cout << "  Found workspace-using config \""
                << config.operation_name << "\" requiring " << std::fixed
                << std::setprecision(2) << ws_mb << " MB of workspace.\n";
      fixed_req = config;
      found_workspace_config = true;
    }
  }

  if (!found_workspace_config) {
    std::cout << "\n No workspace-using algorithm found. Using first config "
                 "anyway.\n";
    std::cout
        << "  (This demonstrates that hipBLASLt can operate without workspace)"
        << "\n\n";
    fixed_req = test_configs[0];
  } else {
    std::cout << "\n";
  }

  // Display test configuration
  std::cout << "Selected Test Configuration:\n";
  std::cout << "  Operation: " << fixed_req.operation_name << "\n";
  std::cout << "  Matrix Shapes: [" << fixed_req.M << " × " << fixed_req.K
            << "] × [";
  if (fixed_req.transB == HIPBLAS_OP_T) {
    std::cout << fixed_req.N << " × " << fixed_req.K << "]^T";
  } else {
    std::cout << fixed_req.K << " × " << fixed_req.N << "]";
  }
  std::cout << " = [" << fixed_req.M << " × " << fixed_req.N << "]\n";

  // Show data types dynamically
  auto type_name = [](hipDataType t) -> const char * {
    switch (t) {
    case HIP_R_16F:
      return "FP16";
    case HIP_R_32F:
      return "FP32";
    case HIP_R_16BF:
      return "BF16";
    default:
      return "other";
    }
  };
  std::cout << "  Input Type: " << type_name(fixed_req.input_type) << "\n";
  std::cout << "  Output Type: " << type_name(fixed_req.output_type) << "\n";
  std::cout << "  Compute Type: FP32 (single precision accumulation)\n";

  // ---- Allocate GPU matrices (type-aware sizes) ----
  std::cout << "Allocating GPU memory for matrices...\n";
  size_t input_elem = element_size_for_type(fixed_req.input_type);
  size_t output_elem = element_size_for_type(fixed_req.output_type);

  size_t matrix_size_a = (size_t)fixed_req.M * fixed_req.K * input_elem;
  size_t matrix_size_b = (size_t)fixed_req.K * fixed_req.N * input_elem;
  size_t matrix_size_c = (size_t)fixed_req.M * fixed_req.N * output_elem;
  double total_matrix_gb =
      (matrix_size_a + matrix_size_b + 2 * matrix_size_c) / 1024.0 / 1024.0 /
      1024.0;

  std::cout << "  Matrix A: " << (matrix_size_a / 1024.0 / 1024.0) << " MB\n";
  std::cout << "  Matrix B: " << (matrix_size_b / 1024.0 / 1024.0) << " MB\n";
  std::cout << "  Matrix C+D: " << (2 * matrix_size_c / 1024.0 / 1024.0)
            << " MB\n";
  std::cout << "  Total: " << std::fixed << std::setprecision(2)
            << total_matrix_gb << " GB\n";

  void *d_A, *d_B, *d_C, *d_D;

  HIP_CHECK(hipMalloc(&d_A, matrix_size_a));
  HIP_CHECK(hipMalloc(&d_B, matrix_size_b));
  HIP_CHECK(hipMalloc(&d_C, matrix_size_c));
  HIP_CHECK(hipMalloc(&d_D, matrix_size_c));

  // Initialize matrices
  HIP_CHECK(hipMemset(d_A, 0, matrix_size_a));
  HIP_CHECK(hipMemset(d_B, 0, matrix_size_b));
  HIP_CHECK(hipMemset(d_C, 0, matrix_size_c));

  std::cout << " GPU matrices allocated successfully\n";

  // ---- Create descriptors ONCE for entire sweep ----
  GEMMDescriptors desc = engine.create_descriptors(fixed_req);
  std::cout << " Descriptors created (reused across all budgets)\n";
  std::cout
      << "ℹ Note: Each test includes 10 warmup + 100 timed iterations\n\n";

  // Table header
  std::cout << std::string(120, '-') << "\n";
  std::cout << std::left << std::setw(16) << "Memory Budget" << std::setw(20)
            << "Algorithm" << std::setw(16) << "Workspace Used" << std::setw(18)
            << "Exec Time" << std::setw(18) << "Performance" << std::setw(30)
            << "Status\n";
  std::cout << std::string(120, '-') << "\n";

  // Initialize validation metrics
  ValidationMetrics metrics;
  metrics.total_tests = memory_available_mb.size();
  size_t last_workspace = SIZE_MAX; // Track workspace tier transitions

  // ---- Run tests for each memory budget ----
  for (size_t mem_mb : memory_available_mb) {
    fixed_req.max_workspace_bytes = mem_mb * 1024 * 1024;

    AlgorithmChoice choice = engine.select_algorithm(fixed_req);

    if (choice.algo_index >= 0) {
      metrics.algo_found_count++;

      // Detect workspace tier transition
      if (last_workspace != SIZE_MAX &&
          choice.workspace_size != last_workspace) {
        metrics.algo_changes++;
        std::cout << std::string(120, '.') << "\n";
      }
      last_workspace = choice.workspace_size;

      // Format budget column
      std::cout << std::setw(16) << (std::to_string(mem_mb) + " MB");

      // Track zero-workspace discovery
      if (choice.workspace_size == 0)
        metrics.found_zero_workspace = true;

      // Display algorithm info
      std::cout << std::setw(20) << choice.algo_info;

      // Display workspace usage
      double ws_mb = choice.workspace_size / (1024.0 * 1024.0);
      std::cout << std::setw(16) << (std::to_string((int)ws_mb) + " MB");

      // Allocate workspace for this specific algorithm
      void *d_workspace = nullptr;
      if (choice.workspace_size > 0) {
        hipError_t ws_err = hipMalloc(&d_workspace, choice.workspace_size);
        if (ws_err != hipSuccess) {
          std::cout << std::setw(18) << "ALLOC FAILED" << std::setw(18)
                    << "N/A"
                    << " FAIL (no workspace memory)\n";
          if (mem_mb == 0)
            metrics.graceful_at_zero = true;
          continue;
        }
      }

      // ---- Execute GEMM with timing ----
      bool exec_success = false;
      double exec_time_ms = 0.0;
      double tflops = 0.0;

      // Warmup (10 iterations to stabilize GPU clocks and caches)
      bool warmup_ok = true;
      for (int w = 0; w < 10; w++) {
        if (!engine.execute_gemm_fast(desc, choice, d_A, d_B, d_C, d_D,
                                      d_workspace)) {
          warmup_ok = false;
          break;
        }
      }
      HIP_CHECK(hipDeviceSynchronize());

      if (!warmup_ok) {
        if (d_workspace)
          HIP_CHECK(hipFree(d_workspace));
        std::cout << std::setw(18) << "N/A" << std::setw(18) << "N/A"
                  << " FAIL (warmup)";
        if (mem_mb == 0)
          metrics.graceful_at_zero = true;
        std::cout << "\n";
        continue;
      }

      // Timed runs (100 iterations for stable average)
      const int timed_iterations = 100;
      std::vector<double> iteration_times;
      iteration_times.reserve(timed_iterations);

      hipEvent_t ev_start, ev_stop;
      HIP_CHECK(hipEventCreate(&ev_start));
      HIP_CHECK(hipEventCreate(&ev_stop));

      for (int iter = 0; iter < timed_iterations; iter++) {
        HIP_CHECK(hipEventRecord(ev_start, 0));
        exec_success = engine.execute_gemm_fast(desc, choice, d_A, d_B, d_C,
                                                d_D, d_workspace);
        HIP_CHECK(hipEventRecord(ev_stop, 0));
        HIP_CHECK(hipEventSynchronize(ev_stop));

        if (!exec_success)
          break;

        float gpu_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&gpu_ms, ev_start, ev_stop));
        iteration_times.push_back(static_cast<double>(gpu_ms));
      }

      HIP_CHECK(hipEventDestroy(ev_start));
      HIP_CHECK(hipEventDestroy(ev_stop));

      // Calculate average and TFLOPS
      if (exec_success && !iteration_times.empty()) {
        double sum_time =
            std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0);
        exec_time_ms = sum_time / iteration_times.size();

        double flops = 2.0 * fixed_req.M * fixed_req.N * fixed_req.K;
        tflops = flops / exec_time_ms / 1e9;

        // ---- Per-tier TFLOPS tracking ----
        metrics.all_tflops.push_back(tflops);
        if (choice.workspace_size > 0) {
          metrics.workspace_tier.tflops.push_back(tflops);
          metrics.workspace_tier.budgets_mb.push_back(mem_mb);
          metrics.workspace_tier.workspace_size_bytes = choice.workspace_size;
        } else {
          metrics.zero_workspace_tier.tflops.push_back(tflops);
          metrics.zero_workspace_tier.budgets_mb.push_back(mem_mb);
        }
      }

      // Free workspace immediately after use
      if (d_workspace)
        HIP_CHECK(hipFree(d_workspace));

      // Display performance metrics
      if (exec_success) {
        char time_buf[32], tflops_buf[32];
        snprintf(time_buf, sizeof(time_buf), "%.2f ms", exec_time_ms);
        snprintf(tflops_buf, sizeof(tflops_buf), "%.1f TFLOPS", tflops);
        std::cout << std::setw(18) << time_buf;
        std::cout << std::setw(18) << tflops_buf;
      } else {
        std::cout << std::setw(18) << "N/A" << std::setw(18) << "N/A";
      }

      // Validate workspace respects budget
      bool within_budget = (choice.workspace_size <= mem_mb * 1024 * 1024);
      if (within_budget && exec_success) {
        metrics.success_count++;
        metrics.successful_budgets_mb.push_back(mem_mb);
        std::cout << " PASS";
      } else if (within_budget && !exec_success) {
        std::cout << " EXEC FAILED";
      } else {
        std::cout << " FAIL (budget)";
      }
      if (mem_mb == 0 && !(within_budget && exec_success))
        metrics.graceful_at_zero = true;

    } else {
      // Format budget column + algorithm selection failed
      std::cout << std::setw(16) << (std::to_string(mem_mb) + " MB");
      std::cout << std::setw(20) << "NONE" << std::setw(16) << "N/A"
                << std::setw(18) << "N/A" << std::setw(18) << "N/A"
                << " FAIL (no algo)";

      // Budget=0 failing is expected and graceful
      if (mem_mb == 0)
        metrics.graceful_at_zero = true;
    }
    std::cout << "\n";
  }

  std::cout << std::string(120, '-') << "\n";

  // ---- Cleanup ----
  engine.destroy_descriptors(desc);
  HIP_CHECK(hipFree(d_D));
  HIP_CHECK(hipFree(d_C));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_A));
  std::cout << "\n GPU matrices and descriptors freed\n";

  // ========================================================================
  // VALIDATION AND ANALYSIS
  // ========================================================================

  std::cout << "\n╔════════════════════════════════════════════════════════════"
               "═══════╗\n";
  std::cout << "║                      VALIDATION REPORT                       "
               "     ║\n";
  std::cout << "╚══════════════════════════════════════════════════════════════"
               "═════╝\n\n";

  // Metric 1: Overall Success Rate
  double success_rate = (metrics.success_count * 100.0) / metrics.total_tests;
  std::cout << "1. Overall Success Rate (alloc + exec + budget compliance)\n";
  std::cout << "   Result: " << metrics.success_count << "/"
            << metrics.total_tests << " (" << std::fixed << std::setprecision(1)
            << success_rate << "%)\n";
  std::cout << "   Algorithms Found: " << metrics.algo_found_count << "/"
            << metrics.total_tests << "\n";

  // Metric 2: High-Memory Performance
  metrics.high_memory_success =
      (!metrics.successful_budgets_mb.empty() &&
       std::count_if(metrics.successful_budgets_mb.begin(),
                     metrics.successful_budgets_mb.end(),
                     [](int mb) { return mb >= 32; }) > 0);
  std::cout << "\n2. High-Memory Baseline (>= 32 MB)\n";
  std::cout << "   Result: "
            << (metrics.high_memory_success ? "SUCCESS" : "FAILED") << "\n";

  // Metric 3: Per-Tier Performance Consistency
  std::cout << "\n3. Performance Consistency (Per-Tier Analysis)\n";
  std::cout << "   Algorithm Tier Transitions: " << metrics.algo_changes
            << "\n\n";

  bool ws_tier_pass = true;
  bool zws_tier_pass = true;

  if (!metrics.workspace_tier.tflops.empty()) {
    ws_tier_pass = print_tier_consistency(metrics.workspace_tier, 3.0);
    std::cout << "\n";
  }

  if (!metrics.zero_workspace_tier.tflops.empty()) {
    zws_tier_pass = print_tier_consistency(metrics.zero_workspace_tier, 3.0);
    std::cout << "\n";
  }

  // Show overall (informational only — not used for pass/fail)
  if (metrics.all_tflops.size() >= 2) {
    double sum =
        std::accumulate(metrics.all_tflops.begin(), metrics.all_tflops.end(), 0.0);
    double mean = sum / metrics.all_tflops.size();
    double var = 0.0;
    for (double t : metrics.all_tflops)
      var += (t - mean) * (t - mean);
    double cv = std::sqrt(var / metrics.all_tflops.size()) / mean * 100.0;

    std::cout << "   Overall (all tiers combined, informational):\n";
    std::cout << "     Mean: " << std::fixed << std::setprecision(1) << mean
              << " TFLOPS, CV: " << std::setprecision(2) << cv << "%\n";
    if (cv > 3.0 && metrics.algo_changes > 0) {
      std::cout << "     (High CV expected — caused by " << metrics.algo_changes
                << " tier transition(s), not instability)\n";
    }
  }

  // Metric 4: Zero-Budget Behavior
  std::cout << "\n4. Zero-Budget Behavior\n";
  if (metrics.graceful_at_zero) {
    std::cout << "   Budget=0: Graceful failure (no algo found, no crash)\n";
  } else if (metrics.found_zero_workspace) {
    std::cout << "   Budget=0: Found zero-workspace algorithm (optimal "
                 "fallback)\n";
  } else {
    std::cout << "   Budget=0: Not tested or unexpected result\n";
  }

  // ========================================================================
  // OVERALL VERDICT
  // ========================================================================

  const double kSuccessRatePassThreshold = 90.0;

  std::cout << "\n" << std::string(78, '=') << "\n";
  std::cout << "OVERALL VERDICT:\n";
  std::cout << std::string(78, '=') << "\n";

  int critical_passes = 0;
  bool perf_consistent = ws_tier_pass && zws_tier_pass;

  if (success_rate >= kSuccessRatePassThreshold)
    critical_passes++;
  if (metrics.high_memory_success)
    critical_passes++;
  if (perf_consistent)
    critical_passes++;

  std::cout << "Critical Criteria Passed: " << critical_passes << "/3\n";
  std::cout << "  • Success Rate: "
            << (success_rate >= kSuccessRatePassThreshold ? "Pass" : "Fail") << " ("
            << std::setprecision(0) << success_rate << "%)\n";
  std::cout << "  • High Memory Baseline: "
            << (metrics.high_memory_success ? "Pass" : "Fail") << "\n";
  std::cout << "  • Per-Tier TFLOPS Consistency (≤3% CV each): "
            << (perf_consistent ? "Pass" : "Fail") << "\n\n";
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char **argv) {
  // Parse command-line arguments
  int M_default = 8192;
  int N_default = 16384;
  int K_default = 8192;

  int M = M_default;
  int N = N_default;
  int K = K_default;

  if (argc >= 2) {
    char *end = nullptr;
    errno = 0;
    long val = strtol(argv[1], &end, 10);

    if (errno != 0 || end == argv[1] || *end != '\0' ||
        val <= 0 || val > INT_MAX / 2) {
      std::cerr << "Error: M must be a positive integer in [1, "
                << INT_MAX / 2 << "]\n";
      std::cerr << "Usage: " << argv[0] << " [M]\n";
      std::cerr << "Example: " << argv[0] << " 8192\n";
      return 1;
    }

    M = static_cast<int>(val);
    N = M * 2;
    K = M;
  }

  std::cout << "Matrix Dimensions:\n";
  std::cout << "  M = " << M << ", N = " << N << ", K = " << K << "\n";
  if (argc < 2) {
    std::cout << "  (using defaults, specify as: " << argv[0]
              << " M to override)\n";
  }
  std::cout << "\n";

  // Display GPU information
  int device;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, device));

  std::cout << "GPU Information:\n";
  std::cout << "  Device: " << prop.name << "\n";
  std::cout << "  Architecture: " << prop.gcnArchName << "\n";
  std::cout << "  Global Memory: "
            << (prop.totalGlobalMem / 1024 / 1024 / 1024) << " GB\n";
  std::cout << "  Compute Units: " << prop.multiProcessorCount << "\n\n";

  SmartGEMMEngine engine;

  std::cout << "\nStarting memory pressure tests...\n";

  demo_dynamic_memory_pressure(engine, M, N, K);

  return 0;
}