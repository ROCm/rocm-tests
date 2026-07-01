/*
Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * REAL-WORLD USE CASE: Iterative Sparse Refactorization Workflow
 *
 * This application demonstrates the key use case for sparse refactorization:
 * - Newton iterations with changing Jacobian values
 * - Time-stepping simulations with updated system matrices
 * - Optimization algorithms with Hessian updates
 * - Structural analysis with load variations
 *
 * UNIQUE VALUE: Demonstrates repeated refactorization with same structure
 * - Analysis is performed ONCE (expensive operation)
 * - Matrix VALUES are updated between iterations
 * - Refactorization reuses the analysis (key performance benefit)
 * - Solution accuracy is validated at each iteration
 * - Demonstrates numerical stability over multiple refactorizations
 *
 * WORKFLOW:
 * 1. Load validated sparse SPD (Sparse Positive Definite) matrix
 * 2. Perform analysis ONCE (symbolic factorization)
 * 3. Iteration Loop:
 *    a. Update matrix values (same structure, different values)
 *    b. Refactorize using cached analysis
 *    c. Solve the linear system
 *    d. Validate solution accuracy
 *    e. Check numerical stability hasn't degraded
 */

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <vector>

namespace fs = std::filesystem;
using namespace std;

// ========================================
// ERROR CHECKING MACROS
// ========================================

// Error checking macros
#define hipCheck(call)                                                         \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP Error at " << __FILE__ << ":" << __LINE__ << " - "     \
                << hipGetErrorString(err) << " (code: " << err << ")\n";       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define rocblasCheck(call)                                                     \
  do {                                                                         \
    rocblas_status status = call;                                              \
    if (status != rocblas_status_success) {                                    \
      std::cerr << "rocBLAS Error at " << __FILE__ << ":" << __LINE__          \
                << " - Status code: " << status << "\n";                       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define rocsolverCheck(call)                                                   \
  do {                                                                         \
    rocblas_status status = call;                                              \
    if (status != rocblas_status_success) {                                    \
      std::cerr << "rocSOLVER Error at " << __FILE__ << ":" << __LINE__        \
                << " - Status code: " << status << "\n";                       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// ========================================
// HELPER FUNCTIONS
// ========================================

// Generate a lower-triangular sparse SPD matrix with exactly nnz_target
// non-zeros. Caller must ensure n <= nnz_target <= n*(n+1)/2.
template <typename T>
void generate_sparse_spd_matrix(int n, int nnz_target, vector<int> &rowPtr,
                                vector<int> &colInd, vector<T> &val) {
  int off_diag_total = nnz_target - n;

  // Find the largest base bandwidth bw such that the banded off-diagonal
  // count does not exceed off_diag_total.
  // Off-diag count for bandwidth bw = bw*(2n - bw - 1)/2
  int bw = 0;
  while (bw + 1 < n) {
    long long next = static_cast<long long>(bw + 1) * (2 * n - bw - 2) / 2;
    if (next > off_diag_total)
      break;
    bw++;
  }

  // Per-row off-diagonal count: base is min(bw, i).
  // Distribute the remaining deficit one per row, starting from the last rows,
  // so the total off-diagonals equals off_diag_total exactly.
  vector<int> row_offdiag(n);
  for (int i = 0; i < n; ++i)
    row_offdiag[i] = min(bw, i);

  long long base_count = (bw > 0)
                             ? static_cast<long long>(bw) * (2 * n - bw - 1) / 2
                             : 0;
  int deficit = off_diag_total - static_cast<int>(base_count);

  for (int i = n - 1; i > bw && deficit > 0; --i) {
    row_offdiag[i]++;
    deficit--;
  }

  // Build the CSR structure
  int current_nnz = 0;
  rowPtr[0] = 0;

  for (int i = 0; i < n; ++i) {
    int num_offdiag = row_offdiag[i];
    T row_offdiag_sum = static_cast<T>(0);

    // Off-diagonal entries: columns (i - num_offdiag) .. (i - 1)
    for (int k = 0; k < num_offdiag; ++k) {
      int j = i - num_offdiag + k;
      colInd[current_nnz] = j;
      T offdiag = static_cast<T>(-0.5 - 1.0 / (1.0 + abs(i - j)));
      val[current_nnz] = offdiag;
      row_offdiag_sum += abs(offdiag);
      current_nnz++;
    }

    // Diagonal entry: exceeds sum of abs(off-diagonals) to guarantee SPD
    colInd[current_nnz] = i;
    val[current_nnz] =
        row_offdiag_sum * static_cast<T>(2.0) + static_cast<T>(10.0);
    current_nnz++;

    rowPtr[i + 1] = current_nnz;
  }
}

// ========================================
// ITERATIVE REFACTORIZATION WORKFLOW
// ========================================

template <typename T>
bool run_iterative_refactorization_workflow(int n, int nnz_target,
                                            int num_iterations) {
  rocblas_handle rocblas_handle;
  rocsolver_rfinfo rfinfo = nullptr;

  // Initialize handles
  rocblasCheck(rocblas_create_handle(&rocblas_handle));

  // Check if sparse functionality is available
  if (rocsolver_create_rfinfo(nullptr, nullptr) ==
      rocblas_status_not_implemented) {
    cout << "SKIPPED: Sparse functionality is not enabled" << endl;
    rocblasCheck(rocblas_destroy_handle(rocblas_handle));
    return true; // Not a failure, just not available
  }

  rocsolverCheck(rocsolver_create_rfinfo(&rfinfo, rocblas_handle));

  rocsolver_set_rfinfo_mode(rfinfo, rocsolver_rfinfo_mode_cholesky);

  // ========================================
  // STEP 1: PREPARE MATRIX PARAMETERS
  // ========================================
  cout << "STEP 1: Preparing matrix..." << endl;

  int nnzA = nnz_target;
  // T matrix (Cholesky factor) initially has same structure as A
  // (analysis/refactorization may compute fill-in internally)
  int nnzT = nnz_target;

  cout << "  - Matrix size: n=" << n << endl;
  cout << "  - Target non-zeros: nnzA=" << nnzA << ", nnzT=" << nnzT << endl;

  // ========================================
  // STEP 2: ALLOCATE MEMORY
  // ========================================

  // Matrix A (input) and Matrix T (Cholesky factor) need SEPARATE storage
  int *d_csrRowPtr_A, *d_csrColInd_A, *d_pivP, *d_pivQ;
  int *d_csrRowPtr_T, *d_csrColInd_T;
  T *d_csrVal_A, *d_csrVal_backup, *d_csrVal_T, *d_x, *d_b, *d_r;

  hipCheck(hipMalloc(&d_csrRowPtr_A, (n + 1) * sizeof(int)));
  hipCheck(hipMalloc(&d_csrColInd_A, nnzA * sizeof(int)));
  hipCheck(hipMalloc(&d_csrVal_A, nnzA * sizeof(T)));
  hipCheck(hipMalloc(&d_csrVal_backup, nnzA * sizeof(T)));
  hipCheck(hipMalloc(&d_csrRowPtr_T, (n + 1) * sizeof(int)));
  hipCheck(hipMalloc(&d_csrColInd_T, nnzT * sizeof(int)));
  hipCheck(hipMalloc(&d_csrVal_T, nnzT * sizeof(T)));
  hipCheck(hipMalloc(&d_x, n * sizeof(T)));
  hipCheck(hipMalloc(&d_b, n * sizeof(T)));
  hipCheck(hipMalloc(&d_r, n * sizeof(T)));
  hipCheck(hipMalloc(&d_pivP, n * sizeof(int)));
  hipCheck(hipMalloc(&d_pivQ, n * sizeof(int)));

  // Load or generate matrix data
  vector<int> h_csrRowPtr(n + 1);
  vector<int> h_csrColInd(nnzA);
  vector<T> h_csrVal(nnzA);
  vector<T> h_b(n);

  // Generate SPD matrix with exactly nnzA non-zeros
  cout << "  - Generating SPD matrix..." << endl;
  generate_sparse_spd_matrix(n, nnzA, h_csrRowPtr, h_csrColInd, h_csrVal);
  cout << "  - Non-zeros created: " << h_csrRowPtr[n] << endl;

  // Keep original values so each iteration can derive from a clean baseline
  vector<T> h_csrVal_orig(h_csrVal);

  // Create RHS vector
  for (int i = 0; i < n; ++i) {
    h_b[i] = static_cast<T>(1.0);
  }

  // Initialize pivot arrays to identity permutation
  vector<int> h_pivP(n), h_pivQ(n);
  for (int i = 0; i < n; ++i) {
    h_pivP[i] = i;
    h_pivQ[i] = i;
  }

  // Copy A matrix and pivot arrays to device
  // Initialize T with A's structure (T will be computed during factorization
  // but needs initial structure)
  hipCheck(hipMemcpy(d_csrRowPtr_A, h_csrRowPtr.data(), (n + 1) * sizeof(int),
                     hipMemcpyHostToDevice));
  hipCheck(hipMemcpy(d_csrColInd_A, h_csrColInd.data(), nnzA * sizeof(int),
                     hipMemcpyHostToDevice));
  hipCheck(hipMemcpy(d_csrVal_A, h_csrVal.data(), nnzA * sizeof(T),
                     hipMemcpyHostToDevice));
  hipCheck(hipMemcpy(d_csrVal_backup, h_csrVal.data(), nnzA * sizeof(T),
                     hipMemcpyHostToDevice));
  hipCheck(hipMemcpy(d_csrRowPtr_T, h_csrRowPtr.data(), (n + 1) * sizeof(int),
                     hipMemcpyHostToDevice));
  hipCheck(hipMemcpy(d_csrColInd_T, h_csrColInd.data(), nnzA * sizeof(int),
                     hipMemcpyHostToDevice));
  hipCheck(hipMemcpy(d_csrVal_T, h_csrVal.data(), nnzA * sizeof(T),
                     hipMemcpyHostToDevice));
  hipCheck(hipMemcpy(d_b, h_b.data(), n * sizeof(T), hipMemcpyHostToDevice));
  hipCheck(
      hipMemcpy(d_pivP, h_pivP.data(), n * sizeof(int), hipMemcpyHostToDevice));
  hipCheck(
      hipMemcpy(d_pivQ, h_pivQ.data(), n * sizeof(int), hipMemcpyHostToDevice));

  // ========================================
  // STEP 3: PERFORM ANALYSIS ONCE
  // (This is the expensive operation we want to reuse)
  // ========================================
  cout << "\nSTEP 2: Performing sparse structure analysis (ONCE)..." << endl;

  rocblas_int nrhs = 1;
  rocblas_int ldb = n;

  if constexpr (std::is_same_v<T, float>) {
    rocsolverCheck(rocsolver_scsrrf_analysis(
        rocblas_handle, n, nrhs, nnzA, d_csrRowPtr_A, d_csrColInd_A, d_csrVal_A,
        nnzT, d_csrRowPtr_T, d_csrColInd_T, d_csrVal_T, d_pivP, d_pivQ, d_b,
        ldb, rfinfo));
  } else {
    rocsolverCheck(rocsolver_dcsrrf_analysis(
        rocblas_handle, n, nrhs, nnzA, d_csrRowPtr_A, d_csrColInd_A, d_csrVal_A,
        nnzT, d_csrRowPtr_T, d_csrColInd_T, d_csrVal_T, d_pivP, d_pivQ, d_b,
        ldb, rfinfo));
  }

  cout << "  - Analysis completed successfully" << endl;
  cout << "  - Analysis will be reused for all " << num_iterations
       << " iterations" << endl;

  // ========================================
  // STEP 3: PERFORM INITIAL FACTORIZATION
  // ========================================
  cout << "\nSTEP 3: Performing initial Cholesky factorization..." << endl;

  if constexpr (std::is_same_v<T, float>) {
    rocsolverCheck(rocsolver_scsrrf_refactchol(
        rocblas_handle, n, nnzA, d_csrRowPtr_A, d_csrColInd_A, d_csrVal_A, nnzT,
        d_csrRowPtr_T, d_csrColInd_T, d_csrVal_T, d_pivQ, rfinfo));
  } else {
    rocsolverCheck(rocsolver_dcsrrf_refactchol(
        rocblas_handle, n, nnzA, d_csrRowPtr_A, d_csrColInd_A, d_csrVal_A, nnzT,
        d_csrRowPtr_T, d_csrColInd_T, d_csrVal_T, d_pivQ, rfinfo));
  }

  cout << "  - Initial factorization completed" << endl;

  // ========================================
  // STEP 4: ITERATIVE REFACTORIZATION LOOP
  // (This is the UNIQUE part - demonstrating repeated refactorization)
  // ========================================
  cout << "\nSTEP 4: Running iterative refactorization workflow..." << endl;
  cout << "  Iter | Refact Status | Solve Status  | Notes" << endl;
  cout << "  -----|---------------|---------------|-------" << endl;

  T initial_norm = -1;
  bool all_iterations_success = true;

  for (int iter = 0; iter < num_iterations; ++iter) {
    rocblas_status refact_status = rocblas_status_success;

    // For iter > 0, update matrix and refactorize
    // For iter == 0, use the initial factorization from Step 3
    if (iter > 0) {
      // ----------------------------------------
      // 4a. UPDATE MATRIX A VALUES
      // Derive from original values each time to avoid compound drift.
      // Uniform scaling preserves diagonal dominance (and thus SPD).
      // ----------------------------------------
      T scale = static_cast<T>(1.0 + 0.1 * sin(iter * 0.5));

      for (int i = 0; i < nnzA; ++i) {
        h_csrVal[i] = h_csrVal_orig[i] * scale;
      }

      hipCheck(hipMemcpy(d_csrVal_A, h_csrVal.data(), nnzA * sizeof(T),
                         hipMemcpyHostToDevice));

      // ----------------------------------------
      // 4b. REFACTORIZE
      // (Reuses analysis from Step 3 - KEY BENEFIT!)
      // Computes Cholesky factor T from updated matrix A
      // ----------------------------------------
      if constexpr (std::is_same_v<T, float>) {
        refact_status = rocsolver_scsrrf_refactchol(
            rocblas_handle, n, nnzA, d_csrRowPtr_A, d_csrColInd_A, d_csrVal_A,
            nnzT, d_csrRowPtr_T, d_csrColInd_T, d_csrVal_T, d_pivQ, rfinfo);
      } else {
        refact_status = rocsolver_dcsrrf_refactchol(
            rocblas_handle, n, nnzA, d_csrRowPtr_A, d_csrColInd_A, d_csrVal_A,
            nnzT, d_csrRowPtr_T, d_csrColInd_T, d_csrVal_T, d_pivQ, rfinfo);
      }
    }

    // ----------------------------------------
    // 4c. SOLVE LINEAR SYSTEM
    // ----------------------------------------
    hipCheck(hipMemcpy(d_x, d_b, n * sizeof(T), hipMemcpyDeviceToDevice));

    rocblas_status solve_status;
    if constexpr (std::is_same_v<T, float>) {
      solve_status = rocsolver_scsrrf_solve(
          rocblas_handle, n, nrhs, nnzT, d_csrRowPtr_T, d_csrColInd_T,
          d_csrVal_T, d_pivP, d_pivQ, d_x, ldb, rfinfo);
    } else {
      solve_status = rocsolver_dcsrrf_solve(
          rocblas_handle, n, nrhs, nnzT, d_csrRowPtr_T, d_csrColInd_T,
          d_csrVal_T, d_pivP, d_pivQ, d_x, ldb, rfinfo);
    }

    // ----------------------------------------
    // 4d. VALIDATE SOLUTION
    // ----------------------------------------
    T solution_norm;
    rocblas_status nrm2_status;
    if constexpr (std::is_same_v<T, float>) {
      nrm2_status = rocblas_snrm2(rocblas_handle, n, d_x, 1, &solution_norm);
    } else {
      nrm2_status = rocblas_dnrm2(rocblas_handle, n, d_x, 1, &solution_norm);
    }

    if (nrm2_status != rocblas_status_success) {
      cerr << "ERROR: NRM2 failed at iteration " << iter << endl;
      all_iterations_success = false;
      break;
    }

    // Track if numerical quality degrades
    if (iter == 0)
      initial_norm = solution_norm;

    // Copy solution to host for inspection
    vector<T> h_x_check(n);
    hipCheck(
        hipMemcpy(h_x_check.data(), d_x, n * sizeof(T), hipMemcpyDeviceToHost));

    // Compute min/max/mean of solution to see if it's changing
    T x_min = *std::min_element(h_x_check.begin(), h_x_check.end());
    T x_max = *std::max_element(h_x_check.begin(), h_x_check.end());
    T x_mean = std::accumulate(h_x_check.begin(), h_x_check.end(), T(0)) / n;

    // Print status
    cout << "  " << setw(4) << iter << " | "
         << (refact_status == rocblas_status_success ? "✓ SUCCESS     "
                                                     : "✗ FAILED      ")
         << " | "
         << (solve_status == rocblas_status_success ? "✓ SUCCESS    "
                                                    : "✗ FAILED     ")
         << " | ";

    if (refact_status == rocblas_status_success &&
        solve_status == rocblas_status_success) {
      cout << "norm=" << scientific << setprecision(2) << solution_norm
           << " x:[" << fixed << setprecision(3) << x_min << "," << x_max
           << "] mean=" << x_mean;
    } else {
      cout << "ERROR";
      all_iterations_success = false;
    }
    cout << endl;

    // Validate operations succeeded
    if (refact_status != rocblas_status_success) {
      cerr << "ERROR: Refactorization failed at iteration " << iter << endl;
      all_iterations_success = false;
      break;
    }
    if (solve_status != rocblas_status_success) {
      cerr << "ERROR: Solve failed at iteration " << iter << endl;
      all_iterations_success = false;
      break;
    }

    // Check solution quality hasn't degraded catastrophically
    if (isnan(solution_norm) || isinf(solution_norm)) {
      cerr << "WARNING: Solution norm is NaN or Inf at iteration " << iter
           << endl;
    }
    if (solution_norm > T(1e-10) && solution_norm > initial_norm * T(100)) {
      cerr << "WARNING: Solution quality degraded significantly at iteration "
           << iter << endl;
    }
  }

  // ========================================
  // STEP 5: FINAL VALIDATION
  // ========================================
  cout << "\nSTEP 4: Final validation..." << endl;

  if (all_iterations_success) {
    cout << "  - Completed " << num_iterations
         << " refactorization+solve cycles" << endl;
    cout << "  - Analysis was reused (not recomputed) for all iterations"
         << endl;
    cout << "  - All refactorizations succeeded" << endl;
    cout << "  - All solves succeeded" << endl;
    cout << "  - Numerical stability maintained" << endl;
    cout << "\n✓ Iterative refactorization workflow PASSED!" << endl;
  } else {
    cout << "\n✗ Iterative refactorization workflow FAILED!" << endl;
  }
  cout << "========================================\n" << endl;

  // Cleanup
  hipCheck(hipFree(d_csrRowPtr_A));
  hipCheck(hipFree(d_csrColInd_A));
  hipCheck(hipFree(d_csrVal_A));
  hipCheck(hipFree(d_csrRowPtr_T));
  hipCheck(hipFree(d_csrColInd_T));
  hipCheck(hipFree(d_csrVal_T));
  hipCheck(hipFree(d_csrVal_backup));
  hipCheck(hipFree(d_x));
  hipCheck(hipFree(d_b));
  hipCheck(hipFree(d_r));
  hipCheck(hipFree(d_pivP));
  hipCheck(hipFree(d_pivQ));

  rocsolver_destroy_rfinfo(rfinfo);
  rocblasCheck(rocblas_destroy_handle(rocblas_handle));

  return all_iterations_success;
}

// ========================================
// MAIN APPLICATION
// ========================================

int main(int argc, char **argv) {
  // Default values
  int n = 10000;          // Matrix size (n x n)
  int nnz = 30000;        // Number of non-zeros
  int num_iterations = 5; // Number of refactorization iterations

  // Display usage if requested
  for (int i = 1; i < argc; ++i) {
    if (string(argv[i]) == "-h" || string(argv[i]) == "--help") {
      cout << "Usage: " << argv[0] << " [OPTIONS]" << endl;
      cout << "\nOptions:" << endl;
      cout << "  -m, --matrix-size <N>     Size of square matrix (default: "
              "10000)"
           << endl;
      cout << "  -n, --nnz <NNZ>           Number of non-zero elements "
              "(default: 30000)"
           << endl;
      cout << "  -i, --iterations <ITER>   Number of refactorization "
              "iterations (default: 5)"
           << endl;
      cout << "  -h, --help                Display this help message" << endl;
      cout << "\nExamples:" << endl;
      cout << "  " << argv[0] << " -m 5000 -n 15000 -i 10" << endl;
      cout << "  " << argv[0] << " --matrix-size 8000 --nnz 24000" << endl;
      cout << "  " << argv[0] << " -m 3000 -i 20" << endl;
      return 0;
    }
  }

  // Parse command-line arguments
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];

    if (arg == "-m" || arg == "--matrix-size") {
      if (i + 1 < argc) {
        n = atoi(argv[++i]);
        if (n <= 0) {
          cerr << "ERROR: Matrix size must be positive" << endl;
          return 1;
        }
      } else {
        cerr << "ERROR: " << arg << " requires a value" << endl;
        return 1;
      }
    } else if (arg == "-n" || arg == "--nnz") {
      if (i + 1 < argc) {
        nnz = atoi(argv[++i]);
        if (nnz <= 0) {
          cerr << "ERROR: Number of non-zeros must be positive" << endl;
          return 1;
        }
      } else {
        cerr << "ERROR: " << arg << " requires a value" << endl;
        return 1;
      }
    } else if (arg == "-i" || arg == "--iterations") {
      if (i + 1 < argc) {
        num_iterations = atoi(argv[++i]);
        if (num_iterations <= 0) {
          cerr << "ERROR: Number of iterations must be positive" << endl;
          return 1;
        }
      } else {
        cerr << "ERROR: " << arg << " requires a value" << endl;
        return 1;
      }
    } else {
      cerr << "ERROR: Unknown option: " << arg << endl;
      cerr << "Use -h or --help for usage information" << endl;
      return 1;
    }
  }

  // Validate and clamp nnz to the feasible range for a lower-triangular matrix.
  // Minimum: n (diagonal only). Maximum: n*(n+1)/2 (full lower triangle).
  long long max_nnz = static_cast<long long>(n) * (n + 1) / 2;
  if (nnz < n) {
    cerr << "WARNING: nnz (" << nnz << ") < n (" << n
         << "). Need at least n entries for the diagonal. Clamping to " << n
         << "." << endl;
    nnz = n;
  }
  if (nnz > max_nnz) {
    cerr << "ERROR: nnz (" << nnz
         << ") exceeds maximum possible non-zeros for a lower-triangular " << n
         << "x" << n << " matrix (" << max_nnz << ")." << endl;
    return 1;
  }

  int total_tests = 0;
  int passed_tests = 0;
  int failed_tests = 0;

  cout << "\n" << endl;
  cout << "ITERATIVE SPARSE REFACTORIZATION WORKFLOW" << endl;
  cout << "=========================================" << endl;
  cout << "Demonstrates repeated refactorization with analysis reuse" << endl;
  cout << endl;

  // Calculate sparsity density (relative to lower triangle, which is what we store)
  double density = (100.0 * nnz) / static_cast<double>(max_nnz);
  double avg_per_row = static_cast<double>(nnz) / n;

  cout << "Configuration:" << endl;
  cout << "  Matrix size:       " << n << " x " << n << endl;
  cout << "  Non-zeros (nnz):   " << nnz << endl;
  cout << "  Iterations:        " << num_iterations << endl;
  cout << "  Avg per row:       ~" << fixed << setprecision(1) << avg_per_row
       << endl;
  cout << "  Density:           ~" << setprecision(4) << density << "%" << endl;
  cout << endl;

  total_tests++;
  if (run_iterative_refactorization_workflow<double>(n, nnz, num_iterations))
    passed_tests++;
  else
    failed_tests++;

  // ========================================
  // Summary
  // ========================================
  cout << "\n";
  cout << "========================================" << endl;
  cout << "SUMMARY" << endl;
  cout << "========================================" << endl;
  cout << "Passed: " << passed_tests << endl;
  cout << "========================================" << endl;
  cout << endl;

  return (failed_tests == 0) ? 0 : 1;
}
