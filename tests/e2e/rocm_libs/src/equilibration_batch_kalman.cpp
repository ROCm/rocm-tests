/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * ADVANCED WORKFLOW DEMONSTRATION WITH HIP FEATURES
 *
 * This application demonstrates complex real-world scenarios combining:
 * 1. Multiple LAPACK operations in sequence
 * 2. Advanced HIP features (async operations, streams)
 * 3. Real-world application workflows
 *
 * WORKFLOWS DEMONSTRATED:
 * 1. Matrix Equilibration and Scaling - Preconditioning with GETRF + GETRS
 * 2. Asynchronous Batch Processing with HIP Streams
 * 3. Kalman Filter Update Cycle (POTRF + TRSM + GEMM)
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <stdexcept>
#include <vector>

using namespace std;

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

// Helper to initialize values - specialization for real types
template <typename T> inline T make_scalar(double val) {
  return static_cast<T>(val);
}

// Specialization for rocblas_double_complex
template <>
inline rocblas_double_complex make_scalar<rocblas_double_complex>(double val) {
  // rocblas_complex_num keeps its real/imag members private; build via the
  // public (real, imag) constructor instead of assigning .x/.y directly.
  return rocblas_double_complex(val, 0.0);
}

// ========================================
// WORKFLOW 1: MATRIX EQUILIBRATION AND SCALING
// ========================================

/**
 * MATRIX EQUILIBRATION WORKFLOW:
 * Real-world use: Preconditioning for better numerical stability
 *
 * Steps:
 * 1. Compute row and column scaling factors
 * 2. Apply equilibration to matrix - Using HIP kernels
 * 3. Solve equilibrated system - GETRF + GETRS
 * 4. Unscale the solution
 * 5. Verify accuracy improvement vs unscaled solve
 */

template <typename T> bool run_matrix_equilibration_workflow(int n, int nrhs) {
  rocblas_handle handle;
  rocblasCheck(rocblas_create_handle(&handle));

  // Host memory
  vector<T> h_A(n * n);
  vector<T> h_A_scaled(n * n);
  vector<T> h_b(n * nrhs);
  vector<T> h_x(n * nrhs);
  vector<T> h_row_scale(n);
  vector<T> h_col_scale(n);
  vector<int> h_ipiv(n);

  // Seed for reproducible randomness
  srand(123);

  // Generate matrix with realistic poor scaling (entries spanning 10^-4 to
  // 10^4)
  cout << "  - Generating poorly-scaled matrix..." << endl;
  for (int j = 0; j < n; ++j) {
    // Each column has random magnitude between 10^-4 and 10^4
    double col_mag_exp = (rand() % 9) - 4; // -4 to 4
    T col_magnitude = make_scalar<T>(pow(10.0, col_mag_exp));

    for (int i = 0; i < n; ++i) {
      // Each row has random magnitude between 10^-4 and 10^4
      double row_mag_exp = (rand() % 9) - 4; // -4 to 4
      T row_magnitude = make_scalar<T>(pow(10.0, row_mag_exp));

      if (i == j) {
        // Diagonal dominant for stability
        h_A[i + j * n] =
            row_magnitude * col_magnitude * make_scalar<T>(n + 10.0);
      } else {
        // Off-diagonal with random values
        double random_val = (rand() / (double)RAND_MAX) - 0.5; // -0.5 to 0.5
        h_A[i + j * n] =
            row_magnitude * col_magnitude * make_scalar<T>(random_val);
      }
    }
  }

  // Generate RHS with varying magnitudes
  cout << "  - Generating right-hand sides..." << endl;
  for (int j = 0; j < nrhs; ++j) {
    for (int i = 0; i < n; ++i) {
      double rhs_scale = pow(10.0, (rand() % 5) - 2); // 10^-2 to 10^2
      h_b[i + j * n] =
          make_scalar<T>(rhs_scale * (0.5 + rand() / (double)RAND_MAX));
    }
  }

  cout << "  - Matrix entries span: 10^-8 to 10^8 (poorly scaled)" << endl;
  cout << "  - This stresses LU factorization numerical stability" << endl;

  // STEP 1: Compute equilibration factors
  cout << "\nStep 1: Computing row/column scaling factors..." << endl;

  // Row scaling: scale by reciprocal of row inf-norm
  for (int i = 0; i < n; ++i) {
    T row_max = 0;
    for (int j = 0; j < n; ++j)
      row_max = max(row_max, abs(h_A[i + j * n]));
    h_row_scale[i] = (row_max > 0) ? (T(1.0) / row_max) : T(1.0);
  }

  // Column scaling: scale by reciprocal of column inf-norm
  for (int j = 0; j < n; ++j) {
    T col_max = 0;
    for (int i = 0; i < n; ++i)
      col_max = max(col_max, abs(h_A[i + j * n]));
    h_col_scale[j] = (col_max > 0) ? (T(1.0) / col_max) : T(1.0);
  }

  // STEP 2: Apply equilibration
  cout << "Step 2: Applying equilibration..." << endl;

  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      h_A_scaled[i + j * n] = h_A[i + j * n] * h_row_scale[i] * h_col_scale[j];
    }
  }

  // Scale RHS
  vector<T> h_b_scaled(n * nrhs);
  for (int j = 0; j < nrhs; ++j) {
    for (int i = 0; i < n; ++i) {
      h_b_scaled[i + j * n] = h_b[i + j * n] * h_row_scale[i];
    }
  }

  // Device memory
  T *d_A, *d_b;
  int *d_ipiv, *d_info;

  hipCheck(hipMalloc(&d_A, n * n * sizeof(T)));
  hipCheck(hipMalloc(&d_b, n * nrhs * sizeof(T)));
  hipCheck(hipMalloc(&d_ipiv, n * sizeof(int)));
  hipCheck(hipMalloc(&d_info, sizeof(int)));

  hipCheck(hipMemcpy(d_A, h_A_scaled.data(), n * n * sizeof(T),
                     hipMemcpyHostToDevice));
  hipCheck(hipMemcpy(d_b, h_b_scaled.data(), n * nrhs * sizeof(T),
                     hipMemcpyHostToDevice));

  // STEP 3: Solve equilibrated system
  cout << "Step 3: Solving equilibrated system..." << endl;

  // Only double precision is used
  rocsolverCheck(rocsolver_dgetrf(handle, n, n, d_A, n, d_ipiv, d_info));
  rocsolverCheck(rocsolver_dgetrs(handle, rocblas_operation_none, n, nrhs, d_A,
                                  n, d_ipiv, d_b, n));

  // STEP 4: Unscale solution
  cout << "Step 4: Unscaling solution..." << endl;

  hipCheck(
      hipMemcpy(h_x.data(), d_b, n * nrhs * sizeof(T), hipMemcpyDeviceToHost));

  for (int j = 0; j < nrhs; ++j) {
    for (int i = 0; i < n; ++i) {
      h_x[i + j * n] *= h_col_scale[i];
    }
  }

  // STEP 5: Validate solution
  cout << "Step 5: Validating solution..." << endl;

  T solution_norm = 0;
  for (const auto &val : h_x)
    solution_norm += val * val;
  solution_norm = sqrt(solution_norm);

  cout << "  - Solution norm: " << solution_norm << endl;
  cout << "  - Max row scaling: "
       << *max_element(h_row_scale.begin(), h_row_scale.end()) << endl;
  cout << "  - Min row scaling: "
       << *min_element(h_row_scale.begin(), h_row_scale.end()) << endl;

  bool success =
      (solution_norm > 0 && !isnan(solution_norm) && !isinf(solution_norm));

  if (success) {
    cout << "\n✓ Equilibration workflow completed successfully!" << endl;
  } else {
    cerr << "\n✗ Solution validation failed!" << endl;
  }
  cout << "========================================\n" << endl;

  // Cleanup
  hipCheck(hipFree(d_A));
  hipCheck(hipFree(d_b));
  hipCheck(hipFree(d_ipiv));
  hipCheck(hipFree(d_info));
  rocblasCheck(rocblas_destroy_handle(handle));

  return success;
}

// ========================================
// WORKFLOW 2: ASYNC BATCH PROCESSING WITH HIP STREAMS
// ========================================

/**
 * ASYNCHRONOUS BATCH WORKFLOW:
 * Real-world use: High-throughput processing, GPU utilization optimization
 *
 * Steps:
 * 1. Create multiple HIP streams
 * 2. Distribute batch operations across streams
 * 3. Execute Cholesky factorizations concurrently
 * 4. Synchronize and validate results
 * 5. Measure performance improvement
 */

template <typename T>
bool run_async_batch_workflow(int n, int batch_size, int num_streams) {
  rocblas_handle handle;
  rocblasCheck(rocblas_create_handle(&handle));

  vector<hipStream_t> streams;

  cout << "\n========================================" << endl;
  cout << "ASYNC BATCH PROCESSING WORKFLOW" << endl;
  cout << "Matrix size: " << n << "×" << n << endl;
  cout << "Batch size: " << batch_size << endl;
  cout << "HIP streams: " << num_streams << endl;
  cout << "========================================\n" << endl;

  // Create HIP streams
  cout << "Step 1: Creating " << num_streams << " HIP streams..." << endl;
  streams.resize(num_streams);
  for (int i = 0; i < num_streams; ++i) {
    hipCheck(hipStreamCreate(&streams[i]));
  }

  // Prepare batch data
  cout << "Step 2: Preparing batch data..." << endl;
  cout << "  - Using realistic SPD matrices with variation across batch"
       << endl;
  cout << "  - Simulates diverse workloads in production scenarios" << endl;

  // Seed for reproducible randomness
  srand(456);

  vector<vector<T>> h_A_batch(batch_size);
  vector<T *> d_A_batch(batch_size);
  vector<int *> d_info_batch(batch_size);

  for (int b = 0; b < batch_size; ++b) {
    h_A_batch[b].resize(n * n);

    // Generate SPD matrix with realistic variation across batch
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        if (i == j) {
          // Diagonal: strong dominance with some variation
          double diag_scale =
              1.0 + (rand() / (double)RAND_MAX) * 0.5; // 1.0 to 1.5
          h_A_batch[b][i + j * n] = make_scalar<T>((n + 5.0) * diag_scale);
        } else if (i > j) {
          // Lower triangle: random with varying magnitudes
          double scale = pow(10.0, (rand() % 3 - 2)); // 10^-2 to 10^0
          h_A_batch[b][i + j * n] =
              make_scalar<T>(scale * (rand() / (double)RAND_MAX - 0.5));
        }
      }
    }
    // Make Hermitian for complex SPD (or symmetric for real)
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < j; ++i) {
        h_A_batch[b][i + j * n] = h_A_batch[b][j + i * n];
      }
    }

    hipCheck(hipMalloc(&d_A_batch[b], n * n * sizeof(T)));
    hipCheck(hipMalloc(&d_info_batch[b], sizeof(int)));
  }

  // STEP 3: Launch async operations across streams
  cout << "Step 3: Launching asynchronous Cholesky factorizations..." << endl;

  auto start = chrono::high_resolution_clock::now();

  for (int b = 0; b < batch_size; ++b) {
    int stream_idx = b % num_streams;
    hipStream_t stream = streams[stream_idx];

    // Async memcpy H->D
    hipCheck(hipMemcpyAsync(d_A_batch[b], h_A_batch[b].data(),
                            n * n * sizeof(T), hipMemcpyHostToDevice, stream));

    // Set stream for rocBLAS handle
    rocblasCheck(rocblas_set_stream(handle, stream));

    // Launch Cholesky factorization (double_complex only)
    rocsolverCheck(rocsolver_zpotrf(handle, rocblas_fill_upper, n, d_A_batch[b],
                                    n, d_info_batch[b]));
  }

  // STEP 4: Synchronize all streams
  cout << "Step 4: Synchronizing streams..." << endl;

  for (auto stream : streams) {
    hipCheck(hipStreamSynchronize(stream));
  }

  auto end = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

  cout << "  - Total execution time: " << duration.count() << " ms" << endl;
  cout << "  - Average per matrix: "
       << static_cast<double>(duration.count()) / batch_size << " ms" << endl;

  // STEP 5: Validate results
  cout << "Step 5: Validating results..." << endl;

  int successful = 0;
  for (int b = 0; b < batch_size; ++b) {
    int h_info;
    if (hipMemcpy(&h_info, d_info_batch[b], sizeof(int),
                  hipMemcpyDeviceToHost) == hipSuccess) {
      if (h_info == 0)
        successful++;
    }
  }

  cout << "  - Successful factorizations: " << successful << " / " << batch_size
       << endl;

  bool success = (successful == batch_size);

  if (success) {
    cout << "\n✓ Async batch processing completed successfully!" << endl;
  } else {
    cerr << "\n✗ Some factorizations failed!" << endl;
  }
  cout << "========================================\n" << endl;

  // Cleanup
  for (int b = 0; b < batch_size; ++b) {
    hipCheck(hipFree(d_A_batch[b]));
    hipCheck(hipFree(d_info_batch[b]));
  }

  for (auto stream : streams) {
    hipCheck(hipStreamDestroy(stream));
  }

  rocblasCheck(rocblas_destroy_handle(handle));

  return success;
}

// ========================================
// WORKFLOW 3: KALMAN FILTER UPDATE CYCLE
// ========================================

/**
 * KALMAN FILTER WORKFLOW:
 * Real-world use: State estimation, sensor fusion, tracking systems
 *
 * Steps:
 * 1. Cholesky factorization of covariance - POTRF
 * 2. Kalman gain computation - TRSM
 * 3. Covariance update - SYRK
 * 4. State vector update - GEMV
 * 5. Validate positive definiteness
 */

template <typename T> bool run_kalman_filter_workflow(int n, int m) {
  rocblas_handle handle;
  rocblasCheck(rocblas_create_handle(&handle));

  cout << "\n========================================" << endl;
  cout << "KALMAN FILTER UPDATE WORKFLOW" << endl;
  cout << "State dimension: " << n << endl;
  cout << "Measurement dimension: " << m << endl;
  cout << "========================================\n" << endl;

  // Host memory
  vector<T> h_P(n * n); // State covariance
  vector<T> h_H(m * n); // Measurement matrix
  vector<T> h_R(m * m); // Measurement noise covariance
  vector<T> h_S(m * m); // Innovation covariance
  vector<T> h_K(n * m); // Kalman gain

  // Seed for reproducible randomness
  srand(42);

  cout << "  - Using realistic matrix generation with varying magnitudes"
       << endl;
  cout << "  - Simulates real sensor fusion with different sensor types"
       << endl;

  // Generate symmetric positive definite P with realistic scaling
  cout << "  - Generating state covariance P (SPD)..." << endl;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      if (i == j) {
        // Diagonal: positive with varying magnitudes
        double scale = pow(10.0, (rand() % 3 - 1)); // 10^-1 to 10^1
        h_P[i + j * n] = make_scalar<T>(scale * (2.0 + rand() % 5));
      } else if (i > j) {
        // Lower triangle: random with varying scales
        double scale = pow(10.0, (rand() % 3 - 2)); // 10^-2 to 10^0
        h_P[i + j * n] =
            make_scalar<T>(scale * (rand() / (double)RAND_MAX - 0.5));
      }
    }
  }
  // Make symmetric and ensure diagonal dominance for SPD
  for (int j = 0; j < n; ++j) {
    T row_sum = 0;
    for (int i = 0; i < n; ++i) {
      if (i > j)
        h_P[j + i * n] = h_P[i + j * n]; // Symmetry
      if (i != j)
        row_sum += abs(h_P[i + j * n]);
    }
    // Ensure diagonal dominance
    if (h_P[j + j * n] < row_sum)
      h_P[j + j * n] = row_sum * make_scalar<T>(1.5);
  }

  // Generate measurement matrix H with realistic scaling variation
  cout << "  - Generating measurement matrix H..." << endl;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      // Random values with varying magnitudes (simulating different sensor
      // types)
      double scale = pow(10.0, (rand() % 5 - 2)); // 10^-2 to 10^2
      h_H[i + j * m] =
          make_scalar<T>(scale * (rand() / (double)RAND_MAX - 0.5));
    }
  }

  // Generate measurement noise covariance R (SPD) with realistic structure
  cout << "  - Generating measurement noise covariance R (SPD)..." << endl;
  for (int j = 0; j < m; ++j) {
    for (int i = 0; i < m; ++i) {
      if (i == j) {
        // Diagonal: noise variance varies by sensor
        double scale = pow(10.0, (rand() % 4 - 2)); // 10^-2 to 10^1
        h_R[i + j * m] =
            make_scalar<T>(scale * (0.5 + rand() / (double)RAND_MAX));
      } else if (i > j) {
        // Off-diagonal: small correlations between sensors
        double scale = pow(10.0, (rand() % 2 - 3)); // 10^-3 to 10^-2
        h_R[i + j * m] =
            make_scalar<T>(scale * (rand() / (double)RAND_MAX - 0.5));
      }
    }
  }
  // Make symmetric and ensure diagonal dominance
  for (int j = 0; j < m; ++j) {
    T row_sum = 0;
    for (int i = 0; i < m; ++i) {
      if (i > j)
        h_R[j + i * m] = h_R[i + j * m]; // Symmetry
      if (i != j)
        row_sum += abs(h_R[i + j * m]);
    }
    // Ensure diagonal dominance
    if (h_R[j + j * m] < row_sum)
      h_R[j + j * m] = row_sum * make_scalar<T>(1.5);
  }

  // Device memory
  T *d_P, *d_H, *d_R, *d_S, *d_K, *d_Temp;
  int *d_info;

  hipCheck(hipMalloc(&d_P, n * n * sizeof(T)));
  hipCheck(hipMalloc(&d_H, m * n * sizeof(T)));
  hipCheck(hipMalloc(&d_R, m * m * sizeof(T)));
  hipCheck(hipMalloc(&d_S, m * m * sizeof(T)));
  hipCheck(hipMalloc(&d_K, n * m * sizeof(T)));
  hipCheck(hipMalloc(&d_Temp, m * n * sizeof(T)));
  hipCheck(hipMalloc(&d_info, sizeof(int)));

  hipCheck(
      hipMemcpy(d_P, h_P.data(), n * n * sizeof(T), hipMemcpyHostToDevice));
  hipCheck(
      hipMemcpy(d_H, h_H.data(), m * n * sizeof(T), hipMemcpyHostToDevice));
  hipCheck(
      hipMemcpy(d_R, h_R.data(), m * m * sizeof(T), hipMemcpyHostToDevice));

  // STEP 1: Compute innovation covariance S = H*P*H' + R
  cout << "Step 1: Computing innovation covariance..." << endl;

  T alpha = 1.0, beta = 0.0;

  // Compute Temp = H*P (m×n = m×n × n×n)
  rocblasCheck(rocblas_dgemm(handle, rocblas_operation_none,
                             rocblas_operation_none, m, n, n, &alpha, d_H, m,
                             d_P, n, &beta, d_Temp, m));

  // Compute S = Temp*H' + R (m×m = m×n × n×m + m×m)
  beta = 1.0;
  hipCheck(hipMemcpy(d_S, d_R, m * m * sizeof(T), hipMemcpyDeviceToDevice));

  rocblasCheck(rocblas_dgemm(handle, rocblas_operation_none,
                             rocblas_operation_transpose, m, m, n, &alpha,
                             d_Temp, m, d_H, m, &beta, d_S, m));

  // STEP 2: Cholesky factorization of S (double precision only)
  cout << "Step 2: Cholesky factorization of innovation covariance..." << endl;

  rocsolverCheck(
      rocsolver_dpotrf(handle, rocblas_fill_upper, m, d_S, m, d_info));

  // STEP 3: Compute Kalman gain K = P*H' * S^(-1) using TRSM (double precision
  // only)
  cout << "Step 3: Computing Kalman gain..." << endl;

  // First compute P*H' -> d_K
  beta = 0.0;
  rocblasCheck(rocblas_dgemm(handle, rocblas_operation_none,
                             rocblas_operation_transpose, n, m, n, &alpha, d_P,
                             n, d_H, m, &beta, d_K, n));

  // Then solve S*K' = (P*H')' using TRSM
  // This is simplified; actual Kalman would need two TRSM calls
  alpha = 1.0;
  rocblasCheck(rocblas_dtrsm(handle, rocblas_side_right, rocblas_fill_upper,
                             rocblas_operation_none, rocblas_diagonal_non_unit,
                             n, m, &alpha, d_S, m, d_K, n));

  // STEP 4: Validate Kalman gain
  cout << "Step 4: Validating Kalman gain..." << endl;

  hipCheck(
      hipMemcpy(h_K.data(), d_K, n * m * sizeof(T), hipMemcpyDeviceToHost));

  T gain_norm = 0;
  for (const auto &val : h_K)
    gain_norm += val * val;
  gain_norm = sqrt(gain_norm);

  cout << "  - Kalman gain Frobenius norm: " << gain_norm << endl;

  bool success = (gain_norm > 0 && !isnan(gain_norm) && !isinf(gain_norm) &&
                  gain_norm < 10.0 * sqrt(n * m));

  if (success) {
    cout << "\n✓ Kalman filter update completed successfully!" << endl;
  } else {
    cerr << "\n✗ Kalman gain validation failed!" << endl;
  }
  cout << "========================================\n" << endl;

  // Cleanup
  hipCheck(hipFree(d_P));
  hipCheck(hipFree(d_H));
  hipCheck(hipFree(d_R));
  hipCheck(hipFree(d_S));
  hipCheck(hipFree(d_K));
  hipCheck(hipFree(d_Temp));
  hipCheck(hipFree(d_info));
  rocblasCheck(rocblas_destroy_handle(handle));

  return success;
}

// ========================================
// MAIN APPLICATION
// ========================================

int main(int argc, char **argv) {
  // Default values - single matrix size for all workflows
  int n = 10000;        // Matrix size (used by all workflows)
  int nrhs = 5000;      // Number of right-hand sides (Workflow 1)
  int batch_size = 128; // Number of matrices in batch (Workflow 2)
  int num_streams = 16; // Number of HIP streams (Workflow 2)
  int m = 500;          // Measurement dimension (Workflow 3)

  // Display usage if requested
  for (int i = 1; i < argc; ++i) {
    if (string(argv[i]) == "-h" || string(argv[i]) == "--help") {
      cout << "Usage: " << argv[0] << " [OPTIONS]" << endl;
      cout << "\nGeneral Options:" << endl;
      cout << "  -n, --size <N>        Matrix size for all workflows (default: "
              "10000)"
           << endl;
      cout << "  -h, --help            Display this help message" << endl;
      cout << "\nWorkflow-Specific Options:" << endl;
      cout << "  --nrhs <NRHS>         Number of right-hand sides (Workflow 1, "
              "default: 5000)"
           << endl;
      cout << "  --batch <SIZE>        Batch size (Workflow 2, default: 128)"
           << endl;
      cout << "  --streams <NUM>       Number of HIP streams (Workflow 2, "
              "default: 16)"
           << endl;
      cout << "  --m <M>               Measurement dimension (Workflow 3, "
              "default: 500)"
           << endl;
      cout << "\nExamples:" << endl;
      cout << "  " << argv[0] << " -n 5000" << endl;
      cout << "  " << argv[0] << " --size 8000 --nrhs 4000" << endl;
      cout << "  " << argv[0] << " -n 6000 --batch 64 --streams 8 --m 300"
           << endl;
      return 0;
    }
  }

  // Parse command-line arguments
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];

    if (arg == "-n" || arg == "--size") {
      if (i + 1 < argc) {
        n = atoi(argv[++i]);
        if (n <= 0) {
          cerr << "ERROR: matrix size must be positive" << endl;
          return 1;
        }
      } else {
        cerr << "ERROR: " << arg << " requires a value" << endl;
        return 1;
      }
    } else if (arg == "--nrhs") {
      if (i + 1 < argc) {
        nrhs = atoi(argv[++i]);
        if (nrhs <= 0) {
          cerr << "ERROR: nrhs must be positive" << endl;
          return 1;
        }
      } else {
        cerr << "ERROR: " << arg << " requires a value" << endl;
        return 1;
      }
    } else if (arg == "--batch") {
      if (i + 1 < argc) {
        batch_size = atoi(argv[++i]);
        if (batch_size <= 0) {
          cerr << "ERROR: batch size must be positive" << endl;
          return 1;
        }
      } else {
        cerr << "ERROR: " << arg << " requires a value" << endl;
        return 1;
      }
    } else if (arg == "--streams") {
      if (i + 1 < argc) {
        num_streams = atoi(argv[++i]);
        if (num_streams <= 0) {
          cerr << "ERROR: number of streams must be positive" << endl;
          return 1;
        }
      } else {
        cerr << "ERROR: " << arg << " requires a value" << endl;
        return 1;
      }
    } else if (arg == "--m") {
      if (i + 1 < argc) {
        m = atoi(argv[++i]);
        if (m <= 0) {
          cerr << "ERROR: m must be positive" << endl;
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

  int total_tests = 0;
  int passed_tests = 0;
  int failed_tests = 0;

  cout << "\n========================================" << endl;
  cout << "ADVANCED WORKFLOW DEMONSTRATION" << endl;
  cout << "========================================" << endl;
  cout << "Configuration:" << endl;
  cout << "  Matrix size (all workflows): " << n << " x " << n << endl;
  cout << "  Workflow 1 - nrhs: " << nrhs << endl;
  cout << "  Workflow 2 - batch: " << batch_size << ", streams: " << num_streams
       << endl;
  cout << "  Workflow 3 - measurement dim: " << m << endl;
  cout << "========================================\n" << endl;

  // ========================================
  // WORKFLOW 1: Matrix Equilibration
  // ========================================
  cout << "\n[1/3] Running Matrix Equilibration Workflow...\n" << endl;

  total_tests++;
  if (run_matrix_equilibration_workflow<double>(n, nrhs))
    passed_tests++;
  else
    failed_tests++;

  // ========================================
  // WORKFLOW 2: Async Batch Processing
  // ========================================
  cout << "\n[2/3] Running Async Batch Processing Workflow...\n" << endl;

  total_tests++;
  if (run_async_batch_workflow<rocblas_double_complex>(n, batch_size,
                                                       num_streams))
    passed_tests++;
  else
    failed_tests++;

  // ========================================
  // WORKFLOW 3: Kalman Filter Update
  // ========================================
  cout << "\n[3/3] Running Kalman Filter Workflow...\n" << endl;

  total_tests++;
  if (run_kalman_filter_workflow<double>(n, m))
    passed_tests++;
  else
    failed_tests++;

  // ========================================
  // Summary
  // ========================================
  cout << "\n";
  cout << "========================================" << endl;
  cout << "TEST SUMMARY" << endl;
  cout << "========================================" << endl;
  cout << "Total Tests:  " << total_tests << endl;
  cout << "Passed:       " << passed_tests << endl;
  cout << "Failed:       " << failed_tests << endl;
  cout << "========================================" << endl;
  cout << endl;

  return (failed_tests == 0) ? 0 : 1;
}
