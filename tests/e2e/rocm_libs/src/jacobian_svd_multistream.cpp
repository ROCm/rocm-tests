/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * STRESS TEST: Multi-Stream GESVDJ (Jacobi SVD)
 *
 * This test specifically targets GESVDJ to diagnose concurrency issues
 * that appear when running many operations across multiple streams.
 * ************************************************************************ */

#include <hip/hip_runtime_api.h>
#include <hipsolver/hipsolver.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

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

#define hipsolverCheck(call)                                                   \
  do {                                                                         \
    hipsolverStatus_t status = call;                                           \
    if (status != HIPSOLVER_STATUS_SUCCESS) {                                  \
      std::cerr << "hipSOLVER Error at " << __FILE__ << ":" << __LINE__        \
                << " - Status code: " << status << "\n";                       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Start with smaller scale to identify where it breaks
constexpr int NUM_STREAMS = 100;
constexpr int OPS_PER_STREAM = 100;
constexpr int TOTAL_OPS = NUM_STREAMS * OPS_PER_STREAM;

// MATRIX_SIZE will be set from command line or default
int MATRIX_SIZE = 64; // Default size

class SVDOperation {
public:
  int stream_id;
  int op_id;
  hipStream_t stream;
  hipsolverHandle_t handle;
  hipsolverGesvdjInfo_t gesvdj_params;

  // Problem data
  int M, N, lda, ldu, ldv;
  double *dA;
  double *dS;
  double *dU;
  double *dV;
  double *dWork;
  int *dInfo;
  int lwork;

  // Host data
  std::vector<double> hA_input;
  std::vector<double> hS;
  std::vector<double> hU;
  std::vector<double> hV;
  std::vector<int> hInfo;

  // Convergence info
  double residual;
  int sweeps;

  // Status
  hipsolverStatus_t status;
  bool verified;

  SVDOperation()
      : stream_id(0), op_id(0), stream(nullptr), handle(nullptr),
        gesvdj_params(nullptr), M(0), N(0), lda(0), ldu(0), ldv(0), dA(nullptr),
        dS(nullptr), dU(nullptr), dV(nullptr), dWork(nullptr), dInfo(nullptr),
        lwork(0), residual(0.0), sweeps(0), status(HIPSOLVER_STATUS_SUCCESS),
        verified(false) {}

  ~SVDOperation() { cleanup(); }

  void init(int sid, int oid, hipStream_t s, hipsolverHandle_t h) {
    stream_id = sid;
    op_id = oid;
    stream = s;
    handle = h;
    M = MATRIX_SIZE;
    N = MATRIX_SIZE;
    lda = MATRIX_SIZE;
    ldu = MATRIX_SIZE;
    ldv = MATRIX_SIZE;

    int min_mn = std::min(M, N);
    size_t size_A = lda * N;
    size_t size_U = ldu * M;
    size_t size_V = ldv * N;

    // Host memory
    hA_input.resize(size_A);
    hS.resize(min_mn);
    hU.resize(size_U);
    hV.resize(size_V);
    hInfo.resize(1);

    // Initialize with random but well-conditioned matrix
    std::mt19937 gen(stream_id * 1000 + op_id + 12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < N; j++) {
      for (int i = 0; i < M; i++) {
        if (i == j)
          hA_input[i + j * lda] = 5.0 + dist(gen); // Dominant diagonal
        else
          hA_input[i + j * lda] = dist(gen);
      }
    }

    // Create GESVDJ parameters
    status = hipsolverCreateGesvdjInfo(&gesvdj_params);
    if (status != HIPSOLVER_STATUS_SUCCESS) {
      std::cerr << "ERROR [Stream " << stream_id << ", Op " << op_id
                << "]: CreateGesvdjInfo failed: " << status << "\n";
      return;
    }

    hipsolverCheck(hipsolverXgesvdjSetTolerance(gesvdj_params, 1.0e-7));
    hipsolverCheck(hipsolverXgesvdjSetMaxSweeps(gesvdj_params, 50));
    hipsolverCheck(hipsolverXgesvdjSetSortEig(gesvdj_params, 1));

    // Device memory
    hipCheck(hipMalloc(&dA, sizeof(double) * size_A));
    hipCheck(hipMalloc(&dS, sizeof(double) * min_mn));
    hipCheck(hipMalloc(&dU, sizeof(double) * size_U));
    hipCheck(hipMalloc(&dV, sizeof(double) * size_V));
    hipCheck(hipMalloc(&dInfo, sizeof(int)));

    // Copy input asynchronously
    hipCheck(hipMemcpyAsync(dA, hA_input.data(), sizeof(double) * size_A,
                            hipMemcpyHostToDevice, stream));

    // Get workspace size - CRITICAL: Check if this works per-stream
    status = hipsolverDgesvdj_bufferSize(handle, HIPSOLVER_EIG_MODE_VECTOR, 0,
                                         M, N, dA, lda, dS, dU, ldu, dV, ldv,
                                         &lwork, gesvdj_params);

    if (status != HIPSOLVER_STATUS_SUCCESS) {
      std::cerr << "ERROR [Stream " << stream_id << ", Op " << op_id
                << "]: bufferSize failed: " << status << "\n";
      return;
    }

    if (lwork == 0) {
      std::cerr << "ERROR [Stream " << stream_id << ", Op " << op_id
                << "]: lwork = 0\n";
      status = HIPSOLVER_STATUS_INVALID_VALUE;
      return;
    }

    hipCheck(hipMalloc(&dWork, sizeof(double) * lwork));
  }

  void launch() {
    status = hipsolverDgesvdj(handle, HIPSOLVER_EIG_MODE_VECTOR, 0, M, N, dA,
                              lda, dS, dU, ldu, dV, ldv, dWork, lwork, dInfo,
                              gesvdj_params);
  }

  void copyResults() {
    int min_mn = std::min(M, N);
    size_t size_U = ldu * M;
    size_t size_V = ldv * N;

    hipCheck(hipMemcpyAsync(hS.data(), dS, sizeof(double) * min_mn,
                            hipMemcpyDeviceToHost, stream));
    hipCheck(hipMemcpyAsync(hU.data(), dU, sizeof(double) * size_U,
                            hipMemcpyDeviceToHost, stream));
    hipCheck(hipMemcpyAsync(hV.data(), dV, sizeof(double) * size_V,
                            hipMemcpyDeviceToHost, stream));
    hipCheck(hipMemcpyAsync(hInfo.data(), dInfo, sizeof(int),
                            hipMemcpyDeviceToHost, stream));
  }

  void getConvergenceInfo() {
    if (gesvdj_params) {
      hipsolverCheck(
          hipsolverXgesvdjGetResidual(handle, gesvdj_params, &residual));
      hipsolverCheck(hipsolverXgesvdjGetSweeps(handle, gesvdj_params, &sweeps));
    }
  }

  bool verify() {
    // Check info
    if (hInfo[0] != 0) {
      std::cout << "  [Stream " << stream_id << ", Op " << op_id
                << "] FAILED: info = " << hInfo[0] << "\n";
      return false;
    }

    // Check sweeps
    if (sweeps == 0) {
      std::cout << "  [Stream " << stream_id << ", Op " << op_id
                << "] FAILED: 0 sweeps (algorithm didn't run!)\n";
      return false;
    }

    // Check singular values are valid
    int min_mn = std::min(M, N);
    bool all_zero = true;
    bool has_negative = false;

    for (int i = 0; i < min_mn; i++) {
      if (std::abs(hS[i]) > 1e-10)
        all_zero = false;
      if (hS[i] < 0)
        has_negative = true;
    }

    if (all_zero) {
      std::cout << "  [Stream " << stream_id << ", Op " << op_id
                << "] FAILED: all singular values are zero\n";
      return false;
    }

    if (has_negative) {
      std::cout << "  [Stream " << stream_id << ", Op " << op_id
                << "] FAILED: negative singular values\n";
      return false;
    }

    // **PROPER SVD VERIFICATION: A = U * Σ * V^T**

    // Step 1: Compute U * Σ (store in temp matrix)
    // For square matrices with full SVD: U is M×M, Σ is M×M diagonal
    std::vector<double> US(M * M, 0.0);
    for (int j = 0; j < M; j++) // Column of U*Σ
    {
      for (int i = 0; i < M; i++) // Row of U*Σ
      {
        // Column j of U*Σ = column j of U times S[j]
        if (j < min_mn) {
          US[i + j * M] = hU[i + j * ldu] * hS[j];
        } else {
          US[i + j * M] = 0.0;
        }
      }
    }

    // Step 2: Compute (U * Σ) * V^T = US * V^T
    // Result is M×N (same as A)
    std::vector<double> A_reconstructed(M * N, 0.0);
    for (int j = 0; j < N; j++) // Column of result
    {
      for (int i = 0; i < M; i++) // Row of result
      {
        double sum = 0.0;
        // (US * V^T)[i,j] = sum_k US[i,k] * V^T[k,j]
        //                 = sum_k US[i,k] * V[j,k]
        for (int k = 0; k < min_mn; k++) {
          // US[i,k] in column-major: US[i + k*M]
          // V[j,k] in column-major: hV[j + k*ldv]
          sum += US[i + k * M] * hV[j + k * ldv];
        }
        A_reconstructed[i + j * lda] = sum;
      }
    }

    // Step 3: Calculate error ||A - A_reconstructed|| / ||A||
    double error_norm = 0.0;
    double A_norm = 0.0;

    for (int j = 0; j < N; j++) {
      for (int i = 0; i < M; i++) {
        int idx = i + j * lda;
        double diff = hA_input[idx] - A_reconstructed[idx];
        error_norm += diff * diff;
        A_norm += hA_input[idx] * hA_input[idx];
      }
    }

    error_norm = std::sqrt(error_norm);
    A_norm = std::sqrt(A_norm);

    double relative_error = error_norm / A_norm;

    // Step 4: Check if error is acceptable
    const double tolerance =
        2.0e-5; // Tolerance for stress testing (allows for numerical variation)

    if (relative_error > tolerance) {
      std::cout << "  [Stream " << stream_id << ", Op " << op_id
                << "] FAILED: SVD reconstruction error = " << std::scientific
                << std::setprecision(6) << relative_error
                << " (tolerance = " << tolerance << ")\n";
      std::cout << "    ||A - U*Σ*V^T|| / ||A|| = " << relative_error << "\n";
      return false;
    }

    verified = true;
    return true;
  }

  void cleanup() {
    if (dA)
      hipCheck(hipFree(dA));
    if (dS)
      hipCheck(hipFree(dS));
    if (dU)
      hipCheck(hipFree(dU));
    if (dV)
      hipCheck(hipFree(dV));
    if (dWork)
      hipCheck(hipFree(dWork));
    if (dInfo)
      hipCheck(hipFree(dInfo));
    if (gesvdj_params)
      hipsolverCheck(hipsolverDestroyGesvdjInfo(gesvdj_params));

    dA = nullptr;
    dS = nullptr;
    dU = nullptr;
    dV = nullptr;
    dWork = nullptr;
    dInfo = nullptr;
    gesvdj_params = nullptr;
  }
};

// Helper function declarations
bool parseCommandLineArgs(int argc, char **argv);
void printConfiguration();
void createStreamsAndHandles(std::vector<hipStream_t> &streams,
                             std::vector<hipsolverHandle_t> &handles);
int initializeOperations(std::vector<SVDOperation> &operations,
                         const std::vector<hipStream_t> &streams,
                         const std::vector<hipsolverHandle_t> &handles);
int launchOperations(
    std::vector<SVDOperation> &operations,
    const std::vector<hipStream_t> &streams,
    std::chrono::high_resolution_clock::time_point &start_time);
void copyAndSyncResults(
    std::vector<SVDOperation> &operations,
    const std::vector<hipStream_t> &streams,
    const std::chrono::high_resolution_clock::time_point &start_time);
void extractConvergenceInfo(std::vector<SVDOperation> &operations);
void printResults(int verified, int failed, int zero_sweeps, int init_errors,
                  int launch_errors);
void cleanupResources(std::vector<hipsolverHandle_t> &handles,
                      std::vector<hipStream_t> &streams);

// Helper function implementations
bool parseCommandLineArgs(int argc, char **argv) {
  if (argc > 1) {
    MATRIX_SIZE = std::atoi(argv[1]);
    if (MATRIX_SIZE <= 0) {
      std::cerr << "ERROR: Invalid matrix size. Must be positive integer.\n";
      std::cerr << "Usage: " << argv[0] << " [matrix_size]\n";
      std::cerr << "  matrix_size: Size of square matrices (default: 64)\n";
      return false;
    }
  }
  return true;
}

void printConfiguration() {
  const size_t size_A = static_cast<size_t>(MATRIX_SIZE) * MATRIX_SIZE;
  const size_t size_U = static_cast<size_t>(MATRIX_SIZE) * MATRIX_SIZE;
  const size_t size_V = static_cast<size_t>(MATRIX_SIZE) * MATRIX_SIZE;
  const size_t size_S = MATRIX_SIZE;

  const size_t gpu_mem_per_op =
      (size_A + size_U + size_V + size_S) * sizeof(double) + sizeof(int) +
      size_A * sizeof(double);
  const size_t host_mem_per_op =
      (size_A + size_U + size_V + size_S) * sizeof(double) + sizeof(int);

  const double total_gpu_mem_gb =
      static_cast<double>(gpu_mem_per_op * TOTAL_OPS) /
      (1024.0 * 1024.0 * 1024.0);
  const double total_host_mem_gb =
      static_cast<double>(host_mem_per_op * TOTAL_OPS) /
      (1024.0 * 1024.0 * 1024.0);
  const double total_mem_gb = total_gpu_mem_gb + total_host_mem_gb;

  std::cout
      << "=================================================================\n";
  std::cout << "Multi-Stream Concurrent Jacobi SVD\n";
  std::cout
      << "=================================================================\n";
  std::cout << "Configuration:\n";
  std::cout << "  - Number of streams: " << NUM_STREAMS << "\n";
  std::cout << "  - Operations per stream: " << OPS_PER_STREAM << "\n";
  std::cout << "  - Total operations: " << TOTAL_OPS << "\n";
  std::cout << "  - Matrix size: " << MATRIX_SIZE << "x" << MATRIX_SIZE << "\n";
  std::cout << "  - Test function: hipsolverDgesvdj (Jacobi SVD)\n";
  std::cout << "  - Max sweeps: 50 per operation\n";
  std::cout << "  - Tolerance: 1.0e-7\n";
  std::cout << "\n";
  std::cout << "Memory Usage:\n";
  std::cout << "  - GPU memory: " << std::fixed << std::setprecision(2)
            << total_gpu_mem_gb << " GB\n";
  std::cout << "  - Host memory: " << std::fixed << std::setprecision(2)
            << total_host_mem_gb << " GB\n";
  std::cout << "  - Total memory: " << std::fixed << std::setprecision(2)
            << total_mem_gb << " GB\n";
  std::cout << std::defaultfloat << std::setprecision(6);
  std::cout << "\n";
  std::cout << "==============================================================="
               "==\n\n";
}

void createStreamsAndHandles(std::vector<hipStream_t> &streams,
                             std::vector<hipsolverHandle_t> &handles) {
  std::cout << "[Phase 1/6] Creating streams and handles...\n";

  for (int s = 0; s < NUM_STREAMS; s++) {
    hipCheck(hipStreamCreate(&streams[s]));
    hipsolverCheck(hipsolverCreate(&handles[s]));
    hipsolverCheck(hipsolverSetStream(handles[s], streams[s]));
  }

  std::cout << "  Created " << NUM_STREAMS << " streams and handles\n\n";
}

int initializeOperations(std::vector<SVDOperation> &operations,
                         const std::vector<hipStream_t> &streams,
                         const std::vector<hipsolverHandle_t> &handles) {
  std::cout << "[Phase 2/6] Initializing " << TOTAL_OPS << " operations...\n";

  int init_errors = 0;

  for (int s = 0; s < NUM_STREAMS; s++) {
    for (int o = 0; o < OPS_PER_STREAM; o++) {
      const int idx = s * OPS_PER_STREAM + o;
      operations[idx].init(s, o, streams[s], handles[s]);

      if (operations[idx].status != HIPSOLVER_STATUS_SUCCESS) {
        std::cout << "  ERROR: Init failed for [Stream " << s << ", Op " << o
                  << "]\n";
        init_errors++;
      }
    }
  }

  std::cout << "  Initialized " << (TOTAL_OPS - init_errors) << "/" << TOTAL_OPS
            << " operations\n";
  if (init_errors > 0) {
    std::cout << "  ⚠ " << init_errors << " initialization errors!\n";
  }
  std::cout << "\n";

  return init_errors;
}

int launchOperations(
    std::vector<SVDOperation> &operations,
    const std::vector<hipStream_t> &streams,
    std::chrono::high_resolution_clock::time_point &start_time) {
  std::cout << "[Phase 3/6] Launching operations...\n";
  std::cout << "  Progress: " << std::flush;

  start_time = std::chrono::high_resolution_clock::now();

  constexpr int BATCH_SIZE = 25;
  int progress_step = TOTAL_OPS / 20;
  if (progress_step < 1) {
    progress_step = 1;
  }

  int errors = 0;

  for (int batch_start = 0; batch_start < TOTAL_OPS;
       batch_start += BATCH_SIZE) {
    const int batch_end = std::min(batch_start + BATCH_SIZE, TOTAL_OPS);

    for (int i = batch_start; i < batch_end; i++) {
      if (operations[i].status == HIPSOLVER_STATUS_SUCCESS) {
        operations[i].launch();

        if (operations[i].status != HIPSOLVER_STATUS_SUCCESS) {
          std::cout << "\n  ERROR: Launch failed for [Stream "
                    << operations[i].stream_id << ", Op " << operations[i].op_id
                    << "]\n";
          errors++;
        }
      }

      if ((i + 1) % progress_step == 0) {
        std::cout << "." << std::flush;
      }
    }

    if (batch_end < TOTAL_OPS) {
      for (const auto &s : streams) {
        hipCheck(hipStreamSynchronize(s));
      }
    }
  }

  std::cout << " done!\n\n";
  return errors;
}

void copyAndSyncResults(
    std::vector<SVDOperation> &operations,
    const std::vector<hipStream_t> &streams,
    const std::chrono::high_resolution_clock::time_point &start_time) {
  std::cout << "[Phase 4/6] Copying results from GPU...\n";

  for (auto &op : operations) {
    if (op.status == HIPSOLVER_STATUS_SUCCESS) {
      op.copyResults();
    }
  }

  for (const auto &s : streams) {
    hipCheck(hipStreamSynchronize(s));
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> elapsed = end - start_time;

  std::cout << "  Completed in " << elapsed.count() << " seconds\n";
  std::cout << "  Average: " << (elapsed.count() * 1000.0 / TOTAL_OPS)
            << " ms/op\n\n";
}

void extractConvergenceInfo(std::vector<SVDOperation> &operations) {
  std::cout << "[Phase 5/6] Extracting convergence information...\n";

  for (auto &op : operations) {
    if (op.status == HIPSOLVER_STATUS_SUCCESS) {
      op.getConvergenceInfo();
    }
  }

  std::cout << "  Done\n\n";
}

// Note: verifyOperations removed - verification is done inline in main to avoid
// const issues

void printResults(int verified, int failed, int zero_sweeps, int init_errors,
                  int launch_errors) {
  std::cout << "\n============================================================="
               "====\n";
  std::cout << "RESULTS:\n";
  std::cout
      << "=================================================================\n";
  std::cout << "  Total operations:        " << TOTAL_OPS << "\n";
  std::cout << "  Initialization errors:   " << init_errors << "\n";
  std::cout << "  Launch errors:           " << launch_errors << "\n";
  std::cout << "  Successfully verified:   " << verified << "\n";
  std::cout << "  Failed verification:     " << failed << "\n";
  std::cout << "  Operations with 0 sweeps: " << zero_sweeps << "\n";
  std::cout << "  Success rate:            " << (100.0 * verified / TOTAL_OPS)
            << "%\n";
  std::cout
      << "=================================================================\n";

  if (zero_sweeps > 0) {
    std::cout << "\nCONCURRENCY BUG DETECTED!\n";
    std::cout << zero_sweeps
              << " operations reported 0 sweeps, meaning GESVDJ\n";
    std::cout << "did not execute despite successful initialization.\n";
    std::cout << "This indicates a race condition or synchronization bug\n";
    std::cout << "in multi-stream GESVDJ execution.\n\n";
  }

  if (verified == TOTAL_OPS && launch_errors == 0 && init_errors == 0) {
    std::cout << "\nSUCCESS: All " << TOTAL_OPS
              << " GESVDJ operations completed correctly!\n";
  } else {
    std::cout << "\nSTRESS TEST FAILED\n";
    std::cout << "  Total failures: " << (init_errors + launch_errors + failed)
              << "\n\n";
  }
}

void cleanupResources(std::vector<hipsolverHandle_t> &handles,
                      std::vector<hipStream_t> &streams) {
  for (auto &h : handles) {
    hipsolverCheck(hipsolverDestroy(h));
  }

  for (auto &s : streams) {
    hipCheck(hipStreamDestroy(s));
  }
}

int main(int argc, char **argv) {
  // Parse command line arguments
  if (!parseCommandLineArgs(argc, argv)) {
    return 1;
  }

  // Print configuration
  printConfiguration();

  // Allocate operations
  std::vector<SVDOperation> operations(TOTAL_OPS);

  // Create streams and handles
  std::vector<hipStream_t> streams(NUM_STREAMS);
  std::vector<hipsolverHandle_t> handles(NUM_STREAMS);
  createStreamsAndHandles(streams, handles);

  // Initialize operations
  const int init_errors = initializeOperations(operations, streams, handles);

  // Launch operations
  std::chrono::high_resolution_clock::time_point start_time;
  const int launch_errors = launchOperations(operations, streams, start_time);

  // Copy results and synchronize
  copyAndSyncResults(operations, streams, start_time);

  // Extract convergence information
  extractConvergenceInfo(operations);

  // Verify results
  int verified = 0;
  int failed = 0;
  int zero_sweeps = 0;

  for (auto &op : operations) {
    if (op.status != HIPSOLVER_STATUS_SUCCESS) {
      failed++;
      continue;
    }

    if (op.sweeps == 0) {
      zero_sweeps++;
    }

    if (op.verify()) {
      verified++;
    } else {
      failed++;
    }
  }

  // Print results
  printResults(verified, failed, zero_sweeps, init_errors, launch_errors);

  // Cleanup
  cleanupResources(handles, streams);

  return (verified == TOTAL_OPS && launch_errors == 0 && init_errors == 0) ? 0
                                                                           : 1;
}
