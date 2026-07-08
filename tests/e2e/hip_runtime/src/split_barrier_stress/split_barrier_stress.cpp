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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <random>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <thread>
#include <vector>

namespace cg = cooperative_groups;

constexpr int HEAVY_THREAD_RATIO = 200;
constexpr int MEDIUM_THREAD_RATIO = 10;
constexpr int HEAVY_ITERATIONS = 20000;
constexpr int MEDIUM_ITERATIONS = 100;
constexpr int LIGHT_BATCH_SIZE = 1000;
constexpr int PHASE2_BATCHES = 3;

#define HIP_CHECK(cmd)                                                         \
  {                                                                            \
    hipError_t error = cmd;                                                    \
    if (error != hipSuccess) {                                                 \
      std::cerr << "HIP error: " << hipGetErrorString(error) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define ROCBLAS_CHECK(cmd)                                                     \
  {                                                                            \
    rocblas_status status = cmd;                                               \
    if (status != rocblas_status_success) {                                    \
      std::cerr << "rocBLAS error at " << __FILE__ << ":" << __LINE__          \
                << std::endl;                                                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define ROCSOLVER_CHECK(cmd)                                                   \
  {                                                                            \
    rocblas_status status = cmd;                                               \
    if (status != rocblas_status_success) {                                    \
      std::cerr << "rocSOLVER error at " << __FILE__ << ":" << __LINE__        \
                << std::endl;                                                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

static __device__ void phase1_matrix_init(double *matrix, int N, int gid,
                                          int total_threads) {
  // EXTREME WORKLOAD IMBALANCE creates opportunity for split barrier benefit:
  // - 0.5% threads: 20000 iterations (HEAVY)
  // - 9.5% threads: 100 iterations (MEDIUM)
  // - 90% threads: instant (LIGHT)
  for (int i = gid; i < N * N; i += total_threads) {
    int row = i / N;
    int col = i % N;

    if (gid < total_threads / HEAVY_THREAD_RATIO) {
      double sum = 0.0;
      for (int k = 0; k < HEAVY_ITERATIONS; k++) {
        sum += sin((double)(row + k) * 0.01) * cos((double)(col + k) * 0.01);
      }
      matrix[i] = sum / (double)HEAVY_ITERATIONS;
      if (row == col)
        matrix[i] += N;
    } else if (gid < total_threads / MEDIUM_THREAD_RATIO) {
      double sum = 0.0;
      for (int k = 0; k < MEDIUM_ITERATIONS; k++) {
        sum += sin((double)(row + k) * 0.1) * cos((double)(col + k) * 0.1);
      }
      matrix[i] = sum / (double)MEDIUM_ITERATIONS;
      if (row == col)
        matrix[i] += N;
    } else {
      matrix[i] =
          (row == col) ? (double)N + 1.0 : (double)((row + col) % 3) - 1.0;
    }
  }
}

static __device__ double phase2_independent_computation(int gid) {
  // INDEPENDENT WORK: 3000 iterations of math that doesn't depend on Phase 1
  // With split barrier: light threads do this WHILE heavy threads finish Phase 1
  // With old barrier: light threads waste time waiting idle, then do this sequentially
  double local_computation = 0.0;

  for (int iter = 0; iter < LIGHT_BATCH_SIZE; iter++) {
    local_computation +=
        sin((double)gid * 0.01 + iter) * cos((double)gid * 0.02 + iter);
  }

  for (int iter = 0; iter < LIGHT_BATCH_SIZE; iter++) {
    local_computation +=
        cos((double)gid * 0.03 + iter) * sin((double)gid * 0.04 + iter);
  }

  for (int iter = 0; iter < LIGHT_BATCH_SIZE; iter++) {
    local_computation +=
        sin((double)gid * 0.05 + iter) + cos((double)gid * 0.06 + iter);
  }

  return local_computation;
}

/**
 * Split barrier kernel: barrier_arrive() allows threads to do independent work
 * while waiting for other threads to complete Phase 1.
 */
static __global__ void unbalanced_matrix_init_kernel_split(double *matrix,
                                                           int *flag, int N) {
  cg::grid_group grid = cg::this_grid();

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;

  for (int i = gid; i < N * N; i += total_threads) {
    int row = i / N;
    int col = i % N;

    if (gid < total_threads / HEAVY_THREAD_RATIO) {
      double sum = 0.0;
      for (int k = 0; k < HEAVY_ITERATIONS; k++) {
        sum += sin((double)(row + k) * 0.01) * cos((double)(col + k) * 0.01);
      }
      matrix[i] = sum / (double)HEAVY_ITERATIONS;
      if (row == col)
        matrix[i] += N;
    } else if (gid < total_threads / MEDIUM_THREAD_RATIO) {
      double sum = 0.0;
      for (int k = 0; k < MEDIUM_ITERATIONS; k++) {
        sum += sin((double)(row + k) * 0.1) * cos((double)(col + k) * 0.1);
      }
      matrix[i] = sum / (double)MEDIUM_ITERATIONS;
      if (row == col)
        matrix[i] += N;
    } else {
      matrix[i] =
          (row == col) ? (double)N + 1.0 : (double)((row + col) % 3) - 1.0;
    }
  }

  auto tok = grid.barrier_arrive();

  // Independent computation overlaps with heavy threads finishing Phase 1
  double local_computation = 0.0;

  for (int iter = 0; iter < LIGHT_BATCH_SIZE; iter++) {
    local_computation +=
        sin((double)gid * 0.01 + iter) * cos((double)gid * 0.02 + iter);
  }

  for (int iter = 0; iter < LIGHT_BATCH_SIZE; iter++) {
    local_computation +=
        cos((double)gid * 0.03 + iter) * sin((double)gid * 0.04 + iter);
  }

  for (int iter = 0; iter < LIGHT_BATCH_SIZE; iter++) {
    local_computation +=
        sin((double)gid * 0.05 + iter) + cos((double)gid * 0.06 + iter);
  }

  if (gid == 0) {
    *flag = (local_computation > -100000.0) ? 1 : 0;
  }

  grid.barrier_wait(std::move(tok));
}

/**
 * Traditional grid.sync() kernel: all threads must wait idle at barrier.
 */
static __global__ void unbalanced_matrix_init_kernel_old(double *matrix,
                                                         int *flag, int N) {
  cg::grid_group grid = cg::this_grid();

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;

  for (int i = gid; i < N * N; i += total_threads) {
    int row = i / N;
    int col = i % N;

    if (gid < total_threads / HEAVY_THREAD_RATIO) {
      double sum = 0.0;
      for (int k = 0; k < HEAVY_ITERATIONS; k++) {
        sum += sin((double)(row + k) * 0.01) * cos((double)(col + k) * 0.01);
      }
      matrix[i] = sum / (double)HEAVY_ITERATIONS;
      if (row == col)
        matrix[i] += N;
    } else if (gid < total_threads / MEDIUM_THREAD_RATIO) {
      double sum = 0.0;
      for (int k = 0; k < MEDIUM_ITERATIONS; k++) {
        sum += sin((double)(row + k) * 0.1) * cos((double)(col + k) * 0.1);
      }
      matrix[i] = sum / (double)MEDIUM_ITERATIONS;
      if (row == col)
        matrix[i] += N;
    } else {
      matrix[i] =
          (row == col) ? (double)N + 1.0 : (double)((row + col) % 3) - 1.0;
    }
  }

  grid.sync();

  double local_computation = 0.0;

  for (int iter = 0; iter < LIGHT_BATCH_SIZE; iter++) {
    local_computation +=
        sin((double)gid * 0.01 + iter) * cos((double)gid * 0.02 + iter);
  }

  for (int iter = 0; iter < LIGHT_BATCH_SIZE; iter++) {
    local_computation +=
        cos((double)gid * 0.03 + iter) * sin((double)gid * 0.04 + iter);
  }

  for (int iter = 0; iter < LIGHT_BATCH_SIZE; iter++) {
    local_computation +=
        sin((double)gid * 0.05 + iter) + cos((double)gid * 0.06 + iter);
  }

  if (gid == 0) {
    *flag = (local_computation > -100000.0) ? 1 : 0;
  }
}

static __global__ void nested_matrix_prep_kernel(double *matrix, int iter_num,
                                                 int N) {
  auto block = cg::this_thread_block();
  auto grid = cg::this_grid();

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;

  auto grid_tok = grid.barrier_arrive();

  for (int i = gid; i < N * N; i += total_threads) {
    int row = i / N;
    int col = i % N;

    if (row == col) {
      matrix[i] = (double)N + (double)(iter_num + 1);
    } else {
      matrix[i] = sin((double)(row + col + iter_num) * 0.1);
    }
  }

  grid.barrier_wait(std::move(grid_tok));

  __shared__ double shared_sum;
  if (threadIdx.x == 0) {
    shared_sum = 0.0;
  }

  auto block_tok = block.barrier_arrive();

  if (gid < N * N) {
    atomicAdd(&shared_sum, matrix[gid] * 0.001);
  }

  block.barrier_wait(std::move(block_tok));
}

/**
 * Compares split barrier (barrier_arrive/barrier_wait) vs traditional
 * grid.sync() performance with unbalanced workload.
 * 
 * PERFORMANCE MEASUREMENT METHODOLOGY:
 * 1. Run each kernel multiple times (num_runs, default 100)
 * 2. Skip first run (warmup) to avoid GPU initialization overhead
 * 3. Measure TWO phases separately:
 *    - Kernel time: Phase 1 (unbalanced init) + Phase 2 (independent work)
 *    - LU time: rocSOLVER factorization on the initialized matrix
 * 4. Compute statistics: average, min, max across all runs
 * 5. Compare total time (kernel + LU) between split and old API
 * 6. Calculate percentage difference: (new - old) / old * 100
 */
bool unbalanced_workload_with_rocsolver(int N = 1024, int num_runs = 100) {
  const int threads = 1024;
  const int blocks = 64;

  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    std::cerr << "Device doesn't support cooperative launch!" << std::endl;
    return false;
  }

  double *d_matrix;
  int *d_ipiv, *d_info, *d_flag;

  HIP_CHECK(hipMalloc(&d_matrix, N * N * sizeof(double)));
  HIP_CHECK(hipMalloc(&d_ipiv, N * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_info, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_flag, sizeof(int)));

  rocblas_handle handle;
  ROCBLAS_CHECK(rocblas_create_handle(&handle));

  void *args[] = {&d_matrix, &d_flag, (void *)&N};
  dim3 grid(blocks);
  dim3 block(threads);

  std::cout << "Using barrier_arrive() and barrier_wait() for synchronization"
            << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  // Store timing results from multiple runs for statistical analysis
  std::vector<long long> split_kernel_times, split_lu_times;

  for (int run = 0; run < num_runs; run++) {
    HIP_CHECK(hipMemset(d_flag, 0, sizeof(int)));

    // MEASUREMENT 1: Kernel execution time (Phase 1 + Phase 2 with split barrier)
    // Includes: unbalanced matrix init + independent computation overlap
    auto kernel_start = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipLaunchCooperativeKernel(
        (void *)unbalanced_matrix_init_kernel_split, grid, block, args, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());
    auto kernel_end = std::chrono::high_resolution_clock::now();

    auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(
                           kernel_end - kernel_start)
                           .count();

    int flag = 0;
    HIP_CHECK(hipMemcpy(&flag, d_flag, sizeof(int), hipMemcpyDeviceToHost));
    if (flag != 1) {
      std::cerr << "Error: Split barrier flag verification failed" << std::endl;
      exit(EXIT_FAILURE);
    }

    // MEASUREMENT 2: LU factorization time on the initialized matrix
    auto lu_start = std::chrono::high_resolution_clock::now();
    ROCSOLVER_CHECK(
        rocsolver_dgetrf(handle, N, N, d_matrix, N, d_ipiv, d_info));
    HIP_CHECK(hipDeviceSynchronize());
    auto lu_end = std::chrono::high_resolution_clock::now();

    auto lu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(lu_end - lu_start)
            .count();

    // Skip first run (warmup) to avoid GPU cache/initialization overhead
    if (run > 0) {
      split_kernel_times.push_back(kernel_time);
      split_lu_times.push_back(lu_time);
    }

    int h_info;
    HIP_CHECK(hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost));
    if (h_info != 0) {
      std::cerr << "Error: LU factorization failed" << std::endl;
      exit(EXIT_FAILURE);
    }

    std::cout << "  Run " << (run + 1) << ": kernel=" << kernel_time
              << " μs, LU=" << lu_time << " μs" << std::endl;
  }

  // STATISTICS: Compute average, min, max from all runs (excluding warmup)
  long long split_kernel_avg = 0, split_lu_avg = 0;
  long long split_kernel_min = split_kernel_times[0],
            split_kernel_max = split_kernel_times[0];
  long long split_lu_min = split_lu_times[0], split_lu_max = split_lu_times[0];

  for (auto t : split_kernel_times) {
    split_kernel_avg += t;
    split_kernel_min = std::min(split_kernel_min, t);
    split_kernel_max = std::max(split_kernel_max, t);
  }
  for (auto t : split_lu_times) {
    split_lu_avg += t;
    split_lu_min = std::min(split_lu_min, t);
    split_lu_max = std::max(split_lu_max, t);
  }
  // Average across (num_runs - 1) since first run was excluded
  split_kernel_avg /= (num_runs - 1);
  split_lu_avg /= (num_runs - 1);

  std::cout << "Using grid.sync() for synchronization" << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  std::vector<long long> old_kernel_times, old_lu_times;

  for (int run = 0; run < num_runs; run++) {
    HIP_CHECK(hipMemset(d_flag, 0, sizeof(int)));

    // MEASUREMENT 1: Kernel execution time (Phase 1 + Phase 2 with traditional barrier)
    // Light threads sit IDLE at grid.sync() waiting for heavy threads
    auto kernel_start = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipLaunchCooperativeKernel(
        (void *)unbalanced_matrix_init_kernel_old, grid, block, args, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());
    auto kernel_end = std::chrono::high_resolution_clock::now();

    auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(
                           kernel_end - kernel_start)
                           .count();

    int flag = 0;
    HIP_CHECK(hipMemcpy(&flag, d_flag, sizeof(int), hipMemcpyDeviceToHost));
    if (flag != 1) {
      std::cerr << "Error: Old API flag verification failed" << std::endl;
      exit(EXIT_FAILURE);
    }

    // MEASUREMENT 2: LU factorization time (same operation as split barrier)
    auto lu_start = std::chrono::high_resolution_clock::now();
    ROCSOLVER_CHECK(
        rocsolver_dgetrf(handle, N, N, d_matrix, N, d_ipiv, d_info));
    HIP_CHECK(hipDeviceSynchronize());
    auto lu_end = std::chrono::high_resolution_clock::now();

    auto lu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(lu_end - lu_start)
            .count();

    // Skip first run (warmup)
    if (run > 0) {
      old_kernel_times.push_back(kernel_time);
      old_lu_times.push_back(lu_time);
    }

    int h_info;
    HIP_CHECK(hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost));
    if (h_info != 0) {
      std::cerr << "Error: LU factorization failed" << std::endl;
      exit(EXIT_FAILURE);
    }

    std::cout << "  Run " << (run + 1) << ": kernel=" << kernel_time
              << " μs, LU=" << lu_time << " μs" << std::endl;
  }

  // STATISTICS: Compute average, min, max for old API (excluding warmup)
  long long old_kernel_avg = 0, old_lu_avg = 0;
  long long old_kernel_min = old_kernel_times[0],
            old_kernel_max = old_kernel_times[0];
  long long old_lu_min = old_lu_times[0], old_lu_max = old_lu_times[0];

  for (auto t : old_kernel_times) {
    old_kernel_avg += t;
    old_kernel_min = std::min(old_kernel_min, t);
    old_kernel_max = std::max(old_kernel_max, t);
  }
  for (auto t : old_lu_times) {
    old_lu_avg += t;
    old_lu_min = std::min(old_lu_min, t);
    old_lu_max = std::max(old_lu_max, t);
  }
  old_kernel_avg /= (num_runs - 1);
  old_lu_avg /= (num_runs - 1);

  std::cout << "========================================" << std::endl;
  std::cout << "PERFORMANCE COMPARISON RESULTS" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::endl;

  std::cout << "┌─────────────────────────────────────────────────────────────┐"
            << std::endl;
  std::cout << "│                   KERNEL TIMING COMPARISON                  │"
            << std::endl;
  std::cout << "├─────────────────────────────────────────────────────────────┤"
            << std::endl;
  std::cout << "│ Metric            │  NEW API (Split) │  OLD API (Sync)     │"
            << std::endl;
  std::cout << "├───────────────────┼──────────────────┼───────────────────┤"
            << std::endl;

  char buffer[200];
  snprintf(buffer, sizeof(buffer), "│ Average (μs)      │  %15lld │  %15lld  │",
           split_kernel_avg, old_kernel_avg);
  std::cout << buffer << std::endl;

  snprintf(buffer, sizeof(buffer), "│ Min (μs)          │  %15lld │  %15lld  │",
           split_kernel_min, old_kernel_min);
  std::cout << buffer << std::endl;

  // TOTAL TIME: Combine kernel + LU factorization for end-to-end performance
  long long split_total = split_kernel_avg + split_lu_avg;
  long long old_total = old_kernel_avg + old_lu_avg;
  snprintf(buffer, sizeof(buffer), "│ Total w/LU (μs)   │  %15lld │  %15lld  │",
           split_total, old_total);
  std::cout << buffer << std::endl;

  std::cout << "└───────────────────┴──────────────────┴───────────────────┘"
            << std::endl;
  std::cout << std::endl;

  // PERFORMANCE DIFFERENCE: Calculate percentage improvement/degradation
  // Formula: (new_time - old_time) / old_time * 100
  // Negative value = improvement, Positive value = degradation

  double total_diff =
      ((double)split_total - (double)old_total) / (double)old_total * 100.0;

  std::cout << "PERFORMANCE ANALYSIS:" << std::endl;
  std::cout << "──────────────────────────────────────" << std::endl;

  std::cout << std::endl;
  std::cout << "Total Time (Kernel + LU):" << std::endl;
  bool test_passed = false;

  // TEST CRITERIA: Split barrier should be faster or equal (shows benefit of overlap)
  // Expected: 4-8% improvement because light threads do useful work during barrier wait
  if (split_total <= old_total) {
    std::cout << "  Split barrier API is faster overall by "
              << std::abs(total_diff) << "%" << std::endl;
    std::cout << "   Total time saved: " << (old_total - split_total) << " μs"
              << std::endl;
    test_passed = true;
  } else {
    std::cout << "   Grid.sync() is faster overall by " << std::abs(total_diff)
              << "%" << std::endl;
  }

  std::cout << std::endl;
  std::cout << "========================================" << std::endl;
  if (test_passed) {
    std::cout << "PASSED: Split barrier API demonstrates equivalent or better "
                 "performance"
              << std::endl;
  } else {
    std::cout
        << "FAILED: Split barrier API shows significant performance degradation"
        << std::endl;
  }
  std::cout << "========================================" << std::endl;

  HIP_CHECK(hipFree(d_matrix));
  HIP_CHECK(hipFree(d_ipiv));
  HIP_CHECK(hipFree(d_info));
  HIP_CHECK(hipFree(d_flag));
  ROCBLAS_CHECK(rocblas_destroy_handle(handle));

  return test_passed;
}

/**
 * Tests nested barriers (grid + block level) with multiple rocSOLVER
 * operations.
 */
void multiple_nested_iterations_with_rocsolver(int N = 512,
                                               int iterations = 100) {
  const int threads = 256;
  const int blocks = 16;

  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));

  if (!device_properties.cooperativeLaunch) {
    std::cerr << "Device doesn't support cooperative launch!" << std::endl;
    return;
  }

  double *d_matrix, *d_tau;
  int *d_ipiv, *d_info;

  HIP_CHECK(hipMalloc(&d_matrix, N * N * sizeof(double)));
  HIP_CHECK(hipMalloc(&d_ipiv, N * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_info, sizeof(int)));
  HIP_CHECK(hipMalloc(&d_tau, N * sizeof(double)));

  rocblas_handle handle;
  ROCBLAS_CHECK(rocblas_create_handle(&handle));

  int iter_num = 0;
  void *args[] = {&d_matrix, &iter_num, (void *)&N};
  dim3 grid(blocks);
  dim3 block(threads);

  std::cout << "Testing multiple nested barriers with matrix operations"
            << std::endl;
  std::cout << "Grid size: " << blocks << " blocks × " << threads << " threads"
            << std::endl;
  std::cout << "Matrix size: " << N << "×" << N << std::endl;
  std::cout << "Iterations: " << iterations << std::endl;
  std::cout
      << "Each iteration: nested barriers (grid + block) + matrix operation"
      << std::endl;
  std::cout << "Total barriers: " << (iterations * 2) << " barriers"
            << std::endl;

  auto total_start = std::chrono::high_resolution_clock::now();

  std::vector<long long> lu_times, qr_times, cholesky_times;
  std::vector<long long> kernel_times;

  std::cout << "\nProgress:" << std::endl;
  
  for (int iter = 0; iter < iterations; iter++) {

    iter_num = iter;

    auto kernel_start = std::chrono::high_resolution_clock::now();
    HIP_CHECK(hipLaunchCooperativeKernel((void *)nested_matrix_prep_kernel,
                                         grid, block, args, 0, 0));
    HIP_CHECK(hipDeviceSynchronize());
    auto kernel_end = std::chrono::high_resolution_clock::now();

    auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(
                           kernel_end - kernel_start)
                           .count();
    kernel_times.push_back(kernel_time);

    if (iter % 3 == 0) {
      auto op_start = std::chrono::high_resolution_clock::now();
      ROCSOLVER_CHECK(
          rocsolver_dgetrf(handle, N, N, d_matrix, N, d_ipiv, d_info));
      HIP_CHECK(hipDeviceSynchronize());
      auto op_end = std::chrono::high_resolution_clock::now();

      auto op_time = std::chrono::duration_cast<std::chrono::microseconds>(
                         op_end - op_start)
                         .count();
      lu_times.push_back(op_time);

      int h_info;
      HIP_CHECK(hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost));
      if (h_info != 0) {
        std::cerr << "Error: LU failed at iteration " << iter
                  << " with info = " << h_info << std::endl;
        exit(EXIT_FAILURE);
      }
      
      std::cout << "  Iter " << (iter + 1) << "/" << iterations 
                << " [LU]: kernel=" << kernel_time << " μs, LU=" << op_time << " μs" 
                << std::endl;

    } else if (iter % 3 == 1) {
      auto op_start = std::chrono::high_resolution_clock::now();
      ROCSOLVER_CHECK(rocsolver_dgeqrf(handle, N, N, d_matrix, N, d_tau));
      HIP_CHECK(hipDeviceSynchronize());
      auto op_end = std::chrono::high_resolution_clock::now();

      auto op_time = std::chrono::duration_cast<std::chrono::microseconds>(
                         op_end - op_start)
                         .count();
      qr_times.push_back(op_time);
      
      std::cout << "  Iter " << (iter + 1) << "/" << iterations 
                << " [QR]: kernel=" << kernel_time << " μs, QR=" << op_time << " μs" 
                << std::endl;

    } else {
      // Make matrix symmetric positive definite for Cholesky
      std::vector<double> h_matrix(N * N);
      HIP_CHECK(hipMemcpy(h_matrix.data(), d_matrix, N * N * sizeof(double),
                          hipMemcpyDeviceToHost));

      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          h_matrix[i * N + j] =
              (i == j) ? (double)(N + 10) : h_matrix[i * N + j] * 0.5478;
        }
      }
      HIP_CHECK(hipMemcpy(d_matrix, h_matrix.data(), N * N * sizeof(double),
                          hipMemcpyHostToDevice));

      auto op_start = std::chrono::high_resolution_clock::now();
      ROCSOLVER_CHECK(
          rocsolver_dpotrf(handle, rocblas_fill_lower, N, d_matrix, N, d_info));
      HIP_CHECK(hipDeviceSynchronize());
      auto op_end = std::chrono::high_resolution_clock::now();

      auto op_time = std::chrono::duration_cast<std::chrono::microseconds>(
                         op_end - op_start)
                         .count();
      cholesky_times.push_back(op_time);

      int h_info;
      HIP_CHECK(hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost));
      if (h_info != 0) {
        std::cerr << "Error: Cholesky failed at iteration " << iter
                  << " with info = " << h_info << std::endl;
        exit(EXIT_FAILURE);
      }
      
      std::cout << "  Iter " << (iter + 1) << "/" << iterations 
                << " [Cholesky]: kernel=" << kernel_time << " μs, Cholesky=" << op_time << " μs" 
                << std::endl;
    }
  }

  auto total_end = std::chrono::high_resolution_clock::now();
  auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        total_end - total_start)
                        .count();

  // Calculate and display statistics
  long long kernel_avg = 0, lu_avg = 0, qr_avg = 0, cholesky_avg = 0;
  
  for (auto t : kernel_times) kernel_avg += t;
  for (auto t : lu_times) lu_avg += t;
  for (auto t : qr_times) qr_avg += t;
  for (auto t : cholesky_times) cholesky_avg += t;
  
  kernel_avg /= iterations;
  lu_avg /= lu_times.size();
  qr_avg /= qr_times.size();
  cholesky_avg /= cholesky_times.size();

  std::cout << "\n============================================" << std::endl;
  std::cout << "NESTED BARRIERS TEST SUMMARY" << std::endl;
  std::cout << "============================================" << std::endl;
  std::cout << "Total iterations completed: " << iterations << std::endl;
  std::cout << "Total time: " << total_time << " ms" << std::endl;
  std::cout << "\nAverage Timings:" << std::endl;
  std::cout << "  Kernel (nested barriers): " << kernel_avg << " μs" << std::endl;
  std::cout << "  LU factorization:         " << lu_avg << " μs (" 
            << lu_times.size() << " runs)" << std::endl;
  std::cout << "  QR factorization:         " << qr_avg << " μs (" 
            << qr_times.size() << " runs)" << std::endl;
  std::cout << "  Cholesky factorization:   " << cholesky_avg << " μs (" 
            << cholesky_times.size() << " runs)" << std::endl;
  std::cout << "============================================" << std::endl;

  HIP_CHECK(hipFree(d_matrix));
  HIP_CHECK(hipFree(d_ipiv));
  HIP_CHECK(hipFree(d_info));
  HIP_CHECK(hipFree(d_tau));
  ROCBLAS_CHECK(rocblas_destroy_handle(handle));
}

int main(int argc, char **argv) {
  int matrix_size_test1 = 1024;
  int iterations_test1 = 100;
  int matrix_size_test2 = 512;
  int iterations_test2 = 100;

  if (argc > 1) {
    matrix_size_test1 = std::atoi(argv[1]);
    if (matrix_size_test1 <= 0) {
      std::cerr << "Error: Matrix size must be positive" << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (argc > 2) {
    iterations_test1 = std::atoi(argv[2]);
    if (iterations_test1 <= 0) {
      std::cerr << "Error: Number of iterations must be positive" << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (argc > 3) {
    matrix_size_test2 = std::atoi(argv[3]);
    if (matrix_size_test2 <= 0) {
      std::cerr << "Error: Matrix size for test 2 must be positive"
                << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (argc > 4) {
    iterations_test2 = std::atoi(argv[4]);
    if (iterations_test2 <= 0) {
      std::cerr << "Error: Number of iterations for test 2 must be positive"
                << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (argc > 5) {
    std::cerr << "Usage: " << argv[0]
              << " [matrix_size_test1] [iterations_test1] [matrix_size_test2] "
                 "[iterations_test2]"
              << std::endl;
    std::cerr << "  matrix_size_test1: Matrix size for unbalanced workload "
                 "test (default: 1024)"
              << std::endl;
    std::cerr << "  iterations_test1:  Number of runs for unbalanced workload "
                 "test (default: 100)"
              << std::endl;
    std::cerr << "  matrix_size_test2: Matrix size for nested iterations test "
                 "(default: 512)"
              << std::endl;
    std::cerr << "  iterations_test2:  Number of iterations for nested "
                 "iterations test (default: 100)"
              << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "========================================" << std::endl;
  std::cout << "CONFIGURATION" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Test 1 (Unbalanced Workload):" << std::endl;
  std::cout << "  Matrix size: " << matrix_size_test1 << " x "
            << matrix_size_test1 << std::endl;
  std::cout << "  Iterations:  " << iterations_test1 << std::endl;
  std::cout << "Test 2 (Nested Iterations):" << std::endl;
  std::cout << "  Matrix size: " << matrix_size_test2 << " x "
            << matrix_size_test2 << std::endl;
  std::cout << "  Iterations:  " << iterations_test2 << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::endl;

  bool all_tests_passed = true;

  std::cout << "Unbalanced Workload" << std::endl;
  std::cout << "========================================" << std::endl;
  bool test1_passed =
      unbalanced_workload_with_rocsolver(matrix_size_test1, iterations_test1);
  all_tests_passed &= test1_passed;
  std::cout << std::endl;

  std::cout << "Multiple Nested Iterations" << std::endl;
  std::cout << "========================================" << std::endl;
  multiple_nested_iterations_with_rocsolver(matrix_size_test2,
                                            iterations_test2);
  std::cout << std::endl;

  std::cout << "========================================" << std::endl;
  if (all_tests_passed) {
    std::cout << "ALL SAMPLES COMPLETED SUCCESSFULLY!" << std::endl;
  } else {
    std::cout << "SAMPLES FAILED" << std::endl;
    std::cout << "  Split barrier API did not show expected performance gains"
              << std::endl;
  }
  std::cout << "========================================" << std::endl;

  return all_tests_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
