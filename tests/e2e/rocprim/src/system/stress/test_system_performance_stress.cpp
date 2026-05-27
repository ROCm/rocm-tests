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

// Contains ONLY these System-Stress QA tests:
//
// 1) MaximumMemoryPressure_100Percent
// 2) MassiveConcurrency_128Streams
// 3) MixedWorkload_ConcurrentDifferentOperations
// 4) LinearScaling_DataSize
//
#include "../../include/test_common.hpp"
#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
namespace tu = test_utils;

extern size_t g_linear_max_elems;
namespace {
inline double gib(size_t bytes) { return bytes / (1024.0 * 1024.0 * 1024.0); }
inline double mib(size_t bytes) { return bytes / (1024.0 * 1024.0); }

__global__ void fill_int_kernel(int *p, size_t n, int v) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    p[i] = v;
}

// Deterministic non-trivial pattern for sort input (keeps values non-negative).
__global__ void fill_sort_pattern_kernel(int *p, size_t n, int seed) {
  size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const int MOD = 1'000'003;
    int x = static_cast<int>((n - 1 - i) % MOD);
    p[i] = x + seed;
  }
}

inline void fill_int(int *p, size_t n, int v, hipStream_t s) {
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(fill_int_kernel, dim3(blocks), dim3(threads), 0, s, p, n,
                     v);
}

inline void fill_sort_pattern(int *p, size_t n, int seed, hipStream_t s) {
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  hipLaunchKernelGGL(fill_sort_pattern_kernel, dim3(blocks), dim3(threads), 0,
                     s, p, n, seed);
}

inline void print_mem_info(const char *prefix) {
  size_t free_mem = 0, total_mem = 0;
  HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
  std::cout << prefix << "Total VRAM: " << gib(total_mem) << " GB, "
            << "Free VRAM: " << gib(free_mem) << " GB\n";
}

// For scan input==1: inclusive scan last element should be N.
inline void verify_scan_last_equals_n(const int *d_scan_out, size_t n) {
  int last = 0;
  HIP_CHECK(hipMemcpy(&last, d_scan_out + (n - 1), sizeof(int),
                      hipMemcpyDeviceToHost));
  ASSERT_EQ(last, static_cast<int>(n)) << "Scan last element mismatch";
}
}  // namespace

// ============================================================================
// System - Stress | MaximumMemoryPressure_100Percent
// ============================================================================

/*
 * Test: MaximumMemoryPressure_100Percent
 *
 * Description (use-case perspective):
 *   Simulates a "worst-case batch" / "near-OOM node" where an application
 * consumes essentially all currently free VRAM and still expects rocPRIM to run
 * without crashing/hanging.
 *
 * What it does:
 *   - Reads current free VRAM.
 *   - Targets allocating 100% of *free* VRAM for a single huge input buffer.
 *   - Backs off as needed so rocPRIM temp storage can also be allocated
 * (otherwise the test can't execute).
 *   - Runs rocPRIM reduce across the massive dataset and prints throughput
 * metrics.
 *
 * Input:
 *   - d_data: huge int array (~100% of free VRAM, adjusted to fit temp)
 *
 * Output / Validation:
 *   - d_output: one int output (not correctness-asserted due to potential
 * overflow at huge N)
 *   - Pass criteria: reduce completes successfully under extreme memory
 * pressure.
 *
 * Bugs it might catch:
 *   - failures in temp sizing/allocation under near-OOM conditions
 *   - hidden extra allocations, leaks, or unexpected memory spikes
 *   - crashes/hangs due to allocator fragmentation or stress
 */
TEST(SystemStressTests, MaximumMemoryPressure_100Percent) {
  std::cout << "\n=== System-Stress: Maximum Memory Pressure (100% of free "
               "VRAM target) ===\n";
  print_mem_info("  Before: ");

  // Allocate output first so free_mem reflects reality.
  int *d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_output, sizeof(int)));

  size_t free_mem = 0, total_mem = 0;
  HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));

  // Target: 100% of current free memory for payload (then back off until temp
  // fits AND cushion remains).
  const double target_frac = .99;
  size_t target_bytes =
      static_cast<size_t>(static_cast<long double>(free_mem) * target_frac);
  target_bytes = (target_bytes / sizeof(int)) * sizeof(int);  // align to int

  // Backoff and minimum payload (updated per request)
  const size_t backoff = 1ULL * 1024ULL * 1024ULL * 1024ULL;     // 1GB
  const size_t min_payload = 1ULL * 1024ULL * 1024ULL * 1024ULL;  // 1GB

  // NEW: operational cushion (headroom required to safely run kernels/runtime)
  const size_t operational_cushion = 4ULL * 1024ULL * 1024ULL * 1024ULL;

  int *d_data = nullptr;
  void *d_temp = nullptr;
  size_t temp_bytes = 0;

  hipStream_t stream = hipStreamDefault;

  size_t data_bytes = target_bytes;

  // Try to find the largest payload that:
  //  1) Allocates successfully
  //  2) Allows rocPRIM temp allocation
  //  3) Leaves >= operational_cushion free VRAM after payload+temp
  while (data_bytes >= min_payload) {
    // Allocate payload
    hipError_t st = hipMalloc(&d_data, data_bytes);
    if (st != hipSuccess) {
      std::cout << "st failure allocate payload failed" << std::endl;
      data_bytes = (data_bytes > backoff) ? (data_bytes - backoff) : 0;
      continue;
    }

    const size_t n = data_bytes / sizeof(int);

    // Query temp
    temp_bytes = 0;
    hipError_t stq = rocprim::reduce(nullptr, temp_bytes, d_data, d_output, n,
                                     rocprim::plus<int>(), stream);
    if (stq != hipSuccess) {
      std::cout << "reduce failed" << std::endl;
      HIP_CHECK(hipFree(d_data));
      d_data = nullptr;
      data_bytes = (data_bytes > backoff) ? (data_bytes - backoff) : 0;
      continue;
    }

    // Allocate temp
    d_temp = nullptr;
    if (temp_bytes > 0) {
      hipError_t stt = hipMalloc(&d_temp, temp_bytes);
      if (stt != hipSuccess) {
        HIP_CHECK(hipFree(d_data));
        d_data = nullptr;
        d_temp = nullptr;

        size_t shrink = std::max(backoff, temp_bytes / 2);
        data_bytes = (data_bytes > shrink) ? (data_bytes - shrink) : 0;
        continue;
      }
    }

    // NEW: Enforce operational cushion AFTER payload+temp allocations
    size_t free_after = 0, total_after = 0;
    HIP_CHECK(hipMemGetInfo(&free_after, &total_after));

    if (free_after < operational_cushion) {
      // Not enough headroom to safely run kernels/rocPRIM; back off and retry
      HIP_CHECK(hipFree(d_data));
      d_data = nullptr;

      if (d_temp) {
        HIP_CHECK(hipFree(d_temp));
        d_temp = nullptr;
      }

      data_bytes = (data_bytes > backoff) ? (data_bytes - backoff) : 0;
      continue;
    }

    // Success
    std::cout << "  Target payload: " << gib(target_bytes) << " GB\n";
    std::cout << "  Actual payload: " << gib(data_bytes) << " GB\n";
    std::cout << "  Temp storage:   " << gib(temp_bytes) << " GB\n";
    std::cout << "  Free after:     " << gib(free_after) << " GB\n";
    std::cout << "  Cushion req:    " << gib(operational_cushion) << " GB\n";
    std::cout << "  Elements:       " << (n / 1e6) << "M ints\n";
    break;
  }

  if (!d_data) {
    HIP_CHECK(hipFree(d_output));
    GTEST_SKIP() << "Unable to allocate payload+temp with required operational "
                    "cushion under current VRAM conditions.";
  }

  const size_t n = data_bytes / sizeof(int);

  // IMPORTANT: initialize without hipMemset(byte-pattern). Use your existing
  // fill_int kernel/helper.
  fill_int(d_data, n, 1, stream);
  HIP_CHECK(hipStreamSynchronize(stream));

  // Timed reduce
  auto start = std::chrono::high_resolution_clock::now();
  HIP_CHECK(rocprim::reduce(d_temp, temp_bytes, d_data, d_output, n,
                            rocprim::plus<int>(), stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  auto end = std::chrono::high_resolution_clock::now();

  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
  double gb_processed = (n * sizeof(int)) / (1024.0 * 1024.0 * 1024.0);
  double throughput = (ms > 0) ? (gb_processed / (ms / 1000.0)) : 0.0;

  std::cout << "✓ Reduce completed under extreme memory pressure\n";
  std::cout << "  Processed:  " << gb_processed << " GB in " << ms << " ms\n";
  std::cout << "  Throughput: " << throughput << " GB/s\n";

  HIP_CHECK(hipFree(d_data));
  HIP_CHECK(hipFree(d_output));
  if (d_temp)
    HIP_CHECK(hipFree(d_temp));

  print_mem_info("  After:  ");
}
// ============================================================================
// System - Stress | MassiveConcurrency_128Streams
// ============================================================================

/*
 * Test: MassiveConcurrency_128Streams
 *
 * Description (use-case perspective):
 *   Simulates a high-concurrency scheduler scenario: many independent work
 * queues (streams) executing in parallel, each running a scan (common in
 * compaction/indexing pipelines).
 *
 * What it does:
 *   - Creates 128 non-blocking streams.
 *   - Runs rocPRIM inclusive_scan on each stream concurrently.
 *   - Target total *input* data ~2GB:
 *       size_per_stream = 4M ints = 16MB input per stream
 *       128 * 16MB = ~2048MB input
 *
 * Input:
 *   - d_inputs[i]: 4M ints per stream filled with 1
 *
 * Output / Validation:
 *   - d_outputs[i]: scan output per stream
 *   - Pass criteria: all scans complete; spot-check correctness on a few
 * streams: last element of scan == N (since input is all ones)
 *
 * Bugs it might catch:
 *   - rocPRIM stream-safety regressions under large in-flight concurrency
 *   - temp storage isolation issues under heavy parallelism
 *   - runtime instability with many streams and large buffers
 */
TEST(SystemStressTests, MassiveConcurrency_128Streams) {
  std::cout << "\n=== System-Stress: Massive Concurrency (128 Streams, ~2GB "
               "input total) ===\n";
  print_mem_info("  Before: ");

  const int num_streams = 128;
  const size_t size_per_stream =
      1u << 22;  // 4M ints (16MB input per stream) => ~2GB total input

  // Basic VRAM feasibility check (rough):
  // Need input + output (each ~2GB total) + temp overhead (varies).
  size_t free_mem = 0, total_mem = 0;
  HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
  const size_t input_bytes_total =
      static_cast<size_t>(num_streams) * size_per_stream * sizeof(int);  // ~2GB
  const size_t output_bytes_total = input_bytes_total;  // another ~2GB
  const size_t rough_need = input_bytes_total + output_bytes_total +
                            (input_bytes_total / 2);  // +~1GB temp guess

  if (free_mem < rough_need) {
    std::cout << "  Skip: free VRAM (" << gib(free_mem) << " GB) < rough need ("
              << gib(rough_need) << " GB)\n";
    GTEST_SKIP() << "Insufficient free VRAM for 128-stream 2GB-input stress "
                    "test right now.";
  }

  std::vector<hipStream_t> streams(num_streams);
  std::vector<int *> d_inputs(num_streams, nullptr);
  std::vector<int *> d_outputs(num_streams, nullptr);
  std::vector<void *> d_temps(num_streams, nullptr);
  std::vector<size_t> temp_bytes(num_streams, 0);

  // Setup allocations
  for (int i = 0; i < num_streams; i++) {
    HIP_CHECK(hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking));

    HIP_CHECK(hipMalloc(&d_inputs[i], size_per_stream * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_outputs[i], size_per_stream * sizeof(int)));

    fill_int(d_inputs[i], size_per_stream, 1, streams[i]);

    temp_bytes[i] = 0;
    HIP_CHECK(rocprim::inclusive_scan(nullptr, temp_bytes[i], d_inputs[i],
                                      d_outputs[i], size_per_stream,
                                      rocprim::plus<int>(), streams[i]));
    if (temp_bytes[i] > 0)
      HIP_CHECK(hipMalloc(&d_temps[i], temp_bytes[i]));
  }

  // Launch all scans (timed)
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_streams; i++) {
    HIP_CHECK(rocprim::inclusive_scan(d_temps[i], temp_bytes[i], d_inputs[i],
                                      d_outputs[i], size_per_stream,
                                      rocprim::plus<int>(), streams[i]));
  }

  for (int i = 0; i < num_streams; i++)
    HIP_CHECK(hipStreamSynchronize(streams[i]));

  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();

  double total_input_gb =
      (static_cast<double>(num_streams) * size_per_stream * sizeof(int)) /
      (1024.0 * 1024.0 * 1024.0);
  double throughput = (ms > 0) ? (total_input_gb / (ms / 1000.0)) : 0.0;

  std::cout << "✓ Completed " << num_streams << " scans in " << ms << " ms\n";
  std::cout << "  Total input processed: " << total_input_gb << " GB\n";
  std::cout << "  Aggregate throughput:  " << throughput << " GB/s\n";

  // Spot-check correctness on a few streams
  for (int i = 0; i < std::min(3, num_streams); i++)
    verify_scan_last_equals_n(d_outputs[i], size_per_stream);

  // Cleanup
  for (int i = 0; i < num_streams; i++) {
    HIP_CHECK(hipFree(d_inputs[i]));
    HIP_CHECK(hipFree(d_outputs[i]));
    if (d_temps[i])
      HIP_CHECK(hipFree(d_temps[i]));
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }

  print_mem_info("  After:  ");
}

// ============================================================================
// System - Stress | MixedWorkload_ConcurrentDifferentOperations
// ============================================================================

/*
 * Test: MixedWorkload_ConcurrentDifferentOperations
 *
 * Description (use-case perspective):
 *   Simulates a heterogeneous workload where different components issue
 * different primitives:
 *   - reductions for metrics/aggregation
 *   - sorts for grouping/indexing
 *   - scans for offsets/compaction
 *   All at the same time. This stresses scheduler + rocPRIM stream-safety under
 * mixed kernel types.
 *
 * What it does:
 *   - Launches 96 concurrent operations:
 *       32 reduce + 32 radix_sort_keys + 32 inclusive_scan
 *   - Separate stream + temp storage per operation (required by rocPRIM
 * concurrency model).
 *
 * Input:
 *   - reduce/scan inputs are filled with 1 (cheap correctness checks).
 *   - sort inputs use a deterministic non-trivial pattern.
 *
 * Output / Validation:
 *   - reduce: output should equal N (sum of ones)
 *   - scan: last element should equal N (inclusive scan of ones)
 *   - sort: for one representative sort (i=0), copy output and verify it is
 * sorted
 *
 * Bugs it might catch:
 *   - cross-talk between different rocPRIM algorithms under concurrency
 *   - temp storage isolation regressions
 *   - stream misuse (work submitted on wrong stream)
 */
TEST(SystemStressTests, MixedWorkload_ConcurrentDifferentOperations) {
  std::cout << "\n=== System-Stress: Mixed Workload (32 reduce + 32 sort + 32 "
               "scan) ===\n";
  print_mem_info("  Before: ");

  const int num_ops = 32;
  const size_t size = 1u << 22;  // 4M ints (16MB) per op array

  // Rough VRAM check (very approximate; temp varies):
  size_t free_mem = 0, total_mem = 0;
  HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
  size_t array_bytes = size * sizeof(int);
  size_t rough_need =
      static_cast<size_t>(num_ops) * (4 * array_bytes) +
      (static_cast<size_t>(num_ops) * array_bytes);  // + temp guess
  if (free_mem < rough_need) {
    std::cout << "  Skip: free VRAM (" << gib(free_mem) << " GB) < rough need ("
              << gib(rough_need) << " GB)\n";
    GTEST_SKIP()
        << "Insufficient free VRAM for mixed workload stress test right now.";
  }

  // Reduce resources
  std::vector<hipStream_t> reduce_streams(num_ops);
  std::vector<int *> reduce_inputs(num_ops, nullptr);
  std::vector<int *> reduce_outputs(num_ops, nullptr);
  std::vector<void *> reduce_temps(num_ops, nullptr);
  std::vector<size_t> reduce_temp_bytes(num_ops, 0);

  // Sort resources
  std::vector<hipStream_t> sort_streams(num_ops);
  std::vector<int *> sort_inputs(num_ops, nullptr);
  std::vector<int *> sort_outputs(num_ops, nullptr);
  std::vector<void *> sort_temps(num_ops, nullptr);
  std::vector<size_t> sort_temp_bytes(num_ops, 0);

  // Scan resources
  std::vector<hipStream_t> scan_streams(num_ops);
  std::vector<int *> scan_inputs(num_ops, nullptr);
  std::vector<int *> scan_outputs(num_ops, nullptr);
  std::vector<void *> scan_temps(num_ops, nullptr);
  std::vector<size_t> scan_temp_bytes(num_ops, 0);

  // Setup
  for (int i = 0; i < num_ops; i++) {
    // Reduce
    HIP_CHECK(
        hipStreamCreateWithFlags(&reduce_streams[i], hipStreamNonBlocking));
    HIP_CHECK(hipMalloc(&reduce_inputs[i], size * sizeof(int)));
    HIP_CHECK(hipMalloc(&reduce_outputs[i], sizeof(int)));
    fill_int(reduce_inputs[i], size, 1, reduce_streams[i]);

    HIP_CHECK(rocprim::reduce(nullptr, reduce_temp_bytes[i], reduce_inputs[i],
                              reduce_outputs[i], size, rocprim::plus<int>(),
                              reduce_streams[i]));
    if (reduce_temp_bytes[i] > 0)
      HIP_CHECK(hipMalloc(&reduce_temps[i], reduce_temp_bytes[i]));

    // Sort (keys-only)
    HIP_CHECK(hipStreamCreateWithFlags(&sort_streams[i], hipStreamNonBlocking));
    HIP_CHECK(hipMalloc(&sort_inputs[i], size * sizeof(int)));
    HIP_CHECK(hipMalloc(&sort_outputs[i], size * sizeof(int)));
    fill_sort_pattern(sort_inputs[i], size, /*seed*/ (i + 1) * 17,
                      sort_streams[i]);

    HIP_CHECK(rocprim::radix_sort_keys(nullptr, sort_temp_bytes[i],
                                       sort_inputs[i], sort_outputs[i], size, 0,
                                       sizeof(int) * 8, sort_streams[i]));
    if (sort_temp_bytes[i] > 0)
      HIP_CHECK(hipMalloc(&sort_temps[i], sort_temp_bytes[i]));

    // Scan
    HIP_CHECK(hipStreamCreateWithFlags(&scan_streams[i], hipStreamNonBlocking));
    HIP_CHECK(hipMalloc(&scan_inputs[i], size * sizeof(int)));
    HIP_CHECK(hipMalloc(&scan_outputs[i], size * sizeof(int)));
    fill_int(scan_inputs[i], size, 1, scan_streams[i]);

    HIP_CHECK(rocprim::inclusive_scan(nullptr, scan_temp_bytes[i],
                                      scan_inputs[i], scan_outputs[i], size,
                                      rocprim::plus<int>(), scan_streams[i]));
    if (scan_temp_bytes[i] > 0)
      HIP_CHECK(hipMalloc(&scan_temps[i], scan_temp_bytes[i]));
  }

  // Launch all ops concurrently
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < num_ops; i++) {
    HIP_CHECK(rocprim::reduce(reduce_temps[i], reduce_temp_bytes[i],
                              reduce_inputs[i], reduce_outputs[i], size,
                              rocprim::plus<int>(), reduce_streams[i]));

    HIP_CHECK(rocprim::radix_sort_keys(sort_temps[i], sort_temp_bytes[i],
                                       sort_inputs[i], sort_outputs[i], size, 0,
                                       sizeof(int) * 8, sort_streams[i]));

    HIP_CHECK(rocprim::inclusive_scan(scan_temps[i], scan_temp_bytes[i],
                                      scan_inputs[i], scan_outputs[i], size,
                                      rocprim::plus<int>(), scan_streams[i]));
  }

  // Synchronize all
  for (int i = 0; i < num_ops; i++) {
    HIP_CHECK(hipStreamSynchronize(reduce_streams[i]));
    HIP_CHECK(hipStreamSynchronize(sort_streams[i]));
    HIP_CHECK(hipStreamSynchronize(scan_streams[i]));
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();

  std::cout << "✓ Completed mixed workload (96 ops) in " << ms << " ms\n";

  // Validate reduce/scan for a few ops
  for (int i = 0; i < std::min(3, num_ops); i++) {
    int red = 0;
    HIP_CHECK(
        hipMemcpy(&red, reduce_outputs[i], sizeof(int), hipMemcpyDeviceToHost));
    ASSERT_EQ(red, static_cast<int>(size)) << "Reduce mismatch for op " << i;

    verify_scan_last_equals_n(scan_outputs[i], size);
  }

  // Validate sort for one representative op (i=0) to catch obvious corruption
  {
    std::vector<int> host_sorted(size);
    HIP_CHECK(hipMemcpy(host_sorted.data(), sort_outputs[0], size * sizeof(int),
                        hipMemcpyDeviceToHost));
    ASSERT_TRUE(std::is_sorted(host_sorted.begin(), host_sorted.end()))
        << "Representative sort output not sorted";
  }

  // Cleanup
  for (int i = 0; i < num_ops; i++) {
    HIP_CHECK(hipFree(reduce_inputs[i]));
    HIP_CHECK(hipFree(reduce_outputs[i]));
    if (reduce_temps[i])
      HIP_CHECK(hipFree(reduce_temps[i]));
    HIP_CHECK(hipStreamDestroy(reduce_streams[i]));

    HIP_CHECK(hipFree(sort_inputs[i]));
    HIP_CHECK(hipFree(sort_outputs[i]));
    if (sort_temps[i])
      HIP_CHECK(hipFree(sort_temps[i]));
    HIP_CHECK(hipStreamDestroy(sort_streams[i]));

    HIP_CHECK(hipFree(scan_inputs[i]));
    HIP_CHECK(hipFree(scan_outputs[i]));
    if (scan_temps[i])
      HIP_CHECK(hipFree(scan_temps[i]));
    HIP_CHECK(hipStreamDestroy(scan_streams[i]));
  }

  print_mem_info("  After:  ");
}

// ============================================================================
// System - Stress | LinearScaling_DataSize
// ============================================================================

/*
 * Test: LinearScaling_DataSize
 *
 * Description (use-case perspective):
 *   Validates performance predictability and algorithmic complexity: reduce
 * should behave like O(n).
 *
 * What it does:
 *   - Measures rocPRIM reduce time across sizes: 1K -> 10M elements.
 *   - Uses deterministic input (all ones) so correctness is easy to validate.
 *   - Computes time-per-element and compares across large sizes to verify
 * approximately O(n).
 *
 * Input:
 *   - d_input: N ints filled with 1
 *
 * Output / Validation:
 *   - d_output should equal N (sum of ones)
 *   - Scaling check (O(n)):
 *       baseline is time/elem at 1M elements (more stable than tiny sizes).
 *       For sizes >= 1M, require time/elem <= 1.5x baseline (loose, avoids
 * flakiness).
 *
 * Bugs it might catch:
 *   - complexity regressions (unexpected superlinear growth)
 *   - temp sizing anomalies at larger sizes
 *   - correctness failures at specific sizes
 */

TEST(SystemStressTests, LinearScaling_DataSize) {
  std::cout << "\n=== System-Stress: Linear Scaling (1K -> user max via "
               "--linear_size) ===\n";

  using T = int;

  // Cap endpoint by free VRAM so it adapts to ASIC + current load.
  size_t free_mem = 0, total_mem = 0;
  HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));

  // Conservative VRAM policy: use up to 70% of free VRAM for size selection.
  // Budget assumes input + temp roughly ~1.5x input (temp varies by
  // implementation).
  const double vram_fraction = 0.70;
  const size_t vram_cap =
      static_cast<size_t>((static_cast<long double>(free_mem) * vram_fraction) /
                          (1.5L * sizeof(T)));

  // Final max is min(user requested, VRAM cap). This keeps test portable across
  // ASICs.
  size_t max_elems = std::min(g_linear_max_elems, vram_cap);

  if (max_elems < (1ULL << 10))
    GTEST_SKIP() << "Not enough free VRAM to run even 1K-element scaling test.";

  std::cout << "  Free VRAM: " << gib(free_mem) << " GB\n";
  std::cout << "  User cap (--linear_size): " << g_linear_max_elems
            << " elems\n";
  std::cout << "  VRAM cap (70% free):      " << vram_cap << " elems\n";
  std::cout << "  Final max used:           " << max_elems << " elems\n";

  // Build sizes: baseline points + 10M checkpoint (if fits) + endpoint
  // max_elems.
  std::vector<size_t> sizes = {
      1ULL << 10,  // 1K
      1ULL << 12,  // 4K
      1ULL << 14,  // 16K
      1ULL << 16,  // 64K
      1ULL << 18,  // 256K
      1ULL << 20,  // 1M (baseline for scaling)
      1ULL << 22,  // 4M
  };

  if (10'000'000ULL <= max_elems)
    sizes.push_back(10'000'000ULL);

  sizes.push_back(max_elems);

  // Remove points > max_elems, dedup, sort
  sizes.erase(std::remove_if(sizes.begin(), sizes.end(),
                             [&](size_t n) { return n > max_elems; }),
              sizes.end());
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());

  const int timed_iters = 10;

  std::cout << std::fixed << std::setprecision(2);
  std::cout
      << "\nElems\t\tAvgTime(ms)\tGB/s\t\tTime/Elem(ns)\tScaling(vs 1M)\n";
  std::cout << "---------------------------------------------------------------"
               "-----------\n";

  double baseline_time_per_elem_ns = -1.0;
  bool baseline_set = false;

  for (size_t n : sizes) {
    int *d_input = nullptr;
    int *d_output = nullptr;

    // If allocation fails, stop (larger sizes likely also fail).
    if (hipMalloc(&d_input, n * sizeof(T)) != hipSuccess) {
      std::cout << "Stop: OOM allocating input at n=" << n << "\n";
      break;
    }
    HIP_CHECK(hipMalloc(&d_output, sizeof(T)));

    // Deterministic input (all ones)
    fill_int(d_input, n, 1, hipStreamDefault);
    HIP_CHECK(hipStreamSynchronize(hipStreamDefault));

    void *d_temp = nullptr;
    size_t temp_bytes = 0;

    hipError_t q = rocprim::reduce(nullptr, temp_bytes, d_input, d_output, n,
                                   rocprim::plus<T>(), hipStreamDefault);
    if (q != hipSuccess) {
      HIP_CHECK(hipFree(d_input));
      HIP_CHECK(hipFree(d_output));
      std::cout << "Stop: temp query failed at n=" << n << "\n";
      break;
    }

    if (temp_bytes > 0) {
      if (hipMalloc(&d_temp, temp_bytes) != hipSuccess) {
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        std::cout << "Stop: OOM allocating temp at n=" << n << "\n";
        break;
      }
    }

    // Warmup
    HIP_CHECK(rocprim::reduce(d_temp, temp_bytes, d_input, d_output, n,
                              rocprim::plus<T>(), hipStreamDefault));
    HIP_CHECK(hipStreamSynchronize(hipStreamDefault));

    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < timed_iters; it++) {
      HIP_CHECK(rocprim::reduce(d_temp, temp_bytes, d_input, d_output, n,
                                rocprim::plus<T>(), hipStreamDefault));
    }
    HIP_CHECK(hipStreamSynchronize(hipStreamDefault));
    auto end = std::chrono::high_resolution_clock::now();

    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();
    double avg_ms = (us / 1000.0) / timed_iters;

    // Correctness sanity: sum of ones == n
    int out = 0;
    HIP_CHECK(hipMemcpy(&out, d_output, sizeof(T), hipMemcpyDeviceToHost));
    ASSERT_EQ(out, static_cast<int>(n))
        << "Reduce correctness failed at n=" << n;

    double gb = (n * sizeof(T)) / (1024.0 * 1024.0 * 1024.0);
    double gbps = (avg_ms > 0.0) ? (gb / (avg_ms / 1000.0)) : 0.0;

    double time_per_elem_ns = (avg_ms * 1e6) / n;  // ms -> ns per elem
    double scaling = 0.0;

    // Baseline at 1M if present
    if (n == (1ULL << 20)) {
      baseline_time_per_elem_ns = time_per_elem_ns;
      baseline_set = true;
    }

    // O(n) scaling check only for sizes >= 1M (avoid tiny-size overhead noise)
    if (baseline_set && n >= (1ULL << 20)) {
      scaling = time_per_elem_ns / baseline_time_per_elem_ns;
      ASSERT_LE(scaling, 1.50)
          << "Scaling factor exceeds O(n) expectation at n=" << n;
    }

    std::cout << n << "\t\t" << avg_ms << "\t\t" << gbps << "\t\t"
              << time_per_elem_ns << "\t\t";

    if (baseline_set && n >= (1ULL << 20))
      std::cout << scaling;
    else
      std::cout << "N/A";

    std::cout << "\n";

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    if (d_temp)
      HIP_CHECK(hipFree(d_temp));
  }

  std::cout << "\n✓ Linear scaling test completed\n";
}
