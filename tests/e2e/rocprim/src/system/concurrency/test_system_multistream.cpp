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

#include "../../include/test_common.hpp"
#include<cstdint>
namespace tu = test_utils;

// ============================================================================
// MULTI-STREAM TESTS
// ============================================================================

// Test concurrent operations on different streams
// ----------------------------------------------------------------------------
// Test: ConcurrentReduceOnMultipleStreams
//
// Purpose (rocPRIM concurrency focus):
//   Stress rocPRIM reduce under concurrent multi-stream execution to validate
//   stream-safety, isolation (no cross-stream corruption), and correctness when
//   multiple reduce operations are in-flight at the same time.
//
// Inputs:
//   - num_streams = 8 independent HIP non-blocking streams
//   - For each stream i:
//       * Host input vector inputs[i] of size N (random ints in [1,10])
//       * Device buffers: d_inputs[i], d_outputs[i]
//       * Unique rocPRIM temp storage buffer: d_temps[i] (size queried per
//       stream)
//   - Iterations: iters (repeat reduce multiple times to increase race
//   exposure)
//
// Output / Validation:
//   - For each stream i: copy GPU scalar sum back from d_outputs[i]
//   - CPU reference sum via std::accumulate(inputs[i])
//   - Assert exact equality (int sum)
//
// Concurrency / Integration Bugs this can catch:
//   - rocPRIM using incorrect stream internally (work launched on wrong stream)
//   - Hidden shared/global state causing cross-stream interference
//   - Temp-storage misuse regressions (accidental internal reuse across calls)
//   - Cross-stream memory clobbering (writing into another stream’s
//   output/temp)
// ----------------------------------------------------------------------------

TEST(SystemMultiStreamTests, ConcurrentReduceOnMultipleStreams) {
  using T = int;
  const size_t size = 1'000'000;
  const int num_streams = 8;
  const int iters = 50;  // stress to catch concurrency issues

  std::cout << "Testing " << num_streams
            << " concurrent rocPRIM reduce ops, iters=" << iters << "...\n";

  std::vector<hipStream_t> streams(num_streams);
  for (int i = 0; i < num_streams; i++)
    HIP_CHECK(hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking));

  std::vector<std::vector<T>> inputs(num_streams);
  std::vector<T *> d_inputs(num_streams, nullptr);
  std::vector<T *> d_outputs(num_streams, nullptr);
  std::vector<void *> d_temps(num_streams, nullptr);
  std::vector<size_t> temp_bytes(num_streams, 0);

  // Setup: allocate + copy + query temp + allocate temp (NO rocPRIM launches
  // yet except size query)
  for (int i = 0; i < num_streams; i++) {
    inputs[i] = tu::generate_random_data<T>(size, 1, 10);

    HIP_CHECK(hipMalloc(&d_inputs[i], size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_outputs[i], sizeof(T)));

    HIP_CHECK(hipMemcpyAsync(d_inputs[i], inputs[i].data(), size * sizeof(T),
                             hipMemcpyHostToDevice, streams[i]));

    HIP_CHECK(rocprim::reduce(nullptr, temp_bytes[i], d_inputs[i], d_outputs[i],
                              size, rocprim::plus<T>(), streams[i]));

    if (temp_bytes[i] > 0)
      HIP_CHECK(hipMalloc(&d_temps[i], temp_bytes[i]));
  }

  // Concurrency window: enqueue lots of reduces on all streams
  for (int iter = 0; iter < iters; ++iter) {
    for (int i = 0; i < num_streams; i++) {
      HIP_CHECK(rocprim::reduce(d_temps[i], temp_bytes[i], d_inputs[i],
                                d_outputs[i], size, rocprim::plus<T>(),
                                streams[i]));
    }
  }

  for (int i = 0; i < num_streams; i++)
    HIP_CHECK(hipStreamSynchronize(streams[i]));

  // Verify
  for (int i = 0; i < num_streams; i++) {
    T gpu_result{};
    HIP_CHECK(
        hipMemcpy(&gpu_result, d_outputs[i], sizeof(T), hipMemcpyDeviceToHost));

    T cpu_result = std::accumulate(inputs[i].begin(), inputs[i].end(), T(0));
    ASSERT_EQ(cpu_result, gpu_result) << "Stream " << i << " mismatch";
  }

  // Cleanup
  for (int i = 0; i < num_streams; i++) {
    HIP_CHECK(hipFree(d_inputs[i]));
    HIP_CHECK(hipFree(d_outputs[i]));
    if (d_temps[i])
      HIP_CHECK(hipFree(d_temps[i]));
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
}

// Test concurrent sorts
// //
// ----------------------------------------------------------------------------
// Test: ConcurrentSortsOnMultipleStreams
//
// Purpose (rocPRIM concurrency focus):
//   Stress rocPRIM radix_sort_keys concurrently on multiple streams to validate
//   stream-safety and correctness under concurrent sort workloads.
//
// Inputs:
//   - num_streams = 8 independent HIP non-blocking streams
//   - For each stream i:
//       * Host input vector inputs[i] of size N with values in [0,10000]
//       * Device buffers:
//           - d_orig[i]  : immutable copy of the original unsorted input
//           - d_in[i]    : per-iteration input restored from d_orig[i]
//           - d_out[i]   : sorted output
//       * Unique rocPRIM temp storage d_temp[i] (queried + allocated once)
//   - Iterations: iters
//       * Each iteration restores d_in from d_orig and enqueues radix sort
//
// Output / Validation:
//   - After synchronize, copy d_out to host
//   - Validate:
//       (1) Monotonic non-decreasing order (sortedness)
//       (2) Multiset/permutation preservation via histogram (0..10000)
//           => ensures output contains exactly the same elements as input
//
// Concurrency / Integration Bugs this can catch:
//   - Cross-stream scratch/temp contamination leading to missing/duplicated
//   keys
//   - Incorrect stream usage leading to ordering violations or stale results
//   - Non-deterministic corruption that still “looks sorted”
//     (histogram check catches this)
// ----------------------------------------------------------------------------
TEST(SystemMultiStreamTests, ConcurrentSortsOnMultipleStreams) {
  using T = int;
  const size_t size = 1'000'000;
  const int num_streams = 8;
  const int iters = 20;  // sorts are heavy; fewer iterations

  std::cout << "Testing " << num_streams
            << " concurrent rocPRIM radix_sort_keys ops, iters=" << iters
            << "...\n";

  std::vector<hipStream_t> streams(num_streams);
  for (int i = 0; i < num_streams; i++)
    HIP_CHECK(hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking));

  std::vector<std::vector<T>> inputs(num_streams);
  std::vector<T *> d_orig(num_streams, nullptr);
  std::vector<T *> d_in(num_streams, nullptr);
  std::vector<T *> d_out(num_streams, nullptr);
  std::vector<void *> d_temp(num_streams, nullptr);
  std::vector<size_t> temp_bytes(num_streams, 0);
  auto make_hist_0_10000 = [](const std::vector<T> &v) -> std::vector<int> {
    std::vector<int> h(10001, 0);
    for (size_t i = 0; i < v.size(); ++i) {
      auto x = v[i];
      if (x < 0 || x > 10000) {
        ADD_FAILURE() << "Value out of range [0..10000]: " << x << " at index "
                      << i;
        continue;  // avoid out-of-bounds
      }
      h[static_cast<int>(x)]++;
    }
    return h;
  };

  // Setup: allocate + upload + query temp + allocate temp
  for (int i = 0; i < num_streams; i++) {
    inputs[i] = tu::generate_random_data<T>(size, 0, 10000);

    HIP_CHECK(hipMalloc(&d_orig[i], size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_in[i], size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_out[i], size * sizeof(T)));

    // Upload original data once
    HIP_CHECK(hipMemcpyAsync(d_orig[i], inputs[i].data(), size * sizeof(T),
                             hipMemcpyHostToDevice, streams[i]));

    // Query temp storage
    HIP_CHECK(rocprim::radix_sort_keys(nullptr, temp_bytes[i], d_in[i],
                                       d_out[i], size, 0, sizeof(T) * 8,
                                       streams[i]));
    if (temp_bytes[i] > 0)
      HIP_CHECK(hipMalloc(&d_temp[i], temp_bytes[i]));
  }

  // Concurrency window: restore unsorted input then sort, repeated
  for (int iter = 0; iter < iters; ++iter) {
    for (int i = 0; i < num_streams; i++) {
      // restore input (device->device copy enqueued on same stream)
      HIP_CHECK(hipMemcpyAsync(d_in[i], d_orig[i], size * sizeof(T),
                               hipMemcpyDeviceToDevice, streams[i]));

      HIP_CHECK(rocprim::radix_sort_keys(d_temp[i], temp_bytes[i], d_in[i],
                                         d_out[i], size, 0, sizeof(T) * 8,
                                         streams[i]));
    }
  }

  // Verify
  for (int i = 0; i < num_streams; i++) {
    HIP_CHECK(hipStreamSynchronize(streams[i]));

    std::vector<T> gpu_out(size);
    HIP_CHECK(hipMemcpy(gpu_out.data(), d_out[i], size * sizeof(T),
                        hipMemcpyDeviceToHost));

    // 1) sortedness
    for (size_t j = 1; j < size; j++)
      ASSERT_LE(gpu_out[j - 1], gpu_out[j])
          << "Stream " << i << " not sorted at " << j;

    // 2) multiset/permutation match via histogram
    auto cpu_hist = make_hist_0_10000(inputs[i]);
    auto gpu_hist = make_hist_0_10000(gpu_out);
    ASSERT_EQ(cpu_hist, gpu_hist) << "Stream " << i << " permutation mismatch";
  }

  // Cleanup
  for (int i = 0; i < num_streams; i++) {
    HIP_CHECK(hipFree(d_orig[i]));
    HIP_CHECK(hipFree(d_in[i]));
    HIP_CHECK(hipFree(d_out[i]));
    if (d_temp[i])
      HIP_CHECK(hipFree(d_temp[i]));
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
}

// ----------------------------------------------------------------------------
// Test: PipelineMultipleStreams (Transform -> Sort -> Reduce)
//
// Purpose (rocPRIM concurrency focus):
//   Validate that a realistic multi-stage workflow (transform then rocPRIM sort
//   then rocPRIM reduce) can run concurrently across multiple streams without
//   cross-stream interference and with correct stream-local ordering.
//
// Inputs:
//   - num_streams = 8 independent HIP non-blocking streams
//   - For each stream sid:
//       * Host input inputs[sid] (random ints in a per-stream range)
//       * Device buffers:
//           - d_inputs[sid]      : raw input
//           - d_transformed[sid] : output of transform (x*2)
//           - d_sorted[sid]      : sorted transformed values
//           - d_reduce_out[sid]  : scalar sum of sorted transformed values
//       * Unique rocPRIM temp storage for sort and reduce allocated in setup
//   - Iterations: iters
//       * In each iter: enqueue transform -> radix_sort_keys -> reduce on same
//       stream
//       * No allocations inside the concurrency window (avoids allocator
//       serialization)
//
// Output / Validation:
//   - Copy d_sorted and d_reduce_out back to host per stream
//   - Validate:
//       (1) Sortedness of d_sorted
//       (2) Permutation of transformed values (histogram) matches CPU
//       transformed input (3) Reduce sum equals CPU sum of transformed input
//       (exact for int)
//
// Concurrency / Integration Bugs this can catch:
//   - rocPRIM launching on wrong stream (breaks stage ordering / wrong results)
//   - Inter-stream cross-talk (one stream’s stage overwriting another’s
//   buffers)
//   - Hidden global-state issues when multiple primitives overlap (sort+reduce)
//   - Pipeline hazards where later stage consumes incomplete earlier-stage data
//     (would show up as permutation/sum mismatch)
// ----------------------------------------------------------------------------

TEST(SystemMultiStreamTests, PipelineMultipleStreams) {
  using T = int;
  const size_t size = 1'000'000;
  const int num_streams = 8;
  const int iters = 10;  // pipeline is heavy

  std::cout << "Testing Transform->Sort->Reduce pipelines on " << num_streams
            << " streams concurrently, iters=" << iters << "...\n";

  std::vector<hipStream_t> streams(num_streams);
  for (int i = 0; i < num_streams; i++)
    HIP_CHECK(hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking));

  std::vector<std::vector<T>> inputs(num_streams);
  std::vector<T *> d_inputs(num_streams, nullptr);
  std::vector<T *> d_transformed(num_streams, nullptr);
  std::vector<T *> d_sorted(num_streams, nullptr);
  std::vector<T *> d_reduce_out(num_streams, nullptr);

  std::vector<void *> d_temp_sort(num_streams, nullptr);
  std::vector<size_t> temp_sort_bytes(num_streams, 0);
  std::vector<void *> d_temp_reduce(num_streams, nullptr);
  std::vector<size_t> temp_reduce_bytes(num_streams, 0);

  // Setup: allocate + upload + query temps + allocate temps
  for (int sid = 0; sid < num_streams; sid++) {
    inputs[sid] = tu::generate_random_data<T>(size, sid*100, (sid+1)*100);

    HIP_CHECK(hipMalloc(&d_inputs[sid], size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_transformed[sid], size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_sorted[sid], size * sizeof(T)));
    HIP_CHECK(hipMalloc(&d_reduce_out[sid], sizeof(T)));

    HIP_CHECK(hipMemcpyAsync(d_inputs[sid], inputs[sid].data(),
                             size * sizeof(T), hipMemcpyHostToDevice,
                             streams[sid]));

    // sort temp
    HIP_CHECK(rocprim::radix_sort_keys(nullptr, temp_sort_bytes[sid],
                                       d_transformed[sid], d_sorted[sid], size,
                                       0, sizeof(T) * 8, streams[sid]));
    if (temp_sort_bytes[sid] > 0)
      HIP_CHECK(hipMalloc(&d_temp_sort[sid], temp_sort_bytes[sid]));

    // reduce temp
    HIP_CHECK(rocprim::reduce(nullptr, temp_reduce_bytes[sid], d_sorted[sid],
                              d_reduce_out[sid], size, rocprim::plus<T>(),
                              streams[sid]));
    if (temp_reduce_bytes[sid] > 0)
      HIP_CHECK(hipMalloc(&d_temp_reduce[sid], temp_reduce_bytes[sid]));
  }

  // Concurrency window: launch pipelines (no allocations here)
  for (int iter = 0; iter < iters; ++iter) {
    for (int sid = 0; sid < num_streams; sid++) {
      HIP_CHECK(rocprim::transform(
          d_inputs[sid], d_transformed[sid], size,
          [] __device__(T x) { return x * 2; }, streams[sid]));

      HIP_CHECK(rocprim::radix_sort_keys(d_temp_sort[sid], temp_sort_bytes[sid],
                                         d_transformed[sid], d_sorted[sid],
                                         size, 0, sizeof(T) * 8, streams[sid]));

      HIP_CHECK(rocprim::reduce(d_temp_reduce[sid], temp_reduce_bytes[sid],
                                d_sorted[sid], d_reduce_out[sid], size,
                                rocprim::plus<T>(), streams[sid]));
    }
  }

  for (int sid = 0; sid < num_streams; sid++)
    HIP_CHECK(hipStreamSynchronize(streams[sid]));

  // Verify (sortedness + histogram + reduce sum)
  auto make_hist_0_1600 = [](const std::vector<T> &v) -> std::vector<int> {
    std::vector<int> h(1601, 0);
    for (size_t i = 0; i < v.size(); ++i) {
      int64_t x = static_cast<int64_t>(v[i]);
      if (x < 0 || x > 1600) {
        ADD_FAILURE() << "Value out of range [0..1600]: " << x << " at index "
                      << i;
        continue;  // avoid out-of-bounds
      }
      h[static_cast<int>(x)]++;
    }
    return h;
  };

  for (int sid = 0; sid < num_streams; sid++) {
    std::vector<T> sorted(size);
    T reduce_result{};
    HIP_CHECK(hipMemcpy(sorted.data(), d_sorted[sid], size * sizeof(T),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&reduce_result, d_reduce_out[sid], sizeof(T),
                        hipMemcpyDeviceToHost));

    for (size_t j = 1; j < size; j++)
      ASSERT_LE(sorted[j - 1], sorted[j])
          << "Pipeline " << sid << " not sorted at " << j;

    // histogram check on transformed values
    std::vector<T> transformed_cpu;
    transformed_cpu.reserve(size);
    for (const auto &v : inputs[sid])
      transformed_cpu.push_back(v * 2);

    auto cpu_hist = make_hist_0_1600(transformed_cpu);
    auto gpu_hist = make_hist_0_1600(sorted);
    ASSERT_EQ(cpu_hist, gpu_hist)
        << "Pipeline " << sid << " permutation mismatch";

    // reduce check
    T expected_sum =
        std::accumulate(transformed_cpu.begin(), transformed_cpu.end(), T(0));
    ASSERT_EQ(reduce_result, expected_sum)
        << "Pipeline " << sid << " reduce mismatch";
  }

  // Cleanup
  for (int sid = 0; sid < num_streams; sid++) {
    HIP_CHECK(hipFree(d_inputs[sid]));
    HIP_CHECK(hipFree(d_transformed[sid]));
    HIP_CHECK(hipFree(d_sorted[sid]));
    HIP_CHECK(hipFree(d_reduce_out[sid]));
    if (d_temp_sort[sid])
      HIP_CHECK(hipFree(d_temp_sort[sid]));
    if (d_temp_reduce[sid])
      HIP_CHECK(hipFree(d_temp_reduce[sid]));
    HIP_CHECK(hipStreamDestroy(streams[sid]));
  }
}

// ----------------------------------------------------------------------------
// Test: StreamDependenciesWithEvents (Sort on stream1 + sort on stream2 ->
// merge on stream3)
//
// Purpose (rocPRIM concurrency focus):
//   Validate rocPRIM correctness when operations execute on different streams
//   with explicit event-based dependencies, ensuring that rocPRIM honors the
//   stream argument and that dependent rocPRIM ops can be safely composed in a
//   multi-stream DAG.
//
// Inputs:
//   - Three HIP non-blocking streams: stream1, stream2, stream3
//   - Two independent inputs (input1, input2) copied to GPU on stream1/stream2
//   - rocPRIM radix_sort_keys on stream1 and stream2 (each with its own temp
//   storage)
//   - hipEventRecord after each sort
//   - stream3 waits on both events via hipStreamWaitEvent
//   - rocPRIM merge on stream3 (with its own temp storage) merges the two
//   sorted ranges
//
// Output / Validation:
//   - Copy merged array (2*size) back to host
//   - Validate:
//       (1) Sortedness of merged output
//       (2) Multiset correctness: histogram(merged) ==
//       histogram(input1)+histogram(input2)
//
// Concurrency / Integration Bugs this can catch:
//   - rocPRIM not respecting stream ordering or using the wrong stream
//   internally
//   - Incorrect interaction with event-based dependencies (merge starts too
//   early)
//   - Cross-stream scratch/temp collisions (sort/merge interference)
//   - Subtle corruption that still yields a sorted array (histogram catches it)
// ----------------------------------------------------------------------------

// Test stream dependencies with events
TEST(SystemMultiStreamTests, StreamDependenciesWithEvents) {
  using T = int;
  const size_t size = 50'000;

  hipStream_t stream1, stream2, stream3;
  HIP_CHECK(hipStreamCreateWithFlags(&stream1, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&stream2, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&stream3, hipStreamNonBlocking));

  hipEvent_t event1, event2;
  HIP_CHECK(hipEventCreate(&event1));
  HIP_CHECK(hipEventCreate(&event2));

  std::vector<T> input1 = tu::generate_random_data<T>(size, 1, 100);
  std::vector<T> input2 = tu::generate_random_data<T>(size, 1, 100);

  T *d_input1, *d_input2, *d_sorted1, *d_sorted2, *d_merged;
  HIP_CHECK(hipMalloc(&d_input1, size * sizeof(T)));
  HIP_CHECK(hipMalloc(&d_input2, size * sizeof(T)));
  HIP_CHECK(hipMalloc(&d_sorted1, size * sizeof(T)));
  HIP_CHECK(hipMalloc(&d_sorted2, size * sizeof(T)));
  HIP_CHECK(hipMalloc(&d_merged, 2 * size * sizeof(T)));

  HIP_CHECK(hipMemcpyAsync(d_input1, input1.data(), size * sizeof(T),
                           hipMemcpyHostToDevice, stream1));
  HIP_CHECK(hipMemcpyAsync(d_input2, input2.data(), size * sizeof(T),
                           hipMemcpyHostToDevice, stream2));

  // Sort stream1
  void *d_temp1 = nullptr;
  size_t temp_bytes1 = 0;
  HIP_CHECK(rocprim::radix_sort_keys(nullptr, temp_bytes1, d_input1, d_sorted1,
                                     size, 0, sizeof(T) * 8, stream1));
  if (temp_bytes1 > 0)
    HIP_CHECK(hipMalloc(&d_temp1, temp_bytes1));
  HIP_CHECK(rocprim::radix_sort_keys(d_temp1, temp_bytes1, d_input1, d_sorted1,
                                     size, 0, sizeof(T) * 8, stream1));
  HIP_CHECK(hipEventRecord(event1, stream1));

  // Sort stream2
  void *d_temp2 = nullptr;
  size_t temp_bytes2 = 0;
  HIP_CHECK(rocprim::radix_sort_keys(nullptr, temp_bytes2, d_input2, d_sorted2,
                                     size, 0, sizeof(T) * 8, stream2));
  if (temp_bytes2 > 0)
    HIP_CHECK(hipMalloc(&d_temp2, temp_bytes2));
  HIP_CHECK(rocprim::radix_sort_keys(d_temp2, temp_bytes2, d_input2, d_sorted2,
                                     size, 0, sizeof(T) * 8, stream2));
  HIP_CHECK(hipEventRecord(event2, stream2));

  // stream3 waits on both
  HIP_CHECK(hipStreamWaitEvent(stream3, event1, 0));
  HIP_CHECK(hipStreamWaitEvent(stream3, event2, 0));

  // Merge on stream3 (FIXED signature: input1, input2, output, sizes)
  void *d_temp3 = nullptr;
  size_t temp_bytes3 = 0;
  HIP_CHECK(rocprim::merge(nullptr, temp_bytes3, d_sorted1, d_sorted2, d_merged,
                           size, size, rocprim::less<T>(), stream3));
  if (temp_bytes3 > 0)
    HIP_CHECK(hipMalloc(&d_temp3, temp_bytes3));

  HIP_CHECK(rocprim::merge(d_temp3, temp_bytes3, d_sorted1, d_sorted2, d_merged,
                           size, size, rocprim::less<T>(), stream3));

  HIP_CHECK(hipStreamSynchronize(stream3));

  std::vector<T> merged(2 * size);
  HIP_CHECK(hipMemcpy(merged.data(), d_merged, 2 * size * sizeof(T),
                      hipMemcpyDeviceToHost));

  // sortedness
  for (size_t i = 1; i < merged.size(); i++)
    ASSERT_LE(merged[i - 1], merged[i]) << "Merged not sorted at " << i;

  // multiset check (histogram since range 1..100)
  auto hist_1_100 = [](const std::vector<T> &v) -> std::vector<int> {
    std::vector<int> h(101, 0);
    for (T x : v) {
      if (x < 1 || x > 100) {
        ADD_FAILURE() << "Value out of range [1..100]: " << x;
        continue;  // avoid out-of-bounds
      }
      h[static_cast<int>(x)]++;
    }
    return h;
  };

  auto h1 = hist_1_100(input1);
  auto h2 = hist_1_100(input2);
  auto hm = hist_1_100(merged);
  for (int k = 1; k <= 100; ++k)
    ASSERT_EQ(hm[k], h1[k] + h2[k])
        << "Merged multiset mismatch at value " << k;

  // Cleanup
  HIP_CHECK(hipFree(d_input1));
  HIP_CHECK(hipFree(d_input2));
  HIP_CHECK(hipFree(d_sorted1));
  HIP_CHECK(hipFree(d_sorted2));
  HIP_CHECK(hipFree(d_merged));
  if (d_temp1)
    HIP_CHECK(hipFree(d_temp1));
  if (d_temp2)
    HIP_CHECK(hipFree(d_temp2));
  if (d_temp3)
    HIP_CHECK(hipFree(d_temp3));
  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipEventDestroy(event2));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  HIP_CHECK(hipStreamDestroy(stream3));
}
