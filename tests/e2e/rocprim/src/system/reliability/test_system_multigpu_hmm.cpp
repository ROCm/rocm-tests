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

// HMM Multi-GPU QA (rocPRIM-on-HMM focused)
//
// Goal of these tests:
//   Validate that rocPRIM device algorithms (sort/reduce/scan) produce correct
//   results when their input/output pointers are allocated with
//   hipMallocManaged (HMM/managed memory), including multi-GPU access patterns.
//
//

// test_system_multigpu_hmm.cpp (updated to avoid GTEST_SKIP/ASSERT in helpers)
//
// Key fix:
// - NEVER call GTEST_SKIP()/ASSERT_* inside helper functions because it only
// returns
//   from the helper, not from the TEST body, which can lead to continued
//   execution and segfaults.
// - Helpers return status/message; TEST body performs GTEST_SKIP()/ASSERT_* and
// returns.

#include "../../include/test_common.hpp"

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <cstdint>
#include <cstring>
#include <future>  // NOLINT(build/c++11)
#include <numeric>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

namespace {

// -------------------- Managed memory gating --------------------

inline bool is_managed_memory_supported(std::string *why_not) {
  int dev = 0;
  hipError_t st = hipGetDevice(&dev);
  if (st != hipSuccess) {
    if (why_not)
      *why_not = std::string("hipGetDevice failed: ") + hipGetErrorString(st);
    return false;
  }

  int managed = 0;
  st = hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory, dev);
  if (st != hipSuccess) {
    if (why_not)
      *why_not = std::string("hipDeviceGetAttribute(ManagedMemory) failed: ") +
                 hipGetErrorString(st);
    return false;
  }

  if (!managed) {
    if (why_not)
      *why_not = "Managed memory (hipMallocManaged) not supported on this "
                 "device/runtime";
    return false;
  }

  return true;
}

inline hipError_t try_malloc_managed(void **p, size_t bytes) {
  *p = nullptr;
  return hipMallocManaged(p, bytes, hipMemAttachGlobal);
}

inline std::string managed_alloc_fail_msg(const char *name, size_t bytes,
                                          hipError_t st) {
  size_t free_vram = 0, total_vram = 0;
  hipError_t mi = hipMemGetInfo(&free_vram, &total_vram);

  std::string msg = "hipMallocManaged failed for ";
  msg += name;
  msg += " (";
  msg += std::to_string(bytes);
  msg += " bytes). HIP: ";
  msg += hipGetErrorString(st);

  if (mi == hipSuccess) {
    msg += ". Free VRAM=";
    msg += std::to_string(free_vram / (1024ULL * 1024ULL));
    msg += " MB";
  }

  msg += ". Likely memlock limit in container; run with --ulimit memlock=-1:-1 "
         "(and often --cap-add IPC_LOCK).";
  return msg;
}

// -------------------- Best-effort prefetch --------------------

inline void prefetch_if_supported(void *ptr, size_t bytes, int device,
                                  hipStream_t stream) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
  (void)hipMemPrefetchAsync(ptr, bytes, device, stream);
#else
  (void)ptr;
  (void)bytes;
  (void)device;
  (void)stream;
#endif
}

// -------------------- Histogram + scan verification helpers
// -------------------- NOTE: Keep these helpers “pure” (no
// ASSERT/ADD_FAILURE/GTEST_*). Return bool and a message; the TEST body
// asserts/fails.

inline std::vector<int> histogram_range(const int *data, size_t n, int lo,
                                        int hi, bool *ok, std::string *err) {
  std::vector<int> h(static_cast<size_t>(hi - lo + 1), 0);
  if (ok)
    *ok = true;

  for (size_t i = 0; i < n; ++i) {
    int x = data[i];
    if (x < lo || x > hi) {
      if (ok)
        *ok = false;
      if (err) {
        *err = "Value out of range [" + std::to_string(lo) + ".." +
               std::to_string(hi) + "]: " + std::to_string(x) + " at index " +
               std::to_string(i);
      }
      // Keep going to avoid out-of-bounds; histogram remains best-effort.
      continue;
    }
    h[static_cast<size_t>(x - lo)]++;
  }

  return h;
}

inline bool verify_inclusive_scan_spot_checks(
    const std::vector<int> &base_sorted, const std::vector<int> &scan_out,
    const std::vector<size_t> &sample_indices, std::string *err) {
  if (base_sorted.size() != scan_out.size()) {
    if (err)
      *err = "Scan output size mismatch";
    return false;
  }

  const size_t n = base_sorted.size();
  if (n == 0)
    return true;

  // Non-decreasing scan (inputs are positive in these tests)
  for (size_t i = 1; i < n; ++i) {
    if (scan_out[i - 1] > scan_out[i]) {
      if (err)
        *err = "Scan output not non-decreasing at index " + std::to_string(i);
      return false;
    }
  }

  // Total sum check
  int64_t total_sum = 0;
  for (int v : base_sorted)
    total_sum += v;

  if (static_cast<int64_t>(scan_out[n - 1]) != total_sum) {
    if (err)
      *err = "Scan last element mismatch (expected total sum)";
    return false;
  }

  // Spot checks
  int64_t running = 0;
  size_t next_sample = 0;
  size_t target = sample_indices.empty() ? n : sample_indices[next_sample];

  for (size_t i = 0; i < n; ++i) {
    running += base_sorted[i];
    if (i == target) {
      if (static_cast<int64_t>(scan_out[i]) != running) {
        if (err)
          *err = "Scan spot-check mismatch at index " + std::to_string(i);
        return false;
      }
      next_sample++;
      if (next_sample >= sample_indices.size())
        break;
      target = sample_indices[next_sample];
    }
  }

  return true;
}

}  // namespace

// ============================================================================
// HMM MULTI-GPU QA TESTS (rocPRIM-on-HMM correctness)
// ============================================================================

TEST(MultiGPUHMMTests, MemoryMigration_AlternatingGPUAccess) {
  // Managed memory capability gate MUST be in TEST body
  {
    std::string why;
    if (!is_managed_memory_supported(&why)) {
      GTEST_SKIP() << why;
      return;
    }
  }

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
    return;
  }

  std::cout << "\n=== rocPRIM-on-HMM Multi-GPU: Alternating GPU Access ===\n";
  std::cout << "Pattern: GPU0(sort) -> GPU1(reduce) -> GPU0(scan)\n";

  using T = int;
  const size_t size = 1 << 22;  // 4M ints (~16MB)

  // Host input (bounded range so histogram check is cheap & strong)
  std::vector<T> input = test_utils::generate_random_data<T>(size, 1, 100);

  int64_t expected_sum = 0;
  for (T v : input)
    expected_sum += v;

  // -------------------------
  // Baseline control on GPU0 with hipMalloc
  // -------------------------
  std::cout << "  Baseline (hipMalloc on GPU0): sort -> reduce -> scan\n";

  HIP_CHECK(hipSetDevice(0));
  hipStream_t s0_base;
  HIP_CHECK(hipStreamCreateWithFlags(&s0_base, hipStreamNonBlocking));

  T *d_in0 = nullptr;
  T *d_sorted0 = nullptr;
  T *d_scan0 = nullptr;
  T *d_sum0 = nullptr;

  HIP_CHECK(hipMalloc(&d_in0, size * sizeof(T)));
  HIP_CHECK(hipMalloc(&d_sorted0, size * sizeof(T)));
  HIP_CHECK(hipMalloc(&d_scan0, size * sizeof(T)));
  HIP_CHECK(hipMalloc(&d_sum0, sizeof(T)));

  HIP_CHECK(hipMemcpyAsync(d_in0, input.data(), size * sizeof(T),
                           hipMemcpyHostToDevice, s0_base));

  void *tmp_sort0 = nullptr;
  size_t tmp_sort0_bytes = 0;
  HIP_CHECK(rocprim::radix_sort_keys(nullptr, tmp_sort0_bytes, d_in0, d_sorted0,
                                     size, 0, sizeof(T) * 8, s0_base));
  HIP_CHECK(hipMalloc(&tmp_sort0, tmp_sort0_bytes));
  HIP_CHECK(rocprim::radix_sort_keys(tmp_sort0, tmp_sort0_bytes, d_in0,
                                     d_sorted0, size, 0, sizeof(T) * 8,
                                     s0_base));

  void *tmp_reduce0 = nullptr;
  size_t tmp_reduce0_bytes = 0;
  HIP_CHECK(rocprim::reduce(nullptr, tmp_reduce0_bytes, d_sorted0, d_sum0, size,
                            rocprim::plus<T>(), s0_base));
  HIP_CHECK(hipMalloc(&tmp_reduce0, tmp_reduce0_bytes));
  HIP_CHECK(rocprim::reduce(tmp_reduce0, tmp_reduce0_bytes, d_sorted0, d_sum0,
                            size, rocprim::plus<T>(), s0_base));

  void *tmp_scan0 = nullptr;
  size_t tmp_scan0_bytes = 0;
  HIP_CHECK(rocprim::inclusive_scan(nullptr, tmp_scan0_bytes, d_sorted0,
                                    d_scan0, size, rocprim::plus<T>(),
                                    s0_base));
  HIP_CHECK(hipMalloc(&tmp_scan0, tmp_scan0_bytes));
  HIP_CHECK(rocprim::inclusive_scan(tmp_scan0, tmp_scan0_bytes, d_sorted0,
                                    d_scan0, size, rocprim::plus<T>(),
                                    s0_base));

  HIP_CHECK(hipStreamSynchronize(s0_base));

  std::vector<T> baseline_sorted(size);
  std::vector<T> baseline_scan(size);
  T baseline_sum = 0;

  HIP_CHECK(hipMemcpy(baseline_sorted.data(), d_sorted0, size * sizeof(T),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(baseline_scan.data(), d_scan0, size * sizeof(T),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&baseline_sum, d_sum0, sizeof(T), hipMemcpyDeviceToHost));

  ASSERT_TRUE(std::is_sorted(baseline_sorted.begin(), baseline_sorted.end()))
      << "Baseline sort failed (hipMalloc)";

  bool ok_hist = true;
  std::string hist_err;
  auto h_in = histogram_range(input.data(), size, 1, 100, &ok_hist, &hist_err);
  ASSERT_TRUE(ok_hist) << hist_err;

  ok_hist = true;
  auto h_out = histogram_range(baseline_sorted.data(), size, 1, 100, &ok_hist,
                               &hist_err);
  ASSERT_TRUE(ok_hist) << hist_err;

  ASSERT_EQ(h_in, h_out) << "Baseline sort permutation mismatch (hipMalloc)";
  ASSERT_EQ(static_cast<int64_t>(baseline_sum), expected_sum)
      << "Baseline reduce mismatch (hipMalloc)";

  std::vector<size_t> samples = {0, 1, 2, 17, 1023, size / 2, size - 1};
  std::string scan_err;
  ASSERT_TRUE(verify_inclusive_scan_spot_checks(baseline_sorted, baseline_scan,
                                                samples, &scan_err))
      << scan_err;

  HIP_CHECK(hipFree(d_in0));
  HIP_CHECK(hipFree(d_sorted0));
  HIP_CHECK(hipFree(d_scan0));
  HIP_CHECK(hipFree(d_sum0));
  HIP_CHECK(hipFree(tmp_sort0));
  HIP_CHECK(hipFree(tmp_reduce0));
  HIP_CHECK(hipFree(tmp_scan0));
  HIP_CHECK(hipStreamDestroy(s0_base));

  // -------------------------
  // Managed memory pipeline (HMM): GPU0(sort) -> GPU1(reduce) -> GPU0(scan)
  // -------------------------
  std::cout << "  Managed (hipMallocManaged): GPU0(sort) -> GPU1(reduce) -> "
               "GPU0(scan)\n";

  T *d_managed_in = nullptr;
  T *d_managed_mid = nullptr;
  T *d_managed_scan = nullptr;
  T *d_managed_sum = nullptr;

  auto free_managed = [&]() {
    if (d_managed_in) {
      HIP_CHECK(hipFree(d_managed_in));
      d_managed_in = nullptr;
    }
    if (d_managed_mid) {
      HIP_CHECK(hipFree(d_managed_mid));
      d_managed_mid = nullptr;
    }
    if (d_managed_scan) {
      HIP_CHECK(hipFree(d_managed_scan));
      d_managed_scan = nullptr;
    }
    if (d_managed_sum) {
      HIP_CHECK(hipFree(d_managed_sum));
      d_managed_sum = nullptr;
    }
  };

  hipError_t st = hipSuccess;

  st = try_malloc_managed(reinterpret_cast<void **>(&d_managed_in),
                          size * sizeof(T));
  if (st != hipSuccess) {
    free_managed();
    GTEST_SKIP() << managed_alloc_fail_msg("d_managed_in", size * sizeof(T),
                                           st);
    return;
  }

  st = try_malloc_managed(reinterpret_cast<void **>(&d_managed_mid),
                          size * sizeof(T));
  if (st != hipSuccess) {
    free_managed();
    GTEST_SKIP() << managed_alloc_fail_msg("d_managed_mid", size * sizeof(T),
                                           st);
    return;
  }

  st = try_malloc_managed(reinterpret_cast<void **>(&d_managed_scan),
                          size * sizeof(T));
  if (st != hipSuccess) {
    free_managed();
    GTEST_SKIP() << managed_alloc_fail_msg("d_managed_scan", size * sizeof(T),
                                           st);
    return;
  }

  st = try_malloc_managed(reinterpret_cast<void **>(&d_managed_sum), sizeof(T));
  if (st != hipSuccess) {
    free_managed();
    GTEST_SKIP() << managed_alloc_fail_msg("d_managed_sum", sizeof(T), st);
    return;
  }

  std::memcpy(d_managed_in, input.data(), size * sizeof(T));

  HIP_CHECK(hipSetDevice(0));
  hipStream_t s0;
  HIP_CHECK(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));

  prefetch_if_supported(d_managed_in, size * sizeof(T), 0, s0);
  prefetch_if_supported(d_managed_mid, size * sizeof(T), 0, s0);

  void *tmp_sort = nullptr;
  size_t tmp_sort_bytes = 0;
  HIP_CHECK(rocprim::radix_sort_keys(nullptr, tmp_sort_bytes, d_managed_in,
                                     d_managed_mid, size, 0, sizeof(T) * 8,
                                     s0));
  HIP_CHECK(hipMalloc(&tmp_sort, tmp_sort_bytes));
  HIP_CHECK(rocprim::radix_sort_keys(tmp_sort, tmp_sort_bytes, d_managed_in,
                                     d_managed_mid, size, 0, sizeof(T) * 8,
                                     s0));
  HIP_CHECK(hipStreamSynchronize(s0));

  HIP_CHECK(hipSetDevice(1));
  hipStream_t s1;
  HIP_CHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));

  prefetch_if_supported(d_managed_mid, size * sizeof(T), 1, s1);
  prefetch_if_supported(d_managed_sum, sizeof(T), 1, s1);

  void *tmp_reduce = nullptr;
  size_t tmp_reduce_bytes = 0;
  HIP_CHECK(rocprim::reduce(nullptr, tmp_reduce_bytes, d_managed_mid,
                            d_managed_sum, size, rocprim::plus<T>(), s1));
  HIP_CHECK(hipMalloc(&tmp_reduce, tmp_reduce_bytes));
  HIP_CHECK(rocprim::reduce(tmp_reduce, tmp_reduce_bytes, d_managed_mid,
                            d_managed_sum, size, rocprim::plus<T>(), s1));
  HIP_CHECK(hipStreamSynchronize(s1));

  HIP_CHECK(hipSetDevice(0));
  prefetch_if_supported(d_managed_mid, size * sizeof(T), 0, s0);
  prefetch_if_supported(d_managed_scan, size * sizeof(T), 0, s0);

  void *tmp_scan = nullptr;
  size_t tmp_scan_bytes = 0;
  HIP_CHECK(rocprim::inclusive_scan(nullptr, tmp_scan_bytes, d_managed_mid,
                                    d_managed_scan, size, rocprim::plus<T>(),
                                    s0));
  HIP_CHECK(hipMalloc(&tmp_scan, tmp_scan_bytes));
  HIP_CHECK(rocprim::inclusive_scan(tmp_scan, tmp_scan_bytes, d_managed_mid,
                                    d_managed_scan, size, rocprim::plus<T>(),
                                    s0));
  HIP_CHECK(hipStreamSynchronize(s0));

  std::cout << "  Validating managed results (correctness only)...\n";

  std::vector<T> managed_sorted(size);
  std::memcpy(managed_sorted.data(), d_managed_mid, size * sizeof(T));
  ASSERT_TRUE(std::is_sorted(managed_sorted.begin(), managed_sorted.end()))
      << "Managed sort output is not sorted";

  ok_hist = true;
  auto h_managed =
      histogram_range(managed_sorted.data(), size, 1, 100, &ok_hist, &hist_err);
  ASSERT_TRUE(ok_hist) << hist_err;

  ASSERT_EQ(h_in, h_managed) << "Managed sort permutation mismatch";
  ASSERT_EQ(static_cast<int64_t>(*d_managed_sum), expected_sum)
      << "Managed reduce mismatch";

  std::vector<T> managed_scan(size);
  std::memcpy(managed_scan.data(), d_managed_scan, size * sizeof(T));
  ASSERT_TRUE(verify_inclusive_scan_spot_checks(managed_sorted, managed_scan,
                                                samples, &scan_err))
      << scan_err;

  std::cout << "     rocPRIM-on-HMM: sort/reduce/scan correct across "
               "GPU0->GPU1->GPU0\n";

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipFree(tmp_sort));
  HIP_CHECK(hipFree(tmp_scan));
  HIP_CHECK(hipStreamDestroy(s0));

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipFree(tmp_reduce));
  HIP_CHECK(hipStreamDestroy(s1));

  free_managed();
}

TEST(MultiGPUHMMTests, MemoryCoherence_PartitionedConcurrentAccess) {
  // Managed memory capability gate MUST be in TEST body
  {
    std::string why;
    if (!is_managed_memory_supported(&why)) {
      GTEST_SKIP() << why;
      return;
    }
  }

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
    return;
  }

  std::cout
      << "\n=== rocPRIM-on-HMM Multi-GPU: Partitioned Concurrent Access ===\n";
  std::cout
      << "Each GPU sorts its partition concurrently using managed pointers.\n";

  using T = int;
  const size_t total_size = 1 << 24;  // 16M ints (~64MB)
  const size_t partition_size = total_size / static_cast<size_t>(device_count);
  const int LO = 1, HI = 1000;

  T *d_managed_in = nullptr;
  T *d_managed_out = nullptr;

  hipError_t st = try_malloc_managed(reinterpret_cast<void **>(&d_managed_in),
                                     total_size * sizeof(T));
  if (st != hipSuccess) {
    GTEST_SKIP() << managed_alloc_fail_msg("d_managed_in",
                                           total_size * sizeof(T), st);
    return;
  }

  st = try_malloc_managed(reinterpret_cast<void **>(&d_managed_out),
                          total_size * sizeof(T));
  if (st != hipSuccess) {
    HIP_CHECK(hipFree(d_managed_in));
    d_managed_in = nullptr;
    GTEST_SKIP() << managed_alloc_fail_msg("d_managed_out",
                                           total_size * sizeof(T), st);
    return;
  }

  std::vector<T> inp = test_utils::generate_random_data<T>(total_size, LO, HI);
  std::memcpy(d_managed_in, inp.data(), total_size * sizeof(T));

  std::vector<hipStream_t> streams(device_count);
  std::vector<void *> temps(device_count, nullptr);
  std::vector<size_t> temp_bytes(device_count, 0);

  for (int dev = 0; dev < device_count; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamCreateWithFlags(&streams[dev], hipStreamNonBlocking));

    T *in_part = d_managed_in + static_cast<size_t>(dev) * partition_size;
    T *out_part = d_managed_out + static_cast<size_t>(dev) * partition_size;

    HIP_CHECK(rocprim::radix_sort_keys(nullptr, temp_bytes[dev], in_part,
                                       out_part, partition_size, 0,
                                       sizeof(T) * 8, streams[dev]));
    HIP_CHECK(hipMalloc(&temps[dev], temp_bytes[dev]));
  }

  auto gpu_sort_task = [&](int dev) {
    HIP_CHECK(hipSetDevice(dev));

    T *in_part = d_managed_in + static_cast<size_t>(dev) * partition_size;
    T *out_part = d_managed_out + static_cast<size_t>(dev) * partition_size;

    prefetch_if_supported(in_part, partition_size * sizeof(T), dev,
                          streams[dev]);
    prefetch_if_supported(out_part, partition_size * sizeof(T), dev,
                          streams[dev]);

    HIP_CHECK(rocprim::radix_sort_keys(temps[dev], temp_bytes[dev], in_part,
                                       out_part, partition_size, 0,
                                       sizeof(T) * 8, streams[dev]));
    HIP_CHECK(hipStreamSynchronize(streams[dev]));
    return true;
  };

  std::cout << "  Launching concurrent sorts on " << device_count
            << " GPUs...\n";
  std::vector<std::future<bool>> futures;
  futures.reserve(device_count);
  for (int dev = 0; dev < device_count; ++dev)
    futures.push_back(std::async(std::launch::async, gpu_sort_task, dev));

  for (int dev = 0; dev < device_count; ++dev) {
    bool ok = futures[dev].get();
    ASSERT_TRUE(ok) << "GPU " << dev << " worker failed";
  }

  std::cout << "  Validating each partition: sortedness + permutation "
               "(histogram)...\n";

  for (int dev = 0; dev < device_count; ++dev) {
    const size_t base = static_cast<size_t>(dev) * partition_size;

    const T *in_part_cpu = inp.data() + base;
    const T *out_part_cpu = d_managed_out + base;

    for (size_t i = 1; i < partition_size; ++i) {
      ASSERT_LE(out_part_cpu[i - 1], out_part_cpu[i])
          << "Partition " << dev << " not sorted at local index " << i;
    }

    bool ok_hist = true;
    std::string hist_err;
    auto h_in = histogram_range(in_part_cpu, partition_size, LO, HI, &ok_hist,
                                &hist_err);
    ASSERT_TRUE(ok_hist) << hist_err;

    ok_hist = true;
    auto h_out = histogram_range(out_part_cpu, partition_size, LO, HI, &ok_hist,
                                 &hist_err);
    ASSERT_TRUE(ok_hist) << hist_err;

    ASSERT_EQ(h_in, h_out) << "Partition " << dev << " permutation mismatch";

    std::cout << "    GPU " << dev << " partition OK\n";
  }

  std::cout << "rocPRIM-on-HMM: concurrent partitioned sorts correct across "
               "all GPUs\n";

  for (int dev = 0; dev < device_count; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipFree(temps[dev]));
    HIP_CHECK(hipStreamDestroy(streams[dev]));
  }

  HIP_CHECK(hipFree(d_managed_in));
  HIP_CHECK(hipFree(d_managed_out));
}
