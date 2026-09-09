/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
*/
// hwq_stream_priority -- HIP stream-priority semantics + DEBUG_HIP_IGNORE_STREAM_PRIORITY.
//
// Queries the device stream-priority range, creates a high-priority and a
// low-priority stream, runs a small workload on each, and verifies results are
// correct.
//
// Scope: priority *reporting* plus functional correctness only. Relative
// scheduling latency / dispatch ordering across priority queues is deliberately
// not asserted here -- that is timing-sensitive and belongs in a separate
// perf-style test.
//
// Two modes:
//
//   (default / semantics): asserts the queue heuristic PRESERVES priority --
//     hipStreamGetPriority() echoes the requested (range-clamped) value for both
//     streams, and both workloads produce correct output.
//
//   --ignore: run under DEBUG_HIP_IGNORE_STREAM_PRIORITY=1. The override tells
//     HIP to ignore the requested priority; this mode only asserts the override
//     is honored gracefully -- streams still create and both workloads execute
//     correctly (no assertion on the echoed priority value).
//
// Prints "hwq_stream_priority: PASSED" on success.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#define HIP_CHECK(cmd)                                                                    \
  do {                                                                                     \
    hipError_t _e = (cmd);                                                                 \
    if (_e != hipSuccess) {                                                                \
      std::printf("HIP error %s at %s:%d\n", hipGetErrorString(_e), __FILE__, __LINE__);  \
      std::printf("hwq_stream_priority: FAILED\n");                                        \
      return 1;                                                                            \
    }                                                                                       \
  } while (0)

__global__ void scale_add(float* a, const float* b, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = a[i];
  for (int k = 0; k < iters; ++k) x = fmaf(x, 1.0f, b[i]);  // x += b, iters times
  a[i] = x;
}

static int run_on_stream(hipStream_t st, int n, int iters, float expect) {
  const size_t bytes = sizeof(float) * static_cast<size_t>(n);
  float *da = nullptr, *db = nullptr;
  HIP_CHECK(hipMalloc(&da, bytes));
  HIP_CHECK(hipMalloc(&db, bytes));
  std::vector<float> h(static_cast<size_t>(n), 0.0f), hb(static_cast<size_t>(n), 1.0f), out(static_cast<size_t>(n));
  HIP_CHECK(hipMemcpyAsync(da, h.data(), bytes, hipMemcpyHostToDevice, st));
  HIP_CHECK(hipMemcpyAsync(db, hb.data(), bytes, hipMemcpyHostToDevice, st));
  const int threads = 256, blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(scale_add, dim3(blocks), dim3(threads), 0, st, da, db, n, iters);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpyAsync(out.data(), da, bytes, hipMemcpyDeviceToHost, st));
  HIP_CHECK(hipStreamSynchronize(st));
  HIP_CHECK(hipFree(da));
  HIP_CHECK(hipFree(db));
  for (int i = 0; i < n; ++i) {
    if (out[i] != expect) {
      std::printf("verify mismatch at %d: got %f want %f\nhwq_stream_priority: FAILED\n", i, out[i], expect);
      return 2;
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  bool ignore_mode = false;
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--ignore") == 0) ignore_mode = true;

  HIP_CHECK(hipSetDevice(0));

  int least = 0, greatest = 0;  // least = lowest priority (numerically largest), greatest = highest
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&least, &greatest));
  std::printf("stream priority range: least(low)=%d greatest(high)=%d ignore_mode=%d\n", least, greatest, ignore_mode);

  hipStream_t high = nullptr, low = nullptr;
  HIP_CHECK(hipStreamCreateWithPriority(&high, hipStreamNonBlocking, greatest));
  HIP_CHECK(hipStreamCreateWithPriority(&low, hipStreamNonBlocking, least));

  int high_got = 0, low_got = 0;
  HIP_CHECK(hipStreamGetPriority(high, &high_got));
  HIP_CHECK(hipStreamGetPriority(low, &low_got));
  std::printf("requested high=%d got=%d ; requested low=%d got=%d\n", greatest, high_got, least, low_got);

  const int n = 1 << 16, iters = 64;
  const float expect = static_cast<float>(iters);  // 0 + 1*iters
  int rc = run_on_stream(high, n, iters, expect);
  if (rc == 0) rc = run_on_stream(low, n, iters, expect);

  HIP_CHECK(hipStreamDestroy(high));
  HIP_CHECK(hipStreamDestroy(low));
  if (rc != 0) return rc;

  if (!ignore_mode) {
    // Semantics mode: the queue heuristic must preserve the requested priority.
    // hipStreamGetPriority echoes the range-clamped request; high must be at
    // least as high-priority (numerically <=) as low.
    if (high_got != greatest || low_got != least) {
      std::printf("priority not preserved (high got %d want %d; low got %d want %d)\n"
                  "hwq_stream_priority: FAILED\n",
                  high_got, greatest, low_got, least);
      return 3;
    }
    if (high_got > low_got) {
      std::printf("priority ordering wrong (high %d > low %d)\nhwq_stream_priority: FAILED\n", high_got, low_got);
      return 3;
    }
    std::printf("priority semantics preserved\n");
  } else {
    std::printf("ignore-override accepted; workloads correct\n");
  }
  std::printf("hwq_stream_priority: PASSED\n");
  return 0;
}
