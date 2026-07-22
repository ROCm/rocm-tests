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
#include "test_utils.h"
#include <cstdio>
#include <vector>

__global__ void touch_kernel(float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += 1.0f;
}

static void run_one_gpu(int dev, int n_streams, int elems) {
  HIP_CHECK(hipSetDevice(dev));
  std::vector<hipStream_t> st(static_cast<size_t>(n_streams));
  for (int i = 0; i < n_streams; ++i) HIP_CHECK(hipStreamCreate(&st[static_cast<size_t>(i)]));
  float *d = nullptr;
  const size_t buf_bytes = sizeof(float) * static_cast<size_t>(elems);
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d), buf_bytes));
  HIP_CHECK(hipMemset(d, 0, buf_bytes));
  const int threads = 256;
  const int blocks = (elems + threads - 1) / threads;
  const double t0 = now_sec();
  for (int r = 0; r < 8; ++r) {
    for (int s = 0; s < n_streams; ++s)
      hipLaunchKernelGGL(touch_kernel, dim3(blocks), dim3(threads), 0, st[static_cast<size_t>(s)], d, elems);
  }
  for (int s = 0; s < n_streams; ++s) HIP_CHECK(hipStreamSynchronize(st[static_cast<size_t>(s)]));
  const double dt = now_sec() - t0;
  std::printf("device=%d streams=%d time_sec=%.6f\n", dev, n_streams, dt);
  HIP_CHECK(hipFree(d));
  for (int s = 0; s < n_streams; ++s) HIP_CHECK(hipStreamDestroy(st[static_cast<size_t>(s)]));
}

int main(int argc, char **argv) {
  if (arg_flag(argc, argv, "--help") || arg_flag(argc, argv, "-h")) {
    std::fprintf(stderr, "Usage: %s [--gpus=count] [--streams=N] [--elems=E] [--p2p]\n", argv[0]);
    print_usage_tail(argv[0]);
    return 0;
  }
  const int gpu_count = arg_int(argc, argv, "--gpus", 1);
  const int n_streams = arg_int(argc, argv, "--streams", 8);
  const int elems = arg_int(argc, argv, "--elems", 1 << 18);
  const bool p2p = arg_flag(argc, argv, "--p2p");

  HIP_CHECK(hipSetDevice(0));
  print_env_header("hwq_per_device_independence_test");

  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  if (gpu_count > ndev) {
    std::fprintf(stderr, "Requested %d GPUs but only %d visible\n", gpu_count, ndev);
    return 2;
  }

  if (p2p && gpu_count >= 2) {
    for (int src = 0; src + 1 < gpu_count; ++src) {
      const int dst = src + 1;
      int can_access = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can_access, dst, src));
      if (!can_access) {
        std::fprintf(stderr, "P2P requested but device %d cannot access device %d\n", dst, src);
        return 2;
      }

      float *d_src = nullptr;
      float *d_dst = nullptr;
      HIP_CHECK(hipSetDevice(src));
      HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_src), sizeof(float) * 4096));
      HIP_CHECK(hipMemset(d_src, 0, sizeof(float) * 4096));
      HIP_CHECK(hipSetDevice(dst));
      HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_dst), sizeof(float) * 4096));
      HIP_CHECK(hipMemcpyPeer(d_dst, dst, d_src, src, sizeof(float) * 4096));
      HIP_CHECK(hipDeviceSynchronize());
      std::printf("p2p_copy src=%d dst=%d bytes=%zu\n", src, dst, sizeof(float) * static_cast<size_t>(4096));

      HIP_CHECK(hipSetDevice(src));
      HIP_CHECK(hipFree(d_src));
      HIP_CHECK(hipSetDevice(dst));
      HIP_CHECK(hipFree(d_dst));
    }
  }

  for (int d = 0; d < gpu_count; ++d) run_one_gpu(d, n_streams, elems);

  std::printf("PASS\n");
  return 0;
}
