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

__global__ void fill_kernel(int *out, int n, int value) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = value;
}

__global__ void heavy_kernel(float *a, const float *b, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = a[i];
  for (int k = 0; k < iters; ++k) x = fmaf(x, b[i], 1.0f);
  a[i] = x;
}

int main(int argc, char **argv) {
  if (arg_flag(argc, argv, "--help") || arg_flag(argc, argv, "-h")) {
    std::fprintf(stderr, "Usage: %s [--n=8] [--elems=E] [--iters=I]\n", argv[0]);
    print_usage_tail(argv[0]);
    return 0;
  }
  const int n_streams = arg_int(argc, argv, "--n", 8);
  const int elems = arg_int(argc, argv, "--elems", 1 << 20);
  const int iters = arg_int(argc, argv, "--iters", 256);

  HIP_CHECK(hipSetDevice(0));
  print_env_header("hwq_null_stream_protection_regr");

  const int out_n = 1024;
  const size_t out_bytes = sizeof(int) * static_cast<size_t>(out_n);
  int *d_out = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_out), out_bytes));

  hipLaunchKernelGGL(fill_kernel, dim3((out_n + 255) / 256), dim3(256), 0, nullptr, d_out, out_n, 7);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<int> check1(static_cast<size_t>(out_n));
  HIP_CHECK(hipMemcpy(check1.data(), d_out, out_bytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < out_n; ++i) {
    if (check1[static_cast<size_t>(i)] != 7) {
      std::fprintf(stderr, "FAIL: first null stream fill incorrect at index %d (got %d, expected 7)\n", i,
                   check1[static_cast<size_t>(i)]);
      return 1;
    }
  }
  std::printf("phase1: initial null-stream fill verified (value=7)\n");

  std::vector<hipStream_t> st(static_cast<size_t>(n_streams));
  for (int i = 0; i < n_streams; ++i) HIP_CHECK(hipStreamCreate(&st[static_cast<size_t>(i)]));
  float *d_a, *d_b;
  const size_t buf_bytes = sizeof(float) * static_cast<size_t>(elems);
  HIP_CHECK(hipMalloc(&d_a, buf_bytes));
  HIP_CHECK(hipMalloc(&d_b, buf_bytes));
  std::vector<float> h(static_cast<size_t>(elems), 1.0f);
  HIP_CHECK(hipMemcpy(d_b, h.data(), buf_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_a, h.data(), buf_bytes, hipMemcpyHostToDevice));
  const int threads = 256;
  const int blocks = (elems + threads - 1) / threads;
  for (int s = 0; s < n_streams; ++s)
    hipLaunchKernelGGL(heavy_kernel, dim3(blocks), dim3(threads), 0, st[static_cast<size_t>(s)], d_a, d_b, elems,
                       iters);
  for (int s = 0; s < n_streams; ++s) HIP_CHECK(hipStreamSynchronize(st[static_cast<size_t>(s)]));
  std::printf("phase2: heavy work on %d explicit streams completed\n", n_streams);

  hipLaunchKernelGGL(fill_kernel, dim3((out_n + 255) / 256), dim3(256), 0, nullptr, d_out, out_n, 42);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<int> host_out(static_cast<size_t>(out_n));
  HIP_CHECK(hipMemcpy(host_out.data(), d_out, out_bytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < out_n; ++i) {
    if (host_out[static_cast<size_t>(i)] != 42) {
      std::fprintf(stderr, "FAIL: second null stream fill incorrect at index %d (got %d, expected 42)\n", i,
                   host_out[static_cast<size_t>(i)]);
      return 1;
    }
  }
  std::printf("phase3: post-heavy null-stream fill verified (value=42)\n");

  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  for (int s = 0; s < n_streams; ++s) HIP_CHECK(hipStreamDestroy(st[static_cast<size_t>(s)]));
  std::printf("PASS\n");
  return 0;
}
