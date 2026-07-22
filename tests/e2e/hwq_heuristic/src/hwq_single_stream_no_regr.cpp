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

__global__ void saxpy_kernel(float *y, const float *x, int n, float a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main(int argc, char **argv) {
  if (arg_flag(argc, argv, "--help") || arg_flag(argc, argv, "-h")) {
    std::fprintf(stderr, "Usage: %s [--n=SIZE] [--passes=P] [--warmup=W]\n", argv[0]);
    print_usage_tail(argv[0]);
    return 0;
  }
  const int n = arg_int(argc, argv, "--n", 1 << 24);
  const int passes = arg_int(argc, argv, "--passes", 32);
  const int warmup = arg_int(argc, argv, "--warmup", 4);

  HIP_CHECK(hipSetDevice(0));
  print_env_header("hwq_single_stream_no_regr");

  float *d_x, *d_y;
  const size_t byte_n = sizeof(float) * static_cast<size_t>(n);
  HIP_CHECK(hipMalloc(&d_x, byte_n));
  HIP_CHECK(hipMalloc(&d_y, byte_n));
  std::vector<float> h(static_cast<size_t>(n), 1.0f);
  HIP_CHECK(hipMemcpy(d_x, h.data(), byte_n, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_y, h.data(), byte_n, hipMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  hipStream_t st;
  HIP_CHECK(hipStreamCreate(&st));

  for (int i = 0; i < warmup; ++i)
    hipLaunchKernelGGL(saxpy_kernel, dim3(blocks), dim3(threads), 0, st, d_y, d_x, n, 2.0f);
  HIP_CHECK(hipStreamSynchronize(st));

  const double t0 = now_sec();
  for (int i = 0; i < passes; ++i)
    hipLaunchKernelGGL(saxpy_kernel, dim3(blocks), dim3(threads), 0, st, d_y, d_x, n, 2.0f);
  HIP_CHECK(hipStreamSynchronize(st));
  const double dt = now_sec() - t0;

  const double elems = static_cast<double>(n) * passes;
  std::printf("single_stream_saxpy_elements=%.0f time_sec=%.6f elements_per_sec=%.3e DEBUG_HIP_DYNAMIC_QUEUES=%d\n",
              elems, dt, elems / dt, getenv_int("DEBUG_HIP_DYNAMIC_QUEUES", -1));

  HIP_CHECK(hipStreamDestroy(st));
  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_y));
  std::printf("PASS\n");
  return 0;
}
