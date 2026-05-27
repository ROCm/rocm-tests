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
#include "test_utils.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void fma_kernel(float *a, const float *b, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = a[i];
  for (int k = 0; k < iters; ++k) x = fmaf(x, b[i], 1.0f);
  a[i] = x;
}

int main(int argc, char **argv) {
  if (arg_flag(argc, argv, "--help") || arg_flag(argc, argv, "-h")) {
    std::fprintf(stderr,
                 "Usage: %s [--compute-streams=4] [--copy-streams=4] [--elems=E] [--iters=I] [--rounds=R] "
                 "[--check-overlap]\n",
                 argv[0]);
    print_usage_tail(argv[0]);
    return 0;
  }

  const int n_compute = arg_int(argc, argv, "--compute-streams", 4);
  const int n_copy = arg_int(argc, argv, "--copy-streams", 4);
  const int elems = arg_int(argc, argv, "--elems", 1 << 22);
  const int iters = arg_int(argc, argv, "--iters", 128);
  const int rounds = arg_int(argc, argv, "--rounds", 8);
  const bool check_overlap = arg_flag(argc, argv, "--check-overlap");

  HIP_CHECK(hipSetDevice(0));
  print_env_header("hwq_compute_copy_overlap_test");

  const int threads = 256;
  const int blocks = (elems + threads - 1) / threads;
  const size_t buf_bytes = sizeof(float) * static_cast<size_t>(elems);

  std::vector<hipStream_t> s_comp(static_cast<size_t>(n_compute));
  std::vector<hipStream_t> s_copy(static_cast<size_t>(n_copy));
  for (int i = 0; i < n_compute; ++i) HIP_CHECK(hipStreamCreate(&s_comp[static_cast<size_t>(i)]));
  for (int i = 0; i < n_copy; ++i) HIP_CHECK(hipStreamCreate(&s_copy[static_cast<size_t>(i)]));

  float *d_comp_a, *d_comp_b;
  HIP_CHECK(hipMalloc(&d_comp_a, buf_bytes));
  HIP_CHECK(hipMalloc(&d_comp_b, buf_bytes));
  std::vector<float> h(static_cast<size_t>(elems), 1.0f);
  HIP_CHECK(hipMemcpy(d_comp_b, h.data(), buf_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_comp_a, h.data(), buf_bytes, hipMemcpyHostToDevice));

  std::vector<float *> d_copy(static_cast<size_t>(n_copy));
  std::vector<float *> h_ping(static_cast<size_t>(n_copy));
  std::vector<float *> h_pong(static_cast<size_t>(n_copy));
  for (int i = 0; i < n_copy; ++i) {
    HIP_CHECK(hipMalloc(&d_copy[static_cast<size_t>(i)], buf_bytes));
    HIP_CHECK(hipMemset(d_copy[static_cast<size_t>(i)], 0, buf_bytes));
    HIP_CHECK(hipHostMalloc(&h_ping[static_cast<size_t>(i)], buf_bytes));
    HIP_CHECK(hipHostMalloc(&h_pong[static_cast<size_t>(i)], buf_bytes));
    for (int j = 0; j < elems; ++j)
      h_ping[static_cast<size_t>(i)][j] = h_pong[static_cast<size_t>(i)][j] = 1.0f;
  }

  auto run_compute_only = [&]() {
    const double t0 = now_sec();
    for (int r = 0; r < rounds; ++r) {
      for (int i = 0; i < n_compute; ++i) {
        hipLaunchKernelGGL(fma_kernel, dim3(blocks), dim3(threads), 0, s_comp[static_cast<size_t>(i)], d_comp_a,
                           d_comp_b, elems, iters);
      }
    }
    for (int i = 0; i < n_compute; ++i) HIP_CHECK(hipStreamSynchronize(s_comp[static_cast<size_t>(i)]));
    return now_sec() - t0;
  };

  auto run_copy_only = [&]() {
    const double t0 = now_sec();
    for (int r = 0; r < rounds; ++r) {
      for (int i = 0; i < n_copy; ++i) {
        const auto si = static_cast<size_t>(i);
        HIP_CHECK(hipMemcpyAsync(d_copy[si], h_ping[si], buf_bytes, hipMemcpyHostToDevice,
                                 s_copy[si]));
        HIP_CHECK(hipMemcpyAsync(h_pong[si], d_copy[si], buf_bytes, hipMemcpyDeviceToHost,
                                 s_copy[si]));
      }
    }
    for (int i = 0; i < n_copy; ++i) HIP_CHECK(hipStreamSynchronize(s_copy[static_cast<size_t>(i)]));
    return now_sec() - t0;
  };

  const double t_comp = run_compute_only();
  HIP_CHECK(hipMemcpy(d_comp_a, h.data(), buf_bytes, hipMemcpyHostToDevice));
  const double t_copy = run_copy_only();
  HIP_CHECK(hipMemcpy(d_comp_a, h.data(), buf_bytes, hipMemcpyHostToDevice));

  const double t0 = now_sec();
  for (int r = 0; r < rounds; ++r) {
    for (int i = 0; i < n_compute; ++i) {
      hipLaunchKernelGGL(fma_kernel, dim3(blocks), dim3(threads), 0, s_comp[static_cast<size_t>(i)], d_comp_a,
                         d_comp_b, elems, iters);
    }
    for (int i = 0; i < n_copy; ++i) {
      const auto si = static_cast<size_t>(i);
      HIP_CHECK(hipMemcpyAsync(d_copy[si], h_ping[si], buf_bytes, hipMemcpyHostToDevice,
                               s_copy[si]));
      HIP_CHECK(hipMemcpyAsync(h_pong[si], d_copy[si], buf_bytes, hipMemcpyDeviceToHost,
                               s_copy[si]));
    }
  }
  for (int i = 0; i < n_compute; ++i) HIP_CHECK(hipStreamSynchronize(s_comp[static_cast<size_t>(i)]));
  for (int i = 0; i < n_copy; ++i) HIP_CHECK(hipStreamSynchronize(s_copy[static_cast<size_t>(i)]));
  const double t_mixed = now_sec() - t0;

  const double sum_parts = t_comp + t_copy;
  std::printf("time_compute_alone_sec=%.6f time_copy_alone_sec=%.6f time_mixed_sec=%.6f sum_parts_sec=%.6f\n",
              t_comp, t_copy, t_mixed, sum_parts);

  int rc = 0;
  if (check_overlap && t_mixed >= sum_parts) {
    std::fprintf(stderr, "FAIL: mixed time not below sum of parts (little overlap)\n");
    rc = 1;
  }

  for (int i = 0; i < n_copy; ++i) {
    HIP_CHECK(hipHostFree(h_ping[static_cast<size_t>(i)]));
    HIP_CHECK(hipHostFree(h_pong[static_cast<size_t>(i)]));
    HIP_CHECK(hipFree(d_copy[static_cast<size_t>(i)]));
  }
  HIP_CHECK(hipFree(d_comp_a));
  HIP_CHECK(hipFree(d_comp_b));
  for (int i = 0; i < n_compute; ++i) HIP_CHECK(hipStreamDestroy(s_comp[static_cast<size_t>(i)]));
  for (int i = 0; i < n_copy; ++i) HIP_CHECK(hipStreamDestroy(s_copy[static_cast<size_t>(i)]));
  if (rc == 0) std::printf("PASS\n");
  return rc;
}
