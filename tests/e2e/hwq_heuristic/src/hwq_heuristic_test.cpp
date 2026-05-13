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
#include <vector>

__global__ void short_kernel(int *flag) {
  if (threadIdx.x == 0 && blockIdx.x == 0) *flag = 1;
}

__global__ void long_kernel(float *a, const float *b, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = a[i];
  for (int k = 0; k < iters; ++k) x = fmaf(x, b[i], 1.0f);
  a[i] = x;
}

static int run_scenario_a() {
  std::printf("scenario_a: balanced load, 4 streams, round-robin expectation\n");
  const int n = 4;
  std::vector<hipStream_t> st(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) HIP_CHECK(hipStreamCreate(&st[static_cast<size_t>(i)]));
  int *d_flag = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_flag), sizeof(int)));
  HIP_CHECK(hipMemset(d_flag, 0, sizeof(int)));
  for (int i = 0; i < 64; ++i) {
    for (int s = 0; s < n; ++s)
      hipLaunchKernelGGL(short_kernel, dim3(1), dim3(32), 0, st[static_cast<size_t>(s)], d_flag);
  }
  for (int s = 0; s < n; ++s) HIP_CHECK(hipStreamSynchronize(st[static_cast<size_t>(s)]));
  HIP_CHECK(hipFree(d_flag));
  for (int s = 0; s < n; ++s) HIP_CHECK(hipStreamDestroy(st[static_cast<size_t>(s)]));
  return 0;
}

static int run_scenario_b() {
  std::printf("scenario_b: imbalanced load, stream 0 short, streams 1-3 long\n");
  const int n = 4;
  std::vector<hipStream_t> st(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) HIP_CHECK(hipStreamCreate(&st[static_cast<size_t>(i)]));
  int *d_flag = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_flag), sizeof(int)));
  int elems = 1 << 18;
  float *d_a, *d_b;
  const size_t buf_bytes = sizeof(float) * static_cast<size_t>(elems);
  HIP_CHECK(hipMalloc(&d_a, buf_bytes));
  HIP_CHECK(hipMalloc(&d_b, buf_bytes));
  std::vector<float> h(static_cast<size_t>(elems), 1.0f);
  HIP_CHECK(hipMemcpy(d_b, h.data(), buf_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_a, h.data(), buf_bytes, hipMemcpyHostToDevice));
  const int threads = 256;
  const int blocks = (elems + threads - 1) / threads;
  for (int k = 0; k < 1000; ++k)
    hipLaunchKernelGGL(short_kernel, dim3(1), dim3(32), 0, st[0], d_flag);
  for (int s = 1; s < n; ++s) {
    for (int k = 0; k < 10; ++k)
      hipLaunchKernelGGL(long_kernel, dim3(blocks), dim3(threads), 0, st[static_cast<size_t>(s)], d_a, d_b, elems,
                         256);
  }
  for (int s = 0; s < n; ++s) HIP_CHECK(hipStreamSynchronize(st[static_cast<size_t>(s)]));
  HIP_CHECK(hipFree(d_flag));
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  for (int s = 0; s < n; ++s) HIP_CHECK(hipStreamDestroy(st[static_cast<size_t>(s)]));
  return 0;
}

static int run_scenario_c() {
  std::printf("scenario_c: sticky queue (submit -> sync -> submit, expect consistent timing)\n");
  hipStream_t st;
  HIP_CHECK(hipStreamCreate(&st));
  int *d_flag = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_flag), sizeof(int)));
  double t1 = 0, t2 = 0;
  for (int phase = 0; phase < 2; ++phase) {
    const double t0 = now_sec();
    for (int k = 0; k < 200; ++k) hipLaunchKernelGGL(short_kernel, dim3(1), dim3(32), 0, st, d_flag);
    HIP_CHECK(hipStreamSynchronize(st));
    const double dt = now_sec() - t0;
    if (phase == 0) t1 = dt;
    else t2 = dt;
  }
  const double ratio = t1 > 0 ? t2 / t1 : 1.0;
  std::printf("scenario_c_phase0_sec=%.6f phase1_sec=%.6f ratio=%.3f\n", t1, t2, ratio);
  HIP_CHECK(hipFree(d_flag));
  HIP_CHECK(hipStreamDestroy(st));
  if (ratio > 2.0) {
    std::fprintf(stderr, "WARN: phase1/phase0 ratio %.2f > 2.0 (possible queue reassignment)\n", ratio);
  }
  return 0;
}

static int run_scenario_d() {
  std::printf("scenario_d: null stream + 8 explicit streams concurrent\n");
  const int n = 8;
  std::vector<hipStream_t> st(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) HIP_CHECK(hipStreamCreate(&st[static_cast<size_t>(i)]));
  int *d_flag = nullptr;
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_flag), sizeof(int)));
  for (int k = 0; k < 32; ++k) {
    hipLaunchKernelGGL(short_kernel, dim3(1), dim3(32), 0, nullptr, d_flag);
    for (int s = 0; s < n; ++s)
      hipLaunchKernelGGL(short_kernel, dim3(1), dim3(32), 0, st[static_cast<size_t>(s)], d_flag);
  }
  HIP_CHECK(hipDeviceSynchronize());
  for (int s = 0; s < n; ++s) HIP_CHECK(hipStreamSynchronize(st[static_cast<size_t>(s)]));
  HIP_CHECK(hipFree(d_flag));
  for (int s = 0; s < n; ++s) HIP_CHECK(hipStreamDestroy(st[static_cast<size_t>(s)]));
  return 0;
}

int main(int argc, char **argv) {
  if (arg_flag(argc, argv, "--help") || arg_flag(argc, argv, "-h")) {
    std::fprintf(stderr, "Usage: %s --scenario=A|B|C|D|all\n", argv[0]);
    print_usage_tail(argv[0]);
    return 0;
  }
  const char *sc = arg_val(argc, argv, "--scenario", "A");
  HIP_CHECK(hipSetDevice(0));
  print_env_header("hwq_heuristic_test");

  bool run_all = (arg_eq(sc, "all") || arg_eq(sc, "ALL"));

  struct {
    char id;
    int (*fn)();
  } scenarios[] = {{'A', run_scenario_a}, {'B', run_scenario_b}, {'C', run_scenario_c}, {'D', run_scenario_d}};

  int rc = 0;
  for (auto &s : scenarios) {
    if (run_all || sc[0] == s.id || sc[0] == (s.id + 32)) {
      int r = s.fn();
      if (r != 0) {
        std::fprintf(stderr, "FAIL scenario %c\n", s.id);
        rc = r;
      } else {
        std::printf("PASS scenario %c\n", s.id);
      }
    }
  }

  if (!run_all && rc == 0) {
    bool found = false;
    for (auto &s : scenarios)
      if (sc[0] == s.id || sc[0] == (s.id + 32)) found = true;
    if (!found) {
      std::fprintf(stderr, "Unknown scenario '%s'\n", sc);
      return 2;
    }
  }
  return rc;
}
