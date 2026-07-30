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
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

__global__ void stress_kernel(float *a, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = a[i];
  for (int k = 0; k < iters; ++k) x = fmaf(x, 1.0001f, 0.0f);
  a[i] = x;
}

static long parse_vm_rss_kb() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      std::istringstream iss(line);
      std::string tag;
      long kb = 0;
      std::string unit;
      iss >> tag >> kb >> unit;
      return kb;
    }
  }
  return -1;
}

int main(int argc, char **argv) {
  if (arg_flag(argc, argv, "--help") || arg_flag(argc, argv, "-h")) {
    std::fprintf(stderr, "Usage: %s --duration=SECS [--phase-sec=600] [--elems=E]\n", argv[0]);
    print_usage_tail(argv[0]);
    return 0;
  }
  const int duration = arg_int(argc, argv, "--duration", 3600);
  const int phase_sec = arg_int(argc, argv, "--phase-sec", 600);
  const int elems = arg_int(argc, argv, "--elems", 1 << 19);

  HIP_CHECK(hipSetDevice(0));
  print_env_header("hwq_robustness");

  const long rss0 = parse_vm_rss_kb();

  const int threads = 256;
  const int blocks = (elems + threads - 1) / threads;
  const int stream_counts[] = {1, 4, 8, 16};
  const int n_phases = static_cast<int>(sizeof(stream_counts) / sizeof(stream_counts[0]));
  float *d_a = nullptr;
  const size_t buf_bytes = sizeof(float) * static_cast<size_t>(elems);
  HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&d_a), buf_bytes));
  HIP_CHECK(hipMemset(d_a, 0, buf_bytes));

  const double t_start = now_sec();
  const double t_end = t_start + duration;
  double next_phase = t_start + phase_sec;
  int phase_idx = 0;
  int inner = 32;

  std::printf("starting stress loop: duration=%ds phases=%d stream_counts=1,4,8,16\n", duration, n_phases);
  std::fflush(stdout);

  while (now_sec() < t_end) {
    if (now_sec() >= next_phase) {
      next_phase += phase_sec;
      phase_idx = (phase_idx + 1) % n_phases;
      std::printf("phase switch: stream_count=%d elapsed_sec=%.0f\n", stream_counts[phase_idx], now_sec() - t_start);
      std::fflush(stdout);
    }
    const int ns = stream_counts[phase_idx];
    std::vector<hipStream_t> st(static_cast<size_t>(ns));
    for (int i = 0; i < ns; ++i) HIP_CHECK(hipStreamCreate(&st[static_cast<size_t>(i)]));
    for (int r = 0; r < 4; ++r) {
      for (int s = 0; s < ns; ++s)
        hipLaunchKernelGGL(stress_kernel, dim3(blocks), dim3(threads), 0, st[static_cast<size_t>(s)], d_a, elems,
                           inner);
    }
    for (int s = 0; s < ns; ++s) HIP_CHECK(hipStreamSynchronize(st[static_cast<size_t>(s)]));
    for (int s = 0; s < ns; ++s) HIP_CHECK(hipStreamDestroy(st[static_cast<size_t>(s)]));
  }

  const long rss1 = parse_vm_rss_kb();
  if (rss0 > 0 && rss1 > 0) {
    const long diff = rss1 > rss0 ? rss1 - rss0 : rss0 - rss1;
    const double drift = static_cast<double>(diff) / static_cast<double>(rss0);
    std::printf("VmRSS_kb_start=%ld end=%ld drift_ratio=%.4f\n", rss0, rss1, drift);
    std::fflush(stdout);
    if (drift > 0.05) std::fprintf(stderr, "WARN: VmRSS drift > 5%%\n");
  }

  HIP_CHECK(hipFree(d_a));
  std::printf("PASS\n");
  std::fflush(stdout);
  return 0;
}
