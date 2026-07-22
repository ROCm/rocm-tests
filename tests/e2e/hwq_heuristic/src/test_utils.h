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
#ifndef HWQ_TEST_UTILS_H
#define HWQ_TEST_UTILS_H

#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

inline void hip_die(hipError_t e, const char *file, int line) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "HIP error %d (%s) at %s:%d\n", static_cast<int>(e),
                 hipGetErrorString(e), file, line);
    std::exit(1);
  }
}

#define HIP_CHECK(x) hip_die((x), __FILE__, __LINE__)

inline double now_sec() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

inline int getenv_int(const char *name, int def) {
  const char *v = std::getenv(name);
  if (!v || !*v) return def;
  return std::atoi(v);
}

inline bool arg_eq(const char *a, const char *b) { return std::strcmp(a, b) == 0; }

inline const char *arg_val(int argc, char **argv, const char *key, const char *def) {
  const size_t n = std::strlen(key);
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], key, n) == 0 && argv[i][n] == '=') return argv[i] + n + 1;
  }
  return def;
}

inline int arg_int(int argc, char **argv, const char *key, int def) {
  const char *v = arg_val(argc, argv, key, nullptr);
  return v ? std::atoi(v) : def;
}

inline bool arg_flag(int argc, char **argv, const char *flag) {
  for (int i = 1; i < argc; ++i)
    if (arg_eq(argv[i], flag)) return true;
  return false;
}

inline void print_usage_tail(const char *exe) {
  std::fprintf(stderr, "Set DEBUG_HIP_DYNAMIC_QUEUES=0|1|2 in the environment before running.\n");
  std::fprintf(stderr, "Example: DEBUG_HIP_DYNAMIC_QUEUES=2 %s ...\n", exe);
}

inline void print_env_header(const char *test_name) {
  hipDeviceProp_t prop{};
  int dev = -1;
  if (hipGetDevice(&dev) == hipSuccess && hipGetDeviceProperties(&prop, dev) == hipSuccess) {
    std::printf("[%s] device=%d name=%s arch=%s\n", test_name, dev, prop.name, prop.gcnArchName);
  }
  int rt = 0;
  (void)hipRuntimeGetVersion(&rt);
  std::printf("[%s] hip_runtime_version=%d DEBUG_HIP_DYNAMIC_QUEUES=%d\n", test_name,
              rt, getenv_int("DEBUG_HIP_DYNAMIC_QUEUES", -1));
}

#endif
