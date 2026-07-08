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

#ifndef _TEST_COMMON_HPP
#define _TEST_COMMON_HPP


#include <hip/hip_runtime.h>

#include <rocrand/rocrand.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>
#include <rocprim/rocprim.hpp>

// ============================================================================
// ERROR CHECKING MACROS
// ============================================================================

#define HIP_CHECK(condition)                                                   \
  {                                                                            \
    hipError_t error = condition;                                              \
    if (error != hipSuccess) {                                                 \
      std::cerr << "HIP error: " << hipGetErrorString(error) << " at "         \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define ROCRAND_CHECK(condition)                                               \
  {                                                                            \
    rocrand_status status = condition;                                         \
    if (status != ROCRAND_STATUS_SUCCESS) {                                    \
      std::cerr << "rocRAND error: " << status << " at " << __FILE__ << ":"    \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// ============================================================================
// ROCRAND UTILITIES
// ============================================================================

namespace test_utils {

// rocRAND generator wrapper for RAII
class RocrandGenerator {
 private:
  rocrand_generator gen_;
  bool initialized_;

 public:
  RocrandGenerator(rocrand_rng_type type = ROCRAND_RNG_PSEUDO_DEFAULT,
                   uint64_t seed = 12345ULL)
      : initialized_(false) {
    ROCRAND_CHECK(rocrand_create_generator(&gen_, type));
    ROCRAND_CHECK(rocrand_set_seed(gen_, seed));
    initialized_ = true;
  }

  ~RocrandGenerator() {
    if (initialized_) {
      rocrand_destroy_generator(gen_);
    }
  }

  rocrand_generator get() { return gen_; }

  // Delete copy constructor and assignment
  RocrandGenerator(const RocrandGenerator &) = delete;
  RocrandGenerator &operator=(const RocrandGenerator &) = delete;
};

// ============================================================================
// RANDOM DATA GENERATION (GPU-based with rocRAND)
// ============================================================================

// Generate random unsigned integers (32-bit)
inline std::vector<unsigned int>
generate_random_data_uint(size_t size, unsigned int min_val = 0,
                          unsigned int max_val = UINT_MAX) {
  RocrandGenerator gen;

  // Allocate device memory
  unsigned int *d_random;
  HIP_CHECK(hipMalloc(&d_random, size * sizeof(unsigned int)));

  // Generate on device
  ROCRAND_CHECK(rocrand_generate(gen.get(), d_random, size));

  // Copy to host
  std::vector<unsigned int> random_values(size);
  HIP_CHECK(hipMemcpy(random_values.data(), d_random,
                      size * sizeof(unsigned int), hipMemcpyDeviceToHost));

  // Scale to range if needed
  std::vector<unsigned int> result(size);
  if (min_val == 0 && max_val == UINT_MAX) {
    result = random_values;
  } else {
    unsigned int range = max_val - min_val + 1;
    for (size_t i = 0; i < size; i++) {
      result[i] = min_val + (random_values[i] % range);
    }
  }

  HIP_CHECK(hipFree(d_random));
  return result;
}

// Generate random integers (signed 32-bit)
inline std::vector<int> generate_random_data_int(size_t size, int min_val = 0,
                                                 int max_val = 1000) {
  auto uints = generate_random_data_uint(size, 0, max_val - min_val);
  std::vector<int> result(size);
  for (size_t i = 0; i < size; i++) {
    result[i] = min_val + static_cast<int>(uints[i]);
  }
  return result;
}

// Generate random floats [0.0, 1.0]
inline std::vector<float> generate_random_data_float(size_t size,
                                                     float min_val = 0.0f,
                                                     float max_val = 1.0f) {
  RocrandGenerator gen;

  float *d_random;
  HIP_CHECK(hipMalloc(&d_random, size * sizeof(float)));

  // Generate uniform [0, 1]
  ROCRAND_CHECK(rocrand_generate_uniform(gen.get(), d_random, size));

  std::vector<float> random_values(size);
  HIP_CHECK(hipMemcpy(random_values.data(), d_random, size * sizeof(float),
                      hipMemcpyDeviceToHost));

  // Scale to range
  std::vector<float> result(size);
  float range = max_val - min_val;
  for (size_t i = 0; i < size; i++) {
    result[i] = min_val + random_values[i] * range;
  }

  HIP_CHECK(hipFree(d_random));
  return result;
}

// Generate random doubles [0.0, 1.0]
inline std::vector<double> generate_random_data_double(size_t size,
                                                       double min_val = 0.0,
                                                       double max_val = 1.0) {
  RocrandGenerator gen;

  double *d_random;
  HIP_CHECK(hipMalloc(&d_random, size * sizeof(double)));

  // Generate uniform [0, 1]
  ROCRAND_CHECK(rocrand_generate_uniform_double(gen.get(), d_random, size));

  std::vector<double> random_values(size);
  HIP_CHECK(hipMemcpy(random_values.data(), d_random, size * sizeof(double),
                      hipMemcpyDeviceToHost));

  // Scale to range
  std::vector<double> result(size);
  double range = max_val - min_val;
  for (size_t i = 0; i < size; i++) {
    result[i] = min_val + random_values[i] * range;
  }

  HIP_CHECK(hipFree(d_random));
  return result;
}

// Generate random normal distribution (float)
inline std::vector<float> generate_random_data_normal(size_t size,
                                                      float mean = 0.0f,
                                                      float stddev = 1.0f) {
  RocrandGenerator gen;

  float *d_random;
  HIP_CHECK(hipMalloc(&d_random, size * sizeof(float)));

  // Generate normal distribution
  ROCRAND_CHECK(
      rocrand_generate_normal(gen.get(), d_random, size, mean, stddev));

  std::vector<float> result(size);
  HIP_CHECK(hipMemcpy(result.data(), d_random, size * sizeof(float),
                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(d_random));
  return result;
}

// Generic template dispatching
template <typename T>
typename std::enable_if<std::is_same<T, int>::value, std::vector<T>>::type
generate_random_data(size_t size, T min_val = 0, T max_val = 1000) {
  return generate_random_data_int(size, min_val, max_val);
}

template <typename T>
typename std::enable_if<std::is_same<T, unsigned int>::value,
                        std::vector<T>>::type
generate_random_data(size_t size, T min_val = 0, T max_val = 1000) {
  return generate_random_data_uint(size, min_val, max_val);
}

template <typename T>
typename std::enable_if<std::is_same<T, float>::value, std::vector<T>>::type
generate_random_data(size_t size, T min_val = 0.0f, T max_val = 1.0f) {
  return generate_random_data_float(size, min_val, max_val);
}

template <typename T>
typename std::enable_if<std::is_same<T, double>::value, std::vector<T>>::type
generate_random_data(size_t size, T min_val = 0.0, T max_val = 1.0) {
  return generate_random_data_double(size, min_val, max_val);
}

// For other integer types (int8_t, int16_t, int64_t, etc.)
template <typename T>
typename std::enable_if<std::is_integral<T>::value &&
                            !std::is_same<T, int>::value &&
                            !std::is_same<T, unsigned int>::value,
                        std::vector<T>>::type
generate_random_data(size_t size, T min_val, T max_val) {
  // Generate as int, then convert
  auto ints = generate_random_data_int(size, static_cast<int>(min_val),
                                       static_cast<int>(max_val));
  std::vector<T> result(size);
  for (size_t i = 0; i < size; i++) {
    result[i] = static_cast<T>(ints[i]);
  }
  return result;
}

}  // namespace test_utils

#endif  // _TEST_COMMON_HPP
