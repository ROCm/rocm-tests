// MIT License
// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Main entry point for rocPRIM Standalone Test Suite
// Single executable with Google Test framework

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <iostream>

#include <cstdlib>
#include <cstring>
#include <string>

// Default endpoint for LinearScaling_DataSize
size_t g_linear_max_elems = 10'000'000ULL;

static void remove_arg(int *argc, char **argv, int idx) {
  for (int i = idx; i + 1 < *argc; ++i) {
    argv[i] = argv[i + 1];
  }
  --(*argc);
}

// Supports:
//   --linear_size=10000000
//   --linear_size 10000000
static void parse_linear_size_flag(int *argc, char **argv) {
  int &argc_ref = *argc;  // keeps body changes minimal

  for (int i = 1; i < argc_ref; ++i) {
    if (std::strncmp(argv[i], "--linear_size", sizeof("--linear_size") - 1) !=
        0) {
      continue;
    }
    std::string value;
    const char *eq = std::strchr(argv[i], '=');

    if (eq) {
      value = std::string(eq + 1);
      remove_arg(argc, argv, i);
    } else {
      if (i + 1 >= argc_ref) {
        remove_arg(argc, argv, i);
        return;
      }
      value = std::string(argv[i + 1]);
      remove_arg(argc, argv, i + 1);
      remove_arg(argc, argv, i);
    }

    char *end = nullptr;
    uint64_t v = std::strtoull(value.c_str(), &end, 10);
    if (end != value.c_str() && v > 0)
      g_linear_max_elems = static_cast<size_t>(v);

    return;
  }
}

int main(int argc, char **argv) {
  // Parse ONLY this custom flag and remove it so gtest doesn't see unknown args
  parse_linear_size_flag(&argc, argv);
  // Initialize Google Test
  ::testing::InitGoogleTest(&argc, argv);

  // Check for HIP devices
  int device_count = 0;
  hipError_t error = hipGetDeviceCount(&device_count);

  if (error != hipSuccess || device_count == 0) {
    std::cerr << "\n❌ Error: No HIP devices found" << std::endl;
    std::cerr << "HIP Error: " << hipGetErrorString(error) << std::endl;
    return 1;
  }

  // Print test suite header
  std::cout << "\n========================================" << std::endl;
  std::cout << "rocPRIM Standalone Test Suite" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "HIP Devices: " << device_count << std::endl;

  // Print device information
  for (int i = 0; i < device_count; i++) {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, i);
    std::cout << "  Device " << i << ": " << prop.name << " (Compute "
              << prop.major << "." << prop.minor << ")" << std::endl;
  }

  std::cout << "========================================" << std::endl;

  // Print usage information
  std::cout << "\nUsage Examples:" << std::endl;
  std::cout << "  Run all tests:           " << argv[0] << std::endl;
  std::cout << "  Run specific suite:      " << argv[0]
            << " --gtest_filter=SystemMultiStreamTests.*" << std::endl;
  std::cout << "  Run specific test:       " << argv[0]
            << " --gtest_filter=MultiGPUHMMTests.*" << std::endl;
  std::cout << "  List all tests:          " << argv[0] << " --gtest_list_tests"
            << std::endl;
  std::cout << "  Run with repeat:         " << argv[0] << " --gtest_repeat=10"
            << std::endl;
  std::cout << "========================================\n" << std::endl;

  // Run all tests
  int result = RUN_ALL_TESTS();

  std::cout << "\n========================================" << std::endl;
  if (result == 0) {
    std::cout << "✅ All tests passed!" << std::endl;
  } else {
    std::cout << "❌ Some tests failed" << std::endl;
  }
  std::cout << "========================================\n" << std::endl;

  return result;
}
