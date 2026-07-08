# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cmake_minimum_required(VERSION 3.21)

if(NOT ROCM_TEST_CMAKE_MODULE_DIR)
  message(FATAL_ERROR "ROCM_TEST_CMAKE_MODULE_DIR is not set; build through the pytest builder fixture.")
endif()
list(APPEND CMAKE_MODULE_PATH "${ROCM_TEST_CMAKE_MODULE_DIR}")
include(rocm_setup)

if(NOT GPU_ARCH)
  message(FATAL_ERROR "GPU_ARCH is not set. Pass -DGPU_ARCH=<arch> (e.g. gfx942) to cmake.")
endif()

# Set HIP architectures before project() so CMake's HIP language module uses
# the same offload target that pytest selected for this run.
set(CMAKE_HIP_ARCHITECTURES "${GPU_ARCH}")

project(hip_split_barrier_sample LANGUAGES CXX HIP)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS OFF)

set(CLR_INCLUDE_DIR "${ROCM_PATH}/include")
if(EXISTS "${CLR_INCLUDE_DIR}/hip/hip_cooperative_groups.h")
  message(STATUS "Found ROCm headers with split barrier support at: ${CLR_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "ROCm development headers not found at: ${CLR_INCLUDE_DIR}")
endif()

find_package(hip CONFIG REQUIRED)
find_package(rocsolver REQUIRED)
find_package(rocblas REQUIRED)

message(STATUS "Building for GPU architecture: ${GPU_ARCH}")

set(TARGET_NAME split_barrier_stress)
set(CPP_SOURCES split_barrier_stress.cpp)
set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE HIP)

add_executable(${TARGET_NAME} ${CPP_SOURCES})
target_compile_options(${TARGET_NAME} PRIVATE -Wno-shift-count-overflow)
target_include_directories(${TARGET_NAME} SYSTEM BEFORE PRIVATE "${CLR_INCLUDE_DIR}")
target_link_libraries(${TARGET_NAME} PRIVATE hip::host roc::rocsolver roc::rocblas)

if(UNIX)
  target_link_libraries(${TARGET_NAME} PRIVATE pthread)
endif()

# Register with CTest (optional)
if(COMMAND hip_sample_add_test)
    hip_sample_add_test(TARGET ${TARGET_NAME}
                       LABELS "sample" "barrier" "synchronization" "cooperative_groups")


    # Set timeout for samples
    if(BUILD_TESTS)
        set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 600)
    endif()
endif()
