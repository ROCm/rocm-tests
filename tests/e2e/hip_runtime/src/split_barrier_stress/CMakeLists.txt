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

cmake_minimum_required(VERSION 3.16.8)

# Set ROCM path before everything
if(UNIX)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE STRING "ROCM Path")
    else()
      set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
    endif()
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
  list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/lib/cmake/hip)
endif()

# Set HIP platform
if(NOT DEFINED ENV{HIP_PLATFORM})
  set(ENV{HIP_PLATFORM} "amd")
endif()

project(hip_split_barrier_sample)

set(CMAKE_BUILD_TYPE Release)

# Use the ROCm development headers with split barrier support
set(CLR_INCLUDE_DIR "${ROCM_PATH}/include")
if(EXISTS "${CLR_INCLUDE_DIR}/hip/hip_cooperative_groups.h")
  message(STATUS "Found ROCm headers with split barrier support at: ${CLR_INCLUDE_DIR}")
  include_directories(BEFORE SYSTEM "${CLR_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "ROCm development headers not found at: ${CLR_INCLUDE_DIR}")
endif()

# Find hip
find_package(HIP REQUIRED)

# Find rocSOLVER and rocBLAS
find_package(rocsolver REQUIRED)
find_package(rocblas REQUIRED)

# Set GPU architecture (can be overridden via -DGPU_ARCH=...)
if(NOT DEFINED GPU_ARCH)
  set(GPU_ARCH "gfx90a" CACHE STRING "GPU architecture for compilation (e.g., gfx90a, gfx908, gfx1030)")
endif()

# Set CMAKE_HIP_ARCHITECTURES if not already defined
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES "${GPU_ARCH}" CACHE STRING "HIP architectures")
endif()

message(STATUS "Building for GPU architecture: ${GPU_ARCH}")

# Set build target
set(TARGET_NAME split_barrier_stress)

# Set source files
set(CPP_SOURCES split_barrier_stress.cpp)

# Use HIP_ADD_EXECUTABLE from FindHIP module
set_source_files_properties(${CPP_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

set(DEVICE_LIB_PATH "${ROCM_PATH}/lib/llvm/amdgcn/bitcode")

# Build first executable (split_barrier_stress)
HIP_ADD_EXECUTABLE(${TARGET_NAME} ${CPP_SOURCES}
  HIPCC_OPTIONS "--offload-arch=${GPU_ARCH};--rocm-device-lib-path=${DEVICE_LIB_PATH};-Wno-shift-count-overflow")

# Link with rocSOLVER and rocBLAS
target_link_libraries(${TARGET_NAME} roc::rocsolver roc::rocblas)

# Add pthread
if(UNIX)
    target_link_libraries(${TARGET_NAME} pthread)
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
