# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)

# Shared ROCm path resolution and CMAKE_PREFIX_PATH setup.
# Does NOT set compiler variables — the hipcc fallback below owns that.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../common/cmake")
include(rocm_setup)

if(NOT GPU_ARCH)
  message(FATAL_ERROR "GPU_ARCH is not set. Pass -DGPU_ARCH=<arch> (e.g. gfx942) to cmake.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler detection: prefer what the conftest passes via -DCMAKE_CXX_COMPILER;
# fall back to hipcc when clang++ is not available. The conftest may inject
# -DCMAKE_CXX_COMPILER=<clangpp> which, as a -D flag, overrides CACHE values here.
if(NOT CMAKE_CXX_COMPILER MATCHES "hipcc")
  set(_HIPCC "${ROCM_PATH}/bin/hipcc")
  if(EXISTS "${_HIPCC}")
    set(CMAKE_CXX_COMPILER "${_HIPCC}" CACHE FILEPATH "HIP compiler" FORCE)
  endif()
endif()

project(hwq_heuristic_test LANGUAGES CXX)

find_package(hip CONFIG REQUIRED)

add_executable(hwq_heuristic_test hwq_heuristic_test.cpp)
target_compile_options(hwq_heuristic_test PRIVATE -O3 "--offload-arch=${GPU_ARCH}")
target_link_libraries(hwq_heuristic_test PRIVATE hip::host)

add_executable(hwq_null_stream_protection_regr hwq_null_stream_protection_regr.cpp)
target_compile_options(hwq_null_stream_protection_regr PRIVATE -O3 "--offload-arch=${GPU_ARCH}")
target_link_libraries(hwq_null_stream_protection_regr PRIVATE hip::host)

add_executable(hwq_compute_copy_overlap_test hwq_compute_copy_overlap_test.cpp)
target_compile_options(hwq_compute_copy_overlap_test PRIVATE -O3 "--offload-arch=${GPU_ARCH}")
target_link_libraries(hwq_compute_copy_overlap_test PRIVATE hip::host)
