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

# CMAKE_HIP_ARCHITECTURES must be set before project() for the CMake HIP language
# module to initialize correctly (sets CMAKE_HIP_LINK_EXECUTABLE and related vars).
set(CMAKE_HIP_ARCHITECTURES "${GPU_ARCH}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler detection: prefer what the conftest passes via -DCMAKE_CXX_COMPILER /
# -DCMAKE_HIP_COMPILER (as -D flags, those override CACHE values below);
# fall back to hipcc when clang++ is not available.
if(NOT CMAKE_CXX_COMPILER MATCHES "hipcc")
  set(_HIPCC "${ROCM_PATH}/bin/hipcc")
  if(EXISTS "${_HIPCC}")
    set(CMAKE_CXX_COMPILER "${_HIPCC}" CACHE FILEPATH "HIP compiler" FORCE)
  endif()
endif()

project(rocm_libs_tests LANGUAGES CXX HIP)

find_package(hip CONFIG REQUIRED)

find_package(hipsolver CONFIG QUIET PATHS "${ROCM_PATH}")
if(hipsolver_FOUND)
  set(HIPSOLVER_LIBS roc::hipsolver)
else()
  set(HIPSOLVER_LIBS hipsolver)
endif()

find_package(rocblas CONFIG QUIET PATHS "${ROCM_PATH}")
if(rocblas_FOUND)
  set(ROCBLAS_LIBS roc::rocblas)
else()
  set(ROCBLAS_LIBS rocblas)
endif()

# Binary 1: Small sliding contact TRSV (rocBLAS)
add_executable(small_sliding_contact small_sliding_contact.cpp)
target_compile_options(small_sliding_contact PRIVATE -O2 "--offload-arch=${GPU_ARCH}")
target_include_directories(small_sliding_contact PRIVATE "${ROCM_PATH}/include")
target_link_libraries(small_sliding_contact PRIVATE hip::host ${ROCBLAS_LIBS})

# Binary 2: Jacobian SVD multistream (hipSOLVER)
set_source_files_properties(jacobian_svd_multistream.cpp PROPERTIES LANGUAGE HIP)
add_executable(jacobian_svd_multistream jacobian_svd_multistream.cpp)
target_compile_options(jacobian_svd_multistream PRIVATE -O2 "--offload-arch=${GPU_ARCH}")
target_include_directories(jacobian_svd_multistream PRIVATE "${ROCM_PATH}/include")
target_link_libraries(jacobian_svd_multistream PRIVATE hip::host ${HIPSOLVER_LIBS})
