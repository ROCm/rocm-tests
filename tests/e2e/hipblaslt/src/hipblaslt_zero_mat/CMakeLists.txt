# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)

if(NOT ROCM_TEST_CMAKE_MODULE_DIR)
  message(FATAL_ERROR "ROCM_TEST_CMAKE_MODULE_DIR is not set; build through the pytest builder fixture.")
endif()
list(APPEND CMAKE_MODULE_PATH "${ROCM_TEST_CMAKE_MODULE_DIR}")
include(rocm_setup)

if(NOT GPU_ARCH)
  message(FATAL_ERROR "GPU_ARCH is not set. Pass -DGPU_ARCH=<arch> (e.g. gfx942) to cmake.")
endif()

set(CMAKE_HIP_ARCHITECTURES "${GPU_ARCH}")

if(NOT DEFINED CMAKE_HIP_COMPILER)
  find_program(_AMDCLANGPP NAMES amdclang++
    HINTS "${ROCM_PATH}/bin" "${ROCM_PATH}/llvm/bin" "${ROCM_PATH}/lib/llvm/bin"
    NO_DEFAULT_PATH)
  if(_AMDCLANGPP)
    set(CMAKE_HIP_COMPILER "${_AMDCLANGPP}" CACHE FILEPATH "HIP compiler")
  endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(hipblaslt_zero_mat LANGUAGES CXX HIP)

find_package(hip CONFIG REQUIRED)

find_package(hipblaslt CONFIG QUIET PATHS "${ROCM_PATH}")
if(hipblaslt_FOUND)
  set(HIPBLASLT_LIBS roc::hipblaslt)
else()
  set(HIPBLASLT_LIBS hipblaslt)
  link_directories("${ROCM_PATH}/lib")
endif()

add_executable(hipblaslt_zero_mat hipblaslt_zero_mat.hip)
target_compile_options(hipblaslt_zero_mat PRIVATE -O2)
target_link_libraries(hipblaslt_zero_mat PRIVATE hip::host ${HIPBLASLT_LIBS})
target_include_directories(hipblaslt_zero_mat PRIVATE "${ROCM_PATH}/include")
