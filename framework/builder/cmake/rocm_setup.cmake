# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Shared ROCm path and search-path setup for CMake-based E2E test builds.
# The pytest builder fixture passes this directory as ROCM_TEST_CMAKE_MODULE_DIR.

cmake_minimum_required(VERSION 3.21)

if(NOT ROCM_PATH)
  if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    set(ROCM_PATH "$ENV{ROCM_PATH}")
  else()
    set(ROCM_PATH "/opt/rocm")
  endif()
endif()

set(ENV{ROCM_PATH} "${ROCM_PATH}")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
list(APPEND CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake/hip")
set(HIP_PLATFORM "amd")
