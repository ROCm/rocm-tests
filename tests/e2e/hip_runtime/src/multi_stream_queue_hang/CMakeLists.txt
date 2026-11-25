# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)

if(NOT ROCM_TEST_CMAKE_MODULE_DIR)
  message(FATAL_ERROR "ROCM_TEST_CMAKE_MODULE_DIR is not set; build through the pytest builder fixture.")
endif()
list(APPEND CMAKE_MODULE_PATH "${ROCM_TEST_CMAKE_MODULE_DIR}")
include(rocm_setup)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_CXX_COMPILER MATCHES "hipcc")
  set(_HIPCC "${ROCM_PATH}/bin/hipcc")
  if(EXISTS "${_HIPCC}")
    set(CMAKE_CXX_COMPILER "${_HIPCC}" CACHE FILEPATH "HIP compiler" FORCE)
  endif()
endif()

project(multi_stream_queue_hang LANGUAGES CXX)

find_package(hip CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_executable(multi_stream_queue_hang multi_stream_queue_hang.cpp)
target_compile_options(multi_stream_queue_hang PRIVATE -O2 -Wall)
target_link_libraries(multi_stream_queue_hang PRIVATE hip::host Threads::Threads)
