# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Standalone CMake project for the rock_mps_test multi-process integration
# binary.  Built in its own subdirectory by the ``_rock_mps_build_dir`` fixture 
# in tests/e2e/hip_runtime/conftest.py.
#
# All translation units are compiled as HIP (LANGUAGE HIP) because several
# roles launch device kernels defined in src/kernels/kernels.cpp across
# translation-unit boundaries — the launch sites must be compiled by the HIP
# compiler.  Optional roles (LIBRARY/COMPILER/MONITOR) are enabled only when
# hipBLASLt / hipRTC / amd_smi are present; otherwise those roles exit 0 with a
# "not available — skipping" message at runtime.

cmake_minimum_required(VERSION 3.21)

if(NOT ROCM_TEST_CMAKE_MODULE_DIR)
  message(FATAL_ERROR "ROCM_TEST_CMAKE_MODULE_DIR is not set; build through the pytest builder fixture.")
endif()
list(APPEND CMAKE_MODULE_PATH "${ROCM_TEST_CMAKE_MODULE_DIR}")
include(rocm_setup)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT GPU_ARCH)
  message(FATAL_ERROR "GPU_ARCH is not set. Pass -DGPU_ARCH=<arch> (e.g. gfx942) to cmake.")
endif()

project(rock_mps_test LANGUAGES CXX HIP)

find_package(hip CONFIG REQUIRED)

# hipBLASLt (optional — LIBRARY role)
find_library(HIPBLASLT_LIB hipblaslt PATHS "${ROCM_PATH}/lib" NO_DEFAULT_PATH)
find_path(HIPBLASLT_INCLUDE hipblaslt/hipblaslt.h PATHS "${ROCM_PATH}/include" NO_DEFAULT_PATH)
if(HIPBLASLT_LIB AND HIPBLASLT_INCLUDE)
  message(STATUS "hipBLASLt found — LIBRARY role enabled")
else()
  message(STATUS "hipBLASLt not found — LIBRARY role disabled")
endif()

# hipRTC (optional — COMPILER role)
find_library(HIPRTC_LIB hiprtc PATHS "${ROCM_PATH}/lib" NO_DEFAULT_PATH)
if(HIPRTC_LIB)
  message(STATUS "hipRTC found — COMPILER role enabled")
else()
  message(STATUS "hipRTC not found — COMPILER role disabled")
endif()

# AMD SMI (optional — MONITOR role)
find_library(AMDSMI_LIB amd_smi PATHS "${ROCM_PATH}/lib" NO_DEFAULT_PATH)
find_path(AMDSMI_INCLUDE amd_smi/amdsmi.h PATHS "${ROCM_PATH}/include" NO_DEFAULT_PATH)
if(AMDSMI_LIB AND AMDSMI_INCLUDE)
  message(STATUS "AMD SMI found — MONITOR role enabled")
else()
  message(STATUS "AMD SMI not found — MONITOR role disabled")
endif()

set(SOURCES
  src/main.cpp
  src/common/validator.cpp
  src/common/health.cpp
  src/common/ipc_channel.cpp
  src/kernels/kernels.cpp
  src/roles/compute.cpp
  src/roles/memory_mover.cpp
  src/roles/library.cpp
  src/roles/compiler.cpp
  src/roles/monitor.cpp
  src/roles/profiler.cpp
  src/roles/ipc_xfer.cpp
)

# Cross-TU kernel launches require every source to be built by the HIP compiler.
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE HIP)

add_executable(rock_mps_test ${SOURCES})

target_compile_options(rock_mps_test PRIVATE
  -O2 -Wall -Wextra -Wno-unused-result "--offload-arch=${GPU_ARCH}")

target_include_directories(rock_mps_test PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${ROCM_PATH}/include
)

target_link_libraries(rock_mps_test PRIVATE hip::host)

if(HIPBLASLT_LIB AND HIPBLASLT_INCLUDE)
  target_compile_definitions(rock_mps_test PRIVATE HAS_HIPBLASLT)
  target_include_directories(rock_mps_test PRIVATE ${HIPBLASLT_INCLUDE})
  target_link_libraries(rock_mps_test PRIVATE ${HIPBLASLT_LIB})
endif()

if(HIPRTC_LIB)
  target_compile_definitions(rock_mps_test PRIVATE HAS_HIPRTC)
  target_link_libraries(rock_mps_test PRIVATE ${HIPRTC_LIB})
endif()

if(AMDSMI_LIB AND AMDSMI_INCLUDE)
  target_compile_definitions(rock_mps_test PRIVATE HAS_AMDSMI)
  target_include_directories(rock_mps_test PRIVATE ${AMDSMI_INCLUDE})
  target_link_libraries(rock_mps_test PRIVATE ${AMDSMI_LIB})
endif()

if(UNIX)
  target_link_libraries(rock_mps_test PRIVATE pthread rt)
endif()
