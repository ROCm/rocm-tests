# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)

# Shared ROCm path resolution and CMAKE_PREFIX_PATH setup.
if(NOT ROCM_TEST_CMAKE_MODULE_DIR)
  message(FATAL_ERROR "ROCM_TEST_CMAKE_MODULE_DIR is not set; build through the pytest builder fixture.")
endif()
list(APPEND CMAKE_MODULE_PATH "${ROCM_TEST_CMAKE_MODULE_DIR}")
include(rocm_setup)

if(NOT GPU_ARCH)
  message(FATAL_ERROR "GPU_ARCH is not set. Pass -DGPU_ARCH=<arch> (e.g. gfx942) to cmake.")
endif()

# Determine wave size: gfx11xx and later use wave32; everything else (gfx9xx,
# gfx10xx wave64) uses wave64. Must match the Makefile default (WAVE_SIZE=64)
# and the Makefile comment: "gfx9xx is what this reproducer is tuned for".
if(GPU_ARCH MATCHES "^gfx11")
  set(WAVE_SIZE 32)
else()
  set(WAVE_SIZE 64)
endif()

project(ipc_module_load_tests LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler detection: prefer what the conftest passes via -DCMAKE_CXX_COMPILER /
# -DCMAKE_HIP_COMPILER; fall back to hipcc when clang++ is not available.
if(NOT CMAKE_CXX_COMPILER MATCHES "hipcc")
  set(_HIPCC "${ROCM_PATH}/bin/hipcc")
  if(EXISTS "${_HIPCC}")
    set(CMAKE_CXX_COMPILER "${_HIPCC}" CACHE FILEPATH "HIP compiler" FORCE)
  endif()
endif()

find_package(hip CONFIG REQUIRED)

# ---- ipc_alltoall_module_load -----------------------------------------------
set_source_files_properties(ipc_alltoall_module_load.cpp PROPERTIES LANGUAGE HIP)
add_executable(ipc_alltoall_module_load ipc_alltoall_module_load.cpp)
target_compile_options(ipc_alltoall_module_load PRIVATE
    -O2
    -Wall
    -Wextra
    "--offload-arch=${GPU_ARCH}"
    -D__AMDGCN_WAVEFRONT_SIZE=${WAVE_SIZE}
)
target_link_libraries(ipc_alltoall_module_load PRIVATE hip::host)

# ---- ipc_dup_import_module_load ---------------------------------------------
set_source_files_properties(ipc_dup_import_module_load.cpp PROPERTIES LANGUAGE HIP)
add_executable(ipc_dup_import_module_load ipc_dup_import_module_load.cpp)
target_compile_options(ipc_dup_import_module_load PRIVATE
    -O2
    -Wall
    -Wextra
    "--offload-arch=${GPU_ARCH}"
    -D__AMDGCN_WAVEFRONT_SIZE=${WAVE_SIZE}
)
target_link_libraries(ipc_dup_import_module_load PRIVATE hip::host)

# ---- noop.hsaco (code object for hipModuleLoad target) ----------------------
# The C++ binaries open "noop.hsaco" as a relative path (CODE_OBJECT constant).
# The custom target compiles noop_kernel.cpp with --genco into the build dir
# so that noop.hsaco is co-located with the binaries at test runtime.
#
# hipcc is resolved through ROCM_PATH so we do not rely on PATH ordering.
if(NOT HIPCC_EXECUTABLE)
  find_program(HIPCC_EXECUTABLE hipcc
    HINTS "${ROCM_PATH}/bin"
    DOC "hipcc compiler wrapper"
  )
endif()
if(NOT HIPCC_EXECUTABLE)
  message(FATAL_ERROR "hipcc not found under ${ROCM_PATH}/bin; cannot build noop.hsaco")
endif()

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/noop.hsaco"
  COMMAND "${HIPCC_EXECUTABLE}"
          --genco
          "--offload-arch=${GPU_ARCH}"
          "-D__AMDGCN_WAVEFRONT_SIZE=${WAVE_SIZE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/noop_kernel.cpp"
          -o "${CMAKE_CURRENT_BINARY_DIR}/noop.hsaco"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/noop_kernel.cpp"
  COMMENT "Compiling noop.hsaco (genco) for ${GPU_ARCH}"
  VERBATIM
)

add_custom_target(noop_hsaco ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/noop.hsaco"
)

# Both IPC binaries require noop.hsaco to exist before they can run.
add_dependencies(ipc_alltoall_module_load noop_hsaco)
add_dependencies(ipc_dup_import_module_load noop_hsaco)
