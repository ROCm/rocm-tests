cmake_minimum_required(VERSION 3.21)

# Shared ROCm path resolution and CMAKE_PREFIX_PATH setup.
# Does NOT set compiler variables — the hipcc fallback below owns that.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../common/cmake")
include(rocm_setup)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler detection: prefer what the conftest passes via -DCMAKE_CXX_COMPILER /
# -DCMAKE_HIP_COMPILER; fall back to hipcc when clang++ is not available.
# The conftest injects these as -D flags which override CACHE values below.
if(NOT CMAKE_CXX_COMPILER MATCHES "hipcc")
  set(_HIPCC "${ROCM_PATH}/bin/hipcc")
  if(EXISTS "${_HIPCC}")
    set(CMAKE_CXX_COMPILER "${_HIPCC}" CACHE FILEPATH "HIP compiler" FORCE)
  endif()
endif()

# BUILD_HOST_ONLY_TESTS and BUILD_HIP_KERNEL_TESTS allow the two conftest
# fixtures to configure and build each subset independently.
option(BUILD_HOST_ONLY_TESTS  "Build host-only (driver-API) tests"  ON)
option(BUILD_HIP_KERNEL_TESTS "Build HIP-kernel (offload) tests"    ON)

# The HIP language is only needed when building kernel tests.
# Declaring it unconditionally causes cmake to require HIP_ARCHITECTURES even
# for the host-only target, which fails when GPU_ARCH is not supplied.
if(BUILD_HIP_KERNEL_TESTS)
  if(NOT GPU_ARCH)
    message(FATAL_ERROR "GPU_ARCH is not set. Pass -DGPU_ARCH=<arch> (e.g. gfx942) to cmake.")
  endif()
  project(hip_runtime_tests LANGUAGES CXX HIP)
else()
  project(hip_runtime_tests LANGUAGES CXX)
endif()

find_package(hip CONFIG REQUIRED)

# driver API only — no offload-arch needed (no GPU kernels)
if(BUILD_HOST_ONLY_TESTS)
  add_executable(hip_invalid_codeobject_load_test hip_invalid_codeobject_load_test.cpp)
  target_compile_options(hip_invalid_codeobject_load_test PRIVATE -O2)
  target_link_libraries(hip_invalid_codeobject_load_test PRIVATE hip::host)
endif()

# HIP kernels — needs offload-arch
if(BUILD_HIP_KERNEL_TESTS)
  set_source_files_properties(multi_stream_serialization.cpp PROPERTIES LANGUAGE HIP)
  add_executable(multi_stream_serialization multi_stream_serialization.cpp)
  target_compile_options(multi_stream_serialization PRIVATE -O3 "--offload-arch=${GPU_ARCH}")
  target_link_libraries(multi_stream_serialization PRIVATE hip::host)
endif()
