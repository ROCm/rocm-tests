# MIT License
# Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# rocPRIM Standalone Test Suite
# This CMakeLists.txt includes only existing tests

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# Shared ROCm path resolution and CMAKE_PREFIX_PATH setup.
# Does NOT set compiler variables — the compiler is passed via -DCMAKE_CXX_COMPILER
# from the conftest (mandatory; conftest calls pytest.fail if clang++ is absent).
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../common/cmake")
include(rocm_setup)

project(rocprim_tests
    VERSION 1.0.0
    DESCRIPTION "rocPRIM System-Level Tests"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(MEMORYCHECK_COMMAND valgrind)
set(MEMORYCHECK_COMMAND_OPTIONS
  "--leak-check=full --show-leak-kinds=all --track-origins=yes --error-exitcode=1"
)
# ============================================================================
# DEPENDENCIES
# ============================================================================

# Module-mode find_package(HIP) — matches AIGQA-Tests/rocPRIM_Tests original.
# Avoids the config-mode chain (hip-config-amd.cmake → find_dependency(AMDDeviceLibs))
# which fails when --blas installs a broken AMDDeviceLibsConfig.cmake forwarding stub.
find_package(HIP REQUIRED)
message(STATUS "Found HIP: ${HIP_VERSION}")

# Find rocPRIM
find_package(rocprim REQUIRED)
message(STATUS "Found rocPRIM")

# Find rocRAND
find_package(rocrand REQUIRED)
message(STATUS "Found rocRAND")

# Google Test
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    message(STATUS "GTest not found, downloading...")
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    add_library(GTest::GTest ALIAS gtest)
    add_library(GTest::Main ALIAS gtest_main)
    message(STATUS "GTest downloaded and configured")
endif()

# ============================================================================
# GPU ARCHITECTURE
# ============================================================================

# GPU_ARCH must be passed from the conftest via -DGPU_ARCH=<arch>.
# Auto-detection via rocminfo at configure time is fragile (requires GPU hardware
# at build time, conflicts with xdist parallelism) and is intentionally removed.
# The conftest (rocprim/conftest.py) passes the value from --gpu-arch; if absent,
# cmake errors here rather than silently compiling for the wrong architecture.
if(NOT GPU_ARCH)
  message(FATAL_ERROR "GPU_ARCH is not set. Pass -DGPU_ARCH=<arch> (e.g. gfx942) to cmake.")
endif()

# Internal alias kept for compatibility with target_compile_options below.
set(AMDGPU_TARGETS "${GPU_ARCH}")

message(STATUS "Building for GPU architecture: ${AMDGPU_TARGETS}")

# ============================================================================
# COMPILER CONFIGURATION
# ============================================================================

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_AMD__")
# --offload-arch flags are NOT added to CMAKE_CXX_FLAGS globally — that would break
# non-HIP targets (samples, GTest) compiled by the system C++ compiler.
# Arch flags propagate to HIP-linked targets via the hip::device imported target
# INTERFACE, which hip-config.cmake populates from AMDGPU_TARGETS automatically.

# ============================================================================
# COLLECT EXISTING TEST FILES
# ============================================================================

message(STATUS "")
message(STATUS "=============================================================")
message(STATUS "rocPRIM Test Suite - Existing Tests Only")
message(STATUS "=============================================================")

set(TEST_SOURCES
    # Main entry point
    main.cpp
    
    # System tests - Concurrency
    system/concurrency/test_system_multistream.cpp
    
    # System tests - Reliability (Multi-GPU HMM)
    system/reliability/test_system_multigpu_hmm.cpp
    
    # System tests - Stress
    system/stress/test_system_performance_stress.cpp
)

message(STATUS "Test Files:")
foreach(source_file ${TEST_SOURCES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")
        message(STATUS "  ✓ ${source_file}")
    else()
        message(FATAL_ERROR "  ✗ ${source_file} NOT FOUND!")
    endif()
endforeach()

# ============================================================================
# CREATE TEST EXECUTABLE
# ============================================================================

add_executable(rocprim_tests ${TEST_SOURCES})

set_target_properties(rocprim_tests PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    LINKER_LANGUAGE CXX
)

target_include_directories(rocprim_tests 
    PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(rocprim_tests 
    PRIVATE 
    hip::device
    roc::rocprim
    roc::rocrand
    GTest::GTest
)

message(STATUS "")
message(STATUS "✓ Test executable: rocprim_tests")
message(STATUS "  - Concurrency: 4 tests (8 streams, parallel)")
message(STATUS "  - Multi-GPU HMM: 2 tests")
message(STATUS "  - Stress: 4 tests (MaxMemoryPressure, MassiveConcurrency, MixedWorkload, LinearScaling)")

# ============================================================================
# CTEST INTEGRATION
# ============================================================================

enable_testing()

# All tests
#add_test(NAME all_tests COMMAND rocprim_tests)
#set_tests_properties(all_tests PROPERTIES TIMEOUT 1200 LABELS "all")

# Concurrency tests
add_test(NAME concurrency_tests 
         COMMAND rocprim_tests --gtest_filter=*SystemMultiStreamTests.*)
set_tests_properties(concurrency_tests PROPERTIES 
    TIMEOUT 600 
    LABELS "concurrency;system")

# Multi-GPU HMM tests
add_test(NAME multigpu_hmm_tests 
         COMMAND rocprim_tests --gtest_filter=*MultiGPUHMMTests.*)
set_tests_properties(multigpu_hmm_tests PROPERTIES 
    TIMEOUT 800 
    LABELS "multigpu;reliability;system"
    ENVIRONMENT "HIP_VISIBLE_DEVICES=0,1")

# Stress tests
add_test(NAME stress_tests 
         COMMAND rocprim_tests --gtest_filter=*SystemStressTests.*)
set_tests_properties(stress_tests PROPERTIES 
    TIMEOUT 1200 
    LABELS "stress;system")

message(STATUS "")
message(STATUS "CTest Labels:")
message(STATUS "  - all: All tests")
message(STATUS "  - concurrency: Multi-stream tests")
message(STATUS "  - multigpu: Multi-GPU HMM tests")
message(STATUS "  - stress: Aggressive stress tests")

# ============================================================================
# CUSTOM TEST TARGETS
# ============================================================================

add_custom_target(run_all
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS rocprim_tests
    COMMENT "Running all tests")

add_custom_target(run_concurrency
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L concurrency
    DEPENDS rocprim_tests
    COMMENT "Running concurrency tests")

add_custom_target(run_multigpu
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L multigpu
    DEPENDS rocprim_tests
    COMMENT "Running multi-GPU tests")

add_custom_target(run_stress
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L stress
    DEPENDS rocprim_tests
    COMMENT "Running stress tests")

message(STATUS "")
message(STATUS "Custom Targets:")
message(STATUS "  - make run_all")
message(STATUS "  - make run_concurrency")
message(STATUS "  - make run_multigpu")
message(STATUS "  - make run_stress")

# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

message(STATUS "")
message(STATUS "=============================================================")
message(STATUS "Build Configuration Complete")
message(STATUS "=============================================================")
message(STATUS "GPU Architectures: ${AMDGPU_TARGETS}")
message(STATUS "")
message(STATUS "Build:")
message(STATUS "  cmake -B build && cmake --build build")
message(STATUS "")
message(STATUS "Run Tests:")
message(STATUS "  ./build/rocprim_tests")
message(STATUS "  ctest --test-dir build -L concurrency")
message(STATUS "  ctest --test-dir build -L multigpu")
message(STATUS "  ctest --test-dir build -L stress")
message(STATUS "=============================================================")
message(STATUS "")
