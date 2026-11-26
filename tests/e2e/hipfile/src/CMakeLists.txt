# hipFile (AIS) System Test
#
# Test suite (CLI: read, write, perf, sizes, align, concurrent, core, all):
#   1. Basic synchronous read   - file -> GPU, verify start/middle/end
#   2. Basic synchronous write  - GPU -> file, POSIX read-back verify
#   3. Performance comparison   - hipFile direct vs traditional CPU-bounce I/O
#   4. Size sweep               - 4K, 64K, 1M, 16M, 64M read and write
#   5. Alignment boundaries     - aligned/unaligned file offset and memory;
#                                 fastpath accept vs reject (statx-derived)
#   6. Concurrent I/O           - multi-threaded, non-overlapping + contended
#
# HIPFILE_PREFIX selects the hipFile install tree (include/, lib/ or lib64/);
# it defaults to ROCM_PATH.

cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm")
endif()

if(NOT DEFINED HIPFILE_PREFIX)
    set(HIPFILE_PREFIX "${ROCM_PATH}")
endif()

if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/llvm/bin/amdclang++")
    set(CMAKE_CXX_COMPILER "${ROCM_PATH}/llvm/bin/amdclang++" CACHE STRING "C++ compiler")
endif()

project(HipFileSystemTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/llvm)

set(HIP_PLATFORM "amd" CACHE STRING "HIP Platform")

find_package(HIP REQUIRED)

# Threads (std::thread in the concurrent test)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_compile_definitions(__HIP_PLATFORM_AMD__=1)

include_directories(
    ${HIPFILE_PREFIX}/include
    ${ROCM_PATH}/include
    ${ROCM_PATH}/include/hip
)

link_directories(
    ${HIPFILE_PREFIX}/lib
    ${HIPFILE_PREFIX}/lib64
    ${ROCM_PATH}/lib
)

# GCC 14 compatibility workaround for ROCm headers
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -D__HIP_NO_STD_CMATH__")

# RPATH so the binary finds hipFile and HIP libraries at runtime
set(CMAKE_INSTALL_RPATH "${HIPFILE_PREFIX}/lib;${HIPFILE_PREFIX}/lib64;${ROCM_PATH}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(hipFile_system_test hipFile_system_test.cpp)

target_link_libraries(hipFile_system_test
    hipfile
    amdhip64
    Threads::Threads
)
