# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# rocm_setup.cmake — shared ROCm path and search-path setup for e2e test builds.
#
# PURPOSE
# -------
# Centralises the ROCM_PATH resolution and CMAKE_PREFIX_PATH extension that
# every test CMakeLists.txt needs but previously copy-pasted independently.
#
# USAGE
# -----
# Include this file BEFORE project() in each test CMakeLists.txt:
#
#   list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/<relpath>/cmake")
#   include(rocm_setup)
#   # ... compiler detection (each area owns this) ...
#   project(my_test LANGUAGES CXX HIP)
#   find_package(hip CONFIG REQUIRED)   # or find_package(HIP REQUIRED)
#
# WHAT THIS MODULE DOES
# ---------------------
# 1. Resolves ROCM_PATH from: -DROCM_PATH arg → ENV{ROCM_PATH} → /opt/rocm
# 2. Writes ROCM_PATH back into ENV{ROCM_PATH} for any subprocesses cmake spawns.
# 3. Appends ROCM_PATH to CMAKE_PREFIX_PATH so find_package(hip CONFIG) can
#    locate hipConfig.cmake under <rocm_path>/lib/cmake/hip/.
# 4. Appends <rocm_path>/lib/cmake/hip to CMAKE_MODULE_PATH for module-mode
#    find_package(HIP REQUIRED) users.
# 5. Sets HIP_PLATFORM=amd (required before find_package(hip CONFIG)).
#
# WHAT THIS MODULE DOES NOT DO
# ----------------------------
# - Does NOT set CMAKE_CXX_COMPILER or CMAKE_HIP_COMPILER.
#   Each CMakeLists.txt owns its compiler detection (or relies on the conftest
#   passing -DCMAKE_CXX_COMPILER / -DCMAKE_HIP_COMPILER via -D flags).
# - Does NOT call find_package(). That must happen after project() once the
#   CMake language has been initialised.
# - Does NOT call project() itself.
#
# WHY NO RECURSIVE LOOP
# ---------------------
# - include() is inline execution — not a subprocess; no cmake re-entry.
# - CMake's HIP language initialisation (CMakeDetermineHIPCompiler.cmake) runs
#   at project() time, not at include() time. Including this file before
#   project() cannot trigger compiler detection.
# - find_package() is only called by the including CMakeLists.txt after
#   project(), so HIP toolchain detection has already completed before it runs.
# - -D command-line values always override CACHE assignments, so compilers
#   passed from the conftest via -DCMAKE_HIP_COMPILER=... are never overwritten
#   by anything in this module.

cmake_minimum_required(VERSION 3.21)   # sanity gate; callers also declare their own

# ---------------------------------------------------------------------------
# 1. Resolve ROCM_PATH
# ---------------------------------------------------------------------------
if(NOT ROCM_PATH)
  if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    set(ROCM_PATH "$ENV{ROCM_PATH}")
  else()
    set(ROCM_PATH "/opt/rocm")
  endif()
endif()

# ---------------------------------------------------------------------------
# 2. Write back to ENV so downstream tool invocations see the resolved value
# ---------------------------------------------------------------------------
set(ENV{ROCM_PATH} "${ROCM_PATH}")

# ---------------------------------------------------------------------------
# 3 + 4. Extend search paths for both config-mode and module-mode HIP
# ---------------------------------------------------------------------------
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
list(APPEND CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake/hip")

# ---------------------------------------------------------------------------
# 5. HIP platform selection — must be set before find_package(hip CONFIG)
# ---------------------------------------------------------------------------
set(HIP_PLATFORM "amd")
