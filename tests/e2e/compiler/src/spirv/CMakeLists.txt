# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Minimal HIP project for the basic SPIR-V compile/run test. The HIP target
# architecture is supplied by the framework via -DCMAKE_HIP_ARCHITECTURES
# (set to amdgcnspirv by the hip_basic_spirv_binary fixture) so the executable
# is emitted as a SPIR-V offload bundle.

cmake_minimum_required(VERSION 3.21)
project(hip_basic_spirv LANGUAGES CXX HIP)

add_executable(vector_add_spirv vector_add.hip)
set_source_files_properties(vector_add.hip PROPERTIES LANGUAGE HIP)
