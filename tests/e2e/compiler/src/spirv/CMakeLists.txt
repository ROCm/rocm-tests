# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21)
project(hip_basic_spirv LANGUAGES CXX HIP)

add_executable(vector_add_spirv vector_add.hip)
