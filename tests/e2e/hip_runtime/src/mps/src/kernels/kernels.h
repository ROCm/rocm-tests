// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

__global__ void kernel_pattern_fill(uint32_t* buf, size_t count, uint32_t pattern);

__global__ void kernel_pattern_verify(const uint32_t* buf, size_t count,
                                      uint32_t pattern, int* error_flag);

__global__ void kernel_compute_burn(float* out, size_t count, int fma_iters);

__global__ void kernel_shared_mem_test(uint32_t* out, size_t count,
                                       uint32_t pattern);
