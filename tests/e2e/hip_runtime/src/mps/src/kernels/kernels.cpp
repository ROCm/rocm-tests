// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "kernels/kernels.h"

__global__ void kernel_pattern_fill(uint32_t* buf, size_t count, uint32_t pattern) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        buf[idx] = pattern ^ static_cast<uint32_t>(idx);
    }
}

__global__ void kernel_pattern_verify(const uint32_t* buf, size_t count,
                                      uint32_t pattern, int* error_flag) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        uint32_t expected = pattern ^ static_cast<uint32_t>(idx);
        if (buf[idx] != expected) {
            atomicAdd(error_flag, 1);
        }
    }
}

__global__ void kernel_compute_burn(float* out, size_t count, int fma_iters) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        float val = static_cast<float>(idx) * 0.001f;
        for (int i = 0; i < fma_iters; i++) {
            val = fmaf(val, 1.00001f, 0.00001f);
        }
        out[idx] = val;
    }
}

__global__ void kernel_shared_mem_test(uint32_t* out, size_t count,
                                       uint32_t pattern) {
    extern __shared__ uint32_t smem[];
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lid = threadIdx.x;

    smem[lid] = pattern ^ static_cast<uint32_t>(idx);
    __syncthreads();

    if (idx < count) {
        out[idx] = smem[lid];
    }
}
