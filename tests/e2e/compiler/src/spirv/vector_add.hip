// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Minimal self-contained HIP vector-add used by test_hip_basic_spirv. Built with
// -DCMAKE_HIP_ARCHITECTURES=amdgcnspirv so it is emitted as a SPIR-V offload
// bundle; HIP JIT-compiles it at load time. Prints "PASSED!" on success.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

namespace {
constexpr int kN = 1 << 16;

#define HIP_CHECK(expr)                                                      \
    do {                                                                     \
        hipError_t _err = (expr);                                           \
        if (_err != hipSuccess) {                                           \
            std::printf("FAILED: %s -> %s\n", #expr, hipGetErrorString(_err)); \
            return 1;                                                        \
        }                                                                    \
    } while (0)
}  // namespace

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const size_t bytes = static_cast<size_t>(kN) * sizeof(float);
    std::vector<float> host_a(kN, 1.0f);
    std::vector<float> host_b(kN, 2.0f);
    std::vector<float> host_c(kN, 0.0f);

    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;
    HIP_CHECK(hipMalloc(&dev_a, bytes));
    HIP_CHECK(hipMalloc(&dev_b, bytes));
    HIP_CHECK(hipMalloc(&dev_c, bytes));

    HIP_CHECK(hipMemcpy(dev_a, host_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_b, host_b.data(), bytes, hipMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (kN + threads - 1) / threads;
    hipLaunchKernelGGL(vector_add, dim3(blocks), dim3(threads), 0, 0, dev_a, dev_b, dev_c, kN);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(host_c.data(), dev_c, bytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(dev_a));
    HIP_CHECK(hipFree(dev_b));
    HIP_CHECK(hipFree(dev_c));

    for (int i = 0; i < kN; ++i) {
        if (host_c[i] != 3.0f) {
            std::printf("FAILED at index %d: got %f, expected 3.0\n", i, host_c[i]);
            return 1;
        }
    }

    std::printf("PASSED!\n");
    return 0;
}
