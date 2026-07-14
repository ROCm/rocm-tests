// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <cstdio>

namespace {
constexpr int kN = 1024;

#define HIP_CHECK(expr)                                                         \
    do {                                                                        \
        hipError_t _err = (expr);                                               \
        if (_err != hipSuccess) {                                               \
            std::printf("FAILED: %s -> %s\n", #expr, hipGetErrorString(_err)); \
            return 1;                                                           \
        }                                                                       \
    } while (0)
}  // namespace

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    float host_a[kN], host_b[kN], host_c[kN];
    for (int i = 0; i < kN; ++i) {
        host_a[i] = 1.0f;
        host_b[i] = 2.0f;
    }

    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;
    const size_t bytes = sizeof(host_a);
    HIP_CHECK(hipMalloc(&dev_a, bytes));
    HIP_CHECK(hipMalloc(&dev_b, bytes));
    HIP_CHECK(hipMalloc(&dev_c, bytes));

    HIP_CHECK(hipMemcpy(dev_a, host_a, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_b, host_b, bytes, hipMemcpyHostToDevice));

    const int threads = 256;
    hipLaunchKernelGGL(add_kernel, dim3((kN + threads - 1) / threads), dim3(threads), 0, 0, dev_a, dev_b, dev_c, kN);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(host_c, dev_c, bytes, hipMemcpyDeviceToHost));

    for (int i = 0; i < kN; ++i) {
        if (host_c[i] != 3.0f) {
            std::printf("FAILED at index %d: got %f, expected 3.0\n", i, host_c[i]);
            return 1;
        }
    }

    HIP_CHECK(hipFree(dev_a));
    HIP_CHECK(hipFree(dev_b));
    HIP_CHECK(hipFree(dev_c));
    std::printf("PASSED!\n");
    return 0;
}
