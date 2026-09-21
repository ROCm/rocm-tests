// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Minimal self-verifying HIP workload used by the multi-instance tests.
//
// Runs a vector-add on the first HIP-visible device, verifies the result on the
// host, prints the device's PCI bus id (so concurrent instances can be shown to
// land on the intended GPU), and prints "multi_instance_app: PASSED" on success.
//
// Instance placement is controlled by the caller via HIP_VISIBLE_DEVICES; this
// program always uses device 0 of whatever is visible to it.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

#define HIP_CHECK(cmd)                                                                    \
  do {                                                                                     \
    hipError_t _e = (cmd);                                                                 \
    if (_e != hipSuccess) {                                                                \
      std::printf("HIP error %s at %s:%d\n", hipGetErrorString(_e), __FILE__, __LINE__);  \
      std::printf("multi_instance_app: FAILED\n");                                         \
      return 1;                                                                            \
    }                                                                                       \
  } while (0)

__global__ void vadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));
  if (count < 1) {
    std::printf("no visible HIP device\nmulti_instance_app: FAILED\n");
    return 1;
  }
  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  constexpr int n = 1 << 20;
  std::vector<float> ha(n, 1.5f), hb(n, 2.5f), hc(n, 0.0f);
  float *da = nullptr, *db = nullptr, *dc = nullptr;
  HIP_CHECK(hipMalloc(&da, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&db, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&dc, n * sizeof(float)));
  HIP_CHECK(hipMemcpy(da, ha.data(), n * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(db, hb.data(), n * sizeof(float), hipMemcpyHostToDevice));

  constexpr int block = 256;
  vadd<<<(n + block - 1) / block, block>>>(da, db, dc, n);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(hc.data(), dc, n * sizeof(float), hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(da));
  HIP_CHECK(hipFree(db));
  HIP_CHECK(hipFree(dc));

  for (int i = 0; i < n; ++i) {
    if (hc[i] != 4.0f) {
      std::printf("verify mismatch at %d: %f\nmulti_instance_app: FAILED\n", i, hc[i]);
      return 1;
    }
  }
  std::printf("device=%s pci=%04x:%02x:%02x\n", prop.name, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
  std::printf("multi_instance_app: PASSED\n");
  return 0;
}
