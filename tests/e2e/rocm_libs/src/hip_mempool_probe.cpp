// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// hip_mempool_probe -- capability probe for the HIP stream-ordered memory pool.
//
// The rocSOLVER / hipSPARSE workloads in this directory allocate scratch through
// hipMallocAsync, whose default pool is backed by HIP virtual-memory management
// (VMM). On driver/kernel stacks that lack VMM support, hsa_amd_vmem_address_reserve
// fails and those allocations error out. This probe performs a single default-pool
// async allocation and reports whether it succeeds, so the test harness can decide
// at runtime whether the VM-heap workaround (DEBUG_HIP_MEM_POOL_VMHEAP=0) is needed
// on the host the test actually lands on.
//
// Output / exit code contract (parsed by the hip_mempool_env fixture):
//     "VMM_POOL=1"       exit 0  -> default async pool works; no workaround needed
//     "VMM_POOL=0 (...)" exit 1  -> async pool allocation failed; apply workaround
//     "VMM_POOL=unknown" exit 2  -> could not probe (no device / setup failure)

#include <hip/hip_runtime.h>

#include <cstdio>

int main() {
  int dev_count = 0;
  if (hipGetDeviceCount(&dev_count) != hipSuccess || dev_count == 0) {
    printf("VMM_POOL=unknown (no device)\n");
    return 2;
  }

  hipStream_t stream;
  if (hipStreamCreate(&stream) != hipSuccess) {
    printf("VMM_POOL=unknown (stream create failed)\n");
    return 2;
  }

  void *ptr = nullptr;
  hipError_t err = hipMallocAsync(&ptr, 1u << 20, stream);
  if (err != hipSuccess) {
    printf("VMM_POOL=0 (%s)\n", hipGetErrorString(err));
    return 1;
  }

  hipFreeAsync(ptr, stream);
  hipStreamSynchronize(stream);
  printf("VMM_POOL=1\n");
  return 0;
}
