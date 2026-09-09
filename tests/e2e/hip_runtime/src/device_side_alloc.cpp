// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Device-side dynamic allocation smoke test.
//
// Exercises in-kernel dynamic memory (``malloc``/``free`` and ``new``/``delete``)
// across five independent scenarios, selected by argv[1]:
//   malloc            -- single-thread in-kernel malloc/free round-trip
//   new               -- single-thread in-kernel new[]/delete[]
//   per_thread        -- every thread allocates, fills, and frees its own buffer
//   per_block         -- one allocation per block, shared by its threads
//   across_kernels    -- allocate in one kernel, consume+free in a second kernel
//
// Each scenario writes a per-element result back to a host-visible buffer; the
// host verifies the expected values. Prints "device_side_alloc <scenario>: PASSED"
// on success (and "... FAILED" otherwise) so the harness can assert on a sentinel.
//
// Re-authored from the standard HIP device-side allocation feature (public HIP
// docs); no internal payload is reused.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>

#define HIP_CHECK(cmd)                                                                    \
  do {                                                                                     \
    hipError_t _e = (cmd);                                                                 \
    if (_e != hipSuccess) {                                                                \
      std::printf("HIP error %s at %s:%d\n", hipGetErrorString(_e), __FILE__, __LINE__);  \
      return false;                                                                        \
    }                                                                                       \
  } while (0)

namespace {

constexpr int kN = 1024;          // elements / threads
constexpr int kPerThreadCount = 8;     // ints each thread allocates
constexpr int kBlock = 128;       // threads per block

// --- Scenario kernels -------------------------------------------------------

__global__ void kMalloc(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i != 0) return;                        // single-thread scenario
  int* p = static_cast<int*>(malloc(n * sizeof(int)));
  if (!p) { for (int k = 0; k < n; ++k) out[k] = -1; return; }
  for (int k = 0; k < n; ++k) p[k] = k * 2;
  for (int k = 0; k < n; ++k) out[k] = p[k];
  free(p);
}

__global__ void kNew(int* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i != 0) return;
  int* p = new int[n];
  if (!p) { for (int k = 0; k < n; ++k) out[k] = -1; return; }
  for (int k = 0; k < n; ++k) p[k] = k * 2;
  for (int k = 0; k < n; ++k) out[k] = p[k];
  delete[] p;
}

__global__ void kPerThread(int* out, int n, int per) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int* p = static_cast<int*>(malloc(per * sizeof(int)));
  if (!p) { out[i] = -1; return; }
  int acc = 0;
  for (int k = 0; k < per; ++k) { p[k] = i + k; acc += p[k]; }
  out[i] = acc;                              // sum_{k}(i+k) = per*i + per*(per-1)/2
  free(p);
}

__global__ void kPerBlock(int* out, int n) {
  __shared__ int* base;
  if (threadIdx.x == 0) base = static_cast<int*>(malloc(blockDim.x * sizeof(int)));
  __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (base == nullptr) { if (i < n) out[i] = -1; return; }
  base[threadIdx.x] = i * 3;
  __syncthreads();
  if (i < n) out[i] = base[threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0) free(base);
}

// across_kernels: kProduce stashes a device pointer + fills it; kConsume reads it
// back and frees it, proving the device heap allocation survives kernel boundaries.
__global__ void kProduce(int** slot, int n) {
  if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
  int* p = static_cast<int*>(malloc(n * sizeof(int)));
  slot[0] = p;
  if (p) for (int k = 0; k < n; ++k) p[k] = k + 7;
}

__global__ void kConsume(int** slot, int* out, int n) {
  if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
  int* p = slot[0];
  if (!p) { for (int k = 0; k < n; ++k) out[k] = -1; return; }
  for (int k = 0; k < n; ++k) out[k] = p[k];
  free(p);
}

// --- Host drivers -----------------------------------------------------------

bool check(const int* h, int n, int (*expect)(int)) {
  for (int k = 0; k < n; ++k)
    if (h[k] != expect(k)) {
      std::printf("mismatch at %d: got %d want %d\n", k, h[k], expect(k));
      return false;
    }
  return true;
}

bool run(const std::string& scenario) {
  // A generous device heap so in-kernel malloc has room.
  HIP_CHECK(hipDeviceSetLimit(hipLimitMallocHeapSize, 64 * 1024 * 1024));

  int* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, kN * sizeof(int)));
  HIP_CHECK(hipMemset(d_out, 0, kN * sizeof(int)));
  int h[kN];
  const int grid = (kN + kBlock - 1) / kBlock;
  bool ok = false;

  if (scenario == "malloc") {
    kMalloc<<<1, 1>>>(d_out, kN);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h, d_out, kN * sizeof(int), hipMemcpyDeviceToHost));
    ok = check(h, kN, [](int k) { return k * 2; });
  } else if (scenario == "new") {
    kNew<<<1, 1>>>(d_out, kN);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h, d_out, kN * sizeof(int), hipMemcpyDeviceToHost));
    ok = check(h, kN, [](int k) { return k * 2; });
  } else if (scenario == "per_thread") {
    kPerThread<<<grid, kBlock>>>(d_out, kN, kPerThreadCount);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h, d_out, kN * sizeof(int), hipMemcpyDeviceToHost));
    ok = check(h, kN, [](int k) { return kPerThreadCount * k + kPerThreadCount * (kPerThreadCount - 1) / 2; });
  } else if (scenario == "per_block") {
    kPerBlock<<<grid, kBlock>>>(d_out, kN);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h, d_out, kN * sizeof(int), hipMemcpyDeviceToHost));
    ok = check(h, kN, [](int k) { return k * 3; });
  } else if (scenario == "across_kernels") {
    int** d_slot = nullptr;
    HIP_CHECK(hipMalloc(&d_slot, sizeof(int*)));
    kProduce<<<1, 1>>>(d_slot, kN);
    HIP_CHECK(hipDeviceSynchronize());
    kConsume<<<1, 1>>>(d_slot, d_out, kN);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h, d_out, kN * sizeof(int), hipMemcpyDeviceToHost));
    ok = check(h, kN, [](int k) { return k + 7; });
    HIP_CHECK(hipFree(d_slot));
  } else {
    std::printf("unknown scenario: %s\n", scenario.c_str());
    HIP_CHECK(hipFree(d_out));
    return false;
  }

  HIP_CHECK(hipFree(d_out));
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  std::string scenario = (argc > 1) ? argv[1] : "malloc";
  bool ok = run(scenario);
  std::printf("device_side_alloc %s: %s\n", scenario.c_str(), ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}
