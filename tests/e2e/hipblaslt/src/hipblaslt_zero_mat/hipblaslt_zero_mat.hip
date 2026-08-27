/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
*/
// hipblaslt_zero_mat -- FP8/BF8 GEMM correctness when one input matrix is zero.
//
// D = alpha * A^T * B + beta * C with alpha=beta=1 and C zeroed, so a zero
// operand makes every element of D exactly 0. A is FP8 (E4M3), B is BF8 (E5M2)
// and C/D are BFloat16, matching the mixed-precision pairing the operation is
// deployed with. Two scenarios selected by argv[1]:
//
//   azero / bzero            (optional iteration count in argv[2])
//
// The FP8 encoding is chosen from the device architecture: FNUZ on gfx94x, the
// OCP-standard variants on gfx950 and newer. Each iteration re-poisons D before
// the GEMM so a zero result is meaningful, then checks every element is finite
// and exactly zero. Prints "hipblaslt_zero_mat <check>: PASSED" per scenario.

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(cmd)                                                                   \
  do {                                                                                    \
    hipError_t _e = (cmd);                                                                \
    if (_e != hipSuccess) {                                                               \
      std::printf("HIP error %s at %s:%d\n", hipGetErrorString(_e), __FILE__, __LINE__); \
      return false;                                                                       \
    }                                                                                      \
  } while (0)

#define HBLT_CHECK(cmd)                                                                  \
  do {                                                                                    \
    hipblasStatus_t _s = (cmd);                                                           \
    if (_s != HIPBLAS_STATUS_SUCCESS) {                                                   \
      std::printf("hipBLASLt error %d at %s:%d\n", (int)_s, __FILE__, __LINE__);          \
      return false;                                                                       \
    }                                                                                      \
  } while (0)

namespace {

constexpr int kM = 8192, kN = 32768, kK = 1024;
constexpr int kDefaultIters = 100;

// Any finite nonzero FP8 byte works for the operand that is not zeroed: 0x3C
// decodes to a small finite value in all four E4M3/E5M2 encodings, so the
// product stays zero without depending on which encoding is in use.
constexpr int kNonZeroByte = 0x3C;

float bf16_to_float(uint16_t raw) {
  uint32_t bits = static_cast<uint32_t>(raw) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

bool run_check(const std::string& name, hipDataType a_type, hipDataType b_type, bool a_zero, int iters) {
  hipblasLtHandle_t handle;
  HBLT_CHECK(hipblasLtCreate(&handle));

  const size_t a_elems = static_cast<size_t>(kM) * kK;  // 1 byte per FP8 elem
  const size_t b_elems = static_cast<size_t>(kK) * kN;
  const size_t cd_elems = static_cast<size_t>(kM) * kN;
  const size_t cd_bytes = cd_elems * sizeof(uint16_t);

  void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
  HIP_CHECK(hipMalloc(&dA, a_elems));
  HIP_CHECK(hipMalloc(&dB, b_elems));
  HIP_CHECK(hipMalloc(&dC, cd_bytes));
  HIP_CHECK(hipMalloc(&dD, cd_bytes));
  HIP_CHECK(hipMemset(dA, a_zero ? 0x00 : kNonZeroByte, a_elems));
  HIP_CHECK(hipMemset(dB, a_zero ? kNonZeroByte : 0x00, b_elems));
  HIP_CHECK(hipMemset(dC, 0x00, cd_bytes));

  float *dScaleA = nullptr, *dScaleB = nullptr;
  float one = 1.0f;
  HIP_CHECK(hipMalloc(&dScaleA, sizeof(float)));
  HIP_CHECK(hipMalloc(&dScaleB, sizeof(float)));
  HIP_CHECK(hipMemcpy(dScaleA, &one, sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dScaleB, &one, sizeof(float), hipMemcpyHostToDevice));

  hipblasLtMatmulDesc_t op;
  HBLT_CHECK(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
  HBLT_CHECK(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
  HBLT_CHECK(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
  HBLT_CHECK(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dScaleA, sizeof(dScaleA)));
  HBLT_CHECK(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dScaleB, sizeof(dScaleB)));

  // op(A) is a transpose, so the stored A is K x M with leading dimension K.
  hipblasLtMatrixLayout_t lA, lB, lC, lD;
  HBLT_CHECK(hipblasLtMatrixLayoutCreate(&lA, a_type, kK, kM, kK));
  HBLT_CHECK(hipblasLtMatrixLayoutCreate(&lB, b_type, kK, kN, kK));
  HBLT_CHECK(hipblasLtMatrixLayoutCreate(&lC, HIP_R_16BF, kM, kN, kM));
  HBLT_CHECK(hipblasLtMatrixLayoutCreate(&lD, HIP_R_16BF, kM, kN, kM));

  hipblasLtMatmulPreference_t pref;
  HBLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
  size_t max_ws = 64 * 1024 * 1024;
  HBLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
                                                   sizeof(max_ws)));

  hipblasLtMatmulHeuristicResult_t heur[16];
  int found = 0;
  hipblasStatus_t hs =
      hipblasLtMatmulAlgoGetHeuristic(handle, op, lA, lB, lC, lD, pref, 16, heur, &found);
  if (hs != HIPBLAS_STATUS_SUCCESS || found == 0) {
    std::printf("no hipBLASLt algorithm for %s (a_type=%d b_type=%d): status=%d found=%d\n", name.c_str(),
                (int)a_type, (int)b_type, (int)hs, found);
    return false;
  }

  void* ws = nullptr;
  size_t ws_size = heur[0].workspaceSize;
  if (ws_size) HIP_CHECK(hipMalloc(&ws, ws_size));

  float alpha = 1.0f, beta = 1.0f;
  std::vector<uint16_t> hD(cd_elems);
  bool ok = true;

  for (int it = 0; it < iters && ok; ++it) {
    HIP_CHECK(hipMemset(dD, 0xCC, cd_bytes));  // poison so a real zero result is meaningful
    HBLT_CHECK(hipblasLtMatmul(handle, op, &alpha, dA, lA, dB, lB, &beta, dC, lC, dD, lD, &heur[0].algo, ws, ws_size,
                               nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(hD.data(), dD, cd_bytes, hipMemcpyDeviceToHost));

    for (size_t i = 0; i < cd_elems; ++i) {
      float z = bf16_to_float(hD[i]);
      if (z != 0.0f || !std::isfinite(z)) {
        std::printf("unexpected output at %zu on iteration %d: %f (check %s)\n", i, it, z, name.c_str());
        ok = false;
        break;
      }
    }
  }

  if (ws) HIP_CHECK(hipFree(ws));
  hipblasLtMatmulPreferenceDestroy(pref);
  hipblasLtMatrixLayoutDestroy(lA);
  hipblasLtMatrixLayoutDestroy(lB);
  hipblasLtMatrixLayoutDestroy(lC);
  hipblasLtMatrixLayoutDestroy(lD);
  hipblasLtMatmulDescDestroy(op);
  HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipFree(dB));
  HIP_CHECK(hipFree(dC));
  HIP_CHECK(hipFree(dD));
  HIP_CHECK(hipFree(dScaleA));
  HIP_CHECK(hipFree(dScaleB));
  hipblasLtDestroy(handle);
  return ok;
}

struct Check {
  const char* name;
  bool a_zero;
};

// gfx94x (MI300 series) implements the FNUZ FP8 encodings; gfx950 and newer
// implement the OCP-standard ones instead. Asking for the wrong pair makes
// hipblasLtMatmulAlgoGetHeuristic return no algorithms at all.
bool use_fnuz_fp8(const char** arch_out) {
  static hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    *arch_out = "unknown";
    return true;
  }
  *arch_out = prop.gcnArchName;
  return std::strncmp(prop.gcnArchName, "gfx94", 5) == 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string sel = (argc > 1) ? argv[1] : "all";
  int iters = (argc > 2) ? std::atoi(argv[2]) : kDefaultIters;
  if (iters <= 0) {
    std::printf("iteration count must be positive: %s\n", argv[2]);
    return 2;
  }

  const char* arch = "unknown";
  const bool fnuz = use_fnuz_fp8(&arch);
  const hipDataType a_type = fnuz ? HIP_R_8F_E4M3_FNUZ : HIP_R_8F_E4M3;
  const hipDataType b_type = fnuz ? HIP_R_8F_E5M2_FNUZ : HIP_R_8F_E5M2;
  std::printf("hipblaslt_zero_mat: arch=%s fp8=%s iters=%d\n", arch, fnuz ? "FNUZ" : "OCP", iters);

  const Check checks[] = {
      {"azero", true},
      {"bzero", false},
  };
  int ran = 0, failed = 0;
  for (const auto& c : checks) {
    if (sel != "all" && sel != c.name) continue;
    ++ran;
    bool ok = run_check(c.name, a_type, b_type, c.a_zero, iters);
    std::printf("hipblaslt_zero_mat %s: %s\n", c.name, ok ? "PASSED" : "FAILED");
    if (!ok) ++failed;
  }
  if (ran == 0) {
    std::printf("unknown check: %s\n", sel.c_str());
    return 2;
  }
  return failed ? 1 : 0;
}
