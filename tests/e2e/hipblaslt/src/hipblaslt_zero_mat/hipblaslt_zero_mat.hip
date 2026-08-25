/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
*/
// hipblaslt_zero_mat -- FP8/BF8 GEMM correctness when one input matrix is zero.
//
// D = alpha * A * B + beta * C, with alpha=1, beta=0. When either A or B is the
// zero matrix the product A*B is zero, so every element of D must be exactly 0.
// Runs four checks (E4M3 and E5M2 inputs x {A zero, B zero}) selected by argv[1]:
//
//   e4m3_azero / e4m3_bzero / e5m2_azero / e5m2_bzero   (or "all")
//
// Output/compute are FP32 so the zero check is exact. FP8 scale pointers are set
// to 1.0 for robustness across hipBLASLt versions. Prints
// "hipblaslt_zero_mat <check>: PASSED" per check.

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdio>
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

constexpr int kM = 256, kN = 256, kK = 256;

bool run_check(const std::string& name, hipDataType in_type, bool a_zero) {
  hipblasLtHandle_t handle;
  HBLT_CHECK(hipblasLtCreate(&handle));

  const size_t a_bytes = static_cast<size_t>(kM) * kK;  // 1 byte per FP8 elem
  const size_t b_bytes = static_cast<size_t>(kK) * kN;
  const size_t d_bytes = static_cast<size_t>(kM) * kN * sizeof(float);

  void *dA = nullptr, *dB = nullptr, *dD = nullptr;
  HIP_CHECK(hipMalloc(&dA, a_bytes));
  HIP_CHECK(hipMalloc(&dB, b_bytes));
  HIP_CHECK(hipMalloc(&dD, d_bytes));
  // Zero operand -> 0x00 (== 0.0 in every FP8 encoding); nonzero operand -> 0x3C bytes.
  HIP_CHECK(hipMemset(dA, a_zero ? 0x00 : 0x3C, a_bytes));
  HIP_CHECK(hipMemset(dB, a_zero ? 0x3C : 0x00, b_bytes));
  HIP_CHECK(hipMemset(dD, 0xFF, d_bytes));  // poison so a real zero result is meaningful

  float *dScaleA = nullptr, *dScaleB = nullptr;
  float one = 1.0f;
  HIP_CHECK(hipMalloc(&dScaleA, sizeof(float)));
  HIP_CHECK(hipMalloc(&dScaleB, sizeof(float)));
  HIP_CHECK(hipMemcpy(dScaleA, &one, sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dScaleB, &one, sizeof(float), hipMemcpyHostToDevice));

  hipblasLtMatmulDesc_t op;
  HBLT_CHECK(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  hipblasOperation_t opN = HIPBLAS_OP_N;
  HBLT_CHECK(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
  HBLT_CHECK(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
  HBLT_CHECK(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dScaleA, sizeof(dScaleA)));
  HBLT_CHECK(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dScaleB, sizeof(dScaleB)));

  hipblasLtMatrixLayout_t lA, lB, lC, lD;
  HBLT_CHECK(hipblasLtMatrixLayoutCreate(&lA, in_type, kM, kK, kM));
  HBLT_CHECK(hipblasLtMatrixLayoutCreate(&lB, in_type, kK, kN, kK));
  HBLT_CHECK(hipblasLtMatrixLayoutCreate(&lC, HIP_R_32F, kM, kN, kM));
  HBLT_CHECK(hipblasLtMatrixLayoutCreate(&lD, HIP_R_32F, kM, kN, kM));

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
    std::printf("no hipBLASLt algorithm for %s (in_type=%d): status=%d found=%d\n", name.c_str(), (int)in_type,
                (int)hs, found);
    return false;
  }

  void* ws = nullptr;
  size_t ws_size = heur[0].workspaceSize;
  if (ws_size) HIP_CHECK(hipMalloc(&ws, ws_size));

  float alpha = 1.0f, beta = 0.0f;
  HBLT_CHECK(hipblasLtMatmul(handle, op, &alpha, dA, lA, dB, lB, &beta, dD, lC, dD, lD, &heur[0].algo, ws, ws_size,
                             nullptr));
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> hD(static_cast<size_t>(kM) * kN);
  HIP_CHECK(hipMemcpy(hD.data(), dD, d_bytes, hipMemcpyDeviceToHost));
  bool ok = true;
  for (size_t i = 0; i < hD.size(); ++i) {
    if (hD[i] != 0.0f) {
      std::printf("nonzero output at %zu: %f (check %s)\n", i, hD[i], name.c_str());
      ok = false;
      break;
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
  HIP_CHECK(hipFree(dD));
  HIP_CHECK(hipFree(dScaleA));
  HIP_CHECK(hipFree(dScaleB));
  hipblasLtDestroy(handle);
  return ok;
}

struct Check {
  const char* name;
  hipDataType in_type;
  bool a_zero;
};

}  // namespace

int main(int argc, char** argv) {
  std::string sel = (argc > 1) ? argv[1] : "all";
  const Check checks[] = {
      {"e4m3_azero", HIP_R_8F_E4M3_FNUZ, true},
      {"e4m3_bzero", HIP_R_8F_E4M3_FNUZ, false},
      {"e5m2_azero", HIP_R_8F_E5M2_FNUZ, true},
      {"e5m2_bzero", HIP_R_8F_E5M2_FNUZ, false},
  };
  int ran = 0, failed = 0;
  for (const auto& c : checks) {
    if (sel != "all" && sel != c.name) continue;
    ++ran;
    bool ok = run_check(c.name, c.in_type, c.a_zero);
    std::printf("hipblaslt_zero_mat %s: %s\n", c.name, ok ? "PASSED" : "FAILED");
    if (!ok) ++failed;
  }
  if (ran == 0) {
    std::printf("unknown check: %s\n", sel.c_str());
    return 2;
  }
  return failed ? 1 : 0;
}
