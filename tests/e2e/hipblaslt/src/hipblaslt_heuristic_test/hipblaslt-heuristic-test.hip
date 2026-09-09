// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <vector>

int main() {
  int exit_code = 0;
  hipblasLtHandle_t handle = nullptr;
  hipblasLtMatmulDesc_t matmulDesc = nullptr;
  hipblasLtMatrixLayout_t ADesc = nullptr, BDesc = nullptr, CDesc = nullptr,
                          DDesc = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;
  constexpr int kMaxRequestedAlgorithms = 128;
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults(
      kMaxRequestedAlgorithms);
  int returnedAlgoCount = 0;
  int algo_state_failures = 0;
  int workspace_violations = 0;
  size_t max_workspace_bytes = 4194304;
  hipblasOperation_t transA = HIPBLAS_OP_N;
  hipblasOperation_t transB = HIPBLAS_OP_N;
  hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;

#define CHECK_HIPBLASLT(call)                                                   \
  do {                                                                           \
    const hipblasStatus_t _status = (call);                                     \
    if (_status != HIPBLAS_STATUS_SUCCESS) {                                    \
      std::cerr << "ERROR: " #call " failed with status " << _status            \
                << std::endl;                                                   \
      exit_code = 1;                                                             \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

  // Initialize hipBLASLt handle
  CHECK_HIPBLASLT(hipblasLtCreate(&handle));

  // Create matrix multiplication descriptor
  CHECK_HIPBLASLT(
      hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));

  CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
      matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
  CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
      matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));
  CHECK_HIPBLASLT(hipblasLtMatmulDescSetAttribute(
      matmulDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

  // Create matrix layouts - M=2304, N=4096, K=768
  CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&ADesc, HIP_R_32F, 2304, 768,
                                              2304));
  CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&BDesc, HIP_R_32F, 768, 4096,
                                              768));
  CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&CDesc, HIP_R_32F, 2304, 4096,
                                              2304));
  CHECK_HIPBLASLT(hipblasLtMatrixLayoutCreate(&DDesc, HIP_R_32F, 2304, 4096,
                                              2304));

  // Create preference handle
  CHECK_HIPBLASLT(hipblasLtMatmulPreferenceCreate(&preference));
  CHECK_HIPBLASLT(hipblasLtMatmulPreferenceSetAttribute(
      preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &max_workspace_bytes, sizeof(max_workspace_bytes)));

  // Request 128 algorithms
  CHECK_HIPBLASLT(hipblasLtMatmulAlgoGetHeuristic(
      handle, matmulDesc, ADesc, BDesc, CDesc, DDesc, preference,
      kMaxRequestedAlgorithms, heuristicResults.data(), &returnedAlgoCount));

  std::cout << "Number of algorithms returned: " << returnedAlgoCount
            << std::endl;
  if (returnedAlgoCount <= 0) {
    std::cerr << "ERROR: No algorithms returned by "
                 "hipblasLtMatmulAlgoGetHeuristic."
              << std::endl;
    exit_code = 1;
    goto cleanup;
  }

  for (int i = 0; i < returnedAlgoCount; i++) {
    if (heuristicResults[i].state != HIPBLAS_STATUS_SUCCESS) {
      ++algo_state_failures;
      std::cerr << "ERROR: Algorithm " << i << " failed with status "
                << heuristicResults[i].state << std::endl;
      continue;
    }
    if (heuristicResults[i].workspaceSize > max_workspace_bytes) {
      ++workspace_violations;
      std::cerr << "ERROR: Algorithm " << i << " workspace size "
                << heuristicResults[i].workspaceSize
                << " exceeds max_workspace_bytes " << max_workspace_bytes
                << std::endl;
      continue;
    }
  }

  if (algo_state_failures > 0 || workspace_violations > 0) {
    std::cerr << "ERROR: Returned " << returnedAlgoCount
              << " algorithms, state failures=" << algo_state_failures
              << ", workspace violations=" << workspace_violations << '.'
              << std::endl;
    exit_code = 1;
  } else {
    std::cout << "PASS: All " << returnedAlgoCount
              << " returned algorithms respect workspace limit "
              << max_workspace_bytes << " bytes." << std::endl;
  }

  // Clean up
cleanup:
  if (preference) {
    (void)hipblasLtMatmulPreferenceDestroy(preference);
  }
  if (DDesc) {
    (void)hipblasLtMatrixLayoutDestroy(DDesc);
  }
  if (CDesc) {
    (void)hipblasLtMatrixLayoutDestroy(CDesc);
  }
  if (BDesc) {
    (void)hipblasLtMatrixLayoutDestroy(BDesc);
  }
  if (ADesc) {
    (void)hipblasLtMatrixLayoutDestroy(ADesc);
  }
  if (matmulDesc) {
    (void)hipblasLtMatmulDescDestroy(matmulDesc);
  }
  if (handle) {
    (void)hipblasLtDestroy(handle);
  }

  return exit_code;
}
