/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

// SPDX-License-Identifier: BSD-2-Clause

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
