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

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <vector>

int main() {
  hipblasLtHandle_t handle;
  hipblasLtMatmulDesc_t matmulDesc;
  hipblasLtMatrixLayout_t ADesc, BDesc, CDesc, DDesc;
  hipblasLtMatmulPreference_t preference;
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults(128);

  // Initialize hipBLASLt handle
  hipblasLtCreate(&handle);

  // Create matrix multiplication descriptor
  hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F);

  hipblasOperation_t transA = HIPBLAS_OP_N;
  hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                  &transA, sizeof(transA));
  hipblasOperation_t transB = HIPBLAS_OP_N;
  hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                  &transB, sizeof(transB));
  hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
  hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                  &epilogue, sizeof(epilogue));

  // Create matrix layouts - M=2304, N=4096, K=768
  hipblasLtMatrixLayoutCreate(&ADesc, HIP_R_32F, 2304, 768, 2304);
  hipblasLtMatrixLayoutCreate(&BDesc, HIP_R_32F, 768, 4096, 768);
  hipblasLtMatrixLayoutCreate(&CDesc, HIP_R_32F, 2304, 4096, 2304);
  hipblasLtMatrixLayoutCreate(&DDesc, HIP_R_32F, 2304, 4096, 2304);

  // Create preference handle
  hipblasLtMatmulPreferenceCreate(&preference);
  size_t max_workspace_bytes = 4194304;
  hipblasLtMatmulPreferenceSetAttribute(
      preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &max_workspace_bytes, sizeof(max_workspace_bytes));

  // Request 128 algorithms
  int returnedAlgoCount = 0;
  hipblasLtMatmulAlgoGetHeuristic(handle, matmulDesc, ADesc, BDesc, CDesc,
                                  DDesc, preference, 128,
                                  heuristicResults.data(), &returnedAlgoCount);

  std::cout << "Number of algorithms returned: " << returnedAlgoCount
            << std::endl;
  for (int i = 0; i < returnedAlgoCount; i++) {
    if (heuristicResults[i].state != HIPBLAS_STATUS_SUCCESS) {
      std::cout << "Algorithm " << i << " failed with status "
                << heuristicResults[i].state << std::endl;
      continue;
    }
    if (heuristicResults[i].workspaceSize > max_workspace_bytes) {
      std::cout << "Algorithm " << i << " workspace size "
                << heuristicResults[i].workspaceSize
                << " exceeds max_workspace_bytes " << max_workspace_bytes
                << std::endl;
      continue;
    }
  }

  // Clean up
  hipblasLtMatmulPreferenceDestroy(preference);
  hipblasLtMatrixLayoutDestroy(DDesc);
  hipblasLtMatrixLayoutDestroy(CDesc);
  hipblasLtMatrixLayoutDestroy(BDesc);
  hipblasLtMatrixLayoutDestroy(ADesc);
  hipblasLtMatmulDescDestroy(matmulDesc);
  hipblasLtDestroy(handle);

  return 0;
}
