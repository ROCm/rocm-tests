/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
// SPDX-License-Identifier: MIT

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
