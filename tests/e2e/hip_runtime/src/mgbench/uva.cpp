// MGBench: Multi-GPU Computing Benchmark Suite
// Copyright (c) 2016, Tal Ben-Nun
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its 
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <map>
#include <random>

#include <gflags/gflags.h>

#include <hip_runtime.h>

DEFINE_uint64(size, 100*1024*1024, "The amount of data to transfer");
DEFINE_uint64(type_size, sizeof(float), "The size of the data chunk to "
              "transfer, e.g. 4 for a 4-byte float");
DEFINE_uint64(repetitions, 100, "Number of repetitions to average");
DEFINE_uint64(block_size, 32, "Copy kernel block size");
DEFINE_bool(fullduplex, false, "True for bi-directional copy");
DEFINE_bool(write, false, "Perform DMA write instead of read");
DEFINE_bool(random, false, "Use random access instead of coalesced");

DEFINE_int32(from, -1, "Only copy from a single GPU index/host (Host is "
             "0, GPUs start from 1), or -1 for all");
DEFINE_int32(to, -1, "Only copy to a single GPU index/host (Host is "
             "0, GPUs start from 1), or -1 for all");

static void HandleError(const char *file, int line, hipError_t err)
{
    printf("ERROR in %s:%d: %s (%d)\n", file, line,
           hipGetErrorString(err), err);
    hipGetLastError();
}

// HIP assertions
#define HIP_CHECK(err) do { hipError_t errr = (err); if(errr != hipSuccess) { HandleError(__FILE__, __LINE__, errr); exit(1); } } while(0)
#define HIP_CHECK_RET(err) do { hipError_t errr = (err); if(errr != hipSuccess) { HandleError(__FILE__, __LINE__, errr); return; } } while(0)

template<typename T>
__global__ void CopyKernel(T *dst_data, const T *src_data, int size)
{
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx >= size)
        return;
    
    dst_data[idx] = src_data[idx];
}

template<typename T , bool WRITE>
__global__ void CopyKernelRandom(T *dst_data, const T *src_data, const int *indices, int size)
{
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx >= size)
        return;

    if (WRITE)
        dst_data[indices[idx]] = src_data[idx];
    else
        dst_data[idx] = src_data[indices[idx]];
}

inline void DispatchCopy(void *dst, const void *src, const size_t& sz, const size_t& type_size,
                         const dim3& grid, const dim3& block, hipStream_t stream)
{
    
  
    switch (type_size)
    {
      case 1: // sizeof(char)
      hipLaunchKernelGGL((CopyKernel<char>), grid, block, 0, stream, (char *)dst, (const char *)src, sz);
          return;
          
      case 2: // sizeof(short)
      hipLaunchKernelGGL((CopyKernel<short>), grid, block,0, stream, (short *)dst, (const short *)src, sz);
          return;

      default:
      case 4: // sizeof(float)
      hipLaunchKernelGGL((CopyKernel<float>), grid, block,0, stream,  (float *)dst, (const float *)src, sz);
          return;
          
      case 8: // sizeof(double)
      hipLaunchKernelGGL((CopyKernel<double>), grid, block,0, stream,  (double *)dst, (const double *)src, sz);
          return;

      case 16: // sizeof(float4)
      hipLaunchKernelGGL((CopyKernel<float4>), grid, block,0, stream, (float4 *)dst, (const float4 *)src, sz);
          return;
    }
   
}

template <bool WRITE>
inline void DispatchCopyRandom(void *dst, const void *src, const int *rnd,
                               const size_t& sz, const size_t& type_size,
                               const dim3& grid, const dim3& block, hipStream_t stream)
{
     
   switch (type_size)
    {
      case 1: // sizeof(char)
    hipLaunchKernelGGL((CopyKernelRandom<char,WRITE>), grid, block, 0, stream, (char *)dst, (const char *)src, rnd, sz);
          return;
          
      case 2: // sizeof(short)
    hipLaunchKernelGGL((CopyKernelRandom<short,WRITE>), grid, block, 0, stream, (short *)dst, (const short *)src, rnd, sz);
          return;

      default:
      case 4: // sizeof(float)
    hipLaunchKernelGGL((CopyKernelRandom<float,WRITE>), grid, block, 0, stream, (float *)dst, (const float *)src, rnd, sz);
          return;
          
      case 8: // sizeof(double)
    hipLaunchKernelGGL((CopyKernelRandom<double,WRITE>), grid, block,0, stream, (double *)dst, (const double *)src, rnd, sz);
          return;

      case 16: // sizeof(float4)
    hipLaunchKernelGGL((CopyKernelRandom<float4,WRITE>), grid, block,0, stream, (float4 *)dst, (const float4 *)src, rnd, sz);
          return;
    }
   
}


void CopySegmentUVA(int a, int b)
{
    void *deva_buff = nullptr, *devb_buff = nullptr;
    void *deva_buff2 = nullptr, *devb_buff2 = nullptr;
    int *devrnd_buff = nullptr;

    hipStream_t a_stream, b_stream;

    size_t sz = FLAGS_size / FLAGS_type_size, typesize = FLAGS_type_size;
    
    // Allocate buffers
    if (a > 0)
    {
        HIP_CHECK_RET(hipSetDevice(a - 1));
        HIP_CHECK_RET(hipMalloc(&deva_buff, FLAGS_size));
        HIP_CHECK_RET(hipMalloc(&deva_buff2, FLAGS_size));
    }
    else
    {
        HIP_CHECK_RET(hipHostMalloc(&deva_buff, FLAGS_size));
        HIP_CHECK_RET(hipHostMalloc(&deva_buff2, FLAGS_size));
    }
    HIP_CHECK_RET(hipStreamCreateWithFlags(&a_stream, hipStreamNonBlocking));
    if (b > 0)
    {
        HIP_CHECK_RET(hipSetDevice(b - 1));
        HIP_CHECK_RET(hipMalloc(&devb_buff, FLAGS_size));
        HIP_CHECK_RET(hipMalloc(&devb_buff2, FLAGS_size));
    }
    else
    {
        HIP_CHECK_RET(hipHostMalloc(&devb_buff, FLAGS_size));
        HIP_CHECK_RET(hipHostMalloc(&devb_buff2, FLAGS_size));
    }

    if (FLAGS_random)
    {
        // Create and allocate random index buffer
        std::vector<int> host_rnd (sz);

        // Randomize
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dist(0, sz - 1);
        for (size_t i = 0; i < sz; ++i)
            host_rnd[i] = dist(gen);

        // Device->Device / Host->Device
        if (b > 0)
        {
            HIP_CHECK_RET(hipSetDevice(b - 1));
        }
        else if (a > 0) // Device->Host
        {
            HIP_CHECK_RET(hipSetDevice(a - 1));
        }

        HIP_CHECK_RET(hipMalloc(&devrnd_buff, sz * sizeof(int)));
        HIP_CHECK_RET(hipMemcpy(devrnd_buff, host_rnd.data(),
                                  sz * sizeof(int), hipMemcpyHostToDevice));
    }

    
    HIP_CHECK_RET(hipStreamCreateWithFlags(&b_stream, hipStreamNonBlocking));

    // Synchronize devices before copying
    if (a > 0)
    {
        HIP_CHECK_RET(hipSetDevice(a - 1));
        HIP_CHECK_RET(hipDeviceSynchronize());
    }
    if (b > 0)
    {
        HIP_CHECK_RET(hipSetDevice(b - 1));
        HIP_CHECK_RET(hipDeviceSynchronize());
    }

    dim3 block_dim (FLAGS_block_size),
         grid_dim((sz + FLAGS_block_size - 1) / FLAGS_block_size);

    // If using UVA to write, simply swap the buffers
    if (FLAGS_write)
    {
        std::swap(deva_buff, devb_buff);
        std::swap(deva_buff2, devb_buff2);
    }
    
    // Copy or Exchange using UVA
    auto t1 = std::chrono::high_resolution_clock::now();
    for(uint64_t i = 0; i < FLAGS_repetitions; ++i)
    {
        if (b > 0)
            HIP_CHECK_RET(hipSetDevice(b - 1));
        else
            HIP_CHECK_RET(hipSetDevice(a - 1));

        if (FLAGS_random)
        {
            if (FLAGS_write)
                DispatchCopyRandom<true>(devb_buff, deva_buff, devrnd_buff,
                                         sz, typesize, grid_dim, block_dim,
                                         b_stream);
            else
                DispatchCopyRandom<false>(devb_buff, deva_buff, devrnd_buff,
                                          sz, typesize, grid_dim, block_dim,
                                          b_stream);
        }
        else
        {
            DispatchCopy(devb_buff, deva_buff, sz, typesize,
                         grid_dim, block_dim, b_stream);
        }
        
        if (FLAGS_fullduplex)
        {
            if (a > 0)
                HIP_CHECK_RET(hipSetDevice(a - 1));
            DispatchCopy(deva_buff2, devb_buff2, sz, typesize,
                         grid_dim, block_dim, a_stream);

        }
    }
    if (a > 0)
    {
        HIP_CHECK_RET(hipSetDevice(a - 1));
        HIP_CHECK_RET(hipDeviceSynchronize());
    }
    if (b > 0)
    {
        HIP_CHECK_RET(hipSetDevice(b - 1));
        HIP_CHECK_RET(hipDeviceSynchronize());
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    double mstime = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0 / FLAGS_repetitions;

    // MiB/s = [bytes / (1024^2)] / [ms / 1000]
    double MBps = (FLAGS_size / 1024.0 / 1024.0) / (mstime / 1000.0);
    
    printf("%.2lf MB/s (%lf ms)\n", MBps, mstime);


    // Swap buffers back, if necessary
    if (FLAGS_write)
    {
        std::swap(deva_buff, devb_buff);
        std::swap(deva_buff2, devb_buff2);
    }
    
    
    // Free buffers
    if (a > 0)
    {
        HIP_CHECK_RET(hipSetDevice(a - 1));
        HIP_CHECK_RET(hipFree(deva_buff));
        HIP_CHECK_RET(hipFree(deva_buff2));
    }
    else
    {
        HIP_CHECK_RET(hipHostFree(deva_buff));
        HIP_CHECK_RET(hipHostFree(deva_buff2));
    }    
    HIP_CHECK_RET(hipStreamDestroy(a_stream));

    if (b > 0)
    {
        HIP_CHECK_RET(hipSetDevice(b - 1));
        HIP_CHECK_RET(hipFree(devb_buff));
        HIP_CHECK_RET(hipFree(devb_buff2));
    }
    else
    {
        HIP_CHECK_RET(hipHostFree(devb_buff));
        HIP_CHECK_RET(hipHostFree(devb_buff2));
    }

    // Free randomized buffer, if exists
    if (FLAGS_random)
    {
        if (b > 0)
        {
            HIP_CHECK_RET(hipSetDevice(b - 1));
        }
        else if (a > 0)
        {
            HIP_CHECK_RET(hipSetDevice(a - 1));
        }
        HIP_CHECK_RET(hipFree(devrnd_buff));
    }
    HIP_CHECK_RET(hipStreamDestroy(b_stream));
}


int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    printf("Inter-GPU DMA %s exchange test\n",
           (FLAGS_fullduplex ? "bi-directional" : "uni-directional"));
    
    int ndevs = 0;
    HIP_CHECK(hipGetDeviceCount(&ndevs));

    if (FLAGS_from >= (ndevs + 1))
    {
        printf("Invalid --from flag. Only %d GPUs are available.\n", ndevs);
        return 1;
    }
    if (FLAGS_to >= (ndevs + 1))
    {
        printf("Invalid --to flag. Only %d GPUs are available.\n", ndevs);
        return 2;
    }
    if (FLAGS_random && FLAGS_fullduplex)
    {
        printf("Cannot enable both --random and --fullduplex flags\n");
        return 3;
    }
    
    printf("Enabling peer-to-peer access\n");

    int tmp = 0;
    
    // Enable peer-to-peer access
    std::map<std::pair<int, int>, bool> canAccessPeer;
    for(int i = 0; i < ndevs; ++i)
    {
        HIP_CHECK(hipSetDevice(i));
        for(int j = 0; j < ndevs; ++j)
            if (i != j)
            {
                hipDeviceEnablePeerAccess(j, 0);
                hipDeviceCanAccessPeer(&tmp, i, j);
                canAccessPeer[std::make_pair(i, j)] = (tmp ? true : false);
            }
    } 

    printf("GPUs: %d (+ host)\n", ndevs);
    printf("Data size: %.2f MB\n", (FLAGS_size / 1024.0f / 1024.0f));
    printf("Data type size: %d bytes\n", (int)FLAGS_type_size);
    printf("Block size: %d\n", (int)FLAGS_block_size);
    printf("Access type: %s\n", (FLAGS_random ? "Randomized" : "Coalesced"));
    printf("Repetitions: %d\n", (int)FLAGS_repetitions);
    printf("\n");
    
    for(int i = 0; i < ndevs + 1; ++i)
    {
        // Skip source GPUs
        if(FLAGS_from >= 0 && i != FLAGS_from)
            continue;

        int start = FLAGS_fullduplex ? i : 0;
        
        for(int j = start; j < ndevs + 1; ++j)
        {
            // Skip self-copies
            if(i == j)
                continue;
            // Skip target GPUs
            if(FLAGS_to >= 0 && j != FLAGS_to)
                continue;

            // Skip cases where host is the target
            if (j == 0)
                continue;
            if (FLAGS_fullduplex && i == 0)
                continue;
                
            if (FLAGS_fullduplex)
            {
                printf("Exchanging between GPU %d and GPU %d: ", i - 1, j - 1);
            }
            else
            {
                if (!FLAGS_write)
                {
                    if (i == 0)
                        printf("Copying from host to GPU %d: ", j - 1);
                    else
                        printf("Copying from GPU %d to GPU %d: ", i - 1, j - 1);
                }
                else
                {
                    if (i == 0)
                        printf("Copying from GPU %d to host: ", j - 1);
                    else
                        printf("Copying from GPU %d to GPU %d: ", j - 1, i - 1);
                }
            }

            // Make sure that DMA access is possible
            if (i > 0 && j > 0)
            {
                if (!canAccessPeer[std::make_pair(i - 1, j - 1)] ||
                    (FLAGS_fullduplex && !canAccessPeer[std::make_pair(j - 1, i - 1)]))
                {
                    printf("No DMA\n");
                    continue;
                }
            }
            
            CopySegmentUVA(i, j);
        }
    }

    return 0;
}
 
