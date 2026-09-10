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

#include <gflags/gflags.h>

#include <hip_runtime.h>

DEFINE_uint64(size, 100*1024*1024, "The amount of data to transfer");
DEFINE_uint64(repetitions, 100, "Number of repetitions to average");

DEFINE_int32(from, -1, "Only copy from a single GPU index, or -1 for all");
DEFINE_int32(to, -1, "Only copy to a single GPU index, or -1 for all");

static void HandleError(const char *file, int line, hipError_t err)
{
    printf("ERROR in %s:%d: %s (%d)\n", file, line,
           hipGetErrorString(err), err);
    exit(1);
}

// HIP assertions
#define HIP_CHECK(err) do { hipError_t errr = (err); if(errr != hipSuccess) { HandleError(__FILE__, __LINE__, errr); } } while(0)

void CopySegment(int a, int b)
{
    void *deva_buff = nullptr, *devb_buff = nullptr;
    void *deva_buff2 = nullptr, *devb_buff2 = nullptr;

    hipStream_t a_stream, b_stream;

    // Allocate buffers
    HIP_CHECK(hipSetDevice(a));
    HIP_CHECK(hipMalloc(&deva_buff, FLAGS_size));
    HIP_CHECK(hipMalloc(&deva_buff2, FLAGS_size));
    HIP_CHECK(hipStreamCreateWithFlags(&a_stream, hipStreamNonBlocking));
    HIP_CHECK(hipSetDevice(b));
    HIP_CHECK(hipMalloc(&devb_buff, FLAGS_size));
    HIP_CHECK(hipMalloc(&devb_buff2, FLAGS_size));
    HIP_CHECK(hipStreamCreateWithFlags(&b_stream, hipStreamNonBlocking));

    // Synchronize devices before copying
    HIP_CHECK(hipSetDevice(a));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipSetDevice(b));
    HIP_CHECK(hipDeviceSynchronize());

    
    
    // Exchange
    auto t1 = std::chrono::high_resolution_clock::now();
    for(uint64_t i = 0; i < FLAGS_repetitions; ++i)
    {
        HIP_CHECK(hipMemcpyPeerAsync(devb_buff, b, deva_buff, a,
                                       FLAGS_size, b_stream));
        HIP_CHECK(hipMemcpyPeerAsync(deva_buff2, a, devb_buff2, b,
                                       FLAGS_size, a_stream));
    }
    HIP_CHECK(hipSetDevice(a));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipSetDevice(b));
    HIP_CHECK(hipDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    double mstime = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0 / FLAGS_repetitions;

    // MiB/s = [bytes / (1024^2)] / [ms / 1000]
    double MBps = (FLAGS_size / 1024.0 / 1024.0) / (mstime / 1000.0);
    
    printf("%.2lf MB/s (%lf ms)\n", MBps, mstime);
    
    // Free buffers
    HIP_CHECK(hipSetDevice(a));
    HIP_CHECK(hipFree(deva_buff));
    HIP_CHECK(hipFree(deva_buff2));
    HIP_CHECK(hipStreamDestroy(a_stream));
    HIP_CHECK(hipSetDevice(b));
    HIP_CHECK(hipFree(devb_buff));
    HIP_CHECK(hipFree(devb_buff2));
    HIP_CHECK(hipStreamDestroy(b_stream));
}


int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    printf("Inter-GPU bi-directional memory exhange test\n");
    
    int ndevs = 0;
    HIP_CHECK(hipGetDeviceCount(&ndevs));

    if (FLAGS_from >= ndevs)
    {
        printf("Invalid --from flag. Only %d GPUs are available.\n", ndevs);
        return 1;
    }
    if (FLAGS_to >= ndevs)
    {
        printf("Invalid --to flag. Only %d GPUs are available.\n", ndevs);
        return 2;
    }
    
    printf("Enabling peer-to-peer access\n");
    
    // Enable peer-to-peer access       
    for(int i = 0; i < ndevs; ++i)
    {
        HIP_CHECK(hipSetDevice(i));
        for(int j = 0; j < ndevs; ++j)
            if (i != j)
                hipDeviceEnablePeerAccess(j, 0);
    } 

    printf("GPUs: %d\n", ndevs);
    printf("Data size: %.2f MB\n", (FLAGS_size / 1024.0f / 1024.0f));
    printf("Repetitions: %d\n", (int)FLAGS_repetitions);
    printf("\n");
    
    for(int i = 0; i < ndevs; ++i)
    {
        // Skip source GPUs
        if(FLAGS_from >= 0 && i != FLAGS_from)
            continue;
        
        for(int j = i; j < ndevs; ++j)
        {
            // Skip self-copies
            if(i == j)
                continue;
            // Skip target GPUs
            if(FLAGS_to >= 0 && j != FLAGS_to)
                continue;

            printf("Exchanging between GPU %d and GPU %d: ", i, j);

            CopySegment(i, j);
        }
    }

    return 0;
}
