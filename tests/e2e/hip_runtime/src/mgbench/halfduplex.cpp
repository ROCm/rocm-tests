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
DEFINE_uint64(chunksize, 0, "If not zero, fragments the data into chunksize-byte chunks");
DEFINE_uint64(repetitions, 100, "Number of repetitions to average");
DEFINE_bool(sync_chunks, false, "If true, synchronizes at the end of each fragment transfer");

DEFINE_int32(from, -1, "Only copy from a single GPU index/host (Host is "
             "0, GPUs start from 1), or -1 for all");
DEFINE_int32(to, -1, "Only copy to a single GPU index/host (Host is "
             "0, GPUs start from 1), or -1 for all");

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

    // Allocate buffers
    HIP_CHECK(hipSetDevice(a));
    HIP_CHECK(hipMalloc(&deva_buff, FLAGS_size));    
    HIP_CHECK(hipSetDevice(b));
    HIP_CHECK(hipMalloc(&devb_buff, FLAGS_size));

    // Create event (for synced fragmentation)
    hipEvent_t hip_event;
    HIP_CHECK(hipEventCreateWithFlags(&hip_event, hipEventDisableTiming));

    // Synchronize devices before copying
    HIP_CHECK(hipSetDevice(a));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipSetDevice(b));
    HIP_CHECK(hipDeviceSynchronize());

    size_t chunk_size = ((FLAGS_chunksize == 0) ? FLAGS_size : FLAGS_chunksize);
    int num_chunks = (FLAGS_size + chunk_size - 1) / chunk_size;
    size_t chunk_remainder = FLAGS_size - (num_chunks - 1) * chunk_size;

    // Copy
    auto t1 = std::chrono::high_resolution_clock::now();
    for(uint64_t i = 0; i < FLAGS_repetitions; ++i)
    {
        char *dstp = (char *)devb_buff, *srcp = (char *)deva_buff;
        size_t curchunk = chunk_size;

        for (int chunk = 0; chunk < num_chunks; ++chunk)
        {
            if (chunk == num_chunks - 1)
                curchunk = chunk_remainder;

            HIP_CHECK(hipMemcpyPeerAsync(dstp, b, srcp, a,
                                           curchunk));

            dstp += chunk_size;
            srcp += chunk_size;

            if (FLAGS_sync_chunks)
            {
                HIP_CHECK(hipEventRecord(hip_event));
                HIP_CHECK(hipEventSynchronize(hip_event));
            }
        }
        HIP_CHECK(hipSetDevice(a));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipSetDevice(b));
        HIP_CHECK(hipDeviceSynchronize());
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    double mstime = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0 / FLAGS_repetitions;

    // MiB/s = [bytes / (1024^2)] / [ms / 1000]
    double MBps = (FLAGS_size / 1024.0 / 1024.0) / (mstime / 1000.0);
    
    printf("%.2lf MB/s (%lf ms)\n", MBps, mstime);
    
    // Destroy event
    HIP_CHECK(hipEventDestroy(hip_event));

    // Free buffers
    HIP_CHECK(hipSetDevice(a));
    HIP_CHECK(hipFree(deva_buff));
    HIP_CHECK(hipSetDevice(b));
    HIP_CHECK(hipFree(devb_buff));
}

void CopyHostDevice(int dev, bool d2h)
{
    void *dev_buff = nullptr, *host_buff = nullptr;

    // Allocate buffers
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipMalloc(&dev_buff, FLAGS_size));    
    HIP_CHECK(hipHostMalloc(&host_buff, FLAGS_size));

    // Create event (for synced fragmentation)
    hipEvent_t hip_event;
    HIP_CHECK(hipEventCreateWithFlags(&hip_event, hipEventDisableTiming));

    // Synchronize devices before copying
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipDeviceSynchronize());

    size_t chunk_size = ((FLAGS_chunksize == 0) ? FLAGS_size : FLAGS_chunksize);
    int num_chunks = (FLAGS_size + chunk_size - 1) / chunk_size;
    size_t chunk_remainder = FLAGS_size - (num_chunks - 1) * chunk_size;

    // Copy
    auto t1 = std::chrono::high_resolution_clock::now();
    for(uint64_t i = 0; i < FLAGS_repetitions; ++i)
    {
        char *devp = (char *)dev_buff, *hostp = (char *)host_buff;
        size_t curchunk = chunk_size;

        for (int chunk = 0; chunk < num_chunks; ++chunk)
        {
            if (chunk == num_chunks - 1)
                curchunk = chunk_remainder;

            if (d2h)
            {
                HIP_CHECK(hipMemcpyAsync(hostp, devp,
                                           curchunk, hipMemcpyDeviceToHost));
            }
            else
            {
                HIP_CHECK(hipMemcpyAsync(devp, hostp,
                                           curchunk, hipMemcpyHostToDevice));
            }

            devp += chunk_size;
            hostp += chunk_size;

            if (FLAGS_sync_chunks)
            {
                HIP_CHECK(hipEventRecord(hip_event));
                HIP_CHECK(hipEventSynchronize(hip_event));
            }
        }
        HIP_CHECK(hipSetDevice(dev));
        HIP_CHECK(hipDeviceSynchronize());
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    double mstime = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0 / FLAGS_repetitions;

    // MiB/s = [bytes / (1024^2)] / [ms / 1000]
    double MBps = (FLAGS_size / 1024.0 / 1024.0) / (mstime / 1000.0);

    printf("%.2lf MB/s (%lf ms)\n", MBps, mstime);

    // Destroy event
    HIP_CHECK(hipEventDestroy(hip_event));

    // Free buffers
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipFree(dev_buff));
    HIP_CHECK(hipHostFree(host_buff));
}


int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    printf("Inter-GPU uni-directional memory transfer test\n");
    
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
    if (FLAGS_chunksize > 0)
    {
        printf("Fragment size: %.2f MB\n", (FLAGS_chunksize / 1024.0f / 1024.0f));
        printf("Fragments per transfer: %d\n",
               (int)((FLAGS_size + FLAGS_chunksize - 1) / FLAGS_chunksize));
    }
    printf("Repetitions: %d\n", (int)FLAGS_repetitions);
    printf("\n");
    
    for(int i = 0; i < ndevs + 1; ++i)
    {
        // Skip source GPUs
        if(FLAGS_from >= 0 && i != FLAGS_from)
            continue;
        
        for(int j = 0; j < ndevs + 1; ++j)
        {
            // Skip self-copies
            if(i == j)
                continue;
            // Skip target GPUs
            if(FLAGS_to >= 0 && j != FLAGS_to)
                continue;

            if (i != 0 && j != 0)
            {
                printf("Copying from GPU %d to GPU %d: ", i - 1, j - 1);
                CopySegment(i - 1, j - 1);
            }
            else if (i == 0 && j != 0)
            {
                printf("Copying from host to GPU %d: ", j - 1);
                CopyHostDevice(j - 1, false);
            }
            else if (i != 0 && j == 0)
            {
                printf("Copying from GPU %d to host: ", i - 1);
                CopyHostDevice(i - 1, true);
            }
        }
    }

    return 0;
}
