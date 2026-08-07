// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// hip_device_count.cpp -- Print hipGetDeviceCount() to stdout.
//
// Used by the partition isolation test to determine how many logical HIP
// devices are visible before assigning normal and buggy workloads to
// disjoint partitions.
//
// Prints the integer count as a single line to stdout.
// Exit 0 on success; exit 1 on hipGetDeviceCount error.

#include <hip/hip_runtime.h>

#include <cstdio>

int main()
{
    int n = 0;
    hipError_t e = hipGetDeviceCount(&n);
    if (e != hipSuccess) {
        fprintf(stderr, "hipGetDeviceCount: %s\n", hipGetErrorString(e));
        return 1;
    }
    printf("%d\n", n);
    return 0;
}
