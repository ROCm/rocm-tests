// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call)                                                        \
    {                                                                           \
        hipError_t err = (call);                                                \
        if (err != hipSuccess) {                                                \
            fprintf(stderr, "[HIP ERROR] %s:%d — %s returned %s (%d)\n",       \
                    __FILE__, __LINE__, #call, hipGetErrorString(err), err);     \
            return 1;                                                           \
        }                                                                       \
    }

#define HIP_LAUNCH_CHECK()                                                     \
    {                                                                           \
        hipError_t err = hipPeekAtLastError();                                  \
        if (err != hipSuccess) {                                                \
            fprintf(stderr, "[HIP LAUNCH ERROR] %s:%d — %s (%d)\n",            \
                    __FILE__, __LINE__, hipGetErrorString(err), err);            \
            (void)hipGetLastError();                                            \
        }                                                                       \
    }

#define HIP_LAUNCH_CHECK_OR(label)                                             \
    {                                                                           \
        hipError_t err = hipPeekAtLastError();                                  \
        if (err != hipSuccess) {                                                \
            fprintf(stderr, "[HIP LAUNCH ERROR] %s:%d — %s (%d)\n",            \
                    __FILE__, __LINE__, hipGetErrorString(err), err);            \
            (void)hipGetLastError();                                            \
            hip_exit_code = err;                                                \
            goto label;                                                         \
        }                                                                       \
    }

#define HIP_CHECK_OR(call, label)                                              \
    {                                                                           \
        hipError_t err = (call);                                                \
        if (err != hipSuccess) {                                                \
            fprintf(stderr, "[HIP ERROR] %s:%d — %s returned %s (%d)\n",       \
                    __FILE__, __LINE__, #call, hipGetErrorString(err), err);     \
            hip_exit_code = err;                                                \
            goto label;                                                         \
        }                                                                       \
    }
