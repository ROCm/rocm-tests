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
**
 * @brief Common utilities for RCCL concurrent collectives testing
 * 
 * This header provides utility functions for:
 * - RCCL initialization and cleanup
 * - HIP stream and memory management
 * - Data initialization with deterministic patterns
 * - Correctness verification for each collective type
 * - Bus bandwidth calculation
 * - Error handling macros
**/

#ifndef RCCL_TEST_UTILS_HPP
#define RCCL_TEST_UTILS_HPP

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <type_traits>
#include <sstream>
#include <stdexcept>

// ============================================================================
// Error Checking Macros
// ============================================================================

#define HIP_CHECK(cmd) do {                                                    \
    hipError_t error = cmd;                                                    \
    if (error != hipSuccess) {                                                 \
        std::ostringstream _oss;                                               \
        _oss << "HIP Error: " << hipGetErrorString(error)                      \
             << " for " << #cmd                                                \
             << " at " << __FILE__ << ":" << __LINE__;                         \
        throw std::runtime_error(_oss.str());                                  \
    }                                                                          \
} while(0)

#define RCCL_CHECK(cmd) do {                                                   \
    ncclResult_t result = cmd;                                                 \
    if (result != ncclSuccess) {                                               \
        std::ostringstream _oss;                                               \
        _oss << "RCCL Error: " << ncclGetErrorString(result)                   \
             << " for " << #cmd                                                \
             << " at " << __FILE__ << ":" << __LINE__;                         \
        throw std::runtime_error(_oss.str());                                  \
    }                                                                          \
} while(0)

// ============================================================================
// Constants and Configuration
// ============================================================================

namespace RcclTestConfig {
    // Tolerance for floating-point comparison
    constexpr float FLOAT32_TOLERANCE = 1e-5f;
    constexpr float FLOAT16_TOLERANCE = 1e-2f;
    
    // Default test parameters
    constexpr size_t DEFAULT_COUNT = 1024 * 1024;  // 1M elements
    constexpr int DEFAULT_WARMUP_ITERS = 5;
    constexpr int DEFAULT_TEST_ITERS = 10;
    
    // Timeout in seconds
    constexpr int DEFAULT_TIMEOUT_SEC = 60;
}

// ============================================================================
// Type utilities (portable type coverage)
// ============================================================================

template <typename T>
constexpr bool kIsIntegral = std::is_integral<T>::value;

template <typename T>
inline double defaultTolerance() {
    if constexpr (std::is_same<T, float>::value) return static_cast<double>(RcclTestConfig::FLOAT32_TOLERANCE);
    if constexpr (std::is_same<T, double>::value) return 1e-12;
    return 0.0; // ints: exact
}

template <typename>
struct AlwaysFalse : std::false_type {};

template <typename T>
inline constexpr const char* typeName() {
    if constexpr (std::is_same<T, float>::value) {
        return "float32";
    } else if constexpr (std::is_same<T, double>::value) {
        return "float64";
    } else if constexpr (std::is_same<T, int>::value) {
        return "int32";
    } else {
        static_assert(AlwaysFalse<T>::value, "Unsupported type for typeName()");
        return "unknown";
    }
}

template <typename T>
inline constexpr ncclDataType_t ncclType() {
    if constexpr (std::is_same<T, float>::value) {
        return ncclFloat;
    } else if constexpr (std::is_same<T, double>::value) {
        return ncclDouble;
    } else if constexpr (std::is_same<T, int>::value) {
        return ncclInt32;
    } else {
        static_assert(AlwaysFalse<T>::value, "Unsupported type for ncclType()");
        return ncclFloat;
    }
}

// ============================================================================
// Data Initialization (Deterministic, Rank-Based Patterns)
// ============================================================================

/**
 * @brief Initialize buffer with rank-based deterministic pattern
 * 
 * Pattern: value = (rank + 1) for all elements
 * This makes expected values easy to compute:
 * - AllReduce sum: n*(n+1)/2 where n = numRanks
 * - AllGather: concatenation of all rank values
 * - Broadcast: root rank's value
 */
template <typename T>
static __global__ void initDataKernelT(T* data, size_t count, int rank) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = static_cast<T>(rank + 1);
    }
}

template <typename T>
inline void initDataGPU(T* d_data, size_t count, int rank, hipStream_t stream) {
    int blockSize = 256;
    int numBlocks = static_cast<int>((count + blockSize - 1) / blockSize);
    hipLaunchKernelGGL(initDataKernelT<T>, dim3(numBlocks), dim3(blockSize), 0, stream,
                       d_data, count, rank);
}

// ============================================================================
// Correctness Verification
// ============================================================================

/**
 * @brief Verification result structure
 */
struct VerificationResult {
    bool passed;
    size_t errorCount;
    size_t firstErrorIndex;
    float expectedValue;
    float actualValue;
    
    void print(const std::string& collectiveName) const {
        if (passed) {
            std::cout << "  " << collectiveName << ": PASSED" << std::endl;
        } else {
            std::cout << "  " << collectiveName << ": FAILED" << std::endl;
            std::cout << "    Errors: " << errorCount << std::endl;
            std::cout << "    First error at index " << firstErrorIndex 
                      << ": expected=" << expectedValue 
                      << ", actual=" << actualValue << std::endl;
        }
    }
};

/**
 * @brief Verify AllReduce result
 * Expected: Each element = sum of (rank+1) for all ranks = n*(n+1)/2
 */
inline VerificationResult verifyAllReduce(
    const float* h_result, 
    size_t count, 
    int numRanks,
    float tolerance = RcclTestConfig::FLOAT32_TOLERANCE
) {
    VerificationResult result = {true, 0, 0, 0.0f, 0.0f};
    
    // Expected value: 1 + 2 + 3 + ... + n = n*(n+1)/2
    float expected = static_cast<float>(numRanks * (numRanks + 1)) / 2.0f;
    
    for (size_t i = 0; i < count; i++) {
        float diff = std::abs(h_result[i] - expected);
        if (diff > tolerance * std::abs(expected) + tolerance) {
            if (result.passed) {
                result.firstErrorIndex = i;
                result.expectedValue = expected;
                result.actualValue = h_result[i];
            }
            result.passed = false;
            result.errorCount++;
        }
    }
    
    return result;
}

template <typename T>
inline VerificationResult verifyAllReduce(
    const T* h_result,
    size_t count,
    int numRanks,
    double tolerance = defaultTolerance<T>()
) {
    VerificationResult result = {true, 0, 0, 0.0f, 0.0f};

    const double expected = static_cast<double>(numRanks) * (static_cast<double>(numRanks) + 1.0) / 2.0;
    for (size_t i = 0; i < count; i++) {
        const double actual = static_cast<double>(h_result[i]);
        bool ok = false;
        if constexpr (kIsIntegral<T>) {
            ok = (actual == expected);
        } else {
            const double diff = std::abs(actual - expected);
            ok = diff <= tolerance * std::abs(expected) + tolerance;
        }

        if (!ok) {
            if (result.passed) {
                result.firstErrorIndex = i;
                result.expectedValue = static_cast<float>(expected);
                result.actualValue = static_cast<float>(actual);
            }
            result.passed = false;
            result.errorCount++;
        }
    }

    return result;
}

/**
 * @brief Verify AllGather result
 * Expected: Concatenation of all ranks' data
 * If each rank sends (rank+1), result should be [1,1,...,2,2,...,3,3,...]
 */
inline VerificationResult verifyAllGather(
    const float* h_result,
    size_t sendCount,  // Elements sent by each rank
    int numRanks,
    int myRank,
    float tolerance = RcclTestConfig::FLOAT32_TOLERANCE
) {
    VerificationResult result = {true, 0, 0, 0.0f, 0.0f};
    
    size_t totalCount = sendCount * numRanks;
    
    for (size_t i = 0; i < totalCount; i++) {
        // Determine which rank this element came from
        int sourceRank = i / sendCount;
        float expected = static_cast<float>(sourceRank + 1);
        
        float diff = std::abs(h_result[i] - expected);
        if (diff > tolerance * std::abs(expected) + tolerance) {
            if (result.passed) {
                result.firstErrorIndex = i;
                result.expectedValue = expected;
                result.actualValue = h_result[i];
            }
            result.passed = false;
            result.errorCount++;
        }
    }
    
    return result;
}

template <typename T>
inline VerificationResult verifyAllGather(
    const T* h_result,
    size_t sendCount,
    int numRanks,
    int /*myRank*/,
    double tolerance = defaultTolerance<T>()
) {
    VerificationResult result = {true, 0, 0, 0.0f, 0.0f};
    const size_t totalCount = sendCount * static_cast<size_t>(numRanks);

    for (size_t i = 0; i < totalCount; i++) {
        const int sourceRank = static_cast<int>(i / sendCount);
        const double expected = static_cast<double>(sourceRank + 1);
        const double actual = static_cast<double>(h_result[i]);

        bool ok = false;
        if constexpr (kIsIntegral<T>) {
            ok = (actual == expected);
        } else {
            const double diff = std::abs(actual - expected);
            ok = diff <= tolerance * std::abs(expected) + tolerance;
        }

        if (!ok) {
            if (result.passed) {
                result.firstErrorIndex = i;
                result.expectedValue = static_cast<float>(expected);
                result.actualValue = static_cast<float>(actual);
            }
            result.passed = false;
            result.errorCount++;
        }
    }

    return result;
}

/**
 * @brief Verify Broadcast result
 * Expected: All elements equal to root's value (root+1)
 */
inline VerificationResult verifyBroadcast(
    const float* h_result,
    size_t count,
    int rootRank,
    float tolerance = RcclTestConfig::FLOAT32_TOLERANCE
) {
    VerificationResult result = {true, 0, 0, 0.0f, 0.0f};
    
    float expected = static_cast<float>(rootRank + 1);
    
    for (size_t i = 0; i < count; i++) {
        float diff = std::abs(h_result[i] - expected);
        if (diff > tolerance * std::abs(expected) + tolerance) {
            if (result.passed) {
                result.firstErrorIndex = i;
                result.expectedValue = expected;
                result.actualValue = h_result[i];
            }
            result.passed = false;
            result.errorCount++;
        }
    }
    
    return result;
}

template <typename T>
inline VerificationResult verifyBroadcast(
    const T* h_result,
    size_t count,
    int rootRank,
    double tolerance = defaultTolerance<T>()
) {
    VerificationResult result = {true, 0, 0, 0.0f, 0.0f};
    const double expected = static_cast<double>(rootRank + 1);

    for (size_t i = 0; i < count; i++) {
        const double actual = static_cast<double>(h_result[i]);
        bool ok = false;
        if constexpr (kIsIntegral<T>) {
            ok = (actual == expected);
        } else {
            const double diff = std::abs(actual - expected);
            ok = diff <= tolerance * std::abs(expected) + tolerance;
        }

        if (!ok) {
            if (result.passed) {
                result.firstErrorIndex = i;
                result.expectedValue = static_cast<float>(expected);
                result.actualValue = static_cast<float>(actual);
            }
            result.passed = false;
            result.errorCount++;
        }
    }

    return result;
}

/**
 * @brief Verify ReduceScatter result
 * Expected: Each rank gets portion of reduced result
 * If input is (rank+1), reduced sum is n*(n+1)/2 for each position
 */
inline VerificationResult verifyReduceScatter(
    const float* h_result,
    size_t recvCount,  // Elements received by this rank
    int numRanks,
    int myRank,
    float tolerance = RcclTestConfig::FLOAT32_TOLERANCE
) {
    VerificationResult result = {true, 0, 0, 0.0f, 0.0f};
    
    // Each element is sum of all ranks' contributions = n*(n+1)/2
    float expected = static_cast<float>(numRanks * (numRanks + 1)) / 2.0f;
    
    for (size_t i = 0; i < recvCount; i++) {
        float diff = std::abs(h_result[i] - expected);
        if (diff > tolerance * std::abs(expected) + tolerance) {
            if (result.passed) {
                result.firstErrorIndex = i;
                result.expectedValue = expected;
                result.actualValue = h_result[i];
            }
            result.passed = false;
            result.errorCount++;
        }
    }
    
    return result;
}

template <typename T>
inline VerificationResult verifyReduceScatter(
    const T* h_result,
    size_t recvCount,
    int numRanks,
    int /*myRank*/,
    double tolerance = defaultTolerance<T>()
) {
    VerificationResult result = {true, 0, 0, 0.0f, 0.0f};
    const double expected = static_cast<double>(numRanks) * (static_cast<double>(numRanks) + 1.0) / 2.0;

    for (size_t i = 0; i < recvCount; i++) {
        const double actual = static_cast<double>(h_result[i]);
        bool ok = false;
        if constexpr (kIsIntegral<T>) {
            ok = (actual == expected);
        } else {
            const double diff = std::abs(actual - expected);
            ok = diff <= tolerance * std::abs(expected) + tolerance;
        }

        if (!ok) {
            if (result.passed) {
                result.firstErrorIndex = i;
                result.expectedValue = static_cast<float>(expected);
                result.actualValue = static_cast<float>(actual);
            }
            result.passed = false;
            result.errorCount++;
        }
    }

    return result;
}

/**
 * @brief Verify AllToAll result
 * Expected: Rank r receives chunk c from rank c, where chunk c contains value (c+1)
 * So rank r's result should be [(0+1), (1+1), (2+1), ...] = [1, 2, 3, ...]
 */
inline VerificationResult verifyAllToAll(
    const float* h_result,
    size_t chunkSize,  // Elements per chunk (one chunk per rank)
    int numRanks,
    int myRank,
    float tolerance = RcclTestConfig::FLOAT32_TOLERANCE
) {
    VerificationResult result = {true, 0, 0, 0.0f, 0.0f};
    
    for (int srcRank = 0; srcRank < numRanks; srcRank++) {
        float expected = static_cast<float>(srcRank + 1);
        size_t startIdx = srcRank * chunkSize;
        
        for (size_t i = 0; i < chunkSize; i++) {
            size_t idx = startIdx + i;
            float diff = std::abs(h_result[idx] - expected);
            if (diff > tolerance * std::abs(expected) + tolerance) {
                if (result.passed) {
                    result.firstErrorIndex = idx;
                    result.expectedValue = expected;
                    result.actualValue = h_result[idx];
                }
                result.passed = false;
                result.errorCount++;
            }
        }
    }
    
    return result;
}

template <typename T>
inline VerificationResult verifyAllToAll(
    const T* h_result,
    size_t chunkSize,
    int numRanks,
    int /*myRank*/,
    double tolerance = defaultTolerance<T>()
) {
    VerificationResult result = {true, 0, 0, 0.0f, 0.0f};

    for (int srcRank = 0; srcRank < numRanks; srcRank++) {
        const double expected = static_cast<double>(srcRank + 1);
        const size_t startIdx = static_cast<size_t>(srcRank) * chunkSize;

        for (size_t i = 0; i < chunkSize; i++) {
            const size_t idx = startIdx + i;
            const double actual = static_cast<double>(h_result[idx]);

            bool ok = false;
            if constexpr (kIsIntegral<T>) {
                ok = (actual == expected);
            } else {
                const double diff = std::abs(actual - expected);
                ok = diff <= tolerance * std::abs(expected) + tolerance;
            }

            if (!ok) {
                if (result.passed) {
                    result.firstErrorIndex = idx;
                    result.expectedValue = static_cast<float>(expected);
                    result.actualValue = static_cast<float>(actual);
                }
                result.passed = false;
                result.errorCount++;
            }
        }
    }

    return result;
}

// ============================================================================
// Timer Utility
// ============================================================================

class GpuTimer {
public:
    GpuTimer() {
        HIP_CHECK(hipEventCreate(&start_));
        HIP_CHECK(hipEventCreate(&stop_));
    }

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    GpuTimer(GpuTimer&&) = delete;
    GpuTimer& operator=(GpuTimer&&) = delete;
    
    ~GpuTimer() {
        (void)hipEventDestroy(start_);
        (void)hipEventDestroy(stop_);
    }
    
    void start(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(start_, stream));
    }
    
    void stop(hipStream_t stream = 0) {
        HIP_CHECK(hipEventRecord(stop_, stream));
    }
    
    // Returns elapsed time in microseconds
    double elapsed() {
        HIP_CHECK(hipEventSynchronize(stop_));
        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, start_, stop_));
        return static_cast<double>(ms) * 1000.0;  // Convert to microseconds
    }
    
private:
    hipEvent_t start_, stop_;
};

/**
 * @brief Print GPU and RCCL information
 */
inline void printSystemInfo(int numGpus) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "System Information" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "Number of GPUs: " << numGpus << std::endl;
    
    for (int i = 0; i < numGpus; i++) {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, i));
        std::cout << "  GPU " << i << ": " << prop.name 
                  << " (" << (prop.totalGlobalMem / (1024*1024*1024)) << " GB)" 
                  << std::endl;
    }
    
    int rcclVersion;
    RCCL_CHECK(ncclGetVersion(&rcclVersion));
    // ncclGetVersion returns encoded version: major*10000 + minor*100 + patch
    int major = rcclVersion / 10000;
    int minor = (rcclVersion % 10000) / 100;
    int patch = rcclVersion % 100;
    std::cout << "RCCL Version: " << major << "." << minor << "." << patch 
              << " (" << rcclVersion << ")" << std::endl;
    
    std::cout << "========================================\n" << std::endl;
}

#endif // RCCL_TEST_UTILS_HPP


