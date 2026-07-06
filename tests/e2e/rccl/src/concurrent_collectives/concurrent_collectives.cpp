/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT

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

 * @brief RCCL concurrent collectives: sanity or weekly mode on independent HIP streams
 *        (no ncclGroupStart/End around collectives; grouping only for ncclCommInitRank).
 *
 * Usage:
 *   ./concurrent_collectives sanity   [iterations] [data_size_mb]
 *   ./concurrent_collectives weekly   [iterations] [data_size_mb]
 */

#include "test_utils.hpp"
#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

struct SanityTestConfig {
    int iterations = 100;
    size_t dataSizeMB = 16;
    int warmupIters = 10;
    bool verbose = true;
};

struct WeeklyTestConfig {
    size_t dataSizeMB = 256;
    int iterations = 1000;
    int warmupIters = 10;
    bool verbose = true;
};

// ============================================================================
// Sanity: AllReduce + AllGather + Broadcast
// ============================================================================

template <typename T>
bool testSanityConcurrent(ncclComm_t commAllReduce, ncclComm_t commAllGather, ncclComm_t commBroadcast,
                          int rank, int numRanks, size_t count, int iterations, int warmupIters,
                          bool verbose) {
    if (rank == 0 && verbose) {
        std::cout << "\n=== Concurrent Collectives (" << typeName<T>() << ") ===" << std::endl;
        std::cout << "Running: AllReduce + AllGather + Broadcast" << std::endl;
    }

    hipStream_t streamAllReduce = nullptr;
    hipStream_t streamAllGather = nullptr;
    hipStream_t streamBroadcast = nullptr;

    T* d_ar_send = nullptr;
    T* d_ar_recv = nullptr;
    T* d_ag_send = nullptr;
    T* d_ag_recv = nullptr;
    T* d_bc_buff = nullptr;

    const ncclDataType_t dtype = ncclType<T>();
    const size_t arBytes = count * sizeof(T);
    const size_t agSendCount = count / numRanks;
    const size_t agRecvCount = count;
    const size_t agSendBytes = agSendCount * sizeof(T);
    const size_t agRecvBytes = agRecvCount * sizeof(T);
    const size_t bcBytes = count * sizeof(T);

    std::vector<T> h_ar_recv(count);
    std::vector<T> h_ag_recv(agRecvCount);
    std::vector<T> h_bc_recv(count);

    bool allPassed = true;
    double totalTime = 0.0;

    try {
        HIP_CHECK(hipStreamCreate(&streamAllReduce));
        HIP_CHECK(hipStreamCreate(&streamAllGather));
        HIP_CHECK(hipStreamCreate(&streamBroadcast));

        HIP_CHECK(hipMalloc(&d_ar_send, arBytes));
        HIP_CHECK(hipMalloc(&d_ar_recv, arBytes));
        HIP_CHECK(hipMalloc(&d_ag_send, agSendBytes));
        HIP_CHECK(hipMalloc(&d_ag_recv, agRecvBytes));
        HIP_CHECK(hipMalloc(&d_bc_buff, bcBytes));

        initDataGPU(d_ar_send, count, rank, streamAllReduce);
        initDataGPU(d_ag_send, agSendCount, rank, streamAllGather);
        initDataGPU(d_bc_buff, count, rank, streamBroadcast);
        HIP_CHECK(hipStreamSynchronize(streamAllReduce));
        HIP_CHECK(hipStreamSynchronize(streamAllGather));
        HIP_CHECK(hipStreamSynchronize(streamBroadcast));

        GpuTimer timer;

        for (int iter = -warmupIters; iter < iterations; iter++) {
            const bool isWarmup = (iter < 0);

            initDataGPU(d_ar_send, count, rank, streamAllReduce);
            initDataGPU(d_ag_send, agSendCount, rank, streamAllGather);
            initDataGPU(d_bc_buff, count, rank, streamBroadcast);
            HIP_CHECK(hipStreamSynchronize(streamAllReduce));
            HIP_CHECK(hipStreamSynchronize(streamAllGather));
            HIP_CHECK(hipStreamSynchronize(streamBroadcast));

            if (!isWarmup) timer.start(streamAllReduce);

            RCCL_CHECK(ncclAllReduce(d_ar_send, d_ar_recv, count, dtype, ncclSum, commAllReduce,
                                     streamAllReduce));
            RCCL_CHECK(ncclAllGather(d_ag_send, d_ag_recv, agSendCount, dtype, commAllGather,
                                     streamAllGather));
            RCCL_CHECK(ncclBroadcast(d_bc_buff, d_bc_buff, count, dtype, 0, commBroadcast,
                                     streamBroadcast));

            HIP_CHECK(hipStreamSynchronize(streamAllReduce));
            HIP_CHECK(hipStreamSynchronize(streamAllGather));
            HIP_CHECK(hipStreamSynchronize(streamBroadcast));

            if (!isWarmup) {
                timer.stop(streamAllReduce);
                totalTime += timer.elapsed();
            }

            if (!isWarmup) {
                HIP_CHECK(hipMemcpy(h_ar_recv.data(), d_ar_recv, arBytes, hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(h_ag_recv.data(), d_ag_recv, agRecvBytes, hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(h_bc_recv.data(), d_bc_buff, bcBytes, hipMemcpyDeviceToHost));

                auto arResult = verifyAllReduce(h_ar_recv.data(), count, numRanks);
                auto agResult = verifyAllGather(h_ag_recv.data(), agSendCount, numRanks, rank);
                auto bcResult = verifyBroadcast(h_bc_recv.data(), count, 0);

                if (!arResult.passed || !agResult.passed || !bcResult.passed) {
                    allPassed = false;
                    if (rank == 0 && verbose) {
                        std::cout << "  Iteration " << iter << " verification:" << std::endl;
                        arResult.print("AllReduce");
                        agResult.print("AllGather");
                        bcResult.print("Broadcast");
                    }
                }
            }
        }

        if (rank == 0 && verbose) {
            double avgTime = totalTime / iterations;
            std::cout << "  Avg time: " << avgTime << " us" << std::endl;
            std::cout << "  Result: " << (allPassed ? "PASSED" : "FAILED") << std::endl;
        }
    } catch (const std::exception& e) {
        allPassed = false;
        if (rank == 0 && verbose) {
            std::cerr << "  ERROR: " << e.what() << std::endl;
        }
    }

    if (d_ar_send) (void)hipFree(d_ar_send);
    if (d_ar_recv) (void)hipFree(d_ar_recv);
    if (d_ag_send) (void)hipFree(d_ag_send);
    if (d_ag_recv) (void)hipFree(d_ag_recv);
    if (d_bc_buff) (void)hipFree(d_bc_buff);
    if (streamAllReduce) (void)hipStreamDestroy(streamAllReduce);
    if (streamAllGather) (void)hipStreamDestroy(streamAllGather);
    if (streamBroadcast) (void)hipStreamDestroy(streamBroadcast);

    return allPassed;
}

static void sanityWorkerThread(int rank, int numRanks, ncclComm_t commAllReduce,
                               ncclComm_t commAllGather, ncclComm_t commBroadcast,
                               const SanityTestConfig& config, std::atomic<unsigned>& failMask) {
    try {
        HIP_CHECK(hipSetDevice(rank));
        const size_t baseBytes = config.dataSizeMB * 1024 * 1024;

        if (rank == 0 && config.verbose) {
            std::cout << "\nTest Parameters:" << std::endl;
            std::cout << "  GPUs: " << numRanks << std::endl;
            std::cout << "  Data size: " << config.dataSizeMB << " MB" << std::endl;
            std::cout << "  Iterations: " << config.iterations << std::endl;
            std::cout << "  Warmup: " << config.warmupIters << std::endl;
        }

        auto runType = [&](auto tag) {
            using TT = decltype(tag);
            size_t count = baseBytes / sizeof(TT);
            count = (count / static_cast<size_t>(numRanks)) * static_cast<size_t>(numRanks);
            return testSanityConcurrent<TT>(commAllReduce, commAllGather, commBroadcast, rank,
                                            numRanks, count, config.iterations,
                                            config.warmupIters, config.verbose);
        };

        const bool ok_f32 = runType(float{});
        const bool ok_f64 = runType(double{});
        const bool ok_i32 = runType(int{});

        unsigned mask = 0u;
        if (!ok_f32) mask |= 1u << 0;
        if (!ok_f64) mask |= 1u << 1;
        if (!ok_i32) mask |= 1u << 2;
        if (mask) failMask.fetch_or(mask);
    } catch (const std::exception& e) {
        if (rank == 0 && config.verbose) {
            std::cerr << "Worker thread error: " << e.what() << std::endl;
        }
        failMask.fetch_or(0x7u);
    }
}

static int runSanity(int argc, char* argv[], const char* prog) {
    SanityTestConfig config;

    try {
        if (argc > 0) {
            long long iters = std::stoll(std::string(argv[0]));
            if (iters <= 0 || iters > static_cast<long long>(std::numeric_limits<int>::max())) {
                throw std::out_of_range("iterations must be in range [1, INT_MAX]");
            }
            config.iterations = static_cast<int>(iters);
        }
        if (argc > 1) {
            long long mb = std::stoll(std::string(argv[1]));
            if (mb <= 0) {
                throw std::out_of_range("data_size_mb must be >= 1");
            }
            config.dataSizeMB = static_cast<size_t>(mb);
        }
    } catch (const std::exception& e) {
        std::cerr << "Usage: " << prog << " sanity [iterations] [data_size_mb]\n"
                  << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    try {
        int numGpus = 0;
        HIP_CHECK(hipGetDeviceCount(&numGpus));

        if (numGpus < 2) {
            std::cerr << "Error: At least 2 GPUs required for collective operations." << std::endl;
            std::cerr << "Available GPUs: " << numGpus << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "========================================" << std::endl;
        std::cout << "RCCL Concurrent Collectives - SANITY TEST" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Testing concurrent collective operations on independent streams" << std::endl;
        std::cout << "without explicit ncclGroupStart/ncclGroupEnd grouping." << std::endl;

        printSystemInfo(numGpus);

        // One communicator set per concurrent stream (AllReduce / AllGather / Broadcast).
        // GroupStart/End below brackets ncclCommInitRank only.
        constexpr int kNumStreams = 3;  // AllReduce, AllGather, Broadcast
        std::vector<std::vector<ncclComm_t>> commsPerStream(
            kNumStreams, std::vector<ncclComm_t>(static_cast<size_t>(numGpus), nullptr));

        for (int s = 0; s < kNumStreams; s++) {
            ncclUniqueId uniqueId;
            RCCL_CHECK(ncclGetUniqueId(&uniqueId));
            RCCL_CHECK(ncclGroupStart());
            for (int rank = 0; rank < numGpus; rank++) {
                HIP_CHECK(hipSetDevice(rank));
                RCCL_CHECK(ncclCommInitRank(&commsPerStream[s][static_cast<size_t>(rank)], numGpus,
                                            uniqueId, rank));
            }
            RCCL_CHECK(ncclGroupEnd());
        }

        std::atomic<unsigned> failMask(0u);

        std::vector<std::thread> threads;
        for (int rank = 0; rank < numGpus; rank++) {
            threads.emplace_back(sanityWorkerThread, rank, numGpus,
                                 commsPerStream[0][static_cast<size_t>(rank)],
                                 commsPerStream[1][static_cast<size_t>(rank)],
                                 commsPerStream[2][static_cast<size_t>(rank)],
                                 std::ref(config), std::ref(failMask));
        }

        for (auto& t : threads) {
            t.join();
        }

        for (auto& commVec : commsPerStream) {
            for (auto& comm : commVec) {
                if (comm) (void)ncclCommDestroy(comm);
            }
        }

        const unsigned mask = failMask.load();
        const int failed = __builtin_popcount(mask & 0x7u);
        const int passed = 3 - failed;

        std::cout << "\n========================================" << std::endl;
        std::cout << "SANITY TEST SUMMARY" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Passed: " << passed << std::endl;
        std::cout << "Failed: " << failed << std::endl;
        std::cout << "Overall: " << (failed == 0 ? "PASSED" : "FAILED") << std::endl;
        std::cout << "========================================" << std::endl;

        return (failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

// ============================================================================
// Weekly: six collectives
// ============================================================================

template <typename T>
bool testAllCollectivesConcurrent(ncclComm_t comms[6], int rank, int numRanks, size_t count,
                                  int iterations, int warmupIters, bool verbose) {
    if (rank == 0 && verbose) {
        std::cout << "\n=== Concurrent Collectives (" << typeName<T>() << ") ===" << std::endl;
        std::cout << "Running: AllReduce + Reduce + AllGather + Broadcast + ReduceScatter + AllToAll"
                  << std::endl;
    }

    hipStream_t streams[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

    size_t arCount = count;
    size_t rCount = count;
    size_t agSendCount = count / numRanks;
    size_t agRecvCount = count;
    size_t bcCount = count;
    size_t rsSendCount = count;
    size_t rsRecvCount = count / numRanks;
    size_t a2aChunkSize = count / numRanks;
    size_t a2aTotalCount = a2aChunkSize * numRanks;

    T* d_ar_send = nullptr;
    T* d_ar_recv = nullptr;
    T* d_r_send = nullptr;
    T* d_r_recv = nullptr;
    T* d_ag_send = nullptr;
    T* d_ag_recv = nullptr;
    T* d_bc_buff = nullptr;
    T* d_rs_send = nullptr;
    T* d_rs_recv = nullptr;
    T* d_a2a_send = nullptr;
    T* d_a2a_recv = nullptr;

    std::vector<T> h_ar_recv(arCount);
    std::vector<T> h_r_recv(rCount);
    std::vector<T> h_ag_recv(agRecvCount);
    std::vector<T> h_bc_recv(bcCount);
    std::vector<T> h_rs_recv(rsRecvCount);
    std::vector<T> h_a2a_recv(a2aTotalCount);

    int rootRank = 0;
    GpuTimer timer;
    std::vector<double> times;
    times.reserve(static_cast<size_t>(iterations));
    bool allPassed = true;
    int failedIters = 0;
    const ncclDataType_t dtype = ncclType<T>();

    try {
        for (int i = 0; i < 6; i++) {
            HIP_CHECK(hipStreamCreate(&streams[i]));
        }

        HIP_CHECK(hipMalloc(&d_ar_send, arCount * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_ar_recv, arCount * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_r_send, rCount * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_r_recv, rCount * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_ag_send, agSendCount * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_ag_recv, agRecvCount * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_bc_buff, bcCount * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_rs_send, rsSendCount * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_rs_recv, rsRecvCount * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_a2a_send, a2aTotalCount * sizeof(T)));
        HIP_CHECK(hipMalloc(&d_a2a_recv, a2aTotalCount * sizeof(T)));

        for (int iter = -warmupIters; iter < iterations; iter++) {
            bool isWarmup = (iter < 0);

            initDataGPU(d_ar_send, arCount, rank, streams[0]);
            initDataGPU(d_r_send, rCount, rank, streams[1]);
            initDataGPU(d_ag_send, agSendCount, rank, streams[2]);
            initDataGPU(d_bc_buff, bcCount, rank, streams[3]);
            initDataGPU(d_rs_send, rsSendCount, rank, streams[4]);
            initDataGPU(d_a2a_send, a2aTotalCount, rank, streams[5]);

            for (int i = 0; i < 6; i++) {
                HIP_CHECK(hipStreamSynchronize(streams[i]));
            }

            if (!isWarmup) timer.start(streams[0]);

            RCCL_CHECK(ncclAllReduce(d_ar_send, d_ar_recv, arCount, dtype, ncclSum, comms[0], streams[0]));
            RCCL_CHECK(ncclReduce(d_r_send, d_r_recv, rCount, dtype, ncclSum, rootRank, comms[1], streams[1]));
            RCCL_CHECK(ncclAllGather(d_ag_send, d_ag_recv, agSendCount, dtype, comms[2], streams[2]));
            RCCL_CHECK(ncclBroadcast(d_bc_buff, d_bc_buff, bcCount, dtype, rootRank, comms[3], streams[3]));
            RCCL_CHECK(ncclReduceScatter(d_rs_send, d_rs_recv, rsRecvCount, dtype, ncclSum, comms[4], streams[4]));
            RCCL_CHECK(ncclAlltoAll(d_a2a_send, d_a2a_recv, a2aChunkSize, dtype, comms[5], streams[5]));

            for (int i = 0; i < 6; i++) {
                HIP_CHECK(hipStreamSynchronize(streams[i]));
            }

            if (!isWarmup) {
                timer.stop(streams[0]);
                times.push_back(timer.elapsed());
            }

            if (!isWarmup) {
                HIP_CHECK(hipMemcpy(h_ar_recv.data(), d_ar_recv, arCount * sizeof(T), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(h_ag_recv.data(), d_ag_recv, agRecvCount * sizeof(T), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(h_bc_recv.data(), d_bc_buff, bcCount * sizeof(T), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(h_rs_recv.data(), d_rs_recv, rsRecvCount * sizeof(T), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(h_a2a_recv.data(), d_a2a_recv, a2aTotalCount * sizeof(T),
                                  hipMemcpyDeviceToHost));
                if (rank == rootRank) {
                    HIP_CHECK(hipMemcpy(h_r_recv.data(), d_r_recv, rCount * sizeof(T), hipMemcpyDeviceToHost));
                }

                auto arResult = verifyAllReduce(h_ar_recv.data(), arCount, numRanks);
                VerificationResult rResult = {true, 0, 0, 0.0f, 0.0f};
                if (rank == rootRank) {
                    rResult = verifyAllReduce(h_r_recv.data(), rCount, numRanks);
                }
                auto agResult = verifyAllGather(h_ag_recv.data(), agSendCount, numRanks, rank);
                auto bcResult = verifyBroadcast(h_bc_recv.data(), bcCount, rootRank);
                auto rsResult = verifyReduceScatter(h_rs_recv.data(), rsRecvCount, numRanks, rank);
                auto a2aResult = verifyAllToAll(h_a2a_recv.data(), a2aChunkSize, numRanks, rank);

                bool iterPassed = arResult.passed && agResult.passed && bcResult.passed && rsResult.passed &&
                                  a2aResult.passed && (rank != rootRank || rResult.passed);

                if (!iterPassed) {
                    allPassed = false;
                    failedIters++;
                    if (rank == 0 && verbose && failedIters <= 3) {
                        std::cout << "  Iteration " << iter << " FAILED:" << std::endl;
                        if (!arResult.passed) arResult.print("AllReduce");
                        if (rank == rootRank && !rResult.passed) rResult.print("Reduce");
                        if (!agResult.passed) agResult.print("AllGather");
                        if (!bcResult.passed) bcResult.print("Broadcast");
                        if (!rsResult.passed) rsResult.print("ReduceScatter");
                        if (!a2aResult.passed) a2aResult.print("AllToAll");
                    }
                }
            }
        }

        if (rank == 0 && verbose) {
            double avgTime =
                times.empty() ? 0.0 : std::accumulate(times.begin(), times.end(), 0.0) / times.size();

            std::cout << "  Statistics over " << iterations << " iterations:" << std::endl;
            std::cout << "    Avg time: " << avgTime << " us" << std::endl;
            std::cout << "    Failed iterations: " << failedIters << "/" << iterations << std::endl;
            std::cout << "  Result: " << (allPassed ? "PASSED" : "FAILED") << std::endl;
        }
    } catch (const std::exception& e) {
        allPassed = false;
        if (rank == 0 && verbose) {
            std::cerr << "  ERROR: " << e.what() << std::endl;
        }
    }

    if (d_ar_send) (void)hipFree(d_ar_send);
    if (d_ar_recv) (void)hipFree(d_ar_recv);
    if (d_r_send) (void)hipFree(d_r_send);
    if (d_r_recv) (void)hipFree(d_r_recv);
    if (d_ag_send) (void)hipFree(d_ag_send);
    if (d_ag_recv) (void)hipFree(d_ag_recv);
    if (d_bc_buff) (void)hipFree(d_bc_buff);
    if (d_rs_send) (void)hipFree(d_rs_send);
    if (d_rs_recv) (void)hipFree(d_rs_recv);
    if (d_a2a_send) (void)hipFree(d_a2a_send);
    if (d_a2a_recv) (void)hipFree(d_a2a_recv);

    for (int i = 0; i < 6; i++) {
        if (streams[i]) (void)hipStreamDestroy(streams[i]);
    }

    return allPassed;
}

static void weeklyWorkerThread(int rank, int numRanks, ncclComm_t comms[6],
                               const WeeklyTestConfig& config, std::atomic<unsigned>& failMask) {
    try {
        HIP_CHECK(hipSetDevice(rank));

        const size_t baseBytes = config.dataSizeMB * 1024 * 1024;

        if (rank == 0 && config.verbose) {
            std::cout << "\nWeekly Test Parameters:" << std::endl;
            std::cout << "  GPUs: " << numRanks << std::endl;
            std::cout << "  Base data size: " << config.dataSizeMB << " MB" << std::endl;
            std::cout << "  Iterations: " << config.iterations << std::endl;
        }

        auto runType = [&](auto tag) {
            using TT = decltype(tag);
            size_t count = baseBytes / sizeof(TT);
            count = (count / static_cast<size_t>(numRanks)) * static_cast<size_t>(numRanks);
            return testAllCollectivesConcurrent<TT>(comms, rank, numRanks, count, config.iterations,
                                                    config.warmupIters, config.verbose);
        };

        const bool ok_f32 = runType(float{});
        const bool ok_f64 = runType(double{});
        const bool ok_i32 = runType(int{});

        unsigned mask = 0u;
        if (!ok_f32) mask |= 1u << 0;
        if (!ok_f64) mask |= 1u << 1;
        if (!ok_i32) mask |= 1u << 2;
        if (mask) failMask.fetch_or(mask);
    } catch (const std::exception& e) {
        if (rank == 0 && config.verbose) {
            std::cerr << "Worker thread error: " << e.what() << std::endl;
        }
        failMask.fetch_or(0x7u);
    }
}

static int runWeekly(int argc, char* argv[], const char* prog) {
    WeeklyTestConfig config;

    try {
        if (argc > 0) {
            long long iters = std::stoll(std::string(argv[0]));
            if (iters <= 0 || iters > static_cast<long long>(std::numeric_limits<int>::max())) {
                throw std::out_of_range("iterations must be in range [1, INT_MAX]");
            }
            config.iterations = static_cast<int>(iters);
        }
        if (argc > 1) {
            long long mb = std::stoll(std::string(argv[1]));
            if (mb <= 0) {
                throw std::out_of_range("data_size_mb must be >= 1");
            }
            config.dataSizeMB = static_cast<size_t>(mb);
        }
    } catch (const std::exception& e) {
        std::cerr << "Usage: " << prog << " weekly [iterations] [data_size_mb]\n"
                  << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    try {
        int numGpus = 0;
        HIP_CHECK(hipGetDeviceCount(&numGpus));

        if (numGpus < 2) {
            std::cerr << "Error: At least 2 GPUs required." << std::endl;
            std::cerr << "Available GPUs: " << numGpus << std::endl;
            return EXIT_FAILURE;
        }

        std::cout << "================================================" << std::endl;
        std::cout << "RCCL Concurrent Collectives - WEEKLY TEST" << std::endl;
        std::cout << "================================================" << std::endl;
        std::cout << "Concurrent collective operations on independent streams" << std::endl;
        std::cout << "without explicit ncclGroupStart/ncclGroupEnd grouping." << std::endl;

        printSystemInfo(numGpus);

        // One communicator set per concurrent stream (6 collectives total).
        // GroupStart/End below brackets ncclCommInitRank only.
        constexpr int kNumStreams = 6;
        std::vector<std::vector<ncclComm_t>> commsPerStream(
            kNumStreams, std::vector<ncclComm_t>(static_cast<size_t>(numGpus), nullptr));

        for (int s = 0; s < kNumStreams; s++) {
            ncclUniqueId uniqueId;
            RCCL_CHECK(ncclGetUniqueId(&uniqueId));
            RCCL_CHECK(ncclGroupStart());
            for (int rank = 0; rank < numGpus; rank++) {
                HIP_CHECK(hipSetDevice(rank));
                RCCL_CHECK(ncclCommInitRank(&commsPerStream[s][static_cast<size_t>(rank)], numGpus,
                                            uniqueId, rank));
            }
            RCCL_CHECK(ncclGroupEnd());
        }

        // Pack per-rank comm pointers into a contiguous array so each worker thread receives
        // the 6 comms it owns (one per stream) without sharing any comm across streams.
        std::vector<std::array<ncclComm_t, 6>> rankComms(static_cast<size_t>(numGpus));
        for (int rank = 0; rank < numGpus; rank++) {
            for (int s = 0; s < kNumStreams; s++) {
                rankComms[static_cast<size_t>(rank)][s] =
                    commsPerStream[s][static_cast<size_t>(rank)];
            }
        }

        std::atomic<unsigned> failMask(0u);

        std::vector<std::thread> threads;
        for (int rank = 0; rank < numGpus; rank++) {
            threads.emplace_back(weeklyWorkerThread, rank, numGpus,
                                 rankComms[static_cast<size_t>(rank)].data(), std::ref(config),
                                 std::ref(failMask));
        }

        for (auto& t : threads) {
            t.join();
        }

        for (auto& commVec : commsPerStream) {
            for (auto& comm : commVec) {
                if (comm) (void)ncclCommDestroy(comm);
            }
        }

        const unsigned mask = failMask.load();
        const int failed = __builtin_popcount(mask & 0x7u);
        const int passed = 3 - failed;

        std::cout << "\n================================================" << std::endl;
        std::cout << "WEEKLY TEST SUMMARY" << std::endl;
        std::cout << "================================================" << std::endl;
        std::cout << "Passed: " << passed << std::endl;
        std::cout << "Failed: " << failed << std::endl;
        std::cout << "Overall: " << (failed == 0 ? "PASSED" : "FAILED") << std::endl;
        std::cout << "================================================" << std::endl;

        return (failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

static void printUsage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " sanity   [iterations] [data_size_mb]   (defaults: 100 iters, 16 MB)\n"
        << "  " << prog << " weekly   [iterations] [data_size_mb]   (defaults: 1000 iters, 256 MB)\n"
        << "  " << prog << " --help | -h | help\n"
        << "\n"
        << "Model: ncclGroupStart/End only around ncclCommInitRank; collectives use separate HIP\n"
        << "streams with no collective-side grouping. Each mode runs float32, float64, int32;\n"
        << "warmup iterations are 10 (fixed).\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    const std::string mode(argv[1]);
    if (mode == "-h" || mode == "--help" || mode == "help") {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }
    if (mode == "sanity" || mode == "--sanity") {
        return runSanity(argc - 2, argv + 2, argv[0]);
    }
    if (mode == "weekly" || mode == "--weekly") {
        return runWeekly(argc - 2, argv + 2, argv[0]);
    }

    std::cerr << "Unknown mode: " << mode << "\n";
    printUsage(argv[0]);
    return EXIT_FAILURE;
}
