// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "roles/memory_mover.h"
#include "common/hip_check.h"
#include "common/validator.h"
#include "common/health.h"
#include "common/ipc_channel.h"
#include "kernels/kernels.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <unistd.h>

// Exercises: HIP memory APIs → ROCr → Thunk (libhsakmt) → KFD ioctls →
//            amdgpu GPU page table updates → SDMA engine.
//
// Three-phase design:
//   Phase 1 (startup): Allocate a "resident set" of large buffers that stay
//     allocated for the entire run, bringing VRAM to the target pressure level.
//     This forces other roles (LIBRARY, COMPILER, COMPUTE) to compete for the
//     remaining free VRAM — testing their OOM handling and allocator resilience.
//
//   Phase 1.5 (optional — --ipc-role producer|consumer): Cross-process shared
//     GPU memory test. Two memory_mover processes share the same physical VRAM
//     via hipIpcGetMemHandle/hipIpcOpenMemHandle. Tests KFD BO reference
//     counting, GPU page table coherency across processes, bidirectional
//     read/write, and safe behavior when the allocator frees while consumer
//     still has the mapping open.
//
//   Phase 2 (main loop): Continuously allocate/copy/verify/free with varied
//     sizes (64B to 512MB) on top of the resident set. This creates a fragmented
//     memory landscape that stresses the page table and SDMA engine.
//
// When other roles run concurrently, this stresses:
//   - GPU page table updates while COMPUTE is dispatching kernels (TLB coherency)
//   - VRAM allocator contention while LIBRARY's rocBLAS allocates workspace
//   - KFD BO (buffer object) reference counting via IPC handle sharing
//   - SDMA engine contention with COMPUTE's async copies

struct ResidentBuffer {
    void* ptr;
    size_t bytes;
};

static size_t get_vram_total_bytes(int gpu_id) {
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, gpu_id) != hipSuccess)
        return 0;
    return props.totalGlobalMem;
}

static size_t get_vram_free_bytes() {
    size_t free_bytes = 0, total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess)
        return 0;
    return free_bytes;
}

static constexpr size_t MB = 1024ULL * 1024;

int run_memory_mover(const RoleConfig& config) {
    HIP_CHECK(hipSetDevice(config.gpu_id));

    printf("[MEMORY_MOVER] PID %d | GPU %d | duration %ds | vram-pressure %d%%\n",
           getpid(), config.gpu_id, config.duration_sec, config.vram_pressure_pct);

    Validator validator;
    HealthMonitor health(config.gpu_id, config.results_dir,
                         config.rss_growth_warn_kb, config.fd_growth_warn);
    IpcChannel ipc(config.results_dir);
    health.start();

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // ---- Phase 1: Resident set allocation ----
    std::vector<ResidentBuffer> resident_set;
    size_t resident_total = 0;

    if (config.vram_pressure_pct > 0) {
        size_t vram_total = get_vram_total_bytes(config.gpu_id);

        printf("[MEMORY_MOVER] VRAM total: %zu MB, target pressure: %d%%\n",
               vram_total / MB, config.vram_pressure_pct);

        // Allocate in 64MB chunks. Re-check free VRAM before each chunk so
        // multiple instances naturally share the pressure target — the second
        // instance sees what the first already allocated.
        const size_t CHUNK = 64 * MB;
        const size_t min_free = vram_total * (100 - config.vram_pressure_pct) / 100;

        (void)hipGetLastError();
        while (true) {
            size_t vram_free = get_vram_free_bytes();
            if (vram_free <= min_free) {
                printf("[MEMORY_MOVER] Resident set: VRAM free (%zu MB) <= target floor (%zu MB), stopping\n",
                       vram_free / MB, min_free / MB);
                break;
            }

            size_t can_take = vram_free - min_free;
            size_t chunk_size = (can_take < CHUNK) ? can_take : CHUNK;
            if (chunk_size < MB) break;

            void* ptr = nullptr;
            hipError_t err = hipMalloc(&ptr, chunk_size);
            if (err != hipSuccess || ptr == nullptr) {
                printf("[MEMORY_MOVER] Resident set: stopped at %zu MB (allocation failed)\n",
                       resident_total / MB);
                break;
            }

            HIP_CHECK(hipMemset(ptr, 0xAB, chunk_size));
            resident_set.push_back({ptr, chunk_size});
            resident_total += chunk_size;
        }

        size_t vram_free_after = get_vram_free_bytes();
        printf("[MEMORY_MOVER] Resident set: %zu chunks, %zu MB held | VRAM free: %zu MB\n",
               resident_set.size(), resident_total / MB,
               vram_free_after / MB);
    }

    auto start_time = std::chrono::steady_clock::now();

    // ---- Phase 1.5: Cross-process shared GPU memory (IPC) ----
    // When --ipc-role is set, two memory_mover processes share the same physical
    // VRAM buffer. The buffer and IPC handle are created once at startup by the
    // producer. The consumer re-imports (open) and closes the handle each round
    // to stress the IPC open/close lifecycle. File-based signals synchronize:
    //
    //   Startup:
    //     Producer: alloc d_shared → export IPC handle → register PID
    //     Consumer: wait for producer PID → register own PID
    //
    //   Round N (repeated for the entire duration):
    //     Producer: fill d_shared with pattern_N → signal "ready_N"
    //     Consumer: wait "ready_N" → import handle (hipIpcOpenMemHandle) →
    //               verify pattern on device → write new pattern → signal "done_N"
    //     Producer: wait "done_N" → verify consumer's pattern → signal round done
    //     Consumer: wait round done → close handle (hipIpcCloseMemHandle)
    //
    //   Cleanup:
    //     Producer: free d_shared after loop ends
    //
    //   This tests:
    //     - GPU page table coherency across two processes
    //     - IPC handle open/close lifecycle under thousands of repeated cycles
    //     - Bidirectional data integrity through shared VRAM
    //     - Page table entry creation/teardown on each consumer open/close

    int ipc_rounds = 0;
    int ipc_errors = 0;

    if (config.ipc_role != IpcRole::NONE) {
        const char* my_role_str = (config.ipc_role == IpcRole::PRODUCER) ? "producer" : "consumer";

        printf("[MEMORY_MOVER] IPC shared memory test: role=%s\n", my_role_str);

        // Startup ordering:
        //   1. Producer cleans up stale files from prior runs
        //   2. Producer does all setup (alloc, export handle)
        //   3. Producer registers PID (signals "I'm ready")
        //   4. Consumer waits for producer PID, then registers its own
        //   5. Producer waits for consumer PID
        // This ensures cleanup can't delete the consumer's PID, and the
        // consumer won't enter the loop until the handle file exists.

        constexpr size_t IPC_BUF_ELEMS = 1024 * 1024;  // 4MB shared buffer
        constexpr size_t IPC_BUF_BYTES = IPC_BUF_ELEMS * sizeof(uint32_t);

        const int ipc_block = 256;
        const int ipc_grid = (IPC_BUF_ELEMS + ipc_block - 1) / ipc_block;

        uint32_t* d_shared = nullptr;
        uint32_t* d_verify_buf = nullptr;
        int* d_err_flag = nullptr;

        if (config.ipc_role == IpcRole::PRODUCER) {
            ipc.cleanup();
            (void)hipGetLastError();

            hipError_t e = hipMalloc(&d_shared, IPC_BUF_BYTES);
            if (e != hipSuccess || !d_shared) {
                fprintf(stderr, "[MEMORY_MOVER-IPC] Producer: failed to allocate shared buffer — skipping IPC test\n");
                goto ipc_done;
            }
            if (!ipc.export_handle(d_shared, "shared")) {
                fprintf(stderr, "[MEMORY_MOVER-IPC] Producer: initial IPC handle export failed — skipping IPC test\n");
                (void)hipFree(d_shared);
                goto ipc_done;
            }

            // Register PID last — this tells the consumer that setup is complete
            // and the handle file is ready to import.
            ipc.register_pid("producer");
            if (!ipc.discover_peer_pid("consumer", 60)) {
                fprintf(stderr, "[MEMORY_MOVER-IPC] Consumer never started — skipping IPC test\n");
                (void)hipFree(d_shared);
                goto ipc_done;
            }
        } else {
            // Consumer waits for producer PID (which means handle file is ready)
            if (!ipc.discover_peer_pid("producer", 60)) {
                fprintf(stderr, "[MEMORY_MOVER-IPC] Producer never started — skipping IPC test\n");
                goto ipc_done;
            }
            ipc.register_pid("consumer");
        }

        (void)hipGetLastError();
        if (hipMalloc(&d_verify_buf, IPC_BUF_BYTES) != hipSuccess) d_verify_buf = nullptr;
        if (hipMalloc(&d_err_flag, sizeof(int)) != hipSuccess) d_err_flag = nullptr;

        while (true) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
                    >= config.duration_sec) {
                break;
            }

            if (!ipc.peer_alive()) {
                fprintf(stderr,
                    "[MEMORY_MOVER-IPC] Peer process died — exiting IPC loop after %d rounds (%d errors)\n",
                    ipc_rounds, ipc_errors);
                ipc_errors++;
                break;
            }

            (void)hipGetLastError();  // drain any sticky error

            std::string round_str = std::to_string(ipc_rounds);

            auto elapsed_now = std::chrono::steady_clock::now() - start_time;
            int elapsed_sec = static_cast<int>(
                std::chrono::duration_cast<std::chrono::seconds>(elapsed_now).count());
            bool near_end = (elapsed_sec > config.duration_sec * 9 / 10);

            if (config.ipc_role == IpcRole::PRODUCER) {
                // --- PRODUCER ---
                uint32_t pattern_a = validator.make_pattern(ipc_rounds, config.role_id, 100);

                hipLaunchKernelGGL(kernel_pattern_fill,
                                   dim3(ipc_grid), dim3(ipc_block), 0, stream,
                                   d_shared, IPC_BUF_ELEMS, pattern_a);
                HIP_LAUNCH_CHECK();
                HIP_CHECK(hipStreamSynchronize(stream));
                HIP_CHECK(hipDeviceSynchronize());

                ipc.post_signal_with_value("producer_ready_" + round_str, pattern_a);
                printf("[MEMORY_MOVER-IPC] Producer round %d: buffer filled (pattern=0x%08X), waiting for consumer...\n",
                       ipc_rounds, pattern_a);

                uint32_t consumer_pattern = 0;
                if (!ipc.wait_signal_read_value("consumer_done_" + round_str, &consumer_pattern, 60)) {
                    if (near_end) {
                        printf("[MEMORY_MOVER-IPC] Producer round %d: consumer timed out (near end of run — not an error)\n", ipc_rounds);
                    } else {
                        fprintf(stderr,
                            "[MEMORY_MOVER-IPC] Producer round %d: *** CONSUMER TIMED OUT (mid-run — possible hang) ***\n", ipc_rounds);
                        ipc_errors++;
                    }
                    ipc_rounds++;
                    continue;
                }

                // Verify consumer's pattern: copy shared buf to local buf to
                // avoid stale L2/TLB reads on cross-process IPC memory.
                HIP_CHECK(hipDeviceSynchronize());
                if (d_verify_buf && d_err_flag) {
                    HIP_CHECK(hipMemcpyAsync(d_verify_buf, d_shared, IPC_BUF_BYTES,
                                             hipMemcpyDeviceToDevice, stream));
                    HIP_CHECK(hipMemsetAsync(d_err_flag, 0, sizeof(int), stream));
                    hipLaunchKernelGGL(kernel_pattern_verify,
                                       dim3(ipc_grid), dim3(ipc_block), 0, stream,
                                       d_verify_buf, IPC_BUF_ELEMS, consumer_pattern, d_err_flag);
                    HIP_LAUNCH_CHECK();
                    HIP_CHECK(hipStreamSynchronize(stream));

                    int dev_errors = 0;
                    HIP_CHECK(hipMemcpy(&dev_errors, d_err_flag, sizeof(int), hipMemcpyDeviceToHost));
                    if (dev_errors > 0) {
                        fprintf(stderr,
                            "[MEMORY_MOVER-IPC] Producer round %d: *** CROSS-PROCESS VERIFY FAILED — %d mismatches (expected consumer pattern 0x%08X) ***\n",
                            ipc_rounds, dev_errors, consumer_pattern);
                        ipc_errors++;
                    } else {
                        printf("[MEMORY_MOVER-IPC] Producer round %d: consumer's pattern verified OK\n",
                               ipc_rounds);
                    }
                }

                // Signal consumer this round is done so it can close its handle
                ipc.post_signal("producer_round_done_" + round_str);
                ipc.wait_signal("consumer_closed_" + round_str, 30);

                ipc.cleanup_round(round_str);
                printf("[MEMORY_MOVER-IPC] Producer round %d: complete\n", ipc_rounds);

            } else {
                // --- CONSUMER ---
                uint32_t producer_pattern = 0;
                printf("[MEMORY_MOVER-IPC] Consumer round %d: waiting for producer...\n", ipc_rounds);
                if (!ipc.wait_signal_read_value("producer_ready_" + round_str, &producer_pattern, 60)) {
                    if (near_end) {
                        printf("[MEMORY_MOVER-IPC] Consumer round %d: producer timed out (near end of run — not an error)\n", ipc_rounds);
                    } else {
                        fprintf(stderr,
                            "[MEMORY_MOVER-IPC] Consumer round %d: *** PRODUCER TIMED OUT (mid-run — possible hang) ***\n", ipc_rounds);
                        ipc_errors++;
                    }
                    ipc_rounds++;
                    continue;
                }

                // Import the IPC handle (same physical VRAM, re-imported each
                // round to test the open/close lifecycle)
                void* d_imported_raw = nullptr;
                if (!ipc.import_handle(&d_imported_raw, "shared", 10)) {
                    fprintf(stderr,
                        "[MEMORY_MOVER-IPC] Consumer round %d: *** IMPORT FAILED ***\n", ipc_rounds);
                    ipc_errors++;
                    ipc_rounds++;
                    continue;
                }
                uint32_t* d_imported = static_cast<uint32_t*>(d_imported_raw);

                printf("[MEMORY_MOVER-IPC] Consumer round %d: handle imported, verifying producer's pattern (0x%08X)...\n",
                       ipc_rounds, producer_pattern);

                HIP_CHECK(hipDeviceSynchronize());
                if (d_verify_buf && d_err_flag) {
                    HIP_CHECK(hipMemcpyAsync(d_verify_buf, d_imported, IPC_BUF_BYTES,
                                             hipMemcpyDeviceToDevice, stream));
                    HIP_CHECK(hipMemsetAsync(d_err_flag, 0, sizeof(int), stream));
                    hipLaunchKernelGGL(kernel_pattern_verify,
                                       dim3(ipc_grid), dim3(ipc_block), 0, stream,
                                       d_verify_buf, IPC_BUF_ELEMS, producer_pattern, d_err_flag);
                    HIP_LAUNCH_CHECK();
                    HIP_CHECK(hipStreamSynchronize(stream));

                    int dev_errors = 0;
                    HIP_CHECK(hipMemcpy(&dev_errors, d_err_flag, sizeof(int), hipMemcpyDeviceToHost));
                    if (dev_errors > 0) {
                        fprintf(stderr,
                            "[MEMORY_MOVER-IPC] Consumer round %d: *** PRODUCER DATA VERIFY FAILED — %d mismatches ***\n",
                            ipc_rounds, dev_errors);
                        ipc_errors++;
                    } else {
                        printf("[MEMORY_MOVER-IPC] Consumer round %d: producer's pattern verified OK\n", ipc_rounds);
                    }
                }

                // Write consumer's pattern into shared buffer (bidirectional test)
                uint32_t pattern_b = validator.make_pattern(ipc_rounds, config.role_id, 200);
                hipLaunchKernelGGL(kernel_pattern_fill,
                                   dim3(ipc_grid), dim3(ipc_block), 0, stream,
                                   d_imported, IPC_BUF_ELEMS, pattern_b);
                HIP_LAUNCH_CHECK();
                HIP_CHECK(hipStreamSynchronize(stream));
                HIP_CHECK(hipDeviceSynchronize());

                ipc.post_signal_with_value("consumer_done_" + round_str, pattern_b);

                // Wait for producer to finish verifying, then close our mapping
                ipc.wait_signal("producer_round_done_" + round_str, 30);
                IpcChannel::close_handle(d_imported);
                ipc.post_signal("consumer_closed_" + round_str);

                printf("[MEMORY_MOVER-IPC] Consumer round %d: complete\n", ipc_rounds);
            }

            ipc_rounds++;
        }

        // Free pre-allocated IPC buffers
        if (d_err_flag) (void)hipFree(d_err_flag);
        if (d_verify_buf) (void)hipFree(d_verify_buf);
        if (d_shared) (void)hipFree(d_shared);

        printf("[MEMORY_MOVER-IPC] Shared memory test done: %d rounds, %d errors\n",
               ipc_rounds, ipc_errors);
    }
    ipc_done:

    // ---- Phase 2: Churn loop with larger allocation sizes ----

    const size_t sizes[] = {
        64,                       // tiny: 64 bytes
        4 * 1024,                 // small: 4KB
        64 * 1024,                // medium: 64KB
        1 * MB,                   // 1MB
        16 * MB,                  // 16MB
        64 * MB,                  // 64MB
        256 * MB,                 // 256MB
        512 * MB,                 // 512MB
    };
    constexpr int NUM_SIZES = sizeof(sizes) / sizeof(sizes[0]);

    int64_t iteration = 0;
    int alloc_failures = 0;

    while (true) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
                >= config.duration_sec) {
            break;
        }

        size_t alloc_size = sizes[iteration % NUM_SIZES];

        // For large allocations under pressure, check if there's enough free VRAM
        // to avoid guaranteed OOM. Still attempt occasionally to test error paths.
        if (alloc_size > 64 * MB && config.vram_pressure_pct > 0) {
            size_t vram_free = get_vram_free_bytes();
            if (alloc_size > vram_free && (iteration % 10) != 0) {
                alloc_size = sizes[iteration % 5]; // fall back to <= 16MB
            }
        }

        size_t elem_count = alloc_size / sizeof(uint32_t);
        if (elem_count == 0) elem_count = 1;
        size_t byte_count = elem_count * sizeof(uint32_t);

        uint32_t pattern = validator.make_pattern((int)iteration, config.role_id, 0);
        int alloc_type = iteration % 3; // rotate: device, pinned, managed

        const char* type_names[] = {"device", "pinned", "managed"};
        printf("[MEMORY_MOVER] #%" PRId64 " | Allocating %zu bytes as %s memory\n",
               iteration, byte_count, type_names[alloc_type]);

        uint32_t* d_buf = nullptr;
        uint32_t* h_buf = nullptr;

        (void)hipGetLastError();
        hipError_t alloc_err = hipSuccess;
        if (alloc_type == 0) {
            alloc_err = hipMalloc(&d_buf, byte_count);
            if (alloc_err == hipSuccess)
                alloc_err = hipHostMalloc(&h_buf, byte_count);
        } else if (alloc_type == 1) {
            alloc_err = hipMalloc(&d_buf, byte_count);
            if (alloc_err == hipSuccess)
                alloc_err = hipHostMalloc(&h_buf, byte_count, hipHostMallocMapped);
        } else {
            alloc_err = hipMallocManaged(&d_buf, byte_count);
            if (alloc_err == hipSuccess)
                h_buf = static_cast<uint32_t*>(malloc(byte_count));
        }

        if (alloc_err != hipSuccess || d_buf == nullptr || h_buf == nullptr) {
            printf("[MEMORY_MOVER] #%" PRId64 " | Allocation failed for %zu bytes (%s) — expected under pressure\n",
                   iteration, byte_count, hipGetErrorString(alloc_err));
            alloc_failures++;
            if (d_buf) { (void)hipFree(d_buf); d_buf = nullptr; }
            if (alloc_type == 2) { free(h_buf); } else if (h_buf) { (void)hipHostFree(h_buf); }
            h_buf = nullptr;
            iteration++;
            continue;
        }

        printf("[MEMORY_MOVER] #%" PRId64 " | Filling host buffer and copying H2D...\n", iteration);
        validator.fill_host(h_buf, elem_count, pattern);

        if (alloc_type == 2) {
            memcpy(d_buf, h_buf, byte_count);
        } else {
            HIP_CHECK(hipMemcpyAsync(d_buf, h_buf, byte_count,
                                     hipMemcpyHostToDevice, stream));
        }

        // Allocate helper buffers — under VRAM pressure these can fail
        int* d_error_flag = nullptr;
        uint32_t* d_buf2 = nullptr;
        uint32_t* h_readback = nullptr;
        bool skip_verify = false;

        if (hipMalloc(&d_error_flag, sizeof(int)) != hipSuccess) {
            printf("[MEMORY_MOVER] #%" PRId64 " | Helper alloc failed (error flag) — skipping verify\n", iteration);
            skip_verify = true;
        }

        int dev_errors = 0;

        if (!skip_verify) {
            printf("[MEMORY_MOVER] #%" PRId64 " | Verifying H2D on device...\n", iteration);
            HIP_CHECK(hipMemsetAsync(d_error_flag, 0, sizeof(int), stream));

            int block = 256;
            int grid = (elem_count + block - 1) / block;
            hipLaunchKernelGGL(kernel_pattern_verify,
                               dim3(grid), dim3(block), 0, stream,
                               d_buf, elem_count, pattern, d_error_flag);
            HIP_LAUNCH_CHECK();
            HIP_CHECK(hipStreamSynchronize(stream));

            HIP_CHECK(hipMemcpy(&dev_errors, d_error_flag, sizeof(int),
                                hipMemcpyDeviceToHost));
            if (dev_errors > 0) {
                fprintf(stderr, "[MEMORY_MOVER] #%" PRId64 " | *** H2D VERIFY FAILED — %d mismatches ***\n",
                        iteration, dev_errors);
            }

            printf("[MEMORY_MOVER] #%" PRId64 " | Copying D2D and verifying...\n", iteration);
            if (hipMalloc(&d_buf2, byte_count) != hipSuccess) {
                printf("[MEMORY_MOVER] #%" PRId64 " | D2D alloc failed (%zu bytes) — skipping D2D/D2H\n",
                       iteration, byte_count);
            } else {
                HIP_CHECK(hipMemcpyAsync(d_buf2, d_buf, byte_count,
                                         hipMemcpyDeviceToDevice, stream));
                HIP_CHECK(hipMemsetAsync(d_error_flag, 0, sizeof(int), stream));
                int block = 256;
                int grid = (elem_count + block - 1) / block;
                hipLaunchKernelGGL(kernel_pattern_verify,
                                   dim3(grid), dim3(block), 0, stream,
                                   d_buf2, elem_count, pattern, d_error_flag);
                HIP_LAUNCH_CHECK();
                HIP_CHECK(hipStreamSynchronize(stream));

                HIP_CHECK(hipMemcpy(&dev_errors, d_error_flag, sizeof(int),
                                    hipMemcpyDeviceToHost));
                if (dev_errors > 0) {
                    fprintf(stderr, "[MEMORY_MOVER] #%" PRId64 " | *** D2D VERIFY FAILED — %d mismatches ***\n",
                            iteration, dev_errors);
                }

                printf("[MEMORY_MOVER] #%" PRId64 " | Copying D2H and verifying on host...\n", iteration);
                if (hipHostMalloc(&h_readback, byte_count) != hipSuccess) {
                    printf("[MEMORY_MOVER] #%" PRId64 " | D2H host alloc failed — skipping D2H verify\n", iteration);
                } else {
                    HIP_CHECK(hipMemcpyAsync(h_readback, d_buf2, byte_count,
                                             hipMemcpyDeviceToHost, stream));
                    HIP_CHECK(hipStreamSynchronize(stream));
                    int host_err = validator.verify_host(h_readback, elem_count, pattern,
                                          iteration, "MEMORY_MOVER-D2H");
                    printf("[MEMORY_MOVER] #%" PRId64 " | H2D=%s D2D=%s D2H=%s\n",
                           iteration,
                           (dev_errors == 0) ? "OK" : "FAIL",
                           (dev_errors == 0) ? "OK" : "FAIL",
                           (host_err == 0) ? "OK" : "FAIL");
                }
            }
        }

        if (iteration % 100 == 0 && alloc_type == 0 && d_buf != nullptr) {
            // IPC handles only work with hipMalloc device allocations (not
            // managed or host-pinned memory). alloc_type==0 guarantees this.
            printf("[MEMORY_MOVER] #%" PRId64 " | Exporting IPC handle...\n", iteration);
            std::string tag = "mover_churn";  // fixed tag — overwrites each time
            if (!ipc.export_handle(d_buf, tag)) {
                printf("[MEMORY_MOVER] #%" PRId64 " | IPC export failed (non-fatal in churn loop)\n", iteration);
            }
        }

        printf("[MEMORY_MOVER] #%" PRId64 " | Freeing all buffers\n", iteration);

        if (h_readback) (void)hipHostFree(h_readback);
        if (d_error_flag) (void)hipFree(d_error_flag);
        if (d_buf2) (void)hipFree(d_buf2);

        if (alloc_type == 2) {
            (void)hipFree(d_buf);
            free(h_buf);
        } else {
            (void)hipFree(d_buf);
            (void)hipHostFree(h_buf);
        }

        iteration++;
    }

    // ---- Cleanup: Free resident set ----
    if (!resident_set.empty()) {
        printf("[MEMORY_MOVER] Freeing resident set (%zu chunks, %zu MB)...\n",
               resident_set.size(), resident_total / MB);
        for (auto& buf : resident_set) {
            (void)hipFree(buf.ptr);
        }
    }

    health.stop();
    (void)hipStreamDestroy(stream);

    int total_errors = validator.total_errors() + ipc_errors;
    double elapsed_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    printf("[MEMORY_MOVER] Finished: %" PRId64 " iterations, %d checks, %d errors, %d alloc_failures, %d ipc_rounds, %d ipc_errors (%.1fs)\n",
           iteration, validator.total_checks(), validator.total_errors(), alloc_failures,
           ipc_rounds, ipc_errors, elapsed_sec);

    return total_errors > 0 ? 1 : 0;
}
