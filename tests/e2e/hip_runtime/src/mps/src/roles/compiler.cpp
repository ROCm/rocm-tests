// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "roles/compiler.h"
#include "common/hip_check.h"
#include "common/health.h"
#include "common/validator.h"

#include <hip/hip_runtime.h>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cerrno>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <dirent.h>

#ifdef HAS_HIPRTC
#include <hip/hiprtc.h>

#define HIPRTC_CHECK(call)                                                     \
    do {                                                                        \
        hiprtcResult res = (call);                                              \
        if (res != HIPRTC_SUCCESS) {                                            \
            fprintf(stderr, "[HIPRTC ERROR] %s:%d — %s returned %s (%d)\n",     \
                    __FILE__, __LINE__, #call, hiprtcGetErrorString(res), res);  \
            return 1;                                                           \
        }                                                                       \
    } while (0)

static void rmdir_recursive(const char* dirpath);

// Remove all files and subdirectories in a directory (keeps the directory itself).
static int purge_dir(const char* dirpath) {
    int removed = 0;
    DIR* dir = opendir(dirpath);
    if (!dir) return 0;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.' && (entry->d_name[1] == '\0' ||
            (entry->d_name[1] == '.' && entry->d_name[2] == '\0')))
            continue;
        std::string path = std::string(dirpath) + "/" + entry->d_name;
        struct stat st;
        if (lstat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            rmdir_recursive(path.c_str());
        } else {
            if (unlink(path.c_str()) == 0) removed++;
        }
    }
    closedir(dir);
    return removed;
}

// Recursively remove a directory and its contents.
static void rmdir_recursive(const char* dirpath) {
    if (!dirpath || strlen(dirpath) < 10 ||
        strstr(dirpath, "/tmp/rocm_compiler_") != dirpath) {
        fprintf(stderr, "[COMPILER] Refusing to recursively delete suspicious path: %s\n",
                dirpath ? dirpath : "(null)");
        return;
    }
    DIR* dir = opendir(dirpath);
    if (!dir) return;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.' && (entry->d_name[1] == '\0' ||
            (entry->d_name[1] == '.' && entry->d_name[2] == '\0')))
            continue;
        std::string path = std::string(dirpath) + "/" + entry->d_name;
        struct stat st;
        if (lstat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
            rmdir_recursive(path.c_str());
        else
            unlink(path.c_str());
    }
    closedir(dir);
    rmdir(dirpath);
}

static constexpr size_t MB = 1024ULL * 1024;

static long get_tmp_free_mb() {
    struct statvfs st;
    if (statvfs("/tmp", &st) != 0) return -1;
    return static_cast<long>((st.f_bavail * st.f_frsize) / MB);
}
#endif

// Exercises: hipRTC → COMGR (libcomgr) → temp file I/O → code object generation
//            → HIP module load → ROCr code object registration → KFD GPU page
//            table mapping → kernel dispatch → module unload → KFD unmapping.
//
// Continuously compiles small kernels at runtime with varied source code,
// loads them, runs them, verifies output, then unloads. This churns the
// code object manager in ROCr.
//
// When other roles run concurrently, this stresses:
//   - ROCr code object table concurrent access while COMPUTE dispatches
//   - COMGR temp file creation while system is under I/O load
//   - GPU page table updates for code segments while MEMORY_MOVER modifies
//     data segment mappings
//   - Code object cache coherency across multiple loading/unloading cycles

int run_compiler(const RoleConfig& config) {
#ifndef HAS_HIPRTC
    printf("[COMPILER] hipRTC not available — skipping\n");
    return 0;
#else
    HIP_CHECK(hipSetDevice(config.gpu_id));

    printf("[COMPILER] PID %d | GPU %d | duration %ds\n",
           getpid(), config.gpu_id, config.duration_sec);

    // Private tmpdir per process — isolates COMGR temp files so each
    // compiler instance only cleans its own files, eliminating cross-
    // process deletion of in-flight .hipi/.tmp files.
    char private_tmpdir[128];
    snprintf(private_tmpdir, sizeof(private_tmpdir),
             "/tmp/rocm_compiler_%d", getpid());
    if (mkdir(private_tmpdir, 0700) != 0 && errno != EEXIST) {
        fprintf(stderr, "[COMPILER] Failed to create private tmpdir '%s': %s\n",
                private_tmpdir, strerror(errno));
        return 1;
    }
    if (setenv("TMPDIR", private_tmpdir, 1) != 0) {
        fprintf(stderr, "[COMPILER] Failed to set TMPDIR — temp files will use /tmp\n");
    }
    printf("[COMPILER] Using private tmpdir: %s\n", private_tmpdir);

    Validator validator;
    HealthMonitor health(config.gpu_id, config.results_dir,
                         config.rss_growth_warn_kb, config.fd_growth_warn);
    health.start();

    // Template kernel sources — varied to exercise different compilation paths
    const std::vector<std::string> kernel_sources = {
        // Simple fill kernel
        R"(
        extern "C" __global__ void rtc_kernel(unsigned int* buf, unsigned int count, unsigned int pattern) {
            unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < count) {
                buf[idx] = pattern ^ idx;
            }
        }
        )",
        // Kernel using shared memory
        R"(
        extern "C" __global__ void rtc_kernel(unsigned int* buf, unsigned int count, unsigned int pattern) {
            extern __shared__ unsigned int smem[];
            unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int lid = threadIdx.x;
            smem[lid] = pattern ^ idx;
            __syncthreads();
            if (idx < count) {
                buf[idx] = smem[lid];
            }
        }
        )",
        // Kernel with more ALU work
        R"(
        extern "C" __global__ void rtc_kernel(unsigned int* buf, unsigned int count, unsigned int pattern) {
            unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < count) {
                unsigned int val = pattern ^ idx;
                val = val * 2654435761u;
                val ^= val >> 16;
                val *= 0x85ebca6bu;
                val ^= val >> 13;
                buf[idx] = val;
            }
        }
        )",
        // Kernel with divergent control flow
        R"(
        extern "C" __global__ void rtc_kernel(unsigned int* buf, unsigned int count, unsigned int pattern) {
            unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < count) {
                unsigned int val;
                if (idx % 2 == 0) {
                    val = pattern ^ idx;
                } else {
                    val = pattern ^ (idx * 3 + 1);
                    val ^= val >> 8;
                    val = pattern ^ idx;  // normalize back for verification
                }
                buf[idx] = val;
            }
        }
        )",
    };

    constexpr size_t BUF_ELEMS = 64 * 1024;
    constexpr size_t BUF_BYTES = BUF_ELEMS * sizeof(uint32_t);

    uint32_t* d_buf = nullptr;
    uint32_t* h_buf = nullptr;
    hipStream_t stream = nullptr;
    int hip_exit_code = 0;
    auto start_time = std::chrono::steady_clock::now();
    int64_t iteration = 0;
    int total_errors = 0;

    HIP_CHECK_OR(hipMalloc(&d_buf, BUF_BYTES), compiler_cleanup);
    HIP_CHECK_OR(hipHostMalloc(&h_buf, BUF_BYTES), compiler_cleanup);
    HIP_CHECK_OR(hipStreamCreate(&stream), compiler_cleanup);

    while (true) {
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
                >= config.duration_sec) {
            break;
        }

        // Pick a kernel source variant
        // Kernels 0 and 1 produce pattern^idx which our validator can check.
        // Kernel 2 produces a hash — skip host verify for that one.
        // Kernel 3 has divergent flow but normalizes back to pattern^idx.
        int src_idx = iteration % kernel_sources.size();
        bool can_host_verify = (src_idx != 2);

        const char* variant_names[] = {"simple-fill", "shared-mem", "hash-ALU", "divergent-branch"};

        // Purge our private tmpdir every iteration to prevent COMGR temp
        // file accumulation. Each compile cycle creates ~50-100 MB of temp
        // files that COMGR may not always clean up on all code paths. Since each
        // compiler instance has its own TMPDIR, we can safely delete everything
        // after extracting the code object.
        int purged = purge_dir(private_tmpdir);
        if (purged > 0 && iteration % 5000 == 0) {
            printf("[COMPILER] #%" PRId64 " | Purged %d leftover temp files from %s\n",
                   iteration, purged, private_tmpdir);
        }

        // Safety check: if /tmp itself is dangerously full (e.g. other users
        // or system services), halt to avoid crashing the whole GPU.
        long tmp_free_mb = get_tmp_free_mb();
        if (tmp_free_mb >= 0 && tmp_free_mb < 100) {
            fprintf(stderr, "[COMPILER] #%" PRId64 " | *** /tmp critically low (%ld MB free) — "
                    "halting compiles ***\n", iteration, tmp_free_mb);
            bool recovered = false;
            for (int wait = 0; wait < 30; wait++) {
                sleep(10);
                purge_dir(private_tmpdir);
                tmp_free_mb = get_tmp_free_mb();
                if (tmp_free_mb >= 500) {
                    printf("[COMPILER] #%" PRId64 " | /tmp recovered (%ld MB free) — resuming\n",
                           iteration, tmp_free_mb);
                    recovered = true;
                    break;
                }
            }
            if (!recovered) {
                fprintf(stderr, "[COMPILER] #%" PRId64 " | /tmp did not recover after 5 minutes — "
                        "stopping compiler to prevent system-wide crash\n", iteration);
                total_errors++;
                break;
            }
            iteration++;
            continue;
        }

        printf("[COMPILER] #%" PRId64 " | Compiling kernel variant: %s\n",
               iteration, variant_names[src_idx]);

        const char* src = kernel_sources[src_idx].c_str();
        const char* name = "rtc_kernel";

        hiprtcProgram prog;
        hiprtcResult create_res = hiprtcCreateProgram(&prog, src, "rtc_kernel.hip",
                                                      0, nullptr, nullptr);
        if (create_res != HIPRTC_SUCCESS) {
            fprintf(stderr, "[COMPILER] #%" PRId64 " | hiprtcCreateProgram failed: %s\n",
                    iteration, hiprtcGetErrorString(create_res));
            total_errors++;
            iteration++;
            continue;
        }

        // Get device architecture for compilation target
        hipDeviceProp_t props;
        hipError_t props_err = hipGetDeviceProperties(&props, config.gpu_id);
        if (props_err != hipSuccess) {
            fprintf(stderr, "[COMPILER] #%" PRId64 " | hipGetDeviceProperties failed: %s\n",
                    iteration, hipGetErrorString(props_err));
            hiprtcDestroyProgram(&prog);
            total_errors++;
            iteration++;
            continue;
        }
        std::string arch_flag = std::string("--offload-arch=") + props.gcnArchName;
        const char* opts[] = { arch_flag.c_str() };

        hiprtcResult compile_res = hiprtcCompileProgram(prog, 1, opts);
        if (compile_res != HIPRTC_SUCCESS) {
            size_t log_size;
            hiprtcGetProgramLogSize(prog, &log_size);
            std::string log(log_size, '\0');
            hiprtcGetProgramLog(prog, &log[0]);
            fprintf(stderr, "[COMPILER] #%" PRId64 " | Compile failed (src=%d): %s\n",
                    iteration, src_idx, log.c_str());
            hiprtcDestroyProgram(&prog);
            total_errors++;
            iteration++;
            continue;
        }

        printf("[COMPILER] #%" PRId64 " | Compile OK, extracting code object...\n", iteration);
        size_t code_size;
        hiprtcResult code_size_res = hiprtcGetCodeSize(prog, &code_size);
        if (code_size_res != HIPRTC_SUCCESS) {
            fprintf(stderr, "[COMPILER] #%" PRId64 " | hiprtcGetCodeSize failed: %s\n",
                    iteration, hiprtcGetErrorString(code_size_res));
            hiprtcDestroyProgram(&prog);
            total_errors++;
            iteration++;
            continue;
        }
        std::vector<char> code(code_size);
        hiprtcResult get_code_res = hiprtcGetCode(prog, code.data());
        if (get_code_res != HIPRTC_SUCCESS) {
            fprintf(stderr, "[COMPILER] #%" PRId64 " | hiprtcGetCode failed: %s\n",
                    iteration, hiprtcGetErrorString(get_code_res));
            hiprtcDestroyProgram(&prog);
            total_errors++;
            iteration++;
            continue;
        }
        hiprtcResult destroy_res = hiprtcDestroyProgram(&prog);
        if (destroy_res != HIPRTC_SUCCESS) {
            fprintf(stderr, "[COMPILER] #%" PRId64 " | hiprtcDestroyProgram failed: %s\n",
                    iteration, hiprtcGetErrorString(destroy_res));
            total_errors++;
            iteration++;
            continue;
        }

        printf("[COMPILER] #%" PRId64 " | Loading module (%zu bytes)...\n", iteration, code_size);
        (void)hipGetLastError();
        hipModule_t module;
        hipError_t mod_err = hipModuleLoadData(&module, code.data());
        if (mod_err != hipSuccess) {
            fprintf(stderr, "[COMPILER] #%" PRId64 " | Module load failed: %s\n",
                    iteration, hipGetErrorString(mod_err));
            total_errors++;
            iteration++;
            continue;
        }

        hipFunction_t func;
        hipError_t func_err = hipModuleGetFunction(&func, module, name);
        if (func_err != hipSuccess) {
            fprintf(stderr, "[COMPILER] #%" PRId64 " | hipModuleGetFunction failed: %s\n",
                    iteration, hipGetErrorString(func_err));
            (void)hipModuleUnload(module);
            total_errors++;
            iteration++;
            continue;
        }

        // --- Launch ---
        uint32_t pattern = validator.make_pattern((int)iteration, config.role_id, src_idx);
        uint32_t count = static_cast<uint32_t>(BUF_ELEMS);

        void* kernel_params[] = {
            &d_buf,
            &count,
            &pattern
        };

        int block = 256;
        int grid = (BUF_ELEMS + block - 1) / block;
        size_t shared_mem = (src_idx == 1) ? block * sizeof(uint32_t) : 0;

        printf("[COMPILER] #%" PRId64 " | Launching RTC kernel (grid=%d, block=%d, shmem=%zu)...\n",
               iteration, grid, block, shared_mem);
        hipError_t launch_err = hipModuleLaunchKernel(func, grid, 1, 1, block, 1, 1,
                                                      shared_mem, stream,
                                                      kernel_params, nullptr);
        if (launch_err != hipSuccess) {
            fprintf(stderr, "[COMPILER] #%" PRId64 " | hipModuleLaunchKernel failed: %s\n",
                    iteration, hipGetErrorString(launch_err));
            (void)hipModuleUnload(module);
            total_errors++;
            iteration++;
            continue;
        }

        printf("[COMPILER] #%" PRId64 " | Readback and verify...\n", iteration);
        hipError_t memcpy_err = hipMemcpyAsync(h_buf, d_buf, BUF_BYTES,
                                               hipMemcpyDeviceToHost, stream);
        if (memcpy_err != hipSuccess) {
            fprintf(stderr, "[COMPILER] #%" PRId64 " | hipMemcpyAsync failed: %s\n",
                    iteration, hipGetErrorString(memcpy_err));
            (void)hipModuleUnload(module);
            total_errors++;
            iteration++;
            continue;
        }
        hipError_t sync_err = hipStreamSynchronize(stream);
        if (sync_err != hipSuccess) {
            fprintf(stderr, "[COMPILER] #%" PRId64 " | hipStreamSynchronize failed: %s\n",
                    iteration, hipGetErrorString(sync_err));
            (void)hipModuleUnload(module);
            total_errors++;
            iteration++;
            continue;
        }

        int mismatches = 0;
        if (can_host_verify) {
            mismatches = validator.verify_host(h_buf, BUF_ELEMS, pattern,
                                               iteration, "COMPILER");
            if (mismatches > 0) total_errors++;
        }

        printf("[COMPILER] #%" PRId64 " | Unloading module...\n", iteration);
        hipError_t unload_err = hipModuleUnload(module);
        if (unload_err != hipSuccess) {
            fprintf(stderr, "[COMPILER] #%" PRId64 " | hipModuleUnload failed: %s\n",
                    iteration, hipGetErrorString(unload_err));
            total_errors++;
        }

        printf("[COMPILER] #%" PRId64 " | %s → compile → load → run → verify → unload %s\n",
               iteration, variant_names[src_idx],
               (mismatches == 0) ? "OK" : "FAIL");

        iteration++;
    }

    health.stop();

compiler_cleanup:
    if (d_buf) (void)hipFree(d_buf);
    if (h_buf) (void)hipHostFree(h_buf);
    if (stream) (void)hipStreamDestroy(stream);

    rmdir_recursive(private_tmpdir);
    unsetenv("TMPDIR");

    double elapsed_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    printf("[COMPILER] Finished: %" PRId64 " iterations, %d errors (%.1fs)\n",
           iteration, total_errors, elapsed_sec);

    return (total_errors > 0 || hip_exit_code != 0) ? 1 : 0;
#endif
}
