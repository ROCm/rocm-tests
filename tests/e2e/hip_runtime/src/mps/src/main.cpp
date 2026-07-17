// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/roles.h"
#include "roles/role_interface.h"
#include "roles/compute.h"
#include "roles/memory_mover.h"
#include "roles/library.h"
#include "roles/compiler.h"
#include "roles/monitor.h"
#include "roles/profiler.h"
#include "roles/ipc_xfer.h"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

static void print_usage(const char* prog) {
    printf(
        "ROCK MPS Test — Multi-process integration test\n"
        "\n"
        "Usage: %s --role <role> [options]\n"
        "\n"
        "Roles:\n"
        "  compute       — HIP kernel launch loop (HIP → ROCr → KFD → HW compute)\n"
        "  memory_mover  — Memory alloc/free/copy/IPC (HIP → Thunk → KFD → SDMA)\n"
        "  library       — hipBLASLt GEMM with varied sizes (hipBLASLt → Tensile → HIP)\n"
        "  compiler      — hipRTC compile/load/run/unload (hipRTC → COMGR → ROCr)\n"
        "  monitor       — AMD SMI GPU queries (libamd_smi → sysfs → amdgpu)\n"
        "  profiler      — Event-based GPU profiling (hipExtLaunchKernelGGL → HSA signals)\n"
        "  ipc_xfer     — Multi-GPU peer copy + direct access (xGMI → KFD → SDMA)\n"
        "\n"
        "Options:\n"
        "  --gpu <id>          GPU device ID (default: 0)\n"
        "  --duration <sec>    Run duration in seconds (default: 60)\n"
        "  --results <dir>     Results output directory (default: /tmp/rock_mps_test)\n"
        "  --peer-gpu <id>     Peer GPU for ipc_xfer role (default: auto-pick next GPU)\n"
        "  --vram-pressure <%%> Target VRAM usage for memory_mover resident set (0-90, default: 0=off)\n"
        "  --ipc-role <role>   IPC shared memory role: producer|consumer (memory_mover only)\n"
        "  --anomaly-ratio <x> Profiler: flag anomaly when kernel time > x * baseline (default: 5.0)\n"
        "  --severe-ratio <x>  Profiler: flag severe anomaly at this ratio (default: 50.0)\n"
        "  --anomaly-pct <%%>  Profiler: FAIL if anomaly rate exceeds this %% (default: 1.0)\n"
        "  --severe-pct <%%>   Profiler: FAIL if severe rate exceeds this %% (default: 0.001)\n"
        "  --rss-warn <MB>     Health: warn if RSS grows by this many MB (default: 100)\n"
        "  --fd-warn <n>       Health: warn if FD count grows by this much (default: 50)\n"
        "  --verbose           Enable verbose per-iteration output\n"
        "  --help              Show this help\n"
        "\n"
        "Examples:\n"
        "  %s --role compute --gpu 0 --duration 60 --verbose\n"
        "  %s --role memory_mover --gpu 0 --duration 300 --vram-pressure 60\n"
        "\n"
        "  Run the full suite on one GPU:  see scripts/run_all_roles.sh\n"
        "  Run on all GPUs:                see scripts/run_all_gpus.sh\n"
        "\n",
        prog, prog, prog);
}

int main(int argc, char** argv) {
    RoleConfig config;
    Role role = Role::UNKNOWN;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--role") == 0 && i + 1 < argc) {
            const char* role_str = argv[++i];
            role = role_from_string(role_str);
            if (role == Role::UNKNOWN) {
                fprintf(stderr, "Error: unknown role '%s'\n"
                        "Valid roles: compute, memory_mover, library, compiler, "
                        "monitor, profiler, ipc_xfer\n", role_str);
                return 1;
            }
        } else if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            config.gpu_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            config.duration_sec = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--results") == 0 && i + 1 < argc) {
            config.results_dir = argv[++i];
        } else if (strcmp(argv[i], "--peer-gpu") == 0 && i + 1 < argc) {
            config.peer_gpu_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--vram-pressure") == 0 && i + 1 < argc) {
            config.vram_pressure_pct = atoi(argv[++i]);
            if (config.vram_pressure_pct < 0 || config.vram_pressure_pct > 90) {
                fprintf(stderr, "Error: --vram-pressure must be 0-90\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--ipc-role") == 0 && i + 1 < argc) {
            const char* val = argv[++i];
            if (strcmp(val, "producer") == 0) {
                config.ipc_role = IpcRole::PRODUCER;
            } else if (strcmp(val, "consumer") == 0) {
                config.ipc_role = IpcRole::CONSUMER;
            } else {
                fprintf(stderr, "Error: --ipc-role must be 'producer' or 'consumer'\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--anomaly-ratio") == 0 && i + 1 < argc) {
            config.anomaly_ratio = atof(argv[++i]);
        } else if (strcmp(argv[i], "--severe-ratio") == 0 && i + 1 < argc) {
            config.severe_ratio = atof(argv[++i]);
        } else if (strcmp(argv[i], "--anomaly-pct") == 0 && i + 1 < argc) {
            config.anomaly_pct_limit = atof(argv[++i]);
        } else if (strcmp(argv[i], "--severe-pct") == 0 && i + 1 < argc) {
            config.severe_pct_limit = atof(argv[++i]);
        } else if (strcmp(argv[i], "--rss-warn") == 0 && i + 1 < argc) {
            config.rss_growth_warn_kb = atol(argv[++i]) * 1024;
        } else if (strcmp(argv[i], "--fd-warn") == 0 && i + 1 < argc) {
            config.fd_growth_warn = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--verbose") == 0) {
            // Accepted for compatibility with scripts; roles log at full detail.
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (role == Role::UNKNOWN) {
        fprintf(stderr, "Error: --role is required\n");
        print_usage(argv[0]);
        return 1;
    }

    if (mkdir(config.results_dir.c_str(), 0700) != 0 && errno != EEXIST) {
        fprintf(stderr, "Error: could not create results directory '%s': %s\n",
                config.results_dir.c_str(), strerror(errno));
        return 1;
    }

    if (config.gpu_id < 0) {
        fprintf(stderr, "Error: --gpu must be >= 0 (got %d)\n", config.gpu_id);
        return 1;
    }

    if (config.duration_sec <= 0) {
        fprintf(stderr, "Error: --duration must be > 0 (got %d)\n", config.duration_sec);
        return 1;
    }

    int gpu_count = 0;
    hipError_t hip_err = hipGetDeviceCount(&gpu_count);
    if (hip_err != hipSuccess || gpu_count <= 0) {
        fprintf(stderr, "ERROR: hipGetDeviceCount failed (error %d) — no GPUs available\n",
                hip_err);
        return 1;
    }

    printf("============================================================\n");
    printf("ROCK MPS Test — Role: %s\n", role_to_string(role));
    printf("  PID:      %d\n", getpid());
    printf("  GPU:      %d of %d\n", config.gpu_id, gpu_count);
    printf("  Duration: %d seconds\n", config.duration_sec);
    if (config.vram_pressure_pct > 0) {
        if (role == Role::MEMORY_MOVER) {
            printf("  VRAM pressure: %d%% (this process holds the resident set)\n", config.vram_pressure_pct);
        } else {
            printf("  VRAM pressure: %d%% (applied by memory_mover on this GPU)\n", config.vram_pressure_pct);
        }
    }
    if (config.ipc_role != IpcRole::NONE) {
        printf("  IPC role:  %s (cross-process shared GPU memory test)\n",
               config.ipc_role == IpcRole::PRODUCER ? "PRODUCER" : "CONSUMER");
    }
    printf("  Results:  %s\n", config.results_dir.c_str());
    printf("============================================================\n");

    if (config.gpu_id >= gpu_count) {
        fprintf(stderr, "ERROR: GPU %d requested but only %d GPU(s) available\n",
                config.gpu_id, gpu_count);
        return 1;
    }

    config.role_id = static_cast<int>(role);

    // Auto-pick peer GPU for ipc_xfer if not specified
    if (role == Role::IPC_XFER && config.peer_gpu_id < 0) {
        config.peer_gpu_id = (config.gpu_id + 1) % gpu_count;
        printf("  Peer GPU: %d (auto-selected)\n", config.peer_gpu_id);
    }

    int exit_code = 0;
    switch (role) {
        case Role::COMPUTE:
            exit_code = run_compute(config);
            break;
        case Role::MEMORY_MOVER:
            exit_code = run_memory_mover(config);
            break;
        case Role::LIBRARY:
            exit_code = run_library(config);
            break;
        case Role::COMPILER:
            exit_code = run_compiler(config);
            break;
        case Role::MONITOR:
            exit_code = run_monitor(config);
            break;
        case Role::PROFILER:
            exit_code = run_profiler(config);
            break;
        case Role::IPC_XFER:
            exit_code = run_ipc_xfer(config);
            break;
        default:
            fprintf(stderr, "Invalid role\n");
            return 1;
    }

    printf("[%s] Exit code: %d\n", role_to_string(role), exit_code);
    return exit_code;
}
