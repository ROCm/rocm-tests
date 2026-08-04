// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

enum class IpcRole { NONE, PRODUCER, CONSUMER };

struct RoleConfig {
    int gpu_id = 0;
    int peer_gpu_id = -1;  // for ipc_xfer: destination GPU (-1 = auto-pick next GPU)
    int duration_sec = 60;
    std::string results_dir = "/tmp/rock_mps_test";
    int role_id = 0;     // numeric ID for pattern generation
    int vram_pressure_pct = 0; // target VRAM resident set percentage (0 = disabled)
    IpcRole ipc_role = IpcRole::NONE; // cross-process shared memory test (memory_mover only)

    // Profiler anomaly thresholds
    float anomaly_ratio = 5.0f;        // kernel time / baseline ratio to flag anomaly
    float severe_ratio = 50.0f;        // ratio threshold for severe anomaly
    float anomaly_pct_limit = 1.0f;    // max allowed anomaly rate (%) before FAIL
    float severe_pct_limit = 0.001f;   // max allowed severe rate (%) before FAIL

    // Health monitor leak detection thresholds
    long rss_growth_warn_kb = 100 * 1024;  // warn if RSS grows by this much (100 MB)
    int fd_growth_warn = 50;               // warn if FD count grows by this much
};
