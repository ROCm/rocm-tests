// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "roles/monitor.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <thread>
#include <string>
#include <vector>
#include <unistd.h>

#ifdef HAS_AMDSMI
#include <amd_smi/amdsmi.h>

#define AMDSMI_CHECK(call)                                                     \
    do {                                                                        \
        amdsmi_status_t st = (call);                                            \
        if (st != AMDSMI_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "[AMDSMI ERROR] %s:%d — %s returned %d\n",          \
                    __FILE__, __LINE__, #call, st);                              \
            total_errors++;                                                     \
        }                                                                       \
    } while (0)

// Returns true if `st` represents a sensor being absent on this topology
// (NOT_SUPPORTED, NO_DATA), as opposed to a real driver or handle failure.
static bool amdsmi_sensor_unavailable(amdsmi_status_t st) {
    return st == AMDSMI_STATUS_NOT_SUPPORTED || st == AMDSMI_STATUS_NO_DATA;
}

struct TempSensor {
    bool            available;
    amdsmi_temperature_type_t type;
    const char*     name;
};

// Probe HOTSPOT (JUNCTION) first — the primary sensor on MI300A/APU topologies.
// Fall back to EDGE for discrete GPUs where HOTSPOT may not exist.
// Returns {available=false} if neither sensor is present on this device.
static TempSensor probe_temp_sensor(amdsmi_processor_handle gpu) {
    static const struct { amdsmi_temperature_type_t type; const char* name; } candidates[] = {
        { AMDSMI_TEMPERATURE_TYPE_HOTSPOT, "HOTSPOT" },
        { AMDSMI_TEMPERATURE_TYPE_EDGE,    "EDGE"    },
    };
    for (auto& c : candidates) {
        int64_t probe_val = 0;
        amdsmi_status_t st = amdsmi_get_temp_metric(gpu, c.type, AMDSMI_TEMP_CURRENT, &probe_val);
        if (st == AMDSMI_STATUS_SUCCESS) {
            return { true, c.type, c.name };
        }
        if (!amdsmi_sensor_unavailable(st)) {
            // Unexpected error on probe — still try next candidate.
            fprintf(stderr, "[MONITOR] probe temp sensor %s: unexpected status %d\n", c.name, st);
        }
    }
    return { false, AMDSMI_TEMPERATURE_TYPE_EDGE, "none" };
}
#endif

// Exercises: AMD SMI (libamd_smi) → sysfs reads → amdgpu driver →
//            SMU (System Management Unit) register reads on GPU hardware.
//
// Continuously queries GPU temperature, clocks, VRAM usage, power, GPU
// activity at high frequency using the AMD SMI API.
//
// When other roles run concurrently, this stresses:
//   - Management plane vs compute plane isolation in the amdgpu driver
//   - sysfs read latency under GPU load (some reads touch HW registers)
//   - KFD process table reads while processes are starting/stopping
//   - Performance counter register reads while PROFILER also accesses them
//   - Potential micro-stalls on the compute engine from SMU register reads

int run_monitor(const RoleConfig& config) {
#ifndef HAS_AMDSMI
    printf("[MONITOR] AMD SMI not available — skipping\n");
    return 0;
#else
    printf("[MONITOR] PID %d | GPU %d | duration %ds | using AMD SMI\n",
           getpid(), config.gpu_id, config.duration_sec);

    amdsmi_status_t init_st = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
    if (init_st != AMDSMI_STATUS_SUCCESS) {
        fprintf(stderr, "[MONITOR] amdsmi_init failed: %d\n", init_st);
        return 1;
    }

    // Enumerate sockets and find the processor handle for our GPU
    uint32_t socket_count = 0;
    amdsmi_status_t enum_st;

    enum_st = amdsmi_get_socket_handles(&socket_count, nullptr);
    if (enum_st != AMDSMI_STATUS_SUCCESS || socket_count == 0) {
        fprintf(stderr, "[MONITOR] amdsmi_get_socket_handles failed: %d (count=%u)\n",
                enum_st, socket_count);
        amdsmi_shut_down();
        return 1;
    }
    std::vector<amdsmi_socket_handle> sockets(socket_count);
    enum_st = amdsmi_get_socket_handles(&socket_count, sockets.data());
    if (enum_st != AMDSMI_STATUS_SUCCESS) {
        fprintf(stderr, "[MONITOR] amdsmi_get_socket_handles (data) failed: %d\n", enum_st);
        amdsmi_shut_down();
        return 1;
    }

    // Collect all GPU processor handles across all sockets
    std::vector<amdsmi_processor_handle> all_gpus;
    for (uint32_t s = 0; s < socket_count; s++) {
        uint32_t dev_count = 0;
        enum_st = amdsmi_get_processor_handles(sockets[s], &dev_count, nullptr);
        if (enum_st != AMDSMI_STATUS_SUCCESS) {
            fprintf(stderr, "[MONITOR] amdsmi_get_processor_handles (count) failed for socket %u: %d\n",
                    s, enum_st);
            continue;
        }
        std::vector<amdsmi_processor_handle> devs(dev_count);
        enum_st = amdsmi_get_processor_handles(sockets[s], &dev_count, devs.data());
        if (enum_st != AMDSMI_STATUS_SUCCESS) {
            fprintf(stderr, "[MONITOR] amdsmi_get_processor_handles (data) failed for socket %u: %d\n",
                    s, enum_st);
            continue;
        }
        for (uint32_t d = 0; d < dev_count; d++) {
            processor_type_t ptype;
            if (amdsmi_get_processor_type(devs[d], &ptype) == AMDSMI_STATUS_SUCCESS
                && ptype == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
                all_gpus.push_back(devs[d]);
            }
        }
    }

    if (config.gpu_id < 0 || (size_t)config.gpu_id >= all_gpus.size()) {
        fprintf(stderr, "[MONITOR] GPU %d not found (detected %zu GPUs)\n",
                config.gpu_id, all_gpus.size());
        amdsmi_shut_down();
        return 1;
    }

    amdsmi_processor_handle gpu = all_gpus[config.gpu_id];
    int total_errors = 0;
    int64_t total_queries = 0;

    // Probe which temperature sensor is available on this GPU topology once,
    // before the main loop. MI300A/APU devices expose HOTSPOT (JUNCTION) but
    // not EDGE; discrete GPUs typically support both.
    TempSensor temp_sensor = probe_temp_sensor(gpu);
    if (temp_sensor.available) {
        printf("[MONITOR] temperature sensor: %s\n", temp_sensor.name);
    } else {
        fprintf(stderr,
                "[MONITOR] temperature sensor unavailable: neither HOTSPOT nor EDGE is supported "
                "on this platform/topology; temp_c will be reported as 0.0 and this topology "
                "absence is not counted as an error\n");
    }

    std::string csv_path = config.results_dir + "/monitor_gpu" +
                           std::to_string(config.gpu_id) +
                           "_pid" + std::to_string(getpid()) + ".csv";
    FILE* csv = fopen(csv_path.c_str(), "w");
    if (csv) {
        fprintf(csv, "time_sec,temp_c,sclk_mhz,mclk_mhz,vram_used_mb,"
                     "power_w,gpu_busy_pct,query_us\n");
    }

    auto start_time = std::chrono::steady_clock::now();
    int64_t iteration = 0;

    while (true) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = now - start_time;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
                >= config.duration_sec) {
            break;
        }
        double time_sec = std::chrono::duration<double>(elapsed).count();

        auto query_start = std::chrono::steady_clock::now();

        // --- Temperature ---
        // Use the sensor probed at startup. If none is available on this topology,
        // skip silently — absence of a sensor is not a monitor failure.
        double temp_c = 0.0;
        if (temp_sensor.available) {
            int64_t temp_val = 0;
            amdsmi_status_t st = amdsmi_get_temp_metric(gpu,
                temp_sensor.type, AMDSMI_TEMP_CURRENT, &temp_val);
            if (st == AMDSMI_STATUS_SUCCESS) {
                temp_c = static_cast<double>(temp_val);
            } else if (!amdsmi_sensor_unavailable(st)) {
                // Unexpected failure (e.g. handle gone, driver error) — real error.
                fprintf(stderr, "[MONITOR] temp query failed: status %d\n", st);
                total_errors++;
            }
            // AMDSMI_STATUS_NOT_SUPPORTED / NO_DATA after a successful probe is
            // treated as transient sensor absence — not counted as an error.
        }

        // --- Clock speeds ---
        amdsmi_clk_info_t gfx_clk = {};
        amdsmi_clk_info_t mem_clk = {};
        AMDSMI_CHECK(amdsmi_get_clock_info(gpu, AMDSMI_CLK_TYPE_GFX, &gfx_clk));
        AMDSMI_CHECK(amdsmi_get_clock_info(gpu, AMDSMI_CLK_TYPE_MEM, &mem_clk));
        uint32_t sclk_mhz = gfx_clk.clk;
        uint32_t mclk_mhz = mem_clk.clk;

        // --- VRAM usage ---
        amdsmi_vram_usage_t vram_info = {};
        AMDSMI_CHECK(amdsmi_get_gpu_vram_usage(gpu, &vram_info));
        double vram_used_mb = static_cast<double>(vram_info.vram_used);

        // --- Power ---
        amdsmi_power_info_t power_info = {};
        AMDSMI_CHECK(amdsmi_get_power_info(gpu, &power_info));
        double power_w = power_info.gfx_voltage != UINT32_MAX
                         ? static_cast<double>(power_info.socket_power) : 0.0;

        // --- GPU activity ---
        amdsmi_engine_usage_t activity = {};
        AMDSMI_CHECK(amdsmi_get_gpu_activity(gpu, &activity));
        uint32_t busy_pct = activity.gfx_activity;

        auto query_end = std::chrono::steady_clock::now();
        long query_us = std::chrono::duration_cast<std::chrono::microseconds>(
            query_end - query_start).count();

        total_queries++;

        if (csv) {
            fprintf(csv, "%.3f,%.1f,%u,%u,%.1f,%.1f,%u,%ld\n",
                    time_sec, temp_c, sclk_mhz, mclk_mhz,
                    vram_used_mb, power_w, busy_pct, query_us);
        }

        if (query_us > 100000) {
            fprintf(stderr, "[MONITOR] #%" PRId64 " | *** SLOW QUERY %ld us at t=%.1fs (driver contention?) ***\n",
                    iteration, query_us, time_sec);
        }

        printf("[MONITOR] #%" PRId64 " | temp=%.0fC sclk=%uMHz mclk=%uMHz vram=%.0fMB "
               "power=%.1fW busy=%u%% query=%ldus\n",
               iteration, temp_c, sclk_mhz, mclk_mhz,
               vram_used_mb, power_w, busy_pct, query_us);

        iteration++;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (csv) fclose(csv);

    amdsmi_shut_down();

    double elapsed_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    printf("[MONITOR] Finished: %" PRId64 " queries, %d errors, avg_interval=%.1fms (%.1fs)\n",
           total_queries, total_errors,
           config.duration_sec * 1000.0 / std::max(total_queries, (int64_t)1), elapsed_sec);

    return total_errors > 0 ? 1 : 0;
#endif
}
