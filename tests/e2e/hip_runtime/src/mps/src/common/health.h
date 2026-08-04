// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <atomic>
#include <thread>
#include <vector>

struct HealthSample {
    double    timestamp_sec;
    long      host_rss_kb;
    int       fd_count;
    long long vram_used_bytes;   // -1 if unavailable
    bool      hip_error_sticky;
};

class HealthMonitor {
public:
    HealthMonitor(int gpu_id, const std::string& results_dir,
                  long rss_growth_warn_kb = 100 * 1024, int fd_growth_warn = 50);
    ~HealthMonitor();

    void start();
    void stop();

private:
    void monitor_loop();
    HealthSample take_sample();
    void write_report() const;

    int gpu_id_;
    std::string results_dir_;
    std::atomic<bool> running_{false};
    bool stopped_{false};
    std::thread thread_;
    std::vector<HealthSample> samples_;

    long baseline_rss_kb_ = 0;
    int  baseline_fd_count_ = 0;
    long rss_growth_warn_kb_;
    int  fd_growth_warn_;
};
