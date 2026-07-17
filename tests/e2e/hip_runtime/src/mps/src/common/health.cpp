// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/health.h"
#include <hip/hip_runtime.h>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstdio>
#include <dirent.h>
#include <unistd.h>

static long get_rss_kb() {
    std::ifstream stat("/proc/self/status");
    std::string line;
    while (std::getline(stat, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            long kb = 0;
            sscanf(line.c_str(), "VmRSS: %ld", &kb);
            return kb;
        }
    }
    return -1;
}

static int get_fd_count() {
    int count = 0;
    DIR* dir = opendir("/proc/self/fd");
    if (!dir) return -1;
    while (readdir(dir)) count++;
    closedir(dir);
    return count - 2; // subtract . and ..
}

static double now_sec() {
    auto t = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<double>(t).count();
}

HealthMonitor::HealthMonitor(int gpu_id, const std::string& results_dir,
                             long rss_growth_warn_kb, int fd_growth_warn)
    : gpu_id_(gpu_id), results_dir_(results_dir),
      rss_growth_warn_kb_(rss_growth_warn_kb), fd_growth_warn_(fd_growth_warn) {}

HealthMonitor::~HealthMonitor() {
    stop();
}

void HealthMonitor::start() {
    running_ = true;
    thread_ = std::thread(&HealthMonitor::monitor_loop, this);
}

void HealthMonitor::stop() {
    if (stopped_) return;
    stopped_ = true;
    running_ = false;
    if (thread_.joinable()) thread_.join();
    write_report();
}

HealthSample HealthMonitor::take_sample() {
    HealthSample s{};
    s.timestamp_sec = now_sec();
    s.host_rss_kb = get_rss_kb();
    s.fd_count = get_fd_count();

    size_t free_mem = 0, total_mem = 0;
    if (hipMemGetInfo(&free_mem, &total_mem) == hipSuccess) {
        s.vram_used_bytes = static_cast<long long>(total_mem - free_mem);
    } else {
        s.vram_used_bytes = -1;
    }

    hipError_t last = hipPeekAtLastError();
    s.hip_error_sticky = (last != hipSuccess);

    return s;
}

void HealthMonitor::monitor_loop() {
    // Delay baseline by 30 seconds so Phase 1 resident set allocation
    // (which legitimately grows RSS/VRAM) doesn't trigger false alarms.
    for (int i = 0; i < 6 && running_; i++) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    if (!running_) return;

    auto baseline = take_sample();
    baseline_rss_kb_ = baseline.host_rss_kb;
    baseline_fd_count_ = baseline.fd_count;
    samples_.push_back(baseline);

    long last_warned_rss_growth = 0;
    int last_warned_fd_growth = 0;

    while (running_) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        if (!running_) break;

        auto s = take_sample();
        samples_.push_back(s);

        long rss_growth = s.host_rss_kb - baseline_rss_kb_;
        int fd_growth = s.fd_count - baseline_fd_count_;

        if (rss_growth > rss_growth_warn_kb_ && rss_growth > last_warned_rss_growth + rss_growth_warn_kb_) {
            fprintf(stderr, "[HEALTH WARN] Host RSS grew by %ld KB since baseline (possible leak)\n",
                    rss_growth);
            last_warned_rss_growth = rss_growth;
        }
        if (fd_growth > fd_growth_warn_ && fd_growth > last_warned_fd_growth + fd_growth_warn_) {
            fprintf(stderr, "[HEALTH WARN] FD count grew by %d since baseline (possible leak)\n",
                    fd_growth);
            last_warned_fd_growth = fd_growth;
        }
        if (s.hip_error_sticky) {
            fprintf(stderr, "[HEALTH WARN] Sticky HIP error detected\n");
        }
    }
}

void HealthMonitor::write_report() const {
    if (samples_.empty()) return;

    std::string path = results_dir_ + "/health_gpu" + std::to_string(gpu_id_) +
                       "_pid" + std::to_string(getpid()) + ".csv";
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "[HEALTH] Failed to write report to %s\n", path.c_str());
        return;
    }

    fprintf(f, "timestamp_sec,host_rss_kb,fd_count,vram_used_bytes,hip_error_sticky\n");
    double t0 = samples_[0].timestamp_sec;
    for (const auto& s : samples_) {
        fprintf(f, "%.3f,%ld,%d,%lld,%d\n",
                s.timestamp_sec - t0, s.host_rss_kb, s.fd_count,
                s.vram_used_bytes, s.hip_error_sticky ? 1 : 0);
    }
    fclose(f);

    // Summary
    auto& first = samples_.front();
    auto& last = samples_.back();
    printf("[HEALTH] Duration: %.1f sec | RSS: %ld→%ld KB | FDs: %d→%d | "
           "VRAM: %lld→%lld bytes\n",
           last.timestamp_sec - first.timestamp_sec,
           first.host_rss_kb, last.host_rss_kb,
           first.fd_count, last.fd_count,
           first.vram_used_bytes, last.vram_used_bytes);
}
