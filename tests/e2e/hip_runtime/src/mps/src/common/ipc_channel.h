// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <string>
#include <sys/types.h>

// File-based IPC handle exchange and signaling between processes.
// Producer writes a hipIpcMemHandle to a file. Consumer reads it.
// Signal files provide lightweight cross-process synchronization.
//
// Peer liveness: each side registers its PID at startup. Wait methods
// poll kill(peer_pid, 0) and return early if the peer process has died,
// avoiding long timeout loops against a crashed peer.
class IpcChannel {
public:
    explicit IpcChannel(const std::string& results_dir);

    // Register this process's PID and discover the peer's PID.
    // Call once at IPC startup. role is "producer" or "consumer".
    void register_pid(const std::string& role);
    bool discover_peer_pid(const std::string& peer_role, int timeout_sec = 30);
    bool peer_alive() const;

    // Producer: export a device pointer's IPC handle to a file
    bool export_handle(void* d_ptr, const std::string& tag);

    // Consumer: import a device pointer from a previously exported handle.
    // Blocks up to timeout_sec waiting for the file to appear.
    bool import_handle(void** d_ptr, const std::string& tag, int timeout_sec = 30);

    // Close an imported handle
    static bool close_handle(void* d_ptr);

    // Cross-process signaling via files. post_signal creates a named file;
    // wait_signal polls until it appears (returns false on timeout or dead peer).
    bool post_signal(const std::string& name);
    bool wait_signal(const std::string& name, int timeout_sec = 60);

    // Write/read a 4-byte pattern value alongside a signal for verification
    bool post_signal_with_value(const std::string& name, uint32_t value);
    bool wait_signal_read_value(const std::string& name, uint32_t* value, int timeout_sec = 60);

    // Remove all IPC artifacts (handle files, signal files) from the directory
    void cleanup();

    // Remove IPC artifacts for a specific round tag (call after both sides complete)
    void cleanup_round(const std::string& round_str);

private:
    std::string dir_;
    pid_t peer_pid_ = 0;

    std::string handle_path(const std::string& tag) const;
    std::string signal_path(const std::string& name) const;
    std::string pid_path(const std::string& role) const;
};
