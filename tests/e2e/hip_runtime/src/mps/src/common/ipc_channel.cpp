// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/ipc_channel.h"
#include <cerrno>
#include <cstdio>
#include <thread>
#include <chrono>
#include <fcntl.h>
#include <sys/stat.h>
#include <signal.h>
#include <unistd.h>
#include <dirent.h>

IpcChannel::IpcChannel(const std::string& results_dir)
    : dir_(results_dir) {}

std::string IpcChannel::handle_path(const std::string& tag) const {
    return dir_ + "/ipc_handle_" + tag + ".bin";
}

std::string IpcChannel::signal_path(const std::string& name) const {
    return dir_ + "/ipc_signal_" + name;
}

std::string IpcChannel::pid_path(const std::string& role) const {
    return dir_ + "/ipc_pid_" + role;
}

void IpcChannel::register_pid(const std::string& role) {
    std::string path = pid_path(role);
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd >= 0) {
        FILE* f = fdopen(fd, "w");
        if (f) {
            fprintf(f, "%d\n", static_cast<int>(getpid()));
            fflush(f);
            fsync(fileno(f));
            fclose(f);
        } else {
            close(fd);
        }
    }
}

bool IpcChannel::discover_peer_pid(const std::string& peer_role, int timeout_sec) {
    std::string path = pid_path(peer_role);
    for (int i = 0; i < timeout_sec * 10; i++) {
        FILE* f = fopen(path.c_str(), "r");
        if (f) {
            int pid = 0;
            if (fscanf(f, "%d", &pid) == 1 && pid > 0) {
                fclose(f);
                peer_pid_ = static_cast<pid_t>(pid);
                printf("[IPC] Discovered peer (%s) PID: %d\n", peer_role.c_str(), pid);
                return true;
            }
            fclose(f);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    fprintf(stderr, "[IPC] Failed to discover peer (%s) PID after %ds\n",
            peer_role.c_str(), timeout_sec);
    return false;
}

bool IpcChannel::peer_alive() const {
    if (peer_pid_ <= 0) return true;  // no peer registered — assume alive
    if (kill(peer_pid_, 0) == 0) return true;
    return errno != ESRCH;  // EPERM = exists but restricted → alive; ESRCH = gone
}

bool IpcChannel::export_handle(void* d_ptr, const std::string& tag) {
    if (!d_ptr) {
        fprintf(stderr, "[IPC] export_handle called with null device pointer\n");
        return false;
    }

    // Clear any sticky error from prior HIP calls (e.g. failed kernel launch)
    // that would cause hipIpcGetMemHandle to return "invalid argument".
    hipError_t sticky = hipGetLastError();
    if (sticky != hipSuccess) {
        fprintf(stderr, "[IPC] Cleared sticky HIP error before IPC export: %s\n",
                hipGetErrorString(sticky));
    }

    hipIpcMemHandle_t handle;
    hipError_t err = hipIpcGetMemHandle(&handle, d_ptr);
    if (err != hipSuccess) {
        fprintf(stderr, "[IPC] hipIpcGetMemHandle failed: %s (d_ptr=%p, tag=%s)\n",
                hipGetErrorString(err), d_ptr, tag.c_str());
        return false;
    }

    std::string path = handle_path(tag);
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        fprintf(stderr, "[IPC] Failed to create handle file: %s\n", path.c_str());
        return false;
    }
    FILE* f = fdopen(fd, "wb");
    if (!f) {
        fprintf(stderr, "[IPC] Failed to open handle file stream: %s\n", path.c_str());
        close(fd);
        return false;
    }
    if (fwrite(&handle, sizeof(handle), 1, f) != 1) {
        fprintf(stderr, "[IPC] Failed to write IPC handle to: %s\n", path.c_str());
        fclose(f);
        return false;
    }
    fflush(f);
    fsync(fileno(f));
    fclose(f);
    return true;
}

bool IpcChannel::import_handle(void** d_ptr, const std::string& tag,
                               int timeout_sec) {
    std::string path = handle_path(tag);

    for (int i = 0; i < timeout_sec * 10; i++) {
        struct stat st;
        if (stat(path.c_str(), &st) == 0 &&
            st.st_size >= static_cast<off_t>(sizeof(hipIpcMemHandle_t))) {
            break;
        }
        if (i % 10 == 9 && !peer_alive()) {
            fprintf(stderr, "[IPC] Peer process (PID %d) is dead — aborting import for '%s'\n",
                    static_cast<int>(peer_pid_), tag.c_str());
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[IPC] Handle file not found after %ds: %s\n",
                timeout_sec, path.c_str());
        return false;
    }

    hipIpcMemHandle_t handle;
    if (fread(&handle, sizeof(handle), 1, f) != 1) {
        fprintf(stderr, "[IPC] Failed to read handle from: %s\n", path.c_str());
        fclose(f);
        return false;
    }
    fclose(f);

    (void)hipGetLastError();
    hipError_t err = hipIpcOpenMemHandle(d_ptr, handle, hipIpcMemLazyEnablePeerAccess);
    if (err != hipSuccess) {
        fprintf(stderr, "[IPC] hipIpcOpenMemHandle failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool IpcChannel::close_handle(void* d_ptr) {
    (void)hipGetLastError();
    hipError_t err = hipIpcCloseMemHandle(d_ptr);
    if (err != hipSuccess) {
        fprintf(stderr, "[IPC] hipIpcCloseMemHandle failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool IpcChannel::post_signal(const std::string& name) {
    std::string path = signal_path(name);
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        fprintf(stderr, "[IPC] Failed to create signal file: %s\n", path.c_str());
        return false;
    }
    FILE* f = fdopen(fd, "w");
    if (!f) {
        close(fd);
        return false;
    }
    fprintf(f, "1\n");
    fflush(f);
    fsync(fileno(f));
    fclose(f);
    return true;
}

bool IpcChannel::wait_signal(const std::string& name, int timeout_sec) {
    std::string path = signal_path(name);
    for (int i = 0; i < timeout_sec * 10; i++) {
        struct stat st;
        if (stat(path.c_str(), &st) == 0 && st.st_size > 0) {
            return true;
        }
        if (i % 10 == 9 && !peer_alive()) {
            fprintf(stderr, "[IPC] Peer process (PID %d) is dead — aborting wait for '%s'\n",
                    static_cast<int>(peer_pid_), name.c_str());
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    fprintf(stderr, "[IPC] Timed out waiting for signal '%s' after %ds\n",
            name.c_str(), timeout_sec);
    return false;
}

bool IpcChannel::post_signal_with_value(const std::string& name, uint32_t value) {
    std::string path = signal_path(name);
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        fprintf(stderr, "[IPC] Failed to create signal file: %s\n", path.c_str());
        return false;
    }
    FILE* f = fdopen(fd, "wb");
    if (!f) {
        fprintf(stderr, "[IPC] Failed to open signal file stream: %s\n", path.c_str());
        close(fd);
        return false;
    }
    if (fwrite(&value, sizeof(value), 1, f) != 1) {
        fprintf(stderr, "[IPC] Failed to write signal value to: %s\n", path.c_str());
        fclose(f);
        return false;
    }
    fflush(f);
    fsync(fileno(f));
    fclose(f);
    return true;
}

bool IpcChannel::wait_signal_read_value(const std::string& name, uint32_t* value,
                                         int timeout_sec) {
    std::string path = signal_path(name);
    for (int i = 0; i < timeout_sec * 10; i++) {
        struct stat st;
        if (stat(path.c_str(), &st) == 0 &&
            st.st_size >= static_cast<off_t>(sizeof(uint32_t))) {
            FILE* f = fopen(path.c_str(), "rb");
            if (f) {
                if (fread(value, sizeof(uint32_t), 1, f) == 1) {
                    fclose(f);
                    return true;
                }
                fclose(f);
            }
        }
        if (i % 10 == 9 && !peer_alive()) {
            fprintf(stderr, "[IPC] Peer process (PID %d) is dead — aborting wait for '%s'\n",
                    static_cast<int>(peer_pid_), name.c_str());
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    fprintf(stderr, "[IPC] Timed out waiting for signal+value '%s' after %ds\n",
            name.c_str(), timeout_sec);
    return false;
}

void IpcChannel::cleanup_round(const std::string& round_str) {
    unlink(signal_path("producer_ready_" + round_str).c_str());
    unlink(signal_path("consumer_done_" + round_str).c_str());
    unlink(signal_path("producer_round_done_" + round_str).c_str());
    unlink(signal_path("consumer_closed_" + round_str).c_str());
}

// Called by the producer ONLY, at startup before any signals are created
// and before the consumer discovers the producer PID. This ordering
// guarantees no concurrent access from the consumer side.
void IpcChannel::cleanup() {
    DIR* dir = opendir(dir_.c_str());
    if (!dir) return;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name.rfind("ipc_handle_", 0) == 0 || name.rfind("ipc_signal_", 0) == 0 ||
            name.rfind("ipc_pid_", 0) == 0) {
            std::string path = dir_ + "/" + name;
            unlink(path.c_str());
        }
    }
    closedir(dir);
}
