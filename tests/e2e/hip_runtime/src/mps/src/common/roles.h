// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

enum class Role {
    COMPUTE,
    MEMORY_MOVER,
    LIBRARY,
    COMPILER,
    MONITOR,
    PROFILER,
    IPC_XFER,
    UNKNOWN
};

inline Role role_from_string(const std::string& s) {
    if (s == "compute")      return Role::COMPUTE;
    if (s == "memory_mover") return Role::MEMORY_MOVER;
    if (s == "library")      return Role::LIBRARY;
    if (s == "compiler")     return Role::COMPILER;
    if (s == "monitor")      return Role::MONITOR;
    if (s == "profiler")     return Role::PROFILER;
    if (s == "ipc_xfer")     return Role::IPC_XFER;
    return Role::UNKNOWN;
}

inline const char* role_to_string(Role r) {
    switch (r) {
        case Role::COMPUTE:      return "COMPUTE";
        case Role::MEMORY_MOVER: return "MEMORY_MOVER";
        case Role::LIBRARY:      return "LIBRARY";
        case Role::COMPILER:     return "COMPILER";
        case Role::MONITOR:      return "MONITOR";
        case Role::PROFILER:     return "PROFILER";
        case Role::IPC_XFER:     return "IPC_XFER";
        default:                 return "UNKNOWN";
    }
}
