// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <string>

class Validator {
public:
    Validator();

    uint32_t make_pattern(int iteration, int role_id, int buffer_id) const;

    // Fill host buffer with expected pattern
    void fill_host(uint32_t* buf, size_t count, uint32_t pattern) const;

    // Verify host buffer matches expected pattern.
    // Returns number of mismatches.
    int verify_host(const uint32_t* buf, size_t count, uint32_t pattern,
                    int iteration, const std::string& context) const;

    int total_errors() const { return error_count_.load(); }
    int total_checks() const { return check_count_.load(); }

private:
    mutable std::atomic<int> error_count_{0};
    mutable std::atomic<int> check_count_{0};
};
