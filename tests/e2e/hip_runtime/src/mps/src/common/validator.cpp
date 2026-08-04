// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/validator.h"
#include <cstdio>

Validator::Validator() = default;

uint32_t Validator::make_pattern(int iteration, int role_id, int buffer_id) const {
    // Simple but unique pattern: mix iteration, role, and buffer id
    uint32_t p = 0xCAFE0000u;
    // Knuth multiplicative hash — both operands are uint32_t, so the
    // multiplication is unsigned integer arithmetic; overflow wraps mod 2^32
    // (well-defined for unsigned types) and the wrap is what spreads bits.
    p ^= static_cast<uint32_t>(iteration) * 2654435761u;
    p ^= static_cast<uint32_t>(role_id) << 16;
    p ^= static_cast<uint32_t>(buffer_id);
    return p;
}

void Validator::fill_host(uint32_t* buf, size_t count, uint32_t pattern) const {
    for (size_t i = 0; i < count; i++) {
        buf[i] = pattern ^ static_cast<uint32_t>(i);
    }
}

int Validator::verify_host(const uint32_t* buf, size_t count, uint32_t pattern,
                           int iteration, const std::string& context) const {
    check_count_.fetch_add(1);
    int mismatches = 0;
    for (size_t i = 0; i < count; i++) {
        uint32_t expected = pattern ^ static_cast<uint32_t>(i);
        if (buf[i] != expected) {
            if (mismatches < 5) {
                fprintf(stderr,
                    "[VERIFY FAIL] %s | iter=%d offset=%zu expected=0x%08X got=0x%08X\n",
                    context.c_str(), iteration, i, expected, buf[i]);
            }
            mismatches++;
        }
    }
    if (mismatches > 0) {
        error_count_.fetch_add(1);
        if (mismatches > 5) {
            fprintf(stderr, "[VERIFY FAIL] %s | iter=%d ... %d total mismatches\n",
                    context.c_str(), iteration, mismatches);
        }
    }
    return mismatches;
}
