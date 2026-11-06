#pragma once

#include <iostream>
#include <string>

inline bool& verbose_output() {
    static bool verbose = false;
    return verbose;
}

inline void begin_test(const std::string& test_name) {
    if (verbose_output()) {
        std::cout << "test: " << test_name << "\n" << std::flush;
    }
}

inline void flow(const std::string& message) {
    if (verbose_output()) {
        std::cout << "[FLOW] " << message << "\n" << std::flush;
    }
}
