#include "result_table.hpp"

#include <iostream>

void ResultTable::add(const std::string& check, const std::string& expected, const std::string& actual, bool pass) {
    rows_.push_back(Row{check, expected, actual, pass ? RowResult::Pass : RowResult::Fail});
}

void ResultTable::skip(const std::string& check, const std::string& expected, const std::string& actual) {
    rows_.push_back(Row{check, expected, actual, RowResult::Skip});
}

bool ResultTable::has_fail() const {
    for (const Row& row : rows_) {
        if (row.result == RowResult::Fail) {
            return true;
        }
    }
    return false;
}

bool ResultTable::has_skip() const {
    for (const Row& row : rows_) {
        if (row.result == RowResult::Skip) {
            return true;
        }
    }
    return false;
}

int ResultTable::exit_code() const {
    if (has_fail()) {
        return 1;
    }
    if (has_skip() && rows_.size() == 1) {
        return 2;
    }
    if (has_skip() && !has_fail()) {
        for (const Row& row : rows_) {
            if (row.result != RowResult::Skip) {
                return 0;
            }
        }
        return 2;
    }
    return 0;
}

void ResultTable::print(const std::string& test_name) const {
    const char* verdict = has_fail() ? "FAIL" : (exit_code() == 2 ? "SKIP" : "PASS");
    std::cout << "[" << verdict << "] " << test_name << "\n";
    for (const Row& row : rows_) {
        const char* mark = row.result == RowResult::Pass ? "PASS" : (row.result == RowResult::Skip ? "SKIP" : "FAIL");
        std::cout << "  " << mark << "  " << row.check << "\n";
        std::cout << "    expected: " << row.expected << "\n";
        std::cout << "    actual:   " << row.actual << "\n";
    }
}
