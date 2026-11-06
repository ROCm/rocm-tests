#pragma once

#include <string>
#include <vector>

class ResultTable {
 public:
    void add(const std::string& check, const std::string& expected, const std::string& actual, bool pass);
    void skip(const std::string& check, const std::string& expected, const std::string& actual);
    void print(const std::string& test_name) const;
    bool has_fail() const;
    bool has_skip() const;
    int exit_code() const;

 private:
    enum class RowResult { Pass, Fail, Skip };

    struct Row {
        std::string check;
        std::string expected;
        std::string actual;
        RowResult result;
    };

    std::vector<Row> rows_;
};
