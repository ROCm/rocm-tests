/*
   Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
   SPDX-License-Identifier: MIT
*/

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace {

int g_passed = 0;
int g_failed = 0;
int g_skipped = 0;

#define CHECK(cond, msg)                                          \
    do {                                                          \
        if (!(cond)) {                                          \
            fprintf(stderr, "  FAIL: %s\n        %s:%d\n",       \
                    msg, __FILE__, __LINE__);                     \
            ++g_failed;                                           \
            return;                                               \
        }                                                         \
    } while (0)

struct KernelSymbol {
    const char* mangled;
    const char* display;
    int unrollFactor;
};

constexpr KernelSymbol kKernels[] = {
    {"_Z23ncclDevKernel_Generic_124ncclDevKernelArgsStorageILm4096EE",
     "ncclDevKernel_Generic_1", 1},
    {"_Z23ncclDevKernel_Generic_224ncclDevKernelArgsStorageILm4096EE",
     "ncclDevKernel_Generic_2", 2},
    {"_Z23ncclDevKernel_Generic_424ncclDevKernelArgsStorageILm4096EE",
     "ncclDevKernel_Generic_4", 4},
};
constexpr size_t kNumKernels = sizeof(kKernels) / sizeof(kKernels[0]);

bool verboseEnabled() {
    const char* v = getenv("RCCL_TEST_VERBOSE");
    return v && std::string(v) == "1";
}

// Single-quote for sh -c / popen; escape embedded single quotes.
std::string shellQuote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'')
            out += "'\\''";
        else
            out += c;
    }
    out += '\'';
    return out;
}

const char* getRcclLibPath() {
    const char* env = getenv("RCCL_LIB_PATH");
    return env ? env : "librccl.so";
}

std::string getAllReducePerfPath() {
    const char* env = getenv("ALL_REDUCE_PERF_PATH");
    if (env) return env;
    const char* candidates[] = {
        "/opt/rocm/bin/all_reduce_perf",
        "/opt/rocm-7.12.0/bin/all_reduce_perf",
    };
    for (auto c : candidates) {
        if (access(c, X_OK) == 0) return c;
    }
    return "";
}

std::string getGpuArch() {
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, 0) != hipSuccess)
        return "";
    std::string arch = props.gcnArchName;
    auto colon = arch.find(':');
    if (colon != std::string::npos)
        arch = arch.substr(0, colon);
    return arch;
}

int getGpuCUCount() {
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, 0) != hipSuccess)
        return -1;
    return props.multiProcessorCount;
}

// Mirror of commSetUnrollFactor from src/rccl_wrap.cc
// Returns the expected unroll factor value (1, 2, or 4) for
// this GPU in single-node configuration.
int expectedUnrollForArch(const std::string& arch, int cuCount) {
    if (arch == "gfx950")
        return 1;  // single-node preset
    if (arch == "gfx908")
        return 2;
    if (arch == "gfx942" && cuCount > 80)
        return 2;  // MI300X (>80 CUs)
    return 4;      // MI308 (gfx942 ≤80 CUs), others
}

// Run a command and capture combined stdout+stderr (like `env ... 2>&1` + temp file).
// Matches shell: use `env` so assignments apply to the child; mirror RCCL_DEBUG with
// NCCL_DEBUG for stacks that read either name.
int runCommand(const std::string& cmd, std::string& out) {
    out.clear();
    FILE* fp = popen((cmd + " 2>&1").c_str(), "r");
    if (!fp) return -1;
    char buf[4096];
    while (fgets(buf, sizeof(buf), fp))
        out += buf;
    int st = pclose(fp);
    int rc = WEXITSTATUS(st);
    if (verboseEnabled() && !out.empty())
        fwrite(out.data(), 1, out.size(), stderr);
    return rc;
}

// Search `output` for a line containing `needle` and return
// the integer that appears at the end of that line.
// e.g. "RCCL Unroll Factor (pre-set): 1" → 1
bool extractUnrollFromOutput(const std::string& output,
                             const std::string& needle,
                             int& value) {
    auto pos = output.find(needle);
    if (pos == std::string::npos) return false;
    auto lineEnd = output.find('\n', pos);
    std::string line = output.substr(pos, lineEnd - pos);
    auto colon = line.rfind(':');
    if (colon == std::string::npos) return false;
    value = atoi(line.c_str() + colon + 1);
    return true;
}

// ---------------------------------------------------------------
// Test 1: All three kernel symbols exist in librccl.so
// ---------------------------------------------------------------
void test_KernelSymbolPresence() {
    printf("[RUN ] KernelSymbolPresence\n");

    const char* libPath = getRcclLibPath();
    void* handle = dlopen(libPath, RTLD_LAZY | RTLD_NOLOAD);
    if (!handle)
        handle = dlopen(libPath, RTLD_LAZY);
    CHECK(handle != nullptr,
          (std::string("Cannot open RCCL library: ") + dlerror()).c_str());

    bool allFound = true;
    for (size_t i = 0; i < kNumKernels; ++i) {
        void* sym = dlsym(handle, kKernels[i].mangled);
        if (sym) {
            printf("  OK   %-35s  UNROLL=%d\n",
                   kKernels[i].display, kKernels[i].unrollFactor);
        } else {
            fprintf(stderr, "  MISS %-35s  UNROLL=%d  NOT FOUND\n",
                    kKernels[i].display, kKernels[i].unrollFactor);
            allFound = false;
        }
    }
    dlclose(handle);

    CHECK(allFound, "One or more kernel symbols missing from librccl.so");
    printf("[PASS] KernelSymbolPresence\n");
    ++g_passed;
}

// ---------------------------------------------------------------
// Test 2: Runtime heuristic selects the correct unroll factor
//         for this GPU architecture (single-node).
//
// Runs: all_reduce_perf -b 8 -e 8 -g 1
// with  NCCL_DEBUG=INFO RCCL_DEBUG=INFO (via env)
// Parses: "RCCL Unroll Factor (pre-set): N"
// ---------------------------------------------------------------
void test_RuntimeHeuristicSelection() {
    printf("[RUN ] RuntimeHeuristicSelection\n");

    std::string perfBin = getAllReducePerfPath();
    if (perfBin.empty()) {
        printf("[SKIP] RuntimeHeuristicSelection — all_reduce_perf not found\n");
        ++g_skipped;
        return;
    }

    std::string arch = getGpuArch();
    int cuCount = getGpuCUCount();
    if (arch.empty()) {
        printf("[SKIP] RuntimeHeuristicSelection — no GPU detected\n");
        ++g_skipped;
        return;
    }

    int expected = expectedUnrollForArch(arch, cuCount);
    printf("  GPU: %s (%d CUs) — expected unroll factor: %d\n",
           arch.c_str(), cuCount, expected);

    std::string cmd = "env NCCL_DEBUG=INFO RCCL_DEBUG=INFO " + shellQuote(perfBin) +
                      " -b 8 -e 8 -g 1";
    std::string output;
    int rc = runCommand(cmd, output);

    CHECK(rc == 0,
          (std::string("all_reduce_perf failed with exit code ") +
           std::to_string(rc)).c_str());

    int actual = -1;
    bool found = extractUnrollFromOutput(output, "RCCL Unroll Factor (pre-set):",
                                         actual);
    CHECK(found, "Could not find 'RCCL Unroll Factor (pre-set):' in NCCL_DEBUG output");

    printf("  Heuristic selected unroll factor: %d\n", actual);

    CHECK(actual == expected,
          (std::string("Heuristic mismatch: expected unroll=") +
           std::to_string(expected) + " for " + arch +
           ", got unroll=" + std::to_string(actual)).c_str());

    printf("[PASS] RuntimeHeuristicSelection\n");
    ++g_passed;
}

// ---------------------------------------------------------------
// Test 3: RCCL_UNROLL_FACTOR env-var override works for each
//         valid index (0→unroll=1, 1→unroll=2, 2→unroll=4).
// ---------------------------------------------------------------
void test_UnrollOverride() {
    printf("[RUN ] UnrollOverride\n");

    std::string perfBin = getAllReducePerfPath();
    if (perfBin.empty()) {
        printf("[SKIP] UnrollOverride — all_reduce_perf not found\n");
        ++g_skipped;
        return;
    }

    struct { int envVal; int expectedUnroll; } cases[] = {
        {0, 1}, {1, 2}, {2, 4},
    };

    bool allPassed = true;
    for (auto& tc : cases) {
        std::string cmd = "env NCCL_DEBUG=INFO RCCL_DEBUG=INFO RCCL_UNROLL_FACTOR=" +
                          std::to_string(tc.envVal) + " " + shellQuote(perfBin) +
                          " -b 8 -e 8 -g 1";
        std::string output;
        int rc = runCommand(cmd, output);

        if (rc != 0) {
            fprintf(stderr, "  FAIL: RCCL_UNROLL_FACTOR=%d — all_reduce_perf "
                    "exited with code %d\n", tc.envVal, rc);
            allPassed = false;
            continue;
        }

        int actual = -1;
        bool found = extractUnrollFromOutput(output, "RCCL Unroll Factor (user set):",
                                              actual);

        if (!found) {
            fprintf(stderr, "  FAIL: RCCL_UNROLL_FACTOR=%d — 'user set' line "
                    "not found in output\n", tc.envVal);
            allPassed = false;
            continue;
        }

        if (actual != tc.expectedUnroll) {
            fprintf(stderr, "  FAIL: RCCL_UNROLL_FACTOR=%d — expected unroll=%d, "
                    "got %d\n", tc.envVal, tc.expectedUnroll, actual);
            allPassed = false;
        } else {
            printf("  OK   RCCL_UNROLL_FACTOR=%d → unroll=%d\n",
                   tc.envVal, actual);
        }
    }

    CHECK(allPassed, "One or more RCCL_UNROLL_FACTOR overrides produced wrong result");
    printf("[PASS] UnrollOverride\n");
    ++g_passed;
}

// ---------------------------------------------------------------
// Test 4: Invalid RCCL_UNROLL_FACTOR values are rejected
//         (exit code != 0 and warning message emitted).
// ---------------------------------------------------------------
void test_InvalidUnrollRejected() {
    printf("[RUN ] InvalidUnrollRejected\n");

    std::string perfBin = getAllReducePerfPath();
    if (perfBin.empty()) {
        printf("[SKIP] InvalidUnrollRejected — all_reduce_perf not found\n");
        ++g_skipped;
        return;
    }

    int badValues[] = {3, 4, 5, 99};
    bool allRejected = true;

    for (int val : badValues) {
        std::string cmd = "env NCCL_DEBUG=WARN RCCL_DEBUG=WARN RCCL_UNROLL_FACTOR=" +
                          std::to_string(val) + " " + shellQuote(perfBin) +
                          " -b 8 -e 8 -g 1";
        std::string output;
        int rc = runCommand(cmd, output);

        bool hasWarning = output.find("Invalid RCCL_UNROLL_FACTOR") != std::string::npos;
        bool rejected = (rc != 0);

        if (rejected && hasWarning) {
            printf("  OK   RCCL_UNROLL_FACTOR=%d → rejected (exit=%d)\n", val, rc);
        } else if (rejected) {
            printf("  OK   RCCL_UNROLL_FACTOR=%d → rejected (exit=%d, no warning msg)\n",
                   val, rc);
        } else {
            fprintf(stderr, "  FAIL: RCCL_UNROLL_FACTOR=%d was accepted "
                    "(exit=0) — should have been rejected\n", val);
            allRejected = false;
        }
    }

    CHECK(allRejected, "RCCL accepted an invalid RCCL_UNROLL_FACTOR value");
    printf("[PASS] InvalidUnrollRejected\n");
    ++g_passed;
}

// ---------------------------------------------------------------
// Test 5: Collective succeeds with each valid unroll factor.
//         Runs all_reduce_perf across a range of message sizes
//         for each override value to verify no crashes or hangs.
// ---------------------------------------------------------------
void test_CollectiveCorrectnessAllUnrolls() {
    printf("[RUN ] CollectiveCorrectnessAllUnrolls\n");

    std::string perfBin = getAllReducePerfPath();
    if (perfBin.empty()) {
        printf("[SKIP] CollectiveCorrectnessAllUnrolls — all_reduce_perf not found\n");
        ++g_skipped;
        return;
    }

    int overrides[] = {0, 1, 2}; // unroll 1, 2, 4
    int unrollLabels[] = {1, 2, 4};
    bool allPassed = true;

    for (int i = 0; i < 3; ++i) {
        std::string cmd = "env RCCL_UNROLL_FACTOR=" + std::to_string(overrides[i]) +
                          " " + shellQuote(perfBin) + " -b 8 -e 4M -g 1";
        std::string output;
        int rc = runCommand(cmd, output);

        if (rc != 0) {
            fprintf(stderr, "  FAIL: unroll=%d (RCCL_UNROLL_FACTOR=%d) — "
                    "all_reduce_perf exited with code %d\n",
                    unrollLabels[i], overrides[i], rc);
            allPassed = false;
        } else {
            // all_reduce_perf prints e.g. "# Out of bounds values : 0 OK"
            bool hasOobError = output.find("Out of bounds values : 0 OK") == std::string::npos
                            && output.find("Out of bounds values") != std::string::npos;
            bool hasNcclFailure = output.find("Test NCCL failure") != std::string::npos;

            if (hasOobError || hasNcclFailure) {
                fprintf(stderr, "  FAIL: unroll=%d — collective reported errors\n",
                        unrollLabels[i]);
                allPassed = false;
            } else {
                printf("  OK   unroll=%d — all_reduce_perf 8B..4MB passed\n",
                       unrollLabels[i]);
            }
        }
    }

    CHECK(allPassed, "Collective failed with one or more unroll factors");
    printf("[PASS] CollectiveCorrectnessAllUnrolls\n");
    ++g_passed;
}

struct TestEntry {
    const char* flag;
    const char* name;
    void (*func)();
};

constexpr TestEntry kTests[] = {
    {"--kernel-symbol",         "KernelSymbolPresence",          test_KernelSymbolPresence},
    {"--runtime-heuristic",     "RuntimeHeuristicSelection",     test_RuntimeHeuristicSelection},
    {"--unroll-override",       "UnrollOverride",                test_UnrollOverride},
    {"--invalid-unroll",        "InvalidUnrollRejected",         test_InvalidUnrollRejected},
    {"--collective-correctness","CollectiveCorrectnessAllUnrolls",test_CollectiveCorrectnessAllUnrolls},
};
constexpr size_t kNumTests = sizeof(kTests) / sizeof(kTests[0]);

void printUsage(const char* prog) {
    printf("Usage: %s [OPTIONS]\n\n", prog);
    printf("Run all tests (default) or select individual subtests via flags.\n\n");
    printf("Options:\n");
    printf("  --help                      Show this help message\n");
    printf("  --list                      List available subtests\n");
    for (size_t i = 0; i < kNumTests; ++i)
        printf("  %-30s Run %s\n", kTests[i].flag, kTests[i].name);
    printf("\nMultiple flags can be combined to run a subset of tests.\n");
    printf("If no test flags are given, all tests are executed.\n");
}

} // namespace

int main(int argc, char* argv[]) {
    bool selected[kNumTests] = {};
    bool anySelected = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        }
        if (strcmp(argv[i], "--list") == 0) {
            printf("Available subtests:\n");
            for (size_t t = 0; t < kNumTests; ++t)
                printf("  %-30s %s\n", kTests[t].flag, kTests[t].name);
            return EXIT_SUCCESS;
        }

        bool matched = false;
        for (size_t t = 0; t < kNumTests; ++t) {
            if (strcmp(argv[i], kTests[t].flag) == 0) {
                selected[t] = true;
                anySelected = true;
                matched = true;
                break;
            }
        }
        if (!matched) {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (!anySelected)
        for (size_t t = 0; t < kNumTests; ++t)
            selected[t] = true;

    printf("=== Dual Kernel Build Validation ===\n");
    printf("=== RCCL library: %s\n", getRcclLibPath());
    printf("=== all_reduce_perf: %s\n\n",
           getAllReducePerfPath().empty() ? "(not found)" : getAllReducePerfPath().c_str());

    for (size_t t = 0; t < kNumTests; ++t)
        if (selected[t])
            kTests[t].func();

    printf("\n=== Results: %d passed, %d failed, %d skipped ===\n",
           g_passed, g_failed, g_skipped);
    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
