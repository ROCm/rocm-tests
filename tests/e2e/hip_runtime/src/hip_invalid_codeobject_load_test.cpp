/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

 * @brief Prevents regression of the null-pointer dereference in HIP runtime
 *        when operator[] inserts a null entry for an invalid code object path.
 */
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

int g_passed = 0;
int g_failed = 0;

#define CHECK(cond, msg)                                          \
    do {                                                          \
        if (!(cond)) {                                            \
            fprintf(stderr, "  FAIL: %s\n        %s:%d\n",       \
                    msg, __FILE__, __LINE__);                     \
            ++g_failed;                                           \
            return;                                               \
        }                                                         \
    } while (0)

#define CHECK_EQ(actual, expected, msg)                           \
    do {                                                          \
        if ((actual) != (expected)) {                             \
            fprintf(stderr, "  FAIL: %s (got %d, expected %d)\n" \
                    "        %s:%d\n",                            \
                    msg, (int)(actual), (int)(expected),          \
                    __FILE__, __LINE__);                          \
            ++g_failed;                                           \
            return;                                               \
        }                                                         \
    } while (0)

bool fileExists(const std::string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

constexpr const char* kNonexistentFile = "/tmp/hip_invalid_codeobj_test.co";
std::string getRocmPath() {
    const char* env = std::getenv("ROCM_PATH");
    if (env && env[0] != '\0')
        return env;
    return "/opt/rocm";
}

// hipBLASLt ships HSACO files under a per-arch subdirectory:
//   ${ROCM_PATH}/lib/hipblaslt/library/<arch>/Kernels.so-000-<arch>.hsaco
std::string getHipblasltLibDir(const std::string& arch) {
    return getRocmPath() + "/lib/hipblaslt/library/" + arch;
}

std::string getGpuArch() {
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, 0) != hipSuccess)
        return "";
    std::string arch = props.gcnArchName;
    auto pos = arch.find(':');
    if (pos != std::string::npos)
        arch = arch.substr(0, pos);
    return arch;
}

void test_LoadNonexistentFile_ReturnsFileNotFound() {
    printf("[RUN ] LoadNonexistentFile_ReturnsFileNotFound\n");

    CHECK(!fileExists(kNonexistentFile),
          "Precondition failed: test file should not exist on disk");

    hipModule_t module;
    hipError_t err = hipModuleLoad(&module, kNonexistentFile);

    CHECK_EQ(err, hipErrorFileNotFound,
             "Expected hipErrorFileNotFound for missing file");

    printf("[PASS] LoadNonexistentFile_ReturnsFileNotFound\n");
    ++g_passed;
}

void test_RepeatedFailedLoads_ReturnConsistentError_NoCrashOnCleanup() {
    printf("[RUN ] RepeatedFailedLoads_ReturnConsistentError_NoCrashOnCleanup\n");

    CHECK(!fileExists(kNonexistentFile),
          "Precondition failed: test file should not exist on disk");

    hipError_t firstErr = hipSuccess;

    for (int i = 0; i < 1000; ++i) {
        hipModule_t module;
        hipError_t err = hipModuleLoad(&module, kNonexistentFile);

        if (err == hipSuccess) {
            fprintf(stderr, "  FAIL: Iteration %d: load should have failed\n", i);
            ++g_failed;
            return;
        }

        if (i == 0) {
            firstErr = err;
        } else if (err != firstErr) {
            fprintf(stderr, "  FAIL: Iteration %d: error code changed from %s to %s\n",
                    i, hipGetErrorName(firstErr), hipGetErrorName(err));
            ++g_failed;
            return;
        }
    }

    (void)hipDeviceReset();

    printf("[PASS] RepeatedFailedLoads_ReturnConsistentError_NoCrashOnCleanup\n");
    ++g_passed;
}

void test_ArchSpecificKernelFile_LoadsSuccessfully() {
    printf("[RUN ] ArchSpecificKernelFile_LoadsSuccessfully\n");

    std::string arch = getGpuArch();
    CHECK(!arch.empty(), "Could not detect GPU architecture");

    // hipBLASLt stores HSACO files under a per-arch subdirectory
    std::string libDir = getHipblasltLibDir(arch);

    if (!fileExists(libDir)) {
        printf("[SKIP] %s not found\n", libDir.c_str());
        return;
    }

    std::string filename = "Kernels.so-000-" + arch + ".hsaco";
    std::string path = libDir + "/" + filename;

    CHECK(fileExists(path),
          (std::string("Kernel file not found: ") + path).c_str());

    hipModule_t module;
    hipError_t err = hipModuleLoad(&module, path.c_str());
    printf("  Loaded %s: %s\n", filename.c_str(), hipGetErrorName(err));

    if (err != hipSuccess) {
        fprintf(stderr, "  FAIL: Failed to load %s: %s - %s\n",
                filename.c_str(), hipGetErrorName(err), hipGetErrorString(err));
        ++g_failed;
        (void)hipDeviceReset();
        return;
    }

    (void)hipModuleUnload(module);
    (void)hipDeviceReset();

    printf("[PASS] ArchSpecificKernelFile_LoadsSuccessfully\n");
    ++g_passed;
}

struct TestEntry {
    const char* name;
    void (*func)();
};

const TestEntry kTests[] = {
    {"LoadNonexistent",   test_LoadNonexistentFile_ReturnsFileNotFound},
    {"RepeatedLoads",     test_RepeatedFailedLoads_ReturnConsistentError_NoCrashOnCleanup},
    {"ArchSpecificLoad",  test_ArchSpecificKernelFile_LoadsSuccessfully},
};

constexpr int kNumTests = sizeof(kTests) / sizeof(kTests[0]);

void printUsage(const char* prog) {
    printf("Usage: %s [OPTIONS] [TEST_NAME ...]\n\n", prog);
    printf("Options:\n");
    printf("  --list, -l         List available test names and exit\n");
    printf("  --test, -t NAME    Run only the named test (repeatable)\n");
    printf("  --help, -h         Show this help message\n\n");
    printf("If no tests are specified, all tests are run.\n");
    printf("Test names are case-sensitive partial matches (substring).\n");
}

} // namespace

int main(int argc, char* argv[]) {
    std::vector<std::string> selected;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        }
        if (arg == "--list" || arg == "-l") {
            printf("Available tests:\n");
            for (int t = 0; t < kNumTests; ++t)
                printf("  %s\n", kTests[t].name);
            return EXIT_SUCCESS;
        }
        if (arg == "--test" || arg == "-t") {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s requires a test name\n", arg.c_str());
                return EXIT_FAILURE;
            }
            selected.push_back(argv[++i]);
        } else if (arg[0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            printUsage(argv[0]);
            return EXIT_FAILURE;
        } else {
            selected.push_back(arg);
        }
    }

    printf("=== HipInvalidCodeObjectLoad Tests ===\n\n");

    int ran = 0;
    for (int t = 0; t < kNumTests; ++t) {
        if (!selected.empty()) {
            bool match = false;
            for (const auto& s : selected) {
                if (std::string(kTests[t].name).find(s) != std::string::npos) {
                    match = true;
                    break;
                }
            }
            if (!match) continue;
        }
        kTests[t].func();
        ++ran;
    }

    if (ran == 0) {
        fprintf(stderr, "No tests matched the given filter(s).\n");
        return EXIT_FAILURE;
    }

    printf("\n=== Results: %d passed, %d failed (of %d run) ===\n",
           g_passed, g_failed, ran);
    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
