#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

__global__ void burn(float* value, int steps) {
    float current = 1.0f;
    for (int step = 0; step < steps; ++step) {
        current = current * 1.000001f + 0.000001f;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        value[0] = current;
    }
}

void burn_device(int device, int seconds) {
    if (hipSetDevice(device) != hipSuccess) {
        std::fprintf(stderr, "hipSetDevice %d failed\n", device);
        return;
    }
    float* value = nullptr;
    if (hipMalloc(&value, sizeof(float)) != hipSuccess) {
        std::fprintf(stderr, "hipMalloc on %d failed\n", device);
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count() < seconds) {
        burn<<<64, 256>>>(value, 50000);
        if (hipDeviceSynchronize() != hipSuccess) {
            break;
        }
    }
    const hipError_t freed = hipFree(value);
    (void)freed;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: xcd_burn <seconds> <hip-device> [hip-device...]\n");
        return 1;
    }
    const int seconds = std::atoi(argv[1]);
    std::vector<std::thread> threads;
    for (int index = 2; index < argc; ++index) {
        threads.emplace_back(burn_device, std::atoi(argv[index]), seconds);
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    return 0;
}
