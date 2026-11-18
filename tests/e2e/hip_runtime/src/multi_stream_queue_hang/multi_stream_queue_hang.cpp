// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
/*
 * multi_stream_queue_hang — multi-thread hipMemset*Async + event/stream waits.
 * Runner sets GPU_MAX_HW_QUEUES=1.
 */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <random>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

constexpr const char* kTestName = "multi_stream_queue_hang";
std::mutex gErrMu;
std::atomic<bool> gThreadFailed{false};

#define CHECK_HIP(call)                                                                            \
  do {                                                                                             \
    hipError_t _e = (call);                                                                        \
    if (_e != hipSuccess) {                                                                        \
      std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,               \
                   hipGetErrorString(_e));                                                         \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

#define CHECK_HIP_THREAD(call)                                                                     \
  do {                                                                                             \
    hipError_t _e = (call);                                                                        \
    if (_e != hipSuccess) {                                                                        \
      std::lock_guard<std::mutex> _lk(gErrMu);                                                     \
      std::fprintf(stderr, "HIP thread error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,        \
                   hipGetErrorString(_e));                                                         \
      gThreadFailed.store(true);                                                                   \
      return;                                                                                      \
    }                                                                                              \
  } while (0)

enum TestType { SameStream = 0, DifferentStreams };

constexpr size_t kMaxProducers = 127;

size_t envSizeOr(const char* name, size_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || parsed == 0UL) {
    return fallback;
  }
  return std::min<size_t>(static_cast<size_t>(parsed), kMaxProducers);
}

// More producers than the legacy hw/4 heuristic so DifferentStreams stresses one AQL ring harder
// under GPU_MAX_HW_QUEUES=1.
size_t defaultProducerCount() {
  const size_t hw = std::thread::hardware_concurrency();
  if (hw == 0) {
    return 32;
  }
  return std::min(std::max(hw, size_t{32}), kMaxProducers);
}

template <typename T>
struct DeviceMemGuard {
  T* ptr = nullptr;
  ~DeviceMemGuard() {
    if (ptr) {
      (void)hipFree(ptr);
      ptr = nullptr;
    }
  }
  DeviceMemGuard() = default;
  DeviceMemGuard(const DeviceMemGuard&) = delete;
  DeviceMemGuard& operator=(const DeviceMemGuard&) = delete;
};

struct EventGuard {
  hipEvent_t event = nullptr;
  ~EventGuard() {
    if (event) {
      (void)hipEventDestroy(event);
      event = nullptr;
    }
  }
  EventGuard() = default;
  EventGuard(const EventGuard&) = delete;
  EventGuard& operator=(const EventGuard&) = delete;
};

template <typename Func, typename T>
void threadCall(Func f, hipStream_t stream) {
  static_assert((std::is_same<Func, hipError_t (*)(void*, int, size_t, hipStream_t)>::value ||
                 std::is_same<Func, hipError_t (*)(hipDeviceptr_t, int, size_t, hipStream_t)>::value ||
                 std::is_same<Func, hipError_t (*)(hipDeviceptr_t, unsigned short, size_t,
                                                   hipStream_t)>::value ||
                 std::is_same<Func, hipError_t (*)(hipDeviceptr_t, unsigned char, size_t,
                                                   hipStream_t)>::value),
                "Func must be hipMemsetAsync or hipMemsetD*Async");

  constexpr bool cast_2_void =
      std::is_same<Func, hipError_t (*)(void*, int, size_t, hipStream_t)>::value;
  using unsigned_t = typename std::make_unsigned<T>::type;

  std::mt19937_64 engine(std::random_device{}());
  auto distribution = std::uniform_int_distribution<unsigned int>(
      0, static_cast<unsigned int>(std::numeric_limits<unsigned_t>::max()));

  DeviceMemGuard<T> mem;
  EventGuard ev;
  constexpr size_t size = 1024;
  constexpr size_t iter = 512;
  CHECK_HIP_THREAD(hipMalloc(&mem.ptr, sizeof(T) * size));
  CHECK_HIP_THREAD(hipEventCreate(&ev.event));

  union {
    T t_val;
    unsigned_t u_val;
  } overlay_val;

  std::vector<T> dst(size, T{});
  for (size_t i = 0; i < iter; i++) {
    if (gThreadFailed.load()) break;
    overlay_val.u_val = static_cast<unsigned_t>(distribution(engine));
    if constexpr (cast_2_void) {
      CHECK_HIP_THREAD(f(static_cast<void*>(mem.ptr), overlay_val.t_val, size, stream));
    } else {
      hipDeviceptr_t dptr = reinterpret_cast<hipDeviceptr_t>(mem.ptr);
      CHECK_HIP_THREAD(f(dptr, overlay_val.t_val, size, stream));
    }
    CHECK_HIP_THREAD(
        hipMemcpyAsync(dst.data(), mem.ptr, size * sizeof(T), hipMemcpyDeviceToHost, stream));
    CHECK_HIP_THREAD(hipEventRecord(ev.event, stream));
    CHECK_HIP_THREAD(hipStreamWaitEvent(stream, ev.event, 0));
    CHECK_HIP_THREAD(hipEventSynchronize(ev.event));
    if (!std::all_of(dst.begin(), dst.end(), [&](T v) { return v == overlay_val.t_val; })) {
      std::lock_guard<std::mutex> lk(gErrMu);
      std::fprintf(stderr, "memset content mismatch\n");
      gThreadFailed.store(true);
      break;
    }
  }
}

template <typename Func, typename T>
void launchThreads(Func f, TestType type) {
  const size_t defaults = defaultProducerCount();
  const size_t num_threads = envSizeOr("MSQH_THREADS", defaults);
  const size_t want_streams = envSizeOr("MSQH_STREAMS", num_threads);
  const size_t num_streams = (type == SameStream) ? 1 : std::max(want_streams, size_t{1});
  std::fprintf(stderr, "launchThreads type=%s threads=%zu streams=%zu\n",
               type == SameStream ? "SameStream" : "DifferentStreams", num_threads, num_streams);

  std::vector<hipStream_t> streams(num_streams, nullptr);
  for (size_t i = 0; i < num_streams; i++) {
    CHECK_HIP(hipStreamCreate(&streams[i]));
  }

  std::vector<std::thread> pool;
  pool.reserve(num_threads);
  for (size_t i = 0; i < num_threads; i++) {
    hipStream_t stream = (type == SameStream) ? streams[0] : streams[i % num_streams];
    pool.emplace_back(threadCall<Func, T>, f, stream);
  }
  for (auto& t : pool) t.join();

  if (gThreadFailed.load()) {
    std::fprintf(stderr, "thread failure in launchThreads\n");
    std::exit(1);
  }

  for (size_t i = 0; i < num_streams; i++) {
    CHECK_HIP(hipStreamDestroy(streams[i]));
  }
}

void runScenario() {
  using MemsetAsync = hipError_t (*)(void*, int, size_t, hipStream_t);
  using MemsetD8 = hipError_t (*)(hipDeviceptr_t, unsigned char, size_t, hipStream_t);
  using MemsetD16 = hipError_t (*)(hipDeviceptr_t, unsigned short, size_t, hipStream_t);
  using MemsetD32 = hipError_t (*)(hipDeviceptr_t, int, size_t, hipStream_t);

  launchThreads<MemsetAsync, char>(hipMemsetAsync, SameStream);
  launchThreads<MemsetAsync, char>(hipMemsetAsync, DifferentStreams);
  launchThreads<MemsetD8, unsigned char>(hipMemsetD8Async, SameStream);
  launchThreads<MemsetD8, unsigned char>(hipMemsetD8Async, DifferentStreams);
  launchThreads<MemsetD16, unsigned short>(hipMemsetD16Async, SameStream);
  launchThreads<MemsetD16, unsigned short>(hipMemsetD16Async, DifferentStreams);
  launchThreads<MemsetD32, int>(hipMemsetD32Async, SameStream);
  launchThreads<MemsetD32, int>(hipMemsetD32Async, DifferentStreams);
}

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--list") == 0 || std::strcmp(argv[i], "--name") == 0) {
      std::printf("%s\n", kTestName);
      return 0;
    }
    if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
      std::printf("Usage: %s [--list|--name]\n", argv[0]);
      std::printf("Env: MSQH_THREADS, MSQH_STREAMS (DifferentStreams; default >=32, cap %zu)\n",
                  kMaxProducers);
      return 0;
    }
  }

  int ndev = 0;
  CHECK_HIP(hipGetDeviceCount(&ndev));
  if (ndev < 1) {
    std::fprintf(stderr, "no HIP devices\n");
    return 1;
  }
  CHECK_HIP(hipSetDevice(0));

  runScenario();
  std::printf("%s PASSED\n", kTestName);
  return 0;
}
