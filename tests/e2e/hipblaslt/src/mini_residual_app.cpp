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
*/

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

// HIP/hipBLASLt are "other"; cpplint may treat as C — suppress include_order.
#include <hip/hip_runtime.h>   // NOLINT(build/include_order)
#include <hipblaslt/hipblaslt.h>   // NOLINT(build/include_order)

namespace {

constexpr int kOk = 0;
constexpr int kHipErr = 1;
constexpr int kLtErr = 2;
constexpr int kCliErr = 3;
constexpr int kNoHeuristics = 5;
constexpr int kTrainFail = 11;
constexpr int kNoPassing = 12;

inline bool HipCheck(hipError_t st, const char* where) {
  if (st != hipSuccess) {
    std::cerr << "[HIP] Error at " << where << ": "
              << hipGetErrorString(st) << std::endl;
    return false;
  }
  return true;
}

inline bool LtCheck(hipblasStatus_t st, const char* where) {
  if (st != HIPBLAS_STATUS_SUCCESS) {
    std::cerr << "[hipBLASLt] Error at " << where
              << ": status=" << static_cast<int>(st) << std::endl;
    return false;
  }
  return true;
}

// ------------------------------ RAII wrappers ------------------------------

struct HipStream {
  hipStream_t s{nullptr};
  HipStream() = default;
  explicit HipStream(unsigned /*flags*/) {
    HipCheck(hipStreamCreate(&s), "hipStreamCreate");
  }
  HipStream(const HipStream&) = delete;
  HipStream& operator=(const HipStream&) = delete;
  HipStream(HipStream&& o) noexcept : s(o.s) { o.s = nullptr; }
  HipStream& operator=(HipStream&& o) noexcept {
    if (this != &o) {
      Destroy();
      s = o.s;
      o.s = nullptr;
    }
    return *this;
  }
  void Destroy() {
    if (s) {
      (void)hipStreamDestroy(s);
      s = nullptr;
    }
  }
  ~HipStream() { Destroy(); }
};

struct HipEvent {
  hipEvent_t e{nullptr};
  HipEvent() = default;
  explicit HipEvent(unsigned flags) {
    if (HipCheck(hipEventCreateWithFlags(&e, flags),
                 "hipEventCreateWithFlags")) {
      /* e set by API */
    } else {
      e = nullptr;
    }
  }
  HipEvent(const HipEvent&) = delete;
  HipEvent& operator=(const HipEvent&) = delete;
  HipEvent(HipEvent&& o) noexcept : e(o.e) { o.e = nullptr; }
  HipEvent& operator=(HipEvent&& o) noexcept {
    if (this != &o) {
      Destroy();
      e = o.e;
      o.e = nullptr;
    }
    return *this;
  }
  void Destroy() {
    if (e) {
      (void)hipEventDestroy(e);
      e = nullptr;
    }
  }
  ~HipEvent() { Destroy(); }
};

struct LtHandle {
  hipblasLtHandle_t h{nullptr};
  LtHandle() { LtCheck(hipblasLtCreate(&h), "hipblasLtCreate"); }
  LtHandle(const LtHandle&) = delete;
  LtHandle& operator=(const LtHandle&) = delete;
  LtHandle(LtHandle&& o) noexcept : h(o.h) { o.h = nullptr; }
  LtHandle& operator=(LtHandle&& o) noexcept {
    if (this != &o) {
      Destroy();
      h = o.h;
      o.h = nullptr;
    }
    return *this;
  }
  void Destroy() {
    if (h) {
      (void)hipblasLtDestroy(h);
      h = nullptr;
    }
  }
  ~LtHandle() { Destroy(); }
};

struct LtMatmulDesc {
  hipblasLtMatmulDesc_t d{nullptr};
  LtMatmulDesc(hipblasComputeType_t ctype, hipDataType dtype) {
    LtCheck(hipblasLtMatmulDescCreate(&d, ctype, dtype), "MatmulDescCreate");
  }
  LtMatmulDesc(const LtMatmulDesc&) = delete;
  LtMatmulDesc& operator=(const LtMatmulDesc&) = delete;
  LtMatmulDesc(LtMatmulDesc&& o) noexcept : d(o.d) { o.d = nullptr; }
  LtMatmulDesc& operator=(LtMatmulDesc&& o) noexcept {
    if (this != &o) {
      Destroy();
      d = o.d;
      o.d = nullptr;
    }
    return *this;
  }
  template <typename T>
  bool SetAttr(hipblasLtMatmulDescAttributes_t attr, const T& val) {
    return LtCheck(
        hipblasLtMatmulDescSetAttribute(d, attr, &val, sizeof(T)),
        "MatmulDescSetAttribute");
  }
  void Destroy() {
    if (d) {
      (void)hipblasLtMatmulDescDestroy(d);
      d = nullptr;
    }
  }
  ~LtMatmulDesc() { Destroy(); }
};

struct LtMatrixLayout {
  hipblasLtMatrixLayout_t l{nullptr};
  LtMatrixLayout(hipDataType dtype, int rows, int cols, int ld) {
    LtCheck(hipblasLtMatrixLayoutCreate(&l, dtype, rows, cols, ld),
            "MatrixLayoutCreate");
  }
  LtMatrixLayout(const LtMatrixLayout&) = delete;
  LtMatrixLayout& operator=(const LtMatrixLayout&) = delete;
  LtMatrixLayout(LtMatrixLayout&& o) noexcept : l(o.l) { o.l = nullptr; }
  LtMatrixLayout& operator=(LtMatrixLayout&& o) noexcept {
    if (this != &o) {
      Destroy();
      l = o.l;
      o.l = nullptr;
    }
    return *this;
  }
  void Destroy() {
    if (l) {
      (void)hipblasLtMatrixLayoutDestroy(l);
      l = nullptr;
    }
  }
  ~LtMatrixLayout() { Destroy(); }
};

struct LtPreference {
  hipblasLtMatmulPreference_t p{nullptr};
  LtPreference() {
    LtCheck(hipblasLtMatmulPreferenceCreate(&p), "MatmulPreferenceCreate");
  }
  LtPreference(const LtPreference&) = delete;
  LtPreference& operator=(const LtPreference&) = delete;
  LtPreference(LtPreference&& o) noexcept : p(o.p) { o.p = nullptr; }
  LtPreference& operator=(LtPreference&& o) noexcept {
    if (this != &o) {
      Destroy();
      p = o.p;
      o.p = nullptr;
    }
    return *this;
  }
  template <typename T>
  bool SetAttr(hipblasLtMatmulPreferenceAttributes_t attr, const T& val) {
    return LtCheck(
        hipblasLtMatmulPreferenceSetAttribute(p, attr, &val, sizeof(T)),
        "MatmulPreferenceSetAttribute");
  }
  void Destroy() {
    if (p) {
      (void)hipblasLtMatmulPreferenceDestroy(p);
      p = nullptr;
    }
  }
  ~LtPreference() { Destroy(); }
};

enum class MemMode { kDevice, kHostMappedC, kManaged };

struct DeviceBuffer {
  void* host{nullptr};
  void* dev{nullptr};
  std::size_t bytes{0};
  MemMode mode{MemMode::kDevice};

  DeviceBuffer() = default;
  DeviceBuffer(std::size_t n, MemMode m) { Reset(n, m); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& o) noexcept {
    host = o.host;
    dev = o.dev;
    bytes = o.bytes;
    mode = o.mode;
    o.host = nullptr;
    o.dev = nullptr;
    o.bytes = 0;
    o.mode = MemMode::kDevice;
  }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      Free();
      host = o.host;
      dev = o.dev;
      bytes = o.bytes;
      mode = o.mode;
      o.host = nullptr;
      o.dev = nullptr;
      o.bytes = 0;
      o.mode = MemMode::kDevice;
    }
    return *this;
  }

  bool Reset(std::size_t n, MemMode m) {
    Free();
    bytes = n;
    mode = m;
    host = nullptr;
    dev = nullptr;
    switch (mode) {
      case MemMode::kDevice: {
        if (!HipCheck(hipMalloc(&host, bytes), "hipMalloc")) return false;
        dev = host;
        break;
      }
      case MemMode::kHostMappedC: {
        if (!HipCheck(hipHostMalloc(&host, bytes, hipHostMallocMapped),
                      "hipHostMalloc"))
          return false;
        if (!HipCheck(hipHostGetDevicePointer(&dev, host, 0),
                      "hipHostGetDevicePointer")) {
          (void)hipHostFree(host);
          host = nullptr;
          return false;
        }
        break;
      }
      case MemMode::kManaged: {
        if (!HipCheck(hipMallocManaged(&host, bytes), "hipMallocManaged"))
          return false;
        dev = host;
        break;
      }
    }
    return true;
  }

  void Free() {
    if (!host) return;
    if (mode == MemMode::kHostMappedC) {
      (void)hipHostFree(host);
    } else {
      (void)hipFree(host);
    }
    host = nullptr;
    dev = nullptr;
    bytes = 0;
  }

  bool Zero() { return HipCheck(hipMemset(dev, 0, bytes), "hipMemset(Zero)"); }
  bool Poison(std::uint8_t v = 0xCC) {
    return HipCheck(hipMemset(dev, v, bytes), "hipMemset(Poison)");
  }
  bool CopyH2D(const void* h) {
    return HipCheck(
        hipMemcpy(dev, h, bytes, hipMemcpyHostToDevice), "hipMemcpy H2D");
  }
  bool CopyD2H(void* h) {
    return HipCheck(
        hipMemcpy(h, dev, bytes, hipMemcpyDeviceToHost), "hipMemcpy D2H");
  }

  ~DeviceBuffer() { Free(); }
};

// ------------------------------ Data type selection --------------------------
// Compile-time: -DHIP_R_8F_E4M3 / -DHIP_R_8F_E5M2 force FP8. If not set,
// runtime tries FP8 first and falls back to BF16.

#if defined(HIP_R_8F_E4M3)
#define A_HIP_DTYPE_COMPILE HIP_R_8F_E4M3
#define A_BYTES_COMPILE 1
#define A_RUNTIME_SELECT 0
#elif defined(HIP_R_8F_E4M3_FNUZ)
#define A_HIP_DTYPE_COMPILE HIP_R_8F_E4M3_FNUZ
#define A_BYTES_COMPILE 1
#define A_RUNTIME_SELECT 0
#else
#define A_HIP_DTYPE_COMPILE HIP_R_16BF
#define A_BYTES_COMPILE 2
#define A_RUNTIME_SELECT 1
#endif

#if defined(HIP_R_8F_E5M2)
#define B_HIP_DTYPE_COMPILE HIP_R_8F_E5M2
#define B_BYTES_COMPILE 1
#define B_RUNTIME_SELECT 0
#elif defined(HIP_R_8F_E5M2_FNUZ)
#define B_HIP_DTYPE_COMPILE HIP_R_8F_E5M2_FNUZ
#define B_BYTES_COMPILE 1
#define B_RUNTIME_SELECT 0
#else
#define B_HIP_DTYPE_COMPILE HIP_R_16BF
#define B_BYTES_COMPILE 2
#define B_RUNTIME_SELECT 1
#endif

// ------------------------------ BF16 helpers ------------------------------

__host__ __device__ inline float Bf16ToF32(std::uint16_t h) {
#if defined(__HIP_DEVICE_COMPILE__)
  std::uint32_t w = (static_cast<std::uint32_t>(h)) << 16;
  return __uint_as_float(w);
#else
  std::uint32_t w = (static_cast<std::uint32_t>(h)) << 16;
  float f;
  std::memcpy(&f, &w, sizeof(float));
  return f;
#endif
}

__host__ __device__ inline std::uint16_t F32ToBf16(float f) {
#if defined(__HIP_DEVICE_COMPILE__)
  std::uint32_t w = __float_as_uint(f);
  return static_cast<std::uint16_t>(w >> 16);
#else
  std::uint32_t w;
  std::memcpy(&w, &f, sizeof(std::uint32_t));
  return static_cast<std::uint16_t>(w >> 16);
#endif
}

// ------------------------------ Device kernels ------------------------------

__global__ void K_QFp32ToFp8E4M3(const float* x, std::uint8_t* y,
                                 std::size_t n, float s) {
  std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = fminf(fmaxf(x[i] * s, -448.f), 448.f);
  y[i] = (v == 0.f) ? 0u : (v > 0.f ? 0x3Cu : 0xBCu);
}

__global__ void K_QFp32ToBf8E5M2(const float* x, std::uint8_t* y,
                                 std::size_t n, float s) {
  std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = fminf(fmaxf(x[i] * s, -57344.f), 57344.f);
  y[i] = (v == 0.f) ? 0u : (v > 0.f ? 0x3Cu : 0xBCu);
}

__global__ void K_QFp32ToBf16(const float* x, std::uint16_t* y,
                              std::size_t n, float s) {
  std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] = F32ToBf16(x[i] * s);
}

__global__ void K_ZeroColsU8(std::uint8_t* A, int rowsK, int /*colsM*/,
                             int ldK, int padCols) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < padCols && r < rowsK) {
    A[c * ldK + r] = 0;
  }
}

__global__ void K_ZeroColsU16(std::uint16_t* A, int rowsK, int /*colsM*/,
                              int ldK, int padCols) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < padCols && r < rowsK) {
    A[c * ldK + r] = 0;
  }
}

__global__ void K_CheckNonFiniteBf16(const std::uint16_t* x,
                                     std::size_t n, int* flag) {
  std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = Bf16ToF32(x[i]);
  if (!isfinite(v)) atomicExch(flag, 1);
}

__global__ void K_CheckZeroRegionBf16(const std::uint16_t* D,
                                      int /*rowsM*/, int colsN,
                                      int ldD, int pad_rows, int* flag) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < pad_rows && c < colsN) {
    if (D[c * ldD + r] != 0) atomicExch(flag, 1);
  }
}

// ------------------------------ App config ------------------------------

struct Config {
  int M = 8192;
  int N = 32768;
  int K = 1024;
  int lda = 0, ldb = 0, ldc = 0, ldd = 0, ldc_diff = 0;
  int iters = 1000;
  float alpha = 1.0f, beta = 1.0f;
  bool transA = true, transB = false;
  bool alpha_zero = false;
  int pad_rows = -1;
  MemMode mem = MemMode::kDevice;
  bool validate = true, verbose = false;

  bool run_all_algos = false;
  int streams = 1;

  int nonzero_iters = 0;
};

inline void Usage(const char* p) {
  std::cout << "Usage: " << p << " [--M 8192 --N 32768 --K 1024]\n"
            << "  [--lda <int> --ldb <int> --ldc <int> --ldd <int> "
            << "--ldc_diff <int>]\n"
            << "  [--iters 1000] [--alpha 1.0] [--beta 1.0] [--alpha0]\n"
            << "  [--transA T|N] [--transB T|N]\n"
            << "  [--memory_mode device|host_mapped_C|managed]\n"
            << "  [--pad_rows <int>]  # -1 => zero whole A; >=0 => zero band\n"
            << "  [--no_validate] [--verbose]\n"
            << "  [--run_all_algos] [--streams N]\n"
            << "  [--nonzero_iters N]\n";
}

inline const char* Need(const char* flag, int* i, int argc, char** argv) {
  if (*i + 1 >= argc) {
    std::cerr << "Missing value after " << flag << std::endl;
    std::exit(kCliErr);
  }
  return argv[++(*i)];
}

// ------------------------------ Quantize / Fill helpers ----------------------

bool QuantizeA(DeviceBuffer* A, std::size_t elems, bool use_bytes) {
  float* tmp = nullptr;
  if (!HipCheck(hipMalloc(&tmp, elems * sizeof(float)), "hipMalloc tmpA")) {
    return false;
  }
  std::vector<float> h(elems);
  std::mt19937 g(12345);
  std::normal_distribution<float> nd(0.f, 0.02f);
  for (auto& v : h) v = nd(g);

  bool ok = HipCheck(
      hipMemcpy(tmp, h.data(), elems * sizeof(float),
                hipMemcpyHostToDevice),
      "Memcpy tmpA");
  if (!ok) {
    (void)hipFree(tmp);
    return false;
  }

  dim3 bl(256), gr((static_cast<std::uint64_t>(elems) + bl.x - 1) / bl.x);
  if (use_bytes) {
    K_QFp32ToFp8E4M3<<<gr, bl>>>(tmp,
                                 static_cast<std::uint8_t*>(A->dev),
                                 elems, 1.0f);
  } else {
    K_QFp32ToBf16<<<gr, bl>>>(tmp,
                              static_cast<std::uint16_t*>(A->dev),
                              elems, 1.0f);
  }
  ok = HipCheck(hipDeviceSynchronize(), "QuantizeA sync");
  (void)hipFree(tmp);
  return ok;
}

bool QuantizeB(DeviceBuffer* B, std::size_t elems, bool use_bytes) {
  float* tmp = nullptr;
  if (!HipCheck(hipMalloc(&tmp, elems * sizeof(float)), "hipMalloc tmpB")) {
    return false;
  }
  std::vector<float> h(elems);
  std::mt19937 g(23456);
  std::uniform_real_distribution<float> ud(-1.f, 1.f);
  for (auto& v : h) v = ud(g);

  bool ok = HipCheck(
      hipMemcpy(tmp, h.data(), elems * sizeof(float),
                hipMemcpyHostToDevice),
      "Memcpy tmpB");
  if (!ok) {
    (void)hipFree(tmp);
    return false;
  }

  dim3 bl(256), gr((static_cast<std::uint64_t>(elems) + bl.x - 1) / bl.x);
  if (use_bytes) {
    K_QFp32ToBf8E5M2<<<gr, bl>>>(tmp,
                                 static_cast<std::uint8_t*>(B->dev),
                                 elems, 1.0f);
  } else {
    K_QFp32ToBf16<<<gr, bl>>>(tmp,
                              static_cast<std::uint16_t*>(B->dev),
                              elems, 1.0f);
  }
  ok = HipCheck(hipDeviceSynchronize(), "QuantizeB sync");
  (void)hipFree(tmp);
  return ok;
}

bool FillCRandomBf16(DeviceBuffer* C, std::size_t elems) {
  float* tmp = nullptr;
  if (!HipCheck(hipMalloc(&tmp, elems * sizeof(float)), "hipMalloc tmpC")) {
    return false;
  }
  std::vector<float> h(elems);
  std::mt19937 g(34567);
  std::normal_distribution<float> nd(0.f, 0.01f);
  for (auto& v : h) v = nd(g);

  bool ok = HipCheck(
      hipMemcpy(tmp, h.data(), elems * sizeof(float),
                hipMemcpyHostToDevice),
      "Memcpy tmpC");
  if (!ok) {
    (void)hipFree(tmp);
    return false;
  }

  dim3 bl(256), gr((static_cast<std::uint64_t>(elems) + bl.x - 1) / bl.x);
  K_QFp32ToBf16<<<gr, bl>>>(tmp,
                            static_cast<std::uint16_t*>(C->dev),
                            elems, 1.0f);
  ok = HipCheck(hipDeviceSynchronize(), "FillC sync");
  (void)hipFree(tmp);
  return ok;
}

// ------------------------------ FP8 support probe --------------------------
// Returns true if FP8 (E4M3 for A, E5M2 for B) is supported (heuristic found).
// Silent: no error logging when FP8 is not supported (expected on older GPUs).
static bool ProbeFp8Support(int rowsA, int colsA, int lda,
                            int rowsB, int colsB, int ldb,
                            int M, int N, int ldc, int ldd,
                            bool transA, bool transB) {
  LtHandle h;
  const hipblasOperation_t opA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  const hipblasOperation_t opB = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  LtMatrixLayout Ad(HIP_R_8F_E4M3, rowsA, colsA, lda);
  LtMatrixLayout Bd(HIP_R_8F_E5M2, rowsB, colsB, ldb);
  LtMatrixLayout Cd(HIP_R_16BF, M, N, ldc);
  LtMatrixLayout Dd(HIP_R_16BF, M, N, ldd);
  LtMatmulDesc desc(HIPBLAS_COMPUTE_32F, HIP_R_32F);
  if (!desc.SetAttr(HIPBLASLT_MATMUL_DESC_TRANSA, opA)) return false;
  if (!desc.SetAttr(HIPBLASLT_MATMUL_DESC_TRANSB, opB)) return false;
  LtPreference pref;
  const std::size_t ws_bytes = (std::size_t{64} << 20);
  if (!pref.SetAttr(HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, ws_bytes))
    return false;
  constexpr int kMaxCands = 1;
  std::vector<hipblasLtMatmulHeuristicResult_t> cands(
      static_cast<std::size_t>(kMaxCands));
  int n_found = 0;
  const hipblasStatus_t st = hipblasLtMatmulAlgoGetHeuristic(
      h.h, desc.d, Ad.l, Bd.l, Cd.l, Dd.l,
      pref.p, kMaxCands, cands.data(), &n_found);
  return (st == HIPBLAS_STATUS_SUCCESS && n_found > 0);
}

// ------------------------------ A zeroing ------------------------------

bool ZeroABandOrAll(void* dA, const Config& cfg, int a_bytes) {
  if (cfg.pad_rows < 0) {
    const std::size_t elemsA =
        static_cast<std::size_t>(cfg.lda) *
        static_cast<std::size_t>(cfg.transA ? cfg.M : cfg.K);
    const std::size_t bytes_total = elemsA * static_cast<std::size_t>(a_bytes);
    return HipCheck(hipMemset(dA, 0, bytes_total), "Memset A all");
  }
  if (cfg.transA) {
    dim3 bc(16, 16);
    dim3 gc((cfg.pad_rows + bc.x - 1) / bc.x,
            (cfg.K + bc.y - 1) / bc.y);
    if (a_bytes == 1) {
      K_ZeroColsU8<<<gc, bc>>>(static_cast<std::uint8_t*>(dA),
                               cfg.K, cfg.M, cfg.lda, cfg.pad_rows);
    } else {
      K_ZeroColsU16<<<gc, bc>>>(static_cast<std::uint16_t*>(dA),
                                cfg.K, cfg.M, cfg.lda, cfg.pad_rows);
    }
    return HipCheck(hipDeviceSynchronize(), "Zero band sync");
  }
  const std::size_t elemsA =
      static_cast<std::size_t>(cfg.lda) *
      static_cast<std::size_t>(cfg.transA ? cfg.M : cfg.K);
  const std::size_t bytes_total_n = elemsA * static_cast<std::size_t>(a_bytes);
  return HipCheck(hipMemset(dA, 0, bytes_total_n), "Memset A N");
}

// ------------------------------ Validation ------------------------------

bool ValidateZeroAndNonFinite(const Config& cfg,
                              DeviceBuffer* D,
                              DeviceBuffer* flag_buf,
                              bool expect_zero) {
  if (expect_zero) {
    if (!flag_buf->Zero()) return false;
    dim3 bz(16, 16);
    const int pad = (cfg.pad_rows >= 0) ? cfg.pad_rows : cfg.M;
    dim3 gz((cfg.N + bz.x - 1) / bz.x, (pad + bz.y - 1) / bz.y);
    K_CheckZeroRegionBf16<<<gz, bz>>>(
        static_cast<const std::uint16_t*>(D->dev),
        cfg.M, cfg.N, cfg.ldd, pad,
        static_cast<int*>(flag_buf->dev));
    if (!HipCheck(hipDeviceSynchronize(), "Zero-region sync")) return false;
    int hflag = 0;
    if (!HipCheck(hipMemcpy(&hflag, flag_buf->dev, sizeof(int),
                            hipMemcpyDeviceToHost),
                  "Memcpy flag zero"))
      return false;
    if (hflag) return false;
  }

  if (!flag_buf->Zero()) return false;
  const std::size_t elemsD =
      static_cast<std::size_t>(cfg.ldd) * static_cast<std::size_t>(cfg.N);
  dim3 bl(256), gr((elemsD + bl.x - 1) / bl.x);
  K_CheckNonFiniteBf16<<<gr, bl>>>(
      static_cast<const std::uint16_t*>(D->dev), elemsD,
      static_cast<int*>(flag_buf->dev));
  if (!HipCheck(hipDeviceSynchronize(), "Non-finite sync")) return false;
  int hflag2 = 0;
  if (!HipCheck(hipMemcpy(&hflag2, flag_buf->dev, sizeof(int),
                          hipMemcpyDeviceToHost),
                "Memcpy flag nf"))
    return false;
  return hflag2 == 0;
}

// ------------------------------ Timing ------------------------------

float TimeAvgMs(int iters, HipStream* st,
                const std::function<hipblasStatus_t()>& fn) {
  const unsigned kEventFlags = 0U;  // hipEventDefault
  HipEvent start(kEventFlags);
  HipEvent stop(kEventFlags);
  if (!start.e || !stop.e) return std::numeric_limits<float>::infinity();
  if (!HipCheck(hipStreamSynchronize(st->s), "Stream sync pre-time"))
    return std::numeric_limits<float>::infinity();
  if (!HipCheck(hipEventRecord(start.e, st->s), "EventRecord start"))
    return std::numeric_limits<float>::infinity();
  for (int i = 0; i < iters; ++i) {
    hipblasStatus_t s = fn();
    if (s != HIPBLAS_STATUS_SUCCESS) {
      (void)hipEventRecord(stop.e, st->s);
      (void)hipEventSynchronize(stop.e);
      return std::numeric_limits<float>::infinity();
    }
  }
  if (!HipCheck(hipEventRecord(stop.e, st->s), "EventRecord stop"))
    return std::numeric_limits<float>::infinity();
  if (!HipCheck(hipEventSynchronize(stop.e), "EventSync stop"))
    return std::numeric_limits<float>::infinity();
  float elapsed = 0.0f;
  if (!HipCheck(hipEventElapsedTime(&elapsed, start.e, stop.e),
                "EventElapsed"))
    return std::numeric_limits<float>::infinity();
  return elapsed / std::max(1, iters);
}

}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (s == "--help" || s == "-h") {
      Usage(argv[0]);
      return kOk;
    }
    if (s == "--M") {
      cfg.M = std::atoi(Need("--M", &i, argc, argv));
    } else if (s == "--N") {
      cfg.N = std::atoi(Need("--N", &i, argc, argv));
    } else if (s == "--K") {
      cfg.K = std::atoi(Need("--K", &i, argc, argv));
    } else if (s == "--lda") {
      cfg.lda = std::atoi(Need("--lda", &i, argc, argv));
    } else if (s == "--ldb") {
      cfg.ldb = std::atoi(Need("--ldb", &i, argc, argv));
    } else if (s == "--ldc") {
      cfg.ldc = std::atoi(Need("--ldc", &i, argc, argv));
    } else if (s == "--ldd") {
      cfg.ldd = std::atoi(Need("--ldd", &i, argc, argv));
    } else if (s == "--ldc_diff") {
      cfg.ldc_diff = std::atoi(Need("--ldc_diff", &i, argc, argv));
    } else if (s == "--iters") {
      cfg.iters = std::atoi(Need("--iters", &i, argc, argv));
    } else if (s == "--alpha") {
      cfg.alpha = std::atof(Need("--alpha", &i, argc, argv));
    } else if (s == "--beta") {
      cfg.beta = std::atof(Need("--beta", &i, argc, argv));
    } else if (s == "--transA") {
      std::string t = Need("--transA", &i, argc, argv);
      cfg.transA = (t == "T" || t == "t");
    } else if (s == "--transB") {
      std::string t = Need("--transB", &i, argc, argv);
      cfg.transB = (t == "T" || t == "t");
    } else if (s == "--alpha0") {
      cfg.alpha_zero = true;
    } else if (s == "--memory_mode") {
      std::string m = Need("--memory_mode", &i, argc, argv);
      if (m == "device") {
        cfg.mem = MemMode::kDevice;
      } else if (m == "host_mapped_C") {
        cfg.mem = MemMode::kHostMappedC;
      } else if (m == "managed") {
        cfg.mem = MemMode::kManaged;
      } else {
        std::cerr << "Unknown memory_mode " << m << std::endl;
        return kCliErr;
      }
    } else if (s == "--pad_rows") {
      cfg.pad_rows = std::atoi(Need("--pad_rows", &i, argc, argv));
    } else if (s == "--no_validate") {
      cfg.validate = false;
    } else if (s == "--verbose") {
      cfg.verbose = true;
    } else if (s == "--run_all_algos") {
      cfg.run_all_algos = true;
    } else if (s == "--streams") {
      cfg.streams = std::atoi(Need("--streams", &i, argc, argv));
    } else if (s == "--nonzero_iters") {
      cfg.nonzero_iters = std::atoi(Need("--nonzero_iters", &i, argc, argv));
    } else {
      std::cerr << "Unknown arg " << s << std::endl;
      Usage(argv[0]);
      return kCliErr;
    }
  }

  if (cfg.lda == 0) cfg.lda = cfg.transA ? cfg.K : cfg.M;
  if (cfg.ldb == 0) cfg.ldb = cfg.transB ? cfg.N : cfg.K;
  if (cfg.ldc == 0) cfg.ldc = cfg.M;
  if (cfg.ldd == 0) cfg.ldd = cfg.M;
  if (cfg.ldc_diff) {
    cfg.ldc += cfg.ldc_diff;
    cfg.ldd += cfg.ldc_diff;
  }

  std::cout << "[Mini Residual App] M=" << cfg.M << " N=" << cfg.N
            << " K=" << cfg.K << " | iters=" << cfg.iters
            << " | alpha=" << (cfg.alpha_zero ? 0.0f : cfg.alpha)
            << " | beta=" << cfg.beta
            << " | mem="
            << (cfg.mem == MemMode::kDevice
                    ? "device"
                    : (cfg.mem == MemMode::kHostMappedC ? "host_mapped_C"
                                                       : "managed"))
            << "\n";

  std::cout << "lda=" << cfg.lda << " ldb=" << cfg.ldb << " ldc=" << cfg.ldc
            << " ldd=" << cfg.ldd << " | transA=" << (cfg.transA ? 'T' : 'N')
            << " transB=" << (cfg.transB ? 'T' : 'N')
            << " | pad_rows=" << cfg.pad_rows << " | streams=" << cfg.streams
            << " | mode=" << (cfg.run_all_algos ? "ALL_ALGOS" : "BEST_ONLY")
            << " | nonzero_iters=" << cfg.nonzero_iters << "\n";

  const int rowsA = cfg.transA ? cfg.K : cfg.M;
  const int colsA = cfg.transA ? cfg.M : cfg.K;
  const int rowsB = cfg.transB ? cfg.N : cfg.K;
  const int colsB = cfg.transB ? cfg.K : cfg.N;

  const std::size_t elemsA =
      static_cast<std::size_t>(cfg.lda) * static_cast<std::size_t>(colsA);
  const std::size_t elemsB =
      static_cast<std::size_t>(cfg.ldb) * static_cast<std::size_t>(colsB);
  const std::size_t elemsC =
      static_cast<std::size_t>(cfg.ldc) * static_cast<std::size_t>(cfg.N);
  const std::size_t elemsD =
      static_cast<std::size_t>(cfg.ldd) * static_cast<std::size_t>(cfg.N);

  hipDataType dtype_a;
  hipDataType dtype_b;
  int bytes_a;
  int bytes_b;
  if (A_RUNTIME_SELECT || B_RUNTIME_SELECT) {
    const bool fp8_ok = ProbeFp8Support(
        rowsA, colsA, cfg.lda, rowsB, colsB, cfg.ldb,
        cfg.M, cfg.N, cfg.ldc, cfg.ldd, cfg.transA, cfg.transB);
    if (fp8_ok) {
      dtype_a = HIP_R_8F_E4M3;
      dtype_b = HIP_R_8F_E5M2;
      bytes_a = 1;
      bytes_b = 1;
      std::cout << "[Mini Residual App] FP8 (E4M3/E5M2) supported "
                << "— using FP8.\n";
    } else {
      dtype_a = HIP_R_16BF;
      dtype_b = HIP_R_16BF;
      bytes_a = 2;
      bytes_b = 2;
      std::cout << "[Mini Residual App] FP8 not supported "
                << "— fallback to BF16.\n";
    }
  } else {
    dtype_a = A_HIP_DTYPE_COMPILE;
    dtype_b = B_HIP_DTYPE_COMPILE;
    bytes_a = A_BYTES_COMPILE;
    bytes_b = B_BYTES_COMPILE;
  }

  DeviceBuffer A(elemsA * static_cast<std::size_t>(bytes_a), MemMode::kDevice);
  DeviceBuffer B(elemsB * static_cast<std::size_t>(bytes_b), MemMode::kDevice);

  if (!QuantizeA(&A, elemsA, bytes_a == 1)) return kHipErr;
  if (!QuantizeB(&B, elemsB, bytes_b == 1)) return kHipErr;

  if (cfg.streams <= 0) cfg.streams = 1;
  std::vector<HipStream> streams;
  streams.reserve(static_cast<std::size_t>(cfg.streams));
  for (int si = 0; si < cfg.streams; ++si) {
    streams.emplace_back(HipStream(0U));
  }

  std::vector<DeviceBuffer> Cs;
  std::vector<DeviceBuffer> Ds;
  Cs.reserve(static_cast<std::size_t>(cfg.streams));
  Ds.reserve(static_cast<std::size_t>(cfg.streams));
  for (int si = 0; si < cfg.streams; ++si) {
    Cs.emplace_back(
        DeviceBuffer(elemsC * sizeof(std::uint16_t), cfg.mem));
    Ds.emplace_back(
        DeviceBuffer(elemsD * sizeof(std::uint16_t), MemMode::kDevice));
    (void)Cs.back().Zero();
    (void)Ds.back().Zero();
  }

  LtHandle h;
  const hipblasOperation_t opA = cfg.transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  const hipblasOperation_t opB = cfg.transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

  LtMatrixLayout Ad(dtype_a, rowsA, colsA, cfg.lda);
  LtMatrixLayout Bd(dtype_b, rowsB, colsB, cfg.ldb);
  LtMatrixLayout Cd(HIP_R_16BF, cfg.M, cfg.N, cfg.ldc);
  LtMatrixLayout Dd(HIP_R_16BF, cfg.M, cfg.N, cfg.ldd);

  LtMatmulDesc desc(HIPBLAS_COMPUTE_32F, HIP_R_32F);
  if (!desc.SetAttr(HIPBLASLT_MATMUL_DESC_TRANSA, opA)) return kLtErr;
  if (!desc.SetAttr(HIPBLASLT_MATMUL_DESC_TRANSB, opB)) return kLtErr;

  LtPreference pref;
  const std::size_t ws_bytes = (std::size_t{64} << 20);
  if (!pref.SetAttr(HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, ws_bytes))
    return kLtErr;

  constexpr int kMaxCands = 32;
  std::vector<hipblasLtMatmulHeuristicResult_t> cands(
      static_cast<std::size_t>(kMaxCands));
  int n_found = 0;
  if (!LtCheck(hipblasLtMatmulAlgoGetHeuristic(
                   h.h, desc.d, Ad.l, Bd.l, Cd.l, Dd.l,
                   pref.p, kMaxCands, cands.data(), &n_found),
               "MatmulAlgoGetHeuristic"))
    return kLtErr;

  if (n_found == 0) {
    std::cerr << "No heuristic algorithm found (adjust workspace or shapes)\n";
    return kNoHeuristics;
  }

  DeviceBuffer dws(ws_bytes, MemMode::kDevice);
  DeviceBuffer dflag(sizeof(int), MemMode::kDevice);

  const float alpha = cfg.alpha_zero ? 0.0f : cfg.alpha;
  const float beta = cfg.beta;

  std::cout << "\n[hipBLASLt] Found " << n_found
            << " heuristic candidate(s). Validating & benchmarking each...\n";

  float best_ms = std::numeric_limits<float>::infinity();
  int best_idx = -1;
  const int bench_iters = std::min(10, std::max(2, cfg.iters >= 20 ? 10 : 5));
  std::vector<int> ok_indices;

  auto run_one = [&](const hipblasLtMatmulAlgo_t& algo, DeviceBuffer& Cbuf,
                     DeviceBuffer& Dbuf, HipStream& st) -> hipblasStatus_t {
    if (!ZeroABandOrAll(A.dev, cfg, bytes_a))
      return HIPBLAS_STATUS_INTERNAL_ERROR;
    (void)Cbuf.Zero();
    (void)Dbuf.Poison();
    return hipblasLtMatmul(h.h, desc.d, &alpha, A.dev, Ad.l, B.dev, Bd.l,
                           &beta, Cbuf.dev, Cd.l, Dbuf.dev, Dd.l, &algo,
                           dws.dev, ws_bytes, st.s);
  };

  for (int i = 0; i < n_found; ++i) {
    const auto& algo = cands[static_cast<std::size_t>(i)].algo;

    // Warmup once on stream 0
    {
      hipblasStatus_t st = run_one(algo, Cs[0], Ds[0], streams[0]);
      if (!HipCheck(hipStreamSynchronize(streams[0].s), "warmup sync"))
        return kHipErr;
      if (st != HIPBLAS_STATUS_SUCCESS) {
        std::cout << "  - Algo[" << i << "]: warmup FAILED (status="
                  << static_cast<int>(st) << ")\n";
        continue;
      }
    }

    // Validate β-path + non-finites (expect zero)
    const bool ok =
        ValidateZeroAndNonFinite(cfg, &Ds[0], &dflag, /*expect_zero=*/true);
    if (!ok) {
      std::cout << "  - Algo[" << i
                << "]: VALIDATION FAILED (β-path / non-finite)\n";
      continue;
    }
    ok_indices.push_back(i);

    // Benchmark average ms on stream 0
    const float avg_ms = TimeAvgMs(bench_iters, &streams[0], [&]() {
      return run_one(algo, Cs[0], Ds[0], streams[0]);
    });

    if (!std::isfinite(avg_ms)) {
      std::cout << "  - Algo[" << i << "]: BENCH FAILED\n";
      continue;
    }

    std::cout << "  - Algo[" << i << "]: PASS | avg=" << avg_ms << " ms over "
              << bench_iters << " iters\n";
    if (avg_ms < best_ms) {
      best_ms = avg_ms;
      best_idx = i;
    }
  }

  if (ok_indices.empty()) {
    std::cerr << "[FAIL] No candidate algorithm passed validation.\n";
    return kNoPassing;
  }

  std::cout << "\n[hipBLASLt] Selected best algorithm: Algo[" << best_idx
            << "] (avg=" << best_ms << " ms)\n";
  std::cout << "Training mode: "
            << (cfg.run_all_algos ? "ALL_ALGOS" : "BEST_ONLY") << " ("
            << ok_indices.size() << " passing), streams=" << cfg.streams
            << ", nonzero_iters=" << cfg.nonzero_iters << "\n";

  // Prepare non-zero phase templates
  if (cfg.nonzero_iters > 0) {
    if (!QuantizeA(&A, elemsA, bytes_a == 1)) return kHipErr;
  }
  std::vector<DeviceBuffer> C_templates;
  if (cfg.nonzero_iters > 0) {
    C_templates.reserve(static_cast<std::size_t>(cfg.streams));
    for (int si = 0; si < cfg.streams; ++si) {
      C_templates.emplace_back(DeviceBuffer(
          elemsC * sizeof(std::uint16_t), MemMode::kDevice));
      if (!FillCRandomBf16(&C_templates.back(), elemsC)) return kHipErr;
    }
  }

  // Training loop (two-phase)
  for (int it = 0; it < cfg.iters; ++it) {
    const bool zero_phase = (it >= cfg.nonzero_iters);

    if (zero_phase) {
      if (!ZeroABandOrAll(A.dev, cfg, bytes_a)) return kHipErr;
    }

    std::vector<int> run_list;
    if (cfg.run_all_algos) {
      run_list = ok_indices;
    } else {
      run_list = {best_idx};
    }

    struct Task {
      int algo_idx;
      int stream_idx;
    };
    std::vector<Task> tasks;
    tasks.reserve(run_list.size());

    for (std::size_t t = 0; t < run_list.size(); ++t) {
      const int algo_idx = run_list[t];
      const int si =
          static_cast<int>(t % static_cast<std::size_t>(cfg.streams));

      if (zero_phase) {
        (void)Cs[si].Zero();
      } else if (cfg.nonzero_iters > 0 &&
                 static_cast<std::size_t>(si) < C_templates.size()) {
        if (!HipCheck(hipMemcpy(Cs[si].dev, C_templates[si].dev,
                                elemsC * sizeof(std::uint16_t),
                                hipMemcpyDeviceToDevice),
                      "Memcpy C template")) {
          return kHipErr;
        }
      }

      (void)Ds[si].Poison();

      const auto& algo = cands[static_cast<std::size_t>(algo_idx)].algo;
      hipblasStatus_t st = hipblasLtMatmul(
          h.h, desc.d, &alpha, A.dev, Ad.l, B.dev, Bd.l, &beta, Cs[si].dev,
          Cd.l, Ds[si].dev, Dd.l, &algo, dws.dev, ws_bytes, streams[si].s);
      if (st != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "[FAIL] hipblasLtMatmul enqueue failed at iter=" << it
                  << " algo=" << algo_idx
                  << " (status=" << static_cast<int>(st) << ")\n";
        std::cout << "Replay: hipblaslt-bench --api_method c"
                  << " -m " << cfg.M << " -n " << cfg.N << " -k " << cfg.K
                  << " --lda " << cfg.lda << " --ldb " << cfg.ldb << " --ldc "
                  << cfg.ldc << " --ldd " << cfg.ldd << " --alpha "
                  << (cfg.alpha_zero ? 0.0f : cfg.alpha) << " --beta "
                  << cfg.beta << " --transA " << (cfg.transA ? 'T' : 'N')
                  << " --transB " << (cfg.transB ? 'T' : 'N')
                  << " --iters 1 -v 1\n";
        return kTrainFail;
      }
      tasks.push_back(Task{algo_idx, si});
    }

    for (const auto& task : tasks) {
      if (!HipCheck(hipStreamSynchronize(streams[task.stream_idx].s),
                    "GEMM stream sync")) {
        return kHipErr;
      }
      if (cfg.validate) {
        const bool ok = ValidateZeroAndNonFinite(
            cfg, &Ds[task.stream_idx], &dflag, /*expect_zero=*/zero_phase);
        if (!ok) {
          std::cerr << "[FAIL] Validation failed at iter=" << it
                    << " algo=" << task.algo_idx
                    << " (stream=" << task.stream_idx
                    << ", zero_phase=" << static_cast<int>(zero_phase)
                    << ")\n";
          return kTrainFail;
        }
      }
    }

    if (cfg.verbose && ((it + 1) % 50 == 0)) {
      std::cout << "[iter " << (it + 1) << "] OK ("
                << (cfg.run_all_algos ? "all algos" : "best")
                << ", streams=" << cfg.streams
                << ", phase=" << (zero_phase ? "ZERO" : "NONZERO") << ")\n";
    }
  }

  std::cout << "[PASS] " << cfg.iters
            << " iterations: non-zero phase (first " << cfg.nonzero_iters
            << ") then zero phase — all checks held.\n";
  return kOk;
}

