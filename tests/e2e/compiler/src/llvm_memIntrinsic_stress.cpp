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
// SPDX-License-Identifier: MIT

// Standalone HIP harness: multi-stream large-buffer stress for LLVM lowering of
// __builtin_memset, __builtin_memcpy, and __builtin_memmove vs host std::mem*
// replay. Optional multi-GPU: stream index s uses HIP device (s % num_devices).
//
// Structure (see namespace mem_intrinsic_stress):
//   StressOp / StressOpKind     — one logical memory operation in the test plan
//   HostReplayEngine            — applies ops on host with std::mem* (golden buffer)
//   StressPlanGenerator         — builds pseudo-random reproducible op sequences (lengths scale with buffer size)
//   StreamResources             — per-stream HIP stream + device pointer + device id (explicit release())
//   LargeBufferStressRunner     — orchestrates setup, kernel enqueue, sync, D2H, compare
//   HarnessCliOptions + parsing — command-line configuration

#include <hip/hip_runtime.h>

#include "utility.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Buffer size limits (must stay consistent with allocator and host vectors).
// ---------------------------------------------------------------------------

/// Maximum single per-stream buffer (matches llvm_Lowering_mem harness cap).
constexpr size_t kMaxBytesPerStream = size_t{1} << 36;

constexpr unsigned kMinBytesPerStream = 64;

/// Default per-stream size unless overridden by --buffer-size-*.
constexpr size_t kDefaultBytesPerStream = size_t{1} << 35; // 32 GiB

// ---------------------------------------------------------------------------
// Multi-thread memset/memcpy partitioning (--multi-thread-enable).
// HIP caps threads per block (typical max 1024).
// ---------------------------------------------------------------------------

constexpr unsigned kDefaultThreadsPerBlock = 256;
constexpr unsigned kMinThreadsPerBlock = 2;
constexpr unsigned kMaxThreadsPerBlock = 1024;

/// Mixed into the RNG seed per stream so each stream’s plan differs but stays reproducible.
constexpr uint64_t kStreamIndexSeedMix = 1315423911ULL;

/// Minimum upper bound on random memset length when the buffer is large enough (legacy tiny-op floor).
constexpr size_t kStressMemsetLenFloor = 64;

// ---------------------------------------------------------------------------
// Human-readable byte strings for logs and --help.
// ---------------------------------------------------------------------------

/// Formats exact byte count plus IEC suffix (KiB / MiB / GiB) for reviewers.
static std::string formatBytesHumanReadable(size_t byte_count) {
  std::ostringstream os;
  os << byte_count << " B";
  if (byte_count >= (1ull << 30)) {
    const double gib = static_cast<double>(byte_count) / static_cast<double>(1ull << 30);
    os << " (" << std::fixed << std::setprecision(gib >= 100.0 ? 1 : (gib >= 10.0 ? 2 : 3)) << gib << " GiB)";
  } else if (byte_count >= (1ull << 20)) {
    const double mib = static_cast<double>(byte_count) / static_cast<double>(1ull << 20);
    os << " (" << std::fixed << std::setprecision(3) << mib << " MiB)";
  } else if (byte_count >= 1024u) {
    const double kib = static_cast<double>(byte_count) / 1024.0;
    os << " (" << std::fixed << std::setprecision(3) << kib << " KiB)";
  }
  return os.str();
}

/// Single-line key=value fields for harness logs (config line and [PASS]).
static void appendStressHarnessFields(std::ostream &os, size_t buf_bytes, unsigned streams, unsigned devices,
                                      unsigned kernels, uint64_t seed, bool interleave, bool mt,
                                      unsigned threads_per_block) {
  os << " buf_B=" << buf_bytes << " streams=" << streams << " devices=" << devices << " kernels=" << kernels
     << " seed=" << seed << " interleave=" << (interleave ? "on" : "off") << " mt=" << (mt ? "on" : "off");
  if (mt)
    os << " tb=" << threads_per_block;
}

// ---------------------------------------------------------------------------
// Launch grid for partitioned memset/memcpy: enough blocks so that
// blockDim.x * gridDim.x threads cover the byte range (each thread one builtin slice).
// ---------------------------------------------------------------------------

static void computePartitionedMemopLaunchDimensions(size_t region_length_bytes, unsigned threads_per_block_x,
                                                    dim3 *grid_dimensions, dim3 *block_dimensions) {
  if (region_length_bytes == 0) {
    *grid_dimensions = dim3(1u);
    *block_dimensions = dim3(1u);
    return;
  }
  unsigned threads_per_block =
      std::max(1u, std::min(threads_per_block_x, kMaxThreadsPerBlock));
  const size_t block_count =
      (region_length_bytes + static_cast<size_t>(threads_per_block) - 1) / static_cast<size_t>(threads_per_block);
  unsigned grid_x = static_cast<unsigned>(block_count);
  if (grid_x < 1u)
    grid_x = 1u;
  *block_dimensions = dim3(threads_per_block);
  *grid_dimensions = dim3(grid_x);
}

} // namespace

// =============================================================================
// Device kernels (__global__ must remain at global scope for HIP).
// One hipLaunchKernelGGL per StressOp. Default path: single CTA, single thread.
// Optional: partitioned memset/memcpy — each thread applies the builtin to a
// disjoint sub-range; combined effect matches one logical memset/memcpy.
// memmove: always single-thread (overlap semantics).
// =============================================================================

__global__ void k_stress_memset(uint8_t *base, size_t offset, size_t len, uint8_t value) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  __builtin_memset(base + offset, value, len);
}

__global__ void k_stress_memcpy_non_overlap(uint8_t *base, size_t dst_off, size_t src_off, size_t len) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  __builtin_memcpy(base + dst_off, base + src_off, len);
}

// Multi-thread memset: partition [0, len) across all threads in the launch grid.
// Each thread calls __builtin_memset only on its disjoint sub-range [region_start, region_end).
// Together these sub-ranges cover [0, len) exactly once, matching one logical memset on
// [offset, offset+len). Threads with no assigned bytes exit early (region_start >= len).
__global__ void k_stress_memset_mt(uint8_t *base, size_t offset, size_t len, uint8_t value) {
  const size_t total_thread_count = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t linear_thread_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (len == 0)
    return;
  // ceil(len / total_thread_count): upper bound on bytes per thread so every byte is covered.
  const size_t chunk_byte_length = (len + total_thread_count - 1) / total_thread_count;
  const size_t region_start = linear_thread_index * chunk_byte_length;
  if (region_start >= len)
    return;
  size_t region_end = region_start + chunk_byte_length;
  if (region_end > len)
    region_end = len;
  __builtin_memset(base + offset + region_start, value, region_end - region_start);
}

// Multi-thread non-overlap memcpy: same partition as k_stress_memset_mt, but copies
// [region_start, region_end) from src_off+… to dst_off+… (non-overlapping ranges in the plan).
__global__ void k_stress_memcpy_non_overlap_mt(uint8_t *base, size_t dst_off, size_t src_off, size_t len) {
  const size_t total_thread_count = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t linear_thread_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (len == 0)
    return;
  const size_t chunk_byte_length = (len + total_thread_count - 1) / total_thread_count;
  const size_t region_start = linear_thread_index * chunk_byte_length;
  if (region_start >= len)
    return;
  size_t region_end = region_start + chunk_byte_length;
  if (region_end > len)
    region_end = len;
  __builtin_memcpy(base + dst_off + region_start, base + src_off + region_start, region_end - region_start);
}

// Single-thread memmove: overlap-safe semantics require one logical memmove over the full range.
// We do not tile memmove across threads (would break ordering for overlapping src/dst).
__global__ void k_stress_memmove(uint8_t *base, size_t dst_off, size_t src_off, size_t len) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;
  __builtin_memmove(base + dst_off, base + src_off, len);
}

namespace mem_intrinsic_stress {

// -----------------------------------------------------------------------------
// StressOp — one logical memory operation in the random plan (host + device).
// -----------------------------------------------------------------------------

enum class StressOpKind {
  Memset,           ///< Fill a destination range with a byte value.
  MemcpyNonOverlap, ///< Non-overlapping src/dst (memcpy semantics).
  MemmoveOverlap    ///< Possibly overlapping ranges (memmove semantics).
};

struct StressOp {
  StressOpKind kind = StressOpKind::Memset;
  size_t dst_off = 0;
  size_t src_off = 0;
  size_t len = 0;
  uint8_t value = 0;
};

// -----------------------------------------------------------------------------
// HostReplayEngine — host validates: replays the same StressOp list with std::mem*
// to build the golden buffer for byte-for-byte comparison after D2H.
// -----------------------------------------------------------------------------

class HostReplayEngine {
public:
  /// Applies a single op to `buffer`; returns false if offsets/lengths are invalid.
  static bool applyOne(std::vector<uint8_t> &buffer, const StressOp &op) {
    const size_t buffer_size = buffer.size();
    switch (op.kind) {
    case StressOpKind::Memset: {
      if (op.dst_off + op.len > buffer_size)
        return false;
      std::memset(buffer.data() + op.dst_off, static_cast<int>(op.value), op.len);
      return true;
    }
    case StressOpKind::MemcpyNonOverlap: {
      if (op.dst_off + op.len > buffer_size || op.src_off + op.len > buffer_size)
        return false;
      if (!rangesAreNonOverlapping(op.dst_off, op.src_off, op.len))
        return false;
      std::memcpy(buffer.data() + op.dst_off, buffer.data() + op.src_off, op.len);
      return true;
    }
    case StressOpKind::MemmoveOverlap: {
      if (op.dst_off + op.len > buffer_size || op.src_off + op.len > buffer_size)
        return false;
      std::memmove(buffer.data() + op.dst_off, buffer.data() + op.src_off, op.len);
      return true;
    }
    }
    return false;
  }

  /// Replays `operations` in order on `buffer` (must be sized to the stream buffer).
  static bool replayAll(std::vector<uint8_t> &buffer, const std::vector<StressOp> &operations) {
    for (const StressOp &op : operations) {
      if (!applyOne(buffer, op))
        return false;
    }
    return true;
  }

private:
  static bool rangesAreNonOverlapping(size_t dst_off, size_t src_off, size_t len) {
    return (dst_off + len <= src_off) || (src_off + len <= dst_off);
  }
};

// -----------------------------------------------------------------------------
// StressPlanGenerator — deterministic pseudo-random op sequences per stream.
// -----------------------------------------------------------------------------

class StressPlanGenerator {
public:
  explicit StressPlanGenerator(size_t buffer_byte_size, uint64_t random_seed, unsigned kernels_per_stream)
      : buffer_byte_size_(buffer_byte_size), random_engine_(random_seed),
        kernels_per_stream_(std::max(1u, kernels_per_stream)) {}

  /// Builds `count` operations; same seed + buffer size + kernels/stream ⇒ same plan.
  std::vector<StressOp> generate(unsigned operation_count) {
    std::vector<StressOp> operations;
    operations.reserve(operation_count);
    for (unsigned i = 0; i < operation_count; ++i)
      operations.push_back(generateNextOperation());
    return operations;
  }

private:
  // Picks one of three op kinds with equal probability: Memset, MemcpyNonOverlap, or MemmoveOverlap.
  StressOp generateNextOperation() {
    std::uniform_int_distribution<int> kind_distribution(0, 2);
    const int kind = kind_distribution(random_engine_);
    if (kind == 0)
      return generateRandomMemset();
    if (kind == 1)
      return generateRandomMemcpyNonOverlap();
    return generateRandomMemmove();
  }

  // Random contiguous fill: length scales with buffer / kernels_per_stream (not fixed 64 B on huge buffers).
  StressOp generateRandomMemset() {
    const size_t max_len = maxLenMemset();
    std::uniform_int_distribution<size_t> len_distribution(1, max_len);
    size_t length = std::max<size_t>(1, len_distribution(random_engine_));
    if (length > buffer_byte_size_)
      length = buffer_byte_size_;
    std::uniform_int_distribution<size_t> offset_distribution(0, buffer_byte_size_ - length);
    const size_t offset = offset_distribution(random_engine_);
    std::uniform_int_distribution<int> value_distribution(0, 255);
    StressOp op;
    op.kind = StressOpKind::Memset;
    op.dst_off = offset;
    op.len = length;
    op.value = static_cast<uint8_t>(value_distribution(random_engine_));
    return op;
  }

  // Random memcpy with disjoint src and dst ranges (memcpy-safe). Length scales with buffer / op count.
  // Retries random dst_off/src_off up to 64 times; if no non-overlapping pair is found, falls back to a fixed memset.
  StressOp generateRandomMemcpyNonOverlap() {
    const size_t max_len = maxLenMemcpyNonOverlap();
    size_t length = std::max<size_t>(1, std::uniform_int_distribution<size_t>(1, max_len)(random_engine_));
    StressOp op;
    op.kind = StressOpKind::MemcpyNonOverlap;
    op.len = length;
    for (int attempt = 0; attempt < 64; ++attempt) {
      std::uniform_int_distribution<size_t> offset_distribution(0, buffer_byte_size_ - length);
      op.dst_off = offset_distribution(random_engine_);
      op.src_off = offset_distribution(random_engine_);
      // Check whether the two ranges don’t overlap.
      // The destination bytes and source bytes don't overlap — safe for memcpy.
      if ((op.dst_off + length <= op.src_off) || (op.src_off + length <= op.dst_off))
        return op;
    }
    // Never found non-overlapping src/dst in 64 tries — can't emit memcpy. Use a simple memset instead.
    op.kind = StressOpKind::Memset;
    op.dst_off = 0;
    op.src_off = 0;
    op.len = std::min(length, buffer_byte_size_);
    op.value = 0xA5;
    return op;
  }

  // Random memmove: length at least 2, scales with buffer / op count; src/dst independent (overlap allowed).
  StressOp generateRandomMemmove() {
    const size_t max_len = maxLenMemmove();
    std::uniform_int_distribution<size_t> len_distribution(2, max_len);
    size_t length = std::max<size_t>(2, len_distribution(random_engine_));
    if (length > buffer_byte_size_)
      length = buffer_byte_size_;
    std::uniform_int_distribution<size_t> offset_distribution(0, buffer_byte_size_ - length);
    const size_t src = offset_distribution(random_engine_);
    const size_t dst = offset_distribution(random_engine_);
    StressOp op;
    op.kind = StressOpKind::MemmoveOverlap;
    op.src_off = src;
    op.dst_off = dst;
    op.len = length;
    return op;
  }

  /// Per-op share of the buffer so N ops can cover the full range in the worst case (proportional stress).
  size_t bufferSharePerOp() const {
    return std::max(size_t{1}, buffer_byte_size_ / static_cast<size_t>(kernels_per_stream_));
  }

  /// Memset: up to max(floor, share), capped by buffer (avoids only ~64 B ops on multi-GiB allocations).
  size_t maxLenMemset() const {
    const size_t share = bufferSharePerOp();
    const size_t cap = std::max(kStressMemsetLenFloor, share);
    return std::max(size_t{1}, std::min(buffer_byte_size_, cap));
  }

  /// Memcpy (non-overlap): bounded by buffer/4 (easier to place) and by per-op share.
  size_t maxLenMemcpyNonOverlap() const {
    const size_t share = bufferSharePerOp();
    const size_t quarter = buffer_byte_size_ / 4;
    const size_t cap = std::min({quarter, share, buffer_byte_size_});
    return std::max(size_t{1}, cap);
  }

  /// Memmove: at least 2 bytes; upper bound scales with buffer/2 and per-op share.
  size_t maxLenMemmove() const {
    const size_t share = bufferSharePerOp();
    const size_t half = buffer_byte_size_ / 2;
    const size_t cap = std::min({half, share, buffer_byte_size_});
    return std::max(size_t{2}, cap);
  }

  size_t buffer_byte_size_;
  std::mt19937_64 random_engine_;
  unsigned kernels_per_stream_;
};

// -----------------------------------------------------------------------------
// StreamResources — per-stream HIP stream, device pointer, owning device id. Call release() when done.
// -----------------------------------------------------------------------------

struct StreamResources {
  std::vector<hipStream_t> streams;
  std::vector<uint8_t *> device_buffers;
  std::vector<int> hip_device_ids;

  explicit StreamResources(unsigned stream_count)
      : streams(stream_count, nullptr), device_buffers(stream_count, nullptr), hip_device_ids(stream_count, 0) {}

  ~StreamResources() = default;

  /// Frees device allocations and destroys streams (safe to call multiple times).
  void release() noexcept {
    for (size_t stream_index = 0; stream_index < device_buffers.size(); ++stream_index) {
      if (device_buffers[stream_index]) {
        HIP_CHECK(hipSetDevice(hip_device_ids[stream_index]));
        HIP_CHECK(hipFree(device_buffers[stream_index]));
        device_buffers[stream_index] = nullptr;
      }
    }
    for (size_t stream_index = 0; stream_index < streams.size(); ++stream_index) {
      if (streams[stream_index]) {
        HIP_CHECK(hipSetDevice(hip_device_ids[stream_index]));
        HIP_CHECK(hipStreamDestroy(streams[stream_index]));
        streams[stream_index] = nullptr;
      }
    }
  }

  StreamResources(const StreamResources &) = delete;
  StreamResources &operator=(const StreamResources &) = delete;
};

// -----------------------------------------------------------------------------
// LargeBufferStressRunner — main test driver: streams, gold, kernels, sync, compare.
// -----------------------------------------------------------------------------

class LargeBufferStressRunner {
public:
  struct Config {
    unsigned num_streams = 1;
    unsigned num_hip_devices = 1;
    unsigned kernels_per_stream = 32;
    uint64_t seed = 0xC0FFEEULL;
    size_t bytes_per_buffer = kDefaultBytesPerStream;
    bool interleave_stream_launches = false;
    bool multi_thread_memops = false;
    unsigned threads_per_kernel = kDefaultThreadsPerBlock;
  };

  explicit LargeBufferStressRunner(Config configuration) : config_(std::move(configuration)) {}

  /// Runs the full stress: returns true on PASS.
  [[nodiscard]] bool run(std::ostream &log) {
    if (config_.num_streams == 0 || config_.kernels_per_stream == 0) {
      log << "[SKIP] streams=" << config_.num_streams << " kernels=" << config_.kernels_per_stream << "\n";
      return true;
    }
    if (config_.num_hip_devices == 0) {
      std::cerr << "Large-buffer stress: --devices must be at least 1.\n";
      return false;
    }

    int hip_device_total = 0;
    HIP_CHECK(hipGetDeviceCount(&hip_device_total));
    if (hip_device_total < 1) {
      std::cerr << "Large-buffer stress: no HIP devices.\n";
      return false;
    }
    if (static_cast<int>(config_.num_hip_devices) > hip_device_total) {
      std::cerr << "Large-buffer stress: --devices=" << config_.num_hip_devices
                << " exceeds hipGetDeviceCount()=" << hip_device_total << ".\n";
      return false;
    }

    StreamResources resources(config_.num_streams);

    // --- Phase 1: create one HIP stream per logical stream, assigned round-robin to devices. ---
    for (unsigned stream_index = 0; stream_index < config_.num_streams; ++stream_index) {
      const int device_id = static_cast<int>(stream_index % config_.num_hip_devices);
      resources.hip_device_ids[stream_index] = device_id;
      HIP_CHECK(hipSetDevice(device_id));
      HIP_CHECK(hipStreamCreate(&resources.streams[stream_index]));
    }

    std::vector<std::vector<uint8_t>> host_golden_buffers(config_.num_streams);
    std::vector<std::vector<StressOp>> per_stream_plans(config_.num_streams);

    // --- Phase 2: allocate device memory, build random plans, compute host gold. ---
    for (unsigned stream_index = 0; stream_index < config_.num_streams; ++stream_index) {
      const int device_id = resources.hip_device_ids[stream_index];
      HIP_CHECK(hipSetDevice(device_id));
      HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&resources.device_buffers[stream_index]), config_.bytes_per_buffer));
      HIP_CHECK(hipMemset(resources.device_buffers[stream_index], 0, config_.bytes_per_buffer));

      host_golden_buffers[stream_index].assign(config_.bytes_per_buffer, 0);

      StressPlanGenerator plan_generator(config_.bytes_per_buffer, config_.seed + stream_index * kStreamIndexSeedMix,
                                         config_.kernels_per_stream);
      per_stream_plans[stream_index] = plan_generator.generate(config_.kernels_per_stream);
      if (!HostReplayEngine::replayAll(host_golden_buffers[stream_index], per_stream_plans[stream_index])) {
        log << "[FAIL] host_replay stream=" << stream_index << " hip_dev=" << resources.hip_device_ids[stream_index]
            << "\n";
        resources.release();
        return false;
      }
    }

    // --- Phase 3: enqueue device kernels (interleaved or per-stream batch order). ---
    if (!enqueueAllKernelLaunches(resources, per_stream_plans)) {
      resources.release();
      return false;
    }

    // --- Phase 4: wait for all streams before readback. ---
    for (unsigned stream_index = 0; stream_index < config_.num_streams; ++stream_index) {
      HIP_CHECK(hipSetDevice(resources.hip_device_ids[stream_index]));
      HIP_CHECK(hipStreamSynchronize(resources.streams[stream_index]));
    }

    // --- Phase 5: D2H each stream and compare to host gold. ---
    //  host_readback : is the buffer read from Device to Host that will be compared against the host_golden_buffers.
    //  host_golden_buffers : is the buffer on the host that was generated by the StressPlanGenerator.
    for (unsigned stream_index = 0; stream_index < config_.num_streams; ++stream_index) {
      std::vector<uint8_t> host_readback(config_.bytes_per_buffer);
      HIP_CHECK(hipSetDevice(resources.hip_device_ids[stream_index]));
      HIP_CHECK(hipMemcpy(host_readback.data(), resources.device_buffers[stream_index], config_.bytes_per_buffer,
                          hipMemcpyDeviceToHost));
      const std::vector<uint8_t> &golden = host_golden_buffers[stream_index];
      if (std::memcmp(host_readback.data(), golden.data(), config_.bytes_per_buffer) != 0) {
        const auto mismatch_pair = std::mismatch(host_readback.begin(), host_readback.end(), golden.begin());
        const size_t byte_index =
            static_cast<size_t>(mismatch_pair.first - host_readback.begin());
        log << "[FAIL] compare stream=" << stream_index << " hip_dev=" << resources.hip_device_ids[stream_index]
            << " byte=" << byte_index << " expected=0x" << std::hex
            << static_cast<unsigned>(golden[byte_index]) << " got=0x"
            << static_cast<unsigned>(host_readback[byte_index]) << std::dec << "\n";
        resources.release();
        return false;
      }
    }

    resources.release();

    log << "[PASS]";
    appendStressHarnessFields(log, config_.bytes_per_buffer, config_.num_streams, config_.num_hip_devices,
                              config_.kernels_per_stream, config_.seed, config_.interleave_stream_launches,
                              config_.multi_thread_memops, config_.threads_per_kernel);
    log << "\n";
    return true;
  }

private:
  Config config_;

  /// Dispatches one HIP kernel for `op` on stream `stream_index`.
  bool dispatchOneOperation(StreamResources &resources, unsigned stream_index, const StressOp &op) {
    HIP_CHECK(hipSetDevice(resources.hip_device_ids[stream_index]));

    static const dim3 kSingleThreadGrid(1);
    static const dim3 kSingleThreadBlock(1);

    switch (op.kind) {
    case StressOpKind::Memset:
      if (config_.multi_thread_memops) {
        dim3 grid_dimensions;
        dim3 block_dimensions;
        ::computePartitionedMemopLaunchDimensions(op.len, config_.threads_per_kernel, &grid_dimensions,
                                                  &block_dimensions);
        hipLaunchKernelGGL(k_stress_memset_mt, grid_dimensions, block_dimensions, 0, resources.streams[stream_index],
                           resources.device_buffers[stream_index], op.dst_off, op.len, op.value);
      } else {
        hipLaunchKernelGGL(k_stress_memset, kSingleThreadGrid, kSingleThreadBlock, 0, resources.streams[stream_index],
                           resources.device_buffers[stream_index], op.dst_off, op.len, op.value);
      }
      break;
    case StressOpKind::MemcpyNonOverlap:
      if (config_.multi_thread_memops) {
        dim3 grid_dimensions;
        dim3 block_dimensions;
        ::computePartitionedMemopLaunchDimensions(op.len, config_.threads_per_kernel, &grid_dimensions,
                                                  &block_dimensions);
        hipLaunchKernelGGL(k_stress_memcpy_non_overlap_mt, grid_dimensions, block_dimensions, 0,
                           resources.streams[stream_index], resources.device_buffers[stream_index], op.dst_off,
                           op.src_off, op.len);
      } else {
        hipLaunchKernelGGL(k_stress_memcpy_non_overlap, kSingleThreadGrid, kSingleThreadBlock, 0,
                           resources.streams[stream_index], resources.device_buffers[stream_index], op.dst_off,
                           op.src_off, op.len);
      }
      break;
    case StressOpKind::MemmoveOverlap:
      hipLaunchKernelGGL(k_stress_memmove, kSingleThreadGrid, kSingleThreadBlock, 0, resources.streams[stream_index],
                         resources.device_buffers[stream_index], op.dst_off, op.src_off, op.len);
      break;
    }

    HIP_CHECK(hipGetLastError());
    return true;
  }

  /// Enqueues all kernels: either round-robin by op index across streams or full streams in order.
  bool enqueueAllKernelLaunches(StreamResources &resources,
                                const std::vector<std::vector<StressOp>> &per_stream_plans) {
    if (config_.interleave_stream_launches) {
      for (unsigned op_index = 0; op_index < config_.kernels_per_stream; ++op_index) {
        for (unsigned stream_index = 0; stream_index < config_.num_streams; ++stream_index) {
          if (!dispatchOneOperation(resources, stream_index, per_stream_plans[stream_index][op_index]))
            return false;
        }
      }
    } else {
      for (unsigned stream_index = 0; stream_index < config_.num_streams; ++stream_index) {
        for (const StressOp &op : per_stream_plans[stream_index]) {
          if (!dispatchOneOperation(resources, stream_index, op))
            return false;
        }
      }
    }
    return true;
  }
};

// -----------------------------------------------------------------------------
// Command line
// -----------------------------------------------------------------------------

struct HarnessCliOptions {
  unsigned streams = 1;
  unsigned num_hip_devices = 1;
  unsigned kernels_per_stream = 32;
  uint64_t seed = 0xC0FFEEULL;
  size_t buffer_bytes = kDefaultBytesPerStream;
  bool interleave_stream_launches = false;
  bool multi_thread_enable = false;
  unsigned threads_per_kernel = kDefaultThreadsPerBlock;
};

static HarnessCliOptions parseHarnessCommandLine(int argc, char **argv) {
  HarnessCliOptions options;
  for (int arg_index = 1; arg_index < argc; ++arg_index) {
    std::string argument = argv[arg_index];
    if (argument.rfind("--streams=", 0) == 0) {
      options.streams = static_cast<unsigned>(std::stoul(argument.substr(10)));
    } else if (argument.rfind("--devices=", 0) == 0) {
      const size_t equals_position = argument.find('=');
      options.num_hip_devices = static_cast<unsigned>(std::stoul(argument.substr(equals_position + 1)));
    } else if (argument.rfind("--kernels-per-stream=", 0) == 0) {
      options.kernels_per_stream = static_cast<unsigned>(std::stoul(argument.substr(21)));
    } else if (argument.rfind("--seed=", 0) == 0) {
      options.seed = static_cast<uint64_t>(std::stoull(argument.substr(7)));
    } else if (argument.rfind("--buffer-size-bytes=", 0) == 0) {
      const size_t equals_position = argument.find('=');
      options.buffer_bytes = static_cast<size_t>(std::stoull(argument.substr(equals_position + 1)));
    } else if (argument.rfind("--buffer-size-2pow=", 0) == 0) {
      const size_t equals_position = argument.find('=');
      const unsigned exponent = static_cast<unsigned>(std::stoul(argument.substr(equals_position + 1)));
      if (exponent > 36) {
        std::cerr << "Invalid --buffer-size-2pow exponent (max 36).\n";
        std::exit(1);
      }
      options.buffer_bytes = (size_t{1} << exponent);
    } else if (argument == "--interleave-stream-launches") {
      options.interleave_stream_launches = true;
    } else if (argument == "--multi-thread-enable") {
      options.multi_thread_enable = true;
    } else if (argument.rfind("--threads-per-kernel=", 0) == 0) {
      options.threads_per_kernel = static_cast<unsigned>(std::stoul(argument.substr(21)));
    } else if (argument == "-h" || argument == "--help") {
      std::cout << "llvm_memIntrinsic_stress — multi-stream large-buffer __builtin_mem* stress\n"
                << "\n"
                << "Options:\n"
                << "  --streams=N              (default 1)\n"
                << "  --devices=D              HIP devices 0..D-1; stream s → device (s % D) (default 1)\n"
                << "  --kernels-per-stream=M   (default 32)\n"
                << "  --seed=S\n"
                << "  --buffer-size-bytes=N    per-stream buffer size in bytes (default 32 GiB; max 2^36 B)\n"
                << "  --buffer-size-2pow=P     per-stream buffer = 2^P bytes (max P=36)\n"
                << "  --interleave-stream-launches  submit op k on all streams before op k+1\n"
                << "  --multi-thread-enable      memset & memcpy use multi-thread kernels; memmove stays 1 thread\n"
                << "  --threads-per-kernel=N    blockDim.x when multi-thread (default " << kDefaultThreadsPerBlock
                << "; max " << kMaxThreadsPerBlock << ")\n"
                << "  -h, --help\n";
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << argument << "\n";
      std::exit(1);
    }
  }
  return options;
}

} // namespace mem_intrinsic_stress

int main(int argc, char **argv) {
  using mem_intrinsic_stress::HarnessCliOptions;
  using mem_intrinsic_stress::LargeBufferStressRunner;
  using mem_intrinsic_stress::parseHarnessCommandLine;

  HarnessCliOptions cli = parseHarnessCommandLine(argc, argv);

  if (cli.num_hip_devices == 0) {
    std::cerr << "--devices must be at least 1.\n";
    return 1;
  }

  const size_t buffer_bytes = cli.buffer_bytes;
  if (buffer_bytes < kMinBytesPerStream) {
    std::cerr << "Per-stream buffer size must be at least " << kMinBytesPerStream << " bytes.\n";
    return 1;
  }
  if (buffer_bytes > kMaxBytesPerStream) {
    std::cerr << "Buffer size exceeds maximum (" << formatBytesHumanReadable(kMaxBytesPerStream) << ").\n";
    return 1;
  }
  if (cli.multi_thread_enable) {
    if (cli.threads_per_kernel < kMinThreadsPerBlock || cli.threads_per_kernel > kMaxThreadsPerBlock) {
      std::cerr << "--threads-per-kernel must be in [" << kMinThreadsPerBlock << ", " << kMaxThreadsPerBlock
                << "].\n";
      return 1;
    }
  }

  std::cout << "=== llvm_memIntrinsic_stress ===\n";
  std::cout << "config:";
  appendStressHarnessFields(std::cout, buffer_bytes, cli.streams, cli.num_hip_devices, cli.kernels_per_stream,
                            cli.seed, cli.interleave_stream_launches, cli.multi_thread_enable,
                            cli.threads_per_kernel);
  std::cout << "\n";

  LargeBufferStressRunner::Config runner_config;
  runner_config.num_streams = cli.streams;
  runner_config.num_hip_devices = cli.num_hip_devices;
  runner_config.kernels_per_stream = cli.kernels_per_stream;
  runner_config.seed = cli.seed;
  runner_config.bytes_per_buffer = buffer_bytes;
  runner_config.interleave_stream_launches = cli.interleave_stream_launches;
  runner_config.multi_thread_memops = cli.multi_thread_enable;
  runner_config.threads_per_kernel = cli.threads_per_kernel;

  LargeBufferStressRunner runner(runner_config);
  if (!runner.run(std::cout)) {
    std::cerr << "[FAIL] llvm_memIntrinsic_stress\n";
    return 1;
  }
  return 0;
}
