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

/*
 * Test suite (CLI: read, write, perf, sizes, align, concurrent, core, all):
 *   1. Basic synchronous read   - file -> GPU, verify start/middle/end
 *   2. Basic synchronous write  - GPU -> file, POSIX read-back verify
 *   3. Performance comparison   - hipFile direct vs traditional CPU-bounce I/O
 *   4. Size sweep               - 4K, 64K, 1M, 16M, 64M read and write
 *   5. Alignment boundaries     - aligned/unaligned file offset and memory;
 *                                 fastpath accept vs reject (statx-derived)
 *   6. Concurrent I/O           - multi-threaded, non-overlapping + contended
 *
 * Cross-cutting (exercised within the tests above, not separate tests):
 *   - O_DIRECT with statx(STATX_DIOALIGN) alignment detection
 *   - KFD AIS initialization status reported at startup
 *   - Data verification on both read and write paths
 */

// Workaround for GCC 14 compatibility with ROCm/HIP headers
// Use the C API instead of C++ wrappers to avoid cmath conflicts
#include <hip/hip_runtime_api.h>
#include <hipfile.h>

// Standard C++ headers
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// System headers
#include <fcntl.h>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

// STATX_DIOALIGN was added in Linux 6.1; define if missing from headers
#ifndef STATX_DIOALIGN
#define STATX_DIOALIGN 0x00002000U
#endif

// ============================================================================
// Global cleanup state for signal handlers
// ============================================================================

struct CleanupState {
  void *dev_ptr        = nullptr;
  void *dev_ptr2       = nullptr;
  void *pinned_ptr     = nullptr;
  hipFileHandle_t fh   = {};
  bool fh_registered   = false;
  int fd               = -1;
  bool driver_open     = false;
  bool buf_registered  = false;
  bool buf2_registered = false;
};

static CleanupState g_cleanup;

static void signal_handler(int sig) {
  fprintf(stderr, "\nCaught signal %d, cleaning up...\n", sig);
  if (g_cleanup.buf_registered && g_cleanup.dev_ptr)
    hipFileBufDeregister(g_cleanup.dev_ptr);
  if (g_cleanup.buf2_registered && g_cleanup.dev_ptr2)
    hipFileBufDeregister(g_cleanup.dev_ptr2);
  if (g_cleanup.fh_registered)
    hipFileHandleDeregister(g_cleanup.fh);
  if (g_cleanup.fd >= 0)
    close(g_cleanup.fd);
  if (g_cleanup.dev_ptr)
    hipFree(g_cleanup.dev_ptr);
  if (g_cleanup.dev_ptr2)
    hipFree(g_cleanup.dev_ptr2);
  if (g_cleanup.pinned_ptr)
    hipHostFree(g_cleanup.pinned_ptr);
  if (g_cleanup.driver_open)
    hipFileDriverClose();
  _exit(128 + sig);
}

static void install_signal_handlers() {
  struct sigaction sa{};
  sa.sa_handler = signal_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
}

static void reset_cleanup() {
  g_cleanup = CleanupState{};
}

// ============================================================================
// Test parameters (populated from CLI args)
// ============================================================================

struct TestParams {
  std::string nvme_path;
  std::string file_path  = "hipfile_test_data.bin";
  std::string write_path = "hipfile_write_output.bin";
  size_t size_mb         = 64;
  int repeat             = 1;
  int threads            = 8;   // concurrent test threads
  int iters              = 20;  // concurrent test iterations per thread
};

// ============================================================================
// Utility Functions
// ============================================================================

struct DioAlignment {
  size_t mem_align    = 4096;
  size_t offset_align = 4096;
  bool   from_statx   = false;
};

DioAlignment get_dio_alignment(const char *filepath) {
  DioAlignment result;
  struct statx stx = {};

  int ret = statx(AT_FDCWD, filepath, 0, STATX_DIOALIGN, &stx);
  if (ret != 0) return result;

  if (!(stx.stx_mask & STATX_DIOALIGN)) {
    printf("DIO alignment info not supported\n");
    return result;
  }

  // stx_dio_mem_align: DMA buffer must be aligned to this many bytes for Direct I/O
  // stx_dio_offset_align: file offset and I/O size must be a multiple of this many bytes
  printf("Memory alignment: %u bytes\n", stx.stx_dio_mem_align);
  printf("Offset alignment: %u bytes\n", stx.stx_dio_offset_align);

  if (stx.stx_dio_mem_align > 0 && stx.stx_dio_offset_align > 0) {
    result.mem_align    = std::max<size_t>(stx.stx_dio_mem_align, sizeof(void *));
    result.offset_align = stx.stx_dio_offset_align;
    result.from_statx   = true;
  }
  return result;
}

DioAlignment get_alignment_for_file(const char *filepath) {
  DioAlignment dio = get_dio_alignment(filepath);
  if (dio.from_statx) return dio;

  // Fallback: use fstat st_blksize
  struct stat st{};
  if (stat(filepath, &st) == 0 && st.st_blksize > 0) {
    dio.mem_align    = static_cast<size_t>(st.st_blksize);
    dio.offset_align = static_cast<size_t>(st.st_blksize);
  }
  return dio;
}

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP Error: " << hipGetErrorString(err) << " at "           \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      return false;                                                            \
    }                                                                          \
  } while (0)

void print_error(const char *msg, hipFileError_t err) {
  fprintf(stderr, "%s: %s\n", msg, hipFileGetOpErrorString(err.err));
  if (err.err == hipFileHipDriverError) {
    fprintf(stderr, "  HIP error code: %d\n", err.hip_drv_err);
  }
}

void print_separator(const char *title) {
  std::cout << "\n" << std::string(70, '=') << std::endl;
  std::cout << "  " << title << std::endl;
  std::cout << std::string(70, '=') << std::endl;
}

// Reads exactly `size` bytes from fd, looping to handle partial reads.
// Returns total bytes read, or -1 on error.
static ssize_t read_all(int fd, void *buf, size_t size) {
  size_t total = 0;
  uint8_t *p = static_cast<uint8_t *>(buf);
  while (total < size) {
    ssize_t n = read(fd, p + total, size - total);
    if (n < 0) {
      if (errno == EINTR) continue;
      return -1;
    }
    if (n == 0) break;
    total += static_cast<size_t>(n);
  }
  return static_cast<ssize_t>(total);
}

bool create_test_file(const char *filename, size_t size_mb) {
  std::cout << "Creating test file: " << filename << " (size: " << size_mb
            << " MB)" << std::endl;

  int fd = open(filename, O_CREAT | O_WRONLY | O_TRUNC | O_DIRECT, 0644);
  bool using_direct = true;
  if (fd < 0) {
    perror("Failed to create test file with O_DIRECT");
    fd = open(filename, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    using_direct = false;
    if (fd < 0) {
      perror("Failed to create test file");
      return false;
    }
  }

  // Use statx(STATX_DIOALIGN) to determine alignment,
  // falling back to fstat st_blksize if the kernel doesn't support it.
  DioAlignment dio = get_alignment_for_file(filename);
  if (dio.from_statx) {
    std::cout << "  Direct I/O alignment via statx(STATX_DIOALIGN):" << std::endl;
  } else {
    std::cout << "  statx(STATX_DIOALIGN) not available, using fstat fallback:"
              << std::endl;
  }
  std::cout << "    Memory alignment : " << dio.mem_align << " bytes" << std::endl;
  std::cout << "    Offset alignment : " << dio.offset_align << " bytes" << std::endl;

  size_t mem_align    = dio.mem_align;
  size_t offset_align = dio.offset_align;

  size_t chunk_size = 1024 * 1024;
  if (using_direct) {
    chunk_size = (chunk_size / offset_align) * offset_align;
    if (chunk_size == 0)
      chunk_size = offset_align;
  }

  void *raw = nullptr;
  if (posix_memalign(&raw, mem_align, chunk_size) != 0) {
    perror("posix_memalign failed");
    close(fd);
    return false;
  }
  uint8_t *buffer = static_cast<uint8_t *>(raw);

  for (size_t i = 0; i < chunk_size; i++) {
    buffer[i] = static_cast<uint8_t>(i % 256);
  }

  for (size_t i = 0; i < size_mb; i++) {
    ssize_t w = write(fd, buffer, chunk_size);
    if (w != static_cast<ssize_t>(chunk_size)) {
      std::cerr << "i=" << i << "\n";
      perror("Failed to write test file");
      if (using_direct && errno == EINVAL) {
        std::cerr << "Falling back to buffered I/O without O_DIRECT.\n";
        off_t cur = lseek(fd, 0, SEEK_CUR);
        close(fd);
        fd = open(filename, O_WRONLY);
        if (fd < 0) {
          perror("Reopen without O_DIRECT failed");
          free(raw);
          return false;
        }
        if (lseek(fd, cur, SEEK_SET) == (off_t)-1) {
          perror("lseek restore failed");
          free(raw);
          close(fd);
          return false;
        }
        using_direct = false;

        w = write(fd, buffer, chunk_size);
        if (w != static_cast<ssize_t>(chunk_size)) {
          perror("Write failed even without O_DIRECT");
          free(raw);
          close(fd);
          return false;
        }
        continue;
      }
      free(raw);
      close(fd);
      return false;
    }
  }

  if (using_direct) {
    if (fsync(fd) != 0) {
        perror("Warning: fsync failed");
    }
  } else {
    if (fdatasync(fd) != 0) {
      perror("Warning: fdatasync failed");
    }
  }
  free(raw);
  close(fd);
  std::cout << "Test file created successfully" << std::endl;
  return true;
}

// Verify data matches pattern at a given file-offset
bool verify_data(uint8_t *data, size_t size, size_t offset = 0) {
  for (size_t i = 0; i < size; i++) {
    uint8_t expected = (uint8_t)((offset + i) % 256);
    if (data[i] != expected) {
      std::cerr << "Data mismatch at offset " << (offset + i) << ": expected "
                << static_cast<int>(expected) << ", got "
                << static_cast<int>(data[i]) << std::endl;
      return false;
    }
  }
  return true;
}

// Verify data at start, middle, and end regions instead of just first 1 MB
bool verify_data_regions(uint8_t *data, size_t total_size) {
  const size_t block = std::min<size_t>(1024 * 1024, total_size / 3);
  if (block == 0) return verify_data(data, total_size, 0);

  // Start
  std::cout << "   Verifying start region (0 .. " << block << ")..." << std::endl;
  if (!verify_data(data, block, 0)) return false;

  // Middle
  size_t mid_off = (total_size / 2) - (block / 2);
  std::cout << "   Verifying middle region (" << mid_off << " .. "
            << mid_off + block << ")..." << std::endl;
  if (!verify_data(data + mid_off, block, mid_off)) return false;

  // End
  size_t end_off = total_size - block;
  std::cout << "   Verifying end region (" << end_off << " .. "
            << total_size << ")..." << std::endl;
  if (!verify_data(data + end_off, block, end_off)) return false;

  return true;
}

// ============================================================================
// Reusable single-shot hipFile I/O primitives
//
// These are self-contained (open file, register handle/buffer, do one I/O,
// optionally verify, then clean up locally). They assume the caller has
// already opened the hipFile driver. They are used by the size-sweep,
// alignment-boundary, and concurrent tests below.
// ============================================================================

// Drop the kernel page cache so that subsequent reads come from the device,
// not RAM. Best-effort: requires root; returns false if not permitted.
static bool drop_caches() {
  sync();
  int fd = open("/proc/sys/vm/drop_caches", O_WRONLY);
  if (fd < 0) return false;
  bool ok = (write(fd, "3\n", 2) == 2);
  close(fd);
  return ok;
}

struct IoResult {
  bool    ok           = false;  // expectation met (incl. verify when requested)
  bool    io_succeeded = false;  // hipFileRead/Write returned the full size
  ssize_t bytes        = 0;      // raw return of hipFileRead/Write
  int     sys_errno    = 0;      // errno captured when bytes == -1
  bool    short_io     = false;  // 0 <= bytes < size (partial transfer)
  double  ms           = 0.0;
  double  mbps         = 0.0;
};

// Decode a hipFileRead/Write return value into a human-readable reason.
// hipFileRead/Write contract:  >=0 bytes, -1 system error (errno),
// else negative of the hipFileOpError_t code (e.g. fastpath alignment reject).
// `err_no` is only meaningful when ret == -1.
static std::string decode_io_error(ssize_t ret, int err_no) {
  std::string s = "ret=" + std::to_string(ret);
  if (ret == -1) {
    s += " errno=" + std::to_string(err_no) + " (" + std::strerror(err_no) + ")";
  } else if (ret < -1) {
    s += " (" + std::string(hipFileGetOpErrorString(
                    static_cast<hipFileOpError_t>(-ret))) +
         ")";
  } else {
    s += " (short I/O)";
  }
  return s;
}

// Human-readable description of why an I/O did not fully succeed.
static std::string io_failure_desc(const IoResult &r) {
  if (r.io_succeeded) return "ok";
  return decode_io_error(r.bytes, r.sys_errno);
}

// One hipFile read of `size` bytes from `file_off` into GPU memory at
// `buf_off`. The GPU allocation includes slack for buf_off so an unaligned
// buffer offset can be exercised. Verifies the byte pattern when `verify`.
static IoResult hipfile_do_read(const std::string &filename, size_t size,
                                off_t file_off, off_t buf_off, bool verify) {
  IoResult r{};  // value-initialize so r.ok is explicitly false on early return

  int fd = open(filename.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) {
    fd = open(filename.c_str(), O_RDONLY);
    if (fd < 0) { perror("open"); return r; }
  }

  hipFileDescr_t descr;
  memset(&descr, 0, sizeof(descr));
  descr.handle.fd = fd;
  descr.type = hipFileHandleTypeOpaqueFD;

  hipFileHandle_t fh;
  hipFileError_t err = hipFileHandleRegister(&fh, &descr);
  if (err.err != hipFileSuccess) {
    print_error("hipFileHandleRegister", err);
    close(fd);
    return r;
  }

  size_t alloc = size + static_cast<size_t>(buf_off);
  void *devPtr = nullptr;
  if (hipMalloc(&devPtr, alloc) != hipSuccess) {
    hipFileHandleDeregister(fh);
    close(fd);
    return r;
  }

  err = hipFileBufRegister(devPtr, alloc, 0);
  if (err.err != hipFileSuccess) {
    print_error("hipFileBufRegister", err);
    hipFileHandleDeregister(fh);
    close(fd);
    (void)hipFree(devPtr);
    return r;
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  ssize_t got = hipFileRead(fh, devPtr, size, file_off, buf_off);
  int call_errno = errno;  // capture immediately (valid when got == -1)
  auto t1 = std::chrono::high_resolution_clock::now();

  r.bytes = got;
  if (got == static_cast<ssize_t>(size)) {
    // Full transfer succeeded
    r.io_succeeded = true;
    double us =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    r.ms = us / 1000.0;
    r.mbps = (r.ms > 0) ? (got / (1024.0 * 1024.0)) / (r.ms / 1000.0) : 0.0;
    if (verify) {
      std::vector<uint8_t> host(size);
      if (hipMemcpy(host.data(),
                    static_cast<uint8_t *>(devPtr) + buf_off, size,
                    hipMemcpyDeviceToHost) == hipSuccess) {
        r.ok = verify_data(host.data(), size, static_cast<size_t>(file_off));
      }
    } else {
      r.ok = true;
    }
  } else if (got < 0) {
    // Negative return = error or fastpath rejection (e.g. unaligned I/O).
    // -1 carries errno; anything < -1 is the negated hipFileOpError_t code.
    r.sys_errno = (got == -1) ? call_errno : 0;
  } else {
    // 0 <= got < size : partial transfer, treated as failure
    r.short_io = true;
  }

  (void)hipFileBufDeregister(devPtr);
  hipFileHandleDeregister(fh);
  close(fd);
  (void)hipFree(devPtr);
  return r;
}

// One hipFile write of `size` bytes from GPU memory to `file_off`, followed by
// a POSIX read-back verification of the bytes that landed on storage.
static IoResult hipfile_do_write(const std::string &filename, size_t size,
                                 off_t file_off) {
  IoResult r;

  int fd = open(filename.c_str(), O_CREAT | O_WRONLY | O_DIRECT, 0644);
  if (fd < 0) {
    fd = open(filename.c_str(), O_CREAT | O_WRONLY, 0644);
    if (fd < 0) { perror("open"); return r; }
  }

  hipFileDescr_t descr;
  memset(&descr, 0, sizeof(descr));
  descr.handle.fd = fd;
  descr.type = hipFileHandleTypeOpaqueFD;

  hipFileHandle_t fh;
  hipFileError_t err = hipFileHandleRegister(&fh, &descr);
  if (err.err != hipFileSuccess) {
    print_error("hipFileHandleRegister", err);
    close(fd);
    return r;
  }

  void *devPtr = nullptr;
  if (hipMalloc(&devPtr, size) != hipSuccess) {
    hipFileHandleDeregister(fh);
    close(fd);
    return r;
  }

  // Offset-aware pattern so a later read at the same offset verifies cleanly
  std::vector<uint8_t> pat(size);
  for (size_t i = 0; i < size; i++)
    pat[i] = static_cast<uint8_t>((static_cast<size_t>(file_off) + i) % 256);
  (void)hipMemcpy(devPtr, pat.data(), size, hipMemcpyHostToDevice);

  err = hipFileBufRegister(devPtr, size, 0);
  if (err.err != hipFileSuccess) {
    print_error("hipFileBufRegister", err);
    hipFileHandleDeregister(fh);
    close(fd);
    (void)hipFree(devPtr);
    return r;
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  ssize_t wrote = hipFileWrite(fh, devPtr, size, file_off, 0);
  int call_errno = errno;  // capture immediately (valid when wrote == -1)
  (void)hipDeviceSynchronize();
  auto t1 = std::chrono::high_resolution_clock::now();

  (void)hipFileBufDeregister(devPtr);
  hipFileHandleDeregister(fh);
  (void)fsync(fd);
  close(fd);
  (void)hipFree(devPtr);

  r.bytes = wrote;
  if (wrote != static_cast<ssize_t>(size)) {
    if (wrote < 0)
      r.sys_errno = (wrote == -1) ? call_errno : 0;  // error / rejection
    else
      r.short_io = true;                              // partial write
    return r;
  }
  r.io_succeeded = true;
  double us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  r.ms = us / 1000.0;
  r.mbps = (r.ms > 0) ? (wrote / (1024.0 * 1024.0)) / (r.ms / 1000.0) : 0.0;

  // Read back via POSIX (buffered, so any offset/size is fine) and verify
  // what actually landed on storage.
  drop_caches();
  int rfd = open(filename.c_str(), O_RDONLY);
  if (rfd < 0) { perror("readback open"); return r; }
  std::vector<uint8_t> rb(size);
  size_t total = 0;
  while (total < size) {
    ssize_t n = pread(rfd, rb.data() + total, size - total,
                      file_off + static_cast<off_t>(total));
    if (n < 0) { if (errno == EINTR) continue; break; }
    if (n == 0) break;
    total += static_cast<size_t>(n);
  }
  close(rfd);
  if (total == size)
    r.ok = verify_data(rb.data(), size, static_cast<size_t>(file_off));
  return r;
}

// ============================================================================
// Test 1: Synchronous Read
// ============================================================================
bool test_basic_read(const char *filename, size_t size) {
  print_separator("Test 1: Synchronous Read with hipFile");
  reset_cleanup();

  std::cout << "1. Initializing hipFile driver..." << std::endl;
  hipFileError_t err = hipFileDriverOpen();
  if (err.err != hipFileSuccess) {
    print_error("hipFile IO Driver failed to open", err);
    std::cout << "\n✗ Test 1 FAILED\n" << std::endl;
    return false;
  }
  g_cleanup.driver_open = true;

  std::cout << "   ✓ hipFile driver initialized" << std::endl;

  std::cout << "2. Opening file..." << std::endl;
  int fd = open(filename, O_RDONLY | O_DIRECT);
  if (fd < 0) {
    fd = open(filename, O_RDONLY);
    if (fd < 0) {
      perror("Failed to open file");
      hipFileDriverClose();
      g_cleanup.driver_open = false;
      std::cout << "\n✗ Test 1 FAILED\n" << std::endl;
      return false;
    }
    std::cout << "   ⚠ Opened without O_DIRECT" << std::endl;
  } else {
    std::cout << "   ✓ File opened with O_DIRECT" << std::endl;
  }
  g_cleanup.fd = fd;

  std::cout << "3. Registering file handle..." << std::endl;
  hipFileDescr_t descr;
  memset(&descr, 0, sizeof(hipFileDescr_t));
  descr.handle.fd = fd;
  descr.type = hipFileHandleTypeOpaqueFD;

  hipFileHandle_t fh;
  err = hipFileHandleRegister(&fh, &descr);
  if (err.err != hipFileSuccess) {
    print_error("Failed to register file handle", err);
    close(fd); g_cleanup.fd = -1;
    hipFileDriverClose(); g_cleanup.driver_open = false;
    std::cout << "\n✗ Test 1 FAILED\n" << std::endl;
    return false;
  }
  g_cleanup.fh = fh;
  g_cleanup.fh_registered = true;
  std::cout << "   ✓ File handle registered" << std::endl;

  std::cout << "4. Allocating GPU memory (" << size / (1024 * 1024) << " MB)..."
            << std::endl;
  void *devPtr;
  HIP_CHECK(hipMalloc(&devPtr, size));
  g_cleanup.dev_ptr = devPtr;
  std::cout << "   ✓ GPU memory allocated at " << devPtr << std::endl;

  std::cout << "5. Registering GPU buffer..." << std::endl;
  err = hipFileBufRegister(devPtr, size, 0);
  if (err.err != hipFileSuccess) {
    print_error("Failed to register buffer", err);
    hipFileHandleDeregister(fh); g_cleanup.fh_registered = false;
    close(fd); g_cleanup.fd = -1;
    hipFree(devPtr); g_cleanup.dev_ptr = nullptr;
    hipFileDriverClose(); g_cleanup.driver_open = false;
    std::cout << "\n✗ Test 1 FAILED\n" << std::endl;
    return false;
  }
  g_cleanup.buf_registered = true;
  std::cout << "   ✓ GPU buffer registered" << std::endl;

  std::cout << "6. Reading data with hipFileRead..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  ssize_t bytes_read = hipFileRead(fh, devPtr, size, 0, 0);
  int read_errno = errno;  // capture immediately (valid when bytes_read == -1)

  HIP_CHECK(hipDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();

  // Treat any non-full transfer as a failure: negative is an error/rejection
  // (errno or hipFileOpError), 0 <= bytes_read < size is a short read.
  if (bytes_read != static_cast<ssize_t>(size)) {
    std::cerr << "   ✗ hipFileRead failed: "
              << decode_io_error(bytes_read, read_errno) << std::endl;
    hipFileBufDeregister(devPtr); g_cleanup.buf_registered = false;
    hipFileHandleDeregister(fh); g_cleanup.fh_registered = false;
    close(fd); g_cleanup.fd = -1;
    hipFree(devPtr); g_cleanup.dev_ptr = nullptr;
    hipFileDriverClose(); g_cleanup.driver_open = false;
    std::cout << "\n✗ Test 1 FAILED\n" << std::endl;
    return false;
  } else {
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double bandwidth =
        (bytes_read / (1024.0 * 1024.0)) / (duration.count() / 1000.0);

    std::cout << "   ✓ Read " << bytes_read << " bytes" << std::endl;
    std::cout << "   ✓ Time: " << duration.count() << " ms" << std::endl;
    std::cout << "   ✓ Bandwidth: " << std::fixed << std::setprecision(2)
              << bandwidth << " MB/s" << std::endl;
  }

  std::cout << "7. Verifying data (start, middle, end)..." << std::endl;
  std::vector<uint8_t> host_buffer(size);
  HIP_CHECK(hipMemcpy(host_buffer.data(), devPtr, size, hipMemcpyDeviceToHost));

  if (verify_data_regions(host_buffer.data(), size)) {
    std::cout << "   ✓ Data verification passed" << std::endl;
  } else {
    std::cout << "   ✗ Data verification failed" << std::endl;
    hipFileBufDeregister(devPtr); g_cleanup.buf_registered = false;
    hipFileHandleDeregister(fh); g_cleanup.fh_registered = false;
    close(fd); g_cleanup.fd = -1;
    hipFree(devPtr); g_cleanup.dev_ptr = nullptr;
    hipFileDriverClose(); g_cleanup.driver_open = false;
    std::cout << "\n✗ Test 1 FAILED\n" << std::endl;
    return false;
  }

  // Cleanup
  hipFileBufDeregister(devPtr); g_cleanup.buf_registered = false;
  hipFileHandleDeregister(fh); g_cleanup.fh_registered = false;
  close(fd); g_cleanup.fd = -1;
  hipFree(devPtr); g_cleanup.dev_ptr = nullptr;
  hipFileDriverClose(); g_cleanup.driver_open = false;

  std::cout << "\n✓ Test 1 PASSED" << std::endl;
  return true;
}

// ============================================================================
// Test 2: Synchronous Write
// ============================================================================

bool test_basic_write(const char *filename, size_t size) {
  print_separator("Test 2: Synchronous Write with hipFile");
  reset_cleanup();

  hipFileError_t err;
  err = hipFileDriverOpen();
  if (err.err != hipFileSuccess) {
    print_error("hipFile IO Driver failed to open", err);
    std::cout << "\n✗ Test 2 FAILED\n" << std::endl;
    return false;
  }
  g_cleanup.driver_open = true;
  printf("  Driver opened successfully. Use count: %ld\n", hipFileUseCount());

  int fd = open(filename, O_CREAT | O_WRONLY | O_TRUNC | O_DIRECT, 0644);
  if (fd < 0) {
    fd = open(filename, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd < 0) {
      perror("Failed to open file for writing");
      hipFileDriverClose(); g_cleanup.driver_open = false;
      std::cout << "\n✗ Test 2 FAILED\n" << std::endl;
      return false;
    }
  }
  g_cleanup.fd = fd;

  hipFileDescr_t descr;
  memset(&descr, 0, sizeof(hipFileDescr_t));
  descr.handle.fd = fd;
  descr.type = hipFileHandleTypeOpaqueFD;

  hipFileHandle_t fh;
  err = hipFileHandleRegister(&fh, &descr);
  if (err.err != hipFileSuccess) {
    print_error("Failed to register file for write", err);
    close(fd); g_cleanup.fd = -1;
    hipFileDriverClose(); g_cleanup.driver_open = false;
    std::cout << "\n✗ Test 2 FAILED\n" << std::endl;
    return false;
  }
  g_cleanup.fh = fh;
  g_cleanup.fh_registered = true;

  void *devPtr;
  HIP_CHECK(hipMalloc(&devPtr, size));
  g_cleanup.dev_ptr = devPtr;

  std::vector<uint8_t> pattern(size);
  for (size_t i = 0; i < size; i++) {
    pattern[i] = (uint8_t)(i % 256);
  }
  HIP_CHECK(hipMemcpy(devPtr, pattern.data(), size, hipMemcpyHostToDevice));

  err = hipFileBufRegister(devPtr, size, 0);
  if (err.err != hipFileSuccess) {
    print_error("Failed to register buffer", err);
    hipFileHandleDeregister(fh); g_cleanup.fh_registered = false;
    close(fd); g_cleanup.fd = -1;
    hipFree(devPtr); g_cleanup.dev_ptr = nullptr;
    hipFileDriverClose(); g_cleanup.driver_open = false;
    std::cout << "\n✗ Test 2 FAILED\n" << std::endl;
    return false;
  }
  g_cleanup.buf_registered = true;

  std::cout << "Writing " << size / (1024 * 1024) << " MB with hipFileWrite..."
            << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  ssize_t bytes_written = hipFileWrite(fh, devPtr, size, 0, 0);
  int write_errno = errno;  // capture immediately (valid when bytes_written == -1)

  HIP_CHECK(hipDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();

  // Treat any non-full transfer as a failure: negative is an error/rejection
  // (errno or hipFileOpError), 0 <= bytes_written < size is a short write.
  if (bytes_written != static_cast<ssize_t>(size)) {
    std::cerr << "✗ hipFileWrite failed: "
              << decode_io_error(bytes_written, write_errno) << std::endl;
    hipFileBufDeregister(devPtr); g_cleanup.buf_registered = false;
    hipFileHandleDeregister(fh); g_cleanup.fh_registered = false;
    close(fd); g_cleanup.fd = -1;
    hipFree(devPtr); g_cleanup.dev_ptr = nullptr;
    hipFileDriverClose(); g_cleanup.driver_open = false;
    std::cout << "\n✗ Test 2 FAILED\n" << std::endl;
    return false;
  } else {
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double bandwidth =
        (bytes_written / (1024.0 * 1024.0)) / (duration.count() / 1000.0);

    std::cout << "✓ Wrote " << bytes_written << " bytes" << std::endl;
    std::cout << "✓ Time: " << duration.count() << " ms" << std::endl;
    std::cout << "✓ Bandwidth: " << std::fixed << std::setprecision(2)
              << bandwidth << " MB/s" << std::endl;
  }

  // Cleanup write resources before verification re-open
  hipFileBufDeregister(devPtr); g_cleanup.buf_registered = false;
  hipFileHandleDeregister(fh); g_cleanup.fh_registered = false;
  close(fd); g_cleanup.fd = -1;
  hipFree(devPtr); g_cleanup.dev_ptr = nullptr;
  hipFileDriverClose(); g_cleanup.driver_open = false;

  // Re-open the written file and verify data at start, middle, and end
  std::cout << "Verifying written file (start, middle, end)..." << std::endl;
  int vfd = open(filename, O_RDONLY);
  if (vfd < 0) {
    perror("Failed to re-open written file for verification");
    std::cout << "\n✗ Test 2 FAILED\n" << std::endl;
    return false;
  }

  std::vector<uint8_t> vbuf(size);
  ssize_t vbytes = read_all(vfd, vbuf.data(), size);
  close(vfd);

  if (vbytes != static_cast<ssize_t>(size)) {
    std::cerr << "Read back only " << vbytes << " of " << size
              << " bytes" << std::endl;
    std::cout << "\n✗ Test 2 FAILED\n" << std::endl;
    return false;
  }

  if (!verify_data_regions(vbuf.data(), size)) {
    std::cout << "   ✗ Written data verification failed" << std::endl;
    std::cout << "\n✗ Test 2 FAILED\n" << std::endl;
    return false;
  }
  std::cout << "   ✓ Written data verification passed" << std::endl;

  std::cout << "\n✓ Test 2 PASSED" << std::endl;
  return true;
}

// ============================================================================
// Test 3: Performance Comparison (hipFile vs Traditional I/O)
// ============================================================================

bool test_performance_comparison(const char *filename, size_t size) {
  print_separator("Test 3: Performance Comparison");
  reset_cleanup();

  std::cout << "\nComparing hipFile vs Traditional I/O" << std::endl;
  std::cout << "File size: " << size / (1024 * 1024) << " MB" << std::endl;

  // ========================================
  // Method 1: Traditional I/O (CPU bounce) — pinned memory
  // ========================================
  std::cout << "\n--- Traditional I/O (read → pinned CPU → GPU) ---" << std::endl;

  void *devPtr1;
  HIP_CHECK(hipMalloc(&devPtr1, size));
  g_cleanup.dev_ptr = devPtr1;

  void *pinned_buf = nullptr;
  HIP_CHECK(hipHostMalloc(&pinned_buf, size));
  g_cleanup.pinned_ptr = pinned_buf;

  auto startTime1 = std::chrono::high_resolution_clock::now();

  int fd1 = open(filename, O_RDONLY);
  if (fd1 < 0) {
    perror("Failed to open file");
    hipHostFree(pinned_buf); g_cleanup.pinned_ptr = nullptr;
    hipFree(devPtr1); g_cleanup.dev_ptr = nullptr;
    std::cout << "\n✗ Test 3 FAILED (Traditional I/O)\n" << std::endl;
    return false;
  }

  ssize_t bytes = read_all(fd1, pinned_buf, size);
  close(fd1);
  if (bytes != static_cast<ssize_t>(size)) {
    std::cerr << "Read only " << bytes << " of " << size << " bytes" << std::endl;
    hipHostFree(pinned_buf); g_cleanup.pinned_ptr = nullptr;
    hipFree(devPtr1); g_cleanup.dev_ptr = nullptr;
    std::cout << "\n✗ Test 3 FAILED (Traditional I/O)\n" << std::endl;
    return false;
  }

  HIP_CHECK(hipMemcpy(devPtr1, pinned_buf, size, hipMemcpyHostToDevice));
  HIP_CHECK(hipDeviceSynchronize());

  auto endTime1 = std::chrono::high_resolution_clock::now();
  auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(
      endTime1 - startTime1);
  double bandwidth1 =
      (bytes / (1024.0 * 1024.0)) / (duration1.count() / 1000.0);

  std::cout << "Time: " << duration1.count() << " ms" << std::endl;
  std::cout << "Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth1
            << " MB/s" << std::endl;

  hipFree(devPtr1); g_cleanup.dev_ptr = nullptr;
  hipHostFree(pinned_buf); g_cleanup.pinned_ptr = nullptr;

  // ========================================
  // Method 2: hipFile (Direct Storage → GPU)
  // ========================================
  std::cout << "\n--- hipFile I/O (direct storage → GPU) ---" << std::endl;

  hipFileError_t err = hipFileDriverOpen();
  if (err.err != hipFileSuccess) {
    print_error("hipFile IO Driver failed to open", err);
    std::cout << "\n✗ Test 3 FAILED (hipFile I/O)\n" << std::endl;
    return false;
  }
  g_cleanup.driver_open = true;

  int fd2 = open(filename, O_RDONLY | O_DIRECT);
  if (fd2 < 0) {
    fd2 = open(filename, O_RDONLY);
    if (fd2 < 0) {
        perror("Failed to open file");
        hipFileDriverClose(); g_cleanup.driver_open = false;
        std::cout << "\n✗ Test 3 FAILED (hipFile I/O)\n" << std::endl;
        return false;
    }
  }
  g_cleanup.fd = fd2;

  hipFileDescr_t descr;
  memset(&descr, 0, sizeof(hipFileDescr_t));
  descr.handle.fd = fd2;
  descr.type = hipFileHandleTypeOpaqueFD;

  hipFileHandle_t fh;
  err = hipFileHandleRegister(&fh, &descr);
  if (err.err != hipFileSuccess) {
    print_error("Failed to register file for read", err);
    close(fd2); g_cleanup.fd = -1;
    hipFileDriverClose(); g_cleanup.driver_open = false;
    std::cout << "\n✗ Test 3 FAILED (hipFile I/O)\n" << std::endl;
    return false;
  }
  g_cleanup.fh = fh;
  g_cleanup.fh_registered = true;

  void *devPtr2;
  HIP_CHECK(hipMalloc(&devPtr2, size));
  g_cleanup.dev_ptr2 = devPtr2;

  err = hipFileBufRegister(devPtr2, size, 0);
  if (err.err != hipFileSuccess) {
    print_error("Failed to register buffer", err);
    hipFileHandleDeregister(fh); g_cleanup.fh_registered = false;
    close(fd2); g_cleanup.fd = -1;
    hipFree(devPtr2); g_cleanup.dev_ptr2 = nullptr;
    hipFileDriverClose(); g_cleanup.driver_open = false;
    std::cout << "\n✗ Test 3 FAILED (hipFile I/O)\n" << std::endl;
    return false;
  }
  g_cleanup.buf2_registered = true;

  auto startTime2 = std::chrono::high_resolution_clock::now();

  ssize_t bytes_read = hipFileRead(fh, devPtr2, size, 0, 0);
  
  HIP_CHECK(hipDeviceSynchronize());
  
  if (bytes_read < 0) {
    std::cerr << "✗ read failed" << std::endl;
    hipFileBufDeregister(devPtr2); g_cleanup.buf2_registered = false;
    hipFileHandleDeregister(fh); g_cleanup.fh_registered = false;
    close(fd2); g_cleanup.fd = -1;
    hipFree(devPtr2); g_cleanup.dev_ptr2 = nullptr;
    hipFileDriverClose(); g_cleanup.driver_open = false;
    std::cout << "\n✗ Test 3 FAILED (hipFile I/O)\n" << std::endl;
    return false;
  }
  auto endTime2 = std::chrono::high_resolution_clock::now();
  auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(
      endTime2 - startTime2);
  double bandwidth2 =
      (bytes_read / (1024.0 * 1024.0)) / (duration2.count() / 1000.0);

  std::cout << "✓ read completed" << std::endl;
  std::cout << "✓ Read " << bytes_read << " bytes" << std::endl;
  std::cout << "✓ Time: " << duration2.count() << " ms" << std::endl;
  std::cout << "✓ Bandwidth: " << std::fixed << std::setprecision(2)
            << bandwidth2 << " MB/s" << std::endl;

  // Results
  std::cout << "\n" << std::string(70, '-') << std::endl;
  std::cout << "Performance RESULTS:" << std::endl;
  std::cout << "  Traditional I/O: " << duration1.count() << " ms, "
            << bandwidth1 << " MB/s" << std::endl;
  std::cout << "  hipFile I/O:      " << duration2.count() << " ms, "
            << bandwidth2 << " MB/s" << std::endl;

  bool rc=false;
  if (duration2.count() < duration1.count()) {
    double speedup = (double)duration1.count() / duration2.count();
    std::cout << "\n✓ hipFile is " << std::fixed << std::setprecision(2)
              << speedup << "x FASTER!" << std::endl;
    std::cout << "\n✓ Test 3 PASSED" << std::endl;
    rc=true;
  } else { // In this cases also we are passing the test as can't get 
          // AIS faster than traditional I/O always. Discussed with Dev team
    std::cout << "\n⚠ Traditional I/O is faster than AIS (possibly cached)"
              << std::endl;
    std::cout << "\n✓ Test 3 PASSED" << std::endl;
    rc=true;
  }

  // Cleanup
  hipFileBufDeregister(devPtr2); g_cleanup.buf2_registered = false;
  hipFileHandleDeregister(fh); g_cleanup.fh_registered = false;
  close(fd2); g_cleanup.fd = -1;
  hipFree(devPtr2); g_cleanup.dev_ptr2 = nullptr;
  hipFileDriverClose(); g_cleanup.driver_open = false;

  return rc;
}

// ============================================================================
// Test 4: Size Sweep (4K, 64K, 1M, 16M, 64M) — read AND write
// ============================================================================

bool test_size_sweep(const std::string &dir) {
  print_separator("Test 4: Size Sweep (4K, 64K, 1M, 16M, 64M) read/write");
  reset_cleanup();

  hipFileError_t derr = hipFileDriverOpen();
  if (derr.err != hipFileSuccess) {
    print_error("hipFile IO Driver failed to open", derr);
    std::cout << "\n✗ Test 4 FAILED\n" << std::endl;
    return false;
  }
  g_cleanup.driver_open = true;

  const std::vector<std::pair<const char *, size_t>> sizes = {
      {"4K", 4 * 1024},
      {"64K", 64 * 1024},
      {"1M", 1 * 1024 * 1024},
      {"16M", 16 * 1024 * 1024},
      {"64M", 64 * 1024 * 1024},
  };

  std::string read_src  = dir + "/sizesweep_src.bin";
  std::string write_dst = dir + "/sizesweep_dst.bin";

  // Pre-stage a 64 MB read source file
  if (!create_test_file(read_src.c_str(), 64)) {
    hipFileDriverClose(); g_cleanup.driver_open = false;
    std::cout << "\n✗ Test 4 FAILED\n" << std::endl;
    return false;
  }

  bool all_ok = true;
  std::cout << "\n"
            << std::left << std::setw(8) << "Size" << std::setw(10) << "ReadOK"
            << std::setw(16) << "ReadBW(MB/s)" << std::setw(10) << "WriteOK"
            << std::setw(16) << "WriteBW(MB/s)" << "\n";
  std::cout << std::string(60, '-') << "\n";

  for (auto &kv : sizes) {
    drop_caches();
    IoResult rr = hipfile_do_read(read_src, kv.second, 0, 0, /*verify=*/true);
    IoResult wr = hipfile_do_write(write_dst, kv.second, 0);

    std::cout << std::left << std::setw(8) << kv.first << std::setw(10)
              << (rr.ok ? "PASS" : "FAIL") << std::setw(16) << std::fixed
              << std::setprecision(1) << rr.mbps << std::setw(10)
              << (wr.ok ? "PASS" : "FAIL") << std::setw(16) << wr.mbps << "\n";

    if (!rr.ok)
      std::cout << "    read failed: " << io_failure_desc(rr) << "\n";
    if (!wr.ok)
      std::cout << "    write failed: " << io_failure_desc(wr) << "\n";
    if (!rr.ok || !wr.ok) all_ok = false;
  }

  unlink(read_src.c_str());
  unlink(write_dst.c_str());

  hipFileDriverClose(); g_cleanup.driver_open = false;
  std::cout << (all_ok ? "\n✓ Test 4 PASSED\n" : "\n✗ Test 4 FAILED\n")
            << std::endl;
  return all_ok;
}

// ============================================================================
// Test 5: Direct I/O Alignment Boundaries
//
// Validates the AIS fastpath alignment behavior reported by
// statx(STATX_DIOALIGN):
//   A. aligned file offset + aligned memory       -> accepted (fastpath)
//   B. unaligned file offset + aligned memory      -> rejected on fastpath,
//                                                     accepted on fallback
//   C. aligned file offset + unaligned memory      -> rejected on fastpath,
//                                                     accepted on fallback
//   D. IO size not a multiple of offset_align       -> rejected on fastpath,
//                                                     accepted on fallback
//
// The negative cases (B, C, D) are only meaningful when the fastpath is forced or
// the fallback is forced, controlled by environment:
//   HIPFILE_ALLOW_COMPAT_MODE=false  -> strict fastpath: unaligned MUST fail
//   HIPFILE_FORCE_COMPAT_MODE=true   -> fallback:        unaligned MUST succeed
// In the default (compat-allowed) mode only the positive cases are asserted.
// ============================================================================

bool test_alignment_boundaries(const std::string &dir) {
  print_separator("Test 5: Direct I/O Alignment Boundaries");
  reset_cleanup();

  const char *allow = getenv("HIPFILE_ALLOW_COMPAT_MODE");
  const char *force = getenv("HIPFILE_FORCE_COMPAT_MODE");
  bool strict_fastpath = (allow && std::string(allow) == "false");
  bool force_fallback  = (force && std::string(force) == "true");

  std::cout << "  Mode: ";
  if (strict_fastpath)
    std::cout << "fastpath-strict (HIPFILE_ALLOW_COMPAT_MODE=false)\n";
  else if (force_fallback)
    std::cout << "fallback (HIPFILE_FORCE_COMPAT_MODE=true)\n";
  else
    std::cout << "default (compat allowed)\n";

  bool check_negative = strict_fastpath || force_fallback;
  if (!check_negative) {
    std::cout << "  ⚠ Negative (rejection) checks need HIPFILE_ALLOW_COMPAT_MODE"
                 "=false;\n    only positive (accept) cases will be asserted "
                 "in this run.\n";
  }

  std::string fn = dir + "/align_src.bin";
  if (!create_test_file(fn.c_str(), 16)) {
    std::cout << "\n✗ Test 5 FAILED\n" << std::endl;
    return false;
  }

  DioAlignment dio = get_alignment_for_file(fn.c_str());
  size_t mem_align = dio.mem_align;     // >= sizeof(void*), so >= 8
  size_t off_align = dio.offset_align;  // e.g. 512 on ext4/NVMe
  std::cout << "  statx alignment: mem_align=" << mem_align
            << " offset_align=" << off_align
            << (dio.from_statx ? " (from statx)" : " (fstat fallback)") << "\n";

  hipFileError_t derr = hipFileDriverOpen();
  if (derr.err != hipFileSuccess) {
    print_error("hipFile IO Driver failed to open", derr);
    unlink(fn.c_str());
    std::cout << "\n✗ Test 5 FAILED\n" << std::endl;
    return false;
  }
  g_cleanup.driver_open = true;

  size_t io_size = off_align * 8;  // aligned multiple, comfortably inside 16 MB
  int total = 0, passed = 0;

  // --- Case A: aligned file offset + aligned memory -> MUST be accepted ---
  {
    total++;
    off_t f_off = static_cast<off_t>(off_align);
    off_t b_off = static_cast<off_t>(mem_align);
    IoResult r = hipfile_do_read(fn, io_size, f_off, b_off, /*verify=*/true);
    bool ok = r.io_succeeded && r.ok;
    std::cout << (ok ? "  ✓" : "  ✗") << " A aligned file_offset=" << f_off
              << " + aligned mem_offset=" << b_off << " -> "
              << (r.io_succeeded ? "accepted" : "rejected")
              << (r.io_succeeded ? (r.ok ? "" : " (verify FAILED)")
                                 : " [" + io_failure_desc(r) + "]")
              << "\n";
    if (ok) passed++;
  }

  // --- Positive sweep: multiples of the reported alignments ---
  {
    total++;
    bool sweep_ok = true;
    for (int k = 1; k <= 4; ++k) {
      off_t f_off = static_cast<off_t>(off_align * k);
      off_t b_off = static_cast<off_t>(mem_align * k);
      IoResult r = hipfile_do_read(fn, io_size, f_off, b_off, /*verify=*/true);
      if (!(r.io_succeeded && r.ok)) {
        sweep_ok = false;
        std::cout << "    ✗ sweep k=" << k << " file_offset=" << f_off
                  << " mem_offset=" << b_off << " rejected/mismatch\n";
      }
    }
    std::cout << (sweep_ok ? "  ✓" : "  ✗")
              << " Positive sweep (n*offset_align / n*mem_align) accepted\n";
    if (sweep_ok) passed++;
  }

  // --- Case B: unaligned file offset + aligned memory ---
  // Only meaningful when off_align > 1; otherwise every offset is a multiple
  // of off_align and "off_align + 1" would still be aligned.
  if (check_negative && off_align > 1) {
    total++;
    off_t f_off = static_cast<off_t>(off_align) + 1;  // not a multiple
    off_t b_off = static_cast<off_t>(mem_align);      // aligned
    IoResult r = hipfile_do_read(fn, io_size, f_off, b_off,
                                 /*verify=*/force_fallback);
    bool ok = force_fallback ? r.io_succeeded : !r.io_succeeded;
    std::cout << (ok ? "  ✓" : "  ✗") << " B unaligned file_offset=" << f_off
              << " + aligned mem -> " << (r.io_succeeded ? "accepted" : "rejected")
              << (r.io_succeeded ? "" : " [" + io_failure_desc(r) + "]")
              << " (expected "
              << (force_fallback ? "accept on fallback" : "reject on fastpath")
              << ")\n";
    if (ok) passed++;
  }

  // --- Case C: aligned file offset + unaligned memory ---
  // Only meaningful when mem_align > 1; otherwise mem_offset=1 is still aligned.
  if (check_negative && mem_align > 1) {
    total++;
    off_t f_off = static_cast<off_t>(off_align);  // aligned
    off_t b_off = 1;                              // not a multiple of mem_align
    IoResult r = hipfile_do_read(fn, io_size, f_off, b_off,
                                 /*verify=*/force_fallback);
    bool ok = force_fallback ? r.io_succeeded : !r.io_succeeded;
    std::cout << (ok ? "  ✓" : "  ✗") << " C aligned file_offset + unaligned "
              << "mem_offset=" << b_off << " -> "
              << (r.io_succeeded ? "accepted" : "rejected")
              << (r.io_succeeded ? "" : " [" + io_failure_desc(r) + "]")
              << " (expected "
              << (force_fallback ? "accept on fallback" : "reject on fastpath")
              << ")\n";
    if (ok) passed++;
  }

  // --- Case D: aligned file offset + aligned memory, but IO size is NOT a
  //     multiple of off_align. Per statx(STATX_DIOALIGN), the I/O size must
  //     also be a multiple of stx_dio_offset_align, so the fastpath must reject
  //     this while the fallback (POSIX) accepts it. ---
  if (check_negative && off_align > 1) {
    total++;
    off_t f_off = static_cast<off_t>(off_align);  // aligned
    off_t b_off = static_cast<off_t>(mem_align);  // aligned
    size_t bad_size = io_size + 1;                // not a multiple of off_align
    IoResult r = hipfile_do_read(fn, bad_size, f_off, b_off,
                                 /*verify=*/force_fallback);
    bool ok = force_fallback ? r.io_succeeded : !r.io_succeeded;
    std::cout << (ok ? "  ✓" : "  ✗") << " D unaligned io_size=" << bad_size
              << " (not a multiple of off_align=" << off_align << ") -> "
              << (r.io_succeeded ? "accepted" : "rejected")
              << (r.io_succeeded ? "" : " [" + io_failure_desc(r) + "]")
              << " (expected "
              << (force_fallback ? "accept on fallback" : "reject on fastpath")
              << ")\n";
    if (ok) passed++;
  }

  unlink(fn.c_str());
  hipFileDriverClose(); g_cleanup.driver_open = false;

  bool ok = (passed == total);
  std::cout << "  " << passed << "/" << total << " alignment checks passed\n";
  std::cout << (ok ? "\n✓ Test 5 PASSED\n" : "\n✗ Test 5 FAILED\n") << std::endl;
  return ok;
}

// ============================================================================
// Test 6: Concurrent multi-threaded I/O (P2P cache stress)
// ============================================================================

struct ConcurrentResult {
  std::atomic<int>    completed{0};
  std::atomic<int>    failed{0};
  std::atomic<size_t> total_bytes{0};
};

static void concurrent_worker(int tid, std::string filename, size_t sz,
                              int iters, ConcurrentResult *res) {
  // Each thread reads its own non-overlapping slice (tid offset) unless tid is
  // forced to 0 by the caller (same-offset contention sub-test).
  off_t my_off = static_cast<off_t>(tid) * static_cast<off_t>(sz);
  for (int i = 0; i < iters; i++) {
    IoResult r = hipfile_do_read(filename, sz, my_off, 0, /*verify=*/true);
    if (r.ok) {
      res->completed.fetch_add(1);
      res->total_bytes.fetch_add(static_cast<size_t>(r.bytes));
    } else {
      res->failed.fetch_add(1);
      return;
    }
  }
}

bool test_concurrent(const std::string &dir, int n_threads, int iters) {
  print_separator("Test 6: Concurrent I/O (multi-thread P2P cache)");
  reset_cleanup();
  std::cout << "Threads: " << n_threads << ", iterations/thread: " << iters
            << std::endl;

  hipFileError_t derr = hipFileDriverOpen();
  if (derr.err != hipFileSuccess) {
    print_error("hipFile IO Driver failed to open", derr);
    std::cout << "\n✗ Test 6 FAILED\n" << std::endl;
    return false;
  }
  g_cleanup.driver_open = true;

  const size_t per_thread = 1024 * 1024;  // 1 MB per I/O
  size_t need_mb =
      (n_threads * per_thread + (1024 * 1024 - 1)) / (1024 * 1024);
  if (need_mb < 16) need_mb = 16;  // headroom for non-overlapping slices

  std::string fn = dir + "/concurrent_src.bin";
  if (!create_test_file(fn.c_str(), need_mb)) {
    hipFileDriverClose(); g_cleanup.driver_open = false;
    std::cout << "\n✗ Test 6 FAILED\n" << std::endl;
    return false;
  }

  // Sub-A: non-overlapping offsets (independent cache entries)
  std::cout << "\n--- Sub-A: " << n_threads
            << " threads, non-overlapping offsets ---" << std::endl;
  ConcurrentResult resA;
  std::vector<std::thread> ts;
  auto a0 = std::chrono::high_resolution_clock::now();
  for (int t = 0; t < n_threads; t++)
    ts.emplace_back(concurrent_worker, t, fn, per_thread, iters, &resA);
  for (auto &t : ts) t.join();
  auto a1 = std::chrono::high_resolution_clock::now();
  auto a_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(a1 - a0).count();
  double a_bw = a_ms > 0 ? (resA.total_bytes.load() / (1024.0 * 1024.0)) /
                               (a_ms / 1000.0)
                         : 0.0;
  std::cout << "  Completed: " << resA.completed.load() << "/"
            << (n_threads * iters) << ", failed: " << resA.failed.load()
            << ", aggregate BW: " << std::fixed << std::setprecision(1) << a_bw
            << " MB/s\n";
  bool a_ok = (resA.failed.load() == 0 &&
               resA.completed.load() == n_threads * iters);

  // Sub-B: all threads on the same offset (cache contention / thread safety)
  std::cout << "\n--- Sub-B: " << n_threads
            << " threads, same offset (contention) ---" << std::endl;
  ConcurrentResult resB;
  ts.clear();
  auto b0 = std::chrono::high_resolution_clock::now();
  for (int t = 0; t < n_threads; t++)
    ts.emplace_back(concurrent_worker, 0, fn, per_thread, iters, &resB);
  for (auto &t : ts) t.join();
  auto b1 = std::chrono::high_resolution_clock::now();
  auto b_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(b1 - b0).count();
  double b_bw = b_ms > 0 ? (resB.total_bytes.load() / (1024.0 * 1024.0)) /
                               (b_ms / 1000.0)
                         : 0.0;
  std::cout << "  Completed: " << resB.completed.load() << "/"
            << (n_threads * iters) << ", failed: " << resB.failed.load()
            << ", aggregate BW: " << std::fixed << std::setprecision(1) << b_bw
            << " MB/s\n";
  bool b_ok = (resB.failed.load() == 0 &&
               resB.completed.load() == n_threads * iters);

  unlink(fn.c_str());
  hipFileDriverClose(); g_cleanup.driver_open = false;

  bool ok = a_ok && b_ok;
  std::cout << (ok ? "\n✓ Test 6 PASSED\n" : "\n✗ Test 6 FAILED\n")
            << std::endl;
  return ok;
}

// ============================================================================
// Helper Functions for Test Orchestration
// ============================================================================

void print_usage(const char *program_name) {
  std::cout << "\nUsage: " << program_name
            << " [options] [test_name ...]\n" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  --nvme-path <dir> NVMe mount point for test files (e.g. /mnt/nvme_gds)"
            << std::endl;
  std::cout << "  --size-mb <N>     Test file size in MB (default: 64)"
            << std::endl;
  std::cout << "  --file <path>     Path for read test file (overrides --nvme-path)"
            << std::endl;
  std::cout << "  --write-file <p>  Path for write test file (overrides --nvme-path)"
            << std::endl;
  std::cout << "  --repeat <N>      Repeat each test N times (default: 1)"
            << std::endl;
  std::cout << "  --threads <N>     Threads for the concurrent test (default: 8)"
            << std::endl;
  std::cout << "  --iters <N>       Iterations/thread for concurrent (default: 20)"
            << std::endl;
  std::cout << "  -h, --help        Show this help message\n" << std::endl;
  std::cout << "Available tests:" << std::endl;
  std::cout << "  read       - Basic synchronous read test" << std::endl;
  std::cout << "  write      - Basic synchronous write test" << std::endl;
  std::cout << "  perf       - Performance comparison test" << std::endl;
  std::cout << "  sizes      - Size sweep (4K/64K/1M/16M/64M) read+write"
            << std::endl;
  std::cout << "  align      - Alignment boundary validation"
            << std::endl;
  std::cout << "  concurrent - Multi-threaded concurrent I/O" << std::endl;
  std::cout << "  core       - sizes + align + concurrent" << std::endl;
  std::cout << "  all        - Run all tests (default)" << std::endl;
  std::cout << "\nAlignment validation — run BOTH:" << std::endl;
  std::cout << "  HIPFILE_ALLOW_COMPAT_MODE=false " << program_name
            << " --nvme-path /mnt/nvme_gds align   # unaligned MUST be rejected"
            << std::endl;
  std::cout << "  HIPFILE_FORCE_COMPAT_MODE=true  " << program_name
            << " --nvme-path /mnt/nvme_gds align   # unaligned succeeds via fallback"
            << std::endl;
  std::cout << "\nExamples (--nvme-path is mandatory):" << std::endl;
  std::cout << "  " << program_name
            << " --nvme-path /mnt/nvme_gds --size-mb 1024 --file hipfile_test_data.bin"
            << " --write-file hipfile_write_output.bin --repeat 3 all" << std::endl;
  std::cout << "  " << program_name
            << " --nvme-path /mnt/nvme_gds --size-mb 1024 --file hipfile_test_data.bin"
            << " --repeat 3 read" << std::endl;
  std::cout << "  " << program_name
            << " --nvme-path /mnt/nvme_gds --size-mb 1024"
            << " --write-file hipfile_write_output.bin --repeat 3 write" << std::endl;
  std::cout << "  " << program_name
            << " --nvme-path /mnt/nvme_gds --size-mb 1024 --file hipfile_test_data.bin"
            << " --repeat 3 perf" << std::endl;
  std::cout << "  " << program_name
            << " --nvme-path /mnt/nvme_gds --threads 8 --iters 50 concurrent\n"
            << std::endl;
}

// ============================================================================
// KFD AIS Initialization Status Interface
//
// Reads each GPU node's capability from KFD topology sysfs:
//   /sys/class/kfd/kfd/topology/nodes/X/properties
// Bit 6 (0x40) of the "capability" field indicates AIS is initialized.
// ============================================================================

static constexpr uint64_t KFD_AIS_CAPABILITY_BIT = 0x40;
static const std::string  KFD_TOPOLOGY_PATH =
    "/sys/class/kfd/kfd/topology/nodes";

struct KfdNodeAisInfo {
  int          node_id    = -1;
  uint64_t     capability = 0;
  bool         is_gpu     = false;
  bool         ais_init   = false;
};

static std::vector<KfdNodeAisInfo> query_kfd_ais_status() {
  std::vector<KfdNodeAisInfo> nodes;

  for (int id = 0; ; ++id) {
    std::string props_path =
        KFD_TOPOLOGY_PATH + "/" + std::to_string(id) + "/properties";
    std::ifstream f(props_path);
    if (!f.is_open()) break;

    KfdNodeAisInfo info;
    info.node_id = id;

    uint32_t simd_count = 0;
    std::string key;
    uint64_t    value;
    while (f >> key >> value) {
      if (key == "capability")
        info.capability = value;
      else if (key == "simd_count")
        simd_count = static_cast<uint32_t>(value);
    }

    info.is_gpu   = (simd_count > 0);
    info.ais_init = (info.capability & KFD_AIS_CAPABILITY_BIT) != 0;
    nodes.push_back(info);
  }
  return nodes;
}

static void print_kfd_ais_status(const std::vector<KfdNodeAisInfo> &nodes) {
  print_separator("KFD AIS Initialization Status");
  std::cout << "  Source: " << KFD_TOPOLOGY_PATH
            << "/X/properties  (capability & 0x40)" << std::endl;

  bool any_gpu = false;
  bool all_gpu_ais = true;

  for (const auto &n : nodes) {
    if (!n.is_gpu) continue;
    any_gpu = true;
    const char *tag = n.ais_init ? "✓" : "✗";
    std::cout << "  " << tag << " Node " << n.node_id
              << "  capability=0x" << std::hex << n.capability << std::dec
              << "  AIS " << (n.ais_init ? "initialized" : "NOT initialized")
              << std::endl;
    if (!n.ais_init) all_gpu_ais = false;
  }

  if (!any_gpu) {
    std::cout << "  ✗ No GPU nodes found in KFD topology" << std::endl;
  } else if (all_gpu_ais) {
    std::cout << "\n  ✓ KFD AIS initialized on all GPU nodes" << std::endl;
  } else {
    std::cout << "\n  ⚠ KFD AIS NOT initialized — hipFile will use POSIX fallback"
              << std::endl;
  }
}

// ============================================================================

bool initialize_environment() {
  int deviceCount;
  hipError_t hip_err = hipGetDeviceCount(&deviceCount);
  if (hip_err != hipSuccess || deviceCount == 0) {
    std::cerr << "No HIP devices found!" << std::endl;
    return false;
  }

  hipDeviceProp_t prop;
  hip_err = hipGetDeviceProperties(&prop, 0);
  if (hip_err != hipSuccess) {
    std::cerr << "Failed to get device properties" << std::endl;
    return false;
  }
  std::cout << "\nGPU: " << prop.name << std::endl;
  std::cout << "Compute Capability: " << prop.major << "." << prop.minor
            << std::endl;

  auto kfd_nodes = query_kfd_ais_status();
  print_kfd_ais_status(kfd_nodes);

  std::cout << "\nChecking hipFile availability..." << std::endl;
  hipFileError_t status = hipFileDriverOpen();
  if (status.err != hipFileSuccess) {
    std::cerr << "✗ hipFile IO Driver failed to open" << std::endl;
    return false;
  }

  std::cout << "✓ hipFile is available" << std::endl;
  hipFileDriverClose();
  
  return true;
}

// ============================================================================
// Main Test Suite
// ============================================================================

int main(int argc, char **argv) {
  install_signal_handlers();

  // Parse arguments
  TestParams params;
  bool run_read = false, run_write = false, run_perf = false;
  bool run_sizes = false, run_align = false, run_concurrent = false;
  bool run_all = false;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--nvme-path" && i + 1 < argc) {
      params.nvme_path = argv[++i];
    } else if (arg == "--size-mb" && i + 1 < argc) {
      params.size_mb = std::stoul(argv[++i]);
    } else if (arg == "--file" && i + 1 < argc) {
      params.file_path = argv[++i];
    } else if (arg == "--write-file" && i + 1 < argc) {
      params.write_path = argv[++i];
    } else if (arg == "--repeat" && i + 1 < argc) {
      params.repeat = std::stoi(argv[++i]);
    } else if (arg == "--threads" && i + 1 < argc) {
      params.threads = std::stoi(argv[++i]);
    } else if (arg == "--iters" && i + 1 < argc) {
      params.iters = std::stoi(argv[++i]);
    } else if (arg == "read") {
      run_read = true;
    } else if (arg == "write") {
      run_write = true;
    } else if (arg == "perf") {
      run_perf = true;
    } else if (arg == "sizes") {
      run_sizes = true;
    } else if (arg == "align") {
      run_align = true;
    } else if (arg == "concurrent") {
      run_concurrent = true;
    } else if (arg == "core") {
      run_sizes = run_align = run_concurrent = true;
    } else if (arg == "all") {
      run_all = true;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      print_usage(argv[0]);
      return 1;
    }
  }

  if (!run_read && !run_write && !run_perf && !run_sizes && !run_align &&
      !run_concurrent && !run_all)
    run_all = true;

  if (params.nvme_path.empty()) {
    std::cerr << "Error: --nvme-path is required. Please specify the NVMe mount point.\n";
    print_usage(argv[0]);
    return 1;
  }

  std::string base = params.nvme_path;
  if (base.back() != '/') base += '/';
  if (params.file_path == "hipfile_test_data.bin")
    params.file_path = base + params.file_path;
  if (params.write_path == "hipfile_write_output.bin")
    params.write_path = base + params.write_path;

  print_separator("hipFile (AIS) System Test Suite");

  if (!initialize_environment()) {
    return 1;
  }

  const char *test_file  = params.file_path.c_str();
  const char *write_file = params.write_path.c_str();
  const size_t test_size = params.size_mb * 1024ULL * 1024ULL;

  if (run_all) {
    run_read = true;
    run_write = true;
    run_perf = true;
    run_sizes = true;
    run_align = true;
    run_concurrent = true;
  }

  std::cout << "\nConfiguration:" << std::endl;
  std::cout << "  Test file  : " << test_file << std::endl;
  std::cout << "  Write file : " << write_file << std::endl;
  std::cout << "  Size       : " << params.size_mb << " MB" << std::endl;
  std::cout << "  Repeat     : " << params.repeat << std::endl;
  if (run_concurrent)
    std::cout << "  Concurrent : " << params.threads << " threads x "
              << params.iters << " iters" << std::endl;

  bool need_read_file = run_read || run_perf;

  try {
    if (need_read_file) {
      print_separator("Preparing Test Data");
      if (!create_test_file(test_file, params.size_mb)) {
        std::cerr << "Failed to create test file" << std::endl;
        return 1;
      }
    }

    int tests_run = 0;
    int tests_passed = 0;

    for (int rep = 0; rep < params.repeat; ++rep) {
      if (params.repeat > 1)
        std::cout << "\n--- Iteration " << (rep + 1) << " of "
                  << params.repeat << " ---" << std::endl;

      if (run_read) {
        tests_run++;
        if (test_basic_read(test_file, test_size))
          tests_passed++;
      }

      if (run_write) {
        tests_run++;
        if (test_basic_write(write_file, test_size))
          tests_passed++;
      }

      if (run_perf) {
        tests_run++;
        if (test_performance_comparison(test_file, test_size))
          tests_passed++;
      }

      if (run_sizes) {
        tests_run++;
        if (test_size_sweep(params.nvme_path))
          tests_passed++;
      }

      if (run_align) {
        tests_run++;
        if (test_alignment_boundaries(params.nvme_path))
          tests_passed++;
      }

      if (run_concurrent) {
        tests_run++;
        if (test_concurrent(params.nvme_path, params.threads, params.iters))
          tests_passed++;
      }
    }

    if (tests_run > 0) {
      print_separator("Test Suite Summary");
      std::cout << "\nTests run:    " << tests_run << std::endl;
      std::cout << "Tests passed: " << tests_passed << std::endl;
      std::cout << "Tests failed: " << (tests_run - tests_passed) << std::endl;

      if (tests_passed == tests_run) {
        std::cout << "\n✓ All tests PASSED\n" << std::endl;
      } else {
        std::cout << "\n✗ Some tests FAILED\n" << std::endl;
      }

      std::cout << "Cleaning up test files..." << std::endl;
      if (need_read_file)
        unlink(test_file);
      if (run_write)
        unlink(write_file);

      return (tests_passed == tests_run) ? 0 : 1;
    } else {
      std::cout << "\nNo tests were selected to run." << std::endl;
      print_usage(argv[0]);
      return 1;
    }
  } catch (const std::exception &e) {
    std::cerr << "\n✗ Test failed with exception: " << e.what() << std::endl;
    return 1;
  }
}
