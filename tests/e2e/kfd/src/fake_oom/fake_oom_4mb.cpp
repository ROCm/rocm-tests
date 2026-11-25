/*
 * System test: cross-GPU dma-buf pin must not fake-OOM a later tiny alloc.
 *
 * GPU0 hipMalloc(PIN_GB) → GPU1 pin via pin_dmabuf.ko → GPU0 fill 2 GiB
 * until free < 2 GiB → hipMalloc(4 MiB).
 *
 * PASS  : 4 MiB admitted (no fake OOM)
 * FAIL  : 4 MiB refused (fake OOM / double-charge pin)
 */
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <vector>

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include "pin_dmabuf_kmod/pin_dmabuf.h"

#ifndef PCI_DEVFN
#define PCI_DEVFN(slot, func) ((((slot) & 0x1f) << 3) | ((func) & 0x07))
#endif

#define EXPORT_CHUNK (2147483648ULL)

static double gib(size_t n) { return (double)n / (1024.0 * 1024.0 * 1024.0); }

static void die(const char *m) {
  fprintf(stderr, "FATAL: %s\n", m);
  exit(2);
}

static void die_hip(hipError_t e, const char *m) {
  fprintf(stderr, "FATAL: %s: %s\n", m, hipGetErrorString(e));
  exit(2);
}

static size_t gpu_free(int gpu) {
  size_t f = 0, t = 0;
  hipError_t e = hipSetDevice(gpu);
  if (e != hipSuccess) {
    die_hip(e, "hipSetDevice");
  }
  e = hipMemGetInfo(&f, &t);
  if (e != hipSuccess) {
    die_hip(e, "hipMemGetInfo");
  }
  return f;
}

static int parse_pci(const char *s, unsigned *domain, unsigned *bus, unsigned *slot,
                     unsigned *func) {
  return sscanf(s, "%x:%x:%x.%x", domain, bus, slot, func) == 4 ? 0 : -1;
}

static void pin_fd_to_gpu(int kfd, int fd, int importer, std::vector<unsigned> *ids) {
  char pci[32];
  hipError_t herr = hipDeviceGetPCIBusId(pci, (int)sizeof(pci), importer);
  if (herr != hipSuccess) {
    die_hip(herr, "hipDeviceGetPCIBusId");
  }
  unsigned domain = 0, bus = 0, slot = 0, func = 0;
  if (parse_pci(pci, &domain, &bus, &slot, &func) != 0) {
    fprintf(stderr, "FATAL: parse PCI %s\n", pci);
    exit(2);
  }
  struct pin_dmabuf_arg req;
  memset(&req, 0, sizeof(req));
  req.fd = fd;
  req.flags = PIN_DMABUF_F_PCI;
  req.pci_domain = (uint16_t)domain;
  req.pci_bus = (uint8_t)bus;
  req.pci_devfn = (uint8_t)PCI_DEVFN(slot, func);
  if (ioctl(kfd, PIN_DMABUF_IOC_PIN, &req) != 0) {
    fprintf(stderr, "FATAL: PIN GPU%d (%s): %s\n", importer, pci, strerror(errno));
    exit(2);
  }
  ids->push_back(req.id);
}

static void stage(const char *name) { printf("\n[%s]\n", name); }

int main(int argc, char **argv) {
  int src = 0;
  int importer = 1;
  double pin_gib = 50.0;
  int opt;
  static struct option opts[] = {
      {"src-gpu", required_argument, NULL, 's'},
      {"import-gpu", required_argument, NULL, 'i'},
      {"pin-gb", required_argument, NULL, 'p'},
      {"help", no_argument, NULL, 'h'},
      {NULL, 0, NULL, 0},
  };
  while ((opt = getopt_long(argc, argv, "s:i:p:h", opts, NULL)) != -1) {
    switch (opt) {
    case 's':
      src = atoi(optarg);
      break;
    case 'i':
      importer = atoi(optarg);
      break;
    case 'p':
      pin_gib = atof(optarg);
      break;
    default:
      printf("Usage: %s [--src-gpu 0] [--import-gpu 1] [--pin-gb 50]\n", argv[0]);
      return opt == 'h' ? 0 : 2;
    }
  }

  if (hsa_init() != HSA_STATUS_SUCCESS) {
    die("hsa_init");
  }
  int ndev = 0;
  hipError_t herr = hipGetDeviceCount(&ndev);
  if (herr != hipSuccess) {
    die_hip(herr, "hipGetDeviceCount");
  }
  if (src < 0 || src >= ndev || importer < 0 || importer >= ndev || src == importer) {
    die("need two distinct HIP GPUs (src != import)");
  }

  int kfd = open("/dev/pin_dmabuf", O_RDWR);
  if (kfd < 0) {
    die("open /dev/pin_dmabuf (insmod pin_dmabuf.ko first)");
  }

  const size_t pin_bytes = (size_t)(pin_gib * 1024.0 * 1024.0 * 1024.0);
  const size_t fill_sz = 2ull << 30;
  const size_t tiny = 4ull << 20;
  size_t idle = 0;
  size_t total = 0;
  (void)hipSetDevice(src);
  (void)hipMemGetInfo(&idle, &total);

  char src_pci[32] = {};
  char imp_pci[32] = {};
  (void)hipDeviceGetPCIBusId(src_pci, (int)sizeof(src_pci), src);
  (void)hipDeviceGetPCIBusId(imp_pci, (int)sizeof(imp_pci), importer);

  printf("========================================================================\n");
  printf("TEST     test_cross_gpu_dmabuf_pin_probe_admitted\n");
  printf("SCENARIO GPU%d alloc + pin via GPU%d import, fill, then 4 MiB probe\n", src,
         importer);
  printf("------------------------------------------------------------------------\n");
  printf("  src GPU               %d  (%s)\n", src, src_pci);
  printf("  import GPU            %d  (%s)\n", importer, imp_pci);
  printf("  pin size              %.0f GiB\n", pin_gib);
  printf("  idle free / total     %.2f / %.2f GiB\n", gib(idle), gib(total));
  printf("------------------------------------------------------------------------\n");

  stage("allocate");
  herr = hipSetDevice(src);
  if (herr != hipSuccess) {
    die_hip(herr, "hipSetDevice src");
  }
  void *pinned = NULL;
  herr = hipMalloc(&pinned, pin_bytes);
  if (herr != hipSuccess) {
    die_hip(herr, "hipMalloc pin buffer");
  }
  size_t after_alloc = gpu_free(src);
  printf("  hipMalloc             %.2f GiB on GPU%d\n", gib(pin_bytes), src);
  printf("  HIP free              %.2f GiB\n", gib(after_alloc));

  void *base = NULL;
  size_t asz = 0;
  herr = hipMemGetAddressRange(&base, &asz, pinned);
  if (herr != hipSuccess) {
    die_hip(herr, "hipMemGetAddressRange");
  }

  stage("pin");
  std::vector<unsigned> ids;
  size_t exported = 0;
  int nfd = 0;
  while (exported < asz) {
    size_t chunk = asz - exported;
    if (chunk > EXPORT_CHUNK) {
      chunk = EXPORT_CHUNK;
    }
    int fd = -1;
    uint64_t off = 0;
    hsa_status_t st =
        hsa_amd_portable_export_dmabuf((char *)base + exported, chunk, &fd, &off);
    if (st != HSA_STATUS_SUCCESS) {
      fprintf(stderr, "FATAL: export_dmabuf %d at +%zu\n", (int)st, exported);
      exit(2);
    }
    pin_fd_to_gpu(kfd, fd, importer, &ids);
    close(fd);
    exported += chunk;
    nfd++;
  }
  size_t after_pin = gpu_free(src);
  printf("  dmabuf chunks         %d pinned on GPU%d\n", nfd, importer);
  printf("  HIP free              %.2f GiB\n", gib(after_pin));
  printf("  delta vs after alloc  %.2f GiB\n", gib(after_alloc) - gib(after_pin));

  stage("fill");
  size_t est_phys = after_alloc;
  std::vector<void *> fillers;
  for (;;) {
    size_t now = gpu_free(src);
    if (now < fill_sz) {
      printf("  stop                  HIP free %.2f GiB < 2 GiB\n", gib(now));
      printf("  fillers               %zu x 2 GiB\n", fillers.size());
      printf("  est_phys              %.2f GiB\n", gib(est_phys));
      break;
    }
    void *p = NULL;
    herr = hipMalloc(&p, fill_sz);
    now = gpu_free(src);
    if (herr != hipSuccess) {
      printf("  hipMalloc(2 GiB)      FAIL after %zu fillers (%s)\n", fillers.size(),
             hipGetErrorString(herr));
      printf("  HIP free              %.2f GiB\n", gib(now));
      printf("  est_phys              %.2f GiB\n", gib(est_phys));
      break;
    }
    fillers.push_back(p);
    est_phys -= fill_sz;
    if (fillers.size() % 8 == 0 || gib(now) < pin_gib + 2.0) {
      printf("  progress              fillers=%zu  HIP free=%.2f GiB  est_phys=%.2f GiB\n",
             fillers.size(), gib(now), gib(est_phys));
    }
  }

  stage("probe");
  size_t before_4m = gpu_free(src);
  void *tiny_p = NULL;
  herr = hipMalloc(&tiny_p, tiny);
  size_t after_4m = gpu_free(src);
  const bool admitted = (herr == hipSuccess);
  if (admitted) {
    printf("  hipMalloc(4 MiB)      admitted\n");
    printf("  HIP free              %.2f -> %.2f GiB\n", gib(before_4m), gib(after_4m));
    (void)hipFree(tiny_p);
  } else {
    printf("  hipMalloc(4 MiB)      refused (%s)\n", hipGetErrorString(herr));
    printf("  HIP free              %.2f GiB\n", gib(before_4m));
    printf("  est_phys              %.2f GiB\n", gib(est_phys));
  }

  for (unsigned id : ids) {
    struct pin_dmabuf_arg req;
    memset(&req, 0, sizeof(req));
    req.id = id;
    (void)ioctl(kfd, PIN_DMABUF_IOC_UNPIN, &req);
  }
  for (void *p : fillers) {
    (void)hipFree(p);
  }
  (void)hipSetDevice(src);
  (void)hipFree(pinned);
  close(kfd);

  printf("------------------------------------------------------------------------\n");
  if (admitted) {
    printf("  Result                 PASS\n");
    printf("                         verdict : no_fake_oom\n");
    printf("                         while pinned : admitted\n");
    printf("                         probe allocated; pin was not charged twice\n");
  } else {
    printf("  Result                 FAIL\n");
    printf("                         verdict : fake_oom\n");
    printf("                         while pinned : refused\n");
    printf("                         4 MiB refused after cross-GPU dma-buf pin\n");
  }
  printf("========================================================================\n");
  return admitted ? 0 : 1;
}
