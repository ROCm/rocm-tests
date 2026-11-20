/*
 * System test helper: one dmabuf imported/pinned once per peer GPU.
 * Prints whether HIP free on the src GPU drops beyond the single alloc.
 */
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include "pin_dmabuf_kmod/pin_dmabuf.h"

#ifndef PCI_DEVFN
#define PCI_DEVFN(slot, func) ((((slot) & 0x1f) << 3) | ((func) & 0x07))
#endif

#define MAX_PIN 16

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

static void stage(const char *name) { printf("\n[%s]\n", name); }

int main(int argc, char **argv) {
  int src = 0;
  double size_gib = 2.0;
  int opt;
  static struct option opts[] = {
      {"src-gpu", required_argument, NULL, 's'},
      {"size-gb", required_argument, NULL, 'z'},
      {"help", no_argument, NULL, 'h'},
      {NULL, 0, NULL, 0},
  };
  while ((opt = getopt_long(argc, argv, "s:z:h", opts, NULL)) != -1) {
    switch (opt) {
    case 's':
      src = atoi(optarg);
      break;
    case 'z':
      size_gib = atof(optarg);
      break;
    default:
      printf("Usage: %s [--src-gpu 0] [--size-gb 2]\n", argv[0]);
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
  if (src < 0 || src >= ndev) {
    die("src-gpu out of range");
  }
  if (ndev < 2) {
    die("need at least two HIP GPUs");
  }

  int kfd = open("/dev/pin_dmabuf", O_RDWR);
  if (kfd < 0) {
    die("open /dev/pin_dmabuf (insmod pin_dmabuf.ko first)");
  }

  size_t bytes = (size_t)(size_gib * 1024.0 * 1024.0 * 1024.0);
  size_t idle = gpu_free(src);
  char src_pci[32] = {};
  (void)hipDeviceGetPCIBusId(src_pci, (int)sizeof(src_pci), src);

  printf("========================================================================\n");
  printf("TEST     test_dmabuf_multi_gpu_import_count\n");
  printf("SCENARIO alloc on GPU%d, pin/import once on every peer GPU\n", src);
  printf("------------------------------------------------------------------------\n");
  printf("  src GPU               %d  (%s)\n", src, src_pci);
  printf("  HIP devices           %d\n", ndev);
  printf("  buffer size           %.2f GiB\n", size_gib);
  printf("  idle free             %.2f GiB\n", gib(idle));
  printf("------------------------------------------------------------------------\n");

  stage("allocate");
  herr = hipSetDevice(src);
  if (herr != hipSuccess) {
    die_hip(herr, "hipSetDevice src");
  }
  void *ptr = NULL;
  herr = hipMalloc(&ptr, bytes);
  if (herr != hipSuccess) {
    die_hip(herr, "hipMalloc");
  }
  size_t after_alloc = gpu_free(src);
  printf("  hipMalloc             %.2f GiB on GPU%d\n", size_gib, src);
  printf("  HIP free              %.2f GiB  (delta %.2f GiB)\n", gib(after_alloc),
         gib(idle) - gib(after_alloc));

  void *base = NULL;
  size_t asz = 0;
  herr = hipMemGetAddressRange(&base, &asz, ptr);
  if (herr != hipSuccess) {
    die_hip(herr, "hipMemGetAddressRange");
  }
  int fd = -1;
  uint64_t off = 0;
  hsa_status_t st = hsa_amd_portable_export_dmabuf(base, asz, &fd, &off);
  if (st != HSA_STATUS_SUCCESS) {
    fprintf(stderr, "FATAL: export_dmabuf %d\n", (int)st);
    exit(2);
  }

  stage("import_pin");
  unsigned ids[MAX_PIN];
  int nids = 0;
  int nimp = 0;
  for (int g = 0; g < ndev; g++) {
    if (g == src) {
      continue;
    }
    char pci[32];
    herr = hipDeviceGetPCIBusId(pci, (int)sizeof(pci), g);
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
      fprintf(stderr, "FATAL: PIN GPU%d (%s): %s\n", g, pci, strerror(errno));
      close(fd);
      exit(2);
    }
    ids[nids++] = req.id;
    nimp++;
    size_t now = gpu_free(src);
    printf("  import #%d             GPU%d (%s)  HIP free=%.2f GiB  "
           "delta_vs_alloc=%.2f GiB\n",
           nimp, g, pci, gib(now), gib(after_alloc) - gib(now));
    if (nids >= MAX_PIN) {
      die("too many importers");
    }
  }
  close(fd);

  size_t final_free = gpu_free(src);
  double charged = gib(after_alloc) - gib(final_free);
  printf("  imports               %d of one %.2f GiB dmabuf\n", nimp, size_gib);
  printf("  HIP free              %.2f GiB\n", gib(final_free));
  printf("  charged vs alloc      %.2f GiB\n", charged);

  for (int i = 0; i < nids; i++) {
    struct pin_dmabuf_arg req;
    memset(&req, 0, sizeof(req));
    req.id = ids[i];
    (void)ioctl(kfd, PIN_DMABUF_IOC_UNPIN, &req);
  }
  size_t after_unpin = gpu_free(src);
  printf("  after unpin           HIP free=%.2f GiB\n", gib(after_unpin));
  (void)hipSetDevice(src);
  (void)hipFree(ptr);
  close(kfd);

  // Informational: pin should not multiply-charge HIP free by importer count.
  const bool ok = charged < (size_gib * 0.25);
  printf("------------------------------------------------------------------------\n");
  if (ok) {
    printf("  Result                 PASS\n");
    printf("                         verdict : pin_not_multi_charged\n");
    printf("                         HIP free drop after peer pins stayed small\n");
  } else {
    printf("  Result                 FAIL\n");
    printf("                         verdict : unexpected_hip_free_drop\n");
    printf("                         HIP free dropped %.2f GiB after peer pins\n", charged);
  }
  printf("========================================================================\n");
  return ok ? 0 : 1;
}
