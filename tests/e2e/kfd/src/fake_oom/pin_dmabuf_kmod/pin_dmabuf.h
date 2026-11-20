/* uapi shared by pin_dmabuf.ko and userspace */
#ifndef PIN_DMABUF_H
#define PIN_DMABUF_H

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/ioctl.h>
#else
#include <stdint.h>
#include <sys/ioctl.h>
typedef int32_t __s32;
typedef uint32_t __u32;
typedef uint16_t __u16;
typedef uint8_t __u8;
#endif

#define PIN_DMABUF_F_PCI 1u /* pci_* selects importer device */

struct pin_dmabuf_arg {
	__s32 fd; /* in: dma-buf fd from hsa_amd_portable_export_dmabuf */
	__u32 id; /* out: cookie for UNPIN */
	__u32 flags;
	__u16 pci_domain;
	__u8 pci_bus;
	__u8 pci_devfn; /* PCI_DEVFN(slot, func) */
};

#define PIN_DMABUF_IOC_MAGIC 0xB1
#define PIN_DMABUF_IOC_PIN _IOWR(PIN_DMABUF_IOC_MAGIC, 1, struct pin_dmabuf_arg)
#define PIN_DMABUF_IOC_UNPIN _IOW(PIN_DMABUF_IOC_MAGIC, 2, struct pin_dmabuf_arg)

#endif
