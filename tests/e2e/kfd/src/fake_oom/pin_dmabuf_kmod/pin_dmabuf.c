/* SPDX-License-Identifier: GPL-2.0 */
/*
 * pin_dmabuf: ioctl wrapper that matches RDMA's dma-buf pin sequence:
 *   dma_buf_get → dynamic_attach(allow_peer2peer) → dma_buf_pin → map
 *
 * hipMalloc → hsa_amd_portable_export_dmabuf(fd) → PIN ioctl.
 * No InfiniBand. A plain dma_buf_attach() to the exporter pins GTT (or
 * skips VRAM), which does not raise adev->vram_pin_size and misses the
 * KFD double-count bug.
 */
#include <linux/dma-buf.h>
#include <linux/dma-resv.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/version.h>
#include <drm/drm_device.h>
#include <drm/drm_gem.h>

#include "pin_dmabuf.h"

#define PIN_DMABUF_MAX 512

struct pin_entry {
	struct dma_buf *dmabuf;
	struct dma_buf_attachment *attach;
	struct sg_table *sgt;
	struct pci_dev *held_pdev;
	bool used;
};

static DEFINE_MUTEX(pin_lock);
static struct pin_entry table[PIN_DMABUF_MAX];

/* dma_buf_dynamic_attach() rejects importer_ops without move_notify.
 * We pin, so the buffer cannot move while attached; nothing to do here.
 */
static void pin_move_notify(struct dma_buf_attachment *attach)
{
	pr_debug("pin_dmabuf: move_notify on pinned attachment\n");
}

static const struct dma_buf_attach_ops pin_importer_ops = {
	.allow_peer2peer = true,
	.move_notify = pin_move_notify,
};

static struct device *exporter_dev(struct dma_buf *dmabuf)
{
	struct drm_gem_object *obj;

	if (!dmabuf || !dmabuf->priv)
		return NULL;
	obj = dmabuf->priv;
	if (!obj->dev || !obj->dev->dev)
		return NULL;
	return obj->dev->dev;
}

/* Prefer another AMD GPU so amdgpu_dma_buf_pin keeps VRAM (P2P), like a NIC. */
static struct device *pick_attach_dev(struct device *exporter,
				      struct pci_dev **held_out)
{
	struct pci_dev *pdev = NULL;

	*held_out = NULL;
	while ((pdev = pci_get_device(PCI_VENDOR_ID_ATI, PCI_ANY_ID, pdev))) {
		unsigned int base = pdev->class >> 16;

		if (&pdev->dev == exporter)
			continue;
		if (base != 0x03 && base != 0x12)
			continue;
		*held_out = pdev; /* reference from pci_get_device */
		return &pdev->dev;
	}
	return exporter;
}

static struct device *lookup_pci_dev(u16 domain, u8 bus, u8 devfn,
				     struct pci_dev **held_out)
{
	struct pci_dev *pdev;

	*held_out = NULL;
	pdev = pci_get_domain_bus_and_slot(domain, bus, devfn);
	if (!pdev)
		return ERR_PTR(-ENODEV);
	*held_out = pdev;
	return &pdev->dev;
}

static void drop_attach(struct pin_entry *e)
{
	if (!e->attach || !e->dmabuf)
		return;
	dma_resv_lock(e->dmabuf->resv, NULL);
	if (e->sgt) {
		dma_buf_unmap_attachment(e->attach, e->sgt, DMA_BIDIRECTIONAL);
		e->sgt = NULL;
	}
	dma_buf_unpin(e->attach);
	dma_resv_unlock(e->dmabuf->resv);
	dma_buf_detach(e->dmabuf, e->attach);
	dma_buf_put(e->dmabuf);
	if (e->held_pdev)
		pci_dev_put(e->held_pdev);
	e->attach = NULL;
	e->dmabuf = NULL;
	e->held_pdev = NULL;
}

static int pin_one(const struct pin_dmabuf_arg *req, u32 *id_out)
{
	struct dma_buf *dmabuf;
	struct dma_buf_attachment *attach;
	struct sg_table *sgt;
	struct device *exp, *dev;
	struct pci_dev *held = NULL;
	int i, ret = 0;

	dmabuf = dma_buf_get(req->fd);
	if (IS_ERR(dmabuf))
		return PTR_ERR(dmabuf);

	exp = exporter_dev(dmabuf);
	if (!exp) {
		dma_buf_put(dmabuf);
		return -EINVAL;
	}
	if (req->flags & PIN_DMABUF_F_PCI) {
		dev = lookup_pci_dev(req->pci_domain, req->pci_bus,
				     req->pci_devfn, &held);
		if (IS_ERR(dev)) {
			dma_buf_put(dmabuf);
			return PTR_ERR(dev);
		}
	} else {
		dev = pick_attach_dev(exp, &held);
	}

	attach = dma_buf_dynamic_attach(dmabuf, dev, &pin_importer_ops, NULL);
	if (IS_ERR(attach)) {
		ret = PTR_ERR(attach);
		if (held)
			pci_dev_put(held);
		dma_buf_put(dmabuf);
		return ret;
	}

	dma_resv_lock(dmabuf->resv, NULL);
	ret = dma_buf_pin(attach);
	if (ret) {
		dma_resv_unlock(dmabuf->resv);
		pr_err("pin_dmabuf: dma_buf_pin failed %d (peer2peer=%d)\n",
		       ret, attach->peer2peer);
		dma_buf_detach(dmabuf, attach);
		if (held)
			pci_dev_put(held);
		dma_buf_put(dmabuf);
		return ret;
	}
	sgt = dma_buf_map_attachment(attach, DMA_BIDIRECTIONAL);
	dma_resv_unlock(dmabuf->resv);
	if (IS_ERR(sgt)) {
		ret = PTR_ERR(sgt);
		pr_err("pin_dmabuf: map_attachment failed %d (peer2peer=%d)\n",
		       ret, attach->peer2peer);
		dma_resv_lock(dmabuf->resv, NULL);
		dma_buf_unpin(attach);
		dma_resv_unlock(dmabuf->resv);
		dma_buf_detach(dmabuf, attach);
		if (held)
			pci_dev_put(held);
		dma_buf_put(dmabuf);
		return ret;
	}

	if (req->flags & PIN_DMABUF_F_PCI)
		pr_info("pin_dmabuf: pin ok attach_dev=%s peer2peer=%d\n",
			dev_name(dev), attach->peer2peer);
	else
		pr_info_once("pin_dmabuf: pin ok attach_dev=%s peer2peer=%d\n",
			     dev_name(dev), attach->peer2peer);

	mutex_lock(&pin_lock);
	for (i = 0; i < PIN_DMABUF_MAX; i++) {
		if (!table[i].used)
			break;
	}
	if (i == PIN_DMABUF_MAX) {
		struct pin_entry tmp = {
			.dmabuf = dmabuf,
			.attach = attach,
			.sgt = sgt,
			.held_pdev = held,
		};

		mutex_unlock(&pin_lock);
		drop_attach(&tmp);
		return -ENOSPC;
	}
	table[i].dmabuf = dmabuf;
	table[i].attach = attach;
	table[i].sgt = sgt;
	table[i].held_pdev = held;
	table[i].used = true;
	mutex_unlock(&pin_lock);
	*id_out = (u32)i;
	return 0;
}

static int unpin_one(u32 id)
{
	struct pin_entry e;

	if (id >= PIN_DMABUF_MAX)
		return -EINVAL;
	mutex_lock(&pin_lock);
	if (!table[id].used) {
		mutex_unlock(&pin_lock);
		return -ENOENT;
	}
	e = table[id];
	memset(&table[id], 0, sizeof(table[id]));
	mutex_unlock(&pin_lock);
	drop_attach(&e);
	return 0;
}

static long pin_dmabuf_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
	struct pin_dmabuf_arg req;
	int ret;

	switch (cmd) {
	case PIN_DMABUF_IOC_PIN:
		if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
			return -EFAULT;
		ret = pin_one(&req, &req.id);
		if (ret)
			return ret;
		if (copy_to_user((void __user *)arg, &req, sizeof(req))) {
			unpin_one(req.id);
			return -EFAULT;
		}
		return 0;
	case PIN_DMABUF_IOC_UNPIN:
		if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
			return -EFAULT;
		return unpin_one(req.id);
	default:
		return -ENOTTY;
	}
}

static int pin_dmabuf_release(struct inode *inode, struct file *file)
{
	int i;

	mutex_lock(&pin_lock);
	for (i = 0; i < PIN_DMABUF_MAX; i++) {
		if (!table[i].used)
			continue;
		drop_attach(&table[i]);
		memset(&table[i], 0, sizeof(table[i]));
	}
	mutex_unlock(&pin_lock);
	return 0;
}

static const struct file_operations pin_dmabuf_fops = {
	.owner = THIS_MODULE,
	.unlocked_ioctl = pin_dmabuf_ioctl,
	.compat_ioctl = pin_dmabuf_ioctl,
	.release = pin_dmabuf_release,
};

static struct miscdevice pin_dmabuf_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "pin_dmabuf",
	.fops = &pin_dmabuf_fops,
	.mode = 0666,
};

static int __init pin_dmabuf_init(void)
{
	return misc_register(&pin_dmabuf_misc);
}

static void __exit pin_dmabuf_exit(void)
{
	misc_deregister(&pin_dmabuf_misc);
}

module_init(pin_dmabuf_init);
module_exit(pin_dmabuf_exit);

MODULE_LICENSE("GPL");
/* 6.8 stringifies the arg; 6.13+ expects a string literal. */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 13, 0)
MODULE_IMPORT_NS("DMA_BUF");
#else
MODULE_IMPORT_NS(DMA_BUF);
#endif
MODULE_AUTHOR("AMD KFD VRAM accounting repro");
MODULE_DESCRIPTION("Pin GEM dma-buf via dma_buf_pin without RDMA");
