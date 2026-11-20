# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Check that an RDMA dma-buf pin of owned VRAM is not charged twice.

Pin the working set, then probe with hipMalloc. Pass when the probe is admitted
while a free buddy block still fits it. Refusal with that block free is fake_oom.

Patterns: pin-all (every chunk), offset (2 MiB subrange), churn (re-register,
then probe again after release). --foreign-probe repeats the probe in a second
process while the pins are held.
"""

import argparse
import contextlib
import ctypes
import ctypes.util
import json
import os
import re
import subprocess
import sys
import time

GIB = 1 << 30
MIB = 1 << 20
UNIT_BYTES = {"KiB": 1 << 10, "MiB": MIB, "GiB": GIB}

HIP_SUCCESS = 0
HIP_ERROR_OUT_OF_MEMORY = 2
HSA_STATUS_SUCCESS = 0
HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 0x1

IBV_ACCESS_LOCAL_WRITE = 1
IBV_ACCESS_REMOTE_WRITE = 2
IBV_ACCESS_REMOTE_READ = 4

# struct ibv_mr { context; pd; addr; length; handle; lkey; rkey; }
IBV_MR_LKEY_OFFSET = 36
IBV_MR_RKEY_OFFSET = 40

# KFD charges VRAM in 2 MiB quanta.  A caching allocator's registered slice
# starts on that boundary, not at the hipMalloc base.
SUBRANGE_OFFSET = 2 * MIB

# Share of the memory that is free right now kept back from the working set.
# On a 192 GiB GPU this is about 12 GiB, which still left a buddy block larger
# than the probe. Reserving only the probe did not.
FREE_RESERVE_DIVISOR = 16
MIN_RESERVE_PROBES = 2

VERDICT_SIGNAL = {
    "admitted": "probe allocated; the pin was not charged twice",
    "fake_oom": "refused while a buddy block still fits the request",
    "genuine_oom": "refused because physical VRAM is short of the request",
    "fragmented": "physical free fits, but no single buddy block does",
    "refused_unverified": "refused; debugfs buddy dump was not readable, so this is not proof of the pin bug",
    "pin_leak": "still refused after every memory region was deregistered",
}

# test name, then the scenario that name denotes. Pass means the probe is admitted.
SCENARIOS = {
    "pin-all": (
        "test_kfd_dmabuf_pin_full_working_set",
        "Allocate the working set from the memory free at start and register every chunk as an RDMA dma-buf "
        "memory region, then allocate the probe. Pass when the probe is admitted "
        "while a free buddy block still fits it.",
    ),
    "offset": (
        "test_kfd_dmabuf_pin_subrange",
        "Register a 2 MiB subrange of each chunk through the HSA dma-buf export, "
        "then allocate the probe. Pass when the probe is admitted.",
    ),
    "churn": (
        "test_kfd_dmabuf_pin_churn",
        "Register and release the working set for the configured number of cycles, "
        "probe while it is pinned, then probe again after the last release. "
        "Pass when both probes are admitted.",
    ),
}
CONTROL_SCENARIO = (
    "test_kfd_vram_alloc_without_pin",
    "Allocate the same working set and probe it with no RDMA registration. "
    "Pass when the probe is admitted. Control for the pin scenarios.",
)

RULE = "=" * 72
THIN = "-" * 72


def memory_text(fields):
    parts = []
    if "free_gib" in fields:
        parts.append(f"HIP free {fields['free_gib']:.3f} GiB")
    if "phys_free_gib" in fields:
        parts.append(f"phys free {fields['phys_free_gib']:.3f} GiB")
    elif "vram_mm_error" in fields:
        parts.append("phys free n/a")
    if "largest_free_block_gib" in fields:
        parts.append(f"largest block {fields['largest_free_block_gib']:.3f} GiB")
    gap = fields.get("accounting_gap_gib")
    if gap is not None:
        parts.append(f"gap {gap:+.3f} GiB")
    return "  |  ".join(parts)


def gib_text(num_bytes):
    return f"{num_bytes / GIB:.3f} GiB"


def line(label, detail):
    print(f"  {label:<22} {detail}", flush=True)


def wrapped(label, text, width=72):
    prefix = f"{label:<9}: "
    indent = " " * len(prefix)
    words = text.split()
    lines = []
    current = ""
    for word in words:
        piece = word if not current else f"{current} {word}"
        if len(prefix) + len(piece) <= width or not current:
            current = piece
        else:
            lines.append(prefix + current)
            prefix = indent
            current = word
    if current:
        lines.append(prefix + current)
    print("\n".join(lines), flush=True)


def report(stage, **fields):
    if stage == "start":
        print(RULE, flush=True)
        wrapped("Test", fields["test_name"])
        wrapped("Scenario", fields["scenario"])
        print(
            f"GPU      : {fields['device']}  {fields['device_name']}",
            flush=True,
        )
        print(
            f"PCI      : {fields['pci_address']}    " f"IB {fields.get('ib_device') or 'none'}",
            flush=True,
        )
        print(
            f"Target   : {fields['target_gib']} GiB ({fields['target_source']})    "
            f"chunk {fields['chunk_bytes'] // MIB} MiB    "
            f"probe {fields['probe_bytes'] // MIB} MiB",
            flush=True,
        )
        print(
            f"Pattern  : {fields['pattern']}    export {fields['export_api']}",
            flush=True,
        )
        print(THIN, flush=True)
        line("Memory", memory_text(fields))
        return

    if stage == "verdict":
        print(THIN, flush=True)
        line("Result", "PASS" if fields["passed"] else "FAIL")
        if fields.get("while_pinned"):
            line("", f"while pinned : {fields['while_pinned']}")
            signal = fields.get("while_pinned_signal")
            if signal:
                line("", signal)
        if fields.get("after_unpin"):
            line("", f"after release : {fields['after_unpin']}")
        if fields.get("foreign"):
            line("", f"foreign probe : {fields['foreign']}")
        if fields.get("expect_failed"):
            line("", f"expected {fields['expected']}, observed {fields['actual']}")
        print(RULE, flush=True)
        return

    detail = {
        "allocated": lambda: f"{fields['chunks']} chunks    {memory_text(fields)}",
        "window": lambda: (
            f"{fields['ranges']} ranges    pinned {gib_text(fields['pinned_bytes'])}    " f"{memory_text(fields)}"
        ),
        "window_empty": lambda: fields.get("note", ""),
        "pin_skipped": lambda: f"registration skipped    {memory_text(fields)}",
        "pinned": lambda: (
            f"{fields['chunks']} regions    pinned {gib_text(fields['pinned_bytes'])}    "
            f"lkey {fields['first_mr_lkey']}    {memory_text(fields)}"
        ),
        "churn_unpinned": lambda: (
            f"cycle {fields['cycle']} released    "
            f"errors {fields['deregistration_errors'] or 'none'}    {memory_text(fields)}"
        ),
        "churn_pinned": lambda: (f"cycle {fields['cycle']}    {fields['ranges']} regions    {memory_text(fields)}"),
        "holding": lambda: f"{fields['seconds']} s",
        "unpinned": lambda: (f"errors {fields['deregistration_errors'] or 'none'}    {memory_text(fields)}"),
        "probe_succeeded": lambda: f"allocated {fields['address']}    {memory_text(fields)}",
        "probe_failed": lambda: f"refused    {fields['error']}    {memory_text(fields)}",
        "probe_after_unpin_succeeded": lambda: (f"allocated {fields['address']}    {memory_text(fields)}"),
        "probe_after_unpin_failed": lambda: (f"refused    {fields['error']}    {memory_text(fields)}"),
        "cleanup": lambda: (
            f"deregistered {fields['deregistered']}    "
            f"freed {fields['freed']}    "
            f"errors {fields['deregistration_errors'] or 'none'}"
        ),
    }.get(stage)
    if stage == "foreign_probe":
        child = fields.get("child") or {}
        if not child:
            detail_text = f"no result    {fields.get('stderr') or ''}".rstrip()
        else:
            outcome = "allocated" if child.get("stage") == "probe_succeeded" else "refused"
            where = child.get("address") or child.get("error") or ""
            detail_text = f"{outcome}    {where}    {memory_text(child)}".rstrip()
        line("Foreign probe", detail_text)
        return
    labels = {
        "allocated": "Working set",
        "window": "Registration",
        "window_empty": "Registration",
        "pin_skipped": "Pin",
        "pinned": "Pinned",
        "churn_unpinned": "Churn",
        "churn_pinned": "Churn",
        "holding": "Hold",
        "unpinned": "Released",
        "probe_succeeded": "Probe",
        "probe_failed": "Probe",
        "probe_after_unpin_succeeded": "Probe after release",
        "probe_after_unpin_failed": "Probe after release",
        "cleanup": "Cleanup",
    }
    line(labels.get(stage, stage), detail() if detail else "")


class Hip:
    def __init__(self):
        self.lib = ctypes.CDLL("libamdhip64.so")
        self.lib.hipGetErrorString.restype = ctypes.c_char_p
        self.lib.hipMemGetHandleForAddressRange.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_ulonglong,
        ]

    def check(self, status, what):
        if status != HIP_SUCCESS:
            message = self.lib.hipGetErrorString(status).decode()
            raise RuntimeError(f"{what} failed: {message} ({status})")

    def set_device(self, device):
        self.check(self.lib.hipSetDevice(ctypes.c_int(device)), "hipSetDevice")

    def device_count(self):
        count = ctypes.c_int()
        self.check(self.lib.hipGetDeviceCount(ctypes.byref(count)), "hipGetDeviceCount")
        return count.value

    def device_name(self, device):
        name = ctypes.create_string_buffer(256)
        self.check(
            self.lib.hipDeviceGetName(name, ctypes.c_int(len(name)), ctypes.c_int(device)),
            "hipDeviceGetName",
        )
        return name.value.decode()

    def pci_address(self, device):
        buf = ctypes.create_string_buffer(32)
        self.check(
            self.lib.hipDeviceGetPCIBusId(buf, ctypes.c_int(len(buf)), ctypes.c_int(device)),
            "hipDeviceGetPCIBusId",
        )
        return buf.value.decode().lower()

    def mem_get_info(self):
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.check(self.lib.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total)), "hipMemGetInfo")
        return free.value, total.value

    def malloc(self, size):
        """Return (pointer, status) so the caller can treat OOM as a result, not an error."""
        pointer = ctypes.c_void_p()
        status = self.lib.hipMalloc(ctypes.byref(pointer), ctypes.c_size_t(size))
        if status == HIP_SUCCESS:
            return pointer.value, status
        if status == HIP_ERROR_OUT_OF_MEMORY:
            return None, status
        self.check(status, "hipMalloc")

    def free(self, pointer):
        self.check(self.lib.hipFree(ctypes.c_void_p(pointer)), "hipFree")

    def export_dmabuf(self, pointer, size):
        fd = ctypes.c_int()
        self.check(
            self.lib.hipMemGetHandleForAddressRange(
                ctypes.byref(fd),
                ctypes.c_void_p(pointer),
                ctypes.c_size_t(size),
                HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                0,
            ),
            "hipMemGetHandleForAddressRange",
        )
        return fd.value, 0


class Hsa:
    """Alternative dma-buf export.

    ``hsa_amd_portable_export_dmabuf`` also returns the offset of the range
    inside the exported buffer object, so it works for pointers that are not at
    the start of their allocation.  HIP has already initialised the runtime by
    the time this is used, and dlopen returns that same loaded library.
    """

    def __init__(self):
        self.lib = ctypes.CDLL("libhsa-runtime64.so.1")
        self.lib.hsa_amd_portable_export_dmabuf.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.lib.hsa_amd_portable_export_dmabuf.restype = ctypes.c_int

    def export_dmabuf(self, pointer, size):
        fd = ctypes.c_int()
        offset = ctypes.c_uint64()
        status = self.lib.hsa_amd_portable_export_dmabuf(
            ctypes.c_void_p(pointer),
            ctypes.c_size_t(size),
            ctypes.byref(fd),
            ctypes.byref(offset),
        )
        if status != HSA_STATUS_SUCCESS:
            raise RuntimeError(f"hsa_amd_portable_export_dmabuf failed: status={status:#x}")
        return fd.value, offset.value


def list_ib_devices():
    """Return RDMA device names visible to libibverbs (empty if none)."""
    try:
        lib = ctypes.CDLL("libibverbs.so.1", use_errno=True)
    except OSError as error:
        raise RuntimeError("libibverbs.so.1 not found; install rdma-core / libibverbs") from error
    lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
    lib.ibv_get_device_name.restype = ctypes.c_char_p
    lib.ibv_get_device_name.argtypes = [ctypes.c_void_p]
    lib.ibv_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    count = ctypes.c_int()
    devices = lib.ibv_get_device_list(ctypes.byref(count))
    if not devices or count.value <= 0:
        return []
    try:
        return [lib.ibv_get_device_name(devices[i]).decode() for i in range(count.value)]
    finally:
        lib.ibv_free_device_list(devices)


class Verbs:
    def __init__(self, requested_name):
        # use_errno so failed ibv_* calls expose the real errno to OSError.
        self.lib = ctypes.CDLL("libibverbs.so.1", use_errno=True)
        if not hasattr(self.lib, "ibv_reg_dmabuf_mr"):
            raise RuntimeError("libibverbs lacks ibv_reg_dmabuf_mr; need a newer rdma-core (dmabuf MR support)")
        self.lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
        self.lib.ibv_get_device_name.restype = ctypes.c_char_p
        self.lib.ibv_get_device_name.argtypes = [ctypes.c_void_p]
        self.lib.ibv_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.ibv_open_device.restype = ctypes.c_void_p
        self.lib.ibv_open_device.argtypes = [ctypes.c_void_p]
        self.lib.ibv_alloc_pd.restype = ctypes.c_void_p
        self.lib.ibv_alloc_pd.argtypes = [ctypes.c_void_p]
        self.lib.ibv_reg_dmabuf_mr.restype = ctypes.c_void_p
        self.lib.ibv_reg_dmabuf_mr.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.ibv_dereg_mr.argtypes = [ctypes.c_void_p]
        self.lib.ibv_dereg_mr.restype = ctypes.c_int

        count = ctypes.c_int()
        devices = self.lib.ibv_get_device_list(ctypes.byref(count))
        if not devices or count.value == 0:
            raise RuntimeError(
                "no RDMA devices found via libibverbs "
                "(check /sys/class/infiniband and provider plugins under /etc/libibverbs.d)"
            )
        try:
            self.available = [self.lib.ibv_get_device_name(devices[i]).decode() for i in range(count.value)]
            chosen = None
            for index, name in enumerate(self.available):
                if name == requested_name:
                    chosen = index
                    break
            if chosen is None:
                raise RuntimeError(f"RDMA device {requested_name!r} not found; available: {self.available}")
            self.name = self.available[chosen]
            self.context = self.lib.ibv_open_device(devices[chosen])
        finally:
            self.lib.ibv_free_device_list(devices)

        if not self.context:
            raise RuntimeError(f"ibv_open_device({self.name}) failed: {os.strerror(ctypes.get_errno())}")
        self.pd = self.lib.ibv_alloc_pd(self.context)
        if not self.pd:
            raise RuntimeError(f"ibv_alloc_pd({self.name}) failed: {os.strerror(ctypes.get_errno())}")

    def register_dmabuf(self, fd, offset, length, iova):
        access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ
        mr = self.lib.ibv_reg_dmabuf_mr(
            self.pd,
            ctypes.c_uint64(offset),
            ctypes.c_size_t(length),
            ctypes.c_uint64(iova),
            ctypes.c_int(fd),
            ctypes.c_int(access),
        )
        if not mr:
            err = ctypes.get_errno()
            raise OSError(
                err,
                f"ibv_reg_dmabuf_mr(len={length}, offset={offset}) failed on {self.name}: " f"{os.strerror(err)}",
            )
        return mr

    def keys(self, mr):
        lkey = ctypes.c_uint32.from_address(mr + IBV_MR_LKEY_OFFSET).value
        rkey = ctypes.c_uint32.from_address(mr + IBV_MR_RKEY_OFFSET).value
        return lkey, rkey

    def deregister(self, mr):
        return self.lib.ibv_dereg_mr(mr)


def driver_memory(pci_addr):
    """Physical VRAM accounting straight from amdgpu, bypassing HIP.

    ``mem_info_vram_used`` is the byte counter every user can read.  The TTM
    dump adds the buddy allocator's free list, which is what distinguishes an
    admission refusal from genuine exhaustion or fragmentation: KFD can reject
    an allocation the buddy allocator could still satisfy from one block.

    Current amdgpu prints ``free_blocks:`` per order and a 4 KiB ``chunk_size``.
    A block at that order is ``chunk_size << order``, not the order's free sum.
    """
    counters = {}
    try:
        with open(f"/sys/bus/pci/devices/{pci_addr}/mem_info_vram_used") as handle:
            counters["phys_used_gib"] = round(int(handle.read()) / GIB, 3)
    except OSError as error:
        counters["phys_used_error"] = str(error)

    try:
        with open(f"/sys/kernel/debug/dri/{pci_addr}/amdgpu_vram_mm") as handle:
            dump = handle.read()
    except OSError as error:
        counters["vram_mm_error"] = str(error)
        return counters

    free_mib = re.search(r"chunk_size:.*free:\s*(\d+)MiB", dump)
    if free_mib:
        counters["phys_free_gib"] = round(int(free_mib.group(1)) / 1024, 3)
    chunk = re.search(r"chunk_size:\s*(\d+)(KiB|MiB|GiB)", dump)
    chunk_bytes = int(chunk.group(1)) * UNIT_BYTES[chunk.group(2)] if chunk else 4096
    orders = [int(order) for order, _blocks in re.findall(r"order-\s*(\d+)\s+free:.*?free_blocks:\s*([1-9]\d*)", dump)]
    if orders:
        order = max(orders)
        counters["largest_free_block_gib"] = round((chunk_bytes << order) / GIB, 3)
        counters["largest_free_block_order"] = order
    return counters


def gpu_memory(hip, pci_addr):
    free_bytes, total_bytes = hip.mem_get_info()
    return {
        "free_gib": round(free_bytes / GIB, 3),
        "used_gib": round((total_bytes - free_bytes) / GIB, 3),
        "total_gib": round(total_bytes / GIB, 3),
        **driver_memory(pci_addr),
    }


def default_ib_device():
    """Pick the first device libibverbs can open providers for.

    Prefer libibverbs over a raw /sys/class/infiniband listing: sysfs can show
    ports whose userspace provider is missing, which would fail later in Verbs.
    """
    names = list_ib_devices()
    if names:
        return names[0]
    sysfs = []
    with contextlib.suppress(OSError):
        sysfs = sorted(os.listdir("/sys/class/infiniband"))
    if sysfs:
        raise RuntimeError(
            f"RDMA devices in sysfs {sysfs} but none via libibverbs; "
            "install matching provider packages (e.g. libmlx5, libbnxt_re)"
        )
    raise RuntimeError("no RDMA devices available (empty /sys/class/infiniband and libibverbs)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0, help="HIP device index")
    parser.add_argument(
        "--ib-device",
        default=None,
        help="RDMA device to register against (default: first under /sys/class/infiniband)",
    )
    parser.add_argument(
        "--target-gib",
        type=int,
        default=0,
        help="Working set in GiB. 0 (default) selects it from the memory free at start",
    )
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=800,
        help="Size of each allocation and each memory region",
    )
    parser.add_argument("--probe-mib", type=int, default=1024)
    parser.add_argument(
        "--export-api",
        choices=("hip", "hsa"),
        default="hip",
        help="Which dma-buf export entry point to use",
    )
    parser.add_argument(
        "--pattern",
        choices=("pin-all", "offset", "churn"),
        default="pin-all",
        help="Which registration shape to apply to the working set",
    )
    parser.add_argument(
        "--pin-gib",
        type=int,
        default=0,
        help="GiB of the working set to register from the tail (0: all)",
    )
    parser.add_argument(
        "--churn-cycles",
        type=int,
        default=50,
        help="Register/deregister rounds before the pinned probe when pattern is churn",
    )
    parser.add_argument(
        "--foreign-probe",
        action="store_true",
        help="While pins are held, try the probe from a second process on the same GPU",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-pin",
        action="store_true",
        help="Control run: allocate the target VRAM without registering it",
    )
    parser.add_argument(
        "--expect",
        choices=("failure", "success", "either"),
        default="either",
        help="Optional gate on the pinned probe. The verdict line is the bug signal",
    )
    parser.add_argument(
        "--hold-seconds",
        type=int,
        default=0,
        help="Keep registrations alive after the probe for external inspection",
    )
    args = parser.parse_args()
    if args.target_gib < 0 or min(args.chunk_mib, args.probe_mib, args.churn_cycles) <= 0:
        parser.error("allocation sizes must be positive")
    if args.pin_gib < 0:
        parser.error("--pin-gib must be >= 0")
    if args.ib_device is None and not args.skip_pin and not args.probe_only:
        args.ib_device = default_ib_device()
    return args


def choose_target_gib(free_bytes, total_bytes, probe_bytes):
    """Return (target_gib, source) from the memory free when the run starts."""
    reserve = max(probe_bytes * MIN_RESERVE_PROBES, free_bytes // FREE_RESERVE_DIVISOR)
    available = (free_bytes - reserve) // GIB
    ceiling = (total_bytes - probe_bytes - GIB) // GIB
    if available < 1 or ceiling < 1:
        raise RuntimeError(
            f"{free_bytes / GIB:.3f} GiB free of {total_bytes / GIB:.3f} GiB, "
            f"not enough for a {probe_bytes / GIB:.3f} GiB probe"
        )
    return min(available, ceiling), "vram"


def accounting_gap(snapshot):
    phys = snapshot.get("phys_free_gib")
    if phys is None:
        return None
    return round(phys - snapshot["free_gib"], 3)


def classify_probe(succeeded, probe_bytes, snapshot):
    """Separate a double-counted pin from exhaustion and from buddy fragmentation.

    The pin bug refuses admission while the TTM buddy allocator still has one
    free block at least as large as the request.  hipMemGetInfo can move with
    the bad counter, so physical free is the reference, not HIP's free number.
    """
    if succeeded:
        return "admitted"
    request_gib = probe_bytes / GIB
    phys = snapshot.get("phys_free_gib")
    largest = snapshot.get("largest_free_block_gib")
    if phys is None:
        return "refused_unverified"
    if phys + 0.05 < request_gib:
        return "genuine_oom"
    if largest is None:
        gap = accounting_gap(snapshot)
        if gap is not None and gap > request_gib:
            return "fake_oom"
        return "refused_unverified"
    if largest + 0.01 < request_gib:
        return "fragmented"
    return "fake_oom"


def allocate_working_set(hip, target_bytes, chunk_bytes):
    pointers = []
    allocated_bytes = 0
    while allocated_bytes < target_bytes:
        size = min(chunk_bytes, target_bytes - allocated_bytes)
        pointer, _status = hip.malloc(size)
        if pointer is None:
            raise RuntimeError(
                f"hipMalloc failed while building the working set "
                f"at chunk {len(pointers)} ({allocated_bytes / GIB:.3f} GiB in)"
            )
        pointers.append((pointer, size))
        allocated_bytes += size
    return pointers


def select_ranges(pointers, pattern, pin_bytes):
    """Pick the buffers that stand in for the registered comm or KV window.

    The tail of the working set is the window: weights stay allocated and
    unpinned, and only the window becomes a dma-buf MR.  ``offset`` then slides
    each region 2 MiB into its allocation, which is the pointer a caching
    allocator actually passes to registration.
    """
    if pin_bytes is None:
        chosen = list(pointers)
    else:
        chosen = []
        remaining = pin_bytes
        for pointer, size in reversed(pointers):
            if remaining <= 0:
                break
            chosen.append((pointer, size))
            remaining -= size
        chosen.reverse()
    if pattern != "offset":
        return chosen
    shifted = []
    for pointer, size in chosen:
        if size > SUBRANGE_OFFSET * 2:
            shifted.append((pointer + SUBRANGE_OFFSET, size - SUBRANGE_OFFSET))
        else:
            shifted.append((pointer, size))
    return shifted


def register_ranges(exporter, verbs, ranges, regions):
    """Append MRs to ``regions`` so a failure mid-loop is still deregistered."""
    for index, (pointer, size) in enumerate(ranges):
        fd, offset = exporter.export_dmabuf(pointer, size)
        try:
            mr = verbs.register_dmabuf(fd, offset, size, pointer)
        except OSError as error:
            os.close(fd)
            raise RuntimeError(f"registration failed at range {index} ({size / GIB:.3f} GiB): {error}") from error
        # The importer holds its own reference, so the exported handle
        # does not need to stay open for the lifetime of the mapping.
        os.close(fd)
        regions.append(mr)


def deregister_regions(verbs, regions):
    errors = []
    for index, mr in reversed(list(enumerate(regions))):
        ret = verbs.deregister(mr)
        if ret != 0:
            errors.append((index, ret))
    regions.clear()
    return errors


def run_probe(hip, pci_addr, probe_bytes, stage):
    pointer, status = hip.malloc(probe_bytes)
    snapshot = gpu_memory(hip, pci_addr)
    succeeded = pointer is not None
    fields = {"accounting_gap_gib": accounting_gap(snapshot), **snapshot}
    if succeeded:
        fields["address"] = hex(pointer)
    else:
        error = hip.lib.hipGetErrorString(status).decode()
        fields["error"] = f"hipMalloc({probe_bytes}) -> {error}"
    report(f"{stage}_{'succeeded' if succeeded else 'failed'}", **fields)
    if succeeded:
        hip.free(pointer)
    return succeeded, snapshot


def foreign_probe(args):
    """A new process, not a fork: HIP is not fork-safe after the parent has a context."""
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--probe-only",
        "--device",
        str(args.device),
        "--probe-mib",
        str(args.probe_mib),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
    payloads = []
    for line in completed.stdout.splitlines():
        if line.startswith("{"):
            payloads.append(json.loads(line))
    report(
        "foreign_probe",
        returncode=completed.returncode,
        child=payloads[-1] if payloads else None,
        stderr=completed.stderr[-400:] if completed.returncode not in (0, 2) else "",
    )
    if not payloads:
        return "refused_unverified", False
    child = payloads[-1]
    succeeded = child.get("stage") == "probe_succeeded"
    kind = classify_probe(succeeded, args.probe_mib * MIB, child)
    return kind, succeeded


def probe_only(args):
    hip = Hip()
    if not 0 <= args.device < hip.device_count():
        raise RuntimeError(f"GPU {args.device} does not exist")
    hip.set_device(args.device)
    pci_addr = hip.pci_address(args.device)
    warmup, _status = hip.malloc(MIB)
    hip.free(warmup)
    probe_bytes = args.probe_mib * MIB
    pointer, status = hip.malloc(probe_bytes)
    snapshot = gpu_memory(hip, pci_addr)
    payload = {
        "stage": "probe_succeeded" if pointer else "probe_failed",
        "pid": os.getpid(),
        "accounting_gap_gib": accounting_gap(snapshot),
        **snapshot,
    }
    if pointer:
        payload["address"] = hex(pointer)
        hip.free(pointer)
    else:
        payload["error"] = f"hipMalloc({probe_bytes}) -> {hip.lib.hipGetErrorString(status).decode()}"
    print(json.dumps(payload), flush=True)
    return 0 if pointer else 2


def prepare_run(args):
    """Open the GPU and report the planned working set.

    Returns hip, pci address, exporter, verbs, and the byte sizes main allocates.
    """
    hip = Hip()
    if not 0 <= args.device < hip.device_count():
        raise RuntimeError(f"GPU {args.device} does not exist")
    hip.set_device(args.device)
    pci_addr = hip.pci_address(args.device)

    # Establish the HIP context before applying memory pressure, so the probe
    # measures allocation admission and nothing else.
    warmup, _status = hip.malloc(MIB)
    hip.free(warmup)

    export_api = "hsa" if args.pattern == "offset" else args.export_api
    exporter = hip if export_api == "hip" else Hsa()
    verbs = None if args.skip_pin else Verbs(args.ib_device)

    probe_bytes = args.probe_mib * MIB
    free_bytes, total_bytes = hip.mem_get_info()
    if args.target_gib == 0:
        target_gib, target_source = choose_target_gib(free_bytes, total_bytes, probe_bytes)
    else:
        target_gib = args.target_gib
        target_source = "flag"
    target_bytes = target_gib * GIB
    chunk_bytes = args.chunk_mib * MIB
    test_name, scenario = CONTROL_SCENARIO if args.skip_pin else SCENARIOS[args.pattern]
    if args.foreign_probe:
        scenario += " A second process repeats the probe while the pins are held."
    report(
        "start",
        test_name=test_name,
        scenario=scenario,
        device=args.device,
        device_name=hip.device_name(args.device),
        target_gib=target_gib,
        target_source=target_source,
        pci_address=pci_addr,
        ib_device=verbs.name if verbs else None,
        rdma_devices_available=verbs.available if verbs else None,
        export_api=export_api,
        pattern=args.pattern,
        target_bytes=target_bytes,
        chunk_bytes=chunk_bytes,
        probe_bytes=probe_bytes,
        **gpu_memory(hip, pci_addr),
    )
    return hip, pci_addr, exporter, verbs, probe_bytes, target_bytes, chunk_bytes


def pin_working_window(args, hip, pci_addr, exporter, verbs, pointers, regions):
    """Register the selected window. Returns the foreign-probe verdict, or None."""
    pin_bytes = args.pin_gib * GIB if args.pin_gib else None
    ranges = select_ranges(pointers, args.pattern, pin_bytes)
    pinned_bytes = sum(size for _pointer, size in ranges)
    report(
        "window",
        ranges=len(ranges),
        pinned_bytes=pinned_bytes,
        **gpu_memory(hip, pci_addr),
    )
    if not ranges:
        report("window_empty", note="no regions selected to register")
        return None

    def pin_window():
        register_ranges(exporter, verbs, ranges, regions)

    pin_window()
    if args.pattern == "churn":
        for cycle in range(args.churn_cycles):
            errors = deregister_regions(verbs, regions)
            report(
                "churn_unpinned",
                cycle=cycle,
                deregistration_errors=errors,
                **gpu_memory(hip, pci_addr),
            )
            pin_window()
            report(
                "churn_pinned",
                cycle=cycle,
                ranges=len(regions),
                **gpu_memory(hip, pci_addr),
            )

    lkey, rkey = verbs.keys(regions[0])
    report(
        "pinned",
        chunks=len(regions),
        pinned_bytes=pinned_bytes,
        first_mr_lkey=lkey,
        first_mr_rkey=rkey,
        **gpu_memory(hip, pci_addr),
    )
    if not args.foreign_probe:
        return None
    foreign_kind, _foreign_ok = foreign_probe(args)
    return foreign_kind


def reprobe_after_unpin(args, hip, pci_addr, verbs, regions, probe_bytes):
    """After a churn release, probe again and treat a stuck refusal as a pin leak."""
    if args.pattern != "churn" or not regions:
        return None
    errors = deregister_regions(verbs, regions)
    report("unpinned", deregistration_errors=errors, **gpu_memory(hip, pci_addr))
    unpinned_ok, unpinned_snapshot = run_probe(hip, pci_addr, probe_bytes, "probe_after_unpin")
    unpinned_kind = classify_probe(unpinned_ok, probe_bytes, unpinned_snapshot)
    if not unpinned_ok and unpinned_kind in ("fake_oom", "refused_unverified"):
        return "pin_leak"
    return unpinned_kind


def cleanup_allocations(hip, verbs, pointers, regions):
    held = len(regions)
    deregistration_errors = deregister_regions(verbs, regions) if held else []
    for pointer, _size in pointers:
        hip.free(pointer)
    report(
        "cleanup",
        deregistered=held - len(deregistration_errors),
        deregistration_errors=deregistration_errors,
        freed=len(pointers),
    )


def verdict_exit(args, probe_succeeded, pinned_kind, unpinned_kind, foreign_kind):
    bug_kinds = {"fake_oom", "pin_leak"}
    confirmed = pinned_kind in bug_kinds or unpinned_kind in bug_kinds or foreign_kind in bug_kinds
    expect_failed = args.expect != "either" and probe_succeeded != (args.expect == "success")
    passed = not confirmed and not expect_failed
    report(
        "verdict",
        passed=passed,
        while_pinned=pinned_kind,
        while_pinned_signal=VERDICT_SIGNAL.get(pinned_kind),
        after_unpin=unpinned_kind,
        foreign=foreign_kind,
        expect_failed=expect_failed,
        expected=args.expect,
        actual="success" if probe_succeeded else "failure",
    )
    return 0 if passed else 1


def main():
    args = parse_args()
    if args.probe_only:
        return probe_only(args)

    hip, pci_addr, exporter, verbs, probe_bytes, target_bytes, chunk_bytes = prepare_run(args)

    pointers = []
    regions = []
    probe_succeeded = False
    pinned_kind = None
    unpinned_kind = None
    foreign_kind = None
    try:
        pointers = allocate_working_set(hip, target_bytes, chunk_bytes)
        report("allocated", chunks=len(pointers), **gpu_memory(hip, pci_addr))
        if args.skip_pin:
            report("pin_skipped", **gpu_memory(hip, pci_addr))
        else:
            foreign_kind = pin_working_window(args, hip, pci_addr, exporter, verbs, pointers, regions)

        probe_succeeded, pinned_snapshot = run_probe(hip, pci_addr, probe_bytes, "probe")
        pinned_kind = classify_probe(probe_succeeded, probe_bytes, pinned_snapshot)

        if args.hold_seconds and regions:
            report("holding", seconds=args.hold_seconds)
            time.sleep(args.hold_seconds)

        unpinned_kind = reprobe_after_unpin(args, hip, pci_addr, verbs, regions, probe_bytes)
    finally:
        cleanup_allocations(hip, verbs, pointers, regions)

    return verdict_exit(args, probe_succeeded, pinned_kind, unpinned_kind, foreign_kind)


if __name__ == "__main__":
    sys.exit(main())
