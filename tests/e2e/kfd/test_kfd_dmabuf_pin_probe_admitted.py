# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_kfd_dmabuf_pin_probe_admitted.py — KFD pinned-VRAM admission after RDMA dma-buf registration.

Ported from:   private source withheld
Upstream ref:  private source withheld; approval recorded
Ported on:     2026-11-19
OSS license:   MIT

Validates:
    1. After an RDMA dma-buf pin of VRAM this process already owns, a later probe
       allocation is still admitted (the pin is not charged twice). Confirmed
       fake_oom or pin_leak fails the run.

The one test runs every pin scenario in order: pin-all, offset, churn.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/kfd/:
    hw.gpu, layer.driver, ci.nightly, e2e.stack, os.linux

Explicit markers:
    runtime.soak
    ci.weekly  (overrides profile ci.nightly; churn is 50 full re-register cycles)
"""

from __future__ import annotations

import pathlib
import shlex

import pytest

_SRC = pathlib.Path(__file__).parent / "src" / "kfd_pinned_fake_vram.py"
_SRC_DIR = str(_SRC.parent)

# Original defaults: live free-memory working set, 800 MiB chunks, 1 GiB probe.
_COMMON_ARGS = "--target-gib 0 --chunk-mib 800 --probe-mib 1024 --device 0"

# name, args, timeout seconds, also require the post-release probe
_SCENARIOS = (
    ("pin-all", f"{_COMMON_ARGS} --pattern pin-all --foreign-probe", 1800, False),
    ("offset", f"{_COMMON_ARGS} --pattern offset --foreign-probe", 1800, False),
    ("churn", f"{_COMMON_ARGS} --pattern churn --churn-cycles 50 --foreign-probe", 7200, True),
)


def _script_on_target(target_executor) -> str:
    """Return the workload path on the execution node (upload when remote)."""
    executor = next(iter(target_executor))
    if hasattr(executor, "upload_tree"):
        remote_src_dir = executor.upload_tree(_SRC_DIR)
        return remote_src_dir + "/" + _SRC.name
    return str(_SRC)


def _require_rdma(target_executor, ld: str) -> None:
    """Skip when this node cannot register a dma-buf memory region."""
    probe = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} python3 -c "
        '\'import ctypes,os; ctypes.CDLL("libibverbs.so.1"); '
        'raise SystemExit(0 if os.listdir("/sys/class/infiniband") else 1)\'',
        timeout=60,
    )
    if not probe.ok:
        pytest.skip("RDMA dma-buf pin requires libibverbs and an InfiniBand device")


def _run(target_executor, ld: str, script: str, args: str, timeout: int):
    binary = shlex.quote(script)
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} python3 {binary} {args}",
        timeout=timeout,
    )
    assert result.ok, (
        f"kfd_pinned_fake_vram failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "Result                 PASS" in result.stdout, f"expected Result PASS:\n{result.stdout[:2000]}"
    assert "Result                 FAIL" not in result.stdout, f"Result FAIL in stdout:\n{result.stdout[:2000]}"
    return result


@pytest.mark.runtime.soak
@pytest.mark.ci.weekly
def test_kfd_dmabuf_pin_probe_admitted(target_executor, ld_path: dict):
    """Run pin-all, offset, and churn; each probe must be admitted."""
    ld = shlex.quote(ld_path["LD_LIBRARY_PATH"])
    _require_rdma(target_executor, ld)
    script = _script_on_target(target_executor)
    for name, args, timeout, check_unpin in _SCENARIOS:
        result = _run(target_executor, ld, script, args, timeout)
        assert "while pinned : admitted" in result.stdout, f"{name}: probe was not admitted:\n{result.stdout[:2000]}"
        if check_unpin:
            assert (
                "after release : admitted" in result.stdout
            ), f"{name}: probe after unpin was not admitted:\n{result.stdout[:2000]}"
