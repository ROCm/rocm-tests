# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_kfd_dmabuf_pin_probe_admitted.py — KFD / amdgpu pin accounting after dma-buf pin.

Ported from:   private source withheld
Upstream ref:  private source withheld; approval recorded
Ported on:     2026-11-19
OSS license:   MIT

Five pytest sub-cases (``@pytest.mark.parametrize``, same pattern as
``test_hip_invalid_codeobject_load`` / ``test_rccl_pat_algo``):

    rdma_pin_all                 — RDMA register full working set, probe admitted
    rdma_offset                  — RDMA register 2 MiB subranges, probe admitted
    rdma_churn                   — RDMA register/release cycles, probe admitted
    cross_gpu_replicate_count    — peer-GPU pin_dmabuf imports do not multi-charge
    cross_gpu_probe_admitted     — after peer pin, 4 MiB probe still admitted

Confirmed fake_oom / pin_leak / refused probe fails the matching sub-case.
A sub-case skips when its own prerequisites are missing.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/kfd/:
    hw.gpu, layer.driver, ci.nightly, e2e.stack, os.linux

Explicit markers:
    runtime.soak, ci.weekly, hw.multi_gpu, gpu_count(2)
"""

from __future__ import annotations

import logging
import pathlib
import shlex

import pytest

logger = logging.getLogger(__name__)

_SRC_DIR = pathlib.Path(__file__).parent / "src"
_RDMA_SCRIPT = _SRC_DIR / "kfd_pinned_fake_vram.py"

_COMMON_ARGS = "--target-gib 0 --chunk-mib 800 --probe-mib 1024 --device 0"

# (id, kind, args_or_none, timeout_s, check_unpin)
# kind: "rdma" | "cross_replicate" | "cross_probe"
_SUBTESTS = (
    (
        "rdma_pin_all",
        "rdma",
        f"{_COMMON_ARGS} --pattern pin-all --foreign-probe",
        1800,
        False,
    ),
    (
        "rdma_offset",
        "rdma",
        f"{_COMMON_ARGS} --pattern offset --foreign-probe",
        1800,
        False,
    ),
    (
        "rdma_churn",
        "rdma",
        f"{_COMMON_ARGS} --pattern churn --churn-cycles 50 --foreign-probe",
        7200,
        True,
    ),
    ("cross_gpu_replicate_count", "cross_replicate", None, 600, False),
    ("cross_gpu_probe_admitted", "cross_probe", None, 1800, False),
)

_PIN_GB = 50
_REPLICATE_GB = 2


def _src_on_target(target_executor) -> pathlib.Path:
    """Return the src/ tree path on the execution node (upload when remote)."""
    executor = next(iter(target_executor))
    if hasattr(executor, "upload_tree"):
        return pathlib.Path(executor.upload_tree(str(_SRC_DIR)))
    return _SRC_DIR


def _rdma_available(target_executor, ld: str) -> bool:
    probe = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} python3 -c "
        '\'import ctypes,os; ctypes.CDLL("libibverbs.so.1"); '
        'raise SystemExit(0 if os.listdir("/sys/class/infiniband") else 1)\'',
        timeout=60,
    )
    return probe.ok


def _assert_pass(result, label: str) -> None:
    assert result.ok, (
        f"{label} failed (exit={result.exit_code}):\n" f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "Result                 PASS" in result.stdout, f"{label}: expected Result PASS:\n{result.stdout[:2000]}"
    assert "Result                 FAIL" not in result.stdout, f"{label}: Result FAIL:\n{result.stdout[:2000]}"


def _cross_gpu_prereqs(target_executor, rock_dir: str) -> str | None:
    """Return a skip reason, or None when the cross-GPU side can run."""
    if not target_executor.run("sudo -n true", timeout=30).ok:
        return "passwordless sudo required to load pin_dmabuf.ko"
    if not target_executor.run("test -d /lib/modules/$(uname -r)/build", timeout=30).ok:
        return "kernel headers/build dir missing for pin_dmabuf.ko"
    gpu_probe = target_executor.run(
        f"env PATH={shlex.quote(rock_dir + '/bin')}:$PATH "
        f"LD_LIBRARY_PATH={shlex.quote(rock_dir + '/lib')}:${{LD_LIBRARY_PATH:-}} "
        "python3 -c 'import ctypes; "
        'h=ctypes.CDLL("libamdhip64.so"); '
        "n=ctypes.c_int(0); "
        "h.hipGetDeviceCount(ctypes.byref(n)); "
        "raise SystemExit(0 if n.value>=2 else 1)'",
        timeout=60,
    )
    if not gpu_probe.ok:
        return "cross-GPU dma-buf pin needs at least 2 HIP GPUs"
    return None


def _build_cross_gpu(target_executor, fake_oom_dir: str, rock_dir: str) -> None:
    """Build pin_dmabuf.ko and the two userspace binaries on the target node."""
    build = target_executor.run(
        f"set -e; cd {shlex.quote(fake_oom_dir)}; "
        f"KDIR=/lib/modules/$(uname -r)/build; "
        f'test -d "$KDIR"; '
        f'make -C pin_dmabuf_kmod KDIR="$KDIR"; '
        f"HIPCC=$(command -v hipcc || true); "
        f"if [ -z \"$HIPCC\" ]; then HIPCC={shlex.quote(rock_dir + '/bin/hipcc')}; fi; "
        f'make ROCM_PATH={shlex.quote(rock_dir)} HIPCC="$HIPCC" all; '
        f"chmod +x fake_oom_4mb replicate_count_dmabufsize",
        timeout=600,
    )
    assert build.ok, (
        f"cross-GPU fake-OOM build failed (exit={build.exit_code}):\n"
        f"stdout: {build.stdout[:2000]}\nstderr: {build.stderr[:1000]}"
    )


def _load_pin_module(target_executor, fake_oom_dir: str) -> None:
    ko = shlex.quote(f"{fake_oom_dir}/pin_dmabuf_kmod/pin_dmabuf.ko")
    target_executor.run("sudo -n rmmod pin_dmabuf 2>/dev/null || true", timeout=60)
    loaded = target_executor.run(f"sudo -n insmod {ko}", timeout=60)
    assert loaded.ok, (
        f"insmod pin_dmabuf.ko failed (exit={loaded.exit_code}):\n"
        f"stdout: {loaded.stdout[:1000]}\nstderr: {loaded.stderr[:1000]}"
    )
    assert target_executor.run("test -e /dev/pin_dmabuf", timeout=30).ok, "/dev/pin_dmabuf missing after insmod"


def _unload_pin_module(target_executor) -> None:
    target_executor.run("sudo -n rmmod pin_dmabuf 2>/dev/null || true", timeout=60)


# target_executor is function-scoped; cache build results in-process so the two
# cross-GPU sub-cases do not rebuild pin_dmabuf every time.
_CROSS_GPU_CACHE: dict[str, object] = {}


def _resolve_cross_gpu(target_executor, rock_dir: str, src: pathlib.Path) -> dict:
    """Return {ok, reason, dir}, building once per process when prereqs allow."""
    cached = _CROSS_GPU_CACHE.get("state")
    if cached is not None:
        return cached  # type: ignore[return-value]

    reason = _cross_gpu_prereqs(target_executor, rock_dir)
    if reason is not None:
        state = {"ok": False, "reason": reason, "dir": None}
        _CROSS_GPU_CACHE["state"] = state
        return state

    fake_oom = str(src / "fake_oom")
    _build_cross_gpu(target_executor, fake_oom, rock_dir)
    state = {"ok": True, "reason": None, "dir": fake_oom}
    _CROSS_GPU_CACHE["state"] = state
    return state


@pytest.mark.runtime.soak
@pytest.mark.ci.weekly
@pytest.mark.hw.multi_gpu
@pytest.mark.gpu_count(2)
@pytest.mark.parametrize(
    ("subtest", "kind", "args", "timeout", "check_unpin"),
    _SUBTESTS,
    ids=[s[0] for s in _SUBTESTS],
)
def test_kfd_dmabuf_pin_probe_admitted(
    target_executor,
    ld_path: dict,
    rock_dir: str,
    subtest: str,
    kind: str,
    args: str | None,
    timeout: int,
    check_unpin: bool,
):
    """One parametrized sub-case of dma-buf pin probe admission."""
    ld = shlex.quote(ld_path["LD_LIBRARY_PATH"])
    src = _src_on_target(target_executor)
    logger.info("subtest=%s kind=%s", subtest, kind)

    if kind == "rdma":
        if not _rdma_available(target_executor, ld):
            pytest.skip("RDMA dma-buf pin requires libibverbs and an InfiniBand device")
        script = shlex.quote(str(src / _RDMA_SCRIPT.name))
        result = target_executor.run(
            f"env LD_LIBRARY_PATH={ld} python3 {script} {args}",
            timeout=timeout,
        )
        _assert_pass(result, subtest)
        assert "while pinned : admitted" in result.stdout, f"{subtest}: probe was not admitted:\n{result.stdout[:2000]}"
        if check_unpin:
            assert (
                "after release : admitted" in result.stdout
            ), f"{subtest}: probe after unpin was not admitted:\n{result.stdout[:2000]}"
        return

    cross = _resolve_cross_gpu(target_executor, rock_dir, src)
    if not cross["ok"]:
        pytest.skip(f"cross-GPU side unavailable: {cross['reason']}")

    fake_oom_dir = cross["dir"]
    assert isinstance(fake_oom_dir, str)
    _load_pin_module(target_executor, fake_oom_dir)
    try:
        if kind == "cross_replicate":
            result = target_executor.run(
                f"env LD_LIBRARY_PATH={ld} "
                f"{shlex.quote(fake_oom_dir + '/replicate_count_dmabufsize')} "
                f"--src-gpu 0 --size-gb {_REPLICATE_GB}",
                timeout=timeout,
            )
            _assert_pass(result, subtest)
            return

        if kind == "cross_probe":
            result = target_executor.run(
                f"env LD_LIBRARY_PATH={ld} "
                f"{shlex.quote(fake_oom_dir + '/fake_oom_4mb')} "
                f"--src-gpu 0 --import-gpu 1 --pin-gb {_PIN_GB}",
                timeout=timeout,
            )
            _assert_pass(result, subtest)
            assert (
                "while pinned : admitted" in result.stdout
            ), f"{subtest}: probe was not admitted:\n{result.stdout[:2000]}"
            return
    finally:
        _unload_pin_module(target_executor)

    raise AssertionError(f"unknown subtest kind: {kind}")
