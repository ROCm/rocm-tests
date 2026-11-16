# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Fixtures for tests/e2e/system_tools/rdc/.

Brings up an unauthenticated RDC daemon and hands each test a GPU group to
diagnose. ``rdcd`` serves clients over gRPC on localhost:50051 and by default
demands SSL mutual authentication, reading a root CA from
``/etc/rdc/client/certs``. The ``-u`` flag drops that requirement, and it is
the mode these tests cover; every ``rdci`` call therefore carries ``-u`` too.

Before a test runs, the daemon has to report the node's full GPU inventory and
the GPU under test has to pass a health check; the same health check runs again
afterwards. Both the daemon and the diagnostic run as root.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import shlex
import time

import pytest

from framework.rocm.libs.stack import get_rocm_version
from tests.common.os_packages import detect_os

logger = logging.getLogger(__name__)

# Ceiling for resolving a binary or issuing a single rdci control call. All of
# these are sub-second, so anything near this means the node is wedged.
_PROBE_TIMEOUT = 60.0

# rdcd binds its port a moment after exec, so readiness is polled rather than
# slept through.
_READY_TIMEOUT = 60.0
_READY_POLL = 2.0
# rdcd's gRPC port, polled directly so the wait costs no rdci output.
_RDCD_PORT = 50051
_LISTEN_POLL = 0.5

# rdcd keeps a single-instance lock at one of these paths depending on the user
# it runs as. A root-owned lock left in /tmp makes a later non-root start fail
# with EACCES, so both are cleared before starting.
_LOCK_FILES = ("/var/run/rdcd.lock", "/tmp/rdcd.lock")

# GPU architectures the RDC diagnostic is qualified on.
_SUPPORTED_ARCHS = frozenset({"gfx942", "gfx950", "gfx1250"})
_MIN_ROCM = (6, 4)

# RDC ships inside ROCm from 7.11.0 onwards. Before that it came from a separate
# OS package, which is installed on demand; from 7.11.0 a missing binary means
# the ROCm install itself is incomplete and installing a package would mask that.
_RDC_BUNDLED_FROM = (7, 11)
_RDC_PACKAGE = "rdc"
_INSTALL_TIMEOUT = 600.0

# The first gfx target rocminfo prints belongs to a GPU agent; the CPU agents
# listed above it carry their model name instead.
_ARCH_RE = re.compile(r"\bgfx[0-9a-f]+\b")
# rdcd prints this once it has dropped the SSL requirement.
_UNAUTH_BANNER = "Accepting Unauthenticated connections only"
# Discovery reports these when it cannot reach the daemon, and they disqualify
# a reply even alongside an otherwise complete inventory.
_DISCOVERY_FAIL_RE = re.compile(r"failed to connect to all addresses|Fail to get device information")

# ``Successfully created group with a group ID 1``
_GROUP_ID_RE = re.compile(r"group ID\s+(\d+)", re.IGNORECASE)

# The GPU the diagnostic is pointed at, in RDC's own enumeration. It is
# independent of whichever GPU the node pool allocated to the test.
_DIAG_GPU_INDEX = 0

# rocminfo prints this when the kernel driver is absent. It, and a failure to
# open the device, are what the health gate reads as "driver not loaded".
_ROCK_NOT_LOADED_RE = re.compile(r"ROCk module is NOT (loaded|live),\s*possibly no GPU devices", re.IGNORECASE)


@dataclass(frozen=True)
class RdcBinaries:
    """Resolved paths to the RDC daemon and its client."""

    rdcd: str
    rdci: str


def _resolve(executor, rock_dir: str, name: str) -> str:
    """Return *name* from the ROCm under test, else from PATH, else ``""``.

    ``--rock-dir`` is preferred so a host carrying several ROCm installs
    exercises the one that was selected rather than whichever copy happens to
    come first on PATH.
    """
    if rock_dir:
        candidate = f"{rock_dir.rstrip('/')}/bin/{name}"
        if executor.run(f"test -x {shlex.quote(candidate)}", timeout=_PROBE_TIMEOUT).ok:
            return candidate

    which = executor.run(f"command -v {shlex.quote(name)}", timeout=_PROBE_TIMEOUT)
    resolved = (which.stdout or "").strip().splitlines()
    return resolved[-1].strip() if which.ok and resolved else ""


def _rocm_version(executor, rock_dir: str) -> tuple[int, int] | None:
    """Return ``(major, minor)`` of the ROCm under test, or ``None``."""
    version = ""
    if rock_dir:
        result = executor.run(
            f"cat {shlex.quote(rock_dir.rstrip('/'))}/.info/version 2>/dev/null",
            timeout=_PROBE_TIMEOUT,
        )
        if result.ok:
            version = (result.stdout or "").strip().split(" ")[0].strip()
    if not version:
        version = get_rocm_version(executor) or ""

    match = re.match(r"(\d+)\.(\d+)", version)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _gpu_arch(report) -> str:
    """Return the GFX target of the node's first GPU agent, or ``""``."""
    if report is None:
        return ""
    match = _ARCH_RE.search(report.stdout or "")
    return match.group(0) if match else ""


def _install_rdc_package(executor) -> None:
    """Install the ``rdc`` OS package, logging rather than raising on failure.

    Used only on the ROCm versions that predate RDC being bundled. The package
    name is the same on every supported distribution, so unlike the shared
    helper no ``-devel`` suffix is applied.
    """
    try:
        os_family = detect_os(executor)
    except RuntimeError as exc:
        logger.warning("Cannot install %s: %s", _RDC_PACKAGE, exc)
        return

    commands = {
        "ubuntu": f"sudo -n apt-get install -y --no-install-recommends {_RDC_PACKAGE}",
        "rhel": f"sudo -n dnf install -y --nogpgcheck {_RDC_PACKAGE}",
        "sles": f"sudo -n zypper install -y --no-gpg-checks {_RDC_PACKAGE}",
    }

    logger.info("Installing %s on %s", _RDC_PACKAGE, os_family)
    result = executor.run(commands[os_family], timeout=_INSTALL_TIMEOUT)
    if not result.ok:
        logger.warning(
            "Installing %s returned %s; the binary check below decides the outcome.\n%s",
            _RDC_PACKAGE,
            result.exit_code,
            (result.stderr or "")[:500],
        )


def _physical_gpu_count(device_list) -> int:
    """Return the number of physical GPUs on the node.

    Counts only the entries ``amd-smi`` reports with partition 0, so a
    partitioned node is counted once per device rather than once per partition.
    """
    if device_list is None:
        pytest.fail("amd-smi is not available, so the number of GPUs rdcd should report cannot be established.")

    output = (device_list.stdout or "").strip()
    try:
        entries = json.loads(output)
    except ValueError:
        pytest.fail(f"'amd-smi list --json' did not return JSON:\n{output[-2000:]}")

    count = sum(1 for entry in entries if entry.get("partition_id") == 0)
    if not count:
        pytest.fail(f"'amd-smi list --json' reported no GPU at partition 0:\n{output[-2000:]}")
    return count


def _node_health_gate(rocminfo_result, device_list) -> None:
    """Apply the three prerequisites a run is blocked on.

    rocminfo has to answer, amd-smi has to answer, and the kernel driver has to
    be loaded. The wider snapshot taken alongside these --- dmesg, GPU
    processes, CPU and disk usage --- is collected as evidence and never
    changes the verdict, so it is left to the framework's own reporting rather
    than asserted on here.
    """
    if rocminfo_result is None:
        pytest.fail("rocminfo is not available, so the node's GPU state cannot be established")

    result = rocminfo_result
    report = (result.stdout or "") + (result.stderr or "")
    assert result.ok, f"rocminfo exited {result.exit_code}:\n{report[-1000:]}"
    assert report.strip(), "rocminfo produced no output, so the node's GPU state is unknown"
    assert not _ROCK_NOT_LOADED_RE.search(report), (
        f"the amdgpu driver is not loaded, so no GPU state is readable:\n{report[-1000:]}"
    )
    assert "Unable to open" not in report, f"rocminfo could not open the device:\n{report[-1000:]}"

    if device_list is None:
        pytest.fail("amd-smi is not available, so the node's GPU state cannot be established")

    assert device_list.ok, f"amd-smi exited {device_list.exit_code}:\n{(device_list.stderr or '')[-1000:]}"
    assert (device_list.stdout or "").strip(), "amd-smi produced no output, so the node's GPU state is unknown"
    logger.info("Node health prerequisites met: rocminfo and amd-smi answer, amdgpu is loaded")


def _log_health(phase: str, result) -> None:
    """Record one health sample for *phase*."""
    logger.info(
        "%s health on GPU %d: temp=%sC ecc=%s vram_free=%sMB",
        phase,
        _DIAG_GPU_INDEX,
        result.temp_c,
        result.ecc_errors,
        result.vram_free_mb,
    )


def _log_partition_mode(executor, rock_dir: str) -> None:
    """Record the node's partition mode, which decides what one GPU index covers.

    Reported rather than asserted on: the diagnostic runs on whatever the node
    is partitioned into, and the mode is context for reading a failure.
    """
    amd_smi = _resolve(executor, rock_dir, "amd-smi")
    if not amd_smi:
        return

    result = executor.run(f"{shlex.quote(amd_smi)} partition --json", timeout=_PROBE_TIMEOUT)
    try:
        payload = json.loads((result.stdout or "").strip())
    except ValueError:
        payload = None
    entries = payload.get("current_partition", []) if isinstance(payload, dict) else []
    if not entries:
        logger.info("Compute partition mode could not be read from amd-smi")
        return

    accelerator = sorted({e.get("accelerator_type") for e in entries if e.get("accelerator_type")})
    memory = sorted({e.get("memory") for e in entries if e.get("memory")})
    logger.info(
        "Partition mode: compute=%s memory=%s",
        ", ".join(accelerator) or "unknown",
        ", ".join(memory) or "unknown",
    )


def _stop_rdcd(executor) -> None:
    """Stop any running rdcd and clear the locks that would block a new one."""
    executor.run("sudo -n systemctl stop rdc 2>/dev/null || true", timeout=_PROBE_TIMEOUT)
    executor.run("sudo -n pkill -x rdcd || true", timeout=_PROBE_TIMEOUT)
    executor.run(f"sudo -n rm -f {' '.join(_LOCK_FILES)} || true", timeout=_PROBE_TIMEOUT)


def _await_listener(executor, timeout: float) -> None:
    """Wait for rdcd's gRPC port to reach LISTEN, giving up after *timeout*.

    ``rdci`` writes its connection failures to stdout, which the per-test
    artifact log records verbatim, so polling with it before the socket exists
    fills that log with errors that are only the wait. ``ss`` with ``grep -q``
    answers the same question silently, and the retry loop runs inside a single
    shell invocation so the wait costs one line in that log rather than one per
    poll. Giving up is not fatal here; the inventory check that follows decides.
    """
    attempts = max(1, int(timeout / _LISTEN_POLL))
    probe = (
        f"for _ in $(seq {attempts}); do "
        f"ss -ltnH 'sport = :{_RDCD_PORT}' | grep -q LISTEN && exit 0; "
        f"sleep {_LISTEN_POLL}; done; exit 1"
    )
    executor.run(probe, timeout=timeout + _PROBE_TIMEOUT)


def _await_unauth_mode(log_path: str, start_offset: int) -> None:
    """Require rdcd to announce that it serves unauthenticated clients only.

    A daemon that came up in the default SSL mode is a different fault from one
    that is merely slow, and a failing ``rdci -u`` call cannot tell them apart.
    The framework forwards the daemon's output to *log_path* on the host
    running the tests, so the file is read directly rather than through the
    executor. It is appended to across runs, hence *start_offset*: only output
    produced by this run can satisfy the check.
    """
    deadline = time.monotonic() + _READY_TIMEOUT
    captured = ""
    while time.monotonic() < deadline:
        try:
            captured = Path(log_path).read_bytes()[start_offset:].decode(errors="replace")
        except OSError:
            captured = ""
        if _UNAUTH_BANNER in captured:
            logger.info("rdcd announced: %s", _UNAUTH_BANNER)
            return
        time.sleep(_LISTEN_POLL)

    pytest.fail(f"rdcd never announced {_UNAUTH_BANNER!r} within {_READY_TIMEOUT:.0f}s:\n{captured[-2000:]}")


def _await_rdcd(executor, rdci: str, expected_gpus: int) -> None:
    """Wait until the daemon serves a full inventory to an unauthenticated client.

    The reply is held to *expected_gpus* rather than to any GPU appearing: a
    daemon that enumerated only part of the node answers discovery perfectly
    well, and a diagnostic run against it would pass while covering less
    hardware than the suite claims. Only the deadline decides the outcome, so a
    daemon that never binds still fails here rather than hanging.
    """
    inventory = (
        re.compile(rf"{expected_gpus}\s+GPUs found"),
        re.compile(r"GPU Index"),
        re.compile(r"Device Information"),
    )

    logger.info("Waiting up to %.0fs for rdcd to serve %d GPUs", _READY_TIMEOUT, expected_gpus)

    deadline = time.monotonic() + _READY_TIMEOUT
    _await_listener(executor, _READY_TIMEOUT)

    # At least one inventory check always runs, so a daemon that never bound
    # still fails on rdci's own diagnostics rather than on an empty message.
    last = ""
    while True:
        result = executor.run(f"{shlex.quote(rdci)} discovery -u -l", timeout=_PROBE_TIMEOUT)
        last = (result.stdout or "") + (result.stderr or "")
        if result.ok and all(pattern.search(last) for pattern in inventory) and not _DISCOVERY_FAIL_RE.search(last):
            logger.info("rdcd is serving unauthenticated clients and reports %d GPUs", expected_gpus)
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(_READY_POLL)

    pytest.fail(
        f"rdcd did not report {expected_gpus} GPUs to an unauthenticated client "
        f"within {_READY_TIMEOUT:.0f}s:\n{last[-2000:]}"
    )


@pytest.fixture
def rdc_binaries(target_executor, rock_dir: str) -> RdcBinaries:
    """Locate ``rdcd`` and ``rdci``, installing the RDC package if that is where they come from.

    On a ROCm that predates :data:`_RDC_BUNDLED_FROM` the binaries arrive in a
    separate OS package, so a missing one is installed and the lookup retried.
    From that version on they are part of ROCm itself and installing a package
    would hide an incomplete install behind a second, possibly mismatched copy.
    """

    def _locate() -> dict[str, str]:
        return {name: _resolve(target_executor, rock_dir, name) for name in ("rdcd", "rdci")}

    resolved = _locate()
    if not all(resolved.values()):
        version = _rocm_version(target_executor, rock_dir)
        if version is not None and version < _RDC_BUNDLED_FROM:
            missing = sorted(name for name, path in resolved.items() if not path)
            logger.info("%s not found on ROCm %d.%d; installing %s", ", ".join(missing), *version, _RDC_PACKAGE)
            _install_rdc_package(target_executor)
            resolved = _locate()

    for name, path in resolved.items():
        if not path:
            pytest.fail(
                f"{name} found neither at {rock_dir}/bin/{name} nor on PATH. It ships "
                "with ROCm, so a node without it is misconfigured rather than out of scope."
            )

    logger.info("Using rdcd=%s rdci=%s", resolved["rdcd"], resolved["rdci"])
    return RdcBinaries(rdcd=resolved["rdcd"], rdci=resolved["rdci"])


@pytest.fixture
def rocminfo_report(target_executor, rock_dir: str):
    """Capture rocminfo once per test, shared by every check that reads it.

    rocminfo prints well over a thousand lines and the executor logs each one,
    so a second invocation would double the test log without adding anything.
    Returns None when rocminfo is absent, leaving each caller to decide whether
    that is a skip or a failure.
    """
    rocminfo = _resolve(target_executor, rock_dir, "rocminfo")
    if not rocminfo:
        return None
    return target_executor.run(shlex.quote(rocminfo), timeout=_PROBE_TIMEOUT)


@pytest.fixture
def amd_smi_device_list(target_executor, rock_dir: str):
    """Capture ``amd-smi list --json`` once per test, shared by its readers.

    Returns None when amd-smi is absent, leaving each caller to decide whether
    that is a skip or a failure.
    """
    amd_smi = _resolve(target_executor, rock_dir, "amd-smi")
    if not amd_smi:
        return None
    return target_executor.run(f"{shlex.quote(amd_smi)} list --json", timeout=_PROBE_TIMEOUT)


@pytest.fixture
def rdc_diag_supported(target_executor, rock_dir: str, rocminfo_report) -> None:
    """Skip unless the node matches the configuration the diagnostic is qualified on."""
    if not target_executor.run("sudo -n true", timeout=_PROBE_TIMEOUT).ok:
        pytest.skip("passwordless sudo is not available; rdcd and rdci diag both run as root")

    version = _rocm_version(target_executor, rock_dir)
    if version is None:
        pytest.skip("the ROCm version could not be determined, so the minimum cannot be checked")
    if version < _MIN_ROCM:
        pytest.skip(
            f"the RDC diagnostic is qualified from ROCm {_MIN_ROCM[0]}.{_MIN_ROCM[1]}; "
            f"this node reports {version[0]}.{version[1]}"
        )

    arch = _gpu_arch(rocminfo_report)
    if arch not in _SUPPORTED_ARCHS:
        pytest.skip(
            f"the RDC diagnostic is qualified on {', '.join(sorted(_SUPPORTED_ARCHS))}; "
            f"this node reports {arch or 'no GPU agent'}"
        )
    logger.info("Node qualifies: ROCm %d.%d on %s", version[0], version[1], arch)


@pytest.fixture
def gpu_health_gate(health_fixture, rocminfo_report, amd_smi_device_list, rdc_diag_supported):
    """Gate the run on the node's health, then sample it again afterwards.

    The pre-run checks block the test: rocminfo and amd-smi have to answer, the
    driver has to be loaded, and the GPU under test has to sit inside the
    configured temperature, ECC and VRAM thresholds.

    The post-run sample is recorded but does not fail the test. A post snapshot
    is taken for triage rather than for gating, and raising here would turn a
    diagnostic that passed into a failure attributed to the wrong thing.
    """
    del rdc_diag_supported

    _node_health_gate(rocminfo_report, amd_smi_device_list)

    before = health_fixture.check(_DIAG_GPU_INDEX)
    assert before.passed, f"GPU {_DIAG_GPU_INDEX} was unhealthy before the diagnostic: {before.message}"
    _log_health("Pre-run", before)

    yield

    after = health_fixture.check(_DIAG_GPU_INDEX)
    _log_health("Post-run", after)
    if not after.passed:
        logger.warning("GPU %d is unhealthy after the diagnostic: %s", _DIAG_GPU_INDEX, after.message)


@pytest.fixture
def rdcd_daemon(
    target_executor,
    rdc_binaries: RdcBinaries,
    rdc_diag_supported,
    gpu_health_gate,
    amd_smi_device_list,
    rock_dir: str,
    request,
):
    """Run an unauthenticated ``rdcd`` for the duration of one test.

    Only one rdcd may hold the lock at a time, so any leftover instance is
    stopped first. It is stopped again on the way out because the background
    handle owns the ``sudo`` wrapper rather than the root daemon beneath it.
    """
    del rdc_diag_supported, gpu_health_gate
    expected_gpus = _physical_gpu_count(amd_smi_device_list)
    _log_partition_mode(target_executor, rock_dir)
    _stop_rdcd(target_executor)

    log_path = f"output/artifacts/executor-logs/{request.node.name}__rdcd.log"
    start_offset = Path(log_path).stat().st_size if Path(log_path).exists() else 0

    command = f"sudo -n {shlex.quote(rdc_binaries.rdcd)} -u"
    logger.info("Starting RDC daemon: %s", command)
    with target_executor.start_background(command, log_path=log_path, console_label="rdcd") as daemon:
        _await_unauth_mode(log_path, start_offset)
        _await_rdcd(target_executor, rdc_binaries.rdci, expected_gpus)
        yield daemon

    _stop_rdcd(target_executor)


@pytest.fixture
def rdc_group(target_executor, rdc_binaries: RdcBinaries, rdcd_daemon):
    """Return a factory making a named GPU group, deleted when the test ends.

    *gpu_index* is an index in RDC's own enumeration, which is independent of
    whichever GPU the node pool allocated to this test.
    """
    del rdcd_daemon
    created: list[str] = []

    def _create(name: str, gpu_index: str = str(_DIAG_GPU_INDEX)) -> str:
        rdci = shlex.quote(rdc_binaries.rdci)

        result = target_executor.run(f"{rdci} group -u -c {shlex.quote(name)}", timeout=_PROBE_TIMEOUT)
        output = (result.stdout or "") + (result.stderr or "")
        match = _GROUP_ID_RE.search(output)
        if not match:
            pytest.fail(f"could not create RDC group {name!r}:\n{output[-2000:]}")
        group_id = match.group(1)
        created.append(group_id)

        added = target_executor.run(
            f"{rdci} group -u -g {group_id} -a {shlex.quote(gpu_index)}", timeout=_PROBE_TIMEOUT
        )
        if not added.ok:
            detail = (added.stdout or "") + (added.stderr or "")
            pytest.fail(f"could not add GPU {gpu_index} to RDC group {group_id}:\n{detail[-2000:]}")

        logger.info("Created RDC group %s (%r) holding GPU %s", group_id, name, gpu_index)
        return group_id

    yield _create

    for group_id in created:
        target_executor.run(f"{shlex.quote(rdc_binaries.rdci)} group -u -d {group_id}", timeout=_PROBE_TIMEOUT)
