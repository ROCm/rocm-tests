# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fixtures for GPU-monitored test execution.

Integrates RVS fixtures with continuous GPU monitoring, post-test analysis,
and HTML report generation.

``rock_dir`` and ``ld_path`` are provided by the framework's ``builder_plugin``
and are NOT re-declared here.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import pytest

from framework.executors.local_executor import run_cmd_get_stdout_stderr
from tests.common.gpu_monitored.analysis import analyze_and_write
from tests.common.gpu_monitored.monitoring import Monitor, count_csv_samples
from tests.common.gpu_monitored.report import write_report
from tests.common.gpu_monitored.validation import capture_dmesg, dmesg_delta, validate_rvs_result
from tests.e2e.rvs.conftest import (
    gpu_conf_dir,
    rvs_binary,
    rvs_find_conf,
    rvs_source,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DMESG_CRITICAL_RE = re.compile(
    r"(Kernel panic|Oops|GPU reset|amdgpu.*ring.*timeout|"
    r"amdgpu.*job timedout|RAS.*error|IOMMU.*fault|"
    r"thermal throttle|Hardware Error)",
    re.IGNORECASE,
)


def _probe_kernel_health(lookback_sec: int = 300) -> dict:
    """Check dmesg for critical GPU events in the last N seconds.

    Returns:
        dict with keys: healthy (bool), events (list), raw_lines (int)
    """
    try:
        rc, stdout, _stderr = run_cmd_get_stdout_stderr(
            "dmesg",
            "--time-format=reltime",
            f"--since=-{lookback_sec}s",
            timeout=10,
            quiet=True,
        )
        if rc != 0:
            rc, stdout, _stderr = run_cmd_get_stdout_stderr(
                "sudo",
                "-n",
                "dmesg",
                "-T",
                timeout=10,
                quiet=True,
            )
        if rc != 0:
            return {"healthy": True, "events": [], "raw_lines": 0, "note": "dmesg unavailable"}

        lines = stdout.splitlines()
        critical = [line.strip() for line in lines if _DMESG_CRITICAL_RE.search(line)]

        return {
            "healthy": len(critical) == 0,
            "events": critical[:20],
            "raw_lines": len(lines),
        }
    except Exception:
        return {"healthy": True, "events": [], "raw_lines": 0, "note": "dmesg unavailable"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def gpu_monitor_interval():
    """Monitoring interval in seconds (configurable via env or default)."""
    return int(os.environ.get("GPU_MONITOR_INTERVAL", "2"))


@pytest.fixture
def run_dir(request, framework_config):
    """Per-test run directory for monitoring output.

    Uses the framework's artifact_dir so results land in
    output/artifacts/gpu_monitored/<test_name>/ alongside other
    framework artifacts.
    """
    test_name = request.node.name
    safe_name = re.sub(r"[^A-Za-z0-9._=-]+", "_", test_name).strip("._")
    d = Path(framework_config.framework.artifact_dir) / "gpu_monitored" / safe_name
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def gpu_monitor(run_dir, rock_dir, gpu_monitor_interval):
    """Factory fixture that returns a Monitor context manager.

    Usage in tests::

        def test_something(gpu_monitor, run_dir):
            with gpu_monitor() as mon:
                # run workload
                pass
            # CSV available at mon.csv_file
    """
    rocm_bin = os.path.join(rock_dir, "bin")
    if rocm_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = rocm_bin + ":" + os.environ.get("PATH", "")

    amd_smi = os.path.join(rock_dir, "bin", "amd-smi")
    if not os.path.isfile(amd_smi):
        amd_smi = "amd-smi"

    def _factory(cu_occupancy: bool = False):
        return Monitor(
            csv_file=run_dir / "power_temp.csv",
            cu_csv=run_dir / "cu_occupancy.csv",
            sample_interval=gpu_monitor_interval,
            enable_cu_occupancy=cu_occupancy,
            amd_smi_path=amd_smi,
        )

    return _factory


@pytest.fixture
def kernel_health_probe():
    """Pre-test kernel health check.

    Returns a function that probes dmesg for critical events.
    Call at the start of a test to verify GPU health before stressing.
    """

    def _probe(lookback_sec: int = 300, strict: bool = False):
        result = _probe_kernel_health(lookback_sec)
        if not result["healthy"]:
            msg = (
                f"Pre-test kernel health check FAILED: "
                f"{len(result['events'])} critical event(s) in last {lookback_sec}s:\n"
                + "\n".join(f"  {e}" for e in result["events"][:5])
            )
            if strict:
                pytest.fail(msg)
            else:
                logger.warning(msg)
        return result

    return _probe


@pytest.fixture
def run_monitored_rvs(
    gpu_monitor,
    run_dir,
    rvs_binary,
    rock_dir,
    ld_path,
    request,
):
    """High-level fixture: run RVS with monitoring, validation, analysis, and report.

    Usage::

        def test_iet(run_monitored_rvs, rvs_find_conf):
            conf = rvs_find_conf("iet_stress.conf")
            result = run_monitored_rvs(conf_file=conf, test_name="iet_stress")
            assert result["passed"]
    """

    def _run(
        conf_file: str,
        test_name: str = "rvs_test",
        timeout: int = 600,
        cu_occupancy: bool = False,
    ) -> dict:
        rvs_bin = rvs_binary
        ld = ld_path["LD_LIBRARY_PATH"]

        cmake_executor = request.config._cmake_executor if hasattr(request.config, "_cmake_executor") else None
        dmesg_before = capture_dmesg(cmake_executor)

        start_time = time.time()
        monitor = gpu_monitor(cu_occupancy=cu_occupancy)
        timed_out = False

        with monitor:
            cmd = f"LD_LIBRARY_PATH={ld} {rvs_bin} -c {conf_file}"
            logger.info("Running RVS: %s", cmd)

            try:
                rc, stdout, stderr = run_cmd_get_stdout_stderr(
                    "bash",
                    "-c",
                    cmd,
                    timeout=timeout,
                    quiet=True,
                )
            except Exception as e:
                timed_out = True
                logger.warning("RVS timed out after %ds: %s", timeout, conf_file)
                rc = -9
                stdout = getattr(e, "stdout", "") or ""
                stderr = getattr(e, "stderr", "") or ""

        duration = time.time() - start_time

        dmesg_after = capture_dmesg(cmake_executor)
        dmesg_new = dmesg_delta(dmesg_before, dmesg_after)

        if timed_out:
            failed = True
            validation_msg = f"TIMEOUT: RVS did not complete within {timeout}s\n"
            _, layers_msg = validate_rvs_result(stdout, stderr, rc, dmesg_new)
            validation_msg += layers_msg
        else:
            failed, validation_msg = validate_rvs_result(stdout, stderr, rc, dmesg_new)

        (run_dir / "stdout.txt").write_text(stdout)
        (run_dir / "stderr.txt").write_text(stderr)
        (run_dir / "validation.txt").write_text(validation_msg)

        summary = {
            "test_name": test_name,
            "conf_file": str(conf_file),
            "passed": not failed,
            "timed_out": timed_out,
            "exit_code": rc,
            "duration_sec": round(duration, 2),
            "validation": validation_msg,
            "samples_collected": count_csv_samples(run_dir / "power_temp.csv"),
        }
        summary_path = run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        analyze_and_write(run_dir, test_name)

        write_report(
            run_dir=run_dir,
            test_name=test_name,
            exit_code=rc,
            duration=int(duration),
        )

        summary["report_path"] = str(run_dir / "report.html")

        logger.info("Test %s: %s (%.1fs)", test_name, "FAIL" if failed else "PASS", duration)
        logger.info("Report: %s", run_dir / "report.html")
        for line in validation_msg.splitlines():
            logger.info("  %s", line)

        return summary

    return _run
