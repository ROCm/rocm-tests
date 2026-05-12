# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
scheduling_plugin.py -- Unified resource-aware test scheduling plugin.

Provides resource-aware test ordering and xdist group assignment via ``DynamicScheduler``.
A single ``--schedule-policy`` flag and a single ``pytest_collection_modifyitems``
hook replace earlier per-plugin ordering flags.

CLI flags added here:
    --schedule-policy {resource-most,resource-least}
        Test ordering policy (default: resource-most).
        resource-most: multinode → multi_gpu DESC → single_gpu
        resource-least: single_gpu → multi_gpu ASC → multinode

    --collect-runtimes PATH
        Write per-test wall-clock durations and outcomes to PATH as JSON at session end.
        Used as a seed for future hint-file input; NOT used for scheduling.

    --vram-headroom-gb GB
        VRAM headroom reserved per GPU in gigabytes (default: 2.0).
        Tests annotated with @pytest.mark.gpu_vram(N) are only assigned to GPUs
        where (total_vram_gb - headroom) >= N.  Read by gpu_plugin and remote_node_plugin.

Hook responsibilities:
    pytest_collection_modifyitems -- delegate to DynamicScheduler; no-op when --no-gpu.
    pytest_runtest_logreport      -- collect (nodeid, duration, outcome) for each test call.
    pytest_sessionfinish          -- flush collected runtimes to --collect-runtimes PATH.
"""

from __future__ import annotations

import datetime
import json
import logging

import pytest

logger = logging.getLogger(__name__)

# Module-level accumulator for runtime data (cleared on each session).
_runtimes: list[dict] = []


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--schedule-policy`` and ``--collect-runtimes`` options.

    Args:
        parser: pytest argument parser.
    """
    group = parser.getgroup("rocm-scheduling", "ROCm dynamic test scheduling")
    group.addoption(
        "--schedule-policy",
        choices=["resource-most", "resource-least"],
        default="resource-most",
        help=(
            "Test scheduling policy. "
            "'resource-most' (default): multinode → multi-gpu DESC → single-gpu — "
            "maximises GPU utilisation by giving high-demand tests first dibs on workers; "
            "single-gpu tests fill remaining free slots via xdist worksteal. "
            "'resource-least': single-gpu → multi-gpu ASC → multinode — "
            "maximises time-to-first-result; heavy tests wait until lightweight ones clear."
        ),
    )
    group.addoption(
        "--collect-runtimes",
        default=None,
        metavar="PATH",
        help=(
            "Write per-test wall-clock durations and outcomes to PATH as JSON at session end. "
            "Informational only — not used for scheduling in this release."
        ),
    )
    group.addoption(
        "--vram-headroom-gb",
        action="store",
        type=float,
        default=2.0,
        metavar="GB",
        help=(
            "VRAM headroom reserved per GPU in gigabytes (default: 2.0).  "
            "Tests annotated with @pytest.mark.gpu_vram(N) are only assigned "
            "to GPUs where (total_vram_gb - headroom) >= N."
        ),
    )


# ---------------------------------------------------------------------------
# Collection-time scheduling
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Assign xdist_group markers and sort items according to ``--schedule-policy``.

    No-op when ``--no-gpu`` is active (``config._node_pool`` is ``None``).

    Args:
        config:  The current pytest configuration (provides CLI options + ``_node_pool``).
        items:   Collected test items, modified in-place.
    """
    from framework.scheduling.dynamic_scheduler import DynamicScheduler, SchedulePolicy

    pool = getattr(config, "_node_pool", None)
    if pool is None:
        # --no-gpu mode: no GPU topology available; skip scheduling and xdist_group assignment.
        logger.debug("scheduling_plugin: no node pool (--no-gpu?); skipping scheduling")
        return

    policy_value = config.getoption("--schedule-policy", default="resource-most")
    policy = SchedulePolicy(policy_value)

    scheduler = DynamicScheduler(pool=pool, policy=policy)
    scheduler.schedule(items)

    recommended = scheduler.recommended_workers()
    logger.info(
        "scheduling_plugin [%s]: %d items scheduled; recommended -n %d",
        policy.value,
        len(items),
        recommended,
    )

    # Print a banner when GPU tests will run sequentially despite multiple available slots.
    numprocesses = getattr(config.option, "numprocesses", None)
    parallel_active = numprocesses is not None and str(numprocesses) not in ("0", "no", "")
    gpu_item_count = sum(1 for i in items if any(m.name in ("hw.gpu", "hw.multi_gpu") for m in i.iter_markers()))
    if recommended > 1 and gpu_item_count > 0 and not parallel_active:
        print(
            f"\n[rocm-test] WARNING: {gpu_item_count} GPU test(s) will run SEQUENTIALLY "
            f"on a {recommended}-GPU node.\n"
            f"           Add -n {recommended} to run in parallel (one test per GPU).\n"
            f"           Requires: pip install pytest-xdist\n"
        )


# ---------------------------------------------------------------------------
# Runtime data collection
# ---------------------------------------------------------------------------


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Accumulate per-test wall-clock duration and outcome after each test call.

    Only records the ``call`` phase (not setup or teardown).

    Args:
        report: pytest test report for the current phase.
    """
    if report.when != "call":
        return

    _runtimes.append(
        {
            "nodeid": report.nodeid,
            "duration_secs": round(report.duration, 3),
            "outcome": report.outcome.upper(),
        }
    )


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Flush accumulated runtimes to ``--collect-runtimes PATH`` as JSON.

    No-op when ``--collect-runtimes`` was not given or no tests were recorded.

    Args:
        session: The current pytest session (provides access to config + options).
    """
    global _runtimes

    path = session.config.getoption("--collect-runtimes", default=None)
    if not path or not _runtimes:
        _runtimes = []
        return

    policy_value = session.config.getoption("--schedule-policy", default="resource-most")
    data = {
        "session": {
            "start_ts": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "policy": policy_value,
            "total_tests": len(_runtimes),
        },
        "tests": _runtimes,
    }

    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info("scheduling_plugin: runtime data written to %s (%d tests)", path, len(_runtimes))
    except OSError as exc:
        logger.warning("scheduling_plugin: could not write runtimes to %s: %s", path, exc)
    finally:
        _runtimes = []
