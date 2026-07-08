# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RCCL concurrent-collectives stress harness.

The first-party MIT harness runs one worker thread per visible GPU and overlaps
collectives on independent streams.  Both modes are kept: ``sanity`` for nightly
coverage and ``weekly`` for the heavier soak run.  Success is a clean exit plus
``Overall: PASSED``.
"""

import re

import pytest

_PASS_RE = re.compile(r"Overall:\s*PASSED", re.IGNORECASE)


def _run_concurrent_collectives(
    *,
    target_executor,
    ld_path: dict,
    concurrent_collectives_binary: str,
    mode: str,
    timeout: float,
) -> None:
    """Run the concurrent_collectives harness in the requested mode."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {concurrent_collectives_binary} {mode}",
        timeout=timeout,
    )
    assert result.ok, (
        f"concurrent_collectives {mode} failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:3000]}\nstderr: {result.stderr[:1000]}"
    )
    assert _PASS_RE.search(result.stdout), (
        f"concurrent_collectives {mode} did not report 'Overall: PASSED':\n"
        f"stdout: {result.stdout[:3000]}\nstderr: {result.stderr[:1000]}"
    )


@pytest.mark.gpu_count(2)
@pytest.mark.runtime.medium
def test_concurrent_distributed_collectives(
    target_executor,
    ld_path: dict,
    require_rccl,
    concurrent_collectives_binary: str,
):
    """concurrent_collectives sanity: overlapping collectives must verify (Overall: PASSED).

    The harness enumerates the GPUs the framework exposed via ``ROCR_VISIBLE_DEVICES``
    (one worker thread per visible GPU), so the ``gpu_count(2)`` marker alone drives
    the device count — no explicit count argument is passed.
    """
    _run_concurrent_collectives(
        target_executor=target_executor,
        ld_path=ld_path,
        concurrent_collectives_binary=concurrent_collectives_binary,
        mode="sanity",
        timeout=1800,
    )


@pytest.mark.gpu_count(2)
@pytest.mark.ci.weekly
@pytest.mark.runtime.soak
def test_concurrent_distributed_collectives_weekly(
    target_executor,
    ld_path: dict,
    require_rccl,
    concurrent_collectives_binary: str,
):
    """concurrent_collectives weekly: heavier legacy WEEKLY workload must verify."""
    _run_concurrent_collectives(
        target_executor=target_executor,
        ld_path=ld_path,
        concurrent_collectives_binary=concurrent_collectives_binary,
        mode="weekly",
        timeout=7200,
    )
