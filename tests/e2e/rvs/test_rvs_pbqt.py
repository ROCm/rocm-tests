# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RVS PBQT -- P2P benchmark and qualification of every GPU-to-GPU link.

Ported from ROCmTest ``tests/TOOLS/RVS/rvs_pbqt.py`` and its parser
``logParser/tests/RVS/rvs_pbqt_parse.py``. PBQT runs unprivileged and needs at
least two GPUs, so a single-GPU host is skipped exactly as the original reported
UNSUPPORTED. For each ordered GPU pair it reports peer reachability and then,
where the action enables it, measured bandwidth::

    [RESULT] [413240.780763] [action_1] p2p [GPU:: 2 - 9354 - ...] [GPU:: 3 - 25466 - ...]
        peers:true distance:40 PCIe:40
    [RESULT] [413241.581084] [action_1] p2p-bandwidth[ 1/56] [GPU:: 2 - ...] [GPU:: 3 - ...]
        bidirectional: true 48.935 GBps duration: 0.9 secs

``peers:false`` is not treated as a failure. It is a legitimate topology answer
for GPUs that genuinely cannot reach each other, and RVS itself still reports the
action as passing; the count is recorded instead so a topology change is visible.

Bandwidth coverage is deliberately not required per pair. Time-bounded actions
stop before every pair is sampled -- on the shipped MI210 config actions 6, 11
and 14 leave 28 to 37 of their 56 pairs at ``(not measured)``, and action 7 sets
``test_bandwidth: false`` so it measures nothing at all. What is enforced is that
every value actually reported is positive, and that the run measured something.

Corrections against the original parser:

* The verdict no longer rests on a condition that cannot fail. The original
  required ``'true' in peers_value or "peers:false" in peers_value``, which is
  satisfied by both ``peers:true`` and ``peers:false`` and so always held; the
  effective test was just a sample count. Peer state and bandwidth validity are
  now checked as separate, meaningful conditions.
* Error detection works. The original's fail pattern ended in ``.w+`` -- a literal
  ``w`` rather than ``\\w+`` -- so ``RVS-ERROR`` lines almost never matched and
  errors passed through silently. Errors now fail the run via the shared check.

Runtime is about two minutes for the shipped MI210 config on 8 GPUs.
"""

from __future__ import annotations

import collections
import logging
import pathlib
import re
import shlex

import pytest

from framework.reporting.allure_reporter import report_metric, step
from tests.e2e.rvs._rvs_log import (
    assert_log_sane,
    assert_summary_passed,
    declared_actions,
    list_gpu_ids,
    parse_summary,
    run_rvs,
)

logger = logging.getLogger(__name__)

_CONF_NAME = "pbqt_single.conf"
_LABEL = "PBQT"
_MIN_GPUS = 2
_RVS_TIMEOUT = 1800.0

_GPU_REF = r"\[GPU::\s*\d+\s*-\s*(\d+)\s*-\s*[^\]]+\]"
_PEER_RE = re.compile(rf"\[([^\]]+)\]\s*p2p\s*{_GPU_REF}\s*{_GPU_REF}\s*peers:(true|false)")
# Only ``[RESULT]`` lines are read. RVS also emits interim ``[INFO  ]`` progress
# lines for the same pairs whose value can still be ``(pending)``, so counting
# those would fold unfinished measurements into the verdict.
_BANDWIDTH_RE = re.compile(
    rf"\[\s*RESULT\s*\].*\[([^\]]+)\]\s*p2p-bandwidth\[[^\]]*\]\s*{_GPU_REF}\s*{_GPU_REF}\s*"
    r"bidirectional:\s*(?:true|false)\s+(\(not measured\)|\(pending\)|-?nan|-?inf|[\d.]+)"
)


def _parse_peers(text: str) -> dict[tuple, bool]:
    """Return ``{(action, gpu_a, gpu_b): peers}`` for every reported pair."""
    peers: dict[tuple, bool] = {}
    for line in text.splitlines():
        if match := _PEER_RE.search(line):
            peers[(match.group(1), match.group(2), match.group(3))] = match.group(4) == "true"
    return peers


def _parse_bandwidth(text: str) -> tuple[list[tuple], int, list[str]]:
    """Return measured samples, the unmeasured count, and any non-finite readings."""
    samples: list[tuple] = []
    skipped = 0
    invalid: list[str] = []
    for line in text.splitlines():
        match = _BANDWIDTH_RE.search(line)
        if not match:
            continue
        action, gpu_a, gpu_b, value = match.groups()
        if value.startswith("("):
            skipped += 1
        elif value.lstrip("-") in ("nan", "inf"):
            invalid.append(f"[{action}] {gpu_a}->{gpu_b}: {value} GBps")
        else:
            samples.append((action, gpu_a, gpu_b, float(value)))
    return samples, skipped, invalid


def _assert_bandwidth_usable(samples: list[tuple], skipped: int, invalid: list[str]) -> None:
    """Reported bandwidths must be finite and positive, and something must be measured."""
    # NaN compares false against every threshold, so it has to be rejected by name
    # rather than left to the positivity check below.
    assert not invalid, "RVS PBQT reported {} non-finite P2P bandwidth value(s):\n{}".format(
        len(invalid), "\n".join(invalid[:10])
    )
    assert samples, (
        f"RVS PBQT measured no P2P bandwidth at all ({skipped} sample(s) reported "
        f"'(not measured)'); the link qualification produced no throughput evidence."
    )
    bad = [f"[{action}] {gpu_a}->{gpu_b}: {gbps} GBps" for action, gpu_a, gpu_b, gbps in samples if gbps <= 0]
    assert not bad, "{} of {} P2P bandwidth sample(s) were not positive:\n{}".format(
        len(bad), len(samples), "\n".join(bad[:10])
    )


def _assert_actions_probed(peers: dict[tuple, bool], declared: set[str]) -> None:
    """Every declared action must have probed at least one GPU pair."""
    probed = {action for action, _, _ in peers}
    missing = sorted(declared - probed)
    assert not missing, f"RVS PBQT action(s) probed no GPU pair: {', '.join(missing)}"


def _qualify_bandwidth(output: str, peers: dict[tuple, bool]) -> None:
    """Parse, report and qualify the P2P bandwidth samples in one pass."""
    samples, skipped, invalid = _parse_bandwidth(output)
    _report_pbqt_metrics(peers, samples, skipped)
    _assert_bandwidth_usable(samples, skipped, invalid)


def _report_pbqt_metrics(peers: dict[tuple, bool], samples: list[tuple], skipped: int) -> None:
    """Publish link and bandwidth coverage so a topology or throughput change shows up."""
    unreachable = sorted(f"[{action}] {gpu_a}->{gpu_b}" for (action, gpu_a, gpu_b), ok in peers.items() if not ok)
    rates = [gbps for _, _, _, gbps in samples]
    report_metric("RVS_PBQT_PAIRS_PROBED", float(len(peers)))
    report_metric("RVS_PBQT_PAIRS_NOT_PEERS", float(len(unreachable)))
    report_metric("RVS_PBQT_BANDWIDTH_SAMPLES", float(len(samples)))
    if rates:
        report_metric("RVS_PBQT_BANDWIDTH_MAX_GBPS", max(rates), "GBps")
        report_metric("RVS_PBQT_BANDWIDTH_MIN_GBPS", min(rates), "GBps")
    logger.info(
        "PBQT: %d pair probe(s) across %d action(s); %d bandwidth sample(s) (%.2f-%.2f GBps), %d not measured",
        len(peers),
        len(collections.Counter(action for action, _, _ in peers)),
        len(samples),
        min(rates) if rates else 0.0,
        max(rates) if rates else 0.0,
        skipped,
    )
    if unreachable:
        logger.warning(
            "%d GPU pair(s) reported peers:false, which RVS still passes but indicates "
            "the pairs cannot reach each other: %s",
            len(unreachable),
            ", ".join(unreachable[:10]),
        )


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.medium
def test_rvs_pbqt(target_executor, rvs_binary, rvs_find_conf, gpu_conf_dir, rvs_env):
    """Every PBQT action must qualify its GPU pairs with positive measured bandwidth."""
    conf = rvs_find_conf(_CONF_NAME, gpu_conf_dir=gpu_conf_dir)
    binary = shlex.quote(str(pathlib.Path(rvs_binary).resolve()))
    conf_path = shlex.quote(str(pathlib.Path(conf).resolve()))

    gpu_ids = list_gpu_ids(target_executor, rvs_env, binary)
    if len(gpu_ids) < _MIN_GPUS:
        pytest.skip(
            f"PBQT qualifies GPU-to-GPU links and needs at least {_MIN_GPUS} GPUs; "
            f"RVS reports {len(gpu_ids)} on this host."
        )

    with step(f"Run RVS PBQT ({_CONF_NAME}) across {len(gpu_ids)} GPUs"):
        output, exit_code = run_rvs(target_executor, rvs_env, binary, conf_path, _RVS_TIMEOUT, _LABEL)

    with step("Qualify peer reachability and P2P bandwidth"):
        assert_log_sane(output, exit_code, _LABEL)

        peers = _parse_peers(output)
        # Without this an empty or unparseable log would assert vacuously.
        assert peers, (
            f"No PBQT peer results found; expected '[<action>] p2p [GPU:: ...] [GPU:: ...] "
            f"peers:true|false' lines from {conf} (exit={exit_code}):\n{output[-2000:]}"
        )

        _qualify_bandwidth(output, peers)

    declared = declared_actions(target_executor, conf_path)
    _assert_actions_probed(peers, declared)
    assert_summary_passed(parse_summary(output), declared, _LABEL, conf)
