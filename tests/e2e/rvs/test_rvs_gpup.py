# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RVS GPUP -- GPU properties query cross-verified against the KFD topology in sysfs.

Ported from ROCmTest ``tests/TOOLS/RVS/rvs_gpup.py`` plus its parser
``logParser/tests/RVS/rvs_gpup_parse.py``. Unlike PEQT, GPUP runs unprivileged
and emits *no* ``true``/``false`` verdict of its own -- it only prints the
property values it read:

    [RESULT] [356229.476743] [RVS-GPUP-TC1] gpup 9354 simd_count 416
    [RESULT] [356229.476744] [RVS-GPUP-TC1] gpup 9354 0 type 2

The first form is a node property, the second an io_link property where the
leading ``0`` is the link index. The verdict therefore has to be produced by
cross-checking every reported ``<key> <value>`` against the kernel's own
KFD topology under ``/sys/.../kfd/topology/nodes/<node>/``, which is what makes
this a test rather than a dump.

Two corrections against the original parser, both load-bearing:

* The node for a GPU is taken from ``rvs -g``, which reports
  ``GPU[ <node> - <gpu_id> ]``. The original captured only ``<gpu_id>`` and
  assumed listing order matched sequential node indices; on an 8x MI210 host
  ``rvs -g`` is ordered by PCI address and maps to nodes 4,5,3,2,8,9,7,6, so
  that assumption pairs properties with the wrong GPU.
* Properties are compared per node and per io_link. The original accumulated
  every node's properties into one list shared by all GPUs, which degraded the
  check into "appears on *some* GPU" and masked the mapping error above.

Verified on 8x MI210: 1720 property lines, all matching their own node/link.
"""

from __future__ import annotations

import collections
import logging
import pathlib
import re
import shlex

import pytest

from framework.reporting.allure_reporter import report_metric, step

logger = logging.getLogger(__name__)

_CONF_NAME = "gpup_single.conf"
_RVS_DEBUG_LEVEL = 3
_RVS_TIMEOUT = 600.0

_KFD_NODES = "/sys/devices/virtual/kfd/kfd/topology/nodes"

# ``[RESULT] [ ts ] [<action>] gpup <gpu_id> <rest>``. Action names are of the
# form ``RVS-GPUP-TC1``, so hyphens must be allowed -- hence [^\]]+ rather than
# the \w+_\d+ used for PEQT's pcie_act_N.
_RESULT_RE = re.compile(r"\[\s*RESULT\s*\]\s*\[\s*[\d.]+\s*\]\s*\[([^\]]+)\]\s+gpup\s+(\d+)\s+(.*)")
_GPU_LIST_RE = re.compile(r"GPU\[\s*(\d+)\s*-\s*(\d+)\s*\]")
_ACTION_DECL_RE = re.compile(r"^\s*-\s*name\s*:\s*(\S+)", re.MULTILINE)
_RVS_ERROR_RE = re.compile(r"RVS-ERROR.*", re.IGNORECASE)
_ABORT_RE = re.compile(r"\bABORT\b")

# One round trip to collect the whole KFD topology; reading each file separately
# would be hundreds of executor calls against a remote node.
_SYSFS_DUMP_CMD = (
    f"for n in {_KFD_NODES}/*; do "
    'if [ -f "$n/properties" ]; then '
    'echo "##NODE ${n##*/}"; cat "$n/properties"; '
    'for l in "$n"/io_links/*; do '
    'if [ -f "$l/properties" ]; then echo "##LINK ${n##*/} ${l##*/}"; cat "$l/properties"; fi; '
    "done; fi; done"
)


def _parse_sysfs_dump(text: str) -> dict[tuple, set[str]]:
    """Index the topology dump as ``{("node", n): {...}, ("link", n, l): {...}}``.

    Values are the raw ``"<key> <value>"`` lines, which is the exact form RVS
    echoes back, so verification is a set membership test with no reformatting.
    """
    index: dict[tuple, set[str]] = {}
    key: tuple | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("##NODE "):
            key = ("node", int(line.split()[1]))
            index.setdefault(key, set())
        elif line.startswith("##LINK "):
            _, node, link = line.split()
            key = ("link", int(node), int(link))
            index.setdefault(key, set())
        elif key is not None:
            index[key].add(line)
    return index


def _parse_gpup_log(text: str) -> list[tuple]:
    """Return ``(action, gpu_id, sysfs_key_suffix, "<key> <value>")`` per property line."""
    entries: list[tuple] = []
    for line in text.splitlines():
        match = _RESULT_RE.search(line)
        if not match:
            continue
        action, gpu_id, rest = match.group(1), match.group(2), match.group(3).strip()
        tokens = rest.split()
        # io_link rows carry the link index ahead of the key; node properties
        # never begin with a digit, so this is unambiguous.
        if len(tokens) >= 3 and tokens[0].isdigit():
            entries.append((action, gpu_id, ("link", int(tokens[0])), " ".join(tokens[1:])))
        else:
            entries.append((action, gpu_id, ("node",), rest))
    return entries


def _verify_entries(entries: list[tuple], gpu_nodes: dict[str, int], sysfs: dict[tuple, set[str]]) -> tuple:
    """Return ``(mismatches, per_action_counts)`` for every reported property.

    A property is correct only when it appears in the sysfs file belonging to
    *that* GPU's node (and that io_link), not merely somewhere in the topology.
    """
    mismatches: list[str] = []
    per_action: collections.Counter = collections.Counter()
    for action, gpu_id, suffix, key_value in entries:
        per_action[action] += 1
        node = gpu_nodes.get(gpu_id)
        if node is None:
            mismatches.append(f"[{action}] gpu {gpu_id}: not listed by 'rvs -g', cannot verify")
            continue
        sysfs_key = ("node", node) if suffix == ("node",) else ("link", node, suffix[1])
        if key_value not in sysfs.get(sysfs_key, set()):
            mismatches.append(f"[{action}] gpu {gpu_id} node {node} {sysfs_key[0]}: {key_value!r} not in sysfs")
    return mismatches, per_action


def _gpu_node_map(executor, rvs_env: str, binary: str) -> dict[str, int]:
    """Map ``gpu_id -> KFD node index`` from ``rvs -g``."""
    result = executor.run(f"env {rvs_env} {binary} -g", timeout=_RVS_TIMEOUT)
    output = (result.stdout or "") + (result.stderr or "")
    mapping = {gpu_id: int(node) for node, gpu_id in _GPU_LIST_RE.findall(output)}
    assert mapping, f"'rvs -g' listed no supported GPUs (exit={result.exit_code}):\n{output[-2000:]}"
    return mapping


def _read_sysfs_index(executor) -> dict[tuple, set[str]]:
    """Collect and index the KFD topology in a single executor round trip."""
    dump = executor.run(_SYSFS_DUMP_CMD, timeout=_RVS_TIMEOUT)
    sysfs = _parse_sysfs_dump(dump.stdout or "")
    assert sysfs, f"No KFD topology found under {_KFD_NODES}; is amdgpu/kfd loaded?"
    return sysfs


def _run_gpup(executor, rvs_env: str, binary: str, conf_path: str) -> tuple[str, int]:
    """Run GPUP and return its combined output and exit code."""
    cmd = f"env {rvs_env} {binary} -c {conf_path} -d {_RVS_DEBUG_LEVEL}"
    logger.info("Running GPUP: %s", cmd)
    result = executor.run(cmd, timeout=_RVS_TIMEOUT)
    return (result.stdout or "") + (result.stderr or ""), result.exit_code


def _assert_log_sane(output: str, exit_code: int) -> None:
    """Reject crashes and RVS-level errors before interpreting any property."""
    assert output.strip(), f"RVS GPUP produced no output (exit={exit_code})"
    assert not _ABORT_RE.search(output), f"RVS GPUP reported ABORT:\n{output[-2000:]}"
    errors = _RVS_ERROR_RE.findall(output)
    assert not errors, "RVS GPUP logged {} error(s):\n{}".format(len(errors), "\n".join(errors[:10]))


def _report_gpup_metrics(entries: list[tuple], per_action: collections.Counter) -> None:
    """Publish property/GPU coverage so a shrinking check surfaces in reports."""
    covered = len({gpu_id for _, gpu_id, _, _ in entries})
    report_metric("RVS_GPUP_PROPERTIES_CHECKED", float(len(entries)))
    report_metric("RVS_GPUP_GPUS_COVERED", float(covered))
    logger.info("GPUP verified %d properties across %d GPUs; actions: %s", len(entries), covered, dict(per_action))


def _assert_actions_covered(executor, conf_path: str, conf: str, seen: collections.Counter) -> None:
    """Every action the config declares must have produced properties.

    Without this an action that silently ran nothing would pass by reporting no
    contradicting values.
    """
    declared = set(_ACTION_DECL_RE.findall(executor.run(f"cat {conf_path}").stdout or ""))
    missing = sorted(declared - set(seen))
    assert not missing, f"GPUP action(s) declared in {conf} produced no properties: {', '.join(missing)}"


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.fast
def test_rvs_gpup(target_executor, rvs_binary, rvs_find_conf, gpu_conf_dir, rvs_env):
    """Every property GPUP reports must match the KFD topology for that GPU's own node."""
    conf = rvs_find_conf(_CONF_NAME, gpu_conf_dir=gpu_conf_dir)
    binary = shlex.quote(str(pathlib.Path(rvs_binary).resolve()))
    conf_path = shlex.quote(str(pathlib.Path(conf).resolve()))

    with step("Map GPU ids to KFD topology nodes"):
        gpu_nodes = _gpu_node_map(target_executor, rvs_env, binary)
        logger.info("GPU id -> KFD node: %s", gpu_nodes)

    with step("Read the KFD topology from sysfs"):
        sysfs = _read_sysfs_index(target_executor)

    with step(f"Run RVS GPUP ({_CONF_NAME})"):
        output, exit_code = _run_gpup(target_executor, rvs_env, binary, conf_path)

    with step("Cross-verify reported properties against sysfs"):
        _assert_log_sane(output, exit_code)
        entries = _parse_gpup_log(output)
        # Without this an empty or unparseable log would assert vacuously.
        assert entries, (
            f"No GPUP property lines found; expected '[<action>] gpup <gpu_id> <key> <value>' "
            f"from {conf} (exit={exit_code}):\n{output[-2000:]}"
        )

        mismatches, per_action = _verify_entries(entries, gpu_nodes, sysfs)
        _report_gpup_metrics(entries, per_action)

    _assert_actions_covered(target_executor, conf_path, conf, per_action)

    assert not mismatches, "{} of {} GPUP properties disagree with the KFD topology:\n{}".format(
        len(mismatches), len(entries), "\n".join(mismatches[:20])
    )
