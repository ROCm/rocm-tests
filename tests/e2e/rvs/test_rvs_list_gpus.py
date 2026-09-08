# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RVS list-GPUs -- cross-check that RVS and the SMI enumerate the same GPUs.

Ported from ROCmTest ``tests/TOOLS/RVS/rvs_list_gpus.py``. This is the only RVS
test that runs no GPU workload and needs no config file: it just asks ``rvs -g``
what RVS can see and compares that against the SMI's own inventory::

    rvs -g       :  0000:03:00.0 - GPU[ 4 - 53592] AMD Instinct MI210
    amd-smi list :  GPU: 0  BDF: 0000:03:00.0  KFD_ID: 53592  NODE_ID: 4

RVS's two bracketed fields are the KFD node id and the KFD id, both of which
``amd-smi list`` reports directly, so the two views are compared on all three
fields they share: the set of KFD ids, and then the node id and BDF of each.

Corrections against the original:

* The comparison is symmetric. The original computed only
  ``rvs_gpu_set.difference(smi_gpu_set)`` and required it to be empty, which
  passes whenever RVS enumerates a *subset* of the GPUs the SMI reports. An RVS
  build that saw three of eight GPUs was therefore reported as PASS -- exactly
  the regression this test exists to catch -- despite the docstring promising
  that "count of GPUs must be the same". Both directions are now checked and
  reported separately, since they mean different things: GPUs missing from RVS
  point at RVS or its KFD access, while extra GPUs point at a stale SMI.
* Node ids and BDFs are compared, not just the id set. The original's SMI map
  stored the SMI's own enumeration index against each KFD id, so it had no node
  to compare and its code comment concluded only KFD ids could be checked.
  ``amd-smi list`` does report ``NODE_ID``, so the stronger check is available:
  a GPU present in both views but attributed to the wrong node would have passed
  the original unnoticed.

``rocm-smi --showid`` is used as a fallback and only carries ``GUID`` (the same
value as ``KFD_ID``), so on that path the id-set check runs alone. If neither SMI
is present there is nothing to cross-check against and the test skips rather than
passing on a single unverified source.
"""

from __future__ import annotations

import logging
import pathlib
import re
import shlex

import pytest

from framework.reporting.allure_reporter import report_metric, step

logger = logging.getLogger(__name__)

_RVS_TIMEOUT = 300.0

# ``0000:03:00.0 - GPU[ 4 - 53592] AMD Instinct MI210``. The BDF prefix is
# optional so a build that omits it still yields the node and KFD id.
_RVS_GPU_RE = re.compile(
    r"(?:([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d+)\s*-\s*)?GPU\[\s*(\d+)\s*-\s*(\d+)\s*\]"
)
_SMI_GPU_RE = re.compile(r"^\s*GPU:\s*(\d+)\s*$")
_SMI_FIELD_RE = re.compile(r"^\s*([A-Z_]+):\s*(\S+)\s*$")
_ROCM_SMI_GUID_RE = re.compile(r"GPU\[(\d+)\]\s*:\s*GUID:\s*(\d+)")


def _parse_rvs_gpus(text: str) -> dict[str, dict]:
    """Return ``{kfd_id: {"node": str, "bdf": str}}`` from ``rvs -g`` output."""
    gpus: dict[str, dict] = {}
    for bdf, node, kfd_id in _RVS_GPU_RE.findall(text):
        gpus[kfd_id] = {"node": node, "bdf": (bdf or "").lower()}
    return gpus


def _parse_amd_smi_list(text: str) -> dict[str, dict]:
    """Return ``{kfd_id: {"node": str, "bdf": str, "index": str}}`` from ``amd-smi list``."""
    gpus: dict[str, dict] = {}
    index = ""
    fields: dict[str, str] = {}

    def flush() -> None:
        if index and (kfd_id := fields.get("KFD_ID")):
            gpus[kfd_id] = {
                "node": fields.get("NODE_ID", ""),
                "bdf": fields.get("BDF", "").lower(),
                "index": index,
            }

    for line in text.splitlines():
        if match := _SMI_GPU_RE.match(line):
            flush()
            index, fields = match.group(1), {}
        elif match := _SMI_FIELD_RE.match(line):
            fields[match.group(1)] = match.group(2)
    flush()
    return gpus


def _parse_rocm_smi_showid(text: str) -> dict[str, dict]:
    """Return ``{guid: {"index": str}}`` from ``rocm-smi --showid`` output.

    ``GUID`` is the same value RVS and amd-smi call the KFD id; no node or BDF is
    reported here, so those checks are skipped on this path.
    """
    return {guid: {"node": "", "bdf": "", "index": index} for index, guid in _ROCM_SMI_GUID_RE.findall(text)}


def _resolve_smi(executor, rock_dir: str, name: str) -> str:
    """Return a runnable path for an SMI tool, preferring the ROCm install."""
    candidate = f"{shlex.quote(rock_dir)}/bin/{name}"
    if executor.run(f"test -x {candidate}").ok:
        return candidate
    return name if executor.run(f"command -v {name}").ok else ""


def _smi_gpu_map(executor, rock_dir: str) -> tuple[dict[str, dict], str]:
    """Return the SMI's GPU inventory and the tool it came from ("" if none ran)."""
    for name, args, parse in (
        ("amd-smi", "list", _parse_amd_smi_list),
        ("rocm-smi", "--showid", _parse_rocm_smi_showid),
    ):
        binary = _resolve_smi(executor, rock_dir, name)
        if not binary:
            continue
        result = executor.run(f"{binary} {args}")
        output = (result.stdout or "") + (result.stderr or "")
        if gpus := parse(output):
            return gpus, name
        logger.warning("%s %s reported no GPUs; trying the next source", name, args)
    return {}, ""


def _assert_ids_match(rvs_gpus: dict[str, dict], smi_gpus: dict[str, dict], source: str) -> None:
    """The two inventories must agree exactly, in both directions."""
    missing = sorted(set(smi_gpus) - set(rvs_gpus), key=int)
    extra = sorted(set(rvs_gpus) - set(smi_gpus), key=int)
    problems = []
    if missing:
        problems.append(
            f"RVS did not enumerate {len(missing)} GPU(s) that {source} reports "
            f"(KFD ids {', '.join(missing)}); RVS or its KFD access is at fault."
        )
    if extra:
        problems.append(
            f"RVS enumerated {len(extra)} GPU(s) that {source} does not report "
            f"(KFD ids {', '.join(extra)}); the SMI view is stale or incomplete."
        )
    assert not problems, "RVS and {} disagree on the GPU inventory ({} vs {} GPUs):\n{}".format(
        source, len(rvs_gpus), len(smi_gpus), "\n".join(problems)
    )


def _assert_topology_matches(rvs_gpus: dict[str, dict], smi_gpus: dict[str, dict], source: str) -> int:
    """Compare node id and BDF for each shared GPU; return the field count checked."""
    mismatches = []
    checked = 0
    for kfd_id in sorted(set(rvs_gpus) & set(smi_gpus), key=int):
        for field in ("node", "bdf"):
            rvs_value, smi_value = rvs_gpus[kfd_id][field], smi_gpus[kfd_id][field]
            # Absent on the rocm-smi path, and optional in the rvs -g line.
            if not rvs_value or not smi_value:
                continue
            checked += 1
            if rvs_value != smi_value:
                mismatches.append(f"KFD {kfd_id}: RVS {field}={rvs_value}, {source} {field}={smi_value}")
    assert not mismatches, "RVS and {} disagree on {} GPU attribute(s):\n{}".format(
        source, len(mismatches), "\n".join(mismatches[:10])
    )
    return checked


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.fast
def test_rvs_list_gpus(target_executor, rvs_binary, rvs_env, rock_dir):
    """RVS and the SMI must report the same GPUs, with matching node ids and BDFs."""
    binary = shlex.quote(str(pathlib.Path(rvs_binary).resolve()))

    with step("Collect the RVS and SMI GPU inventories"):
        rvs_output = target_executor.run(f"env {rvs_env} {binary} -g", timeout=_RVS_TIMEOUT)
        rvs_gpus = _parse_rvs_gpus((rvs_output.stdout or "") + (rvs_output.stderr or ""))
        smi_gpus, source = _smi_gpu_map(target_executor, rock_dir)

    if not source:
        pytest.skip(
            "Neither amd-smi nor rocm-smi is available, so the RVS GPU list has "
            "nothing to be cross-checked against; passing on RVS's own output "
            "alone would verify nothing."
        )

    with step(f"Compare RVS against {source}"):
        assert rvs_gpus, (
            f"RVS reported no GPUs; expected 'GPU[ <node> - <kfd id> ]' entries from "
            f"'rvs -g' while {source} reports {len(smi_gpus)} GPU(s) "
            f"(exit={rvs_output.exit_code}):\n{rvs_output.stdout[-2000:]}"
        )

        _assert_ids_match(rvs_gpus, smi_gpus, source)
        checked = _assert_topology_matches(rvs_gpus, smi_gpus, source)

        report_metric("RVS_LIST_GPUS_RVS_COUNT", float(len(rvs_gpus)))
        report_metric("RVS_LIST_GPUS_SMI_COUNT", float(len(smi_gpus)))
        report_metric("RVS_LIST_GPUS_ATTRS_CHECKED", float(checked))
        logger.info(
            "list-gpus: RVS and %s agree on %d GPU(s) (KFD ids %s); %d node/BDF attribute(s) matched",
            source,
            len(rvs_gpus),
            ", ".join(sorted(rvs_gpus, key=int)),
            checked,
        )

    # Guards against a silent regression to the original's KFD-id-only comparison
    # on a host where amd-smi does supply the richer fields.
    assert checked or source == "rocm-smi", (
        f"No node or BDF attributes were comparable even though {source} normally "
        f"reports them; the cross-check degenerated to the GPU id set alone."
    )
