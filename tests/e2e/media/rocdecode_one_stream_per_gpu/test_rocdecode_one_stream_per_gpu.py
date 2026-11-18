# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_rocdecode_one_stream_per_gpu.py -- one H.264 stream per GPU partition.

The module runs the SPX cases, then switches the node to DPX and runs those
cases, then switches to CPX. ``compute_partition`` from ``partition_plugin``
puts the original compute partition back after the last case, before pytest
leaves this folder.

Each case runs one mode binary. The bare command runs ``partition_layout`` and
then the default decode. The other cases pass the ``--xcd``, ``--keep-decoder``,
``--parallel``, ``--touch-all``, and ``smi_one_xcd`` combinations from the README.
``--sequence`` is not included: its card addresses are specific to one machine.

Each parameter carries ``partition.spx``, ``partition.dpx``, or ``partition.cpx``.
The class fixture calls ``compute_partition.set`` before the skip hook reads
sysfs, so the marker confirms the switch. After that switch, ``no GPU`` is a
failure. Exit 2 is a skip only when a line is the program's
``host mode is other`` text (four-slice / QPX). CPX width is the live slice
count (4, 6, or 8), not a hardcoded 0–7 list.
"""

from __future__ import annotations

import json
import shlex

import pytest

# Matches the ctest TIMEOUT in CMakeLists.txt.
_TIMEOUT_SECS = 600

_SPX = "rocdecode_spx_one_stream_per_card"
_DPX = "rocdecode_dpx_one_stream_per_half"
_CPX = "rocdecode_cpx_one_stream_per_slice"

_PARTITION = {
    "spx": pytest.mark.partition.spx,
    "dpx": pytest.mark.partition.dpx,
    "cpx": pytest.mark.partition.cpx,
}

# After compute_partition.set, skip only the program's exact Other-mode line.
_SKIP_OTHER_PREFIX = "host mode is other"


def _case(binary: str, args: str, partition: str, case_id: str):
    """One binary invocation: ``args`` is the extra command line, or empty."""
    return pytest.param(binary, args, id=case_id, marks=_PARTITION[partition])


# DPX has two slices, so its --xcd values stay in 0 and 1. CPX lists that
# assume XCD 0–3 stay here (valid on 4-, 6-, and 8-XCD SKUs). Per-XCD 0..N-1
# and reverse use the live slice count after the CPX switch.
_CASES = [
    _case(_SPX, "", "spx", "spx-default"),
    _case(_DPX, "", "dpx", "dpx-default"),
    _case(_DPX, "--xcd one", "dpx", "dpx-xcd-one"),
    _case(_DPX, "--xcd 0", "dpx", "dpx-xcd-0"),
    _case(_DPX, "--xcd 1", "dpx", "dpx-xcd-1"),
    _case(_DPX, "--xcd 0,1", "dpx", "dpx-xcd-0-1"),
    _case(_DPX, "--xcd 1,0", "dpx", "dpx-xcd-1-0"),
    _case(_DPX, "--xcd 0,1 --keep-decoder", "dpx", "dpx-keep-decoder"),
    _case(_DPX, "--xcd 1,0 --parallel", "dpx", "dpx-parallel"),
    _case(_DPX, "--touch-all", "dpx", "dpx-touch-all"),
    _case(_CPX, "", "cpx", "cpx-default"),
    _case(_CPX, "--xcd all", "cpx", "cpx-xcd-all"),
    _case(_CPX, "--xcd one", "cpx", "cpx-xcd-one"),
    _case(_CPX, "--xcd reverse", "cpx", "cpx-xcd-reverse"),
    _case(_CPX, "--xcd 0,1", "cpx", "cpx-xcd-0-1"),
    _case(_CPX, "--xcd 0,2", "cpx", "cpx-xcd-0-2"),
    _case(_CPX, "--xcd 0,3", "cpx", "cpx-xcd-0-3"),
    _case(_CPX, "--xcd 3,0", "cpx", "cpx-xcd-3-0"),
    _case(_CPX, "--xcd 0,3 --keep-decoder", "cpx", "cpx-keep-decoder"),
    _case(_CPX, "--xcd 3,0 --parallel", "cpx", "cpx-parallel"),
    _case(_CPX, "--touch-all", "cpx", "cpx-touch-all"),
    _case(_CPX, "--xcd 2 smi_one_xcd", "cpx", "cpx-smi-one-xcd"),
]


def _cpx_slices_per_card(target_executor, ld_path: dict, binaries: dict[str, str]) -> int:
    """HIP devices per card after CPX, from the CPX binary's partition_layout JSON."""
    binary = shlex.quote(binaries[_CPX])
    command = f"LD_LIBRARY_PATH={shlex.quote(ld_path['LD_LIBRARY_PATH'])} {binary} partition_layout --json"
    result = target_executor.run(command, timeout=_TIMEOUT_SECS)
    text = result.stdout or ""
    if not result.ok:
        pytest.fail(
            "could not count CPX slices per card:\n"
            f"stdout: {text[-1000:]}\nstderr: {(result.stderr or '')[-500:]}"
        )
    try:
        payload = json.loads(text[text.find("{") :])
        cards = int(payload["cards"])
        gpus = int(payload["gpus_visible"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        pytest.fail(f"partition_layout --json did not include cards/gpus_visible: {exc}\n{text[-1000:]}")
    if cards < 1 or gpus < 1 or gpus % cards != 0:
        pytest.fail(f"CPX layout is {gpus} GPUs on {cards} cards; cannot derive slices per card")
    return gpus // cards


def _cases(partition: str) -> list:
    """Parameters for one compute mode, in list order. Ids look like ``spx-default``."""
    return [case for case in _CASES if str(case.id).startswith(f"{partition}-")]


def _run_case(target_executor, ld_path: dict, binaries: dict[str, str], binary_name: str, args: str) -> None:
    """Run one README command on the unmasked compute_partition hold."""
    binary = shlex.quote(binaries[binary_name])
    command = f"LD_LIBRARY_PATH={shlex.quote(ld_path['LD_LIBRARY_PATH'])} {binary}"
    if args:
        command = f"{command} {args}"
    result = target_executor.run(command, timeout=_TIMEOUT_SECS)
    text = f"{result.stdout or ''}\n{result.stderr or ''}"
    if result.exit_code == 2:
        lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
        if any(line.startswith(_SKIP_OTHER_PREFIX) for line in lines):
            pytest.skip(f"{command} skipped (exit 2):\n{text.strip()[-1500:]}")
        pytest.fail(
            f"{command} failed (exit 2) after compute_partition.set; "
            "only 'host mode is other' is skippable.\n"
            f"stdout: {(result.stdout or '')[-2000:]}\nstderr: {(result.stderr or '')[-500:]}"
        )
    assert result.ok, (
        f"{command} failed (exit={result.exit_code}):\n"
        f"stdout: {(result.stdout or '')[-2000:]}\nstderr: {(result.stderr or '')[-500:]}"
    )


_MARKS = [
    pytest.mark.hw.multi_gpu,  # Every partition on every card. Overrides the media profile hw.gpu.
    pytest.mark.gpu_count("all"),
    pytest.mark.runtime.fast,  # One case finishes in under 5 minutes. The mode switch is outside the case.
    # compute_partition holds the node; keep every case on one xdist worker.
    pytest.mark.xdist_group("rocdecode_one_stream_per_gpu"),
]


class TestRocdecodeSpx:
    """SPX cases. The fixture runs before the partition plugin reads sysfs."""

    pytestmark = _MARKS

    @pytest.fixture(scope="class", autouse=True)
    def _set_spx(self, compute_partition):
        compute_partition.set("spx")

    @pytest.mark.parametrize(("binary_name", "args"), _cases("spx"))
    def test_rocdecode_one_stream_per_partition(
        self, target_executor, ld_path: dict, rocdecode_one_stream_binaries: dict[str, str], binary_name: str, args: str
    ):
        """Run one SPX README command."""
        _run_case(target_executor, ld_path, rocdecode_one_stream_binaries, binary_name, args)


class TestRocdecodeDpx:
    """DPX cases, after the node has been switched off SPX."""

    pytestmark = _MARKS

    @pytest.fixture(scope="class", autouse=True)
    def _set_dpx(self, compute_partition):
        compute_partition.set("dpx")

    @pytest.mark.parametrize(("binary_name", "args"), _cases("dpx"))
    def test_rocdecode_one_stream_per_partition(
        self, target_executor, ld_path: dict, rocdecode_one_stream_binaries: dict[str, str], binary_name: str, args: str
    ):
        """Run one DPX README command."""
        _run_case(target_executor, ld_path, rocdecode_one_stream_binaries, binary_name, args)


class TestRocdecodeCpx:
    """CPX cases, after the node has been switched off DPX."""

    pytestmark = _MARKS

    @pytest.fixture(scope="class", autouse=True)
    def _set_cpx(self, compute_partition):
        compute_partition.set("cpx")

    @pytest.mark.parametrize(("binary_name", "args"), _cases("cpx"))
    def test_rocdecode_one_stream_per_partition(
        self, target_executor, ld_path: dict, rocdecode_one_stream_binaries: dict[str, str], binary_name: str, args: str
    ):
        """Run one CPX README command."""
        _run_case(target_executor, ld_path, rocdecode_one_stream_binaries, binary_name, args)

    @pytest.mark.partition.cpx
    def test_each_xcd(self, target_executor, ld_path: dict, rocdecode_one_stream_binaries: dict[str, str]):
        """Decode on every CPX slice this SKU actually has."""
        slices = _cpx_slices_per_card(target_executor, ld_path, rocdecode_one_stream_binaries)
        if slices < 1:
            pytest.fail("CPX is set but no PCI functions were found on a compute-partition device")
        for index in range(slices):
            _run_case(target_executor, ld_path, rocdecode_one_stream_binaries, _CPX, f"--xcd {index}")
