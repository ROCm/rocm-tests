# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_rocdecode_one_stream_per_gpu.py -- one H.264 stream per GPU partition.

The module runs the SPX cases, then switches the node to DPX and runs those
cases, then switches to CPX. The directory conftest puts the original compute
partition back after the last case, before pytest leaves this folder.

Each case runs one mode binary. The bare command runs ``partition_layout`` and
then the default decode. The other cases pass the ``--xcd``, ``--keep-decoder``,
``--parallel``, ``--touch-all``, and ``smi_one_xcd`` combinations from the README.
``--sequence`` is not included: its card addresses are specific to one machine.

Each parameter carries ``partition.spx``, ``partition.dpx``, or ``partition.cpx``.
The class fixture sets that mode before ``partition_plugin`` reads sysfs, so the
marker confirms the switch. Exit 2 from the binary is still a skip.
"""

from __future__ import annotations

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


def _case(binary: str, args: str, partition: str, case_id: str):
    """One binary invocation: ``args`` is the extra command line, or empty."""
    return pytest.param(binary, args, id=case_id, marks=_PARTITION[partition])


# DPX has two slices, so its --xcd values stay in 0 and 1. CPX follows the
# README table (xcd_all through the reversed 0-7 order) plus the flag examples.
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
]
_CASES += [_case(_CPX, f"--xcd {index}", "cpx", f"cpx-xcd-{index}") for index in range(8)]
_CASES += [
    _case(_CPX, "--xcd 0,1", "cpx", "cpx-xcd-0-1"),
    _case(_CPX, "--xcd 0,2", "cpx", "cpx-xcd-0-2"),
    _case(_CPX, "--xcd 0,3", "cpx", "cpx-xcd-0-3"),
    _case(_CPX, "--xcd 3,0", "cpx", "cpx-xcd-3-0"),
    _case(_CPX, "--xcd 7,6,5,4,3,2,1,0", "cpx", "cpx-xcd-reverse"),
    _case(_CPX, "--xcd 0,3 --keep-decoder", "cpx", "cpx-keep-decoder"),
    _case(_CPX, "--xcd 3,0 --parallel", "cpx", "cpx-parallel"),
    _case(_CPX, "--touch-all", "cpx", "cpx-touch-all"),
    _case(_CPX, "--xcd 2 smi_one_xcd", "cpx", "cpx-smi-one-xcd"),
]


def _cases(partition: str) -> list:
    """Parameters for one compute mode, in list order. Ids look like ``spx-default``."""
    return [case for case in _CASES if str(case.id).startswith(f"{partition}-")]


def _run_case(target_executor, ld_path: dict, binaries: dict[str, str], binary_name: str, args: str) -> None:
    """Run one README command with every partition visible.

    The session recorded the GPU list before this suite switched modes, so the
    executor's device mask is the old topology. Drop it for the binary. The
    test still acquires the node through ``target_executor``.
    """
    binary = shlex.quote(binaries[binary_name])
    command = (
        "env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES "
        f"LD_LIBRARY_PATH={shlex.quote(ld_path['LD_LIBRARY_PATH'])} {binary}"
    )
    if args:
        command = f"{command} {args}"
    result = target_executor.run(command, timeout=_TIMEOUT_SECS)
    if result.exit_code == 2:
        tail = (result.stdout or "").strip()[-1500:]
        pytest.skip(f"{command} skipped (exit 2): host partition mode does not match.\n{tail}")
    assert result.ok, (
        f"{command} failed (exit={result.exit_code}):\n"
        f"stdout: {(result.stdout or '')[-2000:]}\nstderr: {(result.stderr or '')[-500:]}"
    )


_MARKS = [
    pytest.mark.hw.multi_gpu,  # Every partition on every card. Overrides the media profile hw.gpu.
    pytest.mark.gpu_count("all"),
    pytest.mark.runtime.fast,  # One case finishes in under 5 minutes. The mode switch is outside the case.
]


class TestRocdecodeSpx:
    """SPX cases. The fixture runs before the partition plugin reads sysfs."""

    pytestmark = _MARKS

    @pytest.fixture(scope="class", autouse=True)
    def _set_spx(self, rocdecode_set_compute_partition):
        rocdecode_set_compute_partition("spx")

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
    def _set_dpx(self, rocdecode_set_compute_partition):
        rocdecode_set_compute_partition("dpx")

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
    def _set_cpx(self, rocdecode_set_compute_partition):
        rocdecode_set_compute_partition("cpx")

    @pytest.mark.parametrize(("binary_name", "args"), _cases("cpx"))
    def test_rocdecode_one_stream_per_partition(
        self, target_executor, ld_path: dict, rocdecode_one_stream_binaries: dict[str, str], binary_name: str, args: str
    ):
        """Run one CPX README command."""
        _run_case(target_executor, ld_path, rocdecode_one_stream_binaries, binary_name, args)
