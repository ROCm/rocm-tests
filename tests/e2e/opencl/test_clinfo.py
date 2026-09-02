# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""test_clinfo.py -- clinfo OpenCL device diagnostic validation.

Runs the clinfo OpenCL diagnostic tool on an AMD GPU node and validates its
output with five checks, in order:

    1. clinfo executes and produces output.
    2. ``Number of devices`` is present and greater than zero.
    3. Every device ``Device Type`` is ``CL_DEVICE_TYPE_GPU``.
    4. Every device ``Name`` starts with ``gfx``.
    5. Every device ``Vendor`` is ``Advanced Micro Devices, Inc.``.

tests/e2e/opencl/ has no category profile, so every required marker is declared
explicitly on the test function.
"""

import pytest

_EXPECTED_VENDOR = "Advanced Micro Devices, Inc."
_EXPECTED_DEVICE_TYPE = "CL_DEVICE_TYPE_GPU"
_NAME_PREFIX = "gfx"


def _collect(stdout: str, key: str) -> list[str]:
    """Return values whose colon-delimited field exactly matches ``key``.

    Args:
        stdout: clinfo standard output.
        key: Exact field name to match (e.g. ``Device Type``).

    Returns:
        One stripped value string per matching line, in output order.
    """
    values: list[str] = []
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        field, _, value = line.partition(":")
        if field.strip() == key:
            values.append(value.strip())
    return values


def _device_count(stdout: str) -> int | None:
    """Return the first ``Number of devices`` value as an int, or None if absent.

    Args:
        stdout: clinfo standard output.

    Returns:
        The parsed device count, or ``None`` when no valid count line exists.
    """
    for raw in _collect(stdout, "Number of devices"):
        try:
            return int(raw.split()[0])
        except (ValueError, IndexError):
            continue
    return None


@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.os.linux
@pytest.mark.runtime.fast
def test_clinfo(target_executor, ld_path: dict):
    """Run clinfo and validate its reported OpenCL GPU devices."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} clinfo")

    # 1. clinfo executes.
    assert result.ok, (
        f"clinfo did not execute (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert result.stdout, "clinfo produced no output"

    # 2. Number of devices is present and greater than zero.
    device_count = _device_count(result.stdout)
    assert device_count is not None, f"clinfo did not report a device count:\n{result.stdout[:1000]}"
    assert device_count > 0, f"clinfo reported zero devices:\n{result.stdout[:1000]}"

    # 3. Every device Device Type is CL_DEVICE_TYPE_GPU.
    device_types = _collect(result.stdout, "Device Type")
    assert len(device_types) == device_count, f"expected {device_count} Device Type entries, got {device_types}"
    assert all(
        t == _EXPECTED_DEVICE_TYPE for t in device_types
    ), f"not all devices are {_EXPECTED_DEVICE_TYPE}: {device_types}"

    # 4. Every device Name starts with gfx.
    names = _collect(result.stdout, "Name")
    assert len(names) == device_count, f"expected {device_count} Name entries, got {names}"
    assert all(n.startswith(_NAME_PREFIX) for n in names), f"not all device names start with {_NAME_PREFIX!r}: {names}"

    # 5. Every device Vendor is Advanced Micro Devices, Inc.
    vendors = _collect(result.stdout, "Vendor")
    assert len(vendors) == device_count, f"expected {device_count} Vendor entries, got {vendors}"
    assert all(v == _EXPECTED_VENDOR for v in vendors), f"not all device vendors are {_EXPECTED_VENDOR!r}: {vendors}"
