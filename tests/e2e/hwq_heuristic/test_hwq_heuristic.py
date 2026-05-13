# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_hwq_heuristic.py -- HIP CLR hardware-queue selection heuristic validation.

Unit-level validation of the CLR queue selection logic under four controlled
scenarios that exercise balanced load, imbalanced load, sticky queue assignment,
and null stream isolation.  ``DEBUG_HIP_DYNAMIC_QUEUES=1`` is required to
activate the instrumented queue-selection path.

Binary compiled via CMake from:
    tests/e2e/hwq_heuristic/src/hwq_heuristic_test.cpp

Binary output location:
    output/test-binaries/hwq_heuristic/build/hwq_heuristic_test

Markers auto-injected by CATEGORY_PROFILES in taxonomy.py (for this directory):
    hw.gpu, layer.runtime, ci.nightly, e2e.stack, os.linux

``runtime.fast`` is declared explicitly on the test function.

Scenario reference:
    A — balanced load: 4 streams, short kernels → round-robin queue assignment
    B — imbalanced load: stream 0 gets many short kernels, streams 1-3 get long
        kernels → streams 1-3 land on less-busy queues
    C — sticky queue: single stream, submit+sync+submit → timing ratio < 2.0
        (same queue reused).  Warns instead of failing on timing-sensitive hosts.
    D — null stream isolation: null stream + 8 explicit streams concurrent →
        null stream retains its queue; no HIP errors

Pass criteria (from the binary itself):
    Each scenario prints ``PASS scenario X`` on success.
    Scenario C prints ``WARN`` if timing ratio > 2.0 but does not fail.

Prerequisites:
    - ``--rock-dir`` or ``ROCK_DIR`` env var pointing to a ROCm/TheRock install.
    - At least one AMD GPU visible to the test runner.
    - ``cmake`` on PATH (for the build fixture).
"""

import pytest


@pytest.mark.runtime.fast
@pytest.mark.parametrize("scenario", ["A", "B", "C", "D"])
def test_hwq_heuristic(
    target_executor,
    ld_path: dict,
    hwq_heuristic_binary: str,
    scenario: str,
):
    """Validate HIP queue-selection heuristic for the given scenario.

    Args:
        target_executor:       Executor bound to the allocated GPU.
        ld_path:               ``LD_LIBRARY_PATH`` dict for ROCm libs.
        hwq_heuristic_binary:  Path to the compiled binary.
        scenario:              One of ``"A"``, ``"B"``, ``"C"``, ``"D"``.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} DEBUG_HIP_DYNAMIC_QUEUES=1 " f"{hwq_heuristic_binary} --scenario={scenario}"
    )
    assert result.ok, (
        f"hwq_heuristic scenario {scenario} failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
