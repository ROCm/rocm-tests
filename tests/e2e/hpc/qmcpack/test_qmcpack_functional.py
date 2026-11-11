# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Functional QMCPACK HIP tests.

Builds QMCPACK ``develop`` with HIP offload in four complex/precision
configurations and requires ``ctest -R deterministic`` to pass in each.
MPI stays off. This is not the QMCPACK performance benchmark.

Markers other than ``runtime.soak`` come from the CATEGORY_PROFILE for
``tests/e2e/hpc/qmcpack/`` (``ci.weekly``). Each configuration is its own
test, so one ctest failure does not hide the other three.
"""

import logging
import shlex

import pytest

from tests.e2e.hpc.qmcpack._workload import CONFIGS, CTEST_RUN_TIMEOUT, CTEST_TIMEOUT

logger = logging.getLogger(__name__)

_CTEST_FAIL_BANNER = "The following tests FAILED"
_CPX_ROCBLAS = "ROCBLAS_DEVICE_MEMORY_SIZE=33554432"


def _cpx_rocblas_prefix(target_executor, rock_dir: str) -> str:
    """Return the CPX rocBLAS memory setting, or an empty prefix.

    CPX partitions need a smaller device-memory pool or rocBLAS alloc fails
    inside the deterministic tests. Other partitions leave the default.
    ``amd-smi partition --current`` is the replacement for the removed
    ``rocm-smi --showcomputepartition``.
    """
    amd_smi = shlex.quote(f"{rock_dir}/bin/amd-smi")
    script = (
        f"if [ -x {amd_smi} ]; then bin={amd_smi}; "
        "elif command -v amd-smi >/dev/null 2>&1; then bin=amd-smi; "
        "else exit 1; fi; "
        '"$bin" partition --current'
    )
    probe = target_executor.run(f"bash -lc {shlex.quote(script)}", timeout=60)
    text = (probe.stdout or "") + "\n" + (probe.stderr or "")
    if probe.ok and "CPX" in text:
        return f"{_CPX_ROCBLAS} "
    return ""


def _build_dir_for(qmcpack_functional_build, config_name: str) -> str:
    """Return the build directory for one configuration name."""
    prefix = f"build_{config_name}-"
    matches = [path for path in qmcpack_functional_build.build_dirs if path.rsplit("/", 1)[-1].startswith(prefix)]
    if len(matches) != 1:
        pytest.fail(f"no QMCPACK build directory for {config_name}: {qmcpack_functional_build.build_dirs}")
    return matches[0]


_CONFIG_NAMES = [name for name, _complex, _mixed in CONFIGS]


@pytest.mark.runtime.soak  # type: ignore[attr-defined]
@pytest.mark.parametrize("config_name", _CONFIG_NAMES, ids=_CONFIG_NAMES)
def test_qmcpack_functional(target_executor, rock_dir: str, ld_path: dict, qmcpack_functional_build, config_name: str):
    """One QMCPACK HIP configuration must pass its deterministic ctest."""
    rocblas = _cpx_rocblas_prefix(target_executor, rock_dir)
    ld = ld_path["LD_LIBRARY_PATH"]
    path = f"{rock_dir}/bin:{rock_dir}/llvm/bin:{rock_dir}/lib/llvm/bin"
    build_dir = _build_dir_for(qmcpack_functional_build, config_name)
    name = build_dir.rsplit("/", 1)[-1]
    ctest_log = f"{build_dir}/deterministic.log"
    cmd = (
        f"env ROCM_PATH={shlex.quote(rock_dir)} "
        f"PATH={shlex.quote(path)}:$PATH "
        f"LD_LIBRARY_PATH={shlex.quote(ld)}:${{LD_LIBRARY_PATH:-}} "
        f"{rocblas}"
        f"ctest --test-dir {shlex.quote(build_dir)} -R deterministic "
        f"--output-on-failure --timeout {shlex.quote(str(CTEST_TIMEOUT))} "
        f"--output-log {shlex.quote(ctest_log)}"
    )
    logger.info("QMCPACK %s ctest starting; full log -> %s", name, ctest_log)
    result = target_executor.run(cmd, timeout=CTEST_RUN_TIMEOUT)
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    no_tests = f"QMCPACK {name} ctest ran no tests:\n{output[-2000:]}"
    assert "No tests were found" not in output, no_tests
    assert output.strip(), no_tests
    if _CTEST_FAIL_BANNER in output or not result.ok:
        pytest.fail(
            f"QMCPACK {name} ctest failed (exit={result.exit_code}):\n"
            f"stdout: {output[-4000:]}\nstderr: {(result.stderr or '')[-1000:]}"
        )
