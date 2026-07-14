# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build a basic HIP app for SPIR-V and verify the SPIR-V offload bundle + execution.

Ports the legacy ``hip_basic_spirv`` testcase: compile a HIP program with
``--offload-arch=amdgcnspirv``, confirm the produced binary carries an
``amdgcnspirv`` offload bundle (via ``llvm-objdump --offloading``), then run it
and check it still prints its success marker (HIP JIT-compiles the SPIR-V at
load time).
"""

import pytest

from tests.common.spirv import assert_spirv_offload_bundle


@pytest.mark.runtime.fast
def test_hip_basic_spirv(target_executor, ld_path: dict, rock_dir: str, hip_basic_spirv_binary: str):
    """Verify a SPIR-V-targeted HIP app emits a SPIR-V bundle and runs correctly."""
    binary = hip_basic_spirv_binary

    assert_spirv_offload_bundle(target_executor, rock_dir, binary, "hip_basic_spirv")

    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} {binary}")
    assert result.ok, (
        f"SPIR-V HIP app run failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert (
        "PASSED" in result.stdout or "Passed" in result.stdout
    ), f"SPIR-V HIP app ran but did not report success:\n{result.stdout[:2000]}"
