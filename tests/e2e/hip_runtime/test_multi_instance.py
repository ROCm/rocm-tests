# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_multi_instance.py -- concurrent multi-instance HIP execution via HIP_VISIBLE_DEVICES.

Launches two instances of a self-verifying HIP vector-add workload
(``hip_multi_instance_app``) concurrently and asserts both complete correctly:

- ``test_multi_instance_same_gpu``: both instances pinned to the same visible
  device (``HIP_VISIBLE_DEVICES=0``) — verifies multiple processes share one GPU.
- ``test_multi_instance_two_gpu``: one instance per device
  (``HIP_VISIBLE_DEVICES=0`` / ``=1``) — verifies independent placement on two GPUs.

The framework selects/pins the physical GPU(s) via the allocator; the test only
layers ``HIP_VISIBLE_DEVICES`` on top (the feature under test) and never touches
``ROCR_VISIBLE_DEVICES``.

runtime.fast is declared explicitly.
"""

import pytest


def _launch_two(target_executor, ld: str, app: str, dev0: str, dev1: str, tag: str):
    """Run two app instances concurrently on the given visible devices; return combined stdout."""
    log0, log1 = f"/tmp/multi_instance_{tag}_0.log", f"/tmp/multi_instance_{tag}_1.log"
    cmd = (
        f"env LD_LIBRARY_PATH={ld} HIP_VISIBLE_DEVICES={dev0} {app} >{log0} 2>&1 & "
        f"env LD_LIBRARY_PATH={ld} HIP_VISIBLE_DEVICES={dev1} {app} >{log1} 2>&1 & "
        f"wait; echo '---INSTANCE-0---'; cat {log0}; echo '---INSTANCE-1---'; cat {log1}"
    )
    return target_executor.run(cmd)


def _assert_both_passed(result, ctx: str):
    passed = result.stdout.count("multi_instance_app: PASSED")
    assert passed == 2, f"{ctx}: expected 2 PASSED instances, got {passed}:\n{result.stdout[:2500]}"
    assert "multi_instance_app: FAILED" not in result.stdout, f"{ctx}: an instance FAILED:\n{result.stdout[:2500]}"


@pytest.mark.runtime.fast
def test_multi_instance_same_gpu(target_executor, ld_path: dict, hip_multi_instance_app_binary: str):
    """Two application instances share a single GPU via HIP_VISIBLE_DEVICES=0."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = _launch_two(target_executor, ld, hip_multi_instance_app_binary, "0", "0", "same")
    _assert_both_passed(result, "multi_instance same-gpu")


@pytest.mark.runtime.fast
@pytest.mark.hw.multi_gpu
@pytest.mark.gpu_count(2)
def test_multi_instance_two_gpu(target_executor, ld_path: dict, hip_multi_instance_app_binary: str):
    """Two application instances run one-per-GPU via HIP_VISIBLE_DEVICES=0 and =1."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = _launch_two(target_executor, ld, hip_multi_instance_app_binary, "0", "1", "two")
    _assert_both_passed(result, "multi_instance two-gpu")
