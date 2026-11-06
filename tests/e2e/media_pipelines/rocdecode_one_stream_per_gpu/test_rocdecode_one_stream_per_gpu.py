# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_rocdecode_one_stream_per_gpu.py -- one H.264 stream per GPU partition.

Each mode binary checks the host partition layout, then decodes
``media/sample_stream.h264`` once per partition and checks that the stream's
video-engine time stays on that partition's render node.

Each parameter carries ``partition.spx``, ``partition.dpx``, or ``partition.cpx``.
``partition_plugin`` reads ``current_compute_partition`` and skips a case whose
marker is not the host mode, before the binary starts. The marker does not
change the partition. Exit 2 from the binary is still a skip when the plugin
cannot read a partition file.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/media_pipelines/:
    layer.runtime, ci.nightly, e2e.stack, os.linux
``hw.multi_gpu`` and ``gpu_count("all")`` are set on the test so every partition
on the node is visible. ``runtime.medium`` is declared per function.
"""

from __future__ import annotations

import pytest

# Matches the ctest TIMEOUT in CMakeLists.txt.
_TIMEOUT_SECS = 600

_MODES = [
    pytest.param("rocdecode_spx_one_stream_per_card", id="spx", marks=pytest.mark.partition.spx),
    pytest.param("rocdecode_dpx_one_stream_per_half", id="dpx", marks=pytest.mark.partition.dpx),
    pytest.param("rocdecode_cpx_one_stream_per_slice", id="cpx", marks=pytest.mark.partition.cpx),
]


@pytest.mark.hw.multi_gpu
@pytest.mark.gpu_count("all")
@pytest.mark.runtime.medium
@pytest.mark.parametrize("binary_name", _MODES)
def test_rocdecode_one_stream_per_partition(
    target_executor,
    ld_path: dict,
    rocdecode_one_stream_binaries: dict[str, str],
    binary_name: str,
):
    """Run one mode binary against every GPU visible on the node.

    A host passes the binary whose mode matches its partition setting and skips
    the other two. Skip means no GPU, a different mode, or a layout the binary
    does not score.
    """
    binary = rocdecode_one_stream_binaries[binary_name]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld_path['LD_LIBRARY_PATH']} {binary}",
        timeout=_TIMEOUT_SECS,
    )
    if result.exit_code == 2:
        tail = (result.stdout or "").strip()[-1500:]
        pytest.skip(f"{binary_name} skipped (exit 2): host partition mode does not match.\n{tail}")
    assert result.ok, (
        f"{binary_name} failed (exit={result.exit_code}):\n"
        f"stdout: {(result.stdout or '')[-2000:]}\nstderr: {(result.stderr or '')[-500:]}"
    )
