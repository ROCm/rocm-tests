# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_kfd_gpu_id.py -- DryRun tests for reading the KFD gpu_id in GpuDetector.

KFD keeps gpu_id in topology/nodes/N/gpu_id, not in properties. All tests run
without GPU hardware against a fake KFD tree.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from framework.gpu import detector
from framework.gpu.detector import GpuDetector

pytestmark = [
    pytest.mark.hw.cpu_only,
    pytest.mark.ci.pr,
    pytest.mark.layer.runtime,
    pytest.mark.runtime.fast,
]


@pytest.fixture
def kfd_nodes(tmp_path, monkeypatch):
    """A KFD tree: a CPU node, a GPU in the real layout, one with gpu_id in properties, one with none."""
    nodes = tmp_path / "nodes"
    layout = {
        "0": ("0", "cpu_cores_count 64\ngfx_target_version 0\n"),
        "1": ("41715", "simd_count 1216\ngfx_target_version 90500\nnuma_node 0\n"),
        "2": (None, "gpu_id 22\ngfx_target_version 90402\n"),
        "3": (None, "gfx_target_version 120500\n"),
    }
    for name, (gpu_id, properties) in layout.items():
        (nodes / name).mkdir(parents=True)
        (nodes / name / "properties").write_text(properties)
        if gpu_id is not None:
            (nodes / name / "gpu_id").write_text(f"{gpu_id}\n")
    monkeypatch.setattr(detector, "_KFD_NODES", nodes)
    return nodes


class LocalShell:
    """Stands in for an SshExecutor by running the command on this host."""

    def run(self, cmd, timeout=None):
        proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False, timeout=timeout)
        return SimpleNamespace(exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def test_local_kfd_reads_gpu_id_from_its_own_file(kfd_nodes, tmp_path):
    gpus = GpuDetector(artifact_dir=str(tmp_path))._detect_via_kfd()
    assert [(gpu.index, gpu.arch) for gpu in gpus] == [(0, "gfx950"), (1, "gfx942")]
    assert gpus[0].numa_node == 0


def test_remote_kfd_reads_gpu_id_from_its_own_file(kfd_nodes, tmp_path):
    remote = GpuDetector(artifact_dir=str(tmp_path), ssh_executor=LocalShell())  # type: ignore[arg-type]
    assert [(gpu.index, gpu.arch) for gpu in remote._detect_via_kfd()] == [(0, "gfx950"), (1, "gfx942")]


def test_kfd_diagnostics_print_the_gpu_id_file(kfd_nodes, tmp_path, capsys):
    GpuDetector(artifact_dir=str(tmp_path))._print_kfd_dri_diagnostics()
    out = capsys.readouterr().out
    assert "node 1: gpu_id=41715, gfx_target_version=90500" in out
    assert "node 3: gpu_id=0," in out
