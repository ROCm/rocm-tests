# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_rock_install_contents.py -- an extracted ROCm/TheRock install has core contents.

Realizes the "extract + verify contents" half of TMS rock_tar_download_extract_verify
against the ROCm install under ``--rock-dir`` (which is itself an extracted TheRock
tarball) rather than re-downloading the multi-GB tarball on every run: verifies the
expected ROCm binaries and shared libraries are present and, for binaries, runnable.

hw.cpu_only (no GPU); runtime.fast.
"""

import pathlib

import pytest

# At least one entry from each group must be present (names vary by packaging layout).
_CORE_BINARIES = [
    ["bin/rocminfo"],
    ["bin/amdclang++", "lib/llvm/bin/clang++", "bin/hipcc"],
    ["bin/amd-smi", "bin/rocm-smi"],
]
_CORE_LIBRARIES = [
    ["lib/libamdhip64.so"],
    ["lib/libhsa-runtime64.so"],
]


def _first_present(root: pathlib.Path, candidates: list[str]) -> pathlib.Path | None:
    for rel in candidates:
        p = root / rel
        if p.exists() or list(root.glob(rel + "*")):
            return p
    return None


@pytest.mark.runtime.fast
def test_rock_install_has_core_binaries(rock_dir: str):
    """The extracted ROCm tree exposes core executables (rocminfo, a clang, an smi)."""
    root = pathlib.Path(rock_dir)
    missing = [group for group in _CORE_BINARIES if _first_present(root, group) is None]
    assert not missing, f"missing core ROCm binaries under {rock_dir} (need one of each): {missing}"


@pytest.mark.runtime.fast
def test_rock_install_has_core_libraries(rock_dir: str):
    """The extracted ROCm tree ships the core runtime shared libraries."""
    root = pathlib.Path(rock_dir)
    missing = [group for group in _CORE_LIBRARIES if _first_present(root, group) is None]
    assert not missing, f"missing core ROCm shared libraries under {rock_dir} (need one of each): {missing}"


@pytest.mark.runtime.fast
def test_rock_install_binary_runs(target_executor, ld_path: dict, rock_dir: str):
    """A core ROCm binary from the extracted tree executes (rocminfo --help)."""
    root = pathlib.Path(rock_dir)
    rocminfo = root / "bin" / "rocminfo"
    if not rocminfo.exists():
        pytest.skip("rocminfo not present in this ROCm install")
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} {rocminfo} --help")
    assert result.ok, f"rocminfo --help failed (exit={result.exit_code}):\n{result.stdout[:600]}\n{result.stderr[:600]}"
