# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_ipc_module_load.py -- HIP IPC + hipModuleLoad regression tests.

These C++ programs validate that ``hipModuleLoad`` completes
reliably after repeated ``hipIpcOpenMemHandle`` imports have created a
high-pressure IPC state inside a process.

Validates:
    1. ipc_alltoall_module_load — multi-rank all-to-all IPC import phase
       followed by repeated ``hipModuleLoad`` / ``hipModuleUnload`` cycles.
       Spawns one child process per visible GPU (fork+exec), each of which
       imports every peer's 64 MiB buffer twice to create repeated IPC
       imports, then runs 50 ``hipModuleLoad`` cycles with a 5-second
       per-call latency budget. Requires >= 2 GPUs; exits 77 (skip) on
       single-GPU systems.

    2. ipc_dup_import_module_load — producer/consumer scenario where a
       stress consumer calls ``hipIpcOpenMemHandle`` 16 times on the same
       handle, runs 200 ``hipModuleLoad`` / ``hipModuleUnload`` cycles under
       background SDMA load, while pressure consumers hold their imports
       live. Requires >= 2 GPUs for the primary cross-device IPC regression
       scenario (stress consumer on GPU 0, pressure consumer on GPU 1).

Both binaries are self-orchestrating: the parent process forks and
re-execs itself with env-var role assignments (IPC_AA_* / IPC_DUP_*);
pytest invokes the binary once and waits for the parent to reap all
children and print a final RESULT line to stderr.

The authoritative pass token is ``RESULT PASS`` in the binary's stderr.
Exit code 77 from the binary maps to ``pytest.skip()`` (insufficient GPUs).

GPU architecture gate:
    Runs on gfx90a/gfx942/gfx950. The arch guard skips on other families 
    so regressions on unrelated GPUs do not generate false alerts.
"""

import pytest

# Tests are targeted for below GPU families 
_SUPPORTED_ARCHS = {"gfx90a", "gfx942", "gfx950"}

# Watchdog timeouts from the original run.sh wrappers, plus a 60-second
# buffer to account for executor overhead and fork+exec startup.
_ALLTOALL_TIMEOUT_SECS = 660  # run.sh watchdog 600 s + 60 s buffer
_DUP_IMPORT_TIMEOUT_SECS = 360  # run.sh watchdog 300 s + 60 s buffer


@pytest.mark.hw.multi_gpu
@pytest.mark.runtime.medium
@pytest.mark.gpu_count("all")
def test_ipc_alltoall_module_load(
    target_executor,
    ld_path: dict,
    ipc_alltoall_module_load_binary: str,
    _ipc_module_load_build_dir: str,
    gpu_arch: str | None,
):
    """Run all-to-all IPC import + hipModuleLoad regression on AMD GPUs.

    Acquires every GPU available on one node via ``hw.multi_gpu`` +
    ``gpu_count("all")`` so ``ROCR_VISIBLE_DEVICES`` exposes the full target
    topology to the binary.  The binary
    self-orchestrates multi-rank execution: the parent forks and re-execs
    itself once per visible GPU (IPC_AA_MODE=child), performs an all-to-all
    ``hipIpcOpenMemHandle`` import phase with 2 imports per peer to
    create repeated IPC imports, then drives 50
    ``hipModuleLoad`` / ``hipModuleUnload`` cycles with a 5 s per-call
    latency budget. The parent waits for all ranks and prints one
    ``RESULT PASS`` or ``RESULT FAIL`` line to stderr.

    Exit code 77 means fewer than 2 GPUs are visible; the test is skipped.
    """
    if gpu_arch and gpu_arch not in _SUPPORTED_ARCHS:
        pytest.skip(
            f"IPC module-load test targeted for gfx90a/gfx942/gfx950; "
            f"got {gpu_arch} — skipping to avoid false alert on unrelated hardware"
        )

    ld = ld_path["LD_LIBRARY_PATH"]
    # The binary opens "noop.hsaco" as a relative path; run it from its own
    # build directory so the code object is found correctly.
    build_dir = _ipc_module_load_build_dir
    binary = ipc_alltoall_module_load_binary

    result = target_executor.run(
        f"sh -c 'cd {build_dir} && " f"env LD_LIBRARY_PATH={ld} {binary}'",
        timeout=_ALLTOALL_TIMEOUT_SECS,
    )

    # Exit 77 = "skip: not enough GPUs" signal from the binary.
    if result.exit_code == 77:
        pytest.skip("ipc_alltoall_module_load requires >= 2 GPUs; fewer visible on this node")

    assert result.ok, (
        f"ipc_alltoall_module_load failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:2000]}"
    )
    # The authoritative pass token is emitted to stderr by the parent.
    assert "RESULT PASS" in result.stderr, (
        f"ipc_alltoall_module_load did not emit 'RESULT PASS':\n" f"stderr: {result.stderr[:3000]}"
    )
    # Confirm no rank hung (a single HANG line is a regression indicator).
    assert "HANG" not in result.stderr, (
        f"ipc_alltoall_module_load detected a rank hang:\n" f"stderr: {result.stderr[:3000]}"
    )


@pytest.mark.hw.multi_gpu
@pytest.mark.runtime.medium
@pytest.mark.gpu_count("all")
def test_ipc_dup_import_module_load(
    target_executor,
    ld_path: dict,
    ipc_dup_import_module_load_binary: str,
    _ipc_module_load_build_dir: str,
    gpu_arch: str | None,
):
    """Run duplicate-import IPC + hipModuleLoad regression on AMD GPUs.

    Acquires every GPU available on one node via ``hw.multi_gpu`` +
    ``gpu_count("all")`` so ``ROCR_VISIBLE_DEVICES`` exposes the full target
    topology to the binary.  The binary
    self-orchestrates a producer/consumer scenario: the parent allocates a
    64 MiB buffer on GPU 0, publishes its IPC handle, then forks and
    re-execs itself as n_gpus * 2 consumers (IPC_DUP_MODE=child).
    Each consumer calls ``hipIpcOpenMemHandle`` 16 times on the same handle.
    The stress consumer drives 200 ``hipModuleLoad`` / ``hipModuleUnload``
    cycles under background SDMA load; pressure consumers hold their imports
    live throughout. The parent prints one ``RESULT PASS`` or ``RESULT FAIL``
    line to stderr.

    Exit code 77 means no GPUs are visible; the test is skipped. Two GPUs
    are required so the cross-device IPC path (stress consumer on GPU 0,
    pressure consumer on GPU 1) — the primary regression scenario — is
    exercised.
    """
    if gpu_arch and gpu_arch not in _SUPPORTED_ARCHS:
        pytest.skip(
            f"IPC module-load test targeted for gfx90a/gfx942/gfx950; "
            f"got {gpu_arch} — skipping to avoid false alert on unrelated hardware"
        )

    ld = ld_path["LD_LIBRARY_PATH"]
    # The binary opens "noop.hsaco" as a relative path; run it from its own
    # build directory so the code object is found correctly.
    build_dir = _ipc_module_load_build_dir
    binary = ipc_dup_import_module_load_binary

    result = target_executor.run(
        f"sh -c 'cd {build_dir} && " f"env LD_LIBRARY_PATH={ld} {binary}'",
        timeout=_DUP_IMPORT_TIMEOUT_SECS,
    )

    # Exit 77 = "skip: insufficient GPUs" signal from the binary.
    if result.exit_code == 77:
        pytest.skip("ipc_dup_import_module_load requires >= 2 GPUs; fewer visible on this node")

    assert result.ok, (
        f"ipc_dup_import_module_load failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:2000]}"
    )
    # The authoritative pass token is emitted to stderr by the parent.
    assert "RESULT PASS" in result.stderr, (
        f"ipc_dup_import_module_load did not emit 'RESULT PASS':\n" f"stderr: {result.stderr[:3000]}"
    )
    # Confirm both stages passed.
    assert "STAGE consumer_imports     FAIL" not in result.stderr, (
        f"ipc_dup_import_module_load: consumer_imports stage failed:\n" f"stderr: {result.stderr[:3000]}"
    )
    assert "STAGE module_load_under_load FAIL" not in result.stderr, (
        f"ipc_dup_import_module_load: module_load_under_load stage failed:\n" f"stderr: {result.stderr[:3000]}"
    )
