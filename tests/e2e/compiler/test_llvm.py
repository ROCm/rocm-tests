# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_llvm.py -- LLVM memory-intrinsic stress test for AMD GPU.

Validates that the LLVM lowering of ``__builtin_memset``, ``__builtin_memcpy``,
and ``__builtin_memmove`` produces correct results under multi-stream,
large-buffer conditions on real AMD GPU hardware.

The test binary is compiled from:
    tests/e2e/compiler/src/llvm_memIntrinsic_stress.cpp

Compiled binary is cached at:
    tests/e2e/compiler/build/llvm_mem_intrinsic_stress

Prerequisites:
    - ``--rock-dir`` CLI flag or ``ROCK_DIR`` env var pointing to a TheRock/ROCm
      installation that provides ``bin/hipcc`` and ``lib/``.
    - Real AMD GPU hardware.

Explicit markers (not in profile):
    runtime.medium  -- Expected duration < 30 minutes (variants and multi-GPU).
    hw.multi_gpu    -- Declared explicitly on multi-GPU test to override profile hw.gpu.
    gpu_count(2)    -- Parametric: requires 2 GPU slots (multi-GPU test only).
"""

import pytest


# Scenarios covered by each parametrized id:
#   multi_thread           -- scenario 2: --multi-thread-enable
#   multi_thread_512t      -- scenario 3: --multi-thread-enable --threads-per-kernel=512
#   multi_thread_64k       -- scenario 4: --kernels-per-stream=64 --multi-thread-enable
#   seed_0                 -- scenario 6: --seed=0
#   multi_thread_seed_1    -- scenario 7: --multi-thread-enable --seed=1
#   multi_thread_64k_seed_42 -- scenario 8: --kernels-per-stream=64 --multi-thread-enable --seed=42
#
# All variants use --buffer-size-2pow=28 (256 MiB) rather than the 32 GiB
# default so they complete within the ci.nightly runtime.medium budget.
_VARIANT_ARGS = [
    pytest.param("--multi-thread-enable", id="multi_thread"),
    pytest.param("--multi-thread-enable --threads-per-kernel=512", id="multi_thread_512t"),
    pytest.param("--kernels-per-stream=64 --multi-thread-enable", id="multi_thread_64k"),
    pytest.param("--seed=0", id="seed_0"),
    pytest.param("--multi-thread-enable --seed=1", id="multi_thread_seed_1"),
    pytest.param("--kernels-per-stream=64 --multi-thread-enable --seed=42", id="multi_thread_64k_seed_42"),
]

# Scenarios requiring two HIP devices (--streams=2 --devices=2):
#   multi_gpu_interleave      -- scenario 5: --streams=2 --devices=2 --interleave-stream-launches --multi-thread-enable
#   multi_gpu_interleave_seed -- scenario 9: --streams=2 --devices=2 --interleave-stream-launches --multi-thread-enable --seed=0
_MULTI_GPU_ARGS = [
    pytest.param(
        "--streams=2 --devices=2 --interleave-stream-launches --multi-thread-enable",
        id="multi_gpu_interleave",
    ),
    pytest.param(
        "--streams=2 --devices=2 --interleave-stream-launches --multi-thread-enable --seed=0",
        id="multi_gpu_interleave_seed",
    ),
]


@pytest.mark.runtime.medium
def test_llvm_mem_intrinsic_stress(
    target_executor,
    ld_path: dict,
    llvm_mem_intrinsic_stress_binary: str,
):
    """Run the LLVM memory-intrinsic stress binary on an AMD GPU.

    Exercises multi-stream large-buffer memset/memcpy/memmove kernels and
    validates results against a host-side golden replay.  Uses
    --buffer-size-2pow=28 (256 MiB) for CI runtime safety.  The binary exits
    non-zero on any mismatch or HIP API error and prints ``[PASS]`` to stdout
    on success.
    """
    ld_library_path = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld_library_path} {llvm_mem_intrinsic_stress_binary} --buffer-size-2pow=28"
    )
    assert result.ok, (
        f"llvm_mem_intrinsic_stress failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\n"
        f"stderr: {result.stderr[:500]}"
    )
    assert "[PASS]" in result.stdout, (
        f"llvm_mem_intrinsic_stress exited 0 but '[PASS]' not found in stdout:\n{result.stdout[:2000]}"
    )


@pytest.mark.runtime.medium
@pytest.mark.parametrize("extra_args", _VARIANT_ARGS)
def test_llvm_mem_intrinsic_stress_variants(
    target_executor,
    ld_path: dict,
    llvm_mem_intrinsic_stress_binary: str,
    extra_args: str,
):
    """Parametrized single-GPU variants of the LLVM memory-intrinsic stress test.

    Covers scenarios 2-4 and 6-8: multi-thread kernels, custom thread counts,
    extended kernel counts, and fixed RNG seeds.  Uses --buffer-size-2pow=28
    (256 MiB) for CI runtime safety.  Profile injects hw.gpu, layer.runtime,
    ci.nightly, e2e.stack, os.linux.
    """
    ld_library_path = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld_library_path} {llvm_mem_intrinsic_stress_binary}"
        f" --buffer-size-2pow=28 {extra_args}"
    )
    assert result.ok, (
        f"llvm_mem_intrinsic_stress_variants [{extra_args}] failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\n"
        f"stderr: {result.stderr[:500]}"
    )
    assert "[PASS]" in result.stdout, (
        f"llvm_mem_intrinsic_stress_variants [{extra_args}] exited 0 but '[PASS]' not found in stdout:\n"
        f"{result.stdout[:2000]}"
    )


@pytest.mark.hw.multi_gpu
@pytest.mark.gpu_count(2)
@pytest.mark.runtime.medium
@pytest.mark.parametrize("extra_args", _MULTI_GPU_ARGS)
def test_llvm_mem_intrinsic_stress_multi_gpu(
    target_executor,
    ld_path: dict,
    llvm_mem_intrinsic_stress_binary: str,
    extra_args: str,
):
    """Multi-GPU variants of the LLVM memory-intrinsic stress test (scenarios 5 and 9).

    Exercises --streams=2 --devices=2 --interleave-stream-launches so that
    stream 0 runs on HIP device 0 and stream 1 on HIP device 1.  Uses
    --buffer-size-2pow=28 (256 MiB) per stream for CI runtime safety.
    hw.multi_gpu overrides the profile's hw.gpu; gpu_count(2) tells
    target_executor to acquire two GPU slots and sets ROCR_VISIBLE_DEVICES=0,1
    before the binary is launched.  The binary's --devices=2 is a separate
    concern from ROCR_VISIBLE_DEVICES: it controls how the C++ harness maps
    streams to HIP device ordinals internally.
    """
    ld_library_path = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld_library_path} {llvm_mem_intrinsic_stress_binary}"
        f" --buffer-size-2pow=28 {extra_args}"
    )
    assert result.ok, (
        f"llvm_mem_intrinsic_stress_multi_gpu [{extra_args}] failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\n"
        f"stderr: {result.stderr[:500]}"
    )
    assert "[PASS]" in result.stdout, (
        f"llvm_mem_intrinsic_stress_multi_gpu [{extra_args}] exited 0 but '[PASS]' not found in stdout:\n"
        f"{result.stdout[:2000]}"
    )

