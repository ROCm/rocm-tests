# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prepare the cloned hip-tests MatrixTranspose sample for CRIU checkpoint testing.

A single entry point that, after ``git clone`` of ROCm/hip-tests, patches BOTH files of the
``samples/2_Cookbook/0_MatrixTranspose`` sample in place:

    * ``MatrixTranspose.cpp`` -- add ``<thread>`` / ``<chrono>`` includes and replace the single
      host-side kernel launch with a 100-iteration loop that prints ``Iteration N``, sleeps one
      second, and re-launches the kernel each pass, so the process stays alive and GPU-busy long
      enough to ``criu dump`` it.
    * ``CMakeLists.txt`` -- inject the ROCm device-lib path into ``CMAKE_HIP_FLAGS_INIT`` (before
      the ``project()`` call that enables the HIP language) so the sample builds against TheRock
      and standard ROCm installs alike. The upstream sample already provides the ``../../common``
      include for ``hip_helper.h``, so nothing else is needed.

Both patches are idempotent: re-running on an already-patched tree is a no-op.

Usage:
    python3 patch_matrix_transpose.py <sample-dir>   # dir holding MatrixTranspose.cpp + CMakeLists.txt
"""

from __future__ import annotations

import os
import re
import sys

# --- MatrixTranspose.cpp patch ---------------------------------------------

_IOSTREAM = "#include <iostream>"
_IOSTREAM_WITH_SLEEP = "#include <iostream>\n#include <thread>\n#include <chrono>"

# Replacement launch block: 100 iterations, one per second, re-launching the kernel each pass.
_LOOP = """  // Launching kernel from host
  int z =1;
  while(z <= 100){
      std::cout << "Iteration " << z << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1)); // Sleep for 1 second
      z++;
      hipLaunchKernelGGL(matrixTranspose, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix,
                  gpuMatrix, WIDTH);
  }"""

# Matches the single-launch block: the "... kernel from host" comment followed by the
# hipLaunchKernelGGL(...) call up to its terminating "WIDTH);". Tolerant of the upstream
# "Lauching" typo and of the call being wrapped across several lines.
_LAUNCH_RE = re.compile(
    r"[ \t]*//[^\n]*kernel from host[ \t]*\n[ \t]*hipLaunchKernelGGL\(.*?WIDTH\);",
    re.DOTALL,
)

# --- CMakeLists.txt patch --------------------------------------------------

_HIP_FLAGS_LINE = (
    'set(CMAKE_HIP_FLAGS_INIT "--rocm-path=${ROCM_PATH} '
    '--rocm-device-lib-path=${ROCM_PATH}/lib/llvm/amdgcn/bitcode")'
)
# CMAKE_HIP_FLAGS_INIT must be set BEFORE the project()/enable_language(HIP) call that consumes
# it, so it is inserted immediately above the first project(...) line (ROCM_PATH is already
# resolved by the if(UNIX) block above that line in the upstream CMakeLists).
_PROJECT_RE = re.compile(r"(?m)^(project\()")


def _patch_cpp(path: str) -> None:
    """Add the sleep/relaunch loop to MatrixTranspose.cpp in place; exit non-zero if impossible."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if "while(z <= 100)" in text or "while (z <= 100)" in text:
        print("CPP_ALREADY_PATCHED")
        return
    if "#include <thread>" not in text:
        text = text.replace(_IOSTREAM, _IOSTREAM_WITH_SLEEP, 1)
    text, count = _LAUNCH_RE.subn(_LOOP, text, count=1)
    if count != 1:
        print("CPP_PATCH_FAILED: kernel-launch block not found in MatrixTranspose.cpp", file=sys.stderr)
        sys.exit(1)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print("CPP_PATCH_OK")


def _patch_cmake(path: str) -> None:
    """Inject the ROCm device-lib flag into CMakeLists.txt in place; exit non-zero if impossible."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if "CMAKE_HIP_FLAGS_INIT" in text:
        print("CMAKE_ALREADY_PATCHED")
        return
    text, count = _PROJECT_RE.subn(lambda m: f"{_HIP_FLAGS_LINE}\n\n{m.group(1)}", text, count=1)
    if count != 1:
        print("CMAKE_PATCH_FAILED: project() call not found in CMakeLists.txt", file=sys.stderr)
        sys.exit(1)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print("CMAKE_PATCH_OK")


def main(sample_dir: str) -> None:
    """Patch both sample files under *sample_dir*."""
    _patch_cpp(os.path.join(sample_dir, "MatrixTranspose.cpp"))
    _patch_cmake(os.path.join(sample_dir, "CMakeLists.txt"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: patch_matrix_transpose.py <sample-dir>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
