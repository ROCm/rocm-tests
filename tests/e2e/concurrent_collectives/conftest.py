# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- Build fixture for tests/e2e/concurrent_collectives/.

Compiles concurrent_collectives.cpp via hipcc against RCCL + HIP libs from
the configured ROCm/TheRock install (``rock_dir``).

Binary layout::

    output/test-binaries/concurrent_collectives/concurrent_collectives

The ``compile_binary`` factory (from ``builder_plugin``) is pre-bound to
``rock_dir`` and ``compiler_build_dir`` and handles incremental builds,
xdist file locking, and compiler output streaming automatically.

RCCL-specific link flags are injected via ``extra_flags``; the builder places
them before ``-o`` in the hipcc command, which clang/hipcc accepts for
linker arguments.
"""

from __future__ import annotations

import pytest

_SUBDIR = "concurrent_collectives"
_SRC = "tests/e2e/concurrent_collectives/src/concurrent_collectives.cpp"
_NAME = "concurrent_collectives"


@pytest.fixture(scope="session")
def concurrent_collectives_binary(compile_binary, rock_dir: str) -> str:
    """Compile concurrent_collectives.cpp via hipcc; return absolute binary path.

    Args:
        compile_binary: Session-scoped factory fixture from ``builder_plugin``.
        rock_dir:       Path to the ROCm/TheRock install (``--rock-dir`` /
                        ``ROCK_DIR``).  Used to locate RCCL headers and libs.

    Returns:
        Absolute path to the compiled ``concurrent_collectives`` binary.
    """
    return compile_binary(
        src=_SRC,
        output_name=_NAME,
        std="c++17",
        opt="-O3",
        include_dirs=["tests/e2e/concurrent_collectives/src"],
        extra_flags=[
            "-Wall",
            "-D__HIP_PLATFORM_AMD__",
            "-isystem",
            f"{rock_dir}/include",
            f"-L{rock_dir}/lib",
            "-lrccl",
            "-lpthread",
            "-lamdhip64",
        ],
        subdir=_SUBDIR,
    )
