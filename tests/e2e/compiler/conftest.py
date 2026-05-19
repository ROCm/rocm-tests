# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- Compiler area fixtures for tests/e2e/compiler/.

Binary registry
---------------
Every C++ source file compiled in this area is declared in ``_SPECS`` as a
``CompileSpec`` entry.  Adding a new binary is two steps:

    1. Add one ``CompileSpec`` entry to ``_SPECS``.
    2. Add one ``@pytest.fixture(scope="session")`` that calls ``_build()``.

All per-binary compile options (std, opt, arch, flags, include_dirs) live in
``_SPECS`` — never scattered across test files.  Shared binaries (one .cpp
used by multiple test_*.py files) appear as a single fixture here; pytest
session scope ensures they are compiled exactly once regardless of how many
test files declare them.

Generic flags string
--------------------
``CompileSpec.flags`` accepts any extra compiler flags as a plain string
(e.g. ``"-O3 -ffast-math -DENABLE_LOGGING"``).  ``_build()`` calls
``shlex.split()`` internally before forwarding to ``compile_binary``.  This
avoids list literals at the call site and matches the ergonomics of writing a
shell variable or Makefile ``CXXFLAGS``.

Multi-binary tests
------------------
A test that needs two compiled binaries simply declares both fixtures in its
parameter list.  Because every fixture here is session-scoped, a binary
compiled for one test is already cached when a second test requests it — no
extra compilation cost.

Compilation / execution separation
-----------------------------------
``compile_binary`` delegates to ``BinaryBuilder`` (framework.builder.binary_builder),
which runs hipcc in a CPU-only subprocess.  All GPU device-selection env vars
(HIP_VISIBLE_DEVICES, ROCR_VISIBLE_DEVICES, …) are stripped so the compiler is
never bound to a specific GPU ordinal.

The compiled binary is executed on the GPU by test functions via
``target_executor`` (which injects ROCR_VISIBLE_DEVICES automatically).

Layout
------
tests/e2e/compiler/
├── src/
│   └── llvm_memIntrinsic_stress.cpp   ← unique: test_llvm.py
├── conftest.py  (this file)
└── test_llvm.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import shlex

import pytest

logger = logging.getLogger(__name__)

_SUBDIR = "compiler"
_COMMON_INCLUDE = "tests/common/include"


# ---------------------------------------------------------------------------
# CompileSpec — per-binary compile options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompileSpec:
    """Compile-time options for a single HIP/C++ binary.

    All source paths are relative to the repo root (where pytest is invoked).
    ``include_dirs`` always includes ``tests/common/include``; extend it for
    binaries that need extra headers.

    Attributes:
        src:          Source file path, relative to repo root.
        output_name:  Binary filename written under ``output/test-binaries/compiler/``.
        std:          C++ standard (default ``"c++17"``).
        opt:          Optimisation flag (default ``"-O2"``).
        arch:         GFX target for ``--offload-arch`` (``None`` = hipcc auto-detect).
        include_dirs: ``-I`` paths.  Defaults to ``["tests/common/include"]``.
        flags:        Extra compiler flags as a single string, e.g.
                      ``"-O3 -ffast-math -DENABLE_LOGGING"``.  Split by
                      ``shlex.split()`` before being forwarded to hipcc.
                      Leave empty (default) when no extra flags are needed.
    """

    src: str
    output_name: str
    std: str = "c++17"
    opt: str = "-O2"
    arch: str | None = None
    include_dirs: list[str] = field(default_factory=lambda: [_COMMON_INCLUDE])
    flags: str = ""


# ---------------------------------------------------------------------------
# Binary registry — one entry per .cpp file in src/
#
# Unique  binaries: src used by exactly one test_*.py file.
# Shared  binaries: src declared by multiple test_*.py files; compiled once.
#
# To change compile options for any binary: edit this table only.
# ---------------------------------------------------------------------------

_SPECS: dict[str, CompileSpec] = {
    "llvm_mem_intrinsic_stress": CompileSpec(
        src="tests/e2e/compiler/src/llvm_memIntrinsic_stress.cpp",
        output_name="llvm_mem_intrinsic_stress",
        std="c++17",
        opt="-O2",
    ),
}


# ---------------------------------------------------------------------------
# Internal helper — eliminates the repetition in every fixture body.
# Extra flags are passed as a plain string in CompileSpec.flags and split
# here so callers never deal with list construction.
# ---------------------------------------------------------------------------


def _build(compile_binary, name: str) -> str:
    """Look up *name* in ``_SPECS`` and delegate to the ``compile_binary`` factory.

    Args:
        compile_binary: Session-scoped factory fixture from ``builder_plugin``.
        name:           Key in ``_SPECS``.

    Returns:
        Path to the compiled binary.
    """
    spec = _SPECS[name]
    return compile_binary(
        src=spec.src,
        output_name=spec.output_name,
        include_dirs=spec.include_dirs,
        std=spec.std,
        opt=spec.opt,
        arch=spec.arch,
        extra_flags=shlex.split(spec.flags) if spec.flags else None,
        subdir=_SUBDIR,
    )


# ---------------------------------------------------------------------------
# Session-scoped binary fixtures
#
# Naming convention: <output_name>_binary
# Adding a new binary: one _SPECS entry + one 2-line fixture below.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def llvm_mem_intrinsic_stress_binary(compile_binary) -> str:
    """Compile llvm_memIntrinsic_stress.cpp → binary path.  Used by test_llvm.py."""
    return _build(compile_binary, "llvm_mem_intrinsic_stress")
