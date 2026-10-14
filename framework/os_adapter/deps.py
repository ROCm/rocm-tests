# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
deps.py -- Public API for declaring OS package dependencies on tests.

Usage:
    from framework.os_adapter.deps import OS_DEPS

    # Function-level decorator
    @OS_DEPS("cmake", "libdrm-dev")
    def test_something(target_executor): ...

    # With dict spec and sources
    @OS_DEPS({"dnf": "libtiff-devel", "*": "libtiff-dev"}, src="https://repo/")
    def test_tiff(target_executor): ...

    # Class-level (applies to all methods)
    class TestFoo:
        __OS_DEPS__ = OS_DEPS("cmake", "git")

    # Mid-test callable (via fixture)
    def test_conditional(target_executor, os_deps):
        os_deps("libnuma-dev")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class OsDepsSpec:
    """Stores an OS dependency declaration. Callable as a decorator."""

    pkg_specs: tuple
    src: str | None = None
    sources: dict[str, str] | None = None

    def __call__(self, fn: Callable) -> Callable:
        """Apply as decorator — sets fn.__OS_DEPS__ = self and returns fn."""
        fn.__OS_DEPS__ = self  # type: ignore[attr-defined]
        return fn

    def __repr__(self) -> str:
        parts = [repr(p) for p in self.pkg_specs]
        if self.src:
            parts.append(f"src={self.src!r}")
        if self.sources:
            parts.append(f"sources={self.sources!r}")
        return f"OS_DEPS({', '.join(parts)})"


def OS_DEPS(  # noqa: N802
    *pkg_specs: str | dict, src: str | None = None, sources: dict[str, str] | None = None
) -> OsDepsSpec:
    """Declare OS package dependencies for a test.

    Args:
        pkg_specs: One or more package specs. Each is either:
                   - str: package name (same on all managers; -dev auto-retries -devel on rpm)
                   - dict: per-manager names with "*" as fallback
        src:      Optional source URL applied to all managers.
        sources:  Optional per-manager source URLs (overrides src= per manager).

    Returns:
        OsDepsSpec — callable as a decorator or assignable as a class attribute.
    """
    return OsDepsSpec(pkg_specs=pkg_specs, src=src, sources=sources)
