# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
os_deps_plugin.py -- Per-test OS package auto-install.

Declaration styles (resolution priority: function > class > marker > nothing):

    # Function-level decorator (most specific)
    @OS_DEPS("cmake", "libdrm-dev")
    @OS_DEPS({"dnf": "libtiff-devel", "*": "libtiff-dev"}, src="https://repo/")
    def test_something(target_executor): ...

    # Class-level (applies to all methods unless overridden by function decorator)
    class TestFoo:
        __OS_DEPS__ = OS_DEPS("cmake", "git")

    # Marker (backwards-compat, checked when no __OS_DEPS__ found)
    @pytest.mark.os_deps("cmake", src="https://repo/")
    def test_something(target_executor): ...

    # Callable mid-test
    def test_conditional(target_executor, os_deps):
        if "NEEDS_NUMA" in result.stdout:
            os_deps("libnuma-dev")
        os_deps({"dnf": "libtiff-devel", "*": "libtiff-dev"}, src="https://repo/")

Session cache key: (executor_id, str(pkg_spec), src_key)
Same spec+source on same executor is a no-op after first install.
"""

from __future__ import annotations

from collections.abc import Callable
import logging

import pytest

from framework.os_adapter.pkg_manager import PkgSpec, install_packages

logger = logging.getLogger(__name__)

_INSTALLED: set[tuple[str, str, str]] = set()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "os_deps(*pkg_specs, src=None, sources=None): "
        "Install OS packages before this test. Prefer @OS_DEPS decorator or class __OS_DEPS__. "
        "pkg_specs: str or dict({'mgr': 'name', '*': 'fallback'}). "
        "src=: one URL for all managers. sources={'mgr': 'url'}: per-manager URLs. "
        "Linux: apt/dnf/yum/zypper/apk/pacman. Windows: winget/choco/scoop (auto-bootstrapped).",
    )


@pytest.fixture(autouse=True)
def os_deps(request: pytest.FixtureRequest) -> Callable[..., None]:
    """Install OS packages before test (via __OS_DEPS__ or marker) or on demand (callable).

    Yields:
        Callable: install(*pkg_specs, src=None, sources=None)
    """
    try:
        executor = request.getfixturevalue("target_executor")
    except pytest.FixtureLookupError:
        executor = request.getfixturevalue("dry_run_executor")

    is_dry = _is_dry_run(executor)
    exec_id = _executor_id(executor)

    def _install(*pkg_specs: PkgSpec, src: str | None = None, sources: dict[str, str] | None = None) -> None:
        src_cache_key = _make_src_key(src, sources)
        to_install = [s for s in pkg_specs if (exec_id, str(s), src_cache_key) not in _INSTALLED]
        if not to_install:
            return
        try:
            _dispatch_install(executor, to_install, is_dry=is_dry, src=src, sources=sources)
        except RuntimeError as exc:
            pytest.skip(str(exc))
            return
        if not is_dry:
            for spec in to_install:
                _INSTALLED.add((exec_id, str(spec), src_cache_key))

    # Resolve declaration — function > class > marker
    spec = _resolve_os_deps(request)
    if spec is not None:
        _install(*spec.pkg_specs, src=spec.src, sources=spec.sources)
    else:
        # Backwards-compat: check @pytest.mark.os_deps
        marker = request.node.get_closest_marker("os_deps")
        if marker and marker.args:
            _install(*marker.args, src=marker.kwargs.get("src"), sources=marker.kwargs.get("sources"))

    return _install


def _resolve_os_deps(request: pytest.FixtureRequest):
    """Return OsDepsSpec from function or class __OS_DEPS__, or None."""
    return getattr(request.node.function, "__OS_DEPS__", None) or getattr(request.cls, "__OS_DEPS__", None)


def _dispatch_install(executor, pkg_specs, *, is_dry, src, sources):
    from framework.executors.executor_group import NodeExecutorGroup  # avoid circular at module level

    if isinstance(executor, NodeExecutorGroup):
        for exec_ in executor:
            install_packages(exec_, *pkg_specs, dry_run=is_dry, src=src, sources=sources)
    else:
        install_packages(executor, *pkg_specs, dry_run=is_dry, src=src, sources=sources)


def _make_src_key(src: str | None, sources: dict[str, str] | None) -> str:
    parts = []
    if src:
        parts.append(f"src={src}")
    if sources:
        parts.append("sources=" + ",".join(f"{k}:{v}" for k, v in sorted(sources.items())))
    return "|".join(parts)


def _is_dry_run(executor) -> bool:
    return type(executor).__name__ == "DryRunExecutor" or bool(getattr(executor, "_dry_run", False))


def _executor_id(executor) -> str:
    from framework.executors.executor_group import NodeExecutorGroup  # avoid circular at module level

    if isinstance(executor, NodeExecutorGroup):
        first = next(iter(executor), None)
        return _executor_id(first) if first is not None else "localhost"
    return (
        getattr(executor, "node_id", None)
        or getattr(executor, "hostname", None)
        or getattr(executor, "host", None)
        or "localhost"
    )
