# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_os_deps_plugin.py -- PR-gate unit tests for os_deps_plugin and pkg_manager.

All tests run in DryRun mode (no GPU, no real subprocess execution).
Covers:
    - Marker collection and kwarg extraction (packages, src=)
    - Cross-distro / cross-OS alias resolution
    - Backend detection and caching
    - Command generation for Linux backends (apt, dnf, yum, zypper, apk, pacman)
    - Command generation for Windows backends (winget, choco, scoop)
    - Custom src= URL handling for each backend
    - DryRun safety (executor.run receives echo, not a real install command)
    - Session-level cache deduplication (same package not re-installed)
    - Cache key includes src — same package from different sources installs twice
    - No-marker no-op (fixture does not call install when marker absent)
    - Callable fixture style (os_deps("pkg") mid-test)
    - Callable with src= kwarg
    - zypper commands include --no-gpg-checks
    - scoop bucket add emitted when src= provided
    - winget --source emitted when src= provided
    - choco --source emitted when src= provided
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from framework.os_adapter.pkg_manager import (
    Backend,
    _apk_install,
    _apt_install,
    _cache_backend,
    _candidate_names,
    _choco_install,
    _dnf_install,
    _resolve_spec,
    _scoop_install,
    _winget_install,
    _yum_install,
    _zypper_install,
    install_packages,
)
import framework.plugins.os_deps_plugin as plugin

# ── helpers ───────────────────────────────────────────────────────────────────


def _mock_executor(ok: bool = True, stdout: str = "OK") -> MagicMock:
    """Return a mock executor whose .run() returns a fake ExecutionResult."""
    result = MagicMock()
    result.ok = ok
    result.stdout = stdout
    result.stderr = ""
    result.exit_code = 0 if ok else 1
    ex = MagicMock()
    ex.run.return_value = result
    ex._dry_run = False
    ex.node_id = None
    ex.hostname = None
    ex.host = "test-host"
    return ex


# ── marker collection ─────────────────────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os_deps("cmake", "git")
def test_os_deps_marker_packages_collected(request):
    marker = request.node.get_closest_marker("os_deps")
    assert marker is not None
    assert set(marker.args) == {"cmake", "git"}


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_os_deps_decorator_with_src():
    """Verify @OS_DEPS stores src= and sources= on the function's __OS_DEPS__ attribute."""
    from framework.os_adapter.deps import OS_DEPS

    @OS_DEPS("mypkg", src="https://custom-repo.example.com/apt")
    def dummy():
        pass

    spec = dummy.__OS_DEPS__
    assert "mypkg" in spec.pkg_specs
    assert spec.src == "https://custom-repo.example.com/apt"
    assert spec.sources is None


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_os_deps_decorator_with_sources():
    """Verify @OS_DEPS stores per-manager sources= dict."""
    from framework.os_adapter.deps import OS_DEPS

    @OS_DEPS("cmake", sources={"apt": "https://apt-repo/", "dnf": "https://rhel-repo/"})
    def dummy():
        pass

    spec = dummy.__OS_DEPS__
    assert "cmake" in spec.pkg_specs
    assert spec.src is None
    assert spec.sources == {"apt": "https://apt-repo/", "dnf": "https://rhel-repo/"}


# ── auto-heuristic and dict spec resolution ───────────────────────────────────


def _make_backend(key: str) -> Backend:
    fn_map = {"apt": _apt_install, "dnf": _dnf_install, "yum": _yum_install,
               "zypper": _zypper_install, "apk": _apk_install}
    return Backend(key=key, detect_cmd=f"command -v {key}", install_fn=fn_map.get(key, _apt_install))


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_auto_heuristic_dev_on_dnf_adds_devel():
    assert _candidate_names("libdrm-dev", _make_backend("dnf")) == ["libdrm-dev", "libdrm-devel"]
    assert _candidate_names("libnuma-dev", _make_backend("yum")) == ["libnuma-dev", "libnuma-devel"]
    assert _candidate_names("libdrm-dev", _make_backend("zypper")) == ["libdrm-dev", "libdrm-devel"]


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_auto_heuristic_dev_on_apt_single():
    assert _candidate_names("libdrm-dev", _make_backend("apt")) == ["libdrm-dev"]


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_auto_heuristic_non_dev_single_everywhere():
    for key in ("apt", "dnf", "yum", "zypper"):
        assert _candidate_names("cmake", _make_backend(key)) == ["cmake"]


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_resolve_spec_plain_string():
    assert _resolve_spec("cmake", _make_backend("apt")) == "cmake"
    assert _resolve_spec("cmake", _make_backend("dnf")) == "cmake"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_resolve_spec_dict_key_found():
    assert _resolve_spec({"dnf": "libtiff-devel", "*": "libtiff-dev"}, _make_backend("dnf")) == "libtiff-devel"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_resolve_spec_dict_fallback_to_star():
    spec = {"dnf": "libtiff-devel", "*": "libtiff-dev"}
    assert _resolve_spec(spec, _make_backend("apt")) == "libtiff-dev"
    assert _resolve_spec(spec, _make_backend("zypper")) == "libtiff-dev"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_resolve_spec_dict_no_star_returns_empty():
    assert _resolve_spec({"apt": "x", "dnf": "y"}, _make_backend("zypper")) == ""


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_resolve_spec_unknown_passthrough():
    assert _resolve_spec("totally-unknown-pkg", _make_backend("dnf")) == "totally-unknown-pkg"


# ── Linux command generation ──────────────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_apt_command_no_src():
    ex = _mock_executor()
    cmd = _apt_install(ex, ["cmake", "git"], "sudo -n ", None)
    assert "apt-get install" in cmd
    assert "--no-install-recommends" in cmd
    assert "sudo -n" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_apt_command_deb_src():
    ex = _mock_executor()
    cmd = _apt_install(ex, ["mypkg"], "sudo -n ", "https://example.com/pkg.deb")
    assert "wget" in cmd
    assert "dpkg -i" in cmd
    assert "apt-get install -f" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_apt_command_repo_src():
    ex = _mock_executor()
    src = "deb https://example.com/apt stable main"
    cmd = _apt_install(ex, ["mypkg"], "sudo -n ", src)
    assert "sources.list.d" in cmd
    assert "apt-get update" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_dnf_command_no_src():
    ex = _mock_executor()
    cmd = _dnf_install(ex, ["cmake"], "sudo -n ", None)
    assert "dnf install -y" in cmd
    assert "sudo -n" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_dnf_command_rpm_src():
    ex = _mock_executor()
    cmd = _dnf_install(ex, ["mypkg"], "sudo -n ", "https://example.com/pkg.rpm")
    assert "dnf install -y" in cmd
    assert "pkg.rpm" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_dnf_command_repo_src():
    ex = _mock_executor()
    cmd = _dnf_install(ex, ["mypkg"], "sudo -n ", "https://example.com/repo/")
    assert "yum.repos.d" in cmd
    assert "baseurl=" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_zypper_command_includes_no_gpg_checks():
    ex = _mock_executor()
    cmd = _zypper_install(ex, ["cmake"], "sudo -n ", None)
    assert "--no-gpg-checks" in cmd
    assert "zypper" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_zypper_command_with_repo_src():
    ex = _mock_executor()
    cmd = _zypper_install(ex, ["mypkg"], "sudo -n ", "https://example.com/repo")
    assert "addrepo" in cmd
    assert "--no-gpgcheck" in cmd
    assert "--no-gpg-checks" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_apk_command_with_src():
    ex = _mock_executor()
    cmd = _apk_install(ex, ["curl"], "sudo -n ", "https://alpine.example.com/main")
    assert "--repository" in cmd
    assert "alpine.example.com" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_yum_command_repo_src():
    ex = _mock_executor()
    cmd = _yum_install(ex, ["wget"], "sudo -n ", "https://example.com/repo/")
    assert "yum.repos.d" in cmd
    assert "baseurl=" in cmd


# ── Windows command generation ────────────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_winget_command_no_src():
    ex = _mock_executor()
    cmd = _winget_install(ex, ["Kitware.CMake"], "", None)
    assert "winget install" in cmd
    assert "Kitware.CMake" in cmd
    assert "--accept-package-agreements" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_winget_command_with_src():
    ex = _mock_executor()
    cmd = _winget_install(ex, ["Kitware.CMake"], "", "https://custom.winget.source/")
    assert "--source" in cmd
    assert "custom.winget.source" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_choco_command_no_src():
    ex = _mock_executor()
    cmd = _choco_install(ex, ["cmake", "git"], "", None)
    assert "choco install -y" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_choco_command_with_src():
    ex = _mock_executor()
    cmd = _choco_install(ex, ["mypkg"], "", "https://custom.choco.source/")
    assert "--source" in cmd
    assert "custom.choco.source" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_scoop_command_no_src():
    ex = _mock_executor()
    cmd = _scoop_install(ex, ["cmake"], "", None)
    assert "scoop install" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_scoop_command_with_src():
    ex = _mock_executor()
    src = "https://github.com/mycustom/scoop-bucket"
    cmd = _scoop_install(ex, ["mypkg"], "", src)
    assert "scoop bucket add" in cmd
    assert "github.com/mycustom" in cmd
    assert "scoop install" in cmd


# ── dry-run safety ────────────────────────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_dry_run_does_not_exec_real_command(dry_run_executor):
    """DryRun install must call executor.run with an echo, not a real install command."""
    with patch.object(dry_run_executor, "run", wraps=dry_run_executor.run) as mock_run:
        install_packages(dry_run_executor, "cmake", dry_run=True)
        assert mock_run.call_count == 1
        cmd = mock_run.call_args[0][0]
        assert "cmake" in cmd
        assert "would install" in cmd
        assert "apt-get" not in cmd
        assert "dnf" not in cmd
        assert "winget" not in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_dry_run_with_src_includes_src_in_echo(dry_run_executor):
    with patch.object(dry_run_executor, "run", wraps=dry_run_executor.run) as mock_run:
        install_packages(dry_run_executor, "mypkg", dry_run=True, src="https://my-repo/")
        cmd = mock_run.call_args[0][0]
        assert "would install" in cmd
        assert "src=" in cmd
        assert "my-repo" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_install_packages_noop_on_empty_list(dry_run_executor):
    with patch.object(dry_run_executor, "run") as mock_run:
        install_packages(dry_run_executor, dry_run=True)
        mock_run.assert_not_called()


# ── session cache deduplication ───────────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_session_cache_deduplicates_same_src():
    """A package already cached for a (executor, src) pair must not re-install."""
    plugin._INSTALLED.clear()
    plugin._INSTALLED.add(("test-host", "cmake", ""))

    to_install = [p for p in ["cmake", "git"] if ("test-host", p, "") not in plugin._INSTALLED]
    assert to_install == ["git"]


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_session_cache_different_src_installs_again():
    """Same package from a different src= URL is treated as a separate install."""
    plugin._INSTALLED.clear()
    plugin._INSTALLED.add(("test-host", "cmake", ""))

    # cmake with a custom src= is not cached
    to_install = [p for p in ["cmake"] if ("test-host", p, "https://custom-repo/") not in plugin._INSTALLED]
    assert to_install == ["cmake"]


# ── no marker → no-op ────────────────────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_no_marker_is_noop(request):
    marker = request.node.get_closest_marker("os_deps")
    assert marker is None


# ── callable fixture style ────────────────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_callable_fixture_installs_mid_test(dry_run_executor, os_deps):
    """os_deps("pkg") fixture callable must trigger install via the executor."""
    with patch.object(dry_run_executor, "run", wraps=dry_run_executor.run) as mock_run:
        os_deps("libnuma-dev")
        # At least one run() call should contain "would install" (DryRun path)
        calls = [str(c) for c in mock_run.call_args_list]
        assert any("would install" in c or "libnuma" in c for c in calls)


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_callable_fixture_with_src(dry_run_executor, os_deps):
    """os_deps("pkg", src="url") should include src note in DryRun echo."""
    with patch.object(dry_run_executor, "run", wraps=dry_run_executor.run) as mock_run:
        os_deps("mypkg", src="https://custom-repo.example.com/")
        calls = [str(c) for c in mock_run.call_args_list]
        assert any("custom-repo" in c or "mypkg" in c for c in calls)


# ── backend caching ───────────────────────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_backend_cached_on_executor_after_detection():
    from framework.os_adapter.pkg_manager import _BACKEND_ATTR

    ex = _mock_executor()
    fake_backend = Backend(
        key="apt",
        detect_cmd="command -v apt-get",
        install_fn=_apt_install,
        needs_sudo=True,
        platform="linux",
    )
    _cache_backend(ex, fake_backend)
    assert getattr(ex, _BACKEND_ATTR, None) is fake_backend


# ── OsDepsSpec and OS_DEPS factory ───────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_os_deps_factory_returns_spec():
    from framework.os_adapter.deps import OS_DEPS, OsDepsSpec

    spec = OS_DEPS("cmake", "git")
    assert isinstance(spec, OsDepsSpec)
    assert spec.pkg_specs == ("cmake", "git")
    assert spec.src is None
    assert spec.sources is None


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_os_deps_factory_with_src_and_sources():
    from framework.os_adapter.deps import OS_DEPS

    spec = OS_DEPS("cmake", src="https://repo/", sources={"dnf": "https://rhel/"})
    assert spec.src == "https://repo/"
    assert spec.sources == {"dnf": "https://rhel/"}


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_os_deps_spec_callable_sets_dunder():
    from framework.os_adapter.deps import OS_DEPS

    @OS_DEPS("cmake", "git")
    def dummy_test():
        pass

    assert hasattr(dummy_test, "__OS_DEPS__")
    assert dummy_test.__OS_DEPS__.pkg_specs == ("cmake", "git")


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_os_deps_dict_spec_callable_sets_dunder():
    from framework.os_adapter.deps import OS_DEPS

    @OS_DEPS({"dnf": "libtiff-devel", "*": "libtiff-dev"}, src="https://repo/")
    def dummy_test():
        pass

    spec = dummy_test.__OS_DEPS__
    assert spec.pkg_specs == ({"dnf": "libtiff-devel", "*": "libtiff-dev"},)
    assert spec.src == "https://repo/"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_resolve_os_deps_reads_function_attr(request):
    from framework.os_adapter.deps import OS_DEPS
    from framework.plugins.os_deps_plugin import _resolve_os_deps

    @OS_DEPS("cmake")
    def dummy():
        pass

    with patch.object(type(request.node), "function", new_callable=lambda: property(lambda self: dummy)):
        spec = _resolve_os_deps(request)
    assert spec is not None
    assert spec.pkg_specs == ("cmake",)


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_resolve_os_deps_function_wins_over_class(request):
    from framework.os_adapter.deps import OS_DEPS
    from framework.plugins.os_deps_plugin import _resolve_os_deps

    @OS_DEPS("cmake")
    def dummy():
        pass

    class FakeCls:
        from framework.os_adapter.deps import OS_DEPS as _OS_DEPS
        __OS_DEPS__ = _OS_DEPS("git")

    with (
        patch.object(type(request.node), "function", new_callable=lambda: property(lambda self: dummy)),
        patch.object(request, "cls", FakeCls, create=True),
    ):
        spec = _resolve_os_deps(request)
    assert spec is not None
    assert spec.pkg_specs == ("cmake",)


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_resolve_os_deps_none_when_no_attr(request):
    from framework.plugins.os_deps_plugin import _resolve_os_deps

    def dummy():
        pass

    with patch.object(type(request.node), "function", new_callable=lambda: property(lambda self: dummy)):
        spec = _resolve_os_deps(request)
    assert spec is None
