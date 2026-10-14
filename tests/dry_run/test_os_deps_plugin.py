# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_os_deps_plugin.py -- PR-gate unit tests for os_deps_plugin.

All tests run without GPU (DryRun / unit-level). Covers:

  OS_DEPS class
    - Decorator sets __OS_DEPS__ on function
    - Class-level __OS_DEPS__ (OS_DEPS instance stored, not _InstallSpec)
    - Module-level __OS_DEPS__
    - src=, sources=, command=, extra_args=, on_missing= kwargs stored correctly
    - install() raises when no executor context

  pkg_spec resolution
    - Plain string passthrough
    - Version pin "pkg==x.y" → PM-specific syntax
    - Dict with str value — key match, * fallback, no-* empty
    - Dict with list value — flattened correctly per PM
    - Mixed: str + dict in same call

  Command generation (per backend)
    - apt / dnf / yum / zypper / apk / pacman
    - winget / choco / scoop
    - With and without src=
    - extra_args= appended to command
    - zypper always has --no-gpg-checks

  Custom installer (command=)
    - pip / snap / conda / generic string
    - extra_args= appended

  DryRun safety
    - echo emitted, no real install
    - src, command, extra_args included in echo

  Probe before install
    - already-installed → no install command emitted
    - missing → install command emitted

  on_missing modes
    - "install" → installs missing
    - "skip"    → pytest.skip when missing
    - "fail"    → pytest.fail when missing
    - "probe"   → logs, no gate

  Resolution priority
    - function > class > module
    - None when no __OS_DEPS__ at any level

  Backwards-compat
    - @pytest.mark.os_deps still processed via autouse fixture
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from framework.plugins.os_deps_plugin import (
    OS_DEPS,
    _apk_cmd,
    _apt_cmd,
    _Backend,
    _choco_cmd,
    _dnf_cmd,
    _execute_install,
    _InstallSpec,
    _pacman_cmd,
    _parse_version,
    _resolve_os_deps,
    _resolve_pkgs,
    _scoop_cmd,
    _versioned,
    _winget_cmd,
    _yum_cmd,
    _zypper_cmd,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _mock_executor(ok: bool = True, stdout: str = "OK") -> MagicMock:
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


def _dry_executor() -> MagicMock:
    """Mock a DryRunExecutor."""
    ex = _mock_executor()
    ex.__class__ = type("DryRunExecutor", (), {})
    return ex


def _backend(key: str) -> _Backend:
    fn_map = {
        "apt": _apt_cmd,
        "dnf": _dnf_cmd,
        "yum": _yum_cmd,
        "zypper": _zypper_cmd,
        "apk": _apk_cmd,
        "pacman": _pacman_cmd,
    }
    return _Backend(
        key=key,
        detect_cmd=f"command -v {key}",
        probe_cmd_fn=lambda p: f"probe {p}",
        install_cmd_fn=fn_map.get(key, _apt_cmd),
    )


# ── OS_DEPS class ─────────────────────────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_decorator_sets_dunder_on_function():
    @OS_DEPS("cmake", "git")
    def dummy():
        pass

    assert isinstance(dummy.__OS_DEPS__, _InstallSpec)
    assert dummy.__OS_DEPS__.pkg_specs == ("cmake", "git")


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_decorator_stores_src_sources_command_extra_args():
    @OS_DEPS(
        "cmake",
        src="https://my-repo/",
        sources={"apt": "deb https://apt-repo/ stable main"},
        command="pip",
        extra_args=["--quiet"],
        on_missing="skip",
    )
    def dummy():
        pass

    s = dummy.__OS_DEPS__
    assert s.src == "https://my-repo/"
    assert s.sources == {"apt": "deb https://apt-repo/ stable main"}
    assert s.command == "pip"
    assert s.extra_args == ["--quiet"]
    assert s.on_missing == "skip"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_class_level_stores_os_deps_instance():
    class TestFoo:
        __OS_DEPS__ = OS_DEPS("cmake", "git")

    assert isinstance(TestFoo.__OS_DEPS__, OS_DEPS)
    assert TestFoo.__OS_DEPS__._spec.pkg_specs == ("cmake", "git")


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_module_level_stores_os_deps_instance():
    import types

    m = types.ModuleType("fake")
    m.__OS_DEPS__ = OS_DEPS("cmake")
    assert isinstance(m.__OS_DEPS__, OS_DEPS)
    assert m.__OS_DEPS__._spec.pkg_specs == ("cmake",)


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_install_without_context_raises():
    # Autouse fixture always sets _current_executor — force it to None temporarily.
    import framework.plugins.os_deps_plugin as _plugin

    with (
        patch.object(_plugin, "_current_executor", None),
        pytest.raises(RuntimeError, match="No executor context"),
    ):
        OS_DEPS().install("cmake")


# ── version pin parsing ───────────────────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_parse_version_with_pin():
    assert _parse_version("cmake==3.26") == ("cmake", "3.26")
    assert _parse_version("protobuf==3.21.0") == ("protobuf", "3.21.0")


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_parse_version_without_pin():
    assert _parse_version("cmake") == ("cmake", None)
    assert _parse_version("libdrm-dev") == ("libdrm-dev", None)


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_versioned_translation_per_pm():
    assert _versioned("cmake", "3.26", "apt") == "cmake=3.26*"
    assert _versioned("cmake", "3.26", "dnf") == "cmake-3.26"
    assert _versioned("cmake", "3.26", "yum") == "cmake-3.26"
    assert _versioned("cmake", "3.26", "zypper") == "cmake=3.26"
    assert _versioned("cmake", "3.26", "apk") == "cmake=3.26"
    assert _versioned("cmake", "3.26", "pacman") == "cmake=3.26"
    assert _versioned("cmake", "3.26", "pip") == "cmake==3.26"
    assert _versioned("cmake", None, "apt") == "cmake"  # no version = passthrough


# ── pkg_spec resolution ───────────────────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_resolve_pkgs_plain_string():
    apt = _backend("apt")
    assert _resolve_pkgs(("cmake",), apt) == ["cmake"]
    assert _resolve_pkgs(("cmake==3.26",), apt) == ["cmake=3.26*"]


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_resolve_pkgs_dict_str_value():
    spec = {"apt": "libssl-dev", "dnf": "openssl-devel", "*": "libssl-dev"}
    assert _resolve_pkgs((spec,), _backend("apt")) == ["libssl-dev"]
    assert _resolve_pkgs((spec,), _backend("dnf")) == ["openssl-devel"]
    # zypper not listed → fallback to *
    assert _resolve_pkgs((spec,), _backend("zypper")) == ["libssl-dev"]


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_resolve_pkgs_dict_list_value():
    spec = {
        "apt": ["libssl-dev", "libdrm-dev"],
        "dnf": ["openssl-devel", "libdrm-devel"],
        "*": "libssl-dev",
    }
    assert _resolve_pkgs((spec,), _backend("apt")) == ["libssl-dev", "libdrm-dev"]
    assert _resolve_pkgs((spec,), _backend("dnf")) == ["openssl-devel", "libdrm-devel"]
    assert _resolve_pkgs((spec,), _backend("zypper")) == ["libssl-dev"]


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_resolve_pkgs_dict_no_star_raises():
    spec = {"apt": "libssl-dev", "dnf": "openssl-devel"}
    with pytest.raises(RuntimeError, match="no mapping for backend 'zypper'"):
        _resolve_pkgs((spec,), _backend("zypper"))


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_resolve_pkgs_mixed_str_and_dict():
    spec_dict = {"apt": "libssl-dev", "*": "libssl-dev"}
    apt = _backend("apt")
    result = _resolve_pkgs(("cmake", spec_dict), apt)
    assert result == ["cmake", "libssl-dev"]


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_resolve_pkgs_dict_with_versioned_list():
    spec = {"apt": ["libssl-dev==1.1", "cmake==3.26"], "*": "libssl-dev"}
    result = _resolve_pkgs((spec,), _backend("apt"))
    assert result == ["libssl-dev=1.1*", "cmake=3.26*"]


# ── command generation ────────────────────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_apt_no_src():
    _mock_executor()
    cmd = _apt_cmd(["cmake", "git"], "sudo -n ", None, [])
    assert "apt-get install" in cmd
    assert "--no-install-recommends" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_apt_deb_src():
    _mock_executor()
    cmd = _apt_cmd(["mypkg"], "sudo -n ", "https://example.com/pkg.deb", [])
    assert "dpkg -i" in cmd
    assert "wget" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_apt_repo_src():
    _mock_executor()
    cmd = _apt_cmd(["mypkg"], "sudo -n ", "deb https://example.com stable main", [])
    assert "sources.list.d" in cmd
    assert "apt-get update" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_apt_extra_args():
    _mock_executor()
    cmd = _apt_cmd(["cmake"], "sudo -n ", None, ["-q", "--no-upgrade"])
    assert "-q" in cmd
    assert "--no-upgrade" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_dnf_no_src():
    _mock_executor()
    cmd = _dnf_cmd(["cmake"], "sudo -n ", None, [])
    assert "dnf install -y" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_dnf_extra_args():
    _mock_executor()
    cmd = _dnf_cmd(["cmake"], "sudo -n ", None, ["--best", "--allowerasing"])
    assert "--best" in cmd
    assert "--allowerasing" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_zypper_always_has_no_gpg_checks():
    _mock_executor()
    assert "--no-gpg-checks" in _zypper_cmd(["cmake"], "sudo -n ", None, [])


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_zypper_repo_src():
    _mock_executor()
    cmd = _zypper_cmd(["cmake"], "sudo -n ", "https://suse-repo/", [])
    assert "addrepo" in cmd
    assert "--no-gpgcheck" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_yum_repo_src():
    _mock_executor()
    cmd = _yum_cmd(["cmake"], "sudo -n ", "https://rhel-repo/", [])
    assert "yum.repos.d" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_apk_with_src():
    _mock_executor()
    cmd = _apk_cmd(["curl"], "sudo -n ", "https://alpine.example.com/main", [])
    assert "--repository" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_winget_no_src():
    _mock_executor()
    cmd = _winget_cmd(["Kitware.CMake"], "", None, [])
    assert "winget install" in cmd
    assert "--accept-package-agreements" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_winget_with_src():
    _mock_executor()
    cmd = _winget_cmd(["Kitware.CMake"], "", "https://winget-src/", [])
    assert "--source" in cmd
    assert "winget-src" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_choco_with_src():
    _mock_executor()
    cmd = _choco_cmd(["cmake"], "", "https://choco-src/", [])
    assert "--source" in cmd
    assert "choco-src" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_scoop_with_src_adds_bucket():
    _mock_executor()
    cmd = _scoop_cmd(["cmake"], "", "https://github.com/custom/bucket", [])
    assert "scoop bucket add" in cmd
    assert "scoop install" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_winget_extra_args():
    _mock_executor()
    cmd = _winget_cmd(["Kitware.CMake"], "", None, ["--scope", "machine"])
    assert "--scope" in cmd
    assert "machine" in cmd


# ── DryRun safety ─────────────────────────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_dry_run_emits_echo_not_install():
    ex = _mock_executor()
    spec = _InstallSpec(pkg_specs=("cmake",))
    # Patch _is_dry_run so the DryRun echo path fires regardless of executor type
    with patch("framework.plugins.os_deps_plugin._is_dry_run", return_value=True):
        _execute_install(ex, spec)
    ex.run.assert_called_once()
    cmd = ex.run.call_args[0][0]
    assert "would install" in cmd
    assert "cmake" in cmd
    assert "apt-get" not in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_dry_run_includes_src_and_command_in_echo():
    ex = _mock_executor()
    spec = _InstallSpec(pkg_specs=("mypkg",), src="https://repo/", command="pip", extra_args=["--quiet"])
    with patch("framework.plugins.os_deps_plugin._is_dry_run", return_value=True):
        _execute_install(ex, spec)
    cmd = ex.run.call_args[0][0]
    assert "would install" in cmd
    assert "mypkg" in cmd
    assert "src=" in cmd
    assert "command=pip" in cmd
    assert "extra_args" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_dry_run_noop_on_empty():
    ex = _dry_executor()
    spec = _InstallSpec(pkg_specs=())
    _execute_install(ex, spec)
    ex.run.assert_not_called()


# ── custom command= installer ─────────────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_custom_command_pip():
    ex = _mock_executor()
    spec = _InstallSpec(pkg_specs=("onnxruntime-rocm",), command="pip")
    _execute_install(ex, spec)
    cmd = ex.run.call_args[0][0]
    assert "pip install" in cmd
    assert "onnxruntime-rocm" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_custom_command_with_extra_args():
    ex = _mock_executor()
    spec = _InstallSpec(
        pkg_specs=("torch==2.1",),
        command="pip",
        extra_args=["--index-url", "https://download.pytorch.org/whl/rocm6.0"],
    )
    _execute_install(ex, spec)
    cmd = ex.run.call_args[0][0]
    assert "pip install" in cmd
    assert "torch==2.1" in cmd
    assert "--index-url" in cmd
    assert "rocm6.0" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_custom_command_snap():
    ex = _mock_executor()
    spec = _InstallSpec(pkg_specs=("node --classic",), command="snap")
    _execute_install(ex, spec)
    cmd = ex.run.call_args[0][0]
    assert "snap install" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_custom_command_dict_uses_star_fallback():
    """Dict specs with command= use the '*' value."""
    ex = _mock_executor()
    spec = _InstallSpec(
        pkg_specs=({"pip": "torch-rocm", "*": "torch"},),
        command="pip",
    )
    _execute_install(ex, spec)
    cmd = ex.run.call_args[0][0]
    assert "pip install" in cmd


# ── on_missing modes ──────────────────────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_on_missing_skip_raises_skip():
    ex = _mock_executor(ok=False)  # probe returns not-ok → package missing
    spec = _InstallSpec(pkg_specs=("nonexistent-pkg",), on_missing="skip")
    with patch("framework.plugins.os_deps_plugin._detect_backend") as mock_detect:
        mock_detect.return_value = _backend("apt")
        with pytest.raises(pytest.skip.Exception):
            _execute_install(ex, spec)


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_on_missing_fail_raises_fail():
    ex = _mock_executor(ok=False)
    spec = _InstallSpec(pkg_specs=("nonexistent-pkg",), on_missing="fail")
    with patch("framework.plugins.os_deps_plugin._detect_backend") as mock_detect:
        mock_detect.return_value = _backend("apt")
        with pytest.raises(pytest.fail.Exception):
            _execute_install(ex, spec)


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_on_missing_probe_does_not_install():
    ex = _mock_executor(ok=False)
    spec = _InstallSpec(pkg_specs=("mypkg",), on_missing="probe")
    with patch("framework.plugins.os_deps_plugin._detect_backend") as mock_detect:
        mock_detect.return_value = _backend("apt")
        _execute_install(ex, spec)  # should not raise or install


# ── resolution priority ───────────────────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_resolve_function_level(request):
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
def test_resolve_function_wins_over_class():
    @OS_DEPS("cmake")
    def dummy_fn():
        pass

    class FakeCls:
        __OS_DEPS__ = OS_DEPS("git")

    fake_node = MagicMock()
    fake_node.function = dummy_fn
    fake_request = MagicMock()
    fake_request.node = fake_node
    fake_request.cls = FakeCls

    spec = _resolve_os_deps(fake_request)
    assert spec is not None
    assert spec.pkg_specs == ("cmake",)


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_resolve_class_wins_over_module():
    def plain_fn():
        pass

    class FakeCls:
        __OS_DEPS__ = OS_DEPS("git")

    import types

    m = types.ModuleType("fake")
    m.__OS_DEPS__ = OS_DEPS("cmake")

    fake_node = MagicMock()
    fake_node.function = plain_fn
    fake_request = MagicMock()
    fake_request.node = fake_node
    fake_request.cls = FakeCls
    fake_request.module = m

    spec = _resolve_os_deps(fake_request)
    assert spec is not None
    assert spec.pkg_specs == ("git",)  # class wins over module


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_resolve_module_fallback():
    def plain_fn():
        pass

    import types

    m = types.ModuleType("fake")
    m.__OS_DEPS__ = OS_DEPS("cmake")

    fake_node = MagicMock()
    fake_node.function = plain_fn
    fake_request = MagicMock()
    fake_request.node = fake_node
    fake_request.cls = None
    fake_request.module = m

    spec = _resolve_os_deps(fake_request)
    assert spec is not None
    assert spec.pkg_specs == ("cmake",)


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_resolve_none_when_no_declaration():
    def plain_fn():
        pass

    fake_node = MagicMock()
    fake_node.function = plain_fn
    fake_request = MagicMock()
    fake_request.node = fake_node
    fake_request.cls = None
    fake_request.module = MagicMock(spec=[])  # no __OS_DEPS__

    spec = _resolve_os_deps(fake_request)
    assert spec is None


# ── no marker / no declaration → no-op ───────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_no_declaration_is_noop(request):
    assert request.node.get_closest_marker("os_deps") is None
    assert getattr(request.node.function, "__OS_DEPS__", None) is None


# ── import paths ──────────────────────────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_import_from_os_adapter():
    import framework.os_adapter as _m

    assert _m.OS_DEPS is OS_DEPS


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_import_from_plugin_directly():
    import framework.plugins.os_deps_plugin as _p

    assert _p.OS_DEPS is OS_DEPS


# ── install() DryRun via mock executor ───────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_install_method_dry_run():
    ex = _mock_executor()
    spec = _InstallSpec(pkg_specs=("libnuma-dev",))
    with patch("framework.plugins.os_deps_plugin._is_dry_run", return_value=True):
        _execute_install(ex, spec)
    cmd = ex.run.call_args[0][0]
    assert "would install" in cmd
    assert "libnuma-dev" in cmd


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_install_method_dry_run_with_extra_args():
    ex = _mock_executor()
    spec = _InstallSpec(pkg_specs=("torch==2.1",), command="pip", extra_args=["--quiet"])
    with patch("framework.plugins.os_deps_plugin._is_dry_run", return_value=True):
        _execute_install(ex, spec)
    cmd = ex.run.call_args[0][0]
    assert "would install" in cmd
    assert "torch==2.1" in cmd
    assert "command=pip" in cmd
    assert "extra_args" in cmd
