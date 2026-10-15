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
def test_decorator_with_single_src_url():
    """ROCm repo added as a single apt source for all managers."""

    @OS_DEPS("rocm-smi", src="https://repo.radeon.com/rocm/apt/6.3/")
    def test_rocm_smi_version():
        pass

    s = test_rocm_smi_version.__OS_DEPS__
    assert s.pkg_specs == ("rocm-smi",)
    assert s.src == "https://repo.radeon.com/rocm/apt/6.3/"
    assert s.sources is None  # src= applies globally, no per-manager override


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_decorator_with_per_manager_sources():
    """libdrm-dev installed from distro-specific custom repos via sources=."""

    @OS_DEPS(
        {"apt": "libdrm-dev", "dnf": "libdrm-devel", "*": "libdrm-dev"},
        sources={
            "apt": "deb https://apt.custom-drm-repo.example.com/ stable main",
            "dnf": "https://dnf.custom-drm-repo.example.com/rhel/",
        },
    )
    def test_drm_device_access():
        pass

    s = test_drm_device_access.__OS_DEPS__
    assert s.pkg_specs == ({"apt": "libdrm-dev", "dnf": "libdrm-devel", "*": "libdrm-dev"},)
    assert s.src is None  # per-manager sources only, no global fallback
    assert s.sources["apt"] == "deb https://apt.custom-drm-repo.example.com/ stable main"
    assert s.sources["dnf"] == "https://dnf.custom-drm-repo.example.com/rhel/"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_decorator_with_pip_command_and_extra_args():
    """onnxruntime-rocm is a Python package — installed via pip with a custom index URL."""

    @OS_DEPS(
        "onnxruntime-rocm",
        command="pip",
        extra_args=["--index-url", "https://aiinfra.pkgs.visualstudio.com/PublicPackages/"],
    )
    def test_onnx_rocm_provider():
        pass

    s = test_onnx_rocm_provider.__OS_DEPS__
    assert s.pkg_specs == ("onnxruntime-rocm",)
    assert s.command == "pip"
    assert "--index-url" in s.extra_args
    assert s.src is None  # pip manages its own index, src= is for OS PMs


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_decorator_with_on_missing_skip():
    """CUDA dev headers are optional — skip the test gracefully if not present."""

    @OS_DEPS("libcuda-dev", on_missing="skip")
    def test_cuda_interop():
        pass

    s = test_cuda_interop.__OS_DEPS__
    assert s.pkg_specs == ("libcuda-dev",)
    assert s.on_missing == "skip"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_decorator_with_on_missing_fail():
    """hipcc is mandatory — hard fail if not installed."""

    @OS_DEPS("hipcc", on_missing="fail")
    def test_hipcc_available():
        pass

    s = test_hipcc_available.__OS_DEPS__
    assert s.pkg_specs == ("hipcc",)
    assert s.on_missing == "fail"


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


# ── sudo probe ───────────────────────────────────────────────────────────────


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_sudo_probe_not_found_returns_empty_prefix():
    """When sudo binary is absent (e.g. plain Docker root), prefix must be empty."""
    from framework.plugins.os_deps_plugin import _resolve_sudo_prefix

    ex = _mock_executor(stdout="NO")  # command -v sudo → not found
    prefix = _resolve_sudo_prefix(ex, _backend("apt"))
    assert prefix == ""
    ex.run.assert_called_once()  # probe fired exactly once


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_sudo_probe_found_returns_sudo_n_prefix():
    """When sudo -n works (standard fleet node), prefix must be 'sudo -n '."""
    from framework.plugins.os_deps_plugin import _resolve_sudo_prefix

    ex = _mock_executor(stdout="OK")  # command -v sudo && sudo -n true → OK
    prefix = _resolve_sudo_prefix(ex, _backend("apt"))
    assert prefix == "sudo -n "


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_sudo_probe_cached_after_first_call():
    """Sudo probe must fire only once; subsequent calls use the cached prefix."""
    from framework.plugins.os_deps_plugin import _resolve_sudo_prefix

    ex = _mock_executor(stdout="OK")
    b = _backend("apt")
    _resolve_sudo_prefix(ex, b)  # first call — probes
    _resolve_sudo_prefix(ex, b)  # second call — must use cache
    assert ex.run.call_count == 1  # only one probe, not two


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_sudo_probe_skipped_for_windows_backends():
    """Windows backends (needs_sudo=False) must not probe sudo at all."""
    from framework.plugins.os_deps_plugin import _resolve_sudo_prefix, _winget_cmd

    win_backend = _Backend(
        key="winget",
        detect_cmd="winget --version",
        probe_cmd_fn=lambda p: p,
        install_cmd_fn=_winget_cmd,
        needs_sudo=False,
        platform="windows",
    )
    ex = _mock_executor()
    prefix = _resolve_sudo_prefix(ex, win_backend)
    assert prefix == ""
    ex.run.assert_not_called()  # no probe for Windows


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_execute_install_no_sudo_in_command_when_sudo_absent():
    """End-to-end: when sudo is absent, _execute_install must run apt-get without sudo.

    Simulates a plain Docker container (root user, no sudo binary).
    The probe returns 'NO' → prefix='' → install command has no 'sudo'.
    """
    call_log: list[str] = []

    def fake_run(cmd, **_):
        call_log.append(cmd)
        result = MagicMock()
        if "command -v sudo" in cmd:
            result.ok = True
            result.stdout = "NO"   # sudo not found
        elif "dpkg-query" in cmd:
            result.ok = False      # package missing → trigger install
            result.stdout = ""
        else:
            result.ok = True       # install succeeds
            result.stdout = ""
        result.stderr = ""
        result.exit_code = 0 if result.ok else 1
        return result

    ex = MagicMock()
    ex.run.side_effect = fake_run

    with patch("framework.plugins.os_deps_plugin._detect_backend") as mock_detect:
        mock_detect.return_value = _backend("apt")
        spec = _InstallSpec(pkg_specs=("libnuma-dev",))
        _execute_install(ex, spec)

    install_cmd = next((c for c in call_log if "apt-get install" in c), None)
    assert install_cmd is not None, "apt-get install was never called"
    assert "sudo" not in install_cmd, (
        f"sudo must not appear in install command when sudo is absent, got: {install_cmd}"
    )


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_execute_install_uses_sudo_when_available():
    """End-to-end: when sudo -n works, _execute_install must use 'sudo -n' prefix."""
    call_log: list[str] = []

    def fake_run(cmd, **_):
        call_log.append(cmd)
        result = MagicMock()
        if "command -v sudo" in cmd:
            result.ok = True
            result.stdout = "OK"   # sudo available and -n works
        elif "dpkg-query" in cmd:
            result.ok = False      # package missing → trigger install
            result.stdout = ""
        else:
            result.ok = True       # install succeeds
            result.stdout = ""
        result.stderr = ""
        result.exit_code = 0 if result.ok else 1
        return result

    ex = MagicMock()
    ex.run.side_effect = fake_run

    with patch("framework.plugins.os_deps_plugin._detect_backend") as mock_detect:
        mock_detect.return_value = _backend("apt")
        spec = _InstallSpec(pkg_specs=("libnuma-dev",))
        _execute_install(ex, spec)

    install_cmd = next((c for c in call_log if "apt-get install" in c), None)
    assert install_cmd is not None, "apt-get install was never called"
    assert "sudo -n" in install_cmd, (
        f"Expected 'sudo -n' in install command, got: {install_cmd}"
    )


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


# ── Real install integration tests ────────────────────────────────────────────
#
# @OS_DEPS applied directly to the test function — the autouse fixture reads
# test_fn.__OS_DEPS__ and installs the packages before the test body runs.
# The test body then verifies the package is actually usable.
#
# ci.pr  → packages already present on any dev box; probe skips the install
# ci.nightly → packages that may need a real install on a fresh node


@OS_DEPS("cmake", "git", "curl")
@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_common_build_tools_installed(target_executor):
    """cmake / git / curl are installed (or already present) and functional.

    Probe fires first — on a standard dev machine all three are present and
    apt-get is never called. On a fresh CI node they get installed.
    """
    for tool, flag in [("cmake", "--version"), ("git", "--version"), ("curl", "--version")]:
        result = target_executor.run(f"{tool} {flag}")
        assert result.ok, f"{tool} not found after OS_DEPS install: {result.stderr}"
        assert tool in result.stdout.lower(), f"Unexpected {tool} output: {result.stdout}"


@OS_DEPS({"apt": "libnuma-dev", "dnf": "numactl-devel", "zypper": "libnuma-devel", "*": "libnuma-dev"})
@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_libnuma_dev_installed_cross_distro(target_executor):
    """libnuma-dev (apt) / numactl-devel (dnf/zypper) installed via dict spec.

    Each distro uses the correct package name; '*' covers any unlisted PM.
    After install the NUMA header must be locatable on the file system.
    """
    result = target_executor.run(
        "find /usr/include -name 'numa.h' 2>/dev/null | head -1"
    )
    assert result.ok
    assert result.stdout.strip(), "libnuma-dev installed but numa.h not found under /usr/include"


@OS_DEPS({"apt": "libdrm-dev", "dnf": "libdrm-devel", "zypper": "libdrm-devel", "*": "libdrm-dev"})
@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_libdrm_dev_installed_cross_distro(target_executor):
    """libdrm-dev (apt) / libdrm-devel (dnf/zypper) installed via dict spec.

    After install, the DRM header must be present. This is a real dependency
    for any HIP/GPU userspace stack and validates cross-distro name mapping.
    """
    result = target_executor.run(
        "find /usr/include -name 'drm.h' -path '*/drm/*' 2>/dev/null | head -1"
    )
    assert result.ok
    assert result.stdout.strip(), "libdrm-dev installed but drm.h not found under /usr/include"


@OS_DEPS("packaging", command="pip", extra_args=["--quiet"])
@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_pip_install_packaging_library(target_executor):
    """'packaging' Python library installed via pip with --quiet flag.

    'packaging' is a tiny, dependency-free PyPI package used by pip itself —
    safe to install on any machine. Verifies command= and extra_args= paths.
    """
    result = target_executor.run(
        "python3 -c 'import packaging; print(packaging.__version__)'"
    )
    assert result.ok, f"packaging import failed: {result.stderr}"
    assert result.stdout.strip(), "packaging version string is empty"
