# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared base class for tests that drive the ``rvs`` (RVS) binary.

``rvs_iet_stress`` and ``rvs_tst`` had 46 nearly-identical lines each —
only the config filename, human label, goal string, and ``min_util``
differed. Collapsing them onto a parameterised base:

* Keeps the two ``RvsIetStress`` / ``RvsTst`` classes so
  ``tests/__init__.py::ALL_TESTS`` and downstream imports don't change.
* Removes the duplicated build/available/run bodies.
* Makes it trivial to add a third RVS-based test (e.g. ``rvs_babel``):
  a five-line subclass is all that's required.
"""

from __future__ import annotations

from pathlib import Path

from tests.common.gpu_monitored import rvs
from tests.common.gpu_monitored.config import Config
from tests.common.gpu_monitored.workloads.base import BuildContext, BuildStatus, RunContext, RunResult, Test, TestSpec


class _RvsBased(Test):
    """Parameterised base for ``rvs_iet_stress`` / ``rvs_tst``.

    Subclasses must set the three ``_``-prefixed class attributes
    below. Everything else — build delegation, availability check,
    configuration lookup, exec — is shared.
    """

    # Subclass must override:
    _conf_name: str  # e.g. "iet_stress.conf"
    _human_label: str  # e.g. "IET" (used only in the UNSUPPORTED message)

    # Config resolution mirrors ROCmTest (``rvs.resolve_conf_for_device`` +
    # the vendored ``rvs_config_mapping.csv``): the PCI ``device_id`` selects,
    # per test, the exact conf subdir (``./`` generic, ``./MI300A/`` per-model,
    # or empty -> UNSUPPORTED). This replaces the old GPU-short-name -> single
    # dir guess that misrouted (e.g. MI250 vs MI250X) and mis-reported
    # UNSUPPORTED (AISQA-8989 / AISQA-9134).
    #
    # ``rvs.find_conf`` (GPU-short-name based, with generic fallback) is used
    # ONLY as a safety net when the device_id cannot be detected at all; when
    # ``_gpu_only`` is True that legacy fallback refuses the generic config.
    _gpu_only: bool = False

    def build(self, ctx: BuildContext) -> BuildStatus:
        ok = rvs.build(ctx.config)
        return BuildStatus.OK if ok else BuildStatus.BUILD_FAILED

    def available(self, config: Config) -> bool:
        return rvs.find_bin(config) is not None

    def run(self, ctx: RunContext) -> RunResult:
        rvs_bin = rvs.find_bin(ctx.config)
        if rvs_bin is None:
            print("rvs not built")
            return RunResult(exit_code=1)

        # Validate the shipped mapping before picking a resolution path. The
        # legacy GPU-name lookup below never consults it, so without this a
        # missing or structurally broken mapping went unreported whenever the
        # device ID also happened to be undetectable -- the run would quietly
        # fall back to a generic config.
        component = self._conf_name[:-5] if self._conf_name.endswith(".conf") else self._conf_name
        try:
            rvs.ensure_config_map(component)
        except rvs.ConfigMapUnavailableError as e:
            print(f"  [{self.spec.name}] FAIL: {e}")
            print(
                "  This file is vendored in tests/common/gpu_monitored/; "
                "check the checkout or the copy to the test host."
            )
            return RunResult(exit_code=1)

        device_id = ctx.config.gpu_device_id
        if device_id:
            # Authoritative path: device is identified -> honor the CSV verdict
            # exactly (ROCmTest semantics). Do NOT fall back to the generic
            # config for a *known* device whose cell is empty/missing, so an
            # UNSUPPORTED test stays UNSUPPORTED instead of silently running a
            # possibly-wrong generic config.
            try:
                conf, reason = rvs.resolve_conf_for_device(
                    self._conf_name,
                    device_id,
                    ctx.config.rocm_root,
                )
            except rvs.RvsInstallIncompleteError as e:
                print(f"  [{self.spec.name}] FAIL: {e}")
                print(
                    "  Check that the preinstalled RVS package and its "
                    "configuration tree are complete and from the same "
                    "build."
                )
                return RunResult(exit_code=1)
            except rvs.ConfigMapUnavailableError as e:
                # The mapping ships with the harness. Its absence is our
                # packaging fault, not a property of this host, and must not
                # be laundered into a zero-exit UNSUPPORTED that greens CI
                # for every RVS test on every device.
                print(f"  [{self.spec.name}] FAIL: {e}")
                print(
                    "  This file is vendored in tests/common/gpu_monitored/; "
                    "check the checkout or the copy to the test host."
                )
                return RunResult(exit_code=1)
            if conf is None:
                gpu_label = ctx.config.gpu_short_name or ctx.config.gpu_model or device_id
                print(f"UNSUPPORTED: No {self._human_label} config for " f"{gpu_label} -- {reason}")
                return RunResult(unsupported=True)
        else:
            # Detection failed (no device_id): fall back to the legacy
            # GPU-name-based lookup so a detection gap doesn't hard-fail.
            conf = rvs.find_conf(
                self._conf_name,
                gpu_only=self._gpu_only,
                rocm_root=ctx.config.rocm_root,
                gpu_conf_dir=ctx.config.gpu_conf_dir,
            )
            if conf is None:
                target = ctx.config.gpu_short_name or ctx.config.gpu_arch or "unknown"
                print(f"UNSUPPORTED: No {self._human_label} config found for " f"{target} (GPU device id undetected)")
                return RunResult(unsupported=True)

        print(f"  Using RVS config: {conf}")
        watchdog = getattr(ctx.config, "per_iter_watchdog", 0) or None
        timeout_prefix = f"timeout {watchdog} " if watchdog else ""
        # The redirect is load-bearing, not cosmetic: rvs deadlocks in
        # kfd_wait_on_events when this output goes to a pipe, so the
        # reproducer has to carry it or it will not reproduce the run we did.
        stdout_file = Path(ctx.run_dir) / "rvs_stdout.log"
        reproduce = f"{timeout_prefix}{rvs_bin} -c {conf} -d 3 " f"> {stdout_file.name} 2>&1"
        rc = ctx.exec([rvs_bin, "-c", conf, "-d", "3"], timeout=watchdog, stdout_file=stdout_file)
        if rc == 124:
            print(
                f"  [{self.spec.name}] FAIL: watchdog timeout — RVS did "
                f"not complete within --per-iter-watchdog {watchdog}s"
            )
            return RunResult(exit_code=1, reproduce_cmd=reproduce)
        return RunResult(exit_code=rc, reproduce_cmd=reproduce)

    @staticmethod
    def _make_spec(
        *,
        name: str,
        goal: str,
        workload_profile: dict,
    ) -> TestSpec:
        return TestSpec(name=name, goal=goal, workload_profile=workload_profile)
