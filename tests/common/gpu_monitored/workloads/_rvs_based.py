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

# Ceiling applied when no ``--per-iter-watchdog`` is set. The executor reads a
# timeout of None as "apply the 300s default" rather than "no limit", and RVS
# configs run every device in turn: the stock tst_single config is 120s per
# device, over 16 minutes on an 8-GPU node. An hour keeps the original suite's
# run-to-completion behaviour while still bounding a wedged run.
_DEFAULT_RVS_TIMEOUT = 3600


class _RvsBased(Test):
    """Parameterised base for ``rvs_iet_stress`` / ``rvs_tst``.

    Subclasses must set the three ``_``-prefixed class attributes
    below. Everything else — build delegation, availability check,
    configuration lookup, exec — is shared.
    """

    # Subclass must override. ``_conf_name`` is also what the conftest hands to
    # ``rvs_find_conf``, so it has to name the module's config as RVS ships it.
    _conf_name: str  # e.g. "iet_stress.conf"
    _human_label: str  # e.g. "IET" (used only in the UNSUPPORTED message)

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

        # Resolved by the ``rvs_find_conf`` fixture before the run, so the
        # qualification matrix is consulted once, through the same code path as
        # the RVS module tests, and against the node that will run the workload.
        # An unqualified module never reaches here -- the fixture skips first.
        conf = ctx.config.rvs_conf_path
        if not conf:
            print(f"  [{self.spec.name}] FAIL: no RVS config was resolved for this run")
            return RunResult(exit_code=1)

        print(f"  Using RVS config: {conf}")
        watchdog = getattr(ctx.config, "per_iter_watchdog", 0) or None
        timeout_prefix = f"timeout {watchdog} " if watchdog else ""
        # The redirect is load-bearing, not cosmetic: rvs deadlocks in
        # kfd_wait_on_events when this output goes to a pipe, so the
        # reproducer has to carry it or it will not reproduce the run we did.
        stdout_file = Path(ctx.run_dir) / "rvs_stdout.log"
        reproduce = f"{timeout_prefix}{rvs_bin} -c {conf} -d 3 " f"> {stdout_file.name} 2>&1"
        rc = ctx.exec(
            [rvs_bin, "-c", conf, "-d", "3"],
            timeout=watchdog or _DEFAULT_RVS_TIMEOUT,
            stdout_file=stdout_file,
        )
        if rc == 124:
            limit = f"--per-iter-watchdog {watchdog}s" if watchdog else f"the {_DEFAULT_RVS_TIMEOUT}s default ceiling"
            print(f"  [{self.spec.name}] FAIL: timeout — RVS did not complete within {limit}")
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
