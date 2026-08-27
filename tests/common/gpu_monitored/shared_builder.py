# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-process build-result cache for shared tools (rvs, ...).

A single tool can back multiple tests -- e.g. ``rvs`` is used by both
``rvs_iet_stress`` and ``rvs_tst``. A naive "call the full build/verify
each time" approach would repeat the install check for every dependent
test. This class memoises the verdict per process so the second caller
sees the same answer as the first with no repeated work, and logs the
"install found at …" banner only once.

Historically each tool module carried its own ``_BUILD_ATTEMPTED`` /
``_BUILD_RESULT`` / ``reset_build_state`` trio (plus an internal
``_finish`` helper) to get this behaviour. Those copies were subtly
different (one set ``_BUILD_ATTEMPTED = True`` *before* the build ran,
which silently masked failures). Consolidating into one class makes the
behaviour unambiguous and removes the per-tool boilerplate.

``reset()`` drops the memoised verdict when a caller needs a fresh
install check in the same interpreter (e.g. unit tests).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tests.common.gpu_monitored.config import Config


class SharedToolBuilder:
    """Cache the verdict of an 'install or build from source' step.

    ``install_check(rocm_root) -> bool`` is invoked on every call; when
    it returns True we always take the fast path (no cache needed, the
    binary is already there). The cache only kicks in when the install
    check is False, in which case ``build_fn(config) -> bool`` runs
    exactly once per process lifecycle and its return value is reused
    for every subsequent call until ``reset()`` drops the cache.
    """

    def __init__(
        self,
        *,
        label: str,
        install_check: Callable[[Path], bool],
        build_fn: Callable[[Config], bool],
    ) -> None:
        self._label = label
        self._install_check = install_check
        self._build_fn = build_fn
        self._result: bool | None = None
        self._announced_install = False

    def reset(self) -> None:
        """Drop cached verdict so the next ``build()`` runs from scratch."""
        self._result = None
        self._announced_install = False

    def build(self, config: Config) -> bool:
        """Run the build (once per process) and cache the verdict.

        The cache returns a **boolean status only** — it does not
        validate that the produced binary is still on disk. Every
        caller that proceeds to actually run the tool MUST additionally
        call ``Test.available(config)`` (or the equivalent ``find_bin``
        probe) before spawning a workload, so that a binary cleaned up
        between ``build()`` and ``run()`` is caught even if the cache
        still reports True. The ``rvs`` consumer follows this pattern
        via its module-level ``find_bin`` helper and the
        ``Test.available`` override.
        """
        if self._install_check(config.rocm_root):
            if not self._announced_install:
                print(f"  [build] {self._label}: found at {config.rocm_root}/bin")
                self._announced_install = True
            self._result = True
            return True
        if self._result is not None:
            return self._result
        self._result = bool(self._build_fn(config))
        return self._result
