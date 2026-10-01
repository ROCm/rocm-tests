# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared, env-configurable parameters for the Kokkos HIP RDC test.

Imported by both ``conftest.py`` (clone + build) and ``test_gpu_rdc_kokkos.py``
(ctest run) so build-time and run-time settings can never drift.

Environment overrides:
    KOKKOS_REF            git ref (tag/branch/SHA) to clone. Default ``5.1.1``
                          (supports gfx950 + modern AMD_GFX arch-flag naming).
    KOKKOS_CTEST_TIMEOUT  per-test ctest timeout in seconds (``ctest --timeout N``).
    KOKKOS_CTEST_RUN_TIMEOUT
                          outer wall-clock cap for the whole ctest run (seconds).
    KOKKOS_CTEST_JOBS     ctest parallelism (``ctest -j N``). Unset/empty ->
                          ``$(nproc)`` (resolved on the execution node); set ``1``
                          to force serial. The source build always parallelises
                          make/cmake (``cmake --build --parallel``) automatically.
"""

import os

# Upstream Kokkos performance-portability HPC library.
KOKKOS_URL = os.environ.get("KOKKOS_URL", "https://github.com/kokkos/kokkos.git")

# Pinned default clone ref. 5.1.1 is the evidence-based default: it supports
# current AMD hardware (gfx950/MI350 and the gfx12xx Navi line) and the modern
# ``Kokkos_ARCH_AMD_GFX<NNN>`` arch-flag naming emitted by ``kokkos_arch_flag``
# below. The older 4.2.01 tag predates gfx950 and uses codename arch flags
# (``Kokkos_ARCH_VEGA90A`` etc.), so it cannot configure for newer GPUs — a
# ``-DKokkos_ARCH_AMD_GFX950=ON`` is simply unknown there and cmake fails with
# "no AMD GPU architecture is supported". Override with KOKKOS_REF for a specific
# Kokkos version matched to the target arch.
KOKKOS_REF = os.environ.get("KOKKOS_REF", "5.1.1")

# Per-test ctest timeout (seconds); bounds any single hung/aborted Kokkos unit
# test. Kept well below the outer run cap so one slow test cannot consume the
# whole budget.
CTEST_TIMEOUT = os.environ.get("KOKKOS_CTEST_TIMEOUT", "1800")

# Outer wall-clock cap for the full ctest run. The suite measured ~43 min end to
# end serially, so the default (90 min) leaves headroom for slower GPUs / variance
# even before the -j speed-up below.
CTEST_RUN_TIMEOUT = float(os.environ.get("KOKKOS_CTEST_RUN_TIMEOUT", "5400"))

# ctest parallelism (``ctest -j N``), mirroring the original launch's ``-j`` flag.
# Empty/unset resolves to ``$(nproc)`` on the execution node (see
# ``ctest_parallel_arg``); set KOKKOS_CTEST_JOBS=1 to force serial if parallel
# Kokkos unit tests contend for the single GPU.
CTEST_JOBS = os.environ.get("KOKKOS_CTEST_JOBS", "").strip()


def ctest_parallel_arg() -> str:
    """Return the ``ctest -j`` argument controlling parallel test execution.

    When ``KOKKOS_CTEST_JOBS`` is set, uses that fixed job count; otherwise falls
    back to ``-j$(nproc)`` so the shell resolves the core count on the execution
    node (which may be a remote GPU host, not the coordinator).
    """
    return f"-j{CTEST_JOBS}" if CTEST_JOBS else "-j$(nproc)"


def kokkos_arch_flag(gpu_arch: str | None) -> str:
    """Map a framework GFX arch string to the Kokkos 4.2+ AMD arch cmake flag.

    Kokkos 4.2+ enables an AMD GPU target with ``-DKokkos_ARCH_AMD_GFX<NNN>=ON``
    (e.g. ``gfx950`` -> ``-DKokkos_ARCH_AMD_GFX950=ON``). Returns an empty string
    when ``gpu_arch`` is falsy so callers can guard/require an explicit arch.
    """
    if not gpu_arch:
        return ""
    return f"-DKokkos_ARCH_AMD_{gpu_arch.upper()}=ON"
