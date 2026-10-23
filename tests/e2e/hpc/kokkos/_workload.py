# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared, env-configurable parameters for the Kokkos HIP RDC test.

Imported by both ``conftest.py`` (clone + build) and ``test_gpu_rdc_kokkos.py``
(ctest run) so build-time and run-time settings can never drift.

Environment overrides:
    KOKKOS_REF            git ref (tag/branch/SHA) to clone. Default ``5.2.2``
                          (fixes HIP bfloat16/fp16 API breaks vs 5.1.1).
    KOKKOS_CTEST_TIMEOUT  per-test ctest timeout in seconds (``ctest --timeout N``).
    KOKKOS_CTEST_RUN_TIMEOUT
                          outer wall-clock cap for the whole ctest run (seconds).
"""

import os

# Upstream Kokkos performance-portability HPC library.
KOKKOS_URL = os.environ.get("KOKKOS_URL", "https://github.com/kokkos/kokkos.git")

# Pinned default clone ref. 5.2.2 is the evidence-based default: it supports
# current AMD hardware (gfx950/MI350 and the gfx12xx Navi line), uses the modern
# ``Kokkos_ARCH_AMD_GFX<NNN>`` arch-flag naming, and fixes HIP bfloat16/fp16
# API breaks (``__hip_bfloat16`` conversions, ``__2i16`` to bool) that cause
# Kokkos 5.1.1 to fail to compile against ROCm 7.x and 10.x headers.
# Override with KOKKOS_REF env var for a specific version matched to the target.
KOKKOS_REF = os.environ.get("KOKKOS_REF", "5.2.2")

# Per-test ctest timeout (seconds), matching the original script's
# ``ctest --timeout 50000``. Large on purpose: it bounds a genuinely hung test
# without prematurely killing a slow-but-valid one (e.g. the atomic performance
# benchmark, which the enabled benchmarks run), so a strict ctest pass is judged
# on real results rather than an overtight cap.
CTEST_TIMEOUT = os.environ.get("KOKKOS_CTEST_TIMEOUT", "50000")

# Outer wall-clock cap for the full ctest run. Set above the per-test timeout so
# the executor never pre-empts ctest's own --timeout accounting. The suite runs
# ~60-70 min on gfx950 (the atomic benchmark alone is ~48 min); this is only a
# hang ceiling, not an expected duration.
CTEST_RUN_TIMEOUT = float(os.environ.get("KOKKOS_CTEST_RUN_TIMEOUT", "54000"))


def kokkos_arch_flag(gpu_arch: str | None) -> str:
    """Map a framework GFX arch string to the Kokkos 4.2+ AMD arch cmake flag.

    Kokkos 4.2+ enables an AMD GPU target with ``-DKokkos_ARCH_AMD_GFX<NNN>=ON``
    (e.g. ``gfx950`` -> ``-DKokkos_ARCH_AMD_GFX950=ON``). Returns an empty string
    when ``gpu_arch`` is falsy so callers can guard/require an explicit arch.
    """
    if not gpu_arch:
        return ""
    return f"-DKokkos_ARCH_AMD_{gpu_arch.upper()}=ON"
