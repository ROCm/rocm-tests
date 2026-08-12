# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""GPU monitoring helpers shared across test suites.

Provides monitoring engine, 5-layer validation, post-test analysis,
and HTML report generation utilities.

Submodules are imported directly by consumers (e.g.
``from tests.common.gpu_monitored.monitoring import Monitor``).
"""
