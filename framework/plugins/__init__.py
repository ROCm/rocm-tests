# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
framework.plugins -- pytest plugin modules loaded by the root conftest.

Each plugin module is a self-contained pytest plugin that:
  - Registers its own CLI options via pytest_addoption()
  - Defines fixtures scoped to its domain
  - Optionally hooks pytest_runtest_setup / pytest_configure

Plugins are loaded via pytest_plugins in conftest.py (repo root).
They must not import each other directly; shared state passes via
pytest.Config attributes set in pytest_configure().
"""
