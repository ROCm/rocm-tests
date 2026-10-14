# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
deps.py -- Public re-export of OS_DEPS for convenience.

Import from here:
    from framework.os_adapter.deps import OS_DEPS

The full implementation lives in framework.plugins.os_deps_plugin so
the plugin is self-contained and portable to other repos.
"""

from framework.plugins.os_deps_plugin import OS_DEPS

__all__ = ["OS_DEPS"]
