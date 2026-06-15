# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
framework.config -- Runtime configuration loader.

Modules:
    loader  -- load_config(): code defaults → rocm-test.toml → ENV vars → CLI flags cascade (lowest → highest)
"""
