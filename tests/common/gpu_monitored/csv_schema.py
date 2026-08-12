# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Column names shared by modules that read/write the amd-smi monitoring CSV.

The CSV itself is produced by ``amd-smi monitor --csv`` in ``monitoring.py``
and consumed by:

- ``analyze_monitoring.load_csv`` (health checks, per-GPU stats)
- ``generate_report._parse_rows`` (HTML report)

Keeping the column names in one place means that when amd-smi renames a
field, only this module needs to change.
"""

# Raw amd-smi CSV columns (as emitted by ``amd-smi monitor --csv``)
TIMESTAMP = "timestamp"
GPU = "gpu"
POWER_USAGE = "power_usage"
MAX_POWER = "max_power"
HOTSPOT_TEMP = "hotspot_temperature"
MEM_TEMP = "memory_temperature"
GFX_CLK = "gfx_clk"
GFX_UTIL = "gfx"  # amd-smi labels the gfx utilization column simply "gfx"
MEM_UTIL = "mem"  # and the memory engine column simply "mem"
MEM_CLK = "mem_clock"
VRAM_USED = "vram_used"
VRAM_TOTAL = "vram_total"
VRAM_PCT = "vram_percent"
