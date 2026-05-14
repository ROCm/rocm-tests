# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
framework -- Core test automation framework for AMD ROCm validation.

Sub-packages:
    config      -- Runtime configuration loader (rocm-test.toml → env vars → CLI defaults)
    common      -- Shared utilities: ExecutionResult, Outcome, parse_metric(), retry decorator
    executors   -- Command executor hierarchy (DryRun, Local, SSH, Container, Labeled, Group)
    nodes       -- NodePool fleet manager: NodeSpec, NodeSlot, GpuFileLock, PendingTracker
    scheduling  -- DynamicScheduler and SchedulePolicy for resource-aware xdist scheduling
    builder     -- BinaryBuilder: hipcc compilation with xdist locking + incremental builds
    gpu         -- GpuDetector, GpuAllocator, GpuDrainChecker, GpuBackgroundMonitor
    markers     -- MARKER_SCHEMA taxonomy and MarkerLinter
    os_adapter  -- Linux + Windows GPU enumeration behind AbstractOsAdapter
    plugins     -- pytest plugin modules loaded by root conftest.py
    reporting   -- Allure step-level reporting helpers
    rocm        -- ROCm library helpers: hip, rccl, amd_smi, stack
"""
