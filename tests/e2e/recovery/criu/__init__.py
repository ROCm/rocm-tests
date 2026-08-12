# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CRIU checkpoint/restore recovery test suite.

Holds the workload builds (``conftest.py``) and CRIU checkpoint/restore test functions for
cuda_memtest, the PyTorch MNIST example, LLNL RAJAPerf, and the hip-tests MatrixTranspose sample.
Shared CRIU machinery lives in ``tests.common.criu``.
"""
