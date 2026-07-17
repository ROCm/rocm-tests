# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared, env-configurable run parameters for the TorchVision P1 UT test.

The source checkout, in-tree ops build, and the two pytest UT suites all happen
inside a single command executed by ``target_executor`` (see
``test_torchvision.py``). Running everything through one executor keeps the test
correct across all environment profiles -- local bare-metal, remote SSH, and
Docker container -- because the source tree is always created wherever the GPU
command runs, with no host<->container path mismatch and nothing to volume-mount.

Unlike Apex (which reads only a commit), TorchVision derives BOTH the repository
URL and the commit to check out from the pinned ``related_commits`` manifest --
field 6 is the fork URL (e.g. a ROCm/vision fork) and field 5 is the commit id.
This keeps the clone matched to the exact downstream fork the image was built for.

    * On bare-metal there is no prebuilt PyTorch tree, so both the URL and the
      commit must be supplied by the user (``TORCHVISION_URL`` +
      ``TORCHVISION_COMMIT``); the commit is required, and the URL falls back to
      the default ROCm/vision repository when not set.
    * In a prebuilt-PyTorch container both are looked up from the image's
      ``related_commits`` manifest (each overridable via the env knobs below).

Environment overrides:
    TORCHVISION_URL             Git URL of the torchvision source tree. Overrides
                                the URL read from ``related_commits`` (field 6);
                                defaults to the upstream ROCm/vision repository
                                when neither is available.
    TORCHVISION_COMMIT          torchvision "related commit" id to check out.
                                Required on bare-metal; optional in container mode
                                where it overrides the value read from field 5 of
                                the ``related_commits`` manifest.
    TORCHVISION_RELATED_COMMITS Explicit path to the ``related_commits`` manifest
                                inside a prebuilt-PyTorch container. When empty,
                                well-known locations are probed with a bounded
                                ``find`` fallback.
    TORCHVISION_NUM_GPUS        GPUs to use on one node. Unset (default) uses
                                EVERY GPU the node/container exposes. Set an
                                integer to cap it (e.g. ``TORCHVISION_NUM_GPUS=1``
                                for single-GPU).
    TORCHVISION_WORK_DIR        Writable scratch dir for the checkout + build
                                (default ``/tmp/rocm-tests/torchvision``). Must be
                                writable both on the bare-metal node and inside the
                                container image.
    TORCHVISION_RUN_TIMEOUT     Wall-clock cap for clone + in-tree ops build + both
                                pytest UT suites, in seconds. The build compiles
                                the C++/HIP ops on first run and the two suites
                                exercise many image-transform cases, so the default
                                is generous.
"""

from __future__ import annotations

import os

# Default upstream public torchvision source tree (ROCm/vision fork).
_DEFAULT_TORCHVISION_URL = "https://github.com/ROCm/vision"

# Raw, user-supplied URL override (empty when unset). In container mode a non-empty
# value wins over the manifest's field-6 URL; when empty the manifest URL is used.
TORCHVISION_URL_OVERRIDE = os.environ.get("TORCHVISION_URL", "").strip()

# Effective default URL used when no manifest URL is available (bare-metal, or a
# manifest that omits field 6). A user override always takes precedence.
TORCHVISION_URL = TORCHVISION_URL_OVERRIDE or _DEFAULT_TORCHVISION_URL

# User-supplied torchvision "related commit" id. Required on bare-metal; optional
# in container mode, where it overrides the value read from field 5 of the image's
# related_commits manifest.
TORCHVISION_COMMIT = os.environ.get("TORCHVISION_COMMIT", "").strip()

# Optional explicit path to the related_commits manifest inside a prebuilt PyTorch
# container. When empty, well-known locations are probed with a bounded find(1).
RELATED_COMMITS_PATH = os.environ.get("TORCHVISION_RELATED_COMMITS", "").strip()

# GPUs to use on one node. ``None`` means "every GPU the node/container exposes"
# (resolved at run time); an integer caps the count.
_TORCHVISION_NUM_GPUS_RAW = os.environ.get("TORCHVISION_NUM_GPUS", "").strip()
TORCHVISION_NUM_GPUS = int(_TORCHVISION_NUM_GPUS_RAW) if _TORCHVISION_NUM_GPUS_RAW else None

# Argument for ``@pytest.mark.gpu_count(...)``: an explicit int when the user pins
# TORCHVISION_NUM_GPUS, else the framework "all" sentinel so the bare-metal
# acquisition path reserves every GPU on the node. (Container mode ignores
# gpu_count and controls visibility via ROCR_VISIBLE_DEVICES in the run command --
# see test_torchvision.py.)
GPU_COUNT_ARG = TORCHVISION_NUM_GPUS if TORCHVISION_NUM_GPUS is not None else "all"

# Writable scratch dir for the checkout + ops build. Must be writable on the
# bare-metal node and inside the container image alike.
WORK_DIR = os.environ.get("TORCHVISION_WORK_DIR", "/tmp/rocm-tests/torchvision")

# The two GPU UT suites run under pytest, restricted to the cuda-tagged cases via
# ``-k cuda``. These exercise rotate/affine/perspective/crop/pad/resize/color ops,
# comparing GPU results against CPU/PIL references within tolerance.
TEST_FILES = (
    "test/test_functional_tensor.py",
    "test/test_transforms_tensor.py",
)
PYTEST_SELECTOR = "cuda"

# Whole-workflow wall-clock cap (seconds): clone + first-run ops build + both
# pytest UT suites.
RUN_TIMEOUT = float(os.environ.get("TORCHVISION_RUN_TIMEOUT", "14400"))
