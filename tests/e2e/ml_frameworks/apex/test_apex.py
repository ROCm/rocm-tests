# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_apex.py -- Apex fused-kernel unit-test (L0) suite validation.

Apex is a PyTorch extension that provides high-performance "fused" building
blocks for training deep-learning models, such as fused optimizers (FusedAdam,
FusedLAMB) and fused normalization layers (FusedLayerNorm, FusedRMSNorm). These
components pack several GPU operations into single optimized kernels so that
transformer and other neural-network training can run faster and use memory more
efficiently.

The Apex L0 unit-test suite first compiles these custom GPU kernels and then
exercises them across hundreds of small correctness checks, covering
mixed-precision casts, fused optimizers, MLP and layer-norm layers, and
distributed tensor- and pipeline-parallel routines. A "good" result means the
kernels build successfully and produce numerically correct outputs that match
reference PyTorch implementations, confirming Apex's fused-kernel and
distributed-training features behave correctly on a given ROCm GPU software
stack.

ROCm stack components exercised:
    KFD + amdgpu kernel driver, ROCr + HIP runtime, the hipcc compiler and
    hipify, RCCL (multi-rank collectives for the tensor/pipeline-parallel tests),
    and rocBLAS / hipBLASLt (matrix multiplies in the MLP / fused-dense /
    transformer tests).

Supported architectures:
    gfx1101, gfx1100, gfx950, gfx942, gfx90a, gfx908.

Supported OS / environment profiles:
    Ubuntu 24.04, RHEL 10.1, SLES 15.7; bare-metal and container profiles.

Environment profiles:
    The source checkout, kernel build, and L0 run are performed in a single
    command via ``target_executor`` so the same test runs unchanged on a local
    bare-metal node, a remote SSH node, or inside a Docker/Podman container
    (``--container-mode``). On bare-metal the installed ROCm stack (``rock_dir``)
    is injected into the run environment; in container mode the ROCm stack and
    PyTorch shipped in the image are used as-is.

Apex commit ("related commit"):
    The commit to check out is required input. On bare-metal (no prebuilt PyTorch)
    it must be supplied via ``APEX_COMMIT=<commit>``. In a prebuilt-PyTorch
    container it is read from the image's ``related_commits`` manifest; if that
    manifest is absent the test fails, directing the user to either run against a
    prebuilt-PyTorch image that ships it or pass ``APEX_COMMIT`` explicitly.
    ``APEX_COMMIT`` always overrides the manifest lookup.

GPU count:
    By default the suite uses EVERY GPU the node/container exposes, so the
    multi-device and tensor/pipeline-parallel L0 tests run instead of skipping.
    Set ``APEX_NUM_GPUS=<n>`` to cap it (e.g. ``1`` for single-GPU). On bare-metal
    this drives ``gpu_count`` acquisition; in container mode -- where the executor
    otherwise pins one GPU -- visibility is set via ``ROCR_VISIBLE_DEVICES`` in the
    run command. See ``_workload.py`` for all env knobs.

Markers:
    Injected by the CATEGORY_PROFILES entry for ``tests/e2e/ml_frameworks/apex``
    in taxonomy.py -- hw.gpu + hw.multi_gpu (single- and multi-GPU profiles),
    layer.runtime + layer.math_lib (HIP runtime/compiler and RCCL/rocBLAS/
    hipBLASLt libs), ci.nightly, e2e.stack, os.linux.

    Declared on the test function (profiles never inject these):
        gpu_count(all|N) -- acquire all GPUs (default) or N when APEX_NUM_GPUS set
        runtime.soak     -- kernel compilation plus hundreds of sub-tests runs for
                            hours; scheduled nightly for the ML-framework cadence
"""

import logging
import re

import pytest

from tests.e2e.ml_frameworks.apex._result_parser import parse_unittest_output
from tests.e2e.ml_frameworks.apex._workload import (
    APEX_COMMIT,
    APEX_NUM_GPUS,
    APEX_URL,
    GPU_COUNT_ARG,
    L0_SUBDIR,
    RELATED_COMMITS_PATH,
    RUN_SCRIPT,
    RUN_TIMEOUT,
    WORK_DIR,
)

logger = logging.getLogger(__name__)

# A git ref safe to interpolate into a shell command (no metacharacters).
_SAFE_REF_RE = re.compile(r"^[0-9A-Za-z._/-]+$")
# A filesystem path safe to interpolate into the related_commits lookup snippet.
_SAFE_PATH_RE = re.compile(r"^[0-9A-Za-z._/-]+$")
# Seconds allowed for the (trivial) in-container related_commits lookup.
_RESOLVE_TIMEOUT = 120.0

# Hard-crash signatures. If any appears in the runner output the suite aborted
# mid-run (the process died before finishing), so the run is a failure regardless
# of how many sub-tests were parsed as passing.
_CRASH_MARKERS = (
    "Memory access fault",
    "core dumped",
    "Segmentation fault",
    "HSA_STATUS_ERROR",
    "Aborted (",
    "Fatal Python error",
)

# Shell snippet run inside a prebuilt-PyTorch container to read the Apex commit
# from its ``related_commits`` manifest. It locates the manifest (explicit path,
# then well-known locations, then a bounded find), greps the apex line for this
# OS (distro id from /etc/os-release), and prints field 5 -- the commit. Sentinel
# markers on stdout let the caller distinguish "not found" from "no apex entry".
_RELATED_COMMITS_LOOKUP = r"""
f="{explicit}"
if [ -z "$f" ]; then
  for c in /related_commits /opt/pytorch/related_commits "$PYTORCH_DIR/related_commits"; do
    if [ -f "$c" ]; then f="$c"; break; fi
  done
fi
if [ -z "$f" ] || [ ! -f "$f" ]; then
  found=$(find / -maxdepth 6 -name related_commits -type f 2>/dev/null | head -1)
  [ -n "$found" ] && f="$found"
fi
if [ -z "$f" ] || [ ! -f "$f" ]; then
  echo "__APEX_RC_NOTFOUND__"; exit 0
fi
osid=$(. /etc/os-release 2>/dev/null; echo "$ID")
commit=$(grep -i apex "$f" | grep -i "$osid" | head -1 | cut -d '|' -f 5 | tr -d '[:space:]')
if [ -z "$commit" ]; then
  commit=$(grep -i apex "$f" | head -1 | cut -d '|' -f 5 | tr -d '[:space:]')
fi
if [ -z "$commit" ]; then
  echo "__APEX_RC_NOCOMMIT__:$f"; exit 0
fi
echo "__APEX_COMMIT__:$commit"
"""

# The hw.* / layer.* / ci.* / e2e.* / os.* markers are injected by the
# CATEGORY_PROFILES entry for ``tests/e2e/ml_frameworks/apex`` in taxonomy.py
# (both hw.gpu and hw.multi_gpu, so the suite covers the single- and multi-GPU
# profiles). Only the parametric ``gpu_count`` and the ``runtime.*`` weight -- which
# profiles intentionally never inject -- are declared on the test function.


def _visible_devices_prefix(request) -> str:
    """Return a command prefix that exposes the right number of GPUs, or ``""``.

    Only meaningful in container mode. ``target_executor``'s container path pins a
    single GPU (``ROCR_VISIBLE_DEVICES=0``) regardless of ``gpu_count``, so GPU
    visibility for the L0 run is controlled here instead:

        * ``APEX_NUM_GPUS`` unset -> drop the restriction so every GPU passed into
          the container (via ``--device``) is visible.
        * ``APEX_NUM_GPUS=k``     -> expose exactly GPUs ``0..k-1``.

    On bare-metal / SSH this returns ``""`` -- ``target_executor`` owns the real
    ``ROCR_VISIBLE_DEVICES`` allocation from the acquired ``gpu_count`` slots, and
    the test must not override it.
    """
    if not request.config.getoption("--container-mode", default=False):
        return ""
    if APEX_NUM_GPUS is None:
        return "env -u ROCR_VISIBLE_DEVICES "
    indices = ",".join(str(i) for i in range(APEX_NUM_GPUS))
    return f"env ROCR_VISIBLE_DEVICES={indices} "


def _rocm_env_prefix(request) -> str:
    """Return the ``env VAR=... `` prefix for the L0 runner, or ``""``.

    In container mode the ROCm stack and PyTorch shipped inside the image are
    used as-is -- injecting host paths would point the build at a tree that does
    not exist in the container. On bare-metal / SSH the installed ROCm tree
    (``rock_dir``) and its ``LD_LIBRARY_PATH`` are injected so the kernel build
    and the ROCm compute libraries resolve against the intended stack.

    ``rock_dir`` / ``ld_path`` are resolved lazily (they ``pytest.fail`` when no
    ROCm path is configured) so container runs need neither ``--rock-dir`` nor a
    host ROCm install.
    """
    if request.config.getoption("--container-mode", default=False):
        return ""
    rock_dir = request.getfixturevalue("rock_dir")
    ld = request.getfixturevalue("ld_path")["LD_LIBRARY_PATH"]
    return f"env ROCM_PATH={rock_dir} PATH={rock_dir}/bin:$PATH LD_LIBRARY_PATH={ld}:$LD_LIBRARY_PATH "


def _validate_ref(ref: str) -> str:
    """Return *ref* if it is safe to interpolate into a shell command, else fail."""
    if not _SAFE_REF_RE.match(ref):
        pytest.fail(f"Apex commit id contains unsafe characters: {ref!r}")
    return ref


def _lookup_commit_in_container(target_executor) -> str:
    """Read the Apex commit from the container image's ``related_commits`` manifest.

    Runs the lookup snippet inside the (prebuilt-PyTorch) container and interprets
    its sentinel output. Fails with actionable guidance when the manifest is
    absent or carries no apex entry -- the two states the caller must distinguish.
    """
    explicit = RELATED_COMMITS_PATH
    if explicit and not _SAFE_PATH_RE.match(explicit):
        pytest.fail(f"APEX_RELATED_COMMITS path contains unsafe characters: {explicit!r}")

    result = target_executor.run(_RELATED_COMMITS_LOOKUP.format(explicit=explicit), timeout=_RESOLVE_TIMEOUT)
    out = f"{result.stdout}\n{result.stderr}"

    for line in out.splitlines():
        token = line.strip()
        if token.startswith("__APEX_COMMIT__:"):
            commit = token.split(":", 1)[1].strip()
            logger.info("Apex commit resolved from related_commits manifest: %s", commit)
            return commit

    if "__APEX_RC_NOTFOUND__" in out:
        pytest.fail(
            "related_commits manifest was not found inside the container. Use a prebuilt "
            "PyTorch container image that ships a related_commits file, or pass the Apex "
            "commit explicitly via APEX_COMMIT=<commit>. Set APEX_RELATED_COMMITS=<path> to "
            "point at the manifest directly."
        )
    if "__APEX_RC_NOCOMMIT__" in out:
        pytest.fail(
            "related_commits manifest was found but contains no apex entry for this OS. "
            f"Pass the Apex commit explicitly via APEX_COMMIT=<commit>.\nLookup output:\n{out[-2000:]}"
        )
    pytest.fail(f"Could not resolve the Apex commit from related_commits:\n{out[-2000:]}")
    return ""  # unreachable -- pytest.fail raises


def _resolve_apex_commit(request, target_executor) -> str:
    """Determine the Apex commit to check out.

    Resolution order:
        1. ``APEX_COMMIT`` supplied by the user -- honored in every profile.
        2. Container mode only: the apex entry in the image's ``related_commits``
           manifest.
        3. Otherwise (bare-metal, no commit) fail with guidance -- bare-metal has
           no prebuilt PyTorch / related_commits file, so the commit is required.
    """
    if APEX_COMMIT:
        logger.info("Apex commit supplied by user (APEX_COMMIT): %s", APEX_COMMIT)
        return _validate_ref(APEX_COMMIT)

    if request.config.getoption("--container-mode", default=False):
        return _validate_ref(_lookup_commit_in_container(target_executor))

    pytest.fail(
        "An Apex commit id is required on bare-metal, where no prebuilt PyTorch / "
        "related_commits manifest is present. Provide it on the command line, e.g.\n"
        "  APEX_COMMIT=<commit> pytest tests/e2e/ml_frameworks/apex/test_apex.py "
        "--rock-dir /opt/rocm ..."
    )
    return ""  # unreachable -- pytest.fail raises


@pytest.mark.gpu_count(GPU_COUNT_ARG)
@pytest.mark.runtime.soak
def test_apex_l0_suite(request, target_executor):
    """Clone Apex, build the fused kernels, run the L0 suite, and assert it passes.

    The Apex commit is resolved first (user-supplied ``APEX_COMMIT``, else the
    container image's ``related_commits`` manifest). The checkout, first-run
    kernel build, and the ``unittest`` run then execute in a single
    ``target_executor`` command so the workflow is identical on bare-metal,
    remote SSH, and container profiles -- the source tree is always created
    wherever the GPU command runs. All GPUs are exposed by default (``APEX_NUM_GPUS``
    caps the count). The verbose output is parsed per sub-test so the assertion can
    name any failing or erroring case.
    """
    commit = _resolve_apex_commit(request, target_executor)
    vis_prefix = _visible_devices_prefix(request)
    env_prefix = _rocm_env_prefix(request)
    src_dir = f"{WORK_DIR}/src"

    # A fresh checkout each run avoids stale build state; the L0 runner compiles the
    # fused kernels then drives Python's unittest suite. ``vis_prefix`` sets GPU
    # visibility in container mode (bare-metal leaves ROCR to the executor).
    cmd = (
        "set -e; "
        f"rm -rf {WORK_DIR}; mkdir -p {WORK_DIR}; "
        f"git clone {APEX_URL} {src_dir}; "
        f"cd {src_dir} && git checkout {commit}; "
        f"cd {src_dir}/{L0_SUBDIR} && {vis_prefix}{env_prefix}bash ./{RUN_SCRIPT}"
    )

    gpu_label = "all" if APEX_NUM_GPUS is None else APEX_NUM_GPUS
    logger.info("Apex L0 suite starting in %s (commit=%s, num_gpus=%s)", src_dir, commit, gpu_label)
    result = target_executor.run(cmd, timeout=RUN_TIMEOUT)

    # unittest writes its verbose result lines to stderr; parse both streams.
    combined = f"{result.stdout}\n{result.stderr}"
    summary = parse_unittest_output(combined)

    crash_markers = [m for m in _CRASH_MARKERS if m in combined]

    logger.info(
        "Apex L0 results: passed=%d skipped=%d failed=%d errored=%d unresolved=%d "
        "(ran_total=%d, exit=%s, crash_markers=%s)",
        summary.passed,
        summary.skipped,
        summary.failed,
        summary.errored,
        len(summary.unresolved_names),
        summary.ran_total,
        result.exit_code,
        crash_markers or "none",
    )

    # No parsed results at all means the workflow never reached the suite (clone,
    # checkout, or kernel build failed, or PyTorch is missing) -- surface that.
    assert summary.total > 0 or summary.ran_total > 0, (
        f"Apex L0 suite produced no test results (exit={result.exit_code}); "
        f"the clone, kernel build, or runner likely failed to start:\n"
        f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-4000:]}"
    )

    # A clean run requires: no failed/errored/unresolved sub-tests, a zero exit
    # code, and no GPU crash signature. exit_code and crash_markers are essential
    # backstops -- a fault that aborts the runner mid-test (memory-access fault,
    # core dump) can leave the last sub-test without an outcome, and must never be
    # reported as a pass.
    assert summary.is_clean and result.exit_code == 0 and not crash_markers, (
        f"Apex L0 suite did not complete cleanly "
        f"(exit={result.exit_code}, crash_markers={crash_markers or 'none'}, "
        f"failed={summary.failed}, errored={summary.errored}, "
        f"unresolved={len(summary.unresolved_names)}, "
        f"passed={summary.passed}, skipped={summary.skipped}):\n"
        f"failed: {summary.failed_names[:50]}\n"
        f"errored: {summary.errored_names[:50]}\n"
        f"unresolved (crashed mid-test): {summary.unresolved_names[:50]}\n"
        f"stdout tail: {result.stdout[-3000:]}\nstderr tail: {result.stderr[-3000:]}"
    )
