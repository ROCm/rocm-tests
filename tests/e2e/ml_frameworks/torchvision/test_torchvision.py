# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_torchvision.py -- TorchVision P1 image-transform correctness UT suite.

TorchVision is the PyTorch computer-vision companion library. Its transforms and
functional-tensor operators (rotate, affine, perspective, crop, pad, resize, and
the color/photometric ops) run on the GPU and must produce results that match
their CPU / PIL reference implementations within tolerance. This P1 unit-test
suite is a *correctness* check (not a benchmark): it builds the torchvision
C++/HIP operators in-tree and then runs the cuda-tagged cases of the functional
and transforms tensor suites, comparing GPU output against CPU/PIL references. A
"good" result means the ops build successfully and every selected GPU case
matches its reference within tolerance, confirming the image-transform pipeline
behaves correctly on a given ROCm GPU software stack.

ROCm stack components exercised:
    KFD + amdgpu kernel driver, ROCr + HIP runtime, the HIP device API and the
    hipcc compiler / hipify (the in-tree ``build_ext`` compiles the torchvision
    HIP ops), and the rocBLAS / MIOpen-style compute the resize / affine / warp
    kernels rely on.

Supported OS / environment profiles:
    Ubuntu 24.04, Alibaba Cloud Linux 3, Alibaba Cloud Linux 4, RHEL 10.1,
    SLES 15.7; bare-metal and container profiles.

Environment profiles:
    The source checkout, in-tree ops build, and the two pytest UT suites are
    performed in a single command via ``target_executor`` so the same test runs
    unchanged on a local bare-metal node, a remote SSH node, or inside a
    Docker/Podman container (``--container-mode``). On bare-metal the installed
    ROCm stack (``rock_dir``) is injected into the run environment; in container
    mode the ROCm stack and PyTorch shipped in the image are used as-is. Single-
    and multi-GPU on a single node; nightly cadence.

TorchVision repo URL + commit ("related commit"):
    Unlike Apex, which reads only a commit, TorchVision derives BOTH the repo URL
    and the commit from the pinned ``related_commits`` manifest -- field 6 is the
    fork URL (e.g. a ROCm/vision fork) and field 5 is the commit id -- so the
    clone always matches the exact downstream fork the image was built for. On
    bare-metal (no prebuilt PyTorch / manifest) the commit must be supplied via
    ``TORCHVISION_COMMIT`` and the URL defaults to the ROCm/vision repository
    (override with ``TORCHVISION_URL``). In a prebuilt-PyTorch container both are
    read from the manifest; ``TORCHVISION_COMMIT`` / ``TORCHVISION_URL`` always
    override the manifest lookup. See ``_workload.py`` for all env knobs.

GPU count:
    By default the suite exposes EVERY GPU the node/container exposes. Set
    ``TORCHVISION_NUM_GPUS=<n>`` to cap it (e.g. ``1`` for single-GPU). On
    bare-metal this drives ``gpu_count`` acquisition; in container mode -- where
    the executor otherwise pins one GPU -- visibility is set via
    ``ROCR_VISIBLE_DEVICES`` in the run command.

Markers:
    Injected by the CATEGORY_PROFILES entry for
    ``tests/e2e/ml_frameworks/torchvision`` in taxonomy.py -- hw.gpu +
    hw.multi_gpu (single- and multi-GPU profiles), layer.runtime + layer.math_lib
    (HIP runtime/compiler and the rocBLAS/MIOpen-style compute the transform ops
    use), ci.nightly, e2e.stack, os.linux.

    Declared on the test function (profiles never inject these):
        gpu_count(all|N) -- acquire all GPUs (default) or N when
                            TORCHVISION_NUM_GPUS set
        runtime.soak     -- the in-tree ops build plus the two UT suites run for a
                            long time; scheduled nightly for the ML-framework cadence
"""

import logging
import re

import pytest

from tests.e2e.ml_frameworks.torchvision._result_parser import parse_pytest_output
from tests.e2e.ml_frameworks.torchvision._workload import (
    GPU_COUNT_ARG,
    PYTEST_SELECTOR,
    RELATED_COMMITS_PATH,
    RUN_TIMEOUT,
    TEST_FILES,
    TORCHVISION_COMMIT,
    TORCHVISION_NUM_GPUS,
    TORCHVISION_URL,
    TORCHVISION_URL_OVERRIDE,
    WORK_DIR,
)

logger = logging.getLogger(__name__)

# A git ref safe to interpolate into a shell command (no metacharacters).
_SAFE_REF_RE = re.compile(r"^[0-9A-Za-z._/-]+$")
# A git URL safe to interpolate into a shell command.
_SAFE_URL_RE = re.compile(r"^https?://[0-9A-Za-z._~:/?#@!$&'()*+,;=%-]+$")
# A commit id as validated by the source test: 7-40 hex chars.
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
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

# Shell snippet run inside a prebuilt-PyTorch container to read the torchvision
# repo URL (field 6) and commit (field 5) from its ``related_commits`` manifest.
# It locates the manifest (explicit path, then well-known locations, then a
# bounded find), greps the torchvision line for this OS (distro id from
# /etc/os-release), and prints both fields. Sentinel markers on stdout let the
# caller distinguish "not found" from "no torchvision entry".
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
  echo "__TV_RC_NOTFOUND__"; exit 0
fi
osid=$(. /etc/os-release 2>/dev/null; echo "$ID")
line=$(grep -i torchvision "$f" | grep -i "$osid" | head -1)
if [ -z "$line" ]; then
  line=$(grep -i torchvision "$f" | head -1)
fi
if [ -z "$line" ]; then
  echo "__TV_RC_NOENTRY__:$f"; exit 0
fi
url=$(echo "$line" | cut -d '|' -f 6 | tr -d '[:space:]')
commit=$(echo "$line" | cut -d '|' -f 5 | tr -d '[:space:]')
echo "__TV_URL__:$url"
echo "__TV_COMMIT__:$commit"
"""

# The hw.* / layer.* / ci.* / e2e.* / os.* markers are injected by the
# CATEGORY_PROFILES entry for ``tests/e2e/ml_frameworks/torchvision`` in
# taxonomy.py (both hw.gpu and hw.multi_gpu, so the suite covers the single- and
# multi-GPU profiles). Only the parametric ``gpu_count`` and the ``runtime.*``
# weight -- which profiles intentionally never inject -- are declared on the test.


def _visible_devices_prefix(request) -> str:
    """Return a command prefix that exposes the right number of GPUs, or ``""``.

    Only meaningful in container mode. ``target_executor``'s container path pins a
    single GPU (``ROCR_VISIBLE_DEVICES=0``) regardless of ``gpu_count``, so GPU
    visibility for the UT run is controlled here instead:

        * ``TORCHVISION_NUM_GPUS`` unset -> drop the restriction so every GPU
          passed into the container (via ``--device``) is visible.
        * ``TORCHVISION_NUM_GPUS=k``     -> expose exactly GPUs ``0..k-1``.

    On bare-metal / SSH this returns ``""`` -- ``target_executor`` owns the real
    ``ROCR_VISIBLE_DEVICES`` allocation from the acquired ``gpu_count`` slots, and
    the test must not override it.
    """
    if not request.config.getoption("--container-mode", default=False):
        return ""
    if TORCHVISION_NUM_GPUS is None:
        return "env -u ROCR_VISIBLE_DEVICES "
    indices = ",".join(str(i) for i in range(TORCHVISION_NUM_GPUS))
    return f"env ROCR_VISIBLE_DEVICES={indices} "


def _rocm_env_prefix(request) -> str:
    """Return the ``env VAR=... `` prefix for the UT runner, or ``""``.

    In container mode the ROCm stack and PyTorch shipped inside the image are used
    as-is -- injecting host paths would point the build at a tree that does not
    exist in the container. On bare-metal / SSH the installed ROCm tree
    (``rock_dir``) and its ``LD_LIBRARY_PATH`` are injected so the ops build and
    the ROCm compute libraries resolve against the intended stack.

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
        pytest.fail(f"torchvision commit id contains unsafe characters: {ref!r}")
    if not _COMMIT_RE.match(ref):
        pytest.fail(f"torchvision commit id is not a valid 7-40 char hex sha: {ref!r}")
    return ref


def _validate_url(url: str) -> str:
    """Return *url* if it is a safe http(s) git URL, else fail."""
    if not url.startswith("http"):
        pytest.fail(f"torchvision repo URL is not an http(s) URL: {url!r}")
    if not _SAFE_URL_RE.match(url):
        pytest.fail(f"torchvision repo URL contains unsafe characters: {url!r}")
    return url


def _lookup_in_container(target_executor) -> tuple[str, str]:
    """Read the torchvision repo URL + commit from the container's manifest.

    Runs the lookup snippet inside the (prebuilt-PyTorch) container and interprets
    its sentinel output, returning ``(url, commit)``. Fails with actionable
    guidance when the manifest is absent or carries no torchvision entry -- the two
    states the caller must distinguish.
    """
    explicit = RELATED_COMMITS_PATH
    if explicit and not _SAFE_PATH_RE.match(explicit):
        pytest.fail(f"TORCHVISION_RELATED_COMMITS path contains unsafe characters: {explicit!r}")

    result = target_executor.run(_RELATED_COMMITS_LOOKUP.format(explicit=explicit), timeout=_RESOLVE_TIMEOUT)
    out = f"{result.stdout}\n{result.stderr}"

    url = ""
    commit = ""
    for line in out.splitlines():
        token = line.strip()
        if token.startswith("__TV_URL__:"):
            url = token.split(":", 1)[1].strip()
        elif token.startswith("__TV_COMMIT__:"):
            commit = token.split(":", 1)[1].strip()

    if "__TV_RC_NOTFOUND__" in out:
        pytest.fail(
            "related_commits manifest was not found inside the container. Use a prebuilt "
            "PyTorch container image that ships a related_commits file, or pass the "
            "torchvision commit explicitly via TORCHVISION_COMMIT=<commit> (and optionally "
            "TORCHVISION_URL=<repo>). Set TORCHVISION_RELATED_COMMITS=<path> to point at the "
            "manifest directly."
        )
    if "__TV_RC_NOENTRY__" in out:
        pytest.fail(
            "related_commits manifest was found but contains no torchvision entry for this OS. "
            "Pass the torchvision commit explicitly via TORCHVISION_COMMIT=<commit> (and "
            f"optionally TORCHVISION_URL=<repo>).\nLookup output:\n{out[-2000:]}"
        )
    if commit:
        logger.info("torchvision commit resolved from related_commits manifest: %s", commit)
        if url:
            logger.info("torchvision repo URL resolved from related_commits manifest: %s", url)
        return url, commit

    pytest.fail(f"Could not resolve the torchvision URL/commit from related_commits:\n{out[-2000:]}")
    return "", ""  # unreachable -- pytest.fail raises


def _resolve_url_and_commit(request, target_executor) -> tuple[str, str]:
    """Determine the torchvision repo URL and commit to check out.

    Resolution order:
        1. ``TORCHVISION_COMMIT`` supplied by the user -- honored in every profile.
           The URL then comes from ``TORCHVISION_URL`` (default ROCm/vision).
        2. Container mode only: field 6 (URL) and field 5 (commit) of the
           torchvision entry in the image's ``related_commits`` manifest. A
           user-supplied ``TORCHVISION_URL`` still overrides the manifest URL.
        3. Otherwise (bare-metal, no commit) fail with guidance -- bare-metal has
           no prebuilt PyTorch / related_commits file, so the commit is required.
    """
    if TORCHVISION_COMMIT:
        logger.info("torchvision commit supplied by user (TORCHVISION_COMMIT): %s", TORCHVISION_COMMIT)
        return _validate_url(TORCHVISION_URL), _validate_ref(TORCHVISION_COMMIT)

    if request.config.getoption("--container-mode", default=False):
        manifest_url, manifest_commit = _lookup_in_container(target_executor)
        # A user-supplied URL override wins; otherwise use field 6, falling back to
        # the default ROCm/vision repo if the manifest omitted it.
        url = TORCHVISION_URL_OVERRIDE or manifest_url or TORCHVISION_URL
        return _validate_url(url), _validate_ref(manifest_commit)

    pytest.fail(
        "A torchvision commit id is required on bare-metal, where no prebuilt PyTorch / "
        "related_commits manifest is present. Provide it on the command line, e.g.\n"
        "  TORCHVISION_COMMIT=<commit> pytest "
        "tests/e2e/ml_frameworks/torchvision/test_torchvision.py --rock-dir /opt/rocm ..."
    )
    return "", ""  # unreachable -- pytest.fail raises


@pytest.mark.gpu_count(GPU_COUNT_ARG)
@pytest.mark.runtime.soak
def test_torchvision_p1_ut_suite(request, target_executor):
    """Clone torchvision, build the ops, run the GPU UT suites, and assert they pass.

    The repo URL and commit are resolved first (user-supplied
    ``TORCHVISION_COMMIT`` / ``TORCHVISION_URL``, else the container image's
    ``related_commits`` manifest -- URL from field 6, commit from field 5). The
    checkout, first-run in-tree ops build (``build_ext --inplace``), a
    ``torchvision::nms`` import check, and the two cuda-tagged pytest UT suites
    then execute in a single ``target_executor`` command so the workflow is
    identical on bare-metal, remote SSH, and container profiles -- the source tree
    is always created wherever the GPU command runs. All GPUs are exposed by
    default (``TORCHVISION_NUM_GPUS`` caps the count). The verbose pytest output is
    parsed per case so the assertion can name any failing or erroring case.
    """
    url, commit = _resolve_url_and_commit(request, target_executor)
    vis_prefix = _visible_devices_prefix(request)
    env_prefix = _rocm_env_prefix(request)
    src_dir = f"{WORK_DIR}/src"
    short = commit[:7]

    run_prefix = f"{vis_prefix}{env_prefix}"
    # The two GPU UT suites, restricted to cuda-tagged cases. ``|| true`` is
    # intentionally NOT used: a non-zero pytest exit is a real signal, cross-checked
    # against the parsed per-case outcomes and the crash markers below.
    suite_cmds = " && ".join(
        f"{run_prefix}python -m pytest {test_file} -v -k {PYTEST_SELECTOR}" for test_file in TEST_FILES
    )

    # A fresh checkout each run avoids stale build state. ``build_ext --inplace``
    # compiles the torchvision C++/HIP ops so ``torch.ops.torchvision.nms`` (and
    # the transform ops) resolve; the nms import gates the UT run. ``vis_prefix``
    # sets GPU visibility in container mode (bare-metal leaves ROCR to the executor).
    nms_check = "import torch, torchvision; torch.ops.torchvision.nms; print('torchvision_nms_ok')"
    cmd = (
        "set -e; "
        f"rm -rf {WORK_DIR}; mkdir -p {WORK_DIR}; "
        f"git clone {url} {src_dir}; "
        f"cd {src_dir} && git checkout {commit}; "
        f'git log -1 --format="HEAD is now at %h" | grep -q "HEAD is now at {short}"; '
        f"{run_prefix}python setup.py build_ext --inplace; "
        f'{run_prefix}python -c "{nms_check}" | grep -q torchvision_nms_ok; '
        f"cd {src_dir} && {suite_cmds}"
    )

    gpu_label = "all" if TORCHVISION_NUM_GPUS is None else TORCHVISION_NUM_GPUS
    logger.info(
        "TorchVision P1 UT suite starting in %s (url=%s, commit=%s, num_gpus=%s)",
        src_dir,
        url,
        commit,
        gpu_label,
    )
    result = target_executor.run(cmd, timeout=RUN_TIMEOUT)

    # pytest writes its verbose result lines to stdout; parse both streams so a
    # crash trace on stderr is still seen.
    combined = f"{result.stdout}\n{result.stderr}"
    summary = parse_pytest_output(combined)

    crash_markers = [m for m in _CRASH_MARKERS if m in combined]

    logger.info(
        "TorchVision UT results: passed=%d skipped=%d failed=%d errored=%d unresolved=%d "
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

    # No parsed results at all means the workflow never reached the suites (clone,
    # checkout, ops build, or the nms import check failed) -- surface that.
    assert summary.total > 0 or summary.ran_total > 0, (
        f"TorchVision UT suite produced no test results (exit={result.exit_code}); "
        f"the clone, ops build, nms import check, or runner likely failed to start:\n"
        f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-4000:]}"
    )

    # A clean run requires: no failed/errored/unresolved cases, a zero exit code,
    # and no GPU crash signature. exit_code and crash_markers are essential
    # backstops -- a fault that aborts pytest mid-case (memory-access fault, core
    # dump) can leave the last case without an outcome, and must never be reported
    # as a pass.
    completed_cleanly = summary.is_clean and result.exit_code == 0 and not crash_markers
    assert completed_cleanly, (
        f"TorchVision UT suite did not complete cleanly "
        f"(exit={result.exit_code}, crash_markers={crash_markers or 'none'}, "
        f"failed={summary.failed}, errored={summary.errored}, "
        f"unresolved={len(summary.unresolved_names)}, "
        f"passed={summary.passed}, skipped={summary.skipped}):\n"
        f"failed: {summary.failed_names[:50]}\n"
        f"errored: {summary.errored_names[:50]}\n"
        f"unresolved (crashed mid-test): {summary.unresolved_names[:50]}\n"
        f"stdout tail: {result.stdout[-3000:]}\nstderr tail: {result.stderr[-3000:]}"
    )
