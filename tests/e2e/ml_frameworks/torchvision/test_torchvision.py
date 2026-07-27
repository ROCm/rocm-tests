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

Each suite file is run as its own independent test (parametrized below), so a
failure in one suite never prevents the other from running.

ROCm stack components exercised:
    KFD + amdgpu kernel driver, ROCr + HIP runtime, the HIP device API and the
    hipcc compiler / hipify (the in-tree ``build_ext`` compiles the torchvision
    HIP ops), and the rocBLAS / MIOpen-style compute the resize / affine / warp
    kernels rely on.

Supported architectures:
    gfx1101, gfx1100, gfx950, gfx942, gfx90a, gfx908.

Supported OS / environment profiles:
    Ubuntu 24.04, Alibaba Cloud Linux 3, Alibaba Cloud Linux 4, RHEL 10.1,
    SLES 15.7; bare-metal and container profiles.

Environment profiles:
    Per suite, the source checkout, in-tree ops build, and the pytest UT run are
    performed in a single command via ``target_executor`` so the same test runs
    unchanged on a local bare-metal node, a remote SSH node, or inside a
    Docker/Podman container (``--container-mode``). On bare-metal the installed
    ROCm stack (``rock_dir``) is injected into the run environment; in container
    mode the ROCm stack and PyTorch shipped in the image are used as-is. Single-
    and multi-GPU on a single node; nightly cadence.

Repo URL + commit ("related commit"):
    Both the fork URL and the commit are read from the pinned ``related_commits``
    manifest inside the prebuilt-PyTorch container -- field 6 is the fork URL and
    field 5 is the commit id -- so the clone matches the exact downstream fork the
    image was built for. On bare-metal (no prebuilt PyTorch / manifest) the commit
    must be supplied via ``TORCHVISION_COMMIT`` and the URL defaults to the
    ROCm/vision repository (override with ``TORCHVISION_URL``). ``TORCHVISION_COMMIT``
    / ``TORCHVISION_URL`` always override the manifest lookup.

GPU count:
    By default the suite exposes EVERY GPU the node/container exposes. Set
    ``TORCHVISION_NUM_GPUS=<n>`` to cap it (e.g. ``1`` for single-GPU). On
    bare-metal this drives ``gpu_count`` acquisition; in container mode -- where
    the executor otherwise pins one GPU -- visibility is set via
    ``ROCR_VISIBLE_DEVICES`` in the run command.

Build directory:
    The checkout + ops build live under the framework-managed build dir
    (``<compiler_build_dir>/ml_frameworks/torchvision/<suite>``, default base
    ``output/test-binaries/``) so the build state is reviewable; override with
    ``TORCHVISION_WORK_DIR``.

Markers:
    hw.multi_gpu / layer.math_lib / ci.nightly / e2e.stack / os.linux are injected
    by the CATEGORY_PROFILES entry for this directory in taxonomy.py. gpu_count and
    runtime.fast (the suite completes in a few minutes) are declared on the test.
"""

import logging
import os
import re

import pytest

from tests.e2e.ml_frameworks.torchvision._result_parser import parse_junit_xml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Run parameters (env-configurable). See the module docstring for semantics.
# ---------------------------------------------------------------------------

# Default public torchvision source tree, used when no manifest URL is available.
_DEFAULT_TORCHVISION_URL = "https://github.com/ROCm/vision"

# Raw, user-supplied URL override (empty when unset); wins over the manifest URL.
TORCHVISION_URL_OVERRIDE = os.environ.get("TORCHVISION_URL", "").strip()
TORCHVISION_URL = TORCHVISION_URL_OVERRIDE or _DEFAULT_TORCHVISION_URL

# User-supplied commit id. Required on bare-metal; overrides the manifest in a container.
TORCHVISION_COMMIT = os.environ.get("TORCHVISION_COMMIT", "").strip()

# Optional explicit path to the related_commits manifest inside the container.
RELATED_COMMITS_PATH = os.environ.get("TORCHVISION_RELATED_COMMITS", "").strip()

# GPUs to use on one node: None -> every GPU exposed; an integer caps the count.
_NUM_GPUS_RAW = os.environ.get("TORCHVISION_NUM_GPUS", "").strip()
TORCHVISION_NUM_GPUS = int(_NUM_GPUS_RAW) if _NUM_GPUS_RAW else None

# Argument for @pytest.mark.gpu_count: an explicit int, else the "all" sentinel.
GPU_COUNT_ARG = TORCHVISION_NUM_GPUS if TORCHVISION_NUM_GPUS is not None else "all"

# Optional override for the checkout + build scratch dir (else compiler_build_dir).
WORK_DIR_OVERRIDE = os.environ.get("TORCHVISION_WORK_DIR", "").strip()

# The GPU UT suite files, restricted to the cuda-tagged cases via ``-k cuda``. Each
# runs as an independent test (parametrized below).
TEST_FILES = (
    "test/test_functional_tensor.py",
    "test/test_transforms_tensor.py",
)
PYTEST_SELECTOR = "cuda"

# Whole-workflow wall-clock cap (seconds) per suite: clone + ops build + one UT run.
RUN_TIMEOUT = float(os.environ.get("TORCHVISION_RUN_TIMEOUT", "14400"))

# Sentinels bracketing the JUnit XML report catted onto stdout after the run.
_JUNIT_START = "__TV_JUNIT_START__"
_JUNIT_END = "__TV_JUNIT_END__"

# A git ref/URL/path safe to interpolate into a shell command.
_SAFE_REF_RE = re.compile(r"^[0-9A-Za-z._/-]+$")
_SAFE_URL_RE = re.compile(r"^https?://[0-9A-Za-z._~:/?#@!$&'()*+,;=%-]+$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_SAFE_PATH_RE = re.compile(r"^[0-9A-Za-z._/-]+$")

# Seconds allowed for the (trivial) in-container related_commits lookup.
_RESOLVE_TIMEOUT = 120.0

# Hard-crash signatures. If any appears in the output the run aborted mid-way (the
# process died before writing its report), so the run is a failure regardless of
# how many cases were recorded.
_CRASH_MARKERS = (
    "Memory access fault",
    "core dumped",
    "Segmentation fault",
    "HSA_STATUS_ERROR",
    "Aborted (",
    "Fatal Python error",
)

# Shell snippet run inside a prebuilt-PyTorch container to read the torchvision repo
# URL (field 6) and commit (field 5) from its related_commits manifest. It locates
# the manifest (explicit path, then well-known locations, then a bounded find),
# greps the torchvision line for this OS, and prints both fields with sentinels so
# the caller can distinguish "not found" from "no torchvision entry".
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


def _visible_devices_prefix(request) -> str:
    """Return a command prefix that exposes the right number of GPUs, or ``""``.

    Only meaningful in container mode, where ``target_executor`` pins a single GPU:
    unset ``TORCHVISION_NUM_GPUS`` drops the restriction so every passed-in GPU is
    visible; an integer ``k`` exposes GPUs ``0..k-1``. On bare-metal / SSH this
    returns ``""`` -- ``target_executor`` owns the real ``ROCR_VISIBLE_DEVICES``.
    """
    if not request.config.getoption("--container-mode", default=False):
        return ""
    if TORCHVISION_NUM_GPUS is None:
        return "env -u ROCR_VISIBLE_DEVICES "
    indices = ",".join(str(i) for i in range(TORCHVISION_NUM_GPUS))
    return f"env ROCR_VISIBLE_DEVICES={indices} "


def _rocm_env_prefix(request) -> str:
    """Return the ``env VAR=... `` prefix for the UT runner, or ``""``.

    In container mode the ROCm stack and PyTorch in the image are used as-is. On
    bare-metal / SSH the installed ROCm tree (``rock_dir``) and its
    ``LD_LIBRARY_PATH`` are injected so the ops build and compute libraries resolve
    against the intended stack. ``rock_dir`` / ``ld_path`` are resolved lazily.
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
    its sentinel output, returning ``(url, commit)``. Fails with actionable guidance
    when the manifest is absent or carries no torchvision entry.
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

    Resolution order: (1) a user-supplied ``TORCHVISION_COMMIT`` (URL from
    ``TORCHVISION_URL``, default ROCm/vision); (2) container mode only -- field 6
    (URL) and field 5 (commit) of the image's related_commits manifest, with a
    user URL override winning; (3) otherwise fail with guidance (bare-metal has no
    manifest, so the commit is required).
    """
    if TORCHVISION_COMMIT:
        logger.info("torchvision commit supplied by user (TORCHVISION_COMMIT): %s", TORCHVISION_COMMIT)
        return _validate_url(TORCHVISION_URL), _validate_ref(TORCHVISION_COMMIT)

    if request.config.getoption("--container-mode", default=False):
        manifest_url, manifest_commit = _lookup_in_container(target_executor)
        url = TORCHVISION_URL_OVERRIDE or manifest_url or TORCHVISION_URL
        return _validate_url(url), _validate_ref(manifest_commit)

    pytest.fail(
        "A torchvision commit id is required on bare-metal, where no prebuilt PyTorch / "
        "related_commits manifest is present. Provide it on the command line, e.g.\n"
        "  TORCHVISION_COMMIT=<commit> pytest "
        "tests/e2e/ml_frameworks/torchvision/test_torchvision.py --rock-dir /opt/rocm ..."
    )
    return "", ""  # unreachable -- pytest.fail raises


def _extract_junit(text: str) -> str:
    """Return the JUnit XML report bracketed by the sentinels in *text*, or ``""``."""
    start = text.find(_JUNIT_START)
    end = text.find(_JUNIT_END)
    if start == -1 or end == -1 or end < start:
        return ""
    return text[start + len(_JUNIT_START) : end].strip()


@pytest.mark.gpu_count(GPU_COUNT_ARG)
@pytest.mark.runtime.fast
@pytest.mark.parametrize("test_file", TEST_FILES, ids=lambda f: os.path.basename(f)[len("test_") : -len(".py")])
def test_torchvision_p1_ut_suite(request, target_executor, compiler_build_dir, test_file):
    """Clone torchvision, build the ops, run one cuda-tagged UT suite, assert it passes.

    The repo URL and commit are resolved first (user-supplied ``TORCHVISION_COMMIT``
    / ``TORCHVISION_URL``, else the container image's related_commits manifest). The
    checkout, first-run in-tree ops build (``build_ext --inplace``), a
    ``torchvision::nms`` import check, and the cuda-tagged pytest UT run for
    ``test_file`` then execute in a single ``target_executor`` command so the workflow
    is identical on bare-metal, remote SSH, and container profiles. pytest writes a
    JUnit XML report which is parsed per case so the assertion can name any failing or
    erroring case.
    """
    url, commit = _resolve_url_and_commit(request, target_executor)
    vis_prefix = _visible_devices_prefix(request)
    env_prefix = _rocm_env_prefix(request)
    run_prefix = f"{vis_prefix}{env_prefix}"

    suite = os.path.basename(test_file)[len("test_") : -len(".py")]
    work_dir = WORK_DIR_OVERRIDE or os.path.join(compiler_build_dir, "ml_frameworks", "torchvision", suite)
    src_dir = f"{work_dir}/src"
    junit_xml = f"{work_dir}/junit.xml"
    short = commit[:7]

    # A fresh checkout each run avoids stale build state. ``build_ext --inplace``
    # compiles the torchvision C++/HIP ops so ``torch.ops.torchvision.nms`` (and the
    # transform ops) resolve; the nms import gates the UT run. ``set -e`` fails fast
    # on any setup step; ``set +e`` around pytest lets us capture its exit code and
    # always emit the JUnit report (even on failure) for parsing.
    nms_check = "import torch, torchvision; torch.ops.torchvision.nms; print('torchvision_nms_ok')"
    cmd = "\n".join(
        (
            "set -e",
            f"rm -rf {work_dir}; mkdir -p {work_dir}",
            f"git clone {url} {src_dir}",
            f"cd {src_dir} && git checkout {commit}",
            f'git log -1 --format="HEAD is now at %h" | grep -q "HEAD is now at {short}"',
            f"{run_prefix}python setup.py build_ext --inplace",
            f'{run_prefix}python -c "{nms_check}" | grep -q torchvision_nms_ok',
            "set +e",
            f"cd {src_dir} && {run_prefix}python -m pytest {test_file} -v -k {PYTEST_SELECTOR} "
            f"--junitxml={junit_xml} -p no:cacheprovider",
            "rc=$?",
            f"echo {_JUNIT_START}",
            f"cat {junit_xml} 2>/dev/null",
            f"echo {_JUNIT_END}",
            "exit $rc",
        )
    )

    gpu_label = "all" if TORCHVISION_NUM_GPUS is None else TORCHVISION_NUM_GPUS
    logger.info(
        "TorchVision P1 UT suite starting: file=%s (url=%s, commit=%s, num_gpus=%s, work_dir=%s)",
        test_file,
        url,
        commit,
        gpu_label,
        work_dir,
    )
    result = target_executor.run(cmd, timeout=RUN_TIMEOUT)

    combined = f"{result.stdout}\n{result.stderr}"
    summary = parse_junit_xml(_extract_junit(combined))
    crash_markers = [m for m in _CRASH_MARKERS if m in combined]

    logger.info(
        "TorchVision UT results [%s]: passed=%d skipped=%d failed=%d errored=%d " "(exit=%s, crash_markers=%s)",
        test_file,
        summary.passed,
        summary.skipped,
        summary.failed,
        summary.errored,
        result.exit_code,
        crash_markers or "none",
    )

    # No parsed results at all means the workflow never produced a report (clone,
    # checkout, ops build, nms import check, or the runner failed to start / crashed).
    assert summary.total > 0, (
        f"TorchVision UT suite produced no test results for {test_file} (exit={result.exit_code}); "
        f"the clone, ops build, nms import check, or runner likely failed to start or crashed:\n"
        f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-4000:]}"
    )

    # A clean run requires no failed/errored cases, a zero exit code, and no GPU
    # crash signature -- exit_code and crash_markers are essential backstops so a
    # fault that aborts pytest mid-run can never be reported as a pass.
    completed_cleanly = summary.is_clean and result.exit_code == 0 and not crash_markers
    assert completed_cleanly, (
        f"TorchVision UT suite did not complete cleanly for {test_file} "
        f"(exit={result.exit_code}, crash_markers={crash_markers or 'none'}, "
        f"failed={summary.failed}, errored={summary.errored}, "
        f"passed={summary.passed}, skipped={summary.skipped}):\n"
        f"failed: {summary.failed_names[:50]}\n"
        f"errored: {summary.errored_names[:50]}\n"
        f"stdout tail: {result.stdout[-3000:]}\nstderr tail: {result.stderr[-3000:]}"
    )
