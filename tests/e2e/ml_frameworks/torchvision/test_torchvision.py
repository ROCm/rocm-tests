# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TorchVision P1 image-transform correctness UT suite.

Runs the cuda-tagged tensor UT suites in a container against torchvision ops.
The ops are built once per process (inside the container); subsequent parametrized
test cases reuse the built tree without re-running git-clean or setup.py.
The JUnit report is parsed so the assertion names any failing GPU case.
"""

import logging
import os

import pytest

from tests.e2e.ml_frameworks.torchvision._constants import (
    _RESOLVE_TIMEOUT,
    CONTAINER_MOUNT_FLAGS,
    GPU_COUNT_ARG,
    PYTEST_SELECTOR,
    RUN_TIMEOUT,
    TEST_FILES,
)
from tests.e2e.ml_frameworks.torchvision._result_parser import parse_junit_xml

logger = logging.getLogger(__name__)

# Sentinels bracketing the JUnit XML report catted onto stdout after the run.
_JUNIT_START = "__TV_JUNIT_START__"
_JUNIT_END = "__TV_JUNIT_END__"

# Hard-crash signatures: a match means the run aborted mid-way (the process died
# before writing its report), so the run is a failure regardless of parsed cases.
_CRASH_MARKERS = (
    "Memory access fault",
    "core dumped",
    "Segmentation fault",
    "HSA_STATUS_ERROR",
    "Aborted (",
    "Fatal Python error",
)

# Tracks repos that have already been built this process so the second
# parametrized test reuses the .so without git-cleaning and rebuilding.
_built_repos: set[str] = set()

# Shell command that runs inside the container to:
#   1. Read the torchvision commit from the image's related_commits manifest.
#   2. Check out that exact commit in the bind-mounted clone.
#   3. Build the in-tree ops.
# The manifest lives at /workspace/pytorch/related_commits inside the image.
# git clean removes stale hipify/build artifacts from a prior run on the same tree.
_SETUP_AND_BUILD_CMD = r"""
set -e
repo={repo}
git config --global --add safe.directory "$repo"
cd "$repo"

# Locate the related_commits manifest inside the container.
for c in /workspace/pytorch/related_commits \
          "$PYTORCH_DIR/related_commits" \
          /opt/pytorch/related_commits \
          /related_commits; do
  [ -f "$c" ] && { manifest="$c"; break; }
done
if [ -z "$manifest" ]; then
  tdir=$(python -c 'import os,torch;print(os.path.dirname(os.path.dirname(torch.__file__)))' 2>/dev/null)
  [ -f "$tdir/related_commits" ] && manifest="$tdir/related_commits"
fi
if [ -z "$manifest" ]; then
  manifest=$(find / -maxdepth 6 -name related_commits -type f 2>/dev/null | head -1)
fi

# Extract the torchvision commit from the manifest (field 5, pipe-delimited).
if [ -n "$manifest" ] && [ -f "$manifest" ]; then
  osid=$(. /etc/os-release 2>/dev/null; echo "$ID")
  line=$(grep -i torchvision "$manifest" | grep -i "$osid" | head -1)
  [ -z "$line" ] && line=$(grep -i torchvision "$manifest" | head -1)
  commit=$(echo "$line" | cut -d '|' -f 5 | tr -d '[:space:]')
fi

# Checkout the resolved commit; fall back to current HEAD if not found.
if [ -n "$commit" ]; then
  git fetch origin "$commit" --depth=1 2>/dev/null || true
  git checkout "$commit"
fi

# Build the in-tree ops.
git clean -fdx
python setup.py build_ext --inplace
python -c "import torch, torchvision; torch.ops.torchvision.nms; print('torchvision_nms_ok')" \
  | grep -q torchvision_nms_ok
"""


def _extract_junit(text: str) -> str:
    """Return the JUnit XML report bracketed by the sentinels in *text*, or ``""``."""
    start = text.find(_JUNIT_START)
    end = text.find(_JUNIT_END)
    if start == -1 or end == -1 or end < start:
        return ""
    return text[start + len(_JUNIT_START) : end].strip()


@pytest.mark.container(extra_run_flags=CONTAINER_MOUNT_FLAGS)
@pytest.mark.gpu_count(GPU_COUNT_ARG)
@pytest.mark.runtime.medium
@pytest.mark.parametrize("test_file", TEST_FILES, ids=lambda f: os.path.basename(f)[len("test_") : -len(".py")])
def test_torchvision_p1_ut_suite(target_executor, torchvision_repo, test_file):
    """Build (once) and run one cuda-tagged UT suite; assert it passes.

    On the first call, reads the torchvision commit from the container's
    related_commits manifest, checks it out in the bind-mounted clone, and builds
    the in-tree ops.  Subsequent parametrized calls skip the build and reuse the
    .so.  The JUnit report is parsed so the assertion names any failing GPU case.
    """
    # Build inside the container on the first parametrized call only.
    # _built_repos prevents the second test from re-running git-clean and
    # setup.py, which would wipe the .so the first test already built.
    if torchvision_repo not in _built_repos:
        build_result = target_executor.run(
            _SETUP_AND_BUILD_CMD.format(repo=torchvision_repo),
            timeout=_RESOLVE_TIMEOUT * 10,
        )
        if build_result.exit_code != 0:
            pytest.fail(
                f"torchvision ops build failed (exit={build_result.exit_code}):\n"
                f"stdout: {build_result.stdout[-3000:]}\nstderr: {build_result.stderr[-3000:]}"
            )
        _built_repos.add(torchvision_repo)

    suite = os.path.basename(test_file)[len("test_") : -len(".py")]
    junit = f"junit_{suite}.xml"

    cmd = "\n".join(
        (
            f"cd {torchvision_repo}",
            f"python -m pytest {test_file} -v -k {PYTEST_SELECTOR} --junitxml={junit} -p no:cacheprovider",
            "rc=$?",
            f"echo {_JUNIT_START}",
            f"cat {junit} 2>/dev/null",
            f"echo {_JUNIT_END}",
            "exit $rc",
        )
    )

    logger.info("TorchVision P1 UT suite starting: file=%s", test_file)
    result = target_executor.run(cmd, timeout=RUN_TIMEOUT)

    combined = f"{result.stdout}\n{result.stderr}"
    summary = parse_junit_xml(_extract_junit(combined))
    crash_markers = [m for m in _CRASH_MARKERS if m in combined]

    logger.info(
        "TorchVision UT results [%s]: passed=%d skipped=%d failed=%d errored=%d (exit=%s, crash_markers=%s)",
        test_file,
        summary.passed,
        summary.skipped,
        summary.failed,
        summary.errored,
        result.exit_code,
        crash_markers or "none",
    )

    # Crash markers take priority: if the process was killed by a signal, report
    # the crash explicitly rather than the secondary "no results" symptom.
    assert not crash_markers, (
        f"TorchVision UT runner crashed for {test_file} "
        f"(exit={result.exit_code}, crash_markers={crash_markers}):\n"
        f"stdout tail: {result.stdout[-4000:]}\nstderr tail: {result.stderr[-4000:]}"
    )

    # No parsed results means the ops build or runner failed to produce a report.
    assert summary.total > 0, (
        f"TorchVision UT suite produced no test results for {test_file} (exit={result.exit_code}); "
        f"the ops build or runner likely failed to start:\n"
        f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-4000:]}"
    )

    # Clean run: no failed/errored cases and zero exit.
    completed_cleanly = summary.is_clean and result.exit_code == 0
    assert completed_cleanly, (
        f"TorchVision UT suite did not complete cleanly for {test_file} "
        f"(exit={result.exit_code}, failed={summary.failed}, errored={summary.errored}, "
        f"passed={summary.passed}, skipped={summary.skipped}):\n"
        f"failed: {summary.failed_names[:50]}\n"
        f"errored: {summary.errored_names[:50]}\n"
        f"stdout tail: {result.stdout[-3000:]}\nstderr tail: {result.stderr[-3000:]}"
    )
