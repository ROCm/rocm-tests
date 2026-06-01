# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
reports_plugin.py -- Allure marker-to-label mapping, result classification, and dashboard.

Responsibilities:
    - Map the 6-dimension marker taxonomy to Allure dynamic labels at test
      start (severity, feature, story, epic, layer tags).
    - Wire the 8-outcome classifier into the Allure report as a custom label.
    - Provide the ``outcome_fixture`` which exposes the classified Outcome for
      post-test assertions or downstream reporting.
    - Register ``--allure-log-name NAME`` CLI option: stamps the run with a
      human-readable label written to ``executor.json`` in the allure-results
      directory so it appears as the run title in the Allure dashboard.
    - Register ``--allure-db N`` CLI option: archives each run's results to
      ``build/allure-db/<timestamp>/`` and generates an Allure HTML dashboard
      combining the current run with the previous N build results.

Allure label mapping:
    hw.*     → allure.severity (GPU → CRITICAL, CPU_ONLY → MINOR)
    ci.*     → allure.feature (pr, nightly, weekly)
    layer.*  → allure.story   (driver, runtime, math_lib, ...)
    e2e.*    → allure.epic    (stack, multinode, app, upgrade)
    os.*     → allure.tag
    runtime.*→ allure.tag

The marker-to-Allure mapping runs at test setup so that the labels appear even
if the test fails early. They require an active test context and therefore cannot
run during collection.
"""

from __future__ import annotations

import datetime
import logging
import os
import time

import pytest

from framework.markers.taxonomy import ALLURE_DIMENSION_MAP, HW_SEVERITY_MAP

logger = logging.getLogger(__name__)

# Module-level session log path — set once by pytest_sessionstart (in every
# process: master and xdist workers).  Used by pytest_runtest_logstart to write
# [RUNNING] banners without requiring the hook to receive a config/session param.
_g_session_log_path: str | None = None


def _append_session_log(msg: str) -> None:
    """Append *msg* to the session log file (best-effort, append-safe).

    Uses the module-level ``_g_session_log_path``.  No-op when the path has not
    been initialised (e.g. when ``--no-gpu`` disables session-log setup).

    Args:
        msg: Text to append; a trailing newline is added automatically.
    """
    import pathlib

    if not _g_session_log_path:
        return
    try:
        with pathlib.Path(_g_session_log_path).open("a", encoding="utf-8") as _f:
            _f.write(msg + "\n")
    except OSError:
        pass


def pytest_sessionstart(session: pytest.Session) -> None:
    """Wire the session log path into the module-level global for hook access.

    The path and file setup (truncation on master, directory creation) are handled
    earlier in ``pytest_configure`` so xdist workers cannot race with the master's
    truncation.  Here we only copy the already-resolved absolute path into the
    module-level ``_g_session_log_path`` used by ``pytest_runtest_logstart``.
    """
    global _g_session_log_path
    _g_session_log_path = getattr(session.config, "_session_log_path", None)
    if _g_session_log_path:
        logger.info("Session log: %s", _g_session_log_path)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --allure-log-name and --allure-db options."""
    group = parser.getgroup("rocm-reports", "ROCm reporting options")
    group.addoption(
        "--allure-log-name",
        action="store",
        default=None,
        metavar="NAME",
        help=(
            "Label this Allure run with a custom name (e.g. 'nightly-gfx942-rc1'). "
            "Written to executor.json in the allure-results directory and displayed "
            "as the run title in the Allure dashboard."
        ),
    )
    group.addoption(
        "--allure-db",
        action="store",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Archive this run's allure results and generate an Allure HTML dashboard "
            "combining the current run with the previous N build results. "
            "Requires the allure CLI on PATH. 0 = disabled (default)."
        ),
    )


def pytest_runtest_logstart(nodeid: str, location) -> None:
    """Print a RUNNING status line when a test begins, including the xdist worker ID.

    Fires in the worker process (or in the main process for non-xdist runs).
    ``PYTEST_XDIST_WORKER`` is set by pytest-xdist on each worker to its ID
    (``gw0``, ``gw1``, …).  Falls back to ``"main"`` in single-process mode.

    Format::

        [RUNNING gw0] tests/e2e/compiler/test_llvm.py::test_llvm_stress  (14:32:01)
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    msg = f"[RUNNING {worker}] {nodeid}  ({ts})"
    print(msg, flush=True)
    _append_session_log(msg)


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    """Set up session.log and apply ``reporting.allure_results_dir`` from ``rocm-test.toml``.

    **Session log setup** (master / non-xdist only):
    Resolves ``framework.session_log`` to an absolute path, creates its parent
    directory, and truncates the file to empty.  Storing an absolute path on
    ``config._session_log_path`` before xdist workers are spawned ensures every
    worker receives the exact same path via ``workerinput`` (injected by
    ``_XdistTopologyPlugin.pytest_configure_node``) and writes to the same file
    regardless of their working directory.

    Workers receive ``_session_log_path`` via ``workerinput`` and skip independent
    resolution entirely — see ``pytest_sessionstart``.

    **Allure dir setup:**
    When ``--alluredir`` is not passed on the CLI, sets the Allure results directory
    to the value configured in ``rocm-test.toml`` under ``[reporting] allure_results_dir``.
    The hook runs ``trylast`` so allure-pytest has already registered ``--alluredir``.

    Args:
        config: Active pytest config object.
    """
    import pathlib

    from framework.config.loader import load_config

    cfg = load_config(config_path=config.getoption("--rocm-config", default=None))

    # --- Session log: master truncates before any worker can write ---
    if not hasattr(config, "workerinput"):
        session_log = pathlib.Path(cfg.framework.session_log).resolve()
        session_log.parent.mkdir(parents=True, exist_ok=True)
        session_log.write_text("", encoding="utf-8")
        config._session_log_path = str(session_log)  # type: ignore[attr-defined]
        logger.info("Session log initialised: %s", session_log)
    else:
        # xdist worker: path is injected via workerinput by pytest_configure_node.
        # Fall back to resolving independently only when the injection is absent
        # (e.g., running without the remote_node_plugin).
        injected = config.workerinput.get("_session_log_path", "")
        if injected:
            config._session_log_path = injected  # type: ignore[attr-defined]
        else:
            session_log = pathlib.Path(cfg.framework.session_log).resolve()
            config._session_log_path = str(session_log)  # type: ignore[attr-defined]

    # --- Allure results dir ---
    try:
        alluredir = config.getoption("--alluredir", default=None)
    except (ValueError, AttributeError):
        return

    # CLI value wins — no override needed.
    if alluredir:
        return

    allure_dir = cfg.reporting.allure_results_dir
    for attr in ("allure_report_dir", "alluredir"):
        if hasattr(config.option, attr):
            setattr(config.option, attr, allure_dir)
            logger.info("Allure results dir (from rocm-test.toml): %s", allure_dir)
            break


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Map taxonomy markers to Allure dynamic labels at test setup."""
    try:
        import allure
    except ImportError:
        return

    for marker in item.iter_markers():
        name = marker.name
        if not name or "." not in name:
            continue
        dim, val = name.split(".", 1)
        label_type = ALLURE_DIMENSION_MAP.get(dim)
        if label_type is None:
            continue
        if label_type == "severity":
            allure.dynamic.severity(HW_SEVERITY_MAP.get(val, "normal"))
        elif label_type == "feature":
            allure.dynamic.feature(f"{dim}.{val}")
        elif label_type == "story":
            allure.dynamic.story(val)
        elif label_type == "epic":
            allure.dynamic.epic(val)
        elif label_type == "tag":
            allure.dynamic.tag(f"{dim}.{val}")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call) -> None:  # type: ignore[misc]
    """Attach each phase report (setup/call/teardown) to the item.

    Makes ``item.rep_setup``, ``item.rep_call``, and ``item.rep_teardown``
    available inside fixture teardown so the ``_test_execution_status``
    fixture can read the final pass/fail outcome without the test opting in.
    """
    outcome = yield
    setattr(item, f"rep_{call.when}", outcome.get_result())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _test_execution_status(request: pytest.FixtureRequest) -> None:  # type: ignore[misc]
    """Autouse: emit a structured status line for every test after it runs.

    Runs automatically for every test — no opt-in required.  The status line
    appears in the live log (``--log-cli-level=INFO``) and in caplog, giving
    CI logs a clear per-test summary without enabling verbose executor output.

    Format (single INFO line, or two lines on failure)::

        [PASSED]  tests/e2e/compiler/test_llvm.py::test_llvm_mem_intrinsic_stress  257.1s
                  ci=nightly  hw=gpu  layer=runtime
        [FAILED]  tests/e2e/stack_validation/test_hip_runtime.py::test_foo  0.1s
                  ci=nightly  hw=gpu  layer=runtime  Reason: AssertionError: binary exited 1

    When the *setup* phase fails (e.g. fixture error) and the call phase never
    ran, the status is reported as ``[ERROR (setup)]`` with the setup failure
    reason so the root cause is visible without reading the full traceback.
    """
    t0 = time.monotonic()
    yield
    duration = time.monotonic() - t0

    rep_call = getattr(request.node, "rep_call", None)
    rep_setup = getattr(request.node, "rep_setup", None)

    # Prefer call-phase report; fall back to setup failure
    if rep_call is not None:
        rep = rep_call
        if rep.passed:
            status = "PASSED"
        elif rep.skipped:
            status = "SKIPPED"
        else:
            status = "FAILED"
    elif rep_setup is not None and rep_setup.failed:
        rep = rep_setup
        status = "ERROR (setup)"
    else:
        return  # nothing to report (e.g. collection-only run)

    # Collect taxonomy marker dimensions for context (hw, ci, layer, …)
    dims: dict[str, str] = {}
    for marker in request.node.iter_markers():
        if marker.name and "." in marker.name:
            dim, val = marker.name.split(".", 1)
            dims[dim] = val
    dim_str = ("  " + "  ".join(f"{k}={v}" for k, v in sorted(dims.items()))) if dims else ""

    logger.info("[%s]  %s  %.1fs%s", status, request.node.nodeid, duration, dim_str)

    if not rep.passed and rep.longrepr:
        # Last line of longrepr carries the AssertionError / exception message.
        # Expand escaped \n sequences so multiline stdout/stderr renders line-by-line.
        reason = str(rep.longrepr).strip().splitlines()[-1]
        reason_lines = reason.replace("\\n", "\n").splitlines()
        logger.info("  Reason: %s", reason_lines[0])
        for line in reason_lines[1:]:
            logger.info("    %s", line)


@pytest.fixture
def outcome_fixture(request: pytest.FixtureRequest):
    """Classify the test outcome after execution and expose it for assertions.

    The classified Outcome is also attached to the Allure report as a label.

    Yields:
        A mutable container with ``.value`` populated post-test.
    """
    from framework.common.helpers import Outcome

    class OutcomeHolder:
        """Mutable container for the classified test outcome."""

        value: Outcome = Outcome.PASS

    holder = OutcomeHolder()
    yield holder

    # Auto-detect from actual test result when the test didn't explicitly set the holder.
    # rep_call is stored by pytest_runtest_makereport (tryfirst=True, hookwrapper=True)
    # before fixture teardown runs, so it is available here.
    rep = getattr(request.node, "rep_call", None)
    if rep is not None and holder.value == Outcome.PASS:
        if rep.failed:
            holder.value = Outcome.FAIL
        elif rep.skipped:
            holder.value = Outcome.SKIP

    # Post-test: attach outcome label to Allure
    try:
        import allure

        allure.dynamic.label("outcome", holder.value.value)
    except ImportError:
        pass
    logger.info("Test outcome: %s — %s", holder.value.value, request.node.nodeid)


# ---------------------------------------------------------------------------
# Allure run labelling (--allure-log-name)
# ---------------------------------------------------------------------------


def _write_executor_info(allure_results_dir: str, name: str) -> None:
    """Write ``executor.json`` to *allure_results_dir* so Allure displays *name*
    as the run title in the dashboard and history trend views.

    ``executor.json`` is an Allure-standard file read by ``allure generate``
    when building the HTML report. It does not require the allure CLI to be
    present at write time — only at report-generation time.

    Args:
        allure_results_dir: Path to the allure-results directory for this run.
        name:               Human-readable run label, e.g. ``"nightly-gfx942-rc1"``.
    """
    import json
    import pathlib

    results_path = pathlib.Path(allure_results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    executor = {
        "name": "rocm-test",
        "type": "other",
        "reportName": name,
    }
    executor_file = results_path / "executor.json"
    executor_file.write_text(json.dumps(executor, indent=2))
    logger.info("Allure run labelled %r → %s", name, executor_file)


# ---------------------------------------------------------------------------
# Allure dashboard generation (--allure-db)
# ---------------------------------------------------------------------------


def _build_allure_dashboard(allure_results_dir: str, n_previous: int) -> None:
    """Archive current run's allure results and generate a multi-run HTML dashboard.

    Workflow:
        1. Copy ``allure_results_dir`` → ``build/allure-db/<UTC-timestamp>/``.
        2. Select the last *n_previous* timestamped directories from ``allure-db/``
           that predate the one just archived.
        3. Call ``allure generate --clean -o build/allure-report <prev...> <current>``.

    The ``allure`` CLI must be on PATH (install via ``allure-commandline``).
    If it is absent the function logs a warning and returns without error.

    Args:
        allure_results_dir: Path to the current run's allure-results directory.
        n_previous:         Number of previous archived runs to include in the dashboard.
    """
    import datetime
    import pathlib
    import shutil
    import subprocess

    results_path = pathlib.Path(allure_results_dir)
    db_path = results_path.parent / "allure-db"
    db_path.mkdir(parents=True, exist_ok=True)

    # 1. Archive the current run's results
    run_ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    archive_target = db_path / run_ts
    if results_path.exists() and any(results_path.iterdir()):
        shutil.copytree(results_path, archive_target)
        logger.info("Archived allure results to %s", archive_target)
    else:
        logger.warning("--allure-db: no results found in %s to archive", results_path)

    # 2. Select the last N previously archived runs (excluding the one just written)
    previous_dirs = sorted(p for p in db_path.iterdir() if p.is_dir() and p != archive_target)[-n_previous:]

    # 3. Build source list: N previous archives + current results dir
    sources = [str(p) for p in previous_dirs]
    if results_path.exists():
        sources.append(str(results_path))

    if not sources:
        logger.warning("--allure-db: no result source directories found")
        return

    report_dir = results_path.parent / "allure-report"
    cmd = [*["allure", "generate", "--clean", "-o", str(report_dir)], *sources]
    logger.info("Generating Allure dashboard: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode == 0:
            logger.info(
                "Allure dashboard ready at %s (current run + %d previous run(s))",
                report_dir,
                len(previous_dirs),
            )
        else:
            logger.warning(
                "allure generate failed (rc=%d): %s",
                proc.returncode,
                proc.stderr.strip(),
            )
    except FileNotFoundError:
        logger.warning("allure CLI not found — install allure-commandline to generate the dashboard")
    except Exception as exc:
        logger.warning("Failed to generate Allure dashboard: %s", exc)


def _render_skipped_items(tw, stats) -> None:
    """Print the skipped-tests block with reasons."""
    skipped_items: list[tuple[str, str]] = []
    for report in stats.get("skipped", []):
        reason = ""
        if report.longrepr:
            if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
                reason = str(report.longrepr[2])
            else:
                reason = str(report.longrepr).strip().splitlines()[-1]
        skipped_items.append((report.nodeid, reason))

    if skipped_items:
        tw.write_line("")
        tw.write_line(" Skipped tests:", yellow=True)
        for nid, reason in skipped_items:
            suffix = f"  \u2192  {reason}" if reason else ""
            tw.write_line(f"   \u2022 {nid}{suffix}")


def pytest_terminal_summary(terminalreporter) -> None:
    """Print a tabled test-suite summary grouped by directory at session end.

    Always runs — no CLI flag required.  Groups test results by the parent
    directory of each test file and prints PASS / FAIL / SKIP / ERROR counts
    and total wall-clock duration per directory.  A TOTAL row summarises the
    full session.  When failures or errors are present a secondary list of the
    failing test node IDs is appended below the table.

    Example output::

        ══════════════════════════════════════════════════════════════════
         ROCm Test Suite Summary
        ══════════════════════════════════════════════════════════════════
         Test Directory              PASS  FAIL  SKIP  ERROR    Duration
        ──────────────────────────────────────────────────────────────────
         tests/dry_run                 12     0     1      0       0.5 s
         tests/e2e/compiler             3     0     0      0      45.2 s
         tests/e2e/stack_validation     2     1     0      0       8.7 s
        ──────────────────────────────────────────────────────────────────
         TOTAL  18 tests │ 17 passed │ 1 failed │ 1 skipped │ 0 error │  54.4 s
        ══════════════════════════════════════════════════════════════════

         Failed tests:
           • tests/e2e/stack_validation/test_hip_runtime.py::test_hip_device_count
        ══════════════════════════════════════════════════════════════════

    Args:
        terminalreporter: pytest's TerminalReporter plugin instance.
    """
    from collections import defaultdict
    from pathlib import Path

    stats = terminalreporter.stats

    # outcome stat key → table column label
    outcome_cols = [
        ("passed", "PASS"),
        ("failed", "FAIL"),
        ("skipped", "SKIP"),
        ("error", "ERROR"),
    ]

    # Per-directory accumulators
    dirs: dict[str, dict] = defaultdict(lambda: {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0, "duration": 0.0})
    failed_nodeids: list[str] = []

    for stat_key, col in outcome_cols:
        for report in stats.get(stat_key, []):
            test_path = report.nodeid.split("::")[0]
            d = str(Path(test_path).parent)
            dirs[d][col] += 1
            dirs[d]["duration"] += getattr(report, "duration", 0.0)
            if col in ("FAIL", "ERROR"):
                failed_nodeids.append(report.nodeid)

    if not dirs:
        return

    # Dynamic column width: longest directory path + padding
    dir_w = max(max(len(d) for d in dirs) + 2, len("Test Directory"))
    header = f" {'Test Directory':<{dir_w}} {'PASS':>5} {'FAIL':>5} {'SKIP':>5} {'ERROR':>6}  {'Duration':>10}"
    thin = "─" * len(header)

    tw = terminalreporter
    tw.write_sep("=", "ROCm Test Suite Summary", bold=True)
    tw.write_line(header)
    tw.write_line(thin)

    totals: dict[str, float | int] = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0, "duration": 0.0}
    for d in sorted(dirs):
        row = dirs[d]
        for k in totals:
            totals[k] = totals[k] + row[k]
        tw.write_line(
            f" {d:<{dir_w}} {row['PASS']:>5} {row['FAIL']:>5}"
            f" {row['SKIP']:>5} {row['ERROR']:>6}  {row['duration']:>8.1f} s"
        )

    tw.write_line(thin)
    total_tests = int(totals["PASS"]) + int(totals["FAIL"]) + int(totals["SKIP"]) + int(totals["ERROR"])
    tw.write_line(
        f" TOTAL  {total_tests} tests │ {totals['PASS']} passed │ "
        f"{totals['FAIL']} failed │ {totals['SKIP']} skipped │ "
        f"{totals['ERROR']} error │ {totals['duration']:>7.1f} s"
    )

    if failed_nodeids:
        tw.write_line("")
        tw.write_line(" Failed tests:", red=True)
        for nid in failed_nodeids:
            tw.write_line(f"   \u2022 {nid}")

    _render_skipped_items(tw, stats)

    tw.write_sep("=", "")


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Post-session hook: write run label and/or generate Allure dashboard.

    Runs unconditionally but exits early when neither option is active.
    ``--allure-log-name`` and ``--allure-db`` are independent — either or both
    may be supplied in the same pytest invocation.

    Args:
        session: The pytest session object.
    """
    log_name = session.config.getoption("--allure-log-name", default=None)
    n = session.config.getoption("--allure-db", default=0)

    if not log_name and (not n or n <= 0):
        return

    try:
        from framework.config.loader import load_config

        config_path = session.config.getoption("--rocm-config", default=None)
        cfg = load_config(config_path=config_path)

        if log_name:
            _write_executor_info(cfg.reporting.allure_results_dir, log_name)

        if n and n > 0:
            _build_allure_dashboard(cfg.reporting.allure_results_dir, n_previous=n)

    except Exception as exc:
        logger.warning("Allure post-session hook failed: %s", exc)
