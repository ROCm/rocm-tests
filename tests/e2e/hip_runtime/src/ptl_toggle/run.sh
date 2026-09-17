#!/usr/bin/env bash
# ptl_runtime_toggle_rccl_workload
#
# System-level stress test for the PTL runtime toggle path. Drives the
# per-GPU ptl_enable sysfs node at high frequency while RCCL collectives
# are actively running across all GPUs in the node.
#
# Pre-conditions:
#   - amd-smi available on PATH (PTL telemetry oracle)
#   - mpirun available on PATH
#   - all_reduce_perf available under $ROCM_PATH/bin/ (or via $RCCL_BIN override)
#   - amdgpu loaded with ptl=1 (PTL enabled at load time, runtime changes
#     allowed); ptl=2 (permanent disable) will cause a clean SKIP
#   - rocprof NOT running (rocprof silently suppresses PTL)
#   - Root or write access to /sys/class/drm/card*/device/ptl/ptl_enable
#
# Pass-criteria summary (see verdict block for the full oracle):
#   PASS iff every per-GPU ptl_enable write was acknowledged by sysfs
#   readback, every RCCL iteration exited 0, the dmesg delta shows no
#   critical kernel/driver events during the run window, all GPUs remain
#   enumerable, and the original PTL state was restored.

set -u

# ============================================================
# Config (env-overridable)
# ============================================================
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
RCCL_BIN=${RCCL_BIN:-$ROCM_PATH/bin/all_reduce_perf}
MPIRUN=${MPIRUN:-mpirun}
AMD_SMI=${AMD_SMI:-amd-smi}

# How many full RCCL all_reduce sweeps to drive. The RCCL workload runs
# this many back-to-back `mpirun all_reduce_perf` invocations and then
# exits. Driving the test by sweep count keeps workload coverage
# deterministic across servers; wall time scales with interconnect
# speed but every box still applies the same amount of stress to PTL.
RCCL_ITER_COUNT=${RCCL_ITER_COUNT:-5}

# Cadence of the toggle: every interval, every GPU's ptl_enable is
# flipped (1 -> 0 -> 1 -> ...). 500 ms across N GPUs concurrently.
TOGGLE_INTERVAL_SECS=${TOGGLE_INTERVAL_SECS:-0.5}

# How often to dump an amd-smi static -l consistency snapshot during
# the toggle stress. Every Nth toggler tick.
AMD_SMI_STATIC_EVERY_N_TICKS=${AMD_SMI_STATIC_EVERY_N_TICKS:-20}

# Optional hard watchdog. If > 0, kill the entire test if it hasn't
# returned in this many seconds. Default 0 = disabled (let the RCCL
# iter loop self-terminate). Set explicitly when running under CI or
# any context where you need a guaranteed upper bound on wall time.
WATCHDOG_SECS=${WATCHDOG_SECS:-0}

# Where to drop all per-run artifacts (logs, dmesg snapshot, transition
# trace, RCCL stdout/stderr).
RUN_DIR=${RUN_DIR:-run_$(date +%Y%m%d-%H%M%S)}

# Where the per-GPU PTL toggle is exposed. Globs to multiple cards.
SYSFS_PTL_GLOB=${SYSFS_PTL_GLOB:-/sys/class/drm/card*/device/ptl/ptl_enable}

# RCCL workload knobs. Defaults aim for a few minutes of sustained
# all_reduce traffic on a multi-GPU node.
RCCL_MIN_BYTES=${RCCL_MIN_BYTES:-8M}
RCCL_MAX_BYTES=${RCCL_MAX_BYTES:-1G}
RCCL_FACTOR=${RCCL_FACTOR:-2}
RCCL_ITERS=${RCCL_ITERS:-20}
RCCL_WARMUP=${RCCL_WARMUP:-5}

# Extra mpirun flags. Override on platforms with specific MPI transport
# needs (e.g. UCX-only fabrics: MPIRUN_EXTRA="--mca pml ucx").
MPIRUN_EXTRA=${MPIRUN_EXTRA:-"--allow-run-as-root --bind-to numa"}

# ============================================================
# Logging helpers
# ============================================================
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/run.log"
TOGGLE_TRACE="$RUN_DIR/toggle_trace.csv"
RCCL_LOG="$RUN_DIR/rccl.log"
DMESG_BEFORE="$RUN_DIR/dmesg_before.txt"
DMESG_AFTER="$RUN_DIR/dmesg_after.txt"
DMESG_DELTA="$RUN_DIR/dmesg_delta.txt"
ORIG_PTL_STATE="$RUN_DIR/ptl_state_before.txt"
FINAL_PTL_STATE="$RUN_DIR/ptl_state_after.txt"

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
warn() { echo "[$(date +%H:%M:%S)] WARN: $*" | tee -a "$LOG" >&2; }
die()  { echo "[$(date +%H:%M:%S)] FAIL: $*" | tee -a "$LOG" >&2; exit 1; }
skip() { echo "[$(date +%H:%M:%S)] SKIP: $*" | tee -a "$LOG" >&2; exit 77; }

# ============================================================
# Pre-flight checks
# ============================================================
preflight() {
    log "preflight: checking environment"

    # Infrastructure absence -> SKIP, not FAIL. We can only fail the
    # system-under-test if the test actually ran. Missing tooling means
    # the test was not applicable to this environment.
    command -v "$MPIRUN"  >/dev/null 2>&1 || skip "$MPIRUN not on PATH"
    command -v "$AMD_SMI" >/dev/null 2>&1 || skip "$AMD_SMI not on PATH"
    command -v timeout    >/dev/null 2>&1 || skip "timeout not on PATH (coreutils) -- required to bound amd-smi calls"
    if ! [[ -x "$RCCL_BIN" ]]; then
        if command -v "$RCCL_BIN" >/dev/null 2>&1; then
            RCCL_BIN="$(command -v "$RCCL_BIN")"
        else
            skip "$RCCL_BIN not executable and not on PATH -- set RCCL_BIN to the right path"
        fi
    fi

    shopt -s nullglob
    PTL_NODES=( $SYSFS_PTL_GLOB )
    shopt -u nullglob

    if [[ ${#PTL_NODES[@]} -eq 0 ]]; then
        skip "no per-GPU PTL sysfs nodes match $SYSFS_PTL_GLOB -- this node does not expose runtime PTL toggle"
    fi

    # Read access is enough to snapshot; we need write to toggle.
    # Missing permissions are an environment issue, not a SUT failure.
    for node in "${PTL_NODES[@]}"; do
        [[ -r "$node" ]] || skip "cannot read $node (need root or chmod 666)"
        [[ -w "$node" ]] || skip "cannot write $node (need root or chmod 666)"
    done

    # Rocprof presence is a hard skip: rocprof silently suppresses PTL.
    if pgrep -x rocprof  >/dev/null 2>&1 ||
       pgrep -x rocprofv2 >/dev/null 2>&1 ||
       pgrep -x rocprof-sdk >/dev/null 2>&1; then
        skip "rocprof process running -- PTL would be suppressed; abort to keep the test honest"
    fi

    log "preflight: ok (${#PTL_NODES[@]} per-GPU PTL nodes, mpirun=$MPIRUN, rccl=$RCCL_BIN)"
}

# ============================================================
# GPU / PTL discovery
# ============================================================
detect_n_gpus() {
    # amd-smi list emits one stanza per physical accelerator with a
    # `GPU: <id>` header. That count matches the per-physical-GPU PTL
    # sysfs node count and is the rank count we feed mpirun.
    #
    # Wrap in `timeout` so a wedged GPU that hangs amd-smi can't hang
    # the whole script before the watchdog is even armed.
    local n
    n=$(timeout 30s "$AMD_SMI" list 2>/dev/null | grep -cE '^GPU: [0-9]+$' || true)
    if [[ -z "$n" || "$n" -lt 1 ]]; then
        n=${#PTL_NODES[@]}
    fi
    [[ "$n" -ge 1 ]] || die "could not detect any GPUs via amd-smi or PTL sysfs"
    echo "$n"
}

snapshot_ptl_state() {
    local outfile="$1"
    : >"$outfile"
    for node in "${PTL_NODES[@]}"; do
        local card val
        card=$(echo "$node" | sed -E 's|.*/(card[0-9]+)/.*|\1|')
        val=$(cat "$node" 2>/dev/null || echo "?")
        printf "%s %s %s\n" "$card" "$node" "$val" >>"$outfile"
    done
    log "snapshot_ptl_state -> $outfile"
}

# ptl_enable uses an asymmetric sysfs encoding: it accepts numeric
# 0/1 on write but reports 'disabled'/'enabled' on read. This is the
# canonical Linux pattern for enable-style sysfs attributes (cf.
# /sys/class/leds/*/trigger). Collapse both forms to a canonical
# 0/1 for comparison; anything unrecognized passes through so we
# can still detect a real wedged/garbage readback.
ptl_norm() {
    case "${1:-}" in
        1|enabled)  echo 1 ;;
        0|disabled) echo 0 ;;
        *)          echo "$1" ;;
    esac
}

ensure_ptl_enabled() {
    log "ensure_ptl_enabled: forcing every GPU's ptl_enable to 1"

    # If PTL is permanently disabled at module load (ptl=2), runtime writes are
    # expected to fail; treat that as SKIP (not a SUT failure).
    if [[ -r /sys/module/amdgpu/parameters/ptl ]]; then
        local ptl_mode
        ptl_mode=$(cat /sys/module/amdgpu/parameters/ptl 2>/dev/null || echo "")
        if [[ "$ptl_mode" == "2" ]]; then
            skip "amdgpu loaded with ptl=2 (permanent disable) -- runtime PTL toggle unavailable"
        fi
    fi

    for node in "${PTL_NODES[@]}"; do
        if ! echo 1 >"$node" 2>/dev/null; then
            die "failed to write 1 to $node -- amdgpu refused the change"
        fi
        local r r_norm
        r=$(cat "$node")
        r_norm=$(ptl_norm "$r")
        if [[ "$r_norm" != "1" ]]; then
            die "wrote 1 to $node but readback is '$r' (normalized '$r_norm') -- runtime PTL changes appear blocked"
        fi
    done
    log "ensure_ptl_enabled: ok (all ${#PTL_NODES[@]} GPUs enabled)"
}

# Restore PTL to the exact per-GPU state captured at start.
#
# The snapshot stores the kernel's read-form token ('enabled'/'disabled').
# We don't rely on the kernel accepting that token on write -- we
# normalize to numeric 0/1 (which we know works) before writing, and
# compare the readback in normalized form too.
restore_ptl_state() {
    log "restore_ptl_state: replaying $ORIG_PTL_STATE"
    local restored_ok=1
    while read -r card node val; do
        [[ -z "${val:-}" ]] && continue
        if [[ "$val" == "?" ]]; then continue; fi
        local want
        want=$(ptl_norm "$val")
        if ! echo "$want" >"$node" 2>/dev/null; then
            warn "restore: write '$want' (from '$val') -> $node failed"
            restored_ok=0
            continue
        fi
        local rb rb_norm
        rb=$(cat "$node" 2>/dev/null || echo "?")
        rb_norm=$(ptl_norm "$rb")
        if [[ "$rb_norm" != "$want" ]]; then
            warn "restore: $card readback '$rb' (normalized '$rb_norm') != expected '$want' (from '$val')"
            restored_ok=0
        fi
    done <"$ORIG_PTL_STATE"
    snapshot_ptl_state "$FINAL_PTL_STATE"
    return $((restored_ok == 1 ? 0 : 1))
}

# ============================================================
# Background workers
# ============================================================
RCCL_LOOP_PID=""
TOGGLER_PID=""
STOP_FLAG="$RUN_DIR/stop_workers"
WATCHDOG_FIRED="$RUN_DIR/watchdog_fired"

# Recursively emit the PIDs of every descendant of $1 (post-order, so
# leaves come first). Used by stop_rccl_workload to scope teardown to
# our own process tree -- avoids `pkill -f "$RCCL_BIN"` which would
# also hit any unrelated all_reduce_perf instance on the box.
descendants_of() {
    local pid=$1
    local kids
    kids=$(pgrep -P "$pid" 2>/dev/null || true)
    local k
    for k in $kids; do
        descendants_of "$k"
        echo "$k"
    done
}

start_rccl_workload() {
    local n_ranks="$1"
    log "start_rccl_workload: launching $n_ranks-rank $RCCL_BIN, $RCCL_ITER_COUNT iter(s)"
    (
        set +e
        local iter=0
        # Run RCCL_ITER_COUNT full all_reduce sweeps back to back. The
        # loop exits naturally when the configured iter count is reached;
        # STOP_FLAG is touched by the abort trap to support clean
        # tear-down on Ctrl-C or watchdog. A failed iter (rc != 0) is
        # logged and counted toward the iter total; subsequent iters
        # continue so toggle coverage is preserved, and the verdict
        # block reports any failures via rccl_runs_failed.
        while [[ ! -f "$STOP_FLAG" && $iter -lt $RCCL_ITER_COUNT ]]; do
            iter=$((iter + 1))
            echo "rccl iter $iter/$RCCL_ITER_COUNT: launching" >>"$RCCL_LOG"
            "$MPIRUN" -np "$n_ranks" $MPIRUN_EXTRA \
                "$RCCL_BIN" \
                    -b "$RCCL_MIN_BYTES" -e "$RCCL_MAX_BYTES" \
                    -f "$RCCL_FACTOR" -g 1 \
                    -n "$RCCL_ITERS" -w "$RCCL_WARMUP" \
                >>"$RCCL_LOG" 2>&1
            local rc=$?
            echo "rccl iter $iter: rc=$rc" >>"$RCCL_LOG"
            # Brief gap so the toggler gets at least one tick between
            # iters; harmless if it doesn't.
            [[ $iter -lt $RCCL_ITER_COUNT ]] && sleep 1
        done
        echo "rccl loop: exit after $iter iter(s)" >>"$RCCL_LOG"
    ) &
    RCCL_LOOP_PID=$!
    log "start_rccl_workload: outer loop pid=$RCCL_LOOP_PID"
}

stop_rccl_workload() {
    log "stop_rccl_workload"
    : >"$STOP_FLAG"
    if [[ -z "$RCCL_LOOP_PID" ]]; then
        return 0
    fi
    # Walk the descendant tree under our outer-loop subshell and signal
    # only those processes -- the subshell -> mpirun -> orted -> ranks.
    # Scoped to OUR process tree only; an unrelated all_reduce_perf
    # running on the same host is untouched.
    local kids
    kids=$(descendants_of "$RCCL_LOOP_PID")
    if [[ -n "$kids" ]]; then
        kill -TERM $kids 2>/dev/null || true
    fi
    wait "$RCCL_LOOP_PID" 2>/dev/null || true
    sleep 1
    # Re-collect and SIGKILL anything still hanging on. Anything that
    # got re-parented to init (e.g. if mpirun died before its ranks)
    # would slip past this -- mpirun's PR_SET_PDEATHSIG on orted makes
    # that very unlikely in practice for OpenMPI.
    kids=$(descendants_of "$RCCL_LOOP_PID")
    if [[ -n "$kids" ]]; then
        kill -KILL $kids 2>/dev/null || true
    fi
}

start_toggler() {
    log "start_toggler: interval=${TOGGLE_INTERVAL_SECS}s, amd-smi static snapshot every ${AMD_SMI_STATIC_EVERY_N_TICKS} ticks"
    echo "tick_ts,tick_idx,card,node,desired,readback,ok" >"$TOGGLE_TRACE"
    (
        set +e
        local tick=0
        local desired=0

        # Brief warm-up so the first toggle lands after mpirun has
        # launched the first RCCL iter.
        sleep 2

        # Toggle as long as the RCCL outer-loop subshell is alive. Its
        # lifetime spans "first iter launch" to "last iter completion",
        # which is the window we want to apply PTL stress under. The
        # toggler exits as soon as the RCCL loop process is gone.
        while kill -0 "$RCCL_LOOP_PID" 2>/dev/null; do
            tick=$((tick + 1))
            desired=$((1 - desired))
            local ts
            ts=$(date +%s.%N)
            for node in "${PTL_NODES[@]}"; do
                local card
                card=$(echo "$node" | sed -E 's|.*/(card[0-9]+)/.*|\1|')
                local ok=1
                if ! echo "$desired" >"$node" 2>/dev/null; then ok=0; fi
                local rb rb_norm
                rb=$(cat "$node" 2>/dev/null || echo "?")
                rb_norm=$(ptl_norm "$rb")
                if [[ "$rb_norm" != "$desired" ]]; then ok=0; fi
                printf "%s,%d,%s,%s,%d,%s,%d\n" \
                    "$ts" "$tick" "$card" "$node" "$desired" "$rb" "$ok" \
                    >>"$TOGGLE_TRACE"
            done
            # Periodic amd-smi static -l snapshot for sysfs-vs-amd-smi
            # consistency cross-check. Bounded by `timeout` so a hung
            # amd-smi can't freeze the toggler.
            if (( tick % AMD_SMI_STATIC_EVERY_N_TICKS == 0 )); then
                timeout 30s "$AMD_SMI" static -l \
                    >>"$RUN_DIR/amd_smi_static_tick_${tick}.txt" 2>&1
            fi
            sleep "$TOGGLE_INTERVAL_SECS"
        done

        echo "$tick" >"$RUN_DIR/toggler_total_ticks"
        echo "toggler exit: total=$tick" >>"$LOG"
    ) &
    TOGGLER_PID=$!
    log "start_toggler: pid=$TOGGLER_PID"
}

stop_toggler() {
    log "stop_toggler"
    if [[ -n "$TOGGLER_PID" ]]; then
        kill -TERM "$TOGGLER_PID" 2>/dev/null || true
        wait "$TOGGLER_PID" 2>/dev/null || true
    fi
}

# ============================================================
# Watchdog
# ============================================================
arm_watchdog() {
    if [[ ${WATCHDOG_SECS:-0} -le 0 ]]; then
        log "watchdog: disabled (WATCHDOG_SECS=0)"
        return 0
    fi
    log "watchdog: ${WATCHDOG_SECS}s hard ceiling"
    (
        sleep "$WATCHDOG_SECS"
        warn "WATCHDOG fired -- terminating run"
        : >"$STOP_FLAG"
        : >"$WATCHDOG_FIRED"
        # Signal the parent script so its TERM trap runs stop_rccl_workload()
        # (descendants_of mpirun tree) and restores PTL. Do not SIGTERM the
        # worker subshells directly here -- that can orphan mpirun ranks
        # before the trap walks the tree. Escalate to SIGKILL if the parent
        # does not exit promptly (SIGKILL is uncatchable; TERM path above
        # is what performs cleanup).
        kill -TERM $$ 2>/dev/null || true
        sleep 5
        kill -KILL $$ 2>/dev/null || true
    ) &
    WATCHDOG_PID=$!
}

disarm_watchdog() {
    [[ -n "${WATCHDOG_PID:-}" ]] && kill "$WATCHDOG_PID" 2>/dev/null || true
}

# ============================================================
# Main
# ============================================================
main() {
    log "ptl_runtime_toggle_rccl_workload: starting (run_dir=$RUN_DIR)"

    preflight

    local n_gpus
    n_gpus=$(detect_n_gpus)
    log "detect_n_gpus: $n_gpus"

    log "snapshotting dmesg baseline"
    dmesg --ctime >"$DMESG_BEFORE" 2>/dev/null || \
        sudo dmesg --ctime >"$DMESG_BEFORE" 2>/dev/null || \
        { warn "dmesg unreadable -- dmesg oracle will be skipped"; : >"$DMESG_BEFORE"; }

    snapshot_ptl_state "$ORIG_PTL_STATE"
    ensure_ptl_enabled

    arm_watchdog

    rm -f "$STOP_FLAG"
    start_rccl_workload "$n_gpus"
    start_toggler

    # Wait for the RCCL outer loop to finish its $RCCL_ITER_COUNT iters.
    # The workload self-terminates after completing the configured iter
    # count. Wall time is the sum of the N mpirun iter durations
    # (server-dependent), while workload coverage stays constant at N
    # full RCCL sweeps.
    log "main: waiting for RCCL outer loop to complete $RCCL_ITER_COUNT iter(s) ..."
    wait "$RCCL_LOOP_PID" 2>/dev/null || true

    log "main: waiting for toggler to finish ..."
    wait "$TOGGLER_PID" 2>/dev/null || true

    if [[ -f "$WATCHDOG_FIRED" ]]; then
        disarm_watchdog
        die "watchdog fired -- run exceeded ${WATCHDOG_SECS}s ceiling"
    fi

    disarm_watchdog

    log "snapshotting dmesg after"
    dmesg --ctime >"$DMESG_AFTER" 2>/dev/null || \
        sudo dmesg --ctime >"$DMESG_AFTER" 2>/dev/null || \
        : >"$DMESG_AFTER"
    diff "$DMESG_BEFORE" "$DMESG_AFTER" | grep -E '^>' | sed 's/^> //' \
        >"$DMESG_DELTA" 2>/dev/null || true

    local restored_ok=0
    restore_ptl_state || restored_ok=1

    # ============================================================
    # Verdict
    # ============================================================
    log "computing verdict"

    local total_writes ok_writes bad_writes
    if [[ -s "$TOGGLE_TRACE" ]]; then
        total_writes=$(tail -n +2 "$TOGGLE_TRACE" | wc -l)
        ok_writes=$(tail -n +2 "$TOGGLE_TRACE" | awk -F, '$7==1' | wc -l)
        bad_writes=$(tail -n +2 "$TOGGLE_TRACE" | awk -F, '$7==0' | wc -l)
    else
        total_writes=0; ok_writes=0; bad_writes=0
    fi

    local total_ticks
    total_ticks=$(cat "$RUN_DIR/toggler_total_ticks" 2>/dev/null || echo 0)

    # Use `grep ... | wc -l` everywhere instead of `grep -c`. `grep -c`
    # prints the count on stdout AND exits non-zero when count==0, so the
    # idiom `$(grep -c ... || echo 0)` double-prints ("0\n0") on a clean
    # run and breaks [[ -ne ]] comparisons below. `wc -l` always emits a
    # single integer on stdout regardless of upstream match count.
    #
    # rccl_runs counts only the per-iter "launching" line; the loop also
    # emits an "iter N: rc=X" completion line per iter, so a naive
    # `^rccl iter` regex would double-count (10 lines for 5 iters).
    local rccl_runs rccl_fail
    rccl_runs=$(grep -E '^rccl iter [0-9]+/[0-9]+: launching' "$RCCL_LOG" 2>/dev/null | wc -l)
    rccl_fail=$(grep -E 'iter [0-9]+: rc=[^0]' "$RCCL_LOG" 2>/dev/null | grep -v 'rc=0' | wc -l)

    # Critical kernel/driver events in the before/after dmesg delta: GPU
    # reset/hang, panic/oops/lockup, ring timeout, GPU fault, RAS/IOMMU
    # fault, and PTL-related driver messages.
    local dmesg_critical_re
    dmesg_critical_re='Kernel panic|panic|watchdog:.*(BUG|soft lockup|hard lockup)|soft lockup|hard lockup|BUG:|kernel BUG|Oops:[[:space:]]*[0-9a-fA-F]+|Call Trace|call trace|blocked for more than|hung task|gpu reset|amdgpu.*reset|ring .*timeout|GPU fault|RAS.*error|IOMMU.*fault|DMAR.*fault|segfault|segmentation fault|(amdgpu|drm|kfd).*\b[Pp][Tt][Ll]\b'

    local dmesg_hits
    dmesg_hits=$(grep -Ei "$dmesg_critical_re" "$DMESG_DELTA" 2>/dev/null | wc -l)

    # Bound amd-smi here too: by this point the watchdog is disarmed,
    # so a hung amd-smi (e.g. wedged GPU) would otherwise freeze the
    # verdict block forever. A timeout-induced 0 trips the
    # gpu_count_changed oracle -- which is the right verdict for
    # "amd-smi could not enumerate the GPUs at end of run".
    local gpus_after
    gpus_after=$(timeout 30s "$AMD_SMI" list 2>/dev/null | grep -E '^GPU: [0-9]+$' | wc -l)

    local verdict="PASS"
    local fail_reasons=()
    [[ $bad_writes -ne 0       ]] && { verdict="FAIL"; fail_reasons+=("ptl_writes_bad=$bad_writes"); }
    [[ $rccl_fail  -ne 0       ]] && { verdict="FAIL"; fail_reasons+=("rccl_runs_failed=$rccl_fail"); }
    [[ $dmesg_hits -ne 0       ]] && { verdict="FAIL"; fail_reasons+=("dmesg_critical=$dmesg_hits"); }
    [[ $gpus_after -ne $n_gpus ]] && { verdict="FAIL"; fail_reasons+=("gpu_count_changed_from_${n_gpus}_to_${gpus_after}"); }
    [[ $restored_ok -ne 0      ]] && { verdict="FAIL"; fail_reasons+=("ptl_state_not_restored"); }

    log "==================== VERDICT ===================="
    log "result                    : $verdict"
    [[ ${#fail_reasons[@]} -gt 0 ]] && log "fail_reasons              : ${fail_reasons[*]}"
    log "n_gpus                    : $n_gpus (after=$gpus_after)"
    log "ptl_writes_total          : $total_writes"
    log "ptl_writes_ok             : $ok_writes"
    log "ptl_writes_bad            : $bad_writes"
    log "toggler_ticks_total       : $total_ticks"
    log "rccl_runs_attempted       : $rccl_runs (of $RCCL_ITER_COUNT requested)"
    log "rccl_runs_failed          : $rccl_fail"
    log "dmesg_critical_delta      : $dmesg_hits"
    log "ptl_state_restored        : $([[ $restored_ok -eq 0 ]] && echo yes || echo no)"
    log "run_dir                   : $RUN_DIR"
    log "================================================="

    # Signal the EXIT trap that main completed normally so it doesn't
    # double-run cleanup or print "trap caught" on a healthy PASS/FAIL
    # run. Assignment doesn't change $?, so the verdict comparison
    # below still determines the script's exit code.
    MAIN_EXITED_CLEAN=1
    [[ "$verdict" == "PASS" ]]
}

# Cleanup-on-abort safety net. Fires on EXIT (any reason), INT, TERM.
# On a clean run main() does all of stop_rccl_workload, stop_toggler,
# and restore_ptl_state itself and sets MAIN_EXITED_CLEAN=1 just before
# returning, so the trap body is a no-op and we don't print an alarming
# "trap caught" warning on a PASSing run. On any abnormal exit path
# (signal, die, early failure) the flag stays 0 and the trap takes
# responsibility for tearing the workers down and restoring PTL state.
MAIN_EXITED_CLEAN=0
trap '{
    if [[ ${MAIN_EXITED_CLEAN:-0} -eq 0 ]]; then
        warn "trap caught -- attempting to restore PTL state and stop workers"
        : >"$STOP_FLAG" 2>/dev/null
        stop_rccl_workload 2>/dev/null
        stop_toggler 2>/dev/null
        [[ -s "$ORIG_PTL_STATE" ]] && restore_ptl_state >/dev/null 2>&1
    fi
    [[ -f "${WATCHDOG_FIRED:-}" ]] && exit 124
}' EXIT INT TERM

main
