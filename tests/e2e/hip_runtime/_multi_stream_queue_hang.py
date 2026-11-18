# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Helpers for the multi_stream_queue_hang regression soak.

Hang classification runs on the execution node as a single bash script so the
poll loop (sleep) never lives in ``test_*.py``. On current ROCm builds the hang
is not expected: ``VERDICT=NOHANG`` is success; ``HANG`` / ``WEDGE`` fail.
Optional KFD mqds / ``rdc.py`` paths from the upstream scenario are omitted.
"""

from __future__ import annotations

import os
import shlex

# Defaults match upstream ``manifest.env`` / ``detect_hang.sh``.
DEFAULT_MAX_SECONDS = 1800
# Elevated producer count: more streams/threads on one AQL ring under GPU_MAX_HW_QUEUES=1.
DEFAULT_STREAMS = 64
DEFAULT_THREADS = 64
_FROZEN_SECS = 30
_READY_ROUNDS_OR_SECS = 120
_RUNNABLE_RATIO_PCT = 80
_DSTATE_RATIO_PCT = 50
_CPU_WALL_MIN = 20
_POLL_SECS = 3


def preflight_cmd(binary: str, test_name: str) -> str:
    """Host gates: no GPU/KFD D-state processes; binary ``--list`` identity."""
    quoted_bin = shlex.quote(binary)
    quoted_name = shlex.quote(test_name)
    return f"""
set -eu
PAT='kfd|rocminfo|rocm_agent|multi_stream_queue_hang|hipInfo|amdgpu'
DK=$(ps -eo stat,comm,args 2>/dev/null | awk '$1 ~ /D/' | grep -cE "$PAT" || true)
echo "preflight D-state GPU/KFD-related=$DK"
if [ "$DK" -gt 0 ]; then
  echo "FAIL: D-state on GPU/KFD paths"
  exit 1
fi
test -x {quoted_bin}
name=$({quoted_bin} --list 2>/dev/null | tr -d '\\r')
echo "preflight --list=$name"
[ "$name" = {quoted_name} ] || {{ echo "FAIL: expected {quoted_name}, got $name"; exit 1; }}
echo "PREFLIGHT_OK"
"""


def round_loop_cmd(library_path: str, binary: str, runlog: str) -> str:
    """Continuous rounds: one HW queue plus elevated stream/thread producers."""
    ld = shlex.quote(library_path)
    bin_q = shlex.quote(binary)
    log_q = shlex.quote(runlog)
    threads = int(os.environ.get("ROCM_TEST_MSQH_THREADS", str(DEFAULT_THREADS)))
    streams = int(os.environ.get("ROCM_TEST_MSQH_STREAMS", str(DEFAULT_STREAMS)))
    threads = max(1, min(threads, 127))
    streams = max(1, min(streams, 127))
    return (
        f"env LD_LIBRARY_PATH={ld} GPU_MAX_HW_QUEUES=1 HSA_USE_SVM=0 "
        f"MSQH_THREADS={threads} MSQH_STREAMS={streams} "
        f"bash -c {shlex.quote(_round_loop_body(bin_q, log_q))}"
    )


def _round_loop_body(quoted_binary: str, quoted_runlog: str) -> str:
    # quoted_binary / quoted_runlog are already shell-safe literals.
    return f"""
set -u
i=0
while true; do
  i=$((i+1))
  echo "===== Round $i $(date +%F\\ %T.%3N) =====" >> {quoted_runlog}
  {quoted_binary} >> {quoted_runlog} 2>&1
  rc=$?
  echo "Round $i rc=$rc $(date +%F\\ %T.%3N)" >> {quoted_runlog}
  [ "$rc" -ne 0 ] && break
done
"""


def detect_hang_cmd(binary: str, runlog: str, max_seconds: int) -> str:
    """Classify AQL hang vs KFD DQM wedge vs no-hang within *max_seconds*."""
    exe = shlex.quote(binary)
    log = shlex.quote(runlog)
    return f"""
set -u
EXE={exe}
RUNLOG={log}
MAXS={int(max_seconds)}
FROZEN_NEED={_FROZEN_SECS}
READY_SECS={_READY_ROUNDS_OR_SECS}
RUN_PCT={_RUNNABLE_RATIO_PCT}
D_PCT={_DSTATE_RATIO_PCT}
CW_MIN={_CPU_WALL_MIN}
POLL={_POLL_SECS}

rounds() {{ grep -c '^Round.*rc=' "$RUNLOG" 2>/dev/null || echo 0; }}

getpid() {{
  want=$(readlink -f "$EXE" 2>/dev/null || echo "$EXE")
  for p in $(pgrep -f "$(basename "$EXE")" 2>/dev/null); do
    got=$(readlink -f /proc/$p/exe 2>/dev/null || true)
    [ "$got" = "$want" ] || continue
    echo "$p"
    return 0
  done
  return 1
}}

echo "### detect_hang max=${{MAXS}}s T0=$(date +%FT%T) ###"
T0=$(date +%s)
P=""
FROZEN=0
LASTR=""
LASTC=0
LASTT=0

while [ $(( $(date +%s) - T0 )) -lt "$MAXS" ]; do
  NOW=$(date +%s)
  E=$((NOW - T0))
  {{ [ -z "$P" ] || [ ! -d "/proc/$P" ]; }} && P=$(getpid || true)
  if [ -z "$P" ]; then
    sleep "$POLL"
    continue
  fi

  R=$(rounds)
  R=${{R:-0}}

  TOT=0
  RUN=0
  DST=0
  for t in /proc/$P/task/*/stat; do
    s=$(awk '{{print $3}}' "$t" 2>/dev/null)
    [ -z "$s" ] && continue
    TOT=$((TOT + 1))
    [ "$s" = "R" ] && RUN=$((RUN + 1))
    [ "$s" = "D" ] && DST=$((DST + 1))
  done
  RATIO=0
  [ "$TOT" -gt 0 ] && RATIO=$(( RUN * 100 / TOT ))
  DRATIO=0
  [ "$TOT" -gt 0 ] && DRATIO=$(( DST * 100 / TOT ))

  C2=$(awk '{{print $14+$15}}' /proc/$P/stat 2>/dev/null)
  C2=${{C2:-0}}
  CW=0
  if [ "$LASTT" -gt 0 ] && [ $((NOW - LASTT)) -gt 0 ]; then
    HZ=$(getconf CLK_TCK)
    CW=$(( (C2 - LASTC) / (HZ * (NOW - LASTT)) ))
  fi
  LASTC=$C2
  LASTT=$NOW

  if [ "$R" = "$LASTR" ]; then
    FROZEN=$((FROZEN + POLL))
  else
    FROZEN=0
  fi
  LASTR=$R

  if [ $((E % 30)) -lt "$POLL" ]; then
    printf '[%s] [%5ds] pid=%s R=%s/%s(%s%%) D=%s round=%s frozen=%ss cpu/wall=%sx\\n' \\
      "$(date +%T)" "$E" "$P" "$RUN" "$TOT" "$RATIO" "$DST" "$R" "$FROZEN" "$CW"
  fi

  READY=0
  {{ [ "$R" -ge 1 ] 2>/dev/null || [ "$E" -ge "$READY_SECS" ]; }} && READY=1

  if [ "$READY" -eq 1 ] && [ "$FROZEN" -ge "$FROZEN_NEED" ] && [ "$DRATIO" -ge "$D_PCT" ]; then
    echo "KFD_DQM_WEDGE $(date +%FT%T)"
    echo "VERDICT=WEDGE"
    exit 1
  fi

  if [ "$READY" -eq 1 ] && [ "$FROZEN" -ge "$FROZEN_NEED" ] \\
      && [ "$CW" -gt "$CW_MIN" ] 2>/dev/null && [ "$RATIO" -ge "$RUN_PCT" ]; then
    echo "AQL_HANG elapsed=${{E}}s round=$R frozen=${{FROZEN}}s cpu/wall=${{CW}}x R=$RUN/$TOT"
    echo "VERDICT=HANG"
    exit 1
  fi
  sleep "$POLL"
done

echo "### no hang within ${{MAXS}}s; last round=$LASTR ###"
echo "VERDICT=NOHANG"
exit 0
"""
