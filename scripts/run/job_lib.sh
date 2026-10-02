# shellcheck shell=bash disable=SC2034  # XLA_ENV and LEG_KILLED are for the sourcing script
# Sourced by the multi-node job scripts: legs run with a time cap and a silence limit, a leg's
# XLA_FLAGS, and the checkpoint gate. tests/test_job_scripts.py runs each piece.
#
# The sourcing script sets LEG_DIR (each leg's output is also kept in LEG_DIR/NN-name.log) and
# rc_total (each failed leg or gate adds one). Optional: LEG_POLL_S (10 s between checks on a
# running leg), LEG_KILL_GRACE_S (60 s from TERM to KILL).

LEG_N=0
LEG_KILLED=0

# xla_env [EXTRA]: sets XLA_ENV, the `env` prefix that gives a leg the job's XLA_FLAGS plus
# EXTRA. With both empty XLA_FLAGS is unset: XLA reads a value that is not a `--` flag as a
# file name and aborts.
xla_env () {
  local parts=()
  [ -n "${XLA_FLAGS:-}" ] && parts+=("$XLA_FLAGS")
  [ -n "${1:-}" ] && parts+=("$1")
  if [ ${#parts[@]} -gt 0 ]; then
    XLA_ENV=(env "XLA_FLAGS=${parts[*]}")
  else
    XLA_ENV=(env -u XLA_FLAGS)
  fi
}

# leg_tree PID: PID and every descendant, children first
leg_tree () {
  local c
  for c in $(pgrep -P "$1" 2>/dev/null); do leg_tree "$c"; done
  echo "$1"
}

# leg_kill PID: TERM to the whole tree, KILL to what is left LEG_KILL_GRACE_S later
leg_kill () {
  local pids p t=0 left
  pids=$(leg_tree "$1")
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null
  while [ $t -lt "${LEG_KILL_GRACE_S:-60}" ]; do
    left=""
    for p in $pids; do kill -0 "$p" 2>/dev/null && left=1; done
    [ -z "$left" ] && return 0
    sleep 1; t=$((t + 1))
  done
  # shellcheck disable=SC2086
  kill -KILL $pids 2>/dev/null
  return 0
}

# run_leg [--cap S] [--quiet-limit S] [--expect-fail] NAME CMD...
#   Runs CMD (a command or a shell function) in the background and polls it, so the batch
#   shell's traps (the pre-wall USR1) run during the leg. Its output goes to stdout and to
#   LEG_DIR/NN-NAME.log. --cap kills the leg S seconds after it starts; --quiet-limit kills it
#   once its output has been silent for S seconds. A killed leg sets LEG_KILLED=1 and
#   returns 124. --expect-fail inverts the leg: it passes (0) only when CMD fails by itself.
#   A failed leg adds one to rc_total.
run_leg () {
  local cap="" quiet="" expect_fail=0
  while :; do
    case "$1" in
      --cap) cap=$2; shift 2 ;;
      --quiet-limit) quiet=$2; shift 2 ;;
      --expect-fail) expect_fail=1; shift ;;
      *) break ;;
    esac
  done
  local name=$1; shift
  LEG_N=$((LEG_N + 1)); LEG_KILLED=0
  mkdir -p "$LEG_DIR"
  local out
  out=$LEG_DIR/$(printf %02d "$LEG_N")-${name//[^A-Za-z0-9.-]/_}.log
  : > "$out"; rm -f "$out.rc"
  echo ""; echo "=== LEG $name ($(date +%H:%M:%S)) ==="
  ( "$@" > >(tee -a "$out") 2>&1; echo $? > "$out.rc" ) &
  local pid=$! t0=$SECONDS last=$SECONDS size=0 now why
  while kill -0 "$pid" 2>/dev/null; do
    sleep "${LEG_POLL_S:-10}" & wait $!
    now=$(wc -c < "$out" | tr -d ' ')
    if [ "$now" != "$size" ]; then size=$now; last=$SECONDS; fi
    why=""
    if [ -n "$cap" ] && [ $((SECONDS - t0)) -ge "$cap" ]; then
      why="ran past its $cap s cap"
    elif [ -n "$quiet" ] && [ $((SECONDS - last)) -ge "$quiet" ]; then
      why="printed nothing for $quiet s"
    fi
    if [ -n "$why" ]; then
      echo "=== LEG $name KILLED ($(date +%H:%M:%S)): it $why ==="
      LEG_KILLED=1
      leg_kill "$pid"
      break
    fi
  done
  wait "$pid" 2>/dev/null
  local rc=124
  if [ "$LEG_KILLED" = 0 ]; then rc=$(cat "$out.rc" 2>/dev/null || echo 125); fi
  echo "--- LEG $name rc=$rc ($(date +%H:%M:%S)) ---"
  local ok=$rc
  if [ "$expect_fail" = 1 ]; then
    if [ "$rc" -ne 0 ] && [ "$LEG_KILLED" = 0 ]; then ok=0; else ok=1; fi
  fi
  if [ "$ok" -ne 0 ]; then rc_total=$((rc_total + 1)); fi
  return "$ok"
}

# same_ckpt A B: the same files under two checkpoint dirs, byte for byte. A dir that is
# missing or holds no generation (no */manifest.json) fails.
same_ckpt () {
  local a=$1 b=$2 bad=0 f d
  for d in "$a" "$b"; do
    if [ ! -d "$d" ] || ! compgen -G "$d/*/manifest.json" >/dev/null; then
      echo "  GATE FAIL: no checkpoint generation in $d"
      return 1
    fi
  done
  if ! diff <(cd "$a" && find . -type f | sort) <(cd "$b" && find . -type f | sort) >/dev/null; then
    echo "  GATE FAIL: $a and $b hold different files"
    diff <(cd "$a" && find . -type f | sort) <(cd "$b" && find . -type f | sort) | head -20
    return 1
  fi
  while IFS= read -r f; do
    cmp -s "$a/$f" "$b/$f" || { echo "  differs: $f"; bad=$((bad + 1)); }
  done < <(cd "$a" && find . -type f | sort)
  echo "  $(cd "$a" && find . -type f | wc -l | tr -d ' ') files, $bad differing: $a vs $b"
  [ $bad -eq 0 ]
}

# gate NAME DIR...: every DIR's files equal the first's
gate () {
  local name=$1 first=$2 d ok=0; shift 2
  echo ""; echo "=== GATE $name ==="
  for d in "$@"; do same_ckpt "$first" "$d" || ok=1; done
  if [ $ok -eq 0 ]; then echo "GATE $name PASS"; else echo "GATE $name FAIL"; fi
  return $ok
}
