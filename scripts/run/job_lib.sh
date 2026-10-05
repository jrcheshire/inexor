# shellcheck shell=bash disable=SC2034  # XLA_ENV and LEG_KILLED are for the sourcing script
# Sourced by the job scripts that run legs: legs run with a time cap and a silence limit, a leg's
# XLA_FLAGS, and the checkpoint gate. tests/test_job_scripts.py runs each piece.
#
# The sourcing script sets LEG_DIR (each leg's output is also kept in LEG_DIR/NN-name.log) and
# rc_total (each failed leg or gate adds one). Optional: LEG_POLL_S (10 s between checks on a
# running leg), LEG_KILL_GRACE_S (60 s from TERM to KILL).
# `on_ranks` reads LAUNCH, RANK_EXEC and MEMBIND_RANK, and `summarize` PY, CPU, DRIVER and
# N_RANKS (all from mpi_env_vista.sh), and CARDS and TAG (the script's card dir and prefix).

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

# on_ranks SAMPLES EXTRA CMD...: CMD on every node at once, one process per node, through the
# launch line of every run leg: rank_exec (membind, samplers) and the job's XLA_FLAGS + EXTRA
on_ranks () {
  local samples=$1 extra=$2; shift 2
  xla_env "$extra"
  # shellcheck disable=SC2086
  $LAUNCH "$RANK_EXEC" "${MEMBIND_RANK[@]}" --samples "$samples" -- "${XLA_ENV[@]}" "$@"
}

# summarize ARM FROM: every rank's card of one leg against its node's samples. Arms named
# ref* / det-ref* are one-node runs with a card per node (TAG_ARM<r>_sFROM.json); any other
# arm is one run across the ranks (TAG_ARM_sFROM.rank<r>.json).
summarize () {
  local arm=$1 from=$2 card r
  for ((r = 0; r < N_RANKS; r++)); do
    case $arm in
      ref*|det-ref*) card=$CARDS/${TAG}_${arm}${r}_s${from}.json ;;
      *) card=$CARDS/${TAG}_${arm}_s${from}.rank${r}.json ;;
    esac
    [ -f "$card" ] || continue
    "${CPU[@]}" "$PY" "$DRIVER" summarize --card "$card" \
      --gpu-csv "$CARDS/${TAG}_${arm}_s${from}_r${r}_gpu.csv" \
      --mem-csv "$CARDS/${TAG}_${arm}_s${from}_r${r}_mem.csv" \
      --out "${card%.json}_summary.json" >/dev/null 2>&1 \
      && echo "  summary ${card%.json}_summary.json"
  done
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

# same_ckpt_any_cut A B: `same_ckpt`, except that manifests are compared without
# provenance.device_shapes.window, the one entry a y-block count changes (it sizes a buffer;
# the checkpoint's numbers do not depend on the cut)
same_ckpt_any_cut () {
  local a=$1 b=$2 bad=0 f d
  for d in "$a" "$b"; do
    if [ ! -d "$d" ] || ! compgen -G "$d/*/manifest.json" >/dev/null; then
      echo "  GATE FAIL: no checkpoint generation in $d"
      return 1
    fi
  done
  if ! diff <(cd "$a" && find . -type f | sort) <(cd "$b" && find . -type f | sort) >/dev/null; then
    echo "  GATE FAIL: $a and $b hold different files"
    return 1
  fi
  while IFS= read -r f; do
    if [ "${f##*/}" = manifest.json ]; then
      "${PY:-python3}" - "$a/$f" "$b/$f" <<'EOF_PY' || { echo "  differs: $f"; bad=$((bad + 1)); }
import json, sys
m = [json.load(open(p)) for p in sys.argv[1:]]
for x in m:
    x.get("provenance", {}).get("device_shapes", {}).pop("window", None)
sys.exit(m[0] != m[1])
EOF_PY
    else
      cmp -s "$a/$f" "$b/$f" || { echo "  differs: $f"; bad=$((bad + 1)); }
    fi
  done < <(cd "$a" && find . -type f | sort)
  echo "  $(cd "$a" && find . -type f | wc -l | tr -d ' ') files, $bad differing (manifests without the window shape): $a vs $b"
  [ $bad -eq 0 ]
}

# gate_any_cut NAME DIR...: `gate` with `same_ckpt_any_cut`
gate_any_cut () {
  local name=$1 first=$2 d ok=0; shift 2
  echo ""; echo "=== GATE $name ==="
  for d in "$@"; do same_ckpt_any_cut "$first" "$d" || ok=1; done
  if [ $ok -eq 0 ]; then echo "GATE $name PASS"; else echo "GATE $name FAIL"; fi
  return $ok
}

# manifest_field GENDIR EXPR: EXPR of GENDIR/manifest.json, loaded as `m`
manifest_field () {
  "${PY:-python3}" -c "import json, sys; m = json.load(open(sys.argv[1] + '/manifest.json')); print($2)" "$1"
}

# ckpt_generations CKPT: "OLDER FROM TO", the generation at the lower step and the two steps.
# Fails unless both generations have a manifest, at different steps.
ckpt_generations () {
  local s0 s1
  s0=$(manifest_field "$1/gen0" "m['provenance']['step']" 2>/dev/null) || return 1
  s1=$(manifest_field "$1/gen1" "m['provenance']['step']" 2>/dev/null) || return 1
  if [ "$s0" -lt "$s1" ]; then echo "gen0 $s0 $s1"
  elif [ "$s1" -lt "$s0" ]; then echo "gen1 $s1 $s0"
  else return 1
  fi
}

# ckpt_newest CKPT: "GEN STEP" of the generation at the higher step; fails when none has a
# manifest
ckpt_newest () {
  local g s best=-1 newest=""
  for g in gen0 gen1; do
    s=$(manifest_field "$1/$g" "m['provenance']['step']" 2>/dev/null) || continue
    if [ "$s" -gt "$best" ]; then best=$s newest=$g; fi
  done
  [ -n "$newest" ] && echo "$newest $best"
}

# gate NAME DIR...: every DIR's files equal the first's
gate () {
  local name=$1 first=$2 d ok=0; shift 2
  echo ""; echo "=== GATE $name ==="
  for d in "$@"; do same_ckpt "$first" "$d" || ok=1; done
  if [ $ok -eq 0 ]; then echo "GATE $name PASS"; else echo "GATE $name FAIL"; fi
  return $ok
}

# gate_with NAME CMP FIRST OTHER...: every OTHER against FIRST by the comparison function CMP
gate_with () {
  local name=$1 cmp=$2 first=$3 d ok=0; shift 3
  echo ""; echo "=== GATE $name ==="
  for d in "$@"; do "$cmp" "$first" "$d" || ok=1; done
  if [ $ok -eq 0 ]; then echo "GATE $name PASS"; else echo "GATE $name FAIL"; fi
  return $ok
}

# same_card A B: two P(k) cards (`device_run.py card` / `realization.py card` JSON) equal in
# every field of their summary except its provenance. An unreadable card fails.
same_card () {
  "${PY:-python3}" - "$1" "$2" <<'EOF_PY'
import json, sys
try:
    s = [json.load(open(p))["summary"] for p in sys.argv[1:]]
except (OSError, ValueError, KeyError) as e:
    print(f"  GATE FAIL: unreadable card ({e!r})")
    sys.exit(1)
for x in s:
    x.pop("provenance", None)
bad = sorted(k for k in set(s[0]) | set(s[1]) if s[0].get(k) != s[1].get(k))
print(f"  card fields differing: {', '.join(bad) or 'none'}: {sys.argv[1]} vs {sys.argv[2]}")
sys.exit(1 if bad else 0)
EOF_PY
}

# same_export A B: two complete exports (export.json present; one file per array or one part
# per rank) holding the same particle count and the same whole-array crc32 of every array
same_export () {
  "${PY:-python3}" - "$1" "$2" <<'EOF_PY'
import json, os, sys
try:
    h = [json.load(open(os.path.join(p, "export.json"))) for p in sys.argv[1:]]
except (OSError, ValueError) as e:
    print(f"  GATE FAIL: no complete export ({e!r})")
    sys.exit(1)
same = all(h[0].get(k) == h[1].get(k) for k in ("n_particles", "crc32", "dtype"))
print(f"  {h[1].get('n_particles')} particles, crc32 {h[1].get('crc32')} vs "
      f"{h[0].get('crc32')}: {'same' if same else 'DIFFERENT'}: {sys.argv[1]} vs {sys.argv[2]}")
sys.exit(0 if same else 1)
EOF_PY
}

# card_diff A B: per-bin |p_B / p_A - 1| and |z_B - z_A| of two cards, printed; fails only
# when a card is unreadable or the bins differ
card_diff () {
  "${PY:-python3}" - "$1" "$2" <<'EOF_PY'
import json, sys
import numpy as np
try:
    a, b = (json.load(open(p))["summary"] for p in sys.argv[1:])
except (OSError, ValueError, KeyError) as e:
    print(f"  card_diff: unreadable card ({e!r})")
    sys.exit(1)
if a["k_edges"] != b["k_edges"] or len(a["p"]) != len(b["p"]):
    print("  card_diff: the cards' bins differ")
    sys.exit(1)
dp = np.abs(np.asarray(b["p"]) / np.asarray(a["p"]) - 1.0)
dz = np.abs(np.asarray(b["z_profile"]) - np.asarray(a["z_profile"]))
print(f"  card_diff ({a.get('transform', 'host')} -> {b.get('transform', 'host')}, "
      f"{len(dp)} bins): |dp/p| max {dp.max():.3e} median {np.median(dp):.3e}; "
      f"|dz| max {dz.max():.3e}")
EOF_PY
}

# same_ics A B: two IC generations (manifest present) with the same slab files byte for byte
# and the same manifest except what a rank, card or y-block count and the run record
# (provenance, timings, n_devices, n_ranks, emission_y_blocks, stage_cleanup)
same_ics () {
  "${PY:-python3}" - "$1" "$2" <<'EOF_PY'
import json, os, sys
skip = ("provenance", "stage_s", "emission_s", "n_devices", "n_ranks", "emission_y_blocks",
        "stage_cleanup")
try:
    m = [json.load(open(os.path.join(d, "manifest.json"))) for d in sys.argv[1:]]
except (OSError, ValueError) as e:
    print(f"  GATE FAIL: no complete IC generation ({e!r})")
    sys.exit(1)
bad = [f for f in m[0]["files"] if f not in m[1]["files"] or
       open(os.path.join(sys.argv[1], f), "rb").read() != open(os.path.join(sys.argv[2], f), "rb").read()]
if sorted(m[0]["files"]) != sorted(m[1]["files"]):
    bad.append("the file lists")
fields = sorted(k for k in set(m[0]) | set(m[1]) if k not in skip and m[0].get(k) != m[1].get(k))
print(f"  {len(m[0]['files'])} slab files, differing: {', '.join(bad) or 'none'}; manifest "
      f"fields differing: {', '.join(fields) or 'none'}: {sys.argv[1]} vs {sys.argv[2]}")
sys.exit(1 if bad or fields else 0)
EOF_PY
}
