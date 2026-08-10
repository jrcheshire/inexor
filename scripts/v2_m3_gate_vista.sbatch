#!/bin/bash
#SBATCH -J inexor-m3-gate
#SBATCH -o %x-%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 03:00:00
#
# M-v2-3 exit gate, legs A and B (scripts/v2_m3_engine_gate.py).
#
# WHAT THIS ANSWERS
#   A. Is the engine's force -- read out of slot-ordered T9 state and driven from
#      brick SPANS -- BITWISE the ratified path's, which is driven from the
#      probe's own bucketing? The two membership ORDERS differ, so this is only
#      a meaningful question because M-v2-3 made both paints integer and
#      therefore order-independent. D-v2-16 clause 7's promotion gate had to feed
#      one membership to both arms; this deliberately does not.
#   B. What does the T9 codec cost on the architecture that SHIPS? D-v2-14's
#      ratified 4.123e-4 was measured against a monolithic force with the
#      compression faked in float space -- never through the two-level force,
#      never through the real layout, never with a reordering particle sequence.
#
# WHAT IT DOES NOT ANSWER
#   Performance. The engine's per-tile loop is host numpy and the layout was
#   85.5% of the force on this backend at M-v2-1; nothing here is axis-matched
#   against anything, and the walls below are budget, not measurement.
#
# PRE-REGISTERED, before the numbers exist: leg A is expected to be exactly 0
# differing elements (it is bitwise or it is broken, there is no middle), and
# leg B at cdev K=40 is expected within a factor of a few of 4.123e-4 and at
# least an order under D-v2-9's 3e-2. A miss on B is a finding.
#
# BUDGET, from a measured rate rather than a guess: cdev8 K=40 ran 33 s/step
# locally on 4 CPU threads, so ~25 min; cdev is 8x the particles. 12 h is
# deliberately generous -- billed for actual use, and the last several runtime
# estimates on this project were all low.

set -u

# NOT boilerplate. Jobs 896055 and 896092 ran ZERO science without this line,
# and 896055 reported COMPLETED 0:0 in sacct because `set -e` had been dropped.
export PATH="$HOME/.pixi/bin:$PATH"

cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
echo "commit: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"
echo "host:   $(hostname)   started $(date -Is)"

command -v pixi >/dev/null 2>&1 || { echo "FATAL: pixi not on PATH"; exit 1; }

# The backend is DEMONSTRATED, not inferred. Job 897377 could only infer the GPU
# was live from a 6.83x speedup, and the log recorded no device at all.
pixi run -e gpu python -c "
import jax, sys
d = jax.devices()
print('jax', jax.__version__, 'devices:', d, flush=True)
if d[0].platform == 'cpu':
    sys.exit('FATAL: CPU backend -- this gate must exercise the CUDA scatter')
" || { echo "FATAL: backend check failed"; exit 1; }

export OMP_NUM_THREADS=72   # one Grace socket; the layout is host numpy

# STAMP: the CARDS WRITTEN block counts only files newer than this. Added
# 2026-08-09 (M-v2-4): the glob below had no freshness filter, so a job that
# wrote NOTHING still passed its own "zero cards written" FATAL on a previous
# job's files. That is the 896055 failure -- a green row for a run that produced
# nothing -- surviving in the guard built to catch it.
STAMP="$(mktemp)"; touch "$STAMP"

# No `set -e`: the legs are independent and the cheapest run first, so a failure
# in one must not delete the evidence from the others (M1 job 11's lesson).
NAMES=(); CODES=()
run_leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG ${name} : $* ==="; date -Is
  pixi run -e gpu python "$@"
  local rc=$?
  NAMES+=("$name"); CODES+=("$rc")
  echo "=== LEG ${name} exit ${rc} ==="; date -Is
}

G=scripts/v2_m3_engine_gate.py

# Job 898169 read out 2026-08-09: cdev8-force PASS, cdev8-accum 4.666e-4,
# cdev-accum 7.125e-4, and cgh64-force exit 1 on the gate's own O(N) comparison
# array hitting D-v2-16 clause 1's 2 GiB refusal. Only that leg is re-run.
run_leg cgh64-force  "$G" --config cgh64 --leg force --slack 0.15 --arena-frac 0.05

echo ""
echo "=== LEG SUMMARY ==="
ok=0
for i in "${!NAMES[@]}"; do
  echo "  ${NAMES[$i]}: exit ${CODES[$i]}"
  [ "${CODES[$i]}" -eq 0 ] && ok=$((ok+1))
done

echo ""
echo "=== CARDS WRITTEN (this job only) ==="
n=0
while IFS= read -r f; do
  echo "  $f"; n=$((n+1))
done < <(find runs/v2 -maxdepth 1 -name 'm3_gate_*.json' -newer "$STAMP" 2>/dev/null)
[ "$n" -eq 0 ] && echo "  (none)"
rm -f "$STAMP"

echo ""
echo "legs ok: ${ok} of ${#NAMES[@]};  fresh cards: ${n}"
[ "$ok" -gt 0 ] || { echo "FATAL: zero legs succeeded"; exit 1; }
[ "$n" -eq 0 ] && { echo "FATAL: zero cards written by THIS job"; exit 1; }
exit 0
