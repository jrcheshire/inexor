#!/bin/bash
#SBATCH -J m6-cdev-matched
#SBATCH -p compute
#SBATCH --mem=100G
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -o m6-cdev-matched-%j.log
# Is the engine step LINEAR in particle count now? The matched-knob cdev point.
#
# Job 459 put cgh64 (512^3, nb=32) at 608.67 s/step on the current code, down
# from 2622.57. The obvious next question -- is what remains linear in N -- has
# no answer, because the only cdev wall on record (61 s) came from the trim work
# at `--arena-frac 0.08` against 459's 0.20. That knob sets `n_arena`, so it
# changes the row count every per-step pass walks: the two are not comparable
# and the record now says the 43x should not be quoted as a controlled number.
#
# This job supplies the missing point. Same code, same machine, same cores, same
# legs, same K, and the SAME KNOBS as 459 -- the only thing that moves is the
# configuration.
#
# PRE-REGISTERED, before submission:
#   - LINEAR in N predicts cdev at 608.67 / 8 = **76.1 s/step** (cgh64 has 8x
#     the particles).
#   - 60-95 s/step (within ~25% of that) reads as linear to the precision this
#     pair can support, and production sizing may then proceed from cgh64's
#     608.67 rather than waiting on a further ladder.
#   - **Below ~50 s/step a superlinear term survives** -- more than 12x the wall
#     for 8x the particles -- and it gets its own one-axis hunt rather than a
#     projection. That is the outcome I think most likely: the brick loop runs
#     nb^3 times per step (4,096 at cdev against 32,768 at cgh64), and the
#     post-fix probe ladder was NOT flat (0.91x then 1.67x per doubling of nb).
#   - Above ~95 s/step means cdev is disproportionately EXPENSIVE, which would
#     point at a per-step cost that does not scale with particles at all -- also
#     interesting, also not a projection.
#
# NB this is a two-point comparison and two points cannot tell a line from a
# knee. It is enough to decide whether a hunt is needed; it is not enough to
# quote a production wall from, and section 6 of `m6_scaling_record.md` already
# says nothing should be projected before it.
#
# The peak comes along free and is comparable for once: 459's 11.600 GB at cgh64
# and this at cdev share machine, knobs and code, which no earlier pair did.
#
# Legs, cheapest first, smoke gating:
#   1 smoke        orchestrator end to end, seconds
#   2 cdev build   gen + load
#   3 cdev step K=3
#
# ~10 min expected, 2 h requested. Zero SU. 8 of 28 cores. Scratch ~2 GB (cdev
# stages 1/8 of cgh64's ~13 GB), guard at 8. Cleans up on success.
# NO set -e: independent legs, per-leg exit codes -- except smoke, which gates.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
pixi run -e default python -c "
import jax, platform, sys
d = jax.devices()
print('jax', jax.__version__, d, platform.machine(), sys.platform)
assert d[0].platform == 'cpu', 'expected the CPU backend in the default env'
" || { echo "FATAL: jax import/backend check failed"; exit 1; }

export TMPDIR="${TMPDIR:-/tmp}/m6_cdevmatched_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2

avail=$(df -BG --output=avail "$TMPDIR" | tail -1 | tr -dc '0-9')
echo "scratch available under TMPDIR: ${avail} GB (need ~2, guard at 8)"
[ "${avail:-0}" -lt 8 ] && { echo "FATAL: want >=8 GB of scratch, have ${avail}"; exit 1; }

STAMP="$TMPDIR/.stamp"; touch "$STAMP"

rc_total=0
names=(); codes=()
run_leg() {
  local name="$1"; shift
  echo "=== $name ==="
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "=== $name rc=$rc after $((SECONDS - t0)) s ==="
  names+=("$name"); codes+=("$rc")
  [ $rc -ne 0 ] && rc_total=1
  return 0
}

PEAK="scripts/v2_m6_engine_peak.py"
KNOBS="--slack 0.20 --arena-frac 0.20"   # EXACTLY job 459's, or nothing compares

run_leg "smoke" pixi run -e default python $PEAK \
  --configs smoke --k 3 --legs gen load step $KNOBS \
  --workdir-root "$TMPDIR" --out-suffix _cdevmatched_smoke
[ "${codes[0]}" -ne 0 ] && {
  echo "FATAL: smoke failed; a broken probe would burn the long legs (job 451)"
  exit 1
}

run_leg "cdev-build" pixi run -e default python $PEAK \
  --configs cdev --legs gen load $KNOBS \
  --workdir-root "$TMPDIR" --out-suffix _cdevmatched_build

run_leg "cdev-step-k3" pixi run -e default python $PEAK \
  --configs cdev --k 3 --legs step $KNOBS \
  --workdir-root "$TMPDIR" --out-suffix _cdevmatched_step

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST JOB 459 (cgh64, same code/machine/knobs): 608.67 s/step, 11.600 GB"
echo "LINEAR IN N WOULD PUT THIS LEG AT 76.1 s/step"
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m6_peak_cdevmatched_*.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"

if [ $rc_total -eq 0 ] && [ -n "$SLURM_JOB_ID" ] && [ -d "$TMPDIR" ]; then
  before=$(du -sh "$TMPDIR" 2>/dev/null | cut -f1)
  find "$TMPDIR" -type f -delete
  find "$TMPDIR" -depth -type d -empty -delete
  echo "TMPDIR cleaned on success (was $before): $TMPDIR"
else
  echo "TMPDIR KEPT for inspection (rc=$rc_total): $TMPDIR"
fi
exit $rc_total
