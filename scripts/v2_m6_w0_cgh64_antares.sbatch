#!/bin/bash
#SBATCH -J m6-w0-cgh64
#SBATCH -p compute
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 05:00:00
#SBATCH -o m6-w0-cgh64-%j.log
# W0(a) of the wall plan: phase-time the cgh64 step, which nothing has done.
# Every phase share on record is cdev8/cdev; the cgh64 wall (608.67 s/step,
# job 459, THIS machine) has never been decomposed.
#
# THE PREDICTION ON THE LINE, derived from source before this job ran:
# `coarse_delta_streamed` pays a full n_coarse^3 allocate+widen+add PER CHUNK
# (engine.py:453-464), so the term scales ~N^2 (chunks ~n_bricks ~N, per-chunk
# mesh ~n_coarse^3 ~N). Measured shares are consistent so far: 5.0% at cdev8
# -> 13.8% at cdev (10x absolute for 8x N). Carried to cgh64 (8x N again, and
# n_coarse 128 -> 256) the derivation puts coarse_paint at ~50-60% of the
# step.
#   - If measured share is >= ~40%: the superlinearity attribution stands and
#     Stage 2c (sub-block paint) is confirmed as the largest single wall lever
#     at scale, prerequisite for any C-gh run.
#   - If measured share is <= ~20%: the attribution is WRONG, the W1 payoff
#     was mispriced, and the wall model (W0b) has a defect to find BEFORE any
#     lever work proceeds.
# Either way the full phase table is the denominator the wall plan prices
# every lever against.
#
# Same machine as job 459 deliberately: peak/wall comparisons are not valid
# across machines, and 459's 608.67 s/step is the number this decomposition
# must sum to (the neutrality gate + control arm check that internally).
#
# --mem=64G is ROUTING, not need (cgh64 measured 14.447 GB peak at these
# knobs, job 455): >56G keeps it off deneb-the-node, which is in interactive
# use. Antares has 124 GB. Zero SU. Wall 5 h against ~2.5 h expected
# (warmup K + 4 timed runs x K=3 x ~610 s); estimates have run low.
# NO set -e: per-leg exit codes; smoke gates the long leg (job 451 lesson).

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
pixi run -e default python -c "
import jax, platform, sys
d = jax.devices()
print('jax', jax.__version__, d, platform.machine(), sys.platform)
assert d[0].platform == 'cpu', 'expected the CPU backend in the default env'
" || { echo "FATAL: jax import/backend check failed"; exit 1; }

echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2
STAMP="/tmp/m6_w0_stamp_$SLURM_JOB_ID"; touch "$STAMP"

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

PHASE="scripts/v2_m6_phase_time.py"

# Smoke at the long leg's OWN knobs (job 462 lesson: a smoke must not be
# harsher, or gentler in a way that skips the regime, than the leg it gates).
run_leg "smoke" pixi run -e default python $PHASE \
  --config smoke --k 3 --repeats 2 --slack 0.20 --arena-frac 0.20 \
  --out-suffix _w0_smoke
[ "${codes[0]}" -ne 0 ] && {
  echo "FATAL: smoke failed; a broken probe would burn the long leg (job 451)"
  exit 1
}

# ---- the measurement. Generous slack/arena so the D-007 arena refusal cannot
# ---- end a TIMING leg; same knobs as jobs 455/459, so the wall is comparable.
run_leg "phase-cgh64" pixi run -e default python $PHASE \
  --config cgh64 --k 3 --repeats 2 --slack 0.20 --arena-frac 0.20 \
  --out-suffix _w0_cgh64

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST: cgh64 608.67 s/step (job 459, antares); cdev coarse_paint 13.8% (job 463, deneb)"
echo "PRE-REGISTERED: coarse_paint >= ~40% of the cgh64 step confirms the ~N^2 term; <= ~20% refutes it"
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m6_phase_time_w0_*.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"
rm -f "$STAMP"
exit $rc_total
