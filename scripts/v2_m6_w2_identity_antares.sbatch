#!/bin/bash
#SBATCH -J m6-w2-id
#SBATCH -p compute
#SBATCH --mem=64G
#SBATCH -c 28
#SBATCH -t 06:00:00
#SBATCH -o m6-w2-id-%j.log
# W2 Stage D leg 1 (antares, zero SU): the executor identity at the scale the
# laptop does not run, plus the owed cgh64 coarse_paint A/B in the SAME job
# (the tile_long/coarse_paint cross-job drift is recorded-unexplained, so the
# A/B arms must share a job).
#
# LEGS 1-2 = executor identity. The pool executor must be BITWISE the serial
# loop over a whole K=4 run -- every state array, ids, the cap ladder,
# arena_base. PRE-REGISTERED: n_diff 0 everywhere. A nonzero kills the
# parallel premise and W2 stops; there is nothing to tune. (Laptop already
# read 0 at smoke + cdev8 W=2/4; leg 2 is cdev, the M-v2-3 gate geometry.)
# Antares POOLED WALLS ARE NOT A VERDICT (1.4x ceiling, job 466/467): the
# scaling number comes from the gg job, not this one.
#
# LEGS 3-4 = the Stage 2c A/B at cgh64: paint_subblock on vs off (off = the
# full-mesh-per-chunk arm, bitwise neutral, NOT an operating point), serial
# executor, phase table read on coarse_paint. This is the first decomposition
# arm for the phase's unexplained 132 s/step (job 467: 29.0% of the step; the
# full-mesh N^2 derivation overpredicted 3.4x at 464; cdev payoff measured
# ~2%). PRE-REGISTERED READ: the on-off coarse_paint difference IS what 2c
# bought at cgh64; the card's coarse_subblock_chunks proves each knob applied
# (>0 on, 0 off).
#
# ~3.5-4 h expected, 6 h requested. Zero SU. NO set -e: per-leg exit codes;
# the smoke identity gates the long legs.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
pixi run -e default python -c "
import jax
assert jax.devices()[0].platform == 'cpu', 'expected the CPU backend'
print('jax', jax.__version__)
" || { echo "FATAL: jax import/backend check failed"; exit 1; }

echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2
STAMP="/tmp/m6_w2_stamp_$SLURM_JOB_ID"; touch "$STAMP"

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

IDENT="scripts/v2_m6_w2_identity.py"
PHASE="scripts/v2_m6_phase_time.py"

run_leg "id-smoke-cdev8" pixi run -e default python $IDENT \
  --config cdev8 --k 4 --workers 4 --tile 64 \
  --out runs/v2/m6_w2_identity_cdev8.json
[ "${codes[0]}" -ne 0 ] && {
  echo "FATAL: the identity smoke failed; not burning the long legs"
  exit 1
}

run_leg "id-cdev" pixi run -e default python $IDENT \
  --config cdev --k 4 --workers 4 \
  --out runs/v2/m6_w2_identity_cdev.json

run_leg "2c-ab-cgh64-on" pixi run -e default python $PHASE \
  --config cgh64 --k 3 --repeats 1 --slack 0.20 --arena-frac 0.20 \
  --paint-subblock 1 --out-suffix _2c_cgh64_on

run_leg "2c-ab-cgh64-off" pixi run -e default python $PHASE \
  --config cgh64 --k 3 --repeats 1 --slack 0.20 --arena-frac 0.20 \
  --paint-subblock 0 --out-suffix _2c_cgh64_off

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST: laptop identity n_diff 0 at cdev8 W=2/4; job 467 cgh64 457.5 s/step, coarse_paint 132 s (29.0%)"
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m6_w2_identity_*.json runs/v2/m6_phase_time_2c_cgh64_*.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"
rm -f "$STAMP"
exit $rc_total
