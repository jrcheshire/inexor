#!/bin/bash
#SBATCH -J match-order
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o match-order-%j.log
# Does the coarse match-order defect drive the high-k deficit, and how much of it?
#
# THE DEFECT. The coarse arm paints and gathers TSC, but its match factor is built
# at CIC order, so the long force keeps sinc^2(k d_c / 2) per axis uncorrected.
# Against an Ewald reference at production cells the total force is 3-4.5% low at
# r = 1.6-3.4 Mpc/h, all of it in the long arm (`scripts/v2_force_profile.py`);
# `--coarse-match-order 3` rebuilds the factor at TSC order and the long arm's
# worst error falls from -4.05% to -0.47% of the total.
#
# ARMS: the corrected force at the SAME seed, particle count and bins as the step
# ladder (1016460), so each card differences against its ratified-force twin with
# the realization cancelled -- the seed-to-seed scatter at this box is 2-4% at
# high k (1018128), and a matched pair reads far below it. Cheapest first:
#   m3-k40    40 steps  -> vs ksl_1016460_k40   (the production step count)
#   m3-k120   120 steps -> vs ksl_1016460_k120  (converged; the question)
#
# TIME, from this box's measured 24.4-24.5 s/step (1016874, 1018128); the match
# factor is built once per run, so the step rate does not change:
#   ICs (measured 1.4 min) + 2 cards                          ~2 min
#   m3-k40    40 x 24.5 s                                     ~16 min
#   m3-k120   120 x 24.5 s                                    ~49 min
#                                                             --------
#                                                             ~67 min
#
# MEMORY: 153-157 GB measured at this config and W=16; unchanged by the knob.
#
# BUDGET: gg = 0.33 SU/node-hr -> ~0.4 SU expected, 0.66 worst case (2 h).
#
# OPS (load-bearing on gg): pixi on PATH; JAX_PLATFORMS=cpu (jax 0.10
# hard-raises on cuInit on a GPU-less node); CONDA_OVERRIDE_CUDA; --generator
# host (the device IC generator needs cards).

set -u

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export CONDA_OVERRIDE_CUDA=12.0
export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { rc=$?; echo "FATAL (rc=$rc): pixi not on PATH"; exit 1; }

RUNS=/scratch/10303/jrc4/inexor_runs
WORK=$RUNS/match-order-$SLURM_JOB_ID
CARDS=runs/v2
TAG=mo3_$SLURM_JOB_ID
RLZ=scripts/v2_m6_realization.py
W="--tile-workers 16"
LAYOUT="--slack 0.10 --arena-frac 0.01"
GEO="--config cgh64 --coarse-match-order 3"

mkdir -p "$WORK"
echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
free -g | head -2
df -h "$RUNS" | tail -1
nproc

rc_total=0
finish () {
  echo ""; echo "=== cards ==="
  ls -la "$WORK"/m3-*/realization_pk.json 2>/dev/null
  echo "=== end $(date) rc_total=$rc_total ==="
}
trap finish EXIT

run_leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ($(date +%H:%M:%S)) ==="
  "$@" & local pid=$!
  local rc; wait $pid; rc=$?
  echo "--- LEG $name rc=$rc ($(date +%H:%M:%S)) ---"
  [ $rc -ne 0 ] && rc_total=$((rc_total + 1))
  return $rc
}

run_leg ics \
  pixi run -e gpu python $RLZ ics --config cgh64 --workdir "$WORK/ics" --generator host \
  || { rc=$?; echo "FATAL (rc=$rc): IC generation failed; no arm can run"; exit 1; }

for K in 40 120; do
  A="$WORK/m3-k$K"
  mkdir -p "$A"
  for f in "$WORK"/ics/*; do ln -sfn "$f" "$A/$(basename "$f")"; done
  rm -f "$A/realization_ics.json"

  run_leg "run-m3-k$K" \
    pixi run -e gpu python $RLZ run $GEO --k-steps $K --workdir "$A" $W $LAYOUT \
      --checkpoint-every $K --phase-instrument time \
    || { echo "NOTE: arm k=$K failed or hit the wall; earlier arms stand"; continue; }

  run_leg "card-m3-k$K" \
    pixi run -e gpu python $RLZ card $GEO --k-steps $K --workdir "$A" $W $LAYOUT \
      --heartbeat 60 \
    || echo "NOTE: card for k=$K failed; its checkpoint is at $A/ckpt"

  cp -f "$A/realization_pk.json" "$CARDS/${TAG}_k$K.json" 2>/dev/null || true
done

exit $rc_total
