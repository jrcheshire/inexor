#!/bin/bash
#SBATCH -J ksteps-ladder
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o ksteps-ladder-%j.log
# Does the step count matter at the configuration we run?
#
# WHY. On the accuracy-versus-cost sheet this is the one knob with neither a
# measured outcome nor an analytic bound. Force resolution is being measured by
# the mesh ladder; the two split terms have bounds we have evaluated and found
# negligible (5.2e-5 coarse representation, 1.5e-8 buffer truncation). Forty
# steps is inherited from the gate configuration and rests on a convergence
# check made at a much smaller one, transporting on the argument that the
# preset ladder grows volume at fixed resolution. That is an argument.
#
# AND IT IS THE LARGEST LEVER WE HAVE FOUND. The first pass of this job
# measured 40 against 80 at this configuration: P(40)/P(80) is 0.9998 at
# k = 0.08, 0.9885 at 0.48, 0.9577 at 1.07 and 0.9369 at the band edge. So 40
# steps is converged to 0.5% only out to k = 0.31, to 1% at 0.45, to 2% at
# 0.63. For comparison the mesh ladder's 2x refinement moved the band edge by
# 3.5% and in the WRONG direction; this moves it 6.3% in the direction that
# closes the gap with the emulator. Steps are also the cheapest lever by a
# distance: exactly linear in wall (23.98 and 23.70 s/step across the two arms)
# and ZERO memory, where refining the coarse mesh is disqualifying at 4096^3.
#
# SO THE QUESTION IS NO LONGER "DOES IT MATTER" BUT "HOW MUCH IS ENOUGH", AND
# THAT NEEDS BRACKETING FROM ABOVE. Quoting "use 80" on the strength of 40-vs-80
# would just move the arbitrary number, which is the mistake that put 40 on
# record in the first place. 80 -> 160 sizes 80's own residual error; 60 and 120
# fill the curve so the answer is a tolerance-versus-cost line rather than a
# single recommended count.
#
# ARMS 40 / 60 / 80 / 120 / 160. NOT 20: that arm failed here, and instructively
# -- 195 particles overflowed their bricks against 170 free arena slots, because
# fewer steps means larger displacements per step and the layout sized for 40 is
# too tight. It raised rather than clamping or dropping, which is the design
# working. Including it would mean one arm carrying a different arena fraction,
# i.e. a second axis moving, which is the trap the buffer derivation exists to
# avoid.
#
# ONE IC GENERATION, symlinked into each arm, so the ladder is matched by
# construction. Arms cannot cross-resume even by accident: the coefficients are
# hashed into the checkpoint fingerprint, so a checkpoint from one count is
# refused by another rather than continued onto a different trajectory.
#
# TIME. Cost is LINEAR in the step count and the rate is MEASURED on this node
# class at this exact configuration: 23.70-24.02 s/step across three arms and
# two jobs.
#
#   arm 1    40 steps   ~16 min      (the current production count)
#   arm 2    60 steps   ~24 min
#   arm 3    80 steps   ~32 min
#   arm 4   120 steps   ~48 min
#   arm 5   160 steps   ~64 min
#   ICs (measured 1.4 min) + 5 cards  ~3 min
#                                     --------
#                                     ~3.2 h central
#
# All five run here rather than importing the 40 and 80 cards from job 1016032.
# Those ARE comparable -- the IC generation is byte-identical across jobs,
# checked by md5 -- but reusing them saves 48 minutes and makes the deliverable
# four points plus two imported ones instead of one curve measured under one
# set of conditions.
#
# WALL 5 h against 3.2 h central, on a measured linear rate.
#
# BUDGET: gg = 0.33 SU/node-hr -> 1.65 SU worst case, ~1.05 SU expected.
#
# OPS (load-bearing on gg): pixi on PATH; JAX_PLATFORMS=cpu (jax 0.10
# hard-raises on cuInit on a GPU-less node); CONDA_OVERRIDE_CUDA (the gpu env's
# activation gate); --generator host (the device IC generator needs cards).

set -u

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export CONDA_OVERRIDE_CUDA=12.0
export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }

RUNS=/scratch/10303/jrc4/inexor_runs
WORK=$RUNS/ksteps-$SLURM_JOB_ID
CARDS=runs/v2
TAG=ksl_$SLURM_JOB_ID
RLZ=scripts/v2_m6_realization.py
W="--tile-workers 16"
LAYOUT="--slack 0.10 --arena-frac 0.01"

mkdir -p "$WORK"
echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
free -g | head -2
df -h "$RUNS" | tail -1
nproc

rc_total=0
finish () {
  echo ""; echo "=== cards ==="
  ls -la "$WORK"/k*/realization_pk.json 2>/dev/null
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
  pixi run -e gpu python $RLZ ics --config cgh64 --workdir "$WORK/ics" \
    --generator host \
  || { echo "FATAL: IC generation failed; no arm can run"; exit 1; }

# Cheapest first, each carded before the next starts, so a wall kill costs one
# arm and banks the rest.
for K in 40 60 80 120 160; do
  A="$WORK/k$K"
  mkdir -p "$A"
  for f in "$WORK"/ics/*; do ln -sfn "$f" "$A/$(basename "$f")"; done
  rm -f "$A/realization_ics.json"

  run_leg "run-k$K" \
    pixi run -e gpu python $RLZ run --config cgh64 --k-steps $K \
      --workdir "$A" $W $LAYOUT --checkpoint-every $K \
      --phase-instrument time \
    || { echo "NOTE: arm K=$K failed or hit the wall; earlier arms stand"; continue; }

  run_leg "card-k$K" \
    pixi run -e gpu python $RLZ card --config cgh64 --k-steps $K \
      --workdir "$A" $W $LAYOUT --heartbeat 60 \
    || echo "NOTE: card for K=$K failed; its checkpoint is at $A/ckpt"

  cp -f "$A/realization_pk.json" "$CARDS/${TAG}_k$K.json" 2>/dev/null || true
done

exit $rc_total
