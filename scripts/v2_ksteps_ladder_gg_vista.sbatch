#!/bin/bash
#SBATCH -J ksteps-ladder
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
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
# AND IT IS NOT OBVIOUSLY A SMALL EFFECT. At the 64 Mpc/h dev configuration,
# halving 20 steps to 10 cost power monotonically in k -- 6% at k = 0.5, 15% at
# k = 1.06, 25% at half Nyquist -- the same sign and shape as the force-
# resolution deficit. That is a different box and a much coarser step count, so
# it does not predict 40 versus 80 here; it says the knob is live in exactly
# the band in question, and that attributing the deficit to the mesh alone
# without checking this one would be premature.
#
# ARMS. 20 / 40 / 80 steps at the preset's own mesh, so the ONLY thing varying
# is the number of steps. Every count ends at a = 1 and the coarse mesh is
# untouched, so the three cards share their epoch and their k bins and
# difference with no interpolation -- verified end to end at the dev
# configuration before this was written.
#
# ONE IC GENERATION, symlinked into each arm, so the ladder is matched by
# construction. Arms cannot cross-resume even by accident: the coefficients are
# hashed into the checkpoint fingerprint, so a checkpoint from one count is
# refused by another rather than continued onto a different trajectory.
#
# TIME. Cost is LINEAR in the step count -- measured, 10.52 vs 10.21 s/step
# across two dev-configuration arms -- so unlike the mesh ladder there is no
# scaling uncertainty here. The per-step rate is MEASURED ON THIS NODE CLASS by
# the mesh ladder that ran first: 24.02 s/step at cgh64 with this exact mesh
# and worker count. That supersedes the 52 s/step figure carried in the
# records, which is 2.2x slow for this node and was the basis of the first
# draft of this script.
#
#   arm 1   20 steps    ~8 min         at the measured 24.02 s/step
#   arm 2   40 steps    ~16 min        (also the production reference)
#   arm 3   80 steps    ~32 min
#   ICs (host generator) + 3 cards     ~5 min
#                                      --------
#                                      ~1 h central
#
# WALL 2 h, double the central. The rate is measured on this node class rather
# than inherited, so the remaining uncertainty is node scatter, and the arms
# are cheap enough that a wide margin costs nothing. If the rate is worse than
# 2x anyway, arms 1 and 2 are banked and arm 2 IS the production reference,
# which is the point of running cheapest first.
#
# BUDGET: gg = 0.33 SU/node-hr -> 0.66 SU worst case, ~0.35 SU expected.
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
for K in 20 40 80; do
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
