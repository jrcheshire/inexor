#!/bin/bash
#SBATCH -J zinit-50
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o zinit-50-%j.log
# Does the late 2LPT start carry part of the high-k deficit?
#
# WHERE THIS SITS. Every run on record starts at a = 0.1 (z = 9) with 2LPT.
# DISCO-DJ II (arXiv:2510.05206, App. A.2, Fig. 17) finds a late 2LPT start
# (z_ini = 10) leaves "a noticeable imprint" from the missing third-order terms
# even at 100 steps, where 2LPT from z >= 50 and 3LPT from any start agree. No
# ladder here has moved the start. With steps, both meshes, spacing and box all
# converged, B/B_EE2 is 0.830 at k = 1.56 (corrected match, 120 steps, 1018341).
#
# ONE ARM: z_ini = 50 (a = 1/51), 240 steps, same seed / n_part / bins and
# --coarse-match-order 3. The white noise and the z = 0 linear field are the
# same, so this is the SAME realization as mo3_1018341_k120 and the ratio reads
# below the seed scatter; the low-k bins must read ~1.000, which checks that.
#
# WHY 240 STEPS. The grid is log-spaced in a, and z = 50 spans ln 51 = 3.93
# e-folds against 2.30 from z = 9. The converged z = 9 run has 120 / 2.30 = 52
# steps per e-fold; 240 / 3.93 = 61 is denser everywhere, so the late-time
# integration is at least as resolved. At 160 (41 per e-fold) it would be
# coarser, which would confound the start with the integrator.
#
# TIME, from this config's measured 23.45-24.5 s/step (1018341):
#   ICs (measured 1.4 min) + card                   ~2 min
#   240 x 24.5 s                                    ~98 min
#                                                   --------
#                                                   ~1.7 h
#
# MEMORY: 152-154 GB measured at this config and W=16; the start does not move it.
#
# BUDGET: gg = 0.33 SU/node-hr -> ~0.55 SU expected, 1.0 worst case (3 h).
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
WORK=$RUNS/zinit-$SLURM_JOB_ID
CARDS=runs/v2
TAG=zi50_$SLURM_JOB_ID
RLZ=scripts/v2_m6_realization.py
W="--tile-workers 16"
LAYOUT="--slack 0.10 --arena-frac 0.01"
KSTEPS=240
GEO="--config cgh64 --coarse-match-order 3 --a-init 0.0196078431372549"

mkdir -p "$WORK"
echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
free -g | head -2
df -h "$RUNS" | tail -1
nproc

rc_total=0
finish () {
  echo ""; echo "=== cards ==="
  ls -la "$WORK"/realization_pk.json 2>/dev/null
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
  pixi run -e gpu python $RLZ ics $GEO --workdir "$WORK" --generator host \
  || { rc=$?; echo "FATAL (rc=$rc): IC generation failed"; exit 1; }

run_leg run \
  pixi run -e gpu python $RLZ run $GEO --k-steps $KSTEPS --workdir "$WORK" $W $LAYOUT \
    --checkpoint-every $KSTEPS --phase-instrument time \
  || { rc=$?; echo "FATAL (rc=$rc): the run failed or hit the wall"; exit 1; }

run_leg card \
  pixi run -e gpu python $RLZ card $GEO --k-steps $KSTEPS --workdir "$WORK" $W $LAYOUT \
    --heartbeat 60 \
  || echo "NOTE: the card failed; the checkpoint is at $WORK/ckpt"

cp -f "$WORK/realization_pk.json" "$CARDS/${TAG}_k$KSTEPS.json" 2>/dev/null || true

exit $rc_total
