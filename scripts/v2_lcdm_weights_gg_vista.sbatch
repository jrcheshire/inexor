#!/bin/bash
#SBATCH -J lcdm-weights
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o lcdm-weights-%j.log
# The engine with the LCDM BullFrog weights (D-v2-25) at 512^3 / 256 Mpc/h, scored
# against EE2 beside DISCO-DJ.
#
# ARM: fresh LCDM ICs (growth2 lcdm, the default), 120 steps, coarse match order 3
# (the default), carded with the production estimator. The ICs are left in place:
# the chained DISCO-DJ gh job (`v2_disco_crosscheck_gh_vista.sbatch`, IC_DIR set to
# this job's ics) evolves the same realization, so the two cards difference
# bin-for-bin. At cdev8 the pair agreed to 1.0e-3 (was 13% with EdS weights).
#
# TIME, from this box's measured 24.4-24.5 s/step (1016874, 1018128, 1018341):
#   ICs (measured 0.7-1.4 min) + card                         ~2 min
#   run   120 x 24.5 s                                        ~49 min
#                                                             --------
#                                                             ~51 min
#
# MEMORY: 153-157 GB measured at this config and W=16.
# BUDGET: gg = 0.33 SU/node-hr -> ~0.3 SU expected, 0.66 worst case (2 h).
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
WORK=$RUNS/lcdm-weights-$SLURM_JOB_ID
CARDS=runs/v2
TAG=lcw_$SLURM_JOB_ID
RLZ=scripts/v2_m6_realization.py
K=120
ARGS="--config cgh64 --k-steps $K --coarse-match-order 3 --growth2 lcdm --tile-workers 16 --slack 0.10 --arena-frac 0.01"

git diff --quiet HEAD || { echo "FATAL: the checkout has modified tracked files"; git status -s; exit 1; }
mkdir -p "$WORK"
echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
free -g | head -2
df -h "$RUNS" | tail -1
nproc

leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ($(date +%H:%M:%S)) ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc ($(date +%H:%M:%S)) ---"
  [ $rc -eq 0 ] || { echo "FATAL: leg $name failed (rc=$rc)"; exit $rc; }
}

leg ics pixi run -e gpu python $RLZ ics --config cgh64 --workdir "$WORK/ics" --generator host \
  --growth2 lcdm

A="$WORK/k$K"
mkdir -p "$A"
for f in "$WORK"/ics/*; do ln -sfn "$f" "$A/$(basename "$f")"; done
rm -f "$A/realization_ics.json"

leg run pixi run -e gpu python $RLZ run $ARGS --workdir "$A" --checkpoint-every $K \
  --phase-instrument time
leg card pixi run -e gpu python $RLZ card $ARGS --workdir "$A" --heartbeat 60

cp -f "$A/realization_pk.json" "$CARDS/${TAG}_k$K.json"
echo ""; echo "=== done $(date) ==="; ls -la "$A"/realization_*.json
