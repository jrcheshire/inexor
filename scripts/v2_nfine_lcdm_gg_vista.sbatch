#!/bin/bash
#SBATCH -J nfine-lcdm
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o nfine-lcdm-%j.log
# Is the high-k droop against EE2 force resolution? The 0.125 Mpc/h fine-cell arm
# with the LCDM weights, on the SAME ICs as the 0.25 Mpc/h card.
#
# WHERE THIS SITS. With the LCDM weights (D-v2-25) the 512^3 / 256 Mpc/h realization
# (lcw_1023904_k120) reads B/B_EE2 = 0.966 at k = 1.46 and 0.959 at 1.56. Steps are
# converged (120 -> 240 moves P <= 8.4e-4). The fine-mesh ladder (1018489) raised
# the band edge +2.4% going 0.25 -> 0.125 Mpc/h, but ran the EdS weights.
#
# ONE ARM: nf2048 (0.125 Mpc/h fine cell), 120 steps, coarse match order 3, growth2
# lcdm. The 0.25 Mpc/h arm is lcw_1023904_k120, not re-run: this job symlinks
# 1023904's ICs, so the two cards difference bin-for-bin with the realization
# cancelled. The run refuses ICs of the wrong growth2.
#
# TIME, measured rates:
#   ICs                    reused from 1023904                  0
#   run   120 x 86.6 s     nf2048 rate (1015840)                ~2.9 h
#   card                   0.2 min (1023904)                    ~1 min
#                                                               --------
#                                                               ~2.9 h
#
# MEMORY: nf2048 measured 237.9 GB of the node at W=16 (1015840); ran again in
# 1018489. Tight but it fits; growth2 does not move it.
# BUDGET: gg = 0.33 SU/node-hr -> ~1.0 SU expected, 1.32 max (4 h).
#
# OPS (load-bearing on gg): pixi on PATH; JAX_PLATFORMS=cpu (jax 0.10
# hard-raises on cuInit on a GPU-less node); CONDA_OVERRIDE_CUDA.

set -u

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export CONDA_OVERRIDE_CUDA=12.0
export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { rc=$?; echo "FATAL (rc=$rc): pixi not on PATH"; exit 1; }

RUNS=/scratch/10303/jrc4/inexor_runs
IC_DIR=${IC_DIR:-$RUNS/lcdm-weights-1023904/ics}
WORK=$RUNS/nfine-lcdm-$SLURM_JOB_ID
CARDS=runs/v2
TAG=nfl_$SLURM_JOB_ID
RLZ=scripts/v2_m6_realization.py
K=120
NF=2048
ARGS="--config cgh64 --n-fine $NF --k-steps $K --coarse-match-order 3 --growth2 lcdm --tile-workers 16 --slack 0.10 --arena-frac 0.01"

git diff --quiet HEAD || { echo "FATAL: the checkout has modified tracked files"; git status -s; exit 1; }
[ -d "$IC_DIR" ] || { echo "FATAL: IC_DIR $IC_DIR is missing; the arm must share 1023904's realization"; exit 1; }
mkdir -p "$WORK"
echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
echo "IC_DIR=$IC_DIR"
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

A="$WORK/nf$NF"
mkdir -p "$A"
for f in "$IC_DIR"/*; do ln -sfn "$f" "$A/$(basename "$f")"; done
rm -f "$A/realization_ics.json"

leg run pixi run -e gpu python $RLZ run $ARGS --workdir "$A" --checkpoint-every $K \
  --phase-instrument time
leg card pixi run -e gpu python $RLZ card $ARGS --workdir "$A" --heartbeat 60

cp -f "$A/realization_pk.json" "$CARDS/${TAG}_nf$NF.json"
echo ""; echo "=== done $(date) ==="; ls -la "$A"/realization_*.json
