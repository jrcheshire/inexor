#!/bin/bash
#SBATCH -J inexor-v2-g6
#SBATCH -c 64
#SBATCH --mem=54G
#SBATCH -t 24:00:00
#SBATCH -o %x-%j.log
#
# v2 G6: the low-k architectural floor assessment D-v2-10 deliberately left as
# its own piece of work -- IS THE SPLIT ERROR REALIZATION-STABLE, i.e.
# correctable as a measured transfer? Read the module docstring of
# v2_g6_split_stability.py for what the three measured properties are.
#
# HOME: deneb, deliberately, and this is NOT the D-v2-10 violation it might look
# like. That framing binds PERFORMANCE/Pareto points to the config-table TACC
# homes because a 6 GB card must not shape architecture decisions. This job
# measures a dimensionless ACCURACY ratio (tiled vs mono at identical ICs) with
# no capacity leg at all -- deneb is exactly the correctness ground D-v2-10
# assigns it, it costs no SUs, and it does not contend with the Vista queue.
#
# -c 64 = the whole node as Slurm counts it (CPUTot=64). At -c 32 the XLA-CPU
# Eigen pool still sizes itself from `nproc` = 64 and SPIN-waits, so the cgroup
# runs ~2:1 oversubscribed -- job 40 paid ~5x wall for exactly this. --mem is
# REQUIRED (partition default exceeds deneb's 56 GB -> pends forever, V1 job 9).
#
# NO `set -e`, deliberately (M1 job 11 lesson: independent legs). A seed that
# crashes must not take the other 15 with it, and the aggregator works with
# however many landed. Resumable by construction: a seed whose result JSON
# already exists is skipped, so a resubmit picks up where a kill left off.
#
# WALL: per seed = 2 config gens + validation + floors + one tile force arm +
# 4 evolve arms (mono, T128/b32, T256/b64, T128/b32 shifted) at 20 steps.
# Job 39's C-dev evolve rates (9.9 s/step mono, 32.2 s/step tiled) put that at
# ~40 min/seed = ~11 h for 16 -- and those rates were paid with the -c 32 tax
# above, so this should come in under. 24 h is deliberately generous: TACC-style
# billing does not apply here and a tight wall only forces a resubmit.
#
# NB the two tiled arms cost the SAME wall despite different tiles: total padded
# volume is n_fine^3 (1 + 2b/T)^3, and T128/b32 and T256/b64 share (1 + 2b/T).
# They differ in P (192 vs 384), which is the point -- P is the error's scaling
# variable, so this pair tests whether the transfer's SHAPE is P-invariant.
#
# Submit from ~/src/inexor on deneb:  sbatch scripts/v2_g6_deneb.sbatch

cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export G5_CPU_THREADS=64

SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"

echo "=== env ==="
git log --oneline -1
free -g | head -2
nproc

for S in $SEEDS; do
  OUT="runs/v2/g5_results_cdev_seed${S}.json"
  if [ -f "$OUT" ]; then
    echo "=== seed $S: $OUT exists, skipping ==="
    continue
  fi
  # Seed 0 carries the two degenerate-limit arms (--shift-identity): a tiled arm
  # shifted by a FULL tile period, which maps the lattice onto itself, and a
  # shifted mono arm, which is exactly equivariant. Both must reproduce their
  # unshifted twin or no origin number means anything -- the aggregator hard-
  # fails if they are missing, so this flag is load-bearing, not optional.
  EXTRA=""
  if [ "$S" = "0" ]; then EXTRA="--shift-identity"; fi
  echo "=== seed $S ==="
  pixi run python scripts/v2_g5_two_level_force.py --config cdev \
    --families gauss --skip-coarse --skip-floor-arms --skip-1024 --skip-capacity \
    --pivot-only --evolve-tiles 128:32,256:64 --shift-arm $EXTRA \
    --seed "$S" --out-suffix "_seed${S}"
done

echo "=== aggregate ==="
pixi run python scripts/v2_g6_split_stability.py --config cdev

echo "=== DONE ==="
