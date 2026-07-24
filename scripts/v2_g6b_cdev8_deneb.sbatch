#!/bin/bash
#SBATCH -J inexor-v2-g6b
#SBATCH -c 64
#SBATCH --mem=54G
#SBATCH -t 06:00:00
#SBATCH -o %x-%j.log
#
# v2 G6b: the calibration-transport test (JC, 2026-07-24). Does the split-error
# transfer T-bar(k) measured on a SMALL box match the big-box transfer? If it
# does, the low-k correction is calibrated once, cheaply, and applied to every
# production box -- the memory-expensive monolithic reference is needed ONLY at
# the (small) calibration box. If it drifts with box size, calibration inherits
# the production box's cost. Read v2_g6b_calib_transport.py's docstring.
#
# THIS JOB is the small rung: the cdev8 ensemble (1/8 the volume of cdev, SAME
# fine cell 0.25 / coarse mesh / gate band -- a pure box-transport comparison at
# fixed resolution). The cdev rung already exists (job 188, 16 seeds); cgh64 (8x
# cdev) is an optional third rung on Vista, sequenced only if this cheap test
# passes.
#
# COST: cdev8 is 256^3 fine / 128^3 particles = 1/8 the cells and particles of
# cdev, so mono evolve is ~1-2 s/step (cdev's was 9.9). ~3-4 min/seed x 16
# ~ 1 h. -t 6 h is deliberately generous (no billing here; a tight wall only
# forces a resubmit).
#
# -c 64 / --mem: same reasons as g6 (XLA-CPU spin pool sizes from nproc=64;
# partition default exceeds deneb's 56 GB). NO set -e (independent seeds;
# resumable -- an existing result JSON is skipped).
#
# Submit from ~/src/inexor on deneb:  sbatch scripts/v2_g6b_cdev8_deneb.sbatch

cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export G5_CPU_THREADS=64

SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"

echo "=== env ==="
git log --oneline -1
free -g | head -2

for S in $SEEDS; do
  OUT="runs/v2/g5_results_cdev8_seed${S}.json"
  if [ -f "$OUT" ]; then
    echo "=== seed $S: $OUT exists, skipping ==="
    continue
  fi
  # Seed 0 carries the two degenerate-limit arms (--shift-identity); the
  # aggregator hard-fails without them.
  EXTRA=""
  if [ "$S" = "0" ]; then EXTRA="--shift-identity"; fi
  echo "=== seed $S ==="
  # Tiles scaled to the small box: pivot 64:16 (P=96) + 128:32 (P=192). Their P
  # values differ from cdev's (192/384), which re-confirms P-independence for
  # free at a second box.
  pixi run python scripts/v2_g5_two_level_force.py --config cdev8 \
    --families gauss --skip-coarse --skip-floor-arms --skip-1024 --skip-capacity \
    --pivot-only --evolve-tiles 64:16,128:32 --shift-arm $EXTRA \
    --seed "$S" --out-suffix "_seed${S}"
done

echo "=== aggregate cdev8 ==="
pixi run python scripts/v2_g6_split_stability.py --config cdev8 \
  --out-prefix g6_split_stability_cdev8

echo "=== re-aggregate cdev under the config-suffixed name (per-seed JSONs already on disk) ==="
pixi run python scripts/v2_g6_split_stability.py --config cdev \
  --out-prefix g6_split_stability_cdev

echo "=== transport: cdev8 -> cdev ==="
pixi run python scripts/v2_g6b_calib_transport.py --small cdev8 --big cdev

echo "=== DONE ==="
