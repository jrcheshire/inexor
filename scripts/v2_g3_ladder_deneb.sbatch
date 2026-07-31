#!/bin/bash
#SBATCH --job-name=g3-ladder-cdev8
#SBATCH --output=%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#
# G3 Stage 4: the identity ladder + the two bracket controls, at cdev8.
#
# GEOMETRY. T=64 fine cells with n_fine=256 gives 64 tiles and, at the pivot
# b=16, a padded box P=96 -- a padded fraction P/n_fine = 0.375, i.e. an actual
# tile. That matters: at smoke every non-degenerate configuration had the tile
# covering 50-75% of the box side, the b-scan came out non-monotone, and the
# bracket ordering was meaningless. The tiling error is governed by P/L (Stage
# 1's collapse), so the lever only exists once the tile is small against the box.
#
# RESOURCE FIT (no doomed jobs). cdev8 is n_fine=256: an f64 mesh is 0.13 GB and
# the largest tile solve here is buffer_to_box at P=256, i.e. the same size as
# the monolithic mesh. Peak is a few GB against 24 GB requested and deneb's
# ~56 GB allocatable. Wall, from the measured 13.3 s/mono-evolve at cdev8 and
# the exact cost law (a tiled run costs n_tiles*(P/n_fine)^3 = (P/T)^3 monolithic
# evolves): buffer_to_box 64x = 851 s, b-scan {0,8,16,32,64} = 41x = 545 s,
# brackets 4.4x = 59 s, plus ~5x for build/A0/frame_zero/A1/residual_zero.
# Total ~25 min. 3 h requested (generous per house rule; billed for actual use).
#
# WHY buffer_to_box IS NOT RUN AT cdev. Its cost at b_max is exactly
# (n_fine/T)^3 monolithic evolves: 64x here, but 512x at cdev with T=64, which
# is 30 h. What it certifies -- that the buffer family terminates at the exact
# monolithic answer -- is a property of the tiling MACHINERY, not of the gate's
# T, so cdev8 is where it is established.

set -euo pipefail

# -c 64 = the whole node as Slurm counts it (CPUTot=64). XLA-CPU spins a
# MACHINE-sized spin-waiting pool per process, so a smaller cgroup strangles it.
export OMP_NUM_THREADS=64

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null 2>&1 || { echo "FATAL: pixi not on PATH"; exit 1; }

cd "$HOME/src/inexor"
echo "=== host $(hostname) ==="
echo "=== commit $(git rev-parse --short HEAD) ==="
free -g
date

pixi run python scripts/v2_g3_ladder.py \
    --config cdev8 \
    --tile 64 --buf 16 \
    --b-scan 0 8 16 32 64 \
    --long-mults 2 3 4 6 \
    --k-short-mult 24 \
    --out-suffix _stage4

echo "=== done ==="
date
