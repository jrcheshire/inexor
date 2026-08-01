#!/bin/bash
#SBATCH --job-name=g3-decorr-cdev8
#SBATCH --output=%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#
# Where does a tiled arm still share phases with the monolithic one?
#
# Stage 4 left one question open that a single number should not settle: at
# cdev8/T=64/b=16 the correlation at k_short = 1.18 h/Mpc is r = 0.028, so R_Q
# there is the ratio of two fields with no phase relation. Stage 5 must either
# bring k_short down to where r is appreciable or pay for a less aggressive
# tiling, and those give different A3 compute factors.
#
# THE GRID IS COST-CONTROLLED. A tiled run costs exactly (P/T)^3 monolithic
# evolves, and three (T, b) pairs share each cost:
#   1.95x: T=64/b=8, T=128/b=16
#   3.4x : T=32/b=8, T=64/b=16, T=128/b=32
#   8x   : T=32/b=16, T=64/b=32, T=128/b=64
#   27x  : T=32/b=32, T=64/b=64
# So this answers what a one-dimensional b-scan cannot: AT FIXED COST, does tile
# size matter? Collapse in vol_ratio means geometry is free to choose; separation
# means there is a preferred T at every price.
#
# RESOURCE FIT. cdev8, n_fine=256, f64 mesh 0.13 GB; the largest solve is
# P=256, the size of the monolithic mesh. Peak a few GB against 24 GB requested
# and deneb's ~56 GB allocatable. Wall = sum of vol_ratios = 92 monolithic
# evolves at the measured 13.3 s each = ~20 min, plus build. 2 h requested.

set -euo pipefail
export OMP_NUM_THREADS=64
export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null 2>&1 || { echo "FATAL: pixi not on PATH"; exit 1; }

cd "$HOME/src/inexor"
echo "=== host $(hostname) ==="; echo "=== commit $(git rev-parse --short HEAD) ==="
free -g; date

pixi run python scripts/v2_g3_decorrelation_map.py --config cdev8

echo "=== done ==="; date
