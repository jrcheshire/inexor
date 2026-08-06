#!/bin/bash
#SBATCH --job-name=g3-buffer-criterion
#SBATCH --output=%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#
# RUNG H2 -- the buffer criterion at the PRODUCTION box.
#
# THE QUESTION. With the long range moved to a global coarse solve, does the
# tile's buffer requirement follow the split scale r_s, or does it still grow
# with the box? The failing arm needed a buffer that grew with the box and
# therefore had no cost case; if this one does too, neither does it.
#
# WHY THIS CONFIG AND NOT ANOTHER. cdev8 and cdev share the fine cell
# (0.25 Mpc/h) and the coarse cell (1.0 Mpc/h), so at alpha = 1 they share r_s
# EXACTLY. Running the identical T = 64 buffer ladder at both varies the box
# and nothing else: same tile geometry, same padded sizes, same kernel, 8x the
# volume. That is the only comparison that can answer a box-independence
# question, and it is why the tile is held at 64 fine cells here even though
# cdev's own arms use 128 -- matching the axis matters more than matching the
# production geometry, which H3/H4 cover.
#
# ALREADY MEASURED at cdev8 (local, runs/v2/g3_buffer_criterion_cdev8_T64.json),
# rel rms of the tiled short force against the UNTILED short force:
#   b/r_s   2.0     4.0     7.0     8.0    12.0    16.0
#   rel  4.09e-2 4.55e-3 3.34e-3 3.05e-3 2.16e-3 1.49e-3
# A 9x drop between 2 and 4 r_s, then a slow tail. PREDICTION for cdev: the
# same numbers row by row, within a few percent of each other. A cdev curve
# sitting systematically above cdev8's refutes box-independence and the cost
# case with it.
#
# RESOURCES. The reference is one force_global solve on the 512^3 fine mesh;
# each ladder rung is 512 tiles (T=64 at n_fine=512) of padded size <= 192^3.
# Peak is dominated by the reference solve, comfortably inside 40 G. cdev8's
# ladder took 27 s of solve wall at 64 tiles; cdev has 8x the tiles and ~8x the
# particles, so expect tens of minutes, not hours. -c 32 rather than 64 so this
# does not need the whole node.

set -euo pipefail

export OMP_NUM_THREADS=32
export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null 2>&1 || { echo "FATAL: pixi not on PATH"; exit 1; }

cd "$HOME/src/inexor"
echo "=== host $(hostname) ==="
echo "=== commit $(git rev-parse --short HEAD) ==="
echo "=== free -g ==="; free -g
date

pixi run python scripts/v2_g3_buffer_criterion.py --config cdev --tile 64

echo "=== done ==="
date
