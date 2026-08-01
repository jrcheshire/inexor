#!/bin/bash
#SBATCH --job-name=g3-decorr-cdev
#SBATCH --output=%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#
# Does the T=128 advantage survive the box, or was it box fraction?
#
# THE QUESTION. The cdev8 map (deneb 296) found that usable-k does NOT collapse
# in cost: at a matched 3.38x, k(r=0.5) = 0.520 / 0.517 / 0.951 for T = 32 / 64 /
# 128. Read at face value, T=128/b=32 buys ~2x the trustworthy k at the same
# price as the plan's Stage 5 geometry T=64/b=16. But Stage 1 established that
# the tile force error is governed by the FRACTION of the box the padded tile
# spans, and at cdev8 the winning config has P/n_fine = 0.75 -- its padded tile
# covers three quarters of the box. The advantage and the confound are the same
# number, so the cdev8 ranking cannot pin a Stage 5 geometry.
#
# WHY cdev SEPARATES THEM. cdev and cdev8 share the fine cell exactly (0.25
# Mpc/h; L/n_fine = 128/512 = 64/256), so the SAME (T, b) in fine cells is the
# same tile in Mpc/h -- T=128 is 32 Mpc/h and b=32 is 8 Mpc/h in both boxes.
# What changes is the box: P/n_fine halves at every grid point (T=128/b=32 goes
# 0.750 -> 0.375). Absolute geometry held, box fraction halved.
#
# WHAT EACH OUTCOME MEANS (fixed before the run, so the readout is not post-hoc):
#   - If the three 3.38x configs CONVERGE at cdev, cost is the only variable and
#     the cdev8 T-advantage was a box-fraction artifact. Stage 5 is free to take
#     the cheapest geometry at its target usable-k.
#   - If T=128 still leads by ~2x at matched cost, tile size genuinely matters at
#     fixed absolute buffer and Stage 5 should take T=128/b=32.
#   - Either way the cdev-vs-cdev8 pair at matched (T, b) is a two-point test of
#     the P/L law itself: halving P/L at fixed absolute tile should degrade every
#     config if P/L governs, and leave them alone if the absolute buffer does.
# The organizing variable to plot against is the ABSOLUTE buffer in Mpc/h (~0.47
# at b=2 ... 1.59 at b=16 in the Stage 1 units), matching Stage 1's containment
# scale.
#
# GRID. Unchanged from cdev8 -- the script's GRID is in fine cells and needs no
# edit. One point is GAINED: T=128/b=64 gives P = n_fine = 256 at cdev8, i.e. the
# tile IS the box and r == 1 by construction (it is flagged degenerate and
# dropped from the cost panel); at cdev the same config is P/n_fine = 0.5 and a
# real measurement. So 10 usable configs here against 9 there.
#
# RESOURCE FIT (no doomed jobs).
#   Memory: peak is set by the MONOLITHIC arm at n_fine=512 -- an f64 mesh is
#   1.07 GB, the complex half-grid 1.08 GB, x+v state 0.81 GB. That is the same
#   arm deneb job 282 (G3 floors at cdev) ran to completion at --mem=48G. The
#   largest padded solve in this grid is P=256, an eighth of the monolithic mesh
#   volume, so the tiled legs do not move the peak. 48 G against deneb's ~56 GB
#   allocatable.
#   Wall: the cdev8 run took 1639 s total at 13.3 s/evolve. Split it into physics
#   (92.03 vol units x 13.3 = 1224 s, which scales with the 213.9 s cdev evolve =
#   16.1x) and per-tile overhead (415 s, which scales with tile count = 8x):
#   19682 + 3319 = ~23000 s = 6.4 h. Taking the whole thing at 16.1x bounds it at
#   7.3 h. 12 h requested (generous per house rule; deneb is not billed).
#   Placement: -c 64 fits deneb (CPUTot=64) and NOT antares (28), so this cannot
#   land on the busy node; no -w, no --exclude.

set -euo pipefail

# XLA-CPU spins a MACHINE-sized spin-waiting thread pool per process, so at -c 32
# the cgroup strangles a pool sized for 64 and the job spends its wall in
# contention. See v2_g3_buffer_cost_deneb.sbatch:37.
export OMP_NUM_THREADS=64

# deneb batch shells have no pixi on PATH (job 272 died on this in under a
# second); fail loudly here rather than mid-run.
export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null 2>&1 || { echo "FATAL: pixi not on PATH"; exit 1; }

cd "$HOME/src/inexor"
echo "=== host $(hostname) ==="
echo "=== commit $(git rev-parse --short HEAD) ==="
echo "=== free -g ==="; free -g
date

pixi run python scripts/v2_g3_decorrelation_map.py --config cdev --seed 0

echo "=== done ==="
date
