#!/bin/bash
#SBATCH --job-name=g3-floors-cdev
#SBATCH --output=%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#
# G3 Stage 3 floors at cdev -- the config the estimand is pinned at.
#
# WHY cdev AND NOT cdev8. At cdev8 (L=64) every triangle leg sits past
# k_nl ~ 0.2-0.3, so the box has no linear long mode to squeeze against; the
# Gaussian-sigma diagnostic came back non-discriminating there (measured/analytic
# 14-23x for squeezed AND 19.5x for equilateral, where the plan expected
# agreement on equilateral). At cdev (L=128) the m=1-3 long legs are
# 0.049-0.147 h/Mpc, genuinely linear.
#
# TRIANGLE SET. x = m * P / n_fine; at the Stage 5 geometry T=64/b=16 the padded
# tile is P=96 and n_fine=512, so the lever is 5.33 and x=1 falls at m=5.33.
# m = 1..7 gives five points below the threshold and two above, which is the
# ladder the plan specifies. k_short = 24 k_f = 1.18 h/Mpc, chosen to match
# cdev8's k_short in PHYSICAL units (12 k_f there) so the two configs are
# compared at the same short scale rather than the same multiple of a different
# fundamental.
#
# RESOURCE FIT (no doomed jobs). cdev is n_fine=512: an f64 mesh is 1.07 GB and
# the complex half-grid 1.08 GB, with x+v state 0.81 GB. The cgh64 precedent
# measured a 125 GB working set at n_fine=1024, and cdev is 1/8 that mesh volume
# -> ~15-16 GB expected, against 48 GB requested and deneb's ~56 GB allocatable.
# Over 3x headroom. Wall: cdev8 measured 13.3 s/evolve, cdev is 8x particles and
# 8x cells -> ~2 min/evolve; sections D + A + BC(24 seeds) + F = 27 evolves ->
# ~55-70 min. 4 h requested (generous per house rule; billed for actual use).
#
# Section E (the dynamical null) is deliberately NOT re-run here. Its cdev8
# result was that there is no chaotic amplification at all (rms R_Q ~ eps^0.44,
# no saturation across four decades), and cdev has MORE linear modes, not fewer,
# so the conclusion cannot flip in the direction that would matter. Re-running
# it would cost another 48 evolves (~1.6 h) to re-confirm a negative.

set -euo pipefail

# deneb batch shells have no pixi on PATH (job 272 died on this in under a
# second); the prerequisite guard below fails loudly rather than mid-run.
export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null 2>&1 || { echo "FATAL: pixi not on PATH"; exit 1; }

cd "$HOME/src/inexor"
echo "=== host $(hostname) ==="
echo "=== commit $(git rev-parse --short HEAD) ==="
echo "=== free -g ==="; free -g
date

pixi run python scripts/v2_g3_floors.py \
    --config cdev \
    --sections DABCF \
    --nseed 24 \
    --long-mults 1 2 3 4 5 6 7 \
    --k-short-mult 24 \
    --out-suffix _pin

echo "=== done ==="
date
