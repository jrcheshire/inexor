#!/bin/bash
#SBATCH --job-name=g3-hybrid-evolve
#SBATCH --output=%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#
# RUNG H3 AT THE PRODUCTION BOX. Two arms at cdev, T=128 fine cells:
#   b=16 fine = 4.0 Mpc/h = 4 r_s   (the cheap arm, 1.95x monolithic)
#   b=32 fine = 8.0 Mpc/h = 8 r_s   (3.38x, nearest A3's own 3.2x budget)
#
# WHY BOTH. H2 at cdev8 put the knee at ~4 r_s (rel rms 4.09e-2 at 2 r_s,
# 4.55e-3 at 4 r_s), so b=16 is the interesting cheap point and b=32 is the
# one whose cost matches the route's premise. If the cheap arm works, the cost
# case is made at 1.95x rather than at the 8x the failing scheme needed to get
# anywhere.
#
# PREDICTIONS, fixed before the run:
#
#  1. hybrid auto amplitude sqrt(P/P_mono) ~ 1 at EVERY shell, matching cdev8's
#     1.000 -> 0.989 over k = 1..10 k_f. A deficit at low k refutes H1/H3's
#     attribution at the production box.
#  2. hybrid correlation r = 1.000 at every shell. This is the one that matters
#     most: Stage 5's whole eligibility problem was the tiled arm decorrelating,
#     and at cdev8 the hybrid simply does not.
#  3. TILED arm reproduces the tracked card. runs/v2/g3_stage5_cdev.json has
#     shell_A at the box fundamental = 0.441 (T=128 b=16) and 0.424 (b=32).
#     The tiled arm here must land on those to a few percent -- it is the same
#     configuration and the same seed, so a mismatch means this harness is not
#     measuring what Stage 5 measured and NOTHING else in the run can be read.
#  4. hybrid max|rho_auto| well under the 0.15 bar (cdev8 gave 0.0023). Still
#     PRELIMINARY: one seed, realization-matched, and D-v2-12 clause 4 ratified
#     an ensemble comparison as the reading.
#
# WHAT THIS RUN CANNOT SHOW. The hybrid is evolved in LOCKSTEP -- every
# particle held, coarse long-range solved from true positions each step. That
# is the PHYSICS question. The memory saving that motivated tiling is a
# separate question and this run does not touch it.
#
# RESOURCES. Two independent invocations, each rebuilding ICs and the
# monolithic reference (~214 s at cdev, so ~7 min of duplicated setup -- cheap
# against not adding multi-arm plumbing right before a run). Per arm the
# tiled evolve ran 238 s (b=16) and 440 s (b=32) in job 362, and cdev8 put the
# hybrid within 4% of the tiled wall. Memory is two 512^3 f64 density fields
# plus the evolve; job 362's 58 GB peak was four arms plus a full bispectrum
# scan and does not apply. 48 G against deneb's 56000M cap, which is the real
# ceiling (see g3_stage5_record.md sec. 9).

set -euo pipefail

export OMP_NUM_THREADS=64
export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null 2>&1 || { echo "FATAL: pixi not on PATH"; exit 1; }

cd "$HOME/src/inexor"
echo "=== host $(hostname) ==="
echo "=== commit $(git rev-parse --short HEAD) ==="
echo "=== free -g ==="; free -g
date

for BUF in 16 32; do
    echo "=== T=128 b=${BUF} starting $(date) ==="
    pixi run python scripts/v2_g3_hybrid_evolve.py \
        --config cdev --tile 128 --buf "${BUF}" --nbins 12
done

echo "=== done ==="
date
