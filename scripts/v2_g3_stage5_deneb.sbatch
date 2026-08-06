#!/bin/bash
#SBATCH --job-name=g3-stage5-cdev
#SBATCH --output=%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=54G
#SBATCH --time=24:00:00
#
# G3 Stage 5 -- the mechanism ensemble at cdev.
#
# THE QUESTION. How does the squeezed-bispectrum error depend on the ABSOLUTE
# buffer and on where the small-scale leg sits, and over what region of that
# plane can the gate speak at all?
#
# WHY IT IS ASKED THIS WAY. The pinned gate triangles carry k_short = 1.178
# h/Mpc, and no cdev geometry keeps the arms correlated (r >= 0.5) that far out
# -- 0.47 h/Mpc at an 8 Mpc/h buffer, 0.90 at 16 (g3_decorrelation_record.md).
# Under the ratified conditioning cut that is ZERO gradeable triangles, and R_Q
# cannot be read in the decorrelated regime at all because it saturates and
# stops being monotone in brokenness (g3_ladder_record.md). So the small-scale
# leg is SCANNED and the eligibility boundary is measured, not assumed. The
# buffer ladder is in Mpc/h because the absolute buffer is what sets usable k;
# tile size sets only the price.
#
# WHAT EACH OUTCOME MEANS (fixed before the run, so the readout is not post-hoc):
#   - If a cheap buffer (4-8 Mpc/h) is gradeable at some k_short and lands under
#     15%, A3 is alive at a price worth quoting, and Stage 6 reads the verdict at
#     cgh64 off the same curve.
#   - If only the 16 Mpc/h buffer is ever gradeable, A3 survives only at
#     vol_ratio 8 -- the configurations cheap enough to make it attractive are
#     exactly the ones that decorrelate. That is an A3 RESULT, not a measurement
#     failure, and it is the substance of the Stage 5 design tension.
#   - If nothing is gradeable anywhere, the honest statement is "A3 cannot be
#     graded at cdev at any affordable buffer". The producer prints NOT GRADEABLE
#     and refuses to emit a verdict field; it must never be read as a pass.
#   - Arm D (T=64/b=32) repeats arm B's ABSOLUTE buffer at half the tile size. If
#     their R_Q agree, "absolute buffer sets the physics, tile size sets the
#     price" holds for the seam statistic and not just for the correlation, where
#     it was measured. If they disagree, that rule does not transfer and Stage 6
#     may not use it to pick a geometry.
#
# ARMS (fine cells; 0.25 Mpc/h each at cdev), cost in monolithic evolves:
#   T=128 b=16   4 Mpc/h  P=160  1.95x      T=128 b=64  16 Mpc/h  P=256  8.00x
#   T=128 b=32   8 Mpc/h  P=192  3.38x      T=64  b=32   8 Mpc/h  P=128  8.00x
#   plus the brackets: kill control (b=0) and span check (pure 2LPT).
#
# RESOURCE FIT (no doomed jobs).
#   Memory: the bispectrum band fields dominate, at 2 * n_shell * N^3 * 8 bytes
#   = 2.15 GB per shell at N=512. The producer works one k_short at a time, so
#   n_shell = 7 long + 1 short = 8 -> ~17 GB resident, plus the monolithic mesh
#   (1.07 GB), its complex half-grid (1.08), x+v (0.81) and the two painted
#   fields. That is the same profile deneb job 282 (G3 floors at cdev, 8 shells)
#   completed at 48 G; 54 G here for the extra painted arm, against deneb's ~56
#   GB allocatable. The producer prints measured peak RSS -- read it before
#   scaling the ensemble.
#   Wall: measured cdev tiled walls from job 297 are 442 s at vol 3.375 and
#   1103 s at vol 8.0 (T=128), 1061 s at vol 8.0 (T=64); arm A interpolates to
#   ~260 s; the monolithic evolve is 214 s and the kill control ~150 s. That is
#   ~3200 s of evolve plus the estimator, so ~1.1 h per seed. The pilot is one
#   seed; the ensemble below is 8, so ~9 h. 24 h requested (generous per house
#   rule; deneb is not billed, and a tight wall only forces a resubmit of the
#   expensive monolithic arm).
#   Placement: -c 64 fits deneb (CPUTot=64) and NOT antares (28), so this cannot
#   land on the busy node; no -w, no --exclude.
#
# USAGE.
#   sbatch scripts/v2_g3_stage5_deneb.sbatch pilot      # seed 0, read it first
#   sbatch scripts/v2_g3_stage5_deneb.sbatch ensemble   # seeds 1-8
#
# OUTPUT (runs/ is gitignored -- pull explicitly):
#   runs/v2/g3_stage5_cdev.json           (pilot, seed 0)
#   runs/v2/g3_stage5_cdev_seed{1..8}.json (ensemble)

set -euo pipefail

MODE="${1:-pilot}"

# XLA-CPU spins a MACHINE-sized spin-waiting thread pool per process, so at -c 32
# the cgroup strangles a pool sized for 64. See v2_g3_buffer_cost_deneb.sbatch:37.
export OMP_NUM_THREADS=64

# deneb batch shells have no pixi on PATH (job 272 died on this in under a
# second); fail loudly here rather than mid-run.
export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null 2>&1 || { echo "FATAL: pixi not on PATH"; exit 1; }

cd "$HOME/src/inexor"
echo "=== host $(hostname) ==="
echo "=== commit $(git rev-parse --short HEAD) ==="
echo "=== mode ${MODE} ==="
echo "=== free -g ==="; free -g
date

if [ "$MODE" = "pilot" ]; then
    pixi run python scripts/v2_g3_stage5.py --config cdev --seed 0
else
    # NO set -e INSIDE THE LOOP, deliberately. Each seed writes its own card and
    # the seeds are independent, so one failure must not discard eight hours of
    # measured work (the M1 job 11 lesson: independent legs, cheapest first).
    # Resumable by card existence, so a requeue picks up where it stopped.
    set +e
    for SEED in 1 2 3 4 5 6 7 8; do
        CARD="runs/v2/g3_stage5_cdev_seed${SEED}.json"
        if [ -f "$CARD" ]; then
            echo "=== seed ${SEED}: ${CARD} exists, skipping ==="
            continue
        fi
        echo "=== seed ${SEED} starting $(date) ==="
        pixi run python scripts/v2_g3_stage5.py --config cdev --seed "$SEED" \
            --out-suffix "_seed${SEED}"
        echo "=== seed ${SEED} exit $? at $(date) ==="
    done
    set -e
fi

echo "=== done ==="
date
