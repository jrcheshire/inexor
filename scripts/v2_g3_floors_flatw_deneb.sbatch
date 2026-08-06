#!/bin/bash
#SBATCH --job-name=g3-floors-flatw
#SBATCH --output=%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#
# The equilateral control's floor B at cdev, with a window that reaches it.
#
# THE QUESTION. What is the resolving power of the equilateral control at cdev?
#
# WHY IT IS OWED NOW. Stage 5 reads that control ("does tiling hurt equilateral
# as much as squeezed?"), and the number does not exist. The pin run injected
# T = 1 + 0.05 exp(-(k/k0)^2) with k0 = 6 k_f, while the equilateral sits with
# all three legs at k_short = 24 k_f, where T - 1 = 5.6e-9. So it was perturbed
# by nothing: sigma_B read 1.3e-10 with a "cancellation factor" of 2.5e+09,
# which is the scatter of a quantity that is identically zero by construction,
# not a floor (g3_floors_record.md, "two problems the pin run exposed"). The
# squeezed triangles were fine -- their long leg sits inside the window's
# support -- so this touches the control only, and the gate is untouched.
#
# WHAT CHANGES. --window flat injects a scale-independent 5% transfer, so every
# triangle has a response. Verified at smoke: R_Q = 0.0476 under a pure flat
# window (the 1/T response, as the closed form predicts), rho flat to 4.4e-16,
# W floor 8.4e-17. The gaussian default is UNCHANGED, so g3_floors_cdev_pin.json
# and the cdev8 cards stay comparable; this writes its own card.
#
# WHAT EACH OUTCOME MEANS (fixed before the run):
#   - If sigma_B(R_Q) for the equilateral lands near the squeezed values
#     (1.9e-4 to 5.1e-4 at cdev), the control is as resolvable as the gate
#     triangles and Stage 5 may compare them directly.
#   - If it is much larger, the control is noisier than the thing it controls,
#     and any "tiling hurts squeezed more" statement must carry that error bar
#     explicitly rather than being read off two point estimates.
#
# RESOURCE FIT (no doomed jobs).
#   Memory: identical to deneb job 282, the cdev pin run -- same script, same
#   config, same 8 shells, one monolithic pair per seed. That ran at 48 G.
#   Wall: job 282 measured 213.9 s/seed at cdev for 24 seeds = 1h42m. Sections
#   D + BC only here, so ~1.4 h expected. 6 h requested (generous per house rule).
#   Placement: -c 64 fits deneb (CPUTot=64) and NOT antares (28).
#
# OUTPUT (runs/ is gitignored -- pull explicitly):
#   runs/v2/g3_floors_cdev_flatw.json

set -euo pipefail

export OMP_NUM_THREADS=64
export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null 2>&1 || { echo "FATAL: pixi not on PATH"; exit 1; }

cd "$HOME/src/inexor"
echo "=== host $(hostname) ==="
echo "=== commit $(git rev-parse --short HEAD) ==="
echo "=== free -g ==="; free -g
date

# k-short-mult 24 == 1.178 h/Mpc at cdev -- the SAME physical scale the pin run
# used. It is in units of THIS box's k_f, and copying another config's number
# voided every bracket in deneb job 283.
pixi run python scripts/v2_g3_floors.py \
    --config cdev --sections DBC --nseed 24 \
    --long-mults 1 2 3 4 5 6 7 --k-short-mult 24 \
    --window flat --out-suffix _flatw

echo "=== done ==="
date
