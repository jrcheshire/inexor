#!/bin/bash
#SBATCH -J inexor-v4c-cap-growth
#SBATCH -o %x-%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 01:00:00
# V4c: does per-tile cost grow with TILE COUNT, or with BOX? Submit from
# ~/src/inexor:
#   sbatch scripts/v2_v4c_cap_growth_vista.sbatch
#
# WHY. V4a (job 895315) found per-tile cost is not box-independent: at fixed
# P=192 the steady-state per-tile wall went 34.6 ms (cdev, 64 tiles) -> 58.5 ms
# (cgh64, 512 tiles), with the device phase alone going 10.6 -> 24.1 ms. The
# suspected mechanism is `cap`, the padded per-tile particle count, which is a
# MAX OVER TILES and so grows as tiles are added (1,284,096 -> 2,078,720, a
# factor 1.62); `stage` tracked it almost exactly (1.68x). Every tile then pays
# the padded cost.
#
# BUT THAT PAIR IS CONFOUNDED. Going cdev -> cgh64 changed the tile count (8x)
# AND the box (8x volume, hence 8x the host arrays the gather reads from). The
# device phase grew 2.27x, MORE than cap's 1.62x, so something other than cap
# is contributing and box-size memory pressure is the obvious candidate. The
# two have opposite consequences for V4:
#   - if it is TILE COUNT, C-gh's 4096 tiles carry the growth and the cost
#     extrapolation must include it;
#   - if it is BOX SIZE (host array pressure on the numpy fancy-index gather),
#     the streaming build removes it by construction and the C-gh device floor
#     is the no-growth number, ~196 s per force evaluation.
#
# THE FIX: change the tile count at FIXED BOX. Holding P fixed while shrinking
# T raises the tile count without touching the box, the state arrays, or the
# per-tile geometry:
#
#   P=192 axis:  cdev  T128/b32   64 tiles  [have, V4a leg 1]
#                cgh64 T128/b32  512 tiles  [have, V4a leg 2]
#                cgh64 T64/b64  4096 tiles  [LEG 3 here]  <- 8x tiles, SAME box
#
#   P=320 axis:  cdev  T256/b32    8 tiles  [LEG 1 here]
#                cgh64 T256/b32   64 tiles  [have, V4a leg 3]
#                cgh64 T128/b96  512 tiles  [LEG 2 here]  <- 8x tiles, SAME box
#
# Two independent P values measure the same law, so one anomalous point cannot
# decide it. Geometry legality (padded_size / choose_brick / brick_span) was
# checked locally for all three before submitting.
#
# PRE-REGISTERED READING RULES:
#   - The estimand is STEADY-STATE PER-TILE cost (median over tiles 1.., so
#     compile is excluded) and its `device` component. Totals are not the
#     measurement: these legs deliberately differ in tile count, so of course
#     their totals differ.
#   - If the fixed-box 8x-tile steps reproduce V4a's ~2.3x device growth, the
#     driver is tile count and C-gh must carry it.
#   - If they show little or no growth, V4a's growth was box size, and the
#     C-gh device floor stands at the no-growth number.
#   - `cap` is reported per leg, so the attribution is directly checkable: if
#     per-tile cost tracks cap across all rungs, the mechanism is settled.
#
# NOT AN OPERATING POINT. Leg 2 (T128/b96) has a padded-volume ratio of 15.6
# and leg 3 (T64/b64) of 27, so both do enormous redundant physics per unit of
# box. They are instruments for isolating a cost law, not candidate geometries,
# and must never be quoted as proposed configurations.
#
# NO `set -e`: three independent legs, cheapest first (M1 job 11 lesson).
# States already on Vista scratch, so no IC generation. ~12 min expected.

cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache

echo "=== provenance ==="
git log --oneline -1
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
free -g | head -2

COMMON="--single --kind force --family gauss --which mono --alpha 1.0 --fdtype f64 --a 1.0 --profile 1"

echo
echo "=== leg 1: cdev  T=256 b=32 (P=320, 8 tiles) -- small-box anchor, P=320 axis ==="
pixi run -e gpu python scripts/v2_g5_two_level_force.py $COMMON \
  --config cdev --tag v4c_cdev_T256_b32 --tile 256 --buf 32

echo
echo "=== leg 2: cgh64 T=128 b=96 (P=320, 512 tiles) -- 8x tiles, SAME box as V4a leg 3 ==="
pixi run -e gpu python scripts/v2_g5_two_level_force.py $COMMON \
  --config cgh64 --tag v4c_cgh64_T128_b96 --tile 128 --buf 96

echo
echo "=== leg 3: cgh64 T=64 b=64 (P=192, 4096 tiles) -- 8x tiles, SAME box as V4a leg 2 ==="
pixi run -e gpu python scripts/v2_g5_two_level_force.py $COMMON \
  --config cgh64 --tag v4c_cgh64_T64_b64 --tile 64 --buf 64

echo
echo "=== DONE ==="
