#!/bin/bash
#SBATCH -J inexor-v4a-anatomy
#SBATCH -o %x-%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 01:00:00
# V4a: WHERE THE TILED-FORCE WALL ACTUALLY GOES, and whether the per-tile cost
# is box-independent. Submit from ~/src/inexor:
#   sbatch scripts/v2_v4a_tile_anatomy_vista.sbatch
#
# WHY THIS EXISTS. The V4 go/no-go rests on an extrapolation nobody has
# checked. G5c measured the cgh64 tiled force at 21.36 s (T256/b32, 64 tiles,
# GH200), and C-gh is 64x that volume at the SAME fine cell, so at fixed (T, b)
# in fine cells the per-tile padded solve is identical and only the TILE COUNT
# changes -- 64 -> 4096 tiles, predicting ~1367 s per force evaluation at C-gh.
# Two things could make that number wrong or meaningless:
#
# (1) COMPILE CONTAMINATION. The T256 arm ran only 64 tiles, so a ~10-15 s XLA
#     compile would be a large fraction of its 21.36 s -- and compile does NOT
#     scale with tile count, so including it inflates the C-gh estimate.
# (2) BOX DEPENDENCE. The x64 extrapolation assumes steady-state per-tile cost
#     does not depend on the box. That is the load-bearing assumption and it
#     has never been tested on GPU, where host-side membership and the numpy
#     fancy-index gather both touch O(box) arrays.
#
# And one thing makes the whole rate suspect: at P=320 the tile's four FFTs
# should cost ~2 ms on a GH200 (scaled from the G4b card's 3.818 ms for a
# 512^3 f64 fftn), against a measured ~334 ms per tile. If the tile is running
# ~100x above its FFT floor then "the current rate" is not a property of the
# architecture and no V4 build/no-build call should be made against it.
#
# THE MEASUREMENT. `force_short_tiled` now records four phase timers per tile:
#   member  = host bucket lookup + index/mask build
#   stage   = the numpy fancy-index gather pos_np[idx_pad] + H2D
#   device  = the jitted paint/FFT/gather + D2H
#   scatter = host write-back
# No jit splitting and no extra syncs, so fusion is untouched and the phases
# still sum to the wall the un-instrumented code would have taken. Tile 0
# carries the compile; steady state is the median over tiles 1..
#
# PRE-REGISTERED READING RULES (fixed before the run):
#   - Legs 1 and 2 share (T, b) = (128, 32) hence P = 192, and differ ONLY in
#     box: cdev has 64 tiles, cgh64 has 512 at 8x the volume. If steady-state
#     per-tile cost agrees between them, the x64 extrapolation to C-gh is
#     licensed. If cgh64's is materially higher, it is NOT, and the C-gh
#     number must be measured rather than scaled.
#   - Leg 3 is the C-gh candidate geometry (T256/b32, P=320) and anchors the
#     anatomy to the existing G5c card. Its `device.total_s` need not equal
#     that card's 21.36 s: this leg also writes a .npy the capacity leg does
#     not. Compare the PHASES, not the totals.
#   - `device` dominating means the cost is real work and the extrapolation
#     stands. `stage` or `member` dominating means the cost is host plumbing
#     the streaming build removes anyway, and the C-gh estimate is an upper
#     bound on an artifact.
#
# NOT MEASURED HERE, deliberately: C-gh itself. The probe holds pos_np and the
# force accumulator as full f64 (n,3) host arrays, which at C-gh is 2 x 206 GB
# against a gh node's ~116 GB usable host (D-v2-13). Running C-gh needs the G4
# streaming path wired into the force, and that IS the V4 build, not a job.
#
# NO `set -e`: three independent legs, cheapest first (M1 job 11 lesson).
# Config states already exist on Vista scratch, so no IC generation.

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
echo "=== leg 1: cdev  T=128 b=32 (P=192, 64 tiles) -- the small-box reference ==="
pixi run -e gpu python scripts/v2_g5_two_level_force.py $COMMON \
  --config cdev --tag v4a_cdev_T128_b32 --tile 128 --buf 32

echo
echo "=== leg 2: cgh64 T=128 b=32 (P=192, 512 tiles) -- SAME P, 8x the box ==="
pixi run -e gpu python scripts/v2_g5_two_level_force.py $COMMON \
  --config cgh64 --tag v4a_cgh64_T128_b32 --tile 128 --buf 32

echo
echo "=== leg 3: cgh64 T=256 b=32 (P=320, 64 tiles) -- the C-gh candidate ==="
pixi run -e gpu python scripts/v2_g5_two_level_force.py $COMMON \
  --config cgh64 --tag v4a_cgh64_T256_b32 --tile 256 --buf 32

echo
echo "=== DONE ==="
