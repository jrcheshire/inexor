#!/bin/bash
#SBATCH -J inexor-v2-g5c2
#SBATCH -o %x-%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 08:00:00
# G5c rerun, cgh64 leg ONLY (the FFT_FRIENDLY wrinkle from job 837404).
# Submit from ~/src/inexor on Vista:   sbatch scripts/v2_g5c2_cgh64_vista.sbatch
#
# WHY THIS JOB EXISTS: in job 837404 every tile512 arm of the cgh64 leg died on
#   ValueError: no FFT-friendly padded size >= 576/640/768/1024
# (cgh64 has n_fine=1024; T=512 wants P = 512 + 2b = 576/640/768, and the
# tile-identity check wants P = 1024). Two consequences, both fixed here:
#   1. `validation.tile_identity` + `validation.partition` never ran, so the
#      leg wrote all_ok:false. By the harness's own rule -- a validation FAIL
#      means NO other number in that JSON may be read -- the whole cgh64 card
#      (1/P coefficient 0.66-0.79, evolved |dP/P| 1.90e-2, the ~100x capacity
#      pair) is FORMALLY un-validated, however unambiguous the substance is.
#   2. The largest-tile point is missing from the T-scan, the evolve arms and
#      the capacity legs -- i.e. exactly the "tiling gets cheap at scale" end
#      of the curve the box ladder exists to show.
# `FFT_FRIENDLY` was extended to 1024 in 48ac034 (even 5-smooth sizes appended
# above the FROZEN <=512 block, so every already-measured tile selection is
# byte-identical). This job is the second half of that fix: one clean cgh64
# card with the tile512 arm present.
#
# NOT A NEW MEASUREMENT: the T128/T256 numbers are expected to reproduce job
# 837404's to the digit (same seed, same CPU-f64 path, and the frozen <=512
# padded-size block guarantees they select the same P). If they do not, the
# FFT_FRIENDLY patch perturbed something it promised not to -- that is the
# regression check riding along for free.
#
# PARTITION: -p gh = Grace-Hopper (one GH200: 96 GB HBM3 + ~116 GB LPDDR on a
# 72-core Grace host). NOT -p gg (Grace-Grace, no GPU). ACCOUNT is
# CASE-SENSITIVE: JPL-SPHEREx.
#
# WALL: job 837404's cgh64 leg was ~2.5 h with three evolve arms (mono 857 s,
# T128/b32 3316 s, T256/b32 2608 s at 20 steps); adding T512/b32 (padded 576,
# fft work ratio 1.42, so cheaper per step than either) plus the three tile512
# force arms should land ~3.5-4 h. 8 h is deliberately generous -- TACC bills
# actual use, and a tight wall only risks killing the ladder mid-flight.
#
# NO `set -e`, deliberately (M1 job 11 lesson).
#
# Output (runs/ is gitignored; pull it back explicitly):
#   runs/v2/g5_results_cgh64_vista2.json
# The `_vista2` suffix keeps job 837404's un-validated JSON on disk as the
# provenance of the crash rather than overwriting it.

cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
# CPU physics legs: give the f64 FFTs the Grace cores (the probe default is 8,
# sized for deneb's cgroup).
export G5_CPU_THREADS=64

echo "=== env ==="
git log --oneline -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
# The env install runs ON THE COMPUTE NODE (login-node pixi gets killed).
# --locked: fail loudly if the lock is stale rather than silently re-solving.
pixi install --locked -e gpu
pixi run -e gpu python -c "import jax; print(jax.__version__, jax.devices())"

echo "=== cgh64 leg (validated card + the tile512 point) ==="
# Same flags as job 837404's leg 3 except --evolve-tiles, which now carries
# 512:32 -- tb_list drives BOTH the evolve arms and the GPU capacity legs, so
# one flag buys the missing point in both. The T-scan itself already covers
# T in {128,256,512} x b in {32,64,128} (it is not --pivot-only).
# --skip-1024 stays: the converged 2n floor here would be a 2048^3 mesh, and
# D-v2-9's bar is absolute so it is not needed.
pixi run -e gpu python scripts/v2_g5_two_level_force.py --config cgh64 \
  --families gauss --skip-coarse --skip-floor-arms --skip-1024 \
  --evolve-tiles 128:32,256:32,512:32 --out-suffix _vista2

echo "=== DONE ==="
