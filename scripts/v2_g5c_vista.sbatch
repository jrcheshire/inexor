#!/bin/bash
#SBATCH -J inexor-v2-g5c
#SBATCH -o %x-%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 12:00:00
# G5c on TACC Vista (D-v2-10 follow-up). Submit from ~/src/inexor on Vista:
#   sbatch scripts/v2_g5c_vista.sbatch
#
# WHY THIS JOB EXISTS (D-v2-10): the G5 verdict's margin story leans on two
# things measured only at deneb scale -- (1) the 1/P error coefficient, which
# already moved 3-4x between the kernel-study config and C-dev and is unproven
# under more long-wavelength power; (2) capacity triples on the C-gh hardware
# class. Plus the un-gated G2c rerun at FULL C-dev (t9/t12 separate at 2^17
# levels), which never needed split-force machinery -- only a card bigger than
# 6 GB (the deneb-fit contortion D-v2-10 deleted).
#
# PARTITION: -p gh = Grace-Hopper (one GH200: 96 GB HBM3 + ~116 GB LPDDR on a
# 72-core Grace host). NOT -p gg (Grace-Grace, no GPU). ACCOUNT is
# CASE-SENSITIVE: JPL-SPHEREx.
#
# WALL: 12 h is deliberately generous -- TACC bills actual use; a tight wall
# only risks killing the ladder and forcing a full resubmit. All legs live in
# ONE job (a ~15-minute minimum charge applies per batch job).
#
# NO `set -e`, deliberately (M1 job 11 lesson: one leg's failure aborted the
# rest and the survivors had to be re-run). The legs are independent; each
# orchestrator captures its own worker failures. Leg order = cheapest/most
# blocking first, the long ladder last, so a late crash costs the least.
#
# Leg outputs (runs/ is gitignored; pull these back explicitly):
#   runs/v2/g5_results_cdev_vista.json    -- cross-hardware anchor (job-39 twin)
#   runs/v2/g2c_results_vista_cdev.json   -- G2c at FULL C-dev
#   runs/v2/g5_results_cgh64_vista.json   -- the box-ladder coefficient rung

cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
# CPU physics legs: give the f64 FFTs the Grace cores (the probe's default is
# 8, sized for deneb's cgroup).
export G5_CPU_THREADS=64

echo "=== env ==="
git log --oneline -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
# The env install runs ON THE COMPUTE NODE (login-node pixi gets killed).
# --locked: fail loudly if the lock is stale rather than silently re-solving.
pixi install --locked -e gpu
pixi run -e gpu python -c "import jax; print(jax.__version__, jax.devices())"

echo "=== leg 1: G5c anchor at C-dev (job-39 twin on GH200) ==="
# --pivot-only: replicate the ratified number (tile 128 / buf 32), not the
# scan; the scan already exists from deneb job 39. Floors skipped: the D-v2-9
# bar is absolute.
pixi run -e gpu python scripts/v2_g5_two_level_force.py --config cdev \
  --families gauss --skip-coarse --skip-floor-arms --skip-1024 --pivot-only \
  --out-suffix _vista

echo "=== leg 2: G2c at FULL C-dev (t9 vs t12 separate here; 2^17 levels) ==="
pixi run -e gpu python scripts/v2_g2c_accum_gate.py --config cdev
mv -v runs/v2/g2c_results.json runs/v2/g2c_results_vista_cdev.json || true

echo "=== leg 3: G5c box ladder at cgh64 (the 1/P coefficient rung) ==="
# cgh64 = C-gh at 1/64 volume, same fine cell + spacing (8x C-dev volume ->
# more long-wavelength power at the pinned resolution). Full gauss tile-force
# scan (T x b, 9 points) for the dense force-error coefficient; evolved dP/P
# at two padded periods (P = 192 -> the C-dev transfer test, P = 320) for the
# coefficient in the barred quantity; capacity legs give the C-gh-class
# (B/p, wall) triples incl. a mono arm that FITS here (~28 GB f32 vs 96 HBM).
pixi run -e gpu python scripts/v2_g5_two_level_force.py --config cgh64 \
  --families gauss --skip-coarse --skip-floor-arms --skip-1024 \
  --evolve-tiles 128:32,256:32 --out-suffix _vista

echo "=== DONE ==="
