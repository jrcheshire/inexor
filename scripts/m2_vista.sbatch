#!/bin/bash
#SBATCH -J inexor-m2-s4
#SBATCH -o %x-%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 08:00:00
# M2 S4 on TACC Vista (plan buzzing-seeking-patterson). Submit from ~/src/inexor
# on Vista:  sbatch scripts/m2_vista.sbatch
#
# PARTITION: -p gh is the Grace-Hopper (GH200) partition -- one Hopper GPU,
# 96 GB HBM3, on a Grace ARM host. NOT -p gg: despite the M2 plan text saying
# "Vista GH200 (gg queue)", gg is Vista's Grace-Grace CPU partition and has no
# GPU at all. Vista is all-aarch64 either way (umbrella reference_tacc_practicalities).
#
# ACCOUNT: -A JPL-SPHEREx is CASE-SENSITIVE (the submit filter rejects
# jpl-spherex, and sacctmgr misleadingly displays it lowercased).
#
# WALL: 8 h is deliberately generous -- TACC bills actual use, so a tight wall
# only risks killing the sweep and forcing a resubmit (and Vista GH200 SUs are
# scarce at ~1 SU/GPU-hr). All legs live in ONE job because TACC charges a
# ~15-minute minimum per batch job.
#
# Legs:
#   1. env//device report (this is the FIRST inexor run on aarch64 -- the gpu
#      env's linux-aarch64 platform landed with this milestone, so the install
#      itself is under test)
#   2. full pytest in the gpu env: tier-0 exact reversibility + int-paint
#      determinism are the CUDA re-confirmation arms, and test_adjoint.py now
#      rides along (the S3 CUDA confirmation)
#   3. M1's deferred 512^3 forward smoke + exact roundtrip -- it OOM'd deneb's
#      6 GB 3050 and has been waiting on exactly this env (M1 closed at 256^3)
#   4. the R6 peak-memory sweep + checkpoint A/B (the S4 deliverable)

set -e
cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"

# Give the allocator most of the 96 GB. peak_bytes_in_use measures real
# allocations inside the arena, so preallocation does not distort the profile.
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache

echo "=== env ==="
git log --oneline -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
pixi run -e gpu python -c "import jax; print(jax.__version__, jax.devices())"

echo "=== full suite (gpu env, incl. slow tier-0 + test_adjoint) ==="
OMP_NUM_THREADS=1 pixi run -e gpu python -m pytest tests/ -q

echo "=== M1's deferred 512^3 forward smoke + exact roundtrip ==="
pixi run -e gpu python scripts/m1_smoke.py --roundtrip

echo "=== R6: peak-memory sweep + checkpoint A/B (S4 deliverable) ==="
pixi run -e gpu python scripts/m2_mem_profile.py --n 64,128,256,512 --ab

echo "=== DONE ==="
