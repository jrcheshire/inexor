#!/bin/bash
#SBATCH -J inexor-m2-diag
#SBATCH -w deneb
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 1:00:00
#SBATCH -o %x-%j.log
# M2 S4 diagnostic for Vista job 831091's 4 test_adjoint failures.
# Runs on DENEB on purpose: the failing config is n_mesh=16, which the 3050
# handles trivially, and deneb costs no SUs. The phenomenon under test is
# "differently-compiled programs on CUDA", and deneb is a CUDA device -- Vista
# is not required to reproduce it.
#
# --mem is REQUIRED (partition default 124000M is antares-sized and exceeds
# deneb's 56 GB -> the job pends forever). 8G is ample; n_mesh=16 is tiny.
#
# Deneb's gpu env needs NO reinstall: the M2 S4 lock change (2b6ce08) added the
# linux-aarch64 platform ONLY -- zero linux-64 package changes -- so the
# existing .pixi/envs/gpu still satisfies the lock.
#
# Submit from ~/spherex/inexor on deneb:  sbatch scripts/m2_driver_diag.sbatch

set -e
cd ~/spherex/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

echo "=== env ==="
git log --oneline -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

echo "=== the 4 failing tests, on deneb's CUDA (do they reproduce off Vista?) ==="
# `|| true`: a FAILURE here is the expected observation, not a reason to abort
# before the diagnostic that explains it.
OMP_NUM_THREADS=1 pixi run -e gpu python -m pytest tests/test_adjoint.py -q || true

echo "=== diagnostic ==="
pixi run -e gpu python scripts/m2_driver_diag.py

echo "=== DONE ==="
