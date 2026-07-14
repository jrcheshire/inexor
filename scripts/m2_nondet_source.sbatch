#!/bin/bash
#SBATCH -J inexor-m2-nondet
#SBATCH -w deneb
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 1:00:00
#SBATCH -o %x-%j.log
# M2 S4 diagnostic 2: localize the reverse sweep's CUDA nondeterminism found by
# job 14, and test whether XLA's deterministic-ops flag removes it.
# deneb, not Vista: n_mesh=16, and deneb costs no SUs. --mem REQUIRED (partition
# default 124000M is antares-sized > deneb's 56 GB -> pends forever).
# Submit from ~/spherex/inexor on deneb:  sbatch scripts/m2_nondet_source.sbatch

set -e
cd ~/spherex/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

echo "=== env ==="
git log --oneline -1
nvidia-smi --query-gpu=name,driver_version --format=csv || true

echo
echo "############ ARM 1: default XLA (reproduce + localize) ############"
pixi run -e gpu python scripts/m2_nondet_source.py

echo
echo "############ ARM 2: XLA deterministic ops (the M0 R3 detflag-f32 arm) ############"
# `|| true`: if XLA rejects the flag name, THAT is the finding -- it must not
# abort the job after arm 1 already produced the localization.
XLA_FLAGS="--xla_gpu_deterministic_ops=true" \
  pixi run -e gpu python scripts/m2_nondet_source.py || true

echo "=== DONE ==="
