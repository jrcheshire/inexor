#!/bin/bash
#SBATCH -J inexor-v2-g1
#SBATCH -w deneb
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 24:00:00
# --mem is REQUIRED: the partition default (sized for antares's 128 GB)
# exceeds deneb's 56 GB and makes the job permanently unschedulable.
#SBATCH -o %x-%j.log
# v2 seed V1, gate G1: pallas-vs-XLA CIC paint/gather kernel floor
# (plan snuggly-dreaming-popcorn; kill line = pallas peak > 2x hand-managed).
# Correctness gates run on CUDA here (interpret mode cannot: its atomic_add
# is last-write-wins on duplicate indices).
# Submit from ~/src/inexor on deneb:  sbatch scripts/v2_g1_deneb.sbatch

set -e
cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OMP_NUM_THREADS=1

echo "=== env ==="
git log --oneline -1
pixi run -e gpu python -c "import jax; print(jax.__version__, jax.devices())"

echo "=== G1 kernel floor sweep (dense; correctness + determinism on CUDA) ==="
pixi run -e gpu python scripts/v2_g1_kernel_floor.py

echo "=== chunk-size sensitivity at the C-dev shape ==="
pixi run -e gpu python scripts/v2_g1_kernel_floor.py --shapes 256:512 --chunk 1024
pixi run -e gpu python scripts/v2_g1_kernel_floor.py --shapes 256:512 --chunk 16384

echo "=== DONE ==="
