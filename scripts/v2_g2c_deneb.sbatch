#!/bin/bash
#SBATCH -J inexor-v2-g2c
#SBATCH -w deneb
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 24:00:00
# --mem is REQUIRED (see v2_g1_deneb.sbatch); 24G here because the
# orchestrator computes 512^3 f64 FFT statistics host-side between workers.
#SBATCH -o %x-%j.log
# v2 seed V1, gate G2c: accumulated (quantize-every-step) codec tiers at
# C-dev vs the step/mesh floors, multipole-grade bar (D-v2-8).
# Pre-agreed OOM fallback: rerun with --config cdev8 (same cell/spacing).
# Submit from ~/src/inexor on deneb:  sbatch scripts/v2_g2c_deneb.sbatch

set -e
cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OMP_NUM_THREADS=1

echo "=== env ==="
git log --oneline -1
pixi run -e gpu python -c "import jax; print(jax.__version__, jax.devices())"

echo "=== G2c accumulated codec gate at C-dev (floors first) ==="
pixi run -e gpu python scripts/v2_g2c_accum_gate.py --config cdev

echo "=== DONE ==="
