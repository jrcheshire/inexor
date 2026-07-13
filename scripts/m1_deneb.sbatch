#!/bin/bash
#SBATCH -J inexor-m1-s7
#SBATCH -w deneb
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 8:00:00
#SBATCH -o %x-%j.log
# M1 S7 authoritative CUDA legs (plan tender-stargazing-map):
#   1. full pytest suite in the gpu env (tier-0 exact reversibility + the
#      int-paint determinism tests ARE the CUDA re-confirmation arms)
#   2. 512^3 forward smoke + at-scale exact roundtrip (384^3 pre-agreed
#      fallback, documented in the output json)
# Submit from ~/spherex/inexor on deneb:  sbatch scripts/m1_deneb.sbatch

set -e
cd ~/spherex/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

echo "=== env ==="
git log --oneline -1
pixi run -e gpu python -c "import jax; print(jax.__version__, jax.devices())"

echo "=== full suite (gpu env, incl. slow tier-0 + m0 bridge) ==="
OMP_NUM_THREADS=1 pixi run -e gpu python -m pytest tests/ -q

echo "=== 512^3 smoke + roundtrip (384^3 fallback) ==="
pixi run -e gpu python scripts/m1_smoke.py --roundtrip

echo "=== DONE ==="
