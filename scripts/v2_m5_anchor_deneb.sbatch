#!/bin/bash
#SBATCH -J m5-anchor
#SBATCH -p compute
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o m5-anchor-%j.log
# M-v2-5 deneb anchor: gate legs invariance + e2e at n=512 f64 (the cdev-class
# scale the laptop legs ran at n=64). CPU env by construction (`default` is
# CPU-only jax, so the backend cannot silently become CUDA and no
# CONDA_OVERRIDE_CUDA is needed on any node). Placement by --mem, not -w.
# NO set -e, deliberately: independent legs, cheapest first; per-leg exit
# codes below and exit 1 if any failed (the 896055 lesson: a green Slurm row
# that ran nothing).

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || exit 1
pixi run -e default python -c "import jax; print('jax', jax.__version__, jax.devices())" \
  || { echo "FATAL: jax import failed"; exit 1; }

rc_total=0
for leg in invariance e2e; do
  echo "=== leg $leg n=512 ==="
  pixi run -e default python scripts/v2_m5_ic_gate.py \
    --leg "$leg" --n 512 --bricks 8 --slab 32 --out-suffix _n512
  rc=$?
  echo "=== leg $leg rc=$rc ==="
  [ $rc -ne 0 ] && rc_total=1
done

echo "=== LEG SUMMARY ==="
echo "cards written:"
ls -la runs/v2/m5_gate_*_n512.json 2>/dev/null || echo "  (none)"
exit $rc_total
