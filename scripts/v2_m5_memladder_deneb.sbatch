#!/bin/bash
#SBATCH -J m5-memladder
#SBATCH -p compute
#SBATCH --mem=110G
#SBATCH -t 04:00:00
#SBATCH -o m5-memladder-%j.log
# M-v2-5 leg V: the memory ladder, 3 arms x rungs 256/512/1024 (RUNGS
# overridable). NEEDS the big-memory CPU node: the old-mirror arm at n=1024
# measured ~90 GB (job 896159), so --mem=110G places this on antares
# (28c / 124 GB / no GPU) by memory, never by -w. CPU backend REQUIRED and
# self-enforced (the probe exits fatally on a non-CPU backend: ru_maxrss is
# HOST memory and a CUDA run measures the wrong pool -- deneb 409).
# NO set -e: the probe runs cheapest rung first, so a top-rung OOM still
# yields the lower rungs; the probe's own exit code is the verdict.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || exit 1
pixi run -e default python -c "import jax; print('jax', jax.__version__, jax.devices())" \
  || { echo "FATAL: jax import failed"; exit 1; }

RUNGS="${RUNGS:-256 512 1024}"
echo "=== memory ladder, rungs $RUNGS ==="
# shellcheck disable=SC2086
pixi run -e default python scripts/v2_m5_memladder.py --rungs $RUNGS
rc=$?
echo "=== memladder rc=$rc ==="
echo "cards written:"
ls -la runs/v2/m5_gate_memladder*.json 2>/dev/null || echo "  (none)"
exit $rc
