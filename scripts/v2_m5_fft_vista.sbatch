#!/bin/bash
#SBATCH -J m5-fft-gh
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -o m5-fft-gh-%j.log
# M-v2-5 leg VI: the out-of-core FFT layer at 2048^3 on one GH200 node --
# the ONE Vista job this milestone owes (JC 2026-08-10). The FFTs are
# host-CPU (scipy through ooc_fft) and the noise draw is pinned to the CPU
# backend inside ic.white_plane, so the GH200's role is being the production
# node whose ~116 GB host budget the capacity claim is about. Staging goes
# to $SCRATCH; the script's io-probe phase times one 8.6 GB slab write/read
# BEFORE the big phase commits.
# NO set -e: phases are independent, cheapest first, per-phase status inside
# the script, exit 1 from the script if any phase failed.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || exit 1
pixi run -e gpu python -c "import jax; print('jax', jax.__version__, jax.devices())" \
  || { echo "FATAL: jax import failed"; exit 1; }

WORKDIR="${SCRATCH:?SCRATCH not set}/m5_fft_$SLURM_JOB_ID"
mkdir -p "$WORKDIR"
echo "=== leg VI: 2048^3 out-of-core FFT, workdir $WORKDIR ==="
pixi run -e gpu python scripts/v2_m5_fft_gh.py --n 2048 --workdir "$WORKDIR"
rc=$?
echo "=== fft-gh rc=$rc ==="
echo "cards written:"
ls -la runs/v2/m5_gate_fft-gh*.json 2>/dev/null || echo "  (none)"
# no rm -rf (house rule); the phases clean their own files and $SCRATCH is
# purge-eligible anyway
rmdir "$WORKDIR" 2>/dev/null || echo "workdir not empty, left for inspection: $WORKDIR"
exit $rc
