#!/bin/bash
#SBATCH -J m5-memladder
#SBATCH -A JPL-SPHEREx
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -o m5-memladder-%j.log
# M-v2-5 leg V on a Vista gg node (237 GB LPDDR, CPU -- the G6c home): the
# albireo route needs antares (deneb's 56 GB cannot host the old-mirror arm's
# ~90 GB at n=1024) and antares was occupied. gg is GPU-less, so the `gpu`
# env needs CONDA_OVERRIDE_CUDA at activation (the cb27372 pattern) and jax
# comes up CPU -- which the probe REQUIRES (it refuses any other backend).
# Staging goes to $SCRATCH via TMPDIR. NO set -e: the probe runs cheapest
# rung first and its own exit code is the verdict.
#
# JAX_PLATFORMS=cpu is LOAD-BEARING on jax 0.10: CONDA_OVERRIDE_CUDA only
# gets the env ACTIVATED; the CUDA plugin's discovery then hard-raises on
# cuInit (error 303) on a GPU-less node instead of falling back (job 902236
# died on exactly this). Forcing the platform skips plugin discovery -- the
# probes' standing orchestrator pattern.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || exit 1
export CONDA_OVERRIDE_CUDA=12.0
export JAX_PLATFORMS=cpu
pixi run -e gpu python -c "import jax; d=jax.devices(); print('jax', jax.__version__, d); assert d[0].platform=='cpu', 'expected CPU backend on gg'" \
  || { echo "FATAL: jax import/backend check failed"; exit 1; }

export TMPDIR="${SCRATCH:?SCRATCH not set}/m5_ladder_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"

RUNGS="${RUNGS:-256 512 1024}"
echo "=== memory ladder on $(hostname), rungs $RUNGS, TMPDIR $TMPDIR ==="
free -g | head -2
# shellcheck disable=SC2086
pixi run -e gpu python scripts/v2_m5_memladder.py --rungs $RUNGS
rc=$?
echo "=== memladder rc=$rc ==="
echo "cards written:"
ls -la runs/v2/m5_gate_memladder*.json 2>/dev/null || echo "  (none)"
rmdir "$TMPDIR" 2>/dev/null || echo "TMPDIR not empty, left for inspection: $TMPDIR"
exit $rc
