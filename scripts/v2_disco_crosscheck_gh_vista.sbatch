#!/bin/bash
#SBATCH -J disco-xcheck
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o disco-xcheck-%j.log
# Does a single-mesh PM share the ~14% high-k deficit against EE2?
#
# DISCO-DJ evolves OUR ICs (the z = 9, corrected-match realization that
# mo3_1018341_k120 carded) on its own PM at our fine resolution: 512^3 particles,
# 1024^3 mesh (0.25 Mpc/h), 120 BullFrog steps on the same a-grid, M1 parity
# settings (CIC, ik gradient/Laplacian, no deconvolution). The result is carded
# with the realization script's own estimator, so it differences bin-for-bin
# against mo3_1018341_k120 with the seed scatter cancelled.
#
# LEGS
#   setup    build the disco-mocks `gpu` env (absent on Vista: only `default`
#            is installed) -- pinned jax[cuda12] 0.10.1 + discodj a687972
#   gate     DISCO-DJ imports and sees the GPU
#   smoke    the whole chain at cdev8 (128^3, 256^3 mesh, 120 steps): fresh
#            ICs, export, evolve, card. Low k must match the local cdev8 run.
#   export   cgh64 ICs -> (x, v_d), 6.4 GB
#   evolve   DISCO-DJ f64. Expected device peak ~75 GB (289 B/p measured in f32
#            at pm_ratio 2, doubled); the memory fraction is raised to 0.95
#            (~91 GB). If f64 does not fit, the job STOPS -- f32 is not the
#            parity setting and is a separate decision.
#   card     serial card of DISCO-DJ's final particles
#
# BUDGET: gh = 1 SU/node-hr; expected ~0.5 h, 1.5 h wall.

set -u
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export CONDA_OVERRIDE_CUDA=12.0
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }

INEXOR=$HOME/src/inexor
DISCO=$HOME/src/disco-mocks
RUNS=/scratch/10303/jrc4/inexor_runs
WORK=$RUNS/disco-xcheck-$SLURM_JOB_ID
IC=$RUNS/match-order-1018341/ics
DRV=scripts/v2_disco_crosscheck.py
mkdir -p "$WORK"

echo "=== $(hostname) job $SLURM_JOB_ID $(date) ==="
git -C "$INEXOR" log --oneline -1
git -C "$DISCO" log --oneline -1
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ($(date +%H:%M:%S)) ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc ($(date +%H:%M:%S)) ---"
  [ $rc -eq 0 ] || { echo "FATAL: leg $name failed (rc=$rc)"; exit $rc; }
}

# inexor legs run on the CPU backend; the disco legs on the card
inexor () { (cd "$INEXOR" && JAX_PLATFORMS=cpu pixi run -e gpu python "$@"); }
disco () { (cd "$INEXOR" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  pixi run --manifest-path "$DISCO/pixi.toml" -e gpu python "$@"); }

leg setup bash -c "cd '$DISCO' && pixi run -e gpu setup-gpu"
leg gate disco -c "import jax, discodj; print('discodj', discodj.__file__); print('jax', jax.__version__, jax.devices())"

# ---- smoke: cdev8 end to end
S=$WORK/smoke
leg smoke-ics inexor scripts/v2_m6_realization.py ics --config cdev8 --workdir "$S/ics"
leg smoke-export inexor $DRV export --config cdev8 --ic-dir "$S/ics" --k-steps 120 --out "$S/in.npz"
leg smoke-evolve disco $DRV evolve --in "$S/in.npz" --n-mesh 256 --out "$S/out.npz"
leg smoke-card inexor $DRV card --in "$S/out.npz" -- --config cdev8 --k-steps 120 \
  --workdir "$S" --coarse-match-order 3 --slack 0.10 --arena-frac 0.01 --min-weight 1

# ---- the cross-check
leg export inexor $DRV export --config cgh64 --ic-dir "$IC" --k-steps 120 --out "$WORK/in.npz"
leg evolve disco $DRV evolve --in "$WORK/in.npz" --n-mesh 1024 --out "$WORK/out.npz"
rm -f "$WORK/in.npz"
leg card inexor $DRV card --in "$WORK/out.npz" -- --config cgh64 --k-steps 120 \
  --workdir "$WORK" --coarse-match-order 3 --slack 0.10 --arena-frac 0.01

echo ""; echo "=== done $(date) ==="; ls -la "$WORK" "$S"
