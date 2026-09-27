#!/bin/bash
#SBATCH -J disco-xcheck
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -o disco-xcheck-%j.log
# Cross-check the engine's P(k) against a single-mesh PM: DISCO-DJ evolves our ICs at our
# fine resolution (512^3 particles, 1024^3 mesh = 0.25 Mpc/h, 120 BullFrog steps on the
# same a-grid; CIC, ik gradient/Laplacian, no deconvolution), and the result is carded
# with the realization script's own estimator, so it differences bin-for-bin against the
# engine's card on the same ICs with the seed scatter cancelled.
#
# Requires INEXOR_RUNS (run root), DISCO_SRC (disco-mocks checkout), IC_DIR (cgh64 ICs to
# cross-check), and INEXOR_SRC or submission from the inexor checkout. The account is
# passed with `sbatch -A`:
#   INEXOR_RUNS=$SCRATCH/inexor_runs DISCO_SRC=$HOME/src/disco-mocks IC_DIR=<ic dir> \
#     sbatch -A <account> scripts/compare/disco_crosscheck_gh_vista.sbatch
#
# Legs:
#   guards   both checkouts clean; both envs installed from their locks and disco-mocks'
#            pinned pip layer rebuilt in-job; jax 0.10.1, discodj a687972 and the
#            expected device platform asserted; inexor + the driver importable
#   smoke    the whole chain at cdev8 (128^3, 256^3 mesh, 120 steps): fresh ICs,
#            export, evolve, card
#   export   cgh64 ICs -> (x, v_d), 6.4 GB
#   evolve   DISCO-DJ in f64; expected device peak ~75 GB (289 B/p in f32 at pm_ratio 2,
#            doubled), so the memory fraction is raised to 0.95 (~91 GB). If f64 does not
#            fit the job stops: f32 is a different setting.
#   card     serial card of DISCO-DJ's final particles
# Wall: ~0.5 h expected, 1.5 h requested. SMOKE_ONLY=1 stops after the smoke.
#
# DISCO_BACKEND=cpu runs DISCO-DJ on JAX's CPU backend (disco-mocks' `default` env and
# its `setup` task) with identical settings; cdev8 GPU and CPU cards agree to 1e-6.
# Submit with -p gg and a longer -t: the smoke leg prints its wall scaled by 64 as a
# naive 512^3 estimate before the evolve commits.

set -u
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export CONDA_OVERRIDE_CUDA=12.0
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }

INEXOR=${INEXOR_SRC:-$SLURM_SUBMIT_DIR}
DISCO=${DISCO_SRC:?set DISCO_SRC to the disco-mocks checkout}
RUNS=${INEXOR_RUNS:?set INEXOR_RUNS to the run root, e.g. $SCRATCH/inexor_runs}
WORK=$RUNS/disco-xcheck-$SLURM_JOB_ID
IC=${IC_DIR:?set IC_DIR to the IC directory to cross-check}
DRV=scripts/compare/disco_crosscheck.py
mkdir -p "$WORK"

echo "=== $(hostname) job $SLURM_JOB_ID $(date) ==="
echo "ICs: $IC"
git -C "$INEXOR" log --oneline -1
git -C "$DISCO" log --oneline -1
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || echo "no GPU on this node"

leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ($(date +%H:%M:%S)) ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc ($(date +%H:%M:%S)) ---"
  [ $rc -eq 0 ] || { echo "FATAL: leg $name failed (rc=$rc)"; exit $rc; }
}

# inexor legs run on the CPU backend; the disco legs on the card
inexor () { (cd "$INEXOR" && JAX_PLATFORMS=cpu pixi run -e gpu python "$@"); }
BACKEND=${DISCO_BACKEND:-gpu}
case "$BACKEND" in
  gpu) DENV=gpu; DSETUP=setup-gpu; DPLAT=gpu
       disco () { (cd "$INEXOR" && XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
         pixi run --manifest-path "$DISCO/pixi.toml" -e gpu python "$@"); } ;;
  cpu) DENV=default; DSETUP=setup; DPLAT=cpu
       disco () { (cd "$INEXOR" && JAX_PLATFORMS=cpu \
         pixi run --manifest-path "$DISCO/pixi.toml" -e default python "$@"); } ;;
  *) echo "FATAL: DISCO_BACKEND must be gpu or cpu, got $BACKEND"; exit 1 ;;
esac
echo "DISCO-DJ backend: $BACKEND (env $DENV)"

# ---- guards: nothing about either checkout or env is assumed
clean_checkout () {  # tracked files must be unmodified; untracked logs are fine
  git -C "$1" diff --quiet HEAD || { echo "FATAL: $1 has modified tracked files"; git -C "$1" status -s; return 1; }
  echo "$1 clean at $(git -C "$1" rev-parse --short HEAD)"
}
leg checkout-inexor clean_checkout "$INEXOR"
leg checkout-disco clean_checkout "$DISCO"
leg env-inexor bash -c "cd '$INEXOR' && pixi install -e gpu"
leg env-disco bash -c "cd '$DISCO' && pixi install -e $DENV && pixi run -e $DENV $DSETUP"
leg gate-disco disco -c "
import json, importlib.metadata as md, jax, discodj
url = json.loads(md.distribution('discodj').read_text('direct_url.json') or '{}')
commit = url.get('vcs_info', {}).get('commit_id', '')
print('discodj', md.version('discodj'), commit, discodj.__file__)
print('jax', jax.__version__, jax.devices())
assert jax.__version__ == '0.10.1', jax.__version__
assert commit.startswith('a687972'), commit
assert jax.devices()[0].platform == '$DPLAT', jax.devices()
"
leg gate-inexor inexor -c "import sys; sys.path[:0] = ['scripts/run', 'scripts/compare']; import jax, inexor, realization, disco_crosscheck; print('inexor', inexor.__file__, 'jax', jax.__version__, jax.devices())"

# ---- smoke: cdev8 end to end
S=$WORK/smoke
leg smoke-ics inexor scripts/run/realization.py ics --config cdev8 --workdir "$S/ics"
leg smoke-export inexor $DRV export --config cdev8 --ic-dir "$S/ics" --k-steps 120 --out "$S/in.npz"
T0=$(date +%s)
leg smoke-evolve disco $DRV evolve --in "$S/in.npz" --n-mesh 256 --out "$S/out.npz"
echo "smoke evolve leg $(( $(date +%s) - T0 )) s; x64 (512^3 particles and mesh) ~ $(( ($(date +%s) - T0) * 64 / 60 )) min, a naive linear scaling"
leg smoke-card inexor $DRV card --in "$S/out.npz" -- --config cdev8 --k-steps 120 \
  --workdir "$S" --coarse-match-order 3 --slack 0.10 --arena-frac 0.01 --min-weight 1

# SMOKE_ONLY=1 (e.g. on gh-dev) stops here: env build, GPU import and the chain
if [ "${SMOKE_ONLY:-0}" = 1 ]; then echo "=== SMOKE_ONLY: done $(date) ==="; ls -la "$S"; exit 0; fi

# ---- the cross-check
leg export inexor $DRV export --config cgh64 --ic-dir "$IC" --k-steps 120 --out "$WORK/in.npz"
leg evolve disco $DRV evolve --in "$WORK/in.npz" --n-mesh 1024 --out "$WORK/out.npz"
rm -f "$WORK/in.npz"
leg card inexor $DRV card --in "$WORK/out.npz" -- --config cgh64 --k-steps 120 \
  --workdir "$WORK" --coarse-match-order 3 --slack 0.10 --arena-frac 0.01

echo ""; echo "=== done $(date) ==="; ls -la "$WORK" "$S"
