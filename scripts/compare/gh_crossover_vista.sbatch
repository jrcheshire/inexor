#!/bin/bash
#SBATCH -J gh-crossover
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -o gh-crossover-%j.log
# Wall x memory on ONE GH200: inexor and DISCO-DJ on the same ICs at 256^3, 512^3 and
# 1024^3 (presets cdev, cgh64, c-1024: 0.5 Mpc/h spacing), 120 BullFrog steps, PM mesh twice
# the particle count per side (inexor's fine mesh; DISCO-DJ's single mesh).
#
# Every cost leg runs under scripts/compare/memtrace.py, one instrument for both codes:
# host RSS and each card's allocator bytes every 20 ms, VmHWM and peak_bytes_in_use, and the
# integrals of both over the leg, in $WORK/trace/<leg>.jsonl. Both codes run with the same
# allocator policy (no preallocation, cuda_async), no persistent compilation cache (compile
# is part of the cost), and host memory bound to the CPU NUMA node.
#
# Requires INEXOR_RUNS (run root), DISCO_SRC (disco-mocks checkout) and INEXOR_SRC or
# submission from the inexor checkout. The account is passed with `sbatch -A`:
#   INEXOR_RUNS=$SCRATCH/inexor_runs DISCO_SRC=$HOME/src/disco-mocks \
#     sbatch -A <account> scripts/compare/gh_crossover_vista.sbatch
#
# Legs, in order (inexor legs are fatal, DISCO-DJ legs and cards record rc and continue):
#   guards    clean checkouts, both envs from their locks, pinned jax/discodj, a GPU
#   smoke     cdev8 (128^3) through every leg type below, both codes
#   per size  inexor: ics (device generator) -> preflight -> run (120 steps, checkpoint at
#             120) -> card.  DISCO-DJ: export our ICs -> evolve single -> card -> evolve
#             double -> card.  At 512^3 also DISCO-DJ single under the default preallocating
#             allocator, to show the allocator policy does not change its wall.
#   1024^3    inexor first; then DISCO-DJ export and evolve single, expected to fail on the
#             card (~280 B/p measured, ~300 GB against 96 GB); last, so a host-side failure
#             there costs nothing else.
# Wall, estimated: guards + smoke ~5 min; 256^3 ~10; 512^3 ~20; inexor 1024^3 ~80 (1024785:
# 1:21 with ICs and card); DISCO-DJ 1024^3 ~5. ~2.3 h; 3 h requested.

set -u
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export CONDA_OVERRIDE_CUDA=12.0
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }

INEXOR=${INEXOR_SRC:-$SLURM_SUBMIT_DIR}
DISCO=${DISCO_SRC:?set DISCO_SRC to the disco-mocks checkout}
RUNS=${INEXOR_RUNS:?set INEXOR_RUNS to the run root, e.g. $SCRATCH/inexor_runs}
[ -f "$INEXOR/src/inexor/engine.py" ] || { echo "FATAL: not an inexor checkout: $INEXOR"; exit 1; }
WORK=$RUNS/gh-crossover-$SLURM_JOB_ID
T=$WORK/trace
mkdir -p "$T"

# one allocator policy and no compilation cache for every leg of both codes
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_CLIENT_MEM_FRACTION=0.95
# JAX's default host-memory cap is 64 GB, below what inexor's host streaming needs
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=160
unset JAX_COMPILATION_CACHE_DIR JAX_PLATFORMS CUDA_VISIBLE_DEVICES XLA_PYTHON_CLIENT_MEM_FRACTION

K=120
LAYOUT="--slack 0.10 --alloc-margin 0.10 --arena-frac 0.01"
BIND="numactl --membind=0"
MEMBIND="--membind-nodes 0"
MT=scripts/compare/memtrace.py
DRV=scripts/compare/disco_crosscheck.py
DEV=scripts/run/device_run.py

echo "=== $(hostname) job $SLURM_JOB_ID $(date) ==="
git -C "$INEXOR" log --oneline -1
git -C "$DISCO" log --oneline -1
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
free -g | head -2
numactl --hardware | head -6
env | grep -E '^(XLA_|JAX_)' | sort

rc_total=0
leg () {  # fatal
  local name="$1"; shift
  echo ""; echo "=== LEG $name start $(date +%s) ($(date +%H:%M:%S)) ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc end $(date +%s) ($(date +%H:%M:%S)) ---"
  [ $rc -eq 0 ] || { echo "FATAL: leg $name failed (rc=$rc)"; exit $rc; }
}
soft () {  # records rc and continues
  local name="$1"; shift
  echo ""; echo "=== LEG $name start $(date +%s) ($(date +%H:%M:%S)) ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc end $(date +%s) ($(date +%H:%M:%S)) ---"
  [ $rc -eq 0 ] || rc_total=$((rc_total + 1))
  return 0
}

ipy () { (cd "$INEXOR" && $BIND pixi run --frozen -e gpu python "$@"); }
icpu () { (cd "$INEXOR" && JAX_PLATFORMS=cpu $BIND pixi run --frozen -e gpu python "$@"); }
dpy () { (cd "$INEXOR" && $BIND pixi run --frozen --manifest-path "$DISCO/pixi.toml" -e gpu python "$@"); }
# the cross-check's original allocator settings: BFC, preallocating 95% of the card
dpy_bfc () { (cd "$INEXOR" && env -u XLA_PYTHON_CLIENT_PREALLOCATE -u XLA_PYTHON_CLIENT_ALLOCATOR \
  -u XLA_CLIENT_MEM_FRACTION XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  $BIND pixi run --frozen --manifest-path "$DISCO/pixi.toml" -e gpu python "$@"); }

# ---- guards
clean_checkout () {
  git -C "$1" diff --quiet HEAD || { echo "FATAL: $1 has modified tracked files"; git -C "$1" status -s; return 1; }
  echo "$1 clean at $(git -C "$1" rev-parse --short HEAD)"
}
leg checkout-inexor clean_checkout "$INEXOR"
leg checkout-disco clean_checkout "$DISCO"
leg env-inexor bash -c "cd '$INEXOR' && pixi install --locked -e gpu"
leg env-disco bash -c "cd '$DISCO' && pixi install -e gpu && pixi run -e gpu setup-gpu"
leg checkout-inexor-after-install clean_checkout "$INEXOR"
leg gate-disco dpy -c "
import json, importlib.metadata as md, jax, discodj
url = json.loads(md.distribution('discodj').read_text('direct_url.json') or '{}')
commit = url.get('vcs_info', {}).get('commit_id', '')
print('discodj', md.version('discodj'), commit, discodj.__file__)
print('jax', jax.__version__, jax.devices())
assert jax.__version__ == '0.10.1', jax.__version__
assert commit.startswith('a687972'), commit
assert jax.devices()[0].platform == 'gpu', jax.devices()
"
leg gate-inexor ipy -c "import jax, inexor; print('inexor', inexor.__file__, 'jax', jax.__version__, jax.devices()); assert jax.devices()[0].platform == 'gpu'"

# ---- per size: inexor_size PRESET CARD_FLAGS; disco_export PRESET; disco_arm (below)
inexor_size () {
  local P=$1 W=$WORK/$1 mw=$2
  local ICS=$W/ics CKPT=$W/ckpt PROD=$W/prod
  mkdir -p "$W" "$PROD"
  leg "$P-inexor-ics" ipy $MT --out "$T/$P-inexor-ics.jsonl" -- \
    scripts/run/realization.py ics --config "$P" --workdir "$ICS" --generator device
  leg "$P-inexor-preflight" ipy $MT --out "$T/$P-inexor-preflight.jsonl" -- \
    $DEV preflight --preset "$P" --workdir "$ICS" --cards 1 --card "$W/preflight.json" \
    --scratch "$RUNS" $MEMBIND $LAYOUT
  leg "$P-inexor-run" ipy $MT --out "$T/$P-inexor-run.jsonl" -- \
    $DEV run --preset "$P" --workdir "$ICS" --cards 1 --card "$W/run.json" --k-steps $K \
    --stop-at $K --timed-last $MEMBIND --beat 60 $LAYOUT \
    --checkpoint-dir "$CKPT" --checkpoint-every $K
  ln -sfn "$CKPT" "$PROD/ckpt"
  soft "$P-inexor-card" icpu scripts/run/realization.py card --config "$P" --workdir "$PROD" \
    --k-steps $K $LAYOUT --heartbeat 60 --tile-workers 8 --card-pool $mw
}
disco_export () {
  local P=$1 W=$WORK/$1
  soft "$P-disco-export" icpu $MT --out "$T/$P-disco-export.jsonl" -- \
    $DRV export --config "$P" --ic-dir "$W/ics" --k-steps $K --out "$W/disco_in.npz"
}
disco_arm () {  # PRESET N_MESH PRECISION TAG RUNNER CARD_FLAGS
  local P=$1 NM=$2 PREC=$3 TAG=$4 RUN=$5 mw=$6
  local W=$WORK/$1
  soft "$P-disco-$TAG" $RUN $MT --out "$T/$P-disco-$TAG.jsonl" -- \
    $DRV evolve --in "$W/disco_in.npz" --n-mesh "$NM" --precision "$PREC" --out "$W/disco_$TAG.npz"
  if [ -f "$W/disco_$TAG.npz" ]; then
    soft "$P-disco-$TAG-card" icpu $DRV card --in "$W/disco_$TAG.npz" -- --config "$P" \
      --k-steps $K --workdir "$W/card_$TAG" --coarse-match-order 3 $LAYOUT $mw
    rm -f "$W/disco_$TAG.npz"
  fi
}

# ---- smoke: every leg type at cdev8
inexor_size cdev8 "--min-weight 1"
disco_export cdev8
disco_arm cdev8 256 single single dpy "--min-weight 1"
[ $rc_total -eq 0 ] || { echo "FATAL: the smoke failed ($rc_total legs)"; exit 1; }

# ---- 256^3 and 512^3
inexor_size cdev "--min-weight 1"
disco_export cdev
disco_arm cdev 512 single single dpy "--min-weight 1"
disco_arm cdev 512 double double dpy "--min-weight 1"
rm -f "$WORK/cdev/disco_in.npz"

inexor_size cgh64 ""
disco_export cgh64
disco_arm cgh64 1024 single single dpy ""
disco_arm cgh64 1024 double double dpy ""
disco_arm cgh64 1024 single single-bfc dpy_bfc ""
rm -f "$WORK/cgh64/disco_in.npz"

# ---- 1024^3: inexor, then DISCO-DJ expected to run out of card memory (no card)
inexor_size c-1024 ""
disco_export c-1024
soft "c-1024-disco-single" dpy $MT --out "$T/c-1024-disco-single.jsonl" -- \
  $DRV evolve --in "$WORK/c-1024/disco_in.npz" --n-mesh 2048 --precision single \
  --out "$WORK/c-1024/disco_single.npz"
rm -f "$WORK/c-1024/disco_in.npz" "$WORK/c-1024/disco_single.npz"

echo ""; echo "=== done $(date) rc_total=$rc_total (the 1024^3 DISCO-DJ leg is expected in it) ==="
du -sh "$WORK"/* 2>/dev/null
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,ExitCode -P 2>/dev/null
exit 0
