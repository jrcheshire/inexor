#!/bin/bash
#SBATCH -J gh-same-ics
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:30:00
#SBATCH -o gh-same-ics-%j.log
# Both codes on ONE GH200 from the same ICs as gh_crossover_vista.sbatch (presets cdev 256^3
# and cgh64 512^3, device generator, seed 0), 120 BullFrog steps, PM mesh twice the particle
# count per side:
#   - DISCO-DJ in single precision at its particle chunking (run_nbody `chunk_size`, the
#     knob that bounds its per-particle paint/interpolate temporaries): all particles at
#     once (the crossover's setting) and the chunk sizes in CHUNKS_<preset>; each final state
#     carded against the unchunked one.
#   - inexor at 512^3 with the fine tiles in float64 (production) and float32: wall, peaks,
#     cards. The float64 arm is exported (x, v), and DISCO-DJ's unchunked final (x, v_d) is
#     kept, for the velocity comparison.
#
# Every cost leg runs under scripts/compare/memtrace.py with the crossover's allocator policy
# (no preallocation, cuda_async, no persistent compilation cache) and host memory bound to the
# CPU NUMA node, so the numbers compare with that job's.
#
# Requires INEXOR_RUNS (run root), DISCO_SRC (disco-mocks checkout) and INEXOR_SRC or
# submission from the inexor checkout. The account is passed with `sbatch -A`:
#   INEXOR_RUNS=$SCRATCH/inexor_runs DISCO_SRC=$HOME/src/disco-mocks \
#     sbatch -A <account> scripts/compare/gh_same_ics_vista.sbatch
#
# Legs, in order (guards and inexor run legs are fatal; cards, exports and DISCO-DJ legs
# record rc and continue):
#   guards    clean checkouts; inexor's gpu env through scripts/run/gpu_env.sh (as every
#             GPU job); disco-mocks' env from its lock; pinned jax/discodj; a GPU
#   smoke     cdev8 (128^3): inexor float32 run + card; DISCO-DJ export, one chunked arm,
#             card
#   256^3     ICs; DISCO-DJ export and every chunk arm with its card
#   512^3     ICs; inexor float64 run, card, export; inexor float32 run, card; DISCO-DJ
#             export and every chunk arm with its card
# Wall, estimated: guards ~10 min, smoke ~8, 256^3 ~5, 512^3 ~45 (two inexor runs ~14 min
# each at the crossover's rate). ~70 min; 2.5 h requested.

set -u
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export CONDA_OVERRIDE_CUDA=12.0
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }

INEXOR=${INEXOR_SRC:-$SLURM_SUBMIT_DIR}
DISCO=${DISCO_SRC:?set DISCO_SRC to the disco-mocks checkout}
RUNS=${INEXOR_RUNS:?set INEXOR_RUNS to the run root, e.g. $SCRATCH/inexor_runs}
[ -f "$INEXOR/src/inexor/engine.py" ] || { echo "FATAL: not an inexor checkout: $INEXOR"; exit 1; }
WORK=$RUNS/gh-same-ics-$SLURM_JOB_ID
T=$WORK/trace
mkdir -p "$T"

# chunk sizes per preset, in particles; "all" = DISCO-DJ's default (no chunking)
CHUNKS_cdev8=${CHUNKS_cdev8:-"262144"}
CHUNKS_cdev=${CHUNKS_cdev:-"all 4194304 1048576"}
CHUNKS_cgh64=${CHUNKS_cgh64:-"all 16777216 4194304 1048576"}

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

# ---- guards
clean_checkout () {
  git -C "$1" diff --quiet HEAD || { echo "FATAL: $1 has modified tracked files"; git -C "$1" status -s; return 1; }
  echo "$1 clean at $(git -C "$1" rev-parse --short HEAD)"
}
leg checkout-inexor clean_checkout "$INEXOR"
leg checkout-disco clean_checkout "$DISCO"
# the shared guard: census + imports, else a rebuild through a fresh cache, under the lock the
# other GPU jobs on this checkout take
. "$INEXOR/scripts/run/gpu_env.sh"
leg env-inexor ensure_gpu_env "$INEXOR"
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

# ---- helpers: inexor_ics PRESET; inexor_arm PRESET FINE_DTYPE EXPORT CARD_FLAGS;
# disco_export PRESET; disco_arm PRESET N_MESH CHUNK CARD_FLAGS
inexor_ics () {
  local P=$1
  leg "$P-inexor-ics" ipy $MT --out "$T/$P-inexor-ics.jsonl" -- \
    scripts/run/realization.py ics --config "$P" --workdir "$WORK/$P/ics" --generator device
}
inexor_arm () {
  local P=$1 FD=$2 EXP=$3 mw=$4
  local W=$WORK/$P/inexor-$FD
  local ICS=$WORK/$P/ics CKPT=$W/ckpt PROD=$W/prod
  mkdir -p "$W" "$PROD"
  leg "$P-inexor-$FD-preflight" ipy $MT --out "$T/$P-inexor-$FD-preflight.jsonl" -- \
    $DEV preflight --preset "$P" --workdir "$ICS" --cards 1 --card "$W/preflight.json" \
    --scratch "$RUNS" $MEMBIND $LAYOUT --fine-dtype "$FD"
  leg "$P-inexor-$FD-run" ipy $MT --out "$T/$P-inexor-$FD-run.jsonl" -- \
    $DEV run --preset "$P" --workdir "$ICS" --cards 1 --card "$W/run.json" --k-steps $K \
    --stop-at $K --timed-last $MEMBIND --beat 60 $LAYOUT --fine-dtype "$FD" \
    --checkpoint-dir "$CKPT" --checkpoint-every $K
  ln -sfn "$CKPT" "$PROD/ckpt"
  soft "$P-inexor-$FD-card" icpu scripts/run/realization.py card --config "$P" \
    --workdir "$PROD" --k-steps $K $LAYOUT --fine-dtype "$FD" --heartbeat 60 \
    --tile-workers 8 --card-pool $mw
  if [ "$EXP" = export ]; then
    soft "$P-inexor-$FD-export" icpu scripts/run/realization.py export --config "$P" \
      --workdir "$PROD" --k-steps $K $LAYOUT --fine-dtype "$FD" --heartbeat 60
  fi
}
disco_export () {
  local P=$1
  soft "$P-disco-export" icpu $MT --out "$T/$P-disco-export.jsonl" -- \
    $DRV export --config "$P" --ic-dir "$WORK/$P/ics" --k-steps $K --out "$WORK/$P/disco_in.npz"
}
disco_arm () {
  local P=$1 NM=$2 C=$3 mw=$4
  local W=$WORK/$P TAG=disco-chunk-$3
  local chunk=(); [ "$C" = all ] || chunk=(--chunk-size "$C")
  soft "$P-$TAG" dpy $MT --out "$T/$P-$TAG.jsonl" -- \
    $DRV evolve --in "$W/disco_in.npz" --n-mesh "$NM" --precision single "${chunk[@]}" \
    --out "$W/$TAG.npz"
  if [ -f "$W/$TAG.npz" ]; then
    soft "$P-$TAG-card" icpu $DRV card --in "$W/$TAG.npz" -- --config "$P" \
      --k-steps $K --workdir "$W/card-$TAG" --coarse-match-order 3 $LAYOUT $mw
    # the unchunked final state is kept for the velocity comparison
    [ "$C" = all ] || rm -f "$W/$TAG.npz"
  fi
}

# ---- smoke: every new leg type at cdev8
inexor_ics cdev8
inexor_arm cdev8 float32 no "--min-weight 1"
disco_export cdev8
for c in $CHUNKS_cdev8; do disco_arm cdev8 256 "$c" "--min-weight 1"; done
[ $rc_total -eq 0 ] || { echo "FATAL: the smoke failed ($rc_total legs)"; exit 1; }

# ---- 256^3: DISCO-DJ only (Table 5's 256^3 row)
inexor_ics cdev
disco_export cdev
for c in $CHUNKS_cdev; do disco_arm cdev 512 "$c" "--min-weight 1"; done
rm -f "$WORK/cdev/disco_in.npz"

# ---- 512^3: both inexor precisions, then DISCO-DJ
inexor_ics cgh64
inexor_arm cgh64 float64 export ""
inexor_arm cgh64 float32 no ""
disco_export cgh64
for c in $CHUNKS_cgh64; do disco_arm cgh64 1024 "$c" ""; done
rm -f "$WORK/cgh64/disco_in.npz"

echo ""; echo "=== done $(date) rc_total=$rc_total ==="
du -sh "$WORK"/* 2>/dev/null
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,ExitCode -P 2>/dev/null
exit 0
