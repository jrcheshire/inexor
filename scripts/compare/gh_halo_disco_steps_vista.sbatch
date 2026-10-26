#!/bin/bash
#SBATCH -J gh-halo-disco-steps
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o gh-halo-disco-steps-%j.log
# DISCO-DJ's step and start-epoch arms at inexor's fine cell, on the ICs of a finished
# `gh_halo_check_vista.sbatch` run (cgh64-L128: 512^3 in 128 Mpc/h, 1024^3 mesh = the
# 0.125 Mpc/h fine cell): K = 160 and 240 from the z = 9 ICs, K = 160 from the z = 19 ICs.
# With that run's DISCO-DJ K = 120 arm these match every inexor arm of the halo check step for
# step, so halo shifts with step count or start epoch can be compared code against code.
# Each arm stops at the snapshot steps `run --snapshot-z` takes (`evolve --stop-z`), with a
# z = 0 card; then FoF (`halo_compare.py find`, b = 0.2) on every state, three at a time.
#
# Requires INEXOR_RUNS (run root), DISCO_SRC (disco-mocks checkout), HALO_WORK (the halo-check
# work dir holding cgh64-L128/ics and cgh64-L128/ics_z19) and INEXOR_SRC or submission from
# the inexor checkout. Optional: SNAPZ ("2.2,1,0.5", as the halo check). Account via -A:
#   INEXOR_RUNS=$SCRATCH/inexor_runs DISCO_SRC=$HOME/src/disco-mocks \
#     HALO_WORK=$SCRATCH/inexor_runs/gh-halo-check-1088222 \
#     sbatch -A <account> scripts/compare/gh_halo_disco_steps_vista.sbatch
#
# Legs: guards (clean checkouts, both envs, pinned jax/discodj, a GPU, the FoF env, the ICs)
# stop the job on failure; exports, evolves, cards and FoF record their exit code and continue.
# Wall, from gh-halo-check-1088222 (512^3 m1024 evolve ~40 s at K = 120, export ~20 s, card
# ~2 min, FoF ~2-3 min): guards ~5 min, 12 evolves ~12 min, 3 cards ~7, FoF 12 states ~12:
# ~40 min; 2 h requested.

set -u
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export CONDA_OVERRIDE_CUDA=12.0
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }

INEXOR=${INEXOR_SRC:-$SLURM_SUBMIT_DIR}
DISCO=${DISCO_SRC:?set DISCO_SRC to the disco-mocks checkout}
RUNS=${INEXOR_RUNS:?set INEXOR_RUNS to the run root, e.g. $SCRATCH/inexor_runs}
SRCW=${HALO_WORK:?set HALO_WORK to the gh_halo_check work dir whose ICs are reused}
[ -f "$INEXOR/src/inexor/engine.py" ] || { echo "FATAL: not an inexor checkout: $INEXOR"; exit 1; }
WORK=$RUNS/gh-halo-disco-steps-$SLURM_JOB_ID
HALOS=$WORK/halos
mkdir -p "$HALOS"

P=cgh64-L128
NM=1024
SNAPZ=${SNAPZ:-"2.2,1,0.5"}
LAYOUT="--slack 0.10 --alloc-margin 0.10 --arena-frac 0.05"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_CLIENT_MEM_FRACTION=0.95
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
unset JAX_PLATFORMS CUDA_VISIBLE_DEVICES XLA_PYTHON_CLIENT_MEM_FRACTION

BIND="numactl --membind=0"
DRV=scripts/compare/disco_crosscheck.py
HC=scripts/compare/halo_compare.py

echo "=== $(hostname) job $SLURM_JOB_ID $(date) ==="
git -C "$INEXOR" log --oneline -1
git -C "$DISCO" log --oneline -1
echo "HALO_WORK=$SRCW SNAPZ=$SNAPZ"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
free -g | head -2

rc_total=0
leg () {  # fatal
  local name="$1"; shift
  echo ""; echo "=== LEG $name start $(date +%s) ($(date +%H:%M:%S)) ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc end $(date +%s) ($(date +%H:%M:%S)) ---"
  [ $rc -eq 0 ] || { echo "FATAL: leg $name failed (rc=$rc)"; exit $rc; }
}
soft () {  # records rc and continues; returns it
  local name="$1"; shift
  echo ""; echo "=== LEG $name start $(date +%s) ($(date +%H:%M:%S)) ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc end $(date +%s) ($(date +%H:%M:%S)) ---"
  [ $rc -eq 0 ] || rc_total=$((rc_total + 1))
  return $rc
}

icpu () { (cd "$INEXOR" && JAX_PLATFORMS=cpu $BIND pixi run --frozen -e gpu python "$@"); }
dpy () { (cd "$INEXOR" && $BIND pixi run --frozen --manifest-path "$DISCO/pixi.toml" -e gpu python "$@"); }
# FoF env: kdcount is pip-only (a C extension), in a throwaway pixi env. CC/CXX are unset so
# the build uses the env's own gcc, not the module's (nvc rejects gcc's flags)
hpy () {
  (cd "$INEXOR" && env -u CC -u CXX pixi exec --spec python=3.12 --spec numpy --spec scipy \
    --spec pip --spec c-compiler -- bash -c 'pip install -q kdcount && exec python -I "$@"' _ "$@")
}

# ---- guards
clean_checkout () {
  git -C "$1" diff --quiet HEAD || { echo "FATAL: $1 has modified tracked files"; git -C "$1" status -s; return 1; }
  echo "$1 clean at $(git -C "$1" rev-parse --short HEAD)"
}
have_ics () {
  local d
  for d in "$SRCW/$P/ics" "$SRCW/$P/ics_z19"; do
    [ -f "$d/manifest.json" ] || { echo "FATAL: no ICs at $d"; return 1; }
    echo "ICs: $d ($(ls "$d" | wc -l) files)"
  done
}
leg checkout-inexor clean_checkout "$INEXOR"
leg checkout-disco clean_checkout "$DISCO"
leg ics have_ics
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
leg env-fof hpy -c "import kdcount, scipy, numpy; print('kdcount', kdcount.__file__, 'numpy', numpy.__version__)"

# ---- arms
# disco_arm TAG ICS K A_INIT: export the ICs and a-grid, then DISCO-DJ to z = 0 (with a card)
# and to each snapshot redshift
disco_arm () {
  local TAG=$1 ICS=$2 K=$3 AI=$4
  local IN=$WORK/$P/disco_in_$TAG.npz OUT=$WORK/$P/disco-$TAG-m$NM
  mkdir -p "$OUT"
  soft "$P-disco-export-$TAG" icpu $DRV export --config $P --ic-dir "$ICS" --k-steps "$K" \
    --a-init "$AI" --snapshot-z "$SNAPZ" --out "$IN" || return
  soft "$P-disco-$TAG-m$NM-z0" dpy $DRV evolve --in "$IN" --n-mesh $NM --precision single \
    --out "$OUT/z0.npz" \
    && soft "$P-disco-$TAG-m$NM-card" icpu $DRV card --in "$OUT/z0.npz" -- --config $P \
      --k-steps "$K" --workdir "$OUT/card" --coarse-match-order 3 $LAYOUT
  local Z
  for Z in ${SNAPZ//,/ }; do
    soft "$P-disco-$TAG-m$NM-z$Z" dpy $DRV evolve --in "$IN" --n-mesh $NM --precision single \
      --stop-z "$Z" --out "$OUT/z$Z.npz"
  done
  rm -f "$IN"
}
disco_arm k160 "$SRCW/$P/ics" 160 0.1
disco_arm k240 "$SRCW/$P/ics" 240 0.1
disco_arm z19-k160 "$SRCW/$P/ics_z19" 160 0.05

# ---- FoF on every state, three at a time
echo ""; echo "=== FoF ($(date +%H:%M:%S)) ==="
NAMES=()
for d in "$WORK/$P"/disco-*/; do
  d=${d%/}
  for Z in 0 ${SNAPZ//,/ }; do
    [ -f "$d/z$Z.npz" ] || continue
    name=$P-$(basename "$d")-z$Z
    while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
    ( soft "fof-$name" hpy $HC find "$d/z$Z.npz" --n-mesh 256 --out "$HALOS/$name.npz" ) \
      > "$HALOS/$name.log" 2>&1 &
    NAMES+=("$name")
  done
done
wait
for name in "${NAMES[@]}"; do
  grep -h -E "^(\[find\]|--- LEG)" "$HALOS/$name.log"
  [ -f "$HALOS/$name.npz" ] || rc_total=$((rc_total + 1))
done

echo ""; echo "=== done $(date) rc_total=$rc_total ==="
ls "$HALOS"
du -sh "$WORK"/* 2>/dev/null
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,ExitCode -P 2>/dev/null
exit 0
