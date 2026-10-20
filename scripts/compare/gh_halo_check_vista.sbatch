#!/bin/bash
#SBATCH -J gh-halo-check
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05:00:00
#SBATCH -o gh-halo-check-%j.log
# FoF halos at 0.25 Mpc/h particle spacing (0.125 Mpc/h fine cell), on ONE GH200, at
# z = 0 and the snapshot redshifts SNAPZ:
#   L128  cgh64-L128 (512^3 in 128 Mpc/h; c-hero-L1024 in every resolution knob):
#         inexor from z = 9 ICs at K = 120, 160, 240 and from z = 19 ICs at K = 160 (step and
#         start-epoch convergence of P(k) and of halos); DISCO-DJ from the z = 9 ICs at K = 120
#         on a 1024^3 mesh (inexor's fine cell): the codes compared.
#   L64   cdev-L64 (256^3 in 64 Mpc/h): inexor at K = 120; DISCO-DJ at K = 120 on 512^3 (the
#         same cell) and 1024^3 (half the cell) meshes, and at K = 240 on 1024^3: what the
#         0.125 Mpc/h cell loses against a finer force, and whether that reference is
#         step-converged.
# Every inexor arm writes synchronized snapshots (`run --snapshot-z`) and exports each epoch;
# DISCO-DJ stops at the same step boundaries (`disco_crosscheck.py evolve --stop-z`). Then FoF
# (`halo_compare.py find`, b = 0.2) on every state, three at a time; `halo_compare.py compare`
# on the catalogs is left to the reader. Run cards keep each arm's arena and memory readings.
#
# Requires INEXOR_RUNS (run root), DISCO_SRC (disco-mocks checkout) and INEXOR_SRC or
# submission from the inexor checkout. Optional: SNAPZ ("2.2,1,0.5"), ARENA_FRAC (0.05). The
# account is passed with `sbatch -A`:
#   INEXOR_RUNS=$SCRATCH/inexor_runs DISCO_SRC=$HOME/src/disco-mocks \
#     sbatch -A <account> scripts/compare/gh_halo_check_vista.sbatch
#
# Legs (guards and the smoke are fatal; everything after records rc and continues):
#   guards    clean checkouts; inexor's gpu env (scripts/run/gpu_env.sh); disco-mocks' env from
#             its lock; pinned jax/discodj; a GPU; the FoF env (kdcount)
#   smoke     cdev8 (128^3): inexor run with one snapshot, exports, DISCO-DJ export and two
#             evolves, FoF on all four, a compare, and the periodic self-test
#   L128      ICs (z = 9, z = 19); four inexor arms; DISCO-DJ export and four evolves
#   L64       ICs; one inexor arm; DISCO-DJ two exports and twelve evolves
#   fof       every state; catalogs in $WORK/halos/
# Wall, estimated from gh-same-ics (512^3: ~7 s/step, export 14 s, card 30 s; DISCO-DJ evolve
# ~50 s): guards ~5 min, smoke ~8, L128 inexor 640 steps ~80-100 min plus 16 exports and 4
# cards ~10, DISCO-DJ ~10, L64 ~15, FoF 16 states of 512^3 at ~2.5 min three at a time ~15:
# ~2.5 h; 5 h requested.

set -u
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export CONDA_OVERRIDE_CUDA=12.0
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }

INEXOR=${INEXOR_SRC:-$SLURM_SUBMIT_DIR}
DISCO=${DISCO_SRC:?set DISCO_SRC to the disco-mocks checkout}
RUNS=${INEXOR_RUNS:?set INEXOR_RUNS to the run root, e.g. $SCRATCH/inexor_runs}
[ -f "$INEXOR/src/inexor/engine.py" ] || { echo "FATAL: not an inexor checkout: $INEXOR"; exit 1; }
WORK=$RUNS/gh-halo-check-$SLURM_JOB_ID
HALOS=$WORK/halos
mkdir -p "$HALOS"

SNAPZ=${SNAPZ:-"2.2,1,0.5"}
ARENA=${ARENA_FRAC:-0.05}
LAYOUT="--slack 0.10 --alloc-margin 0.10 --arena-frac $ARENA"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_CLIENT_MEM_FRACTION=0.95
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
# JAX's default host-memory cap is 64 GB, below what inexor's host streaming needs
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=160
unset JAX_PLATFORMS CUDA_VISIBLE_DEVICES XLA_PYTHON_CLIENT_MEM_FRACTION

BIND="numactl --membind=0"
MEMBIND="--membind-nodes 0"
DRV=scripts/compare/disco_crosscheck.py
DEV=scripts/run/device_run.py
REAL=scripts/run/realization.py
HC=scripts/compare/halo_compare.py

echo "=== $(hostname) job $SLURM_JOB_ID $(date) ==="
git -C "$INEXOR" log --oneline -1
git -C "$DISCO" log --oneline -1
echo "SNAPZ=$SNAPZ LAYOUT=$LAYOUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
free -g | head -2
env | grep -E '^(XLA_|JAX_)' | sort

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

ipy () { (cd "$INEXOR" && $BIND pixi run --frozen -e gpu python "$@"); }
icpu () { (cd "$INEXOR" && JAX_PLATFORMS=cpu $BIND pixi run --frozen -e gpu python "$@"); }
dpy () { (cd "$INEXOR" && $BIND pixi run --frozen --manifest-path "$DISCO/pixi.toml" -e gpu python "$@"); }
# FoF env: kdcount is pip-only (a C extension), in a throwaway pixi env
hpy () {
  (cd "$INEXOR" && pixi exec --spec python=3.12 --spec numpy --spec scipy --spec pip \
    --spec c-compiler -- bash -c 'pip install -q kdcount && exec python -I "$@"' _ "$@")
}

# ---- guards
clean_checkout () {
  git -C "$1" diff --quiet HEAD || { echo "FATAL: $1 has modified tracked files"; git -C "$1" status -s; return 1; }
  echo "$1 clean at $(git -C "$1" rev-parse --short HEAD)"
}
leg checkout-inexor clean_checkout "$INEXOR"
leg checkout-disco clean_checkout "$DISCO"
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
leg env-fof hpy -c "import kdcount, scipy, numpy; print('kdcount', kdcount.__file__, 'numpy', numpy.__version__)"

# ---- helpers
# card bins: the small boxes have too few modes per bin for the default minimum weight
mw () { case $1 in cdev8|cdev-L64) echo "--min-weight 1" ;; esac; }
# snapshot_step RUN_CARD Z: the step `run --snapshot-z` took for Z
snapshot_step () {
  python3 -c "import json, sys; c = json.load(open(sys.argv[1]))
print([s['step'] for s in c['snapshots'] if abs(s['z'] - float(sys.argv[2])) < 1e-9][0])" "$1" "$2"
}
# inexor_arm PRESET ICS TAG K A_INIT: run with snapshots, then the z = 0 card and every export
inexor_arm () {
  local P=$1 ICS=$2 TAG=$3 K=$4 AI=$5
  local W=$WORK/$P/$TAG
  mkdir -p "$W/prod"
  soft "$P-$TAG-preflight" ipy $DEV preflight --preset "$P" --workdir "$ICS" --cards 1 \
    --card "$W/preflight.json" --scratch "$RUNS" $MEMBIND $LAYOUT || return
  soft "$P-$TAG-run" ipy $DEV run --preset "$P" --workdir "$ICS" --cards 1 --card "$W/run.json" \
    --k-steps "$K" --stop-at "$K" --timed-every 10 $MEMBIND --beat 60 $LAYOUT \
    --checkpoint-dir "$W/ckpt" --checkpoint-every "$K" \
    --snapshot-z "$SNAPZ" --snapshot-dir "$W/snaps" || return
  ln -sfn "$W/ckpt" "$W/prod/ckpt"
  soft "$P-$TAG-card" icpu $REAL card --config "$P" --workdir "$W/prod" --k-steps "$K" \
    --a-init "$AI" $LAYOUT $(mw "$P") --heartbeat 60 --tile-workers 8 --card-pool
  soft "$P-$TAG-export-z0" icpu $REAL export --config "$P" --workdir "$W/prod" --k-steps "$K" \
    --a-init "$AI" $LAYOUT --heartbeat 60
  local Z S
  for Z in ${SNAPZ//,/ }; do
    S=$(snapshot_step "$W/run.json" "$Z") || { rc_total=$((rc_total + 1)); continue; }
    mkdir -p "$W/z$Z"
    ln -sfn "$W/snaps/step$(printf %04d "$S")" "$W/z$Z/ckpt"
    soft "$P-$TAG-export-z$Z" icpu $REAL export --config "$P" --workdir "$W/z$Z" \
      --k-steps "$K" --a-init "$AI" $LAYOUT --heartbeat 60
  done
}
# disco_export PRESET ICS K: the shared ICs and a-grid, with the snapshot steps
disco_export () {
  local P=$1 ICS=$2 K=$3
  soft "$P-disco-export-k$K" icpu $DRV export --config "$P" --ic-dir "$ICS" --k-steps "$K" \
    --snapshot-z "$SNAPZ" --out "$WORK/$P/disco_in_k$K.npz"
}
# disco_arm PRESET K N_MESH: DISCO-DJ to z = 0 (with a card) and to each snapshot redshift
disco_arm () {
  local P=$1 K=$2 NM=$3
  local IN=$WORK/$P/disco_in_k$K.npz OUT=$WORK/$P/disco-k$K-m$NM
  [ -f "$IN" ] || { echo "no $IN: skipping DISCO-DJ k$K m$NM"; rc_total=$((rc_total + 1)); return; }
  mkdir -p "$OUT"
  soft "$P-disco-k$K-m$NM-z0" dpy $DRV evolve --in "$IN" --n-mesh "$NM" --precision single \
    --out "$OUT/z0.npz" \
    && soft "$P-disco-k$K-m$NM-card" icpu $DRV card --in "$OUT/z0.npz" -- --config "$P" \
      --k-steps "$K" --workdir "$OUT/card" --coarse-match-order 3 $LAYOUT $(mw "$P")
  local Z
  for Z in ${SNAPZ//,/ }; do
    soft "$P-disco-k$K-m$NM-z$Z" dpy $DRV evolve --in "$IN" --n-mesh "$NM" --precision single \
      --stop-z "$Z" --out "$OUT/z$Z.npz"
  done
}
# every evolved state of an arm, "name src" per line, for the FoF pass
states () {
  local P=$1 Z
  for d in "$WORK/$P"/*/; do
    d=${d%/}
    case $(basename "$d") in
      disco-*) for Z in 0 ${SNAPZ//,/ }; do
                 [ -f "$d/z$Z.npz" ] && echo "$P-$(basename "$d")-z$Z $d/z$Z.npz"; done ;;
      *) [ -f "$d/prod/export/export.json" ] && echo "$P-$(basename "$d")-z0 $d/prod/export"
         for Z in ${SNAPZ//,/ }; do
           [ -f "$d/z$Z/export/export.json" ] && echo "$P-$(basename "$d")-z$Z $d/z$Z/export"
         done ;;
    esac
  done
}

# ---- smoke: every leg type at cdev8 (128^3, 0.5 Mpc/h spacing), one snapshot
SNAPZ_RUN=$SNAPZ
SNAPZ=1
leg cdev8-ics ipy $REAL ics --config cdev8 --workdir "$WORK/cdev8/ics" --generator device
inexor_arm cdev8 "$WORK/cdev8/ics" k120 120 0.1
disco_export cdev8 "$WORK/cdev8/ics" 120
disco_arm cdev8 120 256
[ $rc_total -eq 0 ] || { echo "FATAL: the smoke failed ($rc_total legs)"; exit 1; }
leg cdev8-fof-inexor hpy $HC find "$WORK/cdev8/k120/z1/export" --n-mesh 64 --out "$HALOS/cdev8-k120-z1.npz"
leg cdev8-fof-disco hpy $HC find "$WORK/cdev8/disco-k120-m256/z1.npz" --n-mesh 64 --out "$HALOS/cdev8-disco-z1.npz"
leg cdev8-fof-shift hpy $HC find "$WORK/cdev8/k120/z1/export" --n-mesh 64 --shift --out "$HALOS/cdev8-k120-z1-shift.npz"
leg cdev8-compare hpy $HC compare "$HALOS/cdev8-k120-z1.npz" "$HALOS/cdev8-disco-z1.npz" \
  --out "$HALOS/cdev8-compare-z1.json"
leg cdev8-self-test hpy $HC compare "$HALOS/cdev8-k120-z1.npz" "$HALOS/cdev8-k120-z1-shift.npz" \
  --labels plain shifted
SNAPZ=$SNAPZ_RUN

# ---- L128: 512^3 in 128 Mpc/h
P=cgh64-L128
soft "$P-ics-z9" ipy $REAL ics --config $P --workdir "$WORK/$P/ics" --generator device
soft "$P-ics-z19" ipy $REAL ics --config $P --workdir "$WORK/$P/ics_z19" --generator device \
  --a-init 0.05
disco_export $P "$WORK/$P/ics" 120
disco_arm $P 120 1024
rm -f "$WORK/$P/disco_in_k120.npz"
inexor_arm $P "$WORK/$P/ics" k120 120 0.1
inexor_arm $P "$WORK/$P/ics" k160 160 0.1
inexor_arm $P "$WORK/$P/ics" k240 240 0.1
inexor_arm $P "$WORK/$P/ics_z19" z19-k160 160 0.05

# ---- L64: 256^3 in 64 Mpc/h
P=cdev-L64
soft "$P-ics-z9" ipy $REAL ics --config $P --workdir "$WORK/$P/ics" --generator device
inexor_arm $P "$WORK/$P/ics" k120 120 0.1
disco_export $P "$WORK/$P/ics" 120
disco_export $P "$WORK/$P/ics" 240
disco_arm $P 120 512
disco_arm $P 120 1024
disco_arm $P 240 1024
rm -f "$WORK/$P"/disco_in_k*.npz

# ---- FoF on every state, three at a time
echo ""; echo "=== FoF ($(date +%H:%M:%S)) ==="
NAMES=()
for P in cgh64-L128 cdev-L64; do
  while read -r name src; do
    while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
    ( soft "fof-$name" hpy $HC find "$src" --n-mesh 256 --out "$HALOS/$name.npz" ) \
      > "$HALOS/$name.log" 2>&1 &
    NAMES+=("$name")
  done < <(states $P)
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
