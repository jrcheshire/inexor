#!/bin/bash
#SBATCH -J d7g-hero-export
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05:00:00
#SBATCH -o d7g-hero-export-%j.log
# The portable (x, v) export of an evolved 4096^3 realization's final checkpoint, and
# nothing else. The checkpoint is the engine's own T9 encoding; this writes it as two
# plain float32 .npy arrays (~1.65 TB) that any halo finder or analysis code can mmap.
# Independent of the card job (hero_card_vista.sbatch): same checkpoint, neither reads the
# other's output, so they may run concurrently or in either order.
#
# Requires INEXOR_RUNS (run root), SRC_RUN (the realization directory holding
# hero-ckpt/), K_STEPS (schedule length; the newest checkpoint must be at this step),
# PROD_DIR (product directory; the export goes to $PROD_DIR/export), and INEXOR_SRC or
# submission from the inexor checkout. The account is passed with `sbatch -A`:
#   INEXOR_RUNS=$SCRATCH/inexor_runs SRC_RUN=$SCRATCH/inexor_runs/hero-k120 K_STEPS=120 \
#     PROD_DIR=$SCRATCH/inexor_runs/hero-k120-prod \
#     sbatch -A <account> scripts/run/hero_export_vista.sbatch
#
# Wall: legs ~2 min; loading the 687 GB state ~28 min; then max(decode, write), which the
# writer thread overlaps: decode 16,777,216 bricks at ~366 us/brick on Grace = ~1.7 h
# (2.5x slower than on the laptop, and read at +-25%), write 1.65 TB at 706-807 MB/s
# = ~35 min. ~2.2 h central, decode-bound; 5 h requested for the uncertain decode rate.
#
# Layout parameters must match the run, not the script defaults (see
# hero_card_vista.sbatch): slack 0.10 / arena-frac 0.01. Expected host peak ~690-760 GB.
# Host memory is bound to the CPU sockets (`numactl --membind=0,1`). The export must be
# written to $SCRATCH and kept there: the state and export do not fit anywhere else.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
# Run from the inexor checkout: INEXOR_SRC, or the directory sbatch was called from.
cd "${INEXOR_SRC:-$SLURM_SUBMIT_DIR}" && [ -f src/inexor/engine.py ] || {
  echo "FATAL: not an inexor checkout: ${INEXOR_SRC:-$SLURM_SUBMIT_DIR} (set INEXOR_SRC)"; exit 1; }
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
export INEXOR_LOAD_TRACE=1
unset JAX_PLATFORMS
unset CUDA_VISIBLE_DEVICES

RUNS=${INEXOR_RUNS:?set INEXOR_RUNS to the run root, e.g. $SCRATCH/inexor_runs}
SRC=${SRC_RUN:?set SRC_RUN to the realization directory}
K_STEPS=${K_STEPS:?set K_STEPS to the schedule length}
export K_STEPS
WORK=$RUNS/d7g-$SLURM_JOB_ID
PROD=${PROD_DIR:?set PROD_DIR to the product directory}
CARDS=runs/v2
DRIVER=scripts/run/device_run.py
TAG=d7g_$SLURM_JOB_ID
MEM_CSV=$CARDS/${TAG}_mem.csv
BIND="numactl --membind=0,1"
CPU="env JAX_PLATFORMS=cpu"
LAYOUT="--slack 0.10 --arena-frac 0.01"
HB="--heartbeat 60"
mkdir -p "$WORK" "$PROD"
export PROD

ln -sfn "$SRC/hero-ckpt" "$PROD/ckpt"
# the newest generation must be at the schedule's last step; which generation holds it
# depends on the segmentation, so it is read, not assumed
HEAD_MAN=$(python3 -c "
import json, os
best = None
for g in (0, 1):
    p = os.path.join('$PROD/ckpt', 'gen%d' % g, 'manifest.json')
    if os.path.exists(p):
        s = json.load(open(p))['provenance']['step']
        if best is None or s > best[0]:
            best = (s, p)
print(best[1] if best and best[0] == $K_STEPS else '')
")
[ -n "$HEAD_MAN" ] || { echo "FATAL: no step-$K_STEPS manifest under $PROD/ckpt"; exit 1; }
export HEAD_MAN
# An export directory left by a killed run holds full-looking, short arrays. The header is
# written last and removed first, so a directory without it is that case; refuse.
if [ -e "$PROD/export" ] && [ ! -e "$PROD/export/export.json" ]; then
  echo "FATAL: $PROD/export exists without a header -- an interrupted export."
  echo "       Move or remove it deliberately; this job will not write into it."
  exit 1
fi

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
free -g | head -2
df -h "$RUNS" | tail -1
python3 -c "import json;d=json.load(open('$HEAD_MAN'));print('source checkpoint:', d['provenance']['step'], 'of', d['provenance']['n_steps'])"
numactl --hardware | head -4

( while true; do t=$(date +%s.%N)
    for d in /sys/devices/system/node/node[0-9]*; do
      n=${d##*node}; c=0; [ -n "$(tr -d '[:space:]' < $d/cpulist)" ] && c=1
      awk -v t="$t" -v n="$n" -v c="$c" '/MemTotal:/{tot=$4} /MemFree:/{fr=$4}
        /FilePages:/{fp=$4} / Dirty:/{di=$4} / Writeback:/{wb=$4}
        END{print t "," n "," c "," tot "," fr "," fp "," di "," wb}' $d/meminfo
    done
    sleep 5; done ) > "$MEM_CSV" &
MEM_SAMPLER=$!

rc_total=0
finish () {
  kill $MEM_SAMPLER 2>/dev/null
  echo ""; echo "=== products ==="
  du -sh "$PROD/export" 2>/dev/null
  ls -la "$PROD/export" 2>/dev/null | head
  ls -la "$PROD"/realization_export*.json 2>/dev/null
  df -h "$RUNS" | tail -1
  echo "=== sacct ==="
  sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,ExitCode -P 2>/dev/null
  echo "=== end $(date) rc_total=$rc_total ==="
}
trap finish EXIT

run_leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ($(date +%H:%M:%S)) ==="
  "$@" & local pid=$!
  local rc; wait $pid; rc=$?
  echo "--- LEG $name rc=$rc ($(date +%H:%M:%S)) ---"
  [ $rc -ne 0 ] && rc_total=$((rc_total + 1))
  return $rc
}

# Fingerprint guard, before anything expensive: the fingerprint hashes QUADPACK-derived
# coefficients, so it depends on the node's libm (see hero_card_vista.sbatch).
run_leg fingerprint-guard \
  $CPU pixi run -e gpu python -c '
import sys, json, argparse
sys.path.insert(0, "scripts/run")
import jax; jax.config.update("jax_enable_x64", True)
from realization import _cosmo, _coeffs, _engine_config, _geom
from inexor.engine import checkpoint_fingerprint
import os
args = argparse.Namespace(slack=0.10, tile_workers=16, checkpoint_every=10,
                          migrate_pooled=None, eject_kernel="jax")
ec = _engine_config(_geom("c-hero"), args, os.environ["PROD"] + "/ckpt")
co, _ = _coeffs(_cosmo(), int(os.environ["K_STEPS"]))
want = checkpoint_fingerprint(ec, co)
have = json.load(open(os.environ["HEAD_MAN"]))["provenance"]["fingerprint"]
print("fingerprint want", want, "\nfingerprint have", have)
sys.exit(0 if want == have else 1)
' \
  || { echo "FATAL: this node computes a different schedule than the one that wrote the checkpoint"; exit 1; }

run_leg pytest-gpu \
  pixi run -e gpu python -m pytest -q -rs tests/test_export.py tests/test_progress.py \
  || { echo "FATAL: the export gates failed"; exit 1; }

# --- cgh64: the export end to end, on the same filesystem, before the hero leg. Step 3
# suffices: the export has no epoch-dependent branch (unlike the card), and three steps is
# the cheapest run that produces a checkpoint.
run_leg cgh64-ics \
  $BIND pixi run -e gpu python scripts/run/realization.py ics --config cgh64 \
    --workdir "$WORK/cgh64" --generator device \
  && run_leg cgh64-run \
  $BIND pixi run -e gpu python $DRIVER run --preset cgh64 --workdir "$WORK/cgh64" \
    --card $CARDS/${TAG}_cgh64.json --stop-at 3 --membind-nodes 0,1 --beat 60 \
    --checkpoint-dir "$WORK/cgh64-ckpt" --checkpoint-every 3
[ $? -ne 0 ] && { echo "FATAL: the cgh64 control failed"; exit 1; }

mkdir -p "$WORK/cgh64-prod"
ln -sfn "$WORK/cgh64-ckpt" "$WORK/cgh64-prod/ckpt"

run_leg cgh64-export \
  $CPU $BIND pixi run -e gpu python scripts/run/realization.py export --config cgh64 \
    --workdir "$WORK/cgh64-prod" $LAYOUT $HB --allow-partial \
    --export-dir "$WORK/cgh64-prod/export" \
  || { echo "FATAL: the export path is broken at cgh64"; exit 1; }
du -sh "$WORK/cgh64-prod/export"

# --- The 4096^3 export. 1.65 TB, decode-bound, one call.
run_leg hero-export \
  $CPU $BIND pixi run -e gpu python scripts/run/realization.py export --config c-hero \
    --workdir "$PROD" --k-steps "$K_STEPS" $LAYOUT $HB --export-dir "$PROD/export"
rc_export=$?

echo ""
echo "=== export rc=$rc_export ==="
exit $rc_total
