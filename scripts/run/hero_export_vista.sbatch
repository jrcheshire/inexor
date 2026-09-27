#!/bin/bash
#SBATCH -J d7g-hero-export
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05:00:00
#SBATCH -o d7g-hero-export-%j.log
# D7g: the portable (x, v) export off gb 1009133's step-40 checkpoint, and
# NOTHING ELSE. Sibling of v2_d7f_hero_card, split out of the combined D7e job
# that gb 1010938 was cancelled from: the export does not depend on the card,
# reads the same checkpoint, and queueing it behind a card that overran cost it
# its whole wall without producing a byte.
#
# The two are INDEPENDENT and may run at the same time (gb allows two running
# jobs per user) or in either order. Neither reads the other's output.
#
# 1009133's checkpoint is the engine's own T9 encoding, readable by nothing
# outside this package; this is what turns it into two plain float32 .npy
# arrays that any halo finder or analysis code can mmap. ~1.65 TB.
#
# TIME: measured terms x loop counts, with the decode rate taken ON GRACE.
#   legs (fingerprint, pytest, cgh64 ICs, cgh64 run to 3) .......   95 s  measured, 1010730
#   cgh64 export ................................................   12 s  measured, 1010938
#   hero: load the 687 GB state ................................. 1663 s  measured, 1010938
#   hero: max(decode, write), the two being overlapped by the
#     writer thread (7762c89):
#       decode 16,777,216 bricks @ 366 us ........................ 6140 s  see below
#       write 1.65 TB @ 706-807 MB/s ............................. 2100 s  measured, 1009133
#   CENTRAL 2.2 h.  REQUESTED 5:00.  ~2.5 SU at the central.
#   THE DECODE RATE IS THE ONE TO WATCH AND IT IS NOT THE LAPTOP'S. D7e priced
#   it at 133-149 us/brick measured on the M4; 1010938's cgh64 export did 32,768
#   bricks in 12 s = 366 us/brick on Grace at the SAME 4,096 particles/brick,
#   2.5x slower, and that factor is the same class of miss that made D7e's card
#   estimate 6-9x optimistic. The 12 s reading is +-3 s (the leg prints 0.1 min
#   resolution), so treat 366 us as an order, not a rate -- which is why the
#   request is 5:00 against a 2.2 h central rather than 3:00.
#   The write is comfortably the smaller term either way, so the overlap the
#   writer thread buys is worth its whole cost and the leg is DECODE-bound.
#
# THE LAYOUT PARAMETERS ARE NOT THE SCRIPT'S DEFAULTS AND MUST NOT BE. See
# v2_d7f_hero_card for the full note; 1009133 ran slack 0.10 / arena-frac 0.01.
# D7e predicted the export leg peaks ~688 GB (0.67x); the card leg measured 11%
# over its own prediction on 1010938, so expect nearer 760 GB and still clear.
#
# DISK: 1.65 TB into $SCRATCH, which had 2.5 PB free of 13 PB. The $SCRATCH copy
# of gen1 must OUTLIVE this job -- sirius is a 64 GB storage server and is an
# archive target only, so it cannot hold either the state or the export.
#
# HOST MEMORY IS BOUND TO THE CPU SOCKETS, as in every D7 job since 1003511.
#
# PRE-REGISTERED QUESTIONS:
#   1. What does the export cost in parts -- `decode` against `write thread` --
#      and does the depth-one overlap hold on Grace? That writer is new code
#      (7762c89) and has never run above cgh64.
#   2. What is NFS's rate for two sequential 825 GB streams, against the
#      706-807 MB/s the checkpoint's 256 x 2.7 GB files measured?
#   3. Is the decode really 366 us/brick at hero scale, or did the cgh64 reading
#      carry fixed overhead? The heartbeat answers this in the first minute.
#
# Submit from the Vista checkout:
#   sbatch scripts/run/hero_export_vista.sbatch

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
# SRC_RUN / PROD_DIR / K_STEPS override the run this reads; the defaults are 1009133's
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
# the NEWEST complete generation must be the schedule's last step; which generation
# holds it depends on how the run was segmented, so it is read, not assumed
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
# An export directory left by a killed run holds full-looking, short arrays. The
# header is written LAST and removed FIRST, so its absence beside data files is
# exactly that case; refuse rather than write into it.
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

# THE FINGERPRINT GUARD, before anything expensive. Same contract and same
# reason as v2_d7f_hero_card: the fingerprint hashes coefficients out of
# QUADPACK, so it rides on the node's libm and can only be settled here.
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

# --- cgh64: the export end to end, on the same NFS, before the hero leg.
# STEP 3, not 24: the export is epoch-BLIND -- it decodes and writes whatever
# the checkpoint holds, and no branch in it depends on the scale factor. (The
# card is the opposite, which is why its job steps to 24; see v2_d7f.) Three
# steps is the cheapest thing that produces a checkpoint to read.
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

# --- THE 4096^3 EXPORT. 1.65 TB, decode-bound, one call.
run_leg hero-export \
  $CPU $BIND pixi run -e gpu python scripts/run/realization.py export --config c-hero \
    --workdir "$PROD" --k-steps "$K_STEPS" $LAYOUT $HB --export-dir "$PROD/export"
rc_export=$?

echo ""
echo "=== export rc=$rc_export ==="
exit $rc_total
