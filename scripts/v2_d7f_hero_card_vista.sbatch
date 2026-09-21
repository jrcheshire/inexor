#!/bin/bash
#SBATCH -J d7f-hero-card
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d7f-hero-card-%j.log
# D7f: the P(k) card off gb 1009133's step-40 checkpoint, and NOTHING ELSE.
# Successor to v2_d7e_hero_product_vista.sbatch, which paired the card with the
# export in one 5 h job. gb 1010938 ran that pairing and was cancelled 3 h 56 min
# in with the card still inside `pk_summary_card` and the export leg never
# reached. Two changes came out of it and both are here.
#
# (1) THE CARD IS ITS OWN JOB. The export does not depend on the card and reads
#     the same checkpoint, so queueing it behind a card that might overrun buys
#     nothing and costs the export its whole wall. See v2_d7g_hero_export.
#
# (2) THE WALL COMES FROM THE cgh64 ANCHOR, NOT FROM THE PARTS. D7e's estimate
#     was built bottom-up -- decode 2366 s from a rate measured ON THE LAPTOP,
#     plus carries for the paint and the transform -- and came to 56 min against
#     an actual that passed 3 h 23 min without finishing. The anchor that was in
#     the same log all along: the cgh64 card took 60 s ON THIS NODE, and c-hero
#     is exactly 512x it in both particles and coarse cells.
#
# TIME: the cgh64 anchor, scaled, with the load measured.
#   legs (fingerprint, pytest, cgh64 ICs, cgh64 run to 24) ......  196 s  measured, 1010730/938
#   cgh64 card ..................................................   60 s  measured, 1010938
#   hero: load the 687 GB state ................................. 1663 s  measured, 1010938
#   hero: the card itself, 512x the cgh64 anchor ................ 30720 s  8.5 h
#   CENTRAL 9.0 h.
#   The 512x is a FLOOR on two of the three stages and short on the third: the
#   paint is per brick and bricks go 32,768 -> 16,777,216 (512x exactly), the
#   binning is per mode (512x), but the transform carries an extra log N and
#   scales ~700x, which puts the upper end at 11.7 h + load = 12.2 h.
#   REQUESTED 12:00, which is gb's MaxWall and the most this partition allows.
#   ~12 SU. So this run MAY NOT FINISH, and that is a stated bet rather than a
#   padded estimate: what makes it worth taking is that a non-finish is now
#   informative. 1010938 could not say which stage it was in or how far; every
#   loop in the card now reports through `inexor.progress.Heartbeat` and this
#   job prints the per-stage rate at cgh64 BEFORE the hero leg starts, so even a
#   killed hero leg leaves measured hero-scale rates for all three stages.
#   If it does not finish, the lever is not a bigger wall -- 12 h is the cap --
#   it is pooling the paint loop, which `coarse_delta_streamed` already supports
#   via `pool=` and the card does not use.
#
# THE LAYOUT PARAMETERS ARE NOT THE SCRIPT'S DEFAULTS AND MUST NOT BE.
# `load_checkpoint` re-derives the allocation geometry from the occupancy plus
# `--slack` / `--arena-frac` (the checkpoint preserves no geometry, D-v2-19).
# 1009133 ran slack 0.10 / arena-frac 0.01 -> 83.85e9 slots. The script's own
# defaults of 0.20 / 0.20 would provision ~13.7e9 arena rows instead of 687M.
# MEASURED on 1010938 with these values: the card leg peaks 839.5 GB of 1026
# (0.82x), against the 756 GB D7e predicted -- 11% over, and still clear.
#
# HOST MEMORY IS BOUND TO THE CPU SOCKETS, as in every D7 job since 1003511.
#
# PRE-REGISTERED QUESTIONS:
#   1. What are the three stages' rates at hero scale, and how do they compare
#      with 512x the cgh64 rates? This is the first time any of them is measured
#      above cgh64, and it is what prices every later card.
#   2. Does the card finish inside 12 h, and if not, at what fraction?
#   3. What does the card say below k_nonlinear (0.2520 h/Mpc at a = 1)? NO
#      verdict is emitted: the band this run is trusted over is JC's call.
#
# Submit from the Vista checkout:
#   sbatch scripts/v2_d7f_hero_card_vista.sbatch

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
export INEXOR_LOAD_TRACE=1
unset JAX_PLATFORMS
unset CUDA_VISIBLE_DEVICES

RUNS=/scratch/10303/jrc4/inexor_runs
SRC=$RUNS/d7-1009133                 # the run whose step-40 state this reads
WORK=$RUNS/d7f-$SLURM_JOB_ID
PROD=$RUNS/d7-1009133-product
CARDS=runs/v2
DRIVER=scripts/v2_d7_hero_smoke.py
TAG=d7f_$SLURM_JOB_ID
MEM_CSV=$CARDS/${TAG}_mem.csv
BIND="numactl --membind=0,1"
# `_require_cpu` refuses unless jax's first device is CPU, and on this node it is
# not. Scoped per leg rather than exported, so the pytest leg still sees the cards.
CPU="env JAX_PLATFORMS=cpu"
LAYOUT="--slack 0.10 --arena-frac 0.01"
# 60 s between progress lines: at the cgh64 rates that is ~30 lines over a 9 h
# hero leg, and at 16,384 chunks the throttle is a clock read per chunk against
# a chunk costing ~0.4 s.
HB="--heartbeat 60"
mkdir -p "$WORK" "$PROD"
export PROD

# `_ckpt_dir` is `<workdir>/ckpt` and 1009133 wrote `hero-ckpt`, so point one at
# the other rather than moving 1.3 TB or touching the source job's directory.
ln -sfn "$SRC/hero-ckpt" "$PROD/ckpt"
[ -e "$PROD/ckpt/gen1/manifest.json" ] || { echo "FATAL: no step-40 manifest under $PROD/ckpt"; exit 1; }

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
free -g | head -2
df -h "$RUNS" | tail -1
python3 -c "import json;d=json.load(open('$PROD/ckpt/gen1/manifest.json'));print('source checkpoint:', d['provenance']['step'], 'of', d['provenance']['n_steps'])"
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
  ls -la "$PROD"/realization_pk*.json 2>/dev/null
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

# THE FINGERPRINT GUARD, before anything expensive.
# `load_checkpoint` refuses a fingerprint mismatch, and the fingerprint hashes
# `coeffs.tobytes()` -- which come from `scipy.integrate.quad` (QUADPACK), so the
# bytes ride on the platform's libm. It is a SAME-MACHINE contract: the same
# snippet on the osx-arm64 laptop produces a different digest against this
# checkpoint's c28f7507..., with all fifteen fingerprinted CONFIG fields
# identical. Verified matching on c672-003 and c672-017; kept because the
# contract is per node, not per cluster, and it costs one import.
run_leg fingerprint-guard \
  $CPU pixi run -e gpu python -c '
import sys, json, argparse
sys.path.insert(0, "scripts")
import jax; jax.config.update("jax_enable_x64", True)
from v2_m6_realization import _cosmo, _coeffs, _engine_config, _geom
from inexor.engine import checkpoint_fingerprint
import os
args = argparse.Namespace(slack=0.10, tile_workers=16, checkpoint_every=10,
                          migrate_pooled=None, eject_kernel="jax")
ec = _engine_config(_geom("c-hero"), args, os.environ["PROD"] + "/ckpt")
co, _ = _coeffs(_cosmo())
want = checkpoint_fingerprint(ec, co)
have = json.load(open(os.environ["PROD"] + "/ckpt/gen1/manifest.json"))["provenance"]["fingerprint"]
print("fingerprint want", want, "\nfingerprint have", have)
sys.exit(0 if want == have else 1)
' \
  || { echo "FATAL: this node computes a different schedule than the one that wrote the checkpoint"; exit 1; }

# The heartbeat is the newest code in this job and the reason a non-finish is
# worth anything. Its gates run at second 30, with the card's own.
run_leg pytest-gpu \
  pixi run -e gpu python -m pytest -q -rs tests/test_summary.py tests/test_progress.py \
  || { echo "FATAL: the card gates failed"; exit 1; }

# --- cgh64: the card end to end, on the same NFS, before the hero leg.
#
# STEP 24, NOT 3, AND THE EPOCH IS THE REASON. The card leg reads the epoch it
# stops at, and the bins it can compare against linear theory are the ones below
# the nonlinear scale. At step 3 (a = 0.1189) the linear Delta^2 peaks at 0.26,
# so there IS no nonlinear scale and the card takes a branch the hero leg never
# will -- which is how 1010730 shipped a `float(None)` into the guard leg while
# leaving the hero's own path unexercised. Nothing in this schedule is nonlinear
# before step 15; by step 24 (a = 0.3981) the scale is 1.34 h/Mpc and falls
# INSIDE the band, which stops at half Nyquist = 1.57 h/Mpc here, so the smoke
# splits its bins either side of it exactly as the hero card at a = 1 will
# (0.2520 h/Mpc against the same 1.57 ceiling). --stop-at must stay a multiple
# of --checkpoint-every.
#
# It is also THE ANCHOR THIS JOB'S WALL IS BUILT ON, and with the heartbeat it
# now reports its three stages separately. Read those rates against the hero
# leg's: 512x is the scaling this job assumes and has never tested.
run_leg cgh64-ics \
  $BIND pixi run -e gpu python scripts/v2_m6_realization.py ics --config cgh64 \
    --workdir "$WORK/cgh64" --generator device \
  && run_leg cgh64-run \
  $BIND pixi run -e gpu python $DRIVER run --preset cgh64 --workdir "$WORK/cgh64" \
    --card $CARDS/${TAG}_cgh64.json --stop-at 24 --membind-nodes 0,1 --beat 60 \
    --checkpoint-dir "$WORK/cgh64-ckpt" --checkpoint-every 3
[ $? -ne 0 ] && { echo "FATAL: the cgh64 control failed"; exit 1; }

# The checkpoint dir is a SIBLING of the IC dir, not a child: the D7 driver
# refuses to write a checkpoint under the ICs (`5322c25`, containment by path
# component), which is what killed gb 1010700 at second 70.
mkdir -p "$WORK/cgh64-prod"
ln -sfn "$WORK/cgh64-ckpt" "$WORK/cgh64-prod/ckpt"

run_leg cgh64-card \
  $CPU $BIND pixi run -e gpu python scripts/v2_m6_realization.py card --config cgh64 \
    --workdir "$WORK/cgh64-prod" $LAYOUT $HB --allow-partial \
  || { echo "FATAL: the card path is broken at cgh64; the hero leg would find it 9 h in"; exit 1; }

# --- THE 4096^3 CARD. One call, hours long; the heartbeat is the only thing
# that will say where it got to.
run_leg hero-card \
  $CPU $BIND pixi run -e gpu python scripts/v2_m6_realization.py card --config c-hero \
    --workdir "$PROD" $LAYOUT $HB
rc_card=$?

echo ""
echo "=== card rc=$rc_card ==="
exit $rc_total
