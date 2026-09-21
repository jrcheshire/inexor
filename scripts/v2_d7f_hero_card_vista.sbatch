#!/bin/bash
#SBATCH -J d7f-hero-card
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
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
# TIME: the cgh64 anchor, scaled, with the paint pooled.
#   legs (fingerprint, pytest, cgh64 ICs, cgh64 run to 24) ......  196 s  measured, 1010730/938
#   cgh64 card, BOTH arms (serial + pooled) .....................  120 s  serial measured, 1010938
#   hero: load the 687 GB state ................................. 1663 s  measured, 1010938
#   hero: the card itself
#     serial reference = 512x the cgh64 anchor .................. 30720 s  8.5 h
#     of which the streamed paint, pooled at W=16 ............... see below
#   CENTRAL 4.0 h.  REQUESTED 8:00.  ~4 SU at the central.
#
#   WHAT THE POOLING DOES AND WHAT IS STILL A GUESS. The card is three stages
#   and only the first parallelises: the streamed paint over 16,384 chunks,
#   then the out-of-core transform, then the binning. W2's C5 measured the
#   engine's coarse paint pooling 8.5-13.0x at W=16 -- better than the tile
#   loop -- and the card runs the SAME `coarse_delta_streamed` chunks through
#   the same workers. What is NOT measured anywhere is how the 8.5 h splits
#   across the three. At a 60-75% paint share, pooling at 8x takes the card to
#   3.1-4.2 h and the residual is the serial transform and binning; at a 40%
#   share it takes it to 5.6 h and pooling was the wrong lever. THE SPLIT IS
#   WHAT THE cgh64 LEGS BELOW MEASURE, both arms, before the hero leg starts --
#   the heartbeat names the three stages separately, so their ratio is readable
#   off the log rather than assumed. 8:00 covers the pessimistic branch.
#
#   The pooling CANNOT move a number: workers return bounded sub-blocks and the
#   parent accumulates, so integer associativity makes arrival order bitwise the
#   serial order. That is gated in `tests/test_card_pool.py`, at the mesh and at
#   the card, with the dispatch counted so a dropped `pool=` cannot pass.
#
#   MEMORY IS WHY THE POOL IS `paint_only`. A full TilePool allocates three
#   coarse force meshes -- 3 x 2048^3 x f32 = 103.1 GB -- and builds a tile
#   kernel in every worker, and the card reads neither; the card leg measured
#   839.5 GB of 1026 on 1010938, so 103 GB is most of what it has left. Two
#   further conditions, both fixed here rather than hoped for: the state is
#   loaded THROUGH a SharedAllocator so the pool adopts it instead of copying
#   754.6 GB, and `TilePool.close()` no longer copies adopted fields back into
#   private memory on the way out, which was a second 754.6 GB at teardown.
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
#   2. What does the paint pool buy at cgh64 (both arms run below), and does
#      that transport to hero scale, where the chunk count is 512x and the
#      sub-block each worker ships back is the same shape?
#   3. Does the card finish inside 8 h, and if not, at what fraction and in
#      which stage?
#   4. What does the card say below k_nonlinear (0.2520 h/Mpc at a = 1)? NO
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
# W=16 is W2's operating point (W=32 bought 1.2% and broke its own neutrality
# bound); named rather than left to the driver default so the receipt the card
# prints can be read against it.
W="--tile-workers 16"
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

# The pooled paint and the heartbeat are the newest code in this job, and the
# pooled card's bitwise gate is in here: it runs in 6 s at the smoke rung and
# the thing it certifies is the whole basis for pooling the hero leg.
run_leg pytest-gpu \
  pixi run -e gpu python -m pytest -q -rs tests/test_summary.py tests/test_progress.py \
    tests/test_card_pool.py \
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

# BOTH ARMS. The serial one is the anchor this job's wall is scaled from and
# the pooled one is what the hero leg will actually run; their ratio is the
# pooled speedup at a config where both are affordable, and the heartbeat's
# three stage timings inside each are the split the projection needs. The
# IDENTITY is gated in pytest above, not here -- these two arms are timing.
run_leg cgh64-card-serial \
  $CPU $BIND pixi run -e gpu python scripts/v2_m6_realization.py card --config cgh64 \
    --workdir "$WORK/cgh64-prod" $LAYOUT $HB --serial-card --allow-partial \
  || { echo "FATAL: the serial card path is broken at cgh64"; exit 1; }

run_leg cgh64-card-pooled \
  $CPU $BIND pixi run -e gpu python scripts/v2_m6_realization.py card --config cgh64 \
    --workdir "$WORK/cgh64-prod" $LAYOUT $HB $W --card-pool --allow-partial \
  || { echo "FATAL: the pooled card path is broken at cgh64; the hero leg would find it 4 h in"; exit 1; }

# --- THE 4096^3 CARD. One call, hours long; the heartbeat is the only thing
# that will say where it got to.
run_leg hero-card \
  $CPU $BIND pixi run -e gpu python scripts/v2_m6_realization.py card --config c-hero \
    --workdir "$PROD" $LAYOUT $HB $W --card-pool
rc_card=$?

echo ""
echo "=== card rc=$rc_card ==="
exit $rc_total
