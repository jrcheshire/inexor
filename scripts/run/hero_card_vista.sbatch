#!/bin/bash
#SBATCH -J d7f-hero-card
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -o d7f-hero-card-%j.log
# The P(k) card of an evolved realization's final checkpoint (c-hero, 4096^3, unless
# PRESET says otherwise), and nothing else.
# The export reads the same checkpoint independently (hero_export_vista.sbatch), so it
# does not queue behind a card that might overrun.
#
# Requires INEXOR_RUNS (run root), CKPT_DIR (the realization's checkpoint dir, holding
# gen0/ gen1/; SRC_RUN/hero-ckpt when only SRC_RUN is set), K_STEPS (schedule length; the
# newest checkpoint must be at this step), PROD_DIR (product directory; the card is written
# there), and INEXOR_SRC or submission from the inexor checkout. Optional: PRESET (c-hero).
# The account is passed with `sbatch -A`:
#   INEXOR_RUNS=$SCRATCH/inexor_runs SRC_RUN=$SCRATCH/inexor_runs/hero-k120 K_STEPS=120 \
#     PROD_DIR=$SCRATCH/inexor_runs/hero-k120-prod \
#     sbatch -A <account> scripts/run/hero_card_vista.sbatch
#
# Wall comes from the cgh64 anchor, not from summing parts: c-hero is exactly 512x cgh64
# in particles and coarse cells, and the cgh64 card takes ~60 s serial on this node, so
# the serial hero card is ~8.5 h. Legs ~3 min, cgh64 card arms ~2 min, loading the
# 687 GB state ~28 min. The card is three stages (streamed paint over 16,384 chunks,
# out-of-core transform, binning) and only the paint is pooled; the coarse paint pools
# 8.5-13x at W=16. At a 60-75% paint share the card takes 3.1-4.2 h; at 40% it takes
# 5.6 h. The heartbeat times the three stages separately, so the cgh64 legs measure the
# split before the hero leg starts. 8 h covers the pessimistic branch. c-gh is 64x cgh64:
# ~1.1 h serial.
#
# Pooling cannot move a number: workers return bounded sub-blocks and the parent
# accumulates integers, so arrival order is bitwise the serial order (gated in
# tests/test_card_pool.py). The pool is `paint_only` for memory: a full TilePool
# allocates three coarse force meshes (3 x 2048^3 x f32 = 103 GB) the card never reads,
# and the card leg peaks ~840 GB of 1026. The state is loaded through a SharedAllocator
# so the pool adopts it rather than copying ~755 GB.
#
# Layout parameters must match the run, not the script defaults: `load_checkpoint`
# re-derives the allocation geometry from the occupancy plus --slack / --arena-frac
# (a checkpoint stores no geometry). The run used 0.10 / 0.01 (83.85e9 slots); the
# defaults 0.20 / 0.20 would provision ~13.7e9 arena rows instead of 687M.
#
# Host memory is bound to the CPU sockets (`numactl --membind=0,1`). No verdict is
# emitted on the card.

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
CKPT=${CKPT_DIR:-${SRC_RUN:+$SRC_RUN/hero-ckpt}}
[ -n "$CKPT" ] || { echo "FATAL: set CKPT_DIR to the realization's checkpoint dir"; exit 1; }
PRESET=${PRESET:-c-hero}
export PRESET
K_STEPS=${K_STEPS:?set K_STEPS to the schedule length}
export K_STEPS
WORK=$RUNS/d7f-$SLURM_JOB_ID
PROD=${PROD_DIR:?set PROD_DIR to the product directory}
CARDS=runs/v2
mkdir -p "$CARDS"
DRIVER=scripts/run/device_run.py
TAG=d7f_$SLURM_JOB_ID
MEM_CSV=$CARDS/${TAG}_mem.csv
BIND="numactl --membind=0,1"
# `_require_cpu` refuses unless jax's first device is CPU. Scoped per leg, not exported:
# the cgh64 ICs and run use the device engine and must see the GPUs; every card leg,
# including the tests that certify it, must not.
CPU="env JAX_PLATFORMS=cpu"
LAYOUT="--slack 0.10 --arena-frac 0.01"
# W=16: W=32 bought only 1.2%. Named, not left to the driver default, so the card's
# printed receipt can be read against it.
W="--tile-workers 16"
# 60 s between progress lines: a few hundred over the hero leg; the throttle is one clock
# read per ~0.4 s chunk.
HB="--heartbeat 60"
mkdir -p "$WORK" "$PROD"
export PROD

# `_ckpt_dir` is `<workdir>/ckpt` and the run wrote `hero-ckpt`: link rather than move
# 1.3 TB or touch the source directory.
ln -sfn "$CKPT" "$PROD/ckpt"
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

# Fingerprint guard, before anything expensive. `load_checkpoint` refuses a mismatch, and
# the fingerprint hashes `coeffs.tobytes()` from `scipy.integrate.quad`, so the bytes
# depend on the platform's libm (osx-arm64 gives a different digest with identical config
# fields). The contract is per node, so it is checked on every node.
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
ec = _engine_config(_geom(os.environ["PRESET"]), args, os.environ["PROD"] + "/ckpt")
co, _ = _coeffs(_cosmo(), int(os.environ["K_STEPS"]))
want = checkpoint_fingerprint(ec, co)
have = json.load(open(os.environ["HEAD_MAN"]))["provenance"]["fingerprint"]
print("fingerprint want", want, "\nfingerprint have", have)
sys.exit(0 if want == have else 1)
' \
  || { echo "FATAL: this node computes a different schedule than the one that wrote the checkpoint"; exit 1; }

# The card tests, including the pooled card's bitwise gate (~6 s), in the lane they gate.
# `TilePool` refuses a non-CPU parent (`executor.py`) and `cmd_card` calls `_require_cpu`,
# so the card is the CPU lane and these tests must run under $CPU too. JAX_PLATFORMS is
# process-wide and read at backend init, so the lane is chosen here, not per test.
run_leg pytest-cpu-lane \
  $CPU pixi run -e gpu python -m pytest -q -rs tests/test_summary.py tests/test_progress.py \
    tests/test_card_pool.py \
  || { echo "FATAL: the card gates failed"; exit 1; }

# --- cgh64: the card end to end, on the same filesystem, before the hero leg.
#
# Stop at step 24 because the card's branches depend on the epoch: before step 15 there is
# no nonlinear scale (at step 3, a = 0.1189, linear Delta^2 peaks at 0.26), so the card
# would take a branch the hero never does. At step 24 (a = 0.3981) the nonlinear scale is
# 1.34 h/Mpc, inside the band (half Nyquist = 1.57 h/Mpc), so bins split on either side of
# it as the hero card's do at a = 1 (0.2520 h/Mpc). --stop-at must stay a multiple of
# --checkpoint-every. These legs are also the wall anchor; compare their stage rates with
# the hero leg's to test the 512x scaling.
run_leg cgh64-ics \
  $BIND pixi run -e gpu python scripts/run/realization.py ics --config cgh64 \
    --workdir "$WORK/cgh64" --generator device \
  && run_leg cgh64-run \
  $BIND pixi run -e gpu python $DRIVER run --preset cgh64 --workdir "$WORK/cgh64" \
    --card $CARDS/${TAG}_cgh64.json --stop-at 24 --membind-nodes 0,1 --beat 60 \
    --checkpoint-dir "$WORK/cgh64-ckpt" --checkpoint-every 3
[ $? -ne 0 ] && { echo "FATAL: the cgh64 control failed"; exit 1; }

# The checkpoint dir is a sibling of the IC dir, not a child: the driver refuses to write
# a checkpoint under the ICs (containment by path component).
mkdir -p "$WORK/cgh64-prod"
ln -sfn "$WORK/cgh64-ckpt" "$WORK/cgh64-prod/ckpt"

# Both arms: serial (the wall anchor) and pooled (what the hero leg runs). Their ratio is
# the pooled speedup; the stage timings in each give the split. Identity is gated in the
# pytest leg; these arms are timing only.
run_leg cgh64-card-serial \
  $CPU $BIND pixi run -e gpu python scripts/run/realization.py card --config cgh64 \
    --workdir "$WORK/cgh64-prod" $LAYOUT $HB --serial-card --allow-partial \
  || { echo "FATAL: the serial card path is broken at cgh64"; exit 1; }

run_leg cgh64-card-pooled \
  $CPU $BIND pixi run -e gpu python scripts/run/realization.py card --config cgh64 \
    --workdir "$WORK/cgh64-prod" $LAYOUT $HB $W --card-pool --allow-partial \
  || { echo "FATAL: the pooled card path is broken at cgh64; the hero leg would find it 4 h in"; exit 1; }

# --- The 4096^3 card. One call, hours long; the heartbeat says where it got to.
run_leg hero-card \
  $CPU $BIND pixi run -e gpu python scripts/run/realization.py card --config "$PRESET" \
    --workdir "$PROD" --k-steps "$K_STEPS" $LAYOUT $HB $W --card-pool
rc_card=$?

echo ""
echo "=== card rc=$rc_card ==="
exit $rc_total
