#!/bin/bash
#SBATCH -J nfine-ladder
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o nfine-ladder-%j.log
# The force-resolution convergence test the emulator comparison asked for.
#
# WHY. Against EuclidEmulator2 (~1%, boost emitted natively, our cosmology
# inside its training box) gb 1013309's card reads B_inexor/B_EE2 = 0.99-1.01
# out to k ~ 0.2, 0.977 at k = 0.41, 0.859 at k = 1.0, 0.768 at the half-Nyquist
# band edge. The SIGN is what a PM force should do -- it under-predicts
# small-scale power -- but no external reference can say whether that is the
# right amount for a 0.25 Mpc/h force mesh, and the config cannot either: every
# scale it gives sits at k >= 1.57, and `forces.py`'s own analytic split errors
# are 5.2e-5 (coarse representation) and 1.5e-8 (buffer truncation), three
# orders below the effect. It is a convergence question, so measure it.
#
# THE BUFFER IS DERIVED, NOT HELD. `buf` is counted in FINE CELLS, so a ladder
# that refines n_fine at fixed buf shrinks the PHYSICAL buffer and erfc(beta/2)
# runs 1.5e-8 -> 4.7e-3 -> 1.6e-1 across these rungs. The finest arm would be
# 16% wrong from truncation alone and would read as convergence going the WRONG
# WAY. `--n-fine` derives buf to hold beta = 8; every leg prints alpha, beta and
# both error terms, so the receipt carries proof it applied.
#
# ONE IC GENERATION, SYMLINKED INTO EACH ARM. The ICs are byte-identical across
# n_fine (verified locally, same md5 at 128/256/512), so sharing one directory
# makes the ladder matched by construction rather than by re-derivation, and
# saves two generations.
#
# THE COARSE MESH IS THE SAME ON EVERY ARM (n_coarse = 256), so all three cards
# land on identical k bins and differencing them needs no interpolation. Only
# the short-range force changes.
#
# TIME. The anchor is the RECORDED CPU figure, 55.69 s/step at cgh64
# (`runs/v2/m6_scaling_record.md` 5s), less ~5-7% from the M2-M4 migrate ladder
# -> ~52 s/step. NOT the 7.4 s/step from gb 1013309: that leg ran the DEVICE
# engine on four GB200s and does not transport to a GPU-less node.
#
#   arm 1  n_fine  512, buf  16   52 / 2.59 (cdev8 measured)   ~13 min
#   arm 2  n_fine 1024, buf  32   52 s/step, RECORDED          ~35 min
#   arm 3  n_fine 2048, buf  64   bounded under 8.39x arm 2    ~2 h, 4.8 h worst
#   ICs (host generator) + 3 cards                             ~10 min
#                                                              ----------------
#                                                     ~3 h central, ~5.5 h bound
#
# The bound is cdev8's measured 256->512 factor, which came with a 27x rise in
# tile work; arm 3 is only 13.8x, so it should land well under 8.39 -- but the
# cdev8 ladder does NOT fit a constant-plus-tile-work model (fitting its top two
# rungs back-predicts 7.5 s/step for its first against 3.95 measured), so 8.39
# is carried as a bound and not turned into an estimate. 8 h covers the bound.
#
# ARMS RUN CHEAPEST FIRST and each is carded before the next starts, so a wall
# kill costs only arm 3 and leaves two comparison points banked. That is the
# M-v2-4 lesson (Vista 899070: the expensive leg ran first, timed out at 84% of
# the wall, and the two cheapest legs never started).
#
# BUDGET: gg = 0.33 SU/node-hr -> 2.64 SU worst case, ~1 SU expected.
#
# OPS (load-bearing on gg): pixi on PATH; JAX_PLATFORMS=cpu (jax 0.10
# hard-raises on cuInit on a GPU-less node); CONDA_OVERRIDE_CUDA (the gpu env's
# activation gate); --generator host (the device IC generator needs cards).

set -u

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export CONDA_OVERRIDE_CUDA=12.0

RUNS=/scratch/10303/jrc4/inexor_runs
WORK=$RUNS/nfine-$SLURM_JOB_ID
CARDS=runs/v2
TAG=nfl_$SLURM_JOB_ID
RLZ=scripts/v2_m6_realization.py
# W=16 is the operating point the hero legs run. The 55.69 s/step anchor does
# not record its own W, so this is a choice and not a match -- arm 2 measures
# the rate this job actually gets, and that is what arm 3 is read against.
W="--tile-workers 16"
LAYOUT="--slack 0.10 --arena-frac 0.01"

mkdir -p "$WORK"
echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
free -g | head -2
df -h "$RUNS" | tail -1
nproc

rc_total=0
finish () {
  echo ""; echo "=== cards ==="
  ls -la "$WORK"/nf*/realization_pk.json 2>/dev/null
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

# ONE generation, at the preset's own n_fine. The ICs do not depend on it, and
# this is the whole matched-IC guarantee for the ladder.
run_leg ics \
  pixi run -e gpu python $RLZ ics --config cgh64 --workdir "$WORK/ics" \
    --generator host \
  || { echo "FATAL: IC generation failed; no arm can run"; exit 1; }

# Cheapest first. Each arm is carded before the next starts, so a wall kill
# costs one arm and banks the rest.
for NF in 512 1024 2048; do
  A="$WORK/nf$NF"
  mkdir -p "$A"
  # symlink the generation in rather than copying 1.2 GB three times; the
  # checkpoint goes to $A/ckpt, a real directory beside the links
  for f in "$WORK"/ics/*; do ln -sfn "$f" "$A/$(basename "$f")"; done
  rm -f "$A/realization_ics.json"   # each arm writes its own cards

  run_leg "run-nf$NF" \
    pixi run -e gpu python $RLZ run --config cgh64 --n-fine $NF \
      --workdir "$A" $W $LAYOUT --checkpoint-every 40 \
      --phase-instrument time \
    || { echo "NOTE: arm nf=$NF failed or hit the wall; earlier arms stand"; continue; }

  run_leg "card-nf$NF" \
    pixi run -e gpu python $RLZ card --config cgh64 --n-fine $NF \
      --workdir "$A" $W $LAYOUT --heartbeat 60 \
    || echo "NOTE: card for nf=$NF failed; its checkpoint is at $A/ckpt"

  cp -f "$A/realization_pk.json" "$CARDS/${TAG}_nf$NF.json" 2>/dev/null || true
done

exit $rc_total
