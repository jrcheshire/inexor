#!/bin/bash
#SBATCH -J ncoarse-ladder
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o ncoarse-ladder-%j.log
# The last unexamined axis: does the coarse mesh carry the residual?
#
# WHERE THIS SITS. Against EuclidEmulator2 the production configuration reads
# B/B_ref = 0.977 / 0.858 / 0.770 at k = 0.5 / 1.0 / 1.5. Two axes are now
# measured and neither explains it:
#   force mesh   doubling moves the band edge 3.5% and the WRONG way
#   step count   converging it (40 -> 160) moves it 5.7% the right way, taking
#                B/B_ref to 0.991 / 0.898 / 0.827
# So a fully converged integrator is still 17% under at the band edge, and the
# coarse mesh is what is left. It does THREE jobs at once -- the long-range
# solve, the split scale (r_s is one coarse cell at the ratified alpha), and
# the mesh the card measures on, where by k = 1.5 the TSC deconvolution is
# dividing out a factor of 1.85 in power.
#
# THE SPLIT SCALE IS HELD, NOT THE ALPHA. Refining the coarse mesh at fixed
# alpha would shrink r_s and change the force decomposition between arms, which
# is a different experiment. `--n-coarse` derives alpha to hold r_s physically
# fixed at 1.0 Mpc/h: alpha goes 1 -> 2 and the coarse-representation error
# exp(-pi^2 alpha^2) goes 5.2e-05 -> 7.2e-18, so it only ever gets smaller.
# Coarsening is refused by the driver, because it needs alpha below the
# ratified 1.0 where that term is 8.5e-02 -- larger than the effect.
#
# STEPS PINNED AT 120, NOT 40. Today's step ladder puts 120 within 0.5% of 160
# across the whole band, where 40 is within 0.5% only to k = 0.28. Running this
# at the production step count would confound a coarse-mesh effect with an
# unconverged integrator, and the step count is the larger of the two.
#
# BINS PINNED. Refining the coarse mesh moves Nyquist 3.14 -> 6.28, so the
# default band ([0, half Nyquist] of the arm's OWN mesh) differs per arm and
# differencing them would be a resampling rather than a result. --k-max 1.5708
# with 64 bins reproduces the production arm's own default binning exactly.
#
# TWO ARMS, NOT THREE. A 0.25 Mpc/h arm is 64x the coarse cells of production;
# at the measured coarse share that is several hours on an estimate I do not
# trust. One clean doubling answers the question either way.
#
# TIME, from this box's OWN measured phase card (step ladder 1016460):
#   tile_loop 64.4%, coarse_paint 18.5%, coarse_solve 6.9%, migrate 6.3%
# so 25.4% of the step is coarse work and only that part scales.
#
#   arm 1   1.0 Mpc/h cell   23.7 s/step MEASURED         ~47 min
#   arm 2   0.5 Mpc/h cell   ~63 s/step (25.4% x 8)       ~2.1 h
#   ICs (measured 1.4 min) + 2 cards                      ~2 min
#                                                         --------
#                                                         ~3.1 h central
#
# THE ONE UNMEASURED TERM is arm 2's rate: the 8x is the coarse-cell count, and
# the 25.4% share is measured but its scaling is not. Arm 1 runs first and takes
# 47 minutes, so a wrong share shows up before arm 2 commits, and cheapest-first
# means a wall kill costs only arm 2.
#
# MEMORY is a non-issue at this box: 11.5 GB at the refined arm. At 4096^3 it
# would NOT be -- one doubling takes the resident coarse force meshes from
# 103.1 to 824.6 GB and the resident mesh total alone past the whole node. That
# asymmetry is why this experiment matters: a positive result here means the
# fix is not available at production scale and alpha becomes the lever.
#
# BUDGET: gg = 0.33 SU/node-hr -> 1.65 SU worst case, ~1.05 SU expected.
#
# OPS (load-bearing on gg): pixi on PATH; JAX_PLATFORMS=cpu (jax 0.10
# hard-raises on cuInit on a GPU-less node); CONDA_OVERRIDE_CUDA; --generator
# host (the device IC generator needs cards).

set -u

export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export CONDA_OVERRIDE_CUDA=12.0
export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { rc=$?; echo "FATAL (rc=$rc): pixi not on PATH"; exit 1; }

RUNS=/scratch/10303/jrc4/inexor_runs
WORK=$RUNS/ncoarse-$SLURM_JOB_ID
CARDS=runs/v2
TAG=ncl_$SLURM_JOB_ID
RLZ=scripts/v2_m6_realization.py
W="--tile-workers 16"
LAYOUT="--slack 0.10 --arena-frac 0.01"
KSTEPS=120
BINS="--k-max 1.5708 --n-bins 64"

mkdir -p "$WORK"
echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
free -g | head -2
df -h "$RUNS" | tail -1
nproc

rc_total=0
finish () {
  echo ""; echo "=== cards ==="
  ls -la "$WORK"/nc*/realization_pk.json 2>/dev/null
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

run_leg ics \
  pixi run -e gpu python $RLZ ics --config cgh64 --workdir "$WORK/ics" \
    --generator host \
  || { rc=$?; echo "FATAL (rc=$rc): IC generation failed; no arm can run"; exit 1; }

# Cheapest first, each carded before the next starts.
for NC in 256 512; do
  A="$WORK/nc$NC"
  mkdir -p "$A"
  for f in "$WORK"/ics/*; do ln -sfn "$f" "$A/$(basename "$f")"; done
  rm -f "$A/realization_ics.json"

  run_leg "run-nc$NC" \
    pixi run -e gpu python $RLZ run --config cgh64 --n-coarse $NC \
      --k-steps $KSTEPS --workdir "$A" $W $LAYOUT \
      --checkpoint-every $KSTEPS --phase-instrument time \
    || { echo "NOTE: arm n_coarse=$NC failed or hit the wall; earlier arms stand"; continue; }

  run_leg "card-nc$NC" \
    pixi run -e gpu python $RLZ card --config cgh64 --n-coarse $NC \
      --k-steps $KSTEPS --workdir "$A" $W $LAYOUT $BINS --heartbeat 60 \
    || echo "NOTE: card for n_coarse=$NC failed; its checkpoint is at $A/ckpt"

  cp -f "$A/realization_pk.json" "$CARDS/${TAG}_nc$NC.json" 2>/dev/null || true
done

exit $rc_total
