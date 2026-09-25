#!/bin/bash
#SBATCH -J npart-ladder
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o npart-ladder-%j.log
# The mass-resolution axis, first two arms: does the interparticle spacing
# carry the residual?
#
# WHERE THIS SITS. At 120 steps (converged) the realization reads 17% under
# EuclidEmulator2 at k = 1.5. Force mesh, step count and coarse mesh are all
# measured and none of them closes it. Every preset sits at 0.5 Mpc/h spacing,
# so spacing is the one resolution axis no ladder has moved. `--n-part` holds
# the box and both meshes, so the force is identical in physical units across
# arms (fine cell 0.25, coarse cell 1.0, r_s 1.0 Mpc/h, beta 8).
#
# ARMS, cheapest first, each carded before the next starts:
#   np256        1.0 Mpc/h spacing, seed 0, 120 steps
#   np512-s1     0.5 Mpc/h spacing, seed 1, 120 steps
# The 0.5 Mpc/h seed-0 arm is the step ladder's 120-step card (1016460), which
# this does not re-run. A different n_part at the same seed is an UNRELATED
# realization, so the seed-1 arm measures what realization-to-realization
# scatter looks like at this box -- the floor any spacing difference must clear.
#
# BINS NOT PINNED. Every arm has the same coarse mesh, so the default band is
# the same pi/2 edges the step ladder used, and the cards difference against
# 1016460 directly. (--k-max 1.5708 put edges 4e-6 off the lattice shells
# they sit on and moved whole shells between bins.)
#
# TIME:
#   np256     rate UNMEASURED; the 0.5 Mpc/h arm's 24.4 s/step is a ceiling,
#             since the meshes are identical and there are 8x fewer particles
#                                                          <= 49 min
#   np512-s1  24.4 s/step MEASURED (1016874 nc256)          ~50 min
#   ICs (1.3 min at 512^3) x2 + 2 cards                     ~4 min
#                                                          --------
#                                                          <= 1.7 h
#
# MEMORY: 157.3 GB measured at 512^3 W=16 (1016874). The 256^3 arm is the
# second point on the memory-vs-particles curve, which sizes the 1024^3 arm:
# the planner puts 1024^3 at a 222 GB floor and measured 1.9x its floor at
# 512^3, so that arm does not go on gg at W=16 without this job's numbers.
#
# BUDGET: gg = 0.33 SU/node-hr -> 1.0 SU worst case (3 h), ~0.6 expected.
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
WORK=$RUNS/npart-$SLURM_JOB_ID
CARDS=runs/v2
TAG=npl_$SLURM_JOB_ID
RLZ=scripts/v2_m6_realization.py
W="--tile-workers 16"
LAYOUT="--slack 0.10 --arena-frac 0.01"
KSTEPS=120

mkdir -p "$WORK"
echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
free -g | head -2
df -h "$RUNS" | tail -1
nproc

rc_total=0
finish () {
  echo ""; echo "=== cards ==="
  ls -la "$WORK"/np*/realization_pk.json 2>/dev/null
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

# name n_part seed
for ARM in "np256 256 0" "np512-s1 512 1"; do
  set -- $ARM
  NAME=$1; NP=$2; SEED=$3
  A="$WORK/$NAME"
  GEO="--config cgh64 --n-part $NP --seed $SEED"

  run_leg "ics-$NAME" \
    pixi run -e gpu python $RLZ ics $GEO --workdir "$A" --generator host \
    || { echo "NOTE: ICs for $NAME failed; skipping the arm"; continue; }

  run_leg "run-$NAME" \
    pixi run -e gpu python $RLZ run $GEO --k-steps $KSTEPS --workdir "$A" $W $LAYOUT \
      --checkpoint-every $KSTEPS --phase-instrument time \
    || { echo "NOTE: arm $NAME failed or hit the wall; earlier arms stand"; continue; }

  run_leg "card-$NAME" \
    pixi run -e gpu python $RLZ card $GEO --k-steps $KSTEPS --workdir "$A" $W $LAYOUT \
      --heartbeat 60 \
    || echo "NOTE: card for $NAME failed; its checkpoint is at $A/ckpt"

  cp -f "$A/realization_pk.json" "$CARDS/${TAG}_$NAME.json" 2>/dev/null || true
done

exit $rc_total
