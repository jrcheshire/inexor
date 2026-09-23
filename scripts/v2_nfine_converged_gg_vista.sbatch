#!/bin/bash
#SBATCH -J nfine-k120
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o nfine-k120-%j.log
# The fine-mesh ladder again, at converged steps and the corrected match.
#
# WHY AGAIN. The first fine-mesh ladder (1015840) found a finer force mesh gives
# LESS power at high k -- the wrong sign for PM. It ran at 40 steps, which the
# step ladder later measured 6.3% short of converged at the band edge, and on
# the ratified CIC-order match. A sharper force plausibly needs more steps, so
# that result is confounded with the integrator. Every other axis is now
# measured and none closes the ~16.5% left against EuclidEmulator2 at k = 1.56;
# the force profile falls to 88% of Newton at 0.3 Mpc/h and 58% at 0.24, which
# is where halo cores carrying k ~ 1-1.5 live.
#
# ARMS, same seed / n_part / bins as the step ladder, --coarse-match-order 3,
# 120 steps, buf derived to hold beta = 8. Cheapest first:
#   nf512    0.5 Mpc/h fine cell
#   nf2048   0.125 Mpc/h fine cell
# The 0.25 Mpc/h arm is mo3_1018341_k120 (1018341): identical config, not re-run.
#
# TIME, from the first ladder's MEASURED rates (1015840); the match order does
# not change the step rate (1018341: 23.45 vs 24.4 s/step):
#   ICs + 2 cards                                  ~2 min
#   nf512    120 x 23.0 s (8 tiles -> 8 workers)   ~46 min
#   nf2048   120 x 86.6 s                          ~2.9 h
#                                                  --------
#                                                  ~3.7 h
#
# MEMORY: nf2048 measured 237.9 GB of the node (1015840) -- tight, but it ran;
# the knob and the step count do not move it.
#
# BUDGET: gg = 0.33 SU/node-hr -> ~1.2 SU expected, 1.65 worst case (5 h).
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
WORK=$RUNS/nfine-k120-$SLURM_JOB_ID
CARDS=runs/v2
TAG=nfc_$SLURM_JOB_ID
RLZ=scripts/v2_m6_realization.py
W="--tile-workers 16"
LAYOUT="--slack 0.10 --arena-frac 0.01"
KSTEPS=120
FIX="--coarse-match-order 3"

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

run_leg ics \
  pixi run -e gpu python $RLZ ics --config cgh64 --workdir "$WORK/ics" --generator host \
  || { rc=$?; echo "FATAL (rc=$rc): IC generation failed; no arm can run"; exit 1; }

for NF in 512 2048; do
  A="$WORK/nf$NF"
  mkdir -p "$A"
  for f in "$WORK"/ics/*; do ln -sfn "$f" "$A/$(basename "$f")"; done
  rm -f "$A/realization_ics.json"

  run_leg "run-nf$NF" \
    pixi run -e gpu python $RLZ run --config cgh64 --n-fine $NF $FIX --k-steps $KSTEPS \
      --workdir "$A" $W $LAYOUT --checkpoint-every $KSTEPS --phase-instrument time \
    || { echo "NOTE: arm nf=$NF failed or hit the wall; earlier arms stand"; continue; }

  run_leg "card-nf$NF" \
    pixi run -e gpu python $RLZ card --config cgh64 --n-fine $NF $FIX --k-steps $KSTEPS \
      --workdir "$A" $W $LAYOUT --heartbeat 60 \
    || echo "NOTE: card for nf=$NF failed; its checkpoint is at $A/ckpt"

  cp -f "$A/realization_pk.json" "$CARDS/${TAG}_nf$NF.json" 2>/dev/null || true
done

exit $rc_total
