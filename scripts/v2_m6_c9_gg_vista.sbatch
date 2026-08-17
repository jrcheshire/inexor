#!/bin/bash
#SBATCH -J m6-c9-gg
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m6-c9-gg-%j.log
# C9: what the compiled eject is WORTH, on the machine where the mix is real.
#
# 5r measured the ratio at the engine on antares: `migrate` 22.13 -> 20.16
# s/step, 1.10x. It could not measure what that is worth, and said so: `migrate`
# is 5.8% of a serial step on antares and 7.7% at W=8, against 45% of a W=16
# step on gg (5j). 5e already recorded that the phase profile INVERTS across
# machines. So antares owns the ratio and gg owns the leverage, and the C-gh
# projection in 5r is currently a SPLICE -- antares' ratio against gg's mix --
# which is not a measurement of anything.
#
# This job makes it one measurement: both eject arms, serial AND at W=16, on the
# production candidate, in ONE job so no cross-job node drift enters (5k).
#
# PRE-REGISTERED (before submission):
#   a. the `migrate` phase ratio on gg is 1.05-1.25x, i.e. the antares 1.10x
#      transfers. Grace's per-core arithmetic differs from x86's but the phase is
#      the same code on the same clustered state, so a large miss means the
#      ratio is a machine property and neither number describes C-gh.
#   b. at W=16 the STEP-level effect is resolvable where it was not on antares:
#      migrate is ~45% of a W=16 step here (5j), so a 1.10x on it is ~4% of the
#      step, against the +-6-8% cross-leg drift 5k measured. **That is still
#      marginal**, which is why both W=16 arms run in this job rather than
#      against 5j's stored serial reference.
#   c. the pooled jax arm does NOT OOM here. It did on antares inside a 64 GB
#      REQUEST (5r: 8 workers x 6.54 GB anon, plus the parent's XLA arena); a gg
#      node carries 237 GB, so this also gives record owed item 11 -- the pool's
#      incremental footprint -- its first uncapped reading.
# A miss on (a) or (b) is reportable. (c) failing on a 237 GB node would be the
# real result of the job.
#
# THE RECEIPT: every card carries `eject_kernel` and `eject_jax_calls`. A jax leg
# reading 0 measured nothing. 896055 is why this exists.
#
# BUDGET: ~2 h expected against a 4 h request (a TIMEOUT bills the whole wall,
# so the request is generous on purpose and the expectation is stated). gg is
# 0.33 SU/node-hr, so **~0.7 SU expected, 1.3 SU worst case**. Serial cgh64 is
# 284.5 s/step on gg (5j) and the probe runs a traced AND a control pass, so a
# serial leg is ~28 min and a W=16 leg ~7 min.
#
# OPS, all three load-bearing on gg: pixi on PATH (896055/896092 ran ZERO
# science without it), JAX_PLATFORMS=cpu (jax 0.10's CUDA plugin hard-raises on
# cuInit on a GPU-less node) and CONDA_OVERRIDE_CUDA (the `gpu` env enforces
# cuda=12 at ACTIVATION). `-A JPL-SPHEREx` is case-sensitive and Vista refuses
# submission without it.
# NO set -e: per-leg exit codes. rc=2 means instrument non-neutrality, NOT a
# crash -- the smoke reports it and only cgh64 gates on it (the 481 lesson).

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
export JAX_PLATFORMS=cpu
export CONDA_OVERRIDE_CUDA=12.0
pixi run -e gpu python -c "
import jax
assert jax.devices()[0].platform == 'cpu', 'expected the CPU backend'
print('jax', jax.__version__)
" || { echo "FATAL: jax import/backend check failed"; exit 1; }

echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2

rc_total=0
names=(); codes=()
run_leg() {
  local name="$1"; shift
  echo "=== $name ==="
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "=== $name rc=$rc after $((SECONDS - t0)) s ==="
  names+=("$name"); codes+=("$rc")
  [ $rc -ne 0 ] && rc_total=1
  return 0
}

PHASE="scripts/v2_m6_phase_time.py"
SMOKE="--config cdev8 --k 3 --repeats 1 --arena-frac 0.20"
# knobs held EXACTLY at 914085's so the phase table is comparable to 5j
BIG="--config cgh64 --k 3 --repeats 1 --slack 0.20 --arena-frac 0.20"

# 1. every path this job uses, on Grace, at this commit. ~4 min.
run_leg "smoke-jax" pixi run -e gpu python $PHASE $SMOKE \
  --eject-kernel jax --out-suffix _c9_smoke_jax
run_leg "smoke-jax-pooled" pixi run -e gpu python $PHASE $SMOKE \
  --eject-kernel jax --tile-workers 4 --out-suffix _c9_smoke_jax_w4
for i in 0 1; do
  if [ "${codes[$i]}" -eq 2 ]; then
    echo "NOTE: smoke leg ${names[$i]} non-neutral (rc=2); smoke exercises paths,"
    echo "      it does not measure. Continuing."
  elif [ "${codes[$i]}" -ne 0 ]; then
    echo "FATAL: smoke leg ${names[$i]} rc=${codes[$i]}; not burning SU on cgh64"
    echo "=== LEG SUMMARY ==="
    for j in "${!names[@]}"; do echo "  ${names[$j]}: rc=${codes[$j]}"; done
    exit 1
  fi
done

# 2. the serial pair: the ratio, on Grace, against 5j's 284.5 s/step reference
run_leg "cgh64-serial-numpy" pixi run -e gpu python $PHASE $BIG \
  --eject-kernel numpy --out-suffix _c9_serial_numpy
run_leg "cgh64-serial-jax" pixi run -e gpu python $PHASE $BIG \
  --eject-kernel jax --out-suffix _c9_serial_jax

# 3. THE POINT: the W=16 pair, where migrate is ~45% of the step and the
#    step-level effect is largest. Also (c), on a 237 GB node.
run_leg "cgh64-w16-numpy" pixi run -e gpu python $PHASE $BIG \
  --eject-kernel numpy --tile-workers 16 --out-suffix _c9_w16_numpy
run_leg "cgh64-w16-jax" pixi run -e gpu python $PHASE $BIG \
  --eject-kernel jax --tile-workers 16 --out-suffix _c9_w16_jax

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST: antares engine (5r) migrate 22.13 -> 20.16 s/step = 1.10x, step"
echo "         381.46 -> 388.99 (unresolvable, tile force moved 1.7%);"
echo "         gg 5j serial 284.496 s/step, W=16 72.642, migrate 45.3% at W=16."
echo "READ THE RECEIPT: a jax leg with eject_jax_calls=0 measured nothing."
echo "rc=2 is instrument non-neutrality, not a crash."
echo "CARDS WRITTEN BY THIS JOB:"
ls -1 runs/v2/m6_phase_time_c9_*.json 2>/dev/null | sed 's/^/  /'
exit $rc_total
