#!/bin/bash
#SBATCH -J m6-phase-accum
#SBATCH -p compute
#SBATCH -w deneb
#SBATCH --mem=32G
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -o m6-phase-accum-%j.log
# Two questions in one job: where does a step's WALL go, and did the storage
# change move the ACCURACY figure it was gated on.
#
# The milestone's binding constraint moved from memory to wall today. C-gh now
# fits a 237 GB gg node at 0.69x of an arithmetic floor (511.2 -> 164.6 GB over
# the velocity array and the in-place repack), but cgh64 runs at 608.67 s/step,
# which extrapolates to ~11 h/step and ~18 days per realization at K=40. The
# plan's own test for how many realizations to run is "under 5 h each", so the
# gap is ~90x and it is not closing by more of the same.
#
# Nothing has ever timed the phases on the current code. The existing phase
# instrument resets VmHWM at each boundary -- right for a peak, wrong for a
# wall. So `scripts/v2_m6_phase_time.py` is new, and it carries the neutrality
# control the memory one does: the same run with `phase=None`, and a gate that
# refuses to report a breakdown the instrument itself moved.
#
# TWO INSTRUMENT DEFECTS ALREADY CAUGHT AND FIXED LOCALLY, stated because they
# are the reason to trust the third run and not the first:
#   - the traced arm ran first in every repeat and paid the JAX compilation the
#     control then reused, so the gate read +0.66 s of COMPILE as instrument
#     overhead. Fixed with a full-K warmup.
#   - caching the ICs made the arms so reproducible that 2 sigma fell to ~20 ms,
#     below timer noise, and the gate fired on a -0.02 s difference with the
#     traced arm FASTER. A gate that cannot pass on a clean run is not measuring
#     the instrument, so the bound is now 2 sigma OR 2% of the wall, whichever
#     is larger, with both terms on the card.
#
# LEG 2 IS OWED, NOT NEW. Per-brick velocity scales changed the storage
# semantics, so D-v2-21's ratified accumulated codec cost -- **7.125e-4 at C-dev
# K=40 against D-v2-9's 3e-2 bar, a 42x margin** -- no longer describes the
# engine that produced it. It runs at the M-v2-3 gate's OWN defaults (slack
# 0.10, arena_frac 0.02) because that is what the ratified number was measured
# at; changing a knob here would make the comparison meaningless.
#
# PRE-REGISTERED for leg 2, and it can move EITHER WAY:
#   - per-brick scales are up to ~2.5x finer than the global one (measured
#     per-tile ratios r_p50 0.40-0.80 of the global max), which should IMPROVE
#     the figure;
#   - but a migrating particle is now re-expressed at its destination brick's
#     scale, so it sees TWO roundings where one global scale gave it one.
#   - PASS is anything under the 3e-2 bar, which is the ratified gate. A move to
#     within ~2x of 7.125e-4 in either direction is expected and unremarkable.
#     Above ~3e-3 (still passing, but 40x the old figure) I would want the
#     storage decision re-opened rather than recorded as a pass.
#
# Legs, cheapest first, smoke gating:
#   1 smoke          both instruments end to end, ~1 min
#   2 phase cdev8    the breakdown at the cheap config
#   3 phase cdev     the breakdown at the ANCHOR, which is the configuration
#                    every wall figure quoted today was measured at
#   4 accum cdev     the owed accuracy checkpoint, K=40
#
# ~2 h expected, 6 h requested; my wall estimates have been low three times
# today. Zero SU, 8 of 64 cores, no staging so nothing to clean up.
# NO set -e: independent legs, per-leg exit codes -- except smoke, which gates.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
pixi run -e default python -c "
import jax, platform, sys
d = jax.devices()
print('jax', jax.__version__, d, platform.machine(), sys.platform)
assert d[0].platform == 'cpu', 'expected the CPU backend in the default env'
" || { echo "FATAL: jax import/backend check failed"; exit 1; }

echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2
STAMP="/tmp/m6_pa_stamp_$SLURM_JOB_ID"; touch "$STAMP"

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
GATE="scripts/v2_m3_engine_gate.py"

run_leg "smoke" bash -c "
  pixi run -e default python $PHASE --config smoke --k 3 --repeats 2 \
    --slack 0.20 --arena-frac 0.20 --out-suffix _pa_smoke &&
  pixi run -e default python $GATE --config smoke --leg accum --k 3 \
    --out-suffix _pa_smoke"
[ "${codes[0]}" -ne 0 ] && {
  echo "FATAL: smoke failed; a broken probe would burn every long leg (job 451)"
  exit 1
}

# ---- where the wall goes. Generous knobs so the arena refusal cannot end a
# ---- TIMING leg; these are not the accuracy leg's knobs and must not be mixed.
run_leg "phase-cdev8" pixi run -e default python $PHASE \
  --config cdev8 --k 5 --repeats 3 --slack 0.20 --arena-frac 0.20 \
  --out-suffix _pa_cdev8

run_leg "phase-cdev" pixi run -e default python $PHASE \
  --config cdev --k 5 --repeats 2 --slack 0.20 --arena-frac 0.20 \
  --out-suffix _pa_cdev

# ---- the owed accuracy checkpoint, at the M-v2-3 gate's OWN defaults
run_leg "accum-cdev-k40" pixi run -e default python $GATE \
  --config cdev --leg accum --k 40 --out-suffix _pa_cdev_k40

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST: cdev 48.53 s/step (job 460); ratified codec cost 7.125e-4 vs a 3e-2 bar"
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m6_phase_time_pa_*.json runs/v2/*_pa_*.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"
rm -f "$STAMP"
exit $rc_total
