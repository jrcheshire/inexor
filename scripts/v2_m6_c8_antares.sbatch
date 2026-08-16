#!/bin/bash
#SBATCH -J m6-c8-engine
#SBATCH -p compute
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH -t 06:00:00
#SBATCH -o m6-c8-engine-%j.log
# C8: the three things 5q left unmeasured, in one job.
#
# 5q measured both `migrate` changes with `drift_and_migrate` called directly on
# a SYNTHETIC UNIFORM state, single process. Three gaps follow from that and all
# three are answered here, because they share a fixture:
#
#   1. NO ENGINE-LEVEL NUMBER since either change landed. Every figure on record
#      for the compiled eject and the radix sort is the probe's, and the probe
#      runs 1.32x the engine's own migrate at cgh64 (5q, consistent with 5l's
#      1.40x). Ratios should carry; nothing has checked that they do.
#   2. NO CLUSTERED-STATE NUMBER. `_build` makes a uniform state; the engine's is
#      evolved, and clustering is what drives arena occupancy -- the one
#      structural difference 5l flagged for this whole probe family.
#   3. NO POOL INTERACTION. `executor.TilePool` sets worker affinity BEFORE jax
#      exists in the worker, and `eject_jax` imports lazily to preserve that.
#      That is reasoning, not a measurement, and it is the last thing standing
#      between the compiled path and a default flip.
#
# THE KNOB PROVES IT APPLIED: every card carries `eject_kernel` AND an
# `eject_jax_calls` receipt. A jax leg reading 0 calls is a broken instrument,
# not a null result -- this milestone has already banked one card whose arm never
# ran (896055, green Slurm row, zero science).
#
# PRE-REGISTERED (before submission):
#   a. serial cgh64: the compiled arm's migrate phase is 1.10-1.25x the numpy
#      arm's. 5q measured 1.19x on the probe at this config, and the engine's
#      state is clustered where the probe's is uniform, so a large miss means
#      clustering changes the mechanism.
#   b. the pooled pair moves the SAME ratio within +-0.05x. The pool parallelises
#      the tile loop, not migrate (5j: migrate is 45% of a W=16 step precisely
#      because it does NOT pool), so the eject change should be indifferent to W.
#   c. NOTHING in the pooled jax leg deadlocks, crashes on a fork/affinity
#      interaction, or reads 0 calls. This is the actual question; (a) and (b)
#      are the numbers that come with it.
# A miss on (a) or (b) is reportable. A failure of (c) stops the default flip.
#
# WHY ANTARES: --mem=64G routes off deneb (56G allocatable) and covers the ~25 GB
# cgh64 build peak. -c 16 because the pooled legs want 8 workers plus the parent.
# Zero SU. Serial cgh64 is ~375 s/step post-2c, so ~19 min per serial leg and
# less for the pooled ones; ~1.5 h expected, 6 h requested (a TIMEOUT bills the
# whole wall on a charged machine, but this one is free, so the request is
# generous on purpose).
# NO set -e: per-leg exit codes; the cdev8 smoke guards the cgh64 legs.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
pixi run -e default python -c "
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
# cdev8 needs a live arena for the eject path to be exercised at all; 0.20 is the
# smoke convention this probe family already uses (job 450).
SMOKE="--config cdev8 --k 3 --repeats 1 --arena-frac 0.20"
# cgh64 knobs held EXACTLY at the W2 gg job's (914085) so the phase table is
# comparable to 5j's without a knob caveat
BIG="--config cgh64 --k 3 --repeats 1 --slack 0.20 --arena-frac 0.20"

# 1. both arms and the pooled path at laptop scale: every code path this job
#    uses, on THIS machine and commit, in ~2 min.
run_leg "smoke-numpy" pixi run -e default python $PHASE $SMOKE \
  --eject-kernel numpy --out-suffix _c8_smoke_numpy
run_leg "smoke-jax" pixi run -e default python $PHASE $SMOKE \
  --eject-kernel jax --out-suffix _c8_smoke_jax
run_leg "smoke-jax-pooled" pixi run -e default python $PHASE $SMOKE \
  --eject-kernel jax --tile-workers 4 --out-suffix _c8_smoke_jax_w4
for i in 0 1 2; do
  [ "${codes[$i]}" -ne 0 ] && {
    echo "FATAL: smoke leg ${names[$i]} failed (rc=${codes[$i]}); not burning the cgh64 legs"
    echo "=== LEG SUMMARY ==="
    for j in "${!names[@]}"; do echo "  ${names[$j]}: rc=${codes[$j]}"; done
    exit 1
  }
done

# 2. THE ENGINE PAIR at cgh64, serial: gaps 1 and 2 together, since the engine's
#    state is the clustered one by construction.
run_leg "cgh64-serial-numpy" pixi run -e default python $PHASE $BIG \
  --eject-kernel numpy --out-suffix _c8_serial_numpy
run_leg "cgh64-serial-jax" pixi run -e default python $PHASE $BIG \
  --eject-kernel jax --out-suffix _c8_serial_jax

# 3. THE POOL PAIR at W=8: gap 3. Both arms, so the comparison is a pair rather
#    than a single number read against a serial leg from a different run.
run_leg "cgh64-w8-numpy" pixi run -e default python $PHASE $BIG \
  --eject-kernel numpy --tile-workers 8 --out-suffix _c8_w8_numpy
run_leg "cgh64-w8-jax" pixi run -e default python $PHASE $BIG \
  --eject-kernel jax --tile-workers 8 --out-suffix _c8_w8_jax

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST: 5q probe at cgh64 -- baseline 31.270 s, radix 27.566 (1.13x),"
echo "         compiled 23.119 (1.35x stacked, 1.19x over radix), all bitwise."
echo "         Engine migrate on antares post-arena-fix: 23.72 s/step (467/474)."
echo "READ THE RECEIPT FIRST: a jax leg with eject_jax_calls=0 measured nothing."
echo "CARDS WRITTEN BY THIS JOB:"
ls -1 runs/v2/m6_phase_time_c8_*.json 2>/dev/null | sed 's/^/  /'
exit $rc_total
