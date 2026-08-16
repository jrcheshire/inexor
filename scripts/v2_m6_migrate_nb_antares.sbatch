#!/bin/bash
#SBATCH -J m6-migrate-nb
#SBATCH -p compute
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -o m6-migrate-nb-%j.log
# M-v2-6 owed item 10: decompose `migrate`, which section 5j put at 45% of a
# W=16 step on gg and 23.3 h of a projected 51.7 h C-gh realization -- 4.7x the
# whole 5 h bar on its own. It is host numpy, so neither the worker pool nor
# the GPU lane touches it, and every remaining wall fix points at it.
#
# THE HYPOTHESIS, already on the record as section 6's last open shape: both
# `_eject_slab` and `_insert_slab` walk bricks in a Python `for`, so a step runs
# ~2 * nb^3 iterations -- 65,536 at cgh64 and 4.2M at C-gh, since nb goes
# 32 -> 128.
#
# WHY AN ARM AND NOT ARITHMETIC: per-brick overhead (~nb^3) and per-row work
# (~N) predict the SAME 64x from cgh64 to C-gh, so no derivation separates them,
# and the config ladder cannot either -- particles, bricks and coarse cells all
# move together on it (2026-08-13: 64.00x / 64.00x / 61.18x, degenerate by
# construction). nb at FIXED N is the only axis that reads. Box is fixed too, so
# only brick granularity moves: rows/brick 32,768 / 4,096 / 512.
#
# PRE-REGISTERED, and the two poles are far enough apart that no threshold is
# picked. Fitting total_s ~ nb^alpha at fixed N, alpha / 3 is the per-brick
# SHARE of migrate:
#   alpha ~ 3 (share ~100%) -> the Python slab loop IS the cost; the fix is
#                              vectorizing across a slab's bricks.
#   alpha ~ 0 (share ~0%)   -> honest row work; the loop is innocent and the
#                              only lever is parallelism, which means migrate
#                              joins the pool and inherits 5j's bandwidth
#                              ceiling.
#   in between              -> both live, and the share is how much a fix buys.
# The laptop smoke at 262,144 particles read alpha +1.28 / +1.90 (share 43% /
# 63%) at reach 1 / reach 3, so the honest expectation is BOTH TERMS LIVE. The
# share is expected to FALL at production scale, where each brick carries 8x the
# rows the smoke's did against the same per-brick overhead.
#
# THE GATE, and it is the reason leg 2 exists at all: this probe's state is
# SYNTHETIC (uniform positions, Gaussian velocities) while the engine's is
# evolved and clustered, and arena pressure is a clustering property. So the
# probe describing the engine is a hypothesis. Leg 2 reads its cgh64 rung
# against the engine's own measured 23.72 s/step (antares, jobs 467 AND 474,
# which agree to 0.0%) and returns 3 if it misses by more than 2x. On a miss the
# long legs are SKIPPED -- the nb scan would be measuring this probe rather than
# the engine, and the hunt moves engine-side instead. Gate mutation-tested in
# both directions on the laptop (rc=0 on a true bar, rc=3 on a 20x-wrong one).
#
# brick_slack 0.0 throughout: at the 0.10 default a uniform state never
# overflows a brick and the arena stays EMPTY (job 465 measured arena_used = 0
# on every rung), which is the one structural difference from the engine's
# clustered state. --calls 2 measures the SECOND migrate, with the arena already
# resident -- the engine's steady condition, and the regime 5g's churn lived in.
#
# --mem=64G is ROUTING (>56G keeps it off deneb) + headroom over the ~25 GB
# build peak at 512^3. Zero SU. ~1.5-2 h expected, 6 h requested.
# NO set -e: per-leg exit codes; the smoke and the gate each guard what follows.

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
STAMP="/tmp/m6_migrate_nb_stamp_$SLURM_JOB_ID"; touch "$STAMP"

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

PROBE="scripts/v2_m6_migrate_depth.py"
ENGINE_MIGRATE_S=23.72   # antares cgh64, jobs 467 and 474 (identical to 2 dp)

# 1. the reader and the gate, at laptop scale, on THIS machine and commit.
#    Two cluster jobs have died on a line added after the last run (447/448);
#    this leg is 30 s and executes every reporting path the long legs use.
#    nbs8/nbs16 only, deliberately: nbs4 voids by design (nb=4 clamps reach to
#    nb//2 = 2 at f=2.85), and a void rung exits 1 exactly as an uncaught
#    exception does -- so including it would make the smoke's rc unreadable as
#    a gate. Void-free, any non-zero here is a real defect.
run_leg "nbscan-smoke" pixi run -e default python $PROBE \
  --config nbs8 nbs16 --brick-slack 0.0 --calls 2 \
  --out runs/v2/m6_migrate_nb_smoke.json
[ "${codes[0]}" -ne 0 ] && {
  echo "FATAL: smoke failed (rc=${codes[0]}); not burning the long legs"
  exit 1
}

# 2. THE REPRODUCTION GATE -- does the synthetic probe describe the engine?
run_leg "repro-gate" pixi run -e default python $PROBE \
  --config cgh64 --brick-slack 0.0 --calls 2 \
  --expect-s $ENGINE_MIGRATE_S --expect-within 2.0 \
  --out runs/v2/m6_migrate_nb_repro.json
if [ "${codes[1]}" -eq 3 ]; then
  echo "GATE FAILED: the probe does not reproduce the engine's migrate within 2x."
  echo "SKIPPING the nb scan and the profile -- they would measure the probe."
  echo "Next step is engine-side timing, not this instrument."
  echo "=== LEG SUMMARY ==="
  for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
  rm -f "$STAMP"
  exit 1
fi

# 3. the axis: nb at fixed N = 512^3, matched f, matched box
run_leg "nbscan" pixi run -e default python $PROBE \
  --config nb16 nb32 nb64 --brick-slack 0.0 --calls 2 \
  --out runs/v2/m6_migrate_nb_scan.json

# 4. name the lines. Step three of 5g's method (probe exonerates -> one-axis
#    arm -> profile names the line), which is what turned "migrate is slow"
#    into "_build_arena_index, 51% of an occupied migrate".
run_leg "profile" pixi run -e default python $PROBE \
  --config cgh64 --brick-slack 0.0 --calls 2 --profile 30 \
  --out runs/v2/m6_migrate_nb_profile.json

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST: engine migrate $ENGINE_MIGRATE_S s/step at cgh64 (antares 467/474);"
echo "         32.78 s/step on gg at W=16, 45.3% of the step (Vista 914085, 5j);"
echo "         laptop smoke alpha +1.28/+1.90 = per-brick share 43%/63% at 262k particles"
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m6_migrate_nb*.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"
rm -f "$STAMP"
exit $rc_total
