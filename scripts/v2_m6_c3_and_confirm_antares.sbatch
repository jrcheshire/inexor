#!/bin/bash
#SBATCH -J m6-c3-confirm
#SBATCH -p compute
#SBATCH --mem=64G
#SBATCH -c 28
#SBATCH -t 05:00:00
#SBATCH -o m6-c3-confirm-%j.log
# Two follow-ups in one job, both against measured baselines from today:
#
# LEGS 2-3 = C3, the affinity A/B. Job 466 measured the un-pinned pool
# INVERTING on antares -- walls GROW with W (9.05/9.57/10.93/11.50 s at
# W=2/4/8/16, eff 7% at 16) with worker init inflating 0.6 -> 2.4 s -- the
# signature of 16 XLA-CPU spin pools each sized to all 28 cores (the umbrella
# rule: XLA ignores every thread env var; AFFINITY is the only knob, and it is
# Linux-only, which is why the laptop could not test it). Workers now pin to
# disjoint core sets BEFORE jax imports.
# PRE-REGISTERED: pinned W=8 must beat un-pinned W=8 (10.93 s) decisively and
# the plan's bar is efficiency >= 50% at W=8; below that with pinning in
# place, the pool design itself is wrong and W2 stops for a redesign.
# Also re-read: RSS/worker at P=320 (was 5.8-6.0 GB un-pinned, over the 4 GB
# line -- if pinning shrinks per-worker scratch the width arithmetic may
# survive; if not, gg holds ~20 workers, not 35-45).
#
# LEG 4 = the ENGINE-SIDE confirmation of the arena-index fix (record 5g).
# The fix deleted the invalidate-rebuild churn (laptop: occupied-arena migrate
# 12.4-14.7 -> 2.9-3.5 s, equal to the arena-empty control; pre/post bitwise
# chain n_diff 0 on every array).
# PRE-REGISTERED: cgh64 migrate falls from 197.4 s/step (job 464) to the
# ~30 s class and the whole step from 612 to ~450 s, with every OTHER phase
# unchanged within noise -- the fix touches arena bookkeeping only, so a
# moved force/paint phase would mean it is NOT the neutral change the gates
# say. If migrate does NOT fall, the engine-side attribution (the A x
# n_arena bracket) was wrong and W0b reopens. New card suffix; the pre-fix
# card is preserved.
#
# --mem=64G routing + headroom. Zero SU. ~2.5 h expected, 5 h requested.
# NO set -e: per-leg exit codes; smoke gates the long legs.

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
STAMP="/tmp/m6_c3_stamp_$SLURM_JOB_ID"; touch "$STAMP"

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

C2="scripts/v2_m6_c2_pool.py"
PHASE="scripts/v2_m6_phase_time.py"

run_leg "smoke" pixi run -e default python $C2 --config cdev8 --tile 64 \
  --workers 2 --repeats 1 --affinity --out runs/v2/m6_c2_pool_pin_smoke.json
[ "${codes[0]}" -ne 0 ] && {
  echo "FATAL: smoke failed (bitwise or crash); not burning the long legs"
  exit 1
}

run_leg "c3-pinned-cdev8" pixi run -e default python $C2 --config cdev8 --tile 64 \
  --workers 2 4 8 16 --repeats 3 --affinity --out runs/v2/m6_c2_pool_pin_cdev8.json

run_leg "c3-pinned-p320" pixi run -e default python $C2 --config cdev --tile 256 \
  --workers 4 8 --repeats 2 --affinity --out runs/v2/m6_c2_pool_pin_cdev.json

run_leg "phase-cgh64-postfix" pixi run -e default python $PHASE \
  --config cgh64 --k 3 --repeats 2 --slack 0.20 --arena-frac 0.20 \
  --out-suffix _w0b_cgh64_postfix

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST: un-pinned 466 (9.05/9.57/10.93/11.50 s, RSS 5.8-6.0 GB); pre-fix cgh64 612.32 s/step, migrate 197.4 (32.2%)"
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m6_c2_pool_pin_*.json runs/v2/m6_phase_time_w0b_*.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"
rm -f "$STAMP"
exit $rc_total
