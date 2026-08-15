#!/bin/bash
#SBATCH -J m6-w0b-migrate
#SBATCH -p compute
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 05:00:00
#SBATCH -o m6-w0b-migrate-%j.log
# W0b of the wall plan: decompose migrate's ~70x growth (job 464: 197.4 s/step
# at cgh64, the largest phase, vs 2.80 at cdev). One-axis drift ladder at fixed
# config separates the two variables a drift sweep moves together: migrant
# VOLUME (continuous; three reach-1 rungs fit its slope) and staged DEPTH
# (discrete; reach-2/3 rungs read against that fit). Cross-config at matched f
# reads N scaling at REAL volume and depth -- the regime jobs 456/457/461 never
# entered (their reach was pinned at 1 with near-zero drift).
#
# PRE-REGISTERED readings, in order of diagnostic weight:
#   - If cgh64/cdev at matched f is ~8x (linear) at every rung, then N and
#     depth do NOT explain job 464's ~70x, and the difference must live in the
#     ENGINE's migrate call pattern vs this probe's (e.g. per-step scale
#     snapshots, engine-state occupancy structure) -- a different hunt.
#   - If the depth excess at reach 3 is large (total >> volume-law prediction),
#     depth is the mechanism and the fix is scheduling/staging work.
#   - If the volume slope alone carries it, the cost is honest row work and
#     the lever is parallelizing eject/insert (C7), not a defect fix.
# Same machine as jobs 459/464 so the numbers sit beside the phase table.
#
# --mem=64G is ROUTING (>56G keeps it off deneb) + headroom over the ~25 GB
# build peak at 512^3. Zero SU. ~2 h expected, 5 h requested.
# NO set -e: per-leg exit codes; smoke gates the long leg.

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
STAMP="/tmp/m6_w0b_stamp_$SLURM_JOB_ID"; touch "$STAMP"

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

run_leg "smoke" pixi run -e default python $PROBE --config smoke \
  --out runs/v2/m6_migrate_depth_smoke.json
[ "${codes[0]}" -ne 0 ] && {
  echo "FATAL: smoke failed (a void rung or a crash); not burning the long leg"
  exit 1
}

run_leg "depth-cdev-cgh64" pixi run -e default python $PROBE \
  --config cdev cgh64 --out runs/v2/m6_migrate_depth.json

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST: job 464 migrate 197.4 s/step at cgh64 / 2.80 at cdev (~70x for 8x N)"
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m6_migrate_depth*.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"
rm -f "$STAMP"
exit $rc_total
