#!/bin/bash
#SBATCH -J m6-c2-pool
#SBATCH -p compute
#SBATCH --mem=64G
#SBATCH -c 28
#SBATCH -t 02:00:00
#SBATCH -o m6-c2-pool-%j.log
# C2's Linux legs: the scaling and RSS verdicts the laptop cannot give.
# The laptop leg already established the load-bearing half -- bitwise
# n_diff(w) = n_diff(vel_scale) = 0 at W=2/4/8 with ARRIVAL-order writes, and
# realized concurrency 1.98/3.95/7.97 of 2/4/8 (workers never starved, so the
# pickled-return transport is not the bottleneck). Its 34%-at-W=8 efficiency
# is per-tile busy INFLATION (183 -> 537 ms at fixed inputs) = laptop memory
# contention, which is exactly the quantity that must be read on server
# memory instead.
#
# PRE-REGISTERED (the plan's C2 kill criteria, applied HERE, not on macOS):
#   - bitwise nonzero at any W kills the parallel premise outright;
#   - efficiency < 50% at W=8 on the cdev8/tile-64 leg -> redesign before
#     any engine surgery;
#   - RSS/worker > 4 GB on the cdev/tile-256 (P=320, production shape) leg
#     -> the ~35-45-worker gg width arithmetic fails, revisit before W3/W4.
# Antares = 28 cores, 2 sockets; W=16 leaves headroom for the parent.
#
# --mem=64G is ROUTING (>56G keeps it off deneb) + covers 8 workers x ~3 GB
# predicted at P=320. Zero SU. ~30 min expected, 2 h requested.
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
STAMP="/tmp/m6_c2_stamp_$SLURM_JOB_ID"; touch "$STAMP"

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

run_leg "smoke" pixi run -e default python $C2 --config cdev8 --tile 64 \
  --workers 2 --repeats 1 --out runs/v2/m6_c2_pool_smoke.json
[ "${codes[0]}" -ne 0 ] && {
  echo "FATAL: smoke failed (bitwise or crash); not burning the long legs"
  exit 1
}

# the scaling verdict: 64 tiles, W ladder to 16
run_leg "scaling-cdev8" pixi run -e default python $C2 --config cdev8 --tile 64 \
  --workers 2 4 8 16 --repeats 3 --out runs/v2/m6_c2_pool_cdev8.json

# the production-shape RSS/startup readout: P=320, cap ~5.3e6, 8 tiles
run_leg "p320-cdev" pixi run -e default python $C2 --config cdev --tile 256 \
  --workers 4 8 --repeats 2 --out runs/v2/m6_c2_pool_cdev.json

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST: laptop bitwise 0 at W=2/4/8; eff 85/60/34%; per-tile busy 183->537 ms"
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m6_c2_pool_*.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"
rm -f "$STAMP"
exit $rc_total
