#!/bin/bash
#SBATCH -J m6-w2-gg
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m6-w2-gg-%j.log
# W2's EXIT MEASUREMENT: the pooled ENGINE at cgh64 on the production
# candidate. C2 measured the canary's tile loop at 10.4x (W=8) with a ~19x
# ceiling on gg (913729) -- but the canary carried no serial parent phases.
# The real step does (coarse solve, membership, migrate, repack), so this job
# reads what survives: s/step against a SAME-MACHINE serial reference
# (antares' 457.5 does not transfer across architectures), the busy triple
# (realized concurrency / idle -- distinguishes not-enough-work from starved
# from contended), and RSS/worker at the production tile geometry, which is
# what prices the C-gh pool width.
#
# PRE-REGISTERED:
#   - the identity smoke must read n_diff 0 on Grace or everything stops;
#   - the C2-derived hope is the tile+coarse phases compressing ~W-fold at
#     W=8-16; the serial phases (migrate ~5%, membership, solve) cap the
#     step's overall factor -- Amdahl arithmetic from the serial leg's phase
#     table is the honest prediction, written before reading the pooled legs;
#   - W=32 vs W=16 reads the tile-count ceiling (64 tiles at T=256) and
#     Grace bandwidth together; no pre-registered number, it is the scan.
#
# CHARGING: gg = 0.33 SU/node-hr; ~3-3.5 h expected (the serial reference leg
# dominates), 6 h wall requested -> ~1-1.2 SU billed on actuals, 2 worst case.
# Ops: JAX_PLATFORMS=cpu AND CONDA_OVERRIDE_CUDA both LOAD-BEARING on a
# GPU-less gg node (jax 0.10 cuInit hard-raise; pixi cuda check at
# activation). NO set -e: per-leg exit codes; the smoke gates the long legs.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
export JAX_PLATFORMS=cpu
export CONDA_OVERRIDE_CUDA=12.0
pixi run -e gpu python -c "
import jax
assert jax.devices()[0].platform == 'cpu', 'expected the CPU backend'
print('jax', jax.__version__)
" || { echo "FATAL: jax import/backend check failed"; exit 1; }

echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2
STAMP="/tmp/m6_w2gg_stamp_$SLURM_JOB_ID"; touch "$STAMP"

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

IDENT="scripts/v2_m6_w2_identity.py"
PHASE="scripts/v2_m6_phase_time.py"

run_leg "id-smoke-cdev8" pixi run -e gpu python $IDENT \
  --config cdev8 --k 4 --workers 4 --tile 64 \
  --out runs/v2/m6_w2_identity_gg_cdev8.json
[ "${codes[0]}" -ne 0 ] && {
  echo "FATAL: the identity smoke failed on Grace; not burning the long legs"
  exit 1
}

run_leg "serial-cgh64" pixi run -e gpu python $PHASE \
  --config cgh64 --k 3 --repeats 1 --slack 0.20 --arena-frac 0.20 \
  --out-suffix _w2gg_serial

run_leg "pool-cgh64-w8" pixi run -e gpu python $PHASE \
  --config cgh64 --k 3 --repeats 1 --slack 0.20 --arena-frac 0.20 \
  --tile-workers 8 --out-suffix _w2gg_w8

run_leg "pool-cgh64-w16" pixi run -e gpu python $PHASE \
  --config cgh64 --k 3 --repeats 1 --slack 0.20 --arena-frac 0.20 \
  --tile-workers 16 --out-suffix _w2gg_w16

run_leg "pool-cgh64-w32" pixi run -e gpu python $PHASE \
  --config cgh64 --k 3 --repeats 1 --slack 0.20 --arena-frac 0.20 \
  --tile-workers 32 --out-suffix _w2gg_w32

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST: C2 gg 10.4x at W=8 / ~19x ceiling (913729); antares serial 457.5 s/step (467, NOT transferable); RSS 5.6 GB/worker at P=320"
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m6_w2_identity_gg_*.json runs/v2/m6_phase_time_w2gg_*.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"
rm -f "$STAMP"
exit $rc_total
