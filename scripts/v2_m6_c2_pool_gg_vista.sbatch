#!/bin/bash
#SBATCH -J m6-c2-gg
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m6-c2-gg-%j.log
# C2's scaling verdict on the memory system it was designed for. Antares
# measured aggregate tile throughput FLAT in W (~6.4 tiles/s at every worker
# count, idle ~0, per-tile busy inflating linearly 296 -> 2555 ms over
# W=2..16) with affinity pinning changing walls < 5% -- a shared-resource
# ceiling, most plausibly memory bandwidth, on a machine that is neither the
# target nor representative of it. A gg node (2x 72-core Grace, ~500 GB/s
# LPDDR per socket, 237 GB) is the production candidate; this is where the
# plan's kill criterion is actually evaluated.
#
# PRE-REGISTERED (the C2 criteria, applied HERE):
#   - bitwise nonzero at any W kills the parallel premise;
#   - efficiency < 50% at W=8 (pinned) -> the pool design fails ON TARGET
#     hardware and W2 stops for a redesign;
#   - the interesting upside readout is W=32/64: where does Grace's bandwidth
#     cap the pool, read from the busy/idle/concurrency triple;
#   - P=320 RSS/worker prices the gg width arithmetic (antares read 5.6-5.8
#     GB/worker -> ~20 workers; if Grace reproduces it, that stands).
# Pinned and un-pinned ladders both run: antares showed pinning ~neutral on
# walls, but gg is 2 NUMA domains and first-touch placement may matter (C4).
#
# CHARGING: gg = 0.33 SU/node-hr; ~30-40 min expected, 1.5 h wall requested
# -> < 0.5 SU billed on actuals.
# Ops: JAX_PLATFORMS=cpu AND CONDA_OVERRIDE_CUDA are both LOAD-BEARING on a
# GPU-less gg node (jax 0.10 cuInit hard-raise; pixi cuda check at
# activation). NO set -e: per-leg exit codes; smoke gates the long legs.

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
STAMP="/tmp/m6_c2gg_stamp_$SLURM_JOB_ID"; touch "$STAMP"

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

run_leg "smoke" pixi run -e gpu python $C2 --config cdev8 --tile 64 \
  --workers 2 --repeats 1 --affinity --out runs/v2/m6_c2_pool_gg_smoke.json
[ "${codes[0]}" -ne 0 ] && {
  echo "FATAL: smoke failed (bitwise or crash); not burning the long legs"
  exit 1
}

run_leg "gg-pinned" pixi run -e gpu python $C2 --config cdev8 --tile 64 \
  --workers 8 16 32 64 --repeats 3 --affinity \
  --out runs/v2/m6_c2_pool_gg_pinned.json

run_leg "gg-unpinned" pixi run -e gpu python $C2 --config cdev8 --tile 64 \
  --workers 8 16 32 --repeats 3 \
  --out runs/v2/m6_c2_pool_gg_unpinned.json

run_leg "gg-p320" pixi run -e gpu python $C2 --config cdev --tile 256 \
  --workers 8 16 --repeats 2 --affinity \
  --out runs/v2/m6_c2_pool_gg_cdev.json

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST: antares flat ~6.4 tiles/s at every W; laptop 2.7x at W=8; RSS 5.6-5.8 GB/worker at P=320"
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m6_c2_pool_gg_*.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"
rm -f "$STAMP"
exit $rc_total
