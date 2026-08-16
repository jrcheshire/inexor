#!/bin/bash
#SBATCH -J m6-c5-migrate
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m6-c5-migrate-%j.log
# C5: does `migrate` PARALLELISE, or does it inflate like `tile_long`?
#
# Section 5l closed the per-brick lane (the Python slab loop is 2.0% of migrate
# at the 4,096 rows/brick the config table fixes at EVERY rung), so migrate is
# per-row work and parallelism is the only lever left. 5j is why it matters:
# migrate is 32.78 s of a 72.64 s W=16 step on gg and 23.3 h of a projected
# 51.7 h C-gh realization, and it is host numpy -- neither the tile pool nor
# the GPU lane touches it today.
#
# WHY gg AND NOT antares. This is a BANDWIDTH question, and 5h/5i established
# that bottleneck identity is hardware-specific: antares capped at 1.4x and the
# laptop at 2.7x on tile work where Grace reached ~19x. A scaling verdict from
# antares would be uninformative about the target machine. Same reasoning that
# moved the C2 verdict to gg (913729).
#
# PRE-REGISTERED, and it can embarrass me: I expect migrate to scale BETTER
# than tile_long. tile_long inflates 4.13x in worker-seconds at W=32 because
# FFTs are bandwidth-hungry per unit of work; migrate's traffic is small in
# aggregate (generously counting the f64 decode transients, eject moves order
# tens of GB in ~13 s = a few GB/s against gg's ~500 GB/s/socket, order 1% of
# the fabric).
#   KILL CRITERION: efficiency < 50% at W=8 kills the pool lane for migrate,
#   and 51.7 h/realization is then near a floor with an architectural answer.
# If it fails to scale anyway, DRAM bandwidth is NOT the cause and the
# candidates are allocator/page-fault serialisation or per-brick call overhead
# going superlinear under concurrency -- a different fix, worth knowing early.
#
# THE AMDAHL BOUND, fixed before the run so no arm can be over-read: job 478
# measured eject at 12.99 s of migrate's 33.26 s, so PERFECT eject parallelism
# gives migrate 33.3 -> 20.3 s = 1.64x on the phase and no more. `_insert_slab`
# carries the other 19.7 s and needs the disjoint-write premise proved as C2
# proved it for tiles -- a separate job, deliberately NOT this one.
#
# W caps at 32 because the task population IS nb = 32 slabs at cgh64; W=32 is
# already one task per worker and anything above it only adds idle workers.
#
# Transport is an ARM (discard = compute ceiling / counts = + task overhead /
# pickle = what a naive implementation pays), so "the compute scales but the
# transport eats it" cannot hide inside one number.
#
# Ops: the installed Vista env is `gpu`; JAX_PLATFORMS=cpu makes it CPU and
# CONDA_OVERRIDE_CUDA gets past the activation-time cuda check on a GPU-less
# gg node (the 866415 precedent). This canary imports no jax itself, but
# `inexor` does at package import.
# ~15 min expected, 1.5 h requested. gg bills 0.33 SU/node-hour -> ~0.1 SU.
# NO set -e: per-leg exit codes; the smoke gates the long legs.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
echo "=== cwd $(pwd)"
export JAX_PLATFORMS=cpu
export CONDA_OVERRIDE_CUDA=12.0
pixi run -e gpu python -c "
import jax
assert jax.default_backend() == 'cpu', 'expected the CPU backend'
print('jax', jax.__version__)
" || { echo "FATAL: jax import/backend check failed"; exit 1; }

echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2
STAMP="/tmp/m6_c5_stamp_$SLURM_JOB_ID"; touch "$STAMP"

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

PROBE="scripts/v2_m6_c5_migrate_pool.py"

# 1. plumbing on THIS machine and commit: shm adoption, spawn, affinity, all
#    three transports, the card write. Its speedups are HARNESS-DOMINATED by
#    construction and the script says so itself -- this leg proves the paths
#    execute, nothing more.
run_leg "c5-smoke" pixi run -e gpu python $PROBE \
  --config smoke --workers 2 4 --transport discard counts pickle \
  --out runs/v2/m6_c5_smoke.json
[ "${codes[0]}" -ne 0 ] && {
  echo "FATAL: smoke failed (rc=${codes[0]}); not burning the long legs"
  exit 1
}

# 2. THE VERDICT: cgh64, pinned, all three transports.
run_leg "c5-cgh64-pinned" pixi run -e gpu python $PROBE \
  --config cgh64 --workers 4 8 16 32 --transport discard counts pickle \
  --out runs/v2/m6_c5_cgh64_pinned.json

# 3. the affinity A/B -- 5i measured pinning worth 20-49% on gg's two NUMA
#    domains for tile work, and whether that carries to numpy row work is a
#    different question with the same answer shape.
run_leg "c5-cgh64-unpinned" pixi run -e gpu python $PROBE \
  --config cgh64 --workers 4 8 16 32 --transport discard --no-affinity \
  --out runs/v2/m6_c5_cgh64_unpinned.json

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST: eject 12.99 s of migrate 33.26 s at cgh64/nb32 (antares 478) ->"
echo "         phase ceiling 1.64x even at perfect eject parallelism;"
echo "         tile_long busy-inflation 4.13x at W=32 (Vista 914085, 5j);"
echo "         C2 tile pool on gg 10.4x at W=8, pinning worth 20-49% (913729)"
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m6_c5_*.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"
rm -f "$STAMP"
exit $rc_total
