#!/bin/bash
#SBATCH -J m6-c8b-pool
#SBATCH -p compute
#SBATCH --mem=100G
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -o m6-c8b-pool-%j.log
# C8b: the pooled pair 482 lost to an OOM, re-run with room.
#
# 482's `cgh64-w8-jax` was SIGKILLed at 1685 s (JobState=OUT_OF_MEMORY, 3
# oom-kill events, no card). The shape named the cause: 6.54 GB anon + 1.77 GB
# shm per worker, so 8 workers is ~52 GB of private memory before the parent,
# and the matched numpy leg reported rss_mb_max 8710 MB. The compiled eject runs
# in the PARENT (the pool owns the tile loop, not the migration), adding padded
# slab arrays and an XLA CPU arena on top of a pool already near the request.
#
# THE REQUEST WAS THE PROBLEM, NOT THE NODE: antares carries 124 GB and I asked
# for 64. --mem=100G here leaves ~24 GB of node headroom rather than filling it,
# because --mem is a cgroup ceiling and the point is to measure the footprint,
# not to discover a new one.
#
# BOTH ARMS RE-RUN, not just the one that died. The numpy leg completed at 64 GB
# and its number is on record, but a pair measured under different cgroup
# ceilings in different jobs is a comparison with two variables (5k: same-job
# pairing is what settled the +8.6%/+6.0% drift that looked like an effect).
#
# PRE-REGISTERED:
#   a. both legs complete. If the jax arm OOMs at 100 GB the finding is much
#      larger than a memory knob and W=8 is not an operating point for it.
#   b. the migrate ratio is 1.05-1.20x, bracketing 5r's serial 1.10x. The pool
#      does not touch migrate (5j: that is exactly why migrate is 45% of a W=16
#      step), so pooling should leave the ratio alone.
#   c. the jax arm's peak is 1-3 GB above the numpy arm's, parent-side. Larger
#      means the compiled path costs more than its arrays and the accounting is
#      wrong somewhere.
#
# The card carries `eject_jax_calls`: a jax leg reading 0 measured nothing.
# ~50 min expected per leg, ~1.7 h total, 4 h requested. Zero SU.
# NO set -e. rc=2 is instrument non-neutrality, not a crash (the 481 lesson).

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
BIG="--config cgh64 --k 3 --repeats 1 --slack 0.20 --arena-frac 0.20"

run_leg "cgh64-w8-numpy" pixi run -e default python $PHASE $BIG \
  --eject-kernel numpy --tile-workers 8 --out-suffix _c8b_w8_numpy
run_leg "cgh64-w8-jax" pixi run -e default python $PHASE $BIG \
  --eject-kernel jax --tile-workers 8 --out-suffix _c8b_w8_jax

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST: 482 at --mem=64G -- w8-numpy rc=0 (migrate 22.46 s/step,"
echo "         rss_mb_max 8710 MB), w8-jax rc=137 OOM at 1685 s, no card."
echo "         Serial pair on the same machine: 22.13 -> 20.16 s/step = 1.10x."
echo "CARDS WRITTEN BY THIS JOB:"
ls -1 runs/v2/m6_phase_time_c8b_*.json 2>/dev/null | sed 's/^/  /'
exit $rc_total
