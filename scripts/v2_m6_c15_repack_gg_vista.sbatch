#!/bin/bash
#SBATCH -J m6-c15-repack
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m6-c15-repack-%j.log
# C15 = does the repack fast path transfer from the laptop to the production
# geometry? The laptop measured, on the ENGINE's own clustered state at cdev:
# 89-96% of occupied bricks carry no arena residents, the pre-C15 pass B cost
# 26x pass A over the same rows, and skipping the permutation for arena-free
# bricks bought a MEDIAN 3.66x on the phase, bitwise identical output.
#
# cgh64 has 32,768 bricks against cdev's 4,096 at the same 4096 rows/brick, and
# 64x the volume of structure. The laptop number bounds NOTHING here in either
# direction (the standing dev-config lesson), so this leg exists to measure the
# same two things at the config the ratified 4.74 s/step card was measured at.
#
# PRE-REGISTERED (before submission):
#   a. ARENA CONCENTRATION TRANSFERS: arena-free share of occupied bricks lands
#      in 80-98% at cgh64, the cdev band. Materially below it means spill dusts
#      across bricks at scale instead of concentrating, the fast path is rarer
#      than the laptop says, and the payoff shrinks with it. That is the
#      informative miss, not a failure.
#   b. MEDIAN SPEEDUP >= 2.5x over the k=6 steps. Below 1.5x the fast path does
#      not transfer and repack pooling goes back on the table as the main lever.
#   c. THE BITWISE GATE HOLDS AT 32,768 BRICKS. It has run at 512 and 4,096
#      only. A miss here is a state-container defect and outranks every timing.
#   d. THE BASELINE IS THE PHASE THE CARD MEASURED: baseline repack within 20%
#      of C14's 4.74 s/step at matched knobs. If it is not, the speedup is still
#      a valid ratio but the wall arithmetic must NOT be re-derived off it until
#      the discrepancy is explained.
#
# THE RECEIPTS (a knob must prove it applied): the leg prints "fast path taken
# N / 0 times in fast / baseline". A baseline arm with N > 0 means the A/B
# compared the new code with itself, and the script exits 2 rather than
# reporting a speedup it cannot support.
#
# WHY THE EXECUTION KNOBS ARE SET THE WAY THEY ARE: repack is indifferent to
# tile_workers and to whether migrate pooled (C14 proved the pooled pass is
# bitwise the serial one, so the state reaching repack is identical). Both are
# set to the fast configuration purely to keep this job's wall down. The knobs
# that DO matter -- slack 0.20, arena_frac 0.20, config cgh64 -- are held
# exactly at C14's so criterion (d) is a like-for-like comparison.
#
# BUDGET: ~20 min expected against a 1.5 h request. gg is 0.33 SU/node-hr:
# ~0.1 SU expected, 0.5 SU worst case.
#
# OPS (gg, all load-bearing): pixi on PATH; -e gpu (default env not installed
# on Vista); JAX_PLATFORMS=cpu (jax 0.10 hard-raises on cuInit on a GPU-less
# node); CONDA_OVERRIDE_CUDA (env activation gate); -A JPL-SPHEREx
# case-sensitive. NO set -e: per-leg exit codes.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
export JAX_PLATFORMS=cpu
export CONDA_OVERRIDE_CUDA=12.0
pixi run -e gpu python -c "
import jax
assert jax.devices()[0].platform == 'cpu', 'expected the CPU backend'
print('jax', jax.__version__)
" || { echo "FATAL: jax import/backend check failed"; exit 1; }

echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2

rc_total=0
names=(); codes=()
# `run_leg NAME CMD...`. rc=2 here means a GATE REFUSED -- the transcription
# drifted, the arms were not separated, or repack disagreed with its oracle. No
# code is tolerated on this job: every nonzero exit is a real refusal, because
# unlike the phase probe this instrument has no "reportable non-neutrality"
# state. rc=2 must fail the job so the refusal is not read as a pass.
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

C15="scripts/v2_m6_c15_repack.py"

# 1. smoke: the whole reporting path -- engine build, the repack hook, both
#    arms, all four gates, the card write -- at cdev8 on THIS machine and
#    commit, before any cgh64 time is spent. The reporting path is untested
#    code until it has run somewhere.
run_leg "smoke-cdev8" pixi run -e gpu python $C15 \
  --config cdev8 --engine 3 --slack 0.20 --arena-frac 0.20 \
  --tile-workers 2 --migrate-pooled --eject-kernel jax \
  --out runs/v2/m6_c15_repack_engine_cdev8.json
if [ "${codes[0]}" -ne 0 ]; then
  echo "FATAL: smoke leg rc=${codes[0]}; not burning SU on cgh64"
  exit 1
fi

# 2. THE LEG. Knobs held at C14's so the baseline arm is comparable to the
#    ratified 4.74 s/step. k=6 so the arena-free share is read across a
#    trajectory rather than at one step -- it moved 89-96% over five cdev
#    steps and tracks how clustered the state is, so a single step could
#    land anywhere in that band by accident.
run_leg "cgh64-repack-ab" pixi run -e gpu python $C15 \
  --config cgh64 --engine 6 --slack 0.20 --arena-frac 0.20 \
  --tile-workers 16 --migrate-pooled --eject-kernel jax \
  --out runs/v2/m6_c15_repack_engine_cgh64.json

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST: C14 cgh64 W=16 jax pooled: repack 4.74 s/step, step 34.52 s"
echo "         (card m6_phase_time_c14_migpool_jax). Criterion (d) is that the"
echo "         BASELINE arm reproduces 4.74 within 20%."
echo "LAPTOP REFERENCE (cdev, 5 engine steps): 89-96% arena-free, median 3.66x,"
echo "         baseline pass B 26x pass A, fast path still 15-25x above floor."
echo "READ THE RECEIPT: 'fast path taken N / 0 times in fast / baseline'."
echo "CARDS WRITTEN BY THIS JOB:"
ls -1 runs/v2/m6_c15_repack_engine_cdev8.json \
      runs/v2/m6_c15_repack_engine_cgh64.json 2>/dev/null | sed 's/^/  /'
exit $rc_total
