#!/bin/bash
#SBATCH -J m6-rlz-pilot
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m6-rlz-pilot-%j.log
# LEG 1 of the realization: the pilot. Everything at 2048^3 EXCEPT the 35 steps.
#
# WHY A PILOT AND NOT THE RUN. Five things have never executed at this scale:
# a full IC generation end to end (M-v2-5's record lists it under "what this
# does NOT establish" -- the ~1.25 TB staging budget is priced from measured
# disk rates, not demonstrated); the engine's end-to-end peak host memory (open
# since M-v2-4); the pool's own footprint; checkpoint/resume; and the export
# plus the P(k) card, which before today had no caller outside their tests. An
# OOM at hour nine of a 20 h job costs nine hours and returns one bit.
#
# WHAT IT CONVERTS FROM PROJECTION TO MEASUREMENT. Every hour figure in this
# project is `cgh64 x 64 x K=40` arithmetic on a phase card; no realization has
# ever been run. This leg produces the first measured 2048^3 step.
#
# PRE-REGISTERED (before submission):
#   a. TOTAL HOST under 200 GB of the node's 237 during stepping, where TOTAL
#      means parent peak PLUS the pool's worker processes. Above 200 GB, LEG 2
#      IS NOT SUBMITTED until the term is found -- this is the gate, not a
#      reading.
#
#      READ `total_rss_bytes`, NOT `peak_rss_bytes`. The 164.6 GB on record is a
#      parent-side figure and `ru_maxrss` cannot see other processes at all;
#      `v2_m6_engine_peak` is serial-only for exactly that reason. At cdev8 with
#      W=4 the workers held 10.4 GB against the parent's 0.5 -- 21x -- so the
#      worker term is not a correction, it can be the larger half. At W=16 it is
#      the term most likely to break the node fit, and it has never been measured
#      at this scale.
#   b. PER-STEP within 1.5x of the projected 1971 s (32.9 min), i.e. under
#      49 min/step. A miss HIGH is expected rather than surprising: migrate's
#      C-gh term is a FLOOR (its sort is N log N and its staging N^(2/3),
#      neither measured above 512^3), and finding out by how much is half the
#      point of this leg.
#   c. REPACK FAST-PATH SHARE 85-97% at 2.1M bricks. C15 measured 88.7-96.2% at
#      32,768. Below it, spill dusts across bricks at full scale and repack's
#      4.85x is smaller here.
#   d. IC GENERATION COMPLETES and its staging cleanup leaves under 1 GB
#      behind. That stage strands ~687 GB per run if the cleanup does not fire.
#   e. THE POOL RUNS 16 WORKERS. Its memory at this scale is unmeasured and may
#      cap W lower, which would move every wall number on the card.
#
# WHAT IS GATED IN THE CODE (so it is not re-derived from this comment):
# `write_particles` refuses a short file; `pk_summary_card` refuses an empty
# card; `stop_at` refuses to stop off a checkpoint boundary; the driver refuses
# a non-CPU backend and refuses to start from ICs when a checkpoint exists.
#
# BUDGET: ~1.5-2 h expected against a 4 h request. gg is 0.33 SU/node-hr:
# ~0.6 SU expected, 1.3 SU worst case. Scratch has 5.1 PB free and no quota;
# this leg writes ~206 GB of export, ~154 GB of rolling checkpoint, and
# ~687 GB of staging that the cleanup then removes.
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
df -h "$SCRATCH" | tail -1

W="$SCRATCH/inexor_runs/c-gh-r0"
WS="$SCRATCH/inexor_runs/cdev8-smoke"
RLZ="scripts/v2_m6_realization.py"
mkdir -p "$SCRATCH/inexor_runs"

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

# 1. THE SMOKE: all four phases end to end at cdev8, on this machine and
#    commit, before a byte of 2048^3 work. The driver is new and its reporting
#    path has already produced one crash-after-success on the laptop.
rm -rf "$WS"
run_leg "smoke-ics"    pixi run -e gpu python $RLZ ics    --config cdev8 --workdir "$WS"
run_leg "smoke-run-a"  pixi run -e gpu python $RLZ run    --config cdev8 --workdir "$WS" \
  --tile-workers 4 --stop-at 5
run_leg "smoke-run-b"  pixi run -e gpu python $RLZ run    --config cdev8 --workdir "$WS" \
  --tile-workers 4
run_leg "smoke-export" pixi run -e gpu python $RLZ export --config cdev8 --workdir "$WS" \
  --tile-workers 4
run_leg "smoke-card"   pixi run -e gpu python $RLZ card   --config cdev8 --workdir "$WS" \
  --tile-workers 4
for i in 0 1 2 3 4; do
  if [ "${codes[$i]}" -ne 0 ]; then
    echo "FATAL: smoke leg ${names[$i]} rc=${codes[$i]}; not starting 2048^3"
    exit 1
  fi
done

# 2. ICs at 2048^3. The stage that has never run end to end. `keep_stage` is
#    NOT passed, so the cleanup fires and criterion (d) is readable.
run_leg "cgh-ics" pixi run -e gpu python $RLZ ics --config c-gh --workdir "$W"
if [ "${codes[5]}" -ne 0 ]; then
  echo "FATAL: IC generation rc=${codes[5]}; nothing downstream can run"
  echo "=== LEG SUMMARY ==="
  for j in "${!names[@]}"; do echo "  ${names[$j]}: rc=${codes[$j]}"; done
  exit 1
fi
echo "--- staging left behind (criterion d, want < 1 GB) ---"
du -sh "$W/stage" 2>/dev/null || echo "  stage/ is gone, as intended"
du -sh "$W" 2>/dev/null

# 3. FIVE STEPS. The first measured 2048^3 step, and the checkpoint leg 2
#    resumes from.
run_leg "cgh-run-5" pixi run -e gpu python $RLZ run --config c-gh --workdir "$W" \
  --tile-workers 16 --checkpoint-every 5 --stop-at 5

# 4. Export and card AT STEP 5, deliberately: these are path tests at scale,
#    not science. --allow-partial says so out loud. Leg 2 writes the real one
#    into its own directory, so this cannot be mistaken for the product.
if [ "${codes[6]}" -eq 0 ]; then
  run_leg "cgh-export-partial" pixi run -e gpu python $RLZ export --config c-gh \
    --workdir "$W" --tile-workers 16 --allow-partial \
    --export-dir "$W/export_pilot"
  run_leg "cgh-card-partial" pixi run -e gpu python $RLZ card --config c-gh \
    --workdir "$W" --tile-workers 16
else
  echo "STEPPING FAILED: skipping export and card -- they would describe a state"
  echo "the engine did not finish producing."
fi

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "=== DISK ==="
du -sh "$W"/* 2>/dev/null
echo "READ THE CARDS, in this order:"
echo "  $W/realization_ics.json      -> criterion (d), and the IC wall nobody has"
echo "  $W/realization_run_00_05.json -> criteria (a) (b) (c) (e):"
echo "     total_rss_bytes < 200e9 (NOT peak_rss_bytes -- that misses the workers)"
echo "     s_per_step < 2940 | repack fast share | migrate_pooled_workers = 16"
echo "     and projected_full_run_h, which REPLACES the 21.9 h arithmetic"
echo "  $W/realization_export.json / realization_pk.json -> the paths work at scale"
echo "AGAINST: projected 1971 s/step (21.9 h for K=40) and 164.6 GB peak."
echo "LEG 2 (scripts/v2_m6_rlz_run_gg_vista.sbatch) IS NOT AUTOMATIC: submit it"
echo "only after criterion (a) is read and passed."
exit $rc_total
