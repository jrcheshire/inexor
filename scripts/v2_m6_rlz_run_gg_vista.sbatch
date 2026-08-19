#!/bin/bash
#SBATCH -J m6-rlz-run
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 30:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m6-rlz-run-%j.log
# LEG 2: the realization. M-v2-6's deliverable -- a complete 2048^3 mock on ONE
# node, and the first MEASURED realization wall in this project.
#
# DO NOT SUBMIT THIS UNTIL THE PILOT'S CRITERION (a) HAS BEEN READ. The pilot
# (v2_m6_rlz_pilot_gg_vista.sbatch) measures the engine's end-to-end peak host
# memory at 2048^3, which is projected at 164.6 GB against a 237 GB node and has
# never been measured. Submitting this first risks 20 h to learn it does not fit.
# The pilot also sets this job's wall: the -t above is sized from the PROJECTED
# 1971 s/step, and should be re-sized from the pilot's measured one.
#
# THIS SCRIPT IS RESUBMITTABLE, AND THAT IS THE RECOVERY PLAN. The driver reads
# the newest complete checkpoint and continues from it, so a timeout, a node
# failure or a preemption costs at most the segment since the last checkpoint
# (every 5 steps, ~2.7 h at the projected rate). Resubmitting the identical
# script picks the run back up. There is nothing to edit between attempts.
#
# It resumes from the PILOT's step-5 checkpoint in the same workdir, so the
# schedule is one continuous trajectory: each segment is handed the full K=40
# coefficient list and told where to stop, never a truncated schedule, because
# `fused_drifts` makes `coeffs[:n]` a different trajectory rather than a prefix.
# The segments compose back into the uninterrupted run BITWISE
# (test_a_run_split_into_segments_is_bitwise_the_uninterrupted_one).
#
# PRE-REGISTERED (before submission):
#   a. IT FINISHES: 40 of 40 steps, a checkpoint at step 40, an export whose own
#      conservation check passes, and a P(k) card with a non-empty z profile.
#      That conjunction IS the milestone's bar; nothing here is a tuning knob.
#   b. THE MEASURED WALL, against 21.9 h projected. This is the number the leg
#      exists to produce and it is REPORTED, not gated -- a realization that
#      takes 30 h is still the first realization, and the 5 h/realization target
#      is a milestone-level question, not this job's pass criterion.
#   c. PEAK HOST stays under the node across all 40 steps, not just the pilot's
#      five. Clustering grows the arena and the migrate staging, so the pilot's
#      peak is a lower bound on this one.
#   d. THE PRODUCT IS READABLE BY SOMETHING THAT IS NOT THIS CODE: plain .npy,
#      x and v, with a header carrying units, epoch and crc32.
#
# BUDGET: ~20 h expected against a 30 h request (gg QOS caps at 48 h). gg is
# 0.33 SU/node-hr: ~6.6 SU expected, 9.9 SU worst case, against 9,072 available
# on JPL-SPHEREx expiring 2026-11-30. The binding constraint is queue wait for a
# 30 h single-node job, not SUs and not disk.
#
# OPS (gg, all load-bearing): pixi on PATH; -e gpu; JAX_PLATFORMS=cpu (jax 0.10
# hard-raises on cuInit on a GPU-less node); CONDA_OVERRIDE_CUDA; -A JPL-SPHEREx
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
RLZ="scripts/v2_m6_realization.py"

# REFUSE to start without the pilot's ICs and checkpoint. Regenerating 2048^3
# ICs inside a 30 h job because a path was wrong would silently eat an hour and
# start a DIFFERENT realization from the one the pilot measured.
if [ ! -f "$W/manifest.json" ]; then
  echo "FATAL: no IC manifest at $W -- run the pilot first"
  exit 1
fi
if [ ! -f "$W/ckpt/gen0/manifest.json" ] && [ ! -f "$W/ckpt/gen1/manifest.json" ]; then
  echo "FATAL: no checkpoint at $W/ckpt -- the pilot's stepping leg did not finish"
  exit 1
fi
echo "=== resuming from ==="
for g in gen0 gen1; do
  [ -f "$W/ckpt/$g/manifest.json" ] && \
    echo "  $g step=$(grep -o '\"step\"[^,]*' "$W/ckpt/$g/manifest.json" | head -1)"
done

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

# 1. THE RUN. No --stop-at: it goes to step 40 or until the wall ends it, and
#    either way the checkpoints make the next submission continue.
run_leg "cgh-run-to-40" pixi run -e gpu python $RLZ run --config c-gh --workdir "$W" \
  --tile-workers 16 --checkpoint-every 5

# 2. Export and card, only if the run reached the end. `export` refuses a
#    partial state unless --allow-partial, which is NOT passed here: an export
#    at step 33 is a mock at the wrong epoch, and it must not be produced by
#    accident and then found in a directory next year.
if [ "${codes[0]}" -eq 0 ]; then
  run_leg "cgh-export" pixi run -e gpu python $RLZ export --config c-gh \
    --workdir "$W" --tile-workers 16
  run_leg "cgh-card" pixi run -e gpu python $RLZ card --config c-gh \
    --workdir "$W" --tile-workers 16
else
  echo "THE RUN DID NOT REACH STEP 40 (rc=${codes[0]}). This is the expected"
  echo "shape of a timeout, and it is RECOVERABLE: resubmit this same script."
  echo "It will resume from the newest checkpoint. Nothing needs editing."
fi

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "=== THE PRODUCT ==="
ls -lh "$W/export" 2>/dev/null | head
du -sh "$W"/* 2>/dev/null
echo "=== READ ==="
echo "  $W/realization_run_*.json  -> wall_s, s_per_step, peak_rss_bytes per SEGMENT"
echo "  $W/realization_export.json -> the manifest, crc32, units and epoch"
echo "  $W/realization_pk.json     -> the z profile. NO VERDICT is emitted; pick"
echo "     the band with band_verdict(card, k_max) when the k range is decided."
echo "AGAINST: 21.9 h projected for K=40 (arithmetic on the C14+C15 phase card at"
echo "  1 s/step at cgh64 = 0.711 h at 2048^3). No realization had ever been run."
exit $rc_total
