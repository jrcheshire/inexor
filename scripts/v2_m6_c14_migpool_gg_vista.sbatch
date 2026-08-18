#!/bin/bash
#SBATCH -J m6-c14-migpool
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m6-c14-migpool-%j.log
# C14 = the idle-half Stage 2 VERDICT: the pooled migrate (workers write brick
# payloads into shm, the parent replays the arena release/claim interleave
# serially -- drift_and_migrate_pooled) on the production machine, at the
# production scale. Two questions, in order: (1) is the pooled pass BITWISE
# the serial one at cluster scale (cdev, W=4, arena-occupied included), and
# (2) what does it do to the migrate phase at cgh64 W=16, against the same
# commit's serial arm in the same job.
#
# PRE-REGISTERED (before submission):
#   a. IDENTITY: every identity leg reads n_diff=0 on all arrays (ids
#      included), cap ladder + arena_base equal, receipts applied on every
#      step. A miss is rc=1, STOPS the perf legs, and stops adoption -- the
#      parallel premise is then wrong, not tunable.
#   b. cgh64 W=16 jax: the migrate phase drops from ~24.95 s/step (C9 serial
#      card, matched knobs) into a 4-8 s/step band (insert 58-72% of the
#      phase pooling at C5-like efficiency). A miss is reportable and does
#      not void the identity result.
#   c. spill transport is negligible: migrate_pool.spill_bytes_median under
#      100 MB/step at cgh64 (slack 0.20 keeps overflow rare).
#   d. scratch stays cheap: migrate_pool.scratch_mb well under the ~15.5 GB
#      the serial pass stages (2r+2 slabs x ~190 B/particle-row at cgh64).
# THE RECEIPTS (a knob must prove it applied): migrate_pooled_workers = W on
# EVERY step of every pooled leg and 0 on every serial one; on jax legs the
# WORKER-summed migrate_pool.eject_jax_calls_per_step = [nb] (the parent
# counter reads 0 on pooled cards by construction). A pooled leg without its
# receipt measured nothing.
#
# RIDE-ALONG (free, threads note): the C13 insert census confirmation at
# cgh64 -- laptop proved nbs8/nbs16, this confirms the disjoint-write license
# at the production geometry. Independent of the pooled code (it drives the
# SERIAL migrate), so it runs even if identity fails.
#
# BUDGET: ~75 min expected against a 3 h request. gg is 0.33 SU/node-hr:
# ~0.4 SU expected, 1.0 worst case.
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
# `run_leg [-t CODES] NAME CMD...`. CODES is a comma list of NONZERO exit codes
# THIS leg pre-registers as reportable states rather than failures; anything
# else nonzero fails the job. Per-leg and not global on purpose: a code that is
# a known state for one instrument is a real failure for another, and a blanket
# tolerance would turn a genuine refusal into a silent pass.
#   rc=2 = the phase probe reporting its own instrument non-neutrality (5s/5t)
#   rc=3 = the census withholding a verdict (arena unprobed)
# Job 918684 exited 1 with every leg fine and every criterion readable, because
# this list was hardcoded to {3} while two legs legitimately read rc=2.
run_leg() {
  local tol=""
  [ "$1" = "-t" ] && { tol="$2"; shift 2; }
  local name="$1"; shift
  echo "=== $name ==="
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "=== $name rc=$rc after $((SECONDS - t0)) s ==="
  names+=("$name"); codes+=("$rc")
  local ok=0 c
  [ $rc -eq 0 ] && ok=1
  for c in $(echo "$tol" | tr ',' ' '); do [ "$rc" = "$c" ] && ok=1; done
  [ $ok -eq 0 ] && rc_total=1
  return 0
}

IDENT="scripts/v2_m6_w2_identity.py"
PHASE="scripts/v2_m6_phase_time.py"
CENSUS="scripts/v2_m6_c13_insert_census.py"
# knobs held EXACTLY at C9/914085's so the phase rows are comparable to the
# 24.95 s/step serial reference
BIG="--config cgh64 --k 3 --repeats 1 --slack 0.20 --arena-frac 0.20 --eject-kernel jax"

# 1. smoke: every path the big legs use, at cdev8, on THIS machine and
#    commit. The identity smoke is the whole pooled machinery end to end; the
#    phase smoke is the reporting path (rc=2 non-neutrality tolerated there).
run_leg "smoke-ident-w2" pixi run -e gpu python $IDENT \
  --config cdev8 --k 3 --workers 2 --migrate-pooled
run_leg -t 2 "smoke-phase-w4" pixi run -e gpu python $PHASE \
  --config cdev8 --k 3 --repeats 1 --arena-frac 0.20 --eject-kernel jax \
  --tile-workers 4 --migrate-pooled --out-suffix _c14_smoke_migpool
for i in 0 1; do
  rc=${codes[$i]}
  if [ "$i" -eq 1 ] && [ "$rc" -eq 2 ]; then
    # tolerated by the -t above; do NOT reset rc_total here, that would clear
    # an earlier leg's real failure along with this leg's known state
    echo "NOTE: ${names[$i]} non-neutral (rc=2); smoke exercises paths. Continuing."
  elif [ "$rc" -ne 0 ]; then
    echo "FATAL: smoke leg ${names[$i]} rc=$rc; not burning SU on cgh64"
    echo "=== LEG SUMMARY ==="
    for j in "${!names[@]}"; do echo "  ${names[$j]}: rc=${codes[$j]}"; done
    exit 1
  fi
done

# 2. THE IDENTITY GATE at cdev, W=4, plus an arena-saturated cdev8 arm
#    (slack 0.0: overflow every step, so the release AND claim replays are
#    both load-bearing, not idle -- the C13 arena_probed lesson).
run_leg "ident-cdev-w4" pixi run -e gpu python $IDENT \
  --config cdev --k 4 --workers 4 --migrate-pooled
run_leg "ident-cdev8-arena" pixi run -e gpu python $IDENT \
  --config cdev8 --k 3 --workers 4 --slack 0.0 --migrate-pooled \
  --out runs/v2/m6_w2_identity_cdev8_arena_migpool.json
IDENT_OK=1
[ "${codes[2]}" -ne 0 ] && IDENT_OK=0
[ "${codes[3]}" -ne 0 ] && IDENT_OK=0

# 3. the perf pair at cgh64 W=16, same commit, same job -- only if the
#    identity held (perf numbers for a wrong engine rank nothing).
if [ "$IDENT_OK" -eq 1 ]; then
  run_leg -t 2 "cgh64-serial-jax" pixi run -e gpu python $PHASE $BIG \
    --tile-workers 16 --out-suffix _c14_serial_jax
  run_leg -t 2 "cgh64-migpool-jax" pixi run -e gpu python $PHASE $BIG \
    --tile-workers 16 --migrate-pooled --out-suffix _c14_migpool_jax
else
  echo "IDENTITY FAILED: skipping the cgh64 perf pair -- the parallel premise"
  echo "is wrong and its wall numbers would describe a different engine."
fi

# 4. ride-along: the C13 census confirmation at the production geometry
#    (serial migrate; independent of the pooled code). rc=3 = verdict
#    withheld (arena unprobed) -- reportable, not a job failure.
run_leg -t 3 "census-cgh64" pixi run -e gpu python $CENSUS \
  --config cgh64 --calls 2 \
  --out runs/v2/m6_c13_insert_census_cgh64.json

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST: C9 serial cgh64 W=16 jax: migrate 24.95 s/step, step 63.14 s"
echo "         (card m6_phase_time_c9_w16_jax; _c9_serial_jax is the W=1 card)"
echo "         prediction band 4-8 s/step."
echo "READ THE RECEIPTS: migrate_pooled_workers per step (0 serial / W pooled)"
echo "and migrate_pool.eject_jax_calls_per_step = [32] on pooled jax legs."
echo "CARDS WRITTEN BY THIS JOB:"
ls -1 runs/v2/m6_w2_identity_*migpool*.json runs/v2/m6_phase_time_c14_*.json \
      runs/v2/m6_c13_insert_census_cgh64.json 2>/dev/null | sed 's/^/  /'
exit $rc_total
