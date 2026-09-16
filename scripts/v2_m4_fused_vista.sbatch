#!/bin/bash
#SBATCH -J m4-fused
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m4-fused-%j.log
# M4 on a gb node: the device migrate and repack fused into one visit per slab
# (src/inexor/device/fused.py), sized by the destination census the windowed tile
# loop takes (src/inexor/device/window.py), against the separate passes, ON A GB200.
#
# Laptop gates (four forced host devices): the census equals the host migrate's
# per-brick membership on 1, 2 and 4 cards; the fused pass is bitwise the host
# migrate + repack AND the device pair, incl. read-ahead and cross-card overruns and
# an allocation reaching into the old arena; four mechanisms mutation-tested; the
# engine run fused == separate on 1 and 4 cards. None of that is a GPU reading: the
# census only equals the migrate's destinations if the SAME compiled eject kernel
# reproduces on the card, and cross-card timing differs from CPU threads.
# Submit from the Vista checkout:
#   sbatch scripts/v2_m4_fused_vista.sbatch
#
# TIME ASK: ~70 min estimated, 2 h 00 wall, one gb node (~1.08 SU/node-hour:
# ~1.3 SU at the estimate, ~2.2 SU at the wall).
#
#   phase                          each (source)                               total
#   jax devices + GPU pytest,      1001060: 8 files 7:30; this set adds engine
#   10 files                       runs and cards (996857's 6 engine/card files
#                                  took 10:18), so both classes              ~20 min
#   smokes, 32^3 / 64^3            1001060 smoke 26 s; engine smoke ~1 min     ~2 min
#   engine cgh64 K=2: window-sep,  996857: 8 engine arms + solve in 13 min after
#   window-on, cards-sep, cards    pytest, ~1.6 min each; 4 arms               ~10 min
#   device-fused 1024^3, separate  1001060 device-real 1024: build 4.4 min +
#   then fused                     3 x (numpy 3.9 min + device ~10 s); + host
#                                  repack + census per step, UNMEASURED at 1024
#                                  (C15 cdev rate carried: ~10 s) -> ~17.5 min
#                                  each                                        ~35 min
#
# QUESTIONS, in the order the job answers them:
#   1. Do the census, fused, migrate, repack, window and engine device gates pass on
#      the GPU backend? (gate)
#   2. Is the engine run with the fused pass bitwise the separate passes at cgh64, on
#      one card and on four? (gate: hashes + stats)
#   3. Is the fused migrate + repack bitwise the numpy pair at 16.8M rows per slab,
#      and the separate device pair too? (gate)
#   4. What did fusing save? Migrate + repack wall and per-phase tables, separate vs
#      fused, at cgh64 (engine phases) and 1024^3 (synced step); device peak of each.
#
# PRE-REGISTERED:
#   - A pytest failure, a hash/stats mismatch, or a non-bitwise step is a failed
#     gate, read BEFORE any wall. The smokes gate the long legs.
#   - Walls and peaks are REPORTED, NOT GATED. Expectation, falsifiable, written
#     before the run: at 1024^3 the fused step's migrate + repack is below the
#     separate pair's by at least the migrate's slot-range + occupancy downloads
#     (2.62 s of 1001060's current arm) less the census, which the fused arm does
#     NOT pay here (it is the numpy migrate's, host-side); the engine arms DO pay the
#     census inside `tile_loop`, so there the tile loop is expected to grow.
#   - One sample per arm, separate first.
#
# ENV copied verbatim from v2_m3_stalls_ab_vista.sbatch.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=900
unset JAX_PLATFORMS
unset CUDA_VISIBLE_DEVICES

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
free -g | head -2
env | grep -E '^XLA_' | sort

rc_total=0
run_leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ($(date +%H:%M:%S)) ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc ($(date +%H:%M:%S)) ---"
  [ $rc -ne 0 ] && rc_total=$((rc_total + 1))
  return $rc
}

run_leg devices \
  pixi run -e gpu python -c "import jax; print('jax devices:', jax.devices())"
run_leg pytest-gpu \
  pixi run -e gpu python -m pytest -q tests/test_insert_jax.py tests/test_eject_jax.py \
    tests/test_migrate_device.py tests/test_migrate_device_cards.py \
    tests/test_migrate_device_stalls.py tests/test_device_census.py \
    tests/test_fused_migrate_repack.py tests/test_repack_device.py \
    tests/test_device_window.py tests/test_engine_device_cards.py
run_leg smoke-migrate \
  pixi run -e gpu python scripts/v2_d3_device_migrate.py --arms device-fused --real-n 64 \
    --steps 2 --reps 1 --pass fused --out-suffix _m4_gbsmoke
run_leg smoke-engine \
  pixi run -e gpu python scripts/v2_e2e4_device_cards.py --smoke \
    --arms window-sep,window-on,cards-sep,cards --steps 2 --out-suffix _m4_gbsmoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: a GPU gate or a smoke failed, not spending the node on the long legs"
  exit 1
fi

# the engine at cgh64: separate first, gates on hashes + stats inside the script
run_leg engine-cgh64 \
  pixi run -e gpu python scripts/v2_e2e4_device_cards.py \
    --arms window-sep,window-on,cards-sep,cards --engine-n 512 --steps 2 --out-suffix _m4_gb

# the unpriced leg last: 16.8M rows per slab, separate processes, separate first
for pass in separate fused; do
  run_leg "fused-1024-$pass" \
    pixi run -e gpu python scripts/v2_d3_device_migrate.py --arms device-fused \
      --real-n 1024 --steps 2 --pass $pass --out-suffix _m4_1024_${pass}_gb
done

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
