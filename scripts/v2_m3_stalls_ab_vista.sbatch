#!/bin/bash
#SBATCH -J m3-stalls-ab
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m3-stalls-ab-%j.log
# M3 on a gb node: the device migrate pass with its two host stalls removed,
# against the frozen pre-M3 pass (scripts/v2_d3_migrate_legacy.py), ON A GB200.
#
# What changed (src/inexor/device/migrate.py): the slab window is uploaded as a
# view of the host state instead of a zero-padded host copy, with arena residents
# as their own array; the eject reads n_keep, its realized reach and its
# emigrants' count per destination slab in ONE read, and the inserts' census is
# host arithmetic on those counts instead of eager full-size device ops plus a
# blocking read per source slab. Every value reaching the kernels is unchanged,
# so no value test can see either change; tests/test_migrate_device_stalls.py
# watches the host allocation, the read count and the census, each with a
# control that must fail. The transfer-guard control in that file runs ONLY on a
# GPU (on the CPU backend no read is a transfer).
# Submit from the Vista checkout:
#   sbatch scripts/v2_m3_stalls_ab_vista.sbatch
#
# TIME ASK: ~50 min estimated, 1 h 30 wall, one gb node (~1.08 SU/node-hour:
# ~0.9 SU at the estimate, ~1.6 SU at the wall).
#
#   phase                          each (source)                               total
#   jax devices + GPU pytest,      999779 MEASURED 7:47 for 7 files; +1 file
#   8 files                        of 7 small tests                            ~9 min
#   probe smoke, small shapes      999779 measured 12 s                        ~1 min
#   identity: xback 256,           999779 measured 2:34 for xback + device-real
#   device-xback 256               at 512; device-xback is a 256 pass          ~3 min
#   A/B at 512 (cgh64 shape),      999779 device-real 512: build 31 s + 3 x
#   legacy then current            (28 s numpy + <=6 s device) = ~2.3 min      ~5 min
#   A/B at 1024 (4x rows per       UNPRICED, carried n^3 from 512: build ~4 min,
#   slab, 64 slabs), legacy        3 numpy steps ~11 min, device ~1 min, each
#   then current                   arm ~16 min                                 ~32 min
#
# QUESTIONS, in the order the job answers them:
#   1. Do the migrate gates, including the new stall tests and the transfer-guard
#      control, pass on the GPU backend? (gate)
#   2. Is the M3 pass bitwise the numpy migrate on a GB200 with arena residents in
#      the windows (device-xback 256: 30% arena, reach 2, overflow) and at cgh64
#      shape (device-real)? Both arms of both A/B pairs also gate bitwise. (gate)
#   3. What did the stalls cost? Per-phase tables, legacy vs current, same node:
#      legacy "eject: host index + window" against current "eject: host index" +
#      "eject: window"; "insert: census" in both; "eject: reach + scalars" in both
#      (current carries the count program there).
#   4. How do the phases grow from 4.2M to 16.8M rows per slab, at the engine's
#      brick geometry (4096 rows per brick)? The kernel shares at 1024 read on
#      the question record sec. 39 left open.
#
# PRE-REGISTERED:
#   - A pytest failure or a non-bitwise step is a failed gate, read BEFORE any wall.
#     The smoke gates the long legs.
#   - Walls are REPORTED, NOT GATED. Expectation, falsifiable, written before the
#     run: current's "insert: census" is near zero against legacy's, and current's
#     "eject: host index" + "eject: window" is below legacy's combined phase, by a
#     margin that grows from 512 to 1024 (the copy scales with rows per slab).
#     The whole timed step is expected lower, but one timed step per arm is one
#     sample, and legacy runs first.
#   - The 1024 pair is last because it is the unpriced leg.
#
# ENV copied verbatim from v2_m2_insert_ab_vista.sbatch.

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
    tests/test_migrate_device_stalls.py tests/test_slot_state.py \
    tests/test_repack_device.py tests/test_engine_device_backend.py
run_leg smoke \
  pixi run -e gpu python scripts/v2_d3_device_migrate.py \
    --arms xback,device-xback,device-real --xback-n 32 --real-n 64 --nb 8 --steps 2 \
    --reps 1 --out-suffix _m3ab_gbsmoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: a GPU gate or the smoke failed, not spending the node on the long legs"
  exit 1
fi

# identity with arena residents in the device windows
run_leg identity \
  pixi run -e gpu python scripts/v2_d3_device_migrate.py --arms xback,device-xback \
    --xback-n 256 --steps 2 --out-suffix _m3ab_gb

# the A/B at cgh64 shape: separate processes, legacy FIRST so the live pass is not
# the arm that pays for a cold compilation cache
for impl in legacy current; do
  run_leg "ab-512-$impl" \
    pixi run -e gpu python scripts/v2_d3_device_migrate.py --arms device-real \
      --real-n 512 --steps 2 --migrate-impl $impl --out-suffix _m3ab_512_${impl}_gb
done

# the unpriced leg last: 16.8M rows per slab at the engine's brick geometry
for impl in legacy current; do
  run_leg "ab-1024-$impl" \
    pixi run -e gpu python scripts/v2_d3_device_migrate.py --arms device-real \
      --real-n 1024 --steps 2 --migrate-impl $impl --out-suffix _m3ab_1024_${impl}_gb
done

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
