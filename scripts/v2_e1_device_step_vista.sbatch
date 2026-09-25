#!/bin/bash
#SBATCH -J e1-device-step
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:50:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o e1-device-step-%j.log
# E1, the device step executor, on a gb node. ONE bundled job:
#   1. smoke of all five arms at 32^3 on the GPU (the apparatus; stops the job);
#   2. five cgh64 engine arms, K=2 + lead drift, repack every step, each its own
#      process: engine-host, engine-device (compiled), slack0-host,
#      engine-eager, slack0-eager -- the two eager/host pairs gated bitwise,
#      compiled vs eager read at step 1's tile_loop_end, the phase table;
#   3. the GPU pytest set for the executor (LAST: unpriced, and a failure there
#      is a reading, not a reason to lose the engine legs).
# Submit from the Vista checkout:
#   sbatch scripts/v2_e1_device_step_vista.sbatch
#
# TIME ASK: ~34 min estimated, 50 min wall, one gb node (~0.8 SU at the wall,
# ~0.6 at the estimate). Three legs are UNPRICED and run last in their leg:
# the two eager arms and the pytest leg.
#
#   phase                          each (source)                                total
#   smoke, 5 arms at 32^3          laptop MEASURED 45 s (CPU XLA); GPU
#                                  compiles per arm                            ~2 min
#   engine-host, cgh64, K=2        build 31 s (995602) + run 336 s (995813)    ~6.5 min
#   engine-device, cgh64, K=2      build 31 s; tile loop 64 x 17.8 ms/tile
#                                  (5y) + one 1.4 GB state upload per step;
#                                  tile compile ~23 s (994675); paint <1 s,
#                                  solve 1.6, migrate 2.2, repack 0.6 per step;
#                                  lead 6.2 s (995813)                          ~2 min
#   slack0-host, cgh64, K=2        as engine-host, more spills                 ~7 min
#   engine-eager, cgh64, K=2       UNPRICED: the eager tile op by op, 64 tiles
#                                  a step with host result assembly; ~1 s/tile
#                                  assumed                                      ~4 min
#   slack0-eager, cgh64, K=2       UNPRICED, as engine-eager                   ~4 min
#   snapshot writes + comparisons  5 x ~1 GB `w` at 0.96 GB/s, 3 compares      ~1.5 min
#   pytest, 4 files, GPU backend   UNPRICED: 995813's 5 files took 5:24; the
#                                  tile suite compiles per dtype pair           ~7 min
#
# QUESTIONS:
#   - eager lane vs host lane at cgh64 on a GB200: every state hash, every host
#     stat, and step 1's codes and scales at tile_loop_end equal? (rc 3 if not);
#   - the same at brick slack 0, with spills > 0 (the arena fold-in on the GPU);
#   - compiled vs eager at step 1's tile_loop_end: codes within one? how many
#     move, and how far do scales move?
#   - the cgh64 phase table with the paint and the tile loop on the card, beside
#     the host arm's in the same job; device peak per arm.
#
# PRE-REGISTERED:
#   - a smoke failure stops the job before the long legs;
#   - an eager/host mismatch is attributed (tile_loop_end snapshot first) before
#     anything else in the card is read;
#   - walls and peaks are reported, not gated.
#
# Snapshots land in runs/v2/e1_snap/ (~5 GB at cgh64, untracked).
#
# ENV copied verbatim from v2_r2_device_engine_vista.sbatch.

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

rc_total=0
run_leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc ---"
  [ $rc -ne 0 ] && rc_total=$((rc_total + 1))
  return $rc
}

run_leg jax-devices \
  pixi run -e gpu python -c "import jax; print('jax devices:', jax.devices())"
run_leg smoke \
  pixi run -e gpu python scripts/v2_e1_device_step.py --smoke --out-suffix _gbsmoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: the smoke failed, not spending the node on the long legs"
  exit 1
fi

run_leg engine \
  pixi run -e gpu python scripts/v2_e1_device_step.py --engine-n 512 --steps 2 --out-suffix _gb
run_leg pytest-gpu \
  pixi run -e gpu python -m pytest -q tests/test_engine_device_step.py \
    tests/test_engine_device_backend.py tests/test_device_paint_cards.py tests/test_device_tile.py

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
