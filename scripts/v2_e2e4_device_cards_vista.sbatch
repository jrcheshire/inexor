#!/bin/bash
#SBATCH -J e2e4-device-cards
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:50:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o e2e4-device-cards-%j.log
# E2-E4 + R3 on a gb node, ONE bundled job (JC, 2026-09-14: consolidate):
#   1. smoke of every arm at 32^3 on the GPUs (the apparatus; stops the job);
#   2. eight cgh64 arms, each its own process, K=2 + lead drift where it steps:
#      solve-cards (E4), window-off / window-on (E2), r3-w1 / r3-w4 (the
#      migrate + repack split), cards (the whole step on four cards), and the
#      slack-0 pair (compiled lane on four cards vs eager lane on one);
#   3. the GPU pytest set for the new files (LAST: unpriced).
# Submit from the Vista checkout:
#   sbatch scripts/v2_e2e4_device_cards_vista.sbatch
#
# TIME ASK: ~32 min estimated, 50 min wall, one gb node (~0.8 SU at the wall,
# ~0.5 at the estimate). UNPRICED: every four-card arm and the pytest leg.
#
#   phase                          each (source)                                total
#   smoke, 8 arms at 32^3          laptop MEASURED 63 s (CPU XLA); 996685's GPU
#                                  smoke ran 5 arms in ~4.5 min (compiles)      ~6 min
#   state builds, 8 arms, cgh64    31 s each (996685)                           ~4 min
#   solve-cards                    card paint 2.6 s + host solve 2.2 s (996685)
#                                  x3 each path + compiles                      ~1 min
#   window-off                     run 37.7 s (996685 engine-device)            ~0.7 min
#   window-on                      as window-off + 4 plane windows of 10 slabs  ~1 min
#   r3-w1                          2 x device migrate 2.2-3.1 s + 2 x repack
#                                  ~1 s (996685) + compiles                     ~0.5 min
#   r3-w4                          UNPRICED: cross-card hand-offs               ~1 min
#   cards                          UNPRICED: four cards compiling the tile
#                                  concurrently ~30 s each (994675)             ~1.5 min
#   slack0-eager                   run 108 s (996685)                           ~2 min
#   slack0-cards                   UNPRICED                                     ~1.5 min
#   snapshots + comparisons        2 x ~1 GB                                    ~0.5 min
#   pytest, 6 files, GPU backend   UNPRICED; 996685's 4 files took 17:40 with
#                                  the device-tile suite, which is not here     ~12 min
#
# PRE-REGISTERED:
#   - a smoke failure stops the job before the long legs;
#   - bitwise gates (rc 3 if not): solve-cards in-arm; window-on == window-off;
#     r3-w4 == r3-w1; cards == window-on; slack0-cards == slack0-eager, including
#     step 1's codes and scales at tile_loop_end;
#   - a mismatch is attributed before anything else in the card is read;
#   - walls, peaks and splits are reported, not gated.
#
# Snapshots land in runs/v2/e2e4_snap/ (~2 GB at cgh64, untracked).
#
# ENV copied verbatim from v2_e1_device_step_vista.sbatch.

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
  pixi run -e gpu python scripts/v2_e2e4_device_cards.py --smoke --out-suffix _gbsmoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: the smoke failed, not spending the node on the long legs"
  exit 1
fi

run_leg arms \
  pixi run -e gpu python scripts/v2_e2e4_device_cards.py --engine-n 512 --steps 2 --out-suffix _gb
run_leg pytest-gpu \
  pixi run -e gpu python -m pytest -q tests/test_ooc_fft_card_shards.py \
    tests/test_device_window.py tests/test_engine_device_cards.py \
    tests/test_migrate_device_cards.py tests/test_repack_device_cards.py \
    tests/test_engine_device_step.py

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
