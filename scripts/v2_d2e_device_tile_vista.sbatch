#!/bin/bash
#SBATCH -J d2e-tile
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d2e-tile-%j.log
# TIME ASK: ~5.5 min estimated, 15 min wall, one gb node, ~0.25 SU at most.
#
#   phase                     count  each (source)                          total
#   smoke, 4 arms at 32^3       1    ~45 s (993600's smoke leg)            ~45 s
#   xback-cpu, cdev             1    50.2 s whole arm, 13.7 GB peak RSS,
#                                    MEASURED on the M4 laptop 2026-09-12
#                                    (Grace's CPU rate is not measured)    ~50 s
#   xback-gpu, cdev             1    10 s + 7.6 s + ~20 s compile (972737
#                                    P=576: 21 s) + 8 tiles < 1 s          ~45 s
#   short, P=576                1    10 s + 29.5 s build (993139) + 21 s
#                                    compile + 4 x 6 s host decode (0.25
#                                    us/row, B200 cdev card)               ~1.5 min
#   tile, P=576                 1    10 s + 29.5 s + ~25 s compile
#                                    + 4 tiles x ~1 s                      ~1.2 min
#
# TWO QUESTIONS, one card:
#   1. The device tile's GPU-vs-CPU floor, before any cross-backend tolerance is
#      set: `one_tile` alone on identical inputs, then the jitted tile's short
#      force, long force and kicked velocity, in eps x rms; layout must be exact.
#      cdev (P=320, 8 tiles, arena residents), ratified dtypes (f64 fine, f32
#      coarse). The CPU arm also re-runs check A (eager device tile == host
#      tile_task) on this machine.
#   2. What one tile costs a card at the 4096^3 tile shape: cgh64 with tile 512 /
#      buffer 32 gives P=576 and ~23.9M rows per tile. `one_tile` alone (like the
#      gb probe's 94.2 ms/tile) and the jitted tile, each in its own process,
#      device time and peak.
#
# NO BAR is applied to question 1; the card is the floor a bar is set from.
# (993754 ran that: 5:15, ~0.1 SU. Record sec. 18.)
#
# ATTRIBUTION RE-RUN, submitted with
#   sbatch --export=ALL,D2E_ARMS=short+tile+tile-staged,D2E_SUFFIX=_attr_gb
# (`+` separates arms: --export itself splits on commas)
# ONE QUESTION: where do the jitted tile's 566 ms go at P=576 (993754)? Every
# phase ended by a device sync, in one job on one node:
#   short        one_tile alone: input upload / compute / readback of the
#                cap x 3 f64 forces.
#   tile         the path 993754 timed (state arrays uploaded every call): an
#                untimed call (the reference wall), then a timed call split
#                into plan / stage (host) / h2d_tile / h2d_state / compute /
#                d2h / result (host).
#   tile-staged  the same with the state placed on the device once.
# TIME ASK: ~4.5 min estimated, 10 min wall, one gb node, ~0.2 SU at most.
#   phase                 each (source: 993754's arms)                    total
#   smoke, 5 arms at 32^3  ~55 s (993754 smoke, one arm more)            ~55 s
#   short                  10 s start + 28.7 s build + 20.4 s compile
#                          + 4 tiles x ~6 s host decode of inputs        ~83 s
#   tile                   10 s + 28.9 s + 22.8 s compile + 4 tiles x 2
#                          calls x 0.57 s                                ~67 s
#   tile-staged            the same, less the per-call upload, plus one
#                          1.58 GB placement                             ~67 s
# (993817 ran that: 4:17, ~0.1 SU. Record sec. 19.)
#
# STEP-LOOP RE-RUN, submitted with
#   sbatch --export=ALL,D2E_ARMS=tile-staged+loop,D2E_SUFFIX=_loop_gb
# ONE QUESTION: with results written into the device state (tile_loop_device,
# record sec. 20), what does a tile cost at P=576, and what does the once-per-
# step copy-back cost? tile-staged repeats 993817's 389 ms baseline on the same
# node; loop runs whole steps (8 tiles): warm (compile), untimed (the
# reference wall), synced timed (phases accumulated per tile, copy-back once).
# TIME ASK: ~3.5 min estimated, 10 min wall, one gb node, ~0.2 SU at most.
#   phase                 each (source)                                   total
#   smoke, 6 arms at 32^3  ~65 s (993817's smoke, one arm more)          ~65 s
#   tile-staged            10 s start + 28.9 s build + 22.5 s compile
#                          + 4 tiles x 2 calls x 0.39 s (993817)         ~65 s
#   loop                   10 s + 28.9 s build + ~23 s compile in the
#                          warm step + 3 steps x 8 tiles x <= 0.39 s
#                          + 3 copy-backs of ~0.8 GB                     ~75 s
#
# ENV copied verbatim from the D2d gb sbatch (v2_d2d_dead_rows_vista.sbatch).

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

WORK="$SCRATCH/inexor_runs/d2e_xback_$SLURM_JOB_ID"
# which arms run and the card's suffix; 993754 ran the defaults
# (xback,short,tile -> _gb). Override at submit: sbatch --export=ALL,D2E_ARMS=a+b,D2E_SUFFIX=...
D2E_ARMS="${D2E_ARMS:-xback,short,tile}"
D2E_ARMS="${D2E_ARMS//+/,}"
D2E_SUFFIX="${D2E_SUFFIX:-_gb}"
echo "arms: $D2E_ARMS  card suffix: $D2E_SUFFIX"
rc_total=0
run_leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc ---"
  [ $rc -ne 0 ] && rc_total=$((rc_total + 1))
  return $rc
}

run_leg smoke \
  pixi run -e gpu python scripts/v2_d2e_device_tile.py --smoke \
    --arms xback,short,tile,tile-staged,loop --workdir "$WORK/smoke" --out-suffix "${D2E_SUFFIX}smoke"
if [ $rc_total -ne 0 ]; then
  echo "FATAL: smoke failed, not spending the node on the long arms"
  exit 1
fi

run_leg d2e \
  pixi run -e gpu python scripts/v2_d2e_device_tile.py --workdir "$WORK/cdev" \
    --arms "$D2E_ARMS" --out-suffix "$D2E_SUFFIX"

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
