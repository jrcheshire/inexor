#!/bin/bash
#SBATCH -J d2f-bundle
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d2f-bundle-%j.log
# D2f bundle on a gb node: three four-card questions in one job (JC: a job under
# the 15 min minimum wastes the charge).
# Submit from the Vista checkout:
#   sbatch scripts/v2_d2f_bundle_vista.sbatch
#
# TIME ASK: ~18 min estimated, 45 min wall, one gb node, ~0.3 SU (<= 0.75).
#
#   phase                          each (source)                               total
#   smoke, tile split legs, 32^3   9 processes, ~10 s start each (993849)      ~1.5 min
#   smoke, paint + scale, 32^3     5 processes (994608 smoke ~45 s)            ~1 min
#   tile split, threads (before)   start 10 s, cgh64 build 29 s, one-card warm
#                                  23 s + 30 steps 19 s (993849); four warms
#                                  concurrently in one process (UNMEASURED,
#                                  23-60 s) + 30 steps ~6 s                    ~2.5 min
#   tile split, processes          ref group 10 + 29 + 23 + 19 s; four group
#                                  10 + 29 + 23 + 6 s (993849)                 ~2.5 min
#   tile split, threads (after)    as before                                   ~2.5 min
#   paint split, 1024^3 state      build 244-251 s (993294); 2 chunk sizes x
#                                  8 calls: 16 x 67M-row chunks ~2.4 s/call,
#                                  256 x 4.2M-row chunks ~2.7 s/call (secs.
#                                  13-15 rates, spread dead rows)              ~5.5 min
#   4096^3 shards + forward pair   994608's arm ~45 s + 4 more forward calls
#                                  at ~10 s                                    ~1.5 min
#
# QUESTIONS, one card per arm:
#   split-threads / split  Does the tile loop still split ~3.36x across four
#                GB200s when the cards are THREADS of one process (the engine's
#                form) rather than processes? Coarse meshes on each card in both
#                legs, unpinned in both, same node. Threads run before AND after
#                the process legs, so drift over the job shows as a difference
#                between the two threads readings, not inside the ratio.
#   cards-paint  How does the coarse paint split across four cards
#                (`coarse_delta_cards`, one thread per card) at real per-chunk row
#                counts: 67M-row chunks (a 4096^3 quarter-slab chunk's rows) and
#                4.2M-row chunks, one card against four, warmed, 1-4-4-1-1-4?
#   cards-scale  The forward transform from the cards against the host-slab
#                path, now warmed and ABBA (994608 ran each leg once, cold, card
#                leg first).
#
# PRE-REGISTERED:
#   - Every smoke arm passes on the GPUs or the long legs do not run.
#   - split-threads must use four distinct devices (rc 3 otherwise). Threads
#     against processes differing by more than ~1.3x in summed tiles/s means the
#     GIL is the variable and the one-process design is a constraint (the D5
#     criterion). Efficiencies reported both summed and by the slowest card.
#   - cards-paint: the density hash-equal at one and four cards for both chunk
#     sizes (rc 3 otherwise); the split is reported, not predicted.
#   - cards-scale: mass exact, sampled planes bitwise numpy's decode, spectra
#     bitwise, as in 994608; forward medians read only if those hold.
#
# ENV copied verbatim from the D2f gb sbatch (v2_d2f_cards_vista.sbatch).

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
echo "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=$XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB"
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

run_leg smoke-tile \
  pixi run -e gpu python scripts/v2_d2e_device_tile.py --smoke \
    --arms split,split-threads --split-modes unpinned --split-coarse-device \
    --out-suffix _d2f_bundle_smoke
run_leg smoke-paint \
  pixi run -e gpu python scripts/v2_d2d_device_paint.py --smoke --skip-xback \
    --paint-n-part 1024 --scale --out-suffix _d2f_bundle_smoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: smoke failed, not spending the node on the long legs"
  exit 1
fi

run_leg tile-split \
  pixi run -e gpu python scripts/v2_d2e_device_tile.py \
    --arms split,split-threads --split-modes unpinned --split-coarse-device --steps 30 \
    --out-suffix _d2f_threads_gb

run_leg paint-and-forward \
  pixi run -e gpu python scripts/v2_d2d_device_paint.py \
    --skip-xback --chunks none --reps 3 --paint-n-part 1024 --paint-chunks 16384+1024 \
    --scale --out-suffix _d2f_bundle_gb

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
