#!/bin/bash
#SBATCH -J inexor-yblocks
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:30:00
#SBATCH -o inexor-yblocks-%j.log
#SBATCH --signal=B:USR1@600
# The y-block cut on one gb node (4 B200): per preset, device ICs once, then the same steps
# at every y-block count, each from the ICs into its own checkpoint dir. Every count's
# checkpoints must equal the 1-block run's byte for byte, manifests compared without their
# window shape (gate_any_cut); a mismatch reruns 1 and that count under
# --xla_gpu_deterministic_ops=true, to tell nondeterminism from a dependence on the cut. The
# run cards (runs/v2/<TAG>_<preset>_y<n>.json, per-phase card and host peaks, --timed-all)
# and samplers price the planner's per-unit terms.
#
# Requires INEXOR_RUNS and INEXOR_SRC (or submission from the checkout). Optional: PRESETS
# ("cgh64 c-1024"), K (120), STOP (10), EVERY (5), Y_MAX (8; counts are 1, 2, 4, ... up to the
# preset's tile rows), QUIET_S (600).
#   INEXOR_RUNS=$SCRATCH/inexor_runs sbatch -A <account> scripts/run/yblocks_measure_vista.sbatch
#
# REHEARSAL=1 runs the same legs on a laptop CPU at cdev8-tile32 (steps 0 -> 6 of 40, 4
# emulated cards); PLANT_MISMATCH=1 flips one byte of the 2-block arm's checkpoint, so the gate
# must fail and the deterministic rerun must run:
#   REHEARSAL=1 INEXOR_RUNS=runs/rehearsal pixi run --frozen bash \
#     scripts/run/yblocks_measure_vista.sbatch
#
# Wall: cgh64 ~3-5 min per arm, 3 arms; c-1024 ~1 min of ICs, then ~10-15 s/step x 10 steps
# plus load and compile (~5 min) per arm, 4 arms; guards and planner ~5 min. ~1 h central,
# 2.5 h requested (unmeasured compile at each new shape).

export PATH="$HOME/.pixi/bin:$PATH"
cd "${INEXOR_SRC:-${SLURM_SUBMIT_DIR:-$PWD}}" && [ -f src/inexor/engine.py ] || {
  echo "FATAL: not an inexor checkout (set INEXOR_SRC)"; exit 1; }
SRC=$(pwd)
RUNS=${INEXOR_RUNS:?set INEXOR_RUNS to the run root, e.g. \$SCRATCH/inexor_runs}
mkdir -p "$RUNS" && RUNS=$(cd "$RUNS" && pwd)
export PYTHONUNBUFFERED=1

if [ "${REHEARSAL:-0}" = 1 ]; then
  SLURM_JOB_ID=${SLURM_JOB_ID:-rehearsal-$$}
  PRESETS=${PRESETS:-cdev8-tile32} K=${K:-40} STOP=${STOP:-6} EVERY=${EVERY:-3}
  LAYOUT=(--slack 0.20 --alloc-margin 0.10 --arena-frac 0.20)
  # the host generator: at a = 0.1 cdev8-tile32 outruns the device generator's 1-slab window
  ICS_ARGS=(--tile-workers 1 --a-init 0.08)
  PY=python
  export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
  export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=4"
  STUBS=$RUNS/stubs-$SLURM_JOB_ID
  mkdir -p "$STUBS"
  printf '#!/bin/sh\necho "0, 1, 0"\n' > "$STUBS/nvidia-smi"
  # shellcheck disable=SC2016  # the stub's own $1 / $@
  printf '#!/bin/sh\nwhile [ $# -gt 0 ]; do case "$1" in --*) shift ;; *) break ;; esac; done\nexec "$@"\n' \
    > "$STUBS/numactl"
  chmod +x "$STUBS/nvidia-smi" "$STUBS/numactl"
  export PATH="$STUBS:$PATH"
  MEMBIND_CHECK=() MACHINE=(--allow-cpu)
else
  [ "$SLURM_JOB_PARTITION" = gb ] || { echo "FATAL: gb only (4 B200 per node)"; exit 1; }
  command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
  export XLA_CLIENT_MEM_FRACTION=0.95
  export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
  export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=900
  export INEXOR_LOAD_TRACE=1
  export JAX_LOG_COMPILES=1
  unset JAX_PLATFORMS CUDA_VISIBLE_DEVICES XLA_FLAGS
  PY=$SRC/.pixi/envs/gpu/bin/python3.14  # the versioned launcher: a scratch purge can take `python`
  [ -x "$PY" ] || { echo "FATAL: no $PY (install the gpu env first)"; exit 1; }
  export CONDA_PREFIX=$SRC/.pixi/envs/gpu
  LAYOUT=(--slack 0.10 --alloc-margin 0.10 --arena-frac 0.01)
  ICS_ARGS=(--generator device)
  # shellcheck disable=SC2054  # "0,1" is one NUMA node list
  MEMBIND_CHECK=(--membind-nodes 0,1) MACHINE=(--host-gb 1026 --device-gb 199)
fi
PRESETS=${PRESETS:-cgh64 c-1024} K=${K:-120} STOP=${STOP:-10} EVERY=${EVERY:-5}
Y_MAX=${Y_MAX:-8} QUIET_S=${QUIET_S:-600}
# shellcheck disable=SC2054
BIND=(numactl --membind=0,1)  # host allocations on the CPU sockets only
CPU=(env JAX_PLATFORMS=cpu)
DRIVER=$SRC/scripts/run/device_run.py

. "$SRC/scripts/run/job_lib.sh"
WORK=$RUNS/yblocks-$SLURM_JOB_ID
# shellcheck disable=SC2034  # read by job_lib.sh
LEG_DIR=$WORK/legs
CARDS=$SRC/runs/v2
TAG=yblocks_$SLURM_JOB_ID
mkdir -p "$WORK" "$CARDS"

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
echo "=== presets $PRESETS, K=$K, steps 0 -> $STOP every $EVERY, layout ${LAYOUT[*]} ==="
if [ "${REHEARSAL:-0}" != 1 ]; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null
  free -g | head -2; numactl --hardware | head -4
fi
df -h "$WORK" | tail -1
env | grep -E '^(XLA_|JAX_|INEXOR_)' | sort

GPU_CSV=$CARDS/${TAG}_gpu.csv MEM_CSV=$CARDS/${TAG}_mem.csv
( while true; do t=$(date +%s.%N)
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
      2>/dev/null | sed "s/^/$t, /"
    sleep 5; done ) > "$GPU_CSV" < /dev/null &
GPU_SAMPLER=$!
( while true; do t=$(date +%s.%N)
    for d in /sys/devices/system/node/node[0-9]*; do
      [ -d "$d" ] || continue
      n=${d##*node}; c=0; [ -n "$(tr -d '[:space:]' < "$d/cpulist")" ] && c=1
      awk -v t="$t" -v n="$n" -v c="$c" '/MemTotal:/{tot=$4} /MemFree:/{fr=$4}
        /FilePages:/{fp=$4} END{print t "," n "," c "," tot "," fr "," fp}' "$d/meminfo"
    done
    sleep 5; done ) > "$MEM_CSV" < /dev/null &
MEM_SAMPLER=$!

rc_total=0
finish () {
  kill "$GPU_SAMPLER" "$MEM_SAMPLER" 2>/dev/null
  echo ""; du -sh "$WORK"/* 2>/dev/null
  echo "=== sacct ==="
  sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,ExitCode -P 2>/dev/null
  echo "=== end $(date) rc_total=$rc_total ==="
}
trap finish EXIT
stacks () {
  echo "=== USR1 at $(date +%H:%M:%S): to the driver for stack dumps ==="
  pkill -USR1 -f "$DRIVER run" 2>/dev/null
}
trap stacks USR1
stop_job () {
  echo "=== stopping: $1; every output is kept under $WORK ==="
  exit $((rc_total > 0 ? rc_total : 1))
}

# arm PRESET ICS CKPT Y CARD [EXTRA_XLA]: steps 0 -> STOP at Y y-blocks into CKPT
arm () {
  local preset=$1 ics=$2 ckpt=$3 y=$4 card=$5
  xla_env "${6:-}"
  "${XLA_ENV[@]}" "${BIND[@]}" "$PY" "$DRIVER" run --preset "$preset" --workdir "$ics" \
    --cards 4 --y-blocks "$y" --k-steps "$K" --stop-at "$STOP" --timed-all --beat 60 \
    "${MEMBIND_CHECK[@]}" "${LAYOUT[@]}" --checkpoint-dir "$ckpt" --checkpoint-every "$EVERY" \
    --card "$card"
}

for preset in $PRESETS; do
  rows=$("${CPU[@]}" "$PY" -c "from inexor.plan import PRESETS as P; g = P['$preset']; print(g['n_fine'] // g['tile'])") \
    || stop_job "no preset $preset"
  YS=() y=1
  while [ $y -le "$rows" ] && [ $y -le "$Y_MAX" ]; do YS+=("$y"); y=$((y * 2)); done
  D=$WORK/$preset
  echo ""; echo "=== $preset: $rows tile rows, y-blocks ${YS[*]}, in $D ==="
  for y in "${YS[@]}"; do
    run_leg --quiet-limit 300 "plan-$preset-y$y" \
      "${CPU[@]}" "$PY" -m inexor.plan --preset "$preset" --backend device --n-gpus 4 \
      --host-gb 1026 --device-gb 199 --y-blocks "$y" "${LAYOUT[@]}"
  done
  run_leg --quiet-limit 1800 "ics-$preset" \
    "${BIND[@]}" "$PY" "$SRC/scripts/run/realization.py" ics --config "$preset" \
    --workdir "$D/ics" "${ICS_ARGS[@]}" \
    || stop_job "the $preset ICs failed"
  run_leg --quiet-limit 600 "preflight-$preset" "${BIND[@]}" "$PY" "$DRIVER" preflight \
    --preset "$preset" --workdir "$D/ics" --cards 4 --y-blocks "${YS[${#YS[@]} - 1]}" \
    --card "$CARDS/${TAG}_${preset}_preflight.json" --scratch "$WORK" "${MACHINE[@]}" \
    "${MEMBIND_CHECK[@]}" "${LAYOUT[@]}" \
    || stop_job "preflight refused $preset"
  for y in "${YS[@]}"; do
    card=$CARDS/${TAG}_${preset}_y$y.json
    run_leg --quiet-limit "$QUIET_S" \
      "run-$preset-y$y" arm "$preset" "$D/ics" "$D/y$y/ckpt" "$y" "$card" \
      || stop_job "the $preset run at $y y-blocks failed"
    "${CPU[@]}" "$PY" "$DRIVER" summarize --card "$card" --gpu-csv "$GPU_CSV" \
      --mem-csv "$MEM_CSV" --out "${card%.json}_summary.json" >/dev/null 2>&1 \
      && echo "  summary ${card%.json}_summary.json"
  done
  if [ "${REHEARSAL:-0}" = 1 ] && [ "${PLANT_MISMATCH:-0}" = 1 ] && [ -d "$D/y2/ckpt" ]; then
    f=$(cd "$D/y2/ckpt" && find . -type f ! -name manifest.json | sort | head -1)
    echo "=== REHEARSAL: flipping one byte of y2/ckpt/$f ==="
    "$PY" - "$D/y2/ckpt/$f" <<'EOF'
import sys
with open(sys.argv[1], "r+b") as fh:
    fh.seek(-1, 2)
    b = fh.read(1)
    fh.seek(-1, 2)
    fh.write(bytes([b[0] ^ 1]))
EOF
  fi
  others=()
  for y in "${YS[@]:1}"; do others+=("$D/y$y/ckpt"); done
  if gate_any_cut "$preset" "$D/y1/ckpt" "${others[@]}"; then
    continue
  fi
  rc_total=$((rc_total + 1))
  # which counts differ, then 1 and each of them again under deterministic ops
  for y in "${YS[@]:1}"; do
    same_ckpt_any_cut "$D/y1/ckpt" "$D/y$y/ckpt" >/dev/null && continue
    for z in 1 "$y"; do
      [ -d "$D/det-y$z/ckpt" ] && continue
      run_leg --quiet-limit "$QUIET_S" \
        "det-$preset-y$z" arm "$preset" "$D/ics" "$D/det-y$z/ckpt" "$z" \
        "$CARDS/${TAG}_${preset}_det-y$z.json" --xla_gpu_deterministic_ops=true
    done
    gate_any_cut "det-$preset-y$y" "$D/det-y1/ckpt" "$D/det-y$y/ckpt"
  done
  stop_job "the $preset checkpoints differ across y-block counts (the det- gates above: PASS =
  the runs differ but agree under deterministic ops, FAIL = they differ there too)"
done
exit $rc_total
