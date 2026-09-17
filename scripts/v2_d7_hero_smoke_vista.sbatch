#!/bin/bash
#SBATCH -J d7-hero-smoke
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d7-hero-smoke-%j.log
#SBATCH --signal=B:USR1@900
# D7: the first 4096^3 steps -- the whole device step (paint, solve, windowed tile loop
# with the census, fused migrate + repack) on four GB200s, K=3 from the ICs gb 998798
# wrote (/scratch/10303/jrc4/inexor_runs/c-hero-r0), through scripts/v2_d7_hero_smoke.py.
# Submit from the Vista checkout:
#   sbatch scripts/v2_d7_hero_smoke_vista.sbatch
#
# TIME ASK: see the proposal in the session; the legs below are in the order they run.
#
# INSTRUMENTED TO READ WHATEVER HAPPENS:
#   - the driver streams a line per engine phase boundary and a heartbeat every 60 s,
#     prints STEP_JSON per step, rewrites its card at every boundary, and writes a
#     failure card (traceback + every card's allocator stats) on any exception;
#   - nvidia-smi (memory, utilization) and every NUMA node's MemTotal/MemFree sampled
#     every 5 s by THIS script, epoch-stamped, independent of the Python process (an OOM
#     kill or a hang cannot silence them). CPU and GPU (HBM) nodes are separate rows:
#     gb 1002020's `MemAvailable` summed both and hid the CPU-side limit it died on;
#   - 15 min before the wall slurm sends USR1; the trap forwards it to the driver,
#     whose faulthandler dumps every thread's stack, and dumps again 60 s later;
#   - `summarize` lines the samplers up per phase after each run leg, finished or not;
#   - sacct for the job at the end.
#
# PRE-REGISTERED:
#   - preflight refusals (incl. the planner's host peak against the CPU nodes), the GPU
#     window gates, the smoke, and the cgh64 control gate the 4096^3 leg.
#   - Nothing at 4096^3 is gated: this job measures. Compared against the planner
#     (093aa29, printed in the preflight card): host 915.0 GB in-step, 892 GB load;
#     per card 177.3 GB in the step incl. the census, 155.9 GB after the loop.
#   - RERUN of gb 1002020, which died ~7 min into step 1's tile loop on a host overrun
#     (RSS 1,091 GB) from whole-window numpy staging and downloads (fixed in cec9cbc).
#   - Step 1 carries the compiles; step 2 is the steady cost; step 3 is the synced
#     breakdown (a breakdown, not a cost).
#   - One checkpoint is written after step 3 (JC: its write time is what decides
#     whether 40 steps fit one 12 h job). It has no phase boundary of its own: the
#     driver's card records it as `after_last_boundary_s`, and the checkpoint stays on
#     scratch as a resumable step-3 state (not deleted by this job).

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_CLIENT_MEM_FRACTION=0.95
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=900
export INEXOR_LOAD_TRACE=1
export JAX_LOG_COMPILES=1
unset JAX_PLATFORMS
unset CUDA_VISIBLE_DEVICES

RUNS=/scratch/10303/jrc4/inexor_runs
HERO=$RUNS/c-hero-r0
WORK=$RUNS/d7-$SLURM_JOB_ID
CARDS=runs/v2
TAG=d7_$SLURM_JOB_ID
GPU_CSV=$CARDS/${TAG}_gpu.csv
MEM_CSV=$CARDS/${TAG}_mem.csv
DRIVER=scripts/v2_d7_hero_smoke.py
mkdir -p "$WORK"

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
free -g | head -2
df -h "$RUNS" | tail -1
env | grep -E '^(XLA_|JAX_|INEXOR_)' | sort

# --- the samplers: outside the driver, epoch-stamped, for the whole job
( while true; do t=$(date +%s.%N)
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
      2>/dev/null | sed "s/^/$t, /"
    sleep 5; done ) > "$GPU_CSV" &
GPU_SAMPLER=$!
# epoch, node, has_cpus (1/0), MemTotal kB, MemFree kB
( while true; do t=$(date +%s.%N)
    for d in /sys/devices/system/node/node[0-9]*; do
      n=${d##*node}; c=0; [ -n "$(tr -d '[:space:]' < $d/cpulist)" ] && c=1
      awk -v t="$t" -v n="$n" -v c="$c" '/MemTotal:/{tot=$4} /MemFree:/{fr=$4}
        END{print t "," n "," c "," tot "," fr}' $d/meminfo
    done
    sleep 5; done ) > "$MEM_CSV" &
MEM_SAMPLER=$!

finish () {
  kill $GPU_SAMPLER $MEM_SAMPLER 2>/dev/null
  echo ""; echo "=== sacct ==="
  sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,ExitCode -P 2>/dev/null
  echo "=== end $(date) rc_total=$rc_total ==="
}
trap finish EXIT

stacks () {
  echo "=== USR1 at $(date): forwarding to the driver for stack dumps ==="
  pkill -USR1 -f "$DRIVER run" 2>/dev/null
  ( sleep 60; pkill -USR1 -f "$DRIVER run" 2>/dev/null ) &
}
trap stacks USR1

# a leg runs in the BACKGROUND and is waited for, so the USR1 trap runs while it does
# (bash defers traps until a foreground command returns)
rc_total=0
run_leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ($(date +%H:%M:%S)) ==="
  "$@" & local pid=$!
  local rc
  while true; do
    wait $pid; rc=$?
    kill -0 $pid 2>/dev/null || break
  done
  echo "--- LEG $name rc=$rc ($(date +%H:%M:%S)) ---"
  [ $rc -ne 0 ] && rc_total=$((rc_total + 1))
  return $rc
}

summarize () {
  local card="$1"
  run_leg "summarize $(basename "$card" .json)" \
    pixi run -e gpu python $DRIVER summarize --card "$card" --gpu-csv "$GPU_CSV" \
      --mem-csv "$MEM_CSV" --out "${card%.json}_summary.json"
}

run_leg preflight \
  pixi run -e gpu python $DRIVER preflight --preset c-hero --workdir "$HERO" \
    --card $CARDS/${TAG}_preflight.json --scratch "$RUNS" \
  || { echo "FATAL: preflight refused"; exit 1; }

# the changed window staging and write-back, on the GPU backend, before anything runs on it
run_leg pytest-gpu \
  pixi run -e gpu python -m pytest -q tests/test_device_window.py tests/test_device_census.py \
    tests/test_engine_timed_steps.py \
  || { echo "FATAL: the window gates failed on the GPU"; exit 1; }

# the smoke: the same driver and instruments at 32^3
run_leg smoke-ics \
  pixi run -e gpu python scripts/v2_m6_realization.py ics --config smoke \
    --workdir "$WORK/smoke" --generator device \
  && run_leg smoke-run \
  pixi run -e gpu python $DRIVER run --preset smoke --workdir "$WORK/smoke" \
    --card $CARDS/${TAG}_smoke.json --stop-at 2 --timed-last --beat 30
rc_smoke=$?
summarize $CARDS/${TAG}_smoke.json
[ $rc_smoke -ne 0 ] && { echo "FATAL: the smoke failed"; exit 1; }

# the cgh64 control: a known phase table, same node, same driver
run_leg cgh64-ics \
  pixi run -e gpu python scripts/v2_m6_realization.py ics --config cgh64 \
    --workdir "$WORK/cgh64" --generator device \
  && run_leg cgh64-run \
  pixi run -e gpu python $DRIVER run --preset cgh64 --workdir "$WORK/cgh64" \
    --card $CARDS/${TAG}_cgh64.json --stop-at 3 --timed-last --beat 60
rc_control=$?
summarize $CARDS/${TAG}_cgh64.json
[ $rc_control -ne 0 ] && { echo "FATAL: the cgh64 control failed"; exit 1; }

# THE 4096^3 LEG
run_leg hero-run \
  pixi run -e gpu python $DRIVER run --preset c-hero --workdir "$HERO" \
    --card $CARDS/${TAG}_hero.json --stop-at 3 --timed-last --beat 60 \
    --checkpoint-dir "$WORK/hero-ckpt" --checkpoint-every 3
summarize $CARDS/${TAG}_hero.json

exit $rc_total
