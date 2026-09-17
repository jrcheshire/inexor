#!/bin/bash
#SBATCH -J d7b-hero-diag
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:45:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d7b-hero-diag-%j.log
#SBATCH --signal=B:USR1@900
# D7b: where steps 2+ lose their wall at 4096^3. gb 1002247 ran K=3: step 1 856 s, steps
# 2-3 1201 / 1057 s. The coarse solve grew 82 -> 110 -> 138 s with no recompiles; the tile
# loop doubled after step 1 (196 -> 373 s) with 60% of step 3's in the window upload; host
# RSS rose 763 -> ~852 GB in step 1's migrate and stayed; CPU nodes sat at the ~10 GiB
# MemFree floor through steps 2-3's solves. The checkpoint wrote 641 GB in 3757 s.
# Submit from the Vista checkout:
#   sbatch scripts/v2_d7b_hero_diag_vista.sbatch
#
# TIME ASK: see the proposal in the session; the legs below are in the order they run.
#
# WHAT IS NEW AGAINST 1002247 (instruments only; the engine's arithmetic is unchanged):
#   - every step timed (--timed-all), and the coarse solve broken into forward pass1/2,
#     kernel multiply, inverse pass1/2;
#   - per boundary: process RSS by kind (anon / file / shmem / locked / pinned), page
#     faults, /proc/vmstat differences (direct-reclaim stalls and scans, compaction
#     stalls, NUMA misses), per-node file pages / dirty / writeback, and the process's
#     pages per NUMA node from numa_maps (--numa-maps; its seconds are recorded);
#   - the samplers add per-node FilePages / Dirty / Writeback / Mlocked and a vmstat CSV;
#   - after the run: the first 8 slabs of the step-4 state written and timed per part
#     (index / gather / crc32 / write; no manifest, deleted at the end), then gc +
#     malloc_trim(0) with RSS either side.
#
# PRE-REGISTERED QUESTIONS (nothing at 4096^3 is gated; this job measures):
#   1. Is the +90 GB after step 1's migrate locked/pinned, anon that malloc_trim returns,
#      or neither?
#   2. Do steps 2+ show direct-reclaim / compaction stalls that step 1 does not, and in
#      which phases?
#   3. Which part of the coarse solve grows step over step?
#   4. Does step 1's tile window upload already cost what steps 2+ do (step 1 carries
#      the compiles), and does upload time per card follow where its slabs sit by node?
#   5. Checkpoint seconds per slab by part: is the write or the host work the term?

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
TAG=d7b_$SLURM_JOB_ID
GPU_CSV=$CARDS/${TAG}_gpu.csv
MEM_CSV=$CARDS/${TAG}_mem.csv
VMS_CSV=$CARDS/${TAG}_vmstat.csv
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
# epoch, node, has_cpus (1/0), MemTotal kB, MemFree kB, FilePages kB, Dirty kB,
# Writeback kB, Mlocked kB
( while true; do t=$(date +%s.%N)
    for d in /sys/devices/system/node/node[0-9]*; do
      n=${d##*node}; c=0; [ -n "$(tr -d '[:space:]' < $d/cpulist)" ] && c=1
      awk -v t="$t" -v n="$n" -v c="$c" '/MemTotal:/{tot=$4} /MemFree:/{fr=$4}
        /FilePages:/{fp=$4} / Dirty:/{di=$4} / Writeback:/{wb=$4} /Mlocked:/{ml=$4}
        END{print t "," n "," c "," tot "," fr "," fp "," di "," wb "," ml}' $d/meminfo
    done
    sleep 5; done ) > "$MEM_CSV" &
MEM_SAMPLER=$!
# epoch, allocstall (all zones), pgscan_direct, pgsteal_direct, compact_stall, numa_miss,
# pgmajfault -- cumulative counters; a difference is an interval's count
( while true; do t=$(date +%s.%N)
    awk -v t="$t" '/^allocstall/{a+=$2} $1=="pgscan_direct"{sd=$2} $1=="pgsteal_direct"{td=$2}
      $1=="compact_stall"{cs=$2} $1=="numa_miss"{nm=$2} $1=="pgmajfault"{mf=$2}
      END{print t "," a "," sd "," td "," cs "," nm "," mf}' /proc/vmstat
    sleep 5; done ) > "$VMS_CSV" &
VMS_SAMPLER=$!

finish () {
  kill $GPU_SAMPLER $MEM_SAMPLER $VMS_SAMPLER 2>/dev/null
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

# the timed solve and the writer's probe mode, on the GPU backend, before anything runs on it
run_leg pytest-gpu \
  pixi run -e gpu python -m pytest -q -rs tests/test_engine_timed_steps.py \
    "tests/test_icgen.py::test_write_t9_slabs_probe_writes_the_same_leading_slabs_and_no_manifest" \
  || { echo "FATAL: the timing gates failed on the GPU"; exit 1; }

DIAG="--timed-all --numa-maps --trim-probe"

# the smoke: the same driver and instruments at 32^3
run_leg smoke-ics \
  pixi run -e gpu python scripts/v2_m6_realization.py ics --config smoke \
    --workdir "$WORK/smoke" --generator device \
  && run_leg smoke-run \
  pixi run -e gpu python $DRIVER run --preset smoke --workdir "$WORK/smoke" \
    --card $CARDS/${TAG}_smoke.json --stop-at 2 $DIAG --beat 30 \
    --ckpt-probe-slabs 2 --ckpt-probe-dir "$WORK/smoke-ckpt-probe"
rc_smoke=$?
summarize $CARDS/${TAG}_smoke.json
[ $rc_smoke -ne 0 ] && { echo "FATAL: the smoke failed"; exit 1; }

# the cgh64 control: a known phase table, same node, same driver
run_leg cgh64-ics \
  pixi run -e gpu python scripts/v2_m6_realization.py ics --config cgh64 \
    --workdir "$WORK/cgh64" --generator device \
  && run_leg cgh64-run \
  pixi run -e gpu python $DRIVER run --preset cgh64 --workdir "$WORK/cgh64" \
    --card $CARDS/${TAG}_cgh64.json --stop-at 3 $DIAG --beat 60
rc_control=$?
summarize $CARDS/${TAG}_cgh64.json
[ $rc_control -ne 0 ] && { echo "FATAL: the cgh64 control failed"; exit 1; }

# THE 4096^3 LEG
run_leg hero-run \
  pixi run -e gpu python $DRIVER run --preset c-hero --workdir "$HERO" \
    --card $CARDS/${TAG}_hero.json --stop-at 4 $DIAG --beat 60 \
    --ckpt-probe-slabs 8 --ckpt-probe-dir "$WORK/hero-ckpt-probe"
summarize $CARDS/${TAG}_hero.json
# the probe slabs are a partial state with no manifest: not worth keeping
rm -f "$WORK"/hero-ckpt-probe/t9_slab_*.npz && rmdir "$WORK/hero-ckpt-probe" 2>/dev/null

exit $rc_total
