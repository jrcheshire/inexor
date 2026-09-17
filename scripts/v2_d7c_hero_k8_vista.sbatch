#!/bin/bash
#SBATCH -J d7c-hero-k8
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:45:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d7c-hero-k8-%j.log
#SBATCH --signal=B:USR1@900
# D7c: the coarse solve's folded kernel and the IC cache drop, over EIGHT steps at
# 4096^3. gb 1003657 (K=4, memory bound to the CPU sockets) ran steps of 686 / 938 / 792
# / 610 s. Two things came out of it, and this job measures both changes at once because
# they touch different phases:
#   - the coarse solve spent 85 s of its 123 s in a host multiply over the whole
#     half-grid, which exists only because the inverse overwrites its spectrum. That
#     multiply now rides the inverse's own axis-0 pass (`ooc_fft.ArrayKernel`,
#     `EngineConfig.coarse_fold_kernel`, bitwise the old arm on the CPU backend);
#   - the tile loop's window staging was 924 s in step 2 and 55 s in step 4, and steps
#     2-3 are where the kernel was still draining 91 GB of IC page cache. The loader now
#     drops each slab's cache as it reads it (--drop-ic-cache).
# EIGHT steps, because whether step 4's 610 s is the steady cost is exactly the open
# question: four steps cannot tell a settling transient from a cycle.
#
# Submit from the Vista checkout:
#   sbatch scripts/v2_d7c_hero_k8_vista.sbatch
#
# TIME ASK: see the proposal in the session; the legs below are in the order they run.
#
# PRE-REGISTERED QUESTIONS (nothing at 4096^3 is gated; this job measures):
#   1. Is the folded kernel bitwise the host multiply ON A GB200 (the pytest leg below
#      gates it; the CPU backend already says yes), and what is the solve's cost now?
#   2. With the IC cache dropped, does step 2's window staging match step 4's -- i.e.
#      was cache reclaim the cause, or is staging erratic for another reason?
#   3. Over eight steps, is there a steady step cost, and what does K=40 project to
#      against the 12 h wall?
#   4. What does dropping the cache do to the load itself (1687.8 s in 1003657)?
#   5. The checkpoint probe again, on a state eight steps in: same 11.7 s/slab?
#
# HOST MEMORY IS BOUND TO THE CPU SOCKETS (`numactl --membind=0,1`), as in 1003657: the
# cards' HBM appears as CPU-less NUMA nodes and without the binding the kernel put 197 GB
# of the process's memory on two of them (gb 1003511, device OOM at the first step).

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
TAG=d7c_$SLURM_JOB_ID
GPU_CSV=$CARDS/${TAG}_gpu.csv
MEM_CSV=$CARDS/${TAG}_mem.csv
VMS_CSV=$CARDS/${TAG}_vmstat.csv
DRIVER=scripts/v2_d7_hero_smoke.py
# host allocations on the CPU sockets only; device allocations are unaffected
BIND="numactl --membind=0,1"
MEMBIND="--membind-nodes 0,1"
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

numactl --hardware | head -4
run_leg preflight \
  $BIND pixi run -e gpu python $DRIVER preflight --preset c-hero --workdir "$HERO" \
    --card $CARDS/${TAG}_preflight.json --scratch "$RUNS" $MEMBIND \
  || { echo "FATAL: preflight refused"; exit 1; }

# the timed solve and the writer's probe mode, on the GPU backend, before anything runs on it
run_leg pytest-gpu \
  pixi run -e gpu python -m pytest -q -rs tests/test_engine_timed_steps.py tests/test_d7_driver.py \
    tests/test_coarse_solve_folded_kernel.py \
    "tests/test_icgen.py::test_write_t9_slabs_probe_writes_the_same_leading_slabs_and_no_manifest" \
  || { echo "FATAL: the timing gates failed on the GPU"; exit 1; }

DIAG="--timed-all --numa-maps --trim-probe --drop-ic-cache"

# the smoke: the same driver and instruments at 32^3
run_leg smoke-ics \
  $BIND pixi run -e gpu python scripts/v2_m6_realization.py ics --config smoke \
    --workdir "$WORK/smoke" --generator device \
  && run_leg smoke-run \
  $BIND pixi run -e gpu python $DRIVER run --preset smoke --workdir "$WORK/smoke" \
    --card $CARDS/${TAG}_smoke.json --stop-at 2 $DIAG $MEMBIND --beat 30 \
    --ckpt-probe-slabs 2 --ckpt-probe-dir "$WORK/smoke-ckpt-probe"
rc_smoke=$?
summarize $CARDS/${TAG}_smoke.json
[ $rc_smoke -ne 0 ] && { echo "FATAL: the smoke failed"; exit 1; }

# the cgh64 control: a known phase table, same node, same driver
run_leg cgh64-ics \
  $BIND pixi run -e gpu python scripts/v2_m6_realization.py ics --config cgh64 \
    --workdir "$WORK/cgh64" --generator device \
  && run_leg cgh64-run \
  $BIND pixi run -e gpu python $DRIVER run --preset cgh64 --workdir "$WORK/cgh64" \
    --card $CARDS/${TAG}_cgh64.json --stop-at 3 $DIAG $MEMBIND --beat 60
rc_control=$?
summarize $CARDS/${TAG}_cgh64.json
[ $rc_control -ne 0 ] && { echo "FATAL: the cgh64 control failed"; exit 1; }

# THE 4096^3 LEG
run_leg hero-run \
  $BIND pixi run -e gpu python $DRIVER run --preset c-hero --workdir "$HERO" \
    --card $CARDS/${TAG}_hero.json --stop-at 8 $DIAG $MEMBIND --beat 60 \
    --ckpt-probe-slabs 8 --ckpt-probe-dir "$WORK/hero-ckpt-probe"
summarize $CARDS/${TAG}_hero.json
# the probe slabs are a partial state with no manifest: not worth keeping
rm -f "$WORK"/hero-ckpt-probe/t9_slab_*.npz && rmdir "$WORK/hero-ckpt-probe" 2>/dev/null

exit $rc_total
