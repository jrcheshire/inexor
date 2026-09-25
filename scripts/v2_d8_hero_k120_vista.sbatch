#!/bin/bash
#SBATCH -J d8-hero-k120
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 11:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d8-hero-k120-%j.log
#SBATCH --signal=B:USR1@900
# D8: the 4096^3 realization on the 120-step schedule (LCDM weights, D-v2-25), run
# as THREE segments of 40 steps because 120 steps do not fit gb's 12 h MaxWall.
# One script, three submissions, each chained afterok on the one before:
#   sbatch --export=ALL,EXPECT_STEP=0,SEG_STOP=40   ... (depends on the IC job)
#   sbatch --export=ALL,EXPECT_STEP=40,SEG_STOP=80  ...
#   sbatch --export=ALL,EXPECT_STEP=80,SEG_STOP=120 ...
# EXPECT_STEP=0 starts from the ICs and refuses if the checkpoint dir already holds
# a checkpoint; otherwise the driver resumes from the newest checkpoint and refuses
# unless it is at EXPECT_STEP, so a mis-chained segment dies in seconds instead of
# redoing finished steps. Checkpoints every 20 steps: two per segment, so each
# segment ends on its own write and has a mid-segment resume point.
# The segments compose bitwise into the uninterrupted run (engine gate
# test_a_run_split_into_segments_is_bitwise_the_uninterrupted_one; the driver's own
# in tests/test_d7_driver.py). Resume at 4096^3 has not run before this.
#
# Otherwise this is v2_d7d_hero_k40_vista.sbatch (1009133): same env, samplers,
# membind, instruments, pytest set, 32^3 smoke and cgh64 control on every segment,
# so each segment's node gets its own comparable control.
#
# TIME, per segment, from 1009133's measured phases x this segment's loop counts:
#   legs ~300 s; load 1842 s (ICs) / ~1700 s (checkpoint, 1663 measured by the card
#   job); kernel + lead drift ~450 s (segment 1; the kernel alone ~63 s on resume);
#   40 steps x 590-667 s (1009133: steps 4-9 vs 32-39; the arena creep follows the
#   epoch, and segment 3 covers the late epochs) = 23,600-26,700 s;
#   2 checkpoints x 950 s = 1900 s.  CENTRAL ~8.2-9.0 h per segment.
#   REQUESTED 11:00 each against gb's 12 h MaxWall (~9 SU central, max 11).
#
# PRE-REGISTERED QUESTIONS (nothing at 4096^3 is gated; this job measures):
#   1. Does the resume hold at 4096^3: the segment loads the checkpoint, the
#      fingerprint matches on this node, and the first step's time and host peak
#      sit with the steps that preceded it?
#   2. Per-step time on the 120-step schedule against 1009133's 40-step one (smaller
#      steps mean fewer migrants per step; the arena follows the epoch).
#   3. Host peak against 1009133's 951.0 GB of 1026.
#
# HOST MEMORY IS BOUND TO THE CPU SOCKETS (`numactl --membind=0,1`), as in 1009133.

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
HERO=${HERO_IC_DIR:-$RUNS/c-hero-lcdm-r1}
# the realization's home is FIXED across the three segments; each job's own scratch
# (smoke, control) is under it by job id
REAL=${REAL_DIR:-$RUNS/d8-hero-k120}
WORK=$REAL/job-$SLURM_JOB_ID
CKPT=$REAL/hero-ckpt
: "${EXPECT_STEP:?pass EXPECT_STEP (0, 40 or 80) with --export}"
: "${SEG_STOP:?pass SEG_STOP (40, 80 or 120) with --export}"
CARDS=runs/v2
TAG=d8_${SLURM_JOB_ID}_s${EXPECT_STEP}-${SEG_STOP}
GPU_CSV=$CARDS/${TAG}_gpu.csv
MEM_CSV=$CARDS/${TAG}_mem.csv
VMS_CSV=$CARDS/${TAG}_vmstat.csv
DRIVER=scripts/v2_d7_hero_smoke.py
# host allocations on the CPU sockets only; device allocations are unaffected
BIND="numactl --membind=0,1"
MEMBIND="--membind-nodes 0,1"
mkdir -p "$WORK"

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
echo "=== segment: steps $EXPECT_STEP -> $SEG_STOP of 120; ICs $HERO; checkpoints $CKPT ==="
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

# The timed solve and the REWRITTEN checkpoint writer, on the GPU backend,
# before anything runs on it. The writer is the newest code in this job and it
# first fires on the hero state 2.5 h in, so its gates run at second 75.
run_leg pytest-gpu \
  pixi run -e gpu python -m pytest -q -rs tests/test_engine_timed_steps.py tests/test_d7_driver.py \
    tests/test_coarse_solve_folded_kernel.py \
    "tests/test_icgen.py::test_write_t9_slabs_round_trips_an_evolved_state" \
    "tests/test_icgen.py::test_write_t9_slabs_groups_the_arena_once_not_once_per_slab" \
    "tests/test_icgen.py::test_write_t9_slabs_hashes_the_buffers_without_copying_them" \
    "tests/test_icgen.py::test_write_t9_slabs_surfaces_a_failure_from_the_writer_thread" \
    "tests/test_engine.py::test_a_timed_checkpoint_reports_its_own_parts" \
    "tests/test_engine.py::test_a_run_split_into_segments_is_bitwise_the_uninterrupted_one" \
    "tests/test_engine.py::test_a_resumed_segment_writes_first_to_the_generation_it_did_not_load" \
  || { echo "FATAL: the timing gates failed on the GPU"; exit 1; }

DIAG="--timed-all --numa-maps"

# the smoke: the same driver and instruments at 32^3
run_leg smoke-ics \
  $BIND pixi run -e gpu python scripts/v2_m6_realization.py ics --config smoke \
    --workdir "$WORK/smoke" --generator device \
  && run_leg smoke-run \
  $BIND pixi run -e gpu python $DRIVER run --preset smoke --workdir "$WORK/smoke" \
    --card $CARDS/${TAG}_smoke.json --stop-at 2 $DIAG $MEMBIND --beat 30 \
    --checkpoint-dir "$WORK/smoke-ckpt" --checkpoint-every 1
rc_smoke=$?
summarize $CARDS/${TAG}_smoke.json
[ $rc_smoke -ne 0 ] && { echo "FATAL: the smoke failed"; exit 1; }

# The cgh64 control: a known phase table, same node, same driver -- and now one
# REAL checkpoint (134M rows, 1.2 GB, 128 slabs) onto the same NFS scratch the
# hero leg will use, which measures the write bandwidth the 641 GB writes will
# get before the job commits 2.5 h to reaching the first of them.
run_leg cgh64-ics \
  $BIND pixi run -e gpu python scripts/v2_m6_realization.py ics --config cgh64 \
    --workdir "$WORK/cgh64" --generator device \
  && run_leg cgh64-run \
  $BIND pixi run -e gpu python $DRIVER run --preset cgh64 --workdir "$WORK/cgh64" \
    --card $CARDS/${TAG}_cgh64.json --stop-at 3 $DIAG $MEMBIND --beat 60 \
    --checkpoint-dir "$WORK/cgh64-ckpt" --checkpoint-every 3
rc_control=$?
summarize $CARDS/${TAG}_cgh64.json
[ $rc_control -ne 0 ] && { echo "FATAL: the cgh64 control failed"; exit 1; }

# THE 4096^3 SEGMENT. Two checkpoint generations alternate in $CKPT (~1.4 TB beside
# the 641 GB of ICs). The step-120 write is the deliverable: the evolved state, in
# the t9-slabs-2 schema the ICs use, which the card and export jobs read.
for g in 0 1; do
  [ -f "$CKPT/gen$g/manifest.json" ] && python3 -c "import json;p=json.load(open('$CKPT/gen$g/manifest.json'))['provenance'];print('  gen$g: step', p['step'], 'of', p['n_steps'])"
done
run_leg hero-run \
  $BIND pixi run -e gpu python $DRIVER run --preset c-hero --workdir "$HERO" \
    --card $CARDS/${TAG}_hero.json --k-steps 120 --expect-step "$EXPECT_STEP" \
    --stop-at "$SEG_STOP" $DIAG $MEMBIND --beat 60 \
    --checkpoint-dir "$CKPT" --checkpoint-every 20
summarize $CARDS/${TAG}_hero.json
df -h "$RUNS" | tail -1
du -sh "$CKPT"/* 2>/dev/null

exit $rc_total
