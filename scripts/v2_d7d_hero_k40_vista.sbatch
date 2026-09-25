#!/bin/bash
#SBATCH -J d7d-hero-k40
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 11:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d7d-hero-k40-%j.log
#SBATCH --signal=B:USR1@900
# D7d: the 4096^3 PRODUCTION realization, K=40, checkpointing every 10 steps.
# gb 1006175 (K=8) ran clean: steady step ~608 s over steps 3-8, migrate peak
# 906-912 GB of 1026 after the index-count fix, and ZERO direct-reclaim events
# in the whole job. Two changes since, both from that readout:
#   - the IC cache drop is GONE (JC). It cost the load ~270 s (1957 s here
#     against 1688 s without it in 1003657) and did NOT fix step 2's staging,
#     which was the only reason it existed. Bound host memory, no drop, is the
#     configuration 1003657 already ran to completion.
#   - the checkpoint writer no longer scans the 5.5 GB arena index once per
#     slab, gathers through an O(rows) index array, or copies the payload to
#     hash it, and its write overlaps the next slab's gather. Measured 11.1x at
#     512^3 on the laptop; projected ~15 min for the full 256 slabs against 51
#     min projected / 63 min measured before.
#
# The few-slab checkpoint probe is REMOVED: four real full-size writes now
# report their own parts on the timed step's receipt, which is a better
# measurement than an 8-slab probe multiplied by 32 (that read 3076 s against
# the 3757 s job 1002247's full write actually took).
#
# TIME: measured phases from 1006175 x loop counts --
#   legs 203 s; load 1688 s (1003657, no drop); kernel+lead drift 244 s;
#   step 1 659 s; step 2 1024 s; steps 3-40 38 x 608 s = 23,104 s;
#   4 checkpoints ~900 s each = 3600 s.  CENTRAL 8.5 h.
#   Risks, each measured not guessed: the paint/membership creep across steps
#   3-8 extrapolates to +0.9 h if linear rather than settling; and the four
#   checkpoints LEVERAGE the writer 4x -- if Stage 1 does not transport to Grace
#   and they land at the old measured 3757 s, that is +3.2 h. Worst plausible is
#   then 12.6 h, which no wall under this partition's 12 h MaxWall covers.
#   That is not padded away, it is MEASURED EARLY: the cgh64 leg above writes a
#   real 1.2 GB checkpoint onto the same NFS at ~second 250 of the job, so an
#   unimproved writer shows up in four minutes rather than 2.5 h. If it does,
#   kill and resubmit with --checkpoint-every 20. If the wall is hit anyway the
#   step-30 or step-40 generation is a resume point -- untested at 4096^3.
#   REQUESTED 11:00 against gb's 12 h MaxWall.  ~9 SU.
#
# Submit from the Vista checkout:
#   sbatch scripts/v2_d7d_hero_k40_vista.sbatch
#
# PRE-REGISTERED QUESTIONS (nothing at 4096^3 is gated; this job measures):
#   1. What does a full-size checkpoint cost now, in parts? Four of them, on the
#      receipt. This is the number that decides whether the writer is worth
#      optimizing further (concurrent writes) or is now write-bandwidth bound.
#   2. Is the step steady at ~608 s over 40, or does the creep continue? Coarse
#      paint rose 68.0 -> 72.5 s and membership 59.5 -> 65.2 s over steps 3-8,
#      which 6 steps cannot separate from settling.
#   3. Does step 2's staging transient (581 s against ~157 steady) recur without
#      the cache drop, and what does the load cost without it?
#   4. Does the host peak hold? The coarse solve is the binding phase at 936.5 GB
#      of 1026, with 23 GB free across both sockets at its trough.
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

# THE 4096^3 LEG
# THE 4096^3 LEG. Two checkpoint generations alternate in $WORK/hero-ckpt, so
# this needs ~1.3 TB beside the 641 GB of ICs. The last write (step 40) is the
# deliverable: the evolved state, in the same t9-slabs-2 schema the ICs use.
run_leg hero-run \
  $BIND pixi run -e gpu python $DRIVER run --preset c-hero --workdir "$HERO" \
    --card $CARDS/${TAG}_hero.json --stop-at 40 $DIAG $MEMBIND --beat 60 \
    --checkpoint-dir "$WORK/hero-ckpt" --checkpoint-every 10
summarize $CARDS/${TAG}_hero.json
df -h "$RUNS" | tail -1
du -sh "$WORK"/hero-ckpt/* 2>/dev/null

exit $rc_total
