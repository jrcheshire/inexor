#!/bin/bash
#SBATCH -J d8-gh-1card
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o d8-gh-1card-%j.log
#SBATCH --signal=B:USR1@900
# Realizations on ONE GH200: device-generated LCDM ICs, then one run per step count in KS
# from those ICs on the single-card device lane (each checkpointed once, at its end), and
# each run's P(k) card. PRESET defaults to c-1024 (1024^3 in 512 Mpc/h, the same resolution
# as every other preset). A run that fails gets no card and the next step count still runs.
#
# Requires INEXOR_RUNS (run root; outputs under $INEXOR_RUNS/d8-gh-<preset>-<job>/k<K>) and
# INEXOR_SRC or submission from the inexor checkout. Optional: PRESET, KS (120; step counts,
# space-separated), SLACK, ALLOC_MARGIN, ARENA_FRAC, PK_TABLE (a tabulated linear P(k) from
# scripts/run/camb_linear_pk.py instead of EH98), REBUILD_ENV=1 (reinstall the gpu env from the
# lock first; it is also reinstalled when it fails to import, as after a scratch purge). The
# account is passed with `sbatch -A`:
#   INEXOR_RUNS=$SCRATCH/inexor_runs sbatch -A <account> scripts/run/gh_single_card_vista.sbatch
#   sbatch -A <account> -t <wall> --export=ALL,PRESET=c-gh,SLACK=0.05,ALLOC_MARGIN=0.05 ...
# The step-count ladder at 512^3 in 256 Mpc/h (one IC set, five schedules):
#   INEXOR_RUNS=$SCRATCH/inexor_runs PRESET=cgh64 KS="40 60 80 120 160" \
#     sbatch -A <account> scripts/run/gh_single_card_vista.sbatch
#
# A GH200 host is ~116 GB of LPDDR with a hard cliff; MemAvailable also counts the card's
# HBM, so the preflight compares against the CPU NUMA node alone and the process is bound
# to it (--membind 0).
#
# Wall (cgh64 ladder), estimated: ~8 s/step on one GH200 (gh 1029876's 512^3 run), 460
# steps ~1 h, plus five cards (~1-2 min each) and the ICs: ~1.5-2 h central, 4 h requested.
# Wall (c-1024), estimated: ~35-45 s/step (4096^3 on four GB200s scaled by volume, one
# card for four, and GH200 vs GB200 tile force), 120 steps ~70-90 min; ICs ~5 min, pytest
# ~2, checkpoints ~1, card ~5. ~1.5-2 h central; 4 h requested since small-N fixed costs
# are unmeasured. USR1 900 s before the wall triggers driver stack dumps.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
# Run from the inexor checkout: INEXOR_SRC, or the directory sbatch was called from.
cd "${INEXOR_SRC:-$SLURM_SUBMIT_DIR}" && [ -f src/inexor/engine.py ] || {
  echo "FATAL: not an inexor checkout: ${INEXOR_SRC:-$SLURM_SUBMIT_DIR} (set INEXOR_SRC)"; exit 1; }
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_CLIENT_MEM_FRACTION=0.95
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
# JAX's default host-memory cap is 64 GB, below what host streaming needs here
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=160
export INEXOR_LOAD_TRACE=1
export JAX_LOG_COMPILES=1
unset JAX_PLATFORMS
unset CUDA_VISIBLE_DEVICES
. scripts/run/gpu_env.sh
ensure_gpu_env "$PWD" || { echo "FATAL: no working gpu env"; exit 1; }

PRESET=${PRESET:-c-1024}
SLACK=${SLACK:-0.10}
ALLOC_MARGIN=${ALLOC_MARGIN:-0.10}
ARENA=${ARENA_FRAC:-0.01}
KS=${KS:-120}
for K in $KS; do
  [[ "$K" =~ ^[1-9][0-9]*$ ]] || { echo "FATAL: KS holds '$K', not a step count"; exit 1; }
done
RUNS=${INEXOR_RUNS:?set INEXOR_RUNS to the run root, e.g. $SCRATCH/inexor_runs}
WORK=$RUNS/d8-gh-$PRESET-$SLURM_JOB_ID
ICS=$WORK/ics
CARDS=runs/v2
mkdir -p "$CARDS"
TAG=d8gh_${PRESET}_$SLURM_JOB_ID
GPU_CSV=$CARDS/${TAG}_gpu.csv
MEM_CSV=$CARDS/${TAG}_mem.csv
DRIVER=scripts/run/device_run.py
BIND="numactl --membind=0"
MEMBIND="--membind-nodes 0"
CPU="env JAX_PLATFORMS=cpu"
LAYOUT="--slack $SLACK --alloc-margin $ALLOC_MARGIN --arena-frac $ARENA"
mkdir -p "$WORK"

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
echo "=== $PRESET, K in $KS, slack $SLACK, alloc_margin $ALLOC_MARGIN, arena_frac $ARENA ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
free -g | head -2
numactl --hardware | head -6
df -h "$RUNS" | tail -1
env | grep -E '^(XLA_|JAX_|INEXOR_)' | sort

( while true; do t=$(date +%s.%N)
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
      2>/dev/null | sed "s/^/$t, /"
    sleep 5; done ) > "$GPU_CSV" &
GPU_SAMPLER=$!
( while true; do t=$(date +%s.%N)
    for d in /sys/devices/system/node/node[0-9]*; do
      n=${d##*node}; c=0; [ -n "$(tr -d '[:space:]' < $d/cpulist)" ] && c=1
      awk -v t="$t" -v n="$n" -v c="$c" '/MemTotal:/{tot=$4} /MemFree:/{fr=$4}
        /FilePages:/{fp=$4} / Dirty:/{di=$4} / Writeback:/{wb=$4} /Mlocked:/{ml=$4}
        END{print t "," n "," c "," tot "," fr "," fp "," di "," wb "," ml}' $d/meminfo
    done
    sleep 5; done ) > "$MEM_CSV" &
MEM_SAMPLER=$!

rc_total=0
finish () {
  kill $GPU_SAMPLER $MEM_SAMPLER 2>/dev/null
  echo ""; du -sh "$WORK"/* 2>/dev/null
  echo "=== sacct ==="
  sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,ExitCode -P 2>/dev/null
  echo "=== end $(date) rc_total=$rc_total ==="
}
trap finish EXIT
stacks () {
  echo "=== USR1 at $(date): forwarding to the driver for stack dumps ==="
  pkill -USR1 -f "$DRIVER run" 2>/dev/null
}
trap stacks USR1

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

run_leg plan \
  pixi run -e gpu python -m inexor.plan --preset "$PRESET" --backend device --n-gpus 1 \
    --host-gb 116 --device-gb 96 --slack "$SLACK" --alloc-margin "$ALLOC_MARGIN" \
    --arena-frac "$ARENA"

run_leg pytest-gpu \
  pixi run -e gpu python -m pytest -q -rs \
    "tests/test_engine.py::test_a_resumed_segment_writes_first_to_the_generation_it_did_not_load" \
  || { echo "FATAL: the driver gates failed on the GPU"; exit 1; }

run_leg ics \
  $BIND pixi run -e gpu python scripts/run/realization.py ics --config "$PRESET" \
    --workdir "$ICS" --generator device ${PK_TABLE:+--pk-table "$PK_TABLE"} \
  || { echo "FATAL: the IC generation failed"; exit 1; }

run_leg preflight \
  $BIND pixi run -e gpu python $DRIVER preflight --preset "$PRESET" --workdir "$ICS" --cards 1 \
    --card $CARDS/${TAG}_preflight.json --scratch "$RUNS" $MEMBIND \
    --slack "$SLACK" --alloc-margin "$ALLOC_MARGIN" --arena-frac "$ARENA" \
  || { echo "FATAL: preflight refused"; exit 1; }

for K in $KS; do
  CKPT=$WORK/k$K/ckpt
  PROD=$WORK/k$K/prod
  mkdir -p "$PROD"
  run_leg "run-k$K" \
    $BIND pixi run -e gpu python $DRIVER run --preset "$PRESET" --workdir "$ICS" --cards 1 \
      --card $CARDS/${TAG}_k${K}_run.json --k-steps $K --stop-at $K --timed-all --numa-maps \
      $MEMBIND --beat 60 --slack "$SLACK" --alloc-margin "$ALLOC_MARGIN" --arena-frac "$ARENA" \
      --checkpoint-dir "$CKPT" --checkpoint-every $K
  rc_run=$?
  run_leg "summarize-k$K" \
    pixi run -e gpu python $DRIVER summarize --card $CARDS/${TAG}_k${K}_run.json \
      --gpu-csv "$GPU_CSV" --mem-csv "$MEM_CSV" --out $CARDS/${TAG}_k${K}_run_summary.json
  [ $rc_run -ne 0 ] && { echo "=== the $K-step run failed: no card ==="; continue; }

  # the product subcommands read <workdir>/ckpt and refuse a checkpoint under the ICs
  ln -sfn "$CKPT" "$PROD/ckpt"
  run_leg "card-k$K" \
    $CPU $BIND pixi run -e gpu python scripts/run/realization.py card --config "$PRESET" \
      --workdir "$PROD" --k-steps $K $LAYOUT --heartbeat 60 --tile-workers 8 --card-pool
  [ -f "$PROD/realization_pk.json" ] && cp "$PROD/realization_pk.json" $CARDS/${TAG}_k${K}_pk.json
done

exit $rc_total
