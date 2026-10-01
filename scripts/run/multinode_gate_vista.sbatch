#!/bin/bash
#SBATCH -J inexor-mn-gate
#SBATCH -p gh
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH -o inexor-mn-gate-%j.log
#SBATCH --signal=B:USR1@900
# The multi-node byte gate: one realization run three ways on the same two nodes, one GPU
# per node, and the checkpoints compared file by file.
#   ref0, ref1  the single-node run (`--comm serial`), on node 0 and node 1 at the same time,
#               each into its own checkpoint dir: ref0 == ref1 shows the lane repeats
#               bitwise across cards, without which a 1-vs-2 difference could not be read;
#   mpi         the same run on 2 ranks, one per node (`python -m mpi4py ... --comm mpi`).
# Segment 1 runs steps 0 -> SPLIT and compares the step-SPLIT checkpoints. On a mismatch the
# three legs rerun 0 -> SPLIT with --xla_gpu_deterministic_ops=true, are compared, and the
# job stops with every checkpoint kept. Otherwise segment 2 resumes all three to STOP and
# compares the last two generations. The job exits with the number of failed gates and legs.
#
# Before the runs, on node 0 and the CPU backend: the kernel-block and rank-gate tests
# (fatal) and the real-MPI tests against the module's MPI (recorded, not fatal).
#
# Requires INEXOR_RUNS (outputs under $INEXOR_RUNS/mn-gate-<preset>-<job>) and INEXOR_SRC or
# submission from the checkout. Optional: PRESET (c-1024), K (120), SPLIT (20), STOP (K),
# EVERY (SPLIT), SLACK (0.10), ALLOC_MARGIN (0.10), ARENA_FRAC (0.01), DET_TIMEOUT (5400 s).
#   INEXOR_RUNS=$SCRATCH/inexor_runs sbatch -A <account> scripts/run/multinode_gate_vista.sbatch
#
# REHEARSAL=1 runs the same legs on a laptop CPU at cdev8-tile32 (first 6 of 40 steps, split
# at 3), under the env's MPICH:
#   REHEARSAL=1 INEXOR_RUNS=runs/rehearsal scripts/run/mpi_lane.sh --run \
#     bash scripts/run/multinode_gate_vista.sbatch
# (PLANT_MISMATCH=1 flips one byte of ref1's step-SPLIT checkpoint, to see the gate fire.)
#
# Wall (c-1024), from the one-GH200 run 1029876: ICs 105 s, 34 s/step median (late steps
# 38-46 s), checkpoint 15 s. Refs ~13 min + ~62 min; mpi unmeasured, at most the refs' time.
# ~2.4 h central with the guards; 6 h requested. USR1 900 s before the wall reaches every
# rank through the local mpiexec and dumps every thread's stack.

export PATH="$HOME/.pixi/bin:$PATH"
cd "${INEXOR_SRC:-${SLURM_SUBMIT_DIR:-$PWD}}" && [ -f src/inexor/engine.py ] || {
  echo "FATAL: not an inexor checkout (set INEXOR_SRC)"; exit 1; }
SRC=$(pwd)
RUNS=${INEXOR_RUNS:?set INEXOR_RUNS to the run root, e.g. \$SCRATCH/inexor_runs}
mkdir -p "$RUNS" && RUNS=$(cd "$RUNS" && pwd)
export PYTHONUNBUFFERED=1

if [ "${REHEARSAL:-0}" = 1 ]; then
  PRESET=${PRESET:-cdev8-tile32} K=${K:-40} SPLIT=${SPLIT:-3} STOP=${STOP:-6}
  SLACK=${SLACK:-0.20} ARENA=${ARENA_FRAC:-0.20}
  SLURM_JOB_ID=${SLURM_JOB_ID:-rehearsal-$$}
  PY=python
  export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
  export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"
  export XLA_CACHE_ROOT=$RUNS/xla-cache-$SLURM_JOB_ID
  ICS_ARGS=(--tile-workers 1 --a-init 0.08)
  BIND0=() MEMBIND_RANK=() MEMBIND_CHECK=() MACHINE=(--allow-cpu)
else
  [ "$SLURM_JOB_PARTITION" = gh ] || { echo "FATAL: gh only (one GH200 per node)"; exit 1; }
  command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
  export XLA_CLIENT_MEM_FRACTION=0.95
  # JAX's default host-memory cap is 64 GB, below what host streaming needs here
  export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=160
  export INEXOR_LOAD_TRACE=1
  export JAX_LOG_COMPILES=1
  unset JAX_PLATFORMS CUDA_VISIBLE_DEVICES XLA_FLAGS
  ENV=$SRC/.pixi/envs/gpu
  PY=$ENV/bin/python3.14  # the versioned launcher: a scratch purge can take `python`
  [ -x "$PY" ] || { echo "FATAL: no $PY (install the gpu env first)"; exit 1; }
  export CONDA_PREFIX=$ENV
  ICS_ARGS=(--generator device)
  BIND0=(numactl --membind=0) MEMBIND_RANK=(--membind 0) MEMBIND_CHECK=(--membind-nodes 0)
  MACHINE=(--host-gb 116 --device-gb 96)
fi
PRESET=${PRESET:-c-1024} K=${K:-120} SPLIT=${SPLIT:-20} STOP=${STOP:-$K} EVERY=${EVERY:-$SPLIT}
SLACK=${SLACK:-0.10} ALLOC_MARGIN=${ALLOC_MARGIN:-0.10} ARENA=${ARENA:-${ARENA_FRAC:-0.01}}
DET_TIMEOUT=${DET_TIMEOUT:-5400}
. "$SRC/scripts/run/mpi_env_vista.sh"
# after the module: the env's own MPICH (in its bin) must not shadow the module's launcher
[ -n "${ENV:-}" ] && export PATH="$PATH:$ENV/bin"
ON0=(env "JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR//@RANK@/0}")
CPU=(env JAX_PLATFORMS=cpu)
MPIPATH=()
LEG_TIMEOUT=()  # the deterministic-ops legs set it
[ -n "$MPI4PY_DIR" ] && MPIPATH=(env "PYTHONPATH=$MPI4PY_DIR${PYTHONPATH:+:$PYTHONPATH}")

WORK=$RUNS/mn-gate-$PRESET-$SLURM_JOB_ID
ICS=$WORK/ics
CARDS=$SRC/runs/v2
TAG=mngate_${PRESET}_$SLURM_JOB_ID
DRIVER=$SRC/scripts/run/device_run.py
RANK_EXEC=$SRC/scripts/run/rank_exec.sh
LAYOUT=(--slack "$SLACK" --alloc-margin "$ALLOC_MARGIN" --arena-frac "$ARENA")
mkdir -p "$WORK" "$CARDS"

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
echo "=== $PRESET, K=$K, split $SPLIT, stop $STOP, every $EVERY, layout ${LAYOUT[*]} ==="
echo "=== nodes ${SLURM_JOB_NODELIST:-local}, launch '$LAUNCH', work $WORK ==="
for t in mpiexec mpicc ptxas "$PY"; do echo "  $t -> $(command -v "$t" || echo none)"; done
if [ "${REHEARSAL:-0}" != 1 ]; then
  echo "  modules $(module -t list 2>&1 | tr '\n' ' ')"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null
  free -g | head -2; numactl --hardware | head -6
fi
df -h "$RUNS" | tail -1
env | grep -E '^(XLA_|JAX_|INEXOR_|MV2_|MVP_|HYDRA_)' | sort

rc_total=0
finish () {
  echo ""; du -sh "$WORK"/* 2>/dev/null
  echo "=== sacct ==="
  sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,ExitCode -P 2>/dev/null
  echo "=== end $(date) rc_total=$rc_total ==="
}
trap finish EXIT
trap forward_usr1 USR1

# a leg runs in the BACKGROUND and is waited for, so the USR1 trap runs while it does
# (bash defers traps until a foreground command returns)
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

# same_ckpt A B: the same files under two checkpoint dirs, byte for byte
same_ckpt () {
  local a=$1 b=$2 bad=0 f
  if ! diff <(cd "$a" && find . -type f | sort) <(cd "$b" && find . -type f | sort) >/dev/null; then
    echo "  GATE FAIL: $a and $b hold different files"
    diff <(cd "$a" && find . -type f | sort) <(cd "$b" && find . -type f | sort) | head -20
    return 1
  fi
  while IFS= read -r f; do
    cmp -s "$a/$f" "$b/$f" || { echo "  differs: $f"; bad=$((bad + 1)); }
  done < <(cd "$a" && find . -type f | sort)
  echo "  $(cd "$a" && find . -type f | wc -l | tr -d ' ') files, $bad differing: $a vs $b"
  [ $bad -eq 0 ]
}

# gate NAME DIR...: every DIR's files equal the first's
gate () {
  local name=$1 first=$2 d ok=0; shift 2
  echo ""; echo "=== GATE $name ==="
  for d in "$@"; do same_ckpt "$first" "$d" || ok=1; done
  if [ $ok -eq 0 ]; then echo "GATE $name PASS"; else echo "GATE $name FAIL"; fi
  return $ok
}

# run_args ARM FROM STOP: the driver's run arguments for one arm and segment
run_args () {
  local arm=$1 from=$2 stop=$3
  echo --preset "$PRESET" --workdir "$ICS" --cards 1 --k-steps "$K" --stop-at "$stop" \
    --expect-step "$from" --timed-all --beat 60 "${MEMBIND_CHECK[@]}" "${LAYOUT[@]}" \
    --checkpoint-every "$EVERY"
}

# refs ARM FROM STOP [XLA_FLAGS]: the one-node run on every node at once
refs () {
  local arm=$1 from=$2 stop=$3 extra=${4:-}
  # shellcheck disable=SC2046
  "${LEG_TIMEOUT[@]}" $LAUNCH "$RANK_EXEC" "${MEMBIND_RANK[@]}" \
    --samples "$CARDS/${TAG}_${arm}_s${from}_r@RANK@" -- env XLA_FLAGS="${XLA_FLAGS:-} $extra" \
    "$PY" "$DRIVER" run $(run_args "$arm" "$from" "$stop") \
    --card "$CARDS/${TAG}_${arm}@RANK@_s${from}.json" \
    --checkpoint-dir "$WORK/$arm@RANK@/ckpt" --comm serial
}

# ranks ARM FROM STOP [XLA_FLAGS]: the same run across the nodes, one rank each
ranks () {
  local arm=$1 from=$2 stop=$3 extra=${4:-}
  # shellcheck disable=SC2046
  "${LEG_TIMEOUT[@]}" $LAUNCH "$RANK_EXEC" "${MEMBIND_RANK[@]}" \
    --samples "$CARDS/${TAG}_${arm}_s${from}_r@RANK@" -- env XLA_FLAGS="${XLA_FLAGS:-} $extra" \
    "${MPIPATH[@]}" \
    "$PY" -m mpi4py "$DRIVER" run $(run_args "$arm" "$from" "$stop") \
    --card "$CARDS/${TAG}_${arm}_s${from}.json" \
    --checkpoint-dir "$WORK/$arm/ckpt" --comm mpi --comm-timeout 1800
}

# summarize every rank's card of one leg against its node's samples
summarize () {
  local arm=$1 from=$2 card r
  for r in 0 1; do
    case $arm in
      ref*|det-ref*) card=$CARDS/${TAG}_${arm}${r}_s${from}.json ;;
      *) card=$CARDS/${TAG}_${arm}_s${from}.rank${r}.json ;;
    esac
    [ -f "$card" ] || continue
    "${CPU[@]}" "$PY" "$DRIVER" summarize --card "$card" \
      --gpu-csv "$CARDS/${TAG}_${arm}_s${from}_r${r}_gpu.csv" \
      --mem-csv "$CARDS/${TAG}_${arm}_s${from}_r${r}_mem.csv" \
      --out "${card%.json}_summary.json" >/dev/null 2>&1 \
      && echo "  summary ${card%.json}_summary.json"
  done
}

# --- guards and inputs (node 0)
run_leg build-mpi4py build_mpi4py || { echo "FATAL: mpi4py did not build"; exit 1; }
for n in 1 2; do
  run_leg "plan-$n-node" "${CPU[@]}" "$PY" -m inexor.plan --preset "$PRESET" --backend device \
    --n-gpus 1 --n-nodes $n --host-gb 116 --device-gb 96 --slack "$SLACK" \
    --alloc-margin "$ALLOC_MARGIN" --arena-frac "$ARENA"
done
run_leg guard-kernel-block \
  "${CPU[@]}" XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    "$PY" -m pytest -q -rs -p no:cacheprovider tests/test_coarse_kernel_cards.py \
  || { echo "FATAL: the kernel block build is not bitwise on this machine"; exit 1; }
run_leg guard-rank-gates \
  "${CPU[@]}" OMP_NUM_THREADS=1 \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=8" \
    "$PY" -m pytest -q -rs -p no:cacheprovider -n auto tests/test_ranks_coarse.py \
    tests/test_ranks_tile.py tests/test_ranks_migrate.py tests/test_ranks_run.py \
  || { echo "FATAL: the rank gates failed on this machine"; exit 1; }
# not fatal: the tests launch their own mpiexec, whose placement quirks are not the code's
run_leg guard-mpi-lane \
  "${CPU[@]}" OMP_NUM_THREADS=1 "${MPIPATH[@]}" \
    "$PY" -m pytest -q -rs -p no:cacheprovider --basetemp "$WORK/pytest-mpi" \
    tests/test_comm_mpi.py tests/test_driver_mpi.py

run_leg ics \
  "${BIND0[@]}" "${ON0[@]}" "$PY" scripts/run/realization.py ics --config "$PRESET" \
    --workdir "$ICS" "${ICS_ARGS[@]}" \
  || { echo "FATAL: the IC generation failed"; exit 1; }
for n in 1 2; do
  run_leg "preflight-$n-node" \
    "${BIND0[@]}" "${ON0[@]}" "$PY" "$DRIVER" preflight --preset "$PRESET" --workdir "$ICS" --cards 1 \
      --card "$CARDS/${TAG}_preflight_n$n.json" --scratch "$RUNS" --n-nodes $n \
      "${MACHINE[@]}" "${MEMBIND_CHECK[@]}" "${LAYOUT[@]}" \
    || { echo "FATAL: preflight refused"; exit 1; }
done

# --- segment 1: steps 0 -> SPLIT
run_leg "ref 0->$SPLIT" refs ref 0 "$SPLIT"; summarize ref 0
run_leg "mpi 0->$SPLIT" ranks mpi 0 "$SPLIT"; summarize mpi 0
if [ "${REHEARSAL:-0}" = 1 ] && [ "${PLANT_MISMATCH:-0}" = 1 ]; then
  f=$(cd "$WORK/ref1/ckpt" && find . -type f ! -name manifest.json | sort | head -1)
  echo "=== REHEARSAL: flipping one byte of ref1/ckpt/$f ==="
  "$PY" - "$WORK/ref1/ckpt/$f" <<'EOF'
import sys
with open(sys.argv[1], "r+b") as fh:
    fh.seek(-1, 2)
    b = fh.read(1)
    fh.seek(-1, 2)
    fh.write(bytes([b[0] ^ 1]))
EOF
fi
if ! gate "step-$SPLIT" "$WORK/ref0/ckpt" "$WORK/ref1/ckpt" "$WORK/mpi/ckpt"; then
  rc_total=$((rc_total + 1))
  echo "=== the step-$SPLIT checkpoints differ: 0 -> $SPLIT again with deterministic ops ==="
  DET=--xla_gpu_deterministic_ops=true
  LEG_TIMEOUT=(timeout "$DET_TIMEOUT")
  run_leg "det-ref 0->$SPLIT" refs det-ref 0 "$SPLIT" "$DET"; summarize det-ref 0
  run_leg "det-mpi 0->$SPLIT" ranks det-mpi 0 "$SPLIT" "$DET"; summarize det-mpi 0
  gate "step-$SPLIT-deterministic" "$WORK/det-ref0/ckpt" "$WORK/det-ref1/ckpt" \
    "$WORK/det-mpi/ckpt" || rc_total=$((rc_total + 1))
  echo "=== stopping after segment 1: every checkpoint is kept under $WORK ==="
  exit $rc_total
fi

# --- segment 2: SPLIT -> STOP, resumed from the step-SPLIT checkpoints
if [ "$STOP" -gt "$SPLIT" ]; then
  run_leg "ref $SPLIT->$STOP" refs ref "$SPLIT" "$STOP"; summarize ref "$SPLIT"
  run_leg "mpi $SPLIT->$STOP" ranks mpi "$SPLIT" "$STOP"; summarize mpi "$SPLIT"
  gate "step-$STOP" "$WORK/ref0/ckpt" "$WORK/ref1/ckpt" "$WORK/mpi/ckpt" \
    || rc_total=$((rc_total + 1))
fi
exit $rc_total
