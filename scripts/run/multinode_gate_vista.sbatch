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
# Before the runs: the run legs' own launch line on every node (a
# JAX start, an MPI start, and a rank that dies at once, which must end the launch), then on
# node 0 and the CPU backend the kernel-block and rank-gate tests (all fatal) and the real-MPI
# tests against the module's MPI (recorded, not fatal).
#
# Every leg has a time cap; a run leg is also killed once its output (the driver's 60 s
# heartbeat) has been silent for QUIET_S. A failed run leg stops the job, and the gate fails
# on a missing or empty checkpoint dir. Each leg's output is also in $WORK/legs/.
#
# Requires INEXOR_RUNS (outputs under $INEXOR_RUNS/mn-gate-<preset>-<job>) and INEXOR_SRC or
# submission from the checkout. Optional: PRESET (c-1024), K (120), SPLIT (20), STOP (K),
# EVERY (SPLIT), SLACK (0.10), ALLOC_MARGIN (0.10), ARENA_FRAC (0.01), DET_TIMEOUT (5400 s, a
# deterministic-ops leg's cap), LEG_FIXED_S (600) + LEG_STEP_S (100) per step (a run leg's
# cap), QUIET_S (600).
#   INEXOR_RUNS=$SCRATCH/inexor_runs sbatch -A <account> scripts/run/multinode_gate_vista.sbatch
#
# REHEARSAL=1 runs the same legs on a laptop CPU at cdev8-tile32 (first 6 of 40 steps, split
# at 3), under the env's MPICH:
#   REHEARSAL=1 INEXOR_RUNS=runs/rehearsal scripts/run/mpi_lane.sh --run \
#     bash scripts/run/multinode_gate_vista.sbatch
# Stub `nvidia-smi` and `numactl` stand in for the node's, so the samplers and the membind
# paths run. PLANT_MISMATCH=1 flips one byte of ref1's step-SPLIT checkpoint, to see the gate
# fire; PLANT_LEG_FAIL=1 gives the first run leg an unknown XLA flag, which aborts every rank
# at startup, to see the job stop.
#
# Wall (c-1024), from the one-GH200 run 1029876: ICs 105 s, 34 s/step median (late steps
# 38-46 s), checkpoint 15 s. Refs ~13 min + ~62 min; mpi unmeasured, at most the refs' time.
# ~2.4 h central with the guards; 6 h requested. Run-leg caps: 0 -> 20 in 2600 s, 20 -> 120
# in 10600 s. USR1 900 s before the wall reaches every rank through the local mpiexec and
# dumps every thread's stack.

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
  ICS_ARGS=(--tile-workers 1 --a-init 0.08)
else
  ICS_ARGS=(--generator device)
fi
PRESET=${PRESET:-c-1024} K=${K:-120} SPLIT=${SPLIT:-20} STOP=${STOP:-$K} EVERY=${EVERY:-$SPLIT}
SLACK=${SLACK:-0.10} ALLOC_MARGIN=${ALLOC_MARGIN:-0.10} ARENA=${ARENA:-${ARENA_FRAC:-0.01}}
DET_TIMEOUT=${DET_TIMEOUT:-5400}
LEG_FIXED_S=${LEG_FIXED_S:-600} LEG_STEP_S=${LEG_STEP_S:-100} QUIET_S=${QUIET_S:-600}
N_RANKS=${SLURM_NTASKS:-${N_RANKS:-2}}
. "$SRC/scripts/run/mpi_env_vista.sh"
. "$SRC/scripts/run/job_lib.sh"

WORK=$RUNS/mn-gate-$PRESET-$SLURM_JOB_ID
ICS=$WORK/ics
# shellcheck disable=SC2034  # read by job_lib.sh
LEG_DIR=$WORK/legs
CARDS=$SRC/runs/v2
TAG=mngate_${PRESET}_$SLURM_JOB_ID
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

# leg_cap FROM STOP: a run leg's time cap
leg_cap () { echo $((LEG_FIXED_S + LEG_STEP_S * ($2 - $1))); }

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
  on_ranks "$CARDS/${TAG}_${arm}_s${from}_r@RANK@" "$extra" \
    "$PY" "$DRIVER" run $(run_args "$arm" "$from" "$stop") \
    --card "$CARDS/${TAG}_${arm}@RANK@_s${from}.json" \
    --checkpoint-dir "$WORK/$arm@RANK@/ckpt" --comm serial
}

# ranks ARM FROM STOP [XLA_FLAGS]: the same run across the nodes, one rank each
ranks () {
  local arm=$1 from=$2 stop=$3 extra=${4:-}
  # shellcheck disable=SC2046
  on_ranks "$CARDS/${TAG}_${arm}_s${from}_r@RANK@" "$extra" "${MPIPATH[@]}" \
    "$PY" -m mpi4py "$DRIVER" run $(run_args "$arm" "$from" "$stop") \
    --card "$CARDS/${TAG}_${arm}_s${from}.json" \
    --checkpoint-dir "$WORK/$arm/ckpt" --comm mpi --comm-timeout 1800
}

# a failed run leg ends the job: there is nothing to compare
stop_job () {
  echo "=== stopping: the $1 leg failed; every output is kept under $WORK ==="
  exit "$rc_total"
}

# --- guards and inputs (node 0)
run_leg --cap 1800 build-mpi4py build_mpi4py || { echo "FATAL: mpi4py did not build"; exit 1; }
for n in 1 2; do
  run_leg --cap 300 "plan-$n-node" \
    "${CPU[@]}" "$PY" -m inexor.plan --preset "$PRESET" --backend device \
    --n-gpus 1 --n-nodes $n --host-gb 116 --device-gb 96 --slack "$SLACK" \
    --alloc-margin "$ALLOC_MARGIN" --arena-frac "$ARENA"
done
# the run legs' launch line, on every node, before anything long
run_leg --cap 300 guard-launch-ref \
  on_ranks "$CARDS/${TAG}_guard-ref_r@RANK@" "" \
    "$PY" -c "import jax; print('devices', jax.devices())" \
  || { echo "FATAL: the one-node launch line does not start JAX on every node"; exit 1; }
run_leg --cap 300 guard-launch-mpi \
  on_ranks "$CARDS/${TAG}_guard-mpi_r@RANK@" "" "${MPIPATH[@]}" \
    "$PY" -m mpi4py -c "from mpi4py import MPI; import jax; c = MPI.COMM_WORLD
assert c.Get_size() == $N_RANKS, c.Get_size()
print('rank', c.Get_rank(), 'of', c.Get_size(), jax.devices())" \
  || { echo "FATAL: the MPI launch line does not start every rank"; exit 1; }
run_leg --cap 300 --expect-fail guard-dead-rank \
  on_ranks "$CARDS/${TAG}_guard-dead_r@RANK@" "" "$PY" -c "import os; os.abort()" \
  || { echo "FATAL: a rank that dies does not end the launch"; exit 1; }
run_leg --cap 600 guard-kernel-block \
  "${CPU[@]}" XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    "$PY" -m pytest -q -rs -p no:cacheprovider tests/test_coarse_kernel_cards.py \
  || { echo "FATAL: the kernel block build is not bitwise on this machine"; exit 1; }
run_leg --cap 1200 guard-rank-gates \
  "${CPU[@]}" OMP_NUM_THREADS=1 \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=8" \
    "$PY" -m pytest -q -rs -p no:cacheprovider -n auto tests/test_ranks_coarse.py \
    tests/test_ranks_tile.py tests/test_ranks_migrate.py tests/test_ranks_run.py \
  || { echo "FATAL: the rank gates failed on this machine"; exit 1; }
# not fatal: the tests launch their own mpiexec, whose placement quirks are not the code's
run_leg --cap 1200 guard-mpi-lane \
  "${CPU[@]}" OMP_NUM_THREADS=1 "${MPIPATH[@]}" \
    "$PY" -m pytest -q -rs -p no:cacheprovider --basetemp "$WORK/pytest-mpi" \
    tests/test_comm_mpi.py tests/test_driver_mpi.py

run_leg --cap 1800 ics \
  "${BIND0[@]}" "${ON0[@]}" "$PY" scripts/run/realization.py ics --config "$PRESET" \
    --workdir "$ICS" "${ICS_ARGS[@]}" \
  || { echo "FATAL: the IC generation failed"; exit 1; }
for n in 1 2; do
  run_leg --cap 600 "preflight-$n-node" \
    "${BIND0[@]}" "${ON0[@]}" "$PY" "$DRIVER" preflight --preset "$PRESET" --workdir "$ICS" --cards 1 \
      --card "$CARDS/${TAG}_preflight_n$n.json" --scratch "$RUNS" --n-nodes $n \
      "${MACHINE[@]}" "${MEMBIND_CHECK[@]}" "${LAYOUT[@]}" \
    || { echo "FATAL: preflight refused"; exit 1; }
done

# --- segment 1: steps 0 -> SPLIT
RUN=(--cap "$(leg_cap 0 "$SPLIT")" --quiet-limit "$QUIET_S")
PLANT=""
[ "${REHEARSAL:-0}" = 1 ] && [ "${PLANT_LEG_FAIL:-0}" = 1 ] && PLANT=--xla_planted_unknown_flag
run_leg "${RUN[@]}" "ref 0->$SPLIT" refs ref 0 "$SPLIT" "$PLANT"; r=$?; summarize ref 0
[ $r -eq 0 ] || stop_job "ref 0->$SPLIT"
run_leg "${RUN[@]}" "mpi 0->$SPLIT" ranks mpi 0 "$SPLIT"; r=$?; summarize mpi 0
[ $r -eq 0 ] || stop_job "mpi 0->$SPLIT"
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
  RUN=(--cap "$DET_TIMEOUT" --quiet-limit "$QUIET_S")
  run_leg "${RUN[@]}" "det-ref 0->$SPLIT" refs det-ref 0 "$SPLIT" "$DET"; r=$?
  summarize det-ref 0
  [ $r -eq 0 ] || stop_job "det-ref 0->$SPLIT"
  run_leg "${RUN[@]}" "det-mpi 0->$SPLIT" ranks det-mpi 0 "$SPLIT" "$DET"; r=$?
  summarize det-mpi 0
  [ $r -eq 0 ] || stop_job "det-mpi 0->$SPLIT"
  gate "step-$SPLIT-deterministic" "$WORK/det-ref0/ckpt" "$WORK/det-ref1/ckpt" \
    "$WORK/det-mpi/ckpt" || rc_total=$((rc_total + 1))
  echo "=== stopping after segment 1: every checkpoint is kept under $WORK ==="
  exit $rc_total
fi

# --- segment 2: SPLIT -> STOP, resumed from the step-SPLIT checkpoints
if [ "$STOP" -gt "$SPLIT" ]; then
  RUN=(--cap "$(leg_cap "$SPLIT" "$STOP")" --quiet-limit "$QUIET_S")
  run_leg "${RUN[@]}" "ref $SPLIT->$STOP" refs ref "$SPLIT" "$STOP"; r=$?; summarize ref "$SPLIT"
  [ $r -eq 0 ] || stop_job "ref $SPLIT->$STOP"
  run_leg "${RUN[@]}" "mpi $SPLIT->$STOP" ranks mpi "$SPLIT" "$STOP"; r=$?
  summarize mpi "$SPLIT"
  [ $r -eq 0 ] || stop_job "mpi $SPLIT->$STOP"
  gate "step-$STOP" "$WORK/ref0/ckpt" "$WORK/ref1/ckpt" "$WORK/mpi/ckpt" \
    || rc_total=$((rc_total + 1))
fi
exit $rc_total
