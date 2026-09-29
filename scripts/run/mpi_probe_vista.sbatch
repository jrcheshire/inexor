#!/bin/bash
#SBATCH -J inexor-mpi-probe
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:30:00
#SBATCH -o mpi-probe-%j.log
#SBATCH --signal=B:USR1@1500
# Two-node MPI probe (`scripts/run/mpi_probe.py`): one rank per node under ibrun, host MPI from
# the MVAPICH-Plus module (MPI_MODULE, default mvapich-plus/5.1.0, loaded on gcc/15.1.0 +
# cuda/13.0). Its libmpi is libmpi.so.0, not the MPICH ABI the env's conda mpi4py links, so
# the first leg builds mpi4py against the module into MPI4PY_DIR (once; reused after) and the
# legs import it from there; the gpu env itself is untouched. Submit once per partition:
#   INEXOR_SRC=$SCRATCH/src/inexor sbatch -A <account> -p gh scripts/run/mpi_probe_vista.sbatch
#   INEXOR_SRC=$SCRATCH/src/inexor sbatch -A <account> -p gb scripts/run/mpi_probe_vista.sbatch
# Legs, in order: identity, bandwidth, signal, abort, kill; on gb also host-attrib (one node,
# no MPI: the cgh64 device run's per-phase host allocations). The pre-wall USR1 fires ~5 min
# in (1500 s before the 30 min wall). The trap stamps its time and forwards it twice, told
# apart by lag: at once to the local mpiexec (ibrun launches ranks over ssh, so they are not
# Slurm steps), and 30 s later with `scancel -s USR1` (job steps). Every leg records
# receipts, so the signal is kept whichever leg it lands in.
# Every leg prints `GATE <name> PASS|FAIL`; the job exits with the number of failed legs.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "${INEXOR_SRC:-$SLURM_SUBMIT_DIR}" && [ -f scripts/run/mpi_probe.py ] || {
  echo "FATAL: not an inexor checkout: ${INEXOR_SRC:-$SLURM_SUBMIT_DIR} (set INEXOR_SRC)"; exit 1; }
SRC=$(pwd)
case "$SLURM_JOB_PARTITION" in
  gh) MEMS="0"; GPUS=1 ;;
  gb) MEMS="0,1"; GPUS=4 ;;
  *) echo "FATAL: partition $SLURM_JOB_PARTITION is not gh or gb"; exit 1 ;;
esac
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_CLIENT_MEM_FRACTION=0.95
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=160
unset JAX_PLATFORMS CUDA_VISIBLE_DEVICES
# one rank per node keeps every core (ibrun's hydra launch otherwise binds -bind-to core:N);
# host buffers only, so MPI makes no CUDA context of its own
export HYDRA_AFFINITY="-bind-to none" MVP_ENABLE_GPU=0

ENV=$SRC/.pixi/envs/gpu
PY=$ENV/bin/python3.14  # the versioned launcher: a scratch purge can take `python`
[ -x "$PY" ] || { echo "FATAL: no $PY (install the gpu env first)"; exit 1; }
export CONDA_PREFIX=$ENV
type module >/dev/null 2>&1 || . /etc/profile  # a non-interactive submission shell
MPI_MODULE=${MPI_MODULE:-mvapich-plus/5.1.0}
module load gcc/15.1.0 cuda/13.0 "$MPI_MODULE" || { echo "FATAL: module load failed"; exit 1; }
[ -n "$TACC_MV_LIB" ] && command -v mpicc >/dev/null || { echo "FATAL: no MVAPICH env"; exit 1; }
MPI4PY_VERSION=4.1.2
MPI4PY_DIR=${MPI4PY_DIR:-$SCRATCH/inexor-mpi4py/${MPI_MODULE//\//-}-mpi4py$MPI4PY_VERSION-cp314}

OUT=${INEXOR_RUNS:-$SCRATCH/inexor_runs}/mpi-probe-$SLURM_JOB_PARTITION-$SLURM_JOB_ID
mkdir -p "$OUT"
echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
echo "=== partition $SLURM_JOB_PARTITION, nodes $SLURM_JOB_NODELIST, out $OUT ==="
echo "=== modules $(module -t list 2>&1 | tr '\n' ' '); mpi4py from $MPI4PY_DIR ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null
env | grep -E '^(MV2_|MVP_|UCX_|XLA_|JAX_)' | sort

CURRENT=none
FORWARDED=0
stamp_usr1 () {
  # once: if scancel also reached this shell, a second forward would loop
  [ $FORWARDED -eq 1 ] && { echo "=== USR1 again at $(date +%H:%M:%S) (not forwarded) ==="; return; }
  FORWARDED=1
  python3 -c 'import time; print(time.time())' > "$OUT/batch_usr1.txt"
  echo "=== USR1 at $(date +%H:%M:%S) during leg $CURRENT: to the local mpiexec now, and"
  echo "    to the job steps in 30 s ==="
  pkill -USR1 -x mpiexec
  ( sleep 30; echo "=== scancel -s USR1 at $(date +%H:%M:%S) ==="; scancel -s USR1 "$SLURM_JOB_ID" ) &
}
trap stamp_usr1 USR1

rc_total=0
LEG_RC=0
LEG_S=0
# a leg runs in the background and is waited for, so the USR1 trap runs while it does
run_leg () {
  local name="$1"; shift
  CURRENT=$name
  echo ""; echo "=== LEG $name ($(date +%H:%M:%S)) ==="
  local t0; t0=$(date +%s.%N)
  "$@" > "$OUT/$name.log" 2>&1 & local pid=$!
  local rc
  while true; do
    wait $pid; rc=$?
    kill -0 $pid 2>/dev/null || break
  done
  LEG_RC=$rc
  LEG_S=$(python3 -c "import time; print(time.time() - $t0)")
  cat "$OUT/$name.log"
  echo "--- LEG $name rc=$rc (${LEG_S%.*} s) ---"
  CURRENT=none
  return $rc
}
gate_leg () { run_leg "$@" || rc_total=$((rc_total + 1)); }

PROBE="$SRC/scripts/run/mpi_probe.py"
EXPECT=(--out "$OUT" --expect-ranks 2 --expect-gpus "$GPUS" --expect-mems ${MEMS//,/ }
        --expect-mpi MVAPICH --mpi4py-dir "$MPI4PY_DIR")
mpi_leg () {
  local leg=$1 guard=$2; shift 2
  timeout "$guard" ibrun numactl --membind="$MEMS" env PYTHONPATH="$MPI4PY_DIR" \
    "$PY" -m mpi4py "$PROBE" "$leg" "${EXPECT[@]}" "$@"
}

# mpi4py against the module, once per module: built into a job-private dir, then moved into
# place, so two jobs building at once cannot interleave
build_mpi4py () {
  if PYTHONPATH="$MPI4PY_DIR" "$PY" -c "import mpi4py, sys; sys.exit(not mpi4py.__file__.startswith('$MPI4PY_DIR'))" 2>/dev/null; then
    echo "mpi4py already built in $MPI4PY_DIR"; return 0
  fi
  local tmp=$MPI4PY_DIR.tmp.$SLURM_JOB_ID
  mkdir -p "$(dirname "$MPI4PY_DIR")"
  MPICC=$(command -v mpicc) pixi exec --spec python=3.14 --spec pip -- \
    pip install --no-cache-dir --no-binary mpi4py --target "$tmp" "mpi4py==$MPI4PY_VERSION" \
    && { mv -T "$tmp" "$MPI4PY_DIR" 2>/dev/null || [ -d "$MPI4PY_DIR" ]; }
}
run_leg build-mpi4py build_mpi4py || { echo "FATAL: mpi4py did not build"; exit 1; }
gate_leg identity mpi_leg identity 300
gate_leg bandwidth mpi_leg bandwidth 300 --volume-mib 4096
gate_leg signal mpi_leg signal 600 --wait 480 --batch-stamp "$OUT/batch_usr1.txt"
for leg in abort kill; do
  run_leg "$leg" mpi_leg "$leg" 300
  gate_leg "$leg-check" "$PY" "$PROBE" check-fail --out "$OUT" --log "$OUT/$leg.log" \
    --rc "$LEG_RC" --elapsed "$LEG_S" --bound 60
done
if [ "$SLURM_JOB_PARTITION" = gb ]; then
  gate_leg host-attrib timeout 900 numactl --membind="$MEMS" \
    "$PY" "$PROBE" host-attrib --out "$OUT" --preset cgh64 --steps 2
fi

echo ""; echo "=== sacct ==="
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName%20,State,Elapsed,ExitCode -P 2>/dev/null
echo "=== end $(date) failed legs: $rc_total ==="
exit $rc_total
