# shellcheck shell=bash disable=SC2034  # sourced: the variables are for the sourcing script
# Sourced by the multi-node job scripts, from the inexor checkout: the machine and MPI launch
# environment on Vista gh (one GH200 per node) or gb (four GB200 per node, two CPU memory
# nodes), one rank per node under ibrun, host MPI from the MVAPICH-Plus module.
#
# Requires SRC (the checkout) and RUNS (the run root); reads N_RANKS (2) and REHEARSAL. Sets:
#   PY             the gpu env's python (the laptop's in a rehearsal)
#   LAUNCH         the launcher, `ibrun` (rehearsal: `mpiexec -n N_RANKS`)
#   MPI4PY_DIR     mpi4py built against the module, for `python -m mpi4py` legs; MPIPATH is the
#                  `env PYTHONPATH=...` prefix that imports it (empty in a rehearsal)
#   BIND0          the numactl prefix of a node-0 process (ICs, preflight)
#   MEMBIND_RANK   rank_exec.sh's binding of each rank; MEMBIND_CHECK the driver's check of it
#   MACHINE        the preflight's host and card sizes
#   N_CARDS        GPUs per rank: 1 on gh, 4 on gb (a rehearsal: 1)
#   PLAN_MACHINE   the planner's per-node GPU count and host / card sizes (a rehearsal: gh's)
#   ON0, CPU       `env` prefixes: node 0's compile cache; the CPU backend
#   DRIVER, RANK_EXEC  scripts/run/device_run.py and scripts/run/rank_exec.sh
# exports the XLA / JAX variables of a GPU leg and a per-rank JAX_COMPILATION_CACHE_DIR
# (`rank_exec.sh` fills in @RANK@), and defines:
#   build_mpi4py   build mpi4py 4.1.2 against the loaded module into MPI4PY_DIR, once
#   forward_usr1   the batch shell's USR1 trap: to the local mpiexec, which relays it to
#                  every rank (ibrun starts ranks over ssh, so they are not Slurm steps)
#
# REHEARSAL=1 (the laptop, inside `scripts/run/mpi_lane.sh --run`): the CPU backend, no
# modules and no build; mpi4py is the env's own. Stub `nvidia-smi` and `numactl` stand in for
# the node's, so the samplers and the membind paths run.
#
# MVAPICH-Plus 5.1.0 is libmpi.so.0 (not the MPICH ABI conda's mpi4py links), so mpi4py is
# built against the module; the gpu env is untouched.

N_RANKS=${N_RANKS:-2}
if [ "${REHEARSAL:-0}" = 1 ]; then
  SLURM_JOB_ID=${SLURM_JOB_ID:-rehearsal-$$}
  PY=python
  export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
  export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=1"
  export XLA_CACHE_ROOT=$RUNS/xla-cache-$SLURM_JOB_ID
  STUBS=$RUNS/stubs-$SLURM_JOB_ID
  mkdir -p "$STUBS"
  printf '#!/bin/sh\necho "0, 1, 0"\n' > "$STUBS/nvidia-smi"
  # numactl --membind=NODES CMD...: CMD
  cat > "$STUBS/numactl" <<'STUB'
#!/bin/sh
while [ $# -gt 0 ]; do case "$1" in --*) shift ;; *) break ;; esac; done
exec "$@"
STUB
  chmod +x "$STUBS/nvidia-smi" "$STUBS/numactl"
  export PATH="$STUBS:$PATH"
  # MEMBIND_CHECK stays off: the driver's binding check reads Linux /proc
  BIND0=(numactl --membind=0) MEMBIND_RANK=(--membind 0) MEMBIND_CHECK=() MACHINE=(--allow-cpu)
  N_CARDS=1 PLAN_MACHINE=(--n-gpus 1 --host-gb 116 --device-gb 96)

  LAUNCH="mpiexec -n $N_RANKS"
  MPI4PY_DIR=""
  build_mpi4py () { echo "rehearsal: mpi4py from the env"; }
else
  case "$SLURM_JOB_PARTITION" in
    gh|gb) ;;
    *) echo "FATAL: partition $SLURM_JOB_PARTITION is not gh or gb"; exit 1 ;;
  esac
  command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
  export XLA_CLIENT_MEM_FRACTION=0.95
  # JAX's default host-memory cap is 64 GB, below what host streaming needs here
  if [ "$SLURM_JOB_PARTITION" = gb ]; then
    export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=900
  else
    export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=160
  fi
  export INEXOR_LOAD_TRACE=1
  export JAX_LOG_COMPILES=1
  unset JAX_PLATFORMS CUDA_VISIBLE_DEVICES XLA_FLAGS
  ENV=$SRC/.pixi/envs/gpu
  PY=$ENV/bin/python3.14  # the versioned launcher: a scratch purge can take `python`
  [ -x "$PY" ] || { echo "FATAL: no $PY (install the gpu env first)"; exit 1; }
  export CONDA_PREFIX=$ENV
  # host memory on the CPU nodes only (gb: 0 and 1; the GPUs' HBM nodes stay free)
  # shellcheck disable=SC2054  # "0,1" is one numactl node list
  if [ "$SLURM_JOB_PARTITION" = gb ]; then
    BIND0=(numactl --membind=0,1) MEMBIND_RANK=(--membind 0,1)
    MEMBIND_CHECK=(--membind-nodes 0,1) MACHINE=(--host-gb 1026 --device-gb 199) N_CARDS=4
    PLAN_MACHINE=(--n-gpus 4 "${MACHINE[@]}")
  else
    BIND0=(numactl --membind=0) MEMBIND_RANK=(--membind 0) MEMBIND_CHECK=(--membind-nodes 0)
    MACHINE=(--host-gb 116 --device-gb 96) N_CARDS=1 PLAN_MACHINE=(--n-gpus 1 "${MACHINE[@]}")
  fi

  type module >/dev/null 2>&1 || . /etc/profile  # a non-interactive submission shell
  MPI_MODULE=${MPI_MODULE:-mvapich-plus/5.1.0}
  module load gcc/15.1.0 cuda/13.0 "$MPI_MODULE" || { echo "FATAL: module load failed"; exit 1; }
  [ -n "$TACC_MV_LIB" ] && command -v mpicc >/dev/null || { echo "FATAL: no MVAPICH env"; exit 1; }
  LAUNCH=ibrun
  MPI4PY_VERSION=4.1.2
  MPI4PY_DIR=${MPI4PY_DIR:-$SCRATCH/inexor-mpi4py/${MPI_MODULE//\//-}-mpi4py$MPI4PY_VERSION-cp314}
  # one rank per node keeps every core (hydra otherwise binds -bind-to core:N); host buffers
  # only, so MPI makes no CUDA context of its own
  export HYDRA_AFFINITY="-bind-to none" MVP_ENABLE_GPU=0
  # after the module: the env's own MPICH (in its bin) must not shadow the module's launcher
  export PATH="$PATH:$ENV/bin"

  # built into a job-private dir, then moved into place, so two jobs building at once
  # cannot interleave
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
fi

# two processes never write one cache: rank r of every leg uses cache r, so the one-rank
# reference on node r and MPI rank r compile into (and reuse) the same directory
export JAX_COMPILATION_CACHE_DIR=${XLA_CACHE_ROOT:-$SCRATCH/inexor-xla-cache}/rank@RANK@
ON0=(env "JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR//@RANK@/0}")
CPU=(env JAX_PLATFORMS=cpu)
MPIPATH=()
[ -n "$MPI4PY_DIR" ] && MPIPATH=(env "PYTHONPATH=$MPI4PY_DIR${PYTHONPATH:+:$PYTHONPATH}")
DRIVER=$SRC/scripts/run/device_run.py
RANK_EXEC=$SRC/scripts/run/rank_exec.sh

forward_usr1 () {
  echo "=== USR1 at $(date +%H:%M:%S): to the local mpiexec (every rank's stack dump) ==="
  pkill -USR1 -x mpiexec 2>/dev/null
  ( sleep 60; pkill -USR1 -x mpiexec 2>/dev/null ) &
}
