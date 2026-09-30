# shellcheck shell=bash disable=SC2034  # sourced: the variables are for the sourcing script
# Sourced by the multi-node job scripts, from the inexor checkout: the MPI launch environment
# on Vista (gh / gb), one rank per node under ibrun, host MPI from the MVAPICH-Plus module.
#
# Sets LAUNCH (the launcher, `ibrun`), MPI4PY_DIR (mpi4py built against the module, imported
# by `python -m mpi4py` legs with PYTHONPATH), and a per-rank JAX_COMPILATION_CACHE_DIR
# (`rank_exec.sh` fills in @RANK@), and defines:
#   build_mpi4py   build mpi4py 4.1.2 against the loaded module into MPI4PY_DIR, once
#   forward_usr1   the batch shell's USR1 trap: to the local mpiexec, which relays it to
#                  every rank (ibrun starts ranks over ssh, so they are not Slurm steps)
#
# REHEARSAL=1 (the laptop, inside `scripts/run/mpi_lane.sh --run`): no modules and no build;
# LAUNCH is the env's `mpiexec -n 2` and mpi4py is the env's own.
#
# MVAPICH-Plus 5.1.0 is libmpi.so.0 (not the MPICH ABI conda's mpi4py links), so mpi4py is
# built against the module; the gpu env is untouched.

if [ "${REHEARSAL:-0}" = 1 ]; then
  LAUNCH="mpiexec -n ${N_RANKS:-2}"
  MPI4PY_DIR=""
  build_mpi4py () { echo "rehearsal: mpi4py from the env"; }
else
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

forward_usr1 () {
  echo "=== USR1 at $(date +%H:%M:%S): to the local mpiexec (every rank's stack dump) ==="
  pkill -USR1 -x mpiexec 2>/dev/null
  ( sleep 60; pkill -USR1 -x mpiexec 2>/dev/null ) &
}
