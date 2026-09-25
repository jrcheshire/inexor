#!/bin/bash
#SBATCH -J d5c-hostreg
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d5c-hostreg-%j.log
# D5c: price page-locked, numpy-WRITABLE host memory.
# Submit from the Vista checkout:
#   sbatch scripts/v2_d5c_host_register_vista.sbatch
#
# WHERE THIS SITS. D5b killed the cheap route with a measurement: staging each
# crossing through `pinned_host` is 1.23-1.39x SLOWER than handing the driver
# ordinary numpy (115.1 vs 83.0 s/step at W=1, 33.5 vs 27.3 at W=4). JAX arrays
# are immutable, so every crossing allocates a fresh pinned buffer and pays a
# kernel call to page-lock it. The 7.8x memory-kind ceiling is real but only
# reachable by a buffer pinned ONCE and reused -- which needs memory that is
# page-locked AND numpy-writable, the thing JAX cannot express and
# `cudaHostRegister` can, by locking pages that already exist.
#
# THE DECIDER IS NOT WHETHER WE CAN GET THAT MEMORY. It is whether XLA TAKES
# THE FAST PATH when handed it. PJRT may memcpy any host pointer into its own
# staging buffer without ever asking whether it is already page-locked, in
# which case the route is dead however cheap the allocation is. Gate 1 is that
# question, it is the cheapest leg here, and everything after it is only worth
# reading if it comes back positive.
#
# IF IT IS POSITIVE THE PAYOFF IS STRUCTURAL, not incremental: registration is
# a property of the ALLOCATION, not of the transfer call, so the existing
# "pageable" code path gets faster with NO change to the device passes at all.
# The engine registers its buffers once at startup and every crossing in every
# phase benefits -- including the decode and the kick already on the device.
#
# NO NEW DEPENDENCY: libcudart.so.12 already ships inside the gpu env
# (jaxlib's cuda build), reached by ctypes. Nothing is added to a locked
# environment to answer this.
#
# PRE-REGISTERED:
#   - Gate 1 registered rate lands near the pinned ceiling: route ALIVE, and
#     the next question is registering the scratch buffers, not this one.
#   - Gate 1 registered rate is indistinguishable from pageable: XLA ignores
#     registration, THE ROUTE IS DEAD, and pinning is closed as a line of work
#     rather than left open. Gates 2-4 are then reported but decide nothing.
#   - Gate 1 lands between: report the fraction of the ceiling recovered; that
#     is a cost/benefit call for JC, not a measurement.
#   - Registration cost is an AMORTIZATION question, not a wall: it is paid
#     once per run against 40 steps x 4 transforms. The ladder exists to price
#     the ~724 GB the design's host state needs, and to find out whether a
#     registration that size is even permitted.
#   - PORTABLE vs DEFAULT flags are both measured. The design uses four GPUs
#     and default registration binds pages to the current context; if portable
#     costs materially more, the four-GPU design carries that number.
#   - A registration that silently did not happen looks exactly like one that
#     did and bought nothing, so every leg checks `cudaHostRegister` rc, then
#     `cudaHostGetFlags` on the pointer (which succeeds only for a registered
#     one), then that the array still reads back what was written. A leg
#     missing any receipt is VOID, not slow.
#   - Gate 4 is expected to be a PARTIAL win and that is not a defect: pass 1
#     transfers slices OF the spectrum and benefits, pass 2 transfers fresh
#     `ascontiguousarray` temporaries nobody registered. The gap between gate 1
#     and gate 4 IS the price of registering the scratch buffers.
#
# BUDGET: ~15 min expected against 1 h; ~1 SU worst case at gb's 1 SU/node-hr.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=900
unset JAX_PLATFORMS
unset CUDA_VISIBLE_DEVICES

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
free -g | head -2
echo "ulimit -l (compute node): $(ulimit -l)"

rc_total=0
run_leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc ---"
  [ $rc -ne 0 ] && rc_total=$((rc_total + 1))
  return $rc
}

# The smoke runs ON THE GPU at trivial sizes, because there is no CPU form of
# this question: page-locking needs a CUDA context. It exercises every path
# including the ctypes binding and the receipts, for a few seconds of node time.
run_leg smoke \
  pixi run -e gpu python scripts/v2_d5c_host_register.py --smoke \
    --out-suffix _d5csmoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: smoke failed, not running the ladder"
  exit 1
fi

run_leg price \
  pixi run -e gpu python scripts/v2_d5c_host_register.py \
    --n 2048 --plane-n 2048 --sizes-gb 1,4,16,64,128 \
    --slab 32 --plane-batch 1 --pencil-batch 64 --reps 5 \
    --out-suffix _gb

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
