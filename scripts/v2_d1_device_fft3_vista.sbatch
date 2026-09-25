#!/bin/bash
#SBATCH -J d1-fft3
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d1-fft3-%j.log
# D1 attribution: is the transform's wall the BUS, and would pinning fix it?
# Submit from the Vista checkout:
#   sbatch scripts/v2_d1_device_fft3_vista.sbatch
#
# WHERE THIS SITS. D1's own question is answered: the factorized transform is
# correct at 2048^3 (6.676e-06, reproduced on two nodes) and the coarse solve is
# ~120 s/step on one GPU, 11.2% of the 1080 s bar. This job does not re-open
# that. It asks the question D2 needs answered before it builds anything:
# **is pageable host<->device transfer a tax big enough that every device phase
# must be pinned from the start?**
#
# THE SUSPECT, with arithmetic. One transform moves ~137.5 GB across the bus
# (the inverse: pass 2 reads and writes the 34.4 GB spectrum, then the per-plane
# irfft2 reads it and writes the 34.4 GB field). At the measured 31.3 s that is
# 4.4 GB/s, against the 201 GB/s this same hardware streams from PINNED host
# memory (974476). A 46x gap. Pass 2's strided access is already bounded at
# ~11% by its own ladder, so it is not the explanation. And this project has
# measured the pageable tax once before: the gb probe's eject kernel read 14 ms
# on device against 0.528 s end to end, 38x.
#
# THE ARM. `--transfer` does the transform's TRAFFIC and none of its
# arithmetic, in the unit the factorization actually moves -- one 16.8 MB plane,
# not a 2 GiB chunk that would measure a different code path's best case. The
# two modes move identical bytes in identical units and differ only in the
# memory kind. BOTH RUN IN THIS ONE JOB ON THIS ONE NODE: 974807's flat-source
# leg changed the data and the node at the same time and its 1.41x is
# unattributable because of it. Not repeating that.
#
# The timing legs at y=1 and y=64 run here too, with the NOISE source, so the
# pencil knob and the transfer A/B are finally on the same node as each other
# and as the transform they are meant to explain.
#
# PRE-REGISTERED:
#   - If pageable comes back near 4-5 GB/s and pinned near 200, the suspect is
#     confirmed, pinning is the fix, and it is a D2 CONSTRAINT rather than a D1
#     optimization: every device phase gets pinned buffers from the start.
#   - If pinned is also single-digit GB/s at this unit size, the bus is not the
#     cost and the remaining suspect is the 8192 per-plane dispatch round trips
#     -- a different fix (batch the transfers, not the transforms).
#   - If pageable comes back FAST (near 200), then the transform's 31 s is not
#     transfer at all and both suspects are dead. That is a real outcome.
#   - `pinned_host` must appear in the arm's recorded `memory_kinds`, or the
#     pinned leg measured a fallback and says nothing. An env var that is SET is
#     not a cap that APPLIED; a memory kind that is NAMED is not one that was
#     USED.
#
# BUDGET: ~15 min expected against 1 h; ~1.1 SU worst case.

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
echo "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=$XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

rc_total=0
run_leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc ---"
  [ $rc -ne 0 ] && rc_total=$((rc_total + 1))
  return $rc
}

JAX_PLATFORMS=cpu run_leg smoke \
  pixi run -e gpu python scripts/v2_d1_device_fft.py --smoke --transfer \
    --out-suffix _d1smoke3
if [ $rc_total -ne 0 ]; then
  echo "FATAL: smoke failed, not spending a GPU on it"
  exit 1
fi

# 512 is the cheap stack receipt (must read 4.053e-06 again). --timing-n pins
# the transfer and timing legs at 2048 without paying for a third roundtrip.
CUDA_VISIBLE_DEVICES=0 run_leg transfer-ab \
  pixi run -e gpu python scripts/v2_d1_device_fft.py \
    --sizes 512 --timing-n 2048 --transfer \
    --batches 1 --pencil-batches 64 --source noise \
    --no-host-leg --slab 32 --reps 2 --out-suffix _gb3

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
