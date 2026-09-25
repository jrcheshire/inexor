#!/bin/bash
#SBATCH -J d3-escape
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d3-escape-%j.log
# Which operation in the compiled insert breaks the int16 bound on a GB200?
# Vista 995067 died in its smoke: `insert_rows` returned abs_max 49424 on the
# 32^3 xback state, where CPU XLA and numpy read exactly 32767. The bound holds
# for ANY inputs if every operation in the program returns its value, so the
# question is which one does not.
# Submit from the Vista checkout:
#   sbatch scripts/v2_d3_insert_escape_vista.sbatch
#
# TIME ASK: ~2 min estimated, 20 min wall, one gb node, billed at the 15 min
# minimum (~0.25 SU).
#
#   phase                          each (source)                               total
#   capture on the GPU, 32^3       process start ~10 s (993849); 995067 reached
#                                  the escape 40 s into the whole job, pixi
#                                  env resolution included                     ~0.7 min
#   analyze on the GPU             laptop control ran capture + analyze in a
#                                  few seconds; GPU compiles ~10 programs      ~0.7 min
#   analyze, same npz, CPU XLA     same, JAX_PLATFORMS=cpu                     ~0.5 min
#
# PRE-REGISTERED (read in this order; the first disagreement is the finding):
#   - capture: does the escape reproduce, at which call, and do the numpy-eject
#     arm's inputs equal the jax-eject arm's (the eject's part in the inputs)?
#   - analyze, GPU: production jit and eager body against the numpy emulation;
#     then the intermediates -- sort order, brick index, row magnitude, per-brick
#     max, and the per-brick max recomputed in numpy from the program's OWN
#     brick index and magnitudes (a difference there isolates the scatter).
#   - scatter-max alone on random duplicate indices: the kernel's form (int64
#     into zeros) eager and jit, and segment_max int32 as the GPU kick uses it.
#   - analyze, CPU XLA on the GPU-captured inputs: the same-node control; must
#     read all zeros and abs_max 32767, or the inputs themselves are the story.
#   No fix is chosen from this job; it names the operation.
#
# ENV copied verbatim from the D3 migrate sbatch (v2_d3_migrate_vista.sbatch).

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

NPZ=runs/v2/d3_insert_escape_inputs_gb.npz
rc_total=0
run_leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc ---"
  [ $rc -ne 0 ] && rc_total=$((rc_total + 1))
  return $rc
}

run_leg gpu \
  pixi run -e gpu python scripts/v2_d3_insert_escape_diag.py both --npz $NPZ --out-suffix _gb
run_leg cpu-control \
  env JAX_PLATFORMS=cpu pixi run -e gpu python scripts/v2_d3_insert_escape_diag.py analyze \
    --npz $NPZ --out-suffix _gb_cpu

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
