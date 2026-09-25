#!/bin/bash
#SBATCH -J d3-r0-peaks
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d3-r0-peaks-%j.log
# D3b R0 on a gb node: what does one 4096^3 slab's migrate hold on a GB200, eject
# and insert SEPARATELY, and how does the insert's hold grow with the emigrant
# share and the reach? 995264 read 30.75 GiB after both kernels in one process,
# so the insert's own peak was never separated, and its input there was one
# slab's own rows rather than keepers + the emigrants of 2r+1 slabs.
# Submit from the Vista checkout:
#   sbatch scripts/v2_d3_r0_peaks_vista.sbatch
#
# TIME ASK: ~8 min estimated, 45 min wall, one gb node, billed at the 15 min
# minimum (~0.25 SU).
#
#   phase                          each (source)                               total
#   smoke, 3 peak arms at nb=8     ~10 s start each (993849)                   ~0.5 min
#   peak-eject, 268M rows          generate 12.3 s + first call 11.2 s + 0.94 s
#                                  (995264), 2 more calls of which one may
#                                  compile a second program (+11 s)            ~0.7 min
#   peak-insert x 6                generate 13-32 s (995264's 12.3 s per slab
#   (share 0.05/0.2/0.4 x          of rows x 1.1-2.6 slabs); first call ~11 s
#    reach 1/2)                    compile + 1.3-3.3 s per call (995264's
#                                  1.27 s per slab of rows); 10 s start        ~6 min
#
# QUESTION: device peak of each kernel alone, in B per row, at a true 4096^3
# slab; for the insert, at input = keepers (1 - share) of a slab + (2r+1) x share
# of a slab of emigrants, so 1.10 to 2.60 slabs of rows. Each arm is its own
# process, so each peak belongs to one kernel. The eject arm also records how
# many programs row counts 0.3% and 1% below the slab compile.
#
# PRE-REGISTERED:
#   - The smoke passes on the GPU (no CPU fallback) or the long legs do not run.
#   - Peaks are reported, not gated. An out-of-memory at the largest input is a
#     reading, and the arms after it still run (smallest first).
#   - This prices the memory envelope (sec. 27's owed item and plan question 2);
#     it does not choose one.
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
free -g | head -2

rc_total=0
run_leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ==="
  "$@"; local rc=$?
  echo "--- LEG $name rc=$rc ---"
  [ $rc -ne 0 ] && rc_total=$((rc_total + 1))
  return $rc
}

run_leg smoke \
  pixi run -e gpu python scripts/v2_d3_device_migrate.py --arms "" --shape-nb 8 --reps 1 \
    --peaks "eject+insert:0.05:1+insert:0.4:2" --out-suffix _r0_peaks_gbsmoke
if [ $rc_total -ne 0 ]; then
  echo "FATAL: smoke failed, not spending the node on the long legs"
  exit 1
fi

run_leg peaks \
  pixi run -e gpu python scripts/v2_d3_device_migrate.py --arms "" --shape-nb 256 \
    --peaks "eject+insert:0.05:1+insert:0.05:2+insert:0.2:1+insert:0.2:2+insert:0.4:1+insert:0.4:2" \
    --out-suffix _r0_peaks_gb

echo ""
echo "=== rc_total=$rc_total ==="
exit $rc_total
