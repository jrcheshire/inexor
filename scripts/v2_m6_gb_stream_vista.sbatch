#!/bin/bash
#SBATCH -J m6-gb-stream
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m6-gb-stream-%j.log
# THE gb STREAMING RE-RUN: the one premise of the host-state / device-step
# design that job 972737 did NOT measure. Submit from the Vista checkout:
#   sbatch scripts/v2_m6_gb_stream_vista.sbatch
#
# WHY. 972737's streaming legs OOM'd at every host rung of 128 GiB and above
# with "Out of host memory" on a node with 1026 GB, because
# XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB defaults to 64 GB and the gb sbatch did not
# export it -- every earlier streaming job did (gh: 160, S3 hero: 900; record
# g4_record.md sec. 4). So the largest host working set one GPU streamed was
# 64 GiB, against a 675 GiB state at 4096^3. This job sets the flag and runs
# the same ladder past HBM (185 GiB), past one socket (~478 GiB), and up to two
# thirds of host RAM (640 GiB), where gh collapsed at the same fraction.
#
# LEGS:
#   1  one GPU, `staged` arm, 64 -> 640 GiB. The `coherent` arm is DROPPED:
#      on this stack it fails to COMPILE ("INTERNAL: Failed to get configs for:
#      3 out of 3 instructions", 972737 leg stream-1gpu-a), so it measures
#      nothing here; explicit staging out of pinned host is the path.
#   2  all four GPUs at once, `staged`, 64/128/192 GiB each = up to 768 GiB of
#      pinned host in flight (75% of the node) -- the aggregate the readout
#      bills a step against, in the pressure regime the design would run in.
#
# PRE-REGISTERED: the capacity witness is binary (a rung above 185 GiB that
# completes with device peak pinned at 2 chunks streamed); the rate at the
# top completed rung is the number; a collapse like gh's 150-270x fall at 128
# of 192 GiB is a RESULT, not a failure, and names the ceiling the design
# must stay under. Aggregate < the single-GPU rate x4 measures LPDDR supply
# against four C2C links (486 GB/s per socket on paper).
#
# BUDGET: ~20 min expected against 1.5 h (pinning 640 GiB has never been
# timed here); 1.5 SU worst case.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
# THE FLAG 972737 LACKED. 900 GB: below the node's 1026 GB of LPDDR, above the
# 768 GiB the 4x leg asks for in total (each process gets its own cap).
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=900
unset JAX_PLATFORMS
unset CUDA_VISIBLE_DEVICES

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID ==="
echo "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=$XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
free -g | head -2
numactl -H 2>/dev/null | grep -E "node [01] size"

pixi run -e gpu python -c "
import jax
ds = jax.devices()
assert ds[0].platform == 'gpu', ds
print('jax', jax.__version__, len(ds), 'devices:', ds[0].device_kind)
" || { echo "FATAL: jax cuda backend check failed"; exit 1; }

rc_total=0
names=(); codes=()
run_leg() {
  local name="$1"; shift
  echo; echo "=== $name ==="
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "=== $name rc=$rc after $((SECONDS - t0)) s ==="
  names+=("$name"); codes+=("$rc")
  [ $rc -ne 0 ] && rc_total=1
  return $rc
}

G4="scripts/v2_g4_gh_memory.py"

# ---- 1. one GPU, staged, past HBM, past one socket, to two thirds of the node
run_leg stream-1gpu env CUDA_VISIBLE_DEVICES=0 pixi run -e gpu python $G4 \
  --arms staged --ladder 64,128,256,384,512,640 --reps 3 --out-suffix _gb2_b

# ---- 2. four GPUs at once, staged, up to 768 GiB pinned in total
echo; echo "=== stream-4gpu ==="
t0=$SECONDS
pids=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i pixi run -e gpu python $G4 --arms staged --ladder 64,128,192 \
    --reps 3 --out-suffix _gb2x4_gpu$i > "m6-gb-stream-${SLURM_JOB_ID}-4x-gpu$i.log" 2>&1 &
  pids+=($!)
done
rc4=0
for i in 0 1 2 3; do
  wait "${pids[$i]}"; rc=$?
  echo "--- gpu$i rc=$rc"; grep -E "^(staged|HBM cliff)" "m6-gb-stream-${SLURM_JOB_ID}-4x-gpu$i.log"
  [ $rc -ne 0 ] && rc4=1
done
echo "=== stream-4gpu rc=$rc4 after $((SECONDS - t0)) s ==="
names+=("stream-4gpu"); codes+=("$rc4")
[ $rc4 -ne 0 ] && rc_total=1

# ---- 3. the readout with the re-run's streaming cards beside 972737's other cards
run_leg readout env JAX_PLATFORMS=cpu pixi run -e gpu python scripts/v2_m6_gb_probe_readout.py \
  --suffix _gb --stream-suffix _gb2 --preset c-hero --k 40 --n-gpus 4

echo; echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "=== CARDS WRITTEN ==="
ls -la runs/v2/g4_gh_memory_gb2_b.json runs/v2/g4_gh_memory_gb2x4_gpu*.json 2>&1
echo "=== DONE rc_total=$rc_total ==="
exit $rc_total
