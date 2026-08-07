#!/bin/bash
#SBATCH -J inexor-v2-g4-s3
#SBATCH -o %x-%j.log
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 04:00:00
# G4 on STAMPEDE3 h100 -- the C-hero memory-fabric measurement.
# Submit from ~/src/inexor on Stampede3:
#   sbatch scripts/v2_g4_s3_h100.sbatch
#
# WHAT THIS TESTS, and it is a PREDICTION OF MINE, not a known quantity.
# Vista's GH200 streamed host-resident state at 359-429 GB/s because Grace
# and Hopper are coherently packaged (C2C). I claimed C-hero's 1 TB host
# would run ~6-7x slower because it is PCIe-attached (~55-64 GB/s), and that
# claim is currently deciding an architecture question: multi-GPU sharding on
# a gb node (768 GB aggregate HBM, needs sharding inexor does not have)
# versus a bandwidth penalty on C-hero (1 TB host, single-GPU streaming,
# works with today's code). The claim has never been measured. This measures
# it.
#
# MATCHED AXES ARE THE WHOLE POINT. Same script, same 64/88/104/128 GiB
# rungs, same 2 GiB chunk, same arms, same caps, and CUDA_VISIBLE_DEVICES=0
# so it is one GPU exactly as the Vista run was. Then the difference between
# the two cards is the memory fabric and nothing else. Do NOT "improve" the
# ladder here -- an unmatched rung buys nothing.
#
# NODE (from Slurm, not assumed): RealMemory=1031000 MB ~= 1 TB host,
# CPUTot=96. This is the config table's C-hero home, and 1 TB is what makes
# 4096^3 (618.5 GB of T9 state) reachable at all -- it fits nowhere else we
# have. That run is NOT this job; this one settles the rate first.
#
# HBM IS DETECTED per node (nvidia-smi), because the probe's old hardcoded
# 96 GiB is a GH200 number and would answer `exceeds_hbm` and the device-cap
# precondition against the wrong card here.
#
# qh100 caps wall at 2 days. 4 h is generous for a ladder that took 34 min on
# Vista; if PCIe really is ~7x slower the traversals grow but stay minutes.
#
# NO `set -e`: independent rungs; an OOM is data.
#
# Output: runs/v2/g4_gh_memory_s3.json + runs/v2/g4b_capability_s3.json

cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=160
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache

echo "=== provenance ==="
git log --oneline -1
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
free -g | head -2
lscpu | grep -E "^CPU\(s\)|Model name|NUMA node\(s\)"

echo
echo "=== LEG 1: node capability + fp64 (ALL GPUs visible) ==="
pixi run -e gpu python scripts/v2_g4b_fp64_capability.py --out-suffix _s3

echo
echo "=== LEG 2: the matched ladder, ONE GPU (compare against Vista 894036) ==="
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu python scripts/v2_g4_gh_memory.py \
  --ladder 64,88,104,128 --out-suffix _s3

echo
echo "=== DONE ==="
