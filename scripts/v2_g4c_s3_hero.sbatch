#!/bin/bash
#SBATCH -J inexor-v2-g4c-hero
#SBATCH -o %x-%j.log
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 04:00:00
# G4c on Stampede3 h100: turn the fabric ratio into a WALL ratio, and ask
# whether C-hero's actual state size streams at all.
# Submit from ~/src/inexor on Stampede3:
#   sbatch scripts/v2_g4c_s3_hero.sbatch
#
# WHY THIS EXISTS. G4 measured the C2C-vs-PCIe fabric gap at 6.71-6.87x
# (job 3380722 vs Vista 894036), and that number is currently the argument
# against C-hero. But it is BANDWIDTH-BOUND, and the only real-workload
# point we have -- the C-gh paint on the GH200 -- ran at 7.8 GB/s staged,
# 46x BELOW that machine's own ladder rate, because paint is not
# bandwidth-bound. So 6.9x is an upper bound on the end-to-end penalty and
# possibly a very loose one. Sizing C-hero from it would be reading a
# bandwidth ratio as a wall ratio.
#
# LEG 1 -- the matched real point. Same C-gh paint (2048^3 into a 1024^3
# mesh) that Vista ran, so the two are directly comparable and the fabric
# gap converts into the number that actually matters: wall.
#
# LEG 2 -- does C-hero's STATE fit and stream? 4096^3 at the ratified T9
# tier is 618.5 GB = 576 GiB. That is the quantity the hero config has to
# hold, and this node's 1007 GiB host is the only place we have that could.
# Rungs 256/448/576 GiB check flatness on the way up as well as arrival,
# because the GH200 taught us the failure is a CLIFF at physical capacity
# rather than a gradual roll-off (sec. 5.5) -- so "it completed" is not
# enough, the rate has to still be flat.
#
# NOT a res_pm=N force evaluation. A 4096^3 mesh would be 274.9 GB of int32
# on-device and cannot fit 93.6 GiB of HBM; the real thing needs the tiled
# two-level scheme, not this probe. Leg 2 measures STATE STREAMING at hero
# scale, which is the memory question, and claims nothing about the force.
#
# HOST LIMIT RAISED TO 900 GB. The 160 GB used for the 128 GiB ladder would
# cap leg 2 at a quarter of its smallest rung -- exactly how job 894010 was
# voided. The probe reads the cap back and declares the capacity witness
# void if it does not clear the ladder top; 900 GB against a 576 GiB
# (618.5 GB) top, on a 1081 GB node, leaves headroom without crowding the
# node the way the GH200's own LPDDR got crowded.
#
# NO `set -e`: leg 1 is the matched comparison and must not be lost if
# leg 2's large rungs fail. Cheapest first.
#
# Output: runs/v2/g4_gh_memory_s3real.json, runs/v2/g4_gh_memory_s3hero.json

cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=900
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache

echo "=== provenance ==="
git log --oneline -1
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
free -g | head -2

echo
echo "=== LEG 1: matched C-gh real paint point (compare Vista 894036 REAL) ==="
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu python scripts/v2_g4_gh_memory.py \
  --ladder 64 --arms staged --real --out-suffix _s3real

echo
echo "=== LEG 2: C-hero state size -- 256/448/576 GiB (576 GiB = 618.5 GB = T9 at 4096^3) ==="
CUDA_VISIBLE_DEVICES=0 pixi run -e gpu python scripts/v2_g4_gh_memory.py \
  --ladder 256,448,576 --arms staged,coherent --out-suffix _s3hero

echo
echo "=== DONE ==="
