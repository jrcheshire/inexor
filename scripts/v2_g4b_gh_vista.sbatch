#!/bin/bash
#SBATCH -J inexor-v2-g4b-gh
#SBATCH -o %x-%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 01:00:00
# G4b on a Vista GRACE-HOPPER node -- the MATCHED CONTROL for the gb run.
# Byte-derived from v2_g4b_gb_vista.sbatch (partition + out-suffix only), so
# the estimator is identical by construction rather than by inspection. Submit from ~/src/inexor:
#   sbatch scripts/v2_g4b_gh_vista.sbatch
# The gb twin (scripts/v2_g4b_gb_vista.sbatch) MUST also run -- an fp64
# number from one node type alone answers nothing.
#
# WHY: V4's node-ladder call needs two things nobody has written down.
# (1) What is on a gb node. Working figures are ~192 GB HBM/card, 4 cards,
#     2x Grace (144 cores), LPDDR UNKNOWN -- and LPDDR is the one that
#     decides whether a single card can stream 4096^3 (618.5 GB of T9
#     state) or whether it takes multi-GPU sharding inexor does not have.
# (2) Whether GB200 fp64 is usable at all. inexor's force and FFT path is
#     f64 throughout, and Blackwell is reported to cut vector fp64 hard.
#     If it does, capacity won here is paid back in wall, and that is worth
#     knowing BEFORE anyone builds sharding for this node.
#
# NO CUDA_VISIBLE_DEVICES: all 4 cards must be visible so the capability
# dump can report HBM per card and confirm the count.
#
# qgb caps wall at 12 h; 1 h is ~10x what the probe needs and short walls
# backfill sooner, which matters because this queue is expected to sit.
#
# NO `set -e`: the capability dump is the primary deliverable and must
# survive any benchmark failing.

cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache

echo "=== provenance ==="
git log --oneline -1
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
free -g | head -2
lscpu | grep -E "^CPU\(s\)|Model name|NUMA node\(s\)"

echo
pixi run -e gpu python scripts/v2_g4b_fp64_capability.py --out-suffix _gh

echo
echo "=== DONE ==="
