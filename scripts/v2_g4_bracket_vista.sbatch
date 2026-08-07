#!/bin/bash
#SBATCH -J inexor-v2-g4-bracket
#SBATCH -o %x-%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 02:00:00
# G4 follow-up: WHERE is the rate knee? Submit from ~/src/inexor on Vista:
#   sbatch scripts/v2_g4_bracket_vista.sbatch
#
# WHAT THIS RESOLVES. Job 894036 measured both host arms flat to 104 GiB
# (staged 367/360/359, coherent 418/418/429 GB/s) and then COLLAPSED at
# 128 GiB (2.4 and 1.6 GB/s), in all 5 reps, with walls of 42-100 s against
# 0.26-0.31 s one rung down. Allocation did NOT degrade (10.0 s at 128 vs
# 9.8 at 104) and it is not first-touch (at 88/104 only the FIRST timed rep
# is slow; at 128 every rep is), so it is sustained re-fetching.
#
# THE HYPOTHESIS, which is a one-point discrimination and needs a bracket:
# the config table puts this GH200 at 116 GB of LPDDR = 108.0 GiB, and
#   104 GiB = 111.7 GB  FITS
#   128 GiB = 137.4 GB  EXCEEDS
# i.e. the last rung at full rate is the last one that fits physical LPDDR,
# and the only rung that collapses is the only one that does not. If the
# knee sits at ~108 GiB the mechanism is settled; if it sits elsewhere the
# LPDDR story is dead and it is something else. Six rungs across the gap:
# 104 / 108 / 112 / 116 / 120 / 128 GiB, all multiples of the 2 GiB chunk.
#
# ARMS: staged + coherent only. The hbm arm cannot run any of these -- the
# device cap is 90.2 GiB (0.95 of the card) -- so including it would buy six
# guaranteed OOMs and nothing else.
#
# CAPS: same as the verdict job and for the same reason (894010 was VOIDED
# by their absence). The probe reads bytes_limit back off the device and
# declares the capacity witness void if the caps do not clear the ladder.
#
# COST: 894036 did 15 workers in 34 minutes; this is 12 workers with no real
# point, so ~15-25 min. 2 h requested (TACC bills actual use).
#
# NOT gh-dev: that queue is for interactive development turnaround.
#
# NO `set -e`: independent rungs, and a rung that OOMs is data.
#
# Output: runs/v2/g4_gh_memory_bracket.json
#   rsync -av vista:'~/src/inexor/runs/v2/g4_gh_memory_bracket.json' runs/v2/

cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=160
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache

echo "=== provenance ==="
git log --oneline -1
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free -g | head -2
echo "--- LPDDR reference: config table says 116 GB = 108.0 GiB ---"

echo
echo "=== knee bracket: 104/108/112/116/120/128 GiB, host arms only ==="
pixi run -e gpu python scripts/v2_g4_gh_memory.py \
  --ladder 104,108,112,116,120,128 \
  --arms staged,coherent \
  --out-suffix _bracket

echo
echo "=== DONE ==="
