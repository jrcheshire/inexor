#!/bin/bash
#SBATCH -J inexor-v4e-runlength
#SBATCH -o %x-%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 01:00:00
# V4e round 2, standalone: the run-length leg only. Submit from ~/src/inexor:
#   sbatch scripts/v2_v4e_runlength_vista.sbatch
#
# WHY A SEPARATE JOB. Job 896159 ran all seven probe legs and six of them
# produced good cards. Only leg 5 was defective -- it device_put from PAGEABLE
# numpy memory, so its H2D flat-lined at ~19.6 GB/s at every run length and
# could not test D-v2-13 clause 3's pinned_host path at all. Re-running the
# other six would burn SUs and overwrite results that stand.
#
# WHAT CHANGED: gather and H2D are now timed separately, and the H2D is timed
# out of a pinned_host array staged OUTSIDE the timer -- the exact `staged`
# step G4 measured. Round 2's prediction is in the probe docstring: h2d_pinned
# lands within ~2x of 359-367 GB/s and is FLAT in run length, so the design's
# ceiling at a 37 KB brick span is gather-dominated (~11-12 GB/s) and the
# binding constraint is the HOST MEMCPY rather than the transfer. Falsified if
# h2d_pinned is also ~20 GB/s (pinning is not the explanation, something else
# caps this node), or if gather comes back far from round 1's 12.5 GB/s at
# 37 KB (round 1's valid half was not valid either).
#
# NOT a gate. No bar is tested and nothing here is ratified.

set -e
cd ~/src/inexor

# Vista's batch environment does not carry pixi on PATH (jobs 896055/896092).
export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null 2>&1 || {
  echo "FATAL: pixi not on PATH (looked in \$HOME/.pixi/bin)"
  exit 1
}

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=160
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache

echo "=== provenance ==="
git log --oneline -1
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
pixi run -e gpu python -c "import jax; print('jax', jax.__version__, jax.devices())"

echo
echo "=== V4e round 2: gather vs PINNED H2D, by run length ==="
pixi run -e gpu python scripts/v2_v4e_stream_runlength.py \
  --src-gib 16 --stage-mib 64 --tag v4e_gh_pinned

echo
echo "=== card ==="
ls -la runs/v2/v4e_stream_runlength_v4e_gh_pinned.json

echo
echo "=== DONE ==="
