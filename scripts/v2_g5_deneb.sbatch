#!/bin/bash
#SBATCH -J inexor-v2-g5
#SBATCH -w deneb
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
# --mem is REQUIRED (see v2_g1_deneb.sbatch): the partition default is sized for
# antares's 128 GB, exceeds deneb's 56 GB, and the job pends FOREVER without it.
# 32G here because G5's physics legs are CPU f64 and one of them SOLVES a 512^3
# f64 force (~7-8 GB: delta 1.07 + dk 1.08 + three g 3.2 + inv_k2 1.08; the ik_j
# are low-rank (N,1,1) broadcasts and cost nothing), on top of the orchestrator's
# own f64 512^3 painted-force statistics (~6 GB). G2c's 24G covered statistics
# only; G5 adds a full f64 solve in a worker.
#SBATCH -o %x-%j.log
#
# v2 seed V2a, gate G5: two-level (coarse long + fine tile short) PM force vs
# monolithic at C-dev. Floors + the F1/tile-identity kaiser check first, then the
# coarse arm, the tiled arm, the EVOLUTION arm (dP/P -- what D-v2-1 actually
# bars), and the GPU capacity triple.
#
# READ FIRST: runs/v2/g5_kernel_findings.md. The kernel study is DONE and it
# inverted two design assumptions -- the Gaussian family tiles as ~2.0/P (the
# BUFFER IS NOT A KNOB, the tile size is), and kernel matching is FAMILY-SPECIFIC
# (mandatory for gauss, harmful for the windowed families).
#
# ENV: gpu (conda-forge CUDA). G5 is plain XLA -- no Pallas -- so gpu-pallas
# (V1/G1 only, pypi jax[cuda12]) is NOT wanted here.
#
# PLATFORM SPLIT, on purpose: every PHYSICS leg runs CPU f64 so f32/GPU noise can
# never contaminate attribution; the GPU is touched ONLY by the capacity legs, in
# their own subprocesses. The monolithic f32 GPU leg is EXPECTED to OOM the 6 GB
# card at C-dev -- that IS the capacity result ("monolithic does not fit; tiled
# does"), not a failure.
#
# Submit from ~/src/inexor on deneb:  sbatch scripts/v2_g5_deneb.sbatch

set -e
cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OMP_NUM_THREADS=1
# NB the CPU physics workers get OMP_NUM_THREADS=8 from spawn()'s child env: a
# single-threaded 512^3 f64 FFT is ~4 min of pure wall, and those legs' walls are
# not reported measurements, so threading them contaminates nothing.

echo "=== env ==="
git log --oneline -1
pixi run -e gpu python -c "import jax; print(jax.__version__, jax.devices())"
# this gate's headline is a MEMORY claim, so the card and host sizes belong in
# the log, not in someone's memory of what deneb has
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free -g | head -2

echo "=== G5 at C-dev (validation -> floors -> coarse -> tiles -> evolve -> capacity) ==="
pixi run -e gpu python scripts/v2_g5_two_level_force.py --config cdev

# The O(tile) claim is tested by INVARIANCE, not by one number: the same
# (tile, buf) at 1/8 the volume must leave the tiled peak FLAT while mono's
# scales with the box. cdev8 preserves fine cell AND spacing, so every scan
# point -- all expressed in CELLS -- transfers unchanged.
echo "=== capacity scaling: same tile, 1/8 the volume (the O(tile) claim) ==="
pixi run -e gpu python scripts/v2_g5_two_level_force.py --config cdev8 --pivot-only

echo "=== DONE ==="
