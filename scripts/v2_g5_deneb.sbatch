#!/bin/bash
#SBATCH -J inexor-v2-g5
#SBATCH -w deneb
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 24:00:00
# --mem is REQUIRED (see v2_g1_deneb.sbatch): the partition default is sized for
# antares's 128 GB, exceeds deneb's 56 GB, and the job pends FOREVER without it.
# 48G is set by the CONVERGED 1024^3 mesh-floor leg, ~28 GB in f32 (delta 4.3 +
# dk 4.3 + kernel 4.3 + three g 12.9 + inv_k2 2.2). That leg is f32 BY NECESSITY:
# 1024^3 in f64 is ~52 GB against a 56 GB host and will not fit. Safe, because F2
# puts f32-vs-f64 at ~3e-7 -- six orders under the ~1e-1 floor it establishes --
# and the error is common-mode across the pair. Everything else is smaller: the
# 512^3 f64 solve is ~7-8 GB (the ik_j are low-rank (N,1,1) broadcasts and cost
# nothing) and the orchestrator's f64 512^3 statistics ~6 GB. G2c's 24G covered
# statistics only.
#
# WALLTIME: measured, not guessed. Leg costs timed at 128^3/256^3 WITH the JIT
# compile excluded (the naive timing was compile-dominated and gave an
# unphysical 1.2x for an 8x volume step) scale as N^3 log N -- mono 7.4x, tiled
# 4.8x, paint 8.7x per 8x volume. C-dev extrapolates to ~7 min on an M4 Max, so
# call it 15-30 min on deneb, plus the 1024 leg. -t 24:00:00 is deliberately
# generous per convention: we are billed for use, so a tight wall only kills
# jobs and forces reruns.
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
# --skip-1024 here: cdev8's converged floor would be 512^3, which the C-dev leg
# above already measures. This leg exists ONLY for the invariance claim.
pixi run -e gpu python scripts/v2_g5_two_level_force.py --config cdev8 --pivot-only --skip-1024

echo "=== DONE ==="
