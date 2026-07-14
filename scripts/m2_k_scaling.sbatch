#!/bin/bash
#SBATCH -J inexor-m2-kscale
#SBATCH -w deneb
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -t 3:00:00
#SBATCH --exclusive
#SBATCH -o %x-%j.log
# M2 S4: does adjoint peak memory scale with K? The decisive follow-up to Vista
# job 831303's R6 result.
#
# 831303 measured the adjoint at 58.3 GiB at 512^3 against a 4.5 GiB carry --
# 13x, near-constant across 64^3..512^3, projecting the 1024^3 flagship to
# ~466 GiB against an 80 GB claim (~6x over). 466 B/particle at K=10 is
# suspiciously close to 10 x ~45 B/particle, but 831303 fixed K=10 throughout
# and so CANNOT distinguish:
#   flat in K   -> the residual really is O(1); the 13x is a fat per-step
#                  transient. Bad, but a tuning/lifetime problem.
#   linear in K -> per-step state is being kept alive, so reversibility-as-
#                  checkpointing is not delivering step-independent memory.
#                  That is architecture Sec. 8's central premise, and the reason
#                  the whole design exists. An architectural failure.
# Nothing about the paper's memory claim should be concluded before this runs.
#
# deneb, not Vista: 128^3 at K=40 peaks ~3.6 GiB even if peak IS linear in K, so
# it fits the 3050's 6 GB, and deneb is free. 64^3 is carried as a cross-check
# that the K trend is not an artifact of one size. 256^3 is NOT swept: at K=40 a
# linear peak would be ~30 GiB and OOM -- that would map a ceiling, not answer
# the question.
#
# --exclusive: peak-memory measurement, so no co-scheduled job may share the GPU
# (per JC, exclusive is right for benchmarks AND memory-ceiling jobs). --mem is
# still REQUIRED (partition default 124000M > deneb's 56 GB -> pends forever).
#
# Submit from ~/spherex/inexor on deneb:  sbatch scripts/m2_k_scaling.sbatch

set -e
cd ~/spherex/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# No pytest here, but keep the guard: every GPU sbatch carries it (job 16).
export PYTEST_XDIST_AUTO_NUM_WORKERS=1

echo "=== env ==="
git log --oneline -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

echo "=== K sweep at fixed n (the O(1)-in-steps test) ==="
pixi run -e gpu python scripts/m2_mem_profile.py --n 64,128 --k 2,5,10,20,40

echo "=== DONE ==="
