#!/bin/bash
#SBATCH -J inexor-v2-g5b
#SBATCH -w deneb
#SBATCH -c 32
#SBATCH --mem=54G
#SBATCH -t 8:00:00
#SBATCH -o %x-%j.log
#
# v2 seed V2a, gate G5b: the C-dev fine mesh's ABSOLUTE transfer, on a ladder
# that is actually valid. Read the module docstring of v2_g5b_abs_transfer.py.
#
# WHY: job 39's mesh floor refined the MESH at FIXED particles, which drove
# mesh:particle to 4 at its 1024 rung -- four cells inside one interparticle gap
# -- so that rung measures two-body discreteness, not convergence. Its (kc)^2
# check duly failed (1.35 force / 0.76 evolved, expect ~4), and the discrepancy
# GREW under refinement, which mesh truncation error cannot do. So D-v2-1's
# "below the mesh floor" has no bar behind it. This job builds the ladder at
# FIXED mesh:particle = 2 (C-dev's own ratio) -- 128^3/256, 256^3/512, 512^3/1024
# -- where every rung sits in the same discreteness regime.
#
# NO GPU. Every leg is CPU physics; --gres is deliberately absent so this does
# not sit behind a GPU job. It is NOT sent to antares despite its 124 GB: antares
# lacks AVX-512 and the shared env/compilation cache can SIGILL there (umbrella
# memory project_albireo_cluster).
#
# --mem is REQUIRED: the partition default is sized for antares's 124 GB, exceeds
# deneb's 56 GB, and the job pends FOREVER without it (V1, job 9). 54G is set by
# the r512 rung: 1024^3 f32 solve ~26 GB (delta 4.3 + dk 4.3 + kernel 4.3 + three
# g 12.9) + 512^3 particles f32 3.2 GB + the f64 IC state it loads through, ~40
# GB peak. Every rung is f32 BY NECESSITY at 1024 (f64 is ~52 GB against a 56 GB
# host) and f32 EVERYWHERE by choice, so precision is common-mode across the
# ratios; job 39's F2 put f32-vs-f64 at 5.5e-7, six orders under what this
# measures.
#
# WALLTIME: the r512 evolve is ~8x the cells and ~8x the particles of G5's mono
# 512 leg (1.03 s/step), so ~10 s/step x 20 steps ~ 3 min, plus a 512^3 2LPT IC
# gen. Call it 30 min. -t 8:00:00 is deliberately generous per convention: a
# tight wall only kills jobs and forces reruns.
#
# Submit from ~/src/inexor on deneb:  sbatch scripts/v2_g5b_deneb.sbatch

set -e
cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
# The orchestrator's own progress lines are flush=True, but the WORKERS' stdout
# is captured and block-buffered; PYTHONUNBUFFERED closes the "log looks frozen
# for the whole run" gap carried over from the G5 orchestrator.
export PYTHONUNBUFFERED=1

echo "=== env ==="
git log --oneline -1
free -g | head -2
nproc

# Checks first, as their own invocation: matched phase across resolutions is the
# premise of every number below, and a premise that did not run reads exactly
# like one that passed. The script hard-fails on its own, but running the checks
# standalone means a failure costs seconds instead of the whole ladder.
echo "=== checks: degenerate identity + shared modes ==="
pixi run python scripts/v2_g5b_abs_transfer.py --config cdev --checks-only

echo "=== G5b ladder at C-dev (fixed mesh:particle = 2) ==="
pixi run python scripts/v2_g5b_abs_transfer.py --config cdev

echo "=== DONE ==="
