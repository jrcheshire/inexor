#!/bin/bash
#SBATCH -J inexor-p1
#SBATCH -w deneb
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 3:00:00
#SBATCH --exclusive
#SBATCH -o %x-%j.log
# P1 Stage 1 (plan glistening-greeting-possum): the comparative memory curve.
#
# THE QUESTION. S4 measured inexor's adjoint at 466 B/particle where Sec. 9
# budgeted ~68, so the 1024^3 flagship projects ~6x over its 80 GB claim. Claim 2
# is COMPARATIVE, and BOTH competitors are already O(1) in steps (architecture
# Sec. 2's own table; DISCO-DJ's live adjoint is a custom_vjp backsolve whose fwd
# residual is the final state only). So the differentiator is state WIDTH (12 bpp
# int vs f32 float) + bit-exactness -- and a 12-24 B/particle state advantage
# against a ~430 B/particle transient EVERY PM adjoint pays is a 3-5% edge.
# Therefore the decisive question is not "do we win" but: is the ~430 B/particle
# transient OURS or INHERENT? DISCO-DJ is the control that answers it.
#
#   DISCO-DJ ~= inexor -> inherent; the state advantage stays swamped; claim 2
#                         needs restating, not optimizing.
#   DISCO-DJ << inexor -> our fat is ours; the bisect has a named target and a
#                         mature code has proved it reachable.
#
# BOTH ARMS IN ONE JOB, ON ONE CARD. A cross-machine memory comparison is not a
# comparison. inexor's arm is also the harness control: it must reproduce job
# 19's 452.6 (64^3) / 488.2 (128^3) B/particle. If it does not, the harness is
# wrong and the DISCO-DJ number means nothing.
#
# --exclusive: peak-memory measurement; no co-scheduled job may share the card.
# --mem=16G: host-side; REQUIRED regardless (partition default 124000M is
# antares-sized > deneb's 56 GB -> pends forever).
#
# Submit from ~/spherex/inexor on deneb:  sbatch scripts/m2_p1_deneb.sbatch

set -e
cd ~/spherex/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PYTEST_XDIST_AUTO_NUM_WORKERS=1  # every GPU sbatch carries the guard (job 16)

echo "=== env ==="
git log --oneline -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

echo
echo "########## arm 1: inexor (harness control -- must reproduce job 19) ##########"
pixi run -e gpu python scripts/m2_mem_profile.py --n 64,128

echo
echo "########## arm 2: DISCO-DJ, f32 (the fair comparison) ##########"
# Cross-env, m1_run_disco.py pattern: the script is OURS, the interpreter is
# disco-mocks'. -e gpu is required -- their `default` is CPU JAX and a CPU
# backend reports no memory_stats, so it would silently measure nothing.
pixi run --manifest-path ~/src/disco-mocks/pixi.toml -e gpu \
  python scripts/m2_p1_disco_mem.py --n 64,128 --steps 10

echo
echo "########## arm 3: DISCO-DJ, f64 (secondary row) ##########"
# Their harness default is x64. Reported because it is the regime the M1 parity
# work used, NOT because it is the fair memory comparison: f64 state is 48
# B/particle against inexor's 12, which would flatter us for free.
pixi run --manifest-path ~/src/disco-mocks/pixi.toml -e gpu \
  python scripts/m2_p1_disco_mem.py --n 64,128 --steps 10 --f64

echo "=== DONE ==="
