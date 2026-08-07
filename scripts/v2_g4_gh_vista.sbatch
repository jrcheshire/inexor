#!/bin/bash
#SBATCH -J inexor-v2-g4
#SBATCH -o %x-%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 08:00:00
# G4 verdict job (v2 seed V3). Submit from ~/src/inexor on Vista:
#   sbatch scripts/v2_g4_gh_vista.sbatch
# RUN THE SMOKE FIRST (scripts/v2_g4_smoke_ghdev_vista.sbatch). If that job
# reports no host memory kind, this one measures nothing -- do not queue it.
#
# THE QUESTION: C-gh is 2048^3 particles. At the ratified T9 state tier
# (D-v2-8, 9 B/p) that is 77.3 GB of compressed state alone = 81% of the
# GH200's 96 GB HBM before a single mesh exists. So C-gh is real only if the
# ~116 GB of Grace LPDDR is reachable at useful bandwidth. Three arms:
# hbm (control, EXPECTED to die above the cliff), staged (pinned_host +
# explicit device_put, the documented path), coherent (host-resident arrays
# handed straight to jnp, the undocumented path under test).
#
# THE TRAP, and why the ladder crosses 96 GiB rather than sitting under it:
# the coherent arm can silently BE the staged arm if XLA inserts a copy. A
# rate alone cannot tell those apart. So the ladder runs 64/88/104/128 GiB --
# straddling the cliff deliberately -- and an arm that COMPLETES a 128 GiB
# working set cannot have held it all in 96 GB of HBM. That capacity witness
# is binary and needs no bandwidth model; HBM residency (peak_bytes_in_use
# over working set) and the rate against the E2 ceilings corroborate it.
#
# CRITICAL: XLA_PYTHON_CLIENT_PREALLOCATE=false, NOT the
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 that every other GPU sbatch here sets.
# A preallocated pool moves the OOM cliff to wherever the pool was sized and
# makes peak_bytes_in_use report the POOL instead of the working set, which
# would make all three arms read as HBM-resident no matter where the data
# lived. On-demand allocation is what gives the cliff and the residency ratio
# their meaning.
#
# PARTITION: -p gh = Grace-Hopper (one GH200, 96 GB HBM3 + ~116 GB LPDDR on a
# 72-core Grace host). NOT -p gg -- no GPU there, and the gpu env's
# [system-requirements] cuda = "12" is enforced at ACTIVATION, which is how
# job 865844 died. ACCOUNT is case-sensitive: JPL-SPHEREx.
#
# WALL: 12 ladder workers (4 sizes x 3 arms), each dominated by ALLOCATION
# rather than by the reduction -- filling 128 GiB across the C2C link is the
# slow part, the streaming reduce itself is well under a second at any
# plausible rate. Then 3 real C-gh paint workers, where generating 103 GB of
# f32 positions host-side is again the long pole. Estimate 2-4 h; 8 h is
# deliberately generous because TACC bills actual use and a tight wall only
# risks killing the ladder mid-flight (the standing lesson from every Vista
# job in this project).
#
# NO `set -e`, deliberately: the legs are independent and ordered cheapest
# first, so a failure in the real point must not discard the ladder. The hbm
# arm ABOVE the cliff is EXPECTED to fail with an OOM -- that is a designed
# output that brackets where the cliff sits, not a broken job.
#
# Output (runs/ is gitignored; pull it back explicitly):
#   runs/v2/g4_gh_memory.json   + runs/v2/g4_errs/*.err for any dead worker
# Pull with:
#   rsync -av vista:'~/src/inexor/runs/v2/g4_gh_memory.json' runs/v2/

cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache

echo "=== provenance ==="
git log --oneline -1
git status -s | head
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free -g | head -2

echo
echo "=== capability dump ==="
pixi run -e gpu python -c "
import jax
d = jax.devices()[0]
print('jax', jax.__version__, '|', d.platform, d.device_kind)
print('memory kinds:', [m.kind for m in d.addressable_memories()])
"

echo
echo "=== G4 ladder (64/88/104/128 GiB, straddling the 96 GB HBM cliff) + C-gh paint ==="
pixi run -e gpu python scripts/v2_g4_gh_memory.py --ladder 64,88,104,128 --real

echo
echo "=== DONE ==="
