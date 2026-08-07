#!/bin/bash
#SBATCH -J inexor-v2-g4-smoke
#SBATCH -o %x-%j.log
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 02:00:00
# G4 reachability smoke. Submit from ~/src/inexor on Vista:
#   sbatch scripts/v2_g4_smoke_ghdev_vista.sbatch
#
# WHY A SMOKE EXISTS: G4's whole question is whether JAX can put state in
# Grace LPDDR and still compute on it. That is gated on something binary and
# cheap to check -- does this jax/jaxlib even EXPOSE a host memory kind on
# aarch64+CUDA, and does an op on a host-resident array run rather than raise?
# If the answer is no, the long `gh` job measures nothing and must not be
# queued. So this runs UNDER the HBM cliff (4 and 8 GiB) and asks only
# "expressible, and does it stream", never "how fast".
#
# PARTITION: -p gh-dev = the GH200 development queue. qdevelopment caps wall
# at 02:00:00 and one running job per user, so 2 h is the cap and not a
# guess; the smoke itself is minutes. NOT -p gg (Grace-Grace, no GPU: the
# gpu env's [system-requirements] cuda = "12" is enforced at ACTIVATION and
# would fail there the way job 865844 did). ACCOUNT is case-sensitive.
#
# NO `set -e`, deliberately (M1 job 11 lesson: independent legs, cheapest
# first -- the capability dump must still print if the ladder dies).
#
# CRITICAL, and the reason this file does not copy the G5c preamble:
# XLA_PYTHON_CLIENT_PREALLOCATE=false. Every other GPU sbatch here sets
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95, which grabs 95% of HBM up front. That
# would destroy every witness this probe depends on -- the OOM cliff moves to
# wherever the pool was sized, and `peak_bytes_in_use` reports the POOL rather
# than the working set, so the residency witness would read ~1.0 for all three
# arms regardless of where the data actually lived. On-demand allocation is
# what makes the cliff and the residency ratio mean anything.

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
echo "=== capability dump: which memory kinds does this stack expose? ==="
# NECESSARY BUT NOWHERE NEAR SUFFICIENT, and measured to be so: CPU JAX on the
# laptop also reports ['device', 'pinned_host', 'unpinned_host'], where host
# memory is a vacuous concept. So the kind list appearing here proves nothing
# about the coherent path -- the smoke's real deliverable is the ladder below,
# where the arms actually run on CUDA. Read this dump only for what it rules
# OUT: no host kind at all would mean the coherent arm is inexpressible.
pixi run -e gpu python -c "
import jax
d = jax.devices()[0]
print('jax', jax.__version__, '|', d.platform, d.device_kind)
try:
    mems = [(m.kind) for m in d.addressable_memories()]
except Exception as e:
    mems = 'addressable_memories() raised: %r' % (e,)
print('memory kinds:', mems)
print('memory_stats keys:', sorted((d.memory_stats() or {}).keys()))
"

echo
echo "=== G4 smoke ladder (4 + 8 GiB, all three arms, under the HBM cliff) ==="
pixi run -e gpu python scripts/v2_g4_gh_memory.py --smoke

echo
echo "=== DONE ==="
