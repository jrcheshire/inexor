#!/bin/bash
#SBATCH -J inexor-m2-detcost
#SBATCH -w deneb
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH -t 2:00:00
#SBATCH --exclusive
#SBATCH -o %x-%j.log
# M2 S4: (a) validate option B -- do the detflag-marked bit-equality tests pass
# on CUDA under XLA deterministic ops? -- and (b) price the flag for JC's
# production call (option C).
#
# --exclusive is deliberate: (b) is a BENCHMARK, and a co-scheduled job sharing
# the GPU would corrupt the timings. Per JC, exclusive is right for benchmarks
# and memory-ceiling jobs. No -c: --exclusive hands over the whole node (all 64
# cores) and overrides it anyway, so stating -c 4 only misleads. --mem is still
# REQUIRED (partition default is antares-sized, 124000M > deneb's 56 GB -> pends
# forever).
#
# Submit from ~/spherex/inexor on deneb:  sbatch scripts/m2_detflag_cost.sbatch

set -e
cd ~/spherex/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# MANDATORY on any single-GPU node. The `test`/`test-det` pixi tasks use
# `pytest -n auto`, which is right on a laptop/CI but catastrophic here: `auto`
# resolves to the CPU count (64 under --exclusive) and spawns that many xdist
# workers, each initializing JAX and preallocating MEM_FRACTION of the SAME
# card. Job 16 died exactly this way -- one worker took 5050 MiB of the 3050's
# 6144 MiB, the rest got 526/114/86 MiB, the GPU sat at 0% while they thrashed,
# and the tests "failed" from OOM in a way that reads exactly like a real
# nondeterminism failure. xdist honours this variable to resolve `auto`
# (verified: `created: 1/1 worker`). GPU tests want ONE process; the parallelism
# that helps CPU-JAX only fights over one device here.
export PYTEST_XDIST_AUTO_NUM_WORKERS=1

echo "=== env ==="
git log --oneline -1
nvidia-smi --query-gpu=name,driver_version --format=csv || true

echo
echo "############ (a) option B: do the detflag tests pass under the flag? ############"
echo "--- first: plain \`test\` should SKIP them on GPU (visibly, not silently) ---"
pixi run -e gpu test -- -m detflag -rs || true
echo "--- then: \`test-det\` sets XLA_FLAGS via the task env and runs them ---"
pixi run -e gpu test-det

echo
echo "############ (b) option C: price the flag ############"
pixi run -e gpu python scripts/m2_detflag_cost.py --n 64,128,256
XLA_FLAGS="--xla_gpu_deterministic_ops=true" \
  pixi run -e gpu python scripts/m2_detflag_cost.py --n 64,128,256
pixi run -e gpu python scripts/m2_detflag_cost.py --compare

echo "=== DONE ==="
