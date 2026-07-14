#!/bin/bash
#SBATCH -J inexor-m2-attrib
#SBATCH -w deneb
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH --exclusive
#SBATCH -o %x-%j.log
# M2 S4: confirm the peak-memory attribution on CUDA.
#
# The local CPU dump (n=32) decomposed the adjoint's peak to 451.3 B/particle --
# against 452-466 measured on two real GPUs (deneb job 19, Vista job 831303), so
# essentially nothing is unaccounted for. The census says the MESHES are nearly
# free (~21 B/particle) and ~288 B/particle is per-particle scalar/1-column
# arrays (~72 live at once) = the CIC corner weights + flat indices, all
# materialized and simultaneously live through the VJP.
#
# But that dump is CPU. XLA fuses differently per backend, and a CPU dump is not
# evidence about a GPU peak -- the agreement in TOTAL is suggestive, not proof
# that the same buffers dominate. This re-runs the attribution on CUDA at two
# sizes.
#
# n=64 and n=128: the quantity is per-particle and K-independent (both measured),
# so small n is sufficient and the census should be size-invariant. Two sizes
# guard against reading an artifact of one.
#
# --exclusive: memory work, no co-scheduled job on the card. --mem=16G here
# (not 8G): the XLA dump is written from host memory and n=128 dumps are chunky.
# REQUIRED regardless -- the partition default 124000M > deneb's 56 GB -> pends.
#
# Submit from ~/spherex/inexor on deneb:  sbatch scripts/m2_mem_attrib.sbatch

set -e
cd ~/spherex/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export PYTEST_XDIST_AUTO_NUM_WORKERS=1  # every GPU sbatch carries the guard (job 16)

echo "=== env ==="
git log --oneline -1
nvidia-smi --query-gpu=name,driver_version --format=csv || true

for N in 64 128; do
  echo
  echo "########## attribution on CUDA: n=${N} ##########"
  # Dump dir is unique per size AND per job: XLA appends to a dump dir, so a
  # stale one would silently mix a previous run's modules into the census. Job-id
  # scoping gets a guaranteed-fresh dir without deleting anything.
  D=runs/m2/xla_dump_gpu_n${N}_${SLURM_JOB_ID}
  pixi run -e gpu python scripts/m2_mem_attrib.py --n ${N} --steps 5 --top 16 --dump-dir "$D"
done

echo "=== DONE ==="
