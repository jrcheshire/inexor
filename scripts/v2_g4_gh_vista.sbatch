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
#
# THE QUESTION: C-gh is 2048^3 particles. At the ratified T9 state tier
# (D-v2-8, 9 B/p) that is 77.3 GB of compressed state alone = 81% of the
# GH200's 96 GB HBM before a single mesh exists. So C-gh is real only if
# host-resident state is reachable at useful bandwidth. Three arms:
# hbm (control, EXPECTED to die above the cliff), staged (pinned_host +
# explicit device_put, the documented path), coherent (host-resident input,
# jitted op with its output declared on device -- XLA owns the transfer).
#
# WHAT job 894005 ALREADY SETTLED (2026-08-06, GH200, jax 0.10.2), so this
# job does not re-ask it:
#   - memory kinds are ['device', 'pinned_host'] ONLY. There is no
#     `unpinned_host` on this CUDA stack, so NOTHING here can test ATS/
#     coherent access to ordinary pageable LPDDR. The `coherent` arm tests
#     IMPLICIT (XLA-managed) vs EXPLICIT staging out of page-locked host
#     memory. Do not write "coherent path works" on the strength of it.
#   - the instrument reproduces both known ceilings: hbm 3021-3359 GB/s
#     against E2's 3400, staged 364-367 against C2C read's 375. That is what
#     makes the unknown arm's number readable.
#
# LEG 1 IS A GATE, NOT A FORMALITY. Job 894005 found two defects that would
# each have silently ruined this run:
#   (a) the coherent arm did not compile at all (E1200
#       CompileTimeHostOffloadOutputLocationMismatch) -- it measured nothing;
#   (b) the staged arm held EVERY staged chunk on device at once (pk/set read
#       exactly 1.000), so above the cliff it would OOM for reasons having
#       nothing to do with the memory path.
# Both are fixed, but a fix is a claim until it runs here. Leg 1 re-runs the
# small ladder and ABORTS the job if the coherent arm still errors, rather
# than spending hours on legs whose verdict column would be empty.
#
# THE TRAP the ladder is shaped around: the coherent arm can silently BE the
# staged arm if XLA inserts a wholesale copy. A rate alone cannot separate
# them. So the ladder straddles the cliff at 64/88/104/128 GiB -- an arm that
# COMPLETES 128 GiB cannot have held it all in 96 GB of HBM. That capacity
# witness is binary and needs no bandwidth model; HBM residency
# (peak_bytes_in_use over working set) and the rate corroborate it.
#
# CRITICAL: XLA_PYTHON_CLIENT_PREALLOCATE=false, NOT the
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 the other GPU sbatch files set. A
# preallocated pool moves the OOM cliff and makes peak_bytes_in_use report
# the POOL rather than the working set, so all three arms would read as
# HBM-resident no matter where the data actually lived.
#
# PARTITION: -p gh = Grace-Hopper (one GH200, 96 GB HBM3 + ~116 GB LPDDR on a
# 72-core Grace host; job 894005 measured the node at 212 GB total / 190
# free). NOT -p gg -- no GPU there, and the gpu env's [system-requirements]
# cuda = "12" is enforced at ACTIVATION, which is how job 865844 died.
# NOT -p gh-dev: that queue exists for interactive development turnaround and
# is not somewhere to park a batch script, however open it looks.
# ACCOUNT is case-sensitive: JPL-SPHEREx.
#
# WALL: 12 ladder workers (4 sizes x 3 arms), each dominated by ALLOCATION
# rather than the reduction -- filling 128 GiB across the link is the slow
# part, the reduce itself is sub-second at any plausible rate. Then 3 real
# C-gh paint workers, where generating 103 GB of f32 positions host-side is
# the long pole. Estimate 2-4 h; 8 h is deliberately generous because TACC
# bills actual use and a tight wall only risks killing the ladder mid-flight.
#
# NO `set -e`, deliberately: legs are independent and ordered cheapest first,
# so a failure in the real point must not discard the ladder. The hbm arm
# ABOVE the cliff is EXPECTED to OOM -- a designed output that brackets where
# the cliff sits, not a broken job. (Leg 1's gate is an explicit `exit`, which
# is why it does not need set -e to work.)
#
# Output (runs/ is gitignored; pull it back explicitly):
#   runs/v2/g4_gh_memory_smoke.json   (leg 1)
#   runs/v2/g4_gh_memory.json         (legs 2-3)  + runs/v2/g4_errs/*.err
# Pull with:
#   rsync -av vista:'~/src/inexor/runs/v2/g4_gh_memory*.json' runs/v2/

cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# BOTH of these are load-bearing, and job 894010 was VOIDED for want of them.
# Its ladder measured JAX's defaults rather than the GH200: the host arms hit
# XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB's default of 64 GB on a node with 192 GB
# free (staged just fit at the 64 GiB rung; coherent failed the same rung
# needing 128.5 KiB MORE), and the device arm OOM'd at a 68 GiB peak against
# 95.6 GiB of HBM, ~0.75 of the card. Neither ceiling was hardware.
# The probe now reads bytes_limit back off the device and VOIDS the capacity
# witness if the caps do not clear the ladder top -- setting these is a claim,
# the readback is the evidence.
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=160
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
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
echo "=== LEG 1 (gate): small ladder under the cliff ==="
pixi run -e gpu python scripts/v2_g4_gh_memory.py --smoke --out-suffix _smoke

echo
echo "=== LEG 1 gate check ==="
# Abort before the expensive legs if the coherent arm still cannot run. A
# ladder whose verdict arm errors at every rung answers nothing, and there is
# no point paying for it.
pixi run -e gpu python -c "
import json, sys
c = json.load(open('runs/v2/g4_gh_memory_smoke.json'))['configs']
coh = [r for r in c if r['arm'] == 'coherent']
ok = [r for r in coh if r.get('completed')]
print('coherent arm: %d/%d rungs completed' % (len(ok), len(coh)))
for r in coh:
    if not r.get('completed'):
        print('  FAILED %.1f GiB: %s' % (r.get('working_set_gib', 0), str(r.get('error'))[:160]))
if not ok:
    print('GATE FAIL: the coherent arm ran nowhere -- the long legs would')
    print('measure nothing. Fix the arm, then resubmit.')
    sys.exit(1)
print('GATE PASS')
"
if [ $? -ne 0 ]; then
  echo "=== ABORTING before the expensive legs (leg 1 gate failed) ==="
  exit 1
fi

echo
echo "=== LEG 2+3: full ladder (64/88/104/128 GiB) + the C-gh paint point ==="
pixi run -e gpu python scripts/v2_g4_gh_memory.py --ladder 64,88,104,128 --real

echo
echo "=== DONE ==="
