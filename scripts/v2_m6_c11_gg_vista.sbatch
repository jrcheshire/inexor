#!/bin/bash
#SBATCH -J m6-c11-hw
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m6-c11-hw-%j.log
# C11: what the HARDWARE is doing during a cgh64 step -- utilization, not phases.
#
# Every wall number so far says where the engine's seconds go; none says what
# the node did during them. This job produces, per phase at W=16: busy cores
# (of 144, per socket), a MEASURED per-socket bandwidth-contention signal (a
# pinned canary core's triad throughput vs its own in-job idle baseline),
# iowait + disk/Lustre bytes, and PSI stalls -- plus the node's own ceilings
# (STREAM-style triad at 1/16/72/144 cores, cross-NUMA, and scratch disk rates)
# measured in the same job so every fraction divides two same-node numbers.
#
# PRE-REGISTERED (before submission; a miss on (a) redirects the wall story and
# is the point of the job):
#   a. tile-loop window at W=16: at least one socket's canary loses >=20% vs
#      baseline (the 5j busy-inflation read as bandwidth contention predicts
#      it). migrate window: canary contention <10% AND busy cores <=4 -- the
#      node is IDLE there (serial row work), not saturated.
#   b. disk + Lustre traffic during instrumented stepping < 100 MB total:
#      nothing reads disk once ICs exist (a code-reading claim, measured here
#      for the first time).
#   c. the instrument is neutral at cgh64: canary + callback inside the
#      two-term bound (2 sigma or 2% of wall). rc=2 = non-neutral, NOT a crash.
#   d. membind0 arm within 10% of the default arm on step wall (JC: doubts
#      night-and-day; this prices it). RECEIPT: card `mems_allowed` reads "0"
#      there and "0-1" in the default arm -- a knob must prove it applied.
# NB in the membind0 arm the canaries inherit the binding, so the socket-1
# canary reads REMOTE memory -- its baseline is in-arm, so contention stays
# internally consistent, but do not compare its absolute GB/s across arms.
#
# BUDGET: ~1.2 h expected against a 3 h request (a TIMEOUT bills the whole
# wall). gg is 0.33 SU/node-hr: ~0.4 SU expected, 1.0 SU worst case.
#
# OPS (all load-bearing on gg): pixi on PATH; JAX_PLATFORMS=cpu (jax 0.10
# hard-raises on cuInit on a GPU-less node); CONDA_OVERRIDE_CUDA (the gpu env
# enforces cuda=12 at activation); -A JPL-SPHEREx case-sensitive.
# NO set -e: per-leg exit codes. rc=2 instrument non-neutral; rc=4 instrument
# BROKEN (no ticks / no canary windows) -- rc=4 on smoke is fatal, the long
# legs would measure nothing.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
export JAX_PLATFORMS=cpu
export CONDA_OVERRIDE_CUDA=12.0
pixi run -e gpu python -c "
import jax
assert jax.devices()[0].platform == 'cpu', 'expected the CPU backend'
print('jax', jax.__version__)
" || { echo "FATAL: jax import/backend check failed"; exit 1; }

HAVE_NUMACTL=1
command -v numactl >/dev/null || { HAVE_NUMACTL=0; echo "NOTE: no numactl; NUMA legs VOID"; }

echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2
NODE0=$(cat /sys/devices/system/node/node0/cpulist 2>/dev/null)
NODE1=$(cat /sys/devices/system/node/node1/cpulist 2>/dev/null)
echo "node0 cpus: $NODE0 ; node1 cpus: $NODE1"

rc_total=0
names=(); codes=()
run_leg() {
  local name="$1"; shift
  echo "=== $name ==="
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "=== $name rc=$rc after $((SECONDS - t0)) s ==="
  names+=("$name"); codes+=("$rc")
  [ $rc -ne 0 ] && [ $rc -ne 2 ] && rc_total=1
  return 0
}

HW="scripts/v2_m6_c11_hw_util.py"
# engine knobs held EXACTLY at 914085/C9's so the phase walls are comparable
BIG="--config cgh64 --k 3 --repeats 2 --slack 0.20 --arena-frac 0.20"

# 1. smoke: the whole instrument end to end (sampler + canaries + join) on the
#    small config. Gates the long legs; rc=2 tolerated (smoke exercises paths,
#    it does not measure), rc=4 or a crash is fatal.
run_leg "smoke" pixi run -e gpu python $HW run --config cdev8 --k 3 --repeats 1 \
  --tile-workers 4 --arena-frac 0.20 --out-suffix _c11_smoke
if [ "${codes[0]}" -ne 0 ] && [ "${codes[0]}" -ne 2 ]; then
  echo "FATAL: smoke rc=${codes[0]}; not burning SU on cgh64"
  echo "=== LEG SUMMARY ==="
  for j in "${!names[@]}"; do echo "  ${names[$j]}: rc=${codes[$j]}"; done
  exit 1
fi

# 2. ceilings: triad at 1 / 16 (the worker count) / 72 (one socket) / 144
#    cores, and the cross-NUMA rate. ~15 s each.
run_leg "stream-1core"   pixi run -e gpu python $HW stream --workers 1 \
  --cpus "$NODE0" --out runs/v2/m6_c11_stream_1.json
run_leg "stream-16core"  pixi run -e gpu python $HW stream --workers 16 \
  --cpus "$NODE0" --out runs/v2/m6_c11_stream_16.json
run_leg "stream-socket0" pixi run -e gpu python $HW stream --workers 72 \
  --cpus "$NODE0" --out runs/v2/m6_c11_stream_72.json
run_leg "stream-144"     pixi run -e gpu python $HW stream --workers 144 \
  --out runs/v2/m6_c11_stream_144.json
if [ "$HAVE_NUMACTL" -eq 1 ]; then
  run_leg "stream-cross-numa" numactl --cpunodebind=1 --membind=0 \
    pixi run -e gpu python $HW stream --workers 72 \
    --out runs/v2/m6_c11_stream_cross.json
fi

# 3. scratch disk rates (secondary; the fadvise caveat is on the card)
run_leg "disk" pixi run -e gpu python $HW disk --dir "$SCRATCH" --gb 4 \
  --out runs/v2/m6_c11_disk.json

# 4. THE POINT: cgh64 at W=16, instrumented, on the production candidate.
run_leg "cgh64-w16" pixi run -e gpu python $HW run $BIG --tile-workers 16 \
  --out-suffix _c11_gg

# 5. the NUMA arm: identical, state + everything bound to socket 0's memory.
if [ "$HAVE_NUMACTL" -eq 1 ]; then
  run_leg "cgh64-w16-membind0" numactl --membind=0 \
    pixi run -e gpu python $HW run $BIG --tile-workers 16 \
    --out-suffix _c11_gg_membind0
fi

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "rc=2 is instrument non-neutrality, rc=4 is a broken instrument (no data)."
echo "READ THE RECEIPTS: mems_allowed on the run cards; canary n_windows > 0."
echo "CARDS WRITTEN BY THIS JOB:"
ls -1 runs/v2/m6_c11_*.json 2>/dev/null | sed 's/^/  /'
exit $rc_total
