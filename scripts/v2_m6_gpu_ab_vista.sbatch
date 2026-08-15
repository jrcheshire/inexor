#!/bin/bash
#SBATCH -J m6-gpu-ab
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m6-gpu-ab-%j.log
# Does the accelerator earn its place? ONE node, GPU on against GPU off.
#
# Everything previously offered on this question was confounded. The 2.2x that
# made deneb look like it beat a GH200 is x86 against Grace with a GPU on BOTH
# sides. The "3-4% utilization" and "~74% of per-tile cost is host plumbing"
# figures predate the M-v2-6 fixes. And the gg/gh pair on record (897358/897377)
# ran at commit 32b22f2 and drove the PROBE force, not `engine.run`.
#
# A single node with the accelerator disabled removes all of it: same CPU, same
# memory, same clock, same code, same configuration. The H100 is the only
# variable.
#
# **AND THE READING I AM CHECKING IS MY OWN, FROM EARLIER TODAY.** I argued the
# accelerator's ceiling was ~1.14x from a "host plumbing is ~88% of a tile"
# figure that predated the migration fix. Job 463 measured the device-eligible
# phases at **85.6% of the anchor step**, which at the recorded 6.8x device
# speedup implies ~3.7x, not 1.14x. The reasoning was fine and the input was
# stale. This is the measurement that settles which number was right.
#
# PHASE TIMING, NOT A STOPWATCH, and that is an improvement on the original
# design: `v2_m6_phase_time.py` reports per phase on BOTH backends, so instead
# of one ratio this says WHICH phases the accelerator converts and whether the
# 85.6% turns into anything. It carries its own neutrality control (the same run
# with `phase=None`) and refuses to report a breakdown it moved.
#
# PRE-REGISTERED, before submission:
#   - the GPU arm beats the CPU arm on total wall by **2.5-4.5x**. Below 1.5x my
#     1.14x reading was closer to right than the correction and the CPU-only
#     case is back. Above ~5x the 6.8x device figure is understated and the
#     device-eligible fraction is doing more than 85.6% of the work.
#   - `tile_long` and `tile_short` are where the difference is. They are the JAX
#     phases; `tile_decode`, `migrate` and `repack` are host numpy and should
#     read WITHIN NOISE across the two arms. **If the host phases move too, the
#     arms differ in something other than the accelerator and the run is void.**
#   - the CPU arm reproduces job 463's cdev phase SHAPE (tile_long ~49%,
#     tile_short ~23%, coarse_paint ~14%) modulo Grace-vs-x86 per-core speed. A
#     different shape means the node or the environment differs from deneb in a
#     way that is not the CPU.
#
# NOT MEASURED HERE, deliberately: compiling the sub-block gather. It cannot be
# jitted as it stands -- it carries a bounds guard that reads a concrete value
# ("a row's tsc stencil reaches outside the staged sub-block ... reading past
# the block wraps to the far side of the mesh and is silent"), so relaxing
# D-v2-21 means either dropping a D-007-class check on a silent-failure path or
# rewriting the guard into a compilable form. That is a design question, not a
# switch, and the ~1.7x on record is therefore partly the price of a SAFETY
# CHECK rather than purely of bitwise agreement.
#
# Config is cdev and that is not a budget compromise: cdev, cgh64 and C-gh share
# N/tile = 2,097,152 and padded tile side P = 320, so a per-tile timing
# transfers directly. What does NOT transfer is the phase MIX -- at C-gh the
# coarse paint and the migration carry different weight -- so the ARM RATIO is
# the transferable number here, not the wall.
#
# ~40 min expected against a 2 h request; TACC bills actual, so ~0.7 SU, with
# 2 SU the worst case if it times out. One gh node.
# NO set -e: independent arms, per-arm exit codes -- except smoke, which gates.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"

rc_total=0
names=(); codes=()
run_leg() {
  local name="$1"; shift
  echo "=== $name ==="
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "=== $name rc=$rc after $((SECONDS - t0)) s ==="
  names+=("$name"); codes+=("$rc")
  [ $rc -ne 0 ] && rc_total=1
  return 0
}

PHASE="scripts/v2_m6_phase_time.py"

# BACKEND ASSERTED, NOT REQUESTED, in each arm. A knob that did not apply is how
# an A/B reads as "no difference"; here it would read as "the GPU does not help"
# and be believed. Demonstrated per arm before any timing runs.
run_leg "smoke-gpu" bash -c "
  pixi run -e gpu python -c \"
import jax
d = jax.devices()
print('devices', d)
assert d[0].platform == 'gpu', f'GPU ARM IS NOT ON A GPU: {d}'
\" &&
  pixi run -e gpu python $PHASE --config smoke --k 5 --repeats 2 \
    --slack 0.20 --arena-frac 0.20 --out-suffix _gpuab_smoke_gpu"
[ "${codes[0]}" -ne 0 ] && { echo "FATAL: GPU smoke failed"; exit 1; }

run_leg "smoke-cpu" bash -c "
  JAX_PLATFORMS=cpu pixi run -e gpu python -c \"
import jax
d = jax.devices()
print('devices', d)
assert d[0].platform == 'cpu', f'CPU ARM IS NOT ON THE CPU: {d}'
\" &&
  JAX_PLATFORMS=cpu pixi run -e gpu python $PHASE --config smoke --k 5 --repeats 2 \
    --slack 0.20 --arena-frac 0.20 --out-suffix _gpuab_smoke_cpu"
[ "${codes[1]}" -ne 0 ] && { echo "FATAL: CPU smoke failed"; exit 1; }

# ---- the two arms, same everything but the accelerator
run_leg "cdev-gpu" pixi run -e gpu python $PHASE \
  --config cdev --k 3 --repeats 2 --slack 0.20 --arena-frac 0.20 \
  --out-suffix _gpuab_cdev_gpu

run_leg "cdev-cpu" env JAX_PLATFORMS=cpu pixi run -e gpu python $PHASE \
  --config cdev --k 3 --repeats 2 --slack 0.20 --arena-frac 0.20 \
  --out-suffix _gpuab_cdev_cpu

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST job 463 (deneb, CPU): cdev 43.02 s/step, tile_long 48.8%, tile_short 22.9%"
echo "PRE-REGISTERED: GPU arm 2.5-4.5x on wall; host phases within noise or VOID"
echo "CARDS:"
ls -1 runs/v2/m6_phase_time_gpuab_*.json 2>/dev/null || echo "  (none)"
exit $rc_total
