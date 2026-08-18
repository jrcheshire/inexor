#!/bin/bash
#SBATCH -J m6-c12-depth
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m6-c12-depth-%j.log
# C12 = Stage 0 of the idle-half plan: refresh the migrate decomposition on
# CURRENT code, on the PRODUCTION machine. The standing split (5l, antares 478)
# predates the radix sorts and the compiled eject, and it measured the numpy
# path on x86 -- while C11 just showed migrate running on 1-2 cores of an idle
# gg node for ~44% of the W=16 step. This job produces the eject/insert split
# and the insert-internals ranking that order Stages 1-4.
#
# gg, not antares, ON PURPOSE: the ranking's product is a decision about the
# production venue, and bottleneck identity is hardware-specific (the eject
# ratio measured 1.26-1.29x on Grace vs 1.10-1.11x on x86). antares was only
# ever the zero-SU habit.
#
# PRE-REGISTERED (before submission):
#   a. insert dominates the phase with the production kernel:
#      insert_s / (eject_s + insert_s) >= 0.55 at cgh64 (the radix cast sped
#      insert's sorts 1.13x, but eject got the compiled kernel).
#   b. THE RECEIPT: every jax rung carries eject_jax_calls = nb (32 at cgh64)
#      per migrate call. A jax rung reading 0 measured nothing (the phase
#      probe's default-shadowing lesson, now structural).
#   c. the reproduction gate: the synthetic probe lands within 2x of the
#      engine's own gg serial migrate, 24.77 s/step (card
#      m6_phase_time_c9_serial_jax, eject=jax receipt on it). rc=3 = the probe
#      does not describe the engine; the profile leg is then SKIPPED and the
#      hunt moves engine-side.
#   d. reported, not gated: argsort's share of migrate from the profile
#      (ranking only; 5l said ~14% pre-radix).
#
# BUDGET: ~50 min expected against a 2 h request. gg is 0.33 SU/node-hr:
# ~0.3 SU expected, 0.66 worst case.
#
# OPS (gg, all load-bearing): pixi on PATH; -e gpu (the default env is NOT
# installed on Vista); JAX_PLATFORMS=cpu (jax 0.10 hard-raises on cuInit on a
# GPU-less node); CONDA_OVERRIDE_CUDA (env activation gate); -A JPL-SPHEREx
# case-sensitive. NO set -e: per-leg exit codes; rc=3 is the repro gate's own
# refusal, not a crash.

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

echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2

rc_total=0
names=(); codes=()
run_leg() {
  local name="$1"; shift
  echo "=== $name ==="
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "=== $name rc=$rc after $((SECONDS - t0)) s ==="
  names+=("$name"); codes+=("$rc")
  [ $rc -ne 0 ] && [ $rc -ne 3 ] && rc_total=1
  return 0
}

PROBE="scripts/v2_m6_migrate_depth.py"
ENGINE_MIGRATE_S=24.77   # gg cgh64 serial, eject=jax (card m6_phase_time_c9_serial_jax)

# 1. the reader, the gate arithmetic, and the NEW kernel pass-through, at
#    laptop scale on THIS machine and commit (the 447/448/451 class). Both
#    kernels, so a receipt of 0 on the jax arm dies here, not at cgh64.
run_leg "smoke-jax" pixi run -e gpu python $PROBE \
  --config nbs8 nbs16 --brick-slack 0.0 --calls 2 --eject-kernel jax \
  --out runs/v2/m6_migrate_depth_c12_smoke_jax.json
run_leg "smoke-numpy" pixi run -e gpu python $PROBE \
  --config nbs8 --brick-slack 0.0 --calls 2 --eject-kernel numpy \
  --out runs/v2/m6_migrate_depth_c12_smoke_numpy.json
for i in 0 1; do
  if [ "${codes[$i]}" -ne 0 ]; then
    echo "FATAL: smoke leg ${names[$i]} rc=${codes[$i]}; not burning SU on cgh64"
    echo "=== LEG SUMMARY ==="
    for j in "${!names[@]}"; do echo "  ${names[$j]}: rc=${codes[$j]}"; done
    exit 1
  fi
done

# 2. THE REPRODUCTION GATE at cgh64, production kernel. rc=3 skips the profile.
run_leg "repro-gate-jax" pixi run -e gpu python $PROBE \
  --config cgh64 --brick-slack 0.0 --calls 2 --eject-kernel jax \
  --expect-s $ENGINE_MIGRATE_S --expect-within 2.0 \
  --out runs/v2/m6_migrate_depth_c12_repro.json
if [ "${codes[2]}" -eq 3 ]; then
  echo "GATE FAILED: the probe does not reproduce the engine's migrate within 2x."
  echo "SKIPPING the profile -- it would rank the probe, not the engine."
  echo "=== LEG SUMMARY ==="
  for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
  exit 1
fi

# 3. the insert-internals ranking at cgh64 (ranking only, profiler-inflated).
run_leg "profile-jax" pixi run -e gpu python $PROBE \
  --config cgh64 --brick-slack 0.0 --calls 2 --eject-kernel jax --profile 30 \
  --out runs/v2/m6_migrate_depth_c12_profile.json

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST: 5l (antares, numpy path): eject 12.99 / insert 19.68 of 33.26 s;"
echo "         gg engine serial migrate 24.77 s/step (jax, C9)."
echo "READ THE RECEIPT: a jax rung with eject_jax_calls=0 measured nothing."
echo "CARDS WRITTEN BY THIS JOB:"
ls -1 runs/v2/m6_migrate_depth_c12_*.json 2>/dev/null | sed 's/^/  /'
exit $rc_total
