#!/bin/bash
#SBATCH -J m6-c7-ab
#SBATCH -p compute
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 03:00:00
#SBATCH -o m6-c7-ab-%j.log
# C7: do the two `migrate` changes survive at cgh64?
#
# 5o (compiled eject, 1.30x) and 5p (radix-path sort, 1.13x) stack to 1.52x on
# the phase, and BOTH numbers are cdev legs from one laptop. This milestone has
# been bitten twice by exactly that: Stage 2c measured ~2% at cdev and 20.2% at
# cgh64 (5k), and `numpy_batched` INVERTED between arm64 and x86 (5n). Until
# this job runs, 1.52x is a hypothesis about C-gh.
#
# PRE-REGISTERED (written before submission, in the script's docstring too):
#   1. stacked speedup at cgh64 lands 1.3-1.8x on the phase
#   2. every arm is BITWISE identical -- a requirement, not a prediction; a
#      mismatch voids the CHANGE rather than the run
#   3. `radix` alone lands 1.05-1.25x
# A miss on 1 or 3 is reportable. A miss on 2 stops the promotion.
#
# WHY ANTARES: --mem=64G is ROUTING (>56G keeps it off deneb, which has 56G
# allocatable) plus headroom over the ~25 GB build peak at 512^3 (the same
# sizing job 478 used). No GPU is wanted here -- 5n already measured the device
# arm losing on deneb's PCIe, and this is a CPU question. Zero SU.
#
# THE CDEV LEG IS NOT REDUNDANT: both recorded numbers are arm64 laptop
# measurements, and 5n's cross-architecture pair is precisely why an x86 repeat
# of the same config is worth 3 minutes before the 64x leg is believed.
# NO set -e: per-leg exit codes, and the smoke leg guards the rest.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
pixi run -e default python -c "
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
  [ $rc -ne 0 ] && rc_total=1
  return 0
}

PROBE="scripts/v2_m6_c7_migrate_ab.py"

# 1. every reporting path this job uses, on THIS machine and commit. ~10 s.
#    Jobs 447/448 died on a line added after the last run; this is the guard.
run_leg "c7-smoke" pixi run -e default python $PROBE \
  --config smoke --repeats 2 --out runs/v2/m6_c7_smoke.json
[ "${codes[0]}" -ne 0 ] && {
  echo "FATAL: smoke failed (rc=${codes[0]}); not running the long legs"
  echo "=== LEG SUMMARY ==="
  for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
  exit 1
}

# 2. cdev on x86: does the laptop's 1.52x reproduce on a different architecture
#    at the SAME config, before the config axis moves?
run_leg "c7-cdev" pixi run -e default python $PROBE \
  --config cdev --repeats 3 --out runs/v2/m6_c7_cdev.json

# 3. THE QUESTION: 64x the particles, clustered arena occupancy, same arms.
run_leg "c7-cgh64" pixi run -e default python $PROBE \
  --config cgh64 --repeats 2 --out runs/v2/m6_c7_cgh64.json

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST (laptop, arm64, cdev): baseline 2.965 s / radix 2.619 (1.13x) /"
echo "        compiled 1.948 (1.52x), all bitwise"
echo "NB rc=2 from a leg means a BITWISE requirement failed, not a crash."
echo "CARDS WRITTEN BY THIS JOB:"
ls -1 runs/v2/m6_c7_*.json 2>/dev/null | sed 's/^/  /'
exit $rc_total
