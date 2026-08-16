#!/bin/bash
#SBATCH -J m6-c6-jit
#SBATCH -p compute
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -t 00:20:00
#SBATCH -o m6-c6-jit-%j.log
# C6 step 2: does `migrate`'s row kernel want a DEVICE, or just a COMPILER?
#
# THE QUESTION, and why it is not already answered. Section 5m's laptop legs
# measured the kernel at 79x the machine's own single-core traffic floor and a
# jitted CPU twin at 11.6x, BITWISE. That says the phase is compute-bound and
# that compiling it is a drop-in. It does not say where the compiled form should
# run, and that choice decides a venue: if the win is the compiler, Grace CPU
# cores are the target and the S3 h100 queue buys nothing for this phase; if the
# win is the device, the 4096^3 production target (JC: 2048^3 is a stepping
# stone) points at the H100 queue for reasons beyond its ~1 TB host.
#
# WHY THIS IS NOT ANSWERED BY VISTA 912457. That A/B measured the GPU at 1.28x
# and is quoted as the accelerator verdict, but its six HOST phases agree across
# arms to 1.4% -- `migrate`, `lead_drift`, `repack` and `tile_decode` were host
# numpy in BOTH arms. It priced device acceleration of the TILE FORCE. This
# phase was never on a device in either arm, so 1.28x is silent about it.
#
# PRE-REGISTERED (mine, falsifiable, written before submission):
#   - the device arm beats the jitted CPU arm by 2-6x at cdev. The kernel is
#     elementwise over ~1M rows with no reduction and no communication, which is
#     the shape a GPU is best at, but the arrays are small (~75 MB in, ~20 MB
#     out) so launch overhead and the H2D copy are a real share.
#   - it stays BITWISE against numpy. If it does not, the finding is the device
#     arm's, not the compiler's -- the CPU twin already came back bitwise, so a
#     divergence here isolates to the backend rather than to jit.
#   - a MISS in either direction is reportable and does not void the run. The
#     only thing that voids it is the engine gate below.
#
# WHAT WOULD VOID IT: the numpy transcription failing its elementwise check
# against the real `_eject_slab` (exit 2), or a jax arm getting a backend it did
# not ask for (exit 2). Both are in the script and both are two-sided.
#
# RESOURCES: --gres=gpu:1 is what routes this to deneb (antares has no GPU);
# --mem=16G covers the ~6 GB cdev build peak with headroom and stays under
# deneb's 56 GB allocatable. The device arrays are ~95 MB against the 3050's
# 6 GB, so the card is not the constraint at cdev. Zero SU. ~5 min expected,
# 20 min requested.
# NO set -e: per-leg exit codes, and the smoke leg guards the long one.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }

# the backend check is a PRECONDITION, not a report: this job's entire purpose
# is a device arm, so a CPU-only jax here means the run has nothing to measure
# and should die in seconds rather than produce a card with one arm skipped.
pixi run -e gpu python -c "
import jax
d = jax.devices('gpu')
assert d, 'no gpu devices'
print('jax', jax.__version__, 'gpu:', d[0])
print('cpu backend also reachable:', jax.devices('cpu')[0])
" || { echo "FATAL: jax gpu import/backend check failed"; exit 1; }

echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

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

PROBE="scripts/v2_m6_c6_jit.py"

# 1. smoke: every reporting path this job uses, on THIS machine and commit,
#    including the device arm. Two cluster jobs (447/448) have died on a line
#    added after the last run; this leg costs ~40 s.
run_leg "c6-smoke" pixi run -e gpu python $PROBE \
  --config smoke --arms numpy_perbrick numpy_batched jax_cpu jax_dev \
  --out runs/v2/m6_c6_jit_smoke_deneb.json
[ "${codes[0]}" -ne 0 ] && {
  echo "FATAL: smoke failed (rc=${codes[0]}); not running the cdev leg"
  echo "=== LEG SUMMARY ==="
  for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
  exit 1
}

# 2. cdev = 4,096 rows/brick, which `choose_brick` fixes at EVERY rung of the
#    config table (5l), so this is the production granularity and not a scale
#    point that needs extrapolating. --decompose also re-reads the eject
#    breakdown on x86 -- the laptop's is arm64, and 5i/5h measured that scaling
#    verdicts do not cross architectures.
run_leg "c6-cdev" pixi run -e gpu python $PROBE \
  --config cdev --arms numpy_perbrick numpy_batched jax_cpu jax_dev --decompose \
  --out runs/v2/m6_c6_jit_cdev_deneb.json

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "AGAINST: laptop (M4, arm64) cdev -- numpy_perbrick 27.96 ns/row at 80x the"
echo "         floor, jax_cpu 11.64x BITWISE, kernel 35.1% of eject,"
echo "         partition 30.2%, reconstruction 102.8%"
echo "CARDS WRITTEN BY THIS JOB:"
ls -1 runs/v2/m6_c6_jit_*_deneb.json 2>/dev/null | sed 's/^/  /'
exit $rc_total
