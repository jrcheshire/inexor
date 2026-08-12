#!/bin/bash
#SBATCH -J m6-trace
#SBATCH -p compute
#SBATCH --mem=100G
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o m6-trace-%j.log
# M-v2-6 Stage 0b: WHERE the engine's peak is, by phase, with the noise floor
# measured in the same job. Replaces Stage 0's differencing of whole-run maxima,
# which job 445 showed cannot work at cdev: five repeats of one leg scatter by
# sigma 45-115 MB while the terms the gates separated are 67.6 MB (gate 1's
# predicted signal, 0.6 sigma) and 178.7 MB (the isolating arm, 1.6 sigma), and
# the isolating arm read 132 MB HIGHER with 64x less coarse mesh.
#
# --mem=100G is PLACEMENT expressed as a resource: only antares (28 c, 124 GB)
# satisfies it; deneb-the-node has 56 GB. `-c 8` is honest about what the step
# uses -- job 433 measured s/step FLAT across 1..32 threads (0.6% spread), so the
# step is effectively serial and more cores buy nothing.
#
# ONE glibc machine, deliberately. The `trim` arm measures glibc's retained-free
# heap, which macOS does not have in the same form, and the same cdev8 K=20 point
# read 6.484 and 9.855 GB on the laptop minutes apart.
#
# CPU env by construction (`default` is CPU-only jax). The probe REFUSES a
# non-CPU backend and refuses a non-Linux platform: VmHWM and clear_refs are
# procfs, and there is no honest fallback.
# NO set -e, deliberately: independent legs, cheapest first, per-leg exit codes.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
pixi run -e default python -c "
import jax, platform, sys
d = jax.devices()
print('jax', jax.__version__, d, platform.machine(), sys.platform)
assert d[0].platform == 'cpu', 'expected the CPU backend in the default env'
assert sys.platform == 'linux', 'the phase tracer needs procfs'
" || { echo "FATAL: jax import/backend/platform check failed"; exit 1; }

# The high-water reset the whole instrument rests on. If clear_refs is not
# writable (container, hardened kernel, ancient kernel) every phase would report
# the run's cumulative maximum instead of its own and the numbers would look
# plausible -- so it is checked HERE, once, loudly, before any science runs.
pixi run -e default python -c "
def kb(f):
    t = open('/proc/self/status').read()
    return int([l for l in t.splitlines() if l.startswith(f)][0].split()[1])
def reset():
    open('/proc/self/clear_refs','w').write('5\n')
big = bytearray(200 * 1024 * 1024); big[::4096] = b'x' * (len(big) // 4096)
peak = kb('VmHWM:')
del big
reset()
hwm, rss = kb('VmHWM:'), kb('VmRSS:')
print(f'VmHWM {peak/1024:.0f} MB before reset; after reset VmHWM {hwm/1024:.0f} '
      f'MB vs VmRSS {rss/1024:.0f} MB')
# THE defining property, and it does not depend on the allocator returning the
# pages: after a reset the high-water mark restarts from the CURRENT resident
# set. If it does not, every phase would report the run's cumulative maximum and
# the table would look plausible while saying nothing.
assert abs(hwm - rss) <= 4096, 'clear_refs did NOT restart VmHWM; the tracer would be blind'
assert peak > rss, 'VmHWM never rose above the resting set; the counter is not live'
" || { echo "FATAL: /proc/self/clear_refs does not reset VmHWM here"; exit 1; }

export TMPDIR="${TMPDIR:-/tmp}/m6_trace_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD) ==="
free -g | head -2
STAMP="$TMPDIR/.stamp"; touch "$STAMP"

rc_total=0
names=(); codes=()
run_leg() {
  local name="$1"; shift
  echo "=== $name ==="
  "$@"; local rc=$?
  echo "=== $name rc=$rc ==="
  names+=("$name"); codes+=("$rc")
  [ $rc -ne 0 ] && rc_total=1
  return 0
}

TRACE="scripts/v2_m6_peak_trace.py"

# ---- SMOKE FIRST, and it is not ceremony: this orchestrator has never run end
# to end anywhere. Stage 0's dropped a leg silently twice (`repack_only`, then
# `integrity`) and each time the absence read as a clean run. Seconds, and it
# exercises every path the cdev leg will take.
#
# `--arena-frac 0.20`, and the default 0.08 is why job 446's smoke leg died:
# 29 particles overflowed their brick against an arena of 2622 slots with 9
# free (`_to_arena`, the D-007 refusal working as designed -- the layout does
# not clamp or drop). smoke has 32,768 particles, so 8% is 2622 slots and a
# single clustered brick exhausts it; the same 8% at cdev is 1.3e6 slots and
# job 445 ran there fine. NB the window has an upper edge too: at 0.30/0.30
# the repack refuses for the opposite reason (42,835 slots plus a 9,831-slot
# arena against the allocation), measured on the laptop across six (K, arena,
# slack) points -- the same six confirmed the phase hook changes NO outcome.
run_leg "smoke" pixi run -e default python $TRACE \
  --configs smoke --k 5 --repeats 3 --arms trace control trim \
  --arena-frac 0.20 --workdir-root "$TMPDIR" --out-suffix _trace_smoke

# ---- cdev8: the config the Stage 0 peak instrument was developed on, so the
# phase table can be read against numbers already on record.
run_leg "trace-cdev8" pixi run -e default python $TRACE \
  --configs cdev8 --k 5 --repeats 5 --arms trace control trim \
  --workdir-root "$TMPDIR" --out-suffix _trace_cdev8

# ---- cdev: THE ANCHOR. Where the 4.27 GB (254.5 B/p) unattributed residual was
# measured and where `pending` is 537 MB against cdev8's 67 MB, so the terms are
# separable here and not there. Five repeats because the gates are written
# against a sigma and job 445 is the reason.
run_leg "trace-cdev" pixi run -e default python $TRACE \
  --configs cdev --k 5 --repeats 5 --arms trace control trim \
  --workdir-root "$TMPDIR" --out-suffix _trace_cdev

# ---- K=15 at cdev, trace arm only. The K-ladder fits 0.157 GB PER STEP + 6.200
# fixed, so something accumulates; `coarse_delta_streamed` takes a new `pad` --
# hence a new XLA shape -- every step (10 distinct over 10 at smoke, against 1
# for `cap`). Tripling K triples the churn, so if that is the accumulating term
# the per-visit growth in `coarse_paint` grows with it and no other phase's does.
run_leg "trace-cdev-k15" pixi run -e default python $TRACE \
  --configs cdev --k 15 --repeats 3 --arms trace \
  --workdir-root "$TMPDIR" --out-suffix _trace_cdev_k15

echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "CARDS WRITTEN BY THIS JOB:"
n=0
for f in runs/v2/m6_peak_trace_smoke.json runs/v2/m6_peak_trace_cdev8.json \
         runs/v2/m6_peak_trace_cdev.json runs/v2/m6_peak_trace_cdev_k15.json; do
  if [ -f "$f" ] && [ "$f" -nt "$STAMP" ]; then echo "  $f"; n=$((n + 1)); fi
done
[ "$n" -eq 0 ] && { echo "  (none)"; echo "FATAL: zero fresh cards"; exit 1; }
echo "($n fresh cards)"
echo "TMPDIR left for inspection: $TMPDIR"
exit $rc_total
