#!/bin/bash
#SBATCH -J inexor-m1-cgh64-gh
#SBATCH -o %x-%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 04:00:00
#
# M-v2-1 exit gate at cgh64: the layout at 64x the volume it was designed on.
# Submit from ~/src/inexor on Vista:  sbatch scripts/v2_m1_cgh64_vista.sbatch
#
# THE GPU HALF OF A MATCHED PAIR with the `gg` job (897358). Same commit, same
# config, same seed, same slack -- ONLY the force backend differs, so the
# comparison is controlled rather than inferred.
#
# WHY BOTH. The layout answers (peak bucket population, all-in B/p, arena
# fraction) depend only on the TRAJECTORY and should agree across the pair --
# which makes them a free cross-backend consistency check. The OVERHEAD number
# does not, and that is the point: the `gg` job measures the host-side layout
# against a CPU force, and production runs the force on DEVICE. Overhead
# measured against a CPU force flatters the layout by roughly two orders of
# magnitude on the denominator, and would be the wrong baseline in a record.
#
# So this job carries the meaningful overhead figure and `gg` carries the
# fallback; whichever assumption breaks, one of the two still reports.
#
# The trajectories will not be bitwise identical -- CPU and CUDA reductions
# order differently -- so the layout statistics should agree CLOSELY, not
# exactly. A large disagreement would mean something other than roundoff.
#
# Memory: ICs at n_part=512 f64 fit a GH200 (measured, V4d job 896159: `full`
# OOM'd at n=1024 and was fine at 512), and cgh64's tiled force ran on gh
# throughout the V4 campaign.
#
# WHAT THIS ANSWERS
#
#   1. THE uint16 CEILING. The per-bucket occupancy index is uint16 because
#      that is what makes it 0.25 B/p; a wider index is 0.5 and moves D-v2-19
#      clause 5's all-in figure. Peak bucket population measured 5943 at cdev8
#      -- 11.0x under 65535 -- and larger volumes hold rarer, denser peaks.
#      cdev8 / cdev / cgh64 is a 1x / 8x / 64x ladder at FIXED cell and spacing,
#      so the three points are a trend rather than three unrelated numbers.
#   2. WHETHER THE LAYOUT NUMBERS HOLD at 64x volume: all-in B/p, peak arena,
#      brick-level migration, and the repack's cost and scratch.
#
# WHAT IT DOES NOT ANSWER, stated so the record cannot be read as claiming it:
# STREAMING. cgh64's whole T9 state is ~1.2 GB and fits in memory everywhere.
# Streaming only becomes real at C-gh proper (2048^3, 77 GB state), which is
# M-v2-6's business, not this gate's.
#
# A PREDICTION, PRE-REGISTERED so it is not explained after the fact: the layout
# overhead will read WORSE here than the 6.0% measured at cdev8, and that will
# not mean the layout got slower. That figure was taken at 4 CPU threads. The
# force is multithreaded and speeds up on 72 cores; the layout is single-threaded
# numpy and does not. Expect 25-40%. If it instead comes back near 6%, my model
# of where the time goes is wrong and that is worth knowing too.
#
# THREADS: 72, one Grace socket, following G6c. A gg node is two NUMA domains
# and 144 is untested on a bandwidth-bound f64 FFT -- deliberately not changed
# here, since this job is measuring the layout and not the thread scaling.

set -u
export PATH="$HOME/.pixi/bin:$PATH"
export OMP_NUM_THREADS=72

echo "=== $(date -Is)  host $(hostname)  job ${SLURM_JOB_ID:-none} ==="
cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
echo "commit: $(git log --oneline -1)"
echo "branch: $(git branch --show-current)"

# ---- preconditions that ABORT ---------------------------------------------
# Jobs 896055 and 896092 ran ZERO science because pixi was not on PATH, and
# 896055 reported COMPLETED 0:0 in sacct because set -e had been dropped. A
# green Slurm row that ran nothing is worse than a failure. These two checks
# exist so that cannot happen again.
command -v pixi >/dev/null 2>&1 || { echo "FATAL: pixi not on PATH"; exit 1; }
# Print the DEVICE, not just the version. The gh/gg pair exists to compare a
# device force against a CPU one, and job 897377 could only INFER that the GPU
# was live from a 6.83x speedup -- the log recorded the jax version and nothing
# about the backend. A benchmark has to demonstrate its knob is applied.
pixi run -e gpu python -c "
import jax, inexor.layout
d = jax.devices()
print('jax', jax.__version__, 'devices:', d)
assert d[0].platform != 'cpu', f'FATAL: expected a GPU backend, got {d[0].platform}'
" || { echo "FATAL: env cannot import jax + inexor.layout, or is not on a GPU backend"; exit 1; }
free -g | head -2

# ---- the legs --------------------------------------------------------------
declare -a NAMES=() CODES=()
run_leg () {   # run_leg <name> <args...>
  local name="$1"; shift
  echo ""; echo "--- LEG $name  $(date -Is) ---"
  pixi run -e gpu python -u scripts/v2_m1_migration.py "$@"
  local rc=$?
  NAMES+=("$name"); CODES+=("$rc")
  echo "--- LEG $name exit $rc ---"
}

# The operating point: D-v2-19's ratified configuration.
run_leg cgh64 --config cgh64 --steps 20 --slack 0.10 --repack-every 1 \
              --arena-frac 0.02 --out-suffix _cgh64_gh

# ---- summary ---------------------------------------------------------------
echo ""; echo "=== LEG SUMMARY ==="
ok=0
for i in "${!NAMES[@]}"; do
  echo "  ${NAMES[$i]}: exit ${CODES[$i]}"
  [ "${CODES[$i]}" -eq 0 ] && ok=$((ok+1))
done
echo ""; echo "=== CARDS WRITTEN ==="
ls -la runs/v2/m1_migration_*cgh64*.json 2>/dev/null || echo "  (none)"
echo ""; echo "=== $(date -Is)  legs ok: $ok of ${#NAMES[@]} ==="
[ "$ok" -gt 0 ] || { echo "FATAL: zero legs succeeded"; exit 1; }
