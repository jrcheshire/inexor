#!/bin/bash
#SBATCH -J d7h-hero-ic-card
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d7h-hero-ic-card-%j.log
# D7h: the SAME P(k) card, on the ICs the realization started from.
#
# WHY. gb 1013309 carded the step-40 state and its z profile sits 2-3% below
# linear theory over k = 0.04-0.09 h/Mpc (|z| 2.2, 4.3, 4.1 in three adjacent
# bins) before crossing above by k = 0.11. That z is
# `(P/P_lin - 1) * sqrt(N_modes/2)` -- a Gaussian mode-counting bar against
# FIDUCIAL linear theory -- so it carries no allowance for this realization's
# own IC draw, and the deficit cannot be split between the draw and the forty
# steps from one card. Carding `c-hero-r0` at a = 0.1 through the identical
# estimator splits it: linear evolution preserves mode amplitudes, so a deficit
# already present in the ICs propagates to a = 1 unchanged and says nothing
# about the engine, while one that appears only at a = 1 is the evolution.
#
# WHAT IS NEW. `cmd_card --ic-dir`, and nothing else. `load_checkpoint` refuses
# an IC generation by its manifest's `provenance.kind`, so the card needed
# `cmd_run`'s own IC branch; the card lands as `realization_pk_ics.json` so it
# cannot overwrite the step-40 card it is compared against, and `--ic-dir` is
# refused outright if handed a checkpoint (which would load fine and then be
# scored at a = 0.1). The estimator, the mesh, the binning and the paint are
# untouched -- which the cgh64 legs below MEASURE rather than assert.
#
# NO FINGERPRINT GUARD, deliberately. It gates `load_checkpoint`, and the hero
# leg here does not call it. The IC loader's own crc-per-slab check is the
# stronger contract for this path. What no artifact records is that
# `c-hero-r0` is the generation gb 1009133 evolved: that link is 1009133's run
# banner ("from .../c-hero-r0"), not a field in either manifest. The geometry
# guard below catches a wrong-shaped directory and cannot do better than that.
#
# TIME: every term measured on this node class, on this state.
#   fingerprint-free legs (pytest, cgh64 ICs) ...................   45 s  1013309
#   cgh64 IC card, step 0 -- the new path, end to end ...........   15 s  cgh64 pooled card, 1013309
#   cgh64 run to step 24 + its checkpoint card ..................  190 s  1013309
#   hero: load c-hero-r0 through `load_slot_state` ............... 1842 s  MEASURED, 1009133 step-0 load
#   hero: coarse paint, 262,144 chunks / 40.0 per s .............. 6557 s  1013309's measured average
#   hero: transform, 2 x 2,048 planes ............................   66 s  1013309
#   hero: binning, 2,048 slabs / 13.1 per s ......................  155 s  1013309
#                                                                 ------
#                                                                  8870 s = 2.46 h
#
# THE ONE UNMEASURED TERM is whether the paint's rate decay recurs. In 1013309
# the paint held 50.6 chunks/s for 17 minutes, stepped to ~42, and settled at
# ~37 for its last 60%; ~69 GB of anonymous memory accrued linearly across the
# whole paint, but the rate plateaued while the memory did not, so the accrual
# is not established as the cause and nothing here can predict it. At the
# opening rate the hero leg is 2.03 h; at the settled rate, 2.46 h. The 4 h
# wall (JC) leaves the paint 12,087 s once the other legs and the load are
# paid, so it covers the paint down to 21.7 chunks/s against the 36.3 that was
# the slowest minute of 1013309 -- a 1.7x degradation. This job is also the
# second observation of that decay, on an UNCLUSTERED state with an empty
# arena -- the cheapest discriminator available for it.
#
# MEMORY is not a risk here: the IC state is 68.7e9 rows against the step-40
# state's 83.8e9, so the shared segment is ~620 GB against 754.6, and 1013309
# peaked at 933.8 GB of 1026 with that larger one.

set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export INEXOR_LOAD_TRACE=1
unset JAX_PLATFORMS
unset CUDA_VISIBLE_DEVICES

RUNS=/scratch/10303/jrc4/inexor_runs
ICS=$RUNS/c-hero-r0                  # the generation gb 1009133 evolved
WORK=$RUNS/d7h-$SLURM_JOB_ID
PROD=$RUNS/d7-1009133-product        # beside realization_pk.json, on purpose
CARDS=runs/v2
TAG=d7h_$SLURM_JOB_ID
MEM_CSV=$CARDS/${TAG}_mem.csv
BIND="numactl --membind=0,1"
# every card leg is the CPU lane at both ends -- `cmd_card` calls `_require_cpu`
# and `TilePool` VOIDS a non-CPU parent -- INCLUDING the pytest leg that gates
# them, which is what gb 1011375 got wrong and 5cd464b fixed.
CPU="env JAX_PLATFORMS=cpu"
LAYOUT="--slack 0.10 --arena-frac 0.01"
W="--tile-workers 16"
HB="--heartbeat 60"
DRIVER=scripts/v2_d7_hero_smoke.py
mkdir -p "$WORK" "$PROD"
export PROD ICS

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
free -g | head -2
df -h "$RUNS" | tail -1
numactl --hardware | head -4

( while true; do t=$(date +%s.%N)
    for d in /sys/devices/system/node/node[0-9]*; do
      n=${d##*node}; c=0; [ -n "$(tr -d '[:space:]' < $d/cpulist)" ] && c=1
      awk -v t="$t" -v n="$n" -v c="$c" '/MemTotal:/{tot=$4} /MemFree:/{fr=$4}
        /FilePages:/{fp=$4} / Dirty:/{di=$4} / Writeback:/{wb=$4}
        END{print t "," n "," c "," tot "," fr "," fp "," di "," wb}' $d/meminfo
    done
    sleep 5; done ) > "$MEM_CSV" &
MEM_SAMPLER=$!

rc_total=0
finish () {
  kill $MEM_SAMPLER 2>/dev/null
  echo ""; echo "=== products ==="
  ls -la "$PROD"/realization_pk*.json 2>/dev/null
  df -h "$RUNS" | tail -1
  echo "=== sacct ==="
  sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,ExitCode -P 2>/dev/null
  echo "=== end $(date) rc_total=$rc_total ==="
}
trap finish EXIT

run_leg () {
  local name="$1"; shift
  echo ""; echo "=== LEG $name ($(date +%H:%M:%S)) ==="
  "$@" & local pid=$!
  local rc; wait $pid; rc=$?
  echo "--- LEG $name rc=$rc ($(date +%H:%M:%S)) ---"
  [ $rc -ne 0 ] && rc_total=$((rc_total + 1))
  return $rc
}

# THE GEOMETRY GUARD, before anything expensive. It is the only check that can
# be made between these two artifacts: the checkpoint records no IC source, so
# a same-shape wrong-seed directory would pass here. It catches the realistic
# failure -- a path typo landing on cgh64 or c-gh ICs -- and reports the stream
# and epoch so the readout can be read against sec. 38.
run_leg geometry-guard \
  $CPU pixi run -e gpu python -c '
import json, os, sys
ic = json.load(open(os.environ["ICS"] + "/manifest.json"))
ck = json.load(open(os.environ["PROD"] + "/ckpt/gen1/manifest.json"))
keys = ("n_particles", "box_size", "n_part", "bucket_cells", "bricks_per_side")
bad = [k for k in keys if ic.get(k) != ck.get(k)]
for k in keys:
    print(f"  {k:16s} ics {ic.get(k)!r:>16}  ckpt {ck.get(k)!r:>16}")
print("  ic_stream", ic.get("ic_stream"), "a_init", ic.get("a_init"),
      "order", ic.get("order"), "f_NL", ic.get("f_NL"))
print("  NOTE: seed identity is NOT checkable from these manifests; the link is"
      " gb 1009133s run banner.")
sys.exit(1 if bad else 0)
' \
  || { echo "FATAL: --ic-dir does not match the checkpoint geometry"; exit 1; }

run_leg pytest-cpu-lane \
  $CPU pixi run -e gpu python -m pytest -q -rs tests/test_summary.py tests/test_progress.py \
    tests/test_card_pool.py tests/test_realization_driver.py \
  || { echo "FATAL: the card gates failed"; exit 1; }

# --- cgh64: BOTH LEGS OF THE COMPARISON, in this job, at this commit.
#
# The hero comparison differences two cards from two jobs, and the step-40 one
# is imported from gb 1013309 at 5cd464b. That makes it a constant unless this
# job shows the estimator did not move. So cgh64 runs the whole two-epoch
# comparison at a rung where both halves are affordable: card the ICs at step 0
# (the new path), then run to step 24 and card the checkpoint (the old path).
#
# The step-24 card is the parity leg and it has an exact expected value.
# 1013309 ran the same config to the same step at 5cd464b and printed
# k_nonlinear = 1.3382, |z| median 63.28, max 310.86. Anything else means the
# estimator moved and the imported hero card is not comparable.
run_leg cgh64-ics \
  $BIND pixi run -e gpu python scripts/v2_m6_realization.py ics --config cgh64 \
    --workdir "$WORK/cgh64" --generator device \
  || { echo "FATAL: the cgh64 control failed at the ICs"; exit 1; }

mkdir -p "$WORK/cgh64-prod"

run_leg cgh64-ic-card \
  $CPU $BIND pixi run -e gpu python scripts/v2_m6_realization.py card --config cgh64 \
    --workdir "$WORK/cgh64-prod" --ic-dir "$WORK/cgh64" $LAYOUT $HB $W --card-pool \
  || { echo "FATAL: --ic-dir is broken at cgh64; the hero leg would find it 31 min in"; exit 1; }

# The checkpoint dir is a SIBLING of the IC dir, not a child (`5322c25`).
run_leg cgh64-run \
  $BIND pixi run -e gpu python $DRIVER run --preset cgh64 --workdir "$WORK/cgh64" \
    --card $CARDS/${TAG}_cgh64.json --stop-at 24 --membind-nodes 0,1 --beat 60 \
    --checkpoint-dir "$WORK/cgh64-ckpt" --checkpoint-every 3 \
  || { echo "FATAL: the cgh64 control failed in the run"; exit 1; }

ln -sfn "$WORK/cgh64-ckpt" "$WORK/cgh64-prod/ckpt"

run_leg cgh64-ckpt-card \
  $CPU $BIND pixi run -e gpu python scripts/v2_m6_realization.py card --config cgh64 \
    --workdir "$WORK/cgh64-prod" $LAYOUT $HB $W --card-pool --allow-partial \
  || { echo "FATAL: the checkpoint card path is broken at cgh64"; exit 1; }

# --- THE 4096^3 IC CARD. One call, hours long; the heartbeat is the only thing
# that will say where it got to. Writes realization_pk_ics.json into $PROD,
# beside the step-40 realization_pk.json; $ICS is never written to.
run_leg hero-ic-card \
  $CPU $BIND pixi run -e gpu python scripts/v2_m6_realization.py card --config c-hero \
    --workdir "$PROD" --ic-dir "$ICS" $LAYOUT $HB $W --card-pool
rc_card=$?

echo ""
echo "=== ic card rc=$rc_card ==="
exit $rc_total
