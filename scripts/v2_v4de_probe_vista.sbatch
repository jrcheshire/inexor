#!/bin/bash
#SBATCH -J inexor-v4de-probe
#SBATCH -o %x-%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 02:00:00
# V4 architecture-freeze probes. Four independent questions, one node. Submit
# from ~/src/inexor:
#   sbatch scripts/v2_v4de_probe_vista.sbatch
#
# NONE OF THIS IS A GATE. No bar is tested and nothing here is ratified. Every
# leg exists so that a clause of the V4 freeze rests on a measurement instead of
# on arithmetic. The pre-registered reading for each is written BELOW, before
# any leg ran, and each one can come back the other way.
#
# Wall is deliberately generous (billed for actual use). NO `set -e`: the top
# rungs of legs 5 and 6 are expected to OOM and that OOM is itself the result.
#
# ---------------------------------------------------------------------------
# LEGS 1-2: what does the tile padding COST?
#
# Found while planning V4 and fixed in eba91ab: force_short_tiled filled its
# padded rows with np.zeros, so every one of them indexed particle 0, and
# _tile_corner then rewrote every masked row's index to flat 0 regardless. At
# this geometry pad_frac is 0.329 -- ~2e6 dead rows per tile, each hitting 8 CIC
# corners -- so each of the 8 unrolled scatter-adds became ~2e6 f64 atomics
# contending for ONE address. GPU scatter-add is atomic; same-address atomics
# serialize; XLA cannot elide them because the zero weight arrives through a
# data-dependent where().
#
# Forces are bitwise identical under both arms (proved locally against HEAD at a
# clustered 64^3 geometry, and the one-tile identity vs force_global holds at
# 4.29e-16), so the ONLY difference between legs 1 and 2 is contention cost.
#
#   PRE-REGISTERED: `zero` reproduces V4a's 47.8 ms device median at this
#   geometry (job 895315). `cycle` is FASTER. If the two agree to within ~10%,
#   the hot spot is NOT the mechanism, the device phase is genuinely
#   paint/gather-bound, and the V4 record's "far above its FFT floor" reading
#   stands as written -- which makes the Pallas/ffi kernel lane load-bearing
#   rather than optional.
#
# ---------------------------------------------------------------------------
# LEGS 3-4: isolate `cap`, which the pricing record could not.
#
# runs/v2/v4_pricing_record.md names `cap` as THE cost variable but says so from
# a correlation: n_brick co-varied 32 -> 64 in every fixed-box step, so `cap`
# and the brick decomposition always moved together, and the record records the
# isolation as owed ("a pair at fixed n_brick").
#
# --cap-mult is a strictly cleaner instrument than that pair. It inflates the
# padded capacity with members, bricks, tiles, geometry AND kernel all held
# fixed, so the ONLY thing that changes is the number of padded rows. Nothing
# co-varies at all.
#
#   PRE-REGISTERED: if `cap` is the cost variable, device and stage times scale
#   ~linearly in cap_mult. If they are flat, `cap` is NOT the driver, the
#   correlation was the brick decomposition all along, and "design against cap"
#   (v4_pricing_record.md section 7) is wrong and must be retracted.
#   NB legs 3-4 are INSTRUMENTS, never operating points: cap_mult buys nothing
#   and costs padding.
#
# ---------------------------------------------------------------------------
# LEG 5: does a TILE-shaped gather reach the sequential streaming rate?
#
# D-v2-13 clause 1 measured 359-367 GB/s on a 2 GiB SEQUENTIAL stream. A tile
# reads ~1000 brick spans of ~37 KB (T9) to ~98 KB (f64). The entire plan to
# kill the 74% host plumbing assumes the short runs get close to the long-run
# rate, and nothing measured says they do.
#   PRE-REGISTERED: `assemble` (gather to one buffer, ONE device_put) is roughly
#   flat above ~64 KB and within ~3x of 359 GB/s; `per_run` falls off sharply.
#   FALSIFIED IF `assemble` at 37 KB is below ~40 GB/s, i.e. under ~10% of
#   sequential -- then the redesign wins ~10x on `stage`, not ~20x, and the
#   C-gh cost model needs redoing.
#   (A laptop CPU dry run already shows assemble 26 GB/s vs per_run 2.0 GB/s at
#   37 KB. That is the SHAPE, on the wrong hardware; this leg is the number.)
#
# ---------------------------------------------------------------------------
# LEG 6: what does IC generation cost on the HOST?
#
# The only IC number on record is 76 B/p, and it is device-only. ic.py:49-58
# evaluates the transfer function on the FULL 3D rfft half-grid in float64, and
# transfer_eh98 materializes ~15 more arrays of that size. At 2048^3 one is
# 34.4 GB against a ~116 GB host cliff.
#   PRE-REGISTERED: host B/p grows with N and dominates device B/p at every
#   rung. FALSIFIED IF host B/p is flat and small (< ~20) -- then the IC problem
#   is device-only and V4's IC ADR shrinks to the (N^3,3) pile-up alone.
#   This leg is only meaningful ON GPU: on CPU jax's own arrays live in host RAM
#   and the host mark stops isolating the numpy term. The probe reports
#   host_isolates_numpy so a CPU run cannot be misread as the answer.
#
# LEG 7: the largest monolithic rfftn that fits. If 2048^3 fits, C-gh needs no
# out-of-core FFT at all and that whole layer defers to C-hero.
# ---------------------------------------------------------------------------

cd ~/src/inexor || exit 1

export XLA_PYTHON_CLIENT_PREALLOCATE=false
# NOT the 64 GB default: that default silently voided job 894010 (D-v2-13
# provenance note), and legs 5-7 are precisely the ones that would hit it.
export XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=160
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache

echo "=== provenance ==="
git log --oneline -1
hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
free -g | head -2
echo "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=$XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB"

COMMON="--single --kind force --family gauss --which mono --alpha 1.0 --fdtype f64 --a 1.0 --profile 1"
GEOM="--config cgh64 --tile 256 --buf 32"

echo
echo "=== leg 1: padding A/B -- zero fill (the pre-eba91ab contention) ==="
pixi run -e gpu python scripts/v2_g5_two_level_force.py $COMMON $GEOM \
  --tag v4d_padfill_zero --pad-fill zero

echo
echo "=== leg 2: padding A/B -- cycle fill (the shipped default) ==="
pixi run -e gpu python scripts/v2_g5_two_level_force.py $COMMON $GEOM \
  --tag v4d_padfill_cycle --pad-fill cycle

echo
echo "=== leg 3: cap isolation -- cap x1.5, nothing else moves ==="
pixi run -e gpu python scripts/v2_g5_two_level_force.py $COMMON $GEOM \
  --tag v4d_capmult_1p5 --pad-fill cycle --cap-mult 1.5

echo
echo "=== leg 4: cap isolation -- cap x2.0, nothing else moves ==="
pixi run -e gpu python scripts/v2_g5_two_level_force.py $COMMON $GEOM \
  --tag v4d_capmult_2p0 --pad-fill cycle --cap-mult 2.0

echo
echo "=== leg 5: stream run-length sweep (tile-shaped gather vs sequential) ==="
pixi run -e gpu python scripts/v2_v4e_stream_runlength.py \
  --src-gib 16 --stage-mib 64 --tag v4e_gh

echo
echo "=== leg 6: IC memory anatomy, host AND device, cheapest rung first ==="
pixi run -e gpu python scripts/v2_v4d_ic_memory.py \
  --mode ic --rungs 256,512,1024 --tag v4d_gh

echo
echo "=== leg 7: largest monolithic rfftn that fits on this card ==="
pixi run -e gpu python scripts/v2_v4d_ic_memory.py \
  --mode fft --rungs 512,1024,1536,2048 --tag v4d_gh

echo
echo "=== DONE ==="
