#!/bin/bash
#SBATCH -J d7e-hero-product
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o d7e-hero-product-%j.log
# D7e: turn gb 1009133's step-40 checkpoint into the two PRODUCTS -- the P(k)
# card and the portable (x, v) export. 1009133 produced a checkpoint, which is
# the engine's own T9 encoding and readable by nothing outside this package.
#
# Neither leg has ever run at 4096^3. Both are CPU-ONLY (`_require_cpu`), so the
# whole 687 GB state sits on the host and the four GB200s idle; that is what the
# script does today, not a choice this job makes.
#
# TIME: measured terms x loop counts.
#   fingerprint guard + GPU pytest .............................   33 s  measured, 1010730
#   cgh64 ICs ..................................................   23 s  measured, 1010730
#   cgh64 run to step 24 (step 0 10.6, step 1 14.2 compiling,
#     then ~4.1 s/step and 2.2 s per checkpoint) ...............  140 s  measured, 1010730
#   cgh64 card + export, same NFS ..............................  120 s  card measured, export carry
#   hero CARD leg:
#     load the 687 GB state ..................................... 1842 s  measured, 1009133
#     decode 16,777,216 bricks @ 133-149 us/brick ............... 2366 s  MEASURED (laptop)
#     sub-block coarse paint into the 2048^3 int accumulator ....  600 s  carry
#     2048^3 out-of-core forward FFT + binned_power .............  400 s  carry
#   hero EXPORT leg:
#     load again (separate subcommand; see below) ............... 1842 s  measured
#     max(decode 2366, write 1.65 TB @ 706-807 MB/s = 2199) .....	2400 s  decode measured
#   CENTRAL 2.8 h.
#   The three carries are the risk and none of them is padded away:
#     - the host sub-block paint is the least constrained; at 3x, +20 min.
#     - the write rate came from 256 npz files of 2.7 GB, and this writes two
#       sequential 825 GB streams; at half, the export leg is +34 min.
#     - the FFT/binning carry comes from the 417 s/step host coarse solve.
#   Worst plausible 3.8 h. REQUESTED 5:00 against gb's 12 h MaxWall. ~5 SU.
#   Measured early, not padded: the cgh64 legs below run BOTH products end to
#   end onto the same NFS by ~second 320, so a broken product path shows up in
#   five minutes rather than two hours.
#
# THE LAYOUT PARAMETERS ARE NOT THE SCRIPT'S DEFAULTS AND MUST NOT BE.
# `load_checkpoint` re-derives the allocation geometry from the occupancy plus
# `--slack` / `--arena-frac` (the checkpoint deliberately preserves no geometry,
# D-v2-19). 1009133 ran slack 0.10 / arena-frac 0.01 -> 83.85e9 slots, 687 GB.
# v2_m6_realization.py DEFAULTS to 0.20 / 0.20, which on the same state provisions
# ~13.7e9 arena rows instead of 687M and lands near 900 GB before either product
# allocates anything -- the card then wants 34.4 GB of int accumulator and 34.4 GB
# of spectrum on top. With the run's own parameters the card leg peaks ~756 GB
# (0.74x) and the export leg ~688 GB (0.67x).
#
# HOST MEMORY IS BOUND TO THE CPU SOCKETS, as in every D7 job since 1003511.
#
# PRE-REGISTERED QUESTIONS:
#   1. What does the export cost in parts -- decode against write thread -- and
#      does the overlap hold on Grace? The writer thread is new code (7762c89).
#   2. What is NFS's write rate for two sequential 825 GB streams, against the
#      706-807 MB/s the checkpoint's 256 x 2.7 GB files measured?
#   3. Does the host sub-block paint at 2048^3 cost anything like the 600 s carry?
#   4. What does the card say below k_nonlinear? NO verdict is emitted: the band
#      this run is trusted over is JC's call, not the script's.
#
# Submit from the Vista checkout:
#   sbatch scripts/v2_d7e_hero_product_vista.sbatch

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
export INEXOR_LOAD_TRACE=1
unset JAX_PLATFORMS
unset CUDA_VISIBLE_DEVICES

RUNS=/scratch/10303/jrc4/inexor_runs
SRC=$RUNS/d7-1009133                 # the run whose step-40 state this reads
WORK=$RUNS/d7e-$SLURM_JOB_ID
PROD=$RUNS/d7-1009133-product
CARDS=runs/v2
DRIVER=scripts/v2_d7_hero_smoke.py
TAG=d7e_$SLURM_JOB_ID
MEM_CSV=$CARDS/${TAG}_mem.csv
BIND="numactl --membind=0,1"
# `_require_cpu` refuses unless jax's first device is CPU, and on this node it is
# not. Scoped per leg rather than exported, so the pytest leg still sees the cards.
CPU="env JAX_PLATFORMS=cpu"
# 1009133's own layout parameters. See the block above; the defaults are wrong here.
LAYOUT="--slack 0.10 --arena-frac 0.01"
mkdir -p "$WORK" "$PROD"
export PROD

# `_ckpt_dir` is `<workdir>/ckpt` and 1009133 wrote `hero-ckpt`, so point one at
# the other rather than moving 1.3 TB or touching the source job's directory.
ln -sfn "$SRC/hero-ckpt" "$PROD/ckpt"
[ -e "$PROD/ckpt/gen1/manifest.json" ] || { echo "FATAL: no step-40 manifest under $PROD/ckpt"; exit 1; }

echo "=== $(hostname): commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID, $(date) ==="
free -g | head -2
df -h "$RUNS" | tail -1
python3 -c "import json;d=json.load(open('$PROD/ckpt/gen1/manifest.json'));print('source checkpoint:', d['provenance']['step'], 'of', d['provenance']['n_steps'])"
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
  du -sh "$PROD/export" 2>/dev/null
  ls -la "$PROD"/realization_*.json 2>/dev/null
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

# THE FINGERPRINT GUARD, before anything expensive.
# `load_checkpoint` refuses a fingerprint mismatch, and the fingerprint hashes
# `coeffs.tobytes()` -- which come from `scipy.integrate.quad` (QUADPACK), so the
# bytes ride on the platform's libm. It is a SAME-MACHINE contract: the same
# snippet on the osx-arm64 laptop produces
# d9adef05b2242df1740a7459fef6974448c7fe5718e4b5d7fd15c42bc52b35cf against this
# checkpoint's c28f7507..., with all fifteen fingerprinted CONFIG fields identical.
# The real check therefore has to run here. It costs one import, and it turns a
# refusal three minutes in (after the cgh64 legs) into one at second 20 that says
# what is wrong.
run_leg fingerprint-guard \
  $CPU pixi run -e gpu python -c '
import sys, json, argparse
sys.path.insert(0, "scripts")
import jax; jax.config.update("jax_enable_x64", True)
from v2_m6_realization import _cosmo, _coeffs, _engine_config, _geom
from inexor.engine import checkpoint_fingerprint
import os
args = argparse.Namespace(slack=0.10, tile_workers=16, checkpoint_every=10,
                          migrate_pooled=None, eject_kernel="jax")
ec = _engine_config(_geom("c-hero"), args, os.environ["PROD"] + "/ckpt")
co, _ = _coeffs(_cosmo())
want = checkpoint_fingerprint(ec, co)
have = json.load(open(os.environ["PROD"] + "/ckpt/gen1/manifest.json"))["provenance"]["fingerprint"]
print("fingerprint want", want, "\nfingerprint have", have)
sys.exit(0 if want == have else 1)
' \
  || { echo "FATAL: this node computes a different schedule than the one that wrote the checkpoint"; exit 1; }

# The export writer is the newest code in this job (7762c89) and on the hero
# state it first fires 40 minutes in. Its gates run at second 30.
run_leg pytest-gpu \
  pixi run -e gpu python -m pytest -q -rs tests/test_export.py tests/test_summary.py \
  || { echo "FATAL: the product gates failed"; exit 1; }

# --- cgh64: BOTH products end to end, on the same NFS, before the hero legs.
# `v2_m6_realization.py run` is CPU-only too (`_require_cpu`), which at cgh64
# would exercise the host engine rather than the device lane 1009133 ran. The D7
# driver is the proven leg; it defaults to the same slack 0.10 / arena-frac 0.01
# and is pointed at the `ckpt` directory the product subcommands look for.
#
# STEP 24, NOT 3, AND THE EPOCH IS THE REASON. The card leg reads the epoch it
# stops at, and the bins it can compare against linear theory are the ones below
# the nonlinear scale. At step 3 (a = 0.1189) the linear Delta^2 peaks at 0.26,
# so there IS no nonlinear scale and the card takes a branch the hero leg never
# will -- which is how 1010730 shipped a `float(None)` into the guard leg while
# leaving the hero's own path unexercised. Nothing in this schedule is nonlinear
# before step 15; by step 24 (a = 0.3981) the scale is 1.34 h/Mpc and falls
# INSIDE the band, which stops at half Nyquist = 1.57 h/Mpc here, so the smoke
# splits its bins either side of it exactly as the hero card at a = 1 will
# (0.2520 h/Mpc against the same 1.57 ceiling). Costs ~100 s of cgh64 stepping.
# --stop-at must stay a multiple of --checkpoint-every.
run_leg cgh64-ics \
  $BIND pixi run -e gpu python scripts/v2_m6_realization.py ics --config cgh64 \
    --workdir "$WORK/cgh64" --generator device \
  && run_leg cgh64-run \
  $BIND pixi run -e gpu python $DRIVER run --preset cgh64 --workdir "$WORK/cgh64" \
    --card $CARDS/${TAG}_cgh64.json --stop-at 24 --membind-nodes 0,1 --beat 60 \
    --checkpoint-dir "$WORK/cgh64-ckpt" --checkpoint-every 3
[ $? -ne 0 ] && { echo "FATAL: the cgh64 control failed"; exit 1; }

# The checkpoint dir is a SIBLING of the IC dir, not a child: the D7 driver
# refuses to write a checkpoint under the ICs (`5322c25`, containment by path
# component), which is what killed gb 1010700 at second 70. So the product
# subcommands get their own workdir with `ckpt` pointed at it -- the same shape
# as $PROD below, for the same reason.
mkdir -p "$WORK/cgh64-prod"
ln -sfn "$WORK/cgh64-ckpt" "$WORK/cgh64-prod/ckpt"

run_leg cgh64-card \
  $CPU $BIND pixi run -e gpu python scripts/v2_m6_realization.py card --config cgh64 \
    --workdir "$WORK/cgh64-prod" $LAYOUT --allow-partial \
  || { echo "FATAL: the card path is broken at cgh64; the hero leg would find it 2 h in"; exit 1; }

run_leg cgh64-export \
  $CPU $BIND pixi run -e gpu python scripts/v2_m6_realization.py export --config cgh64 \
    --workdir "$WORK/cgh64-prod" $LAYOUT --allow-partial \
    --export-dir "$WORK/cgh64-prod/export" \
  || { echo "FATAL: the export path is broken at cgh64"; exit 1; }
du -sh "$WORK/cgh64-prod/export"

# --- THE 4096^3 PRODUCTS. Card first: it is the cheaper leg and it is the one
# that says whether the realization is right, so a failure in the 1.65 TB write
# cannot cost us the spectrum.
run_leg hero-card \
  $CPU $BIND pixi run -e gpu python scripts/v2_m6_realization.py card --config c-hero \
    --workdir "$PROD" $LAYOUT
rc_card=$?

# The export runs whatever the card said -- the card emits no verdict, so an
# automatic gate on it would be inventing one.
run_leg hero-export \
  $CPU $BIND pixi run -e gpu python scripts/v2_m6_realization.py export --config c-hero \
    --workdir "$PROD" $LAYOUT --export-dir "$PROD/export"
rc_export=$?

echo ""
echo "=== card rc=$rc_card  export rc=$rc_export ==="
exit $rc_total
