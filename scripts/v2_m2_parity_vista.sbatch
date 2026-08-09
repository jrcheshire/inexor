#!/bin/bash
#SBATCH -J inexor-m2-parity
#SBATCH -o %x-%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 03:00:00
#
# M-v2-2 EXIT GATE: the promoted two-level force must be BITWISE the probe at the
# ratified operating point. Submit from ~/src/inexor on Vista:
#     sbatch scripts/v2_m2_parity_vista.sbatch
#
# D-v2-16 clause 7 gates promotion on bitwise parity against `v2_g5_core.py` kept
# UNMODIFIED, at three geometries. Two run on CPU in the unit suite (tile
# identity; a clustered padding-dominated field). This is the third: cgh64 at
# T=256/b=32/P=320, 64 tiles -- the configuration every D-v2-10, D-v2-11 and
# D-v2-12 number was measured at. If the promoted engine is not identical there,
# three ratified records stop describing the shipped artifact.
#
# WHY THE DETERMINISTIC-OPS FLAG IS NOT OPTIONAL HERE. `tile_paint_f64`
# accumulates through .at[].add on an f64 mesh; on CUDA that is an atomic scatter
# with a non-reproducible order, measured in M2 S4 at 156/4096 elements over 8
# identical calls. Comparing two arms under that is a coin flip that sometimes
# passes. The flag removes it -- and because a self-reported knob lies (this
# project has a GPU/CPU pair whose log recorded no device, and a config field
# claiming a setting it never applied), the script DEMONSTRATES determinism as
# leg 0 by painting one tile twice and aborting unless the two are bit-identical.
#
# TIME/MEMORY. Generous wall on purpose -- billed for actual use, and my last
# three runtime estimates on this project were all low. cgh64 is 1.34e8
# particles: two (n,3) f64 force arrays at 3.2 GB each, the probe's bucketing at
# ~1 GB, plus ~12 GB of IC transient, against a gh node's ~194 GB. Device side is
# one P=320 tile at a time, ~2-3 GB against 96 GB of HBM.

set -u

export PATH="$HOME/.pixi/bin:$PATH"
# NOT boilerplate: jobs 896055 and 896092 ran ZERO science without this line, and
# 896055 reported COMPLETED 0:0 in sacct because `set -e` had been dropped.

export XLA_FLAGS="--xla_gpu_deterministic_ops=true"

echo "=== host $(hostname)  job ${SLURM_JOB_ID}  $(date -Is) ==="
cd "$HOME/src/inexor" || { echo "FATAL: no ~/src/inexor"; exit 1; }
echo "commit: $(git rev-parse --short HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"
echo "XLA_FLAGS=$XLA_FLAGS"

if ! command -v pixi >/dev/null 2>&1; then
    echo "FATAL: pixi not on PATH -- this job would run zero science and exit 0"
    exit 1
fi

# Demonstrate the backend rather than infer it. A CPU fallback would still PASS
# the parity check -- deterministically and meaninglessly -- because the whole
# point of running here is the CUDA atomic path.
pixi run -e gpu python -c "
import jax, sys
d = jax.devices()
print('jax', jax.__version__, 'devices', d)
if d[0].platform == 'cpu':
    sys.exit('FATAL: CPU backend -- this gate must exercise the CUDA scatter')
" || { echo "FATAL: jax/GPU precondition failed"; exit 1; }

# NO `set -e`: independent legs, cheapest first (M1 job 11's lesson). cdev8 is
# the scale smoke -- if something breaks between the laptop fixture and a real
# config, it breaks here in a minute rather than after the ICs of a 1.34e8
# particle run.
rc=0

echo ""
echo "--- leg 1: cdev8 scale smoke (T=64, b=32) ---"
pixi run -e gpu python scripts/v2_m2_parity_cgh64.py \
    --config cdev8 --tile 64 --buf 32 \
    --out runs/v2/m2_parity_cdev8.json || rc=1

echo ""
echo "--- leg 2: THE GATE -- cgh64 at the ratified operating point T=256 b=32 ---"
pixi run -e gpu python scripts/v2_m2_parity_cgh64.py \
    --config cgh64 --tile 256 --buf 32 \
    --out runs/v2/m2_parity_cgh64.json || rc=1

echo ""
echo "=== LEG SUMMARY ==="
for f in runs/v2/m2_parity_cdev8.json runs/v2/m2_parity_cgh64.json; do
    if [ -f "$f" ]; then
        echo "  $f:"
        pixi run -e gpu python -c "
import json,sys
d=json.load(open('$f'))
print('    all_ok', d['all_ok'], ' n_diff', d['n_diff'], ' max_delta', d['max_delta'],
      ' peak', d['oracle_peak'], ' vacuous', d['vacuous'])
" 2>/dev/null || echo "    (unreadable)"
    else
        echo "  $f: MISSING"
    fi
done
n=$(ls runs/v2/m2_parity_*.json 2>/dev/null | wc -l)
echo "cards written: $n"
[ "$n" -eq 0 ] && { echo "FATAL: zero legs produced a card"; exit 1; }
echo "exit rc=$rc  $(date -Is)"
exit $rc
