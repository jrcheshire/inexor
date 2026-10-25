#!/bin/bash
#SBATCH -J quant-grid-halo
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00
#SBATCH -o quant-grid-halo-%j.log
# Which quantization moves small halos with the step count: scripts/compare/quant_grid.py at
# cdev8-L32 (128^3 in 32 Mpc/h, cgh64-L128 in every resolution knob) on one gg (Grace-Grace,
# 144-core) node, every leg a separate CPU process pinned to its own 4 cores, all at once.
#
#   seeds 0-3 x K = 120 / 240 from z = 9 x arms engine / ref / refx / refv; refxv at K = 240,
#   seed 0 (the emulation check). Each leg writes its card and its final positions
#   (`--save-x`, the export layout `halo_compare.py find` reads).
#   SEEDS, ARMS and EXTRA ("seed K arm" legs; empty for none) override that grid, e.g.
#   ARMS=refxs EXTRA= for the stochastic-rounding arm alone.
#
# Requires OUT_DIR (outputs under OUT_DIR/job-<id>: <tag>.json cards, x/<tag>/ positions) and
# INEXOR_SRC or submission from the inexor checkout, whose gpu env is used through its python
# with JAX_PLATFORMS=cpu, never through `pixi run`, so this job cannot reinstall an env other
# jobs share.
#   OUT_DIR=$SCRATCH/inexor_runs/quant-grid-halo sbatch -A <account> \
#     scripts/compare/quant_grid_halo_gg_vista.sbatch
# Wall ~1.5-2 h (longest leg: engine, 240 steps).

set -u
SRC=${INEXOR_SRC:-$SLURM_SUBMIT_DIR}
[ -f "$SRC/scripts/compare/quant_grid.py" ] || { echo "FATAL: not an inexor checkout: $SRC"; exit 1; }
OUT=${OUT_DIR:?set OUT_DIR, e.g. $SCRATCH/inexor_runs/quant-grid-halo}/job-${SLURM_JOB_ID:-local}
PY=${PY:-$SRC/.pixi/envs/gpu/bin/python3.14}
CFG=${CFG:-cdev8-L32}
SEEDS=${SEEDS:-0 1 2 3}
ARMS=${ARMS:-engine ref refx refv}
EXTRA=${EXTRA-0 240 refxv}
mkdir -p "$OUT/legs" "$OUT/x"
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
unset XLA_FLAGS CUDA_VISIBLE_DEVICES

echo "=== $(hostname) job ${SLURM_JOB_ID:-local} $(date) ==="
git -C "$SRC" log --oneline -1
nproc
"$PY" -c "import jax, inexor; print('jax', jax.__version__, jax.default_backend(), inexor.__file__); assert jax.default_backend() == 'cpu'" \
  || { echo "FATAL: the env does not import on CPU"; exit 1; }

legs=()
for seed in $SEEDS; do
  for k in 120 240; do
    for arm in $ARMS; do legs+=("$seed $k $arm"); done
  done
done
[ -n "$EXTRA" ] && legs+=("$EXTRA")
echo "${#legs[@]} legs"

i=0
pids=()
for leg in "${legs[@]}"; do
  read -r seed k arm <<<"$leg"
  lo=$((4 * i)); hi=$((lo + 3))
  tag="${CFG}_s${seed}_k${k}_${arm}"
  pin=(); command -v taskset >/dev/null && pin=(taskset -c "$lo-$hi")
  (cd "$SRC" && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "${pin[@]}" \
     "$PY" scripts/compare/quant_grid.py --config "$CFG" --seed "$seed" --a-init 0.1 --k "$k" \
     --arm "$arm" --out "$OUT/$tag.json" --save-x "$OUT/x/$tag" >"$OUT/legs/$tag.log" 2>&1
   echo "rc=$? $tag" >>"$OUT/legs/rc.txt") &
  pids+=($!)
  i=$((i + 1))
done
wait "${pids[@]}"
echo "=== done $(date) ==="
sort "$OUT/legs/rc.txt"
n_ok=$(grep -c "^rc=0 " "$OUT/legs/rc.txt")
n_x=$(ls "$OUT"/x/*/x.npy 2>/dev/null | wc -l)
echo "legs rc 0: $n_ok / ${#legs[@]}; positions written: $n_x"
[ "$n_ok" -eq "${#legs[@]}" ] && [ "$n_x" -eq "${#legs[@]}" ] || exit 1
