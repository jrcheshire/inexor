#!/bin/bash
#SBATCH -J m6-gb-probe
#SBATCH -p gb
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A JPL-SPHEREx
#SBATCH -o m6-gb-probe-%j.log
# THE gb PROBE: price the host-state / device-step design for 4096^3 on ONE
# gb node BEFORE anything is built. Submit from the Vista checkout:
#   sbatch scripts/v2_m6_gb_probe_vista.sbatch
#
# WHY. The CPU engine is at its floor (tile loop + coarse paint = 84% of the
# step, both within 1.5-2x of gg's bandwidth ceiling) and 4096^3 fits no
# node's host at any worker count. The reading of the memory-floor thesis
# that still reaches 4096^3 on one node is: the host is a byte store for the
# ~721 GB T9 state (gb has 1026), and the four Blackwell GPUs do everything
# per step, slabs streamed over C2C. gb's MaxWall is 12 h (qlimits), so the
# design must be FAST: K=40 in one job means <= 1080 s per step. That bar is
# the machine's, not picked. 2 jobs/user, 18 nodes; ~13 idle at submission.
#
# LEGS, cheapest first, each writing a card; the readout sums them:
#   0  preflight: cuda backend, 4 devices, provenance (topology, memory)
#   1  smoke of the NEW probe on this node (gates the rest: the reporting
#      path is untested code until it has run here -- job 923313)
#   2  the new probe, one GPU: eject kernel on a 4096^3-scale slab, TSC int
#      paint of a slab into a 2048^3 device mesh (sub-block and full), device
#      rfftn at 1024/1536/2048^3 (2048 is a DESIGNED OOM at the measured 7x)
#   3  per-tile force anatomy on a B200, P=320 (the 2048^3 tile) and P=576
#      (the 4096^3 preset's tile), f64 -- the V4a leg, same flags
#   4  the CURRENT engine on a B200, serial, cdev K=3: decode/quantize device
#      phases the force leg omits, and the B200 against 912457's gh card
#   5  the 2048^3 out-of-core FFT on gb's 144-core host (the coarse-solve
#      floor if the solve stays on the host; 85.7 s/transform on a GH200 host)
#   6  streaming host->device, one GPU: ladder 64..640 GiB (past HBM, past
#      one socket, up to two thirds of host RAM where gh collapsed)
#   7  streaming on all four GPUs at once (aggregate; two sockets of LPDDR
#      supply four C2C links)
#   8  readout: the per-step sum against the bar, unmeasured terms named
#
# PRE-REGISTERED:
#   a. Receipts first: every device leg's card must read platform gpu; the
#      eject arm must report a leaver fraction in (0, 1); the paint arm's
#      sub-block mass must equal its full-mesh mass; the engine card must be
#      a serial cuda run (tile_workers 1). A leg without its receipt measured
#      nothing.
#   b. The deliverable is the readout's SUM in s/step at 4096^3 K=40 with a
#      perfect 4-GPU split ASSUMED, against 1080 s. Either sign is a result.
#      Above the bar the design needs segments or a different machine and
#      the build is not opened. Below it the sum is a FLOOR: insert, repack,
#      kick and host bookkeeping are named as unmeasured.
#   c. A missing load-bearing card withholds the verdict.
#
# NOT ESTABLISHED HERE: a built pipeline's efficiency, the 4-way split, the
# insert half of the migrate, anything on a node that is not Vista gb.
#
# BUDGET: ~85 min expected against 3 h (gb 1 SU/node-hr -> 3 SU worst case).
# Generous on purpose: pinning 640 GiB of host memory has never been timed
# here, and a TIMEOUT bills the whole wall. Streaming legs run LAST so a slow
# pin cannot starve the cheap legs.
#
# OPS (gb): pixi on PATH; -e gpu; NO JAX_PLATFORMS=cpu except the host-FFT
# leg (that one is host work by construction); NO CUDA_VISIBLE_DEVICES at
# preflight so all four cards are seen, then one card per single-GPU leg;
# XLA_PYTHON_CLIENT_PREALLOCATE=false so device peaks are real; -A is
# case-sensitive. NO set -e: per-leg exit codes, job exits $rc_total.

export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { echo "FATAL: pixi not on PATH"; exit 1; }
cd "$SCRATCH/src/inexor" 2>/dev/null || cd "$HOME/src/inexor" || {
  echo "FATAL: no inexor checkout"; exit 1; }
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache
unset JAX_PLATFORMS
unset CUDA_VISIBLE_DEVICES

echo "=== $(hostname): $(nproc) cores, commit $(git rev-parse --short HEAD), job $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id --format=csv,noheader 2>/dev/null \
  || echo "no nvidia-smi"
nvidia-smi topo -m 2>/dev/null | head -12
free -g | head -2
lscpu | grep -E "^CPU\(s\)|Model name|NUMA node"
numactl -H 2>/dev/null | head -8
df -h "$SCRATCH" | tail -1

pixi run -e gpu python -c "
import jax
ds = jax.devices()
assert ds[0].platform == 'gpu', ds
print('jax', jax.__version__, len(ds), 'devices:', ds[0].device_kind)
print('memory kinds:', [m.kind for m in ds[0].addressable_memories()])
" || { echo "FATAL: jax cuda backend check failed"; exit 1; }

rc_total=0
names=(); codes=()
# `run_leg [-t CODES] NAME CMD...`; CODES = nonzero exit codes this leg
# pre-registers as reportable states (rc=2 = the phase probe reporting its own
# instrument non-neutrality).
run_leg() {
  local tol=""
  [ "$1" = "-t" ] && { tol="$2"; shift 2; }
  local name="$1"; shift
  echo; echo "=== $name ==="
  local t0=$SECONDS
  "$@"; local rc=$?
  echo "=== $name rc=$rc after $((SECONDS - t0)) s ==="
  names+=("$name"); codes+=("$rc")
  local ok=0 c
  [ $rc -eq 0 ] && ok=1
  for c in $(echo "$tol" | tr ',' ' '); do [ "$rc" = "$c" ] && ok=1; done
  [ $ok -eq 0 ] && rc_total=1
  return $rc
}

# `capture_json OUT CMD...`: run CMD, tee its stdout into this log, and write
# the last WORKER_JSON line to OUT as a card (the g5 `--single` leg prints its
# record instead of writing one; the V4a numbers were read off a log, which
# is not a card).
capture_json() {
  local out="$1"; shift
  local tmp; tmp=$(mktemp)
  "$@" | tee "$tmp"
  local rc=${PIPESTATUS[0]}
  grep '^WORKER_JSON ' "$tmp" | tail -1 | sed 's/^WORKER_JSON //' > "$out"
  rm -f "$tmp"
  if [ ! -s "$out" ]; then echo "no WORKER_JSON captured for $out"; rm -f "$out"; return 1; fi
  echo "card: $out"
  return $rc
}

PROBE="scripts/v2_m6_gb_probe.py"
G5="scripts/v2_g5_two_level_force.py"
PHASE="scripts/v2_m6_phase_time.py"
G4="scripts/v2_g4_gh_memory.py"

# ---- 1. smoke of the new probe on THIS node and backend; gates everything
run_leg smoke env CUDA_VISIBLE_DEVICES=0 pixi run -e gpu python $PROBE --smoke --out-suffix _gb_smoke
if [ "${codes[${#codes[@]}-1]}" != "0" ]; then
  echo "FATAL: smoke failed; the reporting path is broken on this node -- not spending the legs"
  exit 1
fi

# ---- 2. the new probe at the 4096^3 preset, one GPU
run_leg probe env CUDA_VISIBLE_DEVICES=0 pixi run -e gpu python $PROBE --preset c-hero --reps 3 \
  --out-suffix _gb

# ---- 3. per-tile force anatomy on a B200 (the V4a leg, same flags)
STATE="runs/v2/g5_states/config_cgh64_a1p0.npz"
if [ ! -f "$STATE" ]; then
  echo "config state $STATE absent (purge?) -- regenerating, CPU f64 as the leg is written"
  run_leg make-config env CUDA_VISIBLE_DEVICES=0 pixi run -e gpu python $G5 --single --kind config \
    --config cgh64 --a 1.0 --seed 0
fi
# f64 only: on the TILED path the probe's --fdtype is not forwarded to
# force_short_tiled (it reaches only the global arm), so an f32 leg here would
# be a knob that does not apply. The f32 tile force is a build item, not a leg.
FORCE_COMMON="--single --kind force --family gauss --which mono --alpha 1.0 --fdtype f64 --a 1.0 --profile 1 --config cgh64"
run_leg force-T256-f64 capture_json runs/v2/m6_gb_force_T256_f64_gb.json \
  env CUDA_VISIBLE_DEVICES=0 pixi run -e gpu python $G5 $FORCE_COMMON \
  --tag gbprobe_T256_f64 --tile 256 --buf 32
run_leg force-T512-f64 capture_json runs/v2/m6_gb_force_T512_f64_gb.json \
  env CUDA_VISIBLE_DEVICES=0 pixi run -e gpu python $G5 $FORCE_COMMON \
  --tag gbprobe_T512_f64 --tile 512 --buf 32

# ---- 4. the current engine on a B200, serial (912457's cdev knobs)
run_leg -t 2 engine-cdev-b200 env CUDA_VISIBLE_DEVICES=0 pixi run -e gpu python $PHASE \
  --config cdev --k 3 --repeats 1 --warmup 1 --slack 0.20 --arena-frac 0.20 \
  --out-suffix _gb_b200_cdev

# ---- 5. the 2048^3 out-of-core FFT on the host (host work; keep jax off the GPU)
FFTDIR="$SCRATCH/inexor_runs/gbprobe_m5"
mkdir -p "$FFTDIR"
run_leg host-ooc-fft-2048 env JAX_PLATFORMS=cpu pixi run -e gpu python scripts/v2_m5_fft_gh.py \
  --n 2048 --workdir "$FFTDIR" --out-suffix _gb

# ---- 6. streaming, one GPU: ceiling control under HBM, then past HBM and past one socket
run_leg stream-1gpu-a env CUDA_VISIBLE_DEVICES=0 pixi run -e gpu python $G4 \
  --arms hbm,staged,coherent --ladder 64,128 --reps 3 --out-suffix _gb_a
run_leg stream-1gpu-b env CUDA_VISIBLE_DEVICES=0 pixi run -e gpu python $G4 \
  --arms staged,coherent --ladder 256,384,512,640 --reps 3 --out-suffix _gb_b

# ---- 7. streaming on all four GPUs at once (unpinned; the topology dump above
#         says which socket each card hangs off, the readout sums the four rates)
echo; echo "=== stream-4gpu ==="
t0=$SECONDS
pids=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i pixi run -e gpu python $G4 --arms staged,coherent --ladder 64,128 \
    --reps 3 --out-suffix _gb4x_gpu$i > "m6-gb-probe-${SLURM_JOB_ID}-4x-gpu$i.log" 2>&1 &
  pids+=($!)
done
rc4=0
for i in 0 1 2 3; do
  wait "${pids[$i]}"; rc=$?
  echo "--- gpu$i rc=$rc"; tail -n 12 "m6-gb-probe-${SLURM_JOB_ID}-4x-gpu$i.log"
  [ $rc -ne 0 ] && rc4=1
done
echo "=== stream-4gpu rc=$rc4 after $((SECONDS - t0)) s ==="
names+=("stream-4gpu"); codes+=("$rc4")
[ $rc4 -ne 0 ] && rc_total=1

# ---- 8. readout (the orchestrator imports no jax; keep it off the GPU anyway)
run_leg readout env JAX_PLATFORMS=cpu pixi run -e gpu python scripts/v2_m6_gb_probe_readout.py \
  --suffix _gb --preset c-hero --k 40 --n-gpus 4

echo; echo "=== LEG SUMMARY ==="
for i in "${!names[@]}"; do echo "  ${names[$i]}: rc=${codes[$i]}"; done
echo "=== CARDS WRITTEN ==="
ls -la runs/v2/m6_gb_probe_gb.json runs/v2/m6_gb_probe_gb_smoke.json \
  runs/v2/m6_gb_force_T*_gb.json runs/v2/m6_phase_time_gb_b200_cdev.json \
  runs/v2/m5_gate_fft-gh_gb.json runs/v2/g4_gh_memory_gb_a.json runs/v2/g4_gh_memory_gb_b.json \
  runs/v2/g4_gh_memory_gb4x_gpu*.json 2>&1
echo "=== DONE rc_total=$rc_total ==="
exit $rc_total
