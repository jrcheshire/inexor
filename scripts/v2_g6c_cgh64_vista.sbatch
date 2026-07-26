#!/bin/bash
#SBATCH -J inexor-v2-g6c
#SBATCH -o %x-%j.log
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -A JPL-SPHEREx
#SBATCH -t 12:00:00
#
# v2 G6c: the THIRD RUNG of the calibration-transport ladder (JC, 2026-07-25).
# Submit from ~/src/inexor on Vista:  sbatch scripts/v2_g6c_cgh64_vista.sbatch
#
# WHY THIS JOB EXISTS. G6 showed the two-level split's low-k error is a
# realization-stable, correctable transfer WITHIN one config (leave-one-out cuts
# k<=0.5 median 8.9e-3 -> 9.9e-4, x9.0). G6b then asked whether T-bar can be
# calibrated on a SMALL box and applied to a big one -- which is the whole
# compute story, because the monolithic reference the calibration needs is the
# memory-expensive run. deneb job 192 (cdev8 -> cdev, 8x volume, fixed cell)
# reported a MIXED verdict in FRACTIONAL units -- shape transports (low-k index
# 1.435 cdev8 vs 1.542 cdev, ~7%) but amplitude appears not to (23% median /
# 37% max of the big-box transfer at k<=0.5, 3.5 sigma_calib).
#
# READ THAT IN ABSOLUTE UNITS BEFORE DRAWING A CONCLUSION FROM IT. D-v2-9's bar
# is an ABSOLUTE |dP/P|, deliberately, and the transfer's own amplitude varies by
# an order of magnitude across the band -- so a fractional statistic exaggerates
# disagreement exactly where the curve is smallest, which is the low-k end the
# whole question is about. Recomputed from the two aggregates (2026-07-25):
#   k<=0.5    uncorrected 4.45e-3 med -> small-box T-bar 1.02e-3 med (max 1.34e-3)
#             own-box leave-one-out 9.9e-4 med
#   full band uncorrected 1.60e-2 med -> small-box T-bar 1.89e-3 med (max 2.93e-3)
#   bar                                                            3.00e-2
# So at 8x the lever arm, calibrating on the small box lands within 3% of what
# calibrating on the box itself achieves (1.02e-3 vs 9.9e-4), and leaves the
# split error at ~1/10 of the bar across the whole gate band. The earlier
# "x9.0 -> x4.4" framing compared gains over DIFFERENT k ranges (cdev's own
# k<=0.5 reaches down to k_f=0.049; the cdev8 overlap starts at 0.098), so the
# denominators differ and the ratio is not the quantity to read. The residuals
# are, and they agree.
#
# THE ONE THING THAT DISCRIMINATES: is that ~1e-3 absolute offset FIXED in box
# ratio (absorb it, calibrate small, done) or does it GROW with the lever arm
# (box-dependent -> cheap calibration is dead)? cdev8 -> cdev is 8x volume.
# This rung adds cgh64 = 64x cdev8, 8x cdev, at the SAME fine cell (0.25), same
# coarse mesh ratio and same gate band -- so cdev8/cdev/cgh64 is a clean 1x/8x/64x
# volume ladder at fixed resolution, and the two transports (8x and 64x) are
# directly comparable. Flat 23% across both = fixed offset. ~2x growth = scaling.
#
# PARTITION: -p gg (Grace-Grace, no GPU), NOT -p gh. Every physics leg in this
# probe is CPU-f64 -- v2_g5_two_level_force.py spawns force/evolve workers with
# platform="cpu" (JAX_PLATFORMS=cpu in the child env), and the ONLY GPU legs are
# the capacity measurements, which --skip-capacity drops. A gh node's GH200 would
# idle for the entire run. gg also has 144 cores/node against gh's 72 Grace cores.
# This is NOT a D-v2-10 violation: that framing binds PERFORMANCE/Pareto points to
# the config-table homes so a bargain-bin card cannot shape architecture; this job
# measures a dimensionless ACCURACY ratio with no capacity leg. cgh64 comes here
# rather than to deneb purely on MEMORY -- the cgh64 monolithic reference is
# ~48 GB against deneb's 56 GB total, which is too thin for the transients.
# ACCOUNT is CASE-SENSITIVE: JPL-SPHEREx.
#
# THREADS: G5_CPU_THREADS=72, i.e. one Grace socket's worth, deliberately not 144.
# The wall estimate below is calibrated on job 837404's 64-thread gh measurements,
# and a gg node is two NUMA domains -- a single process whose first-touch pages
# land on socket 0 while half its Eigen pool runs on socket 1 can lose more to
# remote bandwidth than it gains in cores, on a leg that is memory-bandwidth bound
# f64 FFT. 72 keeps the measured configuration and keeps the wall predictable. If
# a later run wants the other socket, 144 is a one-line change and worth timing
# against this job's per-seed numbers -- do not assume it is faster.
#
# ARMS: one tiled arm, T512/b32, and no --shift-arm.
#   * T512/b32 is the CHEAPEST tiled arm at cgh64, not the most expensive. n_fine
#     is 1024, so T=512 tiles the box into 8 tiles at P = 512 + 2*32 = 576, a
#     padded-volume ratio of 1.42 against n_fine^3; T256/b32 pays 1.95 and
#     T128/b32 pays 3.38. Job 837404 measured T128/b32 at 3316 s and T256/b32 at
#     2608 s for 20 steps, so T512/b32 should land near 1900 s.
#   * Comparing this arm against cdev's T128/b32 (P=192) is legitimate and is
#     exactly what pivot_curve() in v2_g6b_calib_transport.py documents: the low-k
#     transfer is P-INDEPENDENT. Measured twice -- cdev T128(P192) = T256(P384)
#     below the band edge (G6), and cgh64 evolved dP/P T128 = T256 = T512 =
#     1.896e-2 identically (job 861849).
#   * --shift-arm is dropped because tile-origin dependence is already settled at
#     cdev (moving the lattice costs rms 2.6e-5 in-band against 2.4e-3 seed
#     scatter, two orders down) and it is not the box-transport question. The
#     aggregator's origin_dependence() returns None gracefully when no shifted
#     pairs exist. Seed 0 DOES still carry --shift-identity: check_identities()
#     hard-fails without the two degenerate-limit arms, and that guard is
#     load-bearing, not optional.
#
# WHAT TO READ, in this order (and note the criterion is ABSOLUTE, per above):
#   1. the transport residual median |R_cgh64 - R_cdev| and |R_cgh64 - R_cdev8|
#      at k<=0.5, against the 1.02e-3 the 8x rung already paid and against the
#      ~1e-3 per-realization floor that no amount of calibration removes. Flat
#      across 8x and 64x = fixed offset, calibrate small. Growing roughly with
#      box ratio would put 64x near ~8e-3 -- still under the 3e-2 bar, but no
#      longer negligible, and that is the case that kills cheap calibration.
#   2. the low-k index, against cdev 1.542 / cdev8 1.435.
#   3. the fractional statistics LAST, and only as a cross-check.
#
# SEEDS: 6. Per-seed scatter is sigma/|mean| ~ 0.12-0.18 at k<=0.5 (measured at
# cdev, and genuine realization-dependence rather than sample noise -- the
# shared-IC ratio cancels cosmic variance). Six seeds pin the cgh64 mean to
# SEM ~ 0.15/sqrt(6) ~ 6% of the mean; cdev's 16 seeds give ~3.7%; combined
# sigma_calib ~ 7%. So a repeat of G6b's 23% discrepancy would land at ~3 sigma,
# and a null bounds the offset at <~7%. That resolves fixed-offset vs growth,
# which is all this rung is for. More seeds buy precision the decision does not
# need -- the mean is already cheap to pin, and the ~12-14% single-box floor is
# irreducible by calibration seeds anyway.
#
# WALL: per seed = config gen + validation (tile identity at P=1024) + one pivot
# force arm + 2 evolve arms (mono 857 s, T512/b32 ~1900 s at 20 steps) ~ 46 min
# of evolve plus overhead. Seed 0 adds the two identity arms (translated mono +
# full-period-shifted T512) for ~another 46 min. Six seeds ~ 6-6.5 h. -t 12:00:00
# is deliberately generous: TACC bills ACTUAL use, so a tight wall buys nothing
# and only risks killing the ladder mid-flight.
#
# NO `set -e`, deliberately (M1 job 11 lesson: independent legs). A seed that
# crashes must not take the others with it, and the aggregator works with however
# many landed. Resumable by construction -- a seed whose result JSON already
# exists is skipped, so a resubmit picks up where a kill left off.
#
# OUTPUT (runs/ is gitignored; pull it back explicitly):
#   runs/v2/g5_results_cgh64_seed{0..5}.json          per seed
#   runs/v2/g6_split_stability_cgh64.{json,png}       the ensemble aggregate
#   runs/v2/g6b_transport_cdev_cgh64.{json,png}       the 8x transport
#   runs/v2/g6b_transport_cdev8_cgh64.{json,png}      the 64x transport
# The job runs the transport legs itself, so it emits the verdict card directly
# rather than leaving a manual step. Those legs are pure numpy over three small
# JSONs (sub-second) and they need the cdev and cdev8 aggregates, which are born
# on deneb -- STAGE THEM FIRST or the two legs will exit with a clear "missing
# ..." and the ensemble work above them is still saved:
#   scp deneb:~/src/inexor/runs/v2/g6_split_stability_cdev{,8}.json \
#       vista:~/src/inexor/runs/v2/
# (Untracked run outputs, so hand-syncing them is the right route; source always
# travels by git.) Verdict is JC's gate call, not this script's.

cd ~/src/inexor
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu
export G5_CPU_THREADS=72
# WHY THIS OVERRIDE (job 865844 died on it, 2026-07-25). The `gpu` feature
# declares system-requirements cuda = "12", so pixi refuses to ACTIVATE the env
# on any machine lacking the __cuda virtual package -- and a gg node has no GPU.
# That check is pixi's, made before a single line of python runs; JAX_PLATFORMS
# =cpu below cannot help, because nothing gets as far as importing jax. Every
# leg of 865844 printed "unsupported-platform ... missing virtual packages:
# __cuda >= 12" and exited, 11 times. Mocking the virtual package is pixi's own
# documented remedy and is the right one here rather than switching envs: the
# `gpu` env is the ONLY env installed on Vista (no compute-node package fetch),
# it is locked for linux-aarch64, and it is the env that produced the cdev and
# cdev8 aggregates on deneb and the cgh64 card in job 861849 -- so this rung of
# the ladder stays numerically comparable to the rungs it is measured against.
# Nothing touches CUDA: the value is a claim about the machine, not a request
# for a device, and the probe's workers are plain sys.executable subprocesses
# that never re-enter pixi.
export CONDA_OVERRIDE_CUDA=12.0

SEEDS="0 1 2 3 4 5"

echo "=== env ==="
git log --oneline -1
nproc
free -g | head -2
# The env install runs ON THE COMPUTE NODE (login-node pixi gets killed).
# --locked: fail loudly if the lock is stale rather than silently re-solving.
# The `gpu` env is Vista's aarch64 env; JAX_PLATFORMS=cpu above plus the probe's
# own per-worker setting keep every leg on the Grace cores, so no CUDA is touched.
# CONDA_OVERRIDE_CUDA (see above) is what lets it activate on a GPU-less node.
pixi install --locked -e gpu
pixi run -e gpu python -c "import jax; print(jax.__version__, jax.devices())"
# The one place `set -e` discipline WOULD have helped: the env is a prerequisite,
# not an independent leg, so a failure here makes all 11 legs below fail
# identically and bury the cause under repetition (job 865844). Stop instead.
if ! pixi run -e gpu python -c "import jax" >/dev/null 2>&1; then
  echo "!! ABORT: the 'gpu' env will not activate/import on this node -- fix the"
  echo "!! env before spending the ensemble. Diagnostic:"
  pixi run -e gpu python -c "import jax; print(jax.__version__, jax.devices())"
  exit 1
fi

for S in $SEEDS; do
  OUT="runs/v2/g5_results_cgh64_seed${S}.json"
  if [ -f "$OUT" ]; then
    echo "=== seed $S: $OUT exists, skipping ==="
    continue
  fi
  EXTRA=""
  if [ "$S" = "0" ]; then EXTRA="--shift-identity"; fi
  echo "=== seed $S ==="
  pixi run -e gpu python scripts/v2_g5_two_level_force.py --config cgh64 \
    --families gauss --skip-coarse --skip-floor-arms --skip-1024 --skip-capacity \
    --pivot-only --evolve-tiles 512:32 $EXTRA \
    --seed "$S" --out-suffix "_seed${S}"
done

echo "=== aggregate cgh64 ==="
pixi run -e gpu python scripts/v2_g6_split_stability.py --config cgh64 \
  --out-prefix g6_split_stability_cgh64

# The two lever arms, cheapest-informative first. Each needs the corresponding
# staged aggregate; a missing one exits this leg only (no set -e) and leaves the
# ensemble above intact. Read the ABSOLUTE k<=0.5 residual against the 1.02e-3
# the 8x rung paid and against cgh64's own leave-one-out, which the script prints
# alongside; the fractional numbers are the cross-check, not the headline.
for SMALL in cdev cdev8; do
  AGG="runs/v2/g6_split_stability_${SMALL}.json"
  if [ ! -f "$AGG" ]; then
    echo "!! transport ${SMALL} -> cgh64 SKIPPED: $AGG not staged on this host"
    continue
  fi
  echo "=== transport: ${SMALL} -> cgh64 ==="
  pixi run -e gpu python scripts/v2_g6b_calib_transport.py --small "$SMALL" --big cgh64
done

echo "=== DONE ==="
