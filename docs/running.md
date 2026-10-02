# Running inexor

There are two drivers: `scripts/run/realization.py` (the CPU driver, used for every phase on
a CPU and for the card and export phases everywhere) and `scripts/run/device_run.py` (the
GPU stepping driver). There are also Slurm job scripts that chain them together. Commands
below omit the `pixi run --frozen` (or `pixi run -e gpu`) prefix.

## The CPU driver: `realization.py`

```bash
python scripts/run/realization.py {ics,run,export,card} --workdir DIR [--config PRESET] [flags]
```

| Phase | What it does | Reads | Writes (in `--workdir`) |
|---|---|---|---|
| `ics` | 2LPT initial conditions, streamed | nothing | `t9_slab_NNNN.npz`, `manifest.json`, `realization_ics.json` |
| `run` | steps the schedule, or one segment of it, with checkpoints | the ICs, or the newest checkpoint in `ckpt/` | `ckpt/gen0`, `ckpt/gen1`, `realization_run_<from>_<to>.json` |
| `card` | P(k) against bin-averaged linear theory | newest checkpoint (or `--ic-dir`) | `realization_pk.json` (`realization_pk_ics.json` with `--ic-dir`) |
| `export` | writes the particles as plain `.npy` arrays | newest checkpoint | `export/{x,v}.npy`, `export/export.json`, `realization_export.json` |

Each phase runs in its own process. A process's peak memory never resets, so this is the
only way to read each phase's peak host memory separately. It also means that a queue time
limit costs one phase rather than the whole run. Checkpoints always go to `<workdir>/ckpt`.

### Flags

| Flag | Default | Phases | Meaning |
|---|---|---|---|
| `--config` | `cdev8` | all | preset: `smoke`, `cdev8`, `cdev`, `cgh64`, `c-1024`, `c-gh`, `c-hero` |
| `--workdir` | required | all | the run's directory |
| `--k-steps` | 40 | run, card, export | steps from `--a-init` to a = 1 |
| `--a-init` | 0.1 | all | starting scale factor, for the ICs and the step grid |
| `--growth2` | `lcdm` | all | second-order growth for the 2LPT ICs and the integrator weights (`lcdm` or `eds`) |
| `--n-part`, `--n-fine`, `--n-coarse`, `--buf` | preset | all | resolution overrides. The geometry and split error terms are printed |
| `--coarse-match-order` | 3 | all | assignment order the coarse match factor divides out (2 = legacy) |
| `--tile-workers` | 16 | run, card | worker processes. More than 1 means a pooled run, with the state in shared memory |
| `--slack`, `--arena-frac` | 0.20, 0.20 | run, card, export | allocation layout used when a state is loaded |
| `--alloc-margin` | 0.10 | run, card | allocation margin used when a state is loaded **from ICs** |
| `--seed` | 0 | ics | random seed |
| `--generator` | `host` | ics | `device` generates the ICs on the GPUs |
| `--noise`, `--pencil-batch` | `device`, 1 | ics | options for the device generator |
| `--bucket-cells` | 2 | ics | position bucket side in particle cells (recorded in the manifest) |
| `--keep-stage` | off | ics | keep the `stage/` intermediates |
| `--slab` | 32 | ics, card | slab depth for the streamed transforms |
| `--stop-at` | end | run | absolute step to stop before. Must be a multiple of `--checkpoint-every` |
| `--checkpoint-every` | 5 | run | checkpoint cadence, in steps |
| `--migrate-pooled` / `--serial-migrate` | auto | run | run the migrate on the pool, or serially |
| `--eject-kernel` | `jax` | run | `jax` or `numpy` |
| `--phase-instrument` | `time` | run | `time` (seconds per phase) or `peak` (per-phase host high-water mark, Linux only) |
| `--allow-partial` | off | export | export a checkpoint from before the last step |
| `--export-dir` | `<workdir>/export` | export | output directory |
| `--chunk-bricks` | 1024 | export | bricks decoded per chunk |
| `--min-weight` | 100 | card | drop bins with this many modes or fewer |
| `--k-max`, `--n-bins` | half Nyquist, 64 | card | fix the binning to `--n-bins` bins over [0, `--k-max`]. `--n-bins` has no effect without `--k-max` |
| `--ic-dir` | none | card | card the ICs in this directory at a = `--a-init` |
| `--card-pool` / `--serial-card` | pooled if `--tile-workers` > 1 | card | paint on a worker pool (same result bit for bit), or serially |
| `--heartbeat` | 60 | card, export | seconds between progress lines (0 turns them off) |

### Refusals you will meet

- **CPU backend required.** `run`, `export` and `card` (and `ics` unless `--generator device`)
  exit with `FATAL: non-CPU backend` if JAX's first device is not a CPU. In the `gpu` env,
  prefix them with `JAX_PLATFORMS=cpu`.
- **ICs made at a different epoch or growth.** Starting from ICs, `run` (and `card
  --ic-dir`) refuses ICs whose manifest records a different `--a-init` or `--growth2`.
  Regenerate the ICs, or pass the matching value.
- **`--stop-at` off a checkpoint boundary.** The stop step must be a multiple of
  `--checkpoint-every`. With no `--stop-at`, the stop step is `--k-steps`, so `--k-steps`
  must be a multiple of it as well. A stop at or before the step you are resuming from is
  also refused.
- **Export before the last step.** `export` refuses a checkpoint short of `--k-steps` unless
  you pass `--allow-partial`.
- **Resume onto a different configuration.** Loading a checkpoint fails on a fingerprint
  mismatch (see below).
- **Empty card.** `card` fails if no bin has more than `--min-weight` modes. Widen the bins
  (`--k-max` with fewer `--n-bins`) or lower `--min-weight`.
- `card` and `export` with no checkpoint: `no checkpoint to read; run 'run' first`.
- `card --ic-dir` pointed at a checkpoint is refused (it would be scored at the wrong
  epoch). `--n-coarse` coarser than the preset and a non-power-of-two `--n-part` are
  refused.
- `run` on a checkpoint already at `--k-steps` prints `NOTHING TO DO` and exits 0.

## Segments and resume

`run --stop-at N` advances to absolute step N and stops. The next `run` on the same
`--workdir` resumes from the newest checkpoint and prints which source it used:

```bash
python scripts/run/realization.py run --config cdev8 --workdir $W --stop-at 20
python scripts/run/realization.py run --config cdev8 --workdir $W          # 20 -> 40
```

- Each segment receives the full schedule plus a stop step, never a truncated schedule. As a
  result, the segments together reproduce the uninterrupted run bit for bit.
- Checkpoints alternate between two generations, `ckpt/gen0` and `ckpt/gen1`. A generation's
  `manifest.json` is removed first and written last, so a checkpoint torn mid-write is
  skipped and the other generation survives. On resume, the first write goes to the
  generation that was *not* loaded, so the resume point stays intact until a newer
  checkpoint is complete. The loader picks the generation with the higher recorded step.
- **Fingerprint.** Every checkpoint records a hash of the geometry and kernel fields
  (box, particle count, meshes, tile, buffer, split scale, paint and dtypes, and the coarse
  match order) plus the integrator coefficients as raw float64 bytes. The coefficients
  depend on the cosmology, `--a-init`, `--k-steps` and `--growth2`, so a resume, card or
  export must use the same values of all of these. Execution and layout settings are not part
  of the hash (`--tile-workers`, `--slack`, `--arena-frac`, `--eject-kernel`,
  `--checkpoint-every`), so they may change between segments. The coefficients come from
  numerical integration, and their bytes can differ between platforms' math libraries. A
  checkpoint can therefore be refused on a different platform (for example, one written on
  Linux and read on macOS) even with an identical configuration.

## The GPU driver: `device_run.py`

`device_run.py` steps a preset on N GPUs. The particle state stays in host memory, and the
coarse paint and solve, the tile loop and the migrate all run on the cards. It does not
generate ICs and does not produce cards or exports:

- **ICs**: `python scripts/run/realization.py ics --config P --workdir ICS --generator device`
  (in the `gpu` env).
- **Card and export**: the CPU driver, run with `JAX_PLATFORMS=cpu` against the GPU run's
  checkpoint (see below).

| Subcommand | What it does |
|---|---|
| `preflight` | Checks everything before the expensive part and exits 2 on any refusal: device count and backend, the allocator's stats, the IC manifest (particle count, slab count, `growth2`), the planner's host and load peak against the CPU NUMA nodes' memory, the NUMA memory binding, and free scratch space |
| `run` | Loads the ICs or resumes a checkpoint, then steps to `--stop-at`. Prints one line per phase boundary and a heartbeat, and rewrites a JSON card (`--card`) at every boundary. On failure, the traceback and device memory stats go into the card |
| `summarize` | Reads a run card together with the job's sampler CSVs (`--gpu-csv`, `--mem-csv`) and writes per-phase GPU memory, host memory and utilization to `--out`. Does not import jax |

Key flags for `preflight` and `run`:

| Flag | Default | Meaning |
|---|---|---|
| `--preset`, `--workdir`, `--card` | required | preset, IC directory (read only), card JSON path |
| `--cards` | 4 | GPUs to use |
| `--slack`, `--alloc-margin`, `--arena-frac` | 0.10, 0.10, 0.01 | allocation layout |
| `--growth2` | `lcdm` | must match the ICs |
| `--membind-nodes` | none | refuse unless the process runs under `numactl --membind` on these NUMA nodes |
| `--scratch` (preflight) | `$SCRATCH` or `/tmp` | filesystem whose free space is reported |
| `--stop-at` (run) | required | absolute step to stop before |
| `--k-steps` (run) | 40 | length of the whole schedule |
| `--checkpoint-dir`, `--checkpoint-every` (run) | none, 0 | 0 = no checkpoints. Otherwise `--stop-at` must be a multiple of `--checkpoint-every` |
| `--expect-step` (run) | 0 | 0 = start from the ICs, refused if `--checkpoint-dir` already holds a checkpoint. N = resume from the newest checkpoint, refused unless it is at step N |
| `--timed-last`, `--timed-all` (run) | off | synced per-pass timing breakdown on the last step or on every step |
| `--drop-ic-cache` (run) | off | drop each IC slab's page cache as it is read |
| `--beat` (run) | 60 | heartbeat seconds |
| `--comm` (run) | `serial` | `mpi`: one rank per process across nodes (see below) |
| `--comm-timeout` (run) | 1800 | seconds a rank may wait at one exchange before it aborts the job |

`run` also refuses to write a card or checkpoint anywhere under the IC directory. The GPU
driver always starts at a = 0.1 (it has no `--a-init`), and it checks the ICs' `growth2` but
not their epoch.

**Across nodes.** `run --comm mpi` runs one rank per process, one process per node, under
MPI:

```bash
mpiexec -n 2 python -m mpi4py scripts/run/device_run.py run --preset c-gh --workdir $ICS \
    --card run.json --cards 1 --k-steps 120 --stop-at 40 --checkpoint-dir $CKPT \
    --checkpoint-every 40 --comm mpi
```

- Each rank loads its own brick slabs of the ICs or of the checkpoint, writes its own card
  (`run.rank<r>.json`), and prefixes its log lines with `[rank r]`. `python -m mpi4py` makes a
  rank that raises end the whole job.
- All ranks write each checkpoint together, and the files are the same bytes at any rank
  count: a checkpoint from N nodes resumes on M, one included, and the card and export read it
  as a single-node checkpoint.
- Ranks x cards may not exceed the tile planes, and each rank needs at least 2r + 1 brick
  slabs for a drift that reaches r slabs (3 at r = 1): the migrate hands particles to
  immediate neighbours only.
- Each step's record in a rank's card carries that rank's exchanges under `ranks.comm`: calls,
  bytes sent and seconds per operation, and `wait_s`, the time blocked waiting on other ranks.
- `D7_FAIL_AT=<phase> D7_FAIL_RANK=<r>` exercises the failure path on one rank.
- On a cluster, launch each rank through `scripts/run/rank_exec.sh [--membind NODES] [--samples
  PREFIX] -- CMD ...`. It replaces `@RANK@` in the arguments and in exported variables, so one
  launch line gives each rank its own card, checkpoint dir or compilation cache. It also starts
  the node's GPU and NUMA samplers, which stop when the command ends, then becomes the command,
  so the launcher's signals reach it.
- On a laptop, `scripts/run/mpi_lane.sh` runs the tests that need real MPI processes, in a
  throwaway `pixi exec` env (the lock's jax and numpy, plus mpi4py and MPICH).

**Card and export of a GPU run.** The CPU driver reads `<workdir>/ckpt`, so link the GPU
run's checkpoint directory there. Use a product directory that is outside the IC directory:

```bash
ln -sfn $CKPT $PROD/ckpt
JAX_PLATFORMS=cpu python scripts/run/realization.py card --config c-1024 --workdir $PROD \
    --k-steps 120 --slack 0.10 --arena-frac 0.01 --tile-workers 8 --card-pool
```

`--config`, `--k-steps` and `--growth2` must match the run, or the fingerprint check fails.
`--slack` and `--arena-frac` must also match the GPU run (its defaults are 0.10 and 0.01, not
the CPU driver's 0.20 and 0.20). A checkpoint stores particle membership but not the
allocation, and the loader rebuilds the allocation from these two flags.

## Slurm job scripts

The scripts in `scripts/run/` were written for TACC Vista: partition `gb` (one node, four
GB200s) and partition `gh` (one GH200). Each runs checks (preflight, tests, a small control
run) before the expensive leg and stops if they fail.

| Script | Partition | What it does | Required variables |
|---|---|---|---|
| `hero_ics_vista.sbatch` | gb | `c-hero` ICs on four cards with the device generator; refuses an existing manifest | `INEXOR_RUNS`, `HERO_IC_DIR` |
| `hero_steps_vista.sbatch` | gb | one segment of `c-hero` on a 120-step schedule, checkpointing every 20 steps into `$REAL_DIR/hero-ckpt` | `INEXOR_RUNS`, `HERO_IC_DIR`, `REAL_DIR`, `EXPECT_STEP`, `SEG_STOP` |
| `hero_card_vista.sbatch` | gb | P(k) card of the final checkpoint in `$SRC_RUN/hero-ckpt`, written to `$PROD_DIR` | `INEXOR_RUNS`, `SRC_RUN`, `K_STEPS`, `PROD_DIR` |
| `hero_export_vista.sbatch` | gb | export of the same checkpoint to `$PROD_DIR/export` | `INEXOR_RUNS`, `SRC_RUN`, `K_STEPS`, `PROD_DIR` |
| `gh_single_card_vista.sbatch` | gh | a whole realization on one GH200: ICs, 120 steps, card, under `$INEXOR_RUNS/d8-gh-<preset>-<job id>/` | `INEXOR_RUNS`. Optional: `PRESET` (`c-1024`), `SLACK` (0.10), `ALLOC_MARGIN` (0.10), `ARENA_FRAC` (0.01) |
| `multinode_gate_vista.sbatch` | gh, 2 nodes | the multi-node byte gate: the single-node run on each node and the 2-rank run, compared file by file at the split step and at the end (`$INEXOR_RUNS/mn-gate-<preset>-<job id>/`); a split-step mismatch reruns segment 1 with deterministic ops and stops. The run legs' launch line is tried on every node first; every leg has a time cap, a run leg is also killed after `QUIET_S` of silent output, and a failed run leg stops the job | `INEXOR_RUNS`. Optional: `PRESET` (`c-1024`), `K` (120), `SPLIT` (20), `STOP`, `EVERY`, layout as above, `LEG_FIXED_S` (600) + `LEG_STEP_S` (100) per step (a run leg's cap), `QUIET_S` (600) |

Every script also needs `INEXOR_SRC` (the inexor checkout), or it must be submitted from the
checkout. Examples:

```bash
INEXOR_RUNS=$SCRATCH/inexor_runs HERO_IC_DIR=$SCRATCH/inexor_runs/c-hero-lcdm \
  sbatch -A <account> scripts/run/hero_ics_vista.sbatch

INEXOR_RUNS=$SCRATCH/inexor_runs HERO_IC_DIR=$SCRATCH/inexor_runs/c-hero-lcdm REAL_DIR=$SCRATCH/inexor_runs/hero-k120 \
  sbatch -A <account> --export=ALL,EXPECT_STEP=0,SEG_STOP=40 --dependency=afterok:<ics job> \
  scripts/run/hero_steps_vista.sbatch
# then EXPECT_STEP=40,SEG_STOP=80 and EXPECT_STEP=80,SEG_STOP=120, each afterok on the previous

INEXOR_RUNS=$SCRATCH/inexor_runs SRC_RUN=$SCRATCH/inexor_runs/hero-k120 K_STEPS=120 \
  PROD_DIR=$SCRATCH/inexor_runs/hero-k120-prod sbatch -A <account> scripts/run/hero_card_vista.sbatch

INEXOR_RUNS=$SCRATCH/inexor_runs SRC_RUN=$SCRATCH/inexor_runs/hero-k120 K_STEPS=120 \
  PROD_DIR=$SCRATCH/inexor_runs/hero-k120-prod sbatch -A <account> scripts/run/hero_export_vista.sbatch

INEXOR_RUNS=$SCRATCH/inexor_runs sbatch -A <account> scripts/run/gh_single_card_vista.sbatch
```

Notes:

- The scripts exit with an error if a required variable is unset. Pass the account with
  `sbatch -A`. Pass per-submission values with `--export=ALL,VAR=...` or in the environment,
  as shown above.
- The driver cards and sampler CSVs go to `runs/v2/` inside the checkout (created if absent).
- To chain segments, use `--dependency=afterok:<previous job id>`. `sbatch --parsable`
  prints the job id alone, so you can capture it. `EXPECT_STEP` must be the step the previous
  segment stopped at; a mis-chained segment is refused.
- The card and export jobs read the checkpoint independently, so they can run concurrently.
  Both check the checkpoint's fingerprint on the node before loading anything.
- The card and export legs (and the card job's test leg) run in the `gpu` env with
  `env JAX_PLATFORMS=cpu`. The GPU legs unset `JAX_PLATFORMS`.
- To stream host-resident state, the scripts set `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` (900 on
  gb, 160 on gh), plus `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`,
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_CLIENT_MEM_FRACTION=0.95`, and
  `JAX_COMPILATION_CACHE_DIR=$SCRATCH/inexor-xla-cache`.
- Host memory is bound to the CPU NUMA nodes with `numactl --membind=0,1` (gb) or
  `--membind=0` (gh), and the driver checks the binding (`--membind-nodes`). The stepping
  scripts request `--signal=B:USR1@900`, which makes the driver dump every thread's stack
  900 s before the time limit.

**Changes needed for another site:** partition names (`-p gb` / `-p gh`) and walls (`-t`);
`$SCRATCH` (the compilation cache and the examples use it); the pixi location
(`$HOME/.pixi/bin` is prepended to `PATH`); the NUMA node lists in `numactl --membind` and
`--membind-nodes`, which must name the nodes that have CPUs; `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB`
for your host memory; the machine sizes that are hard-coded for a gb node (1026 GB host,
199 GB per card) in `device_ics.py` and in `device_run.py preflight`'s planner call, and
`--host-gb 116 --device-gb 96` in the gh script's planner leg; the preflight's `--need-gb
1500` free-space requirement and its `lfs quota` query (Lustre); and `--cards` / the card
count if a node does not have four GPUs.

## `python -m inexor.export`: any checkpoint to particles

```bash
python -m inexor.export CHECKPOINT_DIR OUT_DIR [--dtype float32|float64] [--chunk-bricks 1024] \
    [--a A [--omega-m OM] [--h H] | --d-time]
```

`CHECKPOINT_DIR` is any directory with a T9 `manifest.json`: `ckpt/gen0`, `ckpt/gen1`, or an
IC directory. Unlike `realization.py export`, this does not need the run's configuration and
does not check a fingerprint or the step. By default, velocities are written in peculiar km/s
at the epoch the checkpoint records:

- `--a` overrides the recorded epoch, and `--omega-m` / `--h` override single cosmology
  fields (these need an epoch);
- `--d-time` writes the engine's native velocity, dx/dD in Mpc/h per unit growth factor;
- a directory that records no epoch (an IC directory, for example) falls back to D-time
  velocities and says so.

The output is `x.npy`, `v.npy` and `export.json`. The header is removed first and written
last, so a directory without it is an interrupted export. `inexor.export.load_particles`
refuses to read one.
