# Running inexor

The portable interface is two drivers and two library entry points:

- `scripts/run/realization.py`, the CPU driver: every phase (ICs, steps, P(k) card, export)
  on a multi-core CPU, plus the card and export of any run's checkpoint;
- `scripts/run/device_run.py`, the GPU driver: steps, ICs, card and export on the GPUs of one
  node, or across nodes under MPI;
- `python -m inexor.plan`, the memory planner (see [getting_started.md](getting_started.md#sizing-a-bigger-run)
  and [configuration.md](configuration.md#memory-planner-python--m-inexorplan));
- `python -m inexor.export`, any checkpoint to plain particle arrays (below).

The Slurm job scripts in `scripts/run/` chain the drivers into the jobs that produced the
paper's runs on TACC Vista. Commands below omit the `pixi run --frozen` (or
`pixi run -e gpu`) prefix.

## The CPU driver: `realization.py`

```bash
python scripts/run/realization.py {ics,run,card,export} --workdir DIR [--config PRESET] [flags]
```

| Phase | What it does | Reads | Writes (in `--workdir`) |
|---|---|---|---|
| `ics` | 2LPT initial conditions, streamed | nothing | `t9_slab_NNNN.npz`, `manifest.json`, `realization_ics.json` |
| `run` | steps the schedule, or one segment of it, with checkpoints | the ICs, or the newest checkpoint in `ckpt/` | `ckpt/gen0`, `ckpt/gen1`, `realization_run_<from>_<to>.json` |
| `card` | P(k) against bin-averaged linear theory | newest checkpoint (or `--ic-dir`) | `realization_pk.json` (`realization_pk_ics.json` with `--ic-dir`) |
| `export` | writes the particles as plain `.npy` arrays | newest checkpoint | `export/{x,v}.npy`, `export/export.json`, `realization_export.json` |

Each phase runs in its own process. A process's peak memory never resets, so this is the only
way to read each phase's peak host memory separately, and a queue time limit costs one phase
rather than the whole run. Checkpoints always go to `<workdir>/ckpt`.

### Flags

| Flag | Default | Phases | Meaning |
|---|---|---|---|
| `--config` | `cdev8` | all | preset: `smoke`, `cdev8`, `cdev8-tile32`, `cdev`, `cgh64`, `c-1024`, `c-gh`, `c-hero`, `c-8192` ([configuration](configuration.md#presets)) |
| `--workdir` | required | all | the run's directory |
| `--k-steps` | 40 | run, card, export | steps from `--a-init` to a = 1 (fingerprinted) |
| `--a-init` | 0.1 | all | starting scale factor, for the ICs and the step grid (fingerprinted) |
| `--growth2` | `lcdm` | all | second-order growth for the 2LPT ICs and the integrator weights (`lcdm` or `eds`) |
| `--n-part`, `--n-fine`, `--n-coarse`, `--buf` | preset | all | resolution overrides; the geometry and split error terms are printed. `--n-fine` re-derives the tile buffer unless `--buf` is given. All are fingerprinted, so `run`, `card` and `export` of one realization must pass the same values as its `ics` |
| `--fine-dtype` | `float64` | run, card, export | precision of the fine tiles, their FFTs and the kick (fingerprinted) |
| `--coarse-match-order` | 3 | all | assignment order the coarse match factor divides out (fingerprinted when not 2) |
| `--drift-rounding` | `stochastic` | all | how drifts land on the position lattice; `nearest` is the v0.1.0 rule (fingerprinted when stochastic, so the card and export of a v0.1.0 run pass `nearest`) |
| `--tile-workers` | 16 | run, card | worker processes. More than 1 means a pooled run, with the state in shared memory |
| `--slack`, `--arena-frac` | 0.20, 0.20 | run, card, export | allocation layout used when a state is loaded |
| `--alloc-margin` | 0.10 | run, card, export | allocation margin used when a state is loaded |
| `--seed` | 0 | ics | random seed |
| `--generator` | `host` | ics | `device` generates the ICs on the GPUs (in the `gpu` env) |
| `--pk-table` | EH98 | ics | make the ICs from a tabulated z = 0 linear P(k) (`scripts/run/camb_linear_pk.py`; see [configuration](configuration.md#linear-power-spectrum)). Refused in the other phases: `run`, `card` and `export` read the table the ICs and checkpoints carry |
| `--noise`, `--pencil-batch` | `device`, 1 | ics | options for the device generator |
| `--bucket-cells` | 2 | ics | position bucket side in particle cells (recorded in the manifest) |
| `--keep-stage` | off | ics | keep the `stage/` intermediates |
| `--slab` | 32 | ics, card | slab depth for the streamed transforms |
| `--stop-at` | end | run | absolute step to stop before. Must be a multiple of `--checkpoint-every` |
| `--checkpoint-every` | 5 | run | checkpoint cadence, in steps |
| `--migrate-pooled` / `--serial-migrate` | auto | run | run the migrate on the pool, or serially |
| `--eject-kernel` | `jax` | run | `jax` or `numpy` |
| `--phase-instrument` | `time` | run | `time` (seconds per phase) or `peak` (per-phase host high-water mark, Linux only) |
| `--allow-partial` | off | card, export | accept a checkpoint that is neither the last step nor a snapshot (its positions are half a step past its epoch) |
| `--export-dir` | `<workdir>/export` | export | output directory |
| `--chunk-bricks` | 1024 | export | bricks decoded per chunk |
| `--min-weight` | 100 | card | drop bins with this many modes or fewer |
| `--k-max`, `--n-bins` | half Nyquist, 64 | card | fix the binning to `--n-bins` bins over [0, `--k-max`]. `--n-bins` has no effect without `--k-max` |
| `--ic-dir` | none | card | card the ICs in this directory at a = `--a-init` |
| `--card-pool` / `--serial-card` | pooled if `--tile-workers` > 1 | card | paint on a worker pool (same result bit for bit), or serially |
| `--heartbeat` | 60 | card, export | seconds between progress lines (0 turns them off) |

The `realization_*.json` cards record each phase's wall time, peak host memory and
configuration. `--tile-workers` sets the size of the worker pool, not the number of cores:
each worker's XLA CPU runtime still uses the cores it finds.

### Refusals you will meet

- **CPU backend required.** `run`, `export` and `card` (and `ics` unless `--generator device`)
  exit with `FATAL: non-CPU backend` if JAX's first device is not a CPU. In the `gpu` env,
  prefix them with `JAX_PLATFORMS=cpu`.
- **ICs made at a different epoch or growth.** Starting from ICs, `run` (and `card
  --ic-dir`) refuses ICs whose manifest records a different `--a-init` or `--growth2`.
  Regenerate the ICs, or pass the matching value.
- **A linear P(k) table for another run.** `ics --pk-table` refuses a table at another
  cosmology, at z != 0, with sigma8 off the cosmology's by more than 1e-3, or not covering
  k = 1e-4 to 1e2 h/Mpc; `run`, `card` and `export` re-check the copy the ICs or checkpoint
  embed, and refuse one whose sha256 no longer matches its arrays.
- **`--stop-at` off a checkpoint boundary.** The stop step must be a multiple of
  `--checkpoint-every`. With no `--stop-at`, the stop step is `--k-steps`, so `--k-steps` must
  be a multiple of it as well. A stop at or before the step you are resuming from is also
  refused.
- **Products from a mid-run checkpoint.** `card` and `export` refuse a checkpoint that is
  neither the last step nor a snapshot unless you pass `--allow-partial`: the drift is fused
  across steps, so its positions are half a step past its epoch. A card of the ICs
  (`--ic-dir`) is not refused.
- **Resume onto a different configuration.** Loading a checkpoint fails on a fingerprint
  mismatch (see below).
- **Empty card.** `card` fails if no bin has more than `--min-weight` modes. Widen the bins
  (`--k-max` with fewer `--n-bins`) or lower `--min-weight`.
- `card` and `export` with no checkpoint: ``no checkpoint to read; run `run` first``.
- `card --ic-dir` pointed at a checkpoint is refused (it would be scored at the wrong epoch).
  `--n-coarse` coarser than the preset and a non-power-of-two `--n-part` are refused.
- `run` on a checkpoint already at `--k-steps` prints `NOTHING TO DO` and exits 0.

## Segments and resume

`run --stop-at N` advances to absolute step N and stops. The next `run` on the same
`--workdir` resumes from the newest checkpoint and prints which source it used:

```bash
python scripts/run/realization.py run --config cdev8 --workdir $W --stop-at 20
python scripts/run/realization.py run --config cdev8 --workdir $W          # 20 -> 40
```

- Each segment receives the full schedule plus a stop step, never a truncated schedule, so the
  segments together reproduce the uninterrupted run bit for bit.
- Checkpoints alternate between two generations, `ckpt/gen0` and `ckpt/gen1`. A generation's
  `manifest.json` is removed first and written last, so a checkpoint torn mid-write is skipped
  and the other generation survives. On resume, the first write goes to the generation that
  was *not* loaded, so the resume point stays intact until a newer checkpoint is complete. The
  loader picks the generation with the higher recorded step.
- **Fingerprint.** Every checkpoint records a hash of the geometry and kernel fields (box,
  particle count, meshes, tile, buffer, split scale, paint orders, dtypes, the coarse match
  order and other kernel parameters), the integrator coefficients as raw float64 bytes, and the snapshot steps if any. The
  coefficients depend on the cosmology, `--a-init`, `--k-steps` and `--growth2`, so a resume,
  card or export must use the same values of all of these, and the same geometry overrides
  and `--fine-dtype`. Execution and layout settings are not part of the hash
  (`--tile-workers`, `--slack`, `--arena-frac`, `--alloc-margin`, `--eject-kernel`,
  `--checkpoint-every`, `--y-blocks`), so they may change between segments. The coefficients
  come from numerical integration, and their bytes can differ between platforms' math
  libraries, so a checkpoint can be refused on a different platform (for example, one written
  on Linux and read on macOS) even with an identical configuration.

## The GPU driver: `device_run.py`

`device_run.py` runs a preset on the GPUs. The particle state stays in host memory, and the
coarse paint and solve, the tile loop and the migrate all run on the cards, streamed slab by
slab. It also generates ICs and makes the card and export on the cards, on one node or across
nodes.

| Subcommand | What it does |
|---|---|
| `preflight` | Checks everything before the expensive part and exits 2 on any refusal: device count and backend, the allocator's stats, the IC manifest (particle count, slab count, `growth2`), the planner's host and load peak against the CPU NUMA nodes' memory, the NUMA memory binding, and free scratch space |
| `run` | Loads the ICs or resumes a checkpoint, then steps to `--stop-at`. Prints one line per phase boundary and a heartbeat, and rewrites a JSON run card (`--card`) at every boundary. On failure, the traceback and device memory stats go into the card |
| `ics` | Device ICs into `--workdir`, with one boundary per generator stage. Under `--comm mpi` every rank generates its own brick slabs and rank 0 writes the manifest; the files are the same at any rank, card and `--y-blocks` count. Refuses a directory that already holds a manifest |
| `card` | The P(k) card of the newest checkpoint in `--checkpoint-dir`, painted and transformed on the cards; rank 0 writes it to `--out` in the same JSON wrapper as `realization.py card` (summary under `summary`, with `transform: "cards"`). The paint is bitwise the CPU card's; the spectrum differs from the CPU card's at the FFT's rounding |
| `export` | The particle export of that checkpoint, decoded on the cards and written as one part per rank to `--export-dir` (format `inexor-particles-2`, see [outputs](outputs.md#one-part-per-rank)); its parts in rank order are the CPU export's bytes. Refuses a non-empty `--export-dir` |
| `summarize` | Reads a run card together with the job's sampler CSVs (`--gpu-csv`, `--mem-csv`) and writes per-phase GPU memory, host memory and utilization to `--out`. Does not import jax |

**Start epoch.** `run` starts the schedule at the epoch recorded in the ICs' manifest (or,
when resuming, at the one recorded in the checkpoint), and `card` and `export` take the
checkpoint's. `--a-init` is optional; if given and different, the run is refused.

Flags of `preflight` and `run`:

| Flag | Default | Meaning |
|---|---|---|
| `--preset`, `--workdir`, `--card` | required | preset, IC directory (read only), run card JSON path |
| `--cards` | 4 | GPUs to use |
| `--y-blocks` | auto | cut each x-slab's card work into this many y-blocks (see below) |
| `--slack`, `--alloc-margin`, `--arena-frac` | 0.10, 0.10, 0.01 | allocation layout |
| `--fine-dtype` | `float64` | precision of the fine tiles (fingerprinted) |
| `--drift-rounding` | `stochastic` | how drifts land on the position lattice; `nearest` is the v0.1.0 rule (fingerprinted when stochastic) |
| `--growth2` | `lcdm` | must match the ICs |
| `--n-fine` | preset | fine mesh cells per side, the tile buffer re-derived (as `realization.py --n-fine`); the ICs must be made with the same, and the run's card and export must pass it |
| `--a-init` | the ICs' or checkpoint's | refused if given and different |
| `--membind-nodes` | none | refuse unless the process runs under `numactl --membind` on these NUMA nodes |
| `--scratch` (preflight) | `$SCRATCH` or `/tmp` | filesystem whose free space is reported |
| `--host-margin` (preflight) | 0.95 | refuse if the planner's host or load peak exceeds this fraction of the CPU nodes' memory |
| `--host-gb`, `--device-gb` (preflight) | 1026, 199 | the planner's host memory per node and memory per card (defaults: a Vista gb node) |
| `--n-nodes` (preflight) | 1 | nodes the run spans; the planner prices the busiest node |
| `--allow-cpu` (preflight) | off | accept the CPU backend |
| `--stop-at` (run) | required | absolute step to stop before |
| `--k-steps` (run) | 40 | length of the whole schedule |
| `--checkpoint-dir`, `--checkpoint-every` (run) | none, 0 | 0 = no checkpoints. Otherwise `--stop-at` must be a multiple of `--checkpoint-every` |
| `--expect-step` (run) | 0 | 0 = start from the ICs, refused if `--checkpoint-dir` already holds a checkpoint. N = resume from the newest checkpoint, refused unless it is at step N |
| `--timed-every N`, `--timed-last`, `--timed-all` (run) | off | synchronized per-pass timing breakdown on every N-th step (absolute step k with (k + 1) % N == 0), the last step, or every step. Untimed steps run without the synchronizations, so their wall time is the step's cost |
| `--snapshot-z`, `--snapshot-dir` (run) | none | synchronized snapshots at the step boundaries nearest these redshifts (comma list), written as `<snapshot-dir>/step<NNNN>/gen0` in checkpoint format. The step before each drifts only its trailing half, so the run differs from one without snapshots by one position rounding per snapshot. Recorded in the checkpoints and their fingerprint, so a resume must give the same `--snapshot-z`. Not under the ICs or `--checkpoint-dir` |
| `--card-sample-ms` (run) | 10 | sample every card's allocator bytes in use and the process's resident memory this often (0 = off); each boundary record carries `card_phase_peak` and `rss_phase_peak`, the largest since the previous boundary (`rss_phase_peak` is None off Linux) |
| `--smaps`, `--numa-maps` (run) | off | at every boundary, resident memory by mapping kind and the largest mappings from `/proc/self/smaps`, or the process's pages per NUMA node (slow on a large state; Linux only) |
| `--cache-probe`, `--trim-probe` (run) | off | after the run, host memory as the compilation and program caches are cleared, then after `malloc_trim(0)` |
| `--drop-ic-cache` (run) | off | drop each IC slab's page cache as it is read |
| `--off-node-gb` (run) | 8 | with `--membind-nodes`: GB of the loaded state allowed on CPU-less (HBM) NUMA nodes |
| `--beat` (run) | 60 | heartbeat seconds |
| `--comm` (run) | `serial` | `mpi`: one rank per process across nodes (see below) |
| `--comm-timeout` (run) | 1800 | seconds a rank may wait at one exchange before it aborts the job |

Flags of `ics`: `--preset`, `--workdir`, `--card`, `--cards` (4), `--seed` (0), `--a-init`
(0.1), `--growth2`, `--pk-table` (EH98; as `realization.py ics`), `--f-nl` (0, local f_NL),
`--window` (1, emission window in brick slabs), `--bucket-cells` (2), `--slab` (32),
`--pencil-batch` (1), `--batch-planes` (16, planes per rank in each plane <-> pencil exchange;
the transient host memory of an exchange), `--y-blocks` (auto, emission units per destination
slab), `--membind-nodes`, `--beat`, `--comm`, `--comm-timeout`. It has no `--n-fine`: ICs at a
non-preset fine mesh come from `realization.py ics --generator device --n-fine N`.

Flags of `card` and `export`: `--preset`, `--checkpoint-dir`, `--k-steps` (required), `--card`
(this process's run card), `--cards`, layout, `--fine-dtype`, `--growth2`, `--n-fine` and
`--a-init` as above (they must match the run), `--membind-nodes`, `--off-node-gb`, `--beat`,
`--comm`, `--comm-timeout`, and `--expect-step` (required: the newest checkpoint must be at
this step). `card` adds `--out` (required), `--k-max` / `--n-bins` / `--min-weight` / `--slab`
as `realization.py card`; `export` adds `--export-dir` (required), `--dtype` (float32) and
`--d-time` (native velocities instead of km/s at the checkpoint's epoch). Both refuse a
checkpoint that is neither the last step nor a snapshot (read from its manifest before the
load) unless `--allow-partial`; a snapshot directory (`<snapshot-dir>/step<NNNN>`) is a
`--checkpoint-dir` like any other. Neither writes under the checkpoint directory.

`run` refuses to write a run card, checkpoint or snapshot anywhere under the IC directory.

**Y-blocks.** Every card working set that would otherwise hold a whole brick x-slab (the tile
window, the destination census, the device migrate and the fused repack) works on (x-slab,
y-block) units: a y-block is a run of whole tile rows, and a tile window holds its block plus
the buffer bricks either side. By default the count is automatic: the fewest blocks that keep
a unit within the rows of a 4096^3 x-slab (`decomp.auto_y_blocks`), so 1 up to 4096^3
particles and 4 at 8192^3; the IC emission applies the same rule to its destination slabs.
`--y-blocks N` sets a count. The bytes are the same at any count, so a checkpoint resumes at
another one. `python -m inexor.plan ... --backend device` prints the count it priced and the
smallest that fits. More units mean more, smaller kernel launches per step.

**Across nodes.** `run`, `ics`, `card` and `export` take `--comm mpi`: one rank per process,
one process per node, under MPI:

```bash
mpiexec -n 2 python -m mpi4py scripts/run/device_run.py run --preset c-gh --workdir $ICS \
    --card run.json --cards 1 --k-steps 120 --stop-at 40 --checkpoint-dir $CKPT \
    --checkpoint-every 40 --comm mpi
```

- Each rank loads its own brick slabs of the ICs or of the checkpoint, writes its own run card
  (`run.rank<r>.json`), and prefixes its log lines with `[rank r]`. `python -m mpi4py` makes a
  rank that raises end the whole job.
- All ranks write each checkpoint together, and the files are the same bytes at any rank
  count: a checkpoint from N nodes resumes on M, one included, and the card and export read it
  as a single-node checkpoint.
- Ranks x cards may not exceed the tile planes, and each rank needs at least 2r + 1 brick
  slabs for a drift that reaches r slabs (3 at r = 1): the migrate hands particles to
  immediate neighbours only.
- Each step's record in a rank's run card carries that rank's exchanges under `ranks.comm`:
  calls, bytes sent and seconds per operation, and `wait_s`, the time blocked waiting on other
  ranks.
- With `--comm mpi`, the card is the same at any rank and card count, and the export's parts in
  rank order are the single-node export's bytes.
- On a cluster, the job scripts launch each rank through `scripts/run/rank_exec.sh
  [--membind NODES] [--samples PREFIX] -- CMD ...`. It replaces `@RANK@` in the arguments and
  in exported variables, so one launch line gives each rank its own run card, checkpoint dir
  or compilation cache. It also starts the node's GPU and NUMA samplers, which stop when the
  command ends, then becomes the command, so the launcher's signals reach it.

**Card and export of a GPU run on the CPU.** The CPU driver reads `<workdir>/ckpt`, so link the
GPU run's checkpoint directory there, in a product directory outside the IC directory:

```bash
ln -sfn $CKPT $PROD/ckpt
JAX_PLATFORMS=cpu python scripts/run/realization.py card --config c-1024 --workdir $PROD \
    --k-steps 120 --slack 0.10 --arena-frac 0.01 --tile-workers 8 --card-pool
```

`--config`, `--k-steps`, `--a-init`, `--growth2`, `--n-fine` and `--fine-dtype` must match the
run, or the fingerprint check fails. Pass the GPU run's `--slack` and `--arena-frac` as well
(its defaults are 0.10 and 0.01, the CPU driver's 0.20 and 0.20): a checkpoint stores particle
membership but not the allocation, and the loader rebuilds the allocation from these flags, so
the CPU defaults provision far more memory than the run used.

## Slurm job scripts

The scripts in `scripts/run/*_vista.sbatch` are the jobs used for the paper's runs on TACC
Vista: partition `gh` (one GH200 per node, about 116 GB of host memory), `gb` (four GB200s per
node, about 1026 GB of host memory) and `gg` (CPU nodes). Each runs checks (preflight, tests, a
small control run or a byte-for-byte gate) before the expensive leg and stops if they fail.
Every production run leg times every fifth step (`--timed-every 5`); the untimed steps carry
the step's cost.

| Script | Partition | What it runs | Required variables | Optional variables (default) |
|---|---|---|---|---|
| `hero_ics_vista.sbatch` | gb | a preset's ICs on four cards (`realization.py ics --generator device`), after `device_ics.py preflight` and a 32^3 device-vs-host smoke; refuses an existing manifest, checks the new one's particle and slab counts | `INEXOR_RUNS`, `IC_DIR` (or `HERO_IC_DIR`) | `PRESET` (`c-hero`), `NEED_GB` (1500, free scratch), `PK_TABLE` |
| `hero_steps_vista.sbatch` | gb | one segment of the 4096^3 `c-hero` run on its 120-step schedule (`device_run.py run`), checkpointing every 20 steps into `$REAL_DIR/hero-ckpt`; submitted three times as 40-step segments, after a 32^3 smoke and a 512^3 control | `INEXOR_RUNS`, `HERO_IC_DIR`, `REAL_DIR`, `EXPECT_STEP`, `SEG_STOP` | |
| `hero_card_vista.sbatch` | gb | P(k) card of the final checkpoint (`realization.py card` on the CPU, pooled), written to `$PROD_DIR` | `INEXOR_RUNS`, `CKPT_DIR` (or `SRC_RUN`, read as `$SRC_RUN/hero-ckpt`), `K_STEPS`, `PROD_DIR` | `PRESET` (`c-hero`) |
| `hero_export_vista.sbatch` | gb | export of the same checkpoint (`realization.py export` on the CPU) to `$PROD_DIR/export` | as the card job | `PRESET` (`c-hero`) |
| `gh_single_card_vista.sbatch` | gh | realizations on one GH200: one set of device ICs, then a run (checkpointed once, at its end) and a CPU card per step count in `KS`, under `$INEXOR_RUNS/gh-1card-<preset>-<job id>/k<K>/`; a run that fails gets no card and the next step count still runs | `INEXOR_RUNS` | `PRESET` (`c-1024`), `KS` (120; e.g. `"40 60 80 120 160"` for a step-count ladder), `N_FINE` (the preset's; ICs, runs and cards all take it), `PK_TABLE`, `SLACK` (0.10), `ALLOC_MARGIN` (0.10), `ARENA_FRAC` (0.01) |
| `multinode_ics_vista.sbatch` | gh or gb, `-p` / `-N` at submission (default gh, 2 nodes) | device ICs across the nodes (`device_run.py ics --comm mpi`) into `$IC_DIR`, then a check of the manifest's particle and slab counts. With `SMALL_PRESET`, first a check that its ICs on one rank and on every rank are the same files | `INEXOR_RUNS`, `IC_DIR` (new) | `PRESET` (`c-gh`), `SEED` (0), `A_INIT` (0.1), `F_NL` (0), `WINDOW` (1), `Y_BLOCKS` (auto), `BATCH_PLANES` (16), `SMALL_PRESET`, `PK_TABLE`, `QUIET_S` (900) |
| `multinode_steps_vista.sbatch` | gh or gb, `-p` / `-N` at submission (default gh, 2 nodes) | one realization across the nodes from ICs made elsewhere, checkpointing every `EVERY` steps into `$REAL_DIR/ckpt`; a lost job is resubmitted with `EXPECT_STEP` at its newest checkpoint. With `CONTROL_REF`, first a control: its older generation (a run that passed the byte gate) is resumed to the newer one's step and must match it byte for byte. Ends by checking the newest checkpoint's step and particle count; with `PROD_DIR`, a passing check is followed in the same allocation by `multinode_products_vista.sbatch` on that checkpoint. The 8192^3 run is two 20-step jobs of this script on 8 gb nodes | `INEXOR_RUNS`, `IC_DIR`, `REAL_DIR` | `PRESET` (`c-gh`), `K` (120), `EXPECT_STEP` (0), `STOP` (`K`), `EVERY` (20), `Y_BLOCKS` (auto), `CONTROL_REF` with `CONTROL_ICS`, `CONTROL_PRESET` (`c-1024`), `SLACK` (0.10), `ALLOC_MARGIN` (0.10), `ARENA_FRAC` (0.01), `QUIET_S` (600), `PROD_DIR` with `WITH_EXPORT` (1), `TIMED` (`--timed-every 5`), `SNAPSHOT_Z` (none; e.g. `2,1,0.5`: snapshots in `$REAL_DIR/snapshots`, and with `PROD_DIR` the products job on each into `$PROD_DIR/step<NNNN>`) |
| `multinode_products_vista.sbatch` | gh or gb, `-p` / `-N` at submission (default gh, 2 nodes) | the P(k) card (`$PROD_DIR/pk.json`) and the export (`$PROD_DIR/export`, one part per rank) of a checkpoint across the nodes, on the cards. With `SMALL_CKPT`, first a check that its card and export on one rank and on every rank match. `REF_EXPORT` (an export of the same checkpoint made another way) must have the same crc32; `REF_CARD`'s per-bin difference is printed, not gated | `INEXOR_RUNS`, `CKPT_DIR`, `PROD_DIR` | `PRESET` (`c-gh`), `K` and `EXPECT_STEP` (from the newest checkpoint), `ALLOW_PARTIAL` (0), `WITH_EXPORT` (1; 0 = the card only), `SMALL_CKPT` with `SMALL_PRESET` (`c-1024`), `REF_EXPORT`, `REF_CARD`, `SLACK` (0.10), `ALLOC_MARGIN` (0.10), `ARENA_FRAC` (0.01), `QUIET_S` (900) |
| `multinode_gate_vista.sbatch` | gh, 2 nodes | the multi-node byte gate: the single-node run on each node and the 2-rank run, compared file by file at the split step and at the end (`$INEXOR_RUNS/mn-gate-<preset>-<job id>/`); a split-step mismatch reruns segment 1 with deterministic ops and stops | `INEXOR_RUNS` | `PRESET` (`c-1024`), `K` (120), `SPLIT` (20), `STOP` (`K`), `EVERY` (`SPLIT`), `SLACK` (0.10), `ALLOC_MARGIN` (0.10), `ARENA_FRAC` (0.01), `QUIET_S` (600) |
| `multinode_scaling_vista.sbatch` | gh or gb, `-N` at submission (default gh, 16 nodes) | strong scaling inside one allocation: `REF_CKPT`'s older generation resumed to its newer one's step at each rank count in `RANK_COUNTS`, each leg's checkpoints compared byte for byte with `REF_CKPT` | `INEXOR_RUNS`, `REF_CKPT` (two generations), `REF_ICS`, `OUT_DIR` | `PRESET` (`c-gh`), `RANK_COUNTS` (`"2 4 8 16"`), `Y_BLOCKS` (auto), layout (`REF_CKPT`'s own: `SLACK`, `ALLOC_MARGIN`, `ARENA_FRAC`), `QUIET_S` (600) |
| `memory_diag_vista.sbatch` | gh, 2 nodes | where host memory goes: a one-GPU run outside MPI with `--smaps` and both probes (its checkpoint compared with `ONE_CKPT` if given), then `STEPS` steps of `REF_CKPT` on two ranks, once as the multi-node jobs run and once with `MPI_OFF` in every rank's environment, the two checkpoints compared | `INEXOR_RUNS`, `ONE_ICS`, `REF_CKPT` (two generations), `REF_ICS`, `OUT_DIR`, `MPI_OFF` (`VAR=value ...`) | `ONE_CKPT`, `ONE_PRESET` (`cgh64`), `ONE_K` (120), `ONE_STOP` (`ONE_K`), `ONE_FINE` (`float64`), `PRESET` (`c-gh`), `STEPS` (10), `SLACK`, `ALLOC_MARGIN`, `ARENA_FRAC`, `QUIET_S` (600) |
| `yblocks_measure_vista.sbatch` | gb | per preset, device ICs, then the same steps at every y-block count (1, 2, 4, ... up to the preset's tile rows, at most `Y_MAX`); every count's checkpoints must equal the 1-block run's | `INEXOR_RUNS` | `PRESETS` (`"cgh64 c-1024"`), `K` (120), `STOP` (10), `EVERY` (5), `Y_MAX` (8), `QUIET_S` (600) |
| `mpi_probe_vista.sbatch` | gh or gb, `-p` at submission, 2 nodes | a two-node MPI probe (`scripts/run/mpi_probe.py`) under the host MPI module: identity, bandwidth, signal, abort and kill legs | none | `INEXOR_RUNS` (`$SCRATCH/inexor_runs`), `MPI_MODULE` (`mvapich-plus/5.1.0`) |

The comparison jobs in `scripts/compare/` (`disco_crosscheck_gh_vista.sbatch`,
`gh_crossover_vista.sbatch`, `gh_same_ics_vista.sbatch` on gh, `quant_grid_gg_vista.sbatch`
on gg) are described in their headers and in [scripts/compare](../scripts/compare/README.md).

Every script also needs `INEXOR_SRC` (the inexor checkout), or it must be submitted from the
checkout. Examples of the 4096^3 run on one gb node:

```bash
INEXOR_RUNS=$SCRATCH/inexor_runs IC_DIR=$SCRATCH/inexor_runs/c-hero-lcdm \
  sbatch -A <account> scripts/run/hero_ics_vista.sbatch

INEXOR_RUNS=$SCRATCH/inexor_runs HERO_IC_DIR=$SCRATCH/inexor_runs/c-hero-lcdm REAL_DIR=$SCRATCH/inexor_runs/hero-k120 \
  sbatch -A <account> --export=ALL,EXPECT_STEP=0,SEG_STOP=40 --dependency=afterok:<ics job> \
  scripts/run/hero_steps_vista.sbatch
# then EXPECT_STEP=40,SEG_STOP=80 and EXPECT_STEP=80,SEG_STOP=120, each afterok on the previous

INEXOR_RUNS=$SCRATCH/inexor_runs SRC_RUN=$SCRATCH/inexor_runs/hero-k120 K_STEPS=120 \
  PROD_DIR=$SCRATCH/inexor_runs/hero-k120-prod sbatch -A <account> scripts/run/hero_card_vista.sbatch

INEXOR_RUNS=$SCRATCH/inexor_runs SRC_RUN=$SCRATCH/inexor_runs/hero-k120 K_STEPS=120 \
  PROD_DIR=$SCRATCH/inexor_runs/hero-k120-prod sbatch -A <account> scripts/run/hero_export_vista.sbatch
```

The 8192^3 run on 8 gb nodes, 40 steps to z = 0 in two 20-step jobs, the second followed by
its card and export:

```bash
R=$SCRATCH/inexor_runs
INEXOR_RUNS=$R IC_DIR=$R/c-8192-ics PRESET=c-8192 \
  sbatch -A <account> -p gb -N 8 -n 8 -t 03:00:00 scripts/run/multinode_ics_vista.sbatch
INEXOR_RUNS=$R IC_DIR=$R/c-8192-ics REAL_DIR=$R/c-8192-k40 PRESET=c-8192 K=40 EVERY=20 \
  ALLOC_MARGIN=0.02 STOP=20 sbatch -A <account> -p gb -N 8 -n 8 -t 12:00:00 \
  scripts/run/multinode_steps_vista.sbatch
INEXOR_RUNS=$R IC_DIR=$R/c-8192-ics REAL_DIR=$R/c-8192-k40 PRESET=c-8192 K=40 EVERY=20 \
  ALLOC_MARGIN=0.02 EXPECT_STEP=20 STOP=40 PROD_DIR=$R/c-8192-k40-product \
  sbatch -A <account> -p gb -N 8 -n 8 -t 12:00:00 scripts/run/multinode_steps_vista.sbatch
```

Notes:

- The scripts exit with an error if a required variable is unset. Pass the account with
  `sbatch -A`, and per-submission values with `--export=ALL,VAR=...` or in the environment.
- Each script's header gives its legs, wall-time estimate and further examples.
- The driver run cards and sampler CSVs go to `runs/cards/` inside the checkout (created if
  absent); multi-node jobs also keep each leg's output under their work directory's `legs/`.
- To chain jobs, use `--dependency=afterok:<previous job id>` (`sbatch --parsable` prints the
  job id alone). `EXPECT_STEP` must be the step the previous segment stopped at; a mis-chained
  segment is refused.
- The single-node card and export jobs read the checkpoint independently, so they can run
  concurrently. Both check the checkpoint's fingerprint on the node before loading anything.
- The CPU legs run in the `gpu` env with `env JAX_PLATFORMS=cpu`; the GPU legs unset
  `JAX_PLATFORMS`.
- Before any leg, the `hero_*`, `gh_single_card`, `multinode_*` and `memory_diag` jobs check
  the `gpu` env on the node (`scripts/run/gpu_env.sh`, sourced directly by the single-node
  jobs and through `scripts/run/mpi_env_vista.sh` by the multi-node ones): every file the env's package records list must exist, and jax, inexor and
  every inexor module must import with a GPU visible. Otherwise, or with `REBUILD_ENV=1`, the
  env is reinstalled from `pixi.lock` through a new, empty package cache under
  `$GPU_ENV_CACHE_ROOT` (default `$SCRATCH`), which guards against files lost to a scratch
  purge.
- To stream host-resident state, the GPU jobs set `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` (900 on
  gb, 160 on gh), plus `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`,
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_CLIENT_MEM_FRACTION=0.95`, and a
  `JAX_COMPILATION_CACHE_DIR` under `$SCRATCH`.
- Host memory is bound to the CPU NUMA nodes with `numactl --membind=0,1` (gb) or
  `--membind=0` (gh), and the driver checks the binding (`--membind-nodes`). The stepping
  scripts request a USR1 signal 900 s before the time limit (`--signal=B:USR1@900`), which
  makes the driver dump every thread's stack.
- The multi-node jobs launch one rank per node with `ibrun`, against the MVAPICH-Plus module
  (`mpi_env_vista.sh` builds mpi4py against it once, into `$SCRATCH`).

**Changes needed for another site:** partition names (`-p gb` / `-p gh`) and walls (`-t`);
`$SCRATCH` (the compilation cache, the env-rebuild cache and the examples use it); the pixi
location (`$HOME/.pixi/bin` is prepended to `PATH`); the NUMA node lists in
`numactl --membind` and `--membind-nodes`, which must name the nodes that have CPUs;
`XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` for your host memory; the machine sizes passed to the
planner and preflight (`--host-gb`, `--device-gb`; `device_ics.py` has a gb node's 1026 GB
host and 199 GB per card built in); `hero_ics_vista.sbatch`'s `NEED_GB` free-space requirement
and `device_ics.py preflight`'s `lfs quota` query (Lustre); `--cards` / the card count if a
node does not have four GPUs; and, for the multi-node jobs, the launcher and MPI module in
`mpi_env_vista.sh`.

### Rehearsal on a laptop

The multi-node job scripts run on a laptop CPU with `REHEARSAL=1`: the same legs at the
`cdev8-tile32` preset (a few steps of a 40-step schedule), under the MPICH of the MPI lane's
throwaway env, with stub `nvidia-smi` and `numactl` so the samplers and binding paths run.
Rehearse the gate first; the other rehearsals start from its ICs and checkpoints:

```bash
REHEARSAL=1 INEXOR_RUNS=runs/rehearsal scripts/run/mpi_lane.sh --run \
    bash scripts/run/multinode_gate_vista.sbatch
G=runs/rehearsal/mn-gate-cdev8-tile32-rehearsal-<pid>
REHEARSAL=1 INEXOR_RUNS=runs/rehearsal IC_DIR=$G/ics CONTROL_REF=$G/mpi/ckpt CONTROL_ICS=$G/ics \
    REAL_DIR=runs/rehearsal/steps-r1 scripts/run/mpi_lane.sh --run \
    bash scripts/run/multinode_steps_vista.sbatch
```

The headers of `multinode_ics_vista.sbatch`, `multinode_products_vista.sbatch`,
`multinode_scaling_vista.sbatch` and `memory_diag_vista.sbatch` give their rehearsal lines,
and the `PLANT_*` variables that make a gate or a leg fail on purpose.
`yblocks_measure_vista.sbatch` rehearses without MPI:
`REHEARSAL=1 INEXOR_RUNS=runs/rehearsal pixi run --frozen bash scripts/run/yblocks_measure_vista.sbatch`.

## `python -m inexor.export`: any checkpoint to particles

```bash
python -m inexor.export CHECKPOINT_DIR OUT_DIR [--dtype float32|float64] [--chunk-bricks 1024] \
    [--a A [--omega-m OM] [--h H] | --d-time] [--allow-partial]
```

`CHECKPOINT_DIR` is any directory with a T9 `manifest.json`: `ckpt/gen0`, `ckpt/gen1`, a
snapshot, or an IC directory. Unlike `realization.py export`, this needs neither the run's
configuration nor a fingerprint check. A run checkpoint that is neither its last step nor a
snapshot is refused (its positions are half a step past its velocities) unless
`--allow-partial` is given. By default, velocities are written in peculiar km/s at the epoch
the checkpoint records:

- `--a` overrides the recorded epoch, and `--omega-m` / `--h` override single cosmology
  fields (these need an epoch);
- `--d-time` writes the engine's native velocity, dx/dD in Mpc/h per unit growth factor;
- a directory that records no epoch (an IC directory, for example) falls back to D-time
  velocities and says so.

The output is `x.npy`, `v.npy` and `export.json`. The header is removed first and written
last, so a directory without it is an interrupted export, and
`inexor.export.load_particles` refuses to read one.
