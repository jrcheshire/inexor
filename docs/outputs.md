# Outputs

Reference for what inexor writes to disk: the T9 slab directories (initial conditions,
checkpoints and snapshots), the P(k) summary card, the portable particle export, and the run
cards the drivers write. Lengths are comoving Mpc/h, wavenumbers h/Mpc, power (Mpc/h)^3.

A run driven by `scripts/run/realization.py --workdir W` (the CPU driver) produces:

```
W/manifest.json, W/t9_slab_0000.npz ...   initial conditions (T9 slab directory)
W/stage/                                  IC intermediates (removed on success unless --keep-stage)
W/ckpt/gen0/, W/ckpt/gen1/                checkpoints (T9 slab directories)
W/export/x.npy, v.npy, export.json        particle export (default --export-dir)
W/realization_*.json                      run cards, one per phase (and per run segment)
```

The GPU driver `scripts/run/device_run.py` takes each location as a flag: the ICs in
`--workdir`, checkpoints in `--checkpoint-dir` (`gen0/`, `gen1/`), snapshots in
`--snapshot-dir` (`step<NNNN>/gen0/`), the P(k) card at `card --out`, the export (one part
per rank) in `export --export-dir`, and each process's run card at `--card`. It refuses to
write a card or checkpoint under the IC directory, and a product under the checkpoint
directory.

## T9 slab directory

Written by `icgen.generate_t9_slabs` / `generate_t9_slabs_device` (ICs) and
`icgen.write_t9_slabs` (checkpoints and snapshots); read by `icgen.load_slot_state`. Schema
string `icgen.SCHEMA = "t9-slabs-2"`.

A directory holds one `t9_slab_{bx:04d}.npz` per x-slab of bricks (`bx` = 0 ..
bricks_per_side - 1; see the preset table in
[configuration.md](configuration.md#presets)) and a `manifest.json`.

### Per-slab arrays

| array | dtype, shape | content |
|---|---|---|
| `meta` | JSON string | slab metadata (below) |
| `occupancy` | int64, `(nb^2 * buckets_per_brick,)` | particle count per bucket, brick-major, for the slab's `nb^2` bricks |
| `off` | uint8, `(n_rows, 3)` | position offset within the particle's bucket, in quanta of `bucket_size / 256` |
| `w` | int16, `(n_rows, 3)` | velocity code; velocity = `w * scale[brick]` |
| `scale` | float64, `(nb^2,)` | per-brick velocity scale, in the engine's native units (Mpc/h per unit growth factor, see [Velocity convention](#velocity-convention)) |

Rows are brick-major: each brick's rows in bucket order, bricks in order. A row's bucket (and
so its position) is implied by where it sits against `occupancy`: the position is
`(bucket_index * 256 + off) * quantum` per axis, with `quantum = box_size / (n_part /
bucket_cells * 256)`.

`meta` fields: `schema`, `bx`, `n_rows`, `bucket_lo` and `brick_lo` (global index of the slab's
first bucket and brick), and `crc32` (a dict with the crc32 of `occupancy`, `off`, `w`,
`scale`).

### Manifest fields common to every slab directory

| field | meaning |
|---|---|
| `schema` | `"t9-slabs-2"` |
| `files` | slab file names, one per x-slab (not necessarily in slab order; the loader orders them by name) |
| `n_particles` | total particles |
| `box_size` | box side, Mpc/h |
| `n_part` | particles per side |
| `bucket_cells` | bucket side in particle cells |
| `bricks_per_side` | bricks per side (= slab files) |
| `provenance` | IC provenance, or the checkpoint record below |

### Write order and refusals

The slabs are written first and the manifest last, so the manifest marks a complete
directory. `write_t9_slabs` removes any existing manifest before rewriting, so a torn
overwrite cannot mix generations. Across ranks every rank writes its own slabs and rank 0
writes the manifest; the files are the same bytes at any rank count.

`load_slot_state` (and `read_manifest`) refuse:

- a directory with no `manifest.json` (an incomplete or interrupted write);
- a manifest whose `schema` is not `"t9-slabs-2"`;
- a file list that does not name every slab exactly once, or a slab whose `meta.bx` differs
  from its file name;
- any array whose crc32 does not match its slab's `meta`;
- slabs whose rows do not add up to the manifest's `n_particles`, or do not match their
  occupancy.

The loader reads the files twice (index, then payload), so only one slab's payload is live at
a time. Capacity (slack, arena) is not stored; it is chosen at load time.

### IC manifest

The IC generators add these fields:

| field | meaning |
|---|---|
| `vel_scale` | global velocity scale at generation (`max|v| / 32767`, native units) |
| `max_displacement` | max 2LPT displacement, Mpc/h |
| `window` | emission window depth, brick slabs (generation refuses if the max displacement reaches it) |
| `a_init` | scale factor of the ICs |
| `order` | LPT order (2) |
| `growth2` | `"lcdm"` or `"eds"` second-order growth |
| `f_NL` | local primordial non-Gaussianity amplitude (`device_run.py ics --f-nl`; 0 otherwise) |
| `fdtype` | generator float dtype (`float32` in the drivers) |
| `slab` | planes per out-of-core FFT slab |
| `ic_stream` | identifier of the noise stream (host-drawn or card-drawn noise differ) |
| `backend` | linear power backend (`eh98` or `table`) |
| `linear_pk` | with `backend: "table"` only: the whole table (`format`, `source`, `sha256`, `z`, `cosmology`, `generator`, `k`, `P`; see [configuration](configuration.md#linear-power-spectrum)) |
| `table_n_points` | nodes of the internal k table |
| `mean_phi2` | mean of phi^2 (null on the device generator at f_NL = 0) |
| `stage_cleanup` | report of removing `stage/`, or `{kept: true, ...}` |
| `provenance` | from the driver: `generator`, `commit`, `host`, `machine`, `numpy`, `when`, `jax`, `x64`, `devices`, `n_devices`, `allocator`, and the XLA variables listed in [configuration](configuration.md#environment-variables); `device_run.py ics` adds `n_ranks` and `comm` |

The device generator also records `generator: "device"`, `emission`, `emission_s`,
`n_devices` (per rank), `pencil_batch` and `stage_s` (per-stage seconds), and, only when
they are above 1, `n_ranks` (generated across nodes) and `emission_y_blocks`. None of these
moves a byte of the slab files: the same seed gives the same files at any rank, card and
y-block count.

### Checkpoints

`engine.run` with `checkpoint_dir` set writes every `checkpoint_every` steps into `gen0/` and
`gen1/` alternately, so the previous generation survives until the new one is complete. A
resumed run first writes the generation it did not load. `engine.load_checkpoint(dir, cfg,
coeffs)` picks the generation with the highest `step` (ignoring a directory without a
manifest or whose `provenance.kind` is not a checkpoint) and refuses one whose fingerprint
differs from the current config and schedule
([configuration](configuration.md#checkpoint-fingerprint)). A state carrying particle ids
cannot be checkpointed (the schema has no ids).

The manifest carries the common fields, `source: "write_t9_slabs"`, and this `provenance`:

| field | meaning |
|---|---|
| `kind` | `"inexor-checkpoint"` |
| `step` | completed steps; the checkpoint's epoch is `a_steps[step]` |
| `n_steps` | steps in the full schedule |
| `cap_shape`, `pad_shape`, `device_shapes` | buffer shapes restored on resume, so the same programs compile (`device_shapes` is `{}` on the CPU lane) |
| `n_arena` | arena rows at write time (the arena size a load gets when it is not given `arena_frac`) |
| `fingerprint` | `engine.checkpoint_fingerprint` of the config, the coefficients and any snapshot steps |
| `a`, `cosmology` | the epoch `a_steps[step]` and the `Cosmology` fields, when `run` was given `epoch=(a_steps, cosmo)` (both drivers do) |
| `linear_pk` | the ICs' tabulated linear P(k), when they were made from one (the drivers pass it on from the IC manifest or the checkpoint they resume) |
| `source` | GPU driver only: the run's ICs, `{ics, a_init, ics_provenance}`, the last holding the IC manifest's `generator`, `commit`, `host`, `when`. The GPU driver's card and export take the schedule's `a_init` from it |
| `snapshot_steps` | the run's snapshot steps, when it has any |
| `snapshot` | `true` on a snapshot |

**Snapshots.** `device_run.py run --snapshot-z Z,...` (`engine.run(snapshot_steps=...,
snapshot_dir=...)`) writes, at the end of each step nearest a requested redshift, a snapshot
`<snapshot_dir>/step<NNNN>/gen0/` in the same format. That step drifts only its trailing half,
so positions and velocities both sit at `a_steps[step]` (`engine.drift_plan`).

**Synchronized states.** The drift is fused across steps, so a regular checkpoint before the
last step holds positions half a step past its epoch. `engine.synchronized(provenance,
n_steps)` is true for the final step (`step >= n_steps`) and for a snapshot, and false
otherwise. The drivers' `card` and `export` and `python -m inexor.export` refuse an
unsynchronized checkpoint unless `--allow-partial` is given. A snapshot cannot be resumed
from; a run resumes from its own `checkpoint_dir`.

## P(k) summary card

`summary.pk_summary_card(st, cfg, cosmo, a_out, ...)` paints the coarse density (TSC, integer
accumulation), takes its spectrum on the host, and bins it against the linear oracle
`D(a_out)^2 P_lin(k)` averaged over each bin's realized modes (never at a bin centre).
`summary.pk_summary_card_cards` does the paint and the transform on the GPUs, across ranks;
its density is bitwise the host card's, and its spectrum differs at the FFT's rounding. Both
return a dict and write nothing. Card id `summary.CARD = "inexor-pk-summary-1"`.

Binning: by default 64 bins of equal width over `[0, k_nyquist / 2]` of the card's coarse
mesh (65 edges); the driver flags `--k-max K --n-bins N` give N bins over `[0, K]` instead, so
runs with different coarse meshes share bins. The k = 0 mode is excluded. Modes are weighted
by their multiplicity in the full grid. The measured power is divided by the TSC window
squared per mode, then the shot noise `V / N` is subtracted. Bins with `n_modes <=
min_weight` (default 100) are dropped from every per-bin list; `k_edges` keeps all requested
edges, so `len(k_edges) - 1` can exceed `n_bins`. A card with no bin left is refused.

| field | meaning, units |
|---|---|
| `card` | `"inexor-pk-summary-1"` |
| `n_coarse`, `box_size`, `n_particles` | mesh cells per side, Mpc/h, total particles |
| `a_out`, `growth_factor` | epoch and `D(a_out)`, with `D(1) = 1` |
| `k_nyquist` | `pi * n_coarse / box_size`, h/Mpc |
| `k_nonlinear` | k where the linear `k^3 P / (2 pi^2)` at `a_out` reaches 1, h/Mpc; null if not reached in the scan |
| `k_nonlinear_scan` | `[1e-3, 10.0]`, the scanned range |
| `shot_noise` | `V / N`, (Mpc/h)^3, subtracted from `p` |
| `deconvolved` | `"tsc"` (TSC window divided out) or null |
| `oracle` | description string of the oracle |
| `linear_pk` | the oracle's linear spectrum: `{"source": "eh98"}`, or the ICs' table's `source` and `sha256` |
| `transform` | `"host"` or `"cards"`: where the spectrum was taken |
| `n_bins` | bins kept |
| `k_edges` | all requested bin edges |
| `k_mean` | mode-weighted mean k per kept bin, h/Mpc |
| `p` | measured power, window-deconvolved and shot-subtracted, (Mpc/h)^3 |
| `p_oracle` | bin-averaged linear power at `a_out`, (Mpc/h)^3 |
| `n_modes` | full-grid mode count per bin |
| `z_profile` | `(p / p_oracle - 1) / sqrt(2 / n_modes)` |
| `window_correction` | mode-mean `W^2` per bin |
| `shot_fraction` | `shot_noise` over the bin's raw (not deconvolved) power |
| `provenance` | caller-supplied dict: empty from `realization.py`; from `device_run.py`, `preset`, `step`, `commit`, `checkpoint`, `n_ranks`, `cards` and, with a tabulated P(k), `linear_pk` |

The card decides no verdict. Linear theory applies only below `k_nonlinear`.
`summary.band_verdict(card, k_max, k_min=0.0, bar=5.0)` returns `max_abs_z` over a band you
name, with `k_min`, `k_max`, `n_bins`, `bar`, `ok` (`max_abs_z < bar`),
`band_reaches_nonlinear` and `k_nonlinear`.

### The card file

Both drivers write the summary under `summary` in one JSON wrapper, card id
`inexor-realization-pk-1`: `realization.py card` to `W/realization_pk.json`
(`realization_pk_ics.json` with `--ic-dir`, which cards the ICs at a = a_init), and
`device_run.py card` to `--out` (rank 0). The wrapper's fields: `config` (preset), `workdir`,
`commit`, `host`, `k_steps`, `growth2`, `when`, `step`, `a_out`, `wall_s`, `load_s`; the CPU
driver adds `machine`, `numpy`, `card_pool_workers`, `pool_spawn_s`, `n_bins_below_k_nl`,
`ic_dir` and `peak_rss_bytes`, and the GPU driver `n_ranks` and `cards`.

### Reading a card

```python
import json
import numpy as np

card = json.load(open("W/realization_pk.json"))["summary"]
k = np.asarray(card["k_mean"])
B = np.asarray(card["p"]) / np.asarray(card["p_oracle"])     # measured / linear
lin = k < card["k_nonlinear"] if card["k_nonlinear"] is not None else np.ones_like(k, bool)

from inexor.summary import band_verdict
print(band_verdict(card, k_max=0.05))
```

## Particle export

`export.write_particles(st, out_dir, dtype=np.float32, a=None, cosmo=None, chunk_bricks=1024,
provenance=None)` writes the state as plain `.npy` arrays. Format id `export.FORMAT =
"inexor-particles-1"`.

| file | dtype, shape | content |
|---|---|---|
| `x.npy` | `dtype` (float32 default), `(N, 3)` | positions, comoving Mpc/h in `[0, box_size)` |
| `v.npy` | `dtype`, `(N, 3)` | velocities (see below) |
| `ids.npy` | int32, `(N,)` | only when the state carries ids (opt-in; the drivers' states do not) |
| `export.json` | JSON | header, removed first and written last (marks a complete export) |

Rows are brick-major (the engine's spatial layout) and carry no Lagrangian identity; row `i`
of one export is not the same particle as row `i` of another. Positions are the engine's own
decode (float64, cast to `dtype`), arena residents included. A file shorter than the state's
particle count is refused at write time.

Header fields: `format`, `n_particles`, `box_size`, `n_part`, `dtype`, `files` (role -> file
name), `crc32` (role -> crc32 of the array bytes), `units` (`position`, `velocity` strings),
`velocity_is_dtime`, `peculiar_velocity_factor`, `a`, `cosmology`, `row_order`, `has_ids`,
`source` (`"write_particles"`), `provenance`. `realization.py export` records `config`,
`step`, `commit` and `workdir` in `provenance`, and `device_run.py export` `preset`, `step`,
`commit`, `checkpoint`, `n_ranks` and `cards`; both add `linear_pk` (the table's `source` and
`sha256`) when the ICs were made from a tabulated P(k). `python -m inexor.export` records
`source_checkpoint` and `epoch_source`.

### Velocity convention

The engine's native velocity is `dx/dD`: Mpc/h per unit linear growth factor. Passing both `a`
and `cosmo` converts to peculiar km/s by the factor

```
v_pec [km/s] = 100 * a * f(a) * D(a) * E(a) * v_D        (export.peculiar_velocity_factor)
```

with `f = dlnD/dlna`, `D(1) = 1` and `E = H/H0` (the h of Mpc/h cancels against H0 = 100 h).
The header then has `velocity_is_dtime: false` and records the factor, `a` and the cosmology;
otherwise `velocity_is_dtime: true` and those three are null. One of `a`/`cosmo` without the
other raises. At a = 1 with the default cosmology the factor is 52.24.

`realization.py export` always writes km/s at the checkpoint's epoch; `device_run.py export`
does too unless `--d-time`.

### Loading

```python
from inexor import export
head, x, v, ids = export.load_particles("W/export")            # memmaps, no crc check
head, x, v, ids = export.load_particles("W/export", mmap=False) # full load, crc32 verified
```

`ids` is None when absent. A missing header or an unknown `format` is refused, and with
`mmap=False` so is a crc32 mismatch. `load_particles` reads both formats.

### One part per rank

`export.write_particle_parts(st, out_dir, comm=None, dtype=np.float32, a=None, cosmo=None,
decode="host", devices=None, ...)` is called by every rank with its own slabs
(`device_run.py export` uses `decode="cards"`, on one node or across nodes). Format id
`export.FORMAT_PARTS = "inexor-particles-2"`. Rank `r` writes `x.r<rrrr>.npy` and
`v.r<rrrr>.npy` (and `ids.r<rrrr>.npy`), its rows starting at row `row0` of the whole export.
The parts in rank order are the single-file arrays row for row, at any rank count and with
either decode.

The header has the single-file fields, except that `files` is replaced by `parts` (one entry
per rank: `rank`, `row0`, `rows`, `files`, `crc32`), `source` is `"write_particle_parts"`,
and `decode` (`"host"` or `"cards"`) is added. Its top-level `crc32` holds the whole array's
crc32 per role (the crc of the parts concatenated), so it equals a single-file export's crc32
for the same particles. Rank 0 removes the header before any part is written and writes it
once every part has closed.

```python
for row0, x, v, ids in export.iter_particle_parts("W/export"):   # either format
    ...
head, x, v, ids = export.load_particles("W/export", mmap=False)  # parts concatenated
```

`load_particles` with `mmap=True` refuses an export of more than one part.

### `python -m inexor.export`

```
python -m inexor.export CHECKPOINT_DIR OUT_DIR [--dtype float32|float64] [--chunk-bricks 1024]
                        [--a A] [--omega-m OM] [--h H] [--d-time] [--allow-partial]
```

`CHECKPOINT_DIR` is any T9 slab directory: a checkpoint generation (for example
`W/ckpt/gen1`), a snapshot, or an IC directory. It needs no run configuration and checks no
fingerprint. A run checkpoint that is neither its last step nor a snapshot is refused unless
`--allow-partial` is given. The output epoch is resolved in this order:

1. `--d-time`: native `dx/dD`; combining it with `--a`, `--omega-m` or `--h` is refused.
2. `--a`: this scale factor, overriding any recorded epoch.
3. The checkpoint's recorded `provenance.a` and `provenance.cosmology`.
4. Otherwise native D-time, with a printed note to pass `--a` for km/s (an IC directory
   records no epoch: its top-level `a_init` comes with no cosmology and is not used).

`--omega-m` and `--h` override single cosmology fields on top of the recorded cosmology (or the
defaults) and are refused when no epoch is in use. The chosen source is recorded as
`provenance.epoch_source`, with `provenance.source_checkpoint`.

## Run cards

### CPU driver: `realization_*.json`

Each `realization.py` phase writes one JSON card into `--workdir`. Every card carries an
envelope: `card` (`"inexor-realization-{kind}-1"`), `config`, `workdir`, `commit`, `host`,
`machine`, `numpy`, `k_steps`, `growth2`, `when`. The rest is per phase.

| file | written by | fields beyond the envelope |
|---|---|---|
| `realization_ics.json` | `ics` | `wall_s`, `peak_rss_bytes`, `bytes_per_particle`, `bricks_per_side`, `manifest` (the IC manifest, an embedded P(k) table reduced to its `source` and `sha256`), `seed`, `slab`, `generator`, `n_part` |
| `realization_run_<from>_<to>.json` | `run`, one per segment (steps zero-padded to two digits) | `from_step`, `to_step`, `n_steps`, `source`, `wall_s`, `s_per_step`, `load_s`, `peak_rss_bytes`, `tile_workers`, `migrate_pooled`, `eject_kernel`, `brick_slack`, `arena_frac`, `checkpoint_every`, `phase_instrument`, `phase`, `per_step_stats`, `a_steps`, `projected_full_run_h`, `worker_rss_bytes`, `total_rss_bytes`, `arena_peak_rows`, `n_arena`, `migrate_pooled_workers` |
| `realization_pk.json` (`realization_pk_ics.json` with `--ic-dir`) | `card` | the P(k) card file above |
| `realization_export.json` | `export` | `step`, `a_out`, `out_dir`, `wall_s`, `peak_rss_bytes`, `manifest` (the export header) |

Reading the run card:

- `peak_rss_bytes` is the parent process's peak host RSS; `worker_rss_bytes` sums the pool
  workers' RSS from the last step, and `total_rss_bytes` is the two together.
- `phase` depends on `phase_instrument`. With `time` (default): seconds per phase over the
  segment (`total_s`, `per_phase`, `per_phase_frac`, `n_steps_seen`, `unknown_phases`). With
  `peak` (Linux only): per-phase host high-water marks of the parent process, from `VmHWM`
  reset at every boundary: `phases` (per phase `peak`, the absolute RSS reached; `delta`, what
  the phase itself added; `visits`; `rss_end`), `order`, `series`, `growth`, `step_ladder`,
  `run_peak`; `peak_rss_bytes` is then `run_peak`.
- `a_steps` is the full scale-factor grid (`k_steps + 1` points).
- `per_step_stats` is the list of per-step stats dicts from `engine.run`: for example `cap`
  (padded rows per tile), `arena_used`, `n_tiles`, the dtypes and backends that ran,
  `checkpoint` (the directory written, or null) and `repack`.

### GPU driver: `--card`

Every `device_run.py` subcommand writes its own card to `--card` (across ranks, one per rank:
`--card run.json` becomes `run.rank<r>.json`), rewritten atomically at every phase boundary, so a killed job
leaves a card up to its last boundary. Card id `inexor-d7-{kind}-1`. Common fields: `preset`,
`workdir`, `commit`, `host`, `job` (`SLURM_JOB_ID`), `started` (epoch seconds), `env` (every
`XLA_*`, `JAX_*`, `INEXOR_*` variable), `ranks`, `finished` (absent if the process did not
finish) and, after an exception, `failure` (the error, traceback, every card's full allocator
stats and the last heartbeat).

A `run` card adds:

| field | content |
|---|---|
| `plan` | the arguments that shape the run (stop step, K, timing, layout, y-blocks, checkpointing) |
| `config` | the resolved engine settings: `n_fine`, `b_fine`, `tile_window`, `fused_pass`, `device_cards`, `y_blocks`, `checkpoint_dir`, `coarse_dtype`, `fine_dtype` |
| `a_init`, `start_step`, `state`, `snapshots` | the schedule's start, the step it started from, the loaded state's size (`n_particles`, `n_bricks`, `rows`, `n_arena`), the snapshot steps |
| `boundaries` | one record per engine phase boundary (below) |
| `beats` | heartbeat records (every `--beat` seconds): host memory, bytes read and written, each card's allocator bytes |
| `steps` | each step's engine stats (the `STEP_JSON` lines of the log); a timed step (`--timed-every`, `--timed-last`, `--timed-all`) carries `timings`, synced per-pass seconds per card, and the other steps `timings: null` |
| `hwm_per_phase` | true when the host high-water mark could be reset at each boundary (Linux), so each boundary's `hwm` is that phase's own peak |
| `run_host_peak` | the largest host high-water over the boundaries, bytes |
| `checkpoints`, `snapshot_dirs` | directories written, per step |
| `program_memory` | XLA's memory breakdown of each compiled tile program |

Each `boundaries` record: `name` (the phase that just ended), `step`, `t`, `dt` (the phase's
seconds), `rss`, `hwm` (host VmHWM over the phase, bytes), `rss_phase_peak` (largest sampled
host RSS in the phase), `avail` (`MemAvailable`), `cards` (per GPU, the allocator's
`in_use` and its running `peak`), `card_phase_peak` (per GPU, the largest sampled bytes in use
during the phase; sampling every `--card-sample-ms`, default 10), `numa`, `mem`, `faults`,
`vmstat`, `instr_s` (the instruments' own seconds, excluded from `dt`), and with `--numa-maps`
/ `--smaps` the process's pages per node / by mapping kind. Step 0's boundaries cover the
load, the kernel build and the lead drift.

An `ics` card has `plan`, `boundaries` (one per generator stage) and `product` (`wall_s`,
`run_host_peak`, and the manifest's `n_particles`, `bricks_per_side`, `max_displacement`,
`vel_scale`, `stage_s`, `emission_s`). `card` and `export` cards have `state`,
`boundaries` and `product` (`wall_s`, `load_s`, and `n_bins` for a card or `timings`, `rows`,
`crc32` for an export). A `preflight` card records `devices`, `allocator`, the IC `manifest`
summary, the `planner` output and its parsed peaks, `numa`, `memory_policy`, `scratch` and
`refusals`.

`device_run.py summarize --card C --gpu-csv G --mem-csv M --out S` lines a run card up
against the job's external samplers and writes, per boundary, `dt`, `host_phase_peak`,
`host_phase_peak_sampled`, `host_avail_min`, `card_max_gib` and `card_util_mean` per GPU,
`gpu_node_used_max_gb` and the sample count, plus `unfinished_tail` for a run that did not
finish.
