# Outputs

Reference for what inexor writes to disk: the T9 slab directories (initial conditions and
checkpoints), the P(k) summary card, the portable particle export, and the per-phase run cards
the driver writes. Lengths are comoving Mpc/h, wavenumbers h/Mpc, power (Mpc/h)^3.

A run driven by `scripts/run/realization.py --workdir W` produces:

```
W/manifest.json, W/t9_slab_0000.npz ...   initial conditions (T9 slab directory)
W/stage/                                  IC intermediates (removed on success unless --keep-stage)
W/ckpt/gen0/, W/ckpt/gen1/                checkpoints (T9 slab directories)
W/export/x.npy, v.npy, export.json        particle export (default --export-dir)
W/realization_*.json                      per-phase run cards
```

## T9 slab directory

Written by `icgen.generate_t9_slabs` / `generate_t9_slabs_device` (ICs) and
`icgen.write_t9_slabs` (checkpoints); read by `icgen.load_slot_state`. Schema string
`icgen.SCHEMA = "t9-slabs-2"`.

A directory holds one `t9_slab_{bx:04d}.npz` per x-slab of bricks (`bx` = 0 ..
bricks_per_side - 1) and a `manifest.json`.

### Per-slab arrays

| array | dtype, shape | content |
|---|---|---|
| `meta` | JSON string | slab metadata (below) |
| `occupancy` | int64, `(nb^2 * buckets_per_brick,)` | particle count per bucket, brick-major, for the slab's bricks |
| `off` | uint8, `(n_rows, 3)` | position offset within the particle's bucket, in quanta of `bucket_size / 256` |
| `w` | int16, `(n_rows, 3)` | velocity code; velocity = `w * scale[brick]` |
| `scale` | float64, `(nb^2,)` | per-brick velocity scale (native D-time units, see Velocity convention) |

Rows are brick-major: each brick's rows in bucket order, bricks in order. A row's bucket (and
so its position) is implied by where it sits against `occupancy`.

`meta` fields: `schema`, `bx`, `n_rows`, `bucket_lo` and `brick_lo` (global index of the slab's
first bucket and brick), and `crc32` (a dict with the crc32 of `occupancy`, `off`, `w`,
`scale`).

### Manifest fields common to every slab directory

| field | meaning |
|---|---|
| `schema` | `"t9-slabs-2"` |
| `files` | slab file names, in order |
| `n_particles` | total particles |
| `box_size` | box side, Mpc/h |
| `n_part` | particles per side |
| `bucket_cells` | bucket side in particle cells |
| `bricks_per_side` | bricks per side |
| `provenance` | free-form dict (IC provenance, or the checkpoint record below) |

### Write order and refusals

The slabs are written first and the manifest last, so the manifest marks a complete
directory. `write_t9_slabs` removes any existing manifest before rewriting, so a torn
overwrite cannot mix generations.

`load_slot_state` (and `read_manifest`) refuse to load:

- a directory with no `manifest.json` (incomplete or interrupted write);
- a manifest whose `schema` is not `"t9-slabs-2"` (a `-1` file with one global velocity scale
  would otherwise decode silently at the wrong scale);
- any slab whose array crc32 does not match its `meta`;
- a slab whose rows do not match its occupancy.

The loader reads the files twice (index, then payload), so only one slab's payload is live at
a time. Capacity (slack, arena) is not stored; it is chosen at load time.

### IC manifest

The IC generators add these fields:

| field | meaning |
|---|---|
| `vel_scale` | global velocity scale at generation (`max|v| / 32767`) |
| `max_displacement` | max 2LPT displacement, Mpc/h |
| `window` | emission window depth, brick slabs (generation refuses if the max displacement reaches it) |
| `a_init` | scale factor of the ICs |
| `order` | LPT order (2) |
| `growth2` | `"lcdm"` or `"eds"` second-order growth |
| `f_NL` | local primordial non-Gaussianity amplitude (the driver uses 0) |
| `fdtype` | generator float dtype (`float32` in the driver) |
| `slab` | planes per out-of-core FFT slab |
| `ic_stream` | noise stream identifier |
| `backend` | linear power backend (`eh98` or `table`) |
| `table_n_points` | nodes of the k table |
| `mean_phi2` | mean of phi^2 (null on the device generator at f_NL = 0) |
| `stage_cleanup` | report of removing `stage/`, or `{kept: true, ...}` |
| `provenance` | from the driver: `generator`, `commit`, `host`, `machine`, `numpy`, `when`, `jax`, `x64`, `devices`, `n_devices`, `allocator`, and the XLA variables listed in [configuration](configuration.md#environment-variables) |

The device generator also records `generator: "device"`, `emission`, `emission_s`,
`n_devices`, `pencil_batch` and `stage_s` (per-stage seconds).

### Checkpoints

`engine.run` with `checkpoint_dir` set writes every `checkpoint_every` steps into
`gen0/` and `gen1/` alternately, so the previous generation survives until the new one is
complete. A resumed run first writes the generation it did not load. The manifest carries
`source: "write_t9_slabs"` and this `provenance`:

| field | meaning |
|---|---|
| `kind` | `"inexor-checkpoint"` (`load_checkpoint` skips anything else) |
| `step` | completed steps |
| `n_steps` | steps in the full schedule |
| `cap_shape`, `pad_shape`, `device_shapes` | buffer shapes restored on resume, so the same programs compile |
| `n_arena` | arena rows at write time (default `arena_frac` on resume) |
| `fingerprint` | `engine.checkpoint_fingerprint` of config + coefficients |
| `a`, `cosmology` | epoch of the checkpoint and its `Cosmology` fields, when `run` was given `epoch=(a_steps, cosmo)` (the driver always does) |

`engine.load_checkpoint(dir, cfg, coeffs)` picks the generation with the highest `step` and
refuses one whose fingerprint differs from the current config and schedule. It also refuses
when no generation has a manifest. A state carrying particle ids cannot be checkpointed (the
schema has no ids).

## P(k) summary card

`summary.pk_summary_card(st, cfg, cosmo, a_out, ...)` paints the coarse density (TSC), takes
its spectrum, and bins it against the linear oracle `D(a_out)^2 P_lin(k)` averaged over each
bin's realized modes. It returns a dict and writes nothing. Card id `summary.CARD =
"inexor-pk-summary-1"`.

| field | meaning, units |
|---|---|
| `card` | `"inexor-pk-summary-1"` |
| `n_coarse`, `box_size`, `n_particles` | mesh cells per side, Mpc/h, total particles |
| `a_out`, `growth_factor` | epoch and `D(a_out)`, with `D(1) = 1` |
| `k_nyquist` | `pi * n_coarse / box_size`, h/Mpc |
| `k_nonlinear` | k where linear `k^3 P / (2 pi^2)` reaches 1, h/Mpc; null if not reached in the scan |
| `k_nonlinear_scan` | `[1e-3, 10.0]`, the scanned range |
| `shot_noise` | `V / N`, (Mpc/h)^3, subtracted from `p` |
| `deconvolved` | `"tsc"` (TSC window divided out) or null |
| `oracle` | description string of the oracle |
| `n_bins` | bins kept |
| `k_edges` | ALL requested bin edges (default 65 edges, 64 bins over `[0, k_nyquist/2]`); bins below `min_weight` are dropped from the per-bin lists, so `len(k_edges) - 1` can exceed `n_bins` |
| `k_mean` | mode-weighted mean k per kept bin, h/Mpc |
| `p` | measured power, window-deconvolved and shot-subtracted, (Mpc/h)^3 |
| `p_oracle` | bin-averaged linear power at `a_out`, (Mpc/h)^3 |
| `n_modes` | full-grid mode count per bin |
| `z_profile` | `(p / p_oracle - 1) / sqrt(2 / n_modes)` |
| `window_correction` | mode-mean `W^2` per bin |
| `shot_fraction` | `shot_noise / raw power` per bin |
| `provenance` | caller-supplied dict (empty from the driver) |

The card decides no verdict. Linear theory applies only below `k_nonlinear`.
`summary.band_verdict(card, k_max, k_min=0.0, bar=5.0)` returns `max_abs_z` over a band you
name, with `k_min`, `k_max`, `n_bins`, `bar`, `ok` (`max_abs_z < bar`),
`band_reaches_nonlinear` and `k_nonlinear`.

### Driver card options

`realization.py card` writes the summary under `summary` in `realization_pk.json` (see below).
`--k-max` and `--n-bins` (default 64) pin the bins to `[0, k_max]` so runs with different
coarse meshes share them; `--min-weight` (default 100) drops bins with fewer modes; `--ic-dir`
cards an IC directory at step 0 instead of the newest checkpoint.

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
| `ids.npy` | int32, `(N,)` | only when the state carries ids |
| `export.json` | JSON | header, removed first and written last (marks a complete export) |

Rows are brick-major (the engine's spatial layout) and carry no Lagrangian identity; row `i`
of one export is not the same particle as row `i` of another. Positions are the engine's own
decode (float64, cast to `dtype`), arena residents included.

Header fields: `format`, `n_particles`, `box_size`, `n_part`, `dtype`, `files` (role -> file
name), `crc32` (role -> crc32), `units` (`position`, `velocity` strings), `velocity_is_dtime`,
`peculiar_velocity_factor`, `a`, `cosmology`, `row_order`, `has_ids`, `source`
(`"write_particles"`), `provenance`.

### Velocity convention

The engine's native velocity is `dx/dD`: Mpc/h per unit linear growth factor. Passing both `a`
and `cosmo` converts to peculiar km/s by the factor

```
v_pec [km/s] = 100 * a * f(a) * D(a) * E(a) * v_D        (export.peculiar_velocity_factor)
```

with `f = dlnD/dlna` and `E = H/H0`. The header then has `velocity_is_dtime: false`, records
the factor, `a` and the cosmology. One of `a`/`cosmo` without the other raises. The driver's
`export` always writes km/s, at the checkpoint's epoch (the final one unless
`--allow-partial`).

### Loading

```python
from inexor import export
head, x, v, ids = export.load_particles("W/export")            # memmaps, no crc check
head, x, v, ids = export.load_particles("W/export", mmap=False) # full load, crc32 verified
```

`ids` is None when absent. A missing header or an unknown `format` is refused.

### `python -m inexor.export`

```
python -m inexor.export CHECKPOINT_DIR OUT_DIR [--dtype float32|float64] [--chunk-bricks 1024]
                        [--a A] [--omega-m OM] [--h H] [--d-time]
```

`CHECKPOINT_DIR` is any T9 slab directory (for example `W/ckpt/gen1`). The output epoch is
resolved in this order:

1. `--d-time`: native `dx/dD`; combining it with `--a`, `--omega-m` or `--h` is refused.
2. `--a`: this scale factor, overriding any recorded epoch.
3. The checkpoint's recorded `provenance.a` and `provenance.cosmology`.
4. Otherwise native D-time, with a printed note to pass `--a` for km/s.

`--omega-m` and `--h` override single cosmology fields on top of the recorded cosmology (or the
defaults) and are refused when no epoch is in use. An IC directory's top-level `a_init` is not
used (it records no cosmology). The chosen source is recorded as `provenance.epoch_source`,
with `provenance.source_checkpoint`.

## Per-phase run cards

Each `realization.py` subcommand writes one JSON card into `--workdir`. Every card carries an
envelope: `card` (`"inexor-realization-{kind}-1"`), `config`, `workdir`, `commit`, `host`,
`machine`, `numpy`, `k_steps`, `growth2`, `when`. The rest is per phase.

| file | written by | top-level fields beyond the envelope |
|---|---|---|
| `realization_ics.json` | `ics` | `wall_s`, `peak_rss_bytes`, `bytes_per_particle`, `bricks_per_side`, `manifest` (the IC manifest), `seed`, `slab`, `generator`, `n_part` |
| `realization_run_{from:02d}_{to:02d}.json` | `run`, one per segment | `from_step`, `to_step`, `n_steps`, `source`, `wall_s`, `s_per_step`, `load_s`, `peak_rss_bytes`, `tile_workers`, `migrate_pooled`, `eject_kernel`, `brick_slack`, `arena_frac`, `checkpoint_every`, `phase_instrument`, `phase`, `per_step_stats`, `a_steps`, `projected_full_run_h`, `worker_rss_bytes`, `total_rss_bytes`, `arena_peak_rows`, `n_arena`, `migrate_pooled_workers` |
| `realization_pk.json` (`realization_pk_ics.json` with `--ic-dir`) | `card` | `step`, `a_out`, `wall_s`, `load_s`, `card_pool_workers`, `pool_spawn_s`, `n_bins_below_k_nl`, `ic_dir`, `peak_rss_bytes`, `summary` (the P(k) card) |
| `realization_export.json` | `export` | `step`, `a_out`, `out_dir`, `wall_s`, `peak_rss_bytes`, `manifest` (the export header) |

Notes on the run card:

- `peak_rss_bytes` is the parent process only; `worker_rss_bytes` sums the pool workers' RSS
  from the last step and `total_rss_bytes` is their sum.
- `phase` depends on `phase_instrument`: with `time`, per-phase seconds (`total_s`,
  `per_phase`, `per_phase_frac`, `n_steps_seen`, `unknown_phases`); with `peak` (Linux only),
  per-phase high-water marks (`phases`, `order`, `series`, `growth`, `step_ladder`,
  `run_peak`, ...), and `peak_rss_bytes` is then `run_peak`.
- `a_steps` is the full scale-factor grid (`k_steps + 1` points).
- `per_step_stats` is the list of per-step stats dicts returned by `engine.run`.
