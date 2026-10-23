# scripts/compare

Tools that compare the engine against other codes and references. None of them is imported
by the package. Paths below are examples: `IC`, `W`, `RUN` stand for an IC directory, a work
directory and a run directory.

| File | What it does |
|---|---|
| `disco_crosscheck.py` | DISCO-DJ (a single-mesh PM) evolved from the engine's initial particles, scored with the engine's P(k) card |
| `disco_crosscheck_gh_vista.sbatch` | that cross-check as one Vista job |
| `gh_crossover_vista.sbatch` | wall time and memory of both codes on one GH200, 256^3 to 1024^3 |
| `gh_same_ics_vista.sbatch` | both codes at 512^3 from the same ICs: DISCO-DJ's particle chunking, inexor's fine-mesh precision, and the inputs of the velocity comparison |
| `gh_halo_check_vista.sbatch` | FoF halos at 0.25 Mpc/h spacing on one GH200: inexor's step and start-epoch arms, DISCO-DJ at the same and at half the force cell, z = 0 and snapshot redshifts |
| `halo_compare.py` | FoF on an evolved state (`find`) and two catalogs compared (`compare`): abundance, matched halos, halo bias |
| `memtrace.py` | host and card memory sampler both codes' cost legs run under |
| `pk_boost_reference.py` | P(k) cards against CAMB HMcode2020, halofit and EuclidEmulator2 |
| `ee2_ratio_figure.py` | P(k) cards over EuclidEmulator2, with sample-variance bands |
| `quant_grid.py`, `quant_grid_gg_vista.sbatch` | the codec's effect on P(k): the engine against unquantized float runs on the same ICs |

The DISCO-DJ legs run in a separate pixi project (`DISCO_SRC` in the job scripts) whose `gpu`
and `default` environments provide DISCO-DJ at commit `a687972` with JAX 0.10.1, installed by
its `setup-gpu` and `setup` tasks; the jobs check both pins before any leg runs.

## disco_crosscheck.py

Runs DISCO-DJ on the engine's initial particles and scores the result with the engine's own
P(k) card, so the two can be differenced bin for bin on the same ICs. Four phases, each a
separate process in the environment it needs; consecutive phases exchange one `.npz`.

| Phase | Env | Reads | Writes |
|---|---|---|---|
| `export` | inexor | an IC directory (`--ic-dir`) | `(x, v_d)`, the a-grid, and `pk_eh98.txt` beside the npz |
| `evolve` | DISCO-DJ | the export npz | final `(x, v_d)` from DISCO-DJ |
| `evolve-mono` | inexor, CPU backend | the export npz | final `(x, v_d)` from inexor's own single-mesh PM |
| `card` | inexor, CPU backend | an evolve npz | `<workdir>/disco_pk.json` |

- **npz contents:** `x` and `v_d` (float64 positions and D-time velocities), `a_steps`, and a
  JSON `meta` (config, `n_part`, `box_size`, `ic_dir`, `k_steps`, `a_init`, `growth2`,
  DISCO-DJ's cosmology dict, and `pk_file`, the absolute path of the EH98 table `evolve`
  reads, so `evolve` must see the same filesystem as `export`). Evolve phases add their
  settings to `meta`.
- **`export`:** `--config`, `--ic-dir`, `--k-steps`, `--out` required; `--a-init`
  (default 0.1) and `--growth2` (`lcdm`, default, or `eds`) must match the ICs' manifest,
  which it checks. The a-grid and coefficients are the realization driver's.
  `--snapshot-z Z,Z,...` records the step boundary `device_run.py run --snapshot-z` takes for
  each redshift (`snapshot_steps` in `meta`).
- **`evolve`:** `--in`, `--n-mesh`, `--out` required; `--precision double` (default) or
  `single`; `--chunk-size N`, the particles DISCO-DJ paints and interpolates at once (must
  divide the particle count; default all at once). DISCO-DJ settings: CIC (`worder=2`), ik
  gradient and Laplacian, no deconvolution, no antialiasing, BullFrog on the exported a-grid,
  momentum `v_d * Fplus(a)`. `--stop-z Z` stops at the export's snapshot step for `Z` (the
  first steps of the same grid; each BullFrog step's coefficients depend only on its own
  interval), and records `stop_step` and `a_out`; an unrecorded `Z` is refused.
- **`evolve-mono`:** `--in`, `--n-mesh`, `--out`. `forces.force_global(which="mono",
  assign="cic")` on one float64 mesh, stepped with `engine.float_run_bullfrog_sync` and the
  engine's BullFrog coefficients: it differs from the engine only in the force split. It
  refuses an export whose a-grid differs from the one it rebuilds.
- **`card`:** `--in` before `--`; everything after `--` is parsed by
  `scripts/run/realization.py card`'s parser (`--config`, `--k-steps`, `--workdir`,
  `--coarse-match-order`, `--slack`, `--alloc-margin`, `--arena-frac`, `--slab`,
  `--min-weight`, the geometry overrides `--n-fine`, `--buf`, `--n-coarse`, `--n-part`). It
  rebuilds a `SlotState` from the particles and runs `summary.pk_summary_card` at the
  state's `a_out` (1.0 unless the evolve stopped early) with default bins.

```bash
pixi run --frozen python scripts/compare/disco_crosscheck.py export --config cgh64 \
    --ic-dir IC --k-steps 120 --out W/disco_in.npz
pixi run --manifest-path <DISCO-DJ project>/pixi.toml -e gpu \
    python scripts/compare/disco_crosscheck.py evolve --in W/disco_in.npz \
    --n-mesh 1024 --out W/disco_out.npz
pixi run --frozen python scripts/compare/disco_crosscheck.py card --in W/disco_out.npz \
    -- --config cgh64 --k-steps 120 --workdir W --coarse-match-order 3
```

## The Vista jobs

All three take their run root and checkouts from environment variables and refuse to start
without the required ones; the account goes on the command line. Each starts with guards:
both checkouts have no modified tracked files, both environments are installed from their
locks, DISCO-DJ's environment reports JAX 0.10.1, a `discodj` install whose recorded commit
starts with `a687972`, and the expected device platform, and inexor imports. Each leg prints
its start, end and exit code.

| Variable | Jobs | Meaning |
|---|---|---|
| `INEXOR_RUNS` | all three, required | run root; the job works in `$INEXOR_RUNS/<job name>-<job id>` |
| `DISCO_SRC` | all three, required | the DISCO-DJ project checkout |
| `IC_DIR` | cross-check, required | the cgh64 IC directory to cross-check |
| `INEXOR_SRC` | all three, optional | the inexor checkout (default: the submission directory) |
| `DISCO_BACKEND` | cross-check | `gpu` (default) or `cpu` |
| `SMOKE_ONLY` | cross-check | `1` stops after the smoke legs |
| `CHUNKS_cdev8`, `CHUNKS_cdev`, `CHUNKS_cgh64` | same-ICs | DISCO-DJ chunk sizes in particles per preset, `all` = no chunking |

### disco_crosscheck_gh_vista.sbatch

```bash
INEXOR_RUNS=<run root> DISCO_SRC=<DISCO-DJ project> IC_DIR=<cgh64 IC dir> \
    sbatch -A <account> scripts/compare/disco_crosscheck_gh_vista.sbatch
```

Legs, in order, each stopping the job on failure:

1. **Guards** (above).
2. **Smoke:** the whole chain at `cdev8` (fresh ICs, export, evolve on a 256^3 mesh, card).
3. **Cross-check:** export the cgh64 ICs, evolve with DISCO-DJ in float64 on a 1024^3 mesh,
   card.

The inexor legs run in inexor's `gpu` environment with `JAX_PLATFORMS=cpu`; the GPU evolve
sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`. `DISCO_BACKEND=cpu` runs DISCO-DJ on JAX's CPU
backend (its `default` environment and `setup` task); submit it to a CPU partition with a
longer wall.

### gh_crossover_vista.sbatch

Wall time and memory of both codes on one GH200 node from the same ICs at 256^3, 512^3 and
1024^3 particles (presets `cdev`, `cgh64`, `c-1024`, all at 0.5 Mpc/h spacing), 120 BullFrog
steps, each code's force mesh twice the particle count per side.

```bash
INEXOR_RUNS=<run root> DISCO_SRC=<DISCO-DJ project> \
    sbatch -A <account> scripts/compare/gh_crossover_vista.sbatch
```

- **Per size, inexor:** ICs (`realization.py ics --generator device`), then
  `device_run.py preflight` and `run` on one card (120 steps, `--timed-every 5`, a
  checkpoint at step 120), then the card (`realization.py card` on the CPU backend).
- **Per size, DISCO-DJ:** export the same ICs, evolve in single and in double precision,
  card each; at 512^3 also single precision under JAX's default preallocating allocator.
  At 1024^3 DISCO-DJ evolves in single precision only, last, where it is expected to run out
  of card memory.
- Every cost leg runs under `memtrace.py` (traces in `$WORK/trace/<leg>.jsonl`). Both codes
  use one allocator policy (`XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`), no persistent compilation cache (compilation
  is part of the cost), host memory bound to the CPU NUMA node (`numactl --membind=0`), and
  `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB=160`. A smoke at `cdev8` runs every leg type first.
- inexor legs stop the job on failure; DISCO-DJ legs and cards record their exit code and
  continue. Cards: `$WORK/<preset>/prod/realization_pk.json` (inexor) and
  `$WORK/<preset>/card_<single|double|single-bfc>/disco_pk.json` (DISCO-DJ).

### gh_same_ics_vista.sbatch

Both codes on one GH200 from the same ICs as the crossover job (device generator, seed 0),
120 steps, with the crossover's allocator policy, memory binding and `memtrace.py`.

```bash
INEXOR_RUNS=<run root> DISCO_SRC=<DISCO-DJ project> \
    sbatch -A <account> scripts/compare/gh_same_ics_vista.sbatch
```

- **DISCO-DJ** in single precision at each chunk size of `CHUNKS_<preset>` (256^3 on a 512^3
  mesh and 512^3 on a 1024^3 mesh), each final state carded.
- **inexor** at 512^3 with the fine tiles in float64 (production) and in float32, each
  through `device_run.py preflight` and `run` (`--timed-last`) and a card; the float64 arm is
  also exported.
- Kept for `scripts/validation/` (see its README): the float64 export,
  `$WORK/cgh64/inexor-float64/prod/export/`, and DISCO-DJ's unchunked final state,
  `$WORK/cgh64/disco-chunk-all.npz`.
- Guards and inexor run legs stop the job on failure; cards, exports and DISCO-DJ legs
  record their exit code and continue. A smoke at `cdev8` runs every leg type first.

### gh_halo_check_vista.sbatch

FoF halos at 0.25 Mpc/h particle spacing (0.125 Mpc/h fine cell) on one GH200, at z = 0 and
the snapshot redshifts `SNAPZ` (default `2.2,1,0.5`):

```bash
INEXOR_RUNS=<run root> DISCO_SRC=<DISCO-DJ project> \
    sbatch -A <account> scripts/compare/gh_halo_check_vista.sbatch
```

- **`cgh64-L128`** (512^3 in 128 Mpc/h, `c-hero-L1024` in every resolution knob): inexor from
  z = 9 ICs at 120, 160 and 240 steps and from z = 19 ICs at 160; DISCO-DJ from the z = 9 ICs
  at 120 steps on a 1024^3 mesh (inexor's fine cell).
- **`cdev-L64`** (256^3 in 64 Mpc/h): inexor at 120 steps; DISCO-DJ at 120 steps on 512^3 and
  1024^3 meshes (the same and half the cell) and at 240 steps on 1024^3.
- Every inexor arm snapshots (`run --snapshot-z`) and exports each epoch, with a z = 0 card;
  DISCO-DJ stops at the same steps (`evolve --stop-z`). FoF (`halo_compare.py find`, b = 0.2,
  256^3 matter mesh) runs on every state, three at a time; catalogs go to
  `$WORK/halos/<preset>-<arm>-z<Z>.npz`. `halo_compare.py compare` on pairs is left to the
  reader.
- A `cdev8` smoke with one snapshot runs every leg type first, including a compare and the
  periodic self-test; guards and smoke stop the job on failure, later legs record their
  exit code and continue. `ARENA_FRAC` (default 0.05) sets the runs' spare arena.

### gh_halo_disco_steps_vista.sbatch

DISCO-DJ's step and start-epoch arms at inexor's fine cell (`cgh64-L128`, 1024^3 mesh), on the
ICs of a finished `gh_halo_check_vista.sbatch` run: 160 and 240 steps from z = 9, 160 from
z = 19. With that run's 120-step DISCO-DJ arm they match every inexor arm step for step.

```bash
INEXOR_RUNS=<run root> DISCO_SRC=<DISCO-DJ project> HALO_WORK=<halo-check work dir> \
    sbatch -A <account> scripts/compare/gh_halo_disco_steps_vista.sbatch
```

- Each arm stops at the snapshot steps (`evolve --stop-z`), with a z = 0 card; FoF runs on
  every state and catalogs go to `$WORK/halos/cgh64-L128-disco-<arm>-m1024-z<Z>.npz`, named
  as the halo check's.
- Guards (checkouts, envs, pinned versions, the ICs) stop the job on failure; later legs record
  their exit code and continue.

## halo_compare.py

Friends-of-friends (kdcount, periodic, linking length b x mean spacing) on one evolved
state, and two such catalogs compared. kdcount is pip-only, so both subcommands run in a
throwaway env:

```bash
pixi exec --spec python=3.12 --spec numpy --spec scipy --spec pip --spec c-compiler -- \
    bash -c 'pip install -q kdcount && python -I scripts/compare/halo_compare.py find \
    RUN/prod/export --out W/inexor.npz'
```

- **`find SRC --out CAT.npz`**: `SRC` is an inexor export directory (format
  `inexor-particles-1`) or a `disco_crosscheck.py evolve` npz. Writes the groups with at least
  `--n-floor` (10) members, their counts and minimum-image COMs, and the matter CIC density
  on an `--n-mesh` (256) mesh, with the epoch, box, `m_p` and timings. `--shift` finds on
  positions moved by L/2 and moves the outputs back.
- **`compare A.npz B.npz [--labels A B] [--out JSON]`**: refuses catalogs of different
  geometry, linking length, mesh or epoch. Per particle-count bin (20, 50, 150, 500, 1500):
  counts and their ratio; mutual-nearest COM matches within `--match-r` (2) R_200m of the
  larger halo, with the match fractions and the matched count ratio; the matter power ratio;
  and, for k <= 0.5 h/Mpc, each catalog's halo bias P_hm/P_mm, their ratio and the
  cross-correlation of the two halo fields. CIC windows are not deconvolved; the ratios
  cancel them. A plain and a `--shift` catalog of one state compare identical.
- Cost at 512^3: FoF ~2 min on one core, ~21 GB peak.

## memtrace.py

Runs a Python script in its own process (`runpy`, as `__main__`) with a sampler thread that
writes one JSON line every `--interval` seconds (default 0.02): host `VmRSS`, `RssAnon`,
`RssFile` from `/proc/self/status`, and each local device's allocator counters
(`memory_stats()`) once the script has initialized a JAX backend. The last line is a summary:
wall, exit code, `VmHWM`, each device's `peak_bytes_in_use`, and the time integrals of RSS and
of device bytes in use. Peaks come from the kernel's and the allocator's counters, not from
the samples. Child processes are not counted. Off Linux only RSS is sampled, through `ps`, at
0.5 s or slower.

```bash
python scripts/compare/memtrace.py --out W/trace.jsonl [--interval 0.02] -- script.py ARGS...
```

## pk_boost_reference.py and ee2_ratio_figure.py

Both compare P(k) cards (`realization_pk.json` from `realization.py card`, the same format
from `device_run.py card --out`, or a cross-check's `disco_pk.json`) against references, as
`P_card / P_reference` with every reference averaged over each card bin's own lattice modes,
the average the card takes of its own P. Evaluating a reference at a bin's mean k instead
leaves the BAO as a +-1-2% zigzag (a bin spans ~40% of a period). The bins are rebuilt from
the card's mesh and `k_edges` (`pk_boost_reference.card_bin_modes`), and a card whose mode
counts are not reproduced is refused. References are run at the `A_s` that matches the
cosmology's `sigma8`, with massless neutrinos; a card made from EH98 ICs shows EH98's
difference from CAMB's linear spectrum at low k.

Run them with `pixi exec` and the specs given in each docstring, not in the project
environment: camb and euclidemu2 are not engine dependencies, and adding them would move
`pixi.lock`. EE2 is parameterized by `A_s`, so both tools take `A_s` from the CAMB solve,
and `ee2_boost` refuses a cosmology outside EE2's training range. The output directory is
created if needed.

- **`pk_boost_reference.py CARD [CARD ...]`**: left panel, the boost `B = P / P_lin` of
  each card (over its own linear oracle) and of CAMB HMcode2020, halofit (Takahashi) and
  EuclidEmulator2 if `euclidemu2` imports (`--no-ee2` skips it), for scale; right panel,
  `P / P_reference`: one card against all three references, with HMcode's quoted 2.5% as the
  band; several cards (one redshift, any boxes) each against EE2 (HMcode without it), with
  that reference's quoted accuracy as the band. A table on stdout. Options: `--labels` and
  `--linestyles` (one entry per card), `--legend-loc` (the ratio panel's legend), `-o`
  (default `runs/figures/pk_boost.png`), `--dpi` (default 200). The cosmology comes from the
  card, or else from `--cosmology FILE` (any JSON with a `cosmology` block; an export header
  written with km/s velocities has one); there is no default.

  ```bash
  pixi exec --spec python=3.12 --spec camb --spec matplotlib --spec numpy --spec scipy -- \
      python scripts/compare/pk_boost_reference.py RUN/realization_pk.json \
      --cosmology RUN/export/export.json -o runs/figures/pk_boost.png
  ```

- **`ee2_ratio_figure.py CARD [CARD ...]`**: `P_inexor / P_EE2` for up to three cards, with
  `P_EE2 = P_lin,CAMB x B_EE2`. Cards may come from different boxes and redshifts; one +-1
  sigma Gaussian sample-variance band per distinct k grid, and EE2's quoted 1% accuracy
  band. `--cosmology` and `-o` are required; `--labels` takes one label per card, `--dpi`
  defaults to 200. Cards must carry `k_edges`. euclidemu2 is pip-only and its wheel needs
  `gsl` in the environment:

  ```bash
  pixi exec --spec python=3.12 --spec camb --spec matplotlib --spec numpy --spec scipy \
      --spec gsl --spec pip -- bash -c "pip install -q euclidemu2; \
      python scripts/compare/ee2_ratio_figure.py RUN_A/realization_pk.json \
      RUN_B/realization_pk.json --cosmology RUN_A/export/export.json \
      --labels 'run A' 'run B' -o runs/figures/inexor_over_ee2.png"
  ```

## quant_grid.py, quant_grid_gg_vista.sbatch

The codec's effect on the power spectrum: the engine against float runs of the same
drift-synchronized BullFrog shape and two-level force, from the same 2LPT ICs (seed 0, LCDM
growth, float64 throughout, CPU backend). One arm per process, one JSON card each.

| Arm | What is quantized |
|---|---|
| `engine` | the engine itself (CPU lane): positions on the T9 lattice, velocities in per-brick int16 codes |
| `ref` | nothing |
| `refx` | positions rounded to the T9 lattice at the ICs and after every drift |
| `refv` | velocities round-tripped through per-brick int16 codes at the ICs and after every kick |
| `refxv` | both; reproduces `engine`, which checks the emulation |

The card holds the fine-mesh P(k) (integer CIC paint, default linear bins) and, for the float
arms, per step the drift in quanta (median `|dx|/q`, fraction below `q/2`) and the kick in
velocity codes (median, fraction below half a code).

```bash
pixi run --frozen python scripts/compare/quant_grid.py --config cdev8 --a-init 0.1 \
    --k 120 --arm engine --out W/cdev8_a0.1_k120_engine.json
```

Options: `--config` (`cdev8`, default: 128^3 in 64 Mpc/h, the production 0.5 Mpc/h spacing;
`cdev8-L128`: 128^3 in 128 Mpc/h, 1 Mpc/h spacing; `smoke`: 32^3), `--a-init` (default
0.1), `--k` (steps, required), `--arm` (required), `--long-paint` (`int`, default, as the
engine, or `f64`: the float arms' coarse paint), `--out` (required).

`quant_grid_gg_vista.sbatch` runs the grid on one CPU node, every leg a separate process
pinned to its own 4 cores, all at once:

- `cdev8`: start epochs z = 9, 19, 49 (`a_init` 0.1, 0.05, 0.02) x 40, 120, 240 steps,
  `engine` and `ref`; `refx`, `refv`, `refxv` at (z = 9, 120 steps) and (z = 49, 240 steps);
  `ref` at (z = 49, 408 steps);
- `cdev8-L128`: (z = 9, 120), (z = 19, 120), (z = 49, 240), `engine` and `ref`.

It needs `OUT_DIR` (results go to `$OUT_DIR/job-<job id>`, one log per leg under `legs/`) and
runs the checkout's installed `gpu` environment's python directly with `JAX_PLATFORMS=cpu`,
never through `pixi run`, so it cannot reinstall an environment other jobs share:

```bash
OUT_DIR=<run root>/quant-grid sbatch -A <account> scripts/compare/quant_grid_gg_vista.sbatch
```
