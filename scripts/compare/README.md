# scripts/compare

Tools that compare the engine against other codes and references. None of them is imported
by the package.

## disco_crosscheck.py

Runs DISCO-DJ (a single-mesh PM) on the engine's initial conditions and scores the result
with the engine's own P(k) card, so the two can be differenced bin for bin on the same ICs.
Four phases, each a separate process in the environment it needs; consecutive phases
exchange one `.npz`.

| Phase | Env | Reads | Writes |
|---|---|---|---|
| `export` | inexor | an IC directory (`--ic-dir`) | `(x, v_d)`, the a-grid, and `pk_eh98.txt` beside the npz |
| `evolve` | DISCO-DJ (disco-mocks checkout) | the export npz | final `(x, v_d)` from DISCO-DJ |
| `evolve-mono` | inexor, CPU backend | the export npz | final `(x, v_d)` from inexor's own single-mesh PM |
| `card` | inexor, CPU backend | an evolve npz | `<workdir>/disco_pk.json` |

- **npz contents:** `x` and `v_d` (float64 positions and D-time velocities), `a_steps`, and a
  JSON `meta` (config, `n_part`, `box_size`, `ic_dir`, `k_steps`, `a_init`, `growth2`, and
  `pk_file`, the absolute path of the EH98 table `evolve` reads, so `evolve` must see the
  same filesystem as `export`). Evolve phases add their settings to `meta`.
- **`export`:** `--config`, `--ic-dir`, `--k-steps`, `--out` required; `--a-init`
  (default 0.1) and `--growth2` (`lcdm` or `eds`) must match the ICs, which it checks. The
  a-grid comes from the realization driver's coefficients.
- **`evolve`:** `--in`, `--n-mesh`, `--out`; `--precision double` (default) or `single`.
  DISCO-DJ settings: CIC (`worder=2`), ik gradient and Laplacian, no deconvolution, no
  antialiasing, BullFrog on the exported a-grid, momentum `v_d * Fplus(a)`.
- **`evolve-mono`:** `--in`, `--n-mesh`, `--out`. `forces.force_global(which="mono",
  assign="cic")` on one float64 mesh, stepped with `engine.float_run_bullfrog_sync` and the
  engine's BullFrog coefficients; refuses an export whose a-grid differs from the one it
  rebuilds.
- **`card`:** `--in` before `--`; everything after `--` is parsed by
  `scripts/run/realization.py card`'s parser (`--config`, `--k-steps`, `--workdir`,
  `--coarse-match-order`, `--slack`, `--alloc-margin`, `--arena-frac`, `--slab`,
  `--min-weight`, geometry overrides). It rebuilds a `SlotState` from the particles and runs
  `summary.pk_summary_card` at `a_out = 1.0` with default bins.

```bash
pixi run --frozen python scripts/compare/disco_crosscheck.py export --config cgh64 \
    --ic-dir IC --k-steps 120 --out W/disco_in.npz
pixi run --manifest-path ~/src/disco-mocks/pixi.toml -e gpu \
    python scripts/compare/disco_crosscheck.py evolve --in W/disco_in.npz \
    --n-mesh 1024 --out W/disco_out.npz
pixi run --frozen python scripts/compare/disco_crosscheck.py card --in W/disco_out.npz \
    -- --config cgh64 --k-steps 120 --workdir W --coarse-match-order 3
```

## disco_crosscheck_gh_vista.sbatch

The whole cross-check as one Vista job. Required environment variables (the script refuses
to start without them):

- `INEXOR_RUNS`: run root; the job works in `$INEXOR_RUNS/disco-xcheck-<jobid>`;
- `DISCO_SRC`: the disco-mocks checkout;
- `IC_DIR`: the cgh64 IC directory to cross-check;
- `INEXOR_SRC`: the inexor checkout, or submit from it (`$SLURM_SUBMIT_DIR` is the fallback).

Optional: `DISCO_BACKEND=gpu` (default; disco-mocks `gpu` env, `setup-gpu` task) or `cpu`
(disco-mocks `default` env, `setup` task, `JAX_PLATFORMS=cpu`; submit to a CPU partition
with a longer wall); `SMOKE_ONLY=1` stops after the smoke legs. The account goes on the
command line:

```bash
INEXOR_RUNS=$SCRATCH/inexor_runs DISCO_SRC=$HOME/src/disco-mocks IC_DIR=<ic dir> \
    sbatch -A <account> scripts/compare/disco_crosscheck_gh_vista.sbatch
```

Legs, in order, each stopping the job on failure:

1. **Guards:** both checkouts have no modified tracked files; both envs are installed from
   their locks (`pixi install -e gpu` for inexor; the disco-mocks env plus its setup task);
   DISCO-DJ's env must report `jax == 0.10.1`, a `discodj` install whose recorded commit
   starts with `a687972`, and the expected device platform; inexor, the realization driver
   and this script must import.
2. **Smoke:** the whole chain at `cdev8` (fresh ICs, export, evolve on a 256^3 mesh, card).
3. **Cross-check:** export the cgh64 ICs, evolve with DISCO-DJ on a 1024^3 mesh, card.

The inexor legs run in inexor's `gpu` env with `JAX_PLATFORMS=cpu`; the GPU evolve sets
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`.

## pk_boost_reference.py and ee2_ratio_figure.py

Both compare a realization's P(k) card (`realization_pk.json`, written by `realization.py
card`) against references. `pk_boost_reference.py` compares nonlinear boosts
`B(k) = P(k) / P_lin(k)`, each side against its own linear theory: the card's
`p / p_oracle` (the oracle is the linear spectrum the ICs were made from) against CAMB's
`P_nl / P_lin` or EuclidEmulator2's emulated boost. `ee2_ratio_figure.py` compares P itself,
both sides averaged over the card's modes (below).

Run them with `pixi exec` and the specs given in each docstring, not in the project env:
camb and euclidemu2 are not engine dependencies, and adding them would move `pixi.lock`.

- **`pk_boost_reference.py CARD`**: the card's boost against CAMB HMcode2020 and halofit
  (Takahashi), plus EuclidEmulator2 if `euclidemu2` imports (`--no-ee2` skips it); a
  two-panel figure (`-o`) and a table on stdout. CAMB is run massless-neutrino and
  rescaled to the card's `sigma8`. The cosmology comes from the card, or else from
  `--cosmology FILE` (any JSON with a `cosmology` block; an export header written with
  km/s velocities has one); there is no default.

  ```bash
  pixi exec --spec camb --spec matplotlib --spec numpy -- \
      python scripts/compare/pk_boost_reference.py RUN/realization_pk.json \
      --cosmology RUN/export/export.json -o figures/pk_boost.png
  ```

- **`ee2_ratio_figure.py CARD [CARD ...]`**: `P_inexor / P_EE2` for up to three cards, with
  `P_EE2 = P_lin,CAMB x B_EE2` averaged over each card bin's own lattice modes (the bins are
  rebuilt from the card's mesh and edges; a card whose mode counts are not reproduced is
  refused). Unlike the boost ratio above, this compares P directly, so a card made from EH98
  ICs shows EH98's difference from CAMB at low k. Cards may come from different boxes and
  redshifts; one +-1 sigma Gaussian sample-variance band per distinct k grid, and EE2's
  quoted 1% accuracy band. `--cosmology` and `-o` are required; `--labels` takes one label
  per card. euclidemu2 is pip-only and its wheel needs `gsl` in the env:

  ```bash
  pixi exec --spec python=3.12 --spec camb --spec matplotlib --spec numpy --spec scipy \
      --spec gsl --spec pip -- bash -c "pip install -q euclidemu2; \
      python scripts/compare/ee2_ratio_figure.py RUN_A/realization_pk.json \
      RUN_B/realization_pk.json --cosmology RUN_A/export/export.json \
      --labels 'run A' 'run B' -o figures/inexor_over_ee2.png"
  ```

EE2 is parameterized by `A_s`, so both tools take `A_s` from the CAMB solve that matches
the card's `sigma8`, and `ee2_boost` refuses a cosmology outside EE2's training range.
