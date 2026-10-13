# scripts/validation

The measurements behind the paper's validation appendix (force accuracy, high-k power
spectrum, initial conditions) and the velocity comparison with DISCO-DJ. Each script is a
command-line tool run from the repository root in the default (CPU) environment:

```bash
pixi run --frozen python scripts/validation/<script>.py ...
```

Outputs go under `runs/validation/` by default (gitignored); every script takes its inputs and
outputs as arguments. None is imported by the package. `_common.py` holds the helpers they
share. The scripts that use JAX select the CPU backend unless `JAX_PLATFORMS` is set, and
enable float64 themselves.

## Inputs

1. **cgh64 initial conditions with their stage** (for `ic_validation.py`): 512^3 particles in
   256 Mpc/h, seed 0, EH98, 2LPT with the LCDM growth, a_init = 0.1.

   ```bash
   pixi run --frozen python scripts/run/realization.py ics --config cgh64 \
       --workdir runs/validation/cgh64 --keep-stage
   ```

   ~35 s and 4.9 GB peak on the laptop; the ICs take 1.3 GB and `stage/` another 11 GB.
2. **The same-ICs run** (for `velocity_compare.py` and `pk_highk.py`): inexor and DISCO-DJ at
   cgh64 from the same initial particles, 120 steps, from
   `scripts/compare/gh_same_ics_vista.sbatch` (see `scripts/compare/README.md`). The inputs
   are the inexor float64 arm's export, `<work>/cgh64/inexor-float64/prod/export/`
   (`x.npy`, `v.npy`, `export.json`), and DISCO-DJ's unchunked final state,
   `<work>/cgh64/disco-chunk-all.npz` (`x`, `v_d`).
3. **An evolved snapshot** (for `force_production.py clustered`): the `x.npy` of any export of
   a run at the preset given by `--config` (`realization.py export`, or
   `python -m inexor.export` on a final checkpoint). The paper used the cgh64 z = 0 state of
   the one-GH200 crossover job (`scripts/compare/gh_crossover_vista.sbatch`).

## Scripts

Runtimes and peak resident memory were measured on the development laptop (Apple M4 Max,
16 cores, 128 GB), with scipy FFTs on at most 8 workers and the JAX scripts under
`XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`.

| Script | Paper result | Command | Time, peak memory | Input |
|---|---|---|---|---|
| `force_accuracy.py` | App. A.1: radial force of a point mass over Newton (two-level and global mesh) and the two-level force's difference from one global fine mesh, by separation, for tile-core and tile-boundary sources; the f32 coarse-mesh side test. Writes `force_accuracy.npz`, `force_curves.json` (the figure's curves) and `summary.txt` | `force_accuracy.py` | 338 s, 14.5 GB (the global-mesh child 12.2 GB) | none (`cdev` geometry, fixed seed) |
| `force_production.py self` | App. A.1: self-force of isolated sources in production precision, in units of the one-cell force | `force_production.py self` | 199 s, 12.7 GB | none |
| `force_production.py clustered` | App. A.1: per-particle two-level force against one global fine mesh on an evolved snapshot (median, p95, p99 of the relative difference; signed component along F; near-face vs core) | `force_production.py clustered --config cgh64 --positions X.npy` | cgh64: ~111 GB peak (the global 1024^3 float64 mesh), not run here; `--config cdev8-tile32` (128^3): 8 s, 3.2 GB | input 3 |
| `pk_highk.py measure` | App. A.2: interlaced-TSC P(k) to the mesh Nyquist of each code's final state (1024^3 mesh); with `--k0 0 --dk 0.02454369260617026 --n-mesh 256`, the check of the card against this estimator on the card's bins | `pk_highk.py measure EXPORT/x.npy --box 256 --n-mesh 1024`, and the same on `disco-chunk-all.npz` | 1024^3: 125 s, 47 GB (inexor, float32 positions); 131 s, 51 GB (DISCO-DJ, float64); 256^3: 95 s, 19 GB | input 2 |
| `pk_highk.py ratio` | App. A.2: max \|P_inexor / P_DISCO-DJ - 1\| for k up to 1.57, 3, 5, 6, 8, 10, 12 h/Mpc | `pk_highk.py ratio INEXOR.json DISCO.json` | < 1 s | two `measure` outputs |
| `velocity_compare.py` | Sec. 5: density, momentum and momentum-divergence spectra of inexor over DISCO-DJ and their cross-correlation, CIC on a 512^3 mesh | `velocity_compare.py --inexor EXPORT --disco disco-chunk-all.npz` | 117 s, 39 GB | input 2 |
| `ic_validation.py pk-lin` | App. A.3: P/P_lin of the staged linear density, chi^2 over 48 bins, mean over all modes | `ic_validation.py pk-lin` | 6 s, 2.8 GB | input 1 |
| `ic_validation.py lpt` | App. A.3: psi1, psi2, u re-derived in float64 vs the staged float32 out-of-core fields; D1, D2 at a_init | `ic_validation.py lpt` | 13 s, 12 GB | input 1 |
| `ic_validation.py div` | App. A.3: div psi1 + delta residual power, on and off the Nyquist planes | `ic_validation.py div` | 2 s, 4.9 GB | input 1 |
| `ic_validation.py s3` | App. A.3: S3 of the linear, Zel'dovich and 2LPT fields at R = 4, 6, 8, 12 Mpc/h, and Delta S3 against the tree level (64 jackknife sub-cubes; `--nsub 2` for 8) | `ic_validation.py s3 --nsub 4` | 58 s, 9.8 GB | input 1 |
| `ic_validation.py ds3-tree` | App. A.3: tree-level Delta S3 on this realization | `ic_validation.py ds3-tree` | 10 s, 9.3 GB | input 1 |
| `ic_validation.py ds3-paired` | App. A.3: Delta S3 minus its tree level on this realization at displacement amplitudes 1, 0.5, 0.25 (paired jackknife) | `ic_validation.py ds3-paired` | 126 s, 9.3 GB | input 1 |
| `fnl_amplitude.py` | App. A.3: amplitude of the local-f_NL squeezed bispectrum over the binned template, with its standard error over seeds | `fnl_amplitude.py --f-nl 2000 --seeds 48`; `--f-nl 2000 --seeds 400`; `--f-nl 100 --seeds 400` | 22 s, 0.9 GB; 174 s, 4.0 GB; 88 s, 4.1 GB | none (32^3 ICs made in memory) |

`ic_validation.py` reads the IC directory from `--ic-dir` (default `runs/validation/cgh64`)
and its geometry (particles, box, a_init, second-order growth) from the IC manifest; it
accepts only Gaussian EH98 ICs.
