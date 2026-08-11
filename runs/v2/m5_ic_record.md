# M-v2-5: streamed ICs + the out-of-core FFT -- the milestone record

**Status: COMPLETE -- every gate green (2026-08-10).** Branch
`jc/m-v2-5-streamed-ics`; ADR D-v2-23. Charter: D-v2-18's ladder row,
re-scoped per D-v2-15 clause 5 (JC-ratified 2026-08-10, the third milestone
whose written exit criterion could not be read literally; the gate docstring
carries the reading). Headline: **the streamed generator reads 8.6 B/p where
the old path reads 95.7 (fitted cubic 8.79, extrapolating to 75.3 GB at
2048^3 against a 116 GB host), a 2048^3 transform the GH200's device cannot
fit runs correctly in 40.4 GB of host, and the loaded state is bitwise the
monolithic build.**

## What was built (all in-package unless noted)

- **The plane-keyed noise stream** (`ic.py`): one axis-0 plane keyed by
  `fold_in`, drawn on the CPU backend explicitly, `IC_STREAM =
  "m5-foldin-1"`. Invariance to slab decomposition is a property of the
  construction; the gate tests the theorem. NOT bit-identical to the old
  monolithic stream at any seed (clause 5), and every card now carries the
  stream identity so readouts refuse to pool across it.
- **The universal-node 1D |k| table** (`cosmology.ICKTable` + `ic_k_table`):
  P log-log, T linear-in-lnk (new -- no tabulated transfer existed), 32768
  nodes on the FIXED range [1e-4, 1e2] h/Mpc, refusal semantics preserved,
  built per run and passed explicitly (no cache to alias).
- **The out-of-core FFT** (`ooc_fft.py`): host-resident k-space, per-plane
  compute unit, slab knobs that are pure loop bounds; spectral multipliers
  slab-built in f64 on `k_components`' conventions; `plan_bytes` +
  `require_fits` as the accounting function with loud refusals;
  `StagedArray` explicit-IO disk staging.
- **Pairwise 2LPT** (`lpt.py`): `resident="mid"|"low"` policies, bitwise
  identical by construction; no six-derivative moment, no (N^3,3) arrays
  outside the monolithic conveniences.
- **The streamed generator + loader** (`icgen.py`): noise -> table colour ->
  f_NL transform -> 2LPT -> T9 slabs on disk through a +-1-brick-slab
  sliding window; `load_slot_state` reassembles a SlotState through the SAME
  `_alloc_geometry` as `SlotState.build`. The velocity scale is the exact
  partition max; the displacement bound is measured then enforced.
- **Engine changes: none** (the smoke proves it: `engine.run` from a loaded
  state, bitwise the built state's trajectory).

## The estimands and the measured results

| leg | estimand | result |
|---|---|---|
| mirror-license (S0, pre-replacement) | probe mirror == `inexor.ic`, bitwise | n_diff 0 every arm; frozen card `m5_gate_mirror-license.json`; post-replacement era: fields differ at every element, `poisson_factor` still 0 (card `_post`) |
| table bar (exit gate B) | max rel table-vs-EH98 P error over the realized \|k\| range, per config | **2.571e-7** at C-dev = C-gh = C-hero (identical BY the universal-node property) vs the 1e-4 bar; h^2 falsifier slope **1.998** over 800..32768 nodes; 65-point re-sweep drift 0%; exact C-dev multiset max 2.288e-7 <= sweep max. Card `m5_gate_table-bar.json` |
| ledger (leg 0) | colour net footprint, subprocess ru_maxrss at n=256 f64 | **595 MB vs the derived 671 MB bar** (white 1x + spec 2x + out 1x + buffers 1x); old path 2190 MB (~16x field). Realized dtypes + random access ok |
| invariance (leg I) | bitwise across slab decompositions, end to end | laptop n=64: white thickness {1,7,32,64} all n_diff 0; TWO FULL GENERATIONS at slab 7 vs N byte-identical (payload n_diff 0, vel_scale and mean_phi2 equal, f_NL=10). **deneb 427, n=512 f64: same, all clean** |
| e2e (leg III) | load(generate) == monolithic SlotState.build | laptop n=64 and **deneb 427 n=512 f64: n_diff 0 on off/w/occupancy/brick_start, vel_scale exactly equal, occupancy peak/mean 5.9** |
| stats (leg IV) | seed-averaged <P_new>/<P_old>, 32 seeds n=128 | max\|z\| **1.79** (bar 4), chi2/dof **0.53** -- indistinguishable from the split-half control (1.71 / 0.60). Plane hashes distinct, moments 5-sigma clean, new != old bitwise at the same seed |
| engine-smoke | K=3 engine steps from loaded vs built state | n_diff 0, decoded rms 19.1 |
| memory ladder (leg V) | fitted cubic coefficient A of net(n) = A n^3 + C, CPU backend, Vista gg | **PASS (902300): A = 8.79 B/p (bar 9.0), C = -0.18 GB, residuals 0.14/-0.15/0.02 GB; extrapolated 2048^3 total 75.3 GB (bar 77.3 = 116/1.5). Top rung raw: streamed 8.6 B/p vs old-mirror 95.7 -- an 11x reduction, and the old arm REPRODUCES D-v2-15 clause 1's ~90 B/p term (its own control).** The first glibc run (902241) read A = 11.74 -- the pre-registered A > 9 finding -- chased to `rfftn_ooc`'s pass-1 double-buffer (3 spec-equivalents at one moment; finding 7 below) and re-measured. new_mono full 75.5 B/p as expected (the monolithic convenience is not the memory product); its psi1 arm reads 39.5 against a ~28 design, unattributed and deliberately not chased (reported only). Card `m5_gate_memladder.json` |
| 2048^3 OOC FFT (leg VI) | capacity + reference-free correctness on a GH200 | **PASS (Vista 902182, 7m46s, rc=0): roundtrip max\|d\|/rms 2.38e-6 (bar 1e-5), Parseval 1.69e-7 (bar 1e-6), peak host 40.4 GB vs plan_bytes' 36.0 (x1.12, bar 1.3), fwd/inv 85.7/85.9 s, io 0.83/1.42 GB/s write/read, invariance on the GH200 clean, f64 refusal fired, and P(k) vs the BIN-AVERAGED oracle max\|z\| 2.90 over 64 bins (bar 5).** The first run (902091) had failed ONLY its P(k) phase at max\|z\| 8.71 with the bin-centre oracle -- finding 4 below; every other number reproduces to the digit across the two jobs. Card `m5_gate_fft-gh.json` |

## Findings (each one caught by a gate this milestone built)

1. **pocketfft results are batch-size dependent at the bit level.** Per-plane
   rfft2 vs the whole-batch call: 341 differing elements on a 32^3 f64 field
   (pass 1); a 5-column axis-0 chunk: 68 (pass 2); worker count: 0. The
   canonical compute unit is therefore ONE plane / one y-pencil-plane, making
   slab thickness an outer loop bound that cannot move a bit. Cost measured:
   ~7x wall vs batched at 256^3 (0.055 vs 0.008 s) -- noise under the
   memory-first philosophy (~40 s per 2048^3 transform, 86 s measured).
2. **Table nodes must be UNIVERSAL, not per-grid.** With nodes derived from
   (n_mesh, box), two resolutions carry two tables and the interpolation
   error at the same physical k stops cancelling: G5b's re-pointed
   shared-modes check failed at 3-6e-9 where the analytic colour left
   1e-15-class residuals. On fixed [1e-4, 1e2] nodes the error is a function
   of physical k alone and cancels in every shared-k comparison
   (cross-resolution matched phase, box-ladder transport); the check reads
   7-8e-16, the FFT-roundoff floor, with the degenerate identity exactly 0.
3. **Memmap-staged writes count as process RSS.** Dirty pages of a written
   memmap are mapped into the process, so the first ladder smoke booked
   reclaimable page cache as footprint (~175 B/p where the heap holds ~8).
   Staging is explicit IO (`StagedArray`) so ru_maxrss measures the truth.
4. **The bin-centre oracle fails at exactly the scale it matters.** Leg VI's
   P(k) phase evaluated P at the bin-mean k; the deterministic Jensen term
   of P's curvature across a bin reaches z = +15 at k ~ 0.17-0.22 at 2048^3
   mode counts (computed exactly) -- invisible at 512^3 (max|z| 2.24),
   8.7 sigma with 64x the modes. The same trap `local_bispectrum_binned`
   exists for, walked into by a new instrument; the oracle now bin-averages
   over the same modes and weights, and the card stores the z profile, not
   only the max.
5. **macOS allocator retention characterized, not chased**: tracemalloc peak
   0.213 GB at n=256 against an RSS net of 0.95 with every phase
   individually on design -- freed spec-sized buffers retained phase over
   phase (5 spec-equivalents at 256 falling to 1.7 at 512, sub-cubic).
   glibc munmaps large frees; the antares fit coefficient A is the
   transferable number, pre-registered as A ~= 8 with A > 9 THERE a real
   finding.
6. **The non-associativity of a slab-summed reduction was measured before it
   shipped**: per-slab subtotals re-associate and moved <phi^2> at 1e-16;
   `sq_sum_by_plane` THREADS one running fold through the slabs, replaying
   the identical addition sequence under any grouping.
7. **The ladder's fit found a real uncounted cubic term and the arithmetic
   named it.** glibc run 902241: A = 11.74 B/p against the 2-spectrum
   (8 B/p) design with C ~= 0 -- three spec-equivalents co-resident, and the
   one such moment was `rfftn_ooc` at slab = n, where `rfft2_slab`
   materialized the whole pass-1 result as an intermediate before the copy
   into the target (source + intermediate + target). Fixed by writing per
   plane directly into the target rows (bitwise-neutral; the suite's
   identities pass unchanged); the re-run (902300) reads A = 8.79, and the
   monolithic colour/linear_density arms each dropped exactly one
   spectrum-worth (-4.1 B/p), corroborating the mechanism. NB the same
   commit's fix to `_psi_from_spec`'s multiplier build did NOT move the
   monolithic psi arm (39.7 -> 39.5), so that attribution is RETRACTED;
   the arm stays a reported, unchased number. The residual A - 8.0 =
   0.79 B/p is inside the pre-registered bar and deliberately not chased.

## Retired / re-ratified (the replace-in-place bill; D-v2-23 carries the table)

- **f_NL differentiability retired with v1** (JC, 2026-08-10): the generator
  is host numpy; a jnp twin would ship two same-seed fields differing in
  last bits. `colour_white` is the seam a gated twin would be built behind.
- **Names kept** (`gaussian_delta`/`linear_density`/`poisson_factor`
  signatures unchanged); the stream change is carried by IC_STREAM + the
  readout refusal.
- **G5b re-pointed, not frozen** (JC's call): `_colour` now calls
  `ic.colour_white`, `_white_hi` draws the new stream; the identity check
  holds by construction (0.000e+00) and stays a drift detector. Its recorded
  cards stand as measured at their commits.
- **The 48-seed bispectrum calibration PASSES unchanged on the new stream**:
  c_cal 0.9726 (was 0.979), c_centre 0.8588 (was 0.862), discrimination
  control intact -- no constant re-ratified.
- `poisson_factor` is a small-n diagnostic behind a 512 ceiling with the
  arithmetic in its refusal. `m1_export_ics.py` marked historical
  (old-stream; the stored runs/m1 references are LOADED, never regenerated,
  so the D-013 arms survive).
- `test_fnl_enters_linearly_and_differentiably` -> `test_fnl_enters_linearly`
  (the grad arm deleted with the rationale in the test);
  `test_gaussian_delta_table_backend_dc_safe` survives as written (the table
  backend now resamples through ICKTable; same semantics).

## What this does NOT establish

- Cross-MACHINE bitwise identity of the noise stream (XLA-CPU erfinv bits
  across x86 / aarch64 / Apple arm64): the plane-0 fingerprint is on every
  card and leg VI records the GH200's; comparison is reported, never gated.
- A full C-gh IC generation end to end (leg VI runs the FFT layer + colour;
  the complete 2048^3 mock is M-v2-6). The ~1.25 TB staging-traffic budget
  is priced from the measured 0.93/1.75 GB/s, not demonstrated.
- C-hero anything.
- The f32-path invariance beyond the unit tests (gates run f64; f32 arms are
  tested at unit scale only).
- The engine's end-to-end peak (unchanged from M-v2-4's open item).

## Ops

- deneb 427 (anchor, ~4 min): clean. albireo 428 (ladder) CANCELLED unrun --
  its --mem=110G admits only antares (deneb-the-node's 56 GB cannot host the
  old arm's ~90 GB at n=1024) and antares sat occupied; the ladder moved to
  a Vista gg node (JC's call).
- Vista jobs: 902091 leg VI (7m45s; failed only its own P(k) instrument,
  finding 4); 902182 leg VI PASS (7m46s); 902236 ladder DIED AT PRECONDITION
  -- jax 0.10's CUDA plugin hard-raises on cuInit (error 303) on a GPU-less
  gg node, so `CONDA_OVERRIDE_CUDA` alone no longer suffices and
  `JAX_PLATFORMS=cpu` is LOAD-BEARING in any gg sbatch; 902241 ladder ran
  clean and delivered the A = 11.74 finding; 902300 ladder PASS (27m52s).
  All with `-A JPL-SPHEREx`.
- Total cluster cost: ~1.2 gg/gh node-hours + ~5 min of deneb.
- The branch was pushed by Claude at JC's explicit authorization
  (2026-08-10); deneb and Vista checkouts synced to it.
