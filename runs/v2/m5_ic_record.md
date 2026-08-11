# M-v2-5: streamed ICs + the out-of-core FFT -- the milestone record

**Status: DRAFT -- two cluster legs pending (deneb 428 memory ladder; Vista
902182 leg VI rerun).** Branch `jc/m-v2-5-streamed-ics`; ADR D-v2-23 to
follow. Charter: D-v2-18's ladder row, re-scoped per D-v2-15 clause 5
(JC-ratified 2026-08-10, the third milestone whose written exit criterion
could not be read literally; the gate docstring carries the reading).

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
| memory ladder (leg V) | fitted cubic coefficient A of net(n) = A n^3 + C, CPU backend, antares | **[PENDING -- deneb 428]** bars: A <= 9.0 B/p, extrapolated 2048^3 < 77.3 GB; tier 2 A ~= 8.0 |
| 2048^3 OOC FFT (leg VI) | capacity + reference-free correctness on a GH200 | first run (902091): **roundtrip max\|d\|/rms 2.38e-6 (bar 1e-5), Parseval 1.69e-7 (bar 1e-6), peak host 40.4 GB vs plan 36.0 (x1.12, bar 1.3), fwd/inv 85.7/86.1 s, io 0.93/1.75 GB/s write/read; invariance on the GH200 clean; f64 refusal fired.** P(k) phase failed at max\|z\| 8.71 -- the INSTRUMENT (below). Rerun with the bin-averaged oracle: **[PENDING -- Vista 902182]** |

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

- deneb 427 (anchor, ~4 min in queue-to-done): clean. 428: pending behind
  the mdnilc array on antares at submit time.
- Vista 902091: 7m45s wall; failed only on the instrument defect above.
  902182 = the rerun. Both submitted with `-A JPL-SPHEREx` (the sbatch now
  carries it).
- The branch was pushed by Claude at JC's explicit authorization
  (2026-08-10); deneb and Vista checkouts synced to it.
