# M1 results — forward PM validation

Working record for milestone M1 (plan: tender-stargazing-map). Sections fill
in as S5-S7 land; the parity gate numbers are ratified with JC only AFTER the
floors below (floor-first protocol; see CLAUDE.md tolerances convention).

Machine key: all S5 numbers measured on the laptop (M4 Max, "andromeda") —
mbody is MLX/Metal-only so the harness lives here; deneb reserved for the S7
CUDA legs (over-subscribed as of 2026-07-13).

## S5 — harness + measured repro floors (2026-07-13)

Harness: `scripts/_m1_common.py` (numpy-only neutral CIC + P(k)/r(k)
estimator, npz exchange schema) + `m1_export_ics.py` / `m1_run_mbody.py` /
`m1_run_disco.py` / `m1_parity.py`, one pixi env per code, cross-process npz
(xcheck.py pattern). All matched configs: L = 256 Mpc/h, a: 0.1 -> 1.0, log
spacing, BullFrog, 2LPT ICs, seed 0, shared a_steps array (np.geomspace —
bitwise the inexor grid; every code integrates on the SAME array).

Sanity identities checked before any run:
- DISCO-DJ q-ordering vs the inexor/mbody Lagrangian grid: EXACT (max diff 0.0).
- a_grid: _m1_common == inexor.integrate.a_grid bitwise (asserted at export).
- CAMB tables: parity-env camb 1.6.5 vs mbody's internal backend agree to
  max |dP/P| = 6.3e-4 (interpolation/k-grid class; either source works, both
  written: `runs/m1/pk_camb.txt`, `runs/m1/pk_camb_mbody.txt`).
- CAMB vs EH98 (context for the S6 deficit axis): max |dP/P| = 3.8%
  (BAO wiggles + small-scale).

### Repro floors (identical-input repeats, x5, worst pair vs run 0)

| code | config | rms dx [cells] | max dP/P | max 1-r |
|---|---|---|---|---|
| mbody (MLX f32, Metal) | 64^3 K=10 | 5.0e-7 | 5.8e-8 | 5.3e-13 |
| mbody (MLX f32, Metal) | 128^3 K=40 | 1.0e-6 | 6.0e-8 | 7.2e-13 |
| DISCO-DJ (JAX f64 CPU) | 64^3 K=10 | 0 (bit-identical) | 0 | 3.3e-16 |
| DISCO-DJ (JAX f64 CPU, fresh-trace) | 64^3 K=10 | 0 (bit-identical) | 0 | 3.3e-16 |
| DISCO-DJ (JAX f64 CPU) | 128^3 K=40 | 0 (bit-identical) | 0 | 2.2e-16 |
| DISCO-DJ (JAX f64 CPU, fresh-trace) | 128^3 K=40 | 0 (bit-identical) | 0 | 2.2e-16 |

(The 1-r "3.3e-16" on bit-identical inputs is the estimator's own float64
epsilon, not a field difference. mbody's floor is the expected Metal
CIC-scatter class from its own docs, ~1e-6 cells.)

### First parity numbers (64^3 K=10, matched-IC injection; NOT gates yet)

| pair | rms dx [cells] | max dP/P | max 1-r |
|---|---|---|---|
| inexor_float (f64) vs mbody | 4.2e-6 | 1.1e-6 | 4.3e-11 |
| inexor_float (f64) vs DISCO-DJ | 3.6e-3 | 1.7e-2 | 1.3e-4 |
| inexor_float (f64) vs inexor int16 (Tier B) | 1.8e-3 | 1.5e-4 | 7.7e-6 |

Readings (preliminary, S6 will formalize):
- **inexor vs mbody sits at mbody's own f32 floor** — the physics port is
  faithful to the source code at its repeatability limit.
- **inexor vs DISCO-DJ deviation is k-INCREASING**: |dP/P| 1.1e-3 in the four
  lowest-k bins -> 4.6e-3 mid -> 1.7e-2 approaching Nyquist; 1-r low-k 2.7e-7.
  The shape says force-kernel / window-implementation class, not growth
  coefficients (those would be k-flat). Low-k (the f_NL regime) dynamics
  agree at the 1e-3 level. Attribution -> S6 matrix.
- **Tier B (int16 vs f64 float)** is k-flat at the few-1e-5 level with a
  1.5e-4 worst bin at 64^3 K=10 — measured against the D-010 1e-4-bar
  convention this needs the proper S6 quantization-gate config (K-vs-2K +
  mesh arm) before any verdict; recorded, not relaxed, not passed.

### 128^3 K=40 parity (gate-grade config)

| pair | rms dx [cells] | max dP/P | max 1-r |
|---|---|---|---|
| inexor_float (f64) vs mbody | 1.6e-5 | 1.4e-6 | 2.3e-10 |
| inexor_float (f64) vs DISCO-DJ | 1.2e-2 | 1.0e-1 | 1.6e-3 |
| inexor_float (f64) vs inexor int16 (Tier B) | 8.3e-3 | 2.8e-4 | 5.6e-5 |

k-band structure:

| pair | low-k (4 bins) | k < 0.4 | mid | near Nyquist (k~1.5) |
|---|---|---|---|---|
| vs mbody, dP/P | 7.3e-7 | 1.2e-6 | 1.4e-6 | 5.7e-7 |
| vs DISCO-DJ, dP/P | 1.8e-3 | 1.6e-2 | 4.8e-2 | 1.0e-1 |
| vs int16, dP/P | 1.3e-4 | 2.0e-4 | 1.9e-4 | 9.0e-5 |

Readings:
- **mbody agreement is k-FLAT at ~1e-6 dP/P over all 40 steps** — at/near
  mbody's own measured floor (6e-8 per repeat, f32 accumulation class).
  Tier-A shape for the ported physics: as good as the reference can resolve.
- **DISCO-DJ deviation grows with k AND with (K, resolution)**: at fixed
  k ~ 0.4-0.8 it is 4.8e-2 here vs 1.7e-2 in the 64^3 K=10 run. Low-k stays
  1.8e-3 / r = 1 - 7e-7. Consistent with a per-step force difference at high
  k (kernel/window handling), compounding over steps — S6 attribution axes:
  deconvolve, worder, grad_kernel_order 0 vs 2, K, res, + growth-table diff.
- **int16 quantization is k-flat at 1-2e-4 dP/P at this config** (worst bin
  2.8e-4) — sits ABOVE the D-010 1e-4 R2-class bar as a raw reading; the
  formal Tier-B verdict belongs to the S6 quantization gate (K-vs-2K + mesh
  arm at the ratified config), where the bar is re-ratified with JC. Recorded
  as measured; nothing relaxed.

## S6 — parity gates + deficit attribution

### DISCO-DJ gap: ATTRIBUTED, closed at machine precision (2026-07-13)

The inexor <-> DISCO-DJ parity gap decomposes into exactly THREE named
convention differences, each verified against discodj 0.0.2 source
(`nbody/acc.py`, `core/kernels.py`, `nbody/steppers/dkd_pi_integrator.py`,
`disco_stepper.py`); probes: `scripts/m1_kernel_probe.py`,
`scripts/m1_force_probe.py`.

1. **Force operator — Nyquist plane in the order-0 gradient kernel.**
   DISCO-DJ zeroes the Nyquist plane of ik in the gradient direction; the
   mbody lineage keeps -i k_nyq. With that one change, the two PM force
   operators agree on the same particle configuration to
   **rms 7e-8 relative (f64 FFT roundoff)** — there is NO other force
   difference (probe: m1_force_probe --side disco/inexor).
2. **BullFrog alpha — true-LCDM D2 vs EdS relation.** DISCO-DJ evaluates
   alpha with its tabulated second-order growth D2plus(a); mbody/inexor use
   the EdS relation E = -(3/7) D^2 on exact LCDM D (documented mbody
   approximation, ~0.8% in D2 at z=0).
3. **Midpoint/drift convention for explicit a-step arrays.** With time_var
   given as an array, DISCO-DJ's internal time is the STEP INDEX, so
   a_mid = (a0+a1)/2 (arithmetic in a; disco_stepper.internal_to_a linear
   interp) and the two D-drift halves are UNEQUAL (dd1/dd2 = 1.0008 at
   64^3 K=10 leading step); mbody/inexor use D_mid = (D0+D1)/2 with equal
   halves.

**Closure test**: replaying inexor's f64 float path with (1)+(2)+(3) —
zeroed-Nyquist kernel + DISCO-DJ's exact dumped runtime coefficients
(m1_force_probe --side coeffs / --side replay) — reproduces disco_final to

| config | rms dx [cells] | max dP/P | max 1-r |
|---|---|---|---|
| 64^3 K=10 | 1.6e-8 | 4.1e-9 | 7.1e-15 |
| 128^3 K=40 | 2.2e-8 | 1.9e-8 | 2.1e-14 |

i.e. **f64 roundoff**. The raw stock-vs-disco numbers in the S5 tables are
therefore CHARACTERIZED CONVENTION DIFFERENCES between two converged
integrators, not errors: the K-scan (K=1/10/80 at 64^3: rms 4.7e-2 /
3.6e-3 / 3.7e-3 cells) shows the raw gap converging to the fixed
operator/coefficient difference, and the growth tables agree to 5e-7 so
none of it is D(a).

Supporting negative results (recorded to prevent re-derivation): zeroing
Nyquist alone halves the displacement rms but moves dP/P by only ~4e-4 —
the P-ratio share of the gap is dominated by (2)+(3), the displacement-rms
share by (1). DISCO-DJ's background has no radiation (Omega_r None), same
as ours.

### Proposed DISCO-DJ Tier-A gate (for ratification)

Gate on the CONVENTION-ADAPTED comparison (the replay above), which pins
the shared physics rather than the convention choices:
- rms displacement <= 1e-6 cells (measured 2.2e-8: ~50x headroom)
- max |dP/P| <= 1e-6, max 1-r <= 1e-12 (measured 1.9e-8 / 2.1e-14)

The raw (stock-convention) comparison is documented, not gated — its size
is set by DISCO-DJ's kernel/coefficient conventions, outside inexor's
control. mbody Tier-A gate proposal: rms <= 1e-4 cells / |dP/P| <= 1e-5
(measured 1.6e-5 / 1.4e-6 at 128^3 K=40 vs its own 1e-6-cell floor).

### 4%-deficit attribution: CLOSED (2026-07-13)

The M0 observation ("evolved P(k) ~4% below linear at k = 0.025, 64^3, one
seed") decomposes into three quantified pieces, none an integrator error
(`scripts/m1_deficit.py`; runs/m1/deficit_matrix.json + deficit_seeds.json):

1. **Bin-center binning artifact (estimator-side, deterministic).** In
   fundamental-width shells the discrete modes sit at |k|/k_f in {1, sqrt2,
   sqrt3, 2, ...} while the theory was read at bin centers; with P(k)
   falling above the turnover, mode-averaged theory sits BELOW bin-center
   theory by -13.4% / -6.8% / -4.4% / -1.1% in the lowest four shells
   (64^3, L=256). Re-measured against MODE-AVERAGED theory, the IC field's
   absolute P(k) is +0.5% +- 2.1% (100 seeds) — consistent with zero. Any
   absolute-vs-theory reading at these bins MUST bin the theory like the
   data (the M0 probe did not; its linspace binning has the same artifact
   class).
2. **Per-realization scatter.** The absolute metric scatters +-12% per seed
   over the k <= 0.05 band (chi^2 statistics of ~tens of modes); single-seed
   absolute readings at the fundamental bins are noise-dominated. The M0
   "-4%" was one draw of artifact + scatter.
3. **Real dynamics: growth suppression -2.6% +- 0.5%** (16-seed mean at
   64^3 K=10 BullFrog 2LPT; seed scatter 2.0%), measured with the
   variance-cancelling growth transfer T(k) = P_f/P_i vs (D_f/D_i)^2 (IC
   realization, CIC window, and discreteness cancel). Attribution evidence:
   - **K-independent** (K = 5/10/20/40 identical to 4 digits) and
     **integrator-independent** (BullFrog/FastPM/exact within 0.15%
     absolute, converging with K) -> not stepping.
   - **ZA vs 2LPT differ by only 0.1%** -> not the LPT transient.
   - **Reproduced by BOTH references to 4 digits** at the shared cell
     (inexor -2.67%, mbody -2.67%, DISCO-DJ -2.65%) -> not an inexor
     artifact; property of the shared PM dynamics.
   - **Amplitude test**: deficit shrinks with IC amplitude but NOT purely
     as amplitude^2 (-2.67% / -1.72% / -1.36% at amp 1 / 0.5 / 0.25) —
     mode-coupling term + an amplitude-independent component.
   - **Resolution/band tests**: band-limiting the 128^3 ICs at the 64^3
     Nyquist changes nothing (-4.73% -> -4.72%); the same band-limited
     realization on the 64^3 mesh gives -5.2% vs 128^3's -4.7% (small,
     OPPOSITE sign to a discreteness story). The apparent 64^3 vs 128^3
     doubling in the raw matrix is realization difference (different
     seed-0 fields per resolution), inside the +-2% seed scatter.
   - Spacing (log vs linear) and EH98-vs-CAMB table: no effect (<0.1%).

   Verdict: a real, small, code-independent nonlinear growth suppression
   at the near-fundamental bins of this (256 Mpc/h) box, 1-loop
   mode-coupling class with large per-realization scatter. A quantitative
   1-loop SPT cross-check of the -2.6% mean is deliberately deferred to
   paper-time (PT kernel formulas to be verified against primary sources
   before use — subagent/memory rule).

   En route library fix: gaussian_delta(backend="table") evaluated the
   colour at the DC mode and tripped the table-range guard — DC-safe
   evaluation + regression test (tests/test_ic.py).

### Tier-B quantization gate measurement (m1_quant_gate.py, 128^3 K=40)

CUBE-style floor comparison, all arms from the same injected ICs vs the
evolve_float f64 reference (runs/m1/quant_gate_n128k40log_bullfrog_lpt2_s0.json):

| arm | rms [cells] | max dP/P | dP/P low-k | k<0.4 | mid | Nyquist |
|---|---|---|---|---|---|---|
| quant (int16 production) | 8.3e-3 | 2.8e-4 | 1.3e-4 | 2.0e-4 | 2.8e-4 | 9.0e-5 |
| step floor (K vs 2K) | 7.7e-4 | 3.2e-3 | 1.8e-6 | 1.7e-4 | 2.8e-3 | 3.2e-3 |
| mesh floor (n vs 2n force) | 2.7e-1 | 5.3e-1 | 1.2e-3 | 5.1e-2 | 4.3e-1 | 5.3e-1 |
| f32-kernel share (f32 vs f64 float) | 2.4e-6 | 4.0e-7 | — | — | — | — |

Readings:
- The int16 error is **k-flat at 1-3e-4** and is **pure phase-space
  quantization** (the f32-kernel share is 4e-7 — three orders below).
- **Below the PM mesh floor in every band** (9x margin at low k, ~2000x
  mid); below the stepping floor everywhere except low k, where BullFrog's
  linear-growth exactness makes the stepping floor artificially tiny
  (1.8e-6) — the governing method floor there is the mesh arm's 1.2e-3.
- Relation to the D-010 1e-4 bar: that bar was ratified on M0-R2's
  like-for-like ladder-budget statistic; this end-to-end Tier-B statistic
  is a different measurement and needs its own gate number (JC's call) —
  quoted candidates: max |dP/P| <= 5e-4 at the production config (measured
  2.8e-4) plus the strict below-mesh-floor invariant per band.

## S7 — deneb CUDA legs + at-scale smoke (2026-07-13)

Authoritative CUDA re-confirmation on deneb's RTX 3050 (the **6 GB** variant),
via Slurm (`scripts/m1_deneb.sbatch`; job 12). Two legs, `set -e` chained:

1. **Full gpu-env pytest** — the tier-0 exact-reversibility and int-paint
   determinism tests ARE the CUDA re-confirmation arms. **Clean** (all passed /
   1 skipped for pyccl outside the parity env). This is the first suite run
   with the M0->package migration bridge (`test_m0_bridge.py`) removed: it was
   the temporary "package bit-matches the frozen `_m0_common` archive" check,
   marked delete-at-M1-close, and its final CUDA run (job 10) confirmed every
   meaningful arm bit-matches on GPU. Its lone failure was an f32-paint
   bit-equality assert, which is invalid on GPU by construction — f32 CIC
   scatter is non-associative, so two differently-compiled `paint_f32` programs
   need not agree bit-for-bit (exactly R3's "f32 nondeterministic everywhere";
   only int paint is bit-deterministic, and only int paint is on the reversible
   path). Removed at M1 close as planned.

2. **At-scale forward smoke + exact roundtrip** (`scripts/m1_smoke.py`,
   `runs/m1/smoke_n256.json`; 2LPT ICs -> int16 BullFrog, perstep driver, K=10,
   L=256 Mpc/h):

   | leg | n | result |
   |---|---|---|
   | forward | 256^3 (16.7M particles) | 3.0 s, x_finite=True, no wrap |
   | roundtrip (K fwd + K rev) | 256^3 | **exact=True, n_diff=0** (2.0 s) |

   The n_diff=0 roundtrip is the product claim — bit-exact reversibility — now
   demonstrated at scale on CUDA, not just in the unit tests. Velocity-frame
   headroom: max|w| climbs monotonically to 27864 / 32767 = **0.85 of the int16
   range** by the final step (`max_abs_w_per_step` in the json). No overflow
   here, but ~15% margin — a longer schedule or higher sigma8 would eat into it;
   the s_w0 policy governs this and stays a per-run diagnostic (W_ABS_WARN).

**512^3 does not fit the 6 GB card**: OOMs at IC generation (the 134M-particle
`max|v0|` reduce needs ~1.5 GiB on top of the 1.6 GB velocity array), even at
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`. Fell back to 256^3 (documented in the
json: `fallback_used=True`, `oom_at_n=512`). The 512^3 headline run is deferred
to a larger-GPU env (Vista aarch64, an M2 backlog item pending the aarch64 pixi
feature).

**Finding — the plan's pre-agreed 384^3 fallback was structurally invalid.**
`BoxConfig` enforces `2^16 % n_mesh == 0` (the exact-Lagrangian-site sublattice
invariant, architecture.md Sec. 3); 65536 / 384 = 170.67, and the only divisors
of 2^16 are powers of two, so **256^3 is the sole valid mesh below 512^3**. The
M0 R3 probe ran 384^3 only because `_m0_common` predates that invariant. Job 11
surfaced this (raised `ValueError` after the 512^3 OOM); the fallback default
was corrected 384 -> 256 (commit 26a235c).

**Slurm ops** (recorded in the umbrella albireo memory): deneb jobs REQUIRE an
explicit `--mem` — the partition default (124000M, sized for antares's 128 GB)
exceeds deneb's 56 GB and makes the job permanently unschedulable (job 9 pended
forever; fixed with `--mem=16G`). The gpu smoke co-schedules beside CPU jobs
(OverSubscribe=OK).

**M1 CLOSED.** Forward PM validated: mbody parity at mbody's own floor (Tier A),
DISCO-DJ gap attributed to 3 conventions and closed at f64 roundoff, the M0
"4% deficit" decomposed, Tier-B int16 quantization gated (D-014), and exact
reversibility confirmed at scale on CUDA. Next: M2 (the adjoint), opening with
its own detailed milestone plan.
