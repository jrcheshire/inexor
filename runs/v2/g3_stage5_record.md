# G3 Stage 5 -- the mechanism ensemble at cdev

**Written before the verdict is known.** The design, the pre-registered
predictions and the reading rules below were fixed on 2026-08-05, before any
cdev Stage 5 arm had been evolved. Only the results sections are filled in
afterwards. Producer `scripts/v2_g3_stage5.py`, readout
`scripts/v2_g3_stage5_readout.py`, jobs `scripts/v2_g3_stage5_deneb.sbatch`.

## Why this is not the Stage 5 the plan described

Two measured results moved it, and both are in the earlier records.

**The pinned gate triangles are decorrelated.** The estimand's small-scale leg
is `k_short = 1.178 h/Mpc`. The tiled arm holds `r >= 0.5` against the
monolithic one only out to 0.47 h/Mpc at cdev with an 8 Mpc/h buffer, and 0.90
with 16 Mpc/h (`g3_decorrelation_record.md`). Under the ratified
`R_CONDITION_CUT = 0.5` that is **zero gradeable triangles at every cdev
geometry**. `R_Q` cannot simply be read there anyway: once two fields
decorrelate the ratio of their bispectra saturates at a bounded value and stops
being monotone in brokenness, so the most broken arm can score best
(`g3_ladder_record.md`).

**The plan's geometry is dominated.** The absolute buffer in Mpc/h sets usable
`k`; tile size sets only the price (paired ratio 0.993 at a fixed 8 Mpc/h buffer
across a 2.37x difference in padded volume). The plan's `T=64/b=16` is beaten by
`T=128/b=32` on both axes at cgh64.

So Stage 5 scans a **buffer ladder in absolute units** against a **ladder of
small-scale legs**, and the eligibility boundary between them is a measured
output. This is the "genuine A3 result rather than a measurement problem" the
Stage 4 record named as the substance of the Stage 5 design decision.

## What is ratified and not re-opened here

Bar 15% (D-v2-7); gate on `R_Q` with `rho` required to agree; estimand pinned at
cdev; reduce by MAX over gate-eligible triangles; the `R` vs `x` curve mandatory
and ungated (inherits D-v2-9 clause 3); conditioning cut `r >= 0.5`; brackets
discriminate on `r`, never on `R_Q`. JC's Stage 5 calls (2026-08-05): gate at
`r >= 0.5` with the `r >= 0.9` subset reported alongside; scan the small-scale
leg; include the position-dependent P(k) statistic and the equilateral floor
re-run; 8 seeds on deneb reusing seeds 1-8.

## The measurement

`cdev` (L = 128 Mpc/h, `n_fine` = 512, `n_part` = 256, fine cell 0.25 Mpc/h),
seed 0 as a pilot then seeds 1-8 -- the same realizations as the correlation
ensemble, so every comparison stays paired.

| arm | T | b | b (Mpc/h) | P | tiles | cost (mono evolves) | role |
|---|---|---|---|---|---|---|---|
| A | 128 | 16 | 4 | 160 | 64 | 1.95 | cheap end |
| B | 128 | 32 | 8 | 192 | 64 | 3.38 | the cheap operating point |
| C | 128 | 64 | 16 | 256 | 64 | 8.00 | the only cdev geometry reaching `r = 0.5` near 0.9 h/Mpc |
| D | 64 | 32 | 8 | 128 | 512 | 8.00 | arm B's ABSOLUTE buffer at half the tile |
| kill | 128 | 0 | 0 | 128 | 64 | 1.00 | the bracket's power |
| span | 2LPT | - | - | - | - | ~0 | the bracket's dynamic range |

`k_short` = 6, 12, 18, 24 `k_f` = 0.295, 0.589, 0.884, 1.178 h/Mpc, all computed
from the same evolved fields, so the scan costs estimator time only. Long legs
`m = 1..7`, `dk = k_f` on every leg -- the binning of the cdev pin run, kept so
that card's `sigma_B` applies to these numbers.

**Arm D is the one addition beyond the buffer ladder.** "Absolute buffer sets
the physics" was measured on the correlation `r(k)` only; the decorrelation
record states explicitly that nothing so far bears on the seam statistic. If
arms B and D agree, the rule transfers to `R_Q` and Stage 6 may pick a geometry
with it. If they disagree, it may not.

## The reading rules, fixed in advance

- **The eligibility map is printed first and appears first in this record.** A
  table of `R_Q` alone cannot be interpreted, because an empty row looks exactly
  like a passing one.
- **A cell with no gradeable triangle reads NOT GRADEABLE and never PASS.** The
  producer refuses to emit a verdict field rather than emitting a passing one.
  Two ways a cell fails to be gradeable, both recorded: the arms are
  decorrelated at that `k_short` (`r < 0.5`), or no triangle at that `k_short`
  is actually squeezed (`k_short < 2 k_long` for every long leg).
- **The estimand is MAX |R_Q| over gate-eligible SQUEEZED triangles.** The
  equilateral is a reported control and never enters it.
- **`rho` must agree with `R_Q` per triangle**, not maximum against maximum --
  two maxima can sit on different triangles and agree numerically while the
  curves disagree everywhere. A divergence flags window contamination, not a
  tiling failure.
- **Ratios aggregate in log**, and a sem band bracketing 1.0 is a null.
- **Wall times compare only within a host** (7.4x on Vista vs 4.0x on deneb for
  the same box and configs).
- **No cost or memory number here is a Pareto point.** deneb is correctness
  ground (D-v2-10); performance comes from the config-table homes.

## Pre-registered predictions

- `R_Q < 0`: the tile cannot feel the long mode's nonlinear response, so a
  positive `R_Q` of the same size is a bug signature, not a result.
- A step at `x = k_long / k_P ~ 1`, the tile's own fundamental.
- Monotone in `b` above the decorrelated regime, and `R_Q -> 0` as `b -> box`.
- The equilateral control hurt less than the squeezed triangles. If both are hurt
  equally the error is a generic small-scale error and the squeezed framing is
  wrong.
- The position-dependent response error smaller than `R_Q` wherever the phases
  have decorrelated, since it is an amplitude statistic and does not see phases.

## Machinery built for this stage

- `inexor.diagnostics.subvolume_response` -- position-dependent P(k), in the
  package with tests (`tests/test_subvolume_response.py`, 10 tests). Sub-volume
  means and per-sub-volume band powers, regressed to give `dlnP/ddelta_bar`.
  Exists because `R_Q` cannot separate amplitude-wrong from phase-wrong: Stage 4
  measured pure 2LPT as 15x better correlated yet carrying twice the `R_Q`.
  - **The sub-volume lattice must not sit inside the tiles**, or it measures
    what the tiling zeroes and both arms are biased together. The obvious guard,
    coprime counts, is **unavailable**: an equal-cube split needs `n_sub | N`,
    `N` is a power of two, and so is the tile count per side, so every admissible
    pair shares a factor. The lattice is offset by half a sub-volume instead, and
    the estimator MEASURES the fraction of sub-volumes crossing a tile wall,
    raising if that fraction is zero.
  - Recovery of an injected coupling is exact to a measured 1.7e-2, and that
    residual scales linearly in the injected amplitude -- it is the O(amp^2)
    truncation of the expected-value formula, not estimator error.
  - The straddle-fraction arithmetic is cross-checked against an independent
    cell-level count and **verified to discriminate**: three mutations of the
    wall logic produce 6, 23 and 6 disagreements. The third is caught only by the
    offsets that put a tile wall on a block's first and last cell.
- `scripts/v2_g3_floors.py --window flat` -- a scale-independent 5% transfer, so
  the equilateral control has a response at all. The gaussian default is
  unchanged and every existing card stays comparable. Verified at smoke:
  `R_Q` = 0.0476 under a pure flat window (the 1/T response the closed form
  predicts), `rho` flat to 4.4e-16, `W` floor 8.4e-17.
- `scripts/v2_g3_stage5.py` cross-checks `k(r=0.5)` for every arm against the
  tracked decorrelation cards and raises on a mismatch. **The tolerance is
  deliberately 1%, not round-off**: those cards were produced on Vista
  (linux-aarch64) and this runs on deneb (linux-64), so an f64 N-body run through
  two XLA-CPU backends is not bitwise comparable and a 1e-6 bound would fire on
  the machine rather than on a regression. 1% is ~12x below the seed-to-seed
  scatter of the quantity and still catches any real change.

## Results

*(to be filled in from the pilot and the 8-seed ensemble; nothing at cdev has
been run yet)*

### 1. Eligibility map

### 2. Max |R_Q| over gate-eligible triangles

### 3. Buffer vs tile size, paired

### 4. Position-dependent P(k)

### 5. Brackets

## Not licensed / owed

- Nothing here is a cgh64 statement. Stage 5 is the MECHANISM ensemble and
  Stage 6 is the geometry verdict; the `x < 1` regime is statistics-starved
  whenever the box is not much larger than the padded tile, which is why they
  are different jobs and must not be read for each other.
- The cgh64 rung still has no error bar (~84 Vista node-hours for 8 seeds), and
  resolving it stays a Stage 6 scoping call.
- The resolution axis is untested: every configuration here is at the 0.25 Mpc/h
  fine cell.
- The position-dependent response is REPORTED, NOT GATED. No bar for it has been
  ratified, and proposing one is a checkpoint decision.
