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

**Pilot only: deneb job 360, seed 0, 2026-08-05, `main @ 490a414`, card
`g3_stage5_cdev.json`. ONE SEED, so nothing below carries an error bar and no
ranking between arms is licensed** (the decorrelation ensemble measured 12-22%
absolute seed scatter on the neighbouring quantity). The 8-seed ensemble was
held pending this map, as designed.

**HEADLINE: the pilot landed outside all three pre-registered outcomes.** The
eligibility half reads as outcome 1 (a cheap buffer IS gradeable at some
`k_short`), but no gradeable cell yields a verdict, because **every one of the
six eligible cells fails the ratified `rho`-agreement precondition**. The
correct statement is therefore neither "A3 is alive at a quotable price" nor
"nothing is gradeable": the gate is ELIGIBLE but UNREADABLE at cdev.

### 1. Eligibility map

`r` at the small-scale leg, threshold 0.5 (`*` = gradeable):

| arm | b [Mpc/h] | cost | 0.295 | 0.589 | 0.884 | 1.178 |
|---|---|---|---|---|---|---|
| T=128 b=16 | 4 | 1.95x | 0.700\* | 0.368 | 0.080 | -0.038 |
| T=128 b=32 | 8 | 3.38x | 0.708\* | 0.393 | 0.113 | -0.005 |
| T=128 b=64 | 16 | 8.00x | 0.921\* | 0.762\* | 0.517\* | 0.276 |
| T=64 b=32 | 8 | 8.00x | 0.774\* | 0.430 | 0.119 | -0.011 |

Six of sixteen cells are eligible. **The pinned gate leg (`k_short` = 1.178
h/Mpc) is gradeable NOWHERE, including at the 8x buffer** (`r` = 0.276). Stage
5's founding premise was inferred from the correlation cards; it is now measured
directly on the seam configuration, and it holds.

The `k(r=0.5)` anchors reproduce the tracked Vista cards (0.452 / 0.472 / 0.900
/ 0.517 h/Mpc) within the 1% cross-arch tolerance, so the cross-machine
comparison is sound.

### 2. Max |R_Q| over gate-eligible triangles

| arm | 0.295 | 0.589 | 0.884 | 1.178 |
|---|---|---|---|---|
| T=128 b=16 | 0.547 `!rho` | -- | -- | -- |
| T=128 b=32 | 0.700 `!rho` | -- | -- | -- |
| T=128 b=64 | 0.106 `!rho` | 0.161 `!rho` | 0.231 `!rho` | -- |
| T=64 b=32 | 0.898 `!rho` | -- | -- | -- |

`--` is NOT GRADEABLE and never a pass. `!rho` = the cell fails the
per-triangle `rho`-agreement requirement, so its `R_Q` is not a verdict.

**The `rho` failures are genuine curve-level disagreement, not the
normalization.** `max_rho_deviation` divides each triangle's `|R_Q - rho|` by
that triangle's own `|R_Q|`, which diverges at a zero crossing, so the printed
122 and 229 are inflated. The ABSOLUTE differences are not:

| arm, `k_short` | `R_Q` range | `rho` range | max abs diff |
|---|---|---|---|
| b=32, 0.295 | -0.375 to -0.700 | +0.057 to -0.369 | 0.43 |
| b=64, 0.589 | -0.161 to +0.153 | +0.293 to +0.823 | 0.80 |
| b=64, 0.884 | -0.198 to +0.231 | +1.970 to +3.269 | 3.23 |

**`R_Q` and `rho` diverge because `rho` diverges, and section 8 identifies the
cause: in this regime `rho` measures `1/r^2` and nothing else.** The first
reading of this card said the divergence showed a squeezed bispectrum wrong by
a factor of 2 to 4 at b=64. **That reading was wrong and is retracted.** The
large `rho` values are a restatement of the decorrelation already reported in
the eligibility map, not an independent measurement of tiling error. See
section 8.

What survives from this section: no cell yields a verdict under the ratified
precondition, and the raw disagreement between the two statistics is large in
absolute terms. WHY it is large is section 8's subject, and the answer changes
which statistic is at fault.

### 3. Buffer vs tile size, paired

Not computable: the paired section needs >= 2 seed cards and the pilot is one.

The pre-registered arm-D question ("does absolute buffer set the physics for
the SEAM statistic, as it does for the correlation?") **cannot be answered from
this card**. Arms B and D share an 8 Mpc/h buffer and give `R_Q` = 0.700 vs
0.898, a 28% disagreement, and `rho` = 0.369 vs 0.862, a factor 2.3 -- but BOTH
cells are `!rho`, so those are refused numbers and quoting them as a comparison
would be reading a statistic the gate has already rejected. On `r`, where the
numbers are admissible, arm D reaches `k(r=0.5)` = 0.517 against arm B's 0.472,
a 9.5% difference at 2.4x the cost. One seed, no error bar. **Stage 6 may not
use the transfer rule to pick a geometry on this evidence.**

### 4. Position-dependent P(k)

REPORTED, NOT GATED (no bar ratified). Tiled/mono response ratio minus 1, at
`n_sub` = 4 (straddle fraction 1.000, so every sub-volume crosses a tile wall
as intended):

| arm | k=0.29 | k=0.59 | k=0.88 | k=1.18 |
|---|---|---|---|---|
| T=128 b=16 | -0.171 | +0.316 | +0.186 | +0.304 |
| T=128 b=32 | -0.207 | -0.038 | -0.031 | -0.003 |
| T=128 b=64 | -0.018 | +0.044 | +0.075 | +0.084 |
| T=64 b=32 | -0.086 | +0.046 | +0.047 | +0.167 |

The statistic remains finite and small at `k` = 1.18 where `R_Q` is refused
outright, which is the property it was built for: it is an amplitude statistic
and does not see the decorrelated phases. Whether it is a usable substitute
gate is a checkpoint decision, not something this card settles.

### 5. Brackets

On `r`, never on `R_Q`:

| k | pivot | kill (b=0) | 2LPT |
|---|---|---|---|
| 0.049 | 0.9985 | 0.9980 | 1.0000 |
| 0.098 | 0.9642 | **0.9649** | 0.9997 |
| 0.147 | 0.9238 | 0.8723 | 0.9992 |
| 0.196 | 0.8932 | 0.8418 | 0.9981 |
| 0.245 | 0.7818 | 0.5858 | 0.9943 |
| 0.295 | 0.7737 | 0.5584 | 0.9904 |
| 0.344 | 0.6981 | 0.4597 | 0.9813 |

- 2LPT more correlated than the pivot at every shell: **True**.
- Kill control less correlated than the pivot at every shell: **False**, and by
  7e-4 at a single shell (k = 0.098), against a quantity whose seed scatter is
  percent-level. **Treat as unresolved at one seed, not as a bracket failure.**
  It is the one result here that a second seed would settle cheaply.

### 6. Pre-registered predictions, graded

- **`R_Q < 0`: the prediction FAILS, but the code is sound.** Positive `R_Q` of
  squeezed size does appear (+0.153, +0.231). Section 8 shows it is the
  documented `1/T` response of `Q` operating on the real tiling, concentrated
  exactly on the triangles the mechanism predicts. **The pre-registration was
  wrong, not the estimator** -- it assumed `Q`'s power normalization cancels,
  which contradicts the project's own Stage-4 finding. Retire the prediction;
  do not chase a bug.
- **Monotone in `b`: FAILS at one seed.** At `k_short` = 0.295, `R_Q` runs
  0.547 (b=4) -> 0.700 (b=8) -> 0.106 (b=16). The b=4/b=8 inversion is 28% and
  within plausible seed scatter; unresolved.
- **A step at `x` ~ 1**: not assessed here.
- **Equilateral hurt less than squeezed**: not assessable, see the floors note
  below.
- **Position-dependent response smaller than `R_Q` where phases decorrelate:
  HOLDS where it can be checked.** At b=32 it is 0.003-0.031 at the two deep
  legs where `R_Q` is refused entirely.

### 7. The companion floors run (deneb 361) does NOT deliver the equilateral floor

Job 361 ran `v2_g3_floors.py --window flat`, 24 seeds, sections D + BC.

**Section D is a clean and genuinely new identity pass.** The flat window is a
scale-independent 5% transfer, and the estimator reproduces its closed form
exactly at every triangle INCLUDING the equilateral, which the gaussian window
could never reach (it had `T - 1` = 5.6e-9 there): `R_B` = 0.157625000000001
against an expected 1.05^3 = 0.157625 (agreement ~2e-16); `R_Q` =
-0.047619047619048 against the predicted 1/1.05 - 1 = -1/21 exactly, confirming
the documented 1/T response of `Q` under a uniform window; `rho` flat to
5.6e-16, confirming its window-invariance at the equilateral for the first
time; `W` floor 3.7e-16.

**Section BC's "MEASUREMENT" verdicts must NOT be quoted as resolving power.**
It reports `sigma_B(R_Q)` = 1.8e-16 to 3.1e-16 on all eight triangles with a
cancellation factor of 1.2e15 to 2.3e15, and marks all eight `resolvable:
true` against `bar/3` = 0.05. Those sigmas are floating-point noise. A
scale-flat multiplicative window makes `R_Q` EXACTLY seed-independent by
construction -- the realization cancels identically in the ratio -- so the
seed-to-seed scatter is machine epsilon and the cancellation factor is
approximately 1/eps.

This is the same defect class as the gaussian-window failure it was meant to
repair, in a different disguise: under `gauss` the control was the scatter of
an identically-zero quantity (1.3e-10, cancellation 2.5e9); under `flat` it is
the scatter of an exactly-constant one (2e-16, cancellation 1.8e15). The
`--window flat` flag DID fix the response (section D above), which is what it
was named for, but the response and the floor are different quantities and only
the first was repaired.

**The codebase already predicted this.** `section_E`'s docstring states that
BC's window-arm construction "is nearly noiseless by design ... the
resolvability verdict it produces is optimistic to the point of being wrong. A
reference that CANNOT exhibit the behaviour under test reads as agreement."
The real floor is section E, the dynamical null that perturbs the INITIAL
density and evolves both arms, and **361 did not run it**. The equilateral
control still has no measured floor at cdev.

### 8. The bug signature is not a bug, and `rho` is measuring the wrong thing

Resolved 2026-08-06 from the pilot card alone, no new compute. All three gate
statistics are algebraically determined by two measured quantities, so the
decomposition is exact rather than inferred:

```
T(k)   = <d_t d_m*> / <|d_m|^2>  =  r(k) * sqrt(P_t/P_m)   [CROSS transfer]
R_B    = B_t/B_m - 1                      T_prod = T(k1)T(k2)T(k3)
rho    = (1 + R_B)/T_prod - 1
R_Q    = (1 + R_B) * D - 1,  where  D := qdenom(P_m)/qdenom(P_t)
```

`D` is not stored but is recovered exactly as `(1+R_Q)/(1+R_B)`. The
`rho`-`R_B`-`T_prod` identity was checked on every triangle of every cell and
holds to < 1e-9, so the card is internally consistent.

**Finding A: the positive `R_Q` is the documented `1/T` response, not a
defect.** `Q` normalizes by each arm's OWN power, so `R_Q` carries the power
deficit through `D` with the opposite sign to the bispectrum deficit. Wherever
`D * (1 + R_B) > 1`, `R_Q` goes positive. `D` is a per-triangle constant,
essentially independent of `k_short` (b=16 triangle sq1: 5.99 / 5.71 / 5.90 /
5.92 across the four legs) and strongly dependent on `x = k_long / k_P`:

| x | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 |
|---|---|---|---|---|---|---|---|
| D (b=64) | 2.50-2.82 | 1.22-1.32 | 1.14-1.22 | 1.02-1.10 | 1.13-1.22 | 1.13-1.23 | 1.15-1.20 |

At `x <= 1` the long mode is at or below the padded tile's own fundamental, so
the tiled arm structurally cannot represent it and loses a factor 2.5 to 6 of
power there. That deficit lands in `Q`'s DENOMINATOR and inflates `Q_t`.
**Of the 19 positive `R_Q` values in the card, 17 sit at `x <= 1`**, and the
two that do not are +0.013 (x = 2.5) and +0.043 (x = 3.5) -- an order of
magnitude below the sizeable positives (+0.10 to +0.38) and consistent with
noise about zero. So every positive `R_Q` of SQUEEZED SIZE is at `x <= 1`,
which is the mechanism's own prediction and not a pattern a coding error would
produce.

The estimator is independently validated on this exact path: job 361's section
D reproduces the closed-form `R_Q` = 1/1.05 - 1 = -1/21 to ~2e-16 under a
scale-flat window. The `R_Q` code is analytically correct.

**So the PRE-REGISTRATION was wrong, not the code.** It reasoned that the tile
cannot feel the long mode's nonlinear response and concluded `R_Q < 0`. That
step is only valid if `Q`'s power normalization cancels between the arms, and
the project had ALREADY MEASURED that it does not (`R_Q` responds as `1/T`,
Stage 4 / `g3_ladder_record.md`). The prediction contradicted an established
result of this gate at the moment it was written. The physical intuition
survives on the RAW ratio: **`R_B` < 0 on 90 of 92 estimand triangles**,
spanning -0.87 to +0.07, which is the pre-registered "tile cannot feel the long mode"
statement measured on a quantity whose normalization does not fight it.

**Finding B, and this is the more consequential one: in the decorrelated regime
`rho` measures `1/r^2` and nothing else.** Because `T = r * sqrt(P_t/P_m)`, when
the tiled arm retains its power but loses phase coherence, `T -> r`, so
`T_prod -> r(k_long) * r(k_short)^2 ~ r_short^2` and
`rho ~ (1 + R_B)/r_short^2 - 1`. Measured against that prediction:

| arm | `k_short` | `r` | `1/r^2 - 1` | median `rho` |
|---|---|---|---|---|
| T=128 b=64 | 0.295 | 0.921 | 0.18 | 0.05 |
| T=128 b=64 | 0.589 | 0.762 | 0.72 | 0.66 |
| T=128 b=64 | 0.884 | 0.517 | 2.74 | 2.67 |
| T=128 b=64 | 1.178 | 0.276 | 12.13 | 10.93 |
| T=128 b=16 | 0.884 | 0.080 | 155.07 | 142.40 |
| T=128 b=32 | 0.884 | 0.113 | 76.69 | 83.58 |
| T=128 b=32 | 1.178 | -0.005 | (diverges) | 35980 |

Sixteen for sixteen, agreeing to 10-20% wherever `r` > 0 and diverging to 1e4
where `r` crosses zero. **`rho`'s "signal" is a restatement of the eligibility
map.** It carries essentially no independent information about the tiling error
in this regime, and its apparent verdict that the b=64 arm is wrong by a factor
of 2 to 4 is an artifact of dividing by a decorrelation-damped transfer. This
is the `1/P`-diverging-where-the-signal-is-damped failure mode, in a statistic
built specifically to be window-invariant.

**Finding C: the root cause is shared, and it is the transfer's definition.**
`T` is a CROSS spectrum, so it conflates two physically different things: the
tiled arm LOSING POWER (`sqrt(P_t/P_m)` < 1) and the tiled arm DECORRELATING
(`r` < 1). Every gate statistic inherits the confusion. `rho` divides by it and
diverges; `R_B` is multiplied by it and is biased low; `R_Q` is contaminated by
its power part through `D` and flips sign. The `r >= 0.5` conditioning cut does
not repair this, because at `r` = 0.5 the `1/r^2` inflation is already a factor
of 4.

**What this does NOT establish.** That an auto-spectrum ratio
`sqrt(P_t/P_m)` would separate the two cleanly is a hypothesis, not a
measurement -- it is the obvious candidate but it has not been tried, and
whether the resulting statistic still responds to a genuine seam error is
exactly the thing the brackets exist to test. One seed throughout. Whether the
gate statistic should change is a checkpoint decision and is NOT taken here.

### 9. The auto-transfer re-run (deneb 362, 2026-08-06)

Job 362 re-ran the pilot at `main @ abd4661`, same config and seed as 360,
5224 s. It exists to measure `rho_auto` on real evolved fields, which fixtures
and the smoke box cannot do.

**The refactor is exactly neutral.** `v2_g3_card_repro.py` compared 112 arrays
across the four arms on `R_Q`, `R_B`, `rho`, `W`, `T_prod`, `n_tri` and the
shell arrays: **worst relative deviation 0.000e+00** against a 1e-12 bound.
Bitwise, not merely within tolerance, a day apart on the same host. The four
`k(r=0.5)` anchors also reproduce (0.4517 / 0.4722 / 0.9000 / 0.5172), and
eligibility is unchanged at 6 of 16 cells. The 362 card therefore SUPERSEDES
360 at the canonical path -- identical numbers plus `rho_auto`, `A_prod` and
`shell_A` -- and every number in sections 1-8 above stands unaltered.

**`rho_auto` does not inherit the decorrelation.** Across the b=64 arm's
`k_short` scan, where `r` falls from 0.921 to 0.517:

| `k_short` | `r` | `rho` | `1/r^2 - 1` | `rho_auto` |
|---|---|---|---|---|
| 0.295 | 0.921 | 0.289 | 0.179 | 0.398 |
| 0.589 | 0.762 | 0.823 | 0.721 | 0.250 |
| 0.884 | 0.517 | 3.270 | 2.742 | 0.225 |

`rho` climbs by 11x tracking `1/r^2 - 1`; `rho_auto` is flat, and slightly
DECREASING. Section 8's diagnosis is confirmed on evolved fields rather than
on fixtures, and the replacement behaves as designed.

**`rho_auto` discriminates the brackets.** Computed from the stored `shell_T`
and `shell_r` via the identity `A = T/r`, so no extra run was needed. Max over
the three estimand triangles at `k_short` = 6 k_f:

| control | `R_Q` | `rho` | `rho_auto` | `R_B` |
|---|---|---|---|---|
| span_check (2LPT) | 0.4233 | 0.4880 | 0.4982 | 0.7310 |
| pivot | 0.8975 | 0.8620 | 0.9237 | 0.9612 |
| kill_control (b=0) | 1.4017 | 1.6831 | 1.1858 | 1.0245 |

The ratified criterion -- a maximally broken tiling must read WORSE than the
configuration under test -- passes for `rho_auto` (1.186 > 0.924), and it
passes PER TRIANGLE rather than only on the max: kill > pivot > span_check
holds on sq1, sq2 and sq3 individually. That is the form Stage 4 demanded and
the form `R_Q` could not express.

**What this does NOT establish, and it is the obvious next question.** The
brackets run at `k_short` = 6 k_f only (`k_shorts[0]`), which is exactly the
regime where `r` is 0.7-0.92 and `rho` is still healthy -- all four statistics
pass there. So this measures that `rho_auto` HAS discriminating power, not that
it RETAINS it at the deep legs where `rho` fails. Brackets at `k_short` = 18
k_f would settle it and need a new run (`--bracket-k-short 18`).

**Also unresolved: `rho_auto` puts every cdev arm far outside the 15% bar**
(0.22 to 0.92 at `k_short` = 6 k_f, i.e. 22% to 92%). If it became the gate
statistic the verdict would be a clear fail rather than the unreadable cell
grid of section 2. One seed, no error bar, and the bar was ratified against
`R_Q`, so whether it transfers to a different statistic is itself a checkpoint
question.

**Housekeeping hazard, pre-existing, and DO NOT "fix" it by raising `--mem`.**
Both runs peaked at ~58 GB host RSS (360: 57.83, 362: 57.97) against a
`--mem=54G` request. The first reading of this was that the request should go
up to ~60 G; that is wrong and would produce a doomed job. Deneb's Slurm
config is `RealMemory=56000` MB (`CfgTRES=cpu=64,mem=56000M`), so **56000M is
the largest satisfiable request** and anything above it pends forever. 54G =
55296M already sits just under that cap.

What is actually happening: the node has 61 GB physical against 56 GB declared
to Slurm, and there is no cgroup memory enforcement, so a job silently
overruns both its request and `RealMemory` and survives on the ~5 GB Slurm does
not know about. The correct posture is to leave `--mem=54G` and know the
profile has ~0 GB of headroom, not to request memory that cannot be granted.
Co-scheduling is already prevented by `-c 64` taking every core, which is the
placement mechanism the sbatch header documents.

### 10. The bracket scan, pre-registered before the cdev run (2026-08-06)

Section 9 left one question open: the brackets had only ever been read at
`k_shorts[0]`, the most-correlated leg, so they showed `rho_auto` HAS
discriminating power but not that it KEEPS it where `rho` fails.
`bracket_controls` now takes an optional `tri_sets` and reads all four
`k_short` off the SAME three evolved control fields, so the scan costs
estimator time only (+3 s on the smoke). The pre-existing single-set outputs
are bit-identical, verified by diffing a smoke card against a baseline taken
before the change: worst absolute deviation 0.000e+00 on every bracket
quantity, with `shell_r`'s centre list widening 7 -> 8 by design.

**Smoke preview** (T=32/b=8, tiny box, NOT a cdev statement), `kill/pivot`
ratio; > 1 means the maximally broken tiling reads worse, as it must:

| `k_short` (k_f) | pivot `r` | `R_Q` | `rho` | `rho_auto` | `R_B` |
|---|---|---|---|---|---|
| 4 | 0.597 | 0.947 NO | 3.555 | 1.050 | 1.056 |
| 6 | 0.404 | 0.904 NO | 9.244 | 1.027 | 1.049 |
| 8 | 0.066 | 0.945 NO | 406.4 | 1.047 | 1.053 |

**Predictions for cdev, fixed before the run:**

1. **`R_Q` fails to rank at every `k_short`.** On the smoke it already fails at
   the SHALLOWEST leg (0.947), which is stronger than expected -- the
   saturation is not confined to the decorrelated regime. If cdev reproduces
   this, `R_Q`'s problem is worse than "cannot be read where the arms
   decorrelate" and the ratified gate statistic is in question on its own terms.
2. **`rho` ranks, and the ordering is worthless.** Its `kill/pivot` grows
   3.6 -> 9.2 -> 406 as `r` falls, and its kill magnitude reaches 6.6e4 against
   a 0.15 bar. It gets the ORDER right for the wrong reason: the kill control
   decorrelates more, so its `1/r^2` inflation is larger, and brokenness
   correlates with decorrelation. A correct ranking does not rescue a statistic
   whose magnitude is five orders off the bar. **Do not read this row as
   `rho` winning.**
3. **`rho_auto` ranks with a modest, roughly constant margin** (1.03-1.05 on
   the smoke), tracking `R_B` closely -- which is expected, since `A -> 1` under
   pure decorrelation and `rho_auto -> R_B` by construction.

**The risk this run is designed to expose.** `rho_auto` is bounded, but bounded
is not the same as discriminating: as decorrelation completes, `B_t` becomes
independent of `B_m`, `R_B -> -1` for every arm, and the margin over 1.0 can
collapse. A 3-5% margin at one seed is NOT resolved. **If `kill/pivot` -> 1.0
at the deep legs, the honest conclusion is that `rho_auto` fixes the divergence
but does NOT extend the gate's reach, and the `r` conditioning cut is still
required.** That is a real possible outcome and it is not a failed run.

### 11. Why the tiling loses large-scale power (2026-08-06, resolved)

Section 9's anomaly -- the tiled arm at `A(k_f)` = 0.394 against pure 2LPT's
0.980 -- is explained, and it is structural rather than a defect.

**Mechanism, read off the integrator.** `cola_step_bullfrog` updates the
residual velocity as `u <- alpha*u + bcoef*g + c_k1*psi1 + c_k2*psi2`. COLA
works because at large scales the true force `g` very nearly cancels the
analytic frame terms, leaving the residual small there so the background
carries the long modes untouched. A tile's `g` comes from `force_global` on a
PERIODIC PADDED BOX, which supports no mode longer than itself, while `psi1`
and `psi2` are the GLOBAL LPT displacements and keep their long-wavelength
content. The cancellation therefore fails at large scales and the residual
grows a spurious piece that fights the background.

**Measured, one force evaluation per arm, no evolution**
(`scripts/v2_g3_tile_force_scale.py`). The prediction fixed before the run was
that the tile force loses amplitude specifically below the padded tile's own
fundamental `k_P` and tracks the monolithic force above it, with a flat loss
or a knee elsewhere refuting it. Smoke, `sqrt(P_tile/P_mono)` on the
Lagrangian grid:

| arm | `k_P` [k_f] | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|---|
| T=16 b=8 | 2.00 | 0.186 | 0.364 | 0.571 | 0.537 |
| T=16 b=16 | 1.33 | 0.468 | 0.882 | 0.871 | 0.847 |
| T=32 b=8 | 1.33 | 0.542 | 0.934 | 0.776 | 0.857 |

The knee moves with `k_P` and not with the buffer: the two arms sharing
`k_P` = 1.33 recover together at k=2, while the `k_P` = 2.00 arm is still at
0.364 there.

**Confirmed at the production box for free**, from `shell_A` already on the
tracked card, across four arms at three different `k_P`:

| arm | `k_P` [k_f] | below `k_P` | first shell above |
|---|---|---|---|
| T=128 b=16 | 3.20 | 0.441, 0.691, 0.702 | 0.920 |
| T=128 b=32 | 2.67 | 0.424, 0.736 | 0.903 |
| T=128 b=64 | 2.00 | 0.633 | 0.929 |
| T=64 b=32 | 4.00 | 0.394, 0.573, 0.676 | 0.839 |

**What this changes.** The controlling variable is the PADDED TILE SIZE
relative to the box, not the buffer as such, so no affordable buffer converges
the long modes and the literature's larger buffer would not either. The
missing ingredient is precisely a long-range force, which the two-level split
ratified in D-v2-10/D-v2-11 already supplies. A tiled arm drawing its
long-wavelength force from the global coarse mesh rather than from its own
periodization has no reason to show this deficit. **That is a mechanism-backed
hypothesis, not a measurement**, and it is the one experiment worth running
before A3 is closed permanently.

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

### Opened by the pilot (2026-08-06)

- **The gate is eligible but unreadable at cdev.** Six cells pass the `r >= 0.5`
  conditioning cut and all six fail the `rho`-agreement precondition. Section 8
  shows the precondition is failing mostly because **`rho` diverges as `1/r^2`**,
  so it is not an independent check on `R_Q` in this regime and the six
  refusals should not be read as six tiling failures. Whether the gate statistic
  changes is a CHECKPOINT CALL, deliberately not taken here.
- **RESOLVED 2026-08-06: the positive `R_Q` is not a bug** (section 8). It is
  the documented `1/T` response of `Q`, concentrated at `x <= 1` where the tiled
  arm cannot represent the long mode. The pre-registered `R_Q < 0` prediction is
  retired as unsound; the physical intuition behind it survives on `R_B`.
- **Candidate for the checkpoint, untested:** replace the cross transfer with an
  auto-spectrum ratio `sqrt(P_t/P_m)`, which does not carry `r` and so would not
  diverge on decorrelation. Whether such a statistic still RESPONDS to a genuine
  seam error is unmeasured and is exactly what the brackets exist to decide.
  Costs one readout on the existing card plus a bracket check, not a new run.
- **The equilateral control still has no floor at cdev.** Section BC cannot
  produce one under any window; it needs section E (the dynamical IC-perturbation
  null), which has not been run at cdev.
- **The `max_rho_deviation` criterion normalizes per triangle by `|R_Q|`** and
  therefore diverges at a zero crossing. The verdicts here survive an absolute
  reading, so no conclusion changes, but the criterion should be restated in
  absolute terms (or against the bar) before it gates anything.
- **The kill-control bracket reads False on a 7e-4 inversion at one shell**, at
  one seed. Unresolved, and cheap to settle.
