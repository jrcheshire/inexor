# Proposal: give the tiles a global long-range force

**Status: SHELVED 2026-08-06 (JC), after H0-H3 and H5 were measured.** Read
this line before the ladder below: rungs H0, H1, H2 (cdev8 leg), H3 and H5 ran
and are reported here; H2's production leg and H4 were not pursued and the
frozen-background knobs named in H5 are untested.

**Why shelved, not failed.** H0-H3 PASSED and the numbers are good. But the
arm they validate is the SYNCHRONOUS TWO-LEVEL scheme already ratified in
D-v2-10 -- the design study calls synchronous two-level and independent tiles
"different products" (Sec. f/g) -- so this work did not repair A3, it measured
A2 with a tiled fine mesh. Reviving A3 as a product means making the
frozen-background arm good enough, and that is a NEW product decision belonging
to the V4 architecture freeze with every number on the table, not a
continuation of G3. A3's own kill line already fired (D-v2-12).

**What to pick up from, if it is ever revived:** H5 measured a fully
independent-tile arm at `max|rho_auto|` = 0.107 against the 0.15 bar (one seed,
cdev8), and names the knob that should move it -- a larger `alpha`, which
confines the frozen long-range kernel to the scales 2LPT actually models, at
the cost of a longer-reach short kernel and a bigger buffer. Start there rather
than from scratch.

**The result worth carrying forward regardless** is the buffer sizing, which is
what G3 owed A2 as well as A3 (`docs/plan-plan-v2.md` V2 exit: "squeezed-B
error number -> A3 verdict + A2 buffer sizing"). See H3.

Written 2026-08-06 after D-v2-12's mechanism finding. Supersedes nothing;
D-v2-12 clause 1 stands for the configuration it tested.

## The claim to be tested

The tiled arm's large-scale power deficit is caused by each tile solving the
FULL gravitational kernel inside its own periodic padded box, which supports no
mode longer than that box. Supply the long-range part from a global coarse
solve and take only the short-range part from the tile, and the deficit should
not exist.

## Why this is a rewire and not new physics

**The failing configuration asks each tile for something a tile cannot give.**
`make_tile_force` calls `force_global(..., which="mono", ...)` on the padded
box: the monolithic kernel, i.e. the whole 1/k^2 range, evaluated periodically
on a box of side P. Everything longer than P is absent by construction, and
D-v2-12's measurement shows exactly that -- the amplitude knee tracks
`k_P = 2*pi/(P*cell)` across four arms at three different `k_P`.

**The machinery to fix it is already in the tree and already ratified.**

- `v2_g5_core.force_short_tiled` computes `(1-S(k)) * ik/k^2` on fine tiles with
  host accumulation, brick bucketing, capacity and ownership handled. It is the
  fine half of the two-level split ratified in D-v2-10/D-v2-11.
- `split_factor` guarantees the recombination identity STRUCTURALLY: `short` is
  literally `1.0 - S` sharing the same `S` array, so `long + short == 1` in
  floating point. For `family="compact"`, `K_long = mono - K_short` by
  construction. There is no tuning that can make the two halves fail to sum.
- The long-range half is a global solve on the COARSE mesh
  (`n_coarse` = 128 against `n_fine` = 512 at cdev), which is the whole point
  of the two-level scheme and is what D-v2-9's bar was set on.

So the proposed arm is **the already-ratified two-level force with its fine
level tiled**, rather than a new scheme. The arm that failed was a different
thing: the full kernel on tiles, with no global long-range term at all.

**The COLA frame needs no change.** The frame terms cancel the FULL force at
large scales. Once long + short sums to the full kernel again, the cancellation
that makes the scheme work is restored, and it is restored by an identity
rather than by a fit.

**Kernel family: `gauss` + TSC + matching, and NOT `compact`** -- the opposite
of what the docstrings alone suggest. `split_kernels` says the gaussian split
"tiles as ~2.0/P (kernel ringing)" while compact "tiles exactly for
buffer >= r_out", which reads as an argument for compact. The measured table in
`runs/v2/g5_kernel_findings.md` says otherwise, because exact tiling is bought
with a much worse coarse representation:

| family | coarse floor | tiling error | total at P=192 | asymptote |
|---|---|---|---|---|
| gauss (TSC+match) | 4.66e-3 | 2.0/P | ~1.1e-2 | 4.7e-3 |
| compact / hybrid | ~4.4e-2 | exact | ~4.4e-2 | ~4.4e-2 |

Gaussian wins ~4x at cdev-scale P and ~9x asymptotically. Note also that
matching is FAMILY-SPECIFIC: mandatory for gauss (1.35-1.6x) and actively
harmful for the windowed families (2.8-4x worse with CIC).

**Both families' errors are ~1e-2 in force, three orders below the 15% bar
this proposal is aimed at.** The failing arm does not have a percent-level
kernel-ringing problem; it has no long-range force at all. Either family fixes
that, so the family choice is a second-order optimisation already studied, not
a question this proposal needs to re-open.

## The ladder

Each rung has a prediction fixed in advance and a stated falsifier. **H1 is the
checkpoint: if it fails, the mechanism attribution in D-v2-12 is wrong and the
rest is not worth running.**

| rung | what | cost |
|---|---|---|
| H0 | wiring identity | seconds |
| H1 | **the decisive force test** | minutes |
| H2 | buffer criterion | minutes |
| H3 | density, one evolve | ~1 h, small box |
| H4 | the gate, production box | ~1.5 h deneb |

**H0 -- wiring identity, no evolution.** With `n_coarse = n_fine` and a single
tile covering the box (T = n_fine, b = 0), the hybrid force must reproduce the
monolithic force to roundoff. *Predict* max relative deviation < 1e-12.
*Falsifier:* anything larger means the recombination or the gather is
mis-wired, and no later rung means anything. Degenerate limit first, as always.

**H1 -- the decisive test, no evolution.** Re-run
`scripts/v2_g3_tile_force_scale.py` against the hybrid force at the arms
D-v2-12 measured. *Predict* the amplitude ratio is flat near 1 down to the box
fundamental, with NO knee at `k_P` -- the signature that produced 0.441 /
0.424 / 0.633 / 0.394 at k = 1 k_f should be gone. *Falsifier:* a knee still
sitting at `k_P`. This costs one force evaluation per arm and settles the
proposal's premise before any evolution is paid for.

**H2 -- what buffer is actually required.** Scan buffer at fixed `r_s` and
measure where the short-range periodization error becomes negligible.
*Predict* the requirement is set by the short kernel's own range
`R ~ beta * r_s`, not by the box -- which is where the cost win comes from,
since the failing arm needed a buffer scaling with the box and so had none.
*Falsifier:* a required buffer that grows with box size.

**Risk R7 is already answered, and the answer is the pessimistic one.**
`force_short_tiled`'s docstring warns that sharp-k truncation gives
oscillatory ~1/x kernel tails which would make the periodization error decay as
a POWER LAW in P rather than erfc. The measured gaussian tiling error is
**2.0/P** -- a power law, exactly as feared. So the buffer requirement for
`gauss` is NOT set by `r_s` alone and does grow with the padded box, and
`runs/v2/g5_kernel_findings.md` records this as the real tension: "error falls
only with bigger tiles, and big tiles are what M3 exists to avoid."

That tension is real but it is not fatal here, and the difference of scale is
the whole point: 2.0/P at P=192 is ~1e-2 in force, against a squeezed-B bar of
0.15 and against a long-mode power deficit of 60-85% in the arm that failed.
H2 should therefore be read as "how much buffer to stay well under the bar",
not as "does the buffer requirement vanish". One recorded false alarm is worth
knowing about: an apparent R7-like buffer plateau turned out to be trap 2 in
that record, a discarded last cell layer, not kernel ringing.

**H3 -- density.** One evolve at a small config. *Predict* the auto amplitude
ratio `sqrt(P_t/P_m)` is ~1 at low k rather than 0.39-0.63. *Falsifier:* force
fixed at H1 but density still depressed, which would point at the residual or
the reassembly rather than the force.

**H4 -- the gate.** Squeezed bispectrum against the monolithic arm at the
production box, read with `rho_auto` (D-v2-12 clause 3), against D-v2-7's 15%
bar. Note the comparison should be ENSEMBLE-based per D-v2-12 clause 4, not
realization-matched.

## Results: H0 and H1 PASS (2026-08-06)

Run with `scripts/v2_g3_tile_force_scale.py --force hybrid`, which composes the
long-range coarse solve and `force_short_tiled` exactly as
`v2_g5_two_level_force.force_two_level` does. `alpha` = 1.0 so
`r_s` = one coarse cell; gauss family; TSC + matching on the coarse solve.

**H0 -- wiring identity: PASS at 4.8e-16**, both forms, against a 1e-12 bound.
`long + short` reproduces `mono` on the fine mesh, and so does
`long + short_tiled` with one tile covering the box. The probe REFUSES to
continue if either fails.

**H1 -- the decisive test: PASS.** Amplitude ratio at cdev8, at the box
fundamental (k = 1 k_f), before and after:

| arm | P | `k_P` [k_f] | mono | hybrid |
|---|---|---|---|---|
| T=64 b=16 | 96 | 2.67 | 0.096 | 0.998 |
| T=64 b=32 | 128 | 2.00 | 0.332 | 0.998 |
| T=64 b=64 | 192 | 1.33 | 0.647 | 0.998 |
| T=32 b=32 | 96 | 2.67 | 0.087 | 0.998 |

The knee at `k_P` is gone. The hybrid is flat at 0.998 -> 0.987 across k = 1 to
10 k_f in every arm, and the correlation with the monolithic force is 1.000 to
three decimals at every shell -- against a mono arm whose correlation ran
0.654, 0.977, 0.970 and **-0.246** at the fundamental. The worst arm recovers
by a factor 11.5.

**The residual 0.2-1.3% is arm-independent and rises with k**, which is the
signature of the shared coarse long-range solve's own representation error, not
of tiling. It is three orders below the 15% bar.

**The arm knob was verified to apply**, since four arms agreeing to three
decimals is exactly what a silently-ignored parameter looks like. Measured
directly on `g_short`: the arms differ from the b=64 reference by 7.1e-3,
4.1e-3 and 5.8e-3 relative, monotone in buffer, i.e. real periodization error
that is simply too small to move the shell average. At smoke, where the buffers
are only 2-4 `r_s`, the arms do visibly differ (0.976 / 0.985 / 0.982).

**Early read on H2, not a substitute for running it.** At cdev8 the buffers are
4, 8 and 16 `r_s` and the 4-`r_s` arm already sits within 0.7% of the 16-`r_s`
one. That is consistent with the requirement being set by `r_s` rather than by
the box, which is the H2 prediction, but H2 still owes the actual scan and the
2.0/P law still applies.

## Results at the PRODUCTION box (deneb 363, 364, 2026-08-06)

**The gating reproduction check passed first.** The sbatch pre-registered that
the tiled arm must land on the tracked card's `shell_A` at the box fundamental
or nothing else in the run could be read. Predicted 0.441 and 0.424; measured
0.441 and 0.424. The harness is measuring what Stage 5 measured.

**H3 at cdev**, T=128, both buffers, k = 1..12 k_f:

| | k=1 | k=4 | k=8 | k=12 | r at k=12 | max\|rho_auto\| | wall |
|---|---|---|---|---|---|---|---|
| tiled, b=16 | 0.441 | 0.920 | 0.884 | 0.918 | 0.368 | 0.679 | 234 s |
| tiled, b=32 | 0.424 | 1.222 | 1.050 | 1.017 | 0.393 | 0.673 | 434 s |
| hybrid, b=16 | 1.000 | 0.999 | 0.996 | 0.994 | 1.000 | **0.0051** | 452 s |
| hybrid, b=32 | 1.000 | 0.999 | 0.996 | 0.994 | 1.000 | **0.0051** | 571 s |

All four predictions hold at the production box. The hybrid sits at 0.0051
against the 0.15 bar, a **29x margin**, where the failing arm is 4.5x OVER it.
Correlation is 1.000 at every shell out to 0.589 h/Mpc against the failing
arm's decay to 0.368.

**THE BUFFER SIZING ANSWER, which is what G3 owed A2.** The two hybrid arms are
identical to three decimals in amplitude and identical in `rho_auto`, so **4
`r_s` of buffer is already ample and 8 buys nothing**. The cheap arm runs 22.6
s/step against the monolithic 9.9 s/step recorded in `cost_of_memory.md`, i.e.
**2.3x monolithic compute** -- inside A3's 3.2x premise, and cheaper than the
3.23x the ratified two-level config was measured at.

**H2's box-independence prediction was HALF right, and the half that failed
does not matter.** Predicted: the same rel(b) row by row at cdev8 and cdev
within a few percent. Measured:

| b/`r_s` | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| cdev8 | 4.09e-2 | 4.55e-3 | 3.05e-3 | 1.49e-3 |
| cdev (8x volume) | 3.97e-2 | 5.45e-3 | 4.10e-3 | 2.75e-3 |
| ratio | 0.97 | 1.20 | 1.34 | 1.85 |

The STEEP part of the curve -- the part that sets how much buffer you need --
agrees to 3%, and the knee sits at 4 `r_s` in both boxes. So **the buffer
REQUIREMENT is box-independent**, which is the cost case. What differs is the
residual FLOOR once buffer is ample, 1.85x higher at 8x the volume. That is a
real box dependence and it is recorded as one, but it lives in a quantity
already ~50x under the bar, and H3 shows it does not propagate: the 4 and 8
`r_s` arms give the SAME `rho_auto` to four decimals. Extrapolating the trend
to 64x volume lands near 5e-3, still far under.

## H5: can tile independence be recovered? Partly, and it is not free

The lockstep hybrid solves the coarse long-range force from the true evolved
positions each step, which forces a per-step synchronisation across all tiles.
The frozen-background arm instead sources that field from `x_LPT(D_mid)` --
known for every particle at every step without evolving anything -- while still
gathering each particle's force at its TRUE position. If it agreed, all coarse
force fields could be precomputed up front and every tile could run its whole
schedule independently.

cdev8, T=64, b=16 fine (4 r_s), one seed, `alpha` = 1.0:

| | k=1 | k=4 | k=7 | k=10 | r at k=10 | max\|rho_auto\| |
|---|---|---|---|---|---|---|
| tiled (failing) | 0.432 | 0.880 | 0.798 | 0.791 | 0.128 | 0.625 |
| hybrid lockstep | 1.000 | 0.996 | 0.992 | 0.989 | 1.000 | 0.0023 |
| hybrid frozen bg | 0.995 | 0.959 | 0.920 | 0.896 | 0.989 | 0.107 |

**It works where it was supposed to and costs where it was not obvious.** The
large-scale deficit is gone (0.995 at the fundamental against the failing arm's
0.432) and the correlation stays above 0.989 at every shell. But small-scale
power drifts down to 0.896 by 10 k_f, and `rho_auto` lands at 0.107 against the
0.15 bar -- a 1.4x margin where lockstep has 65x. **That is not a pass**: one
seed, no error bar, and the ratified reading is an ensemble at the production
box.

**Why a LONG-range approximation shows up at SMALL scales.** Two compounding
reasons, the second of which is a knob.

1. A slightly wrong long-range force perturbs every trajectory, and 20 steps of
   nonlinear dynamics amplify that perturbation preferentially at small scales.
   This is the same amplification `v2_g3_floors.section_E` exists to measure.
2. At `alpha` = 1.0 the split scale is one coarse cell = 1.0 Mpc/h, so the
   "long" kernel still carries force down to k ~ 1 h/Mpc -- well inside the
   regime where 2LPT is a poor description. The frozen background is therefore
   being asked to supply force on scales it does not model well.

**Two knobs, both untested, both cheap.** Reason 2 predicts that a LARGER
`alpha` confines the long kernel to larger scales, where 2LPT is accurate, and
should improve the frozen arm -- at the cost of a longer-reach short kernel and
so a bigger buffer. That is a direct trade between tile independence and buffer
cost, and it is the interesting one. Separately, re-syncing the background
every N steps interpolates continuously between frozen (N = 20, fully
independent) and lockstep (N = 1). Neither is measured here.

**Architecturally**, the honest statement today is: full tile independence is
achievable and gets the large scales right, but at `alpha` = 1.0 it spends most
of the accuracy margin to do it. A synchronous pipeline keeps the margin.

## What could still sink it, stated up front

- **The 2.0/P tiling error is a power law, so buffer cost does not vanish.**
  See H2. It should sit ~1e-2 where the bar is 0.15, but "should" is doing work
  there and H2 is what checks it.
- **The G5 kernel numbers are from n_fine=128, force error only.** That record
  says so explicitly and lists an evolved snapshot and the evolution arm as
  still owed. The table above orders the families; it does not license a
  production error budget.
- **It re-introduces a global object.** The long-range solve is over the whole
  box, so the scheme is no longer purely tile-local. The coarse mesh itself is
  small (128^3 f64 = 16 MB at cdev), but the global particle set must be
  paintable, and **the memory saving that motivated tiling in the first place
  has to be re-priced, not assumed.** A version of this that costs the same
  memory as the monolithic arm answers nothing.
- **It inherits the two-level split's own floor.** The hybrid can be no more
  accurate than the ratified split, whose low-k error D-v2-11 characterises as
  a correctable transfer with a ~1e-3 single-box floor. That is far below the
  15% bar, so it should not bind, but it is a floor and not zero.
- **A3's compute budget still applies.** The route was premised at ~3.2x
  monolithic compute. The hybrid adds a global coarse solve per step on top of
  the tile solves. Cheap, but not free, and it must be counted.

## What this does NOT do

It does not reopen D-v2-12. That verdict is about the configuration measured,
and it stands whatever happens here. If H1 through H4 pass, the correct outcome
is a NEW arm with its own verdict, not a retraction -- the thing that failed
and the thing proposed are different schemes, and conflating them would make
the record unreadable.
