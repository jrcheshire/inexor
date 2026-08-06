# Proposal: give the tiles a global long-range force

**Status:** proposal, nothing built. Written 2026-08-06 after D-v2-12's
mechanism finding. Supersedes nothing; D-v2-12 clause 1 stands for the
configuration it tested.

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
