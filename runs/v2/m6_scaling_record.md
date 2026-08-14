# M-v2-6: the migration's write-back was O(N x n_bricks^2), and the engine's wall was mostly that

Companion to `m6_peak_record.md`, which is about MEMORY. This one is about TIME,
and it opens with a measurement that broke its own pre-registration.

Jobs: **455** (antares, the sizing run that found it), **456** (deneb, the
one-axis attribution), **457** (deneb, the same probe after the fix), **459**
(antares, the end-to-end confirmation; section 5b). 458 was refused by its own
scratch guard and ran nothing -- see section 5b's note on why that mattered.

**Headline: the engine step at cgh64 went 2622.57 -> 608.67 s, 4.31x, because
77% of it was one scan.** The memory half of 459 missed its pre-registered band
AND is confounded; section 5b says so plainly rather than reporting the wall
alone.

Commits: `10d1a2d` (the fix), `e435538` (the probe), `fb3a5bf` (a fixture guard).

## 0. The reductions, stated once

- `s_per_step` is `wall_s / k_steps` over a whole engine step -- force, kick,
  fused drift, migrate, repack -- not the migration alone. This matters in
  section 5 and it is the distinction the attribution rests on.
- `insert_s` / `eject_s` are wall time inside `SlotState._insert_slab` and
  `_eject_slab`, summed over every call in one `drift_and_migrate`, median of 3
  repeats. Timing lives in the probe, never in the package.
- `nb` = bricks per side. `n_bricks` = nb^3. Bricks per x-slab = nb^2.

## 1. What job 455 measured, and how it broke its own bar

The job existed to size a charged Vista request, and pre-registered **100-800
s/step** at cgh64 with the clause that outside that band "the reasoning behind
the Vista request is wrong rather than imprecise and the sizing needs redoing,
not scaling."

| config | particles | nb | s/step | machine |
|---|---|---|---|---|
| cdev | 256^3 = 1.68e7 | 16 | 61 | antares |
| cgh64 | 512^3 = 1.34e8 | 32 | **2622.57** | antares |

**43x the wall for 8x the particles**, and 3.3x past the top of the band. Per
the pre-registration nothing was scaled from it. Peak host RSS was 14.447 GB,
which DID sit inside that job's 11-40 GB bracket.

NB job 455 ran with `--slack 0.20 --arena-frac 0.20`, deliberately generous so a
sizing run could not die on the D-007 arena refusal at a value never exercised
at this configuration. Its peak is therefore not comparable to any other peak on
record; its wall is what it measured.

## 2. The derivation, written before any of it was measured

`_insert_slab` selected each brick's rows inside the brick loop:

    for b in range(lo_b, hi_b):            # nb^2 bricks in this slab
        sel_k = keep["dest"] // p3 == b    # FULL SCAN of the slab's keepers
        sel_i = imm["dest"] // p3 == b     # FULL SCAN of the slab's immigrants

Every brick reads every row of its slab. A slab holds ~N/nb rows, contains nb^2
bricks, and there are nb slabs, so per step:

    nb x nb^2 x N/nb  =  N x nb^2

which is **N^(5/3)**, not N. Feeding the two rows of section 1 in -- including
the measured staging depth, 2 at cdev against 3 at cgh64 -- predicts **42x**
against the measured 43x.

**That agreement is why the next thing was a measurement and not a patch.** One
ratio matching one derivation has a coincidence budget, and this project's
record is five wrong causes proposed and three shipped before one was measured
first. The configuration ladder cannot settle it either: particles, bricks and
coarse cells all move together on it (measured 2026-08-13, 64.00x / 64.00x /
61.18x smoke -> cdev8, degenerate by construction).

## 3. Job 456 -- the one-axis attribution

`scripts/v2_m6_insert_scaling.py`, deneb, staging depth **pinned at 1 and
asserted per rung** (see section 4 for why that is the whole design).

**Arm A, the claim.** Particles fixed at 256^3, brick count doubling:

| nb | insert_s | step | eject_s |
|---|---|---|---|
| 8 | 3.793 | -- | 1.160 |
| 16 | 10.756 | 2.84x | 1.269 |
| 32 | 39.626 | 3.68x | 2.493 |

Fitting a constant plus a quadratic term to the first two rungs gives ~1.4 s of
brick-count-independent work plus 0.037 s per nb^2, and **that model was used to
predict 38.6 s at nb=32 BEFORE the rung ran.** Measured 39.626, within 2.6%. The
shape was called in advance rather than fitted after.

**Arm B, the control.** Brick count fixed at 16, particles varied: 1.445 s at
128^3 against 10.756 s at 256^3 = **7.44x for 8x the particles**, linear.

So cost rises with the SQUARE of the brick count and only LINEARLY with
particles. That is the signature of a per-brick scan over slab rows, and it is
not the signature of "bigger is slower".

**Arm B crashed** after its first rung on `n_part=192`: the T9 lattice needs
`n_part * 256 / bucket_cells` to be a power of two or the periodic wrap would
saturate (D-007), so 192 is not constructible and the layout refused correctly.
The defect was a FIXTURE I never evaluated, and the local smoke could not have
caught it because every value the smoke uses is legal. Same class as jobs 447
and 448, which died on print statements added after the previous run. The 7.44x
above is therefore recovered from two rungs of a failed leg, not from a clean
arm. `fb3a5bf` adds `_validate_rungs`, which constructs every rung's layout
before the first build and names all offending rungs at once.

## 4. The confound the design exists to kill

`brick_reach` is `ceil(|c_drift| * vel_scale * INT16_MAX / (box / nb))`, so at a
fixed drift the staging depth grows **linearly with nb** -- and deeper staging
means more immigrant rows for each brick to scan. An arm that let depth float
would have measured `nb^2 x depth` and could not have separated them.

So the drift is SOLVED per rung from that bound (`0.9 * extent / (s *
INT16_MAX)`) rather than tuned, depth is reported per rung, and a rung whose
depth is not 1 makes the run **void rather than noisy**. All rungs of 456 and
457 read depth 1.

A consequence worth stating: with depth pinned at 1 and the drift tiny, almost
nothing migrates, so `imm` is nearly empty and these numbers are dominated by
the KEEPER scan. Job 455 ran at depth 2-3, where the immigrant half is several
times larger again. **The measured saving is therefore a floor on the real one.**

## 5. Job 457 -- the fix, measured on the same machine with the same probe

`_group_by_brick` does one grouping pass and returns a permutation plus CSR
offsets; the loop slices. Order within a brick is preserved, which is what makes
it bitwise neutral rather than merely equivalent -- the destination velocity
scale is a max over the brick's rows and the encode that follows is
order-dependent through it.

| nb | insert before | insert after | speedup |
|---|---|---|---|
| 8 | 3.793 | 1.281 | 3.0x |
| 16 | 10.756 | 1.170 | 9.2x |
| 32 | 39.626 | **1.951** | **20.3x** |

Steps per doubling went **2.84x / 3.68x -> 0.91x / 1.67x**: the quadratic term
is gone.

**An unplanned control, and it is the strongest thing in this record.** `eject`
was not touched. It reads 1.161 / 1.275 / 2.515 against 456's 1.160 / 1.269 /
2.493 -- reproducing to within 1% at all three rungs. Machine, fixture and
conditions are therefore identical between the two jobs and the insert change is
the only variable.

## 5b. Job 459 -- the end-to-end confirmation, and one prediction missed

Identical to 455 in config, knobs, legs, machine and cores.

| | job 455 (before) | job 459 (after) | change |
|---|---|---|---|
| s/step | 2622.57 | **608.67** | **-2013.9 s, 4.31x** |
| peak host RSS | 14.447 GB | 11.600 GB | -2.847 GB |

**PREDICTION B (wall): passed, by a wide margin.** Pre-registered at >=250 s off
and falsified below 150. Measured 2014 s off. The 250 s floor was the KEEPER
scan alone scaled to this configuration's particle count; the realized saving is
~8x that, which is what the immigrant half predicts at depth 3, where seven
slabs of emigrants are staged instead of the probe's three.

**This ANSWERS section 6's first entry.** The scan was **77% of the whole engine
step** at cgh64 -- force, kick, drift, migrate and repack included. Applying the
same term to cdev (8x fewer particles, nb 16 vs 32, depth 2 vs 3 => 44.8x less
scan) puts its scan at ~45 s of its 61 s step, 74%. The two agree, so "the scan
explains the 43x" now rests on a measurement rather than on a matching ratio.

**PREDICTION A (memory): FAILED its own band.** Predicted ~10.1 GB, i.e. a 3-5
GB drop, from `kick_pending` priced at 4.295 GB here. Measured 2.847 GB. Below
the band, so it is recorded as a miss.

**And the comparison is CONFOUNDED, which is my design error.** Job 455 predates
BOTH the per-brick velocity scales and the grouping fix, so two interventions sit
between it and 459 and the memory delta cannot be attributed to either alone.
The wall is safe -- the grouping is measured directly and in isolation by
456/457, and the velocity change has no plausible wall effect at this size -- but
the memory number is not. This repository already carries the lesson in
`engine.py`'s own comment: *an instrument and an intervention in one commit
cannot be told apart afterwards.* I did it at the job level instead.

The likely reading, and it is UNMEASURED: a peak is a maximum over the step, and
job 446 established that the engine's peak is set in `tile_short`, not in the
kick. `pending` grew through the tile loop and was largest at its end, so
deleting it removes only however much had accumulated at the moment the peak was
actually set -- not its final 4.295 GB. That would explain a partial drop
without anything being wrong. Confirming it needs a run with only one of the two
changes, which does not exist.

## 6. What is NOT established

- ~~That this explains job 455's 43x.~~ **SETTLED by job 459: it does.** The
  scan was 77% of a cgh64 step, and the same term put at cdev's configuration
  gives 74% of its step. See section 5b.
- **What the 2.847 GB of memory belongs to.** Two interventions sit between 455
  and 459, so the drop cannot be attributed to either. Section 5b.
- **That the remaining 608.67 s/step is now linear in N.** It is 10x cdev's
  pre-fix 61 s for 8x the particles, but cdev's post-fix wall has never been
  measured and the two runs also differ in `arena_frac` (0.08 against 0.20),
  which changes the row count every per-step pass walks. A matched-knob cdev
  point is the cheap next measurement and nothing should be projected before it.
- **That the 43x itself was a clean comparison.** cdev's 61 s came from the trim
  work at different `arena_frac`, so the headline ratio that started this was
  knob-mismatched. It does not affect sections 3 and 5, whose arms hold every
  knob fixed, but the 43x should not be quoted as a controlled number.
- **Any production projection.** Three points after the fix, and the residual is
  not flat (0.91x then 1.67x), so at least two effects remain -- one of them
  almost certainly the per-brick Python loop, which runs nb^3 times per step.
  Two points cannot tell a line from a knee. Extending the ladder is cheap and
  has not been done.
- **The speedup at any configuration other than 256^3 / depth 1.**
- **Anything about eject's own growth**, which is 2.15x across the same span and
  untouched by this work. It makes one call per brick and n_bricks cubes, so
  per-call overhead is the obvious candidate -- unmeasured.

## 7. Owed

1. ~~Read out job 459.~~ DONE, section 5b. Wall passed; the memory prediction
   missed its band and its comparison is confounded.
2. **A matched-knob cdev point on the current code**, which is what says whether
   the step is now linear in N. Cheap, and everything about production sizing
   waits on it.
3. Extend the brick ladder past nb=32 before any production wall is quoted.
4. `eject`'s nb-growth, if it ever matters next to what remains.
5. The per-brick loop itself: nb^3 Python iterations per step is the shape the
   residual points at, and it is a different fix from this one.
6. A velocity-change-only run, if the memory attribution ever needs to be
   clean. Not owed for its own sake -- the term is gone either way and the
   planner prices it -- but the 2.847 GB stays unattributed until then.
