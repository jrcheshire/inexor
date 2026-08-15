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

## 5c. Job 460 -- the matched-knob cdev point, and a confound I failed to control

Same code, machine, cores, legs, K and knobs as 459; only the configuration
moves. This is the first cdev/cgh64 pair on record that is comparable at all.

| | cdev (460) | cgh64 (459) | ratio |
|---|---|---|---|
| particles | 1.68e7 | 1.34e8 | 8.0x |
| s/step | **48.53** | 608.67 | **12.54x** |
| peak host RSS | 6.851 GB | 11.600 GB | 1.69x |
| mean staged slabs (2r+1) | 3.67 | 6.33 | 1.73x |
| `cap` (per-tile capacity) | 4.19-5.28e6 | 5.28-6.66e6 | ~1.1x |

12.54x for 8x the particles is an effective exponent of **1.22** -- down from
1.67 before the fix, and by this job's own pre-registration ("below ~50 s/step a
superlinear term survives") that is the superlinear branch.

**It should not be read that way, and the reason is a confound of mine.**
Staging depth was NOT held fixed between the two runs: cdev drifted at reach
[1,2,1] and cgh64 at [3,3,2]. That is 1.73x more staged data per step, and
8 x 1.73 = **13.8x**, which brackets the measured 12.54x with nothing left over
for an algorithmic term. Section 4 of this record explains at length why an
unpinned depth makes an arm unreadable, and pins it in the probe for exactly
that reason. I then built this pair and let it float.

**The depth difference is physical rather than a defect.** Brick extent is
IDENTICAL at both configurations -- 8 Mpc/h -- because the config table holds
the fine cell fixed and grows the box, so nb scales with L. Depth differs
because the larger box carries higher peak velocities, so a particle crosses
more bricks per step. That is the simulation being bigger.

**One clean result falls out**, and it is the first confirmation of it on
matched runs rather than by derivation: `cap` barely moves between the two
configurations, so the per-tile working set does not grow with the box. That is
why the peak rises only 1.69x for 8x the particles, and it is the planner's own
claim that cdev, cgh64 and C-gh share particles-per-tile and padded tile side.

## 5d. Job 463 -- where the wall goes, and the accuracy checkpoint discharged

`scripts/v2_m6_phase_time.py`, deneb, instrument neutral at both configurations
(overhead +0.211 s of a 1.184 s bound at cdev8, -0.407 of 4.310 at cdev).

| phase | cdev8 | cdev (anchor) |
|---|---|---|
| `tile_long` (the UNCOMPILED sub-block gather) | 50.4% | **48.8%** |
| `tile_short` (the jitted short-range force) | 33.5% | 22.9% |
| `coarse_paint` | 5.0% | 13.8% |
| `migrate` | 2.2% | 6.5% |
| `tile_decode` | 7.0% | 4.2% |
| `tile_reduce` | 0.9% | 1.6% |
| everything else | <1% each | <1% each |

**The migration is DONE as an optimization target**: 6.5% of the anchor step and
2.2% of the cheap one, after being 77% of it this morning.

**`tile_long` is now the engine**, at about half the step at both rungs. It is
the gather D-v2-21 refuses to compile on a 2.2e-16 bitwise break, and it costs
MORE than the jitted force it feeds.

**This falsifies an Amdahl argument made earlier the same day.** I argued the
accelerator's ceiling was ~1.14x, from a "host plumbing is ~88% of a tile"
figure that predated the migration fix. Measured, the device-eligible phases
(the JAX ones) are **85.6% at the anchor and 88.9% at cdev8**, so at the
recorded 6.8x device speedup the ceiling is **~3.7x**. The reasoning was sound
and the input was stale; the GPU question is not settled toward CPU and the
single-node accelerator-on/off test is the instrument. **That test has now run
and the answer is 1.28x, not ~3.7x -- section 5e. The 6.8x device speedup does
not generalize across the JAX phases; eligibility is not speedup.**

**A term the cheap configuration hid:** `coarse_paint` is 5.0% at cdev8 and
13.8% at the anchor -- 10x in absolute terms for 8x the particles, mildly
superlinear. That is the phase allocating a full `n_coarse^3` mesh per chunk,
re-verified live on 2026-08-14, and it is one of the few items that would help
wall and memory together.

**Parallelism ceiling, and it is lower at the anchor than the cheap rung
suggests.** The tile phases are 77.5% of the anchor step against 91.8% at
cdev8, so parallelising that loop alone caps at **4.4x** rather than 12x -- 8
large tiles instead of 64 small ones leaves the coarse paint and the migration
proportionally bigger. NB the tile-parallelism and accelerator levers are
largely THE SAME 86% exploited two ways, not multiplicative.

**The accuracy checkpoint owed by the per-brick velocity scales: PASSED, and it
IMPROVED.** `v2_m3_engine_gate.py --leg accum --config cdev --k 40`, at the
M-v2-3 gate's own knobs so the comparison is like-for-like:

    ratified (global scale)   7.125e-4   42.1x under D-v2-9's 3e-2 bar
    per-brick scales          4.120e-4   72.8x under the same bar

A 1.73x improvement, inside the "within ~2x either way" band pre-registered
before the run and in the predicted direction: finer per-brick scales (r_p50
0.40-0.80 of the global max) beat the extra rounding a migrant now takes at its
destination brick's scale. Nowhere near the ~3e-3 point at which the sbatch
said the storage decision should be re-opened rather than recorded as a pass.

**Incidental, and not a controlled comparison:** the anchor phase leg reads
43.02 s/step on deneb against job 460's 48.53 on antares at the same knobs but
K=5 against K=3, so deneb is roughly 1.13x faster per step. Smaller than the
machine-choice discussion assumed. A matched pair would be cheap and has not
been run.

## 5e. Vista 912457 -- the single-node accelerator A/B: the GPU buys 1.28x, and the pre-registration MISSED

`scripts/v2_m6_gpu_ab_vista.sbatch` driving `v2_m6_phase_time.py` at commit
`024523a`: one gh node, the same checkout and knobs in both arms (cdev, K=3,
repeats 2, slack 0.20, arena 0.20), differing ONLY in `JAX_PLATFORMS`. The
backend is asserted per arm before its leg runs (sbatch lines 97/108), so
neither arm can silently be the other. Pre-registered: **2.5-4.5x on wall, and
the host phases within noise across arms or the run is VOID.**

**The run is VALID and the band was missed by 2x at its floor.** All four legs
rc=0; `instrument_neutral: true` in both arms (overhead -0.083 s GPU / -0.124 s
CPU against material bounds of 1.46 / 1.87 s); the six host phases agree across
arms to 1.4% in aggregate (31.26 s GPU-arm vs 31.69 s CPU-arm over the K=3 run;
worst absolute difference `migrate` at 0.22 s, worst relative `membership` at
13% on a 0.11 s absolute). So the VOID clause did not fire, and the verdict
stands:

    CPU-only   31.14 s/step
    GPU on     24.39 s/step        ratio 1.277x   (pre-registered 2.5-4.5x)

Per phase, seconds over the whole K=3 run:

| phase | CPU arm | GPU arm | ratio |
|---|---|---|---|
| `tile_short` | 31.79 | 9.47 | 3.36x faster |
| `tile_long` | 16.60 | 6.52 | 2.55x faster |
| `coarse_paint` | 13.29 | 25.84 | **1.94x SLOWER** |
| `migrate` | 14.99 | 14.77 | 1.01 (host) |
| `tile_decode` | 7.67 | 7.49 | 1.02 (host) |
| `lead_drift` + `tile_reduce` + `repack` + `membership` | 9.03 | 8.99 | 1.00 (host) |

**Where the 3.7x went.** The tile forces accelerate roughly as the ceiling
assumed (3.36x and 2.55x against the 6.8x input), but `coarse_paint` -- JAX,
device-eligible, 14.2% of the CPU-arm step -- runs 1.94x slower on the GPU and
gives back 12.6 s of the 32.4 s the tile forces save. It is now **35.3% of the
GPU-arm step, the single largest phase there**. The 5d ceiling arithmetic
treated "device-eligible" as "accelerates at the measured device speedup", and
that 6.8x was measured on the tile force alone; carried to a phase with a
different structure it is not even the right sign.

**The mechanism of the `coarse_paint` regression is NOT measured.** The
candidate is structural -- the phase allocates a full `n_coarse^3` mesh per
chunk and streams host-resident chunks, so a GPU backend adds a transfer per
chunk to work that is allocation-bound -- but that is a reading of the code,
not an attribution. It does not need chasing on its own: the phase is already
the named wall+memory target, and any fix that stops materializing the full
mesh per chunk changes both arms.

**Charging verdict at the Stage 1b bar.** gg = 0.33 against gh = 1.0
SU/node-hr, so the GPU must clear ~3x on wall to win on charging. It measured
1.28x. Even granting the full D-v2-21 gather-compile lever (~1.7x on record,
not exercised in either arm here) the stack is ~2.2x, still under the bar --
and the CPU side keeps its own memory advantage (237 GB gg against the 116 GB
gh cliff, which the C-gh state alone now exceeds). **On this record the
production node is CPU-only unless something changes the coarse_paint story.**

**Incidental, not controlled:** the Vista Grace CPU arm (31.14 s/step) beats
deneb's job-463 anchor (43.02 s/step) by 1.38x at the same knobs but K=3
against K=5, and the profile INVERTS across machines -- deneb is
`tile_long`-dominated (48.8%) where Grace is `tile_short`-dominated (34.0%).
The machine-choice discussion should use matched-K pairs before quoting either
number.

**What 5e does NOT establish:** the ratio at cgh64 or C-gh (the phase mix moves
with configuration -- `coarse_paint` was 5.0% at cdev8 and 13.8% at cdev, so
its GPU penalty plausibly GROWS toward production, but that is a projection);
the `coarse_paint` regression's mechanism; anything about the tile-parallelism
lever, which is untouched by this A/B and is now the largest one standing.

## 5f. Job 464 -- the cgh64 phase table: NEITHER pre-registered branch fired, and `migrate` is the largest phase

`scripts/v2_m6_w0_cgh64_antares.sbatch` (antares, commit `9339e78`, K=3,
repeats 2, slack/arena 0.20 -- jobs 455/459's knobs). Valid: both legs rc=0,
`instrument_neutral: true`, and the control arm reproduces job 459
independently (612.32 vs 608.67 s/step, 0.6%), which is what licenses reading
the phases against that wall.

    migrate        197.36 s/step   32.2%     tile_decode   16.90   2.8%
    tile_long      176.93 s/step   28.9%     lead_drift    10.11   1.7%
    coarse_paint   125.03 s/step   20.4%     tile_reduce    6.95   1.1%
    tile_short      75.39 s/step   12.3%     repack         2.91   0.5%

**The pre-registration -- `coarse_paint` >= ~40% confirms the ~N^2 term,
<= ~20% refutes it -- landed at 20.4%, in the dead zone, and the derivation
OVERPREDICTED.** Derived: 64x growth cdev -> cgh64 (8x chunks x 8x mesh
cells). Measured: 125.03 vs 5.94 s/step = 21.0x raw across machines (~18.6x
after the ~1.13x deneb-vs-antares correction from 5c, which is incidental,
not controlled). So the phase IS superlinear (~2.3x excess over the 8x
particle ratio) but ~3.4x less than the full-mesh arithmetic says --
something inside the phase (the chunk's own scatter? the brick decode?) is a
large share the derivation did not model. **Consequence: Stage 2c stays
justified (bitwise, removes a 20% term and the per-chunk transient), but its
C-gh payoff must NOT be quoted from the N^2 arithmetic until W0b decomposes
the phase.**

**THE FINDING, unpredicted: `migrate` is the LARGEST phase at cgh64 --
32.2%, 197.36 s/step -- against 6.5% (2.80 s/step) at cdev (deneb 463).**
That is ~70x raw for 8x the particles (~62x machine-corrected). Section 5d's
"the migration is DONE as an optimization target" was a statement about cdev
and does not survive at scale. Known candidate contributors, none of which is
a measured decomposition: staging depth [3,3,2] vs [1,2,1] (~1.73x, physical
-- section 5c); the per-brick Python loop (nb^3 = 32,768 calls vs 4,096);
eject's own 2.15x growth (section 6, unchased). The config ladder cannot
attribute this (degenerate by construction); it needs one-axis arms, and
naming migrate's exponent is now the W0b work that gates any C-gh wall
number.

**The device-eligible share SHRINKS with scale: ~62% at cgh64** (tile_long
28.9 + tile_short 12.3 + coarse_paint 20.4) **against 85.6% at cdev** -- the
accelerator ceiling, already measured at 1.28x realized (5e), gets worse at
the configuration that matters. The parallel-tile-loop lever covers
tile_long + tile_short + tile_decode + tile_reduce = 45.1% at cgh64; 2c
targets 20.4%; the remaining 32.2% is migrate's own problem.

**NOT established:** the C-gh phase table (extrapolating an unmodeled ~70x
is exactly the trap the pre-fix section 5 numbers fell into); the
migrate-internal split (eject / insert / per-brick overhead / staging);
which term inside coarse_paint carries the excess. Smoke-config shares
(tile_long 74.1%) are dispatch overhead at tiny tiles and transfer to
nothing.

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
2. ~~A matched-knob cdev point.~~ DONE (job 460, section 5c).
2b. ~~Where the wall goes.~~ DONE (job 463, section 5d). ~~The accuracy
   checkpoint the velocity change owed.~~ DONE and PASSED, 4.120e-4.
3. Extend the brick ladder past nb=32 before any production wall is quoted.
4. `eject`'s nb-growth, if it ever matters next to what remains.
5. The per-brick loop itself: nb^3 Python iterations per step is the shape the
   residual points at, and it is a different fix from this one.
6. A velocity-change-only run, if the memory attribution ever needs to be
   clean. Not owed for its own sake -- the term is gone either way and the
   planner prices it -- but the 2.847 GB stays unattributed until then.
7. ~~The single-node accelerator A/B.~~ DONE (Vista 912457, section 5e): the
   GPU buys 1.28x against a pre-registered 2.5-4.5x, and `coarse_paint`
   regresses 1.94x on device. The `coarse_paint` mechanism is owed only if a
   GPU path is ever pursued; the phase is already the wall+memory target on
   CPU.
8. **The migrate decomposition (from job 464, section 5f).** One-axis arms
   for eject vs insert vs the per-brick Python overhead vs staging depth --
   the phase is 32.2% at cgh64, grew ~70x for 8x particles, and no C-gh wall
   number is quotable until its exponent has a mechanism. This is W0b of the
   wall plan and it now gates W1's C-gh payoff claim too (the coarse_paint
   derivation missed by 3.4x; both phases need the model corrected against
   the smoke/cdev/cgh64 three-point ladder).
