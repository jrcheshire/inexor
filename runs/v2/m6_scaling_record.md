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

## 5g. Job 465 + the arena A/B: migrate's ~70x is the ARENA INDEX CHURN, attributed in three steps

**Step 1 -- the exoneration (antares job 465, `v2_m6_migrate_depth.py`,
commit `00ba689`; every rung valid, spreads <= 0.1 s).** A five-point drift
ladder at fixed config separates migrant VOLUME (three reach-1 rungs) from
staged DEPTH (reach-2/3 rungs read against the volume fit), at cdev and
cgh64:

- cgh64/cdev at matched drift-fraction: **7.89-7.98x on every rung** --
  exactly linear in N, eject and insert separately.
- Depth excess: **1.00x at every reach, both configs** -- staged depth costs
  nothing beyond the migrants it carries. Volume slope 0.11 s/Mrow at both.
- The probe migrating **100.5M particles (75% of the box)** at cgh64 costs
  **30.5 s** where the engine's migrate at a few-percent volume costs 197.4
  s/step (5f). N, volume and depth are all EXONERATED; the pre-registered
  branch 1 fired: the cost lives in a condition the probe did not share.

**Step 2 -- the one-axis arena arm (laptop, valid stand-in: the control
reproduces antares 2.6-3.5 s at cdev).** The one structural difference found:
job 465's uniform states never overflow a brick, so `arena_used = 0` on
every rung, while the engine's clustered state runs with ~0.9M arena
residents at cdev. Rebuilt with `brick_slack=0.0` (arena forced occupied,
`--calls 2` so the measured call runs on a pre-populated arena, the engine's
steady condition): **total 2.6-3.5 -> 12.4-14.7 s at identical N, volume and
depth -- ~4.5x, in BOTH eject (1.5 -> 7.0 s) and insert (1.3 -> 7.2 s).**
Cards `m6_migrate_depth_arena_{ctl,on}.json`.

**Step 3 -- the attribution (cProfile, one arena-occupied cdev migrate,
13.8 s):**

    _build_arena_index   2,008 calls   7.01 s   51% of the migrate
    _to_arena            1,988 calls   2.99 s   (its own nonzero free-list scan)

The mechanism, verified in source: `_eject_slab` releases ONE brick's arena
rows and calls `_invalidate_arena_index()` (state.py:1021); the next brick's
`decode_brick` calls `arena_slots_of_brick`, which rebuilds the WHOLE
brick->arena index -- an O(n_arena) pass (state.py:726-741). With A
arena-holding bricks that is A x O(n_arena) per migrate: at the profiled
configuration 2,008 x 3.4M ~ 7e9 element-visits = the 7.01 s measured. At
the engine's cgh64 (n_arena 26.8M, A plausibly 5-15k) that is 1.4-4e11 --
**100-300 s, the size of 5f's missing ~170 s** -- and the term scales as
A x n_arena, both growing with N, which is the ~70x-for-8x shape. The scan
fixed in section 3 has a sibling, and it is the arena index.

**The fix this points at (NOT yet built):** the invalidation is a
sledgehammer where the release is exact -- ejecting brick b's arena rows
removes exactly key b from the index, an O(1) dict update; `_to_arena`'s
claim adds rows to exactly one key. The free-list `np.nonzero` scan wants a
maintained stack. Same class as the `_insert_slab` fix: targeted rewrite,
gated on an identity (post-migrate arena content elementwise equal, ids
included -- this container lost a particle once).

**NOT established:** A (arena-holding brick count) at the engine's cgh64 --
the 100-300 s is a bracket from a plausible range, not a measurement; the
engine-side confirmation is the phase table re-run after the fix.

**THE FIX IS IN (2026-08-15): surgical per-key index updates + a lazy
free-list.** Eject drops exactly key b instead of invalidating the cache;
`_to_arena` claims off a maintained ascending free-list, rebuilt at most once
per eject->claim transition; the repack paths keep the full invalidation.
Three gates, because the first one has a measured blind spot:

1. **Cache-purity A/B** (`test_the_arena_caches_are_pure_and_match_a_rebuild_
   under_migration`): fast path vs an oracle arm that rebuilds before EVERY
   index read, elementwise state equality over a 4-migrate chain, plus the
   maintained index/free-list against fresh rebuilds. Mutation-tested: a
   stale-release mutation FAILS it; **a wrong-claim-order mutation PASSES it**,
   because both arms share the claim policy -- a purity test cannot see a
   consistently-applied policy change. Known limit, hence gate 2.
2. **Pre-fix/post-fix bitwise chain** (worktree of the parent commit, same
   seeded slack-0 state, 4 migrates, 5,162 arena residents):
   off/w/occupancy/arena_bucket/vel_scale/ids all n_diff 0. The fix IS the
   old behavior, claim policy included.
3. Full suite 440/1 + test-det 16.

**Payoff at the laptop arm:** arena-occupied cdev migrate 12.4-14.7 s ->
**2.90-3.47 s**, within ~10% of the arena-EMPTY control (2.6-3.5 s) -- the
occupancy penalty is deleted, volume slope back to 0.13 s/Mrow. Card
`m6_migrate_depth_arena_fixed.json`. **Owed: the engine-side confirmation**,
a cgh64 phase re-run (prediction: migrate 197.4 -> ~30 s/step class, step
612 -> ~450 s).

## 5h. Job 467 -- the arena fix confirmed at the engine, and the C3 affinity arm falsified

**Leg 4, the confirmation (antares, commit `1ae14cf`, same knobs as 464;
instrument neutral; leg 7,018 s):**

    cgh64, s/step:          pre-fix (464)    post-fix (467)
    step                        612.32           457.45      1.34x
    migrate                     197.36            23.72      8.3x  (32.2% -> 5.2%)
    tile_long                   176.93           192.09      +8.6%
    coarse_paint                125.03           132.53      +6.0%
    tile_short                   75.39            74.51      -1.2%
    tile_decode                  16.90            17.30      +2.4%

**The pre-registered core landed: migrate fell to the predicted class (~30 s;
measured 23.7) and the step to the predicted ~450 s class.** The 5g
attribution is confirmed at the engine: the arena index churn WAS the ~170 s.
The cumulative cgh64 ladder now reads **2622.6 (pre-scan-fix) -> 612.3 (scan
fix, `10d1a2d`) -> 457.5 (arena fix) = 5.73x in two attributed fixes.**

**Neutrality clause: PARTIALLY met, recorded rather than absorbed.**
tile_short/decode/reduce are within noise, but tile_long +8.6% and
coarse_paint +6.0% exceed the within-job sigma (~2.9 s/step). The books
balance (migrate's -173.6 vs the step's -154.9; the difference IS those
drifts), nothing in the fix touches either phase, and cross-job scatter at
cgh64 has never been characterized (one run per version, different days) --
the likely reading is node-state scatter, but it stays UNEXPLAINED on this
record. A same-day A/B pair would characterize it if the neutrality claim
ever needs to be airtight.

**Legs 2-3, C3: per-worker affinity does NOT fix the pool scaling.** Pinned
walls at cdev8/64 tiles: 9.54/9.19/10.30/10.95 s at W=2/4/8/16 against
un-pinned 466's 9.05/9.57/10.93/11.50 -- within ~5%, efficiency still 7% at
W=16 (it did cut idle-worker RSS, 836 vs 1583 MB at W=16). Bitwise 0 at
every width in every arm, still. The diagnostic triple names the shape:
idle ~0 while per-tile busy inflates ~linearly with W (296/570/1256/2555 ms
at W=2/4/8/16 vs 195 serial) -- **aggregate throughput is FLAT at ~6.4
tiles/s regardless of W: a shared-resource ceiling, not a dispatch defect.**
Leading candidate: memory bandwidth (the laptop at ~500+ GB/s plateaus at
2.7x; antares caps at 1.4x). Confound noted: the serial baseline may recruit
XLA intra-op parallelism a 1-core-pinned worker cannot, inflating the
per-tile ratio -- but that cannot flatten AGGREGATE throughput.
**Consequence: antares is the WRONG VENUE for the C2 scaling verdict**
(bottleneck identity is hardware-specific); the verdict belongs on a gg node
(~500 GB/s/socket, first-touch NUMA control), which is also the target.
P=320 RSS re-read pinned: 5.6-5.8 GB/worker, unchanged -- the gg width
arithmetic is ~20 workers per 120 GB, not 35-45, if gg reproduces it.

**The post-fix phase mix at cgh64:** tile_long 42.0%, coarse_paint 29.0%,
tile_short 16.3%, migrate 5.2%, decode 3.8%. The tile phases sum to 63.6%;
**coarse_paint is now the largest single non-tile term and carries the
superlinear residual (5f), so Stage 2c is the next fix on the ladder.**

## 5i. Vista 913729 -- C2 on gg: the pool design PASSES on the target hardware

(After two ~10 s env casualties, 913639/913656: the scratch purge had eaten
the Vista env's stale-atime symlinks -- `bin/python` gone, `python3.14`
intact -- and the rebuild then hit a half-extracted `libprotobuf` in the
$HOME rattler cache. `pixi clean cache` + reinstall on idev, JC's hands per
the TACC rule. The hard-fail preamble caught both for ~zero SU.)

**The pre-registered bar -- efficiency >= 50% at W=8, evaluated HERE --
passes decisively.** Honest metric is tiles/s (the serial baseline runs
single-threaded and slightly slow on gg, so the naive efficiency prints
exceed 100%; and the script's own "redesign" verdict lines apply the 50% bar
at MAX W, which was never the criterion -- read the cards, not the verdict
strings):

    pinned, cdev8/64 tiles:   serial 4.4 tiles/s
      W=8    45 tiles/s  (10.4x)   busy/tile 176 ms   idle ~0
      W=16   68 tiles/s  (15.5x)   busy/tile 233 ms
      W=32   78 tiles/s  (17.9x)   busy/tile 380 ms
      W=64   85 tiles/s  (19.4x)   busy/tile 709 ms   idle 2.95 s

- **The ceiling is ~19-20x at this config**, emerging as busy-inflation
  beyond W=16 -- the same bandwidth signature as antares, but at 14x the
  throughput. The machine matters exactly as 5h said: antares capped at
  1.4x, the laptop at 2.7x, Grace at ~19x.
- **Pinning WINS on gg** (unlike antares): un-pinned reads 7.0x / 11.4x /
  14.7x at W=8/16/32 against pinned 10.4x / 15.5x / 17.9x -- 20-49%. Grace's
  two NUMA domains are real; affinity stays (C4 answered).
- **Bitwise n_diff = 0 at every width in every arm** -- the third
  architecture (after macOS-arm64 and x86) on which the disjoint-write
  premise has now held exactly.
- **P=320: 7.47x at W=8 over 8 tiles** (quantization-perfect), RSS 5.6
  GB/worker, matching antares -- the gg width arithmetic at production tile
  shape stands at ~20 workers in the ~120 GB budget.
- Caveats: 64 tiles quantizes the high-W rungs (W=64 = one tile per worker),
  so the ceiling number is approximate until a larger tile population is
  measured; and the serial baseline's own thread budget on gg (nproc read 1
  in the batch shell) makes cross-machine SERIAL comparisons unreliable --
  tiles/s within one machine is the only number quoted here.

**Consequence: W2's engine surgery is green-lit by the plan's own criterion.**
The pool skeleton (persistent spawn workers, shm state, parent-applied
writes, per-worker affinity) is the design as canaried; the coarse-paint
chunks are pool-eligible by the same associativity argument and should ride
the same executor.

## 5j. Vista 914085 -- the W2 exit measurement on gg: 3.92x end to end, and the wall is now `migrate`

**Job 914085, gg i614-021, 1:25:40, ~0.47 SU, commit `afabe49`. Five legs, four
rc=0 and one rc=2; the job's `exit 1` is the sbatch's any-leg rule firing on
that rc=2, NOT a crash.** `scripts/v2_m6_phase_time.py:313` returns 2 on exactly
one condition, `instrument_neutral is False`, and the W=32 leg's traced-vs-control
overhead was +4.724 s against its own 4.211 s material bound. All five cards were
written. The W=32 **phase table** is therefore not readable at the pre-registered
tolerance; its wall is (the control arm, 210.56 s, is the neutral number and
agrees with the traced arm to 2.2%).

**Pre-registration honesty note.** The plan said Amdahl the serial leg's phase
table BEFORE reading the pooled legs. The job log carried all five legs in one
tail and they were read together. The arithmetic below uses only the serial
table as input, but it was NOT read blind and is not claimed as a blind
prediction.

### The serial reference, measured on the same machine in the same job

cgh64, K=3, `s_per_step` **284.496** (traced 853.487 s, control 857.320 s,
overhead -3.832 against a 17.146 bound, neutral). This leg exists because
antares' 457.5 s/step does not transfer across architectures, and it is the only
baseline any ratio here is taken against.

    tile_short      225.882 s   26.5%      migrate        93.757 s   11.0%
    tile_long       214.358 s   25.1%      tile_decode    61.048 s    7.2%
    coarse_paint    190.809 s   22.4%      tile_reduce    27.075 s    3.2%
                                            lead_drift    27.031 s    3.2%
                                            repack        11.547 s    1.4%

Pool-eligible work (the four tile phases + `coarse_paint`) = 719.172 of 853.487 s
= **84.26%**. Serial residue = 134.315 s = **15.74%**, so the **Amdahl ceiling is
6.35x** and no worker count can beat it.

### What the pool actually delivered

| W | Amdahl | s/step | speedup | % of Amdahl | tile loop s/step | tile speedup |
|---|---|---|---|---|---|---|
| serial | 1.00x | 284.496 | 1.000x | -- | 176.121 | 1.00x |
| 8 | 3.806x | 81.516 | **3.490x** | 91.7% | 27.011 | 6.52x |
| 16 | 4.761x | 72.642 | **3.916x** | 82.3% | 20.020 | 8.80x |
| 32 | 5.443x | 71.760 | **3.965x** | 72.8% | 18.296 | 9.63x |

**W=32 buys 1.2% over W=16** (72.642 -> 71.760 s/step) for twice the workers, and
is the leg that broke its own neutrality bound. The operating point is W=8 or
W=16, and the gap between them is 12.2% of wall.

`coarse_paint` pooled BETTER than the tile loop at every width -- 63.603 s/step
serial -> 7.500 / 5.573 / 4.897, i.e. **8.48x / 11.41x / 12.99x** against the tile
loop's 6.52 / 8.80 / 9.63. Stage C's decision to put the coarse chunks on the same
executor is vindicated by its own number, and the phase that section 5h called
"the next fix on the ladder" is no longer the target.

### The mechanism: the workers do MORE work as W grows, and it is one phase

The pool triple makes this direct. Aggregate worker-seconds per step against the
serial cost of the identical work:

    per step        serial     W=8       W=16      W=32     inflation @32
    decode          20.349    22.156    23.426    25.890      1.27x
    short           75.294    92.152   122.027   189.247      2.51x
    long            71.453    90.340   161.909   295.089      4.13x
    quant            9.025     7.795     8.074     8.497      0.94x
    busy total     176.121   212.357   315.307   518.638      2.94x
    idle             --        3.854     6.287    34.225

**`tile_long` is the whole story: 4.13x the worker-seconds at W=32 for identical
physics, while `quant` gets slightly FASTER and `decode` is nearly flat.** The
long-range arm is the FFT-heavy one and the most bandwidth-hungry; this is the
same busy-inflation signature 5h measured on antares and 5i measured on the C2
canary, now confirmed inside the engine on the target machine. Pool efficiency
(serial work over W x tile wall) is **81.5% / 55.0% / 30.1%**.

So the shortfall against Amdahl is not dispatch overhead and not load imbalance
at W=8 or W=16 (idle is 3.9 and 6.3 s/step). It is a shared-resource ceiling
inside one phase. At W=32 imbalance does appear (idle 34.2 s/step, 6.2% of
worker-seconds) on top of it.

### The C2 canary's 10.4x does not survive contact with the engine

5i measured 10.4x at W=8 on the canary; the engine's tile loop gets **6.52x**.
The canary timed the tile force alone, while the engine's loop carries decode and
quantize with it and drives them from shm-adopted state. **Quote 6.52x, not 10.4x,
for the engine.** The canary's ~19x ceiling is likewise an upper bound on a
narrower quantity; the engine's own tile-loop curve is flattening by W=32 (9.63x)
and there is no measurement above it.

### `migrate` is now the wall, and pooling made it slightly worse

    per step       serial    W=8     W=16    W=32
    migrate        31.252   32.794  32.780  33.643    (+4.9% / +4.9% / +7.6%)
    share of step   11.0%    40.3%   45.3%   47.5%

The whole serial residue inflated ~5% under pooling (134.289 -> 140.462 / 140.535
/ 142.965 s over 3 steps), uniformly across `migrate`, `lead_drift` and `repack`.
Candidate causes are the shm rebind and first-touch NUMA placement of pages the
workers now also read; **UNATTRIBUTED, and it is a ~5% tax on 16% of the step, so
it does not change any decision here.**

What does change a decision: `migrate` was 5.2% of the step at 5h's antares
reading and is **45% of it at W=16 on gg**. It is host numpy, so it is untouched
by both the pool and the GPU lane. Every further wall fix points at it.

### What this projects to at C-gh, and what licenses the projection

Carrying cgh64 to C-gh is **64x in particles at the ratified K=40**. The tile term
carries by V4's finding that per-tile cost tracks `cap` and not tile count, with
`cap` fixed across the config table; `coarse_paint` carries with the coarse mesh.
**`migrate` has no such license** -- its sort is N log N and 5d's staging term is
N^(2/3) -- so the number below is a floor for that phase, not an estimate.

    gg, per realization at C-gh (64x, K=40):
      serial                   202.3 h     40.5x the 5 h bar
      pooled W=8                58.0 h     11.6x
      pooled W=16               51.7 h     10.3x
      pooled W=32               51.0 h     10.2x

      of the W=16 figure:  migrate 23.3 h   tile loop 14.2 h
                           lead_drift 6.7   coarse_paint 4.0   repack 2.8

**`migrate` alone is 4.7x the whole bar.** That is the readout's single most
consequential line: the tile loop could go to zero and the engine would still miss
5 h/realization by 7.4x.

### Owed out of this section

- The **W5 operating-point checkpoint with JC**: W=8 vs W=16, trim on/off, and
  the realization wall + SU re-derived from the numbers above.
- `migrate`'s own decomposition at cgh64 post-arena-fix. 5g attributed the
  ~170 s churn; what the residual 31 s/step is made of has never been broken down,
  and it is now the largest term in the engine.

**NOT established by this job:**

- **Any pool footprint at C-gh.** `rss_mb_max` is `VmHWM` of the single largest
  worker (`executor.py:56` -> `engine.py:1035` -> `phase_time.py:254`), and the
  shm-adopted state pages count in EVERY worker's VmHWM while existing once
  physically. **The 9024 / 8408 / 7799 MB figures therefore cannot be multiplied
  by W**, and the pool's incremental footprint is unmeasured. 5i's "~20 workers
  in the ~120 GB budget" also predates the state's fall to 164.6 GB on a 237 GB
  node, which leaves ~72 GB of headroom, not 120. **W may be memory-capped below
  16 at C-gh and this job could not see it** -- cgh64's state is 1/64 of C-gh's.
  This is a W5 input and needs its own measurement.
- The tile-loop ceiling above W=32.
- Whether the ~5% residue inflation persists at C-gh scale or is a cgh64 artifact.
- The W=32 phase table (instrument non-neutral; the wall stands).

## 5k. Antares 474 -- Stage 2c at cgh64: 2.66x on the phase, and the cdev number did not transfer

**Four legs, all rc=0, both phase-time arms instrument-neutral, commit
`afabe49`.** The A/B is the `EngineConfig.paint_subblock` knob with both arms in
ONE job, which is also what controls the cross-job drift 5h left unexplained.

    cgh64, s/step        2c OFF     2c ON     ratio
    step                 450.970   375.346    1.202x
    coarse_paint         130.343    48.937    2.663x
    tile_long            188.336   194.660    0.968x
    tile_short            73.590    73.030    1.008x
    migrate               23.723    23.700    1.001x
    tile_decode           17.480    17.610    0.993x
    chunks/step                0       512    (knob proven applied)

**The payoff is 2.66x on `coarse_paint` and 20.2% on the whole step.** The
laptop-at-cdev measurement recorded in owed item 9 was **~2%**, and it did not
transfer -- the phase's composition is genuinely different at the two
configurations, which is what 5f warned about from the other direction ("do not
quote 2c's payoff from the N^2 arithmetic"). The arithmetic overpredicted at
cdev and roughly landed at cgh64. **The general form: a wall payoff measured at
the small config is not a bound on the large one in either direction.**

**Cross-job reproduction, which is the other thing this job bought.** Against
job 467's post-arena-fix table, 474's `off` arm reads `tile_long` 194.66 vs
192.09 s/step (+1.3%), `coarse_paint` 130.34 vs 132.53 (-1.7%), `migrate` 23.72
vs 23.72 (0.0%), step 450.97 vs 457.45 (-1.4%). **5h's unexplained +8.6% /
+6.0% drift does not reappear**, so the likely reading there was node-state
scatter and the paired-arm design is what makes this one airtight.

**The antares cgh64 ladder now reads 2622.6 -> 612.3 -> 457.5 -> 375.3 s/step.**

**No double counting with 5j:** the gg cards carry `paint_subblock: true`, so
2c was already ON for every number in that section and the 51.7 h C-gh
projection already includes this win.

**Identity legs:** pooled-vs-serial n_diff 0 at cdev8 and cdev (`m6_w2_identity_
{cdev8,cdev}.json`), the fourth and fifth architectures-plus-configs on which the
executor's disjoint-write premise has held exactly.

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
8. ~~The migrate decomposition (from job 464, section 5f).~~ DONE, sections
   5g/5h: the arena index churn, attributed in three steps, fixed, and
   confirmed at the engine (197.4 -> 23.7 s/step).
9. **Stage 2c is BUILT and bitwise (2026-08-15), and its WALL payoff at cdev
   on the laptop is ~2%** -- 5.70 vs 5.81 s with the knob proven applied (64
   vs 0 sub-block chunks; `EngineConfig.paint_subblock`, the `pad_ladder` A/B
   pattern; gates: the streamed-vs-monolithic pin, a 4-case unit identity
   incl. periodic wrap + masked pads + the full-axis degenerate, a
   containment guard with a passing and a failing direction, suite 443/1).
   So the full-mesh-per-chunk term is NOT what dominates coarse_paint at
   cdev-on-M4 -- exactly the trap 5f warned about ("do not quote 2c's payoff
   from the N^2 arithmetic") -- and the phase's internal composition at
   cgh64/antares is UNMEASURED. The memory half is structural regardless
   (the per-chunk full-mesh transient, 4.3 + 8.6 GB at C-gh, is deleted).
   **Owed: the cgh64 wall A/B via the knob** (one phase-time leg per arm),
   which is also the coarse_paint decomposition's first one-axis arm.
   **DONE, antares 474 -- see section 5k. The cdev reading did NOT transfer:
   the cgh64 payoff is 2.66x on the phase and 20.2% on the step.** The
   sentence first written here, that 2c "reads as a memory result plus an
   attribution, not a wall fix", was written from the cdev number before 474's
   `off` arm finished and is WRONG at cgh64. Corrected rather than deleted,
   because the mistake is the transferable part: a wall payoff measured at
   cdev-on-M4 does not carry to cgh64.
10. **`migrate`'s decomposition at cgh64, post-arena-fix.** 5j puts it at 45%
   of a W=16 step and 4.7x the whole 5 h bar on its own at C-gh. 5g attributed
   and removed the ~170 s index churn; the residual ~31 s/step has never been
   broken down. This is now the largest term in the engine and the only one
   that both the pool and the GPU lane leave untouched.
11. **The pool's incremental memory footprint**, which 5j could not measure:
   per-worker `VmHWM` double-counts the shm state, and cgh64's state is 1/64 of
   C-gh's. W may be memory-capped below 16 at C-gh. A W5 input.
