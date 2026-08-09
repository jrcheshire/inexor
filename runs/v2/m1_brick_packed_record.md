# Brick-packed layout: the budget comes back

> **RESOLVED AFTER THE FACT, 2026-08-08 (D-v2-20).** The open uint16-vs-arena
> question this record leaves below is CLOSED, and by neither of the two routes
> it lists: the index is **uint32, 0.50 B/p, all-in ~10.54, 1.28x under the
> cliff**. The cgh64 tail re-run this record calls for was NOT performed and is
> no longer owed -- the ceiling was removed rather than measured, because the
> only extrapolation available failed its own validation by 10x and three points
> were never going to bound a peak two rungs away. Every occupancy number below
> stands as measured; only the decision they were gathering evidence for has
> moved. The `uint16` figures in the tables are therefore historical, and
> reproducible via `--index-dtype uint16`.
>
> Two things surfaced while implementing it, both recorded in D-v2-20: only
> `build` guarded the narrowing, so `migrate` and `repack` would have wrapped
> SILENTLY; and the same class of ceiling sits in the int32 `key`, where it is
> refused rather than widened because `key` cannot be resident at C-gh in either
> width.

**Measurement record, not a verdict.** D-v2-14 clause 3 is ratified and nothing
here amends it. Branch `jc/v2-brick-packed`, cdev8, real two-level force, 20
steps a = 0.1 -> 1.0, seed 0. cgh64 NOT run.

## The result

| term | B/p |
|---|---|
| payload (T9) | 9.000 |
| bucket index (uint16) | 0.250 |
| brick boundaries | 0.002 |
| slack, 10% pooled per brick | 0.901 |
| arena, peak 0.57% of particles | 0.051 |
| **total** | **10.204** |

**Against D-v2-14 clause 2's 10.15, that is +0.5%.** At C-gh, 87.7 GB against
the ~116 GB cliff = **1.32x under**, which is the headroom the ratified figure
claimed and the per-bucket layout could not deliver.

For comparison, the per-bucket layout at its best measured setting is ~13.4 B/p
(12.38 as reported plus the 1.00 B/p `bucket_start` array, which the earlier
record did not count -- see below).

Cost: repack 27 ms/step, scratch 0.52 MB, total layout overhead **6.0% of the
force**. Peak arena 0.57% of particles against the per-bucket design's 22%.

## Why it works: three terms, not one

1. **Granularity.** Per-bucket, `ceil()` gives every occupied bucket at least
   one whole spare slot -- 12.5% of payload at ~8 particles per bucket, whatever
   the setting. Pooled over a brick's ~4096 particles, 10% means 10%: measured
   slots hold at exactly 1.100x N.
2. **The boundary array disappears.** `BrickLayout` stores an int64 slot
   boundary per bucket: 8.6 GB at C-gh, **1.00 B/p that the earlier record never
   counted**. Here bucket boundaries are a prefix sum of `occupancy` within a
   brick, so the index already paid for at 0.25 B/p does both jobs. 1.00 -> 0.002.
3. **Migration is 7x smaller at brick level** (4.4% vs 32%), because a brick is
   512 buckets and a particle must travel much further to leave one.

## What had to be combined, and a claim of mine that was wrong

I proposed brick pooling as an ALTERNATIVE to repacking. It is not; they fix
different things and both are required.

- Frozen capacity fails at EVERY granularity. A brick hosting a collapsing halo
  outgrew even 50% spare by step 6. My argument that bricks average over 512x
  the volume and therefore cannot concentrate was too optimistic: they are much
  more stable than buckets, and "more stable" is not "stable".
- With repack every step, a fixed brick fraction STILL overflows, and the
  overflow rises with the fraction -- 37 particles at 10%, 248 at 15%, 461 at
  20% -- because more spare lets the run reach heavier clustering before failing.
  No fixed fraction is the answer.
- But 461 of 2.1e6 is 0.02%: a rare-event problem, not a sizing one. A 1-2%
  arena absorbs it for 0.05 B/p, and peak use came in at 0.57%.

So: pooling buys the cheap slack, repack tracks the growth, the arena catches
the rare collapsing brick. Remove any one and it fails.

## Defects found, all mine, none of which would have raised

4. **repack scattered arena residents.** The main runs are in (brick, bucket)
   order so slot order IS key order -- but an arena particle sits past every
   brick run and arrives LAST regardless of its bucket, so placing by slot order
   put it in the wrong bucket's span. It only triggers once something overflows:
   steps 0-5 were clean and it fired at step 6. **Nothing would have raised** --
   positions in those buckets would have decoded against the wrong origin and
   the physics would have been quietly wrong. Caught only by the invariant that
   checks every particle against its DERIVED span, which exists because that
   derivation is the load-bearing trick of this design. Fixed by ordering by key
   when the arena is non-empty.

   Earlier three, on the per-bucket work: counts read off slots (dropped exactly
   the overflowing particles the ladder counted), buffered output (a 15-minute
   death read as exit 0 with an empty log), and an arena insertion costing 23x
   the physics. The pattern is consistent and worth stating: none of the four
   would have raised on its own.

## What this does NOT license

- **No statement about C-gh.** One config at 1/4096 of its volume, one seed.
- The uint16 per-bucket index has **11.0x headroom** at cdev8's peak cell
  population of 5943. Larger volumes hold rarer, denser peaks; whether C-gh
  stays under 65535 is unmeasured and is the second thing cgh64 must report.
- Layout overhead is 6.0% of the force ON CPU AT THIS SIZE. Whether it holds
  when the state streams rather than fits in memory is untested.
- **`migrate`'s sort is FIXED, 2026-08-08 (M-v2-2): 1.67x end to end, layout
  bitwise unchanged.** The cost was `np.argsort(kind="stable")`, which is a radix
  sort in numpy only for 1- and 2-byte integer types -- the 30-bit bucket ordinal
  gets timsort instead. Splitting the key into two uint16 digits puts both passes
  on the radix implementation: **the sort alone goes 5.0x** (89.6 -> 17.0 ms at
  2.1e6 rows; 3063 -> 216 ms at 16e6), and `migrate` end to end goes 1.67x,
  so the sort was ~40% of it rather than nearly all of it. The permutation is
  IDENTICAL to argsort's on random keys and on seven adversarial patterns, so no
  trajectory bit moves -- verified end to end by rebuilding the layout from
  scratch and comparing `slot_to_particle` elementwise.
  **The speedup is FLAT in N**: 1.68 / 1.67 / 1.68 / 1.66x at 3.3e4 / 2.6e5 /
  2.1e6 / 1.7e7 particles, a 512x range. I expected it to GROW as the O(M log M)
  term took over and it does not, which is what licenses carrying it to C-gh
  rather than re-measuring there. Measured on the M4 laptop; the transfer to a
  Grace CPU is untested.
  Also fixed: `np.isin(new_brick, affected)` -> a bool lookup table, 34.0 -> 9.8
  ms at 16e6 rows. **That correction is small, and it corrects this record**: the
  "argsort + `isin`" attribution above overstates `isin`, which was never more
  than a few percent of the step.
- `repack` sorts by key when the arena is non-empty, which is O(N log N). At
  cdev8 that is invisible; at C-gh it wants the merge it deserves, since only
  the few arena residents are out of order.
- The 6.0% is dominated by `migrate` (0.23 s) rather than `repack` (0.027 s),
  and neither is optimized.

---

# cgh64: the exit gate, as a matched CPU/GPU pair

Jobs **897358** (`gg`, CPU force, 2h23m) and **897377** (`gh`, GPU force,
1h43m), both COMPLETED 0:0, commit `32b22f2`, 512^3 particles in L=256 --
**64x the volume the layout was designed on, at C-gh's own cell and spacing**.
~4.1 node-hours, against my 1-2 estimate.

## The pair agrees EXACTLY, and that is a stronger check than it looks

| | gg (CPU force) | gh (GPU force) |
|---|---|---|
| all-in B/p | 10.292 | 10.292 |
| main | 1.100x N | 1.100x N |
| peak arena | 1.548% | 1.548% |
| peak bucket population | 13774 | 13774 |
| p99.9 occupancy | 706 | 706 |
| peak migrants | 69.6% | 69.6% |

**I predicted "close but not bitwise" and was wrong.** Quantization filters
roundoff completely: backend differences are ~1e-15 relative, the position
quantum is 3.9e-3 Mpc/h, so a particle would have to sit within ~1e-13 of a
bucket boundary to be assigned differently -- a ~1e-9 event across 1.3e8
particles. The layout statistics are therefore backend-invariant BY
CONSTRUCTION, which is a useful property in its own right.

## Memory: the design holds at 64x volume

**10.292 B/p, +1.4% on D-v2-14 clause 2's 10.15.** Across the 1x/8x/64x ladder
the figure moves 10.204 -> 10.234 -> 10.292, and `main` is **1.100x N at every
rung** -- the pooled spare is exactly the requested 10% at all three volumes,
which is the property per-bucket allocation could not deliver at any setting.

At C-gh that projects to ~88-90 GB against the ~116 GB cliff.

## Overhead: the finding this run was for, and my prediction was wrong

| | force s/step | layout s/step | overhead |
|---|---|---|---|
| gg (CPU force) | 146.75 | 19.01 | **13.0%** |
| gh (GPU force) | 21.48 | 18.36 | **85.5%** |

Force speedup CPU -> GPU: **6.83x**. The layout costs the SAME on both (19.01 vs
18.36 s) -- it is host-side numpy and does not care about the backend, which is
exactly why the CPU-relative figure is the wrong one to quote.

**On the backend production will use, the layout is 85.5% of the force cost.**
I pre-registered 25-40% and was wrong by ~2x. The reasoning was right in
direction and wrong in size: I expected the force to gain far more from the GPU
than 6.83x, forgetting that V4 already measured ~74% of per-tile cost as HOST
plumbing, so the device speedup is capped by work that never left the CPU.

This is an M-v2-2/M-v2-3 item, not a layout defect: `migrate` dominates
(~18 s of the ~20 s), and its cost is an `argsort` plus `isin` over 1.34e8
particles that a counting sort should largely remove.

## The uint16 index: the distribution is invariant, only the extreme grows

| config | p99 | p99.9 | peak |
|---|---|---|---|
| cdev8 (128^3) | 121 | 704 | 5943 |
| cdev (256^3) | 123 | 688 | 7581 |
| cgh64 (512^3) | 122 | 706 | 13774 |

**p99 and p99.9 are flat to ~2% across 64x volume.** The occupancy distribution
is not changing; the peak grows only because more samples are drawn from the
same distribution. My earlier "the growth ratio is accelerating, headroom is
1.4x" was fitting a trend to the single noisiest statistic in the dataset -- the
max is one bucket, and its two ratio estimates (1.276x, 1.817x) differ by 2.5x
for that reason.

Extrapolating the last rung gives ~45000 at C-gh against 65535, but that rests
on the weakest number available and should not be read to two figures. What is
solid: **99.9% of buckets are under ~710**, and realistically ONE bucket in the
box approaches the ceiling.

Two ways to handle it, JC's call, neither implemented **at the time this was
written -- SETTLED 2026-08-08 as the first, see the banner at the top**:
- **uint32 index everywhere** (**ADOPTED, D-v2-20**): +0.25 B/p, all-in ~10.54,
  ~91 GB at C-gh, 1.28x under the cliff. Simple, costs a quarter byte per
  particle forever.
- **Let the arena catch it** (**not taken**): a bucket over 65535 spills its
  excess exactly as a full brick does, capping the stored count. Given p99.9
  ~ 706 this is ~one bucket, so the cost rounds to zero and the machinery
  already exists. Rejected on inspection rather than measurement: it is not the
  free reuse it looks like, because `repack` pulls every arena resident back
  into a brick run and asserts as much, so a genuinely overflowing bucket needs
  permanent-resident semantics and a cap-aware mask in BOTH `migrate` and
  `repack` -- the two functions that produced four of this milestone's six
  instrument defects -- to save 0.25 B/p out of a 1.32x margin.

## What this does NOT establish

- **Streaming is untested and this run could not test it.** cgh64's whole T9
  state is ~1.2 GB and fits in memory. Streaming is real only at C-gh proper
  (77 GB), which is M-v2-6.
- **One seed, one cosmology, one schedule.**
- **The GPU backend was INFERRED, not demonstrated.** The log records
  `jax 0.10.2` and no device line; the 6.83x speedup is strong circumstantial
  evidence, since both nodes carry the same 72-core Grace CPU, but a benchmark
  should demonstrate its knob is live. Both sbatch scripts now print
  `jax.devices()` and the gh one ASSERTS a non-CPU backend.

## Defects in my own instruments, this run

5. **The gh job reported `CARDS WRITTEN: (none)` while the card was written
   fine.** `--out-suffix _cgh64` duplicates the config name the probe already
   inserts, so the file is `m1_migration_cgh64_cgh64_gh.json` and the listing
   pattern missed it. A false negative in a check whose whole job is to prove
   the run produced something. Pattern widened.
6. **The device was not recorded** (above).

Running total for the layout work: six defects of mine, and the consistent
property is that **none of them would have raised on its own.**
