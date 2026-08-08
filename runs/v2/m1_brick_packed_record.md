# Brick-packed layout: the budget comes back

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
- `repack` sorts by key when the arena is non-empty, which is O(N log N). At
  cdev8 that is invisible; at C-gh it wants the merge it deserves, since only
  the few arena residents are out of order.
- The 6.0% is dominated by `migrate` (0.23 s) rather than `repack` (0.027 s),
  and neither is optimized.
