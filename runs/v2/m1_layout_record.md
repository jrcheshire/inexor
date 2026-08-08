# M-v2-1: the layout, and what measuring its slack turned up

**This is a measurement record, not a verdict.** It reports what two configs
show and what they do not license. D-v2-14 clause 3 is ratified; if anything
here moves it, that is JC's amendment to make.

Provenance: `scripts/v2_m1_migration.py`, cards
`runs/v2/m1_migration_{smoke,cdev8}.json`, laptop CPU, f64 throughout,
20 steps a = 0.1 -> 1.0 log-spaced, seed 0. `cgh64` NOT YET RUN.

## 0. What was being measured, and why it was owed

D-v2-14 clause 2 prices the T9 tier at 10.15 B/p all-in, of which **0.90 is
"slack and arena"**, and states plainly that this term is an ESTIMATE from a
hand argument about migration rates and that measuring it is an exit condition
of this milestone. It is 7.7 GB of C-gh's 87.2, against 1.33x headroom on the
~116 GB LPDDR cliff (D-v2-13), so the size of the term is not cosmetic.

The layout implements clause 3 as written: capacity is per-bucket, frozen at
build time, and the operation is eject-and-reinsert because "a full re-sort is
not affordable (a second 77 GB scatter target is over the ceiling)". Overflow
escalates slack -> arena -> loud refusal and may never clamp.

## 1. The trajectory is the ratified one

The probe drives `scripts/v2_g5_core.py` unmodified -- the coarse global long
force plus the tiled short force, gauss + TSC + matching, alpha = 1.0 -- per
D-v2-16 clause 7, which keeps that file the oracle. Both halves of the
composition were checked rather than assumed, at cdev8, seed 0:

- **ICs bitwise identical** to `make_config_leg`'s recipe, positions and
  velocities, `x_sha1 66d73be45428ba3c` both ways.
- **One force evaluation bitwise identical** to `v2_g5_two_level_force`'s
  `force_two_level` at its recorded pivot (alpha 1.0, gauss, TSC, matching):
  `max |mine - theirs| = 0.000e+00`. Checked with a dynamic-range guard --
  `|g| rms = 4.97e-1`, not the all-zero arrays that made an earlier V4 parity
  check pass vacuously.

Same starting state, same force, same shared `float_step_bullfrog`: the
composed trajectory is the ratified one, which is a stronger statement than a
P(k) comparison would have given.

One caveat that does not affect the conclusion: run-to-run nondeterminism on
macOS-arm64 CPU means the exact digits need not reproduce ACROSS processes
(umbrella `reference-jax-macos-cpu-nondeterminism`; the two force calls above
agreed within one process). Migrant fractions and occupancy counts are
insensitive at that level, and cgh64 runs on linux-aarch64 regardless.

## 2. The headline: the failure is not the size of the slack, it is the policy

Capacity frozen at IC time cannot track structure formation.

| | smoke (L=32) | cdev8 (L=64) |
|---|---|---|
| bucket | 2.0 Mpc/h | **1.0 Mpc/h = C-gh's** |
| spacing | 1.0 Mpc/h | **0.5 Mpc/h = C-gh's** |
| initial max occupancy | 30 | 74 |
| final max occupancy | 1129 (**37x**) | 5621 (**76x**) |
| final migrant fraction | 22% | 50% |
| **arena residency at the ratified 10% slack** | **47%** | **62%** |

At cdev8 the migrant fraction rises monotonically 12.9% -> 53% and the arena
fills to 61.5% of every particle in the box. **Slack is not the lever**: at a
100% target -- 9.00 B/p, ten times the ratified estimate -- 53% of particles are
still in the arena, because the problem is not Poisson migration noise but the
occupancy DISTRIBUTION transforming as halos form.

| slack target | slack B/p | peak arena | arena B/p | all-in B/p |
|---|---|---|---|---|
| 0.05 | 1.135 | 61.8% | 5.565 | 15.95 |
| **0.10 (ratified)** | **1.355** | **61.5%** | **5.538** | **16.15** |
| 0.25 | 2.594 | 59.8% | 5.384 | 17.23 |
| 0.50 | 4.725 | 57.2% | 5.148 | 19.13 |
| 1.00 | 9.000 | 53.0% | 4.767 | 23.02 |

Read against D-v2-14 clause 2's 10.15 B/p, every row is over the ~116 GB cliff
at C-gh. An arena holding 60% of the particles is not an overflow region; it is
a second layout, and particles in it are out of place, which is exactly the
contiguity the design exists to provide.

Two mechanisms, separable:

- **Granularity + spread.** A 10% target realizes 1.36 B/p, not 0.90, because a
  whole spare slot is the smallest unit a bucket can be given and occupancy is
  spread (3 to 14 around a mean of 8 at build). Predicting from the mean alone
  understates it. This is the smaller effect.
- **Structure growth.** Dense buckets grow 76x while their capacity is frozen.
  This is the effect that breaks the policy, and no slack fraction fixes it.

## 3. cdev8 is in C-gh's regime, which is what makes it worth reporting

cdev8's bucket is 1.0 Mpc/h and its spacing 0.5 Mpc/h -- **both exactly C-gh's**
(the config table holds the fine cell fixed and grows volume). So the occupancy
statistics are not a different regime, only a smaller sample of one. Going up in
volume adds rarer and denser peaks: max occupancy already grew 30 -> 74 at build
and 1129 -> 5621 at a=1 across a factor 8 in volume, which is why the number that
matters at C-gh is not this one.

**The uint16 exposure is now quantified and is not comfortable.** The per-bucket
index is uint16 because that is what makes it 0.25 B/p. cdev8's peak occupancy
5621 leaves **11.7x headroom** on 65535, at 1/4096 of C-gh's volume. Whether
C-gh's rarest peaks stay under is unmeasured and is a second thing the cgh64 run
must report.

## 4. What this does NOT license

- **No statement about C-gh.** Two configs, one seed each, one cosmology, and
  the largest is 1/4096 of C-gh's volume.
- **No accuracy claim.** Nothing here touches physics; the layout watches the
  trajectory and does not alter it.
- **No amendment.** Clause 3 is ratified. The options visible from here are a
  periodic rebuild (which needs the "second 77 GB scatter" objection re-priced,
  since a blocked or streamed re-sort may not need a full-size second buffer),
  capacity sized from the FINAL density rather than the initial (cheap -- 2LPT
  at a=1 is available at IC time), or promoting the arena to a first-class
  second tier. Choosing among them is not this record's business.
- **The slack ladder assumes build-time capacity.** Under any rebuild policy the
  ladder above does not apply and would have to be re-measured.

## 5. Defects in my own instruments

Three, all mine, none of which would have raised on their own.

1. **Bucket counts were read off `particle_to_slot`.** A particle in the arena
   has a slot past every bucket run, so `searchsorted` filed it under the last
   bucket or off the end -- silently dropping exactly the overflowing particles
   the ladder exists to count. It reported **0.000% arena demand at a 10% target
   while the live layout stood at 47% arena occupancy**, and the contradiction
   between two numbers that should have agreed is the only reason it surfaced.
   Counts now come from positions, with an assert that they conserve.
2. **Buffered output.** The first cdev8 attempt died after ~15 minutes leaving
   an empty log and **exit code 0** -- indistinguishable from a clean run that
   produced nothing, the failure already on record for Vista jobs 896055 and
   896092. Per-step output is now flushed.
3. **The arena insertion cost 23x the physics.** Measured at cdev8 step 0:
   force 3.97 s, migrate 91.00 s. `_to_arena` scanned the whole arena for a free
   slot once per particle. Fixed to one scan per step plus a scatter:
   **91.00 s -> 0.25 s, a factor of 364.** At cgh64 the unfixed version would
   have been ~100 minutes a step -- not a slow measurement but no measurement.

Worth keeping from (3): it was caught because the per-step line prints the force
and migrate timings side by side, so the ratio was legible at a glance. An
instrument that reports its own cost next to the cost of what it measures
catches this class for free; one that reports only its answer does not.

## 6. The in-place repack: clause 3's objection does not hold, but the budget is still tight

Clause 3 rejects periodic re-sorting because "a second 77 GB scatter target is
over the ceiling". That is true of a sort, and **this is not a sort**: `migrate`
already keeps every particle in the right bucket, bucket order is a fixed
spatial ordering, and what degrades is only the CAPACITY distribution.
Restoring it is a monotone rearrangement, doable in place in two passes
(compact ascending, expand descending), chunked so the temporary is O(chunk).

Measured at cdev8 under the REAL two-level force, 20 steps, slack 0.20:

| repack cadence | overflow at a=1 | trend |
|---|---|---|
| never (clause 3 as ratified) | 61.5% | climbing |
| every 2 steps | 32.8% | climbing |
| **every step** | **16.5%** | **plateaued** (16.1% at step 14, 16.5% at step 19) |

The qualitative change is the plateau: a bounded steady state instead of a
runaway. Cost is **58 ms against a 4.13 s force step, 1.4%**. Working memory is
**0.52 MB and flat** in clustering and in time -- set by the chunk size, not by
N, which is the number the "second 77 GB" objection should be read against.
Main allocation held at **1.258x N across the whole run**, no drift.

**The residual 16.5% is not a repack failure and no cadence fixes it.** With 50%
of particles changing bucket per step, a bucket of 8 with 20% slack (capacity
10) overflows on its third arrival, within a single step. Cadence controls the
runaway; per-bucket slack controls the single-step overflow. Two independent
knobs, and the second one costs memory directly.

**Where that leaves the budget.** Main 1.258x N plus an arena sized for the
16.5% peak is ~1.43x N of 9-byte slots, so **~12.8-13.1 B/p all-in against
D-v2-14 clause 2's 10.15** -- about 29% over. At C-gh that is ~112 GB against
the ~116 GB cliff: 1.03x under, where the ratified figure claimed 1.33x. So the
repack removes the failure mode and does NOT by itself restore the budget.

### 6a. The slack sweep, and where the floor actually comes from

Five settings, every-step repack, uniform 10% allocation margin, cdev8, real
force. All-in = (main + PEAK arena) x 9 + index, since the arena must be sized
for the worst step:

| slack | main xN | peak arena | all-in B/p | peak inside the run |
|---|---|---|---|---|
| **0.02** | 1.127 | 22.0% | **12.38** | yes, step 17 |
| 0.05 | 1.142 | 21.4% | 12.46 | yes |
| 0.10 | 1.177 | 20.3% | 12.67 | yes |
| 0.20 | 1.260 | 18.1% | 13.22 | yes |
| 0.35 | 1.419 | 15.1% | 14.38 | yes |

**No interior minimum.** Monotonic, cheapest at the bottom. **12.38 B/p against
the ratified 10.15, +22%** -- ~106 GB at C-gh against the ~116 GB cliff, 1.09x
margin where D-v2-14 clause 2 claimed 1.33x.

**The arena TURNS OVER rather than plateauing.** Every setting peaks at step 17
and ends ~1.5% below its peak (slack 0.02: 3.7 -> 22.0 -> 20.6%, per-step
increments +1.65, +1.20, +0.51, +0.37, -0.64, -0.82). So the allocation is
bounded by a measured peak, not an extrapolation, and a = 1 is where production
runs end anyway. My earlier "plateaued" was read off two steps and was
imprecise; a quarter-mean test then called it "still climbing" because the early
ramp dominates the average. Both were wrong about the shape; the series is a
rise and turnover.

**The 12.5% floor is granularity, and the `min_spare` rule that appeared to
cause it is INERT.** `np.ceil` of any positive value is already >= 1, so
`max(ceil(s*count), 1)` has always equalled `ceil(s*count)` -- verified
elementwise, and a run at `min_spare=0` reproduced `min_spare=1` to every digit
(main 1.127, arena 22.0%, 12.38 B/p). I had claimed that rule was the largest
remaining term and worth ~1.1 B/p. It is worth nothing. The real cause is that a
bucket cannot be given a FRACTION of a slot: any nonzero slack costs one whole
slot per occupied bucket, which at ~8 particles per bucket is 12.5% of payload.

**The lever that follows, untested:** allocate spare at BRICK granularity rather
than per bucket. A brick is ~512 buckets / ~4096 particles, so one shared spare
slot amortizes to 1/4096 instead of 1/8. A bucket needing room takes it from the
brick's pool by shifting its neighbours along -- a move of at most 4096 entries
inside a 37 KB block, the same local operation the repack already performs. That
would remove the granularity floor and make slack a genuine tunable, which is
where the remaining ~2 B/p would have to come from. Not built: it changes how
the layout allocates, and that is a design call.

## 7. What the cgh64 run must report

- the arena fraction and peak occupancy at C-gh's own bucket and spacing, in a
  volume 64x cdev8's
- whether peak occupancy stays under the uint16 index ceiling
- the migrant fraction per step, which sets the exchange the streaming path pays
- per-step wall for force, migrate and check separately, so the instrument's
  cost stays visible against the physics
