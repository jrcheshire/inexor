# M-v2-6 Stage 0b: the engine's peak is set in the tile short-range force, and Stage 0's method could not have found it

**Result: the peak is ATTRIBUTED, and its growth with K is a WARM-UP that
SATURATES** (section 3, corrected from job 450; section 3b measures K=40 and
K=60 and closes it -- the earlier "linear, unsaturating, +4 GB at K=40" reading
is retracted, and **at the ratified K=40 the anchor lands at 7.727 GB, flat
against 7.888 at K=15**). At cdev (256^3, K=5) the run peak is set in
`tile_short`, the per-tile short-range force, whose own increment is **3.432 GB
= 26 sigma** of the run-to-run scatter of the peak itself. `kick_pending`, which
`python -m inexor.plan` names as THE binding term at C-gh (274.9 GB), is
**537 MB here and does not bind**: the two phases where it would appear increment
0.000 and 0.140 GB. The decomposition is complete against an independent control
(0.9 sigma at cdev, 0.12 at cdev8).

Branch `jc/m-v2-6-capacity`. Measurements: **antares job 446** at
`db1d13d`, four legs, ~3 h 40 m, zero SU. Cards `runs/v2/m6_peak_trace_*.json`
(gitignored, also on `deneb:~/src/inexor/runs/v2/`). Instrument
`scripts/v2_m6_peak_trace.py`; the phase hook is `engine.step`/`engine.run`'s
`phase=` argument. Stage 0's instrument is `scripts/v2_m6_engine_peak.py` and
job 445.

**All four legs of 446 returned rc=1 and every failure was the probe, not the
engine** (two defects, section 6). Every number below was recomputed offline
from the per-visit series the cards persist, so no re-run was needed; the cards'
own `aggregate` and `verdict` blocks predate the fixes and **must not be
quoted**. A corrected-probe re-run is owed for a card whose verdict can be read
directly.

## 0. The reductions, stated once

| quantity | reduction |
|---|---|
| run peak | max over the boundary readings in a run, then **median** over repeats |
| phase increment | **max** over that phase's visits in a run, then median over repeats |
| sigma | sd of the corrected run peak over repeats, same leg |
| step ladder | max over the readings inside a step, split at each `coarse_paint`; median over repeats |
| slope | OLS through the median ladder, with the per-repeat spread quoted beside it |

The phase increment is a MAX and not a mean on purpose: a peak is set by one
allocation moment, not by an average one. `tile_short` is visited 8 times a step
at cdev and its per-visit increments run 1.6 GB with 3.4 GB outliers, so a mean
would report a number no moment of the run ever reached.

The control arm runs `phase=None` and so has no boundaries; its peak is its own
`ru_maxrss`, which nothing reset. That is what makes it an independent
measurement of the same quantity rather than a second copy of the trace arm.

## 1. Where the peak is (cdev, 256^3, K=5, 5 repeats; sigma = 131 MB)

| phase | own increment | sigma | visits/run |
|---|---|---|---|
| **`tile_short`** | **3.432 GB** | **26.2** | 40 |
| `tile_long` | 1.492 GB | 11.4 | 40 |
| `membership` | 1.318 GB | 10.1 | 5 |
| `tile_decode` | 1.295 GB | 9.9 | 40 |
| `coarse_paint` | 0.399 GB | 3.0 | 5 |
| `coarse_solve` | 0.189 GB | 1.4 | 5 |
| `tile_reduce` | 0.158 GB | 1.2 | 40 |
| `reconcile` | 0.140 GB | 1.1 | 5 |
| `lead_drift` | 0.093 GB | 0.7 | 1 |
| `migrate` | 0.006 GB | 0.0 | 5 |
| `tile_loop_end` | 0.000 GB | 0.0 | 5 |
| `repack` | -0.001 GB | -0.0 | 5 |

The ordering is stable across configs and across K: at cdev8 the top four are
the same four (`tile_short` 0.287, `coarse_paint` 0.224, `tile_long` 0.162,
`membership` 0.082 GB), and the cdev K=15 leg reproduces cdev K=5 to within a
percent on every term above 0.1 GB (`tile_short` 3.417, `tile_long` 1.510,
`membership` 1.318, `tile_decode` 1.300).

**`kick_pending` does not bind where we can measure it.** Its derived 32 B/p is
537 MB at cdev, a sixth of `tile_short`'s increment, and `reconcile` (which
holds it) increments 0.140 GB while `tile_loop_end` increments 0.000. This is
not a straight contradiction of `inexor.plan`: `pending` is O(N) while the tile
transients scale with `cap`, and C-gh moves both. It does mean the plan's
binding-term claim rests on arithmetic that has now been contradicted at the
only scale anyone has measured, and it needs re-pricing rather than restating.

**`mesh_bytes` under-counts this phase by 1.20 GB.** Its two fine-mesh terms sum
to 2.235 GB (`tile_kernels` 0.791 + `tile_workspace` 1.443) against
`tile_short`'s measured 3.432. At cdev8 the same comparison is 0.144 modelled
against 0.287 measured, short by 0.143 GB, so the model is short by about a
factor of 1.5 at both rungs rather than by a fixed offset. The sub-terms of the
tile force are named nowhere in the package. This is the gap Stage 0's gate 1
was groping for and could not resolve.

## 2. Run peaks, and what the allocator is holding

| config | trace | control | trim |
|---|---|---|---|
| cdev8 K=5 | 2.010 GB (sd 0.093) | 1.997 (0.112) | 1.366 (0.038) |
| cdev K=5 | 7.461 GB (sd 0.131) | 7.576 (0.125) | 6.585 (0.030) |
| cdev K=15 | 8.180 GB (sd 0.355, n=3) | -- | -- |

- **Gate A (instrument neutrality):** trace minus control is -115 MB at cdev
  (-0.9 sigma) and +13 MB at cdev8 (+0.12 sigma). The hook does not move the
  peak it measures.
- **Gate B (the decomposition is complete):** the top phase reaches the peak to
  within the same 115 MB / 0.9 sigma, measured against the CONTROL arm rather
  than against the trace arm's own run peak (section 6, defect 2).
- **The control independently reproduces job 445's four cdev measurements**
  (7.471 to 7.575 GB against 7.576 median here), which is the cross-check that
  the two probes measure the same quantity by two different routes.
- **glibc retention is 12% at cdev and 32% at cdev8** (trace vs trim). It
  SHRINKS with config size, so one config's fraction must not be carried to
  another. The `trim` arm is an instrument and never an operating point: it
  bounds what is reclaimable, it does not propose reclaiming it.

## 3. The K-growth is a WARM-UP, not a leak, and the peak is the FLOOR rising

**Superseded reading, corrected 2026-08-12 from job 450.** This section
originally reported +103 MB/step "linear and unsaturating" and extrapolated
+4 GB at K=40. Both halves of that are wrong, and the data to see it was already
in job 446: nobody split the ladder.

**(a) The climb is in the resident FLOOR, and the within-step transient
SHRINKS.** Job 450 persists each step's starting RSS beside its peak:

| config, arm | step-start RSS | slope | within-step transient | slope |
|---|---|---|---|---|
| cdev, ladder off | 0.556 -> 2.747 GB | +108 +- 8 MB/step | 6.094 -> 5.384 GB | **-12 +- 5** |
| cdev, ladder on | 0.563 -> 2.649 GB | +108 +- 10 | 6.068 -> 5.185 GB | **-27 +- 5** |
| cdev8, ladder off | 0.337 -> 2.465 GB | +109 +- 6 | 1.185 -> 0.225 GB | **-27 +- 1** |
| cdev8, ladder on | 0.337 -> 1.875 GB | +64 +- 2 | 1.159 -> 0.150 GB | **-26 +- 1** |

No phase's own increment grows enough to matter: the largest trend at cdev is
`tile_long` at +14 to +17 MB/step against a 108 MB/step floor rise, and most
phase trends are NEGATIVE. So the peak does not climb because any phase's
working set grows. It climbs because the floor underneath it does.

**(b) Most of that floor is glibc's, not the engine's.** Job 446's trim arm cuts
the floor rise by 3.5-4x at K=5 (cdev 354 -> 97 MB/step, cdev8 306 -> 103), which
is the same allocator retention section 2 measures as a static 12-32%, seen in
its accumulating form.

**(c) At the anchor config the climb SATURATES, and that is why the
extrapolation was wrong.** Splitting the cdev K=15 ladder in half:

| config, arm | steps 1-8 | steps 8-15 | decay |
|---|---|---|---|
| cdev, ladder on | +146 MB/step | **+28** | 118 +- 12 (81%, 10 sigma) |
| cdev, ladder off | +155 | **+59** | 96 +- 14 (62%) |
| cdev8, ladder on | +19 | +54 | **-35 +- 9 (steepens)** |
| cdev8, ladder off | +64 | +89 | **-25 +- 9 (steepens)** |

The cdev median ladder makes it plain: 6.64 -> 7.63 GB over the first eight
steps, then 7.63 -> 7.86 over the next seven, with the last five steps moving
-15, +11, +68, -8, +53, -9 MB against a 131 MB run-to-run sigma. A single OLS
slope through that curve is dominated by the early rise and says nothing about
where it lands. **`+103 MB/step` was a straight line fitted to a knee.**

**cdev8 does the opposite and steepens**, so this is not one mechanism at both
rungs, and neither ladder is long enough to say where either lands. That
disagreement is the argument for MEASURING K=40 rather than fitting anything
through K=15.

**What this changes.** The capacity question M-v2-6 exists to answer is whether a
complete mock fits on one node at D-v2-14's ratified K=40. Nothing on record
measures a run longer than 15 steps. The "+4 GB at K=40" that fell out of this
section was an extrapolation through a knee at one config and through a
steepening curve at the other, and it should not be quoted in either direction
until K=40 has been run.

## 3a. The original reading, kept for the mechanism

The cdev K=15 median ladder, 15 steps:

```
6.64 6.83 6.92 7.27 7.29 7.40 7.43 7.45 7.53 7.63 7.87 7.94 8.03 8.12 8.18 GB
```

**+103 MB/step by OLS through the median ladder** (per-repeat slopes 77 / 144 /
103, mean 108), consistent with job 445's independent 0.157 GB/step fit. The
"linear and unsaturating, so about +4 GB at K=40" that followed is RETRACTED:
section 3 shows the same ladder decays by 81% between its first and second
halves, and an OLS slope cannot see that.

**How much survives `malloc_trim`, at both configs:**

| config | trace slope | trim slope | difference | trim retains |
|---|---|---|---|---|
| cdev K=5 | 168 +- 14 MB/step | 99 +- 5 | 70 +- 15 (4.7 sigma) | 59% |
| cdev8 K=5 | 124 +- 17 MB/step | 71 +- 5 | 53 +- 17 (3.1 sigma) | 57% |

(per-repeat slopes, mean +- sem over 5 repeats; a first-to-last reduction that
assumes no linearity gives the same split, 184/115 and 121/69 MB/step.)

**So the majority of the growth is live memory and a reproducible ~40% is
allocator retention.** Both configs agree on the fraction. This CORRECTS the
first reading of this leg, which quoted the trim arm at +111 MB/step against the
trace arm's +110 and concluded the growth survived trimming entirely; under the
reductions in section 0 that comparison mixed a K=15 trace slope with a K=5 trim
slope. The conclusion that matters is unchanged -- a retained executable family
is live memory, and trimming cannot reach it -- but the expected signal of any
fix is the live ~60%, not the whole slope.

Note the K=5 legs give a slope from five points, so their per-repeat scatter is
large (136 to 209 MB/step at cdev). The K=15 leg is the trustworthy slope; the
K=5 legs are where the trace/trim contrast can be read, because only they have a
trim arm.

## 3b. K=40 and K=60, MEASURED (antares 452) -- the peak does NOT grow with K, and both rungs are ONE mechanism

**Job 452 at `6d68bfa`, five legs all rc=0, ~4.9 h, zero SU. Cards
`runs/v2/m6_peak_klong_*.json`.** Provenance matches 450 and 446 (antares, CPU
backend, 28 cores, jax 0.10.2), so the cross-job comparisons below are on one
machine and one backend. Reductions are section 0's, applied unchanged; slopes
are SPLIT, never one OLS through the range.

**(a) The number the capacity question rests on: cdev at K=40 is 7.727 GB**
(sd 0.282, n=3), against job 450's matched cdev K=15 of 7.888 GB -- flat to
slightly LOWER at 2.7x the steps. Section 3's retracted extrapolation implied
~11.4 GB. Within 452 alone the plateau needs no cross-job comparison: the last
ten steps of the K=40 ladder span 0.088 GB against a 282 MB run-to-run sigma.

| config | K | trace | control | trim |
|---|---|---|---|---|
| cdev8 | 40 | 2.147 GB (sd 0.044) | 2.036 (0.069) | 1.276 (0.021) |
| cdev8 | 60 | 2.254 GB (sd 0.183) | -- | -- |
| **cdev** | **40** | **7.727 GB (sd 0.282)** | -- | -- |

**(b) The floor's rise decelerates at EVERY config, so section 3's "cdev8 does
the opposite" is SUPERSEDED.** Floor slope by thirds of each run:

| config, K | first third | second third | last third | floor at end |
|---|---|---|---|---|
| cdev, K=40 | +79.0 | +8.5 | +4.4 MB/step | 2.542 GB |
| cdev8, K=40 | +37.5 | +29.7 | +9.0 | 1.943 GB |
| cdev8, K=60 | +28.0 | +22.4 | +3.9 | 1.957 GB (flat: 1.967 -> 1.957) |

cdev turns over by ~step 13 and cdev8 by ~step 40-45, and cdev8's K=60 floor
ENDS where its K=40 floor ends. So the two rungs are the same saturating
warm-up at different timescales, and section 3's "not one mechanism at both
rungs" was an artifact of a 15-step window catching cdev8 mid-rise while cdev
had already turned over. **A window too short to reach saturation can make one
curve look like two mechanisms** -- the sequel to that section's own lesson
about fitting a line to a knee, and the reason K=60 was worth its leg.

**(c) Both gates PASS at K=40, the first evaluation beyond K=5.** On the cdev8
K=40 leg (the only one with a control arm): gate A reads **-1.6 sigma**, the
trace arm sitting 111 MB ABOVE the control, which is the safe direction for an
instrument; gate B passes; `tile_short` still sets the peak. Instrument
neutrality at the ratified cadence is now measured rather than assumed.

**(d) The unattributed jump class is ATTRIBUTED, on the first job carrying the
per-step fields.** At cdev8 K=40 the largest single step-to-step move in the
peak is **+169 MB at step 22 = 3.8 sigma**, and `cap_per_step` changes at
exactly step 22 (330,281 -> 416,128) -- the one remaining rung crossing in the
run. So the `cap` ladder's residual cost is one bounded discrete event per
crossing, not a diffuse climb, and job 450's +136 MB step-11 jump is of this
class. The trim arm's largest move (+67 MB at step 20) lands on the one
`coarse_pad` crossing, the same way.

**(e) `malloc_trim` is priced, but NOT the version anyone proposed.** The trim
arm cuts the cdev8 K=40 peak 2.147 -> 1.276 GB (**-41%**) and the floor slope
+21.0 -> +4.2 MB/step (5x), at 14.96 against 12.98 s/step (**+15% wall**). Two
things must travel with those numbers:

- **The arm trims at EVERY phase boundary, which is 263 calls per step at cdev8**
  (10,521 series entries over 40 steps), not once per step. The proposed
  operating point -- a `malloc_trim` at the step boundary -- is a different
  intervention by a factor of 263 in call count, and neither its recovery nor
  its wall cost is measured. The 41%/+15% pair does not price it.
- **Retention shrinks with config size** (12% at cdev K=5, 32% at cdev8 K=5,
  41% at cdev8 K=40), so 41% must not be carried to C-gh, and **cdev trim at
  K=40 was not run** -- the anchor's retention at the ratified cadence is
  unmeasured, and that is the number a promote-to-default decision rests on.

## 4. The leading cause, measured and now fixed

`coarse_delta_streamed` sized its chunk buffer from the current occupancy
(`engine.py`, the `pad` term), so it keyed a **new XLA shape every step**.
Executables and their buffers are cached for the life of the process, so an
unbounded shape family is an unbounded leak -- the same defect Stage 0 fixed for
`cap` and left in place here.

Measured at the smoke config, on the laptop, which is valid because a shape
COUNT is exact arithmetic where a peak RSS is not:

| | unquantized | on the ladder |
|---|---|---|
| distinct chunk shapes over 10 steps | **10** | **1** |
| distinct chunk shapes over 15 steps | **14** | **1** |
| distinct `cap` shapes over 10 steps | 10 | 1 (since Stage 0) |

Fixed in `e2aebfe` by putting the chunk buffer on the same `capacity_shape`
ladder with the same `cap_rungs` knob, carried monotonically across steps.
Padding costs 24.7% of rows, inside the derived 2^(1/3) - 1 = 26.0% bound.
Bitwise neutral by the masking argument the `cap` padding already stands on, and
gated as an identity on the mesh rather than argued.

**This is a candidate cause, not a measured fix.** Attributing the slope to
shape churn is the hypothesis; the A/B against the measured +103 MB/step is the
one antares job still owed. The instrument and the intervention are in separate
commits deliberately -- an instrument and an intervention in one commit cannot
be told apart afterwards.

## 5. Why Stage 0 could not have found any of this

Stage 0 differenced MAXIMA between arms. **A maximum carries no timestamp**, so
differencing two of them assumes both were set at the same moment by the same
phase, and nothing checked it.

Job 445 contains five independent measurements of one identical leg (cdev, K=5,
f64 coarse): the `step` leg 7.503 GB, the K-ladder's K=5 rung 7.280, and repeats
at 7.538 / 7.575 / 7.471. That is **sigma = 115 MB over all five, 45 MB over the
four tightest**, and nobody had computed it. Against that scatter:

- gate 1's predicted f64-f32 signal is **67.6 MB = 0.6 sigma**, and its +-25%
  tolerance is +-16.9 MB, a quarter of the noise;
- the isolating arm removes 178.7 MB of modelled coarse mesh (1.6 sigma) and
  read **132 MB HIGHER** than the leg it was isolating from.

So the 2.034 mesh-model ratio on record is one draw of a noisy difference, not a
finding, and "the mesh model does not describe this machine" is NOT established
by it. (Section 1 shows the model IS short, by 1.20 GB on the tile force phase
-- a real effect that this method could not have separated from its own noise.)
Compounding it, half of gate 1's predicted signal (33.8 of 67.6 MB) is terms
`mesh_bytes` itself labels TRANSIENT while the analysis summed every term as
resident.

**Do not re-run Stage 0's differencing arms**, and do not quote its numbers.

**Correction to the earlier record**: Stage 0's cdev anchor was relayed as "four
gates fail on their own terms". Only **gate 1** returned false. **Gate 2 PASSED**
(`gate2_sees_on_bpp_term: true`, a one-sided floor at 0.8 x 32 B/p that the
254.5 B/p residual clears 8x) and was vacuous at that margin. The repack
transient (2.2 vs 12.6) and the state gap (13.74 vs 12.66) are
REPORTED-NOT-GATED by the script's own pre-registration. Gate 3 needs a second
config.

## 6. Two probe defects, both mine, both caught by my own gates

1. **`/proc/self/clear_refs` resets `mm->hiwater_rss`, which BOTH `VmHWM` and
   getrusage's `ru_maxrss` report.** A traced arm's end-of-run `ru_maxrss` is
   therefore the peak since the LAST boundary, not the run. I had written a
   comment asserting the two were independent. Effect: the cdev8 trace arm read
   1.824 GB against 2.010 actual, i.e. 0.19 GB BELOW the untraced control --
   which looks exactly like the instrument suppressing the peak, the thing gate A
   exists to catch, and gate A passed it anyway on a large sigma. **Gate B caught
   it by failing in the physically impossible direction**: a phase's high-water
   exceeding the process's. Fixed by accumulating the run peak inside the tracer
   as a max over boundary readings, with `maxrss_raw` and `hiwater_was_reset`
   kept on the card.
2. **Fixing that made gate B vacuous.** The run peak became the max over phase
   peaks BY CONSTRUCTION, so "does the top phase reach the run peak" compared a
   number with itself and would have returned zero for any engine and any defect.
   Now measured against the CONTROL arm, with a guard test that hands the trace
   arm a peak exactly equal to its top phase and requires failure anyway.

Also corrected in-session: `phase_growth` originally trended each phase's
ABSOLUTE peak, which rises for every phase alike as the process ratchets up. It
reported +1233 to +1308 MB for four unrelated phases -- one climb restated
twelve times. It now reads each phase's own increment, with the climb carried
once by the step ladder.

## 7. What is NOT established

- **Anything at production scale.** Every number here is cdev or cdev8. C-gh
  moves `cap`, K, tile count and N together, and the two term families
  (`pending` O(N) against the tile transients scaling with `cap`) move
  differently under it.
- **That the shape churn causes the K-growth.** Measured: the churn exists (10
  shapes over 10 steps) and the growth exists (+103 MB/step). The link is
  untested; that is the owed A/B.
- **The memory cost per retained executable family**, which is what would turn
  the shape count into a predicted GB/step.
- **Any peak on macOS.** Darwin reads about 3x low and one laptop point read
  6.484 and 9.855 GB minutes apart. Peak comparisons must live on one glibc
  machine, and the trace probe refuses non-Linux rather than falling back.
- **The sub-terms of the tile force**, which is what the 1.20 GB shortfall is
  made of.
- **What a STEP-boundary `malloc_trim` recovers or costs.** Section 3b(e): the
  trim arm calls it 263x per step, so it bounds what is reclaimable by the most
  aggressive possible schedule and prices nothing cheaper.
- **The anchor's retention at the ratified cadence.** cdev trim exists at K=5
  only (12%); cdev8's grew from 32% to 41% between K=5 and K=40, so the anchor's
  cannot be inferred from either.

## 8. Owed

1. **DONE (job 450), and the hypothesis mostly failed.** The `pad` A/B: the
   knob applied cleanly (15 chunk shapes -> 3, `cap` unchanged at 2 in both
   arms), and it moves cdev8 hard but the anchor barely. cdev8 slope 82 +- 5 ->
   38 +- 3 MB/step (-54%, 7.1 sigma) and peak 2.676 -> 2.065 GB (-23%); cdev
   slope 97 +- 10 -> 80 +- 8 (-17%, **1.3 sigma**) and peak 8.221 -> 7.888 GB
   (-4%, 2.3 sigma). The off arm reproduces 446 (97 +- 10 against 103 MB/step,
   8.221 against 8.180 GB), so the comparison is sound. **Keep the fix** -- it is
   free, bitwise neutral, and removes twelve executable families -- but describe
   it as removing a term, not as fixing the growth.
2. **DONE (job 452): K=40 and K=60 are measured, and the peak does not grow with
   K.** Section 3b. cdev lands at 7.727 GB against 7.888 at K=15; both rungs
   saturate; both gates pass at long K; the jump class is attributed to a `cap`
   rung crossing. **Now owed out of it:** a cdev trim leg at K=40 (the anchor's
   retention at the ratified cadence), and a step-boundary trim arm, because the
   measured arm trims 263x per step and so does not price the proposed default.
3. **Re-priced (2026-08-13): `kick_pending` SURVIVES as the C-gh binding term**,
   and the reason is that `cap` does not grow. The config table holds the fine
   cell fixed and grows volume at T=256/b=32, so cdev, cgh64 and C-gh share
   N/tile = 2,097,152 and P = 320: the tile transient is the SAME 2.235 GB
   modelled (3.432 measured) at all three, while `kick_pending` grows 512x to
   274.9 GB. Crossover at n_part ~ 475, so **the anchor is the last rung where
   the tile force wins** and section 1's finding never contradicted the planner.
   `kick_pending` alone is 2.37x a 116 GB host. Section 3b adds that the tile
   transient is K-invariant too, so everything growing from the anchor to C-gh
   is the O(N) family. Three defects fixed in `plan.py` on the way (transients
   were not candidates for the largest-term line; `tile_buffers` vanished
   without a `cap`; `--buf` shadowed the preset table). Still owed: name the
   `mesh_bytes` sub-terms of the tile force.
4. Stage 2a (per-brick velocity scales) removes the `pending` term, which this
   measurement shows is smaller than the tile transient at cdev -- new
   information for its priority, not a reason to drop it, since it is O(N).
