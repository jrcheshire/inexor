# M-v2-4: the f32 coarse force mesh buys 1.83x measured, and costs 1e-5 where the mesh itself costs 1e-2

**Result: adopt.** Branch `jc/m-v2-4-f32-mesh`. Accuracy on deneb (jobs 399, 400,
408, 410, 411) and Vista `gh` (899070, partial); the memory ladder on antares
(415, 416, 417). Cards `runs/v2/m4_gate_*.json`. D-v2-22 is drafted from this
record. Nothing here amends a ratified decision; it supplies the measurement
D-v2-16 clause 3 asserted without one.

Two headline numbers, and they are answers to different questions:

| question | answer |
|---|---|
| what does f32 on the coarse mesh BUY | **1.830x** peak host memory at n_coarse=1024, measured |
| what does it COST | **1.3e-5 to 1.7e-4** on P(k) in band, against a 3.0e-3 budget |
| what does the f64 reference ALREADY cost | **3.590e-2** from its own coarse mesh being finite |

## The gate, and why the charter's could not be run

D-v2-18's row reads "re-run `v2_g3_floors.py` unchanged". That script imports
`force_global` from the FROZEN probe `v2_g5_core.py`, so a dtype threaded through
the package never reaches it, and it drives that force as `which="mono"`, a
single-level monolithic solve that never touches the two-level coarse arm this
milestone changes. Its existing `fdtype` arm casts only the initial `x, v`, and
`float_step_bullfrog` promotes back to f64 on the first operation, so the 1.1e-6
on record measures the f32 rounding of one array. Second milestone running whose
written exit criterion could not be read literally (M-v2-3 was the first).

Replaced by a two-tier bar, pre-registered in the probe docstring before any run:

- **tier 1, the budget share: 3.0e-3.** A fifth of D-v2-9's 3e-2, the share the
  architecture allots this term.
- **tier 2, the expectation: a few x 1e-6**, from the size of f32 rounding alone,
  with **1e-4 named as the level demanding investigation**.

Tier 1 alone is unfailable at these margins, which is the objection that ruled
out gating on a mesh floor in the first place. Tier 2 is what discriminates, and
it is the tier that missed.

## Part 1 -- accuracy, the arm that ships (f32 coarse, f64 fine)

E = max over in-band bins of |P_f32/P_f64 - 1|, band k <= 0.2 k_Nyq(fine)
= 2.513 h/Mpc, D-v2-9's band, estimator `v2_m3_engine_gate._pk` imported
unmodified so E is directly comparable to M-v2-3's codec figure.

### cdev8 (128^3 particles, L=64, n_coarse=64), 15 cards

| K | n | mean E | scatter | mean / 1e-5 | sigma above 1e-5 |
|---|---|---|---|---|---|
| 10 | 1 | 1.321e-05 | -- | 1.3 | -- |
| 20 | 4 | 3.617e-05 | 41% | 3.6 | 3.6 |
| 40 | 6 | 7.493e-05 | 67% | 7.5 | 3.2 |
| 80 | 4 | 5.666e-05 | 73% | 5.7 | 2.3 |

### cdev (256^3, L=128, n_coarse=128), the second volume rung

| K | seed | E_max | E_median | margin under 3e-3 |
|---|---|---|---|---|
| 40 | 0 | 4.002e-05 | 8.514e-06 | 75x |

**Every one of 16 cards passes tier 1**, by 18x at worst and 227x at best.
**None reaches tier 2.** Two of 15 cdev8 cards sit above the 1e-4 investigation
line (K=40 seed 5 at 1.709e-4, K=80 seed 2 at 1.160e-4); both are single seeds
inside a per-rung scatter of 41-73%, and JC called them closed on that basis
(2026-08-10).

**K-dependence: none readable.** deneb 399's n=1 ladder appeared to put K=40 ~10x
above its neighbours. deneb 400's replicates killed that: K=40 sits 1.78 sigma
above K=20 and 0.63 above K=80, so the apparent anomaly was seed 0 drawing high
at K=40 and low at both ends. With 41-73% per-rung scatter the ladder is not
monotone and no slope is readable.

**Volume dependence: none across 8x.** cdev8 K=40 means 7.49e-5 against cdev
K=40 at 4.00e-5, same order, no growth. The third rung (512^3) was **dropped by
JC** (2026-08-10): it has not grown across the two rungs measured, and a
paper-grade final validation is the context where a run at that scale earns its
cost.

**E_max is a max over 25 or 51 bins** and so an extreme-value statistic with a
heavy tail; medians sit 3-8x lower and far closer to the tier-2 prediction. This
is noted once and the statistic is NOT swapped post hoc -- tier 2 was
pre-registered on the max.

## Part 2 -- the floors, which is what tier 2's miss has to be read against

Job 410, cdev8 K=40, both arms at f64, computed by `leg_floors` through the same
`_dpp` helper as Part 1, so same estimand, same band, same config.

| floor | value | E is below it by |
|---|---|---|
| coarse mesh, n_coarse vs n_coarse/2 | **3.590e-02** | 210x to 2718x |
| time step, K vs 2K | **4.783e-02** | 280x to 3622x |

The coarse-mesh floor is the one that bears: it is the error the f64 reference
already carries because its own coarse mesh is finite, and it is the only "mesh
floor" about the arm this milestone changes (G2c's 1.3e-1 is the FINE mesh).

**This is context, not a bar, and it is deliberately not converted into one.**
Gating on it would be the post-hoc statistic swap the milestone already declined
when it refused to gate on a mesh floor. Tier 2 is recorded as missed. What the
floors establish is that the miss has no physical consequence: the term is three
orders below an error already accepted in the same quantity.

## Part 3 -- the ladder, which is the milestone's deliverable

Three arms, because an f64-before vs f32-after comparison would credit the dtype
with S6's slabbed decode: `f64_whole` (the pre-M-v2-4 decode), `f64_slab` (what
the engine pays today), `f32_slab` (what the milestone buys). One subprocess per
(rung, arm), because `ru_maxrss` is a high-water mark that never resets.

| n_coarse | f64 peak | f32 peak | ratio | control f64_whole/f64_slab |
|---|---|---|---|---|
| 128 | 0.65 GiB | 0.65 GiB | 1.0000 | 1.0000 |
| 256 | 1.48 GiB | 0.93 GiB | 1.594 / 1.615 | 0.9960 / 0.9961 |
| 512 | 11.25 GiB | 6.23 GiB | 1.8049 / 1.8051 / 1.8052 | 1.0002 / 1.0001 / 1.0000 |
| **1024** | **88.36 GiB** | **48.29 GiB** | **1.8297** | 1.0000 |

Multiple entries are independent jobs (415/416 at the lower rungs, 416/417 at
512). The top rung reproduces to 4 significant figures across jobs; n=256 moves
1.4%, which is the expected structure, since the fixed baseline is a large share
of the peak there and a negligible one at 512. The control at ~1.0 everywhere
confirms the headline is not being credited with the slabbed decode.

### The peak decomposes exactly, and the pre-registered mechanism was wrong

Fitting peak = baseline + int64 accumulator + float working set:

| term | size | dtype-dependent |
|---|---|---|
| interpreter + jax baseline | 0.21-0.22 GiB, constant in n | no |
| int64 paint accumulator | 8.0 B/cell | **no** |
| float working set (kernels, dk, three force meshes) | 40.2 B/cell at f32, 80.3 at f64 | yes |

The float term reads 40.2 B/cell at n=512 and 40.1 at n=1024, consistent to
0.2%, and the model reproduces both measured ratios to four digits (1.8058 vs
1.8058; 1.8298 vs 1.8298).

**The pre-registration said the ratio approaches 2.0 from below as the fixed
baseline dilutes. The mechanism is wrong.** The baseline is 0.22 GiB and does
dilute to nothing, but what caps the ratio is the int64 accumulator, which scales
as n^3 exactly as the float payload does and therefore never dilutes. The
harness's asymptote is (8 + 2x40.1)/(8 + 40.1) = **1.834**, and n=1024 at 1.8297
is already 99.8% of the way there. The trend is not "still climbing toward 2.0";
it has converged to 1.83.

### What that ceiling belongs to, and it is not the engine

`ladder_worker` holds `mesh`, the int64 accumulator, as a live local while
`coarse_force_meshes` runs. **The engine does not**: `engine.py:372` calls
`coarse_delta_streamed`, where the accumulator is a function-local freed on
return, and only then does line 375 call `coarse_force_meshes` on the decoded
delta. So the 8 B/cell term is co-resident with the solve in the harness and not
in the engine.

Netting it out gives **1.960 at n=512 and 1.9945 at n=1024** -- the
pre-registered 2.0, recovered. Both numbers are reported because they answer
different questions: 1.830 is what was measured, 1.994 is what the engine's own
sequencing implies, and the second is a code reading rather than a measurement.

## What this leg does NOT measure

- **The engine's end-to-end peak.** This measures the coarse-mesh working set and
  its dtype scaling in isolation. The engine's true peak may be set elsewhere,
  for instance during the streamed paint where the accumulator IS resident
  alongside brick decode buffers. Different measurement, not performed.
- **Production scale.** The largest rung measured is n_coarse=1024. The C-gh
  saving is this measurement carried up by arithmetic. Reported as a bound.
- **Out-of-core.** A coarse mesh larger than host memory is unmeasured at any
  rung, and n=2048 (~720 GiB f64) is out of reach of any host available.
- **Wallclock.** The f32 arm's solve is faster in every rung (13.40 s vs 21.61 s
  at n=1024) but this is an XLA-CPU harness on one node and no wallclock claim is
  made from it. JC's rationale for adoption includes that f32 is the direction
  accelerator hardware is prioritizing, which is a forward-looking argument and
  not a measurement of ours.

## Also measured, and NOT a proposal: the fine arm

Job 411, cdev K=40, f32 on the FINE mesh with the coarse at f64: E_max 5.025e-05,
median 1.190e-05, 59.7x under the bar. Recorded as information. The carried
finding is that f32 pays on the COARSE mesh at hero scale and not on the tiled
fine mesh, so no fine-mesh adoption is proposed here.

## The census went nonzero for the first time, and it is benign

`cells_inexact_f32_max = 2` at cdev (2 of 2,097,152 coarse cells), where every
cdev8 card reads 0. Those cells hold an integer sum whose exact value would not
survive a round trip through f32.

**Nothing performs that conversion.** `engine.py:340` decodes as
`s.astype(np.float64) * scale / mean - 1.0`: the integer sum goes to f64, and
only the result, a density contrast of order unity, narrows to the requested
dtype. The counter is a headroom canary, opt-in, and its docstring says so.

It counts round-trip failures rather than cells above 2^24 deliberately, since
`< 2^24` is sufficient for exactness and not necessary (`5000 * 2^12 =
625 * 2^15` is exact at 2.05e7). That refinement is the product of the earlier
integer-ceiling work, not a gap in it. It is a different quantity from D-v2-20's
bucket index, where narrowing to uint16 wrapped modularly so 65536 stored as 0.

## Corrections issued during this milestone

1. **Three `fdtype` parameters in the package existed and could not apply**: one
   died at its only caller, one was defeated by f64 weights promoting the
   accumulation, one was dropped by a positional call. On this codebase a dtype
   parameter is not evidence of a dtype path.
2. **D-v2-16 clause 3's 12.9 / 103 GB are f32 figures** while the shipped code
   ran f64 at 25.8 / 206. The architecture record labelled the dtype; the ADR and
   a code comment dropped it. Corrected additively, clause text verbatim.
3. **deneb 399's K=40 anomaly was seed scatter**, killed by 400's replicates.
4. **The pre-registered approach-to-2.0 mechanism was wrong**, above.

## Instrument defects found in this milestone's own apparatus

Four, against zero in the code under test. Recorded because the class recurs.

1. **A CPU-measured anchor pooled with 14 GPU cards** read as a clean fourth rung
   and moved the least-squares slope +0.32 -> +0.80, against a pre-registration
   where ~1.0 blocks adoption. Fixed by `_provenance()` plus
   `check_comparability()`, which refuses on disagreement and reports
   pre-provenance cards as UNKNOWN rather than passing them (`e4bf778`). The 14
   deneb cards and the cdev card stay UNKNOWN: re-running costs ~2.5 h and
   changes no number, and backfilling an inferred backend is worse than UNKNOWN.
2. **Vista 899070 chained six legs into one job**, cgh64-k10 blew a 2.3 h
   estimate, and 84% of a 6 h 41 min wall produced nothing. The four owed legs
   were re-cut one per job, with walls from measured card walls rather than
   scaling guesses. The cdev gated card above is what that job banked before it
   died.
3. **The mesh ladder's sbatch forbade the only backend that measures it.**
   `ru_maxrss` is host memory and on CUDA the mesh is in VRAM, so all three arms
   read 1.21 GiB and every ratio came out exactly 1.000 (deneb 409, rc=0 over 5
   of 9 dead arms). The precondition is inverted and the env is now `default`,
   which is CPU-only by construction.
4. **The replacement's own content guard failed a run that measured** (415),
   because it treated any rung at exactly 1.0 as the void signature. A rung reads
   1.0 whenever the peak is set by the import and compile footprint rather than
   the arrays. The discriminator is the TOP rung, where the payload is largest
   against the fixed baseline; when the host is genuinely blind, every rung reads
   1.0 including that one. Mutation-tested against the real card plus five
   mutations before re-submitting.

## Owed

Nothing on the milestone. Carried forward unchanged from M-v2-3: `SlotState.repack`
allocates O(N) where D-v2-19 clause 3 establishes a monotone in-place form, and
the parity instrument compares an O(N) array so cgh64 is its ceiling.

New and small: the 8 B/cell int64 accumulator is now the visible next term on the
coarse arm, and `v2_m3_engine_gate.py` still lacks the provenance stamp that
`v2_m4_f32_mesh_gate.py` gained.
