# V4 architecture record: the measurements the freeze rests on

**This is a measurement record, not a verdict.** Nothing here is ratified. The
decision records drafted from it (D-v2-14..18, `docs/decisions.md`) are
PROPOSED and await JC. Where a number is an extrapolation or an estimate it
says so in the line that carries it.

Companion to `runs/v2/v4_pricing_record.md` (2026-08-07), which priced V4's
premise. This record measures what the freeze needs to decide, and it
**corrects two conclusions in that document** (sections 1 and 2 below).

## 0. Provenance

| job | machine | what | state |
|---|---|---|---|
| 896055 | Vista gh | 7-leg probe | **VOID** -- ran nothing, see section 8 |
| 896092 | Vista gh | T9 quantum ladder | **VOID** -- ran nothing, see section 8 |
| 896159 | Vista gh | 7-leg probe, 18m13s | legs 1-4, 6-7 stand; leg 5 defective |
| 896160 | Vista gh | T9 quantum ladder at cdev, 11m49s | stands |
| 896408 | Vista gh | V4e round 2 (pinned H2D) | stands; supersedes 896159 leg 5 |

Code: `eba91ab` (force-path defect fixes), `0a24ed6` (probes), `8746a63`
(sbatch fixes), `e3d6f27` + `d3c5362` (V4e round 2). Cards:
`runs/v2/v4d_ic_memory_v4d_gh_{ic,fft}.json`,
`runs/v2/v4e_stream_runlength_v4e_gh_pinned.json`, `runs/v2/g2c_results.json`
(overwritten by 896160 -- see section 8).

All force legs: `cgh64`, T=256, b=32, P=320, 64 tiles, f64, one GH200. Times
are medians over tiles 1.., so compile is excluded (V4a's lesson).

## 1. The tile padding was a same-address atomic hot spot

`force_short_tiled` filled `idx_pad` with `np.zeros`, so every padded row
indexed particle 0, and `_tile_corner` sent every masked row to flat index 0
regardless. `pad_frac` at this geometry is 0.329: about 2e6 dead rows per tile,
each touching 8 CIC corners, all scatter-adding onto ONE address. GPU
scatter-add is atomic, same-address atomics serialize, and XLA cannot elide
them because the zero weight arrives through a data-dependent `where`.

The two halves are coupled: cycling the fill does nothing while the index is
rewritten, and dropping the rewrite does nothing while every pad row carries
particle 0's position. Fixed together in `eba91ab`. The index rewrite was never
what made the index safe -- `% nx` is, since `u = mod(pos - origin, L)` bounds
`base` by `n_fine` -- and only the weight zeroing is load-bearing, so the force
is unchanged by construction.

| arm | device median | note |
|---|---|---|
| `zero` (pre-fix) | 47.15 ms | V4a job 895315 recorded 47.8 ms: reproduced to 1.4% |
| `cycle` (shipped) | 21.26 ms | **2.218x faster** |

The control reproducing V4a to 1.4% at identical `cap` and geometry is what
makes this readable: the only thing that differs is the fill.

**CORRECTION to `v4_pricing_record.md` section 2.** That document reads: "the
device phase is itself far above its FFT floor, i.e. paint/gather-dominated --
which is what G1's Pallas work targets." More than half of that excess was
zero-weight atomic contention on padding, not paint work. The device phase is
26% smaller than the arithmetic there assumed, and the custom-kernel lane is
correspondingly less urgent. The sentence should be read as superseded.

**A cost the fix introduces, not free.** `stage` rose 13% (75.88 -> 85.92 ms):
cycling makes the host gather touch `cap` scattered real rows instead of
re-reading row 0 out of cache. Net win 12.4 ms/tile. Steady per-tile improved
only 1.07x (186.20 -> 173.79 ms) because host plumbing now dominates harder
still, at ~88% of the tile cost.

## 2. `cap` is the cost variable, now isolated

`v4_pricing_record.md` section 2 named `cap` from a correlation and recorded
the isolation as owed ("a pair at fixed `n_brick`"). A capacity multiplier is
strictly cleaner than that pair: members, bricks, tiles, geometry and kernel
are ALL held fixed and only the padded row count moves.

| | cap | device | stage | steady |
|---|---|---|---|---|
| x1.0 | 5,986,304 | 1.000 | 1.000 | 1.000 |
| x1.5 | 8,979,456 | 1.210 | 1.487 | 1.255 |
| x2.0 | 11,972,608 | 1.540 | 1.973 | 1.540 |

**`stage` is linear in `cap`** (1.487 against 1.500; 1.973 against 2.000).
**Device goes as roughly `cap^0.5`**, which decomposes sensibly: the FFT is
fixed at P^3 regardless of `cap`, and only paint and gather scale with it.
Solving the two device points gives a fixed part near 9.8 ms and a
`cap`-proportional part near 11.5 ms, so about half the device time at the
operating point is `cap`-driven.

"Design against `cap`" therefore STANDS and is now measured. The owed
fixed-`n_brick` measurement is discharged by a better instrument and should be
struck from the owed list.

**Not licensed:** `cap_mult` legs are instruments, never operating points. And
the exponents are two points each at one geometry; they support "linear" and
"sublinear", not a fitted power law.

## 3. The streaming redesign is gather-bound, not transfer-bound

Round 1 (896159 leg 5) is **withdrawn**: it staged from pageable numpy memory,
so its H2D flat-lined near 19.6 GB/s at every run length and could not test
D-v2-13 clause 3's `pinned_host` path. See section 8.

Round 2 (896408), gather and pinned H2D timed separately:

| run | gather | H2D pageable | H2D pinned | ideal |
|---|---|---|---|---|
| 4 KB | 2.8 | 21.1 | 176.2 | 2.8 |
| 16 KB | 7.9 | 19.8 | 196.4 | 7.6 |
| **37 KB** | **12.2** | 20.2 | **203.8** | **11.5** |
| 98 KB | 15.7 | 20.3 | 207.5 | 14.6 |
| 256 KB | 17.9 | 19.9 | 209.6 | 16.5 |
| 1 MB | 18.4 | 19.5 | 221.4 | 17.0 |
| 4 MB | 17.4 | 17.5 | 216.7 | 16.1 |

Three pre-registered checks, all passed: pinned H2D within ~2x of D-v2-13's
359-367 GB/s and flat above 16 KB; gather at 37 KB reproducing round 1's
12.5 GB/s to 2.4%, which retroactively validates round 1's other half; and
`ideal` at 37 KB equal to the gather rate.

**Pinned H2D is 17x faster than the memcpy feeding it.** The binding constraint
is the host gather.

37 KB is not an arbitrary rung: it is the C-gh brick span at T9
(4096 particles x 9 B = 36.9 KB), and `choose_brick` forces brick <= buffer, so
b=32 fixes the brick at 32 fine cells.

**What it is worth** (arithmetic on measured numbers): `stage` is 85.9 ms
moving 143.7 MB (`cap` x 3 x 8 B at f64), an effective 1.67 GB/s. At 11.5 GB/s
that is ~12.5 ms, a **6.9x** win from access pattern alone; carrying T9's
9 B/p instead of f64's 24 gives 53.9 MB and ~4.7 ms, **18x against today**.

**The `n_brick` trade, now quantified.** Gather saturates near 18 GB/s at
>= 256 KB runs. Reaching that needs b ~ 64, taking the padded-volume ratio from
1.95 to 3.375: **1.7x more device work for 1.5x better gather, roughly a wash.**
b=32 stands.

**`per_run` versus `assemble`** at 37 KB is 0.4 against 11.5 GB/s, a 29x
penalty for the loop written by accident. The design fork is settled.

**`ideal` is a ceiling, not a built thing.** jax exposes no way to memcpy into a
pinned buffer in place, so this probe gathers into pageable memory and stages
to pinned outside the timer. Production must gather directly into
`cudaHostAlloc`-backed memory. That is an ffi-level build item and the 6.9x is
contingent on it.

## 4. The T9 position quantum: the tier survives at a resolution never measured

`v2_g2c_accum_gate.py:35` says the gate that ratified T9 "measures
REPRESENTATION error only (storage layout is a build decision)". V4 takes that
decision, and the arithmetic does not work as assumed. The gated arm quantizes
at `fine_cell/256`, which an int8 carries only if its bucket is ONE FINE CELL
(verified: the t9 arm's bucket is 0.25 Mpc/h, exactly the fine cell). At C-gh
the fine mesh is 4096^3 = 6.9e10 cells against 8.6e9 particles -- eight cells
per particle -- so a per-cell index costs ~69 GB at one byte per entry, more
than the 77 GB of state it indexes. CUBE's sorted-by-cell layout works because
CUBE runs one particle per cell; our fine mesh is 2x the particle grid per side.

An affordable index needs a coarser bucket, which coarsens the quantum. Job
896160, cdev, K=40, velocity codec held identical (int16 max-range) across all
arms so the position quantum is the only variable:

| arm | bucket | quantum | index at C-gh | dP/P | dP0/P0 | dP2/P0 | worst vs 3e-2 |
|---|---|---|---|---|---|---|---|
| `t9` | 0.25 Mpc/h | fine/256 | ~69 GB, does not fit | 1.314e-4 | 1.070e-4 | 2.600e-4 | 115x |
| `t9c1` | 0.5 Mpc/h | fine/128 | 17.2 GB | 3.318e-5 | 1.567e-4 | 3.121e-4 | 96x |
| **`t9c2`** | **1.0 Mpc/h** | **fine/64** | **2.15 GB** | 4.123e-4 | 5.992e-4 | 2.035e-3 | **15x** |
| `t9c4` | 2.0 Mpc/h | fine/32 | 0.27 GB | 3.651e-4 | 6.588e-4 | 8.431e-4 | 36x |

Floors at the same K: step K=40 vs 80 gives dP/P 8.252e-2; mesh 512 vs 256
gives 1.016e-1. Every arm sits ~2.5 orders under both.

`t9` reproducing the ratified 1.3e-4 exactly is the control that makes the rest
readable.

**All four clear D-v2-9's absolute 3e-2 bar.** The tier the layout can actually
deliver is fine.

**What this does NOT establish.** One seed, one config. And the ladder is
**non-monotonic**: `t9c1` beats `t9` on dP/P, `t9c4` beats `t9c2` on dP2/P0. We
are at a floor where the arms are not cleanly ordered, so "c=2 is better than
c=4" is NOT shown -- only that all of them pass. c=2 is the recommendation on
margin plus a 2.15 GB index, not on a measured ordering.

**9 B/p was never an all-in number.** The bucket index and the migration slack
are real bytes:

| item | B/p | GB at C-gh |
|---|---|---|
| T9 payload (int8 x3 + int16 x3) | 9.00 | 77.3 |
| bucket index (1024^3 uint16) | 0.25 | 2.15 |
| brick CSR (128^3 x 16 B) | 0.004 | 0.03 |
| slack + arena (10% of payload, **UNMEASURED**) | 0.90 | 7.7 |
| **total** | **10.15** | **87.2** |

87.2 GB against the ~116 GB cliff is 1.33x under, versus 1.50x for the 9 B/p
figure. D-v2-8's B/p column needs re-deriving with this, which is what D-v2-8
said V4 would do. **The 10% slack is an estimate, not a measurement** -- see
section 7.

## 5. IC generation: the host term is real, and it is flat

The only IC number on record (76 B/p, v1 R6) is DEVICE ONLY. `ic.py:49-58`
evaluates the transfer function on the full 3D rfft half-grid in float64, and
`transfer_eh98` materializes ~15 more arrays of that size.

| n | host peak | host B/p (net) | device B/p at `full` |
|---|---|---|---|
| 256 | 2.19 GB | 126.5 | 80.1 |
| 512 | 12.14 GB | 95.4 | 76.0 |
| 1024 | 89.79 GB | 89.6 | `full` OOM'd |

**Pre-registration partly wrong.** I predicted host B/p would GROW with N. It
does not: it is asymptotically flat near 90 B/p, which in hindsight is what a
fixed number of half-grid arrays must give. The magnitude and the dominance
over the device term are confirmed.

Consequences:
- ~90 B/p at 2048^3 extrapolates to **~773 GB against a 116 GB ceiling**. The
  IC host term is the binding constraint at C-gh, and it was never counted.
- Device at `full` reproduces the recorded 76 B/p (76.0 at n=512).
- `full` at n=1024 OOM'd, which is the six simultaneous (N^3,3) arrays at
  `lpt.py:153-158` biting exactly where predicted.
- The fix for the host term is cheap and both halves already exist in the tree:
  P(k) is a function of |k| only, `_eh98_amplitude` already uses a 4000-point
  log grid, and `linear_power(backend="table")` already log-log interpolates.

## 6. Monolithic FFT capacity: 2048^3 does not fit

| n | field | fwd | inv | device peak |
|---|---|---|---|---|
| 512 | 0.50 GB | 0.032 s | 0.114 s | 3.51 GB |
| 1024 | 4.00 GB | 0.063 s | 0.280 s | 28.04 GB |
| 1536 | 13.5 GB | -- | -- | **OOM at a 27.04 GiB allocation** |

The largest monolithic `jnp.fft.rfftn` on a GH200 is between 1024^3 and 1536^3.
**C-gh needs an out-of-core FFT layer; it cannot defer to C-hero.** The
workspace ratio is the number to design against: a 4 GB field peaks at 28 GB,
**7x the field**.

## 7. What this does NOT license

- **No C-gh force measurement.** Every force leg is `cgh64`, C-gh at 1/64
  volume. The 6.9x and 18x in section 3 are arithmetic on measured rates, not a
  measured C-gh `stage`.
- **No wall/step or realization claim at C-gh or C-hero.** Unchanged from
  D-v2-13: nothing has run above 1/64 volume.
- **The 10% slack in section 4 is an estimate.** It rests on a hand argument
  about migration rates (order 3 x displacement/brick side per step, ~10% at
  C-gh) that has never been measured. It is 7.7 GB of an 87.2 GB budget. The
  measurement belongs in the codec milestone.
- **The T9 ladder is one seed at one config**, and non-monotonic at these
  levels. It licenses "the tier survives", not any ordering among c1/c2/c4.
- **`cap` exponents are two points each.** "Linear" and "sublinear", not a fit.
- **`ideal` in section 3 is a ceiling** contingent on an ffi-level gather that
  does not exist.
- **Accuracy is untouched** except by section 4. Nothing here bears on
  D-v2-9's bar for the split, D-v2-11's transport, or any fidelity question
  outside the state codec.

## 8. Corrections, defects, and void jobs

**Void jobs.** 896055 and 896092 ran ZERO science: Vista's batch environment
does not carry pixi on PATH and every call died rc 127. 896055 is the dangerous
one -- it has no `set -e` (deliberately, so an expected OOM would not kill later
legs), so it exited 0 and **`sacct` reports it COMPLETED 0:0 in 6 seconds**. A
green Slurm record that ran nothing is exactly the failure mode this project's
gate discipline exists to prevent. Recorded here as void so the record cannot
later be cited as provenance, the same treatment 894010 has in D-v2-13.
Fixed in `8746a63`: a `command -v pixi` precondition and a jax-import check
that both abort, per-leg exit codes, a LEG SUMMARY, a listing of cards actually
written, and `exit 1` if zero legs succeeded.

**Defects found in this session's own instruments:**

1. **V4e round 1 staged from pageable memory** and its falsifier was calibrated
   against that defect rather than against the design. Read literally it would
   have reported the streaming redesign falsified at 7.6 GB/s. The tell was
   present in the data -- H2D flat at ~19.6 GB/s across a 1000x span of run
   length is not a run-length effect -- and should have been caught before
   submission. Superseded by 896408.
2. **The first bitwise-parity check of the `eba91ab` fixes passed vacuously**
   on all-zero arrays, because `r_s=None` makes `split_factor` use 0.0 and the
   short kernel is then identically zero. It now asserts real dynamic range and
   that a 1e-9 position shift breaks the equality.
3. **The IC probe's B/p is only meaningful on GPU**; on CPU jax's arrays live in
   host RAM and the host mark stops isolating the numpy term. The card reports
   `host_isolates_numpy` so a CPU run cannot be misread.

**Defects found in the tree:**

4. **`choose_brick` was missing `c | b_realized`** (fixed in `eba91ab`). Without
   it the brick union overshoots the padded box. The V4a card shows overhang 0
   at five of six legs and 3,044,340,012 at the one leg where the condition
   fails (T128/b96). **That leg's `cap` of 10,168,320 is inflated ~1.7x by the
   defect, and it is the top rung of `v4_pricing_record.md`'s cost-law table.**
   No operating geometry moves. With the condition enforced the union is
   exactly the padded box, which promotes `n_out == 0` from a diagnostic to a
   contract -- and the two docstrings that disagree about this
   (`tile_paint_f64:794` says contract, `:1064` says diagnostic) should be
   reconciled in favour of contract at promotion.
5. **`paint_tsc_f64` accumulates through order-dependent f64 `.at[].add`**, and
   `v2_g3_hybrid_evolve.py:54` defaults `--assign-long tsc`. D-v2-10 passed at
   gauss + TSC + matching, so **the ratified coarse arm violates D-006's
   deterministic-paint rule**. `paint_tsc_int` is a required deliverable that no
   existing document names. It is also a precondition for a brick-sorted layout,
   which reorders particles every step.
6. **Coarse force meshes do not fit at C-hero.** Tile-local consumption of the
   long-range force needs three coarse components readable per tile:
   3 x 1024^3 f32 = 12.9 GB at C-gh (fine), 3 x 2048^3 f32 = 103 GB at C-hero
   against 96 GB of HBM. Derived, not measured.
7. `runs/v2/g2c_results.json` was overwritten by 896160 and now holds only the
   T9 arms. The ratified G2c numbers live in `docs/decisions.md` and the
   worklogs, so nothing authoritative is lost.
8. Carried, unfixed: `m2_p1_disco_mem`'s summary printer misreports in
   `--forward-only` mode; jax 0.10.1 (disco-mocks) vs 0.10.2 (inexor).

## 9. What the freeze inherits

- **The state tier survives at a quantum nobody had measured**, at ~10.15 B/p
  all-in rather than 9. Bucket c=2 on margin, not on a measured ordering.
- **The force is never materialized globally.** Its only consumer is the
  elementwise kick and ownership is a partition, so the kick applies
  tile-locally. That deletes both O(N) f64 host arrays -- the 2 x 206 GB that
  have kept C-gh unrunnable.
- **The host gather is the streaming constraint**, worth 6.9x on `stage` at f64
  and 18x at T9, contingent on an ffi-level pinned gather.
- **IC generation needs both a cheap fix and an expensive one**: the 1D
  transfer table removes ~773 GB of host allocation, and the out-of-core FFT
  layer is required at C-gh rather than deferrable.
- **`paint_tsc_int` and the coarse sub-block staging are unnamed deliverables**
  that the freeze must name.
- **b=32 stands** on a quantified trade rather than on the 4 r_s sizing alone.
