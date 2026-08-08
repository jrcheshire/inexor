# V4 pricing record — what V4 costs, and what it beats

**Purpose.** Price V4's premise *before* building it: is the memory win real
against a competitor, and is the wall acceptable at the config-table home?
Both questions are answered here. This is a pricing record, not a gate: no
bar is tested and nothing here is ratified. The V4 architecture freeze is
where decisions get made.

Organised for lookup. §1 is the competitor comparison, §2 the cost law, §3 the
C-gh estimate that falls out of it, §4 the gb node readout, §5 what this does
NOT license, §6 corrections and defects.

> **SUPERSEDED IN TWO PLACES (2026-08-07, `runs/v2/v4_architecture_record.md`).**
> Read §2 with both of these in hand:
>
> 1. **The "paint/gather-dominated" reading of the device phase is wrong.**
>    §2 finding (4) says the device phase is "far above its FFT floor, i.e.
>    paint/gather-dominated -- which is what G1's Pallas work targets." More
>    than half of that excess was zero-weight atomic contention: the tile
>    padding indexed particle 0 and `_tile_corner` sent every masked row to
>    flat index 0, so ~2e6 dead rows per tile contended on ONE address. Fixed
>    in `eba91ab`; measured 47.15 -> 21.26 ms, **2.218x**, at identical `cap`
>    and geometry (job 896159 legs 1-2). The custom-kernel lane is
>    correspondingly less urgent.
> 2. **The top rung of §2's `cap` table is inflated by a defect.** `cgh64`
>    T128/b96's `cap` of 10,168,320 is ~1.7x too large because `choose_brick`
>    was missing the `c | b_realized` condition, so the brick union overshot
>    the padded box (union side 384 against P = 320) and that leg alone carries
>    3,044,340,012 overhang against 0 at every other leg. Fixed in `eba91ab`.
>    No operating geometry moves. **Do not fit a cost law through that point.**
>
> §2's central claim survives both and is now ISOLATED rather than
> correlational: `stage` is linear in `cap` and device goes as ~`cap^0.5`, with
> members, bricks, tiles and kernel all held fixed (job 896159 legs 3-4). The
> fixed-`n_brick` pair §2 records as owed is discharged by that instrument.

## Provenance

| job | machine | elapsed | what | commit |
|---|---|---|---|---|
| 894167 | Vista gb | 00:00:59 | fp64 capability, GB200 (G4's in-flight leg) | 31adeba |
| 895315 | Vista gh | 00:01:54 | V4a tile-phase anatomy, 3 legs | c20fcc7 |
| 895316 | Vista gh | 00:01:51 | V4b DISCO-DJ forward capacity ladder | c20fcc7 |
| 895439 | Vista gh | 00:07:18 | V4c cap-growth, tile count at fixed box, 3 legs | 73d261c |

Cost: ~0.19 node-hours across the three V4 jobs. Node for 895316 was
`c620-111`, GH200 120GB, 97871 MiB HBM, 71.25 GiB usable to JAX at the
default 0.75 memory fraction.

Cards: `runs/v2/v4_tile_anatomy.json` (the six anatomy legs),
`runs/m2/p1_disco_mem_v4b_fwd_capacity.json` (the ladder),
`runs/v2/g4b_capability_gb.json` (the GB200 dump).

894167 discharges the item D-v2-13's record left in flight. That record is
ratified and is deliberately not edited here; a forward pointer in it is
owed if wanted.

## 1. DISCO-DJ forward capacity — the comparison v2 actually claims

The only competitor number we owned was an *adjoint* one at 128^3 (DISCO-DJ
f32 437 B/p, f64 816, inexor v1 488). That measures the wrong thing twice:
v2 is a forward engine, and matched-N bytes-per-particle is not the claim.
The claim is **the largest box one card holds**.

Forward-only, f32 (DISCO-DJ's default and favourable case), `res_pm = 2 x
n_part` to match the config table's `n_fine = 2 x n_part` per side. Without
that matching DISCO-DJ would carry an 8x smaller mesh than the arm it is
compared against.

| particles | res_pm | forward peak | B/p | wall |
|---|---|---|---|---|
| 64^3 | 128 | 0.095 GiB | 389.4 | 1.85 s |
| 128^3 | 256 | 0.604 GiB | 309.2 | 1.73 s |
| 256^3 | 512 | 4.510 GiB | 288.6 | 2.43 s |
| 384^3 | 768 | 14.889 GiB | 282.3 | 4.39 s |
| 512^3 | 1024 | **35.039 GiB** | **280.3** | 8.02 s |
| 640^3 | 1280 | **OOM** (cuFFT plan) | — | — |

Converges to ~280 B/p. The forward-ran guard (particles actually moved, all
finite) passed at every rung, so no rung is a flattering no-op.

**DISCO-DJ's forward ceiling on one GH200 is between 512^3 and 640^3.**

At the matched config (512^3 particles, 1024^3 fine mesh, one GH200, f32 —
the `cost_of_memory.md` §V2a capacity legs, NOT the f64 anatomy legs below):

| arm | device peak | B/p | vs DISCO-DJ |
|---|---|---|---|
| DISCO-DJ forward | 35.0 GiB | 280.3 | — |
| inexor mono force | 48.3 GB | 360.2 | **0.78x (we lose)** |
| inexor tiled T256/b32 | 1.78 GB | 13.29 | 21.1x |
| inexor tiled T128/b32 | 0.44 GB | 3.31 | 84.7x |

**Read it as: the win is entirely the tiling.** Monolithic to monolithic we
lose, 360 against 280. That is the honest framing and the better argument —
the architecture is the result, not the implementation. Quoting the tiled
number alone invites exactly the challenge the mono row answers in advance.

**Estimand caveat that must travel with these numbers.** DISCO-DJ's 35 GiB is
a full 10-step forward evolve; ours is one force evaluation with the O(box)
host accumulator excluded and reported separately as host RSS. Not the same
quantity. The **ceiling** comparison is the robust one, because it is a
yes/no about identical hardware: DISCO-DJ dies between 512^3 and 640^3, and
G4 ran a 2048^3 paint on the same card at 3.91 B/p.

## 2. The cost law — per-tile cost tracks `cap`, not tile count, not box

`force_short_tiled` now records four per-tile phase timers: `member` (host
bucket lookup + index/mask build), `stage` (the numpy fancy-index gather
`pos_np[idx_pad]` + H2D), `device` (jitted paint/FFT/gather + D2H), `scatter`
(host write-back). No jit splitting and no added syncs, so fusion is untouched
and the phases still sum to the wall the un-instrumented code would take.
Verified default-preserving: forces bitwise identical with and without.

All six rungs, f64, one GH200. Medians over tiles 1.., so compile is excluded.

| box | P | tiles | cap | member | stage | device | scatter | **steady/tile** |
|---|---|---|---|---|---|---|---|---|
| cdev | 192 | 64 | 1,284,096 | 1.0 | 15.0 | 10.6 | 8.0 | **34.6 ms** |
| cgh64 | 192 | 512 | 2,078,720 | 1.2 | 25.2 | 24.1 | 8.0 | **58.5 ms** |
| cgh64 | 192 | 4096 | 2,213,888 | 1.1 | 24.5 | 25.5 | 2.7 | **53.7 ms** |
| cdev | 320 | 8 | 4,915,200 | 5.4 | 63.7 | 30.9 | 56.1 | **156.0 ms** |
| cgh64 | 320 | 64 | 5,986,304 | 4.9 | 75.7 | 47.8 | 57.8 | **186.3 ms** |
| cgh64 | 320 | 512 | 10,168,320 | 7.8 | 120.4 | 122.6 | 22.2 | **273.1 ms** |

Fixed-box 8x tile-count steps (V4c's design: raise tile count by shrinking T
at fixed P = T + 2b, so box, state arrays and per-tile geometry are untouched):

| axis | step | cap | device | steady |
|---|---|---|---|---|
| P=192 | 512 -> 4096 tiles | 1.07x | **1.05x** | 0.92x |
| P=320 | 64 -> 512 tiles | 1.70x | **2.56x** | 1.47x |

Confounded box+tile steps, for contrast:

| axis | step | cap | device | steady |
|---|---|---|---|---|
| P=192 | cdev 64 -> cgh64 512, 8x box | 1.62x | 2.29x | 1.69x |
| P=320 | cdev 8 -> cgh64 64, 8x box | 1.22x | 1.55x | 1.19x |

**Findings.**

1. **Tile count per se costs nothing.** The P=192 fixed-box step is the
   decisive rung: 8x the tiles at essentially unchanged `cap` leaves device
   time flat (1.05x) and steady-state cost slightly *lower* (0.92x). Across
   the whole table tile count varies 512-fold (8 to 4096) while device time
   varies 11.6-fold, and it tracks `cap` (7.9-fold), not tile count.
2. **`cap` is the cost variable.** Every rung where `cap` rose, device rose
   more; every rung where `cap` held, device held.
3. **`cap` is a design choice, not a physical cost.** It is the padded
   per-tile particle capacity, set by a global max over tiles and by the
   brick decomposition. Both are ours to change.
4. **Host plumbing dominates.** At the C-gh candidate geometry, steady state
   is `stage` 75.7 + `scatter` 57.8 + `member` 4.9 = 138.4 ms of host against
   47.8 ms of device, i.e. **74% host**. The device phase is itself far above
   its FFT floor (four P=320 f64 FFTs should cost ~2 ms, scaling the G4b card
   measurement below), so it is paint/gather-dominated — which is what G1's
   Pallas work targets.

**Attribution limit.** None of the three fixed-box pairs held the brick
decomposition constant: `n_brick` went 32 -> 64 in both steps. So `cap` and
brick geometry co-vary in every step measured. "Device tracks `cap`" is
supported by the correlation across six points and by an obvious mechanism
(you paint and gather `cap` padded particles per tile), but it is **not
isolated**. Isolating it needs a pair at fixed `n_brick`.

**Internal consistency check that passed.** `scatter` is the host write-back
of owned rows, so it must fall as tiles are added. It does: 8.0 -> 2.7 ms and
57.8 -> 22.2 ms, both ~0.36x for an 8x tile increase. The phases measure what
they claim.

## 3. The C-gh cost estimate

C-gh = 2048^3 particles in L = 1024 Mpc/h, fine mesh 4096^3 (tiled), coarse
mesh 1024^3, at T=256/b=32, hence P=320 and **4096 tiles**. One GH200.

Going cgh64 -> C-gh at fixed (T, b) holds P, density and per-tile brick
geometry fixed; only the number of supersets `cap` maxes over changes,
64 -> 4096. The P=192 fixed-box pair brackets that at ~1.1x over an 8x range,
so **1.1-1.7x over 64x** is the working bracket.

| basis | per force evaluation | K=40 per realization |
|---|---|---|
| device only, 4096 x 47.8 ms x [1.1, 1.7] | 215-333 s | **2.4-3.7 h** |
| current form, 4096 x 186.3 ms x [1.1, 1.7] | 14-22 min | 9.3-14.4 h |

Assumes one force evaluation per step, correct for BullFrog.

**Excluded from both rows, all of which push them up:** the global coarse
solve (one 1024^3 FFT-based force per step at C-gh), the state update, IC
generation, and the fact that streaming does not make data movement free — it
removes the numpy gather and makes transfers overlappable, so the 74% host
share is *the size of the term attacked*, not the saving.

**Superseded:** a pre-anatomy estimate of ~1367 s per force evaluation
(15.2 h per realization) derived by dividing the G5c capacity card's 21.36 s
by its 64 tiles. That per-tile figure includes XLA compile, measured here at
8.9 s of a 29.8 s wall at 64 tiles. Compile does not scale with tile count and
must be excluded from any extrapolation.

## 4. GB200 fp64 (job 894167)

Read by D-v2-13's pre-registered rules: the GEMM f64:f32 ratio against the
two-part Hopper baseline, and the FFT by **wall**, never the printed GB/s
ratio (complex128 moves 2.00x the bytes, so that ratio reads >1 for a healthy
part).

| part | GEMM f64:f32 @4096 | @8192 | abs f64 TFLOP/s | abs f32 TFLOP/s |
|---|---|---|---|---|
| GB200 | **0.091** | **0.047** | 36.4 / 38.2 | 399 / 810 |
| GH200 | 0.193 | 0.157 | 59.8 / 64.6 | 310 / 412 |
| S3 H100 | 0.192 | 0.156 | 60.6 / 63.7 | 316 / 407 |

| part | FFT f64/f32 wall @256^3 | @512^3 | abs f64 wall @512^3 |
|---|---|---|---|
| GB200 | 1.394 | 1.541 | **2.265 ms** |
| GH200 | 1.613 | 1.787 | 3.818 ms |
| S3 H100 | 1.720 | 1.917 | 6.454 ms |

**The two statistics disagree, and the one that bears on inexor says gb wins.**
GB200 fp64 GEMM is genuinely throttled: 2-3x below Hopper's own ratio and
0.59x Hopper in absolute f64, while its f32 is 1.96x. But the FFT shows no
fp64 penalty at all — gb sits *further* below the 2.00 bandwidth-scaling bar
than Hopper, and in absolute f64 wall a GB200 does the 512^3 FFT **1.69x
faster than a GH200** and 2.85x faster than an S3 H100. inexor's force path is
FFT and CIC and contains no GEMM, so the GEMM row characterizes the part
without predicting us.

Corollary for fp64 emulation (Ozaki-style splitting into low-precision
tensor-core matmuls): it targets ALU-bound *matmul*, and our f64 FFT is
demonstrably not ALU-bound on the part with the weakest fp64 ALU. Not a lever
for us today. Worth watching only as insurance against further fp64 cuts.

**Caveat:** 256^3 and 512^3 are small enough that launch latency has a share,
and the production coarse mesh is 1024^3-2048^3. Re-read at size before
anything leans on the ranking.

**New node facts.** A gb node is 4x GB200 at 189,471 MiB HBM each, 144
Neoverse-V2 cores, and 1691 GiB host — but `numactl` shows that host split
across **two CPU NUMA domains of ~478 GiB each**, plus HBM exposed as further
NUMA nodes. A single-GPU streaming job likely sees ~478 GiB, not 1.69 TB.
Reading the headline total would overstate by 3.5x, the same shape of error as
pricing a CPU run with a device number (the "~48 GB" cgh64 trap).

## 5. What this does NOT license

- **No wall comparison against DISCO-DJ.** Our tiled force runs far above its
  FFT floor with 74% of its cost in host plumbing, so timing it against
  DISCO-DJ's production path today would measure our probe. `wall_forward_s`
  in the ladder card is provenance only.
- **No C-gh measurement.** C-gh was not run and cannot be with this probe:
  `pos_np` and the accumulator are full f64 (n,3) host arrays, 2 x 206 GB at
  C-gh against a gh node's ~116 GB usable host (D-v2-13 clause 2). Running it
  needs D-v2-13 clause 3's streaming path wired into the force, which is the
  V4 build. §3 is an extrapolation with a stated bracket, not a measurement.
- **No C-hero number.** Nothing here ran above cgh64.
- **No claim that streaming recovers the 74%.** See §3.
- **Legs V4c-2 (T128/b96) and V4c-3 (T64/b64) are not operating points.**
  Padded-volume ratios 15.6 and 27. They are instruments for isolating a cost
  law and must never be quoted as candidate geometries.
- **The `cap` attribution is not isolated** (see §2).
- **Accuracy is untouched.** Nothing here bears on D-v2-9's bar, D-v2-11's
  transport, or any fidelity question.

## 6. Corrections and defects

**Corrections made during this arc.**

1. "~1367 s per force evaluation at C-gh, 15 h per realization" — wrong,
   compile-contaminated. See §3.
2. "`cap` is a max over tiles, so it grows with tile count" — the strong form
   is refuted by the 4096-tile rung, where 8x the tiles moved `cap` by 7%.
   The extreme-value effect is real but weak; what moves `cap` is the brick
   decomposition. This matters because it changes what to fix.
3. A "2.5-3 h" C-gh estimate quoted before V4c rested on a growth factor
   picked rather than measured. §3's bracket is the measured replacement; the
   old number happens to sit at its low end, which does not retroactively
   justify it.
4. The gb fp64 note retracted at V3 (one uncontrolled ichnaea row) now has a
   controlled measurement. The retraction stands as to provenance; the
   direction it guessed is right for GEMM and wrong for FFT, and only the FFT
   bears on us.

**Defects found, not fixed.**

- **`cap` is a global max over tiles and brick-decomposition-driven.** Every
  tile pays the worst tile's padding. Per-tile or bucketed capacity is the
  lever, and §2 says this is the cost variable, so it should be settled
  together with the streaming design rather than after it.
- **`m2_p1_disco_mem` summary printer** reports "no device stats: CPU?" in
  `--forward-only` mode because it keys on `peak_adjoint`, which is now
  `None`. The JSON is correct; only the printed table is wrong.
- **Version skew in the comparison:** disco-mocks' gpu env on Vista is jax
  0.10.1, inexor's is 0.10.2. Recorded for provenance; not thought to matter
  for a capacity ceiling, and untested.

## 7. What V4 inherits from this

- The premise prices out. The memory claim is 21-85x against a real
  competitor with a 64x volume ceiling difference on the same card, and the
  wall is not the obstacle it appeared to be.
- **Design against `cap`, not against the wall.**
- Choose the C-gh geometry on `cap`, not on padded-volume ratio. T256/b32
  wins on both in this table, but the two criteria are not the same and are
  already visibly diverging.
- The streaming force build carries an untested risk: G4's 359-367 GB/s was a
  *sequential chunked* stream, while the tiled force gathers by tile
  membership, which is scattered in the global particle ordering and degrades
  as particles move.
