# The host-state / device-step design: 4096^3 on one Vista gb node

Opened 2026-09-06, branch `jc/device-step-4096`. This is the design record that
`m6_scaling_record.md` sec. 5y "Owed" names. Its companion is that record's
sec. 5y and 5z, which price the design and are not repeated here.

**Status: priced; D1 built and CLOSED (secs. 7-9).** Sections 1-6 are
arithmetic over a design plus readings carried from 5y/5z; the one thing they
add that was in neither is the MEMORY budget, which changes what the binding
constraint is. Section 7 is the first MEASUREMENT of any part of this design:
the coarse solve's transform, which is correct at 2048^3 and 78x its projected
wall.

## 1. The design

The CPU engine is at its bandwidth floor and 4096^3 fits no node's host at any
worker count (1487 GB at W=1 against a gb node's 1026). The only host that
holds the state is gb's: 1026 GB of LPDDR over two ~478 GiB sockets, 4x GB200
at 185 GiB each, GPUs 0-1 on socket 0 and 2-3 on socket 1.

- **The host is a byte store.** It holds the T9 state and nothing else that
  scales with N. It does not hold the coarse mesh and does not run the tile
  loop.
- **The four GPUs do every per-step phase** -- decode, tile force, kick,
  quantize, migrate, coarse paint, coarse solve -- with slabs streamed over
  C2C out of pinned host memory.
- **The coarse mesh is decomposed along x** across the four cards, which is
  also what the plane-factorized FFT needs.
- **The bar is the machine's**: gb's MaxWall is 12 h, so at K=40 a step has
  1080 s. Not picked.

The constraint that shapes every part of it: **host plumbing is 90% of a tile
as the engine stands** (5y: stage 0.41 + scatter 0.51 s against 0.094 s of
device work at P=576). A design that keeps any per-particle host pass keeps
the wall. That is the whole reason this is a new execution backend and not a
few kernels swapped into the existing one.

## 2. The budget (this record's own contribution)

`python -m inexor.plan --preset c-hero --backend device --host-gb 1026
--device-gb 199 --arena-frac 0.01`, at commit `63a4a17`.

| column | peak | against | ratio |
|---|---|---|---|
| host, the run | 886.4 GB | 1026 GB | 0.86x |
| host, the LOAD stage | **892.0 GB** | 1026 GB | **0.87x, BINDING** |
| per GPU (resident + worst phase) | 181.1 GB | 199 GB | **0.91x** |

**It fits, and the CPU column at the same config does not** (1381.7 GB, 1.35x).
That contrast is the design's premise and if it ever inverts the premise is
gone.

Both columns are tight, and three terms are worth naming:

1. **The slab window is 43.5 GB and is the largest single per-GPU term** --
   larger than the sharded coarse force mesh (25.8 GB). It is **18 x-slabs at
   c-hero, DERIVED from `layout.brick_span`**: a tile draws from 18 bricks per
   side (512/32 across the tile plus one brick of pad each side), so walking
   tiles in x-order needs 18 consecutive slabs live. It is a membership
   contract read as a residency requirement, not a tuning knob, and
   `choose_brick`'s `c | b_fine` condition is what makes the union exactly the
   padded box rather than a 1.7x superset.
2. **`coarse_solve` is the worst per-GPU phase at 60.2 GB, and 51.6 of it is
   the monolithic FFT's `coarse_kernel_build_f64` + `coarse_fft_workspace`.**
   The plane-factorized form replaces exactly those. So that rung buys per-card
   headroom, not only the 417 s/step it was scoped for.
3. **`slack + alloc_margin` is 129.9 GB of the host column.** The state prices
   at **11.56 B/p** here against the **10.54 B/p = 724 GB** the design's own
   arithmetic (and 5y/5z) assumes. The difference is entirely those two knobs
   at their defaults of 0.10 / 0.10. Which value is right is a decision, not an
   arithmetic fact: a brick that overflows its slack goes to the arena, and a
   full arena refuses the run.

`DEVICE_PLACEMENT` in `plan.py` is a **design assertion, not a reading of
code** -- no device executor exists. When one does, the table moves into it,
the way `MESH_PHASE` lives in `engine` because a term's phase is a property of
the code that allocates it. A mesh term with no entry raises.

## 3. What the budget found that was not known

**4096^3 ICs do not fit a gg node.** The out-of-core FFT at the 4096^3 IC grid
peaks at **556.5 GB** on the `derivative` policy (275.0 GB spectral array plus
a 275.0 GB copy) and **281.5 GB** on `forward` alone. A Vista gg node is
255.1 GB. Neither policy fits. The carried note that 4096^3 ICs "scale to ~12 h
on gg" prices a wall for a job that cannot run on that machine at all.

The choices are gb's 1026 GB host -- which puts a second ~12 h job on a queue
whose MaxWall is 12 h and which allows 2 jobs per user -- or staging the
spectral array to disk through `ooc_fft.StagedArray`, which buys a real
writeback cost the current path deliberately avoids. **Unpriced, undecided,
and it is the largest open piece of the deliverable.**

## 4. The per-particle host passes the build has to delete

Each of these is on the host today and each is per particle:

| phase | today | site |
|---|---|---|
| `tile_decode` | numpy `decode_bricks` | `state.py:1062`, `engine.py:1021` |
| coarse paint accumulate | host int64 mesh, `mesh[np.ix_] += sub` | `engine.py:749, 802` |
| coarse sub-block staging | host `np.take` per axis | `forces.py:1115` |
| coarse gather | jax, but syncs on a host bounds check mid-flight | `forces.py:1186` |
| kick | numpy f64 | `engine.py:1078` |
| quantize + per-brick scale | numpy run scan, `np.rint` | `engine.py:1088` |
| migrate insert | numpy always | `state.py:1439` |
| repack | numpy always | `state.py:1735` |

Reused rather than rewritten: `one_tile` (`forces.py:1231`, jitted, fixed `cap`
and `P^3` shapes so XLA holds one executable), `paint_tsc_int_subblock`
(`painting.py:317`), `eject_jax.eject_rows` (`eject_jax.py:138`), and the
staged pinned-host streaming mechanics of `scripts/v2_g4_gh_memory.py:174-212`
-- 2 GiB chunks, `device_put` per chunk, and **the per-chunk
`block_until_ready` is load-bearing**: without it XLA keeps every staged copy
alive and the device peak equals the whole working set.

## 5. What is NOT established

- **The four-way split.** Charged as an exact quarter in the budget above, and
  assumed perfect in 5y's 114 s/step. Neither is measured. The only reading is
  685 GB/s aggregate against 4x a single stream's 201, i.e. 0.85x, on the
  streaming leg alone, unpinned.
- **Every unmeasured term of the 114 s floor**: insert, repack, kick, host
  bookkeeping.
- **That 675 GiB streams.** The ladder reached 640 GiB on one GPU and stopped
  there by choice; the state is 5% above the top rung on a monotone trend.
- **XLA intra-jit scratch**, invisible to the budget on either column. Vista
  923139 lost ~79 GB inside a phase the CPU column priced at 30.
- **gb only.** Horizon's gb is a 240 GiB host. Say so beside any claim.

## 6. Owed

1. ~~The plane-factorized device FFT.~~ **BUILT AND MEASURED, sec. 7**:
   correct at 2048^3 (6.68e-06) and streaming (0.0034x of the spectrum). Its
   WALL is 78x the projection and the step floor moves with it. Still owed off
   it: the pass-2 access pattern (`pencil_batch`, the axis 974643 never varied)
   and a four-GPU reading.
2. A decision on `slack` / `alloc_margin` at c-hero (sec. 2, item 3).
3. The IC stage at 4096^3: a machine and a policy (sec. 3).
4. **The per-GPU budget's `coarse_solve` line re-read against sec. 7.** Sec. 2
   prices that phase at 60.2 GB from the MONOLITHIC form's terms
   (`coarse_kernel_build_f64` + `coarse_fft_workspace`, 51.6 of the 60.2). The
   factorized form does not allocate them -- it peaked at 117.5 MB -- so the
   per-GPU column is now conservative by ~50 GB and the 0.91x should fall once
   `DEVICE_PLACEMENT` learns the factorized terms. Not corrected here: the
   budget must not be edited to match a measurement of code that has not yet
   replaced the code it prices.

## 7. D1, Vista 974643 -- the factorized device FFT is CORRECT at 2048^3, and the coarse solve is 78x its projection

`f9f973f`, 2026-09-06, gb node c672-004 (4x GB200), 35:20 wall, all legs rc=0,
~0.6 SU. `scripts/v2_d1_device_fft_vista.sbatch`, one GPU. Card:
`runs/v2/d1_device_fft_gb.json`. jax 0.10.2.

### The two witnesses that decided it, both PASS

| n | roundtrip max\|d\|/rms | device peak | peak / spectrum |
|---|---|---|---|
| 512^3 | 4.053e-06 | 8.4 MB | 0.0156 |
| 1024^3 | 6.199e-06 | 29.4 MB | 0.0068 |
| **2048^3** | **6.676e-06** | **117.5 MB** | **0.0034** |

**The factorized form does NOT inherit the monolithic form's silent-wrong
class.** The monolithic `jnp.fft.rfftn` read 3.8e+3 at 1536^3 on this stack;
factorized, 2048^3 reads 6.68e-06 -- the same order as 512 and 1024, rising
mildly with n as accumulated rounding should. This is the result the rung
existed to get, and it is what keeps the device coarse solve alive.

**And it streams.** 117.5 MB of device high-water against a 34.4 GB spectrum.
The spectrum stayed on the host, which is the property the per-GPU budget of
sec. 2 assumes.

### The wall, and the correction to it

Every reading was repeatable to the third digit (forward 77.8/77.8/77.8).

| leg (one GB200 unless stated) | forward | inverse |
|---|---|---|
| device, plane_batch 1 / 4 / 16 | 77.78 / 77.53 / 77.76 s | 31.32 / 30.72 / 30.84 s |
| host out-of-core, same node, same job | 114.13 s | 69.98 s |

**The forward legs time the host RNG, and the number to quote is the inverse.**
A forward has exactly one thing an inverse does not: generating the field, 8.6e9
gaussians at 2048^3. Device forward-minus-inverse is 46.5 s; host
forward-minus-inverse is 44.1 s. Two different backends agree to 5% on an
excess that must be identical if it is the shared host generator, so the
transform costs **31.3 s on device and 70.0 s on the host**, and:

- **coarse solve = 4 transforms = ~125 s/step on ONE GPU, 11.6% of the 1080 s
  bar.** Device is **2.23x** the host form, not the 1.9x the readout printed
  from the contaminated legs.
- The gb probe projected **0.4 s/step** for this term, already divided by four
  GPUs -- 1.6 s on one. Measured is **78x that**. The projection was scaled
  `n^3 log n` device compute; this transform is not compute-bound.
- **The step floor moves.** 5y's 114 s/step carried the coarse solve at 0.4 s.
  At a perfect 4-way split it becomes ~145 s/step (0.13x of the bar); with no
  split, ~239 s/step (0.22x). It still fits, but the margin falls from 9.5x to
  4-7x and the coarse solve goes from a rounding error to the second-largest
  term after the tile force.

### Two defects in the instrument, both mine, both found by reading the card

1. **The forward leg timed field generation inside the transform** (above). The
   headline it printed, 169.7 s/step, is ~45 s of host RNG the engine will
   never do -- its forward's source is the painted density mesh, already in
   memory. Same class as 5y finding 5, where a host phase was scaled as device
   work; caught the same way, by reading the card rather than the summary.
2. **The ladder varied the dead axis.** plane_batch (pass 1) read
   77.78 / 77.53 / 77.76 -- identical digits -- while pencil_batch (pass 2)
   stayed pinned at 1 in all four legs. Pass 2 is the strided one: it reads
   `spec[:, y, :]` across the whole 34.4 GB spectrum 2048 times, with 16.8 MB
   between consecutive x rows. **The axis that could move the number is the one
   nothing varied.**

The same numpy pass-2 code on the laptop at n=512 reads 0.403 s at
pencil_batch=1 against 0.118 at 8. That is a small config on different
hardware and bounds the payoff on Grace in NEITHER direction; it says only that
the axis is alive, which is why the ladder is worth a job.

### What this does NOT establish

- **Any four-GPU number.** Every reading is one GB200. The 4-way split is
  assumed in the step-floor arithmetic above exactly as it was in 5y.
- **Where the 31.3 s goes.** Pass 2's access pattern is the leading suspect and
  it is a suspect, not a finding. The next candidate is the per-plane
  host<->device transfer.
- **f64.** All legs are f32, which is what the coarse mesh runs (D-v2-22).
- The 1024^3 leg reads 6.20e-06 against the 2.9e-06 that 972737's MONOLITHIC
  1024^3 read. Different factorization, different rounding; same order, so the
  stack has not moved. Do not quote them as the same measurement.

## 8. D1 follow-up, Vista 974807 -- generation measured, the pass-2 knob is weak on Grace, and the "clean" control was not clean

`0e4ae7b`, 2026-09-06, gb node **c672-002** (974643 ran on c672-004 -- see the
confound below), 19:00 wall, all legs rc=0, ~0.35 SU.
`scripts/v2_d1_device_fft2_vista.sbatch`, one GPU. Card:
`runs/v2/d1_device_fft_gb2.json`.

**Reproducibility receipt first:** the roundtrip residuals came back at
4.053e-06 (512^3) and 6.676e-06 (2048^3) -- the same digits as 974643, on a
different node. The correctness result of sec. 7 reproduces.

### Field generation, measured

**51.29 s** at 2048^3 (both reps 51.3). Sec. 7 inferred it at 46.5 / 44.1 s from
the device and host forward-minus-inverse gaps; the inference was right in shape
and ~10-15% low. So on real data:

| term | s | basis |
|---|---|---|
| forward transform | **26.5** | 974643's noise-source forward 77.78 minus this 51.29 |
| inverse transform | **31.3** | 974643's inverse, which generates no field at all |
| **coarse solve = 1 fwd + 3 inv** | **120.5 s/step, one GPU** | **11.2% of the 1080 s bar** |
| the same at a perfect 4-way split | 30.1 s/step | unmeasured split |

That leaves sec. 7's headline standing: ~120 rather than ~125 s/step, and the
step floor still moves from 114 to roughly 145 s/step at a 4-way split.

### The pass-2 knob is real but small, and the laptop did not transfer

| pencil_batch | 1 | 8 | 64 | 256 |
|---|---|---|---|---|
| per step (flat source) | 85.14 s | 90.51 | **76.62** | 76.77 |

1.11x from y=1 to y=64, 1.18x across the whole ladder, and y=8 is *slower* than
y=1. **The same numpy code on the laptop at n=512 read 3.4x.** It did not
transfer -- a small config on different hardware bounded the payoff in neither
direction and the record said so before the job ran. **Pass 2's strided access
is NOT where the transform's time goes on Grace.** The axis is now varied and
answered; it is worth ~11% and no more.

### The defect: the flat source was not an inert control
### [SUPERSEDED BY SEC. 9 -- the variable was the NODE, not the source. The
### flat source WAS inert (1.04x same-node). Read sec. 9 before this section.]

The flat-source legs read inverse **22.21 s** where 974643's noise-source legs
read **31.32 s** for the same operation, same library code (`git diff` over
`src/` between the two commits is empty), same n, slab and knobs. **The inverse
generates no field, so the change it was supposed to be blind to moved it by
1.41x.**

**Two things differ between those readings, not one: the DATA and the NODE**
(c672-002 against c672-004). The 1.41x is therefore UNATTRIBUTED and this
record does not assign it. What follows regardless is that the flat-source
numbers are not a clean measurement of the real transform, so every figure in
the table above is taken from the noise-source job -- and the leg that was
always clean is the INVERSE, which never generated a field in either job. The
flat source solved a problem the inverse leg did not have and introduced one it
did not have either.

### Where the time actually goes -- a suspect with arithmetic, not a finding

One transform moves **~137.5 GB** across host<->device. The inverse: pass 2
reads and writes the 34.4 GB spectrum (68.8), then the per-plane irfft2 reads
the spectrum and writes the 34.4 GB field (68.8). The forward is the same total
the other way round. At 31.3 s that is **4.4 GB/s effective, against the 201
GB/s this same hardware streams from PINNED host memory** (974476, sec. 5z) --
a **46x** gap.

> CORRECTION (same day): this paragraph first read ~275 GB and 8.8 GB/s, both
> exactly 2x too large -- the expression counted the forward AND the inverse as
> one transform. The gap is 46x, not the 23x first written here. The conclusion
> is unchanged and stronger; the arithmetic was wrong and is corrected rather
> than quietly replaced.

**This project has already measured that tax once**: the gb probe's eject
kernel read 14 ms on device against 0.528 s end to end through pageable
transfers, 38x (5y). Every transfer in this FFT path is pageable -- numpy in,
`jnp.asarray`, `np.asarray` out. Pinning the spectrum is the obvious fix and it
is something the design needs anyway, since the host state is pinned by
construction.

It is a SUSPECT. Two other candidates are untested: the 2048 per-plane dispatch
round trips, and pass 2's `np.ascontiguousarray` host copy (which the ladder
above bounds at ~11%, so it is not the whole story either way).

### Owed off this section

1. The pinned-buffer arm. If it recovers even half of the 23x, the coarse solve
   stops being the second-largest term in the step floor.
2. A four-GPU reading. Every number in secs. 7-8 is one GB200.
3. Nothing on `pencil_batch`: the axis is answered.

## 9. D1 attribution, Vista 975164 -- 91% of the transform is the bus, pinning is 6.8x, and sec. 8's confound resolves to the NODE

`38cf5bf`, 2026-09-06, gb node **c672-002** (the same node as 974807 -- that is
what resolves sec. 8), 11:36 wall, all legs rc=0, ~0.2 SU.
`scripts/v2_d1_device_fft3_vista.sbatch`, one GPU. Card:
`runs/v2/d1_device_fft_gb3.json`. 512^3 roundtrip reproduced at 4.053e-06 for
the third time.

### The transfer A/B, same node, same job

| mode | bytes | wall | rate | memory kind used |
|---|---|---|---|---|
| pageable (numpy -> `jnp.asarray` -> `np.asarray`) | 137.5 GB | 21.07 s | **6.5 GB/s** | numpy |
| pinned (`pinned_host` memory space) | 137.5 GB | 3.10 s | **44.4 GB/s** | `pinned_host` |

Device kinds available: `['device', 'pinned_host']` -- the pinned leg used the
kind it claimed, so it is a reading and not a silent fallback.

**91% of the transform is the bus.** The inverse on this node reads 23.06 s and
the identical traffic in identical 16.8 MB units reads 21.07 s pageable. The
suspect of sec. 8 is CONFIRMED: this factorization is not compute-bound, it is
not bound by pass 2's stride (~11%), it is bound by pageable host<->device
copies.

**Pinning is 6.8x on the bus** and projects the transform from 23.06 to
**~5.1 s (4.5x)**, i.e. a coarse solve of **~20 s/step on one GPU, 1.9% of the
bar** -- back to a minor term. Projected, not measured: the pinned rate is a
microbenchmark of the traffic, and a pinned FFT path has not been built.

**The UNIT is a second constraint, not just the kind.** Pinned reads 44.4 GB/s
at a 16.8 MB plane against the **201 GB/s the same hardware reached at 2 GiB
chunks** (5z) -- 378 us per transfer over 8192 of them. So a device phase that
pins its buffers but moves them a plane at a time still leaves 4.5x on the
table. **Both constraints are D2's, not D1's:** every device phase pins, and
every device phase moves in the largest unit its algorithm allows.

### Sec. 8's 1.41x was the NODE, and my suspicion was wrong

Sec. 8 recorded that the flat-source inverse (22.21 s) and the noise-source
inverse (31.32 s) differed by 1.41x on an operation that generates no field,
named the two variables that moved together -- the DATA and the NODE -- and
declined to assign it. This job holds the node fixed:

| comparison | held fixed | varied | ratio |
|---|---|---|---|
| 22.21 (974807 flat) vs 23.06 (975164 noise) | node c672-002 | the data | **1.04x** |
| 31.32 (974643 noise) vs 23.06 (975164 noise) | the data | node c672-004 -> c672-002 | **1.36x** |

**The data barely matters and the node is the whole effect.** The flat source
was an inert control after all; the variable I failed to hold was the machine.
Sec. 8's suspicion that "the flat source was not clean" is **withdrawn** -- it
was clean, and the sentence in sec. 8 saying the flat legs "are not a clean
measurement of the real transform" is wrong. Left standing there with this
correction pointing at it rather than edited away, because the mistake is the
transferable part: two jobs differing in one deliberate knob also differed in a
node nobody chose, and only a same-node arm could tell them apart.

**1.36x of node-to-node spread on identical work is itself a result.** Any
single-node reading of this transform carries it, including every number in
secs. 7 and 8.

### The coarse solve, restated with the node named

On c672-002, noise source, generation (52.02 s) subtracted:

| pencil_batch | forward | inverse | coarse solve = 1 fwd + 3 inv |
|---|---|---|---|
| 1 | 17.15 s | 23.06 s | **86.3 s/step, 8.0% of the bar** |
| 64 | 14.13 s | 19.70 s | **73.2 s/step, 6.8% of the bar** |

Sec. 8's 120.5 s/step was the same measurement on the slower node at y=1. **The
honest range for the coarse solve on one GPU is 73-120 s/step, 7-11% of the
bar, spanning node and knob** -- and ~20 s/step if pinned. The pencil knob
reproduces at 1.10x here against 974807's 1.11x.

### What D1 has established, and what closes it

Correct at 2048^3 (three jobs, two nodes, 4.053e-06 / 6.676e-06 to the digit).
Streams (0.0034x of the spectrum resident). Costs 73-120 s/step on one GPU as
built, ~20 s/step pinned, against a 1080 s bar. The rung is DONE.

Owed forward into D2, not into D1:
1. **Pin every device buffer.** Established at 6.8x on this phase.
2. **Move in the largest unit the algorithm allows.** 44.4 GB/s at a plane
   against 201 GB/s at 2 GiB; the slab window (2.4 GB) is the natural unit.
3. **A four-GPU reading.** Every number in secs. 7-9 is one GB200.
4. **Name the node on every future reading of this class.** 1.36x.

## 10. D5 pulled forward, Vista 991162 + 991236 -- the transform splits 2.8-3.0x across four GPUs, not 4x

`6760488` and `38d4bec`, 2026-09-11/12, gb nodes **c672-010** and **c672-016**,
11:09 and 12:56 wall, all legs rc=0, **~0.4 SU total**.
`scripts/v2_d5_four_gpu_vista.sbatch`, four GB200s, all four visible in every
leg. Cards: `runs/v2/d5_four_gpu_gb.json`. 512^3 roundtrip reproduced at
4.053e-06 for the fourth and fifth time.

Pulled ahead of D2c at JC's direction: D0's budget and 5y's 114 s/step floor
both rest on a 4-way split that had never been measured, and D2c..D2e would
have been built on top of it.

### The measurement, and why it is a contention measurement

The factorization is embarrassingly parallel with **no inter-device
communication** -- pass 1 independent per plane, pass 2 per y-pencil-plane,
spectrum host-resident throughout -- so nothing in the algorithm stops a 4x
split. But D1 measured this path at 91% host bus, and the bus is the one
resource four GPUs share. Strong scaling at fixed total work, widths 1/2/4, all
in ONE job on ONE node.

**The identity arm gates everything else and passed at every width**: the W=4
spectrum is BITWISE equal to the W=1 spectrum at 2048^3, forward and inverse
(n_diff 0 of 8.6e9 f32 words). Partitioning a loop cannot change a value, so
this is an identity rather than a tolerance; without it a wall measured at W=4
would be a wall for a different transform. Every leg also records which devices
did work, and a width served by fewer devices fails the job rather than
printing a number -- a W=4 leg that quietly ran on device 0 is
indistinguishable from "it did not scale".

### The split, c672-016 (991236, the corrected job)

| | W=1 | W=2 | W=4 | T(1)/T(4) |
|---|---|---|---|---|
| forward | 20.79 s | 11.64 | 6.86 | 3.03x |
| inverse | 21.68 s | 12.06 | 7.13 | 3.04x |
| inverse, pass 2 | 7.63 s | 4.53 | 2.60 | **2.93x** |
| inverse, pass 1 | 13.37 s | 7.34 | 4.11 | **3.25x** |
| host spectrum copy | 2.97 s | 3.21 | 3.04 | **1.00x** |
| coarse solve (1 fwd + 3 inv) | **85.8 s/step** | 47.8 | **28.2 s/step** | **3.04x** |
| % of the 1080 s bar | 7.9% | 4.4% | **2.6%** | |

991162 on c672-010 read the same quantities 1.28x faster throughout (67.1 /
37.9 / 23.8 s/step) for a split of **2.80x**. So the split is **2.8-3.0x on two
nodes**, and the node spread that D1 measured at 1.36x reproduces at 1.28x on a
third and fourth node.

**The exact-quarter charge is wrong.** Any term in 5y's 114 s/step floor priced
by dividing a single-card time by four should be divided by ~2.9 instead.
D0's per-GPU column is NOT affected: that column is memory, and each GPU still
holds its quarter of the working set however the wall splits.

**But the absolute number moves the right way anyway.** 28.2 s/step against
D1's 73-120 s/step on one GPU. The coarse solve stops being a large term -- for
a different reason than the design assumed.

### The sub-linearity is a shared resource, not a serial section

An Amdahl fit on the inverse gives 2.28 s serial against 19.4 s parallel and
predicts W=2 to 0.7% on a point it was not fitted to. My first reading of that
named a suspect with arithmetic: pass 2's strided host gather, ~103 GB of host
traffic per pass at 2048^3.

**The per-pass split REFUTES it.** Pass 2 scales 2.93x; a 2.4 s serial section
inside it would have capped it at 2.06x. Fitting each pass separately puts
~0.9 s in pass 2 and ~1.0 s in pass 1 -- spread across both roughly in
proportion to their size, which is the signature of a shared resource (four
threads do not get four times the host memory bandwidth) and not of
unsplittable code.

**Consequence: there is no serial section to go delete.** The missing 1.0-1.2x
is bought by moving less host traffic, not by restructuring the loop. This is
why the per-pass instrumentation was added rather than the fit being trusted.

### The transfer proxy, and two defects in it that were mine

| mode | W=1 | W=2 | W=4 threads | W=4 processes | T(1)/T(4) |
|---|---|---|---|---|---|
| pageable | 3.8 GB/s | 7.1 | 13.9 | 14.2 | 3.69x |
| pinned | 37.3 GB/s | 71.3 | 108.9 | **139.5** | 2.92x |

The first version of this arm got BOTH modes wrong and **neither error was
visible in the rate it printed**. Pageable did a second `device_put` where D1
reused the device array, so it moved three plane-crossings per unit instead of
two and read 1.5x low. Pinned asked `device_put` to send an already-pinned
array back to `pinned_host`, which returns the IDENTICAL OBJECT and moves
nothing, so the D2H leg never ran and the rate read 2x high. Both errors were
width-independent, so 991162's speedups survived them; its absolute rates did
not.

Corrected, **the pinned proxy agrees with D1**: 37.3 GB/s here against 44.4 on
c672-002, a 1.19x gap inside the node spread. That agreement is the evidence
the fix took. A prior claim that the corrected PAGEABLE figure matched D1's 6.5
GB/s is **withdrawn** -- that arithmetic assumed H2D and D2H cost the same,
which this job contradicts; normalized for the node the corrected reading is
~4.8 GB/s against D1's 6.5, a node-sized gap rather than a match.

`_crossing_receipt` now runs before any timing and FAILS the leg: it checks the
D2H leg returned a distinct object in the expected memory kind, which is
exactly what the pinned no-op could not have satisfied. A transfer that
silently did not happen looks identical to a fast one, and my own instrument is
not exempt from that.

### Pinning is the remaining lever, and it is large

Pinned buys **7.8x on the bus at W=4** (108.9 against 13.9 GB/s). Carrying D1's
attribution that the transform is 91% bus, a pinned transform on four GPUs
projects to **~5.8 s/step, 0.5% of the bar**. PROJECTED, on a 91% measured at
W=1 pageable and not remeasured at width -- and no pinned FFT path exists,
because doing it honestly needs the slab window itself allocated in pinned host
memory, which is pipeline work.

**If that lands, the 34.4 GB host spectrum copy becomes the largest term in the
coarse solve**: flat at ~3.0 s across every width (it is serial host work), and
at three inverses per step that is ~9.1 s/step against a ~5.8 s/step transform.
Whether the solve can avoid making it is worth settling before anything else in
this phase is optimized.

The driver control, with the bytes finally right: pageable **1.02x** (threads
and processes identical), pinned **1.28x** in favour of processes. The GIL
costs nothing while transfers are slow and ~28% once they are fast.

### What D5 has established, and what is owed

Established: the split is 2.8-3.0x not 4x, on two nodes, under a bitwise
identity gate; the shortfall is shared host bandwidth in both passes rather
than a serial section; pinning is worth 7.8x on the bus at width.

Owed forward:
1. **Does an explicit copy into a reusable pinned buffer beat the driver's
   pageable path?** The engine's host buffers are numpy and JAX's pinned arrays
   are immutable, so the allocation cannot simply be swapped. The pageable path
   already pays a host copy inside the driver; the question is whether the
   driver is doing something worse than a plain memcpy. This decides the route
   and is one cheap arm.
2. The 91% bus attribution remeasured AT WIDTH, before the ~5.8 s/step
   projection is quoted as anything but a projection.
3. Whether the coarse solve can avoid the 34.4 GB spectrum copy.
4. Name the node on every reading of this class. Four nodes now, 1.28-1.36x.

## 11. D5b + D5c -- pinning is CLOSED for this engine, and D1's first inherited constraint is refuted

`f8edf30` / `7567b60` / `9f06942`, 2026-09-12, gb nodes c672-005, c672-008 and
c672-004. Jobs 992589 (FAILED, correctly), 992621 and 992680, ~0.7 SU.
Cards: `runs/v2/d5_four_gpu_gb.json`, `runs/v2/d5c_host_register_gb.json`.

D1 recorded two constraints "every later phase inherits": **pin every host
buffer that crosses**, and **move in the largest unit the algorithm allows**.
The first is now REFUTED for this engine. The second is untouched, and is
therefore the whole of the remaining opportunity.

### D5b: staging each crossing is SLOWER than ordinary numpy

| policy | W=1 | W=4 |
|---|---|---|
| pageable | **83.0 s/step** | **27.3 s/step** |
| staged (via `pinned_host`) | 115.1 s/step | 33.5 s/step |
| staged / pageable | **1.39x slower** | **1.23x slower** |

Uniform across both passes at W=4 (+23% each), so it is the crossing itself and
not one pass's access pattern. Bitwise-gated: the policy changes where a buffer
lives, not what is computed.

**Why.** JAX arrays are immutable, so there is no reusable pinned buffer to
copy into: every crossing allocates a fresh one and pays a kernel call to
page-lock it. The proxy's pinned arm pins ONCE outside its timed loop and reads
36.1 GB/s; the staged transform pins on every one of 8192 crossings and lands
below pageable. Same memory kind, same job, same node, opposite results, and
the only difference is amortized versus per-crossing pinning.

**So the 7.8x memory-kind ceiling is real but belongs only to a buffer pinned
once and reused.**

### D5c: XLA does not take the fast path for registered memory

The surviving route was memory that is page-locked AND numpy-writable, which
JAX cannot express and `cudaHostRegister` can, by locking pages that already
exist. c672-004, page size 65536, `memlock` unlimited, libcudart via ctypes
with no dependency added.

| flags | pageable | registered | ceiling | speedup | of the way |
|---|---|---|---|---|---|
| portable | 7.2 GB/s | 7.4 | 63.3 | **1.03x** | **0%** |
| default | 7.1 GB/s | 7.3 | 67.2 | **1.03x** | **0%** |

Confirmed end to end: the real 2048^3 inverse reads 21.68 s unregistered and
21.71 s registered, **1.00x**, with both passes flat. Three gates agree.

**The allocation side was never the problem.** Registration runs at 187-284
GB/s (128 GB in 0.45 s; the design's ~724 GB host state would page-lock in
~3 s), and the compute node permits it without limit. XLA simply does not ask
whether a host pointer is already locked -- it stages every one of them.

### What this closes, and what it does not

**CLOSED: pinning as a line of work for this engine.** Not argued, measured, on
three independent gates. The 7.8x is reachable only by a JAX-owned `pinned_host`
buffer, which is immutable, and the engine's state is numpy written in place by
construction (T9, the arena, migrate, repack). From numpy-resident state the
ceiling is unreachable, and the two ways of trying both cost more than they
save.

**NOT closed, and stated narrowly:** this is a property of THIS engine's
architecture against THIS jax/XLA, not a general claim about pinning. An engine
whose host state were JAX-resident and functionally updated would reach the
ceiling; that is a different engine, and the design study's own rule applies --
name the workflow before building the capability.

**What survives as the remaining lever.** D5 established that the four-GPU
shortfall is shared host bandwidth spread across both passes rather than a
serial section, so there is nothing to delete. D5c now removes the bus as
something that can be made faster. Together those leave exactly one direction:
**move less host traffic**, which is D1's second constraint and is untouched.
The nearest concrete target is the 34.4 GB spectrum copy -- flat ~3.0 s at
every width, ~9.1 s/step across three inverses, and the largest single host
term now that pinning cannot shrink the others.

**The coarse solve is not a problem either way**: 27.3 s/step at W=4, 2.5% of
the 1080 s bar, against D1's 73-120 s/step on one GPU.

### Owed

1. Whether the solve can avoid the spectrum copy (it is now the biggest host
   term, and the only one with an obvious fix).
2. The synthetic traffic proxy remains 1.83x slower than the inverse whose
   traffic it replicates, and the commitment suspect was REFUTED by a control
   leg (84.2 vs 83.0 s/step, 1.01x). Recorded as unexplained. It is off the
   decision path and is not worth a job.
3. Nothing further on pinning.

## 12. D2d, Vista 993139 -- the device coarse paint is bitwise across backends, and a whole x-slab chunk does not fit a card

`573d55f`, 2026-09-12, gb node c672-004, one GB200 read (`jax.devices()[0]`),
COMPLETED rc=0 in 8:08, ~0.14 SU. Script `scripts/v2_d2d_device_paint.py`,
sbatch `v2_d2d_device_paint_vista.sbatch`; cards
`runs/v2/d2d_device_paint_{gb,smoke}.json` (force-added). Every arm a fresh
subprocess, so each device high-water is its own (baseline `bytes_in_use` 0).

**What was built** (`6c3cd17`, `7882701`, `73ee946`): `device/paint.py` decodes
a chunk of consecutive bricks from one contiguous slot slice plus its arena
residents, checks stencil containment as device scalars and paints the integer
sub-block on the device. The accumulator is a seam; this job used the host
int64 mesh. Also found and fixed on the way: `device.decode`'s bucket search
built a (rows, 512) table -- ~1.1 TB for an x-slab chunk, ~100 GB for a 4096^3
tile -- now one searchsorted, O(rows).

### Identity: PASS on every gate

- **Cross-backend.** A `JAX_PLATFORMS=cpu` process hashed
  `engine.coarse_delta_streamed` at cdev (256^3); the GPU's
  `coarse_delta_device` hash-equals it for a plain state and for one with
  **61,879 arena residents**. The laptop suite could only show device == host on
  one backend.
- **Per chunk.** Each chunk's block equals host `decode_bricks` through the same
  kernel, at all three sizes below. The smoke leg (32^3, on the GPU) passed the
  same gates first.

### Memory: a clean per-row rate, 2.2x the traced floor

Chunks built at production brick geometry (4096 rows, 512 buckets per brick),
so their rows are 4096^3 chunks' rows:

| chunk | real rows | padded rows | device peak | B per padded row |
|---|---|---|---|---|
| 1/16 x-slab (512^3, 4096 bricks) | 16.8M | 16.8M | 4.56 GB | 272.0 |
| 1/4 x-slab (512^3, 16384 bricks) | 67.1M | 84.6M | 22.39 GB | 264.8 |
| whole x-slab (1024^3, 65536 bricks) | 268.4M | 338.2M | 90.03 GB | 266.2 |

Flat over 20x in rows. The traced program under last-use freeing reads 121 B/row
(what `PAINT_CHUNK_B_PER_ROW` charged until this job, plus 3 for the window);
the eager program as it runs holds 2.2x that. **The planner now charges the
measured 266** (`PAINT_CHUNK_B_PER_ROW`), and keeps 121 as
`PAINT_CHUNK_TRACED_B_PER_ROW`, the laptop gate on program growth.

**At 4096^3 (`inexor.plan --backend device`, measured rate):**

| chunk | paint on a card | per-card total | verdict |
|---|---|---|---|
| whole x-slab | 90.0 GB | 228.4 GB | **DOES NOT FIT, 1.15x** |
| 1/4 x-slab | 22.5 GB | 160.9 GB | fits, 0.81x |
| 1/16 x-slab | 5.6 GB | 144.0 GB | fits, 0.72x |

By the rule pre-registered in the sbatch (the reading picks the chunk size),
**the device paint's default is now a quarter of an x-slab**
(`device.paint.default_chunk_bricks`). Host column unchanged at 0.93x.

### Time: the paint is now the largest measured term in the step

| chunk | median s/chunk (3 reps) | host window prep | ns per real row |
|---|---|---|---|
| 1/16 x-slab | 0.234 | 0.016 | 13.9 |
| 1/4 x-slab | 0.798 | 0.034 | 11.9 |
| whole x-slab | 2.590 | 0.165 | 9.6 |

**[CORRECTION, sec. 13] These chunks match 4096^3 chunks in ROWS, not in block
shape.** `_chunk_cuboid` tiles a chunk by the brick grid, so at 4096^3 (256
bricks per side) an x-slab chunk paints an 11 x 2048 x 2048 cell block, thin in
x, while the test's whole x-slab (1024^3, 64 per side) paints 131 x 512 x 512.
Memory follows rows and projects cleanly; chunk TIME may depend on the block's
shape, so the per-step times below are arithmetic that assumes it does not.

Warm (first) calls 5.2-7.7 s: eager per-op compilation, once per shape per
process. Carried to a 4096^3 step, ONE card, serial: 958 / 817 / 663 s at
1/16 / 1/4 / whole. **At the quarter-slab default that is ~204 s/step on four
cards at an exact quarter split and ~282 s at D5's 2.9x** -- 19-26% of the
1080 s bar, against the coarse solve's 27.3 s/step at W=4. Both four-card
figures are arithmetic on a one-card reading; the paint's own split is not
measured.

**The gb probe projected this phase at 12.6 s/step** (sec. 5y: 0.196 s per
268M-row slab for the sub-block paint alone, split four ways); a whole slab
measures 2.59 s here. So most of the chunk's time is plausibly the decode and
containment, not the paint kernel -- an INFERENCE across two jobs and two nodes,
not attributed inside this one.

### Environment note

The gpu env prints `cuBLAS < 13.2 (120902 found) has a known issue ... executing
a cuBLAS kernel concurrently with another kernel (e.g. on another stream) can
lead to silent data corruption.` This job is single-stream and its gates are
bitwise, so it is not implicated. Not checked: whether any device path in this
engine calls cuBLAS (the transforms are cuFFT, the paint and gather are
scatter/gather), which matters for every multi-stream layout, D5's threads
included.

### Owed

1. **Jit the chunk, gated bitwise BEFORE the jitted numbers are read.** The
   traced floor (121 against 266 eager) says memory has room to fall; how much
   wall moves is not predicted here.
2. Attribute the 2.59 s inside one job: decode vs containment vs paint vs
   readback.
3. Whether cuBLAS is on any device path (environment note).
4. Where the coarse mesh lives (host vs sharded on cards): still open, and the
   accumulator seam is where either lands.

## 13. D2d jit, Vista 993294 -- the jitted paint is bitwise on a GB200 and holds 4.3x less; it is the default

`9556ffc`, 2026-09-12, gb node c672-016, one GB200 read, COMPLETED rc=0 in
15:19, ~0.26 SU. Sbatch `v2_d2d_jit_paint_vista.sbatch`; cards
`runs/v2/d2d_device_paint_{jit_gb,jitsmoke}.json` (force-added). Eager and jit
arms at every chunk size in ONE job, each its own subprocess.

**What was built** (`9556ffc`): `coarse_delta_device(jit=True)` compiles decode
+ containment + integer paint into one program per step. Chunk inputs are padded
to fixed per-step shapes (`step_shapes`) and the sub-block origin is traced;
`decode_rows` became an eager wrapper over a pure-jnp `decode_core`, and
`paint_tsc_int_subblock` takes its origin as an int array.

### Identity: PASS on every gate, so jit is adopted

- The GPU's JITTED density hash-equals a CPU-only host engine at cdev (256^3),
  plain and with 61,879 arena residents -- as does the GPU eager density, and
  the CPU's own jitted density. The risk this gated was XLA fusing the TSC
  weight arithmetic and moving a rounded integer weight; it did not, at tens of
  millions of rows x 27 corners.
- Every jitted chunk block (up to 268M rows) equals host `decode_bricks`
  through the eager kernel. Each jit arm traced exactly once.

**`coarse_delta_device` now defaults to `jit=True`**, per the sbatch's
pre-registration; `jit=False` is the eager path and keeps its own tests.

### Memory and time, same job and node

| chunk rows (padded) | eager B/row | jit B/row | eager s/chunk | jit s/chunk | jit host prep |
|---|---|---|---|---|---|
| 16.8M (16.8M) | 293.0 | 72.1 | 0.216 | 0.038 | 0.019 |
| 67.1M (84.6M) | 269.7 | 62.2 | 0.804 | 0.459 | 0.051 |
| 268.4M (338.2M) | 266.0 | 62.2 | 2.707 | 1.829 | 0.243 |

The eager arms reproduce sec. 12 on a different node (269.7 vs 264.8 and 266.0
vs 266.2 B/row; 0.804 vs 0.798 and 2.707 vs 2.590 s). **Jit holds 4.3x less on
the card**, and below the traced program's 121 B/row -- that reading has no
operator fusion, so it is not a floor for the compiled program
(`PAINT_CHUNK_TRACED_B_PER_ROW` is renamed in meaning, not value: it still gates
program growth on the laptop). **The planner now charges jit's 72**
(`PAINT_CHUNK_B_PER_ROW`; eager's 266 kept as `PAINT_CHUNK_EAGER_B_PER_ROW`).

**At 4096^3, per card:**

| chunk | eager | jit |
|---|---|---|
| whole x-slab | 1.15x, does not fit | 0.82x |
| 1/4 x-slab (default) | 0.81x | 0.73x |
| 1/16 x-slab | 0.72x | 0.70x |

**Time, stated with its caveat.** Carried to a 4096^3 step on ONE card as
chunks x s/chunk: jit 156 / 470 / 468 s at 1/16 / 1/4 / whole, eager 885 / 823 /
693 s. On four cards that is ~120-160 s at the quarter-slab default (exact
quarter vs D5's 2.9x) against the 1080 s bar. **None of this is measured at
4096^3's block shape** (sec. 12 correction): the test blocks are thick in x
(35-131 cells) where every 4096^3 chunk block is 11 cells thick.

**UNEXPLAINED: jit's per-row time is not flat.** The device part of a chunk
(total minus host prep) is 1.1 ns per padded row at 16.8M and 4.8 / 4.7 at
84.6M / 338M, where eager's device part is 12.0 / 9.1 / 7.5. The two larger
arms agree across two different states (512^3 and 1024^3) and a 4x difference
in block size, so block size alone does not explain it; the 16.8M arm is the
only one whose padded row count is a power of two (2^24), carrying 16 rows of
padding against 26% at the other two. Not attributed. It matters because it decides the chunk size: taken at
face value the 1/16 chunk is 3x faster per step.

**The default chunk stays a quarter-slab** until the time is measured at the
block shape 4096^3 actually paints.

### Owed

1. **Chunk time at 4096^3's block shape.** `_chunk_cuboid` only makes thin-in-x
   blocks when a chunk is at most one brick plane thick, i.e. at `chunk_bricks
   <= nb^2`. A 1024^3 state (64 bricks per side) gives thin blocks at 4096
   (1 x 64 x 64 bricks = 16.8M rows) and 1024 bricks (4.2M), so the same
   16.8M rows can be timed thin (1024^3) and thick (512^3, sec. 13 table) in
   one job. Rows and shape cannot both match 4096^3 below 4096^3.
2. **The 1/16 per-row anomaly**: pad at an exact power of two vs padded, same
   rows, one job.
3. Carried: the paint's own four-card split; cuBLAS on device paths (sec. 12);
   where the coarse mesh lives.

## 14. D2d chunk shape, Vista 993350 -- padding, not block shape, sets the jitted chunk's time; and the host plan copies the whole index

`a93b3be`, 2026-09-12, gb node c672-004, COMPLETED rc=0 in 9:14, ~0.15 SU.
Sbatch `v2_d2d_chunk_shape_vista.sbatch`; cards
`runs/v2/d2d_device_paint_{shape_gb,shapesmoke}.json` (force-added). Seven
jitted arms, one node, each its own subprocess; the pad override is an
instrument. Device time = median chunk time minus the host window prep
(timed once, separately).

| state | block (cells) | rows | pad | device s | ns / padded row | ns / real row | B / padded row |
|---|---|---|---|---|---|---|---|
| 1024^3 | 11x512x512 | 16.78M | +26.0% | 0.099 | 4.69 | 5.91 | 71.1 |
| 512^3 | 35x256x256 | 16.78M | +0.0% (2^24) | 0.019 | 1.13 | 1.13 | 72.1 |
| 512^3 | 35x256x256 | 16.78M | +26.0% | 0.099 | 4.67 | 5.89 | 71.0 |
| 512^3 | 35x256x256 | 16.78M | +100% (2^25) | 0.328 | 9.78 | 19.56 | 113.5 |
| 512^3 | 131x256x256 | 67.11M | +26.0% | 0.395 | 4.67 | 5.88 | 62.2 |
| 512^3 | 131x256x256 | 67.11M | +0.0% | 0.074 | 1.11 | 1.11 | 64.0 |
| 512^3 | 131x256x256 | 67.11M | +100% (2^27) | 1.302 | 9.70 | 19.40 | 103.5 |

Every block equals host decode through the eager kernel; every arm traced once.
The two repeats of 993294's arms land within 3% of it on a different node.

### What it establishes

- **Block shape does not move the device time.** 16.78M rows into the thin
  block every 4096^3 chunk has (11 cells in x, from a 1024^3 state) and into a
  thick one, at the same ~21.14M pad: 0.099 s both. The sec. 12 correction's
  caveat is lifted for time at this row count; it is one row count.
- **Power of two is not the variable**: the 2^25 and 2^27 pads are the slowest
  arms.
- **Padding is.** Unpadded, the rate is 1.11-1.13 ns per row at both sizes.
  26% padding costs ~5.3x the chunk's device time and 100% ~17x, identically at
  16.8M and 67M rows, so on the card a padded row costs far more than a real
  one. **Sec. 13's "per-row time is not flat" was this**: its fast arm was the
  only one whose ladder happened to pad by 16 rows.
- **Memory**: 62-72 B per padded row at up to 26% padding, so the planner's 72
  holds for the ladder's range; 104-114 at 2x, which is not an operating point.

**The mechanism is NOT attributed.** Every padded row scatter-adds a zero
weight into block cell 0 on all 27 corners, so the paint's scatter carries one
heavily duplicated index -- the same class as the tiled force's zero-filled
padding (`eba91ab`, 2.2x there) -- and a duplicate-index scatter on the GPU is
the leading suspect. **The laptop cannot attribute it**: a jitted CPU paint at
4M rows costs 24-32 ns per padded row, flat across 0 / 26 / 100% padding, and
spreading the dead rows over the block (still weight zero, bitwise the shipped
kernel's output) moves it 5-9%. The CPU shows no padding penalty, so the test
has to run on a card.

### Host prep scales with the STATE, and that one is attributed

Host window prep reads 0.091 s on the 1024^3 state against 0.019 s on 512^3 for
the same 4096 bricks. `device.decode.tile_decode_plan` opened with
`np.asarray(st.occupancy, dtype=np.int64).reshape(-1, p3)[bricks]`, which widens
the ENTIRE uint32 index to int64 before slicing: measured on the laptop, the
plan's host peak is 0.33 / 2.17 MB at 64^3 / 128^3 for the same 16 bricks,
against 8 B x buckets = 0.26 / 2.10 MB. At 4096^3 that is **68.7 GB per chunk
call**. D2a's "O(bricks + arena), never O(rows)" was true of the plan's output
and false of its work, and its test gated sizes only. Fixed by slicing first;
gated by host allocation against the whole index
(`test_the_host_plan_does_not_copy_the_whole_index`).

### Owed

1. **The padding cost on a card, attributed**: the same 16.8M-row chunk at
   26% and 100% padding with dead rows sent to cell 0 (as shipped) and spread
   over the block, bitwise identical by construction. If spreading closes it,
   ship it behind that gate; if not, the cost is elsewhere in the padded rows.
2. Host prep re-measured after the index fix.
3. Carried: the paint's four-card split; cuBLAS on device paths; where the
   coarse mesh lives.

## 15. D2d dead rows, Vista 993600 -- the padding cost is the duplicated scatter index; spreading the dead rows removes it

`dbb852b`, 2026-09-12, gb node c672-012, COMPLETED rc=0 in 3:54, ~0.1 SU.
Sbatch `v2_d2d_dead_rows_vista.sbatch`; cards
`runs/v2/d2d_device_paint_{dead_gb,deadsmoke}.json` (force-added). Five jitted
arms on the 512^3 state, 4096-brick chunk (16.78M rows, block 35x256x256), one
node, each its own subprocess. Device time = median of 3 chunk times minus the
host window prep, as in sec. 14.

| pad | dead rows | device s | ns / padded row | ns / real row | vs unpadded | B / padded row |
|---|---|---|---|---|---|---|
| +0.0% (2^24) | cell0 | 0.0190 | 1.13 | 1.13 | 1.00x | 72.1 |
| +26.0% | cell0 | 0.0986 | 4.67 | 5.88 | 5.20x | 74.2 |
| +26.0% | spread | 0.0219 | 1.04 | 1.31 | 1.15x | 71.0 |
| +100% (2^25) | cell0 | 0.3533 | 10.53 | 21.06 | 18.6x | 113.5 |
| +100% (2^25) | spread | 0.0255 | 0.76 | 1.52 | 1.34x | 69.5 |

Every block equals host decode; every arm traced once.

### What it establishes

- **The pre-registered first reading fired.** Spreading the dead rows brings a
  padded row to 0.76-1.04 ns, at or under the unpadded 1.13, so the cost was the
  one heavily duplicated scatter index (every dead row into block cell 0 on all
  27 corners). Spread is 4.5x faster than cell0 at 26% padding and 13.9x at 100%.
- **The cell0 arms reproduce sec. 14**: 0.0190 / 0.0986 against 0.019 / 0.099;
  at 100% 0.353 against 0.328, 1.08x, inside node spread.
- **Memory moves too.** At 100% padding cell0 holds 113.5 B per padded row and
  spread 69.5, back under the planner's 72. So sec. 14's "104-114 at 2x" was also
  the duplicated index, not a property of padding.
- Padding is not free under spread (1.34x the unpadded device time for twice the
  rows), but a padded row now costs less than a real one.

**Host window prep** read 0.0096-0.0097 s per arm against sec. 14's 0.019-0.020
on the same 512^3 state and chunk, after the index fix (`bc7b43b`). Different
node, and the 512^3 state is not where the fix bites; the 1024^3 re-read is
still owed.

### Owed

1. ~~Make "spread" the default~~ DONE: `painting.paint_tsc_int_subblock` and
   every `device.paint` entry point default to "spread", behind
   `test_spreading_the_dead_rows_moves_no_bit`. The host engine's paint takes
   the same default; its output is bitwise unchanged, and its CPU time under
   "spread" is not measured at a production row count (sec. 14 saw 5-9% at 4M
   rows on the laptop). The traced per-row reading rises 121 -> 125 B/row, the
   int32 per-row dead index exactly (traced both ways on one chunk), and
   `PAINT_CHUNK_TRACED_B_PER_ROW` moves with it; the card rate the planner
   charges is unaffected (69.5-71.0 above).
2. Chunk size under jit, re-read now that padding no longer penalizes it.
3. Host prep at 1024^3 after the index fix.
4. Carried: the paint's four-card split; cuBLAS on device paths; where the
   coarse mesh lives.

## 16. D2e eager -- one tile of the kick on the device is bitwise `tile_task` on the CPU backend

Local only, no cluster time. `src/inexor/device/tile.py`
(`tile_task_device`), tests `tests/test_device_tile.py`. Same arguments and
same returned dict as `engine.tile_task`, so `engine.apply_result` consumes it
unchanged.

**What it composes.** `device.decode.decode_rows` (D2a); the tile-local shift
and ownership, the latter a jnp twin of `forces.owned_mask_from_bricks`
(integer, so exact); the caller's `one_tile`; `forces.gather_coarse_subblock`
reading the tile's rows in row order with `live=owned` and its deferred guard
(D2c); the kick written with the host's operands in the host's order; and
`device.kick.quantize_per_brick`, split out of D2b's `kick_and_quantize`
unchanged apart from the fix below. Padded rows handed to the short arm are the
host's own (real rows cycled), so the tile paint sees exactly the host's input.
The coarse sub-blocks are still staged on the host and copied per tile: the
seam for where the coarse mesh lives.

### The gate: PASS, bitwise

Real `tile_task` and real jitted `one_tile` on both sides, random coarse
meshes, a state built with no brick spare and migrated so arena residents are
present (asserted). Equal: owned slots, int16 codes, written bricks, per-brick
scales, owned and overhang counts.
- Three tiles (first, middle, last) at all four fine/coarse dtype pairs,
  f64/f64, f64/f32, f32/f32, f32/f64. The kick promotes an f32 arm the way the
  host's numpy does; `kick_and_quantize` casts both arms to f64 first and would
  differ from the host whenever the fine arm is f32, which is why the tile
  writes the kick itself.
- **Every tile of a step, applied**, at f64 fine / f32 coarse (the driver's
  setting): `w` and `vel_scale` bitwise the host's, every particle written
  exactly once.
- Anti-vacuity on each arm: doubling the coarse meshes, and zeroing the short
  force, each move the codes. A halo of zero cells is refused by the deferred
  guard. Every written brick's extreme lands on +-32767.

### What the gate found: scalar division on CPU XLA is a reciprocal multiply

The first run failed on per-brick scales alone, 1 ulp low, in 6 of 64 tiles at
f64/f32 and 8 at f32/f32, with every input bitwise equal: short force, long
force, their sum, the kicked velocities, and the brick maxima out of
`segment_max`. `vmax / 32767` came back as `vmax * (1/32767)`. Measured
directly on 200,000 values (jax 0.10.2, CPU): dividing by a Python float, a 0-d
array, a literal under jit, or **a 0-d runtime argument to a jitted program**
all give the reciprocal form in 1,183 elements, and so does `full_like` built
inside jit. Only a full-shape divisor that exists as a runtime array matches
numpy. The per-row codes had the same exposure along one axis
(`v / scales[seg][:, None]`: 161,074 of 600,000 values one ulp off) and `rint`
hid it, but not for a value on a rounding tie.

Fixed in `quantize_per_brick`: both divisors are full-shape arrays. **D2b's own
gate had passed with the scalar form**, on random forces that happened not to
hit a sensitive brick. GPU untested.

### Owed

1. **Check B, jit**: one compiled tile program per step, bitwise the eager path
   on CPU or the difference reported in eps; the full-shape divisors must be
   passed in as arguments, since building them inside the program folds back to
   the reciprocal form.
2. **Check C, a GB200 job** (its own proposal): the GPU-vs-CPU floor of
   `one_tile` alone before any tolerance is set; per-tile device time and card
   memory at the 4096^3 tile shape.
3. The same division exposure wherever a device path must reproduce numpy:
   `codec.encode_positions` (`x / quantum`) and `codec.encode_velocities` face
   it when D3 puts the migrate insert on the device.
4. Carried: a per-tile slab window (the tile's bricks are not a contiguous slot
   range, so this path decodes against the whole `off` / `w`); coarse mesh
   placement; the four-card split.

## 17. D2e jit -- one compiled program per run, NOT bitwise the eager tile, accepted on a measured floor

Local only, no cluster time. `tile_task_device(jit=True, shapes=tile_step_shapes(st))`.
Decode, tile-local shift, ownership, short force, coarse gather, kick and
quantize are one program at fixed per-step shapes. The tile index, origins,
arena base, row count and kick coefficients are runtime values, so every tile
of every step reuses one executable: one trace across a whole step and a
second step with new coefficients (asserted). The coefficients enter as 0-d
arrays cast to the dtype the host's Python-float operand takes, and the
quantize's scale divisor is passed in (sec. 16). `forces.gather_coarse_subblock`
now accepts a traced origin and block; the integers are unchanged and every
test file using it passes.

### The pre-registered rule did not fire: jit is not bitwise

Compiled against eager, stage by stage over all 64 tiles at all four
fine/coarse dtype pairs: decoded positions and velocities, tile-local
coordinates and the short force are **bitwise**. **Every difference starts in
the coarse gather**:

| coarse arm | long-force values differing | worst tile | velocity codes differing |
|---|---|---|---|
| f64 | 50,042 of 98,304 | 4.48 eps x rms | 0 |
| f32 | 49,860 of 98,304 | **5.88 eps x rms** | 13-21 of 98,304, each by 1 |

(eps of the arm's own dtype; rms of that tile's eager long force.) The same
gather was measured non-bitwise under jit on 2026-08-09 (86 of 189 at 2.2e-16).
**Not fixable cheaply**: `jax.lax.optimization_barrier` after the coordinate
division, the offset, the TSC weights, the masking, the per-axis and corner
products, or the accumulation changes nothing (same count at every placement),
nor does `--xla_cpu_enable_fast_math=false`.

### Decision (JC, 2026-09-12): accept jit on a tolerance

The eager path stays the bitwise oracle against `engine.tile_task` (sec. 16);
jit is gated against eager in `tests/test_device_tile.py` by:
- **the long force within `JIT_LONG_FORCE_EPS` = 6 eps x rms**: the measured
  worst (5.88) rounded up to a whole eps, JC's choice among the measured-worst,
  per-dtype-exact and multi-seed options. One state's worst case is its whole
  basis.
- **exact**: short force, owned slots, written bricks, owned and overhang counts;
- **identities**: no code moves by more than 1, and each brick's scale moves by
  at most its largest kicked-velocity change / 32767 plus one ulp (a max is
  1-Lipschitz; met with equality at f32, never exceeded);
- **anti-vacuity**: coarse meshes nudged by 1e-13 (~450 f64 eps) fail the floor.

GPU against CPU cannot be bitwise either (the FFTs); that tolerance is check C's
and is set from its own measured floor, not from this one.

### Owed

1. **Check C, a GB200 job** (its own proposal): the GPU-vs-CPU floor of
   `one_tile` alone, then of the compiled tile; per-tile device time and card
   memory at the 4096^3 tile shape (P=576) against the probe's 94.2 ms.
2. Whether 6 eps holds across states: the floor was measured on one state.
3. Carried from sec. 16: the division exposure in `codec.encode_positions` /
   `encode_velocities` for D3; a per-tile slab window; coarse mesh placement;
   the four-card split.

## 18. D2e on a GB200, Vista 993754 -- the device tile's GPU-vs-CPU floor, and one tile at the 4096^3 tile shape

`1a41cb0`, 2026-09-12, gb node c672-011, COMPLETED rc=0 in 5:15, ~0.1 SU.
Sbatch `v2_d2e_device_tile_vista.sbatch`, script `v2_d2e_device_tile.py`; cards
`runs/v2/d2e_device_tile_{gb,smoke}.json` (force-added), log copied to
`runs/v2/d2e-tile-993754.log`. Every arm its own process.

### Check A on Grace: PASS

On the node's CPU backend at cdev (P=320, 8 tiles, 61,879 arena residents,
ratified f64 fine / f32 coarse), the eager device tile equals `engine.tile_task`
exactly on every tile: owned slots, codes, written bricks, scales. The same arm
on the M4 laptop also passed on all 8 tiles (50.2 s, 13.7 GB peak RSS).

### The GPU-vs-CPU floor at cdev (P=320), no bar applied

GPU against the CPU arm's saved outputs. `one_tile` alone is fed the CPU arm's
exact inputs.

| quantity | values differing | worst tile, eps x rms | worst rms-relative |
|---|---|---|---|
| layout (owned slots, written bricks, counts) | **0** | -- | -- |
| short force, `one_tile` alone (f64) | 91.8% | **19.6** | 6.0e-16 |
| short force, inside the jitted tile | 91.8% | 19.6 (identical to alone) | 6.0e-16 |
| long force (f32 coarse) | 50.7% | **8.2** | 6.4e-8 |
| kicked velocity (f64) | 86.3% | -- | 2.1e-8 |
| velocity codes | 5,549 of 50,331,648 (0.011%), each by 1 | -- | -- |
| per-brick scales | -- | -- | max relative 4.5e-8 |

(The kicked velocity's eps x rms column in the card is in f64 eps while its
difference comes from the f32 long force, so it reads ~1.5e9 and is not a
measure of anything; the rms-relative figure is.)

- **The jitted tile adds nothing to the short force's cross-backend floor**:
  inside the tile it is identical to `one_tile` alone on the same inputs.
- **The long force's floor is 8.2 eps x rms against the CPU jit-vs-eager 5.88
  (sec. 17)**: the GPU comparison is jit against jit, so it is a different
  pair and not a sum of the two.
- **The floor grows with P**: the 32^3 smoke (P=32) read 11.3 for the short
  force against 19.6 at P=320. **P=576 was not compared across backends.**

### One tile at the 4096^3 tile shape (cgh64, tile 512, buffer 32: P=576)

cap 26,632,171 rows, largest tile 23,888,100: 4096^3's tile shape and row count.
One GB200.

| | `one_tile` alone | jitted device tile |
|---|---|---|
| compile (first tile) | 20.4 s | 22.8 s |
| per tile, median of 3 | **126 ms** | **566 ms** |
| of which host prep (decode plan + coarse staging) | -- | 30 ms |
| device memory held before the first call | 4.60 GB | 4.60 GB |
| peak over that | 13.1 GB | 17.5 GB, **16.0 GB** excluding the state upload |
| per padded row, excluding the upload | -- | 599 B |

- **Neither per-tile time is device time.** The 126 ms includes reading back
  the cap x 3 f64 short forces (639 MB). The 566 ms includes uploading the
  whole state's `off`/`w`/`vel_scale`/`arena_bucket`, **1.58 GB per call**, which
  this development path does and the 4096^3 design does not (it streams a
  window), plus the result readback. **So the 566 ms is not a per-tile rate
  for the design and no step wall is projected from it.** The probe's 94.2 ms
  was device-only on a different paint (f64 gauss), so 126 ms is not a
  like-for-like comparison with it either.
- **Memory against the planner** (`inexor.plan --preset c-hero --backend
  device`): the 4.60 GB held before any call is the planner's `tile_kernels`,
  4.602 GB, to the MB. Its `tile_workspace` charge is **8.414 GB, against 13.1
  measured for `one_tile` alone (1.56x) and 16.0 for the jitted tile (1.90x)**.
  The per-GPU total (144.5 GB, 0.73x) does not move: its worst phase is
  another one at 57.7 GB.
- **The persistent compilation cache refused both programs** as 4.6 GB
  executables (over protobuf's 2 GiB). That is the size of the three P=576
  f64 short kernels (3 x 1.53 GB), which `make_tile_force_fn` closes over, so
  they are compiled in as constants. The run is unaffected (only the cache
  write fails); whether the constants also occupy device memory beside the
  4.60 GB held is not attributed here.

### Owed

1. **Attribute the 566 ms**: device compute, the 1.58 GB upload and the result
   readback, separately, before any per-step projection.
2. **The GPU-vs-CPU floor at P=576**, then a cross-backend tolerance set with
   JC from it.
3. **`tile_workspace` in the planner** re-read against 13.1 / 16.0 GB, once the
   tile path that prices it replaces the one the charge was written for.
4. **The short kernels as program arguments** rather than closure constants.
5. Carried: `JIT_LONG_FORCE_EPS` measured on one state (sec. 17); the division
   exposure in `codec` for D3; a per-tile slab window; coarse mesh placement;
   the four-card split.

## 19. D2e attribution, Vista 993817 -- of sec. 18's 566 ms per tile, 71 ms is device compute; the largest term is a host pass

`5283433`, 2026-09-12, gb node c672-011 (the node 993754 ran on), COMPLETED rc=0
in 4:17, ~0.1 SU. Submitted with `D2E_ARMS=short+tile+tile-staged`; cards
`runs/v2/d2e_device_tile_attr_{gb,gbsmoke}.json` (force-added), log copied to
`runs/v2/d2e-tile-993817.log`. P=576, cap 26,632,171, one GB200, median of 3
tiles after a warm one. Every phase is ended by a device sync
(`tile_task_device(timings=)`); an untimed call on the same tile precedes each
timed one.

**The instrument is neutral**: timed 577 ms against untimed 570 (state uploaded
each call), 391 against 389 (state placed once). **The untimed 570 ms
reproduces 993754's 566.**

| phase, per tile | state uploaded every call | state placed once |
|---|---|---|
| host: decode plan | 8 ms | 8 ms |
| host: coarse sub-block staging | 25 ms | 22 ms |
| upload: per-tile inputs | 12 ms | 12 ms |
| **upload: state arrays (1.58 GB)** | **183 ms** | 0 |
| **device compute, the whole tile program** | **71 ms** | **71 ms** |
| readback of results | 62 ms | 62 ms |
| **host: result assembly** | **214 ms** | **214 ms** |
| total (untimed) | 570 ms | 389 ms |

`one_tile` alone at the same shape: **29 ms device compute**, 87 ms to read back
its cap x 3 f64 forces (639 MB), 95 ms to upload its inputs.

### What it establishes

- **Device compute is 71 ms of a tile; 29 ms of that is the short force.** The
  other 42 ms is the decode, ownership, coarse gather, kick and quantize in the
  same program. Everything else in the 566 ms is host work or bus traffic.
- **The largest term is the host result assembly, 214 ms, 37% of the tile.**
  It is `device.tile._result`: two boolean-mask gathers over the 26.6M padded
  rows (owned slots, int64, and their codes, (cap, 3) int16) that build the
  `engine.apply_result` contract. That is a per-particle host pass, the class
  the design exists to delete (sec. 1). The split between the two gathers is
  not measured.
- **The state upload is 183 ms for 1.58 GB (8.6 GB/s)**, and placing the state
  once removes exactly that: 570 -> 389 ms, with every other phase unchanged
  to within 3 ms. At 4096^3 the state is streamed as a window, not uploaded per
  tile, so this term belongs to the development path.
- **Readback is 62 ms** for the cap-sized slots, owned mask and codes (~400 MB).
  It moves padded rows the host then discards.
- The gb probe's 94.2 ms per tile was device-only on the f64 paint; this short
  force is the integer paint inside a compiled program, 29 ms. The two are not
  the same program, so the ratio is not attributed to either difference.

### Owed

1. **The result contract, off the host**: compact owned slots and codes on the
   device (or write codes into the device-side window) so neither the 214 ms
   host gathers nor the padded-row readback remain. Fixed shapes rule out a
   per-tile compaction; the write target is the D3 / streaming design.
2. The four-card split of the tile loop; the GPU-vs-CPU floor at P=576; the
   short kernels as program arguments; `tile_workspace` in the planner (sec.
   18).
3. Carried: `JIT_LONG_FORCE_EPS` on one state; the `codec` division exposure for
   D3; a per-tile slab window; coarse mesh placement.

## 20. D2e step loop -- the kick written into the device state, bitwise the host-apply path

Local only, no cluster time. `device.tile.tile_loop_device(st, one_tile, C,
g_coarse, members, shapes)`, tests in `tests/test_device_tile.py`. JC chose a
step-level function over a per-tile write option, and approved the sec. 17
tolerance as the fallback if exactness failed. It did not fail.

**What it does.** Each tile's compiled program writes its codes and per-brick
scales into the device state and returns only scalars (owned count, overhang,
stencil bounds). The program donates `w` and `vel_scale`, so the card never
holds two copies. After the loop every stencil guard is resolved and, for a
full step, the owned total checked against the particle count; only then are
`st.w` and `st.vel_scale` copied back to the host, once. That removes both
sec. 19 terms that moved results: the 62 ms readback of padded rows and the
214 ms host result assembly.

**How the write stays exact.** Codes: the new code minus the stored code, in
int32, zeroed on rows the tile does not own, narrowed to int16 (modular, the
codec's rule) and scatter-ADDED at the rows' slots. Adding it back gives the new
code exactly; padding, all at slot 0, adds zero, so repeated indices cannot
collide the way a scatter-set's would. Scales: a set on the tile's own bricks,
which are unique within a tile, keeping the stored scale where the tile owns no
rows. `stage_state_on_device` copies rather than views the host arrays, because
a donated buffer aliasing numpy memory would be overwritten under the host.

**Gate: PASS, bitwise.** Every tile of a step at f64 fine / f32 coarse, arena
residents present:
- `w` and `vel_scale` after the device loop equal the jitted tile's host
  results applied tile by tile with `engine.apply_result`; owned total equals
  the particle count; the step writes something.
- A skipped tile's owned rows and bricks keep their stored codes and scales, and
  the host arrays are untouched while the device state changes.
- A loop missing a tile is refused as a step, before the host is written.

### Owed

1. ~~A gb timing job~~ DONE, sec. 21.
2. Carried from sec. 19: the four-card split of the tile loop; the GPU-vs-CPU
   floor at P=576; the short kernels as program arguments; `tile_workspace` in
   the planner; the `codec` division exposure for D3; a per-tile slab window;
   coarse mesh placement.

## 21. D2e step loop on a GB200, Vista 993837 -- 110 ms per P=576 tile with results kept on the device, against 389

`7b0d059`, 2026-09-12, gb node c672-016, COMPLETED rc=0 in 3:18, ~0.1 SU.
Submitted with `D2E_ARMS=tile-staged+loop`; cards
`runs/v2/d2e_device_tile_loop_{gb,gbsmoke}.json` (force-added), log copied to
`runs/v2/d2e-tile-993837.log`. P=576, cap 26,632,171, 8 tiles of a 512^3
state, one GB200, state placed on the device once in both arms.

**The baseline reproduces on a different node**: the per-tile path with host
results reads 389 ms untimed (993817: 389), with the same phases (compute 72,
readback 61, host result assembly 213 ms).

| whole step of 8 tiles, `tile_loop_device` | reading |
|---|---|
| warm step (compile) | 23.06 s |
| **untimed step** | **0.883 s = 110 ms per tile**, one copy-back included |
| synced timed step | 1.142 s |
| timed, per tile: plan / staging / per-tile upload / compute | 6.6 / 22.6 / 8.8 / **76.3 ms** |
| timed, once per step: copy-back of `w` + `vel_scale` (1.02 GB) | **226.8 ms** |
| owned rows | 134,217,728 of 134,217,728 |
| device peak over baseline | 17.57 GB (per-tile path: 17.56) |

### What it establishes

- **389 -> 110 ms per tile, 3.5x, with results written into the device state.**
  The readback of padded rows and the host result assembly are gone, as
  intended, and nothing else grew to replace them.
- **The write costs ~4 ms of device compute**: 76.3 ms against the per-tile
  program's 72 in the same job.
- **Donation holds on the card**: the peak is the per-tile path's to 0.01 GB,
  so no second copy of `w` or `vel_scale` exists.
- **The synced timings are NOT neutral here**, unlike the per-tile arm (sec.
  19): the timed step is 259 ms (29%) slower than the untimed one. A sync after
  every tile evidently stops the host from preparing the next tile while the
  device computes the last. So the phase split gives proportions, and the
  untimed 110 ms is the per-tile cost to quote. How the untimed step divides
  between host prep, compute and the copy-back is not measured.
- **The copy-back is per step, not per tile**: 227 ms for 1.02 GB here. At
  4096^3 the state returns as a streamed window, so this figure prices the
  development path's whole-state copy, not the design's.
- **The largest per-tile term that is not compute is the host coarse
  sub-block staging, 22.6 ms**, the seam for where the coarse mesh lives.

### Owed

1. How the untimed step divides (host prep overlapping device compute), if a
   per-step projection ever needs it.
2. The coarse sub-block staging (22.6 ms per tile) once the coarse mesh's
   placement is decided.
3. Carried: the four-card split of the tile loop; the GPU-vs-CPU floor at
   P=576; the short kernels as program arguments; `tile_workspace` in the
   planner; `JIT_LONG_FORCE_EPS` on one state; the `codec` division exposure for
   D3; a per-tile slab window.

## 22. Four-GPU split of the device tile loop, and host coarse staging at 2048^3, Vista 993849

`166c1ec`, 2026-09-13, gb node c672-011, COMPLETED rc=0 in 5:44, ~0.1 SU.
Submitted with `D2E_ARMS=split+stage`, bundled into one job at JC's direction;
cards `runs/v2/d2e_device_tile_split_{gb,gbsmoke}.json` (force-added), log
copied to `runs/v2/d2e-tile-993849.log`.

### The tile loop scales 3.36x across four GB200s

P=576 (cgh64, tile 512), `tile_loop_device` with results on the device and no
copy-back, 30 timed steps per process after a warm step. One-GPU reference: all
8 tiles on GPU 0. Four GPUs: one process per card, 2 tiles each, timed steps
released together by a barrier. Each arm has its own reference in the same job.

| | one GPU | four GPUs, combined | per card in the four | efficiency |
|---|---|---|---|---|
| pinned (`numactl`) | 12.48 tiles/s = 80.1 ms/tile | 41.99 tiles/s | 94.8-95.8 ms/tile | **3.36x** |
| unpinned | 12.37 tiles/s = 80.9 ms/tile | 40.98 tiles/s | 94.9-99.7 ms/tile | **3.31x** |

- **3.36x, where the coarse FFT's split was 2.8-3.0x (sec. 10).** Each card
  runs its tiles ~19% slower beside three others than alone; which shared
  resource that is (host prep threads, bus, memory bandwidth) is not attributed.
- **Pinning is worth ~1.5%.** CPU binding is proven by the receipts: pinned
  workers could run on exactly their card's socket (cores 0-71 for GPUs 0-1,
  72-143 for GPUs 2-3), unpinned on all 144. **Memory binding is NOT proven**:
  every worker, pinned or not, reports `Mems_allowed_list` `0-2,10,18,26`, which
  is the cpuset and not `--membind`'s policy, so the receipt cannot see it.
- **The single-GPU rate cross-checks sec. 21**: 80.1 ms/tile here with no
  copy-back, against 993837's untimed step less its copy-back, (883 - 227) / 8
  = 82 ms.
- **At this throughput, 4096 tiles take 4096 / 41.99 = 97.5 s.** This is the
  development path's tile loop only: it decodes against a whole 512^3 state
  already on each card, where the 4096^3 design streams a window, and it
  includes no copy-back, migrate or coarse solve. It is the first measured
  replacement for 5y's projected 96.5 s tile force, which covered the short
  force alone and assumed a perfect four-way split.

### Host staging of the coarse sub-blocks does not slow at the real mesh size

Three 2048^3 f32 meshes (103.1 GB, every page written in 5.9 s, pinned to
socket 0; 1,722 GB available), 32 tile positions at 4096^3 geometry including
wrapping corners, extent 132, three sub-blocks per tile:

| mesh | per tile, median | p90 |
|---|---|---|
| 2048^3, pass 1 | **22.7 ms** | 22.9 ms |
| 2048^3, pass 2 | 22.7 ms | 22.8 ms |
| 256^3, same process | 24.6 ms | -- |

- **The 2048^3 mesh is no slower than the 256^3 one (0.92x)**, and the first
  pass is no slower than the second: the gather's cost is the ~27.6 MB it
  delivers, not the size of the mesh it reads from. It reproduces sec. 21's
  22.6 ms.
- **For the coarse mesh placement decision**: keeping the mesh on the host costs
  ~23 ms of host staging per tile, flat in mesh size, plus the sub-blocks' share
  of the per-tile upload. That is ~28% of a tile's 80 ms on one card. The
  device-side gather that card placement would use instead is still unmeasured,
  and so is the paint accumulator's side of the decision (sec. 12).

### Owed

1. What the four-card slowdown (~19% per card) is spent on.
2. Whether `--membind` applied (read the policy, e.g. `numa_maps`, not the
   cpuset).
3. The device-side sub-block gather's cost, for the coarse mesh placement
   decision (JC's).
4. Carried: the GPU-vs-CPU floor at P=576; the short kernels as program
   arguments; `tile_workspace` in the planner; `JIT_LONG_FORCE_EPS` on one
   state; the `codec` division exposure for D3; a per-tile slab window.

## 23. The coarse mesh on the card, Vista 993866 -- the device gather is 0.19 ms at the 4096^3 shard, and the step loop does not slow

`3790d66`, 2026-09-13, gb node c672-011, COMPLETED rc=0 in 7:45.
Submitted with `D2E_ARMS=loop+loop-shard+dgather+numa+xback576`, bundled at JC's
direction; cards `runs/v2/d2e_device_tile_shard_{gb,gbsmoke}.json`, log copied to
`runs/v2/d2e-tile-993866.log`. Every arm its own process; the smoke leg passed
every arm first.

### The step loop with the coarse mesh on the card (cgh64, tile 512, P=576, one GB200)

`tile_loop_device`, results on the device, one copy-back per step (1.02 GB).
`loop` stages each tile's three sub-blocks on the host and uploads them;
`loop-shard` holds the coarse meshes as a resident shard and gathers the
sub-blocks inside the tile program (`coarse_shard=`, bitwise host staging,
sec. 22's commit `f43646f`). Same node, same process shape.

| | coarse on host | coarse on card |
|---|---|---|
| untimed step, 8 tiles (the reference wall) | 873.2 ms = 109.1 ms/tile | **850.7 ms = 106.3 ms/tile** |
| synced timed, per tile: plan | 6.5 ms | 6.5 ms |
| stage (host) | **23.0 ms** | 0.3 ms |
| h2d_tile | 9.1 ms | 5.5 ms |
| compute | 75.3 ms | 75.4 ms |
| copy-back, once per step | 229.6 ms | 231.2 ms |
| device peak over baseline | 17.56 GB | 17.52 GB |

- **On one card, placement moves the untimed wall by 2.6%**, 2.8 ms/tile, where
  the synced phases say the host path spends 26 ms/tile more. The gap is the
  sec. 21 trap in the other direction: unsynced, host staging overlaps the
  previous tile's device compute, so most of its 23 ms is not on the wall. One
  untimed reading per arm; 993837's host-path reading was 883 ms, so run-to-run
  spread is of the same order as the difference.
- **Compute is identical** (75.3 vs 75.4 ms): the gather inside the program
  costs nothing visible.
- **Not measured: the host path under four cards.** Sec. 22's four-card tile
  loop lost ~19% per card to an unattributed shared resource; host staging is a
  per-tile host-memory pass that four processes would share. Neither arm here
  was run four-wide.

### The device gather at the real shard size

One card's 4096^3 coarse shard: three f32 meshes of 516 x 2048 x 2048 (x origin
-2, so 512 owned planes plus two ghost planes each side), **25.97 GB, built on
the device in 0.59 s**; compile 0.11 s. 32 tile positions at 4096^3 geometry
including wrapping corners, extent 132, three blocks per tile.

| | per tile, median | p90 |
|---|---|---|
| device gather (`subblock_device`) | **0.19 ms** | 0.20 ms |
| upload of host-staged blocks (27.6 MB) | 3.86 ms | 3.89 ms |
| host staging from a 2048^3 mesh (sec. 22, for reference) | 22.7 ms | 22.9 ms |

- **Every block equals its formula bitwise, and a wrong offset differs**
  (anti-vacuity), so the gather reads the right cells.
- Device peak over baseline 26.04 GB: the shard plus 64 MB. **The planner's
  `coarse_force_resident` charges 25.770 GB per card, the owned planes only;
  the ghost planes add 0.20 GB per card**, too small to move any verdict.

### Memory binding applies

A 2 GB touched allocation under `numactl` bound to socket 1 reads policy
`bind:1` in `numa_maps`, all 32,771 pages on N1; unbound it reads `default`.
**`--membind` takes effect, and `numa_maps` is the receipt that can see it**
(sec. 22's `Mems_allowed_list` could not). The unbound process's pages also
landed on N1 (first touch where it ran), so placement alone would not have
distinguished the two.

### The GPU-vs-CPU floor at P=576, one tile, no bar applied

Check A on Grace at P=576: the eager device tile equals `engine.tile_task`
exactly (16,777,547 owned; CPU jit 26.9 s). GPU against that CPU arm:

| quantity | values differing | eps x rms | rms-relative |
|---|---|---|---|
| layout | **0** | -- | -- |
| short force, alone and in the jitted tile (f64) | 93.2% | **26.9** | 7.3e-16 |
| long force (f32 coarse) | 50.7% | **8.2** | 6.4e-8 |
| kicked velocity (f64) | 88.1% | -- (sec. 18's note) | 2.1e-8 |
| velocity codes | 5,547 of 50,332,641, each by 1 | -- | -- |
| per-brick scales | -- | -- | max relative 4.6e-8 |

- **The short force's floor keeps growing with P: 11.3 (P=32), 19.6 (P=320),
  26.9 (P=576). The long force's does not: 8.2 at both P=320 and P=576.** One
  tile at P=576, so a tolerance set from this carries one reading.

### The placement decision, as measured (JC's call)

Coarse meshes, their kernel prefactor and match factor, the paint accumulator
and the coarse delta are the terms that move with placement.
`inexor.plan --preset c-hero --backend device` shards them (`DEVICE_PLACEMENT`);
moving the planner's own per-card figures x 4 onto the host is arithmetic, not a
planner run:

| | host | per card |
|---|---|---|
| sharded on the cards (planner, today) | 956.3 GB, **0.93x** of 1026 | 144.5 GB, **0.73x** of 199 |
| on the host: + resident meshes, pref, match factor (137.5 GB) | 1093.7 GB, 1.07x | -- |
| + the paint phase's accumulator and delta (103.1 GB) | 1196.8 GB, **1.17x** | -- |
| + the solve phase's delta and copy, summed as the CPU column sums (68.7 GB) | 1265.5 GB, **1.23x** | -- |

On time, one card: card placement is 2.6% faster on the untimed wall, with an
identical compute term. The four-card host-path cost is the unmeasured side.

**DECIDED (JC, 2026-09-13): the coarse mesh lives on the cards**, sharded as
`plan.DEVICE_PLACEMENT` already charges it. Host placement does not fit a gb
node's host at 1.17-1.23x; card placement is not slower.

### Owed

1. A card-resident paint accumulator behind the `HostInt64Accumulator` seam
   (`device/paint.py`), now that the cards hold the coarse mesh.
2. [DONE] The planner charges each shard's ghost planes (`plan.shard_halo_planes`):
   `coarse_force_resident` is now exactly the 25,971,130,368 bytes this job held.
3. Carried: the cross-backend tolerance (JC) from the floor above; the four-card
   ~19% per-card slowdown; the short kernels as program arguments;
   `tile_workspace` in the planner; `JIT_LONG_FORCE_EPS` on one state; the
   `codec` division exposure for D3; a per-tile slab window.

## 24. D2f on a gb node, Vista 994608 -- the density on the cards is bitwise the CPU host at one and four GB200s, and accumulating a 4096^3 step costs ~3 s

`fd7a70f`, 2026-09-13, gb node c672-017, COMPLETED rc=0 in 2:53. Sbatch
`v2_d2f_cards_vista.sbatch`; cards `runs/v2/d2d_device_paint_d2f_{gb,smoke}.json`,
log copied to `runs/v2/d2f-cards-994608.log`. Every arm its own process; the
32^3 smoke passed every arm on the GPUs first.

**What was built** (`adbe030`, `adfe9f5`): `device.paint.coarse_delta_cards` --
per-card int64 accumulators with 1 + 2 ghost planes, a ghost fold into the
owning card, the density decoded on each card with a plane-shaped runtime
divisor -- and `ooc_fft.forward_from_card_planes`, which the factorized solve
uses when handed the cards' shards. Gated bitwise on the laptop at 1, 2 and 4
cards (including four forced host devices).

### Identity: PASS on every gate

At cdev (128^3 coarse mesh), against a `JAX_PLATFORMS=cpu` process's
`engine.coarse_delta_streamed`:

| state | 1 GB200 | 4 GB200s | ghost planes with mass (1 / 4 cards) |
|---|---|---|---|
| plain | hash-equal | hash-equal, 4 devices used, 16 chunks each | 3 / 12 |
| 61,879 arena residents | hash-equal | hash-equal, 4 devices used, 16 chunks each | 3 / 12 |

- **The three force meshes solved from the cards hash-equal the solve of the
  same density from host memory on the GPU**, at both widths and both states.
- The CPU process's own jitted paint hash-equals its host engine (both states).

### Cost at 4096^3 shard shapes (synthetic blocks, four GB200s)

Four accumulators of 515 x 2048 x 2048 int64 on x-ranges of 512 planes; 1024
quarter-slab blocks of 11 x 515 x 2048 (the 4096^3 chunk block), 256 per card,
each add synced.

| phase | reading |
|---|---|
| allocate four accumulators on the cards | 0.88 s, 17,280,532,480 B each (= the planner's 17.281 GB) |
| one add | **1.05 ms median** on every card; first add 0.10-0.15 s |
| all 1024 adds, one thread per card | **0.43 s** wall |
| ghost fold, 12 planes | 0.97 s |
| decode, 2048 planes on 4 cards | 1.51 s |
| device peak per card | **25.90 GB**: accumulator 17.28 + density 8.59 + 34 MB |

- **Receipts.** Owned mass 11,880,366,080 = blocks x block cells, exactly.
  Sampled planes on every card, including each card's first plane (which
  carries a folded ghost), decode **bitwise against numpy** -- the first GPU
  reading of the plane-shaped divisor. An overlap cell (value >= 2) was among
  them.
- **Accumulate + fold + decode is ~3 s per 4096^3 step**, 0.3% of the 1080 s
  bar; the paint itself (secs. 13-15) is the term that matters on this side.
- **The per-card peak matches the planner's paint-phase charge** of accumulator
  plus density to 34 MB, which is one decode plane of f64.

### The forward transform from the cards

| | from the cards | from host slabs, same four devices |
|---|---|---|
| pass 1 | **3.50 s** | 4.75 s |
| pass 2 | 5.65 s | 5.25 s |
| forward | **9.15 s** | 10.00 s |

Spectra bitwise equal. Getting the density to the host for the second leg took
6.02 s (34.4 GB), which the card path does not pay at all.

- **Pass 1 is 1.25 s faster from the cards**: the planes no longer cross the
  bus inbound. Pass 2 is the same host-resident work in both legs (5.65 vs 5.25).
- **Caveat on the ratio.** One call per leg, no warm-up, and the card leg ran
  FIRST, so any per-op compilation of the eager transforms landed on it. The
  pass 1 difference is therefore a lower bound on the saving, not a measurement
  of it. The host leg's 10.00 s forward is also above sec. 10's 6.86 s on another
  node, a gap this job does not attribute.
- **Against the host-mesh design at step level**: that design pays the density's
  host round trip (6.02 s out here) plus pass 1's inbound planes; the card design
  pays neither.

### Owed

1. [DONE, sec. 25] A warm-started forward pair (both legs warmed, alternated).
   It reads a 0.50 s pass 1 saving, SMALLER than the 1.25 s above: the cold
   reading overstated the saving rather than bounding it from below.
2. Carried: the cross-backend tolerance for the tile (JC); the four-card ~19%
   per-card tile slowdown; the tile loop's split under threads rather than
   processes; the short kernels as program arguments; `tile_workspace` in the
   planner; the `codec` division exposure for D3; a per-tile slab window; the
   inverse transforms still read and write host slabs.

## 25. D2f bundle, Vista 994675 -- one thread per card splits the tile loop 3.6-3.8x, as processes do; the paint splits 2.2-2.3x; the warm forward saves 0.5 s from the cards

`dfd5bfa`, 2026-09-13, gb node c672-015, COMPLETED rc=0 in 13:13. Sbatch
`v2_d2f_bundle_vista.sbatch`, bundled at JC's direction; cards
`runs/v2/d2e_device_tile_d2f_{threads_gb,bundle_smoke}.json` and
`runs/v2/d2d_device_paint_d2f_bundle_{gb,smoke}.json`, log copied to
`runs/v2/d2f-bundle-994675.log`. Every arm its own process; both smoke legs
passed on the GPUs first.

### The tile loop split: threads against processes (cgh64, tile 512, P=576, coarse on the cards, unpinned, 30 steps)

`split-threads` runs one process with one thread per card (`tile_loop_device(device=)`,
`dfd5bfa`); `split` runs one process per card (`CUDA_VISIBLE_DEVICES`). Each arm
takes its own one-card reference in the same shape. The threads arm ran before
AND after the process legs.

| | one card, tiles/s | four cards, summed tiles/s | split, summed | split, by the slowest card |
|---|---|---|---|---|
| threads, before | 13.03 | 47.03 (per card 11.64-11.96) | **3.61x** | 3.57x |
| processes | 12.89 | 47.87 (per card 11.83-12.09) | **3.71x** | -- |
| threads, after | 12.97 | 48.80 (per card 12.07-12.32) | **3.76x** | 3.72x |

- **Pre-registered criterion: threads and processes more than ~1.3x apart means
  the GIL is the variable. They are 0.98x and 1.02x apart**, straddling the
  processes reading, and the before/after drift (3.61 -> 3.76) is larger than
  either gap. **The GIL is not the variable; one process with a thread per card
  is admissible for the executor** (the D5 criterion).
- **4096 tiles cost ~84-88 s** on the development path (48.8 summed, 46.5 by the
  slowest card), against sec. 22's 97.5 s.
- **Per card, four-wide is 5-10% below a lone card** (11.6-12.3 against
  12.9-13.0), where sec. 22 read ~19%. The two jobs differ in where the coarse
  meshes live (sec. 22 staged them from the host per tile, a host pass four
  processes share) and in pinning; this job ran no host-staged four-wide leg, so
  the placement is a candidate for the difference, not its measured cause.
- Steady per-step walls are flat over 30 steps: 0.619-0.622 s for 8 tiles on one
  card; 0.165-0.170 s for 2 tiles per card four-wide.
- **Compilation.** Each process compiles the 4.60 GB tile program (23 s on one
  card) and the persistent cache refuses it (`GpuExecutableProto` > 2 GiB), so no
  run reuses it. Under threads, cards 1-3 compiled concurrently in 29.7-31.0 s
  each; card 0 reused the reference leg's in-process executable (2.8-2.9 s). A
  one-time cost per run, not per step.
- The 32^3 smoke split read 1.39-2.04x: tiles of that size are overhead, not a
  reading of the split.

### The coarse paint across four cards (`coarse_delta_cards`, one thread per card, n_part=1024, 64 bricks/side)

State built in 243.6 s. Medians of three after one warm call per width.

| chunk | rows per chunk (padded to) | chunks | one card | four cards | split |
|---|---|---|---|---|---|
| 16384 bricks | 67,109,413 (84,551,871) | 16, 4 per card | 2.00 s | 0.90 s | **2.23x** |
| 1024 bricks | 4,194,760 (5,284,492) | 256, 64 per card | 2.27 s | 1.00 s | **2.28x** |

- **Density hash-equal between one and four cards at both chunk sizes.**
- **The paint does not divide by four.** Any paint term priced by dividing a
  one-card time by four is ~1.75x optimistic. This arm records no phase
  breakdown, so the shortfall is NOT attributed; host-side per-chunk work shared
  by the four threads is the leading candidate, as it was for the transform
  (sec. 10's 2.8-3.0x), and it is a candidate only.
- The 16x smaller chunk is 13% slower on one card and splits the same.
- Device peak 6.20 GB on card 0, 5.29 GB on cards 1-3.

### 4096^3 shard shapes, repeated (four cards)

| phase | 994608 (sec. 24) | 994675 |
|---|---|---|
| allocate four accumulators | 0.88 s | 0.87 s |
| one add, median | 1.05 ms | **1.07 ms** |
| all 1024 adds | 0.43 s | 0.41 s |
| ghost fold, 12 planes | 0.97 s | 0.89 s |
| decode | 1.51 s | 1.42 s |
| device peak per card | 25.90 GB | 25.90 GB |

Mass exact (11,880,366,080), sampled planes bitwise numpy's decode (an overlap
cell among them). **Accumulate + fold + decode reproduces at ~2.7 s per step.**

### The forward transform, warmed and ABBA

Each leg warmed once, then cards-host-host-cards; medians of the two reps.

| | from the cards | from host slabs |
|---|---|---|
| warm call | 5.81 s | 5.74 s |
| pass 1 | **2.15 s** | 2.65 s |
| pass 2 | 3.10 s | 3.10 s |
| forward | **5.25 s** | 5.75 s |

Spectra bitwise. Density to the host for the second leg: 4.64 s.

- **Reading from the cards saves 0.50 s, all of it in pass 1**; pass 2 is the
  same host-resident work in both legs to 5 ms.
- **This corrects sec. 24**, which called its 1.25 s pass 1 saving a lower bound.
  Warm, the saving is 0.50 s: the cold single calls (9.15 / 10.00 s) carried ~4 s
  of first-call cost each, unevenly. The host leg's 5.75 s is also below sec. 10's
  6.86 s, so the "10.00 above 6.86" gap sec. 24 left unattributed was the cold
  start.
- At step level the card design still skips the density's host copy (4.64 s
  here) and the inbound planes.

### Owed

1. The paint's four-card shortfall, attributed by phase, if a step projection
   needs the paint term to better than the measured 2.2-2.3x.
2. Carried: the cross-backend tolerance for the tile (JC); the short kernels as
   program arguments (the 4.60 GB executables, which also defeat the persistent
   cache); `tile_workspace` in the planner; the `codec` division exposure; a
   per-tile slab window; the inverse transforms still read and write host slabs.

## 26. The compiled insert escapes int16 on a GB200, Vista 995067 + 995228 -- the jitted int16 row magnitude is wrong; the scatter is not

`e74c23e` (995067, gb node c672-002, FAILED rc=1 in 0:40) and `040f4f5` (995228,
c672-018, COMPLETED rc=0 in 0:54). Logs copied to `runs/v2/d3-migrate-995067.log`
and `runs/v2/d3-escape-995228.log`; cards `runs/v2/d3_insert_escape_diag_{gb,gb_cpu}.json`,
captured inputs `runs/v2/d3_insert_escape_inputs_gb.npz`.

### What failed

995067 (the D3 migrate job, `scripts/v2_d3_migrate_vista.sbatch`) stopped in its
GPU smoke, in the first arm: `drift_and_migrate` on the 32^3 xback state with the
compiled eject and insert raised the int16 guard in `_insert_slab_jax`, **code
49424 against 32767**. The same state reads exactly 32767 in numpy and under CPU
XLA. The smoke gate held: no timing leg ran, nothing was written past the guard.

**The escape could not come from the inputs.** The kernel's brick scale is
max(|w| * s_old) / 32767 over the rows it then rescales by s_old / s_b, so every
rescaled code is <= 32767 for ANY inputs unless an operation in the program
returns a wrong value.

### Where, measured on the captured call (995228)

The first escaping call (call 0, 15,898 rows, 4,091 bound for the slab, 64 bricks)
captured on the GPU; no code is -32768.

| reading | GPU | CPU XLA, same inputs, same node |
|---|---|---|
| inputs from the jax eject vs the numpy eject | **identical** (both escape) | -- |
| production program, jit | **abs_max 49424; 27 of 64 scales differ** | 32767; 0 |
| production program, eager body | 32767; 0 | 32767; 0 |
| sort order / brick index | equal / equal | equal / equal |
| **row magnitude `abs(w).max(1).astype(f64) * s_old`** | **5,293 of 15,898 rows differ** | 0 |
| per-brick max | 27 differ | 0 |
| per-brick max recomputed in numpy from the program's OWN brick index and magnitudes | **0 differ** | 0 |
| scatter-max alone (int64 `.at[].max` eager and jit, `segment_max` int32; 4,096-1M rows, sorted and not) | 0 in every form | 0 |

- **The scatter is clean; the jitted int16 row magnitude is not.** Every
  downstream error (27 scales too small, the escape) follows from it.
- **Not identified: the operation inside that expression, or the mechanism.** No
  simple wrong formula reproduces the GPU's counts on these inputs (tested on the
  laptop: one component's |w|, signed max, |min|, int8 truncation, min of |w| --
  nearest 7,933 and 7,965 rows against 5,293). The GPU's magnitude values were
  not saved by this job.
- Every other abs/max reduction on the device path is over floats (`kick`,
  `codec`) or an int64 count (`paint`); the insert was the only int16 one.

### The fix

`insert_jax` widens the codes to f64 before abs and max:
`jnp.abs(w_s.astype(jnp.float64)).max(axis=1) * s_s`. Every int16 is exact in f64,
so the value is numpy's by construction. `tests/test_insert_jax.py` +
`tests/test_eject_jax.py` 16 passed; the D3 probe's laptop smoke passes all three
arms; the diagnostic on the GPU-captured inputs under CPU XLA reads 0 in every
form. **Not yet shown on a GPU**: the next D3 job runs the magnitude piece by piece
(int16 abs, max, cast, times scale; int32- and f64-widened; with and without the
sort gather) on the captured inputs first, then the unchanged smoke gate.

### Owed

1. [DONE, sec. 27] The GPU reading of the fixed program and of the magnitude
   pieces.

## 27. D3 on a GB200, Vista 995264 -- the compiled migrate is bitwise numpy's; at a 4096^3 slab the eject is transfer and the insert is half compute; at cgh64 the host is 65% of the step

`d461cbe`, 2026-09-13, gb node c672-018, COMPLETED rc=0 in 5:08. Sbatch
`v2_d3_migrate_vista.sbatch`; cards `runs/v2/d3_insert_escape_diag_gb_forms.json`,
`runs/v2/d3_device_migrate_{gbsmoke,gb,gb_lowdrift}.json`, log copied to
`runs/v2/d3-migrate-995264.log`. Every arm its own process; the GPU smoke passed
all three arms before the long legs.

### The int16 magnitude, piece by piece (sec. 26's inputs, one GPU)

Rows differing from numpy of 15,898, each form its own jitted program:

| form | alone | after the sort gather |
|---|---|---|
| `abs(w)`, int16 | 0 | 0 |
| `abs(w).max(1)`, int16 | 0 | **5,293** |
| `... .astype(f64)` | 0 | **5,293** |
| `... * s_old` (the pre-fix kernel) | **5,293** | **5,293** |
| int32-widened `* s_old` | 0 | 0 |
| **f64-widened `* s_old` (the fix)** | **0** | **0** |

- **The int16 max over axis 1 is exact alone and wrong once another operation is
  compiled beside it** (a gather in front, or a multiply after), always the same
  5,293 rows. Widening to int32 or f64 first is exact in every arrangement. What
  the compiler does to the fused int16 reduction is not identified.
- **The fixed production program reads 0 scales different and abs_max 32767** on
  the inputs that escaped in 995067.

### Identity: PASS

`xback` at cdev (256^3, nb=16, drift 1.9 bricks), numpy eject + insert against
the compiled pair on the GPU:

| step | numpy | compiled | state and stats |
|---|---|---|---|
| 0 | 4.5 s | 5.7 s (compiles) | **bitwise** |
| 1 | 4.6 s | 2.8 s | **bitwise** |

Receipts: 32 eject and 32 insert calls; realized reach 2 on a non-all-to-all
schedule; 188,179 rows overflowed into the arena; 89,782 residents re-homed.
**The first GPU reading of the migrate against numpy.**

### cgh64, one compiled migrate step divided (nb=32, 4,194,304 rows per slab)

`eject_rows` / `insert_rows` return numpy, so each call's wall is upload + compute
+ readback, synchronous; host = step minus calls. Built in 30.8 s.

| step | numpy | compiled | eject calls (32) | insert calls (32) | **host** | new shapes (eject / insert) |
|---|---|---|---|---|---|---|
| 0 | 26.87 s | 17.51 s | 3.47 s | 4.12 s | 9.92 s | 2 / 2 |
| 1 | 27.14 s | **15.17 s** | 1.94 s | 3.39 s | **9.84 s** | 0 / 1 |

- **Bitwise the numpy migrate at both steps.** Reach 1, no overflow.
- **Steady, the compiled step is 1.79x the serial numpy one, and the host is 65%
  of it**: 308 ms per slab of host work against 61 ms of eject and 106 ms of
  insert calls. What the host 9.8 s is made of is NOT measured here; the per-brick
  Python loops in `_eject_slab_jax` (slot resolution, arena lookups, 32,768 bricks
  at ~300 us each) are a candidate, not a finding.
- A padded row count is a new shape: step 1 still compiled one insert program.

### One 4096^3 slab (65,536 bricks, 268,435,456 rows), synthetic rows

Medians of three; the split program's outputs equal the public call's, repeats
equal.

| | drift 0.5 brick (18.1% leavers) | drift 0.13 brick (5.0% leavers) |
|---|---|---|
| **eject, public call** | **0.941 s** | 0.947 s |
| prep / upload / compute / readback | 0.013 / 0.518 / **0.015** / 0.396 | 0.013 / 0.552 / 0.014 / 0.393 |
| **insert, public call** | **1.269 s** | 1.254 s |
| prep / upload / compute / readback | 0.013 / 0.202 / **0.628** / 0.437 | 0.013 / 0.208 / 0.606 / 0.439 |
| first call (compile) eject / insert | 11.2 / 11.0 s | 3.6 / 3.2 s (cache) |
| rows written by the insert, spilled | 251,015,983, 0 | 263,902,910, 0 |

Device peak 30.75 GiB after both kernels (the insert's own peak is not separated).

- **The eject is transfer: 15 ms of compute in a 0.94 s call.** The insert is
  half compute (0.61-0.63 s of 1.25-1.27).
- **The leaver fraction moves neither kernel** (0.941 vs 0.947; 1.269 vs 1.254).
- **Reproduction of the gb probe's eject (5y):** compute 14-15 ms (14.0), peak
  30.75 GiB (30.8), first call 11.2 s (10.6) reproduce. **End to end does not:
  0.94 s against 0.528 s**, with the upload alone 0.52-0.55 s. Not attributed.
- Each kernel captures a 2.15 GB constant (JAX warning at lowering). That is
  exactly `idx_all = jnp.arange(n_pad)` at 268,439,552 rows x 8 B, built outside
  the traced function in both kernels; a candidate for the warning, not traced.

### What this prices at c-hero -- arithmetic, not a measurement

- **Kernel calls from host-resident state, one GPU:** 256 slabs x (0.94 + 1.27 s)
  = **~566 s/step**, of which compute ~163 s and transfers ~400 s.
- **Host bookkeeping, if linear in rows:** cgh64's 9.84 s x 512 = **~5,000 s/step**
  serial. Linearity is not measured; rows and bricks both scale 512x at fixed
  4096 rows per brick, so a per-brick and a per-row cost cannot be told apart
  from this pair.
- Together ~5,600 s/step serial, against ~1,900 s/step for the pooled numpy
  migrate scaled the same way (cgh64 3.78 s x 512, scaling record). The compiled
  kernels inside the pool are unmeasured.
- **The migrate at 4096^3 is not a GPU-compute problem.** Its cost sits in the
  host bookkeeping and in moving slabs across the bus; GPU compute is ~3% of the
  serial projection.

### Owed

1. The composition of cgh64's 9.8 s host share (a profile of one compiled step),
   which decides between vectorizing the bookkeeping on the host and moving it
   onto the cards with the slab window.
2. The end-to-end eject gap to 5y (0.94 vs 0.528 s).
3. `idx_all` as a traced `arange` rather than a 2.15 GB captured constant.
4. Carried from sec. 25.

## 28. The compiled migrate's host share is mostly per-row, laptop profile at 256^3

`scripts/v2_d3_host_profile.py`, M4 laptop, CPU XLA, card
`runs/v2/d3_host_profile_laptop.json`. Particles FIXED at 256^3 (16.7M), bricks
varied, drift half a brick (reach 1 at every rung). One warm compiled migrate,
then one under cProfile with `eject_rows` / `insert_rows` timed; host = step minus
those calls. Profiled walls: cProfile inflates Python-call-heavy code, so the
per-brick term below is an upper bound.

| bricks per side | bricks | rows per brick | step | kernel calls | **host** |
|---|---|---|---|---|---|
| 8 | 512 | 32,768 | 3.86 s | 3.16 s | **0.71 s** |
| 16 (production 4096 rows/brick) | 4,096 | 4,096 | 4.02 s | 3.19 s | **0.83 s** |
| 32 | 32,768 | 512 | 4.37 s | 2.85 s | **1.52 s** |

- **Per-brick slope, each segment separately: 33 us/brick (512 -> 4,096) and
  24 us/brick (4,096 -> 32,768).** Not fitted as one line.
- **At production geometry the per-brick term is ~0.10-0.14 s of the 0.83 s
  host**; the rest, ~0.7 s or ~42 ns/row, does not move with the brick count.
- Largest own times in the host (nb=16): `_insert_slab_jax` 0.26 s (flat across
  rungs), `_eject_slab_jax` 0.23 s (0.21 / 0.23 / 0.38 s), `bucket_ijk_from_key`
  0.15 s over 4,096 calls (0.12 / 0.15 / 0.34 s). Both slab functions' own time is
  their numpy array work (gathers from and writes into the state, concatenations);
  which lines, not resolved. `numpy.asarray`'s 2.4-3.1 s is the kernels' results
  being forced, inside the timed calls.
- **At 4096^3 on these rates** (laptop, not Grace): per row ~6.9e10 x 42 ns =
  ~2,900 s/step, per brick 1.68e7 x 24-33 us = ~400-550 s/step. Against cgh64 on
  the GB200's Grace (sec. 27), the same rates give ~6.6 s where 9.84 s was
  measured: same order, different machine and occupancy.
- **Reading:** vectorizing the per-brick Python would remove at most ~15% of the
  host share at production geometry. The bulk is per-row movement of the state in
  and out of the kernels -- the host side of the same traffic the transfers carry.

### Owed

1. Line-level attribution inside `_insert_slab_jax` / `_eject_slab_jax` if the
   host path is kept.
2. The same rungs on Grace, if the laptop rates are ever used for a bill.

## 29. D3b R0 -- the insert holds ~80 B per padded input row on a GB200, the eject 114; repack projects to 430-630 s/step at 4096^3

Plan `~/.claude/plans/serene-nibbling-sparrow.md`, rung R0. Kernel change
`9b1c3d6`; probes `bef34e0`. GPU readings Vista 995435 (gb node c672-008,
COMPLETED rc=0 in 3:56), cards `runs/v2/d3_device_migrate_r0_peaks_{gbsmoke,gb}.json`,
log `runs/v2/d3-r0-peaks-995435.log`. Host readings laptop, card
`runs/v2/d3_replay_cost_laptop.json`. Each GPU arm its own process, so each peak
belongs to one kernel; the GPU smoke passed all three arms first.

### The kernels (`9b1c3d6`)

- **`_padded(n)` is `capacity_shape(n + 1)` at 12 rungs per octave** (<= 6%
  extra rows) instead of a multiple of 4096; the `+ 1` keeps a padded row in
  every call. `idx_all` is traced inside both kernels, not a captured constant:
  no captured-constant warning at 1M rows, where the old closure form warns
  (control run in the same process). Eject/insert/slot_state/executor tests 87
  passed, 1 skipped.

### Device peaks at a 4096^3 slab (65,536 bricks, 268,435,456 rows)

**Eject alone:** 30.30 GiB = **121.2 B/row = 114.4 B per padded row**
(284,397,459 padded rows).

| row count | padded to | new program | wall |
|---|---|---|---|
| 268,435,456 (the slab) | 284,397,459 | yes | 2.84 s |
| 267,630,149 (-0.3%) | 268,435,456 | yes | 2.30 s |
| 265,751,101 (-1%) | 268,435,456 | no | 1.89 s |

- **The nominal c-hero slab is exactly 2^28 rows, a rung of the ladder, so the
  `+ 1` pushes it one rung up and slabs just under it take the rung below: two
  programs.** Slab counts that straddle 2^28 will compile both; nothing more.
- **The walls are ~2x sec. 27's 0.94 s eject**, including the call that
  compiled nothing (1.89 s). A candidate, not traced: every call now pads, so
  `eject_rows` concatenates a full copy of each input on the host, where at an
  exact multiple of 4096 sec. 27's call padded nothing. Irrelevant to R1 (inputs
  built on the device); owed if the host path's wall is ever quoted.

**Insert alone**, input = keepers (1 - share) of a slab + (2r+1) x share of
emigrants:

| share | reach | input rows (slabs) | device peak | **B / padded input row** | B / slab row | 2nd call |
|---|---|---|---|---|---|---|
| 0.05 | 1 | 295,278,999 (1.10) | 22.28 GiB | 79.4 | 89.1 | 2.41 s |
| 0.05 | 2 | 322,122,543 (1.20) | 26.08 GiB | 82.8 | 104.3 | 2.75 s |
| 0.20 | 1 | 375,809,637 (1.40) | 29.36 GiB | 83.0 | 117.4 | 2.85 s |
| 0.20 | 2 | 483,183,819 (1.80) | 38.48 GiB | 81.5 | 153.9 | 3.73 s |
| 0.40 | 1 | 483,183,819 (1.80) | 38.48 GiB | 81.5 | 153.9 | 3.75 s |
| 0.40 | 2 | 697,932,183 (2.60) | **52.31 GiB** | 78.4 | 209.2 | 5.39 s |

- **The insert's peak is linear in its input: 78-83 B per padded input row**
  over 1.1-2.6 slabs. The 0.20/2 and 0.40/1 specs are the same input size and
  seed and read identical peaks.
- **The inputs are the host kernel's expanded form** (int64 destination, f64
  per-row scale). An on-device core with slab-local narrower indices (R1) is a
  candidate to lower both kernels' B/row; not measured.
- The emigrant share and reach at 4096^3 are NOT measured; the table brackets
  them. Both grow with box size and step size.

### Host: arena replay and repack (laptop, 256^3, bricks varied, arena-occupied)

| | 512 bricks | 4,096 (production 4096 rows/brick) | 32,768 |
|---|---|---|---|
| arena residents (bricks holding them) | 31,499 (243) | 90,268 (2,016) | 247,431 (15,344) |
| releases, every brick | 0.63 us/brick | 0.54 | 0.49 |
| claims, one call per spilling brick | 13.2 us/brick | 8.3 | 6.3 |
| **repack** | 0.095 s (5.7 ns/row) | **0.152 s (9.1 ns/row)** | 0.302 s (18.0 ns/row) |

- Built at zero brick slack with one numpy migrate so residents exist; wall
  clock, no profiler.
- **Claims were timed after all releases.** The real replay alternates them slab
  by slab, and each alternation dirties `_arena_free`, whose rebuild is a scan of
  the whole arena. That term is NOT in these numbers.
- Repack's per-brick slope, each segment separately: 16 us/brick (512 -> 4,096)
  and 5.2 us/brick (4,096 -> 32,768).

### At 4096^3 -- arithmetic, not a measurement

- **Repack: ~430-630 s/step** (Grace C16 cgh64 0.85 s/step / 134M rows = 6.3
  ns/row, and the laptop's 9.1 ns/row, each x 6.87e10). **5-7 h of a 40-step run
  on the host**, whatever the migrate does.
- Releases ~8 s/step (0.5 us x 1.68e7 bricks). Claims 6-13 us per spilling brick;
  the number of spilling bricks at 4096^3 is unmeasured. The free-list rebuild
  per alternation, at ~1 ns per arena slot (6.9e8 slots at 1%) x 256 slabs, could
  reach ~180 s/step -- the reason R1's replay must not rebuild it per slab.
- **Per-GPU envelope, one slab at a time, staging not included:** eject 32.5 GB;
  insert 23.9-56.2 GB across the bracket. Within the ~112 GB left beside the
  planner's resident terms (the envelope JC set, decision 2 below) every bracketed
  case fits. Within the planner's ~54 GB verdict headroom, which also charges the
  in-step phases summed, share 0.40 / reach 2 (56.2 GB) would not.

### Decided at the R0 checkpoint (JC, 2026-09-13)

1. **A GPU repack is high priority**, next after R1 and ahead of four cards and
   fusion.
2. ~~The device migrate is designed to fit beside the tile window (~54 GB per
   card).~~ **Revised the same day (JC): the envelope is ~112 GB per card**, 199 GB
   less the planner's resident terms (87.0 GB: slab window 43.5, coarse force
   shard 26.0, tile kernels 4.6, coarse prefactor 4.3, match factor 4.3, stream
   chunks 4.3). The ~54 GB offered first was the planner's verdict headroom,
   which also charges the coarse paint (32.0), coarse solve (17.4) and tile loop
   (8.4) transients summed; none of them coexists with the migrate, which runs
   after all three. The design still refuses by name above its envelope. The
   verdict's summed-phase convention and the underpriced `tile_workspace` (8.4
   charged, 13-16 measured) are unchanged in `inexor.plan`.
3. **The device and pooled migrates share one arena-replay helper**, both gated
   against the serial numpy migrate.

### Owed

1. The emigrant share and reach at production step size (they size the insert).
2. The replay with releases and claims interleaved, and its free-list rebuild.
3. The host pad copy in `eject_rows` / `insert_rows` (candidate for the ~2x wall).

## 30. D3b R1 on a GB200, Vista 995602 -- the device migrate is bitwise numpy's; a cgh64 step is 1.72 s against 27.6; its device peak is 438 B per slab row

`55bdbfb`, 2026-09-13, gb node c672-005, COMPLETED rc=0 in 3:38. Sbatch
`v2_d3_r1_device_migrate_vista.sbatch`; cards
`runs/v2/d3_device_migrate_r1_{gbsmoke,gb}.json`, log copied to
`runs/v2/d3-r1-migrate-995602.log`.

**What was built** (`eca3529`, `1f221b3`): `state._replay_arena_pass`, the pooled
migrate's replay moved verbatim and shared; `device.migrate.drift_and_migrate_device`
-- per slab one upload of the slot range, its arena residents and occupancy, a
device program enumerating rows in the reference order, the unchanged compiled
eject/insert kernels on device arrays, keepers/emigrants staged on the device,
written rows scattered into the slab's original bytes and one slice copied back,
releases and claims replayed on the host. Laptop gate 5 passed with five mutants
caught (skipped write-back, reversed source order, residents descending within a
brick, census off by one, one extra written row).

### Identity on the GPU: PASS

- `tests/test_migrate_device.py` + `test_insert_jax.py` + `test_eject_jax.py` on
  the GPU backend: **21 passed**.
- Probe smoke (32^3 arena state, 64^3): bitwise at both steps.

| arm | step | numpy | device | new programs | state and stats |
|---|---|---|---|---|---|
| cdev, zero slack, 30% arena, reach 2, ids | 0 | 4.41 s | 8.46 s | 5 | **bitwise** (overflow 89,782) |
| | 1 | 4.52 s | 4.31 s | 3 | **bitwise** (overflow 98,397) |
| cgh64, slack 0.10, 1% arena, reach 1 | 0 | 27.38 s | 6.36 s | 5 | **bitwise** |
| | 1 | 27.61 s | **1.72 s** | 0 | **bitwise** |

- **The steady cgh64 step is 1.72 s: 16x the serial numpy migrate and 8.8x sec.
  27's compiled-kernel host path (15.17 s)**, 54 ms per slab of 4.19M rows, on one
  card. Not attributed by phase.
- The cdev arena arm's step 1 still compiled 3 programs (the arena grew onto new
  ladder rungs), so neither of its readings is steady.
- **Device peak: 0.99 GiB (cdev), 1.71 GiB (cgh64) = 438 B per slab row at
  cgh64.** R0 put the kernels at 114 (eject) and ~80 (insert) B per padded row,
  so most of the peak is arrays the driver holds at once. Candidates, by
  arithmetic on the code and NOT measured: the rows program's outputs kept alive
  through the eject (~50 B/row), three staged slabs of eject outputs at reach 1
  (dest, off, w, src, plus a retained int64 `d_slab`: ~33 B/row each), the
  windows of pending inserts (~9 B/row each), and the insert's compacted buffers
  (~25 B/row) beside its workspace.

### At 4096^3 -- arithmetic, not a measurement

- **Memory: 438 B x 268M rows = ~117 GB per card if the peak scales with slab
  rows, against the ~112 GB envelope (JC, sec. 29 as revised).** R1 does not fit
  the envelope as built.
- **Wall: 1.72 s x 512 = ~880 s/step on one card if linear in rows.** Per-slab
  fixed costs and per-row costs are not separated, so this is not a projection.

### Owed (R1 does not close until these are read)

1. The device peak attributed by what is held, then the driver's retention cut to
   the envelope (free rows outputs after the eject, compute destination slabs on
   demand, stage only what the insert reads), with a refusal naming the envelope.
2. The cgh64 step wall split by phase (upload, device programs, readback, host
   index work, replay), which a 4096^3 projection needs.

## 31. D3b R1 re-run, Vista 995638 -- still bitwise; the cgh64 peak is 320 B per slab row after the retention cuts; the step splits 65% device, 32% bus and host

`2e8ce24`, 2026-09-13, gb node c672-002, COMPLETED rc=0 in 4:04. Sbatch
`v2_d3_r1_device_migrate_vista.sbatch` (re-run); cards
`runs/v2/d3_device_migrate_r1cut_{gbsmoke,gb}.json`, log copied to
`runs/v2/d3-r1-migrate-995638.log`. Sec. 30's `_r1_*` cards are the before.

**What changed since sec. 30.** Retention cuts (`7ccfd95`, laptop probe
`scripts/v2_d3_retention.py`: held 248 / 287 / 324 -> 150 / 181 / 169 B per slab
row at the eject / insert / write-back calls at 256^3): row buffers freed after
the eject, destination slabs computed on demand, a staged slab cut to its
emigrants after its own insert, the insert's compacted inputs freed before the
write-back. `timings=` synced phase split and spill rows gathered on the ladder
(`2e8ce24`). Laptop gate 5 passed and the five mutants caught after each change.

### Identity on the GPU: PASS

GPU-backend pytest (device migrate + insert/eject gates) **21 passed**; smoke
bitwise; both long arms **bitwise at all three steps**, the synced one included.

| arm | step | numpy | device | new programs | device peak (sec. 30) |
|---|---|---|---|---|---|
| cdev, zero slack, 30% arena, reach 2, ids | 0 | 4.52 s | 5.90 s | 10 | |
| | 1 | 4.61 s | 1.74 s | 4 | |
| | timed | 4.64 s | 1.05 s | -- | **0.77 GiB** (0.99) |
| cgh64, slack 0.10, 1% arena, reach 1 | 0 | 27.14 s | 6.42 s | 8 | |
| | 1 | 27.19 s | **1.63 s** | 0 | |
| | timed | 27.28 s | 1.66 s | -- | **1.25 GiB = 320 B/slab row** (1.71, 438) |

- **The cgh64 peak fell 27% (438 -> 320 B per slab row)** and the steady step
  1.72 -> 1.63 s. Synced timing moved the cgh64 step by 0.03 s.
- The cdev arena arm is not comparable per row (ids, reach 2 so five source slabs,
  30% arena) and was still compiling at step 1; its synced step is 1.05 s.

### Where the cgh64 step goes (synced, 1.66 s, 32 slabs)

| phase | s | share |
|---|---|---|
| insert kernel | 0.477 | 29% |
| insert: compact inputs | 0.215 | 13% |
| insert: slot range to host | 0.213 | 13% |
| eject: upload | 0.163 | 10% |
| eject: host index + window | 0.134 | 8% |
| insert: census (per-source counts) | 0.118 | 7% |
| insert: shrink staged | 0.076 | 5% |
| eject: reach + scalars | 0.071 | 4% |
| eject: rows program | 0.063 | 4% |
| write program + scalars | 0.037 | 2% |
| occupancy + scales to host | 0.027 | 2% |
| setup, eject kernel, final census, replay, spills | 0.063 | 4% |

- **Device programs 1.07 s (65%); host and bus 0.54 s (32%); replay and
  bookkeeping 0.05 s (3%).**
- **The insert kernel is 28x the eject kernel** (0.477 vs 0.017 s), as at c-hero
  (sec. 27: 0.63 vs 0.015 s per slab).
- **Census and reach (0.19 s together) are eager counts each ended by a host
  readback**, three per insert and one per eject; folding them into the programs
  is a candidate saving, not measured.

### At 4096^3 -- arithmetic, not a measurement

- **Memory: 320 B x 268M rows = ~86 GB per card, inside the ~112 GB envelope**
  (sec. 29 as revised), if the peak scales with slab rows.
- **Wall, one card:** the insert kernel measured at a c-hero slab is 0.63 s
  (sec. 27), 256 slabs = ~160 s; every other cgh64 phase scaled x512 (rows per
  slab x64, slabs x8) is ~600 s. Order **~500-800 s/step on one card**; the
  device-program phases below 4.2M rows per slab are unlikely to scale linearly,
  so the upper end is the linear bound. Four cards are unmeasured.

### Status of R1

Done: the bitwise gate on a GB200, cgh64 wall against 15.2 s (1.63 s), device
peak against R0 and the envelope (320 B/slab row, ~86 GB by arithmetic).

**The envelope refusal, added after this job (laptop only).**
`drift_and_migrate_device(device_budget_bytes=)` estimates each slab's device
footprint before its eject and its insert -- every array the pass already holds,
plus 114.4 (eject) or 83.0 (insert) B per padded row, the R0 kernel peaks with
inputs included -- and refuses with the budget, the held bytes and the kernel term
named. The largest estimate is reported on the receipt with or without a budget.
Tests: a 1 KB budget refuses with the state untouched; a generous budget is bitwise
and reports its estimate (7 passed; the five mutants still caught).
**Not validated: the estimate against a measured device peak** -- owed to the next
GPU job, which reads both in one process. **Moved to R3 (JC):** the permuted
slab-order arm, which the four-card boundary-first schedule is what needs.

## 32. D3b R1b + R2 on the laptop -- the repack on the device and the engine routed to the device passes, both bitwise; one bundled gb job proposed

`f593d9c` (R1b), `e888ec8` (R2), `bffd8dd` (empty-slab skip), 2026-09-13. Laptop
only (CPU XLA); the GPU readings ride ONE job, `scripts/v2_r2_device_engine_vista.sbatch`
(JC, 2026-09-13: stop profiling one rung per job).

### R1b: `device.repack.repack_device`

Per x-slab: the slab's old slot range and its arena residents go up once; one
program computes every row's destination WITHOUT a sort (a live row of rank i in
bucket w lands at `new_start[b] + i + n_res(b, bucket < w)`, a resident at
`new_start[b] + occ_cum[b, w] + j` with j its rank among the brick's residents
sorted by (bucket, slot), both from one searchsorted over the residents sorted by
bucket), scatters the slab's whole new block (spare zeroed, ids -1), bincounts the
new occupancy, and the block comes back as one slice. A block write can overrun a
later slab's OLD range, so every later slab it intersects is uploaded first (a
read-ahead window, on the receipt). The arena cannot be overrun: the allocation
refusal bounds `n_alloc_new` by the old `arena_base`. Empty slabs are skipped
(`empty_slabs` on the receipt); a one-slab probe state has 255.

**Gate** (`tests/test_repack_device.py`, 7): bitwise the host `repack` on every
field and its return dict (`bricks_fast` / `bricks_merged` included) with a
populated arena, an overrunning write (read-ahead fired), repeated migrate+repack
steps without ids, a quarter-filled box (empty slabs), and residents PLANTED
below their brick's live rows -- the engine's own spill only ever parks a brick's
highest-bucket rows, so no migrated fixture exercises the lower-bucket term, and
the first mutant (dropping it) survived until that fixture existed. Three mutants
caught after it.

**Unmeasured:** device bytes per row of the program (the receipt reports what the
driver holds: windows + block), and the wall at a 4096^3 slab.

### R2: `EngineConfig(migrate_backend="device")`

One router each for the migrate (step and lead drift) and the repack; receipts
`migrate_backend`, `migrate_device` (None on the host) and `repack["repack_device"]`
on every card; `migrate_device_budget_bytes` reaches the envelope refusal;
`validate()` refuses an explicit `migrate_pooled=True` beside it and the device
backend without x64. A tile pool may still drive the tile loop; the device pass
ignores it (on a GPU node the pool refuses anyway).

**Gate** (`tests/test_engine_device_backend.py`, 6): a K=3 run bitwise the host
run with a repack every step and the receipts in both directions (device passes
ran 4 + 3 times, the host arm touched none); the same beside a two-worker tile
pool; a split-run resume bitwise the uninterrupted device run and the host run;
the budget knob reaching the pass and refusing; the three refusals.

Side change, bitwise: the lead drift's pooled call now forwards
`migrate_eject_inflight` like the step's.

### Planner: the device passes priced, the summed convention untouched

Under `--backend device`, `step_bytes` charges the HOST for what the device
passes actually build there -- two slab-sized numpy buffers each (window +
block, 9 B/row, from the code) -- instead of the host migrate staging and
repack scratch. The per-GPU column gains an AFTER-THE-TILE-LOOP table held
apart from the in-step phases (JC, sec. 29: the migrate runs after paint, solve
and tile loop have released their transients): the migrate at 320 B per slab
row (sec. 31) and the repack at 3 slab windows (program scratch UNMEASURED,
labelled so), max of the two, with its own verdict against `resident` alone.

| c-hero on a gb node | before (sec. 11) | now |
|---|---|---|
| host, lower bound on the run's peak | 956.3 GB (0.93x), binding | **874.2 GB (0.85x)** |
| the LOAD stage | 892.0 GB (0.87x) | 892.0 GB (0.87x), **binding again** |
| per GPU, resident + worst in-step phase | 144.8 GB (0.73x) | 144.8 GB (0.73x) |
| per GPU, resident + after-the-loop | -- | **172.9 GB (0.87x)**, migrate 85.9 |

The host drop is the 51.0 GB migrate staging and the 40.6 GB repack scratch
leaving, less 2 x 4.8 GB of windows. The after-the-loop figure is ARITHMETIC on
320 B/slab row; sec. 31 says the device-program phases below 4.2M rows per slab
are unlikely to scale linearly, in either direction.

### The bundled job (proposal, not submitted)

`scripts/v2_r2_device_engine.py` + `_vista.sbatch`: GPU pytest (5 files); the
probe smoke; engine A/B at cgh64 K=2 (device backend first, host second, each
its own process, every state field hashed and the stripped stats compared by the
orchestrator); migrate-budget (R1's owed estimate-vs-measured, one process);
repack-slab (a 268M-row slab at nb=256 with 4,096 residents planted, device
repack bitwise the host, synced phases, device peak). Laptop smoke: all four
arms rc=0, R2 gate hashes and stats equal. Time ask in the sbatch header: ~25
min, 45 min wall, the tile loop at cgh64 the unmeasured bound.
