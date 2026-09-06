# The host-state / device-step design: 4096^3 on one Vista gb node

Opened 2026-09-06, branch `jc/device-step-4096`. This is the design record that
`m6_scaling_record.md` sec. 5y "Owed" names. Its companion is that record's
sec. 5y and 5z, which price the design and are not repeated here.

**Status: priced; one rung built and measured (D1, secs. 7-8).** Sections 1-6 are
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

One transform moves ~275 GB across host<->device (pass 1 reads the 34.4 GB
field and writes the 34.4 GB spectrum; pass 2 reads and writes it again; both
directions). At 31.3 s that is **8.8 GB/s effective, against the 201 GB/s this
same hardware streams from PINNED host memory** (974476, sec. 5z) -- a 23x gap.

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
