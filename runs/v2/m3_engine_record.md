# M-v2-3 exit gate: the engine IS the ratified force plus a codec

**Result, and it is a pass on all three parts.** Branch `jc/m-v2-3-engine-core`,
Vista jobs 898129 (cancelled), 898169 (3 of 4 legs) and 898242 (the fourth),
`gh` nodes, ~1.5 node-hours total. Cards `runs/v2/m3_gate_*.json`. Nothing here
amends a ratified decision; D-v2-21 is drafted from it.

## The gate

D-v2-18's row for M-v2-3 reads "correctness vs the v1 parity arms where configs
overlap". **No v1 parity configuration overlaps a v2 one** -- the v1 arms are
single-level gravity at 2.0-4.0 Mpc/h with one particle per force cell, every v2
configuration runs 0.25 Mpc/h at two force cells per particle, and D-v2-9's
transferability argument is grounded on that fine cell. The v1 *quantized* arm
cannot be re-measured at all: its codec was deleted at the retirement and T9 is a
different tier in kind. JC replaced the row with a three-part gate (2026-08-08).

### Part 1 -- the engine's force is BITWISE the ratified path's

| leg | particles | elements compared | differing | max abs delta | oracle peak |
|---|---|---|---|---|---|
| cdev8 | 2,097,152 | 6,291,456 | **0** | **0.0** | 2.352 |
| **cgh64** | **134,217,728** | **402,653,184** | **0** | **0.0** | 3.136 |

`partition_ok` true and `n_overhang` 0 on both; oracle non-vacuous at both.
cgh64 is T=256/b=32, the configuration every D-v2-10, D-v2-11 and D-v2-12 number
was measured at, and the same 402,653,184 elements the M-v2-2 gate compared.

**What makes this a different claim from M-v2-2's.** That gate had to drive both
arms from ONE membership, because the layout's tile members are the same SET in a
different ORDER and order changes an f64 scatter-add sequence. This gate
deliberately does not: the engine reads positions out of slot-ordered T9 state
and drives the force from brick spans, the reference reads a global array and
drives it from the probe's own bucketing, and the two agree to the bit. That is
only possible because both paints became integer in this milestone -- it is a
direct consequence of the D-006 work, not an independent result.

### Part 2 -- the codec's cost on the architecture that ships

| config | K | max \|dP/P\| in band | median | margin under 3e-2 | engine wall |
|---|---|---|---|---|---|
| cdev8 | 40 | 4.666e-04 | 1.195e-04 | 64.3x | 728 s |
| **cdev** | **40** | **7.125e-04** | 2.456e-04 | **42.1x** | 2035 s |

Band k <= 2.513 h/Mpc (0.2 k_Nyq of the fine mesh), D-v2-9's.

**PRE-REGISTERED before the run**, in the probe's docstring and the sbatch
header: "cdev K=40 lands within a factor of a few of 4.123e-4 and at least an
order under D-v2-9's 3e-2. A miss is a finding." Measured 7.125e-4 = **1.73x**
the ratified figure, 42x under the bar. The pre-registration holds.

**Why this is a new measurement and not a repeat.** D-v2-14's ratified 4.123e-4
came from `v2_g2c_accum_gate.py`, which quantizes in FLOAT SPACE against a
MONOLITHIC force. It never saw the two-level force, the real storage layout, or a
reordering particle sequence. This runs the actual engine against a
never-quantized driver of the same drift-synchronized shape, so the codec is the
only difference between the arms.

### Part 3 -- the v1 arms as a regression on the shared code

| arm vs `mbody_final` | config | rms cells | max \|dP/P\| |
|---|---|---|---|
| `inexor_float` (STORED) | n64k10 | 4.196e-06 | 1.109e-06 |
| `inexor_float` (STORED) | n128k40 | 1.628e-05 | 1.380e-06 |
| `inexor_replumb` (today) | n64k10 | 4.196e-06 | 1.109e-06 |
| `inexor_replumb` (today) | n128k40 | 1.628e-05 | 1.380e-06 |

D-013's bars are rms <= 1e-4 cells and |dP/P| <= 1e-5; both arms clear both at
both configurations, and the stored arm reproduces D-013's recorded 1.6e-5 /
1.4e-6 **to the digit**.

**The re-plumbed driver reproduces the stored v1 reference EXACTLY** -- rms
0.000e+00 cells, max |dP/P| 0.000e+00, with 1-r at 2-3e-16, the estimator's own
self-comparison floor. So the shared force/paint stack has not moved since v1,
and D-013 is re-runnable rather than only re-readable.

**This took a wrong turn first, and the correction is the useful part.** The
re-plumb initially used `paint="int"`, the deterministic integer paint, and
missed the stored reference by 5.320e-05 cells with 1-r = 1.75e-08 -- small,
systematic, and reported here as an unexplained drift in shared code. It is
not. `evolve_float`'s signature (recovered from git at `450f468^`) is
`(..., paint="f32", fdtype=jnp.float32)` and `m1_parity.py` never passed
`paint`, so **the stored v1 reference used the differentiable FLOAT CIC paint**.
The integer paint quantizes corner weights to 2^-12, which is a real
mass-assignment difference of exactly that size and shape. Kernel dtype had been
excluded by measurement and was never the variable.

Two things worth keeping from the wrong turn: a decayed default is invisible in a
call site that omits the argument, so reconstructing a deleted function means
reading its SIGNATURE and not just its body; and "unexplained discrepancy in
ratified shared code" was the expensive hypothesis to leave standing -- the
deleted function was in git history the whole time and the check cost minutes.

## What this does NOT license

- **Nothing at C-gh.** Part 1's instrument compares two forces elementwise, which
  is an O(N) array -- 3.0 GiB at cgh64 and 206 GB at C-gh. **cgh64 is this gate's
  ceiling in this form**; a hero-scale parity check must compare per tile and
  accumulate statistics rather than arrays.
- **No performance claim.** The walls above are budget, not measurement, and
  nothing here is axis-matched against the probe.
- **No statement about the RSD bar** (D-v2-8 clause 5 still carries retired
  language) and none about f32 meshes (M-v2-4, its own gate).
- Part 2 is one seed at each configuration.

## Defects in my own instruments

Seven, and the pattern is that five were caught by a guard, a paired test or a
control rather than by inspection.

1. **`check()` as first written was a gate that cannot fail.** Decode a slot,
   assert its position falls in the bucket the slot implies -- an identity, since
   the offset is a uint8 stored relative to that bucket, so the recovered bucket
   is the same one for every byte value. It passes on arbitrarily corrupted
   offsets. Replaced by a structural check; the vacuity is pinned by a test.
2. **The exchange moved payload between slots and not the id column**, so every
   id pointed at whoever previously occupied the slot. Caught by the id-based
   destination test -- which is what D-v2-14 clause 5's opt-in tier is for here.
3. **Arena residents were re-homed twice** -- `decode_brick` returns them, so
   ejection already handles them, and insert pulled them in again. 32,771
   particles reachable against 32,768 stored.
4. **`cap` was sized from the brick run length**, excluding arena residents that
   `decode_bricks` returns. Same class as D-v2-19 clause 4's 98.4% loss, but it
   under-sized capacity and the force refused, which is the good version.
5. **The accumulated-quantization reference used the MONOLITHIC force**, so it
   measured the split error plus the codec -- and D-v2-9's bar IS the split-error
   bar. It reported 6.96e-2 at `smoke`. With the reference on the same two-level
   force: 1.004e-3.
6. **The cgh64 leg hit D-v2-16 clause 1's own refusal** on the gate's O(N)
   comparison array. The guard was right; raised at that one call site.
7. **The re-plumbed v1 arm used the wrong paint** (`int` where `evolve_float`
   defaulted to `f32`), and I reported the resulting 5.3e-5 offset as an
   unexplained drift in ratified shared code. It was my harness. Caught by
   reading the deleted function's SIGNATURE out of git -- which the control had
   already pointed at, since the stored arm reproduced D-013 to the digit and
   therefore located the problem on today's side, not in the stored data.

Two fixtures also could not reach the regime they claimed (a crossing test whose
drift never cleared a bucket, and an overflow test converging on the box centre,
which with 2 bricks per side is the corner where all eight meet and is therefore
balanced).

## Performance, measured because a job died of it

Job 898129 was cancelled at 34:48: the engine ran 183-226 s/step at cdev8 where
33 s/step had been measured locally, so the cdev leg extrapolated to ~18 h
against a 12 h wall. **I guessed the cause wrong (single-threaded host numpy) and
profiled instead.**

| fix | cdev8 step |
|---|---|
| start | 32.65 s |
| pad the long-range read to one XLA shape | 13.21 s |
| mask the streamed coarse paint | 11.43 s |
| group the arena once instead of per brick | **10.04 s** |

The first was 2,107 XLA compilations in a single step: the long-range read is
called once per tile with that tile's own row count, so every tile keyed a new
shape -- exactly the trap `make_tile_force_fn` documents for the short arm and
which the short arm avoids with `cap`. The third was M-v2-1's third instrument
defect again: an O(n_arena) scan for one brick's answer, asked per brick.

**cdev measures 35.61 s/step, not the ~80 s particle-count scaling gives**, because
decode redundancy is set by the tile-to-brick ratio: cdev8 decodes 64 bricks per
8-brick tile core (8x), cdev about 2x. Extrapolating cdev8 by particle count would
have repeated the first job's mistake in a new disguise.

**REFUSED, on a measurement.** Jitting the gather is the largest remaining term
(4.85 s of 11.43 s) and worth ~1.7x. Under jit XLA reassociates the corner
accumulation and the result stopped being bitwise the global gather -- 86 of 189
elements at 2.220e-16. Physically irrelevant, fatal to the parity gate, and the
same trade that function's docstring already records rejecting at 8.9e-16.
Reverted; the contract is pinned by
`test_jitting_the_subblock_gather_would_break_its_bitwise_contract`.

## Owed

- `SlotState.repack` allocates O(N); D-v2-19 clause 3 establishes the monotone
  in-place form at O(chunk). Correct at development configurations, 91 GB of
  transient at C-gh.
- `cap_probe` 4,335,616 vs `cap_engine` 4,335,570 at cgh64 -- the two membership
  computations size their capacity 46 apart in 4.3M while the force is bitwise
  identical, so it is not about which particles are included. Not chased.
- The host exchange and per-tile decode are ~5.6 s of the 10.04 s step.

## Cost

Three Vista `gh` jobs: 898129 cancelled at 34:48, 898169 at ~1 h, 898242 at
1 min 55 s. Roughly 1.5 node-hours, of which the cancelled job is 0.6.
