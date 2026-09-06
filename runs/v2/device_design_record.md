# The host-state / device-step design: 4096^3 on one Vista gb node

Opened 2026-09-06, branch `jc/device-step-4096`. This is the design record that
`m6_scaling_record.md` sec. 5y "Owed" names. Its companion is that record's
sec. 5y and 5z, which price the design and are not repeated here.

**Status: designed and priced, nothing built.** Every number below is
arithmetic over a design or a reading carried across from 5y/5z. The one thing
this record adds that was not in either is the MEMORY budget, and it changes
what the binding constraint is.

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

1. The plane-factorized device FFT (also the largest per-GPU phase, sec. 2).
2. A decision on `slack` / `alloc_margin` at c-hero (sec. 2, item 3).
3. The IC stage at 4096^3: a machine and a policy (sec. 3).
4. This record's budget re-read against a measurement, the first time any part
   of the device pipeline runs at scale.
