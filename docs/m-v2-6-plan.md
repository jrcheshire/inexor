# M-v2-6 -- capacity: a complete 2048^3 dark-matter mock on one node

**Status:** OPEN. Branch `jc/m-v2-6-capacity`, no PR. The memory removals are
DONE and production now fits a CPU-only node on the arithmetic floor; **the
binding constraint has moved to wall**, where it is ~90x off this plan's own
bar. See the ledger below.

**Provenance.** This refreshes the planning session of 2026-08-11, whose original
lives at `~/.claude/plans/enumerated-foraging-wadler.md` and is superseded by this
file rather than edited. The design, the rejected alternatives and the gates are
carried over unchanged; what moved is recorded in "What changed" below. The
measurements are NOT here -- they live in `runs/v2/m6_peak_record.md` (memory)
and `runs/v2/m6_scaling_record.md` (time).

## The gate, re-scoped

The build ladder's written criterion (`docs/decisions.md:942`,
`docs/plan-plan-v2.md:408`) reads in full:

> | M-v2-6 | capacity | **a complete 2048^3 mock on one Vista gh node** |

No clause, no bar, no definition of "complete" or "mock". It is the fourth
milestone running whose written criterion could not be read literally. Re-scoped
with JC on 2026-08-11:

- **Product = dark matter only, persisted.** ICs -> K steps -> the evolved state
  written to disk -> a large-scale P(k) computed without ever building a
  full-size array. Halos, bias, HMF, b1 and the disco-mocks read-back stay in
  M-v2-7 (`decisions.md:943`), which a literal reading would swallow.
- **Not presumed to be a `gh` node.** The written row names one; the machine is
  Stage 1b's verdict, and the arithmetic below now bears on it directly.
- Completion + memory + wall + SU is the capacity statement; accuracy comes from
  the Stage 3 streaming parity number.

The re-scope needs its own ADR (see Stage 6); it is not yet written.

## Status ledger

The stage numbers are this document's internal labels and mean nothing outside
it. The "what" column is the real name of each piece of work; use that when
talking about it.

| stage | what it actually is | status |
|---|---|---|
| 0 | build a tool that measures the engine's total memory, which nothing did | **DONE** + a Stage 0b the plan did not anticipate |
| 1a | check whether thread count explained deneb beating a GH200 | **DONE, answered: no.** The step is serial |
| 1b | time the same run on several machines to decide where production goes | **NOT RUN.** Needs a Slurm proposal, and see the note below |
| 2a | remove the 275 GB velocity array via per-brick scales | **BUILT 2026-08-14** (`8179cec`); accuracy checkpoint owed |
| 2b | remove the scratch buffer in the periodic re-layout | **DONE 2026-08-14** (`c647e8e`): 11.1 -> 2.1 B/row measured, 115.4 -> 21.8 GB |
| 2c | stop building a full-size mesh for every small chunk of particles | open, premise re-verified 2026-08-14 |
| 2d | lower the memory spike while loading ICs from disk | open |
| 3 | rewrite the correctness check so it needs no 206 GB array | open |
| 4 | write the output: save state, compute P(k), budget the disk | open |
| 5 | do the capacity runs | open, gated on 1b + 2 |
| 6 | write the record and the re-scoping ADR | open |
| P | portability (`inexor.plan`, parameters, running-elsewhere) | **1 of 3 done** |

**THE BINDING CONSTRAINT IS NOW WALL, NOT MEMORY.** Production fits a CPU-only
node on the arithmetic floor (see the next section), but cgh64 runs at 608.67
s/step, which extrapolates to ~11 h/step and ~18 days per realization at the
ratified K=40 -- against this plan's own "under 5 h each" test, a ~90x gap. The
largest untested lever is that the step is SERIAL (the thread sweep was flat
from 1 to 32 threads) while tiles and bricks are independent by construction, so
tens of cores sit idle. That is the same order as the gap and it is a hypothesis,
not a plan: job 463 is timing the phases, because nothing has, and today has
twice punished acting on a derivation that had not been measured.

**A wall-clock problem surfaced on 2026-08-14, was attributed, and is FIXED.**
Job 455 measured cgh64 (512^3) at **2622.6 s/step**, 3.3x past its own
pre-registered band, so nothing was sized from it. `_insert_slab` was scanning
every row of a slab once per brick -- `N x n_bricks^2`, i.e. N^(5/3). A one-axis
arm confirmed it (job 456: particles fixed, insert 3.793 -> 10.756 -> 39.626 s
as bricks per side went 8 -> 16 -> 32, with the last rung predicted at 38.6 s
before it ran), the fix landed (`10d1a2d`), and the end-to-end confirmation
(job 459, identical to 455) reads **608.67 s/step -- 4.31x, with the scan
accounting for 77% of a whole engine step.** Full numbers, including a memory
prediction that MISSED its band and a confounded comparison, are in
`runs/v2/m6_scaling_record.md`.

**Is the step linear in N now? Effectively yes.** Job 460's matched-knob cdev
point reads 48.53 s/step against cgh64's 608.67 -- 12.54x for 8x the particles,
an exponent of 1.22. That is NOT an algorithmic term: staging depth was not held
fixed between the two runs (reach [1,2,1] against [3,3,2]), and 8 x 1.73 = 13.8x
brackets it with nothing left over. Depth grows because a larger box carries
faster particles, which is physics, not code. Job 461 confirmed the migration is
linear in N once depth IS pinned. Details and the confound in
`runs/v2/m6_scaling_record.md` section 5c.

Discharged along the way, and not to be re-proposed:

- Stage 0's `bytes_per_particle` accounting hole (the `alloc_margin` allocation,
  ~0.99 B/p) is **closed**: `state.py:1100-1125` counts it in `slack`, and
  `inexor.plan` reports it as `slack + alloc_margin`. It is why the all-in figure
  reads 11.47 B/p where D-v2-20 recorded 10.54.
- **Portability item 1 is built**: `python -m inexor.plan` shipped 2026-08-11 and
  was corrected from source on 2026-08-13. Items 2 (every budget a parameter) and
  3 (`docs/running-elsewhere.md`) are still open.
- Stage 0's differencing arms are **retired**: a maximum carries no timestamp, so
  differencing two maxima assumes both were set by the same phase and nothing
  checked it. Do not re-run them and do not quote their numbers.

## The memory arithmetic, and what it says about the target node

`python -m inexor.plan --preset c-gh --cap 5284492`, at the branch head. The
lower bound is `state resident + mesh resident + every per-particle per-step term
+ the single largest mesh transient` = **511.2 GB**.

| block | GB | note |
|---|---|---|
| state, resident | 98.5 | 11.47 B/p all-in |
| mesh, resident through the tile loop | 18.0 | |
| largest single mesh transient | 12.9 | `coarse_kernel_build_f64` |
| `kick_pending` | **274.9** | REMOVED (`8179cec`) |
| `repack_scratch` | **93.5 -> 115.4 measured -> 21.8** | REWRITTEN (`c647e8e`) |
| `migrate_staging` | 12.8 | N^(2/3), corrected 2026-08-13 |
| `tile_buffers` | 0.6 | |

**Removing the largest term was necessary and not sufficient**, which is worth
keeping because a reading that stopped there would have concluded the opposite.
It took both removals to reach a node:

| after | lower bound | vs gh (116 GB) | vs gg (237 GB) |
|---|---|---|---|
| at the start of M-v2-6 | 511.2 GB | 4.41x | 2.16x |
| the velocity array removed (`8179cec`) | 236.3 GB | 2.04x | 1.00x -- at the edge |
| the repack coefficient MEASURED, not derived | 258.2 GB | 2.23x | 1.09x |
| **the repack rewritten in place (`c647e8e`)** | **164.6 GB** | **1.42x** | **0.69x -- FITS** |

**Production fits a CPU-only node, and that is new.** Two removals did it: the
274.9 GB velocity array, and the repack scratch at 115.4 -> 21.8 GB. The middle
row is worth keeping -- correcting the repack coefficient from a DERIVED 9 B/row
to a MEASURED 11.1 made the picture temporarily WORSE, which is what an honest
accounting does.

**The largest single term is now `t9_payload` itself**, so there is nothing
further to remove that is not the simulation; any further reduction is a codec
question. A `gh` node remains out of reach and the reason is structural rather
than incidental: state plus resident mesh alone clears 116 GB.

Standing caveat: this is an arithmetic lower bound, measured to read ~1.9x low
against the one configuration where it has been checked. **Fitting on paper is
not fitting**, and the capacity run is what would settle it.

Two consequences the 08-11 plan could not have drawn:

1. **A `gh` node cannot host C-gh under this arithmetic even after 2a and 2b.**
   State alone (98.5) plus mesh resident (18.0) is 116.5 GB against a 116 GB hard
   cliff, before a single transient. So the written charter's "one Vista gh node"
   is not merely unratified, it is arithmetically out of reach without further
   removals -- and D-v2-13 measured that cliff as a ~100x collapse in one 4 GB
   step, not a soft edge.
2. **Stage 1b's verdict is partly forced by memory rather than by SU.** The plan
   framed 1b as a charging question with a 3x wall bar. On these numbers gg's
   237 GB is the only Vista arm C-gh fits on at all after 2a+2b. The SU
   comparison still decides gg against a Stampede3 x86 arm; it no longer decides
   gg against gh.

Caveats, stated because the tool states them: this is a lower bound from
arithmetic, not a measurement; it assumes one transient peaks at a time; and the
engine's true peak has been measured at development scale only. `kick_pending`
and `repack_scratch` are summed as co-resident, which is correct today --
`pending` is dead after the reconciliation loop but stays REFERENCED until `step`
returns (`engine.py:764`), which spans `drift_and_migrate`.

## Stage 2a -- the 275 GB term (BUILT 2026-08-14, `8179cec`)

**Done, gate green (429 passed / 1 skipped, determinism tier 16, lint clean).**
The measured effect is in the planner: C-gh's lower bound went 511.2 -> 236.3 GB
and the largest single term is now `repack_scratch` at 93.5 GB. What replaced
the array is 16.8 MB, counted in both the container's own figure and the
planner's rather than described.

**The design below had a hole and the build is where it is handled.** Under one
global scale `_rescale_w` could not overflow int16 and said so as a theorem: the
scale was a max over a PARTITION. Per brick that is false -- a fast particle
drifting out of a dense brick into a quiet one needs more range than the quiet
brick's own maximum provides -- and D-007 forbids the clamp, so a wrap is silent
corruption rather than imprecision. The plan's "no second rounding is introduced
anywhere" was wrong for migrants for the same reason.

The obvious fix does not work: covering a brick's neighbours would bound who can
arrive, but computing it needs every neighbour's new velocities before any of
them are encoded, and holding those IS the 275 GB array. So a brick's scale is
fixed in `_insert_slab`, the first and only point where its full post-migration
membership exists. Ejection no longer rescales; it stages each emigrant's source
brick (4 B, emigrants only) so the insert can take a true max and express
everything at it in one rounding. `_rescale_w` now refuses out of range instead
of asserting it cannot happen.

**Still owed:** the accuracy checkpoint below. It can move either way -- scales
are up to ~2.5x finer, and migrants now see two roundings where one global scale
gave them one -- so it needs the anchor run rather than an argument.

The rest of this section is the design as ratified, kept for its rejected
alternatives.

Generalize `vel_scale` from a scalar to one f64 per brick (`n_bricks` x 8 B =
16.8 MB at C-gh). Each brick's velocities are quantized **once**, at its own
tile's scale; decode uses that brick's scale. `pending` then disappears, because
nothing needs to wait for a global reduction, and no second rounding is
introduced anywhere.

It is also *more accurate* than today: measured per-tile scale ratios are r_p50
0.40-0.80 and r_p99 0.90-0.99 of the global max, so per-brick quantization is up
to ~2.5x finer and recovers the ~1.12x on velocity RMS that M-v2-3 measured
losing.

**Call sites, re-verified against the branch head 2026-08-14** (the 08-11 plan's
line numbers had all drifted; every site still exists):

| site | what |
|---|---|
| `engine.py:659,728,754` | `pending` init / append / consume |
| `engine.py:763,771,776` | `st.vel_scale = s_new`, `drift_and_migrate(vel_scale_new=)`, stats |
| `engine.py:302` | `kick_pending` term in `mesh_bytes` |
| `state.py:408` | the `SlotState.vel_scale` field |
| `state.py:154` | `_rescale_w` |
| `state.py:228,242` | `brick_reach`, which reads the scale to bound the drift |
| `state.py:250,272-273,385` | `drift_and_migrate`'s old/new scale handling |
| `state.py:643,805,827` | `decode_brick`, `decode_bricks`, `write_velocities` |
| `state.py:664` | `v = self.w[slots] * self.vel_scale` |
| `state.py:845,886` | `_eject_slab` and the `_rescale_w` call inside it |
| `codec.py:327,359` | `T9State.vel_scale`, `decode` |
| `icgen.py:287,382` | the IC-side exact partition-max scale, and the loader |

**Migrants are the one place a rescale survives.** A particle crossing into
another brick is re-encoded at the destination's scale, through the `_rescale_w`
call `_eject_slab` already makes.

**Schema.** Per-brick scales change what a slab file means, so states written
before and after are not interchangeable. Bump `SCHEMA` at `icgen.py:54` from
`t9-slabs-1` to `t9-slabs-2` so old slabs refuse loudly rather than load with a
silently different meaning. Slabs are cheap to regenerate and staged ones are
cleaned between jobs anyway.

**Rejected, with reasons** (carried from the 08-11 plan, unchanged):

- a second force pass -- the force cannot be retained, D-v2-16 cl.1 deletes it
  (206 GB);
- predict-and-refuse -- struck in M-v2-3 as unimplementable, since the refusal is
  only detectable after the force is consumed;
- spilling `pending` to the staging tier -- 275 GB at 0.83-1.42 GB/s is 5-9 min
  of pure I/O *per step*;
- double-rounding at a single global scale -- works, but strictly worse now that
  storage semantics are open ("not married to the storage semantics whatsoever",
  JC 2026-08-11).

**Gates.** Identities and degenerate limits, no picked thresholds:

1. Per-brick scales reproduce a single-scale run **bitwise** wherever all bricks
   share a scale (the degenerate limit).
2. Force parity unaffected: 0 of 6,291,456 elements at cdev8.
3. `load(generate) == SlotState.build` bitwise still holds, at the new schema.
4. **Checkpoint 4:** the re-measured accumulated codec cost against the ratified
   **7.125e-4 at C-dev K=40** (D-v2-9's bar is 3e-2, a 42x margin). Expectation
   is an improvement; a regression is a finding. This moves either way, so it is
   a checkpoint, not a footnote.

**Readout: the laptop, not the cluster.** The 08-11 plan said to re-run Stage 0's
instrument after each Stage 2 fix. That instrument cannot read this one: at cdev
`kick_pending` is 537 MB against a 282 MB RSS sigma, ~1.9 sigma. Use
`scripts/v2_m6_host_bytes.py` instead -- it measures numpy allocation exactly and
bit-identically across runs, `pending` is numpy, and it costs ~90 s on the laptop
with no cluster job at all. Traps: the first run in a process compiles and reads
2.09x high (`--warmup` defaults to 1), and cdev needs `--arena-frac 0.20` on this
laptop where 0.08 refuses.

## Stage 2b -- `repack`'s O(N) scratch (93.5 GB)

Unchanged and re-verified: `state.py:1063-1065` allocates `np.zeros_like(off)`,
`np.zeros_like(w)` and `np.full_like(ids)`, and its own docstring says the
in-place form is owed. Port the chunked compact-forward / expand-backward from
`layout.BrickPackedLayout.repack` (`layout.py:614`), which already reports
`scratch_bytes`. D-v2-19 cl.3 established this is a **monotone rearrangement**,
not a sort: O(chunk) scratch, 0.13-0.52 MB independent of N.

`SlotState` moves 9 B payload rows rather than an int64 permutation, so the
backward buffer is wider, and the arena fold-in has no analogue in `layout.py`.
Keep `argsort` here deliberately: post-repack slot order *is* key order, measured
sortedness exactly 1.0000, timsort's best case, where radix is ~10x slower.

**Gate: a pure identity** -- the new repack's final `off`/`w`/`brick_start`/
`occupancy` elementwise identical to the current one's at cdev8 and cdev.

Its priority rose with the arithmetic above: 2a alone leaves C-gh at 1.00x a gg
node, and 2b is what turns that edge into 0.60x.

## Stage 2c -- the coarse paint decomposition

**Premise re-verified 2026-08-14 and it has NOT moved.** I expected the masking
and pad-ladder work to have eroded it; it did not. `engine.py:396-434` allocates
one persistent `n_coarse^3` int64 accumulator (8.59 GB at C-gh, the term
`inexor.plan` calls `coarse_accumulator`), and then *per chunk* calls
`paint_tsc_int` which returns a full `n_coarse^3` mesh, widens it to int64 and
adds it in. At C-gh `chunk_bricks=64` over 128^3 bricks is **32,768 chunks per
step**, each one a full-mesh allocation plus 27 scatter passes for what is
spatially a thin pencil of cells.

Paint each chunk into a coarse **sub-block** instead. A chunk is a flat brick
run, hence a spatial pencil, so its TSC footprint is a thin slab of coarse cells
with a computable bounding box. The machinery to copy is
`coarse_subblock_origin_extent` / `stage_coarse_subblock` (`forces.py`), already
doing exactly this shape on the *gather* side at a measured 3,500x saving.

**Gate: bitwise.** Integer addition is associative and the corners and masks are
unchanged, so the accumulated mesh must be bit-identical to today's; the existing
streamed-vs-monolithic pin in `tests/test_engine.py` is the regression.

Light `chunk_bricks` re-tuning afterward, deneb only, at cdev8/cdev. State the
extrapolation limit plainly: chunk count scales with `n_bricks`, so a
deneb-scale optimum does not transfer to C-gh unchanged.

## Stage 2d -- `load_slot_state`'s peak

`icgen.py:342-388`. Two things are co-resident that need not be: the int64
`occupancy` and its narrowed copy through `_to_index` (`:379`), and the `slabs`
list holding every slab's `off`/`w` while `off_all`/`w_all` are allocated at
1.21N (`:363-364`). Build `occupancy` directly in the index dtype, and order the
allocations so the slab payload is released as it is placed.

**Gate:** the load phase read separately by the peak instrument, and
`load(generate) == SlotState.build` bitwise still holds.

## Stage 3 -- the streaming parity instrument

Rewrite `leg_force_parity` (`scripts/v2_m3_engine_gate.py`) so it never
materializes an O(N) array. `force_short_tiled` already accepts a
`sink(idx, g_owned)` callable, so the reference arm can be driven per tile while
the leg folds running statistics -- `n_diff`, `max|delta|`, element count,
per-tile digests -- instead of building `x_q` (206 GB), `row_of_slot` (83 GB) and
two force arrays.

This preserves M-v2-3's stronger claim: the arms are driven from **different**
membership orders and agree bitwise anyway, which only works because both paints
are integer.

**Gate -- a regression, not a new threshold:** the streaming instrument must
reproduce the existing cgh64 card **bitwise**, 0 of 402,653,184 elements, at
T=256/b=32. `scripts/v2_g3_card_repro.py` is the two-card diff tool. Only then
does it run at 2048^3.

While here: `v2_m3_engine_gate.py` writes cards with no `commit`, no
`provenance`, no `slurm_job_id` -- the same gap that moved a slope +0.32 -> +0.80
in M-v2-4. Route it through the shared `_write`.

## Stage 4 -- the output stage (dark matter only)

**(a) The writer.** `save_slot_state` / `write_t9_slabs(st)` as the exact inverse
of `load_slot_state`, same slab schema (now `t9-slabs-2`), same per-array crc32.
The evolved state already holds `(occupancy, off, w)` in slot order, so this is
close to free. **Gate: `load(save(st))` is bitwise `st`.**

This is also what makes the capacity run **restartable**, which matters
independently: a multi-hour realization need not fit one wall, and a timeout
bills the whole wall.

**(b) The usable summary.** Large-scale P(k) with no full-size array: reuse
`coarse_delta_streamed` (already yields a 1024^3 f32 delta, 4.3 GB) -> `ooc_fft`
-> the **bin-AVERAGED** oracle comparison from M-v2-5 leg VI. Bin-centre is not
an option: it failed at 8.7 sigma at exactly 2048^3 mode counts (deterministic
Jensen term, z = +15 at k ~ 0.2). Store the **z profile**, not just `max|z|` --
storing only the max is a recorded instrument defect from that session.

**(c) Disk accounting, which nothing does today.** `generate_t9_slabs` stages 20
full 2048^3 f32 files at 34.4 GB each, about **687 GB of scratch**, deletes none,
and there is no `plan_bytes` analogue for disk and no cleanup of `stage/`. Add a
disk `plan_bytes` + `require_fits` twin and stage cleanup. **The budget is a
parameter, never a read of `$SCRATCH`.** House rule: `rmdir`, never `rm -rf`.

## Stage 5 -- the capacity runs

On whichever machine Stage 1b picks, subject to the arithmetic above. ICs at
2048^3 -> K steps -> written state + P(k) card + the Stage 3 parity number.
Report the ratified triple (D-v2-8 cl.8): **peak B/p, wall/step, node-hours and
SU per realization.**

**Run count, per JC: 3 realizations if one costs under 5 h, plus one run at a
second (T,b) -- four runs maximum, and deliberately not a matrix.**

- **Second (T,b) point = T=512/b=32.** Padded-volume ratio 1.42 against
  T=256/b=32's 1.95 (cheaper wall), at a larger `cap` and ~P^3 more memory per
  tile, and `cap` is the measured cost variable. Accuracy is not the question:
  evolved dP/P is measured P-independent (T128 = T256 = T512 = 1.896e-2 at
  cgh64). This discharges D-v2-16 cl.4's "PROVISIONAL until measured at C-gh".
- **New input since 08-11:** the anchor's wall at the ratified cadence is
  measured -- cdev K=40 at 62.10 s/step (antares 453), and the peak does NOT grow
  with K (7.727 GB at K=40 against 7.888 at K=15). The under-5-h test gets
  applied at checkpoint 5 against a measured s/step, not against
  `v4_pricing_record.md`'s 2.4-3.7 h device-only range.
- **New decision that may change the operating config:** `malloc_trim` at the
  anchor is 15.5% of peak for +1.6% wall, and the trade improves toward
  production. If ratified it belongs in Stage 5's pre-registration.
- **Walls from measurement, generously.** One M-v2-5 job burned 6 h 41 min (84%
  of its wall) producing nothing.
- **One leg per job**, cheapest first, `set -e` deliberately dropped.
- K is **not** a ratified config parameter -- pin it in the run's
  pre-registration rather than inheriting 40 silently.
- Not a gate under any reading: D-v2-8 cl.7's 100-realization production batch.

## Stage 6 -- the record and the re-scope

- `runs/v2/m6_capacity_record.md`, cards force-added past the `runs/` ignore.
  (`runs/v2/m6_peak_record.md` already exists and covers Stages 0/0b.)
- A new ADR carrying the **re-scoped gate**, the **per-brick velocity scale**
  semantics change, and the **correction to `engine.py:39-45`'s "nothing O(N) in
  floats" claim**. Precedent is **supersede, not edit** (D-v2-10 -> D-v2-11).
- **Correct the "only thing between this engine and a capacity run" language** for
  `kick_pending`, per the table above.
- **Forward obligation on M-v2-7, written down rather than left implicit:** the
  D-v2-11 transfer correction is a required *production* step for a mock and
  lives only in `scripts/v2_g6b_calib_transport.py`, never in the package.
  M-v2-6's output is explicitly not a production mock.
- If Stage 1b favours the GPU: record the **gather-jit contract** as a live,
  ~1.7x-valued open call rather than a closed refusal.
- `docs/running-elsewhere.md`: the per-site variable list, and a plain statement
  of which recorded numbers are properties of Vista (the 116 GB cliff, the 237 GB
  gg node, the staging bandwidths, the SU rates) rather than of inexor.
- Fix the **ladder annotation gap**: M-v2-1..M-v2-5 carry no closure annotation in
  either table copy, and `plan-plan-v2.md:131-136`'s capacity table is still
  uint16 / 87.2 GB / 1.33x, superseded by D-v2-20.

## Checkpoints

1. ~~After Stage 0~~ **DONE.** The instrument sees the 32 B/p term
   (`gate2_sees_on_bpp_term: true`, an 8x clearance).
2. ~~After Stage 1a~~ **DONE, answered: thread count does not explain the 2x.**
   The sweep is flat to 0.6% across 1-32 threads; the step is serial and per-core
   speed decides. The plan's open question 2 resolves in favour of the CPU case.
3. **Before Stage 1b** -- a Slurm proposal (count, nature, queue, wallclock, SU)
   for three TACC arms.
4. **After Stage 2a** -- the re-measured codec cost against the ratified
   7.125e-4, since the storage semantics changed. This needs a cdev K=40 run, so
   it arrives with an albireo Slurm proposal attached.
5. **Before Stage 5** -- a Slurm proposal with walls derived from measured
   s/step, and the under-5-h test applied to decide 1 vs 3 realizations.

## Verification

- `pixi run test` (428 passed / 1 skipped at the last close) and
  `pixi run test-det` (16 items -- it silently collected **zero** tests once,
  after a retirement deleted the marked population). Both are the pre-push gate,
  with `pixi run lint`.
- New tests, each asserting an identity or a shape rather than a picked
  threshold: per-brick scales reproducing a single-scale run in the degenerate
  limit; the new repack elementwise-identical to the old; the sub-block coarse
  paint bitwise the current streamed paint; the streaming parity leg reproducing
  the cgh64 card bitwise; `load(save(st))` bitwise.
- Each Stage 2 removal measured by `scripts/v2_m6_host_bytes.py` on the laptop,
  not by an RSS peak on a cluster.
- Cards read only through the comparability-checking readout; pre-provenance
  cards report UNKNOWN and are never backfilled.
- **macOS peaks are unusable** for RSS comparisons (Darwin reads ~3x low; one
  laptop point read 6.484 and 9.855 GB minutes apart). The host-byte instrument
  is exempt because it counts allocation, not RSS.

## Ops

- Branch `jc/m-v2-6-capacity`. Push only after the pre-push gate passes locally.
  Remotes pull from GitHub over HTTPS.
- Deneb checkout is `~/src/inexor`; Vista's `~/src/inexor` is a **symlink to
  scratch**, so outputs there are purge-eligible.
- Placement by resource, never `-w`: `--mem=100G` routes to antares (124 GB) vs
  deneb-the-node (56 GB); `-c 32` routes to deneb (64 c) vs antares (28 c).
- Never leave a detached background process on a shared remote machine.
- Every sbatch keeps the current preamble: `export PATH="$HOME/.pixi/bin:$PATH"`,
  a `command -v pixi` hard-fail, a jax import + **backend assertion**, per-leg
  exit codes, a LEG SUMMARY, a card-freshness stamp, and `exit 1` if zero legs
  succeeded. `set -e` stays dropped so one failed leg does not delete the
  evidence from the others.
- **The smoke leg gates the long legs** (`6d68bfa`), after job 451 caught a
  defect in its smoke leg and ran every long leg anyway.
- Ops trap for any `gg` sbatch: `JAX_PLATFORMS=cpu` **and**
  `CONDA_OVERRIDE_CUDA=12.0` are both required (jax 0.10 cuInit hard-raise).

## Open questions

1. **The charter says "one Vista gh node" and the arithmetic says that node
   cannot hold C-gh** -- state plus resident mesh is 116.5 GB against a 116 GB
   hard cliff, before any transient and after both large removals. Does M-v2-6's
   re-scope change the target to a gg node outright, or does the milestone keep
   gh as a target and take on further removals to reach it?
2. **Does that change your appetite for Stage 1b?** Its charging comparison was
   framed as gg-vs-gh at a 3x wall bar. If gh cannot host the run, 1b becomes a
   gg-vs-Stampede3-x86 question, which is a different job and possibly a cheaper
   one.
3. **`malloc_trim` as an operating point** is still your call, now with the
   anchor number (15.5% of peak for +1.6% wall, improving with tile size). It
   belongs in Stage 5's pre-registration either way, so a decision before then is
   enough.
4. **Schema bump to `t9-slabs-2`** for the per-brick scales: taken as the default
   here, on the grounds that a silently different meaning is worse than a loud
   refusal and slabs are cheap to regenerate. Say so if you would rather it stay
   backward-compatible with a scalar broadcast.
5. **The second (T,b) point assumes the operating point survives Stage 1b.** If
   gg's 237 GB is the target, most of the memory pressure that made T=256/b=32
   attractive is gone, and the informative second point may become a *larger*
   tile than T=512.
