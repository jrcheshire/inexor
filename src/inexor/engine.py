"""The v2 engine: BullFrog PM on T9 state in slot order (M-v2-3).

`integrate.py` provides per-step coefficients and a stateless float stepper;
`forces.py` provides two independent arms; `state.py` provides the container and
the exchange. Nothing composed them, and `integrate.py:14-16` says so: the v1
drivers were deleted at the retirement and "the v2 engine's own driver lands at
M-v2-3". This is that driver.

## The step, and why it is not the reference stepper's shape

BullFrog DKD carries a velocity that is SHARED across the step boundary: the
trailing half-drift of step n and the leading half-drift of step n+1 both use
`v_{n+1}`, so they fuse exactly into one drift of `(dD_n + dD_{n+1})/2`. The
engine therefore carries state DRIFT-SYNCHRONIZED -- positions at step midpoints,
velocities at step boundaries -- and one step is

    force at the stored positions  ->  kick  ->  ONE fused drift + migrate

rather than drift, force, kick, drift. Two reasons, and the second is the one
that matters:

  1. The layout runs once per step instead of twice. It is 85.5% of the force on
     the production backend, so that is ~40% of the step.
  2. The state is quantized ONCE per step, which is the cadence D-v2-14's
     ratified accumulated-error figure was measured at. The two-drift form
     quantizes twice and that number would stop describing the engine.

Measured before it was built (JC's call): moving the quantization from the step
boundary to the midpoint costs nothing. At cdev8, K=40, the ratified bucket
against its own never-quantized reference, max |dP/P| 2.311e-4 at the midpoint
against 2.821e-4 at the boundary, both ~2 orders under D-v2-9's absolute 3e-2.

The price is that the engine is NOT bitwise `float_step_bullfrog`:
`mod(mod(x+a)+b)` and `mod(x+a+b)` are equal in R and not in float (24,867 of
100,000 components, max 2.8e-14). `float_run_bullfrog_sync` below is the matched
reference the parity gate compares against; `float_step_bullfrog` is untouched,
being an mbody port under a stability contract.

## Nothing O(N) in floats, anywhere

The long arm is solved on the coarse mesh and read per tile out of a staged
sub-block (D-v2-16 cl.3); the short arm is per tile already; the kick applies
tile-locally and writes velocities back through the layout (cl.1). The two
global `(n,3)` f64 arrays that would otherwise appear are 206 GB each at C-gh,
and deleting them is what makes that configuration runnable.
"""

import numpy as np

from .codec import INT16_MAX, assert_int16_range
from .forces import (
    CAP_RUNGS_PER_OCTAVE,
    COARSE_HALO,
    capacity_shape,
    coarse_force_meshes,
    owned_mask_from_bricks,
    coarse_subblock_origin_extent,
    gather_coarse_subblock,
    make_tile_force_fn,
    stage_coarse_subblock,
    tile_capacity,
    tile_origin_extent,
)
from .layout import assert_brick_divides_buffer, choose_brick
from .painting import check_tsc_paint_headroom, paint_tsc_int
from .state import drift_and_migrate

__all__ = ["EngineConfig", "coarse_delta_streamed", "float_run_bullfrog_sync", "run", "step"]


def _dtype_name(x, what):
    """Normalize a mesh dtype to its name at CONSTRUCTION, numpy only.

    Accepts "float32", np.float32 or jnp.float32 alike and stores the name, so
    `EngineConfig` keeps its no-jax-at-construction rule. Failing here rather
    than at first use means a typo cannot survive as far as a cluster job.
    """
    dt = np.dtype(x)
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"{what} must be float32 or float64, got {dt.name}")
    return dt.name


class EngineConfig:
    """Geometry and kernel choices for a run. Plain attributes, no jax."""

    def __init__(
        self,
        box_size,
        n_part,
        n_fine,
        n_coarse,
        n_tile,
        b_fine,
        alpha=1.0,
        paint_short="int",
        paint_long="int",
        frac_bits=12,
        chunk_bricks=64,
        brick_slack=0.10,
        repack_every=1,
        coarse_dtype="float64",
        fine_dtype="float64",
        cap_rungs=CAP_RUNGS_PER_OCTAVE,
        pad_ladder=True,
    ):
        self.box_size = float(box_size)
        self.n_part = int(n_part)
        self.n_fine = int(n_fine)
        self.n_coarse = int(n_coarse)
        self.n_tile = int(n_tile)
        self.b_fine = int(b_fine)
        self.alpha = float(alpha)
        # D-006 on BOTH arms. The coarse paint's integer twin was a named
        # deliverable (D-v2-16 cl.2); the tiled short arm had the same defect and
        # no document named it, which M-v2-3's planning session found. The layout
        # reorders particles every step, so an order-dependent primal paint is
        # not reproducible even on one machine.
        self.paint_short = str(paint_short)
        self.paint_long = str(paint_long)
        self.frac_bits = int(frac_bits)
        self.chunk_bricks = int(chunk_bricks)
        # D-v2-19 clause 3: frozen capacity fails at EVERY granularity, so the
        # repack is required rather than an optimization. Built without it, the
        # engine hit the D-007 refusal with a full arena by step 10 at `smoke`.
        self.brick_slack = float(brick_slack)
        self.repack_every = int(repack_every)
        # M-v2-4. TWO knobs, defaulting independently, deliberately NOT coupled:
        # the coarse mesh is the gated arm (it grows with the box and is the
        # binding resident term at hero scale) and the fine tile mesh is
        # measured (P^3, box-independent). Moving one must move one error term
        # or the gate's reading is unattributable -- which is also why
        # `fine_dtype` does not follow `coarse_dtype` by default: setting one
        # knob would then silently set two.
        self.coarse_dtype = _dtype_name(coarse_dtype, "coarse_dtype")
        self.fine_dtype = _dtype_name(fine_dtype, "fine_dtype")
        # Rungs per octave for the per-tile buffer SHAPE ladder (M-v2-6 Stage 0).
        # `cap` moves every step, so an unquantized shape leaks one XLA executable
        # family per step. A knob rather than a constant because it trades padded
        # rows against retained executables and the balance is machine-dependent:
        # more rungs means less padding and more compilations.
        self.cap_rungs = int(cap_rungs)
        # The A/B knob for the SECOND shape ladder, and it exists only so the
        # coarse chunk buffer can be turned back to its pre-fix behaviour with
        # `cap` left on its ladder in both arms. Moving `cap_rungs` would move
        # both ladders at once and the reading would be unattributable, which is
        # the same reason `fine_dtype` does not follow `coarse_dtype`.
        # False is NOT an operating point: it is the arm whose slope the fix is
        # measured against.
        self.pad_ladder = bool(pad_ladder)

    @property
    def np_coarse_dtype(self):
        return np.dtype(self.coarse_dtype)

    @property
    def np_fine_dtype(self):
        return np.dtype(self.fine_dtype)

    @property
    def n_total(self):
        return self.n_part**3

    @property
    def coarse_cell(self):
        return self.box_size / self.n_coarse

    @property
    def fine_cell(self):
        return self.box_size / self.n_fine

    @property
    def r_s(self):
        """The split scale. `alpha = r_s / d_coarse` is the dimensionless knob,
        so C-dev results transfer to configurations where cells do not."""
        return self.alpha * self.coarse_cell

    def mesh_bytes(self):
        """Per-step MESH anatomy in bytes, as a function of the two dtypes.

        **Deliberately not folded into `bytes_per_particle`.** That reports a
        per-particle budget and the mesh is not per-particle; a mesh term
        divided by N would shrink as the box grows, which is the opposite of
        what the coarse mesh does. The `scaffold=0.0` precedent there is the
        same instinct -- a term that vanishes from a table is indistinguishable
        from one that was never counted.

        This exists because that is exactly what happened. D-v2-16 clause 3
        quoted 12.9 GB at C-gh and 103 GB at C-hero; those are the f32 numbers
        (`v4_architecture_record.md` item 6 says so), and the shipped code was
        f64 throughout, so the engine paid 25.8 and 206 GB from the freeze until
        M-v2-4. Nothing in the package could have shown that, because nothing
        counted the mesh at all. `test_the_coarse_meshes_match_the_ratified_
        budget` is the check that would have.

        Terms are labelled `resident` (live simultaneously during the tile loop)
        or `transient` (peak while a phase runs). The kernel BUILD stays f64 at
        either dtype -- it is the precision island -- so it does not shrink, and
        it is reported separately rather than hidden inside the kernel term.
        """
        cw = self.np_coarse_dtype.itemsize
        fw = self.np_fine_dtype.itemsize
        nc = self.n_coarse
        cells = nc**3
        half = nc * nc * (nc // 2 + 1)
        from .forces import padded_size

        p = padded_size(self.n_tile, self.b_fine, n_fine=self.n_fine)[0]
        pcells = p**3
        phalf = p * p * (p // 2 + 1)
        slab = max(1, min(nc, 32))
        return dict(
            # --- coarse, transient
            coarse_accumulator=cells * 8,          # int64 host, dtype-independent
            coarse_decode_slab=slab * nc * nc * 8,  # one f64 slab (M-v2-4)
            coarse_kernel_build_f64=3 * half * 8,   # k2_true/k2_safe/fac, the island
            coarse_kernels=3 * half * 2 * cw,
            coarse_fft_workspace=half * 2 * cw,
            # `pref = (fac / k2_safe).astype(fdtype)` is a fourth full half-grid
            # and was in neither arm's accounting (`forces.split_kernels`).
            coarse_kernel_pref=half * cw,
            # --- coarse, resident through the tile loop
            coarse_delta=cells * cw,
            coarse_force_resident=3 * cells * cw,
            # `g_coarse = [np.asarray(g) for g in g_coarse]` rebinds only after
            # the comprehension completes, so the jax originals and their numpy
            # copies are live TOGETHER. M-v2-6 measured `coarse_solve` at 14.19 MB
            # (cdev8) and 110.89 (cdev) against 6.29 and 50.33 for one copy; the
            # coarse_div arm moved it 7.3x for an 8x change in cells, which is
            # what identifies this as a cells term rather than a particle one.
            coarse_force_copy_transient=3 * cells * cw,
            # --- fine, resident through the tile loop
            tile_kernels=3 * phalf * 2 * fw,
            # THE BUILD, which the tile arm never counted though the coarse arm
            # always did. `split_kernels` holds `k2_true`, `k2_safe` and `fac` as
            # full f64 half-grids plus `pref` at the fine dtype while it
            # materialises the three complex kernels, so the peak of a build is
            # 80*phalf at f64 where this dict modelled 48. The `ik_j` are LOW-RANK
            # broadcasts ((nx,1,1) etc.) and cost nothing, which is why the factor
            # is 5/3 and not larger.
            #
            # Measured, and the reason this is a derivation rather than a fit:
            # `membership` (the phase that calls `make_tile_force_fn`) reads 85.60
            # MB at cdev8 against 51.12 modelled and 1319.45 at cdev against
            # 791.28 -- ratios of 1.675 and 1.667 against a derived 5/3. The
            # config ladder alone could not have found this: particles, coarse
            # cells and phalf scale together on every rung, and it took cdev's
            # 15.48x phalf against 8.00x particles to separate them.
            tile_kernel_build_f64=3 * phalf * 8,
            tile_kernel_pref=phalf * fw,
            # --- fine, transient per tile
            tile_workspace=pcells * (fw + 4 + 3 * fw) + phalf * 2 * fw,
        )

    def step_bytes(self, n_particles, n_rows=None, cap=None):
        """Per-step HOST terms that scale with PARTICLES, not with the mesh.

        The companion to `mesh_bytes`, and it exists for the same reason: nothing
        counted these either. M-v2-6 Stage 0 measured the engine's end-to-end peak
        for the first time and found 8.2 GB at cdev8 where state plus
        `mesh_bytes` plus the tile buffers modelled ~0.25 GB, so a planner built
        only on `mesh_bytes` would have sized C-gh and been wrong by a factor of
        thirty. A term that is not in the table cannot be traded against anything.

        Terms, each pointing at the line that allocates it:

        `brick_scales` -- one f64 per brick, the array that REPLACED
        `kick_pending`. That term held `(slots int64, v_new f64)` for every owned
        row until a global velocity scale could be reconciled -- 32 B/p, 274.9 GB
        at C-gh, the largest single term in the configuration and the one the
        module docstring wrongly claimed did not exist ("Nothing O(N) in floats,
        anywhere"). Per-brick scales delete the wait that forced it: a brick's
        rows are all kicked in one tile, so its scale is known immediately. What
        is left is `n_bricks * 8` -- 16.8 MB at C-gh, five orders down, and it is
        RESIDENT rather than per-step, so it is carried in the state table
        instead. The line stays here reading zero because a term that vanishes
        from a table is indistinguishable from one that was never counted.

        `repack_scratch` -- `SlotState.repack` allocates `zeros_like` of `off` and
        `w` while the originals stay live (`state.py:936-938`), so 9 B per ROW,
        ~91 GB at C-gh. D-v2-19 clause 3 establishes the in-place form at
        O(chunk); this is what it is worth.

        `tile_buffers` -- the per-tile host working set, `cap`-sized. Needs a
        measured `cap`; omitted when not supplied rather than guessed.
        """
        n = int(n_particles)
        rows = int(n_rows) if n_rows is not None else int(round(n * 1.21))
        # MIGRATION STAGING IS N^(2/3), NOT N, and that is the whole reason it is
        # worth a line. `state.drift_and_migrate` walks x-slabs and releases a
        # staged slab as soon as every write that could reach it has happened, so
        # the rows in flight are a handful of SLABS rather than the state: a slab
        # is N / bricks_per_side, and bricks_per_side grows as N^(1/3), so this
        # term grows as N^(2/3). Carrying it as an O(N) term would overstate it by
        # 8x at C-gh.
        #
        # The coefficient is measured, the SHAPE is derived, and two configs agree
        # on the coefficient which is what tests the shape: 181.7 B per
        # slab-particle at cdev8 and 198.9 at cdev, an 8x change in N (`migrate`
        # own-increment 47.62 and 208.58 MB, M-v2-6 host-byte instrument).
        # `lead_drift` is the SAME function called once before the loop and runs
        # 77.3 / 92.7 B on the same basis; it is not added here because it
        # completes before the step's peak and so is never co-resident with it.
        nb = max(1, self.n_fine // self.n_brick)
        out = dict(
            kick_pending=0,
            repack_scratch=rows * 9,
            migrate_staging=int(round(190.0 * n / nb)),
        )
        if cap is not None:
            out["tile_buffers"] = int(cap) * (8 + 1 + 24 + 24 + 1 + 8 + 24 + 24)
        return out

    @property
    def n_brick(self):
        return choose_brick(self.n_tile, self._b_realized, self.n_fine)

    @property
    def _b_realized(self):
        from .forces import padded_size

        return padded_size(self.n_tile, self.b_fine, n_fine=self.n_fine)[1]

    @property
    def tiles_side(self):
        return self.n_fine // self.n_tile

    @property
    def tiles(self):
        s = self.tiles_side
        return [(i, j, k) for i in range(s) for j in range(s) for k in range(s)]

    def validate(self):
        assert_brick_divides_buffer(self.n_tile, self._b_realized, self.n_brick, self.n_fine)
        if self.paint_long == "int":
            check_tsc_paint_headroom(self.n_total, self.frac_bits)
        self._refuse_f64_without_x64()
        return True

    def _refuse_f64_without_x64(self):
        """An f64 mesh without x64 is not an f64 mesh (M-v2-4).

        The library never enables x64 and every caller opts in, so this refuses
        nothing that works today. What it prevents is the one configuration that
        lies about itself: asking for f64, silently getting f32, and reading the
        result as an f64 reference. That is the reference arm of M-v2-4's own
        gate, where a silent degradation would not make the comparison fail --
        it would make it read ZERO, which is a pass.

        In `validate()` rather than in `step()` so an ad-hoc single step stays
        possible; `run()` calls `validate()`.
        """
        import jax

        if jax.config.jax_enable_x64:
            return
        wants = [n for n in ("coarse_dtype", "fine_dtype") if getattr(self, n) == "float64"]
        if wants:
            raise ValueError(
                f"{' and '.join(wants)} ask for float64 while jax_enable_x64 is False, so "
                "the mesh would silently be float32 and an 'f64 reference' would not be "
                "one. Call jax.config.update('jax_enable_x64', True) in the driver (the "
                "library never sets it), or ask for float32 explicitly."
            )


# ===========================================================================
# the long arm, streamed
# ===========================================================================


def coarse_delta_streamed(st, cfg, stats=None, census=False, pad_shape=0):
    """delta on the coarse mesh, accumulated brick by brick.

    Integer addition is associative, so a chunked accumulation is **bitwise**
    what a single call over every position produces -- that is a property to
    test, not to hope for, and it is the reason the coarse paint must be the
    integer one before the paint can be streamed at all. With the f64 paint the
    chunking would change the answer.

    `stats`, if a dict, receives `coarse_peak_int` -- the max accumulated cell
    sum, which the int32 refusal below already computes, so it is free.

    `pad_shape` is the previous step's chunk shape, carried forward so the shape
    is monotone across a run for the same reason `cap` is; see the comment on
    `pad` below and `forces.capacity_shape`.

    `census=True` additionally counts cells whose integer sum is NOT exactly
    representable in f32 (`coarse_cells_inexact_f32`). It is OPT-IN because it
    costs two extra passes over the mesh and is a gate instrument, not a
    production need. **It counts round-trip failures rather than cells above
    2^24**: `< 2^24` is sufficient for exactness, not necessary, and
    `5000 * 2^12 = 625 * 2^15` is exact at 2.05e7 -- a magnitude threshold
    overstates the loss, sometimes by a lot.
    """
    if cfg.paint_long != "int":
        raise ValueError(
            "the streamed coarse paint requires the integer accumulator: an f64 "
            "accumulation is order-dependent, so chunking it changes the result"
        )
    import jax.numpy as jnp

    n = cfg.n_coarse
    mesh = np.zeros((n, n, n), dtype=np.int64)
    bricks = list(range(st.n_bricks))
    groups = [bricks[i : i + cfg.chunk_bricks] for i in range(0, len(bricks), cfg.chunk_bricks)]
    rows = [sum(st.brick_member_count(b) for b in gg) for gg in groups]
    # ONE shape for every chunk. A varying row count recompiles per chunk, which
    # profiled at 2.91 s of a 13.21 s step -- the same trap as the long-range
    # read. Padding is masked, not filled: an unmasked pad row would add mass.
    #
    # ONE shape ACROSS STEPS too, which the max alone does not give: occupancy
    # shifts every step, so `max(rows)` took ten distinct values over ten steps
    # at the smoke config (against one for `cap` since Stage 0 put that on the
    # ladder), and every buffer keyed on it keys a new XLA shape whose executable
    # is cached for the life of the process. That is the leading measured cause
    # of the run peak's +110 MB/step at cdev (M-v2-6 Stage 0b, antares 446: the
    # growth is linear over 15 steps, unsaturating, and survives malloc_trim, so
    # it is live memory rather than allocator slack). Same ladder, same rungs
    # knob, and the same masking argument makes it bitwise neutral.
    pad_true = int(max(rows)) if rows else 0
    pad = (
        capacity_shape(pad_true, rungs=cfg.cap_rungs, floor_shape=pad_shape)
        if cfg.pad_ladder
        else pad_true
    )
    for gg, m in zip(groups, rows):
        if m == 0:
            continue
        _, x, _ = st.decode_bricks(gg)
        xp = np.zeros((pad, 3), dtype=np.float64)
        xp[:m] = x
        lv = np.zeros(pad, dtype=bool)
        lv[:m] = True
        mesh += np.asarray(
            paint_tsc_int(jnp.asarray(xp), n, cfg.box_size, cfg.frac_bits, live=lv),
            dtype=np.int64,
        )
    peak = int(np.abs(mesh).max())
    if peak >= 2**31:
        raise ValueError(
            f"the accumulated coarse paint reached {peak}, past int32. "
            "Lower frac_bits -- this is D-007-class corruption, not imprecision."
        )
    # SLABBED DECODE (M-v2-4). The old form was
    #     counts_from_int(mesh.astype(np.int32), ...) / mean - 1.0
    # which materializes an int32 copy of the whole mesh plus two full f64
    # meshes on top of the int64 accumulator -- about 30 GB at C-gh, all of it
    # transient, none of it counted anywhere. Slabbing makes the transient one
    # slab (0.27 GB at C-gh) and lets the result land straight at the requested
    # dtype. Every operation is elementwise, so the slabbed result is
    # bit-identical to the whole-array form; `tests/test_engine.py` pins the
    # streamed paint against the monolithic one and would catch it if not.
    #
    # The decode stays f64 and the narrowing happens after `- 1.0`; see
    # `painting.density_tsc` for how little that buys and why it is kept anyway.
    mean = float(cfg.n_total) / float(n) ** 3
    scale = 2.0 ** -cfg.frac_bits
    out = np.empty((n, n, n), dtype=cfg.np_coarse_dtype)
    slab = max(1, min(n, 32))
    inexact = 0
    for i0 in range(0, n, slab):
        s = mesh[i0 : i0 + slab]
        out[i0 : i0 + slab] = s.astype(np.float64) * scale / mean - 1.0
        if census:
            inexact += int(np.count_nonzero(s.astype(np.float32).astype(np.int64) != s))
    if stats is not None:
        # both, for the same reason `cap`/`cap_true` are both reported: one hides
        # the padding cost, the other hides the shape churn, and the churn leaked
        stats["coarse_pad"] = pad
        stats["coarse_pad_true"] = pad_true
        stats["coarse_peak_int"] = peak
        if census:
            stats["coarse_cells_inexact_f32"] = inexact
            stats["coarse_exact_decode_ok"] = inexact == 0
    return out


# ===========================================================================
# one step
# ===========================================================================


def _diagnose_partition(st, cfg):
    """Where the ownership deficit is, for the assertion's message.

    Failure path only. The bare count ("16777215 against 16777216") is
    unattributable and cost three wrong explanations before this existed: a
    tile-local float gap, a position-vs-storage mismatch, and a lost row. Each was
    consistent with the count and none was the cause. What actually distinguishes
    them is WHICH rows are unclaimed and what is unusual about them, and the state
    knows.
    """
    b_real = cfg._b_realized
    nb = cfg.n_fine // cfg.n_brick
    claimed = np.zeros(st.off.shape[0], dtype=np.int32)
    misaligned = []
    for t in cfg.tiles:
        members = st.tile_bricks(t, cfg.n_tile, b_real, cfg.n_brick, cfg.n_fine)
        slots, _, _ = st.decode_bricks(members)
        counts = [st.brick_member_count(b) for b in members]
        if int(np.sum(counts)) != len(slots):
            misaligned.append((tuple(int(q) for q in t), int(np.sum(counts)), len(slots)))
            continue
        brick_of_row = np.repeat(np.asarray(members, dtype=np.int64), counts)
        own = owned_mask_from_bricks(brick_of_row, t, cfg.n_tile, cfg.n_brick, nb)
        # np.add.at, NOT `claimed[idx] += 1`. Fancy-index += increments a repeated
        # index ONCE, so the buffered form cannot see the very duplication this is
        # looking for -- the first version of this diagnostic reported zero
        # double-claims on a state whose distinct-slot count was short by one.
        np.add.at(claimed, np.asarray(slots)[own], 1)
    # counts vs DISTINCT slots: `SlotState.check` compares counts (state.py:587),
    # so an aliased slot passes it -- occupancy still sums to n_particles while
    # one slot is reachable through two bricks and the reachable SET is short.
    seen = np.zeros(st.off.shape[0], dtype=np.int64)
    total_decoded = 0
    for b in range(st.n_bricks):
        s = np.asarray(st.decode_brick(b)[0])
        total_decoded += len(s)
        np.add.at(seen, s, 1)
    live = seen > 0
    aliased = np.nonzero(seen > 1)[0]
    unclaimed = np.nonzero(live & (claimed == 0))[0]
    twice = np.nonzero(claimed > 1)[0]
    # THREE censuses that must be identical, printed rather than inferred. Job 440
    # showed decoded == distinct == n_particles - 1 with no aliasing, so a row is
    # unreachable while `check` passes -- and `check` sums OCCUPANCY. Whichever
    # pair disagrees localizes the defect: occupancy vs member_count is a counting
    # bug, member_count vs decode is a span/decode bug.
    p3 = st.buckets_per_brick
    occ = st.occupancy.astype(np.int64)
    occ_per_brick = occ.reshape(st.n_bricks, p3).sum(axis=1)
    mc_per_brick = np.array(
        [st.brick_member_count(b) for b in range(st.n_bricks)], dtype=np.int64
    )
    dec_per_brick = np.array(
        [len(st.decode_brick(b)[0]) for b in range(st.n_bricks)], dtype=np.int64
    )
    bad = np.nonzero((occ_per_brick != mc_per_brick) | (mc_per_brick != dec_per_brick))[0]
    detail = [
        (int(b), int(occ_per_brick[b]), int(mc_per_brick[b]), int(dec_per_brick[b]))
        for b in bad[:4]
    ]
    lines = [
        f"  CENSUS occupancy {int(occ_per_brick.sum()) + st.arena_used} "
        f"(incl. arena), brick_member_count {int(mc_per_brick.sum())}, "
        f"decode {int(dec_per_brick.sum())}, n_particles {st.n_particles}",
        f"  bricks where the three disagree: {len(bad)} "
        f"(brick, occ, member_count, decode)={detail}",
        f"  distinct live rows {int(live.sum())}, decoded WITH duplicates "
        f"{total_decoded}, n_particles {st.n_particles}, "
        f"arena_used {st.arena_used} of {st.n_arena}",
        f"  ALIASED slots (reachable through more than one brick) {len(aliased)}"
        f"{': ' + str([int(q) for q in aliased[:5]]) if len(aliased) else ''}",
        f"  unclaimed {len(unclaimed)}, claimed-more-than-once {len(twice)}",
    ]
    for s in aliased[:5]:
        s = int(s)
        bricks_with = [b for b in range(st.n_bricks)
                       if s in set(int(q) for q in st.decode_brick(b)[0])]
        lines.append(
            f"  aliased slot {s}: seen {int(seen[s])}x in bricks {bricks_with[:6]}, "
            f"off={[int(q) for q in st.off[s]]}, "
            f"in_arena={s >= st.arena_base}"
        )
    if misaligned:
        lines.append(
            f"  BRICK COUNT vs DECODE LENGTH disagree on {len(misaligned)} tiles: "
            f"{misaligned[:3]}"
        )
    for s in unclaimed[:5]:
        s = int(s)
        in_arena = s >= st.arena_base
        bucket = int(st.arena_bucket[s - st.arena_base]) if in_arena else None
        brick = int(np.searchsorted(st.brick_start, s, side="right") - 1)
        lines.append(
            f"  slot {s}: in_arena={in_arena} arena_bucket={bucket} "
            f"brick_by_span={brick} off={[int(q) for q in st.off[s]]}"
        )
    return "\n".join(lines)


def _no_phase(_name):
    """The default phase hook: does nothing, allocates nothing, returns nothing."""


def step(st, cfg, coeff, c_drift, collect=None, census=False, cap_shape=0, pad_shape=0,
         phase=None):
    """One drift-synchronized BullFrog step. Mutates `st`; returns diagnostics.

    `coeff = (alpha, beta_over_Dmid)` from `bullfrog_float_coeffs` columns 1 and
    2; `c_drift` is the FUSED drift for this step.

    `census=True` turns on the coarse decode census; see
    `coarse_delta_streamed`. Off by default: it is a gate instrument and costs
    two extra passes over the coarse mesh.

    `phase`, if given, is called with a boundary NAME after each phase of the
    step completes. It exists because a peak is a max and a max carries no
    timestamp: M-v2-6 Stage 0 attributed the peak by differencing whole-run
    maxima between arms, and at cdev the terms it was trying to separate
    (67-179 MB) sat inside the run-to-run scatter of the maximum itself
    (sigma 45-115 MB over five repeats of one leg, antares 445), so no
    difference of maxima could be read. Naming the boundaries lets a caller
    take a high-water mark PER PHASE instead, which is a measurement of where
    the peak is rather than an inference from what it is not. The hook takes no
    payload and returns nothing on purpose -- it must not be able to perturb
    the step, and the default is a function that does nothing at all.
    """
    import jax.numpy as jnp

    ph = phase if phase is not None else _no_phase

    alpha_k, bcoef = float(coeff[0]), float(coeff[1])

    # --- long arm: solve once, globally, on the coarse mesh
    mesh_stats = {}
    delta = coarse_delta_streamed(st, cfg, stats=mesh_stats, census=census, pad_shape=pad_shape)
    ph("coarse_paint")
    # `coarse_force_meshes` infers from delta.dtype and REFUSES a mismatch, so
    # the dtype cannot silently disagree with what the config asked for
    g_coarse = coarse_force_meshes(
        jnp.asarray(delta),
        cfg.n_coarse,
        cfg.box_size,
        "long",
        r_s=cfg.r_s,
        match=(cfg.coarse_cell, cfg.fine_cell),
        fdtype=cfg.np_coarse_dtype,
    )
    g_coarse = [np.asarray(g) for g in g_coarse]
    ph("coarse_solve")

    # --- membership, and the capacity one jitted program needs
    b_real = cfg._b_realized
    members = {
        t: st.tile_bricks(t, cfg.n_tile, b_real, cfg.n_brick, cfg.n_fine) for t in cfg.tiles
    }
    # brick_member_COUNT, not brick_live_count: a brick's arena residents are
    # part of its membership and `decode_bricks` returns them, so sizing `cap`
    # off the run length alone under-counts and the force refuses.
    counts = [sum(st.brick_member_count(b) for b in members[t]) for t in cfg.tiles]
    cap_true = tile_capacity(counts)
    # QUANTIZE THE SHAPE, do not use the true max. `cap_true` moves every step as
    # occupancy shifts (measured at cdev8: ten distinct values over ten steps,
    # 285,554 -> 397,319), so every buffer keyed on it keys a new XLA shape and
    # the executable cache grows without bound -- a peak host RSS linear in K at
    # 0.331 GB/step. `capacity_shape` puts it on a geometric ladder and
    # `cap_shape` carries the previous value so it is monotone across the run.
    # Padding is masked exactly as the within-step padding already is, so this is
    # bitwise neutral; `tests/test_engine.py` pins that.
    cap = capacity_shape(cap_true, rungs=cfg.cap_rungs, floor_shape=cap_shape)

    one_tile, geom = make_tile_force_fn(
        cfg.n_fine, cfg.box_size, cfg.n_total, cfg.n_tile, cfg.b_fine,
        r_s=cfg.r_s, paint=cfg.paint_short, frac_bits=cfg.frac_bits,
        fdtype=cfg.np_fine_dtype,
    )
    cell = geom["cell"]
    ph("membership")

    tile_scales, n_owned, n_overhang = [], 0, 0
    # NO `pending`. It held `(slots int64, v_new f64)` for every owned row until
    # the last tile had been kicked, because a GLOBAL scale cannot be known
    # before then -- 32 B/p, 274.9 GB at C-gh, and the largest single term in the
    # whole configuration. Per-brick scales remove the wait rather than the
    # array: a brick's rows are all kicked in one tile, so its scale is known the
    # moment that tile is done and its codes can be written immediately.
    for t in cfg.tiles:
        slots, x, v = st.decode_bricks(members[t])
        # the brick each decoded row came from, in the order decode_bricks
        # concatenates: this is what ownership is read off, NOT the position
        brick_of_row = np.repeat(
            np.asarray(members[t], dtype=np.int64),
            [st.brick_member_count(b) for b in members[t]],
        )
        m = len(slots)
        if m == 0:
            continue
        if m > cap:
            raise RuntimeError(f"tile {t}: {m} members > cap {cap}")
        idx = np.resize(np.arange(m), cap) if m else np.zeros(cap, dtype=np.int64)
        live = np.zeros(cap, dtype=bool)
        live[:m] = True
        origin, _ = tile_origin_extent(t, cfg.n_tile, b_real, cell)
        xg = x[idx]
        u = jnp.mod(jnp.asarray(xg) - jnp.asarray(origin), cfg.box_size)
        # ownership from the brick each row is STORED IN, which is the same thing
        # membership is built from, so the two cannot disagree. Two earlier rules
        # both re-derived it from a coordinate and both lost exactly one row of
        # 16,777,216 at cdev (antares 431 tile-local, 436 global-position). See
        # `forces.owned_mask_from_bricks` for why this one cannot.
        own_rows = owned_mask_from_bricks(
            brick_of_row, t, cfg.n_tile, cfg.n_brick, cfg.n_fine // cfg.n_brick
        )
        own = np.zeros(cap, dtype=bool)
        own[:m] = own_rows
        own &= live
        ph("tile_decode")
        g_short, owned, n_out = one_tile(u, jnp.asarray(live), jnp.asarray(own))
        g_short = np.asarray(g_short)[:m]
        owned = np.asarray(owned)[:m]
        n_overhang += int(n_out)
        ph("tile_short")
        if not owned.any():
            continue

        # the long force at the SAME owned rows, out of a staged sub-block
        o_cells, extent = coarse_subblock_origin_extent(
            t, cfg.n_tile, cfg.n_coarse, cfg.n_fine, halo=COARSE_HALO
        )
        sub = [stage_coarse_subblock(g, o_cells, extent) for g in g_coarse]
        # Padded to `cap` with a live mask, for the SAME reason the short arm is:
        # a per-tile row count keys a new XLA shape, so every tile recompiles.
        # Profiled before the fix at 2,107 compilations and 24.1 s of a 32.7 s
        # step -- 74% of it, 18.5 s inside backend_compile_and_load. One shape
        # serves every tile.
        n_own = int(owned.sum())
        xo = np.zeros((cap, 3), dtype=np.float64)
        xo[:n_own] = x[owned]
        lv = np.zeros(cap, dtype=bool)
        lv[:n_own] = True
        g_long = np.asarray(
            gather_coarse_subblock(
                *sub, jnp.asarray(xo), o_cells, cfg.coarse_cell, cfg.n_coarse,
                assign="tsc", live=lv,
            )
        )[:n_own]

        ph("tile_long")
        g_tot = g_short[owned] + g_long
        v_new = alpha_k * v[owned] + bcoef * g_tot
        n_owned += int(owned.sum())
        # QUANTIZE AND WRITE HERE, per brick, instead of holding `v_new`.
        # `decode_bricks` concatenates brick by brick and ownership is read off
        # the brick a row is STORED in, so the owned rows are whole brick blocks
        # and stay contiguous under the mask -- which is what lets a run scan
        # replace a sort. The assertion below is on that contiguity, because it
        # is load-bearing and free to check: if a brick ever appeared in two
        # runs, the second run would silently overwrite the first one's scale
        # and decode every row of it wrong.
        slots_o, bricks_o = slots[owned], brick_of_row[owned]
        cut = np.flatnonzero(np.diff(bricks_o)) + 1
        run_lo = np.concatenate(([0], cut))
        run_hi = np.concatenate((cut, [len(bricks_o)]))
        if len(np.unique(bricks_o)) != len(run_lo):
            raise AssertionError(
                f"tile {t}: owned rows are not grouped by brick "
                f"({len(run_lo)} runs over {len(np.unique(bricks_o))} bricks). The "
                "per-brick scale depends on a brick's rows being contiguous."
            )
        for lo_r, hi_r in zip(run_lo, run_hi):
            vb = v_new[lo_r:hi_r]
            s_b = float(np.max(np.abs(vb))) / INT16_MAX
            s_b = s_b if s_b > 0.0 else 1.0
            w_b = np.rint(vb / s_b)
            assert_int16_range(w_b)
            st.write_velocities(slots_o[lo_r:hi_r], w_b.astype(np.int16))
            st.vel_scale[int(bricks_o[lo_r])] = s_b
            tile_scales.append(s_b)
        ph("tile_reduce")

    # `pending` is at its largest HERE and nowhere else: it grows by one tile's
    # owned rows per iteration and is consumed below. A boundary at the end of
    # the loop is the only place a high-water mark can price it.
    ph("tile_loop_end")

    if n_owned != st.n_particles:
        # SELF-DIAGNOSING, because the bare count sent me down three wrong
        # explanations. Computed only on the failure path, so the happy path pays
        # nothing. It names where the deficit is rather than leaving it to be
        # guessed from the geometry.
        raise AssertionError(
            f"the tiles own {n_owned} rows against {st.n_particles} particles: ownership "
            "is supposed to be a partition, so this is a geometry error.\n"
            + _diagnose_partition(st, cfg)
        )
    if n_overhang:
        raise AssertionError(
            f"{n_overhang} rows fell outside their padded tile. Since choose_brick gained "
            "its divisibility condition the brick union is EXACTLY the padded box, so a "
            "nonzero overhang means the decomposition is wrong rather than wasteful."
        )

    # Nothing to reconcile: every brick's codes were written at its own scale
    # inside the loop. The phase boundary stays so the trace keeps its shape and
    # a peak comparison against every card on record is still like-for-like.
    ph("reconcile")

    stats = drift_and_migrate(st, c_drift)
    ph("migrate")
    # both: `cap` is the SHAPE every buffer took, `cap_true` the max over tiles it
    # was quantized from. Reporting only one of them hides either the padding cost
    # or the shape churn, and the shape churn is what leaked.
    # `vel_scale` stays a SCALAR in the stats dict -- the max over bricks, which
    # `drift_and_migrate` already put there -- so every reader of a card keeps
    # working. `vel_scale_min` is beside it: the two together are what say how
    # much the per-brick scales actually spread, and a single number cannot.
    stats.update(cap=cap, cap_true=cap_true, n_tiles=len(cfg.tiles),
                 vel_scale_kick_max=float(max(tile_scales)) if tile_scales else 1.0)
    # the REALIZED dtypes, read off the arrays rather than echoed from the
    # config: a receipt that repeats what it was told cannot catch a knob that
    # did not apply, which is the whole failure mode this milestone is built
    # against
    stats["coarse_dtype"] = np.dtype(delta.dtype).name
    stats["fine_dtype"] = str(geom["fdtype"])
    stats.update(mesh_stats)
    stats["repack"] = None
    if collect is not None:
        collect(stats)
    return stats


def fused_drifts(coeffs):
    """The drift-synchronized schedule from `bullfrog_float_coeffs`.

    Returns `(lead, fused)`: a leading half-drift that puts positions on the
    first midpoint, then one drift per step -- `h_k + h_{k+1}` after each kick,
    with `h_{K-1}` alone on the last so the trajectory lands on the same endpoint
    the boundary form does.
    """
    h = np.asarray(coeffs)[:, 0]
    return float(h[0]), np.concatenate([h[:-1] + h[1:], h[-1:]])


def run(st, cfg, coeffs, collect=None, census=False, phase=None):
    """Advance `st` over a whole schedule. `coeffs` from `bullfrog_float_coeffs`.

    `phase` is forwarded to `step`; see its docstring. The boundaries `run`
    itself adds are the lead drift and the repack, so that every allocation in
    the run falls inside exactly one named phase and the phases sum to the run.
    """
    cfg.validate()
    ph = phase if phase is not None else _no_phase
    lead, fused = fused_drifts(coeffs)
    drift_and_migrate(st, lead)  # onto the first midpoint
    ph("lead_drift")
    out = []
    # both buffer shapes are carried ACROSS steps and only ever grow, so the run
    # visits at most a few shapes instead of one per step: `cap` from Stage 0,
    # `coarse_pad` from Stage 0b, which measured the second one still churning
    cap_shape = 0
    pad_shape = 0
    for k in range(len(fused)):
        stats = step(st, cfg, (coeffs[k][1], coeffs[k][2]), float(fused[k]), collect,
                     census=census, cap_shape=cap_shape, pad_shape=pad_shape, phase=phase)
        cap_shape = int(stats["cap"])
        pad_shape = int(stats["coarse_pad"])
        if cfg.repack_every and (k + 1) % cfg.repack_every == 0:
            stats["repack"] = st.repack(brick_slack=cfg.brick_slack)
            ph("repack")
        out.append(stats)
    return out


def float_run_bullfrog_sync(x, v, coeffs, force_fn, box_size):
    """The drift-synchronized float reference the parity gate compares against.

    Algebraically identical to looping `float_step_bullfrog`, and deliberately
    NOT bitwise it: `mod(mod(x+a)+b)` and `mod(x+a+b)` differ in float. That is
    the whole reason this exists rather than the gate reusing the stepper --
    comparing the engine against a driver of a different SHAPE would report a
    roundoff difference as an engine defect.

    `float_step_bullfrog` is untouched; it is an mbody port with a stability
    contract and the probes drive it.
    """
    import jax.numpy as jnp

    lead, fused = fused_drifts(coeffs)
    x = jnp.mod(x + lead * v, box_size)
    for k in range(len(fused)):
        g = force_fn(x)
        v = coeffs[k][1] * v + coeffs[k][2] * g
        x = jnp.mod(x + float(fused[k]) * v, box_size)
    return x, v
