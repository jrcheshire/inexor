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

from .codec import INT16_MAX
from .forces import (
    CAP_RUNGS_PER_OCTAVE,
    COARSE_HALO,
    capacity_shape,
    coarse_force_meshes,
    coarse_subblock_origin_extent,
    gather_coarse_subblock,
    make_tile_force_fn,
    stage_coarse_subblock,
    tile_capacity,
    tile_origin_extent,
)
from .layout import assert_brick_divides_buffer, choose_brick
from .painting import check_tsc_paint_headroom, paint_tsc_int
from .state import drift_and_migrate, reconcile_velocity_scale

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
            # --- coarse, resident through the tile loop
            coarse_delta=cells * cw,
            coarse_force_resident=3 * cells * cw,
            # --- fine, resident through the tile loop
            tile_kernels=3 * phalf * 2 * fw,
            # --- fine, transient per tile
            tile_workspace=pcells * (fw + 4 + 3 * fw) + phalf * 2 * fw,
        )

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


def coarse_delta_streamed(st, cfg, stats=None, census=False):
    """delta on the coarse mesh, accumulated brick by brick.

    Integer addition is associative, so a chunked accumulation is **bitwise**
    what a single call over every position produces -- that is a property to
    test, not to hope for, and it is the reason the coarse paint must be the
    integer one before the paint can be streamed at all. With the f64 paint the
    chunking would change the answer.

    `stats`, if a dict, receives `coarse_peak_int` -- the max accumulated cell
    sum, which the int32 refusal below already computes, so it is free.

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
    pad = int(max(rows)) if rows else 0
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
        stats["coarse_peak_int"] = peak
        if census:
            stats["coarse_cells_inexact_f32"] = inexact
            stats["coarse_exact_decode_ok"] = inexact == 0
    return out


# ===========================================================================
# one step
# ===========================================================================


def step(st, cfg, coeff, c_drift, collect=None, census=False, cap_shape=0):
    """One drift-synchronized BullFrog step. Mutates `st`; returns diagnostics.

    `coeff = (alpha, beta_over_Dmid)` from `bullfrog_float_coeffs` columns 1 and
    2; `c_drift` is the FUSED drift for this step.

    `census=True` turns on the coarse decode census; see
    `coarse_delta_streamed`. Off by default: it is a gate instrument and costs
    two extra passes over the coarse mesh.
    """
    import jax.numpy as jnp

    alpha_k, bcoef = float(coeff[0]), float(coeff[1])

    # --- long arm: solve once, globally, on the coarse mesh
    mesh_stats = {}
    delta = coarse_delta_streamed(st, cfg, stats=mesh_stats, census=census)
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

    tile_scales, n_owned, n_overhang = [], 0, 0
    pending = []  # (slots, v_new) held until the global scale is known
    for t in cfg.tiles:
        slots, x, v = st.decode_bricks(members[t])
        m = len(slots)
        if m == 0:
            continue
        if m > cap:
            raise RuntimeError(f"tile {t}: {m} members > cap {cap}")
        idx = np.resize(np.arange(m), cap) if m else np.zeros(cap, dtype=np.int64)
        live = np.zeros(cap, dtype=bool)
        live[:m] = True
        origin, _ = tile_origin_extent(t, cfg.n_tile, b_real, cell)
        u = jnp.mod(jnp.asarray(x[idx]) - jnp.asarray(origin), cfg.box_size)
        g_short, owned, n_out = one_tile(u, jnp.asarray(live))
        g_short = np.asarray(g_short)[:m]
        owned = np.asarray(owned)[:m]
        n_overhang += int(n_out)
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

        g_tot = g_short[owned] + g_long
        v_new = alpha_k * v[owned] + bcoef * g_tot
        n_owned += int(owned.sum())
        # per-tile scale: a reduction over a buffer already resident. The max
        # over tiles is EXACTLY the global scale because ownership is a partition.
        tile_scales.append(float(np.max(np.abs(v_new))) / INT16_MAX)
        pending.append((slots[owned], v_new))

    if n_owned != st.n_particles:
        raise AssertionError(
            f"the tiles own {n_owned} rows against {st.n_particles} particles: ownership "
            "is supposed to be a partition, so this is a geometry error"
        )
    if n_overhang:
        raise AssertionError(
            f"{n_overhang} rows fell outside their padded tile. Since choose_brick gained "
            "its divisibility condition the brick union is EXACTLY the padded box, so a "
            "nonzero overhang means the decomposition is wrong rather than wasteful."
        )

    s_new = reconcile_velocity_scale(tile_scales)
    for slots, v_new in pending:
        w = np.rint(v_new / s_new)
        if np.abs(w).max() > INT16_MAX:
            raise ValueError(
                f"velocity code {np.abs(w).max():.0f} escapes int16 after reconciliation. "
                "That is supposed to be impossible: s_new is the max over a PARTITION of "
                "the particles, so no row can exceed it. The partition is broken."
            )
        st.write_velocities(slots, w.astype(np.int16))
    st.vel_scale = s_new

    stats = drift_and_migrate(st, c_drift, vel_scale_new=s_new)
    # both: `cap` is the SHAPE every buffer took, `cap_true` the max over tiles it
    # was quantized from. Reporting only one of them hides either the padding cost
    # or the shape churn, and the shape churn is what leaked.
    stats.update(cap=cap, cap_true=cap_true, n_tiles=len(cfg.tiles), vel_scale=s_new)
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


def run(st, cfg, coeffs, collect=None, census=False):
    """Advance `st` over a whole schedule. `coeffs` from `bullfrog_float_coeffs`."""
    cfg.validate()
    lead, fused = fused_drifts(coeffs)
    drift_and_migrate(st, lead)  # onto the first midpoint
    out = []
    # the buffer shape is carried ACROSS steps and only ever grows, so the run
    # visits at most a few shapes instead of one per step (M-v2-6 Stage 0)
    cap_shape = 0
    for k in range(len(fused)):
        stats = step(st, cfg, (coeffs[k][1], coeffs[k][2]), float(fused[k]), collect,
                     census=census, cap_shape=cap_shape)
        cap_shape = int(stats["cap"])
        if cfg.repack_every and (k + 1) % cfg.repack_every == 0:
            stats["repack"] = st.repack(brick_slack=cfg.brick_slack)
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
