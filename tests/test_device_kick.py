"""D2b: the kick and per-brick quantize on the device, gated against `tile_task`.

THE ORACLE IS THE REAL HOST CODE, not a transcription of it. `engine.tile_task`
takes its short-force kernel as an argument, so a stub `one_tile` returning
forces this file chooses -- plus zero coarse meshes, which make the long arm
contribute exactly zero -- drives the genuine host kick, run scan and quantize
over inputs the device arm is then given verbatim. Re-spelling those four lines
as the oracle would have been testing the transcription, in a repo whose own
records say two implementations of one formula is how they drift apart.

Bitwise, and it can be: `max` over floats is exact and associative, so a
segmented max cannot disagree with a run scan whatever order it reduces in, and
the division and round-half-even that follow are IEEE-exact.
"""

import numpy as np
import pytest

from inexor import engine, forces, state
from inexor.codec import T9Layout

pytest.importorskip("jax")

# The geometry `tests/test_engine.py` uses, verbatim. Hand-picking a smaller one
# produced a tile whose brick span exceeded the brick grid, which
# `layout.brick_span` refuses for a good reason (a tile that wraps visits the
# same brick twice and paints its particles twice) -- and the failure surfaced as
# a stencil-containment error three layers down rather than as "your geometry is
# inconsistent". Take a validated one.
L_BOX, N_PART, N_FINE, N_COARSE, N_TILE, B_FINE = 32.0, 32, 64, 16, 16, 8
TILE = (0, 0, 0)


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _t9():
    return T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=2)


def _cfg():
    return engine.EngineConfig(box_size=L_BOX, n_part=N_PART, n_fine=N_FINE,
                               n_coarse=N_COARSE, n_tile=N_TILE, b_fine=B_FINE)


def _nb(cfg):
    return N_FINE // cfg.n_brick


def _built(**kw):
    cfg = _cfg()
    rng = np.random.default_rng(0)
    g = (np.arange(N_PART) + 0.5) * (L_BOX / N_PART)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    x = np.mod(q + rng.normal(scale=0.25 * L_BOX / N_PART, size=q.shape), L_BOX)
    v = np.random.default_rng(1).normal(scale=0.5, size=(len(x), 3))
    return cfg, state.SlotState.build(x, v, _t9(), _nb(cfg), **kw)


def _header(cfg, cap):
    """`C` exactly as `engine.step` builds it (engine.py:1260)."""
    return dict(cap=int(cap), n_tile=cfg.n_tile, n_brick=cfg.n_brick,
                n_fine=cfg.n_fine, n_coarse=cfg.n_coarse, box=cfg.box_size,
                coarse_cell=cfg.coarse_cell, cell=cfg.fine_cell,
                b_real=int(cfg._b_realized), alpha_k=0.87, bcoef=1.31)


def _run_host(cfg, st, bricks, cap, g_short_full):
    """The real `tile_task`, with a stub short force and zero coarse meshes.

    The stub returns the caller's forces at the padded shape and echoes the
    ownership mask it is handed, which is what the real `one_tile` does; the
    zero coarse meshes make `gather_coarse_subblock` contribute exactly zero, so
    `g_tot` is the array this file chose and the kick is the genuine one.
    """
    import jax.numpy as jnp

    def one_tile(u, live, own):
        return jnp.asarray(g_short_full), own, 0

    g_coarse = [np.zeros((N_COARSE,) * 3) for _ in range(3)]
    return engine.tile_task(st, one_tile, _header(cfg, cap), g_coarse, TILE,
                            bricks)


def _run_device(cfg, st, bricks, cap, g_short_full, res_owned_rows):
    from inexor.device import decode as ddec
    from inexor.device import kick as dkick

    plan = ddec.tile_decode_plan(st, bricks)
    dec = ddec.decode_rows(plan, st.off, st.w, st.vel_scale, st.arena_bucket,
                           st.arena_base, _t9(), _nb(cfg), cap)
    C = _header(cfg, cap)
    out = dkick.kick_and_quantize(
        v=dec["v"], g_short=g_short_full,
        g_long_rows=np.zeros((cap, 3)),
        owned=res_owned_rows, brick_index=dec["brick_index"],
        n_bricks=len(bricks), alpha_k=C["alpha_k"], bcoef=C["bcoef"])
    return plan, dec, out


def _setup(seed=3):
    """A state, a tile's bricks, a cap, chosen forces, and the owned mask.

    Ownership is taken from the host run rather than recomputed, so both arms
    are kicking exactly the same rows -- an ownership mask derived twice is the
    fault `forces.owned_mask_from_bricks` exists to prevent.
    """
    cfg, st = _built(arena_frac=0.25)
    bricks = st.tile_bricks(TILE, cfg.n_tile, cfg._b_realized, cfg.n_brick,
                            cfg.n_fine)
    m = len(st.decode_bricks(bricks)[0])
    cap = forces.capacity_shape(m + 6)
    g_short_full = np.random.default_rng(seed).normal(scale=0.5, size=(cap, 3))
    res = _run_host(cfg, st, bricks, cap, g_short_full)
    return cfg, st, bricks, cap, g_short_full, res


def _owned_rows_from(st, bricks, res, cap):
    """The host's owned mask as a row-aligned boolean of length `cap`."""
    slots, _x, _v = st.decode_bricks(bricks)
    own = np.zeros(cap, dtype=bool)
    own[: len(slots)] = np.isin(slots, res["slots_o"])
    return own


# ------------------------------------------------------------------- the gate


def test_the_kick_and_quantize_are_bitwise_the_host_tile_task():
    cfg, st, bricks, cap, g_short_full, res = _setup()
    assert not res["empty"], "vacuous: the host tile produced nothing"
    assert res["n_owned"] > 0
    own = _owned_rows_from(st, bricks, res, cap)
    assert own.sum() == res["n_owned"]

    _plan, dec, out = _run_device(cfg, st, bricks, cap, g_short_full, own)
    w_dev = np.asarray(out["w_codes"])
    # host `w_codes` are packed to owned rows in row order; the device keeps
    # row alignment, so select the same rows to compare like with like
    assert np.array_equal(w_dev[own], res["w_codes"]), "velocity codes differ"

    # the per-brick scales, for the bricks the host actually wrote
    scales_dev = np.asarray(out["scales"])
    counts = np.asarray(out["owned_counts"])
    got = {int(bricks[i]): float(scales_dev[i])
           for i in range(len(bricks)) if counts[i] > 0}
    want = {int(b): float(s)
            for b, s in zip(res["run_bricks"], res["run_scales"])}
    assert got == want, "per-brick velocity scales differ"


def test_the_comparison_can_fail():
    """Anti-vacuity: a different short force must move the codes."""
    cfg, st, bricks, cap, g_short_full, res = _setup()
    own = _owned_rows_from(st, bricks, res, cap)
    _p, _d, out = _run_device(cfg, st, bricks, cap, g_short_full, own)
    bumped = g_short_full.copy()
    bumped[np.flatnonzero(own)[0], 0] += 1.0
    _p, _d, out2 = _run_device(cfg, st, bricks, cap, bumped, own)
    assert not np.array_equal(np.asarray(out["w_codes"]),
                              np.asarray(out2["w_codes"]))


def test_the_scale_is_the_brick_max_and_the_extremes_land_on_int16_ends():
    """The T9 contract: scale = max|v| / 32767, so the extremes are exactly

    representable and nothing needs clamping. D-007 forbids a saturating op on
    integer state, so a code outside int16 would have to WRAP -- silently
    misrepresenting the fastest particles in the brick, which are the ones that
    matter. Checked rather than assumed.
    """
    from inexor.device import kick as dkick

    cfg, st, bricks, cap, g_short_full, res = _setup()
    own = _owned_rows_from(st, bricks, res, cap)
    _p, _d, out = _run_device(cfg, st, bricks, cap, g_short_full, own)
    w32 = np.asarray(out["w32"])[own]
    assert np.abs(w32).max() == dkick.INT16_MAX, (
        "the extreme did not land on 32767, so the scale is not the max")
    dkick.assert_int16_range_device(w32)


def test_a_brick_with_no_owned_rows_takes_scale_one_and_is_reported_empty():
    """`segment_max` seeds an empty segment at -inf; left alone that would make

    the scale -inf/32767 and every code in the brick a nan. The all-zero branch
    of `encode_velocities` is what it must match instead, and the caller needs
    to be told the brick is empty so it does not write a scale for a brick this
    tile does not own.
    """
    cfg, st, bricks, cap, g_short_full, res = _setup()
    own = _owned_rows_from(st, bricks, res, cap)
    # force one brick to have no owned rows
    _p, dec, _o = _run_device(cfg, st, bricks, cap, g_short_full, own)
    bi = np.asarray(dec["brick_index"])
    victim = int(bi[np.flatnonzero(own)[0]])
    own2 = own & (bi != victim)
    assert own2.sum() < own.sum(), "vacuous: no rows were removed"
    _p, _d, out = _run_device(cfg, st, bricks, cap, g_short_full, own2)
    scales = np.asarray(out["scales"])
    counts = np.asarray(out["owned_counts"])
    assert counts[victim] == 0
    assert scales[victim] == 1.0, "an empty brick must take the unit scale"
    assert np.isfinite(scales).all(), "segment_max's -inf seed leaked"
    assert np.asarray(out["w_codes"])[bi == victim].max() == 0


def test_masked_and_padded_rows_write_zero_codes():
    cfg, st, bricks, cap, g_short_full, res = _setup()
    own = _owned_rows_from(st, bricks, res, cap)
    _p, _d, out = _run_device(cfg, st, bricks, cap, g_short_full, own)
    w = np.asarray(out["w_codes"])
    assert (w[~own] == 0).all(), "a row this tile does not own carries a code"


def test_the_segmented_max_does_not_need_contiguous_runs():
    """The host form is only correct because a brick's rows are contiguous --

    it scans runs, and a brick appearing twice would let the second run
    overwrite the first one's scale. The device form reduces on the brick index
    and has no such assumption, so a permuted row order must give the same
    per-brick scales. That is a real robustness difference and worth pinning,
    not a restatement of the gate above.
    """
    from inexor.device import kick as dkick

    cfg, st, bricks, cap, g_short_full, res = _setup()
    own = _owned_rows_from(st, bricks, res, cap)
    _p, dec, out = _run_device(cfg, st, bricks, cap, g_short_full, own)

    rng = np.random.default_rng(7)
    perm = rng.permutation(cap)
    C = _header(cfg, cap)
    out_p = dkick.kick_and_quantize(
        v=np.asarray(dec["v"])[perm], g_short=g_short_full[perm],
        g_long_rows=np.zeros((cap, 3)), owned=own[perm],
        brick_index=np.asarray(dec["brick_index"])[perm], n_bricks=len(bricks),
        alpha_k=C["alpha_k"], bcoef=C["bcoef"])
    assert np.array_equal(np.asarray(out["scales"]),
                          np.asarray(out_p["scales"]))
