"""The device lane's host-side reads on a node-local `SlotState`.

On one rank's cut (`restrict_to_slabs`), the per-tile decode plan, the coarse paint's chunk
inputs and the migrate's per-slab eject index must equal the whole state's on owned bricks,
up to row numbering (the cut renumbers rows from 0 and has its own arena), and refuse bricks
outside the owned range.
"""

import numpy as np
import pytest

pytest.importorskip("jax")

from inexor.device import decode as ddecode  # noqa: E402
from inexor.device import migrate as dmig  # noqa: E402
from inexor.device import paint as dpaint  # noqa: E402
from tests.test_partial_state import (  # noqa: E402
    NB,
    evolved_state,
    rank_slabs,
    restrict_to_slabs,
)


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def whole():
    st = evolved_state()
    assert st.arena_used > 0, "vacuous: no arena residents"
    return st


def _arena_rows(st, slots):
    """(off, w, bucket) of absolute arena `slots`, for comparing across numberings."""
    s = np.asarray(slots, dtype=np.int64)
    return st.off[s], st.w[s], st.arena_bucket[s - int(st.arena_base)]


def _cuts(whole, n_ranks):
    return [restrict_to_slabs(whole, s) for s in rank_slabs(NB, n_ranks)]


@pytest.mark.parametrize("n_ranks", [1, 2, 4])
def test_the_decode_plan_of_owned_bricks_is_the_whole_states(whole, n_ranks):
    nb2 = NB * NB
    for part in _cuts(whole, n_ranks):
        lo, hi = part.owned_slabs
        # one brick column per owned slab, and a scattered set in non-ascending order
        rng = np.random.default_rng(lo)
        bricks = rng.permutation(np.arange(lo * nb2, hi * nb2))[:37]
        got = ddecode.tile_decode_plan(part, bricks)
        want = ddecode.tile_decode_plan(whole, bricks)
        for k in ("bricks", "occ", "live_counts", "arena_counts", "member_counts",
                  "row_offsets", "n_rows", "p3"):
            np.testing.assert_array_equal(got[k], want[k], err_msg=k)
        np.testing.assert_array_equal(got["starts"] - part.brick_start[bricks],
                                      want["starts"] - whole.brick_start[bricks])
        for i in range(len(bricks)):
            n = int(got["arena_counts"][i])
            for a, b in zip(_arena_rows(part, got["arena_slots"][i, :n]),
                            _arena_rows(whole, want["arena_slots"][i, :n])):
                np.testing.assert_array_equal(a, b)
        assert got["arena_counts"].sum() > 0 or n_ranks > 1


def test_a_decode_plan_outside_the_owned_bricks_refuses(whole):
    part = _cuts(whole, 4)[1]
    blo, bhi = part.owned_bricks
    for b in (blo - 1, bhi):
        with pytest.raises(IndexError, match="outside this state's owned bricks"):
            ddecode.tile_decode_plan(part, [blo, b])


@pytest.mark.parametrize("n_ranks", [1, 2, 4])
def test_paint_chunk_inputs_of_owned_chunks_are_the_whole_states(whole, n_ranks):
    L = dpaint.default_chunk_bricks(NB)
    rows_w = dpaint.chunk_rows(whole, L)
    pad = int(rows_w.max())
    shapes = dpaint.step_shapes(whole, L, pad)
    seen = 0
    for part in _cuts(whole, n_ranks):
        blo, bhi = part.owned_bricks
        rows_p = dpaint.chunk_rows(part, L)
        own = np.zeros(len(rows_w), dtype=bool)
        own[blo // L:bhi // L] = True
        np.testing.assert_array_equal(rows_p[own], rows_w[own])
        assert not rows_p[~own].any()
        for gi in np.flatnonzero(own):
            bricks = np.arange(gi * L, (gi + 1) * L, dtype=np.int64)
            got = dpaint.slab_window_fixed(part, bricks, shapes)
            want = dpaint.slab_window_fixed(whole, bricks, shapes)
            for k in ("starts", "occ", "live_counts", "arena_slots", "row_offsets",
                      "bricks", "off", "arena_bucket", "n_rows"):
                np.testing.assert_array_equal(got[k], want[k], err_msg=k)
            seen += 1
    assert seen == len(rows_w)


@pytest.mark.parametrize("n_ranks", [1, 2, 4])
def test_the_eject_index_of_owned_slabs_is_the_whole_states(whole, n_ranks):
    ar_w = dmig.pass_arena_index(whole)
    for part in _cuts(whole, n_ranks):
        ar_p = dmig.pass_arena_index(part)
        for s in range(*part.owned_slabs):
            got = dmig._slab_index(part, s, *ar_p)
            want = dmig._slab_index(whole, s, *ar_w)
            for k in ("lo_b", "hi_b", "span", "occ", "live", "ar_offsets", "row_offsets",
                      "n_rows", "n_ar", "a_cap", "ar_bucket", "cap"):
                np.testing.assert_array_equal(got[k], want[k], err_msg=k)
            for a, b in zip(_arena_rows(part, got["rows_a"]), _arena_rows(whole, want["rows_a"])):
                np.testing.assert_array_equal(a, b)
        blo = part.owned_bricks[0]
        if blo:
            with pytest.raises(IndexError):
                dmig._slab_index(part, part.owned_slabs[0] - 1, *ar_p)
