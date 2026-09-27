"""The coarse mesh painted across cards, gated bitwise against `engine.coarse_delta_streamed`.

Integer addition is independent of which card adds a chunk, and the ghost fold is more integer
addition, so every card count must match to the bit. Card counts above the backend's device
count replicate handles (partition, threads, fold and per-card decode still run); with
`--xla_force_host_platform_device_count=4` or four GPUs it is a true multi-device gate.
"""

import numpy as np
import pytest

pytest.importorskip("jax")

from inexor import engine, state  # noqa: E402
from inexor.codec import T9Layout  # noqa: E402
from inexor.device import paint as dpaint  # noqa: E402

# tests/test_engine.py's validated smoke geometry.
L_BOX, N_PART, N_FINE, N_COARSE, N_TILE, B_FINE = 32.0, 32, 64, 16, 16, 8


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _cfg(**kw):
    return engine.EngineConfig(box_size=L_BOX, n_part=N_PART, n_fine=N_FINE,
                               n_coarse=N_COARSE, n_tile=N_TILE, b_fine=B_FINE, **kw)


def _state(cfg, seed=0, arena=False):
    rng = np.random.default_rng(seed)
    g = (np.arange(N_PART) + 0.5) * (L_BOX / N_PART)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    x = np.mod(q + rng.normal(scale=0.25 * L_BOX / N_PART, size=q.shape), L_BOX)
    v = np.random.default_rng(seed + 1).normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=2)
    nb = N_FINE // cfg.n_brick
    if not arena:
        return state.SlotState.build(x, v, t9, nb, arena_frac=0.05)
    st = state.SlotState.build(x, v, t9, nb, brick_slack=0.0, arena_frac=0.30)
    state.drift_and_migrate(st, 2.0)
    return st


def _devices(w):
    import jax

    devs = jax.devices()
    return [devs[i % len(devs)] for i in range(w)]


def _nontrivial(delta):
    assert float(np.abs(delta).max()) > 0.1, "degenerate density; comparison is vacuous"


# ------------------------------------------------------------------ the gate


@pytest.mark.parametrize("arena", [False, True])
@pytest.mark.parametrize("w", [1, 2, 4])
def test_card_density_is_bitwise_the_host_streamed_density(w, arena):
    cfg = _cfg()
    st = _state(cfg, 4, arena=arena)
    if arena:
        assert st.arena_used > 0, "VACUOUS: no arena residents"
    want = engine.coarse_delta_streamed(st, cfg)
    host = {}
    dpaint.coarse_delta_device(st, cfg, stats=host)
    s = {}
    shards = dpaint.coarse_delta_cards(st, cfg, devices=_devices(w), stats=s)
    assert s["coarse_cards"] == w and len(shards) == w
    assert s["coarse_device_chunks"] == host["coarse_device_chunks"], \
        "the cards did not paint every chunk the host-mesh path paints"
    assert all(c > 0 for c in s["coarse_card_chunks"]), \
        f"a card painted nothing ({s['coarse_card_chunks']}): it did not participate"
    assert s["coarse_ghost_planes_nonzero"] > 0, \
        "VACUOUS: no chunk wrote past a card's edge, so the fold moved nothing"
    got = dpaint.gather_card_delta(shards)
    assert got.dtype == want.dtype
    assert np.array_equal(got, want), f"{w} card(s) moved a bit of the density"
    _nontrivial(got)


def test_the_shards_tile_the_mesh_and_live_on_their_cards():
    cfg = _cfg()
    st = _state(cfg, 2)
    devs = _devices(4)
    shards = dpaint.coarse_delta_cards(st, cfg, devices=devs)
    assert shards[0]["lo"] == 0 and shards[-1]["hi"] == N_COARSE
    for a, b in zip(shards, shards[1:]):
        assert a["hi"] == b["lo"]
    for s, dev in zip(shards, devs):
        assert s["delta"].shape == (s["hi"] - s["lo"], N_COARSE, N_COARSE)
        assert s["delta"].devices() == {dev}


def test_chunk_size_does_not_move_a_bit_on_cards():
    cfg = _cfg()
    st = _state(cfg, 4, arena=True)
    nb = st.bricks_per_side
    want = engine.coarse_delta_streamed(st, cfg)
    s_q, s_s = {}, {}
    q = dpaint.coarse_delta_cards(st, cfg, devices=_devices(4), stats=s_q)
    sl = dpaint.coarse_delta_cards(st, cfg, devices=_devices(4), stats=s_s,
                                   chunk_bricks=nb * nb)
    assert s_q["coarse_device_chunks"] > s_s["coarse_device_chunks"], \
        "the chunk-size knob did not change the number of chunks"
    assert np.array_equal(dpaint.gather_card_delta(q), want)
    assert np.array_equal(dpaint.gather_card_delta(sl), want)


def test_census_matches_the_host():
    cfg = _cfg()
    st = _state(cfg, 4, arena=True)
    h, s = {}, {}
    engine.coarse_delta_streamed(st, cfg, stats=h, census=True)
    dpaint.coarse_delta_cards(st, cfg, devices=_devices(2), stats=s, census=True)
    assert s["coarse_peak_int"] == h["coarse_peak_int"]
    assert s["coarse_cells_inexact_f32"] == h["coarse_cells_inexact_f32"]


def test_skipping_the_fold_moves_the_density():
    """Control: the ghost planes carry real mass, so skipping the fold must fail the gate."""
    cfg = _cfg()
    st = _state(cfg, 2)
    want = engine.coarse_delta_streamed(st, cfg)
    got = dpaint.gather_card_delta(
        dpaint.coarse_delta_cards(st, cfg, devices=_devices(2), fold=False))
    assert not np.array_equal(got, want)


# ------------------------------------------------- the contracts around it


def test_a_block_outside_the_card_is_refused():
    import jax.numpy as jnp

    acc = dpaint.CardInt64Accumulator(N_COARSE, 0, 4)
    sub = jnp.zeros((5, 5, 5), dtype=jnp.int32)
    acc.add(sub, np.array([15, 0, 0]), np.array([5, 5, 5]))  # x planes -1..3: in range
    with pytest.raises(ValueError, match="not inside"):
        acc.add(sub, np.array([8, 0, 0]), np.array([5, 5, 5]))


def test_a_full_x_axis_chunk_is_refused_across_cards():
    cfg = _cfg()
    st = _state(cfg, 2)
    L = st.bricks_per_side**3
    with pytest.raises(ValueError, match="full x axis"):
        dpaint.coarse_delta_cards(st, cfg, devices=_devices(2), chunk_bricks=L)
    got = dpaint.coarse_delta_cards(st, cfg, devices=_devices(1), chunk_bricks=L)
    assert np.array_equal(dpaint.gather_card_delta(got),
                          engine.coarse_delta_streamed(st, cfg))


def test_decode_before_fold_is_refused():
    acc = dpaint.CardInt64Accumulator(N_COARSE, 0, N_COARSE)
    with pytest.raises(RuntimeError, match="before fold"):
        dpaint._delta_on_cards([acc], _cfg())


def test_a_second_fold_is_refused():
    accs = [dpaint.CardInt64Accumulator(N_COARSE, 0, 8),
            dpaint.CardInt64Accumulator(N_COARSE, 8, N_COARSE)]
    dpaint.fold_ghosts(accs)
    with pytest.raises(RuntimeError, match="twice"):
        dpaint.fold_ghosts(accs)


def test_a_peak_past_int32_is_refused():
    acc = dpaint.CardInt64Accumulator(N_COARSE, 0, N_COARSE)
    acc.mesh = acc.mesh.at[dpaint.ACC_GHOST_LO, 0, 0].set(2**31)
    acc.folded = True
    with pytest.raises(ValueError, match="past int32"):
        dpaint._delta_on_cards([acc], _cfg())


def test_x64_off_is_refused():
    import jax

    cfg = _cfg()
    st = _state(cfg, 2)
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            dpaint.coarse_delta_cards(st, cfg)
    finally:
        jax.config.update("jax_enable_x64", prev)


# ------------------------------------------------ the solve reads the cards


@pytest.mark.parametrize("coarse_dtype", ["float32", "float64"])
@pytest.mark.parametrize("w", [1, 4])
def test_the_solve_from_the_cards_is_bitwise_the_host_density_solve(w, coarse_dtype):
    from inexor import forces

    cfg = _cfg(coarse_dtype=coarse_dtype)
    st = _state(cfg, 4, arena=True)
    host = engine.coarse_delta_streamed(st, cfg)
    shards = dpaint.coarse_delta_cards(st, cfg, devices=_devices(w))
    want = forces.coarse_force_meshes(host, N_COARSE, L_BOX, "long", r_s=cfg.r_s)
    got = forces.coarse_force_meshes(shards, N_COARSE, L_BOX, "long", r_s=cfg.r_s)
    for i in range(3):
        assert got[i].dtype == want[i].dtype == np.dtype(coarse_dtype)
        assert np.array_equal(got[i], want[i]), f"component {i}, {w} card(s)"
    assert float(np.abs(want[0]).max()) > 0, "vacuous: zero force"


def test_a_monolithic_solve_of_card_shards_is_refused():
    from inexor import forces

    cfg = _cfg()
    shards = dpaint.coarse_delta_cards(_state(cfg, 2), cfg, devices=_devices(2))
    with pytest.raises(ValueError, match="factorized"):
        forces.coarse_force_meshes(shards, N_COARSE, L_BOX, "long", r_s=cfg.r_s,
                                   transform="monolithic")
