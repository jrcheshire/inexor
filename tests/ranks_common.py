"""Shared set-up for the across-ranks tests (`tests/test_ranks_*.py`).

`cdev8-tile32` has 8 tile planes and 32 brick slabs, so it hosts up to 4 ranks x 2 cards.
`whole_state` is a jittered lattice at that geometry migrated into a populated arena (cached:
copy it before mutating); `rank_cfg` is the production device lane; `rank_devices` gives a
rank disjoint emulated devices while the backend has enough.
"""

import copy
import functools

import numpy as np

from inexor import state
from inexor.codec import T9Layout
from inexor.plan import PRESETS, engine_config

PRESET = "cdev8-tile32"


def rank_cfg(cards, **kw):
    return engine_config(PRESET, coarse_backend="device", tile_backend="device",
                         migrate_backend="device", device_cards=cards, tile_workers=1, **kw)


@functools.lru_cache(maxsize=None)
def _whole(seed, drift):
    p = PRESETS[PRESET]
    n, box = p["n_part"], p["box"]
    rng = np.random.default_rng(seed)
    q = (np.arange(n) + 0.5) * (box / n)
    x = np.stack(np.meshgrid(q, q, q, indexing="ij"), axis=-1).reshape(-1, 3)
    x = np.mod(x + rng.normal(scale=0.3 * box / n, size=x.shape), box)
    v = rng.normal(scale=0.5, size=x.shape)
    cfg = rank_cfg(1)
    st = state.SlotState.build(x, v, T9Layout(box_size=box, n_part=n, bucket_cells=2),
                               cfg.n_fine // cfg.n_brick, brick_slack=0.0, arena_frac=0.3,
                               with_ids=False)
    state.drift_and_migrate(st, drift)
    assert st.arena_used > 0, "vacuous: no arena residents"
    return st


def whole_state(seed=11, drift=0.3):
    """A fresh copy of the cached whole state (built with x64 on, whoever asks first)."""
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        return copy.deepcopy(_whole(seed, drift))
    finally:
        jax.config.update("jax_enable_x64", prev)


def rank_devices(rank, cards):
    """Rank `rank`'s cards (None for one card, jax's default device)."""
    import jax

    if cards == 1:
        return None
    devs = jax.devices()
    return [devs[(rank * cards + k) % len(devs)] for k in range(cards)]
