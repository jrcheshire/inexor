"""Shared set-up for the across-ranks tests (`tests/test_ranks_*.py`).

`cdev8-tile32` has 8 tile planes and 32 brick slabs, so it hosts up to 4 ranks x 2 cards.
`whole_state` is a jittered lattice at that geometry migrated into a populated arena (cached:
copy it before mutating); `rank_cfg` is the production device lane; `rank_devices` gives a
rank disjoint emulated devices while the backend has enough.

`RANKS_MARKS` go on every rank test module. They assert bit equality of float results, so on
GPU they are `detflag` tests; on macOS-arm64 the multithreaded XLA-CPU pool makes a jitted
FFT's bytes vary run to run at some sizes (cdev8-tile32's 48^3 tile mesh among them), so there
they need `--xla_cpu_multi_thread_eigen=false` and skip visibly without it (`pixi run
test-ranks` sets it).
"""

import copy
import functools
import hashlib
import os
import sys

import numpy as np
import pytest

from inexor import state
from inexor.codec import T9Layout
from inexor.plan import PRESETS, engine_config

PRESET = "cdev8-tile32"

STABLE_FFT_FLAG = "--xla_cpu_multi_thread_eigen=false"
RANKS_MARKS = [
    pytest.mark.detflag,
    pytest.mark.skipif(
        sys.platform == "darwin" and STABLE_FFT_FLAG not in os.environ.get("XLA_FLAGS", ""),
        reason=f"macOS XLA-CPU's threaded FFT is not bitwise stable run to run; bit-equality "
               f"gates need XLA_FLAGS={STABLE_FFT_FLAG} (pixi run test-ranks)"),
]


def need_devices(n):
    """Skip unless the backend has `n` devices (the config refuses more cards than that)."""
    import jax

    if len(jax.devices()) < n:
        pytest.skip(f"needs {n} jax devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)")


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


def hashes(ckpt_dir):
    """{gen/file: sha256} over the complete generations (with a manifest) under `ckpt_dir`."""
    out = {}
    for gen in ("gen0", "gen1"):
        d = os.path.join(ckpt_dir, gen)
        if not os.path.exists(os.path.join(d, "manifest.json")):
            continue
        for name in sorted(os.listdir(d)):
            with open(os.path.join(d, name), "rb") as fh:
                out[f"{gen}/{name}"] = hashlib.sha256(fh.read()).hexdigest()
    return out
