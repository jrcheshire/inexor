"""T9 emission on the cards (`device.emit.emit_t9_slabs_cards`) vs host `icgen._emit_t9_slabs`.

On the CPU backend every slab file's payload must be bitwise the host's. Card counts above the
device count replicate handles; `--xla_force_host_platform_device_count=4` gives a true
multi-device gate.
"""

import json
import os

import jax
import numpy as np
import pytest

from inexor import icgen
from inexor.codec import T9Layout
from inexor.device import emit

ARRAYS = ("occupancy", "off", "w", "scale")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _devices(w):
    devs = jax.devices()
    return [devs[i % len(devs)] for i in range(w)]


def _fields(n, box, nb, dt, seed=0, reach=0.9):
    """Displacements up to `reach` brick slabs in every axis, planted seam crossers, v ~ N(0,1)."""
    dt = np.dtype(dt)
    rng = np.random.default_rng(seed)
    depth = box / nb
    u = [rng.uniform(-reach, reach, (n, n, n)).astype(dt) * dt.type(depth) for _ in range(3)]
    u[0][0, :, :] = dt.type(-reach * depth)       # first plane back across the seam
    u[0][-1, :, :] = dt.type(reach * depth)       # last plane forward across it
    v = [rng.standard_normal((n, n, n)).astype(dt) for _ in range(3)]
    return u, v


def _host(tmp, u, v, t9, n, box, nb, dt):
    d = str(tmp / "host")
    os.makedirs(d)
    written, total = icgen._emit_t9_slabs(d, [icgen._HostField(a) for a in u],
                                          [icgen._HostField(a) for a in v], t9, n, box, nb,
                                          np.dtype(dt), n, 1)
    return d, written, total


def _cards(tmp, u, v, t9, n, box, nb, dt, w, name="cards", planes=emit.PLANES_PER_CALL):
    d = str(tmp / name)
    os.makedirs(d)
    shards = emit.shards_from_host(u[0], emit.card_slab_ranges(n, nb, _devices(w), 1))
    t = {}
    written, total = emit.emit_t9_slabs_cards(d, shards, u[1], u[2],
                                              [icgen._HostField(a) for a in v], t9, n, box,
                                              nb, np.dtype(dt), 1, planes_per_call=planes,
                                              timings=t)
    return d, written, total, t


def _payload(d, names):
    out = {}
    for f in names:
        with np.load(os.path.join(d, f)) as z:
            out[f] = dict(meta=json.loads(str(z["meta"])), **{k: z[k] for k in ARRAYS})
    return out


def _same(a, b):
    assert list(a) == list(b)
    for f in a:
        assert a[f]["meta"] == b[f]["meta"], f"{f}: meta differs"
        for k in ARRAYS:
            assert a[f][k].dtype == b[f][k].dtype
            assert np.array_equal(a[f][k], b[f][k]), f"{f}:{k} moved bits"


def _cpu_only():
    if jax.devices()[0].platform != "cpu":
        pytest.skip("card == host emission bitwise is a CPU-backend gate")


@pytest.mark.parametrize("n,nb,dt", [(32, 4, np.float32), (32, 4, np.float64),
                                     (64, 8, np.float32)])
@pytest.mark.parametrize("w", [1, 2, 4])
def test_card_emission_is_bitwise_the_host_emission(tmp_path, n, nb, dt, w):
    _cpu_only()
    box = float(n)
    t9 = T9Layout(box, n, 2)
    u, v = _fields(n, box, nb, dt)
    hd, hw, ht = _host(tmp_path, u, v, t9, n, box, nb, dt)
    cd, cw, ct, timings = _cards(tmp_path, u, v, t9, n, box, nb, dt, w)
    assert ht == ct == n**3
    assert sorted(hw) == cw == [f"t9_slab_{d:04d}.npz" for d in range(nb)]
    _same(_payload(hd, cw), _payload(cd, cw))
    assert set(timings) == {"upload_s", "source_s", "dest_s", "write_s"}


def test_card_counts_and_plane_chunks_are_bitwise_on_any_backend(tmp_path):
    n, nb, dt = 32, 4, np.float32
    box = float(n)
    t9 = T9Layout(box, n, 2)
    u, v = _fields(n, box, nb, dt, seed=1)
    d1, names, _, _ = _cards(tmp_path, u, v, t9, n, box, nb, dt, 1, "one")
    d4, _, _, _ = _cards(tmp_path, u, v, t9, n, box, nb, dt, 4, "four", planes=8)
    d2, _, _, _ = _cards(tmp_path, u, v, t9, n, box, nb, dt, 2, "two", planes=1)
    ref = _payload(d1, names)
    _same(ref, _payload(d4, names))
    _same(ref, _payload(d2, names))


def test_the_seam_and_slab_crossers_are_exercised(tmp_path):
    """Control: the fixture must move rows across slabs, and across x = 0, both ways."""
    n, nb = 32, 4
    box = float(n)
    t9 = T9Layout(box, n, 2)
    u, v = _fields(n, box, nb, np.float32)
    cd, names, _, _ = _cards(tmp_path, u, v, t9, n, box, nb, np.float32, 1)
    rows = [len(_payload(cd, [f])[f]["off"]) for f in names]
    assert rows != [n**3 // nb] * nb, "no slab gained or lost a row"
    assert sum(rows) == n**3


def test_a_row_past_the_window_is_refused_on_both_paths(tmp_path):
    n, nb, dt = 32, 4, np.float32
    box = float(n)
    t9 = T9Layout(box, n, 2)
    u, v = _fields(n, box, nb, dt)
    # plane 10 sits in slab 1 at x = 10; +1.9 bricks lands in slab 3, two slabs away
    u[0][10, 3, 3] = dt(1.9 * box / nb)
    with pytest.raises(RuntimeError, match="window"):
        _host(tmp_path, u, v, t9, n, box, nb, dt)
    with pytest.raises(RuntimeError, match="window"):
        _cards(tmp_path, u, v, t9, n, box, nb, dt, 2)


def test_a_non_power_of_two_quantum_is_refused(tmp_path):
    n, nb = 32, 4
    box = 30.0
    t9 = T9Layout(box, n, 2)
    u, v = _fields(n, box, nb, np.float32)
    with pytest.raises(ValueError, match="power of two"):
        _cards(tmp_path, u, v, t9, n, box, nb, np.float32, 1)


def test_card_slab_ranges_cover_each_cards_sources():
    r = emit.card_slab_ranges(64, 8, _devices(4), 1)
    assert [(x["lo"], x["hi"]) for x in r] == [(0, 2), (2, 4), (4, 6), (6, 8)]
    assert [(x["x0"], x["nx"]) for x in r] == [(-8, 32), (8, 32), (24, 32), (40, 32)]
