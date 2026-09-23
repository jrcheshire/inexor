"""The coarse match factor's assignment order.

The coarse arm paints and gathers TSC while the ratified factor divides out a
CIC window, so the long force keeps sinc^2 per axis of the coarse window.
`coarse_match_order=3` corrects it; 2 stays the default so every oracle and gate
remains bitwise. These tests pin both halves: the default is the ratified
expression unchanged, and the corrected arm meets its Ewald target where the
ratified one measurably does not.
"""

import argparse
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))

from inexor import forces  # noqa: E402
from inexor.engine import EngineConfig, _FINGERPRINTED, checkpoint_fingerprint  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _cfg(**kw):
    return EngineConfig(box_size=64.0, n_part=64, n_fine=256, n_coarse=64, n_tile=64,
                        b_fine=32, **kw)


def test_the_default_is_the_ratified_two_tuple():
    assert _cfg().coarse_match == (1.0, 0.25)


def test_order_two_spelled_out_is_bitwise_the_ratified_factor():
    a = forces.coarse_kernel_parts(16, 16.0, "long", r_s=1.0, match=(1.0, 0.25))
    b = forces.coarse_kernel_parts(16, 16.0, "long", r_s=1.0, match=(1.0, 0.25, 2, 2))
    assert np.array_equal(a["mf"], b["mf"])


def test_order_three_is_the_tsc_over_cic_ratio():
    """An identity, not a tolerance: the factor is W_cic(fine)^2 / W_tsc(coarse)^2
    with W = prod sinc^order, and differs from the ratified one by exactly the
    sinc^2 per axis the ratified one leaves behind."""
    n, L, dc, df = 16, 16.0, 1.0, 0.25
    m2 = forces.coarse_kernel_parts(n, L, "long", r_s=1.0, match=(dc, df))["mf"]
    m3 = forces.coarse_kernel_parts(n, L, "long", r_s=1.0, match=(dc, df, 3, 2))["mf"]
    w_extra = forces.assignment_window((n,) * 3, dc, 1) ** 2
    np.testing.assert_allclose(m2, m3 * w_extra, rtol=1e-13)
    assert not np.array_equal(m2, m3)


def test_a_malformed_match_is_refused():
    with pytest.raises(ValueError):
        forces.coarse_kernel_parts(16, 16.0, "long", r_s=1.0, match=(1.0, 0.25, 3))


def test_only_orders_two_and_three_are_accepted():
    with pytest.raises(ValueError):
        _cfg(coarse_match_order=4)


def test_the_default_fingerprint_is_the_pre_knob_one():
    """Every checkpoint on disk, the 4096^3 generations included, was written
    before the knob existed and must still resume."""
    import hashlib
    import json

    cfg = _cfg()
    co = np.linspace(0.0, 1.0, 7)
    h = hashlib.sha256()
    h.update(json.dumps({k: getattr(cfg, k) for k in _FINGERPRINTED}, sort_keys=True).encode())
    h.update(np.ascontiguousarray(co, dtype=np.float64).tobytes())
    assert checkpoint_fingerprint(cfg, co) == h.hexdigest()
    assert checkpoint_fingerprint(_cfg(coarse_match_order=3), co) != h.hexdigest()


def test_the_driver_passes_the_order_to_the_engine():
    import v2_m6_realization as rlz

    a = argparse.Namespace(slack=0.10, tile_workers=1, checkpoint_every=5,
                           migrate_pooled=None, eject_kernel="jax", coarse_match_order=3)
    assert rlz._engine_config(rlz._geom("cdev8"), a, "/tmp/nowhere").coarse_match_order == 3
    del a.coarse_match_order
    assert rlz._engine_config(rlz._geom("cdev8"), a, "/tmp/nowhere").coarse_match_order == 2


def _long_arm_error(order):
    """Mean long-arm error as a fraction of the total Ewald force, over
    r = 1.6-3.4 Mpc/h, at production cells (fine 0.25, coarse 1.0, r_s 1.0) in
    a 32 Mpc/h box. The window where the ratified arm is most wrong."""
    import v2_force_profile as fp

    L, n_fine, n_coarse = 32.0, 128, 32
    fine, coarse, r_s = L / n_fine, L / n_coarse, 1.0
    n_total = 64**3
    m = L**3 / n_total
    rng = np.random.default_rng(3)
    src = rng.uniform(0.0, L, size=3)
    u = rng.normal(size=(400, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    r = rng.uniform(1.6, 3.4, size=400)
    tests = np.mod(src + u * r[:, None], L)
    d = fp._min_image(tests - src, L)

    def long(x):
        match = (coarse, fine) if order == 2 else (coarse, fine, order, 2)
        g, _ = forces.force_global(np.asarray(x, float), n_coarse, L, n_total, "long",
                                   r_s=r_s, match=match, assign="tsc", paint="int")
        return np.asarray(g)

    g_src = long(np.vstack([tests, src[None]]))[:-1] - long(tests)
    tot = m * fp.ewald_total(d, L)
    ref_long = tot - m * fp.ewald_real(d, L, 1.0 / (2.0 * r_s), n_img=1)
    rhat = d / np.linalg.norm(d, axis=1)[:, None]
    fr = lambda g: -(g * rhat).sum(1)  # noqa: E731
    return float(np.mean((fr(g_src) - fr(ref_long)) / fr(tot)))


def test_the_corrected_long_arm_meets_its_ewald_target():
    """Against a continuum reference, not a parity. The bar is 1% of the total
    force; measured -0.47% for the corrected arm and -3.71% for the ratified one
    (2026-09-23)."""
    err3 = _long_arm_error(3)
    assert abs(err3) < 1e-2, f"order-3 long arm off its Ewald target by {err3:+.4f}"


def test_the_ewald_gate_can_fail():
    """Control: the same gate on the ratified factor must FAIL, or the test
    above cannot tell the two arms apart."""
    err2 = _long_arm_error(2)
    assert err2 < -2e-2, f"ratified long arm reads {err2:+.4f}; the gate lost its power"


if __name__ == "__main__":
    for o in (2, 3):
        print(o, _long_arm_error(o))
