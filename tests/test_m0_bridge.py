"""TEMPORARY migration bridge (delete at M1 close): the package's migrated
functions must BIT-MATCH the frozen _m0_common.py archive on fixed seeds, so
the M1 package provably computes the same numbers the M0 verdicts certified.

Marked slow: not part of the fast gate; runs in the full suite while the
migration is live. scripts/_m0_common.py is a frozen archive -- if this test
ever needs a fix, fix the PACKAGE, never the archive.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import _m0_common as m0  # noqa: E402

from inexor import codec, cosmology, diagnostics, ic, integrate, lpt, painting  # noqa: E402
from inexor.config import PLANCK, BoxConfig  # noqa: E402

pytestmark = pytest.mark.slow

N, L = 32, 200.0


def test_cosmology_bits():
    k = np.geomspace(1e-4, 50.0, 512)
    assert np.array_equal(cosmology.transfer_eh98(k, PLANCK), m0.transfer_eh98(k, m0.PLANCK))
    assert np.array_equal(cosmology.linear_power(k, PLANCK), m0.linear_power(k, m0.PLANCK))
    for a in (0.1, 0.5, 1.0):
        assert cosmology.growth_factor_a(a, PLANCK) == m0.growth_factor_a(a, m0.PLANCK)
        assert cosmology.growth_rate_a(a, PLANCK) == m0.growth_rate_a(a, m0.PLANCK)


def test_tables_and_ladder_bits():
    a_steps = m0.a_grid(0.1, 1.0, 10, "log")
    assert np.array_equal(a_steps, integrate.a_grid(0.1, 1.0, 10, "log"))
    lad_old = m0.ladder_constants(a_steps, m0.PLANCK, 2.0, 0.5)
    lad_new = integrate.ladder_for_schedule(a_steps, PLANCK, 2.0, 0.5)
    for name in ("alphas", "betas", "dD", "D_mid", "P", "c1", "c2", "kappa"):
        assert np.array_equal(getattr(lad_old, name), getattr(lad_new, name)), name
    for a0, a1 in ((0.1, 0.2), (0.5, 0.9)):
        assert integrate.kick_factor(a0, a1, PLANCK) == m0.kick_factor(a0, a1, m0.PLANCK)
        assert integrate.drift_factor(a0, a1, PLANCK) == m0.drift_factor(a0, a1, m0.PLANCK)
        a_c = 0.5 * (a0 + a1)
        assert integrate.fastpm_kick_factor(a0, a1, a_c, PLANCK) == m0.fastpm_kick_factor(
            a0, a1, a_c, m0.PLANCK
        )
        assert integrate.fastpm_drift_factor(a0, a1, a_c, PLANCK) == m0.fastpm_drift_factor(
            a0, a1, a_c, m0.PLANCK
        )


def test_ic_and_za_bits():
    key = jax.random.PRNGKey(7)
    d_old = m0.linear_delta0(key, N, L, m0.PLANCK)
    d_new = ic.gaussian_delta(key, N, L, PLANCK)
    assert jnp.array_equal(d_old, d_new)
    psi_old = m0.za_psi(d_old, N, L)
    psi_new = lpt.zeldovich_displacement(d_new, L)
    assert jnp.array_equal(psi_old, psi_new)
    x_old, v_old = m0.za_ics(d_old, N, L, 0.1, m0.PLANCK)
    x_new, v_new = lpt.za_ics(d_new, L, 0.1, PLANCK)
    assert jnp.array_equal(x_old, x_new) and jnp.array_equal(v_old, v_new)


def test_paint_and_force_bits():
    pos = jax.random.uniform(jax.random.PRNGKey(3), (20000, 3), minval=0.0, maxval=L)
    assert jnp.array_equal(m0.paint_int(pos, N, L, 12), painting.paint_int(pos, N, L, 12))
    assert jnp.array_equal(m0.paint_f32(pos, N, L), painting.paint_f32(pos, N, L))
    f_old = m0.make_force_fn(N, L, N**3, paint="int")
    f_new = __import__("inexor.forces", fromlist=["make_force_fn"]).make_force_fn(
        BoxConfig(n_mesh=N, box_size=L), paint="int"
    )
    assert jnp.array_equal(f_old(pos), f_new(pos))


def test_step_bits_through_pm():
    key = jax.random.PRNGKey(5)
    d = ic.gaussian_delta(key, 16, 100.0, PLANCK)
    x_phys, v = lpt.za_ics(d, 100.0, 0.1, PLANCK)
    a_steps = integrate.a_grid(0.1, 1.0, 8, "log")
    box = BoxConfig(n_mesh=16, box_size=100.0)
    unit_old = m0.ladder_constants(a_steps, m0.PLANCK, 1.0, 1.0)
    s_w0 = m0.s_w0_policy(float(jnp.max(jnp.abs(v))), unit_old.P[-1])
    lad_old = m0.ladder_constants(a_steps, m0.PLANCK, s_w0, box.s_x)
    lad_new = integrate.ladder_for_schedule(a_steps, PLANCK, s_w0, box.s_x)
    cs_old, cs_new = m0.step_consts(lad_old), integrate.step_consts(lad_new)
    force_old = m0.make_force_fn(16, 100.0, 16**3, paint="int")
    force_new = __import__("inexor.forces", fromlist=["make_force_fn"]).make_force_fn(
        box, paint="int"
    )
    xo, wo = m0.encode_x(x_phys, box.s_x), m0.encode_w(v, s_w0)
    xn, wn = codec.encode_x(x_phys, box.s_x), codec.encode_w(v, s_w0)
    assert jnp.array_equal(xo, xn) and jnp.array_equal(wo, wn)
    for k in range(8):
        co = m0.StepConsts(cs_old.c1[k], cs_old.c2[k], cs_old.kappa[k])
        cn = integrate.StepConsts(cs_new.c1[k], cs_new.c2[k], cs_new.kappa[k])
        xo, wo = m0.step_fwd(xo, wo, co, force_old, box.s_x)
        xn, wn = integrate.step_fwd(xn, wn, cn, force_new, box.s_x)
        assert jnp.array_equal(xo, xn) and jnp.array_equal(wo, wn), f"step {k}"


def test_estimator_equivalence_not_bits():
    """The pk estimator DELIBERATELY changed binning (mbody convention); assert
    both recover the same spectrum, not the same bins."""
    d = np.asarray(ic.gaussian_delta(jax.random.PRNGKey(2), 64, 500.0, PLANCK))
    kc_o, pk_o, _ = m0.pk_estimator(d, 500.0, n_bins=12, k_max=0.6 * np.pi * 64 / 500.0)
    kc_n, pk_n, _ = diagnostics.pk_estimator(d, 500.0, kmax=0.6 * np.pi * 64 / 500.0)
    p_ref_o = m0.linear_power(kc_o, m0.PLANCK)
    p_ref_n = cosmology.linear_power(kc_n, PLANCK)
    assert np.median(np.abs(pk_o / p_ref_o - 1.0)) < 0.2
    assert np.median(np.abs(pk_n / p_ref_n - 1.0)) < 0.2