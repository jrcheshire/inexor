"""lpt.py: kernel identities, 2LPT sign/skewness, and the D-time velocity
identity verified against a finite difference of the displaced positions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from inexor.config import PLANCK
from inexor.cosmology import growth_factor_2, growth_factor_a, growth_rate_2, growth_rate_a
from inexor.forces import make_force_fn
from inexor.ic import gaussian_delta
from inexor.lpt import (
    divergence,
    lagrangian_grid,
    lpt2_source,
    lpt_ics,
    second_order_displacement,
    za_ics,
    zeldovich_displacement,
)

N, L = 32, 200.0


@pytest.fixture(scope="module")
def delta0():
    return gaussian_delta(jax.random.PRNGKey(1), N, L, PLANCK)


def _mask_nyquist(field_k_shape=None):
    """Real-mesh mask that is 1 away from Nyquist-affected modes: we compare
    div Psi1 to -delta0 after removing the Nyquist planes of both."""
    kx = np.fft.fftfreq(N) * N
    keep = np.abs(kx) != N // 2
    mask = np.ones((N, N, N // 2 + 1), dtype=bool)
    mask &= keep[:, None, None]
    mask &= keep[None, :, None]
    mask[:, :, -1] = False
    return mask


def _strip_nyquist(mesh):
    dk = np.fft.rfftn(np.asarray(mesh, np.float64))
    dk[~_mask_nyquist()] = 0.0
    return np.fft.irfftn(dk, s=(N, N, N), axes=(0, 1, 2))


def test_div_psi1_equals_minus_delta(delta0):
    psi1 = zeldovich_displacement(delta0, L)
    div = _strip_nyquist(divergence(psi1, L))
    ref = _strip_nyquist(delta0)
    scale = np.max(np.abs(ref))
    assert np.max(np.abs(div + ref)) < 2e-5 * scale  # f32 FFT round-off class


def test_force_equals_za_identity(delta0):
    """The force solve and the ZA displacement apply the same ik/k^2 kernel via
    different code paths: ZA of delta equals the force path's spectral composition
    (`k_components`) on the same delta, at f32 roundoff."""
    from inexor.forces import k_components

    psi1 = zeldovich_displacement(delta0, L)
    ikx, iky, ikz, inv_k2 = k_components(N, L)
    dk = jnp.fft.rfftn(delta0)
    gx = jnp.fft.irfftn(dk * ikx * inv_k2, s=(N, N, N))
    gy = jnp.fft.irfftn(dk * iky * inv_k2, s=(N, N, N))
    gz = jnp.fft.irfftn(dk * ikz * inv_k2, s=(N, N, N))
    scale = float(jnp.max(jnp.abs(psi1)))
    for i, g in enumerate((gx, gy, gz)):
        assert float(jnp.max(jnp.abs(g.reshape(-1) - psi1[:, i]))) < 1e-6 * scale


def test_2lpt_source_quadratic_scaling(delta0):
    s1 = lpt2_source(delta0, L)
    s2 = lpt2_source(2.0 * delta0, L)
    assert jnp.allclose(s2, 4.0 * s1, rtol=1e-4, atol=1e-6 * float(jnp.max(jnp.abs(s1))))


def test_2lpt_skewness_sign(delta0):
    """2LPT positions must have MORE skewed density than ZA at the same D
    (collapse enhancement) -- the sign test that validates the Psi2 sign
    convention."""
    a_late = 1.0  # exaggerate the second-order effect
    x1, _ = lpt_ics(delta0, L, a_late, PLANCK, order=1)
    x2, _ = lpt_ics(delta0, L, a_late, PLANCK, order=2)
    from inexor.painting import density_contrast

    d1 = density_contrast(x1, N, L, N**3, paint="f32")
    d2 = density_contrast(x2, N, L, N**3, paint="f32")

    def skew(d):
        d = np.asarray(d, np.float64)
        return float(np.mean(d**3) / np.mean(d**2) ** 1.5)

    assert skew(d2) > skew(d1)


def test_v_d_identity_against_finite_difference(delta0):
    """v_D = dx/dD1 verified against a finite difference of the (unwrapped)
    2LPT displacement over a small growth interval -- confirms the derived
    coefficient -(D2 f2)/(D1 f1) (== +(6/7) D1 in the EdS-approx relation)."""
    a0 = 0.1
    da = 1e-3
    D1 = growth_factor_a(a0, PLANCK)
    psi1 = np.asarray(zeldovich_displacement(delta0, L), np.float64)
    psi2 = np.asarray(second_order_displacement(delta0, L), np.float64)

    def displacement_at(a):
        d1 = growth_factor_a(a, PLANCK)
        d2 = growth_factor_2(a, PLANCK)
        return d1 * psi1 - d2 * psi2

    dD = growth_factor_a(a0 + da, PLANCK) - growth_factor_a(a0 - da, PLANCK)
    v_fd = (displacement_at(a0 + da) - displacement_at(a0 - da)) / dD
    _, v = lpt_ics(delta0, L, a0, PLANCK, order=2)
    v = np.asarray(v, np.float64)
    scale = np.max(np.abs(v))
    assert np.max(np.abs(v - v_fd)) < 2e-3 * scale
    # and in the EdS approximation the coefficient reduces to +(6/7) D1
    coef = -(growth_factor_2(a0, PLANCK, "eds") * growth_rate_2(a0, PLANCK, "eds")) / (
        D1 * growth_rate_a(a0, PLANCK)
    )
    assert coef == pytest.approx((6.0 / 7.0) * D1, rel=1e-12)


def test_za_velocity_constant_in_d(delta0):
    _, v_early = za_ics(delta0, L, 0.1, PLANCK)
    _, v_late = za_ics(delta0, L, 0.9, PLANCK)
    assert jnp.array_equal(v_early, v_late)  # v_D == Psi1, exactly a-independent


def test_lagrangian_grid_forces_vanish(delta0):
    from inexor.config import BoxConfig

    q = lagrangian_grid(N, L)
    g = make_force_fn(BoxConfig(n_mesh=N, box_size=L), paint="f32")(q)
    assert float(jnp.max(jnp.abs(g))) == 0.0


def test_low_and_mid_residency_are_bitwise_identical(delta0, tmp_path):
    """The disk-staged "low" residency policy is a pure memory knob. Both policies must execute the identical
    per-element op sequence, so delta2 and the full lpt_ics state agree BIT
    FOR BIT -- and the six staged derivative files existing on disk is the
    structural proof that "low" never held them resident."""
    d0 = np.asarray(delta0, np.float64)
    a_mid = lpt2_source(d0, L, fdtype=np.float64, resident="mid")
    a_low = lpt2_source(d0, L, fdtype=np.float64, resident="low", workdir=str(tmp_path))
    assert np.array_equal(a_low, a_mid), "residency policy moved delta2 bits"
    staged = sorted(p.name for p in tmp_path.glob("phi_*.npy"))
    assert staged == ["phi_00.npy", "phi_01.npy", "phi_02.npy",
                      "phi_11.npy", "phi_12.npy", "phi_22.npy"]

    x_mid, v_mid = lpt_ics(d0, L, 0.1, PLANCK, order=2, fdtype=np.float64)
    x_low, v_low = lpt_ics(d0, L, 0.1, PLANCK, order=2, fdtype=np.float64,
                           resident="low", workdir=str(tmp_path))
    assert np.array_equal(x_low, x_mid) and np.array_equal(v_low, v_mid)

    # anti-vacuity: a perturbed density must move delta2
    d1 = d0.copy()
    d1[3, 4, 5] += 1e-6
    assert not np.array_equal(lpt2_source(d1, L, fdtype=np.float64), a_mid)
    # refusals: an unknown policy and a workdir-less "low" must both be loud
    with pytest.raises(ValueError, match="resident"):
        lpt2_source(d0, L, fdtype=np.float64, resident="high")
    with pytest.raises(ValueError, match="workdir"):
        lpt2_source(d0, L, fdtype=np.float64, resident="low")
