"""ic.py: spectrum recovery, f_NL round trip + linearity, COBE-scale phi,
tree-level template sanity (the full bispectrum-estimator-vs-template oracle
is an M4-class [slow] item once an estimator exists; f_NL's structural
properties are exactly testable without one)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from inexor.config import PLANCK
from inexor.cosmology import linear_power
from inexor.diagnostics import pk_estimator
from inexor.ic import (
    IC_STREAM,
    gaussian_delta,
    linear_density,
    local_bispectrum_template,
    poisson_M,
    primordial_potential,
    white_noise,
    white_plane,
    white_slab,
)

N, L = 64, 500.0


@pytest.fixture
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def test_pk_recovery_within_sample_variance():
    # M0 self-check 5 promoted (mbody-binned estimator now)
    delta0 = gaussian_delta(jax.random.PRNGKey(2), N, L, PLANCK)
    kc, pk, nm = pk_estimator(np.asarray(delta0), L, kmax=0.6 * np.pi * N / L)
    p_ref = linear_power(kc, PLANCK)
    sel = nm > 50
    rel = np.abs(pk[sel] / p_ref[sel] - 1.0)
    exp = 3.0 * np.sqrt(2.0 / nm[sel])  # ~3 sigma of the chi^2 sample variance
    assert (rel < np.maximum(exp, 0.1)).all()


def test_fnl_zero_round_trips_to_gaussian():
    key = jax.random.PRNGKey(3)
    dg = gaussian_delta(key, N, L, PLANCK)
    d0 = linear_density(key, N, L, PLANCK, f_NL=0.0)
    scale = float(jnp.max(jnp.abs(dg)))
    assert float(jnp.max(jnp.abs(d0 - dg))) < 1e-4 * scale  # FFT round-off class


def test_fnl_enters_linearly_and_differentiably():
    key = jax.random.PRNGKey(4)

    def field(f_nl):
        return linear_density(key, N, L, PLANCK, f_NL=f_nl)

    d0, d10, d20 = field(0.0), field(10.0), field(20.0)
    # linear in f_NL: equal increments
    inc1 = np.asarray(d10 - d0, np.float64)
    inc2 = np.asarray(d20 - d10, np.float64)
    assert np.max(np.abs(inc2 - inc1)) < 1e-3 * np.max(np.abs(inc1))
    # jax.grad through f_NL matches the (exact) linear slope
    g = jax.grad(lambda f: jnp.sum(field(f) ** 2))(10.0)
    fd = (float(jnp.sum(field(10.0 + 1.0) ** 2)) - float(jnp.sum(field(10.0 - 1.0) ** 2))) / 2.0
    # f32 loss + f32 FD numerator: ~1e-2 agreement is the precision floor here;
    # exact linearity is separately proven by the equal-increments assert above
    assert float(g) == pytest.approx(fd, rel=2e-2)


def test_primordial_potential_cobe_scale():
    phi = primordial_potential(jax.random.PRNGKey(5), N, L, PLANCK)
    rms = float(jnp.sqrt(jnp.mean(phi**2)))
    # physical primordial potential ~ 1e-5 (COBE); generous half-decade window
    assert 3e-6 < rms < 1e-4


def test_poisson_M_shape_and_limits():
    k = np.array([0.0, 1e-3, 0.1, 1.0])
    M = poisson_M(k, PLANCK)
    assert M.shape == k.shape
    assert M[0] == 0.0  # k^2 kills k=0
    assert np.all(np.diff(M[1:]) > 0)  # rising with k over this range


def test_poisson_M_table_path_matches_analytic():
    """M-v2-5: the ICKTable transfer path agrees with the analytic one at
    interp-error class, preserves shape, and keeps the k=0 -> M=0 contract."""
    from inexor.cosmology import ic_k_table

    tab = ic_k_table(PLANCK, 256, 128.0)
    k = np.array([0.0, 0.05, 0.1, 0.5, 1.0, 5.0])
    M_tab = poisson_M(k, PLANCK, table=tab)
    M_ana = poisson_M(k, PLANCK)
    assert M_tab.shape == k.shape
    assert M_tab[0] == 0.0
    assert np.allclose(M_tab[1:], M_ana[1:], rtol=1e-4)
    # 2D shape preservation through the table path
    assert poisson_M(np.full((2, 3), 0.1), PLANCK, table=tab).shape == (2, 3)


# ---------------------------------------------------------------------------
# Plane-keyed white noise (M-v2-5, D-v2-15 clause 5)
# ---------------------------------------------------------------------------


def test_white_noise_is_invariant_to_slab_thickness(x64):
    """The decomposition-invariance theorem, at f64 and f32: any tiling of the
    same (seed, N, dtype) assembles the identical array, bit for bit. The
    thickness set includes 7, which does not divide 32 -- the ragged final
    slab is the off-by-one class."""
    key = jax.random.PRNGKey(3)
    n = 32
    for fdt in (np.float64, np.float32):
        ref = white_noise(key, n, fdt)
        for t in (1, 7, 16, 32):
            parts = [white_slab(key, lo, min(lo + t, n), n, fdt) for lo in range(0, n, t)]
            assembled = np.concatenate(parts, axis=0)
            assert assembled.dtype == np.dtype(fdt)
            assert np.array_equal(assembled, ref), f"thickness {t} broke invariance at {fdt}"


def test_white_slab_is_random_access(x64):
    """A mid-field slab request never generates preceding planes and is bitwise
    the same rows of a full build -- the property that makes the construction
    streamable and spot-checkable at scale."""
    key = jax.random.PRNGKey(3)
    n = 32
    ref = white_noise(key, n, np.float64)
    assert np.array_equal(white_slab(key, 13, 20, n, np.float64), ref[13:20])


def test_white_noise_invariance_can_fail(x64):
    """Anti-vacuity: a different base seed and a shifted plane index must both
    break the equality, or the invariance test asserts nothing."""
    key = jax.random.PRNGKey(3)
    n = 32
    ref = white_noise(key, n, np.float64)
    other_seed = white_noise(jax.random.PRNGKey(4), n, np.float64)
    assert not np.array_equal(other_seed, ref)
    assert not np.array_equal(white_plane(key, 1, n, np.float64), white_plane(key, 2, n, np.float64))
    shifted = np.concatenate(
        [white_slab(key, 1, n, n, np.float64), white_plane(key, 0, n, np.float64)[None]], axis=0
    )
    assert not np.array_equal(shifted, ref)


def test_white_noise_moments_sane():
    w = white_noise(jax.random.PRNGKey(0), 64, np.float32)
    n_samp = w.size
    assert abs(float(w.mean())) < 5.0 / np.sqrt(n_samp)
    assert abs(float(w.var()) - 1.0) < 5.0 * np.sqrt(2.0 / n_samp)


def test_white_noise_refuses_silent_f64_degradation():
    """An f64 request without x64 must refuse, not silently return f32."""
    if jax.config.jax_enable_x64:
        pytest.skip("x64 already on in this process")
    with pytest.raises(RuntimeError, match="without jax_enable_x64"):
        white_plane(jax.random.PRNGKey(0), 0, 8, np.float64)


def test_ic_stream_constant_exists():
    assert IC_STREAM.startswith("m5-foldin")


def test_bispectrum_template_squeezed_divergence():
    b_squeezed = local_bispectrum_template([(1e-3, 0.1, 0.1)], PLANCK, f_NL=1.0)[0]
    b_equil = local_bispectrum_template([(0.1, 0.1, 0.1)], PLANCK, f_NL=1.0)[0]
    assert b_squeezed > 100.0 * abs(b_equil)  # 1/M(k1) ~ 1/k1^2 divergence
    # linear in f_NL by construction
    b2 = local_bispectrum_template([(0.1, 0.1, 0.1)], PLANCK, f_NL=2.0)[0]
    assert b2 == pytest.approx(2.0 * b_equil, rel=1e-12)


def test_gaussian_delta_table_backend_dc_safe():
    # regression (S6 deficit matrix): the table backend refuses k outside its
    # range, and the |k| grid contains the DC mode -- gaussian_delta must
    # evaluate the colour DC-safely (the DC colour is zeroed regardless)
    k_t = np.geomspace(1e-4, 1e2, 800)  # the density of the real pk_*.txt dumps
    table = (k_t, linear_power(k_t, PLANCK))
    d_tab = gaussian_delta(jax.random.PRNGKey(0), N, L, PLANCK, table=table, backend="table")
    d_eh = gaussian_delta(jax.random.PRNGKey(0), N, L, PLANCK)
    assert float(jnp.mean(d_tab)) == pytest.approx(0.0, abs=1e-6)
    # same white noise + a table OF the eh98 spectrum -> near-identical field
    # (only log-log interpolation error differs)
    rms = float(jnp.sqrt(jnp.mean((d_tab - d_eh) ** 2) / jnp.mean(d_eh**2)))
    assert rms < 1e-3
