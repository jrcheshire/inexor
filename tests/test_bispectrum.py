"""Scoccimarro bispectrum estimator (diagnostics.bispectrum) at f64.

Deterministic bounds sit at ~100x the f64 floor measured on this code, with the measured value
quoted beside each. The seeded f_NL test uses 48 seed pairs, enough that it rejects the
bin-centre oracle rather than merely accepting the binned one. x64 is process-global, so it is
enabled per test and restored.
"""

import math
from functools import partial

import numpy as np
import pytest

from inexor.config import Cosmology
from inexor.diagnostics import (
    _band_fields,
    _shell_spec,
    _theta_stack,
    band_power,
    bispectrum,
    bispectrum_core,
    local_bispectrum_binned,
    pk_estimator,
)

L_BOX = 200.0
N_SMALL = 16


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _plane_wave_field(n, m1, m2, m3, a):
    """a * sum_i cos(q_i . x) on the mesh, q = (2 pi / L) m, phase (2 pi / N) m."""
    ax = (2.0 * np.pi / n) * np.arange(n)

    def cos(m):
        return np.cos(m[0] * ax[:, None, None] + m[1] * ax[None, :, None] + m[2] * ax[None, None, :])

    return a * (cos(m1) + cos(m2) + cos(m3))


def _brute_n_tri(n, centers, dk_mult=1.0):
    """Count full-grid mode triplets with |k_i| in shell i and k1+k2+k3 = 0 (mod N).

    Independent of the FFT J-product, so a real cross-check on n_tri. dk_mult (shell width in
    units of k_f) is a scalar or a 3-sequence, honoured on all three shells.
    """
    w = np.asarray(dk_mult, dtype=np.float64)
    w = np.full(3, float(w)) if w.ndim == 0 else w
    idx = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    gx, gy, gz = np.meshgrid(idx, idx, idx, indexing="ij")
    mag = np.sqrt(gx**2 + gy**2 + gz**2)
    masks = [(mag >= centers[i] - 0.5 * w[i]) & (mag < centers[i] + 0.5 * w[i]) for i in range(2)]
    m1 = np.stack([gx[masks[0]], gy[masks[0]], gz[masks[0]]], axis=1)
    m2 = np.stack([gx[masks[1]], gy[masks[1]], gz[masks[1]]], axis=1)
    count = 0
    for a in m1:
        c3 = (-(a + m2)) % n
        signed = np.where(c3 <= n // 2, c3, c3 - n)
        mag3 = np.sqrt((signed**2).sum(axis=1))
        count += int(((mag3 >= centers[2] - 0.5 * w[2]) & (mag3 < centers[2] + 0.5 * w[2])).sum())
    return count


def test_brute_n_tri_honours_per_shell_width():
    """The brute counter honours a per-shell width: widening only shell 3 must change the
    count, or the per-shell n_tri cross-check below is vacuous."""
    narrow = _brute_n_tri(N_SMALL, [3, 4, 5], dk_mult=[1.0, 1.0, 1.0])
    wide3 = _brute_n_tri(N_SMALL, [3, 4, 5], dk_mult=[1.0, 1.0, 3.0])
    assert wide3 > narrow, "widening shell 3 must admit more triplets"
    assert _brute_n_tri(N_SMALL, [3, 4, 5], dk_mult=1.0) == narrow


def test_normalization_deterministic():
    """Closed 3-4-5 plane-wave triangle: each shell holds one mode pair, so
    B = L^6 a^3 / (4 n_tri) exactly. Pins the V^2/N^9 prefactor, data path and count."""
    n, a = N_SMALL, 0.5
    kf = 2.0 * np.pi / L_BOX
    field = _plane_wave_field(n, (3, 0, 0), (0, 4, 0), (-3, -4, 0), a)
    b, n_tri = bispectrum(field, L_BOX, [(3 * kf, 4 * kf, 5 * kf)])

    n_brute = _brute_n_tri(n, [3, 4, 5])
    assert abs(n_tri[0] - n_brute) < 0.5

    predicted = L_BOX**6 * a**3 / (4.0 * n_brute)
    rel = abs(b[0] / predicted - 1.0)
    # measured f64 floor 6.7e-16 at N=16 (1.7e-15 at N=32); bound ~150x the N=16 value
    assert rel < 1e-13, f"plane-wave normalization off by {rel:.3e}"


def test_n_tri_matches_brute_force_at_per_shell_dk():
    """n_tri against the independent counter at a non-uniform per-leg dk, which catches a
    per-shell width reaching the counter but not the estimator (or the reverse)."""
    n = N_SMALL
    kf = 2.0 * np.pi / L_BOX
    dk = (1.0 * kf, 1.0 * kf, 3.0 * kf)
    _, n_tri = bispectrum(field_ones(n), L_BOX, [(3 * kf, 4 * kf, 5 * kf)], dk=dk)
    n_brute = _brute_n_tri(n, [3, 4, 5], dk_mult=[1.0, 1.0, 3.0])
    assert abs(n_tri[0] - n_brute) < 0.5, f"n_tri {n_tri[0]} vs brute {n_brute}"
    # n_tri is a near-exact integer: the J-product is a mode count.
    assert abs(n_tri[0] - round(float(n_tri[0]))) < 1e-6 * n_tri[0]


def field_ones(n):
    """A field whose value is irrelevant -- n_tri depends only on the J fields."""
    return np.zeros((n, n, n), dtype=np.float64)


def test_non_closing_triple_is_guarded():
    """A bin triple that cannot close returns n_tri = 0 and B = NaN, not a round-off
    number that reads as a measurement; a closing triple in the same call is unaffected."""
    n = N_SMALL
    kf = 2.0 * np.pi / L_BOX
    rng = np.random.default_rng(0)
    field = rng.normal(size=(n, n, n))
    b, n_tri = bispectrum(field, L_BOX, [(1 * kf, 2 * kf, 8 * kf), (3 * kf, 4 * kf, 5 * kf)])
    assert n_tri[0] == 0.0
    assert np.isnan(b[0])
    assert n_tri[1] > 0
    assert np.isfinite(b[1])


def test_triple_product_reduction_matches_fsum():
    """jnp.sum of the triple product against math.fsum: the zero-mean band fields cancel
    heavily, and this checks f64 pairwise summation is accurate enough without compensation."""
    import jax.numpy as jnp

    n = N_SMALL
    kf = 2.0 * np.pi / L_BOX
    rng = np.random.default_rng(1)
    field = rng.normal(size=(n, n, n))
    centers, widths, tri_idx = _shell_spec([(3 * kf, 4 * kf, 5 * kf)], L_BOX)
    theta = _theta_stack(n, L_BOX, centers, widths)
    i_f = np.asarray(_band_fields(field, theta), dtype=np.float64)

    prod = (i_f[tri_idx[0, 0]] * i_f[tri_idx[0, 1]] * i_f[tri_idx[0, 2]]).ravel()
    fast = float(jnp.sum(jnp.asarray(prod)))
    exact = math.fsum(prod.tolist())
    rel = abs(fast - exact) / max(abs(exact), 1e-300)
    # measured worst 4.7e-16 over N in (16, 32) x 4 seeds; bound ~200x that
    assert rel < 1e-13, f"reduction differs from fsum by {rel:.3e}"


def test_band_power_agrees_with_pk_estimator_away_from_nyquist():
    """band_power equals pk_estimator on the same shells (else R_Q mixes binnings). Compared
    on interior bins, away from the edge bins."""
    n, ell = 32, 128.0
    rng = np.random.default_rng(2)
    field = rng.normal(size=(n, n, n))
    k_c, p_ref, n_ref = pk_estimator(field, ell)
    sel = slice(1, len(k_c) - 1)
    p_new, n_new = band_power(field, ell, k_c[sel])
    assert np.array_equal(n_new, n_ref[sel].astype(np.float64)), "mode counts differ"
    assert np.allclose(p_new, p_ref[sel], rtol=1e-12), "band powers differ"


def test_core_is_jittable_and_differentiable():
    """bispectrum_core is jit/grad/vmap-safe and jit matches eager bitwise; gradient consumers
    rely on this, and a host-side branch in the traced region would only raise on wrapping.
    B is cubic in delta, so doubling the field must scale B by 8."""
    import jax
    import jax.numpy as jnp

    n, a = N_SMALL, 0.5
    kf = 2.0 * np.pi / L_BOX
    centers, widths, tri_idx = _shell_spec([(3 * kf, 4 * kf, 5 * kf)], L_BOX)
    theta = jnp.asarray(_theta_stack(n, L_BOX, centers, widths))
    ti = jnp.asarray(tri_idx)
    alpha = L_BOX**6 / float(n) ** 9
    fld = jnp.asarray(_plane_wave_field(n, (3, 0, 0), (0, 4, 0), (-3, -4, 0), a))

    b_eager, nt_eager = bispectrum_core(fld, theta, ti, alpha, n)
    b_jit, nt_jit = jax.jit(partial(bispectrum_core, n_mesh=n))(fld, theta, ti, alpha)
    assert float(b_jit[0]) == float(b_eager[0]), "jit and eager must agree bitwise"
    assert float(nt_jit[0]) == float(nt_eager[0])

    grad = jax.grad(lambda d: bispectrum_core(d, theta, ti, alpha, n)[0][0])(fld)
    assert grad.shape == fld.shape
    assert grad.dtype == jnp.float64
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.sqrt(jnp.mean(grad**2))) > 0.0, "gradient must not be identically zero"

    batched = jax.vmap(lambda d: bispectrum_core(d, theta, ti, alpha, n)[0])(
        jnp.stack([fld, 2.0 * fld])
    )
    assert float(batched[1, 0] / batched[0, 0]) == pytest.approx(8.0, rel=1e-12)


@pytest.mark.slow
def test_squeezed_matches_binned_template():
    """Matched-phase squeezed f_NL signal against the bin-averaged oracle.

    [B(+f) - B(-f)]/2 at shared phase cancels the Gaussian cosmic-variance term. The oracle is
    local_bispectrum_binned: the bin-centre template misstates the steep squeezed long side, a
    binning systematic that would read as estimator error.
    """
    import jax
    import jax.numpy as jnp

    from inexor import ic
    from inexor.ic import local_bispectrum_template

    n, ell, f_nl, n_seed = 32, 256.0, 2000.0, 48
    cosmo = Cosmology()
    kf = 2.0 * np.pi / ell
    tris = [(m * kf, 8 * kf, 8 * kf) for m in (2, 3, 4)]

    def measure(seed, f):
        d = ic.linear_density(jax.random.PRNGKey(seed), n, ell, cosmo, f_NL=f, fdtype=jnp.float64)
        b, _ = bispectrum(np.asarray(d, np.float64), ell, tris)
        return b

    sig = np.mean([0.5 * (measure(s, f_nl) - measure(s, -f_nl)) for s in range(n_seed)], axis=0)
    tmpl = local_bispectrum_binned(n, ell, cosmo, tris, f_nl)

    assert np.all(tmpl > 0), "oracle must be positive for f_NL > 0 in the squeezed limit"
    c_cal = float(np.sum(sig * tmpl) / np.sum(tmpl**2))

    # Discrimination control, and why n_seed is 48: at 16 seeds the bin-centre oracle also
    # passes. Measured centre/binned = [1.155, 1.081, 1.002] over the bins; at 48 seeds
    # (SEM/signal 4.4-5.5%) binned calibrates to 0.979 and centre to 0.862, outside the band.
    c_centre = float(np.sum(sig * local_bispectrum_template(tris, cosmo, f_nl)) /
                     np.sum(local_bispectrum_template(tris, cosmo, f_nl) ** 2))
    assert not (0.90 < c_centre < 1.10), (
        f"bin-centre oracle calibrates to {c_centre:.4f}, inside the band -- this test "
        "has lost its power to reject the wrong oracle; re-measure the seed count"
    )

    # measured 0.979 at 48 seeds (0.992 / 1.003 at 16 / 32); band ~4x that spread
    assert 0.90 < c_cal < 1.10, f"calibration {c_cal:.4f}"
    # measured max per-bin deviation 12.9% (consistent with the uncancelled O(f_NL^3) term at
    # f_NL = 2000); bound ~2.3x that
    assert np.all(np.abs(sig / tmpl - 1.0) < 0.30), f"per-bin {sig / tmpl}"
    # the squeezed 1/k_long^2 divergence, ordering only
    assert sig[0] > sig[1] > sig[2] > 0
