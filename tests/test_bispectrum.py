"""Scoccimarro bispectrum estimator (diagnostics.bispectrum), ported from mbody
fields.py:174 / ic.py:151 to numpy-coefficients + JAX-fields at f64.

TOLERANCES ARE NOT INHERITED FROM MBODY. mbody's plane-wave bound is 1e-4
carrying a measured ~5e-8 float32 floor; at f64 the same check reaches ~1e-13,
so keeping 1e-4 would ship a test that has lost its power. Every deterministic
bound here is set at 100x a floor measured on this code, and the measured value
is quoted next to it. The seeded bound cannot transfer at all -- mbody's fields
come from MLX Threefry and JAX's stream is different -- so it is re-measured
over 48 seed pairs, a count chosen so the test can REJECT the bin-centre oracle
rather than merely accept the binned one (see the discrimination control).

x64 is process-global in jax 0.10.2 (no jax.experimental.enable_x64 context
manager), so it is flipped per test and restored, rather than at import: this
module must not change the dtype defaults other test modules run under.
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
    """Enable x64 for this test only, then restore. See the module docstring."""
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
    """Count full-grid mode-triplets with |k_i| in shell i and k1+k2+k3 = 0 (mod N).

    Independent of the FFT J-product, so it is a real cross-check on the
    estimator's n_tri rather than a restatement of it.

    GENERALIZED from mbody tests/test_bispectrum.py:43, which takes one scalar
    dk_mult and then HARDCODES 0.5 for the third shell (its line 57). That is
    harmless there because its only call site uses dk_mult = 1.0, where
    0.5*dk_mult == 0.5. It is not harmless here: the gate needs a per-shell dk,
    and the un-generalized version would silently count the third shell at a
    width the estimator never used, certifying the port against binning it does
    not implement. dk_mult is a scalar or a 3-sequence, honoured on all three.
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
    """The generalization itself, before anything is certified against it.

    mbody's version ignores dk_mult on the third shell, so widening only shell 3
    would leave its count unchanged. If this test passes vacuously the per-shell
    cross-check below is worthless, so assert the count actually MOVES.
    """
    narrow = _brute_n_tri(N_SMALL, [3, 4, 5], dk_mult=[1.0, 1.0, 1.0])
    wide3 = _brute_n_tri(N_SMALL, [3, 4, 5], dk_mult=[1.0, 1.0, 3.0])
    assert wide3 > narrow, "widening shell 3 must admit more triplets"
    # And the un-generalized behaviour (0.5 fixed on shell 3) is what `narrow` is.
    assert _brute_n_tri(N_SMALL, [3, 4, 5], dk_mult=1.0) == narrow


def test_normalization_deterministic():
    """Closed 3-4-5 triangle of distinct-magnitude modes.

    Each shell holds exactly one populated mode pair, so I_i = a cos(q_i x) and
    sum_x I1 I2 I3 = a^3 N^3 / 4, giving B = L^6 a^3 / (4 n_tri) exactly. Tests
    the V^2/N^9 prefactor, the data path and the count together.
    """
    n, a = N_SMALL, 0.5
    kf = 2.0 * np.pi / L_BOX
    field = _plane_wave_field(n, (3, 0, 0), (0, 4, 0), (-3, -4, 0), a)
    b, n_tri = bispectrum(field, L_BOX, [(3 * kf, 4 * kf, 5 * kf)])

    n_brute = _brute_n_tri(n, [3, 4, 5])
    assert abs(n_tri[0] - n_brute) < 0.5

    predicted = L_BOX**6 * a**3 / (4.0 * n_brute)
    rel = abs(b[0] / predicted - 1.0)
    # Measured f64 floor: 6.7e-16 at N=16 (this test), 1.7e-15 at N=32. Bound is
    # ~150x the N=16 value. mbody's f32 floor was ~5e-8 under a 1e-4 bound, so
    # inheriting 1e-4 here would have left 11 orders of slack.
    assert rel < 1e-13, f"plane-wave normalization off by {rel:.3e}"


def test_n_tri_matches_brute_force_at_per_shell_dk():
    """n_tri against the independent counter, at a NON-uniform per-leg dk.

    The uniform-dk case is covered above; this is the one that would catch a
    per-shell width that reached the counter but not the estimator (or the
    reverse), which is the specific failure the generalization exists to avoid.
    """
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
    """A bin triple that cannot form a triangle returns n_tri = 0 and B = NaN.

    mbody documents n_tri = 0 here but still evaluates alpha * S / norm, which
    in f32 round-off returns a large finite number that reads as a measurement.
    At f64 the shell product can reach exact zero and raise instead. Neither is
    acceptable, so the port guards and returns NaN.
    """
    n = N_SMALL
    kf = 2.0 * np.pi / L_BOX
    rng = np.random.default_rng(0)
    field = rng.normal(size=(n, n, n))
    b, n_tri = bispectrum(field, L_BOX, [(1 * kf, 2 * kf, 8 * kf), (3 * kf, 4 * kf, 5 * kf)])
    assert n_tri[0] == 0.0
    assert np.isnan(b[0])
    # the closing companion in the same call must be unaffected
    assert n_tri[1] > 0
    assert np.isfinite(b[1])


def test_triple_product_reduction_matches_fsum():
    """jnp.sum against math.fsum on the actual band fields.

    mbody reduces on a CPU stream in f64 because the triple product of zero-mean
    band-limited fields cancels heavily and f32 accumulation is unsafe. At f64
    with pairwise summation that escape hatch should be unnecessary -- this is
    the check that says so rather than assuming it.
    """
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
    # Measured worst 4.7e-16 over N in (16, 32) x 4 seeds; several draws are
    # bit-exact. So f64 pairwise summation needs no CPU-stream escape hatch, and
    # mbody's accurate_sum has no analogue to port. Bound is ~200x the worst.
    assert rel < 1e-13, f"reduction differs from fsum by {rel:.3e}"


def test_band_power_agrees_with_pk_estimator_away_from_nyquist():
    """band_power must be pk_estimator on the same shell, or R_Q mixes binnings.

    This is the assertion that keeps the two estimators in this module on ONE
    convention. It is stated away from Nyquist deliberately: the top bin is
    where a closed-vs-half-open edge would disagree, and _shell_index exists so
    that it does not.
    """
    n, ell = 32, 128.0
    rng = np.random.default_rng(2)
    field = rng.normal(size=(n, n, n))
    k_c, p_ref, n_ref = pk_estimator(field, ell)
    # drop the first and last resolved bins; compare the interior
    sel = slice(1, len(k_c) - 1)
    p_new, n_new = band_power(field, ell, k_c[sel])
    assert np.array_equal(n_new, n_ref[sel].astype(np.float64)), "mode counts differ"
    assert np.allclose(p_new, p_ref[sel], rtol=1e-12), "band powers differ"


def test_core_is_jittable_and_differentiable():
    """bispectrum_core is jit/grad/vmap-safe, and jit agrees with eager BITWISE.

    Asserted rather than assumed: the estimator is shared with ichnaea, which
    needs gradients, and "jit-able" degrades silently -- a stray host-side
    branch or a numpy call inside the traced region raises only when someone
    first tries to wrap it, which by then is someone else's bug.

    The cubic scaling is a free correctness check riding along: B is trilinear
    in delta, so doubling the field must multiply B by exactly 8.
    """
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
    """Matched-phase squeezed f_NL signal against the BIN-AVERAGED oracle.

    The antisymmetric combination [B(+f) - B(-f)]/2 at shared phase cancels the
    Gaussian cosmic-variance term, which otherwise swamps the signal at any
    affordable seed count. The oracle is local_bispectrum_binned, not
    ic.local_bispectrum_template: the bin-centre template biases the steep
    squeezed long side low, so using it would fold a known binning systematic
    into the calibration and read as an estimator error.
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

    # DISCRIMINATION CONTROL, and the reason n_seed is 48 rather than 16. At 16
    # seeds the bin-CENTRE template also passes (c_cal 0.875 against a 0.85-1.15
    # band), so the test could not tell the two oracles apart and its use of the
    # binned one rested on argument rather than measurement. Measured here:
    # centre/binned = [1.155, 1.081, 1.002] across the three bins, i.e. the
    # bin-centre template is 15.5% high on the most squeezed long side and the
    # systematic vanishes as k_long grows -- the k^2 shell-density effect, right
    # sign and right shape. At 48 seeds (7.8 s) SEM/signal is 4.4-5.5% and the
    # two separate: binned 0.979, centre 0.862. The band below EXCLUDES the
    # centre value, so using the wrong oracle fails rather than passes.
    c_centre = float(np.sum(sig * local_bispectrum_template(tris, cosmo, f_nl)) /
                     np.sum(local_bispectrum_template(tris, cosmo, f_nl) ** 2))
    assert not (0.90 < c_centre < 1.10), (
        f"bin-centre oracle calibrates to {c_centre:.4f}, inside the band -- this test "
        "has lost its power to reject the wrong oracle; re-measure the seed count"
    )

    # measured 0.979 at 48 seeds (0.992 / 1.003 at 16 / 32); band is ~4x the
    # spread across those seed counts, and excludes c_centre by construction
    assert 0.90 < c_cal < 1.10, f"calibration {c_cal:.4f}"
    # measured max per-bin deviation 12.9% at 48 seeds against a 4.4-5.5% SEM;
    # the residual is consistent with the O(f_NL^3) term the antisymmetric
    # combination does not cancel at f_NL = 2000. Bound is ~2.3x the measured.
    assert np.all(np.abs(sig / tmpl - 1.0) < 0.30), f"per-bin {sig / tmpl}"
    # the squeezed 1/k_long^2 divergence, ordering only
    assert sig[0] > sig[1] > sig[2] > 0
