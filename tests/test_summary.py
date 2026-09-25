"""summary.py: the per-run P(k) accuracy statement (M-v2-6 Stage 4).

The load-bearing gate is a NULL: on a field whose spectrum is known exactly,
the z profile must be standard normal. That is what validates the whole
normalization chain at once -- hermitian mode weights, the bin-averaged oracle,
and the Gaussian sigma -- and a wrong weight shows up as a uniform sqrt(2)
inflation that no comparison against another estimator of ours would catch.
"""

import numpy as np
import pytest

from inexor import ooc_fft, state, summary
from inexor.codec import T9Layout
from inexor.config import Cosmology
from inexor.diagnostics import cic_window, tsc_window

N, BOX = 64, 64.0


def _white(seed=0, n=N, box=BOX):
    """Unit-variance white noise and its exact spectrum.

    P = sigma^2 * (L/N)^3 for a per-cell variance sigma^2: with
    P = |delta_k|^2 L^3 / N^6 and <|delta_k|^2> = N^3 sigma^2. FLAT, so the
    bin-averaged and bin-centre oracles coincide and the null tests the
    normalization alone rather than the averaging."""
    f = np.random.default_rng(seed).normal(size=(n, n, n))
    return f, (box / n) ** 3


NULL_EDGES = np.linspace(0.0, 0.5 * np.pi * N / BOX, 129)
NULL_SEEDS = 4


def _pooled_null_z(oracle_scale=1.0, seeds=NULL_SEEDS):
    """z over `seeds` independent white-noise realizations, pooled.

    One realization gives ~59 usable bins, and the sampling error on std(z) is
    1/sqrt(2m) -- too loose at that size to see a 25% error in the mode
    weighting (measured: the bar is +-0.37 and the defect moves std to 0.71).
    Pooling four realizations is what gives the null its power, and it is
    cheaper than a bigger grid.
    """
    zs = []
    for seed in range(seeds):
        f, p_true = _white(seed=seed)
        spec = ooc_fft.rfftn_ooc(f)
        zs.append(
            summary.binned_power(
                spec, N, BOX,
                p_of_k=lambda k, p=p_true * oracle_scale: np.full_like(k, p),
                edges=NULL_EDGES,
            )["z"]
        )
    return np.concatenate(zs)


def test_white_noise_null_is_standard_normal():
    """THE gate on the whole normalization chain: power normalization, mode
    weights and the Gaussian sigma, on a field whose spectrum is exact.

    Bars are the sampling errors of the statistics themselves, not picked: over
    `m` pooled bins, mean(z) has SE = 1/sqrt(m) and std(z) has SE = 1/sqrt(2m),
    both checked at 4 SE.

    What it catches, measured by planting each: dropping the hermitian weight
    entirely (std 0.71 against a 1 +- 0.18 bar) and a power normalization off by
    a factor of the grid size. What it does NOT catch is the kz = 0 / Nyquist
    planes being weighted like the rest -- those are ~6% of the half grid, so it
    moves std to 1.04 and sits under the noise. That case is covered exactly by
    `test_hermitian_weights_are_the_full_grid_multiplicities` instead, which is
    the right instrument for a discrete fact.
    """
    z = _pooled_null_z()
    m = len(z)
    assert m > 200, f"only {m} pooled bins; the null has no power"
    assert abs(z.mean()) < 4.0 / np.sqrt(m), f"mean z = {z.mean():.3f} over {m} bins"
    assert abs(z.std() - 1.0) < 4.0 / np.sqrt(2 * m), f"std z = {z.std():.3f} over {m} bins"


def test_the_null_can_fail():
    """A gate that cannot fail reads as a pass, and this one is loose enough
    that the question is real. An oracle wrong by 10% must trip it; one wrong by
    1% must not, or the bar would be catching noise rather than the defect."""
    assert abs(_pooled_null_z(oracle_scale=1.10).mean()) > 4.0 / np.sqrt(200)
    z1 = _pooled_null_z(oracle_scale=1.01)
    assert abs(z1.mean()) < 4.0 / np.sqrt(len(z1))


def test_hermitian_weights_are_the_full_grid_multiplicities():
    """Exact, because the null cannot see it. A half-grid element stands for
    two full-grid modes except on the self-conjugate kz planes -- kz = 0 always,
    and kz = Nyquist on an even grid, which hold each mode beside its own
    conjugate. Weighting those like the rest is a ~6% error in the mode count,
    which is 3% on every z and invisible to a spread test."""
    for n in (8, 9, 16, 17):
        _, kz, w = summary._mode_grid(n, 32.0)
        assert len(w) == n // 2 + 1 == len(kz)
        assert w[0] == 1.0, "kz = 0 is self-conjugate"
        assert w[-1] == (1.0 if n % 2 == 0 else 2.0), "Nyquist exists only on an even grid"
        assert np.all(w[1 : n // 2] == 2.0)
        # the weights must sum to the full grid's plane count, which is the
        # whole point of them
        assert w.sum() == n


def test_the_measurement_itself_is_unbiased():
    """Independently of the z scaling: the binned power of white noise is its
    known flat spectrum."""
    f, p_true = _white()
    spec = ooc_fft.rfftn_ooc(f)
    res = summary.binned_power(spec, N, BOX, p_of_k=lambda k: np.full_like(k, p_true))
    assert abs(np.mean(res["p"]) / p_true - 1.0) < 0.01


def test_bin_centre_oracle_carries_a_deterministic_bias():
    """The M-v2-5 leg VI finding, reproduced as arithmetic rather than as a
    field. P's curvature across a bin means its average over the realized modes
    is not its value at the bin's mean k, and the gap is DETERMINISTIC: it does
    not average down with more modes, it grows in sigma units as sqrt(N_modes).
    That is how a correct code reads as an 8.7 sigma failure.

    FOUR bins, not the production 64: the gap is P's curvature across a bin
    scaled by sqrt(N_modes), so a small box needs wide bins to show what a
    production box shows with narrow ones."""
    edges = np.linspace(0.0, 0.5 * np.pi * N / BOX, 5)
    zeros = np.zeros((N, N, N // 2 + 1), dtype=np.complex128)

    def steep(k):
        return k**-3.0

    res = summary.binned_power(zeros, N, BOX, p_of_k=steep, edges=edges)
    z_jensen = (steep(res["k_mean"]) / res["p_oracle"] - 1.0) * np.sqrt(res["n_modes"] / 2.0)
    assert np.abs(z_jensen).max() > 3.0, (
        f"bin-centre oracle biased by only {np.abs(z_jensen).max():.2f} sigma here"
    )


def test_the_bin_centre_bias_grows_as_sqrt_of_the_mode_count():
    """The half that makes it dangerous: it is not noise, so a bigger run makes
    it WORSE. Isolating that needs the bins held fixed in k while the mode count
    moves, and the mode count in a fixed k shell goes as the BOX volume, not as
    the grid -- at fixed box, refining the mesh adds modes above the band and
    none inside it. Comparing bins at different k instead would confound the
    count with P's curvature, which changes across the band too.

    Doubling the box is 8x the modes, so the predicted ratio is sqrt(8) = 2.83.
    Read on the second bin from the top; the last bin is edge-affected."""
    edges = np.linspace(0.0, 0.7, 5)
    got = {}
    for n, box in ((64, 64.0), (128, 128.0)):
        zeros = np.zeros((n, n, n // 2 + 1), dtype=np.complex128)
        r = summary.binned_power(zeros, n, box, p_of_k=lambda k: k**-3.0,
                                 edges=edges, min_weight=10.0)
        got[box] = ((r["k_mean"] ** -3.0 / r["p_oracle"] - 1.0) * np.sqrt(r["n_modes"] / 2.0),
                    r["n_modes"])

    z_a, n_a = got[64.0]
    z_b, n_b = got[128.0]
    predicted = np.sqrt(n_b[-2] / n_a[-2])
    assert abs(z_b[-2] / z_a[-2] / predicted - 1.0) < 0.05, (
        f"z_jensen scaled {z_b[-2] / z_a[-2]:.3f}, sqrt(mode ratio) predicts {predicted:.3f}"
    )
    assert abs(predicted / np.sqrt(8.0) - 1.0) < 0.05


@pytest.mark.parametrize("slab", [1, 7, 16, N])
def test_slab_is_a_memory_knob_only(slab):
    f, p_true = _white()
    spec = ooc_fft.rfftn_ooc(f)
    ref = summary.binned_power(spec, N, BOX, p_of_k=lambda k: np.full_like(k, p_true), slab=N)
    got = summary.binned_power(spec, N, BOX, p_of_k=lambda k: np.full_like(k, p_true), slab=slab)
    for key in ("k_mean", "p", "n_modes", "z", "p_oracle"):
        np.testing.assert_allclose(got[key], ref[key], rtol=1e-13, atol=0)


def test_tsc_window_is_the_cic_window_at_exponent_three_halves():
    """`paint_tsc_int` is a quadratic spline, so its window is `sinc^3` per axis
    where CIC's is `sinc^2`. Both are exactly 1 at k = 0 and the ratio is the
    3/2 power everywhere, which pins the exponent rather than the shape."""
    w_t, w_c = tsc_window(16, 32.0), cic_window(16, 32.0)
    assert w_t[0, 0, 0] == 1.0 and w_c[0, 0, 0] == 1.0
    ok = w_c > 1e-12
    np.testing.assert_allclose(w_t[ok], w_c[ok] ** 1.5, rtol=1e-12)

    # and the two differ enough at half Nyquist that using the wrong one shows
    knyq = np.pi * 16 / 32.0
    s = np.sinc(0.5 * knyq / (2 * knyq))
    assert abs(s**3 / s**2 - 1.0) > 0.02


def test_corrections_are_applied_in_the_painted_order_and_reported():
    """A painted discrete field has <|delta|^2> = W^2 (P + 1/nbar), so the
    estimator is raw/W^2 - shot and NOT (raw - shot)/W^2. Checked against a
    hand-computed bin on a spectrum of ones, where every step is arithmetic."""
    n, box = 16, 32.0
    spec = np.ones((n, n, n // 2 + 1), dtype=np.complex128)
    shot = 1e-4

    def win(kx_slab, kx, kz):
        return np.full((len(kx_slab), len(kx), len(kz)), 0.5)

    plain = summary.binned_power(spec, n, box, min_weight=0.0)
    both = summary.binned_power(spec, n, box, window=win, shot_noise=shot, min_weight=0.0)
    np.testing.assert_allclose(both["p"], plain["p"] / 0.25 - shot, rtol=1e-12)
    # the corrections are on the card rather than hidden inside `p`
    np.testing.assert_allclose(both["window_correction"], 0.25, rtol=1e-12)
    np.testing.assert_allclose(both["shot_fraction"], shot / plain["p"], rtol=1e-12)


def test_nonlinear_scale_finds_the_analytic_crossing():
    """Delta^2 = k^3 P / (2 pi^2) = 1. For P = A k^-3 that is A = 2 pi^2 / 1
    independent of k, so use a slope that actually crosses: P = A k^-2 gives
    k_nl = 2 pi^2 / A."""
    A = 50.0
    k_nl = summary.nonlinear_scale(lambda k: A * k**-2.0)
    assert abs(k_nl / (2.0 * np.pi**2 / A) - 1.0) < 1e-3
    # a spectrum that never crosses returns None rather than an extrapolation
    assert summary.nonlinear_scale(lambda k: 1e-12 * k**-2.0) is None


def test_the_scanned_range_is_on_the_card_beside_the_none():
    """A None means "no crossing in here" and is unreadable without "here".
    The card's range must be the one the function actually scanned, so bracket
    it: a crossing just inside NL_SCAN_K's ceiling is found, one just outside
    is not."""
    lo, hi = summary.NL_SCAN_K
    inside = 2.0 * np.pi**2 / (0.9 * hi)  # A giving k_nl = 0.9 * hi
    outside = 2.0 * np.pi**2 / (1.1 * hi)
    assert summary.nonlinear_scale(lambda k: inside * k**-2.0) == pytest.approx(
        0.9 * hi, rel=1e-3)
    assert summary.nonlinear_scale(lambda k: outside * k**-2.0) is None
    assert lo < hi


# --------------------------------------------------------- the card, end to end

L_BOX, N_PART, N_FINE, N_COARSE, N_TILE, B_FINE = 32.0, 32, 64, 16, 16, 8


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _smoke_state():
    from inexor import engine

    cfg = engine.EngineConfig(
        box_size=L_BOX, n_part=N_PART, n_fine=N_FINE, n_coarse=N_COARSE,
        n_tile=N_TILE, b_fine=B_FINE,
    )
    rng = np.random.default_rng(0)
    g = (np.arange(N_PART) + 0.5) * (L_BOX / N_PART)
    q = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    x = np.mod(q + rng.normal(scale=0.25 * L_BOX / N_PART, size=q.shape), L_BOX)
    v = rng.normal(scale=0.5, size=x.shape)
    t9 = T9Layout(box_size=L_BOX, n_part=N_PART, bucket_cells=2)
    st = state.SlotState.build(x, v, t9, N_FINE // cfg.n_brick, arena_frac=0.05)
    return st, cfg


def _smoke_card(st, cfg, **kw):
    """The smoke mesh is 16^3, so its bins have tens of modes rather than
    millions; the production default would drop every one of them."""
    edges = np.linspace(0.0, 0.5 * np.pi * cfg.n_coarse / cfg.box_size, 5)
    return summary.pk_summary_card(
        st, cfg, Cosmology(), a_out=1.0, edges=edges, min_weight=20.0, **kw
    )


def test_the_card_refuses_to_be_empty():
    """The production default at a smoke mesh keeps no bin. That must refuse:
    a zero-bin card satisfies every length and finiteness check a caller makes
    and carries no measurement. Found by a sibling test failing, not by this
    one -- the vacuous version of the test below passed."""
    st, cfg = _smoke_state()
    with pytest.raises(ValueError, match="not a card"):
        summary.pk_summary_card(st, cfg, Cosmology(), a_out=1.0)


def test_card_composes_over_a_real_state():
    st, cfg = _smoke_state()
    card = _smoke_card(st, cfg)

    assert card["n_bins"] > 0, "vacuous: every assertion below holds on an empty card"
    assert card["card"] == summary.CARD
    assert card["n_bins"] == len(card["z_profile"]) == len(card["k_mean"]) == len(card["p"])
    assert card["deconvolved"] == "tsc"
    assert card["k_nonlinear_scan"] == [float(summary.NL_SCAN_K[0]),
                                        float(summary.NL_SCAN_K[1])]
    assert card["shot_noise"] == pytest.approx(L_BOX**3 / st.n_particles)
    assert card["growth_factor"] == pytest.approx(1.0)
    assert np.isfinite(card["z_profile"]).all() and np.isfinite(card["p"]).all()
    # the profile is the product; nothing on the card is a verdict
    assert not any(k in card for k in ("ok", "pass", "max_abs_z"))
    # the corrections travel with it
    assert np.all(np.asarray(card["window_correction"]) <= 1.0)
    assert np.all(np.asarray(card["window_correction"]) > 0.0)


def test_card_accepts_a_delta_the_caller_already_has():
    """The engine can hand over the last step's coarse field; paying for the
    paint twice at C-gh is 4.3 GB and a full streamed pass."""
    from inexor import engine

    st, cfg = _smoke_state()
    delta = engine.coarse_delta_streamed(st, cfg)
    a = _smoke_card(st, cfg, delta=delta)
    b = _smoke_card(st, cfg)
    assert a["n_bins"] > 0
    np.testing.assert_allclose(a["z_profile"], b["z_profile"], rtol=1e-12)

    with pytest.raises(ValueError, match="want"):
        _smoke_card(st, cfg, delta=np.zeros((4, 4, 4)))


def test_band_verdict_names_its_band_and_flags_the_nonlinear_reach():
    st, cfg = _smoke_state()
    card = _smoke_card(st, cfg)
    k = np.asarray(card["k_mean"])
    assert len(k) > 1

    v = summary.band_verdict(card, k_max=k[len(k) // 2])
    assert v["k_max"] == k[len(k) // 2] and v["n_bins"] > 0
    assert v["max_abs_z"] == pytest.approx(
        np.abs(np.asarray(card["z_profile"])[k <= v["k_max"]]).max()
    )
    if card["k_nonlinear"] is not None:
        far = summary.band_verdict(card, k_max=k.max())
        assert far["band_reaches_nonlinear"] == bool(k.max() > card["k_nonlinear"])
    with pytest.raises(ValueError, match="no bins"):
        summary.band_verdict(card, k_max=1e-6)


def test_the_slab_built_window_is_the_full_grid_one():
    """The card builds the TSC window per slab to avoid a second full-grid f64
    array beside the spectrum (17 GB at C-gh). Two expressions of one function
    is exactly how they drift, so pin them."""
    n, box = 12, 24.0
    k_nyq = np.pi * n / box
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=box / n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=box / n)
    ref = tsc_window(n, box)
    for lo in range(0, n, 5):
        hi = min(lo + 5, n)
        got = summary.tsc_window_slab(kx[lo:hi], kx, kz, k_nyq)
        np.testing.assert_allclose(got, ref[lo:hi], rtol=0, atol=0)
