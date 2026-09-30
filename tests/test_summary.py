"""summary.py: the per-run P(k) accuracy card.

The central gate is a null: on white noise, whose spectrum is exact, the z profile is
standard normal. That checks power normalization, hermitian mode weights and the Gaussian
sigma together; a wrong weight shows as a uniform sqrt(2) inflation.
"""


import numpy as np
import pytest

from inexor import ooc_fft, state, summary
from inexor.codec import T9Layout
from inexor.config import Cosmology
from inexor.diagnostics import cic_window, tsc_window

N, BOX = 64, 64.0


def _white(seed=0, n=N, box=BOX):
    """Unit-variance white noise and its exact, flat spectrum P = sigma^2 (L/N)^3."""
    f = np.random.default_rng(seed).normal(size=(n, n, n))
    return f, (box / n) ** 3


NULL_EDGES = np.linspace(0.0, 0.5 * np.pi * N / BOX, 129)
NULL_SEEDS = 4


def _pooled_null_z(oracle_scale=1.0, seeds=NULL_SEEDS):
    """z pooled over `seeds` white-noise realizations. One gives ~59 bins, a 4-SE std bar of
    +-0.37, too loose to see a dropped hermitian weight (std 0.71).
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
    """Pooled null z has mean 0 and std 1 within 4 SE (1/sqrt(m), 1/sqrt(2m)).

    Catches a dropped hermitian weight (std 0.71) and a grid-size power error. Mis-weighting only
    the kz = 0 / Nyquist planes (std 1.04) is pinned exactly by the multiplicity test below.
    """
    z = _pooled_null_z()
    m = len(z)
    assert m > 200, f"only {m} pooled bins; the null has no power"
    assert abs(z.mean()) < 4.0 / np.sqrt(m), f"mean z = {z.mean():.3f} over {m} bins"
    assert abs(z.std() - 1.0) < 4.0 / np.sqrt(2 * m), f"std z = {z.std():.3f} over {m} bins"


def test_the_null_can_fail():
    """The null trips on an oracle 10% off and passes one 1% off."""
    assert abs(_pooled_null_z(oracle_scale=1.10).mean()) > 4.0 / np.sqrt(200)
    z1 = _pooled_null_z(oracle_scale=1.01)
    assert abs(z1.mean()) < 4.0 / np.sqrt(len(z1))


def test_hermitian_weights_are_the_full_grid_multiplicities():
    """A half-grid kz plane counts twice except the self-conjugate ones (kz = 0, and
    Nyquist on an even grid); the weights sum to n. Exact, since the null cannot see a
    ~6% mode-count error."""
    for n in (8, 9, 16, 17):
        _, kz, w = summary._mode_grid(n, 32.0)
        assert len(w) == n // 2 + 1 == len(kz)
        assert w[0] == 1.0, "kz = 0 is self-conjugate"
        assert w[-1] == (1.0 if n % 2 == 0 else 2.0), "Nyquist exists only on an even grid"
        assert np.all(w[1 : n // 2] == 2.0)
        assert w.sum() == n


def test_the_measurement_itself_is_unbiased():
    """Binned white-noise power matches its flat spectrum to 1%, independent of z."""
    f, p_true = _white()
    spec = ooc_fft.rfftn_ooc(f)
    res = summary.binned_power(spec, N, BOX, p_of_k=lambda k: np.full_like(k, p_true))
    assert abs(np.mean(res["p"]) / p_true - 1.0) < 0.01


def test_bin_centre_oracle_carries_a_deterministic_bias():
    """A bin-centre oracle is biased by P's curvature across the bin by >3 sigma here (four
    wide bins so a small box shows it).
    """
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
    """The bin-centre bias in sigma grows as sqrt(N_modes): bins fixed in k, box doubled
    (predicted ratio sqrt(8)), read on the second-highest bin (the last is edge-affected).
    """
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


def _win(kx, ky, kz):
    return summary.tsc_window_slab(kx, ky, kz, np.pi * N / BOX)


@pytest.mark.parametrize("slab", [1, 7, 16, N])
def test_slab_is_a_memory_knob_only(slab):
    """Bitwise: each ky-plane's partial is summed alone, whatever the block."""
    f, p_true = _white()
    spec = ooc_fft.rfftn_ooc(f)
    kw = dict(p_of_k=lambda k: np.full_like(k, p_true), window=_win, shot_noise=1e-3)
    ref = summary.binned_power(spec, N, BOX, slab=N, **kw)
    got = summary.binned_power(spec, N, BOX, slab=slab, **kw)
    for key in ("k_mean", "p", "n_modes", "z", "p_oracle", "window_correction",
                "shot_fraction"):
        np.testing.assert_array_equal(got[key], ref[key])


@pytest.mark.parametrize("cuts", [(0, 5, 11, N), (0, 1, N - 1, N), (0, N // 2, N)])
def test_any_split_of_the_planes_is_bitwise_one_part(cuts):
    """Partials over disjoint ky-ranges (as ranks would hold them), combined in any order,
    are bitwise the single pass."""
    f, p_true = _white()
    spec = ooc_fft.rfftn_ooc(f)
    kw = dict(p_of_k=lambda k: np.full_like(k, p_true), window=_win, shot_noise=1e-3)
    ref = summary.binned_power(spec, N, BOX, slab=3, **kw)
    parts = [summary.binned_power_partials(np.ascontiguousarray(spec[:, a:b]), N, BOX, y0=a,
                                           slab=4, **kw)
             for a, b in zip(cuts[:-1], cuts[1:])]
    for order in (parts, parts[::-1]):
        got = summary.combine_partials(order)
        for key in ("k_mean", "p", "n_modes", "z", "p_oracle"):
            np.testing.assert_array_equal(got[key], ref[key])


def test_partials_must_cover_every_plane_once():
    f, _ = _white()
    spec = ooc_fft.rfftn_ooc(f)
    a = summary.binned_power_partials(spec[:, :5], N, BOX, y0=0)
    b = summary.binned_power_partials(spec[:, 4:], N, BOX, y0=4)
    with pytest.raises(ValueError, match="exactly once"):
        summary.combine_partials([a, b])
    with pytest.raises(ValueError, match="exactly once"):
        summary.combine_partials([a])
    c = summary.binned_power_partials(spec[:, 5:], N, BOX, y0=5, shot_noise=1.0)
    with pytest.raises(ValueError, match="disagree"):
        summary.combine_partials([a, c])


def test_tsc_window_is_the_cic_window_at_exponent_three_halves():
    """TSC's window is sinc^3 per axis and CIC's sinc^2: both 1 at k = 0 and
    W_tsc = W_cic^1.5 everywhere, which pins the exponent."""
    w_t, w_c = tsc_window(16, 32.0), cic_window(16, 32.0)
    assert w_t[0, 0, 0] == 1.0 and w_c[0, 0, 0] == 1.0
    ok = w_c > 1e-12
    np.testing.assert_allclose(w_t[ok], w_c[ok] ** 1.5, rtol=1e-12)

    # and they differ by >2% at half Nyquist, so using the wrong one shows
    knyq = np.pi * 16 / 32.0
    s = np.sinc(0.5 * knyq / (2 * knyq))
    assert abs(s**3 / s**2 - 1.0) > 0.02


def test_corrections_are_applied_in_the_painted_order_and_reported():
    """<|delta|^2> = W^2 (P + 1/nbar), so the estimator is raw/W^2 - shot, not
    (raw - shot)/W^2; checked on a spectrum of ones. Both corrections are reported."""
    n, box = 16, 32.0
    spec = np.ones((n, n, n // 2 + 1), dtype=np.complex128)
    shot = 1e-4

    def win(kx, ky, kz):
        return np.full((len(kx), len(ky), len(kz)), 0.5)

    plain = summary.binned_power(spec, n, box, min_weight=0.0)
    both = summary.binned_power(spec, n, box, window=win, shot_noise=shot, min_weight=0.0)
    np.testing.assert_allclose(both["p"], plain["p"] / 0.25 - shot, rtol=1e-12)
    # the corrections are on the card rather than hidden inside `p`
    np.testing.assert_allclose(both["window_correction"], 0.25, rtol=1e-12)
    np.testing.assert_allclose(both["shot_fraction"], shot / plain["p"], rtol=1e-12)


def test_nonlinear_scale_finds_the_analytic_crossing():
    """Delta^2 = k^3 P / (2 pi^2) = 1; for P = A k^-2 the crossing is k_nl = 2 pi^2 / A."""
    A = 50.0
    k_nl = summary.nonlinear_scale(lambda k: A * k**-2.0)
    assert abs(k_nl / (2.0 * np.pi**2 / A) - 1.0) < 1e-3
    # a spectrum that never crosses returns None rather than an extrapolation
    assert summary.nonlinear_scale(lambda k: 1e-12 * k**-2.0) is None


def test_the_scanned_range_is_on_the_card_beside_the_none():
    """A None is read against the scanned range, so NL_SCAN_K must be the range actually
    scanned: a crossing just inside its ceiling is found, one just outside is not."""
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
    """Card with 4 bins and min_weight=20: the 16^3 smoke mesh has tens of modes per
    bin, and the production default would drop every bin."""
    edges = np.linspace(0.0, 0.5 * np.pi * cfg.n_coarse / cfg.box_size, 5)
    return summary.pk_summary_card(
        st, cfg, Cosmology(), a_out=1.0, edges=edges, min_weight=20.0, **kw
    )


def test_the_card_refuses_to_be_empty():
    """A zero-bin card (production defaults at a smoke mesh) refuses: it would pass
    every length and finiteness check while carrying no measurement."""
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
    """A caller-supplied coarse delta gives the same card as repainting (which costs
    4.3 GB and a streamed pass at C-gh); a wrong-shape delta refuses."""
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
    """The per-slab TSC window (built to avoid a second full-grid f64 array, 17 GB at
    C-gh) equals the full-grid one exactly."""
    n, box = 12, 24.0
    k_nyq = np.pi * n / box
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=box / n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=box / n)
    ref = tsc_window(n, box)
    for lo in range(0, n, 5):
        hi = min(lo + 5, n)
        got = summary.tsc_window_slab(kx[lo:hi], kx, kz, k_nyq)
        np.testing.assert_allclose(got, ref[lo:hi], rtol=0, atol=0)
