"""integrate.py: coefficient oracles, linear-mode growth pins, integer
reversibility (micro + tier-0), STE twin bit-match, public API.

Promotes M0 self-checks 1, 7, 8 and the R1 probe's essential arms. Exact
integer equality everywhere reversibility is asserted (house rule). The
[slow] tier-0 test on CPU is the CI regression guard; the SAME test run on
deneb (pixi run -e gpu pytest -k tier0) is the authoritative CUDA arm.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from inexor.codec import s_w0_policy
from inexor.config import PLANCK, BoxConfig, QuantConfig, TimeConfig
from inexor.cosmology import growth_factor_a
from inexor.forces import make_force_fn
from inexor.ic import gaussian_delta
from inexor.integrate import (
    KDKConsts,
    StepConsts,
    _bullfrog_weights,
    a_grid,
    bullfrog_float_coeffs,
    bullfrog_table,
    drift_factor,
    evolve,
    evolve_float,
    fastpm_drift_factor,
    fastpm_kick_factor,
    kdk_table,
    kick_factor,
    ladder_for_schedule,
    p_to_v,
    replay_roundtrip,
    run_perstep,
    run_scan,
    simulate,
    step_consts,
    step_fwd,
    step_float,
    step_kdk_fwd,
    step_kdk_rev,
    step_rev,
    v_to_p,
)
from inexor.lpt import lagrangian_grid, za_ics

Q = QuantConfig()


# ----------------------------------------------------------------- oracles


def test_bullfrog_eds_closed_form():
    """M0 self-check 1 / the roadmap M1 GATE: < 1e-12 vs the published EdS
    closed form (mbody test_integrate.py:472 oracle)."""
    err = 0.0
    for n in [1.0, 2.0, 3.5, 7.0, 20.0]:
        D0, dD = n * 0.05, 0.05
        alpha, beta, _, _ = _bullfrog_weights(D0, D0 + dD)
        m = D0 / dD
        alpha_ref = (4 * m * (4 * m + 1) - 5) / (4 * m * (4 * m + 7) + 7)
        beta_ref = (24 * m + 12) / (4 * m * (4 * m + 7) + 7)
        err = max(err, abs(alpha - alpha_ref), abs(beta - beta_ref))
    assert err < 1e-12


def test_ladder_guard_full_composition():
    """M0 self-check 2 at the composition level (build_ladder alone is tested
    in test_codec): rejects linear-0.04 K=11 under EdS growth, accepts
    log-0.1 K=10 under exact LCDM at ~5-bit cost."""
    with pytest.raises(ValueError, match="unfit schedule"):
        ladder_for_schedule(a_grid(0.04, 1.0, 11, "linear"), PLANCK, 1.0, 1.0, D_of_a=lambda a: a)
    lad = ladder_for_schedule(a_grid(0.1, 1.0, 10, "log"), PLANCK, 1.0, 1.0)
    assert lad.n_steps == 10
    assert 4.0 < lad.bits_consumed < 6.5


def test_fastpm_reduces_to_exact_small_step():
    """mbody pattern: over a tiny interval the growth-corrected FastPM kernels
    converge to the exact background integrals."""
    a0, a1 = 0.5, 0.5005
    a_c = 0.5 * (a0 + a1)
    assert fastpm_kick_factor(a0, a1, a_c, PLANCK) == pytest.approx(
        kick_factor(a0, a1, PLANCK), rel=1e-5
    )
    assert fastpm_drift_factor(a0, a1, a_c, PLANCK) == pytest.approx(
        drift_factor(a0, a1, PLANCK), rel=1e-5
    )


def test_v_p_conversion_round_trip():
    v = jnp.asarray([[1.0, -2.0, 3.0]])
    a = 0.3
    assert jnp.allclose(p_to_v(v_to_p(v, a, PLANCK), a, PLANCK), v, rtol=1e-6)


# ------------------------------------------------- linear-mode growth pins
# mbody _grow_linear_mode methodology: a linear mode's geometric force equals
# its displacement (both track D), so the COEFFICIENT TABLES alone determine
# the growth -- a scalar recurrence, exact math, no PM/CIC resolution effects
# (those belong to the S6 deficit matrix, not unit tests).


def _grow_linear_mode_kdk(integrator, K, a_i=0.1, a_f=1.0):
    from inexor.cosmology import E_of_a, growth_rate_a

    a = a_grid(a_i, a_f, K, "log")
    Di = growth_factor_a(a_i, PLANCK)
    x = Di
    p = a_i**2 * E_of_a(a_i, PLANCK) * Di * growth_rate_a(a_i, PLANCK)
    for k1, dr, k2 in kdk_table(a, PLANCK, integrator):
        p = p + k1 * x
        x = x + dr * p
        p = p + k2 * x
    return x


def _grow_linear_mode_bullfrog(K, a_i=0.1, a_f=1.0):
    t = bullfrog_table(a_grid(a_i, a_f, K, "log"), PLANCK)
    x, v = t.D_steps[0], 1.0  # x = D Psi with Psi = 1; v_D = Psi = 1
    for dD_half, alpha, bcoef in bullfrog_float_coeffs(t):
        x = x + dD_half * v
        v = alpha * v + bcoef * x
        x = x + dD_half * v
    return x


def test_fastpm_linear_mode_exact_growth():
    """FastPM's defining property: a linear mode grows as D(a) exactly at ANY
    step count (pins the previously-unexercised fastpm table)."""
    D_f = growth_factor_a(1.0, PLANCK)
    for K in (2, 4, 8):
        assert abs(_grow_linear_mode_kdk("fastpm", K) / D_f - 1.0) < 1e-4


def test_bullfrog_linear_mode_exact_growth():
    """BullFrog's Zel'dovich consistency: exact linear growth per step."""
    D_f = growth_factor_a(1.0, PLANCK)
    for K in (2, 4, 8):
        assert abs(_grow_linear_mode_bullfrog(K) / D_f - 1.0) < 1e-6


def test_exact_kdk_deficit_and_convergence():
    """The exact-KDK fallback has a real low-K growth deficit, converging
    toward D as K rises. Measured (this toy, a 0.1 -> 1.0): log spacing
    0.61% at K=2 -> 4.8e-6 at K=32; mbody's documented ~2%+ K=2 number is
    its LINEAR-spacing default (7.7% here) -- spacing matters more than K.
    """
    D_f = growth_factor_a(1.0, PLANCK)
    err = {K: abs(_grow_linear_mode_kdk("exact", K) / D_f - 1.0) for K in (2, 8, 32)}
    assert err[2] > 3e-3
    assert err[2] > err[8] > err[32]
    assert err[32] < 1e-4


# ------------------------------------------------ integer micro-replays


def _toy_force(xp):
    """Deterministic elementwise pseudo-force (M0 self-check 7)."""
    return jnp.sin(xp * 0.37) * 55.0 + jnp.cos(xp[:, ::-1] * 0.11) * 21.0


def _rand_int_state(seed, n=4096):
    key = jax.random.PRNGKey(seed)
    kx_, kw_ = jax.random.split(key)
    x0 = jax.random.randint(kx_, (n, 3), 0, 2**16, dtype=jnp.int32).astype(jnp.uint16)
    w0 = jax.random.randint(kw_, (n, 3), -(2**14), 2**14, dtype=jnp.int32).astype(jnp.int16)
    return x0, w0


def test_micro_replay_bullfrog_100_steps():
    """M0 self-check 7: 100 synthetic BullFrog steps forward + reverse, exact."""
    x0, w0 = _rand_int_state(4)
    kc_ = jax.random.PRNGKey(5)
    cs = StepConsts(
        c1=jnp.abs(jax.random.normal(kc_, (100,), dtype=jnp.float32)) * 0.03,
        c2=jnp.abs(jax.random.normal(kc_, (100,), dtype=jnp.float32)) * 0.03,
        kappa=jnp.ones((100,), dtype=jnp.float32) * 0.7,
    )
    x, w = x0, w0
    for k in range(100):
        c = StepConsts(cs.c1[k], cs.c2[k], cs.kappa[k])
        x, w = step_fwd(x, w, c, _toy_force, s_x=1.0)
    for k in reversed(range(100)):
        c = StepConsts(cs.c1[k], cs.c2[k], cs.kappa[k])
        x, w = step_rev(x, w, c, _toy_force, s_x=1.0)
    assert jnp.array_equal(x, x0) and jnp.array_equal(w, w0)


def test_micro_replay_kdk_100_steps():
    """The new KDK kernel under the same synthetic regime, exact bits."""
    x0, w0 = _rand_int_state(6)
    kc_ = jax.random.PRNGKey(7)
    cs = KDKConsts(
        kappa1=jnp.abs(jax.random.normal(kc_, (100,), dtype=jnp.float32)) * 0.4,
        c=jnp.abs(jax.random.normal(kc_, (100,), dtype=jnp.float32)) * 0.03,
        kappa2=jnp.abs(jax.random.normal(kc_, (100,), dtype=jnp.float32)) * 0.4,
    )
    x, w = x0, w0
    for k in range(100):
        c = KDKConsts(cs.kappa1[k], cs.c[k], cs.kappa2[k])
        x, w = step_kdk_fwd(x, w, c, _toy_force, s_x=1.0)
    for k in reversed(range(100)):
        c = KDKConsts(cs.kappa1[k], cs.c[k], cs.kappa2[k])
        x, w = step_kdk_rev(x, w, c, _toy_force, s_x=1.0)
    assert jnp.array_equal(x, x0) and jnp.array_equal(w, w0)


def test_micro_replay_pm_and_ste_twin():
    """M0 self-check 8: 16^3 BullFrog K=8 PM exact-bit replay + the STE float
    twin's primal bit-matches the integer trajectory."""
    N, L, K = 16, 100.0, 8
    box = BoxConfig(n_mesh=N, box_size=L)
    delta0 = gaussian_delta(jax.random.PRNGKey(5), N, L, PLANCK)
    x_phys, v = za_ics(delta0, L, 0.1, PLANCK)
    a_steps = a_grid(0.1, 1.0, K, "log")
    unit = ladder_for_schedule(a_steps, PLANCK, 1.0, 1.0)
    s_w0 = s_w0_policy(float(jnp.max(jnp.abs(v))), unit.P[-1])
    lad = ladder_for_schedule(a_steps, PLANCK, s_w0, box.s_x)
    cs = step_consts(lad)
    force = make_force_fn(box, paint="int")
    from inexor.codec import encode_w, encode_x

    x0i, w0i = encode_x(x_phys, box.s_x), encode_w(v, s_w0)
    xi, wi = x0i, w0i
    xf = x0i.astype(jnp.float32)
    wf = w0i.astype(jnp.float32)
    twin_mismatch = 0
    for k in range(K):
        c = StepConsts(cs.c1[k], cs.c2[k], cs.kappa[k])
        xi, wi = step_fwd(xi, wi, c, force, box.s_x)
        xf, wf = step_float(xf, wf, c, force, box.s_x)
        twin_mismatch += int(jnp.sum(xf != xi.astype(jnp.float32)))
        twin_mismatch += int(jnp.sum(wf != wi.astype(jnp.float32)))
    for k in reversed(range(K)):
        c = StepConsts(cs.c1[k], cs.c2[k], cs.kappa[k])
        xi, wi = step_rev(xi, wi, c, force, box.s_x)
    assert jnp.array_equal(xi, x0i) and jnp.array_equal(wi, w0i)
    assert twin_mismatch == 0


# --------------------------------------------------------- drivers + API


def test_scan_and_perstep_drivers_agree_bitwise():
    N, L, K = 16, 100.0, 6
    box = BoxConfig(n_mesh=N, box_size=L)
    delta0 = gaussian_delta(jax.random.PRNGKey(8), N, L, PLANCK)
    x_phys, v = za_ics(delta0, L, 0.1, PLANCK)
    a_steps = a_grid(0.1, 1.0, K, "log")
    unit = ladder_for_schedule(a_steps, PLANCK, 1.0, 1.0)
    s_w0 = s_w0_policy(float(jnp.max(jnp.abs(v))), unit.P[-1])
    lad = ladder_for_schedule(a_steps, PLANCK, s_w0, box.s_x)
    cs = step_consts(lad)
    force = make_force_fn(box, paint="int")
    from inexor.codec import encode_w, encode_x

    x0i, w0i = encode_x(x_phys, box.s_x), encode_w(v, s_w0)
    xs, ws = run_scan(x0i, w0i, cs, step_fwd, force, box.s_x)
    xp, wp = run_perstep(x0i, w0i, cs, step_fwd, force, box.s_x, donate=False)
    # NOTE macOS-arm64 CPU jit quirks memory: differently-compiled programs may
    # tie-flip; R1 certified both drivers pass replay independently on CUDA.
    # Bitwise agreement here is a CPU regression canary, not a design gate.
    assert jnp.array_equal(xs, xp) and jnp.array_equal(ws, wp)


@pytest.mark.parametrize("integrator", ["bullfrog", "fastpm", "exact"])
def test_replay_roundtrip_all_integrators(integrator):
    N, L = 16, 100.0
    box = BoxConfig(n_mesh=N, box_size=L)
    t = TimeConfig(a_init=0.1, a_final=1.0, n_steps=5, integrator=integrator)
    delta0 = gaussian_delta(jax.random.PRNGKey(9), N, L, PLANCK)
    x0, v0 = za_ics(delta0, L, 0.1, PLANCK)
    ok, n_diff = replay_roundtrip(box, t, Q, PLANCK, x0, v0)
    assert ok, f"{integrator}: {n_diff} components differ after roundtrip"


def test_evolve_api_shapes_and_quantization_budget():
    N, L = 32, 200.0
    box = BoxConfig(n_mesh=N, box_size=L)
    t = TimeConfig(a_init=0.1, a_final=1.0, n_steps=4)
    delta0 = gaussian_delta(jax.random.PRNGKey(10), N, L, PLANCK)  # physical amplitude
    x0, v0 = za_ics(delta0, L, 0.1, PLANCK)
    x_f, v_f, (xi, wi), max_w = evolve(
        box, t, Q, PLANCK, x0, v0, return_int_state=True, monitor=True, driver="perstep"
    )
    assert x_f.shape == v_f.shape == (N**3, 3)
    assert xi.dtype == jnp.uint16 and wi.dtype == jnp.int16
    assert len(max_w) == 4 and max(max_w) < int(0.95 * 32767)
    assert float(jnp.max(x_f)) < L and float(jnp.min(x_f)) >= 0.0
    # quantized vs never-quantized reference (Tier-B class, scaled down):
    # position drift stays a small fraction of the rms displacement
    from inexor.diagnostics import min_image_rms

    x_ref, _ = evolve_float(box, t, PLANCK, x0, v0, paint="int")
    q = np.asarray(lagrangian_grid(N, L), np.float64)
    disp = np.asarray(x_ref, np.float64) - q
    disp -= L * np.round(disp / L)
    rms_disp = float(np.sqrt(np.mean(disp**2)))
    assert min_image_rms(x_f, x_ref, L) < 2e-2 * rms_disp


def test_simulate_smoke_all_paths():
    box = BoxConfig(n_mesh=16, box_size=100.0)
    t = TimeConfig(a_init=0.1, a_final=0.5, n_steps=3)
    for kwargs in (dict(), dict(f_NL=20.0), dict(lpt_order=1), dict(quantized=False)):
        x_f, v_f = simulate(box, t, Q, PLANCK, seed=0, **kwargs)[:2]
        assert bool(jnp.all(jnp.isfinite(x_f))) and bool(jnp.all(jnp.isfinite(v_f)))


# ------------------------------------------------------------ tier-0 [slow]


@pytest.mark.slow
@pytest.mark.parametrize("driver", ["scan", "perstep"])
def test_tier0_reversibility(driver):
    """R1 promoted: 64^3, K=10, exact integer replay over seeds, both drivers.
    CPU here is the CI regression guard; the deneb CUDA invocation of THIS
    test is the authoritative run (recorded in docs/m1-results.md)."""
    N, L, K = 64, 256.0, 10
    box = BoxConfig(n_mesh=N, box_size=L)
    t = TimeConfig(a_init=0.1, a_final=1.0, n_steps=K)
    for seed in range(5):
        delta0 = gaussian_delta(jax.random.PRNGKey(seed), N, L, PLANCK)
        x0, v0 = za_ics(delta0, L, 0.1, PLANCK)
        ok, n_diff = replay_roundtrip(box, t, Q, PLANCK, x0, v0, driver=driver)
        assert ok, f"seed {seed}: {n_diff} diffs"


@pytest.mark.slow
def test_tier0_wrap_adversarial():
    """R1's wrap-adversarial arm: s_w0/64 forces w to wrap int16 mid-run;
    replay must STILL be exact (D-007 demonstrated)."""
    N, L, K = 64, 256.0, 10
    box = BoxConfig(n_mesh=N, box_size=L)
    t = TimeConfig(a_init=0.1, a_final=1.0, n_steps=K)
    delta0 = gaussian_delta(jax.random.PRNGKey(0), N, L, PLANCK)
    x0, v0 = za_ics(delta0, L, 0.1, PLANCK)
    ok, n_diff = replay_roundtrip(box, t, Q, PLANCK, x0, v0, s_w0_div=64.0)
    assert ok, f"wrap-adversarial: {n_diff} diffs"
