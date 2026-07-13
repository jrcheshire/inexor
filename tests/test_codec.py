"""codec.py: lattice primitives, STE boundary, ladder guard (promotes M0
self-checks 2 and 3 into permanent tests; adds the arch Sec. 4 |m|>1 lemma).

Reversibility assertions are EXACT integer equality, never tolerances
(house rule)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from inexor.codec import (
    INT16_MAX,
    U16_MOD,
    W_ABS_WARN,
    build_ladder,
    dequant_w,
    dequant_x,
    encode_w,
    encode_x,
    iadd,
    imask_add,
    imask_sub,
    isub,
    rint_i,
    s_w0_policy,
    ste_round,
    ste_wrap_s,
    ste_wrap_u,
    w_headroom_bits,
)


def _rand_state(seed, n=4096):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.integers(0, U16_MOD, size=(n, 3)), dtype=jnp.uint16)
    w = jnp.asarray(rng.integers(-(2**15), 2**15, size=(n, 3)), dtype=jnp.int16)
    inc = jnp.asarray(rng.integers(-(2**17), 2**17, size=(n, 3)), dtype=jnp.int32)
    return x, w, inc


def test_isub_inverts_iadd_exactly_under_wrap():
    x, w, inc = _rand_state(0)
    # increments far beyond the int16 range force wraps; inversion must be exact
    assert jnp.array_equal(isub(iadd(x, inc), inc), x)
    assert jnp.array_equal(isub(iadd(w, inc), inc), w)


def test_imask16_bit_identical_to_uint16_path():
    x, _, inc = _rand_state(1)
    x32 = x.astype(jnp.int32)
    masked = imask_add(x32, inc, 16)
    assert jnp.array_equal(masked.astype(jnp.uint16), iadd(x, inc))
    assert jnp.array_equal(imask_sub(masked, inc, 16), x32)


def test_rint_i_half_even_and_int32_routing():
    z = jnp.asarray([0.5, 1.5, 2.5, -0.5, -1.5, 40000.7])
    out = rint_i(z)
    assert out.dtype == jnp.int32
    assert np.array_equal(np.asarray(out), [0, 2, 2, 0, -2, 40001])


def test_encode_decode_round_trip():
    rng = np.random.default_rng(2)
    s_x, s_w0 = 256.0 / U16_MOD, 3e-2
    # stay > 0.5*s_x below the box edge: values that ROUND to 2^16 wrap to 0
    # (correct periodic physics, but it breaks the abs-error metric used here)
    x_phys = jnp.asarray(rng.uniform(0.0, 255.99, size=(1024, 3)), dtype=jnp.float32)
    xi = encode_x(x_phys, s_x)
    assert xi.dtype == jnp.uint16
    assert float(jnp.max(jnp.abs(dequant_x(xi, s_x) - x_phys))) <= 0.5 * s_x * (1 + 1e-5)
    v = jnp.asarray(rng.normal(0.0, 200.0 * s_w0, size=(1024, 3)), dtype=jnp.float32)
    wi = encode_w(v, s_w0)
    assert wi.dtype == jnp.int16
    assert float(jnp.max(jnp.abs(dequant_w(wi, s_w0) - v))) <= 0.5 * s_w0 * (1 + 1e-5)


def test_ste_primal_and_identity_jacobian():
    z = jnp.asarray([0.3, 1.7, -2.4])
    assert jnp.array_equal(ste_round(z), jnp.rint(z))
    assert jnp.allclose(jax.grad(lambda t: ste_round(t).sum())(z), 1.0)
    zu = jnp.asarray([5.0, 70000.0, -3.0])
    assert jnp.array_equal(ste_wrap_u(zu, 2.0**16), jnp.mod(zu, 2.0**16))
    assert jnp.allclose(jax.grad(lambda t: ste_wrap_u(t, 2.0**16).sum())(zu), 1.0)
    zs = jnp.asarray([40000.0, -40000.0, 12.0])
    ws = ste_wrap_s(zs, 2.0**16)
    assert float(jnp.max(ws)) < 2.0**15 and float(jnp.min(ws)) >= -(2.0**15)
    assert jnp.allclose(jax.grad(lambda t: ste_wrap_s(t, 2.0**16).sum())(zs), 1.0)


def test_multiplicative_lemma_invertible_above_unity():
    """Arch Sec. 4 lemma: w' = rint(m w) IS invertible via rint(w'/m) when |m| > 1
    (unused by BullFrog -- the ladder exists because its |alpha| < 1)."""
    w = np.arange(-(2**15), 2**15, dtype=np.int64)
    for m in (1.5, 2.0, 7.3, -1.01):
        w_scaled = np.rint(m * w)
        back = np.rint(w_scaled / m)
        assert np.array_equal(back, w), f"lemma fails at m={m}"


def _fake_weights(K, alphas):
    a_steps = np.geomspace(0.1, 1.0, K + 1)
    D_steps = a_steps.copy()  # EdS-like placeholder; build_ladder doesn't care
    betas = 1.0 - np.asarray(alphas)
    dD = np.diff(D_steps)
    D_mid = D_steps[:-1] + 0.5 * dD
    return a_steps, D_steps, np.asarray(alphas, float), betas, dD, D_mid


def test_ladder_guard_refuses_near_zero_alpha():
    K = 5
    args = _fake_weights(K, [0.5, 0.4, 0.02, 0.6, 0.7])  # step 2 below the 0.05 floor
    with pytest.raises(ValueError, match="unfit schedule"):
        build_ladder(*args, s_w0=1.0, s_x=1.0)


def test_ladder_accepts_and_prices_fit_schedule():
    K = 4
    alphas = [0.5, 0.5, 0.5, 0.5]
    lad = build_ladder(*_fake_weights(K, alphas), s_w0=2.0, s_x=0.5)
    assert lad.n_steps == K
    assert lad.P[0] == 1.0
    assert lad.P[-1] == pytest.approx(0.5**4)
    assert lad.bits_consumed == pytest.approx(4.0)  # log2(1/0.0625)
    # constants follow the Sec. 4 formulas
    assert np.allclose(lad.kappa, lad.betas / (lad.D_mid * lad.P[1:] * 2.0))
    assert np.allclose(lad.c1, 0.5 * lad.dD * 2.0 * lad.P[:-1] / 0.5)
    assert np.allclose(lad.c2, 0.5 * lad.dD * 2.0 * lad.P[1:] / 0.5)


def test_s_w0_policy_formula():
    assert s_w0_policy(10.0, 0.02) == pytest.approx(2.5 * 10.0 / (0.9 * 32767.0 * 0.02))


def test_w_headroom_monitor():
    w = jnp.asarray([[100, -200, 50]], dtype=jnp.int16)
    assert float(w_headroom_bits(w)) == pytest.approx(np.log2(INT16_MAX / 200.0), rel=1e-5)
    w_hot = jnp.asarray([[W_ABS_WARN + 10, 0, 0]], dtype=jnp.int16)
    assert float(w_headroom_bits(w_hot)) < np.log2(1.0 / 0.9) + 1e-3
