"""codec.py: the integer lattice primitives and the STE boundary.

Trimmed 2026-08-08 with the v1 retirement. What went with it: the w-frame
ladder guard and pricing tests, the global uint16 encode/decode round trip, the
s_w0 policy formula, the int16 headroom monitor, the `imask_*` width knob, and
the arch Sec. 4 |m| > 1 invertibility lemma -- every one of them a statement
about machinery that no longer exists. The T9 codec brings its own gates at
M-v2-1, including a round trip and a wrap-never-clamp assertion.

What remains covers the primitives v2 still stands on. Reversibility assertions
are EXACT integer equality, never tolerances (house rule).
"""

import jax
import jax.numpy as jnp
import numpy as np

from inexor.codec import iadd, isub, rint_i, ste_round, ste_wrap_s, ste_wrap_u

U16_MOD = 2**16


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


def test_rint_i_half_even_and_int32_routing():
    z = jnp.asarray([0.5, 1.5, 2.5, -0.5, -1.5, 40000.7])
    out = rint_i(z)
    assert out.dtype == jnp.int32
    assert np.array_equal(np.asarray(out), [0, 2, 2, 0, -2, 40001])


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
