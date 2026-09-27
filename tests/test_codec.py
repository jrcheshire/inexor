"""codec.py: the integer rounding primitive. Assertions are exact integer equality."""

import jax.numpy as jnp
import numpy as np

from inexor.codec import rint_i

U16_MOD = 2**16


def test_rint_i_half_even_and_int32_routing():
    z = jnp.asarray([0.5, 1.5, 2.5, -0.5, -1.5, 40000.7])
    out = rint_i(z)
    assert out.dtype == jnp.int32
    assert np.array_equal(np.asarray(out), [0, 2, 2, 0, -2, 40001])


