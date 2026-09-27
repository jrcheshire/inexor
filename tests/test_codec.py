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

import jax.numpy as jnp
import numpy as np

from inexor.codec import rint_i

U16_MOD = 2**16


def test_rint_i_half_even_and_int32_routing():
    z = jnp.asarray([0.5, 1.5, 2.5, -0.5, -1.5, 40000.7])
    out = rint_i(z)
    assert out.dtype == jnp.int32
    assert np.array_equal(np.asarray(out), [0, 2, 2, 0, -2, 40001])


