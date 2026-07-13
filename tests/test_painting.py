"""painting.py: CIC paints (int + f32 twins), reads, and the one-interface
wrapper (promotes M0 self-check 6; R3's determinism check becomes a permanent
test -- trivially true on CPU, authoritative when run on deneb CUDA)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from inexor.codec import rint_i
from inexor.painting import (
    _CORNERS,
    _cic_pieces,
    _corner_flat_weight,
    check_int_paint_headroom,
    cic_read_vector,
    counts_from_int,
    density_contrast,
    paint_f32,
    paint_int,
)

N, L = 32, 200.0


def _positions(seed, n=20000):
    return jax.random.uniform(jax.random.PRNGKey(seed), (n, 3), minval=0.0, maxval=L)


def test_paint_int_vs_f32_fixed_point_bound():
    # M0 self-check 6, verbatim bound: each corner deposit errs <= 2^-(F+1);
    # a cell collects ~lambda*8 deposits (lambda ~ 5 here) -> 32 deposits' worth.
    pos = _positions(3)
    F = 12
    mi = counts_from_int(paint_int(pos, N, L, frac_bits=F), F)
    mf = paint_f32(pos, N, L)
    assert float(jnp.max(jnp.abs(mi - mf))) < 32 * 2.0 ** -(F + 1)


def test_f32_mass_conservation_exact_class():
    pos = _positions(4, n=4096)
    total = float(jnp.sum(paint_f32(pos, N, L)))
    assert total == pytest.approx(4096.0, abs=5e-2)  # f32 summation round-off only


def test_int_paint_per_particle_weight_sum():
    # R3's mass-conservation arm: the 8 quantized corner weights of one particle
    # sum to 2^F up to 8 half-ulp roundings (measured 7.3e-4 relative at F=12).
    pos = _positions(5, n=8192)
    F = 12
    base, frac = _cic_pieces(pos, N, L)
    wsum = jnp.zeros((8192,), jnp.int32)
    for corner in _CORNERS:
        _, w = _corner_flat_weight(base, frac, corner, N)
        wsum = wsum + rint_i(w * np.float32(2.0**F))
    rel = jnp.max(jnp.abs(wsum - 2**F)) / 2.0**F
    assert float(rel) <= 8 * 2.0 ** -(F + 1) / 1.0


def test_periodic_edge_deposit():
    # a particle just inside the far box edge deposits into cell 0 by wrap
    pos = jnp.asarray([[L - 1e-3, 0.5 * L / N, 0.5 * L / N]])
    counts = paint_f32(pos, N, L)
    assert float(counts.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(counts[0].sum() + counts[-1].sum()) == pytest.approx(1.0, abs=1e-6)


def test_cic_read_vector_matches_manual_gather():
    pos = _positions(6, n=1000)
    key = jax.random.PRNGKey(7)
    gx, gy, gz = (jax.random.normal(k, (N, N, N)) for k in jax.random.split(key, 3))
    vec = cic_read_vector(gx, gy, gz, pos, N, L)
    base, frac = _cic_pieces(pos, N, L)
    for i, g in enumerate((gx, gy, gz)):
        flatg = g.reshape(-1)
        acc = jnp.zeros((1000,))
        for corner in _CORNERS:
            flat, w = _corner_flat_weight(base, frac, corner, N)
            acc = acc + w * flatg[flat]
        assert jnp.allclose(vec[:, i], acc, rtol=1e-6, atol=1e-6)


def test_paint_int_repeat_and_retrace_determinism():
    """R3's core check as a permanent test: repeated dispatch AND a fresh trace
    give bit-identical int meshes. Trivial on CPU; run on deneb CUDA (pixi run
    -e gpu pytest -k retrace) for the authoritative atomics arm."""
    pos = _positions(8)
    ref = paint_int(pos, N, L)
    for _ in range(3):
        assert jnp.array_equal(paint_int(pos, N, L), ref)
    fresh = jax.jit(lambda p: paint_int(p, N, L))(pos)
    assert jnp.array_equal(fresh, ref)


def test_density_contrast_uniform_grid_is_exactly_zero():
    # one particle per cell at exact cell corners: frac == 0, so BOTH paints
    # give exactly one count per cell -> delta == 0 identically
    d = L / N
    coords = jnp.arange(N, dtype=jnp.float32) * d
    qx, qy, qz = jnp.meshgrid(coords, coords, coords, indexing="ij")
    q = jnp.stack([qx.reshape(-1), qy.reshape(-1), qz.reshape(-1)], axis=1)
    for paint in ("int", "f32"):
        delta = density_contrast(q, N, L, N**3, paint=paint)
        assert float(jnp.max(jnp.abs(delta))) == 0.0


def test_density_contrast_validates_paint_arg():
    with pytest.raises(ValueError, match="paint"):
        density_contrast(_positions(9, n=8), N, L, 8, paint="tsc")


def test_headroom_guard_fires():
    # 2^20 particles in one cell x 2^12 = 2^32 >= 2^31: accumulator would overflow
    with pytest.raises(ValueError, match="headroom"):
        check_int_paint_headroom(2**30, frac_bits=12, max_cell_particles=2.0**20)
    check_int_paint_headroom(1024**3, frac_bits=12)  # flagship class w/ arch budget: fine
