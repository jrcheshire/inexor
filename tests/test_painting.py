"""painting.py: CIC paints (int and f32 twins), reads, the `density_contrast` wrapper,
and the sub-block TSC paint. The determinism test is trivial on CPU and meaningful on a
GPU, where the paint uses atomics."""

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
    # each corner deposit errs <= 2^-(F+1); a cell collects ~8 * lambda deposits
    # (lambda ~ 0.6 particles per cell here), bounded by 32 deposits' worth
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
    # the 8 quantized corner weights of one particle sum to 2^F up to 8 half-ulp
    # roundings (measured 7.3e-4 relative at F=12)
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
    """Repeated dispatch and a fresh trace give bit-identical int meshes. Trivial on CPU;
    on a GPU (`pixi run -e gpu pytest -k retrace`) it exercises the atomics."""
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
    check_int_paint_headroom(1024**3, frac_bits=12)  # 1024^3 at the default cell cap: fine


# ------------------------------------------------- the sub-block TSC paint


def test_the_subblock_tsc_paint_is_bitwise_the_global_one():
    """`paint_tsc_int_subblock`, placed back into the box, is bitwise `paint_tsc_int`:
    interior block, masked pad rows, a block wrapping the periodic boundary, and a
    full-axis block."""
    from inexor import painting

    N, box = 16, 32.0
    cell = box / N
    rng = np.random.default_rng(7)

    def _case(origin, extent, n_pad):
        # positions whose TSC bases stay in [origin+1, origin+extent-2] per
        # axis (mod N); a full axis (extent == N) draws the whole box
        cols = []
        for a in range(3):
            if int(extent[a]) >= N:
                cols.append(rng.uniform(0.0, box, size=96))
            else:
                lo = (origin[a] + 1 - 0.4) * cell
                hi = (origin[a] + int(extent[a]) - 2 + 0.4) * cell
                cols.append(np.mod(rng.uniform(lo, hi, size=96), box))
        x = np.stack(cols, axis=-1)
        xp = np.concatenate([x, np.zeros((n_pad, 3))]) if n_pad else x
        lv = np.zeros(len(xp), dtype=bool)
        lv[: len(x)] = True
        full = np.asarray(
            painting.paint_tsc_int(jnp.asarray(xp), N, box, 12, live=jnp.asarray(lv))
        )
        sub = np.asarray(
            painting.paint_tsc_int_subblock(
                jnp.asarray(xp), tuple(origin), tuple(extent), N, box, 12,
                live=jnp.asarray(lv),
            )
        )
        placed = np.zeros((N, N, N), dtype=np.int32)
        ax = [(np.arange(int(extent[a])) + int(origin[a])) % N for a in range(3)]
        placed[np.ix_(*ax)] = sub
        assert np.array_equal(placed, full), (origin, extent)
        assert full.sum() > 0, "no mass painted; comparison is vacuous"

    _case((3, 3, 3), (8, 8, 8), n_pad=0)  # interior
    _case((3, 3, 3), (8, 8, 8), n_pad=32)  # masked pad rows
    _case((13, 13, 13), (8, 8, 8), n_pad=0)  # wraps the periodic boundary
    _case((0, 13, 3), (16, 8, 8), n_pad=0)  # x is the degenerate full axis
