"""forces.py: kernel identity vs an independent reference, exact-zero force on
the uniform grid, paint-path agreement, and the shared-object cache contract
(architecture Sec. 5: step_fwd/step_rev must receive the SAME force object)."""

import jax
import jax.numpy as jnp
import numpy as np

from inexor.config import BoxConfig
from inexor.forces import k_components, make_force_fn

N, L = 32, 200.0
BOX = BoxConfig(n_mesh=N, box_size=L)


def test_kernel_matches_independent_reference():
    """ik/k^2 kernel vs a from-scratch reference built with a different
    composition (meshgrid instead of broadcast reshapes). Machine precision at
    the storage dtype: the library never enables x64, so jnp.asarray narrows
    the returned arrays to f32 -- compare at f32 tolerances. The force == ZA
    IDENTITY test (same kernel via lpt.za_psi) lands with S3."""
    ikx, iky, ikz, inv_k2 = k_components(N, L, np.float64)
    kvec = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kzv = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    KX, KY, KZ = np.meshgrid(kvec, kvec, kzv, indexing="ij")
    k2 = KX**2 + KY**2 + KZ**2
    k2[0, 0, 0] = 1.0
    assert np.allclose(np.asarray(ikx).imag * np.ones_like(k2), KX, rtol=1e-6, atol=1e-6)
    assert np.allclose(np.asarray(iky).imag * np.ones_like(k2), KY, rtol=1e-6, atol=1e-6)
    assert np.allclose(np.asarray(ikz).imag * np.ones_like(k2), KZ, rtol=1e-6, atol=1e-6)
    assert np.allclose(np.asarray(inv_k2), 1.0 / k2, rtol=1e-6)


def test_uniform_grid_force_exactly_zero():
    # unperturbed Lagrangian grid -> delta == 0 exactly -> force == 0 exactly
    d = L / N
    coords = jnp.arange(N, dtype=jnp.float32) * d
    qx, qy, qz = jnp.meshgrid(coords, coords, coords, indexing="ij")
    q = jnp.stack([qx.reshape(-1), qy.reshape(-1), qz.reshape(-1)], axis=1)
    for paint in ("int", "f32"):
        g = make_force_fn(BOX, paint=paint)(q)
        assert float(jnp.max(jnp.abs(g))) == 0.0


def test_mean_force_near_zero_random_positions():
    # k=0 mode nulled -> mean force vanishes to paint/read round-off
    pos = jax.random.uniform(jax.random.PRNGKey(0), (20000, 3), minval=0.0, maxval=L)
    g = make_force_fn(BOX, paint="f32")(pos)
    rms = float(jnp.sqrt(jnp.mean(g**2)))
    assert float(jnp.max(jnp.abs(jnp.mean(g, axis=0)))) < 1e-3 * max(rms, 1e-30)


def test_int_vs_f32_paint_force_agreement():
    pos = jax.random.uniform(jax.random.PRNGKey(1), (20000, 3), minval=0.0, maxval=L)
    gi = make_force_fn(BOX, paint="int")(pos)
    gf = make_force_fn(BOX, paint="f32")(pos)
    scale = float(jnp.max(jnp.abs(gf)))
    assert float(jnp.max(jnp.abs(gi - gf))) < 5e-3 * scale


def test_force_fn_cache_identity():
    # equal args -> the SAME closure object (step_fwd/step_rev sharing contract)
    f1 = make_force_fn(BOX, paint="int")
    f2 = make_force_fn(BoxConfig(n_mesh=N, box_size=L), paint="int")
    assert f1 is f2
    f3 = make_force_fn(BoxConfig(n_mesh=16, box_size=L), paint="int")
    assert f3 is not f1
