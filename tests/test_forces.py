"""forces.py: kernel vs an independent reference, exact-zero force on the uniform grid,
paint-path agreement, and the coarse long-range solve (hoisted kernel build, factorized
transform, oversize guard)."""

import pytest

import jax
import jax.numpy as jnp
import numpy as np

from inexor.config import BoxConfig
from inexor.forces import k_components, make_force_fn

N, L = 32, 200.0
BOX = BoxConfig(n_mesh=N, box_size=L)


def test_kernel_matches_independent_reference():
    """ik/k^2 kernel matches a meshgrid-built reference. The library never enables x64,
    so the returned arrays are f32: tolerances are f32 round-off."""
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


# ============================================ the hoisted coarse kernel build


def _reference_coarse_solve(delta, n, box, r_s, match, fdtype):
    """Oracle: build all three complex kernels, apply the match, then solve.

    Same products as the hoisted per-component solve, only formed at a different time,
    so the two must agree bitwise.
    """
    from inexor.forces import cic_match_factor, split_kernels

    kers = split_kernels((n,) * 3, box / n, "long", r_s=r_s, fdtype=fdtype)
    mf, _ = cic_match_factor((n,) * 3, match[0], match[1], clip=None)
    kers = tuple(k * mf.astype(fdtype, copy=False) for k in kers)
    dk = jnp.fft.rfftn(jnp.asarray(delta))
    return [np.asarray(jnp.fft.irfftn(dk * jnp.asarray(k), s=(n,) * 3)) for k in kers]


@pytest.fixture
def x64():
    """f64 arms need the caller's opt-in; library code never toggles it."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.mark.parametrize("n", [16, 24, 32])
@pytest.mark.parametrize("fdtype", [np.float32, np.float64])
def test_the_hoisted_coarse_build_is_bitwise_the_build_it_replaced(n, fdtype, x64):
    """The hoisted kernel build is bitwise the all-at-once build.

    `coarse_kernel_parts` keeps only real half-grids and forms each complex kernel inside
    the solve as `(pref * ik) * mf`; folding `mf` into `pref` would reassociate and fail.
    """
    from inexor.forces import coarse_force_meshes, coarse_kernel_parts

    fdtype = np.dtype(fdtype)
    box = float(n)
    r_s = 2.0 * box / n
    match = (box / n, box / (4 * n))
    rng = np.random.default_rng(11)
    delta = (rng.standard_normal((n, n, n)) * 1e-3).astype(fdtype)

    want = _reference_coarse_solve(delta, n, box, r_s, match, fdtype)
    parts = coarse_kernel_parts(n, box, "long", r_s=r_s, match=match, fdtype=fdtype)
    # The oracle transforms monolithically; hold the transform so only the hoist varies.
    got = coarse_force_meshes(jnp.asarray(delta), n, box, "long", r_s=r_s,
                              match=match, fdtype=fdtype, parts=parts,
                              transform="monolithic")
    for i, (a, b) in enumerate(zip(want, got)):
        assert np.array_equal(a, b), (
            f"component {i} at n={n}/{fdtype.name} is not bitwise the pre-hoist "
            f"build: max|d| = {np.abs(a - b).max():.3e}"
        )


def test_the_hoisted_build_and_the_per_call_build_agree():
    """`parts=None` must reach the same numbers as a hoisted build, or a
    standalone `step` and a `run` would not be the same engine."""
    from inexor.forces import coarse_force_meshes, coarse_kernel_parts

    n, box = 24, 24.0
    r_s, match = 2.0, (1.0, 0.25)
    rng = np.random.default_rng(3)
    delta = (rng.standard_normal((n, n, n)) * 1e-3).astype(np.float32)
    dj = jnp.asarray(delta)
    a = coarse_force_meshes(dj, n, box, "long", r_s=r_s, match=match,
                            fdtype=np.float32)
    parts = coarse_kernel_parts(n, box, "long", r_s=r_s, match=match,
                               fdtype=np.float32)
    b = coarse_force_meshes(dj, n, box, "long", r_s=r_s, match=match,
                            fdtype=np.float32, parts=parts)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_parts_refuse_a_geometry_they_were_not_built_for():
    """Parts built for another geometry are refused, not coerced: dtype would promote and
    shapes broadcast silently."""
    from inexor.forces import coarse_force_meshes, coarse_kernel_parts

    parts = coarse_kernel_parts(16, 16.0, "long", r_s=2.0, fdtype=np.float32)
    delta = np.zeros((24, 24, 24), dtype=np.float32)
    with pytest.raises(ValueError, match="cache outlived the configuration"):
        coarse_force_meshes(jnp.asarray(delta), 24, 24.0, "long", r_s=2.0,
                            fdtype=np.float32, parts=parts)


def test_the_solve_writes_into_the_buffers_it_is_given():
    """`out=` buffers are returned by identity: `stage_step` skips its copy on `a is buf`,
    so a solve that allocated its own would leave workers reading a stale shm mesh."""
    from inexor.forces import coarse_force_meshes

    n, box = 16, 16.0
    rng = np.random.default_rng(5)
    delta = (rng.standard_normal((n, n, n)) * 1e-3).astype(np.float32)
    sinks = [np.zeros((n, n, n), dtype=np.float32) for _ in range(3)]
    got = coarse_force_meshes(jnp.asarray(delta), n, box, "long", r_s=2.0,
                              match=(1.0, 0.25), fdtype=np.float32, out=sinks)
    assert all(g is s for g, s in zip(got, sinks))
    assert any(np.any(s != 0) for s in sinks)


def test_the_parts_hold_only_real_half_grids():
    """`parts` holds 8 B per half-grid element (two real grids), not the 24 B of three
    complex kernels; a complex array here would silently triple the cost."""
    from inexor.forces import coarse_kernel_parts

    n = 32
    parts = coarse_kernel_parts(n, 32.0, "long", r_s=2.0, match=(1.0, 0.25),
                                fdtype=np.float32)
    half = n * n * (n // 2 + 1)
    kept = parts["pref"].nbytes + parts["mf"].nbytes
    assert not np.iscomplexobj(parts["pref"]) and not np.iscomplexobj(parts["mf"])
    assert kept == 8 * half, f"{kept / half:.2f} B/half held, expected 8.00"
    # ik grids must stay low-rank broadcasts, not three more full grids
    for a in parts["iks"]:
        assert a.size <= n, f"ik grid is full-rank ({a.size} elements)"


def test_the_monolithic_coarse_solve_refuses_the_silent_wrong_size(monkeypatch):
    """The coarse solve refuses a device transform above 2**31 elements (a 2048^3 mesh is
    4x that), but only off CPU: the bound is a cuFFT wrong-result limit. Both halves are
    pinned, since a guard firing everywhere is as wrong as one firing nowhere. The refusal is
    also pinned on a CPU host by stubbing `jax.default_backend` (read at call time).
    """
    import jax

    from inexor import forces, ooc_fft

    below = 1024  # measured correct on GPU (2.9e-6)
    at_or_above = 2048
    assert below**3 < ooc_fft.MAX_DEVICE_TRANSFORM_ELEMENTS <= at_or_above**3, (
        "the test sizes no longer bracket the bound")

    forces.refuse_oversize_coarse_solve(below)  # must not raise, any backend

    if jax.default_backend() == "cpu":
        forces.refuse_oversize_coarse_solve(at_or_above)  # exempt, must not raise
    else:
        with pytest.raises(ValueError, match="MEASURED to return a wrong result"):
            forces.refuse_oversize_coarse_solve(at_or_above)

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    forces.refuse_oversize_coarse_solve(below)  # below the bound: still passes on gpu
    with pytest.raises(ValueError, match="MEASURED to return a wrong result"):
        forces.refuse_oversize_coarse_solve(at_or_above)


def _coarse_parity_setup(fdtype, n=32, box=64.0, seed=5):
    from inexor import forces

    delta = np.random.default_rng(seed).standard_normal((n, n, n)).astype(fdtype)
    kw = dict(r_s=2.0)
    mono = forces.coarse_force_meshes(delta, n, box, "long", transform="monolithic", **kw)
    fact = forces.coarse_force_meshes(delta, n, box, "long", transform="factorized", **kw)
    return delta, mono, fact


@pytest.mark.parametrize("fdtype", [np.float32, np.float64])
def test_the_factorized_coarse_solve_agrees_with_monolithic_at_roundoff(fdtype):
    """Factorized and monolithic coarse solves agree to round-off, in units of eps.

    A different transform order rounds differently, so bitwise is unavailable; instead the
    disagreement must scale with eps (measured ~6 eps at f64, ~7 at f32). The 50 eps cap
    is an order above that and far below the ~1e-3 of a real Poisson-solve error.
    """
    import jax


    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", fdtype is np.float64)
    try:
        _, mono, fact = _coarse_parity_setup(fdtype)
        eps = float(np.finfo(fdtype).eps)
        for i in range(3):
            m, f = np.asarray(mono[i]), np.asarray(fact[i])
            assert f.dtype == np.dtype(fdtype), "the factorized solve changed precision"
            rel = float(np.max(np.abs(m - f)) / np.std(m))
            assert rel < 50 * eps, (
                f"component {i} disagrees by {rel:.3e} = {rel / eps:.1f} eps at "
                f"{np.dtype(fdtype).name}; roundoff between two factorizations is "
                "a few eps, so this is a difference in the computation")
        # anti-vacuity: the same metric fails on a single element perturbed by 100 eps x std
        m = np.asarray(mono[0])
        bad = np.array(fact[0], copy=True)
        bad.flat[bad.size // 2] += 100 * eps * float(np.std(m))
        rel_bad = float(np.max(np.abs(m - bad)) / np.std(m))
        assert rel_bad > 50 * eps, f"a 100 eps perturbation reads {rel_bad / eps:.1f} eps"
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_the_per_slab_kernel_is_bitwise_the_whole_grid_kernel():
    """Per-slab kernels concatenate bitwise to the whole-grid kernel.

    The kernel is elementwise `(pref * ik) * mf` with no reduction across x, so slicing
    is exact; this keeps the round-off gate above from hiding a per-slab kernel error.
    """
    from inexor import forces

    n, box, fdtype = 24, 48.0, np.float32
    parts = forces.coarse_kernel_parts(n, box, "long", r_s=2.0, match=(box / n, box / n),
                                       fdtype=fdtype)
    cdtype = np.complex64
    for axis in range(3):
        whole = forces.coarse_kernel_slab(parts, axis, 0, n, cdtype)
        for slab in (1, 5, 8, n):
            got = np.concatenate(
                [forces.coarse_kernel_slab(parts, axis, lo, min(lo + slab, n), cdtype)
                 for lo in range(0, n, slab)], axis=0)
            assert got.dtype == whole.dtype
            assert np.array_equal(got, whole), f"axis {axis} slab {slab} moved bits"


def test_the_factorized_solve_is_bitwise_invariant_to_slab_thickness():
    """Slab thickness is only a loop bound: it must not change a bit, or a checkpoint
    resumed with a different window would change the physics."""
    from inexor import forces

    n, box, fdtype = 32, 64.0, np.float32
    delta = np.random.default_rng(9).standard_normal((n, n, n)).astype(fdtype)
    ref = forces.coarse_force_meshes(delta, n, box, "long", r_s=2.0,
                                     transform="factorized", slab=n)
    for slab in (1, 7, 8):
        got = forces.coarse_force_meshes(delta, n, box, "long", r_s=2.0,
                                         transform="factorized", slab=slab)
        for i in range(3):
            assert np.array_equal(np.asarray(got[i]), np.asarray(ref[i])), (
                f"slab={slab} moved bits in component {i}")


def test_an_unknown_transform_is_refused():
    """A typo'd transform must not fall through to the monolithic default and be
    measured as if the factorized path had run."""
    from inexor import forces

    delta = np.zeros((8, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="transform must be"):
        forces.coarse_force_meshes(delta, 8, 16.0, "long", r_s=2.0, transform="ooc")


@pytest.mark.parametrize("fdtype", [np.float32, np.float64])
def test_the_hoisted_build_is_bitwise_on_the_factorized_path_too(fdtype, x64):
    """The hoist is bitwise on the default factorized path too, with the per-call build
    as oracle (the monolithic test above cannot cover this path)."""
    from inexor.forces import coarse_force_meshes, coarse_kernel_parts

    fdtype = np.dtype(fdtype)
    n, box = 24, 24.0
    r_s, match = 2.0 * box / n, (box / n, box / (4 * n))
    delta = (np.random.default_rng(11).standard_normal((n, n, n)) * 1e-3).astype(fdtype)

    kw = dict(r_s=r_s, match=match, fdtype=fdtype, transform="factorized")
    built_per_call = coarse_force_meshes(delta, n, box, "long", **kw)
    parts = coarse_kernel_parts(n, box, "long", r_s=r_s, match=match, fdtype=fdtype)
    hoisted = coarse_force_meshes(delta, n, box, "long", parts=parts, **kw)
    for i, (a, b) in enumerate(zip(built_per_call, hoisted)):
        assert np.array_equal(np.asarray(a), np.asarray(b)), (
            f"component {i} at n={n}/{fdtype.name}: hoisting the kernel build "
            "moved bits on the factorized path")
