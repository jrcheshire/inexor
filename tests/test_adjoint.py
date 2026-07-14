"""M2 S1: the exact-replay adjoint (custom_vjp) -- runs, shapes, driver
agreement, float-twin match, nested-grad composition, and the documented
outer-jit boundary. Fidelity GATES (measured floors, both losses, both IC
params) are S3 (scripts/m2_grad_gate.py); here we assert correctness at a
loose, comfortably-passing tolerance well inside the R4 7.5e-2 gate.
"""

import jax
import jax.numpy as jnp
import pytest

from inexor import PLANCK, BoxConfig, QuantConfig, TimeConfig
from inexor.adjoint import adjoint_grad_fnl, adjoint_grad_ic, evolve_grad
from inexor.ic import gaussian_delta, linear_density
from inexor.integrate import evolve_float
from inexor.losses import band_power_loss, field_l2_loss, fundamental_k_edges
from inexor.lpt import lpt_ics

BOX = BoxConfig(n_mesh=16, box_size=100.0)
QUANT = QuantConfig()
COSMO = PLANCK
INTEGRATORS = ("bullfrog", "fastpm", "exact")
DRIVERS = ("scan", "perstep")


def _ics(seed=0, fdtype=jnp.float32):
    key = jax.random.PRNGKey(seed)
    d0 = gaussian_delta(key, BOX.n_mesh, BOX.box_size, COSMO, fdtype=fdtype)
    x0, v0 = lpt_ics(d0, BOX.box_size, 0.1, COSMO, order=2, fdtype=fdtype)
    return x0.astype(fdtype), v0.astype(fdtype)


def _loss(x_f, v_f):
    return jnp.sum(x_f**2) + 0.5 * jnp.sum(v_f**2)


def _time(integrator, K=6):
    return TimeConfig(a_init=0.1, a_final=1.0, n_steps=K, integrator=integrator)


@pytest.mark.parametrize("integrator", INTEGRATORS)
@pytest.mark.parametrize("driver", DRIVERS)
def test_grad_runs_and_shapes(integrator, driver):
    time = _time(integrator)
    x0, v0 = _ics()

    def loss(x0_, v0_):
        return _loss(*evolve_grad(BOX, time, QUANT, COSMO, driver, jnp.float32, x0_, v0_))

    gx, gv = jax.grad(loss, argnums=(0, 1))(x0, v0)
    assert gx.shape == x0.shape and gv.shape == v0.shape
    assert bool(jnp.all(jnp.isfinite(gx))) and bool(jnp.all(jnp.isfinite(gv)))
    # a nonzero gradient (the map is not degenerate)
    assert float(jnp.linalg.norm(gx)) > 0 and float(jnp.linalg.norm(gv)) > 0


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_scan_perstep_grads_identical(integrator):
    """Both drivers produce a bit-identical integer trajectory (M1), so the
    residual -- and therefore the gradient -- is identical."""
    time = _time(integrator)
    x0, v0 = _ics()

    def grad_with(driver):
        def loss(x0_, v0_):
            return _loss(*evolve_grad(BOX, time, QUANT, COSMO, driver, jnp.float32, x0_, v0_))

        return jax.grad(loss, argnums=(0, 1))(x0, v0)

    gxs, gvs = grad_with("scan")
    gxp, gvp = grad_with("perstep")
    assert jnp.array_equal(gxs, gxp) and jnp.array_equal(gvs, gvp)


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_grad_matches_float_twin(integrator):
    """STE adjoint (through the quantized path) matches the smooth float-path
    gradient at O(quantization). Global metrics (single-component rel is noise-
    dominated at near-zero gradients -- see the S1 check)."""
    time = _time(integrator)
    x0, v0 = _ics()

    def loss_q(x0_, v0_):
        return _loss(*evolve_grad(BOX, time, QUANT, COSMO, "scan", jnp.float32, x0_, v0_))

    def loss_f(x0_, v0_):
        return _loss(*evolve_float(BOX, time, COSMO, x0_, v0_, paint="f32", fdtype=jnp.float32))

    gx, gv = jax.grad(loss_q, argnums=(0, 1))(x0, v0)
    gxf, gvf = jax.grad(loss_f, argnums=(0, 1))(x0, v0)
    for a, b in ((gx, gxf), (gv, gvf)):
        a, b = a.ravel(), b.ravel()
        ratio = jnp.linalg.norm(a) / jnp.linalg.norm(b)
        corr = jnp.corrcoef(a, b)[0, 1]
        assert abs(float(ratio) - 1.0) < 2e-2  # comfortably inside R4 7.5e-2
        assert float(corr) > 0.99


def test_composes_in_nested_grad():
    """evolve_grad nested inside a larger jax differentiation -- the concrete
    upgrade over mbody's eager-only manual adjoint entries."""
    time = _time("bullfrog")
    x0, v0 = _ics()

    def outer(scale):
        xf, vf = evolve_grad(BOX, time, QUANT, COSMO, "scan", jnp.float32, scale * x0, v0)
        return jnp.sum(xf**2)

    g = jax.grad(outer)(jnp.float32(1.0))
    assert bool(jnp.isfinite(g)) and float(jnp.abs(g)) > 0


def test_outer_jit_is_unsupported_boundary():
    """Documented boundary (shared with the forward evolve): an outer jax.jit
    over the orchestration hits the host-side float(max|v0|) ladder build.
    Pinned so it is not silently regressed or assumed away."""
    time = _time("bullfrog")
    x0, v0 = _ics()

    def loss(x0_, v0_):
        return _loss(*evolve_grad(BOX, time, QUANT, COSMO, "scan", jnp.float32, x0_, v0_))

    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.jit(jax.grad(loss))(x0, v0)


# ---------------------------------------------------------------------------
# S2: IC-parameter wrappers + differentiable losses
# ---------------------------------------------------------------------------

_KEDGES = fundamental_k_edges(BOX.n_mesh, BOX.box_size, n_bins=6)


def _bandpower_loss_field(x_f, v_f):
    # target 0 -> loss = mean(P^2), a smooth functional with nonzero IC grads
    return band_power_loss(x_f, BOX, _KEDGES, jnp.zeros(len(_KEDGES) - 1, dtype=x_f.dtype))


def _ic_grad_via_float(loss_field, integrator, theta):
    """The smooth float-path reference d(loss)/d[f_NL, amplitude]."""
    time = _time(integrator)

    def g(th):
        d0 = th[1] * linear_density(
            jax.random.PRNGKey(0), BOX.n_mesh, BOX.box_size, COSMO, f_NL=th[0], fdtype=jnp.float32
        )
        x0, v0 = lpt_ics(d0, BOX.box_size, time.a_init, COSMO, order=2, fdtype=jnp.float32)
        x_f, v_f = evolve_float(BOX, time, COSMO, x0, v0, paint="f32", fdtype=jnp.float32)
        return loss_field(x_f, v_f)

    return jax.grad(g)(jnp.asarray(theta, jnp.float32))


@pytest.mark.parametrize("integrator", ("bullfrog", "fastpm"))
def test_adjoint_grad_ic_matches_float_path(integrator):
    """d/d[f_NL, amplitude] through the quantized adjoint matches the smooth
    float-path gradient (both f32), comfortably inside the R4 7.5e-2 gate."""
    time = _time(integrator)
    theta = (5.0, 1.0)
    g_adj = adjoint_grad_ic(
        _bandpower_loss_field, BOX, time, QUANT, COSMO, theta=theta, driver="scan"
    )
    g_flt = _ic_grad_via_float(_bandpower_loss_field, integrator, theta)
    assert bool(jnp.all(jnp.isfinite(g_adj)))
    for i in range(2):
        rel = abs(float(g_adj[i]) - float(g_flt[i])) / (abs(float(g_flt[i])) + 1e-12)
        assert rel < 5e-2, (i, float(g_adj[i]), float(g_flt[i]))
        assert float(g_adj[i]) * float(g_flt[i]) > 0  # same sign


def test_adjoint_grad_fnl_equals_ic_component():
    """The scalar f_NL wrapper equals the f_NL component of the length-2 grad."""
    time = _time("bullfrog")
    g_fnl = adjoint_grad_fnl(_bandpower_loss_field, BOX, time, QUANT, COSMO, f_NL=5.0)
    g_ic = adjoint_grad_ic(_bandpower_loss_field, BOX, time, QUANT, COSMO, theta=(5.0, 1.0))
    assert float(g_fnl) == pytest.approx(float(g_ic[0]), rel=1e-6, abs=1e-8)


def test_field_l2_loss_differentiable():
    """A field-level L2 loss also yields a finite, nonzero IC gradient."""
    time = _time("bullfrog")
    key = jax.random.PRNGKey(1)
    d0 = linear_density(key, BOX.n_mesh, BOX.box_size, COSMO, f_NL=0.0, fdtype=jnp.float32)
    x_t, _ = lpt_ics(d0, BOX.box_size, time.a_init, COSMO, order=2, fdtype=jnp.float32)
    from inexor.losses import density_f32

    target = density_f32(x_t, BOX)
    g = adjoint_grad_ic(
        lambda xf, vf: field_l2_loss(xf, BOX, target), BOX, time, QUANT, COSMO, theta=(5.0, 1.0)
    )
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.linalg.norm(g)) > 0
