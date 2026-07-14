"""NOVEL CORE 2: the discrete exact-replay adjoint (architecture.md Sec. 8).

`evolve_grad` is a genuine `jax.custom_vjp` around the quantized forward map:
physical floats in, physical floats out, integer state internal. Its reverse
rule is the exact transpose of the discrete forward map inexor actually
executed (D-003, discretise-then-optimise -- NOT a continuous/Diffrax
backsolve). The reverse pass is mbody's `_reversible_ic_grad` five-stage replay
skeleton, but the per-step `one_step`/`reverse_step` are inexor's integer
kernels and the VJP is taken through the STE float twin, all in LATTICE UNITS:

  fwd residual = FINAL integer state only (O(1) in steps -- reversibility IS the
    checkpointing; no cross-step jax.checkpoint).
  per reverse step (scan reverse=True):
    1. x_prev, w_prev = step_rev(x, w, c, force_int, s_x)   # bit-exact replay
    2. xf, wf = x_prev.astype(f), w_prev.astype(f)          # lattice-unit floats
    3. (xbar, wbar) = vjp(step_float(., ., c, force_f32))   # STE twin, at x_prev
    4. within-step jax.checkpoint around the paint/read (the FFT solve is
       linear -> no saved activations).

Paint decision (D-006, finalized): the primal/replay path uses the
deterministic paint="int" force; the backward VJP uses the differentiable
paint="f32" force (the int paint needs no VJP rule).

Gradient scope (STE, architecture.md Sec. 8): gradients flow ONLY through the
step_float dynamics and the linear encode/decode scales. The w-frame scale s_w0
is a quantization hyperparameter (a policy from max|v0|); its dependence on the
inputs is a discretization effect the STE deliberately drops, so the whole
ladder/consts bundle is a stop-gradient constant w.r.t. the differentiation.
This makes `evolve_grad` an eager orchestrator (like `evolve`): the host-side
float64 ladder build runs on concrete inputs. `jax.grad(loss . evolve_grad)`
composes around it -- including nested inside a larger jax differentiation --
which is the concrete upgrade over mbody's eager-only manual `adjoint_grad_*`
entry points (those cannot be nested in an outer jax.grad at all). Boundary
(shared with the forward `evolve`): an OUTER `jax.jit` over the orchestration
is NOT supported -- the s_w0 policy needs a concrete `float(max|v0|)`, so under
jit v0 is a tracer and setup raises ConcretizationTypeError. The jitted units
are the internal drivers (run_scan's lax.scan; run_perstep's per-step jit); the
top-level orchestration stays eager by design.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .codec import dequant_w, dequant_x, encode_w, encode_x
from .forces import make_force_fn
from .integrate import (
    _bullfrog_setup,
    _G_f,
    _kdk_setup,
    p_to_v,
    run_perstep,
    run_scan,
    step_float,
    step_fwd,
    step_kdk_float,
    step_kdk_fwd,
    step_kdk_rev,
    step_rev,
    v_to_p,
)


def _setup(box, time, quant, cosmo, x0_f, v0_f, fdtype):
    """Host-side ladder build + encode (concrete inputs). Returns everything the
    forward driver and the reverse sweep need. Mirrors evolve's setup so the
    residuals are exactly the forward state, not a recomputation with drift."""
    frac_bits = quant.frac_bits
    force_int = make_force_fn(box, fdtype=fdtype, paint="int", frac_bits=frac_bits)
    if time.integrator == "bullfrog":
        v_max0 = float(jnp.max(jnp.abs(v0_f)))
        ladder, consts = _bullfrog_setup(box, time, quant, cosmo, v_max0, fdtype)
        s_w0 = ladder.s_w0
        s_out = s_w0 * ladder.P[-1]  # decode scale for v_D at a_final
        x_i, w_i = encode_x(x0_f, box.s_x), encode_w(v0_f, s_w0)
        kernels = (step_fwd, step_rev, step_float)
    else:  # fastpm / exact: fixed-scale w = p / s_w0
        p0 = v_to_p(v0_f, time.a_init, cosmo)
        p_max0 = float(jnp.max(jnp.abs(p0)))
        _, consts, s_w0, _ = _kdk_setup(box, time, quant, cosmo, p_max0, fdtype)
        s_out = s_w0
        x_i, w_i = encode_x(x0_f, box.s_x), encode_w(p0, s_w0)
        kernels = (step_kdk_fwd, step_kdk_rev, step_kdk_float)
    return force_int, consts, s_w0, s_out, x_i, w_i, kernels


def _decode(box, time, cosmo, x, w, s_out, fdtype):
    """Integer final state -> physical (x_f, v_f). Matches evolve exactly."""
    x_f = dequant_x(x, box.s_x, fdtype)
    v_f = dequant_w(w, s_out, fdtype)
    if time.integrator != "bullfrog":
        v_f = p_to_v(v_f, time.a_final, cosmo)
    return x_f, v_f


def _forward(box, time, quant, cosmo, x0_f, v0_f, driver, fdtype):
    """Shared forward: encode -> drive (paint=int) -> decode. Returns
    ((x_f, v_f), residual) where residual carries the FINAL integer state +
    consts + scales for the reverse sweep."""
    force_int, consts, s_w0, s_out, x_i, w_i, kernels = _setup(
        box, time, quant, cosmo, x0_f, v0_f, fdtype
    )
    step_f = kernels[0]
    # copy: run_perstep donates its inputs; the encoded ICs must survive if a
    # caller reuses them (the R1 GPU lesson, integrate.replay_roundtrip).
    if driver == "scan":
        x, w = run_scan(x_i.copy(), w_i.copy(), consts, step_f, force_int, box.s_x, fdtype=fdtype)
    elif driver == "perstep":
        x, w = run_perstep(x_i.copy(), w_i.copy(), consts, step_f, force_int, box.s_x, fdtype=fdtype)
    else:
        raise ValueError(f"driver must be 'scan' or 'perstep', got {driver!r}")
    x_f, v_f = _decode(box, time, cosmo, x, w, s_out, fdtype)
    residual = (x, w, consts, s_w0, s_out)
    return (x_f, v_f), residual


# ============================================================================
# The custom_vjp
# ============================================================================


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4, 5))
def evolve_grad(box, time, quant, cosmo, driver, fdtype, x0_f, v0_f):
    """Differentiable quantized evolution: d(x_f, v_f)/d(x0_f, v0_f) via the
    exact-replay adjoint. Floats in, floats out; integer state internal. The
    static args (box/time/quant/cosmo/driver/fdtype) are nondifferentiable.

    x0_f: physical positions (n, 3) in [0, L); v0_f: D-time velocities dx/dD
    (lpt.lpt_ics convention). driver: "scan" (test/small) | "perstep".
    """
    out, _ = _forward(box, time, quant, cosmo, x0_f, v0_f, driver, fdtype)
    return out


def _evolve_grad_fwd(box, time, quant, cosmo, driver, fdtype, x0_f, v0_f):
    return _forward(box, time, quant, cosmo, x0_f, v0_f, driver, fdtype)


def _evolve_grad_bwd(box, time, quant, cosmo, driver, fdtype, residual, cot):
    x, w, consts, s_w0, s_out = residual
    x_f_bar, v_f_bar = cot
    force_int = make_force_fn(box, fdtype=fdtype, paint="int", frac_bits=quant.frac_bits)
    force_f32 = make_force_fn(box, fdtype=fdtype, paint="f32", frac_bits=quant.frac_bits)
    s_x = box.s_x

    # Seed lattice cotangents at the DECODE boundary (STE: rounding = identity):
    #   x_f = x_lat * s_x                     -> x_lat_bar = s_x * x_f_bar
    #   bullfrog: v_f = w_lat * s_out         -> w_lat_bar = s_out * v_f_bar
    #   kdk:      v_f = w_lat * s_w0 / G_f(a_f) -> w_lat_bar = (s_w0/G_f) * v_f_bar
    xbar = s_x * x_f_bar
    if time.integrator == "bullfrog":
        wbar = s_out * v_f_bar
        step_r, step_flt = step_rev, step_float
    else:
        gf_final = _G_f(time.a_final, cosmo)
        wbar = (s_w0 / gf_final) * v_f_bar
        step_r, step_flt = step_kdk_rev, step_kdk_float

    def rev_body(carry, c):
        xc, wc, xb, wb = carry
        x_prev, w_prev = step_r(xc, wc, c, force_int, s_x, fdtype=fdtype)  # bit-exact replay
        xf = x_prev.astype(fdtype)  # lattice-unit floats (step_float applies s_x)
        wf = w_prev.astype(fdtype)
        # STE twin linearized at the replayed state; jax.checkpoint rematerializes
        # the per-step paint/read activations during its VJP (~18% peak saving).
        twin = jax.checkpoint(lambda a, b: step_flt(a, b, c, force_f32, s_x, fdtype=fdtype))
        _, vjp = jax.vjp(twin, xf, wf)
        xb, wb = vjp((xb, wb))
        return (x_prev, w_prev, xb, wb), None

    (_, _, xbar_ic, wbar_ic), _ = jax.lax.scan(
        rev_body, (x, w, xbar, wbar), consts, reverse=True
    )

    # Unwind the ENCODE boundary (STE: rounding = identity):
    #   x_i_lat = x0_f / s_x                  -> x0_bar = x_lat_bar / s_x
    #   bullfrog: w_i_lat = v0_f / s_w0       -> v0_bar = w_lat_bar / s_w0
    #   kdk:      w_i_lat = v0_f * G_f(a_i)/s_w0 -> v0_bar = w_lat_bar * G_f(a_i)/s_w0
    x0_bar = xbar_ic / s_x
    if time.integrator == "bullfrog":
        v0_bar = wbar_ic / s_w0
    else:
        gf_init = _G_f(time.a_init, cosmo)
        v0_bar = wbar_ic * (gf_init / s_w0)
    return x0_bar, v0_bar


evolve_grad.defvjp(_evolve_grad_fwd, _evolve_grad_bwd)
