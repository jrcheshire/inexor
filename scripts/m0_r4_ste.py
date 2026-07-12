"""M0 R4 probe: STE gradient fidelity through quantized dynamics (roadmap R4).

THE existential probe (decisions.md D-002). Three gradient estimates of the
same scalar loss on the INT pipeline:

  g_STE   -- the mini-adjoint: bit-exact step_rev replay + per-step jax.vjp of
             the ste_round float twin (architecture.md Sec. 8 in miniature).
             Two drivers, cross-checked: eager per-step loop (primary) and the
             M2-shaped custom_vjp + lax.scan.
  g_FD    -- staircase-aware finite differences of the INT pipeline's decoded
             loss: eps sweep -> plateau validity -> least-squares regression
             slope with a standard error.
  g_float -- jax.grad of a never-quantized f32 BullFrog sim: NOT an oracle for
             the INT pipeline, but the q -> 0 anchor both must approach.

Quantization width swept WITHOUT dtype games: positions on a B-bit lattice via
imask_add (B in {16,14,12,10}; int32 storage), velocities via s_w0 * 2^m
(m in {0,2,4,6}). (B=16, m=0) is the production int16/uint16 pipeline and is
asserted bit-identical to it. Kill signature: gradient error NOT O(q) (log-log
slope p << 1) or growing uncontrolled with K -> pivot per D-002.

Run:  pixi run python scripts/m0_r4_ste.py [--quick] [--outdir runs/m0/r4]
CPU, f32 (x64 stays OFF). Every config starts by asserting bit-exact replay.
"""

# ruff: noqa: E402
import argparse
import json
import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _m0_common as mc

OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
F32 = jnp.float32


# ----------------------------------------------------------------------------
# problem setup
# ----------------------------------------------------------------------------


class Setup:
    """Fixed fiducial problem: base delta0, IC directions, losses, schedules."""

    def __init__(self, N, L, K, seed, cosmo=mc.PLANCK, a_init=0.1):
        self.N, self.L, self.K, self.cosmo, self.a_init = N, L, K, cosmo, a_init
        key = jax.random.PRNGKey(seed)
        kb, k1, k2, k3, kref = jax.random.split(key, 5)
        self.delta0_base = mc.linear_delta0(kb, N, L, cosmo)
        sig = float(jnp.std(self.delta0_base))
        # Colored unit-perturbation directions, normalized to rms(delta0).
        self.dirs = []
        for kk in (k1, k2, k3):
            u = mc.linear_delta0(kk, N, L, cosmo)
            self.dirs.append(u * (sig / float(jnp.std(u))))
        self.a_steps = mc.a_grid(a_init, 1.0, K, "log")
        lad_unit = mc.ladder_constants(self.a_steps, cosmo, 1.0, 1.0)
        self.P_final = lad_unit.P[-1]
        # Band-power mask, k in [0.1, 0.3] h/Mpc (hermitian weights).
        kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
        kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
        kk3 = np.sqrt(
            kx.reshape(N, 1, 1) ** 2 + kx.reshape(1, N, 1) ** 2 + kz.reshape(1, 1, -1) ** 2
        )
        wts = np.full(kk3.shape, 2.0)
        wts[:, :, 0] = 1.0
        if N % 2 == 0:
            wts[:, :, -1] = 1.0
        band = ((kk3 >= 0.1) & (kk3 <= 0.3)).astype(np.float64) * wts
        self.n_modes = float(band.sum())
        self._band = jnp.asarray(band.astype(np.float32))
        # Frozen reference field for the MSE loss: float run at A = 1.05.
        self.delta_ref = None  # filled by build_ref()

    def ic_fn(self, kind, theta, fdtype=F32):
        """theta -> (x0_phys, v0), differentiable. kind: 'amp' | ('dir', i)."""
        if kind == "amp":
            d0 = theta * self.delta0_base
        else:
            d0 = self.delta0_base + theta * self.dirs[kind[1]]
        return mc.za_ics(d0, self.N, self.L, self.a_init, self.cosmo, fdtype)

    def loss_band(self, x_phys):
        """Mean band power of the CIC delta of final positions (n_p == n_mesh)."""
        delta = mc.paint_f32(x_phys, self.N, self.L) - 1.0
        dk = jnp.fft.rfftn(delta)
        p3 = (dk.real**2 + dk.imag**2) * (self.L**3 / self.N**6)
        return jnp.sum(self._band * p3) / self.n_modes

    def loss_mse(self, x_phys):
        delta = mc.paint_f32(x_phys, self.N, self.L) - 1.0
        return jnp.mean((delta - self.delta_ref) ** 2)


# ----------------------------------------------------------------------------
# INT pipeline (B-bit positions via imask, int16 velocities via s_w0 * 2^m)
# ----------------------------------------------------------------------------


class IntPipe:
    """One quantization config: scan-jitted primal + eager and scan adjoints."""

    def __init__(self, su, B=16, m=0, c_growth=2.0):
        self.su, self.B, self.m = su, B, m
        self.s_x = su.L / 2.0**B
        s_pol = mc.s_w0_policy(self._vmax0(), su.P_final, c_growth=c_growth)
        self.s_w0 = s_pol * 2.0**m
        self.lad = mc.ladder_constants(su.a_steps, su.cosmo, self.s_w0, self.s_x)
        self.consts = mc.step_consts(self.lad)
        self.force = mc.make_force_fn(su.N, su.L, su.N**3)
        self._build()

    def _vmax0(self):
        x0, v0 = self.su.ic_fn("amp", 1.0)
        return float(jnp.max(jnp.abs(v0)))

    def _build(self):
        su, B, s_x = self.su, self.B, self.s_x
        # ONE compiled force executable serves every bit-sensitive path (FD
        # primal, replay, adjoint forward). Measured this session (macOS CPU):
        # two differently-compiled programs of the same step flip ~1 rint
        # half-tie per ~1e5 components/step -- the architecture.md Sec. 5
        # residual hazard, live even on CPU. Eager integer ops around a single
        # jitted force kill the whole hazard class inside this probe. (The
        # custom_vjp+scan driver below deliberately keeps its own compiled
        # programs -- it is the M2 structure under test, compared with
        # tolerance, and its tie-flip count is REPORTED as R1 evidence.)
        force = jax.jit(self.force)
        self.force_jit = force
        xmod = 2**B

        def step_fwd1(x, w, c):
            x1 = mc.imask_add(x, mc.rint_i(c.c1 * w.astype(F32)), B)
            g = force(x1.astype(F32) * s_x)
            w1 = mc.iadd(w, mc.rint_i(c.kappa * g))
            x2 = mc.imask_add(x1, mc.rint_i(c.c2 * w1.astype(F32)), B)
            return x2, w1

        def step_rev1(x2, w1, c):
            x1 = mc.imask_sub(x2, mc.rint_i(c.c2 * w1.astype(F32)), B)
            g = force(x1.astype(F32) * s_x)
            w = mc.isub(w1, mc.rint_i(c.kappa * g))
            x = mc.imask_sub(x1, mc.rint_i(c.c1 * w.astype(F32)), B)
            return x, w

        def run_fwd(x0i, w0i):
            x, w = x0i, w0i
            cs = self.consts
            for k in range(self.lad.n_steps):
                c = mc.StepConsts(cs.c1[k], cs.c2[k], cs.kappa[k])
                x, w = step_fwd1(x, w, c)
            return x, w

        self.run_fwd = run_fwd

        def step_float1(xf, wf, c):
            return mc.step_float(xf, wf, c, force, s_x, x_bits=B)

        @jax.jit
        def step_vjp(xf, wf, c, xbar, wbar):
            _, vjp = jax.vjp(lambda a, b: step_float1(a, b, c), xf, wf)
            return vjp((xbar, wbar))

        self.step_fwd1, self.step_rev1, self.step_vjp = step_fwd1, step_rev1, step_vjp

        def encode(x_phys, v0):
            x0i = mc.rint_i(x_phys / s_x) & (xmod - 1)
            w0i = mc.rint_i(v0 / self.s_w0).astype(jnp.int16)
            return x0i, w0i

        self.encode = encode

        def encode_ste(x_phys, v0):
            x0f = mc.ste_wrap_u(mc.ste_round(x_phys / s_x), float(xmod))
            w0f = mc.ste_round(v0 / self.s_w0)
            return x0f, w0f

        self.encode_ste = encode_ste

        # --- custom_vjp evolve (the M2 structure): its OWN compiled scan
        # programs for fwd and bwd, exactly the Sec. 8 skeleton. Compared to
        # the eager driver with tolerance; primal tie-flips reported.
        def scan_fwd(x0i, w0i, consts):
            def body(carry, c):
                x, w = carry
                x1 = mc.imask_add(x, mc.rint_i(c.c1 * w.astype(F32)), B)
                g = self.force(x1.astype(F32) * s_x)
                w1 = mc.iadd(w, mc.rint_i(c.kappa * g))
                x2 = mc.imask_add(x1, mc.rint_i(c.c2 * w1.astype(F32)), B)
                return (x2, w1), None

            (x, w), _ = lax.scan(body, (x0i, w0i), consts)
            return x, w

        self.scan_fwd = jax.jit(scan_fwd)

        @jax.custom_vjp
        def evolve(x0f, w0f, consts):
            x, w = scan_fwd(x0f.astype(jnp.int32), w0f.astype(jnp.int16), consts)
            return x.astype(F32), w.astype(F32)

        def evolve_fwd(x0f, w0f, consts):
            out = evolve(x0f, w0f, consts)
            return out, (out[0].astype(jnp.int32), out[1].astype(jnp.int16), consts)

        def evolve_bwd(res, cots):
            xN, wN, consts = res
            xbar, wbar = cots

            def body(carry, c):
                x, w, xb, wb = carry
                x1 = mc.imask_sub(x, mc.rint_i(c.c2 * w.astype(F32)), B)
                g = force(x1.astype(F32) * s_x)
                wp = mc.isub(w, mc.rint_i(c.kappa * g))
                xp = mc.imask_sub(x1, mc.rint_i(c.c1 * wp.astype(F32)), B)
                _, vjp = jax.vjp(
                    lambda a, b: step_float1(a, b, c), xp.astype(F32), wp.astype(F32)
                )
                xb, wb = vjp((xb, wb))
                return (xp, wp, xb, wb), None

            (x0, w0, xb0, wb0), _ = lax.scan(body, (xN, wN, xbar, wbar), consts, reverse=True)
            zero_c = jax.tree.map(jnp.zeros_like, consts)
            return xb0, wb0, zero_c

        evolve.defvjp(evolve_fwd, evolve_bwd)
        self.evolve = evolve

        # loss-of-theta primal (returns loss AND final ints for FD flips).
        # IC+encode and the loss are jitted; the steps run through run_fwd so
        # FD, replay, and the adjoint linearize on the IDENTICAL trajectory.
        def make_loss_of_theta(kind, loss_name):
            loss = su.loss_band if loss_name == "band" else su.loss_mse

            @jax.jit
            def ic_enc(theta):
                x_phys, v0 = su.ic_fn(kind, theta)
                return encode(x_phys, v0)

            loss_jit = jax.jit(lambda x: loss(x.astype(F32) * s_x))

            def f(theta):
                x0i, w0i = ic_enc(theta)
                x, w = run_fwd(x0i, w0i)
                return loss_jit(x), x, w

            return f

        self.make_loss_of_theta = make_loss_of_theta

    # --- gradients ---------------------------------------------------------

    def replay_assert(self, x0i, w0i):
        """Bit-exact replay gate: fwd K, rev K, exact int equality (house rule)."""
        x, w = x0i, w0i
        K = self.lad.n_steps
        cs = self.consts
        for k in range(K):
            c = mc.StepConsts(cs.c1[k], cs.c2[k], cs.kappa[k])
            x, w = self.step_fwd1(x, w, c)
        for k in reversed(range(K)):
            c = mc.StepConsts(cs.c1[k], cs.c2[k], cs.kappa[k])
            x, w = self.step_rev1(x, w, c)
        ok = bool(jnp.array_equal(x, x0i) and jnp.array_equal(w, w0i))
        if not ok:
            raise AssertionError("bit-exact replay FAILED -- adjoint linearization invalid")

    def field_grad_eager(self, kind, theta, loss_name):
        """Mini-adjoint, eager driver: (dL/dtheta, field cotangents, loss)."""
        su, s_x = self.su, self.s_x
        loss = su.loss_band if loss_name == "band" else su.loss_mse
        x_phys, v0 = su.ic_fn(kind, theta)
        x0i, w0i = self.encode(x_phys, v0)
        self.replay_assert(x0i, w0i)
        K, cs = self.lad.n_steps, self.consts
        x, w = x0i, w0i
        for k in range(K):
            c = mc.StepConsts(cs.c1[k], cs.c2[k], cs.kappa[k])
            x, w = self.step_fwd1(x, w, c)
        lval, xbar = jax.value_and_grad(lambda xl: loss(xl * s_x))(x.astype(F32))
        wbar = jnp.zeros_like(xbar)
        for k in reversed(range(K)):
            c = mc.StepConsts(cs.c1[k], cs.c2[k], cs.kappa[k])
            xp, wp = self.step_rev1(x, w, c)
            xbar, wbar = self.step_vjp(xp.astype(F32), wp.astype(F32), c, xbar, wbar)
            x, w = xp, wp
        # through the encode STE (identity Jacobian) to physical IC cotangents
        gx0 = xbar / s_x
        gv0 = wbar / self.s_w0
        # contract with the IC map to get dL/dtheta
        _, ic_vjp = jax.vjp(lambda t: su.ic_fn(kind, t), jnp.asarray(theta, F32))
        (gtheta,) = ic_vjp((gx0, gv0))
        return float(gtheta), (np.asarray(gx0), np.asarray(gv0)), float(lval)

    def grad_scan_driver(self, kind, theta, loss_name):
        """custom_vjp + lax.scan driver: dL/dtheta and lattice-unit field cots."""
        su, s_x = self.su, self.s_x
        loss = su.loss_band if loss_name == "band" else su.loss_mse

        def full(theta):
            x_phys, v0 = su.ic_fn(kind, theta)
            x0f, w0f = self.encode_ste(x_phys, v0)
            xN, wN = self.evolve(x0f, w0f, self.consts)
            return loss(xN * s_x)

        g = jax.grad(full)(jnp.asarray(theta, F32))
        return float(g)


# ----------------------------------------------------------------------------
# float anchor pipeline (never quantized)
# ----------------------------------------------------------------------------


class FloatPipe:
    def __init__(self, su):
        self.su = su
        self.force = mc.make_force_fn(su.N, su.L, su.N**3)
        lad = mc.ladder_constants(su.a_steps, su.cosmo, 1.0, 1.0)
        self.dD2 = jnp.asarray(0.5 * lad.dD, F32)
        self.alpha = jnp.asarray(lad.alphas, F32)
        self.bcoef = jnp.asarray(lad.betas / lad.D_mid, F32)

    def grad(self, kind, theta, loss_name):
        su, L = self.su, self.su.L
        force = self.force
        loss = su.loss_band if loss_name == "band" else su.loss_mse
        dD2, alpha, bcoef = self.dD2, self.alpha, self.bcoef

        @jax.jit
        def full(theta):
            x, v = su.ic_fn(kind, theta)

            def body(carry, cs):
                x, v = carry
                d2, al, bc = cs
                x = jnp.mod(x + d2 * v, L)
                g = force(x)
                v = al * v + bc * g
                x = jnp.mod(x + d2 * v, L)
                return (x, v), None

            (x, v), _ = lax.scan(body, (x, v), (dD2, alpha, bcoef))
            return loss(x)

        val, g = jax.value_and_grad(full)(jnp.asarray(theta, F32))
        gfield = jax.grad(lambda t: full(t))(jnp.asarray(theta, F32))  # scalar path only
        del gfield
        return float(g), float(val)

    def field_grad(self, kind, theta, loss_name):
        """Full IC-field gradient of the float pipeline (for cosine vs g_STE)."""
        su, L = self.su, self.su.L
        force = self.force
        loss = su.loss_band if loss_name == "band" else su.loss_mse
        dD2, alpha, bcoef = self.dD2, self.alpha, self.bcoef

        @jax.jit
        def full_xv(x0, v0):
            def body(carry, cs):
                x, v = carry
                d2, al, bc = cs
                x = jnp.mod(x + d2 * v, L)
                g = force(x)
                v = al * v + bc * g
                x = jnp.mod(x + d2 * v, L)
                return (x, v), None

            (x, v), _ = lax.scan(body, (x0, v0), (dD2, alpha, bcoef))
            return loss(x)

        x0, v0 = su.ic_fn(kind, theta)
        gx, gv = jax.grad(full_xv, argnums=(0, 1))(x0, v0)
        return np.asarray(gx), np.asarray(gv)


# ----------------------------------------------------------------------------
# staircase-aware finite differences
# ----------------------------------------------------------------------------


def fd_analysis(loss_of_theta, theta0, eps_list, loss_scale, n_reg=11):
    """Eps sweep -> plateau validity -> regression FD (slope, SE) or None."""
    sweep = []
    for eps in eps_list:
        lp, xp_, wp_ = loss_of_theta(theta0 + eps)
        lm, xm_, wm_ = loss_of_theta(theta0 - eps)
        flips = int(jnp.sum(xp_ != xm_)) + int(jnp.sum(wp_ != wm_))
        sweep.append(dict(
            eps=float(eps), g=float((lp - lm) / (2 * eps)), flips=flips,
            dL=float(abs(lp - lm)),
        ))
    # plateau: widest contiguous window, >= 1 decade, g within 10% of window
    # median, flips >= 1e4, dL >= 100 * f32 loss noise at both ends
    valid = [s for s in sweep]
    best = None
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            win = valid[i : j + 1]
            if win[-1]["eps"] / win[0]["eps"] < 10.0 - 1e-9:
                continue
            gs = np.array([s["g"] for s in win])
            med = np.median(gs)
            if med == 0 or np.max(np.abs(gs - med)) > 0.10 * abs(med):
                continue
            if win[0]["flips"] < 1e4 or win[0]["dL"] < 100 * 1e-7 * loss_scale:
                continue
            if best is None or win[-1]["eps"] / win[0]["eps"] > best[1]:
                best = ((i, j), win[-1]["eps"] / win[0]["eps"])
    if best is None:
        return dict(sweep=sweep, valid=False)
    i, j = best[0]
    E = sweep[j]["eps"]
    thetas = np.linspace(theta0 - E, theta0 + E, n_reg)
    ls = np.array([float(loss_of_theta(t)[0]) for t in thetas])
    A = np.vstack([thetas - theta0, np.ones_like(thetas)]).T
    coef, res, *_ = np.linalg.lstsq(A, ls, rcond=None)
    slope = coef[0]
    dof = max(n_reg - 2, 1)
    se = float(np.sqrt((res[0] if len(res) else 0.0) / dof / np.sum((thetas - theta0) ** 2)))
    return dict(sweep=sweep, valid=True, window=(sweep[i]["eps"], E), slope=float(slope), se=se)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def cosine(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default="runs/m0/r4")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    N = 32 if args.quick else 64
    L, K0 = 256.0, 8
    su = Setup(N, L, K0, args.seed)
    fp = FloatPipe(su)

    # frozen MSE reference: float run at A = 1.05
    x0, v0 = su.ic_fn("amp", 1.05)
    dD2, alpha, bcoef = fp.dD2, fp.alpha, fp.bcoef
    x, v = x0, v0
    for k in range(K0):
        x = jnp.mod(x + dD2[k] * v, L)
        g = fp.force(x)
        v = alpha[k] * v + bcoef[k] * g
        x = jnp.mod(x + dD2[k] * v, L)
    su.delta_ref = mc.paint_f32(x, N, L) - 1.0

    q_configs = [(16, 0), (14, 2), (12, 4), (10, 6)] if not args.quick else [(16, 0), (12, 4)]
    # K=2 is EXCLUDED: log-0.1 K=2 has alpha_1 = 0.018 and the ladder guard
    # (correctly) refuses it -- few-step early-start BullFrog schedules sit
    # near the alpha zero-crossing (n = D0/dD ~ 0.46). Gate-review note:
    # the int16+BullFrog ladder needs K >= 3 from a_i = 0.1.
    K_sweep = [3, 4, 12] if not args.quick else [4]
    params = [("amp", 1.0, "A"), (("dir", 0), 0.0, "u0"), (("dir", 1), 0.0, "u1")]
    if args.quick:
        params = params[:2]
    losses = ["band", "mse"] if not args.quick else ["band"]
    eps_list = np.geomspace(1e-6, 1e-1, 10)

    results = dict(config=dict(N=N, L=L, K0=K0, seed=args.seed), entries=[])
    t0 = time.time()

    # --- production-path bit-identity assert: imask B=16 pipeline == uint16/int16
    pipe0 = IntPipe(su, B=16, m=0)
    xph, v00 = su.ic_fn("amp", 1.0)
    x0i, w0i = pipe0.encode(xph, v00)
    xu, wu = mc.encode_x(xph, pipe0.s_x), mc.encode_w(v00, pipe0.s_w0)
    cs = pipe0.consts
    xa, wa = x0i, w0i
    xb, wb = xu, wu
    for k in range(K0):
        c = mc.StepConsts(cs.c1[k], cs.c2[k], cs.kappa[k])
        xa, wa = pipe0.step_fwd1(xa, wa, c)
        # same single compiled force executable on both paths (like vs like)
        xb, wb = mc.step_fwd(xb, wb, c, pipe0.force_jit, pipe0.s_x)
    same = bool(
        jnp.array_equal(xa, xb.astype(jnp.int32)) and jnp.array_equal(wa, wb.astype(jnp.int16))
    )
    print(f"[gate] imask(B=16) trajectory == uint16/int16 production path: "
          f"{'PASS' if same else 'FAIL'}")
    if not same:
        raise SystemExit("production-path bit-identity failed; fix before any verdict")

    # --- q sweep at K = K0 (+ K sweep at production q)
    pipes = {}
    for B, m in q_configs:
        pipes[(B, m, K0)] = pipe0 if (B, m) == (16, 0) else IntPipe(su, B=B, m=m)
    for K in K_sweep:
        su_k = Setup(N, L, K, args.seed)
        su_k.delta_ref = su.delta_ref
        pipes[(16, 0, K)] = IntPipe(su_k, B=16, m=0)

    # scan-program vs per-step-program primal comparison: rint tie-flips
    # between two compilations of the same step (CPU evidence for R1's
    # driver question; architecture.md Sec. 5 residual hazard).
    tie_flips = {}
    for (B, m, K), pipe in pipes.items():
        xph_t, v_t = pipe.su.ic_fn("amp", 1.0)
        x0t, w0t = pipe.encode(xph_t, v_t)
        xs, ws = pipe.scan_fwd(x0t, w0t, pipe.consts)
        xp_, wp_ = pipe.run_fwd(x0t, w0t)
        tie_flips[f"B{B}_m{m}_K{K}"] = int(jnp.sum(xs != xp_)) + int(jnp.sum(ws != wp_))
    results["scan_vs_perstep_tie_flips"] = tie_flips
    ncomp = 2 * 3 * N**3
    print(f"[info] scan-vs-perstep primal tie-flips per {ncomp} components: {tie_flips}")

    for (B, m, K), pipe in pipes.items():
        suk = pipe.su
        for loss_name in losses:
            for kind, th0, pname in params:
                tt = time.time()
                lot = pipe.make_loss_of_theta(kind, loss_name)
                lval = float(lot(jnp.asarray(th0, F32))[0])
                g_ste, (gx0, gv0), _ = pipe.field_grad_eager(kind, th0, loss_name)
                g_scan = pipe.grad_scan_driver(kind, th0, loss_name)
                g_flt, _ = (FloatPipe(suk) if K != K0 else fp).grad(kind, th0, loss_name) \
                    if K != K0 else fp.grad(kind, th0, loss_name)
                fd = fd_analysis(lot, th0, eps_list, abs(lval))
                # cosine of field gradients vs the float anchor (production q only)
                cos = None
                if (B, m) == (16, 0):
                    fgx, fgv = (FloatPipe(suk) if K != K0 else fp).field_grad(
                        kind, th0, loss_name
                    )
                    cos = cosine(np.concatenate([gx0.ravel(), gv0.ravel()]),
                                 np.concatenate([fgx.ravel(), fgv.ravel()]))
                ent = dict(
                    B=B, m=m, K=K, loss=loss_name, param=pname, loss_val=lval,
                    q_rel=2.0 ** (16 - B),
                    g_ste=g_ste, g_scan=g_scan, g_float=g_flt,
                    driver_rel_diff=abs(g_ste - g_scan) / max(abs(g_ste), 1e-30),
                    fd_valid=fd["valid"],
                    g_fd=fd.get("slope"), fd_se=fd.get("se"),
                    fd_window=fd.get("window"), fd_sweep=fd["sweep"],
                    cos_vs_float=cos,
                    wall_s=time.time() - tt,
                )
                results["entries"].append(ent)
                fdtxt = (f"g_FD={fd['slope']:+.4e}+-{fd['se']:.1e}" if fd["valid"]
                         else "FD INVALID (no plateau)")
                print(f"  [B={B} m={m} K={K:2d} {loss_name:4s} {pname:3s}] "
                      f"g_STE={g_ste:+.4e} g_scan={g_scan:+.4e} g_flt={g_flt:+.4e} "
                      f"{fdtxt} cos={cos if cos is None else round(cos, 6)} "
                      f"({ent['wall_s']:.0f}s)")

    # ------------------------------------------------------------------
    # verdict: O(q) slope + int16 relative error + K trend
    # ------------------------------------------------------------------
    verdict = {}
    for loss_name in losses:
        for _, _, pname in params:
            es, qs = [], []
            for B, m in q_configs:
                e = [x for x in results["entries"]
                     if x["B"] == B and x["m"] == m and x["K"] == K0
                     and x["loss"] == loss_name and x["param"] == pname]
                if e and e[0]["fd_valid"]:
                    err = abs(e[0]["g_ste"] - e[0]["g_fd"])
                    if err > 3 * e[0]["fd_se"]:  # below FD SE = indistinguishable
                        es.append(err)
                        qs.append(e[0]["q_rel"])
            slope = None
            if len(es) >= 2:
                slope = float(np.polyfit(np.log(qs), np.log(es), 1)[0])
            # anchor-based O(q) slope: |g_STE - g_float| is measurable at EVERY
            # q (FD-based e(q) starves: sub-SE at fine q, no plateau at coarse
            # q). Conflates STE error with the physical quantized-vs-float
            # difference, but both vanish as q -> 0 and the kill question is
            # "does gradient error scale with q" -- this answers it robustly.
            qa, ea = [], []
            for B, m in q_configs:
                e = [x for x in results["entries"]
                     if x["B"] == B and x["m"] == m and x["K"] == K0
                     and x["loss"] == loss_name and x["param"] == pname]
                if e and abs(e[0]["g_ste"] - e[0]["g_float"]) > 0:
                    qa.append(e[0]["q_rel"])
                    ea.append(abs(e[0]["g_ste"] - e[0]["g_float"]))
            slope_anchor = (float(np.polyfit(np.log(qa), np.log(ea), 1)[0])
                            if len(ea) >= 3 else None)
            e16 = [x for x in results["entries"]
                   if x["B"] == 16 and x["m"] == 0 and x["K"] == K0
                   and x["loss"] == loss_name and x["param"] == pname]
            rel16 = None
            if e16 and e16[0]["fd_valid"]:
                rel16 = abs(e16[0]["g_ste"] - e16[0]["g_fd"]) / abs(e16[0]["g_fd"])
                sig16 = abs(e16[0]["g_ste"] - e16[0]["g_fd"]) / max(e16[0]["fd_se"], 1e-30)
            else:
                sig16 = None
            verdict[f"{loss_name}/{pname}"] = dict(
                oq_slope=slope, oq_slope_anchor=slope_anchor,
                rel_err_int16=rel16, nsigma_int16=sig16
            )
    results["verdict"] = verdict

    with open(out / "r4_results.json", "w") as fh:
        json.dump(results, fh, indent=1)
    make_figures(results, out, q_configs, K_sweep + [K0], losses, params)

    print("\nR4 verdict inputs (threshold = measure-then-negotiate with JC):")
    for k, v in verdict.items():
        print(f"  {k}: O(q) slope p = {v['oq_slope']} "
              f"(anchor-based p = {v['oq_slope_anchor']}), "
              f"rel err @ int16 = {v['rel_err_int16']}, "
              f"nsigma vs FD SE = {v['nsigma_int16']}")
    print(f"total wall: {(time.time() - t0) / 60:.1f} min; outputs: {out.resolve()}")


def make_figures(results, out, q_configs, Ks, losses, params):
    ents = results["entries"]
    # Fig 1: FD staircase diagnostics at production q (per param, band loss)
    fig, axes = plt.subplots(1, len(params), figsize=(4.2 * len(params), 3.6), squeeze=False)
    for j, (_, _, pname) in enumerate(params):
        ax = axes[0][j]
        e = [x for x in ents if x["B"] == 16 and x["m"] == 0 and x["K"] == results["config"]["K0"]
             and x["loss"] == "band" and x["param"] == pname]
        if e:
            sw = e[0]["fd_sweep"]
            ax.semilogx([s["eps"] for s in sw], [s["g"] for s in sw], "o-", color=OI[0], ms=4)
            if e[0]["fd_valid"]:
                ax.axvspan(*e[0]["fd_window"], color=OI[2], alpha=0.15)
                ax.axhline(e[0]["g_fd"], color=OI[2], lw=1)
            ax.axhline(e[0]["g_ste"], color=OI[1], lw=1, ls="--")
            ax.axhline(e[0]["g_float"], color="gray", lw=1, ls=":")
        ax.set_xlabel("eps")
        ax.set_title(f"param {pname} (band)", fontsize=9)
        if j == 0:
            ax.set_ylabel("central FD  g(eps)")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "r4_fig1_fd_staircase.png", dpi=150)
    plt.close(fig)

    # Fig 2: e(q) vs q log-log (headline)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for li, loss_name in enumerate(losses):
        for pi, (_, _, pname) in enumerate(params):
            qs, es = [], []
            for B, m in q_configs:
                e = [x for x in ents if x["B"] == B and x["m"] == m
                     and x["K"] == results["config"]["K0"]
                     and x["loss"] == loss_name and x["param"] == pname]
                if e and e[0]["fd_valid"]:
                    qs.append(e[0]["q_rel"])
                    es.append(abs(e[0]["g_ste"] - e[0]["g_fd"]) / abs(e[0]["g_fd"]))
            if qs:
                ax.loglog(qs, es, marker="o", color=OI[pi % len(OI)],
                          ls="-" if loss_name == "band" else "--",
                          label=f"{loss_name}/{pname}")
    qq = np.array([1, 64.0])
    ax.loglog(qq, 1e-3 * qq, "k:", lw=1, label="slope 1 guide")
    ax.set_xlabel("quantization step / production int16 step")
    ax.set_ylabel("|g_STE - g_FD| / |g_FD|")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "r4_fig2_oq_scaling.png", dpi=150)
    plt.close(fig)

    # Fig 3: error and cosine vs K at production q
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    for pi, (_, _, pname) in enumerate(params):
        Ks_s, rels, coss = [], [], []
        for K in sorted(set(Ks)):
            e = [x for x in ents if x["B"] == 16 and x["m"] == 0 and x["K"] == K
                 and x["loss"] == "band" and x["param"] == pname]
            if e and e[0]["fd_valid"]:
                Ks_s.append(K)
                rels.append(abs(e[0]["g_ste"] - e[0]["g_fd"]) / abs(e[0]["g_fd"]))
                coss.append(e[0]["cos_vs_float"])
        if Ks_s:
            axes[0].semilogy(Ks_s, rels, "o-", color=OI[pi % len(OI)], label=pname)
            axes[1].plot(Ks_s, [1 - c for c in coss], "o-", color=OI[pi % len(OI)])
    axes[0].set_ylabel("|g_STE - g_FD| / |g_FD|")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("1 - cos(g_STE, g_float) [fields]")
    axes[1].set_xlabel("K (steps)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "r4_fig3_vs_K.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
