"""G3 Stage 3 -- the three floors, measured on MONOLITHIC PAIRS ONLY.

THE ORDERING CONSTRAINT IS THE POINT. D-v2-7's 15% bar is kept, but the ESTIMAND
(which triangles, max vs median, which denominator, the conditioning cut) is
pinned only after these floors are known -- and the floors are measured with no
tiled arm in existence, so the choice cannot leak the answer. Nothing in this
script may import or construct an sCOLA tile. "Arm B" is always a monolithic
field perturbed by a KNOWN field-level operation, which is what makes every
number here checkable against a closed form.

THREE FLOORS, NEVER CONFLATED:

  A instrument      what a null perturbation reads -- bit-repro pair, a
                    translated monolithic field, f64 vs f32.
  B resolving power seed-to-seed sigma of the RATIO at shared ICs. The number
                    that decides whether 15% is a measurement or a bound.
  C cosmic variance seed-to-seed sigma of B_mono itself. Context only.

The cancellation factor sigma_C / sigma_B is the headline: G6 measured 100-500x
for P(k) because a shared-IC ratio cancels cosmic variance. NEAR 1 MEANS THE TWO
ARMS ARE NOT SHARING LONG MODES -- an IC plumbing bug, separately confirmable
via r(k_long) from cross_r, which must be ~1.

THE STATISTICS.

  R_B = B_t / B_m - 1                      headline, but contaminated: D-v2-9
                                           already gates a P(k) error, and 2%
                                           low in P is 4-6% low in B with no
                                           coupling failure at all.
  R_Q = Q_t / Q_m - 1,  Q = B/(P1P2+P2P3+P3P1)     the gate statistic.
  W   = B[eps, m, m] / B[m, m, m]          the mechanism discriminator, with
                                           eps = delta_t - T(k) delta_m the
                                           residual after removing the measured
                                           per-shell transfer, placed on the
                                           LONG leg.

W IS ZERO ONLY UP TO WITHIN-SHELL VARIATION OF THE WINDOW, and that is a floor
this script measures rather than asserts. T is estimated per SHELL (a per-MODE
estimate would make eps identically zero by construction and W vacuous), so a
window varying inside a shell leaves a residual that W sees. Narrow shells
shrink it; section D reports its size so no later W is read below its own floor.

Usage:
    pixi run python scripts/v2_g3_floors.py --config smoke --sections DA
    pixi run python scripts/v2_g3_floors.py --config cdev8 --nseed 24
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from v2_g5_two_level_force import A_CONTROL, A_PIVOT, N_STEPS, geometry  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(HERE), "runs", "v2")

# Squeezed configurations: long leg at a few times the box fundamental, short
# legs fixed. Plus an equilateral control at the same k_short -- if tiling (or
# any perturbation) hurts both equally the error is generic, not a squeezed
# coupling failure, and the whole squeezed framing is wrong.
# Defaults are the cdev8 exploration set. The GATE set is pinned at cdev, where
# x = m * P / n_fine with P = 96 at the Stage 5 geometry T=64/b=16 puts x = 1 at
# m = 5.33, so m = 1..7 straddles the threshold with five points below and two
# above. Overridable so one script serves both configs and the deneb job states
# its own set rather than inheriting a constant tuned for another box.
LONG_MULTS = (2, 3, 4, 6)
K_SHORT_MULT = 12


def _triangles(kf, long_mults=None, k_short_mult=None):
    lm = LONG_MULTS if long_mults is None else tuple(long_mults)
    ks = K_SHORT_MULT if k_short_mult is None else int(k_short_mult)
    squeezed = [(m * kf, ks * kf, ks * kf) for m in lm]
    equilateral = [(ks * kf, ks * kf, ks * kf)]
    return squeezed + equilateral, ["sq%d" % m for m in lm] + ["equi"]


# ===========================================================================
# statistics
# ===========================================================================


def shell_transfer(delta_t, delta_m, box_size, centers, dk):
    """Per-shell transfer T(k) = P_tm / P_mm, and the correlation r(k).

    T is what a deterministic window would be; r is the decorrelation check the
    cancellation factor's interpretation depends on (r ~ 1 at k_long means the
    arms genuinely share long modes).
    """
    from inexor.diagnostics import _k_grid, _shell_mask

    n = delta_m.shape[0]
    tk = np.fft.rfftn(delta_t)
    mk = np.fft.rfftn(delta_m)
    _, _, k_mag = _k_grid(n, box_size)
    t_out = np.empty(len(centers))
    r_out = np.empty(len(centers))
    for i, c in enumerate(centers):
        msk = _shell_mask(k_mag, c - 0.5 * dk, c + 0.5 * dk)
        p_tm = float(np.real(tk[msk] * np.conj(mk[msk])).sum())
        p_mm = float((np.abs(mk[msk]) ** 2).sum())
        p_tt = float((np.abs(tk[msk]) ** 2).sum())
        t_out[i] = p_tm / p_mm if p_mm > 0 else np.nan
        r_out[i] = p_tm / np.sqrt(p_tt * p_mm) if p_tt * p_mm > 0 else np.nan
    return t_out, r_out


def residual_field(delta_t, delta_m, box_size, centers, dk):
    """eps = delta_t - T(k) delta_m, with T piecewise constant on the shells.

    Modes outside every shell keep delta_t - delta_m; they never enter a
    bispectrum leg, so their treatment is cosmetic, but leaving them at zero
    would misrepresent eps if it were ever power-spectrum-ed.
    """
    from inexor.diagnostics import _k_grid, _shell_mask

    n = delta_m.shape[0]
    tk = np.fft.rfftn(delta_t)
    mk = np.fft.rfftn(delta_m)
    _, _, k_mag = _k_grid(n, box_size)
    t_shell, _ = shell_transfer(delta_t, delta_m, box_size, centers, dk)
    ek = tk - mk
    for c, t in zip(centers, t_shell):
        msk = _shell_mask(k_mag, c - 0.5 * dk, c + 0.5 * dk)
        ek[msk] = tk[msk] - t * mk[msk]
    return np.fft.irfftn(ek, s=(n, n, n), axes=(0, 1, 2))


def stats(delta_t, delta_m, box_size, tris, dk=None):
    """(R_B, R_Q, W, B_m, B_t, n_tri) for one arm pair on one triangle set."""
    from inexor.diagnostics import band_power, bispectrum

    kf = 2.0 * np.pi / box_size
    w = kf if dk is None else dk
    centers = sorted({float(k) for tri in tris for k in tri})

    b_m, n_tri = bispectrum(delta_m, box_size, tris, dk=dk)
    b_t, _ = bispectrum(delta_t, box_size, tris, dk=dk)
    p_m, _ = band_power(delta_m, box_size, centers, dk=w)
    p_t, _ = band_power(delta_t, box_size, centers, dk=w)
    pm = dict(zip(centers, p_m))
    pt = dict(zip(centers, p_t))

    def qdenom(p, tri):
        a, b, c = (p[float(k)] for k in tri)
        return a * b + b * c + c * a

    q_m = np.array([b_m[i] / qdenom(pm, t) for i, t in enumerate(tris)])
    q_t = np.array([b_t[i] / qdenom(pt, t) for i, t in enumerate(tris)])

    eps = residual_field(delta_t, delta_m, box_size, centers, w)
    b_emm, _ = bispectrum((eps, delta_m, delta_m), box_size, tris, dk=dk)

    # rho: the window-DIVIDED ratio. R_Q was expected to cancel a deterministic
    # window and MEASURABLY DOES NOT (it responds as 1/T for a uniform window --
    # the same reason tree-level Q ~ 1/b under linear bias). rho divides the
    # measured per-shell transfer out explicitly, so it is the estimand that is
    # window-invariant by construction. Reported here so the checkpoint can
    # choose between them on measured window-response rather than on intent.
    t_shell, _ = shell_transfer(delta_t, delta_m, box_size, centers, w)
    ts = dict(zip(centers, t_shell))
    t_prod = np.array([ts[float(t[0])] * ts[float(t[1])] * ts[float(t[2])] for t in tris])

    return dict(
        R_B=b_t / b_m - 1.0,
        R_Q=q_t / q_m - 1.0,
        rho=(b_t / b_m) / t_prod - 1.0,
        T_prod=t_prod,
        W=b_emm / b_m,
        B_m=b_m,
        B_t=b_t,
        n_tri=n_tri,
        P_m=p_m,
        centers=np.asarray(centers),
    )


def gaussian_sigma_B(b_m, n_tri, p_m, centers, tris, box_size):
    """Analytic Gaussian sigma(B) = sqrt(s123 * V * P1 P2 P3 / n_tri).

    s123 = 6 equilateral, 2 isoceles, 1 scalene. Reported ALONGSIDE the measured
    scatter because the two must DIFFER a lot for squeezed configurations and
    agree for equilaterals: if they agree for squeezed triangles the estimator
    is not seeing squeezed physics, only Gaussian mode counting.
    """
    pm = dict(zip([float(c) for c in centers], p_m))
    out = np.empty(len(tris))
    for i, tri in enumerate(tris):
        ks = [float(k) for k in tri]
        uniq = len(set(ks))
        s123 = 6.0 if uniq == 1 else (2.0 if uniq == 2 else 1.0)
        prod = pm[ks[0]] * pm[ks[1]] * pm[ks[2]]
        out[i] = np.sqrt(s123 * box_size**3 * prod / n_tri[i]) if n_tri[i] > 0 else np.nan
    return out


# ===========================================================================
# field construction (monolithic only)
# ===========================================================================


def evolved_field(cfg, seed, fdtype=None, ic_rel_eps=0.0, ic_pert_seed=90001):
    """One monolithic evolved density field. THE ONLY simulation path here.

    ic_rel_eps > 0 multiplies the INITIAL linear density by (1 + eps * xi) with
    xi a unit-variance field from an independent key. That is a DYNAMICAL null:
    physically the same simulation to within eps, but the difference is then
    propagated through 20 steps of nonlinear evolution rather than applied to
    the final field. It is the only construction here that can exhibit chaotic
    amplification, which is precisely what a field-level window cannot do and
    what a tiled arm certainly will.
    """
    import jax
    import jax.numpy as jnp

    from inexor import ic, lpt, painting
    from inexor.config import Cosmology
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table, float_step_bullfrog
    from v2_g5_core import force_global

    if fdtype is None:
        fdtype = jnp.float64
    g = geometry(cfg)
    ell, n_fine, n_part = g["L"], g["n_fine"], g["n_part"]
    cosmo = Cosmology()
    # THE ICs ARE ALWAYS BUILT AT f64 AND CAST AFTERWARDS. Passing fdtype into
    # linear_density changes the jax.random STREAM, not just its precision, so
    # an f32 "precision" arm built that way is a DIFFERENT REALIZATION -- it
    # decorrelates from the f64 arm at every k, including the box fundamental
    # (measured 1-r ~ 1 at k = k_f, rms difference 1.39x the field itself).
    # That reads as a catastrophic precision floor and is nothing of the kind.
    # fdtype below controls the EVOLUTION only, which is the thing under test.
    d0 = ic.linear_density(
        jax.random.PRNGKey(seed), n_part, ell, cosmo, f_NL=0.0, fdtype=jnp.float64
    )
    if ic_rel_eps:
        xi = jax.random.normal(jax.random.PRNGKey(ic_pert_seed), d0.shape, dtype=jnp.float64)
        d0 = d0 * (1.0 + ic_rel_eps * xi)
    x_ic, v_ic = lpt.lpt_ics(d0, ell, A_CONTROL, cosmo, order=2, fdtype=jnp.float64)
    table = bullfrog_table(a_grid(A_CONTROL, A_PIVOT, N_STEPS, "log"), cosmo)

    def force(pos):
        out, _ = force_global(pos, n_fine, ell, n_part**3, "mono", assign="cic")
        return jnp.asarray(out)

    x, v = jnp.asarray(x_ic, fdtype), jnp.asarray(v_ic, fdtype)
    for c in bullfrog_float_coeffs(table):
        x, v = float_step_bullfrog(x, v, tuple(np.asarray(c, np.float64)), force, ell)
    # paint="int" is the deterministic primal path (D-006). It matters here
    # specifically: the bit-repro rung of floor A is meaningless on a paint that
    # is not reproducible, so the null would read as instrument noise that is
    # really just the f32 scatter-add.
    d = painting.density_contrast(x, n_fine, ell, n_part**3, paint="int")
    return np.asarray(d, np.float64), g


def apply_window(delta, box_size, amp=0.05, k0_mult=6.0):
    """delta_t(k) = T(k) delta_m(k) with T = 1 + amp exp(-(k/k0)^2). KNOWN answer."""
    from inexor.diagnostics import _k_grid

    n = delta.shape[0]
    _, _, k_mag = _k_grid(n, box_size)
    k0 = k0_mult * 2.0 * np.pi / box_size
    tk = 1.0 + amp * np.exp(-((k_mag / k0) ** 2))
    return np.fft.irfftn(tk * np.fft.rfftn(delta), s=(n, n, n), axes=(0, 1, 2)), tk


def translate(delta, shift):
    """Rigid translation by whole cells. B is translation-invariant, so any
    response is pure instrument."""
    return np.roll(delta, shift, axis=(0, 1, 2))


# ===========================================================================
# sections
# ===========================================================================


def section_D(cfg, seed, out, long_mults=None, k_short_mult=None):
    """Discriminators against KNOWN answers, before they judge anything."""
    print("\n=== D: discriminator validation (known-answer, field-level) ===")
    d_m, g = evolved_field(cfg, seed)
    ell = g["L"]
    kf = 2.0 * np.pi / ell
    tris, names = _triangles(kf, long_mults, k_short_mult)
    print(f"  k_f = {kf:.4f} h/Mpc; k_long = "
          + ", ".join(f"{float(t[0]):.3f}" for t in tris[:-1])
          + f"; k_short = {float(tris[0][1]):.3f} h/Mpc")

    d_t, tk_grid = apply_window(d_m, ell)
    s = stats(d_t, d_m, ell, tris)

    # exact expected R_B: the window at the shell centres
    from inexor.diagnostics import _k_grid, _shell_mask

    _, _, k_mag = _k_grid(d_m.shape[0], ell)
    centers = sorted({float(k) for t in tris for k in t})
    t_bar = {}
    for c in centers:
        msk = _shell_mask(k_mag, c - 0.5 * kf, c + 0.5 * kf)
        t_bar[c] = float(tk_grid[msk].mean())
    expect = np.array([t_bar[float(t[0])] * t_bar[float(t[1])] * t_bar[float(t[2])] - 1.0
                       for t in tris])

    print(f"  {'tri':>6s} {'R_B':>11s} {'expect':>11s} {'resid':>10s} {'R_Q':>11s} "
          f"{'rho':>11s} {'W':>11s}")
    for i, nm in enumerate(names):
        print(f"  {nm:>6s} {s['R_B'][i]:11.6f} {expect[i]:11.6f} "
              f"{s['R_B'][i] - expect[i]:10.2e} {s['R_Q'][i]:11.6f} "
              f"{s['rho'][i]:11.3e} {s['W'][i]:11.3e}")
    print("  R_B should track the shell-averaged window product; the residual is")
    print("  within-shell variation of T, which is also W's floor (next line).")
    print(f"  W floor under a DETERMINISTIC window: max |W| = {np.abs(s['W']).max():.3e}")
    print(f"  R_Q under a pure window: max |R_Q| = {np.abs(s['R_Q']).max():.4f} -- NOT zero.")
    print("  Q = B/(P1P2+P2P3+P3P1) scales as 1/T under a uniform window (same")
    print("  algebra as tree-level Q ~ 1/b under linear bias), so R_Q REDUCES a")
    print("  P(k) contamination relative to R_B and flips its sign, but does not")
    print("  cancel it. rho divides the measured transfer out and IS window-flat.")
    print(f"  rho under a pure window: max |rho| = {np.abs(s['rho']).max():.3e}")

    out["D"] = dict(
        names=names, R_B=list(s["R_B"]), R_B_expect=list(expect), R_Q=list(s["R_Q"]),
        rho=list(s["rho"]), W=list(s["W"]), W_floor=float(np.abs(s["W"]).max()),
        RQ_max_abs=float(np.abs(s["R_Q"]).max()),
        rho_max_abs=float(np.abs(s["rho"]).max()),
    )
    return d_m, g, tris, names


def section_A(cfg, seed, d_m, g, tris, names, out):
    """Floor A: what a NULL perturbation reads."""
    print("\n=== A: instrument floor (null perturbations) ===")
    ell = g["L"]
    rows = {}

    s = stats(d_m.copy(), d_m, ell, tris)
    rows["bit_repro"] = s
    sh = (g["n_fine"] // 4, g["n_fine"] // 8, 3)
    rows["translated"] = stats(translate(d_m, sh), translate(d_m, sh), ell, tris)
    rows["translate_vs_orig"] = stats(translate(d_m, sh), d_m, ell, tris)

    import jax.numpy as jnp

    d32, _ = evolved_field(cfg, seed, fdtype=jnp.float32)
    rows["f32_vs_f64"] = stats(d32, d_m, ell, tris)

    print(f"  {'null':>18s} {'max|R_B|':>11s} {'max|R_Q|':>11s} {'max|rho|':>11s} "
          f"{'max|W|':>11s}")
    for k, s in rows.items():
        print(f"  {k:>18s} {np.abs(s['R_B']).max():11.3e} {np.abs(s['R_Q']).max():11.3e} "
              f"{np.abs(s['rho']).max():11.3e} {np.abs(s['W']).max():11.3e}")
    print("  translate_vs_orig is the physics check: B is translation-invariant,")
    print("  so a nonzero value there is estimator error, not a real difference.")

    out["A"] = {k: dict(R_B=list(v["R_B"]), R_Q=list(v["R_Q"]), rho=list(v["rho"]),
                        W=list(v["W"]))
                for k, v in rows.items()}
    out["A_summary"] = {k: dict(max_R_B=float(np.abs(v["R_B"]).max()),
                                max_R_Q=float(np.abs(v["R_Q"]).max()),
                                max_rho=float(np.abs(v["rho"]).max()),
                                max_W=float(np.abs(v["W"]).max())) for k, v in rows.items()}


def section_BC(cfg, nseed, tris, names, out):
    """Floors B and C, and the cancellation factor."""
    print(f"\n=== B/C: resolving power and cosmic variance ({nseed} seeds) ===")
    rq, rb, bm, wv, rlong = [], [], [], [], []
    t0 = time.perf_counter()
    for s in range(nseed):
        d_m, g = evolved_field(cfg, 1000 + s)
        ell = g["L"]
        d_t, _ = apply_window(d_m, ell)
        st = stats(d_t, d_m, ell, tris)
        rq.append(st["R_Q"])
        rb.append(st["R_B"])
        bm.append(st["B_m"])
        wv.append(st["W"])
        _, r = shell_transfer(d_t, d_m, ell, st["centers"], 2.0 * np.pi / ell)
        rlong.append(r[0])
        if s == 0:
            g0, p_m0, ntri0, cen0 = g, st["P_m"], st["n_tri"], st["centers"]
        print(f"  seed {1000 + s}: {time.perf_counter() - t0:6.1f}s", end="\r")
    rq, rb, bm, wv = map(np.asarray, (rq, rb, bm, wv))
    print(f"  {nseed} seeds in {time.perf_counter() - t0:.1f}s" + " " * 20)

    sig_B = rq.std(axis=0, ddof=1)
    sig_B_RB = rb.std(axis=0, ddof=1)
    sig_C = bm.std(axis=0, ddof=1) / np.abs(bm.mean(axis=0))
    cancel = sig_C / sig_B
    sig_g = gaussian_sigma_B(bm.mean(axis=0), ntri0, p_m0, cen0, tris, g0["L"])
    sig_g_frac = sig_g / np.abs(bm.mean(axis=0))

    print(f"\n  {'tri':>6s} {'sigma_B(R_Q)':>13s} {'sigma_B(R_B)':>13s} {'sigma_C':>10s} "
          f"{'cancel':>9s} {'sig_G/|B|':>10s} {'C/G':>7s} {'n_tri':>9s}")
    for i, nm in enumerate(names):
        print(f"  {nm:>6s} {sig_B[i]:13.4e} {sig_B_RB[i]:13.4e} {sig_C[i]:10.4f} "
              f"{cancel[i]:9.1f} {sig_g_frac[i]:10.4f} {sig_C[i] / sig_g_frac[i]:7.2f} "
              f"{ntri0[i]:9.0f}")
    print(f"\n  r(k_long) across seeds: mean {np.mean(rlong):.8f} min {np.min(rlong):.8f}")
    print("  (must be ~1: if it is not, the arms do not share long modes and the")
    print("   cancellation factor is meaningless)")

    bar = 0.15
    print(f"\n  RESOLVABILITY at the D-v2-7 bar {bar:.0%} (criterion sigma_B <= bar/3):")
    for i, nm in enumerate(names):
        ok = sig_B[i] <= bar / 3.0
        print(f"    {nm:>6s} sigma_B(R_Q) = {sig_B[i]:.4f}  "
              f"{'MEASUREMENT' if ok else 'BOUND ONLY'}  (bar/3 = {bar / 3:.4f})")

    out["BC"] = dict(
        nseed=nseed, names=names,
        sigma_B_RQ=list(sig_B), sigma_B_RB=list(sig_B_RB), sigma_C=list(sig_C),
        cancellation=list(cancel), sigma_gauss_frac=list(sig_g_frac),
        n_tri=list(ntri0), r_klong_mean=float(np.mean(rlong)),
        r_klong_min=float(np.min(rlong)),
        W_mean=list(wv.mean(axis=0)), W_sigma=list(wv.std(axis=0, ddof=1)),
        resolvable=[bool(v <= bar / 3.0) for v in sig_B],
    )


def inject_coupling(delta, box_size, k_long_center, dk, g):
    """delta_t = delta_m (1 + g * delta_L / rms(delta_L)), delta_L the long shell.

    A long mode modulating local small-scale amplitude -- the physical term an
    independent tile cannot reproduce, because the tile does not contain the
    long mode at all. This is the injection W exists to detect, as distinct from
    a window (which W must ignore) and a decorrelation (which W also ignores).
    """
    from inexor.diagnostics import _k_grid, _shell_mask

    n = delta.shape[0]
    _, _, k_mag = _k_grid(n, box_size)
    dk_grid = np.fft.rfftn(delta)
    msk = _shell_mask(k_mag, k_long_center - 0.5 * dk, k_long_center + 0.5 * dk)
    d_long = np.fft.irfftn(np.where(msk, dk_grid, 0.0), s=(n, n, n), axes=(0, 1, 2))
    d_long = d_long / np.sqrt((d_long**2).mean())
    return delta * (1.0 + g * d_long)


def section_F(cfg, seed, tris, names, out, g_ladder=(0.01, 0.02, 0.04, 0.08)):
    """W's POSITIVE-response test: a known quadratic long-short coupling.

    Replaces the plan's "scramble long-mode phases and confirm W returns the
    injected size", which cannot work for any W that is genuinely zero under a
    deterministic window. A phase ROTATION is still mode-diagonal, so it cancels
    out of W exactly as a window does; a full phase SCRAMBLE decorrelates the
    arms, and the cross-bispectrum of an independent field with the reference
    vanishes in expectation. Neither exercises W.

    W responds to new mode COUPLING, so the injection is a coupling. The known
    answer is the SCALING, not an amplitude: W must be linear in g with zero
    intercept, and must sit far above the window floor measured in section D. A
    statistic that is flat in g, or comparable to its own null, has no
    sensitivity and must not be reported as a discriminator.

    READ THE TWO OUTPUTS DIFFERENTLY. The linearity is EXACT BY CONSTRUCTION:
    delta_t - delta_m = g delta_m d_L is exactly O(g), and T - 1 is O(g) as
    well, so eps and therefore W are exactly linear in g. A constant W/g is
    consequently a PLUMBING check -- it would catch an O(g^2) contamination or a
    botched transfer subtraction, and nothing more. The physics result is the
    ratio W(g)/W_floor: that is what says W can tell a coupling from a window,
    which is the only reason it is in the deliverable.
    """
    print("\n=== F: W positive response to an injected long-short coupling ===")
    d_m, g_geo = evolved_field(cfg, seed)
    ell = g_geo["L"]
    kf = 2.0 * np.pi / ell
    k_long = float(tris[0][0])
    rows = []
    for g in g_ladder:
        d_t = inject_coupling(d_m, ell, k_long, kf, g)
        st = stats(d_t, d_m, ell, tris)
        rows.append((g, st))
    w_floor = out.get("D", {}).get("W_floor", float("nan"))

    print(f"  injecting at k_long = {k_long:.4f} h/Mpc; W window floor = {w_floor:.3e}")
    print(f"  {'g':>7s} " + " ".join(f"{'W[' + nm + ']':>12s}" for nm in names))
    for g, st in rows:
        print(f"  {g:7.3f} " + " ".join(f"{st['W'][i]:12.4e}" for i in range(len(names))))
    print(f"  {'W/g':>7s} " + " ".join(f"{'':>12s}" for _ in names))
    for g, st in rows:
        print(f"  {g:7.3f} " + " ".join(f"{st['W'][i] / g:12.4e}" for i in range(len(names))))

    w0 = np.array([r[1]["W"] for r in rows])
    gs = np.array([r[0] for r in rows])
    lin = [float(np.polyfit(gs, w0[:, i], 1)[0]) for i in range(len(names))]
    resid = [float(np.abs(w0[:, i] / gs - np.mean(w0[:, i] / gs)).max() / abs(np.mean(w0[:, i] / gs)))
             for i in range(len(names))]
    print(f"\n  {'tri':>6s} {'dW/dg':>12s} {'W/g spread':>12s} {'W(g=max)/floor':>15s}")
    for i, nm in enumerate(names):
        print(f"  {nm:>6s} {lin[i]:12.4e} {resid[i]:12.3f} "
              f"{abs(w0[-1, i]) / w_floor:15.1f}")
    out["F"] = dict(names=names, g=list(gs), W=[list(r) for r in w0],
                    dWdg=lin, W_over_g_spread=resid, W_floor=w_floor,
                    k_long=k_long)


def section_E(cfg, nseed, tris, names, out, eps_ladder=(1e-10, 1e-8, 1e-6)):
    """Floor B', the DYNAMICAL resolving power -- the one that binds.

    Section BC measures the seed-to-seed scatter of R for an arm B built by
    applying a window to the FINAL field. That construction is nearly noiseless
    by design: R is an almost deterministic function of the window, so its
    scatter is tiny and the resolvability verdict it produces is optimistic to
    the point of being wrong. A reference that CANNOT exhibit the behaviour
    under test reads as agreement.

    A tiled arm does not differ from a monolithic one by a post-hoc window. It
    differs during evolution, and the difference is then amplified by 20 steps
    of nonlinear dynamics. Floor A's f32-vs-f64 rung already shows what that
    costs (R_Q = 7% at cdev8 from precision alone).

    So: perturb the INITIAL linear density by a relative eps and evolve both
    arms in full. eps -> 0 is a physical null, so any surviving R is pure
    amplification. The ladder measures how it scales -- a Lyapunov-like growth
    means R saturates at a level set by the dynamics, NOT by eps, and that
    saturation level is the real floor against the 15% bar.
    """
    print(f"\n=== E: DYNAMICAL null -- IC perturbation, both arms evolved ({nseed} seeds) ===")
    print("  (section BC's window arm cannot chaotically amplify; this one can)")
    bar = 0.15
    rows = {}
    t0 = time.perf_counter()
    for eps in eps_ladder:
        rq, rb, rho_v = [], [], []
        for s in range(nseed):
            d_a, g = evolved_field(cfg, 2000 + s)
            d_b, _ = evolved_field(cfg, 2000 + s, ic_rel_eps=eps)
            st = stats(d_b, d_a, g["L"], tris)
            rq.append(st["R_Q"])
            rb.append(st["R_B"])
            rho_v.append(st["rho"])
            print(f"  eps={eps:.0e} seed {2000 + s}: {time.perf_counter() - t0:6.1f}s", end="\r")
        rows[eps] = (np.asarray(rq), np.asarray(rb), np.asarray(rho_v))
    print(" " * 60, end="\r")

    print(f"  {'eps':>8s} {'tri':>6s} {'mean R_Q':>11s} {'rms R_Q':>11s} {'rms R_B':>11s} "
          f"{'rms rho':>11s} {'vs bar/3':>9s}")
    for eps, (rq, rb, rho_v) in rows.items():
        for i, nm in enumerate(names):
            rms = float(np.sqrt((rq[:, i] ** 2).mean()))
            print(f"  {eps:8.0e} {nm:>6s} {rq[:, i].mean():11.4e} {rms:11.4e} "
                  f"{float(np.sqrt((rb[:, i] ** 2).mean())):11.4e} "
                  f"{float(np.sqrt((rho_v[:, i] ** 2).mean())):11.4e} "
                  f"{'OK' if rms <= bar / 3 else 'EXCEEDS':>9s}")
    print("\n  Read the eps-ladder, not any single row: if rms R_Q is flat in eps the")
    print("  difference has saturated at a dynamics-set level and THAT is the floor.")
    print("  If it scales linearly in eps, the perturbation is still in the linear")
    print("  regime and the floor is below anything measured here.")

    out["E"] = {f"{eps:.0e}": dict(
        names=names,
        mean_R_Q=list(rq.mean(axis=0)),
        rms_R_Q=list(np.sqrt((rq**2).mean(axis=0))),
        rms_R_B=list(np.sqrt((rb**2).mean(axis=0))),
        rms_rho=list(np.sqrt((rho_v**2).mean(axis=0))),
    ) for eps, (rq, rb, rho_v) in rows.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="smoke", choices=("smoke", "cdev8", "cdev", "cgh64"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nseed", type=int, default=24)
    ap.add_argument("--sections", default="DABC", help="subset of D, A, BC, E, F")
    ap.add_argument("--neps-seed", type=int, default=8, help="seeds for section E (2 evolves each)")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--long-mults", type=int, nargs="+", default=None,
                    help="k_long shell centres in units of k_f (default: the cdev8 set)")
    ap.add_argument("--k-short-mult", type=int, default=None,
                    help="k_short shell centre in units of k_f")
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)

    print(f"=== G3 Stage 3 floors: {args.config} ===")
    print("MONOLITHIC PAIRS ONLY -- no tiled arm exists in this script by design.")
    out = dict(config=args.config, seed=args.seed, sections=args.sections)

    d_m, g, tris, names = section_D(
        args.config, args.seed, out, args.long_mults, args.k_short_mult
    )
    out["geometry"] = {k: float(v) if isinstance(v, (int, float, np.generic)) else v
                       for k, v in g.items()}
    if "A" in args.sections:
        section_A(args.config, args.seed, d_m, g, tris, names, out)
    if "BC" in args.sections:
        section_BC(args.config, args.nseed, tris, names, out)
    if "F" in args.sections:
        section_F(args.config, args.seed, tris, names, out)
    if "E" in args.sections:
        section_E(args.config, args.neps_seed, tris, names, out)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"g3_floors_{args.config}{args.out_suffix}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
