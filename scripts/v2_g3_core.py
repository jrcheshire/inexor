"""v2 gate G3 (seed V2b): the sCOLA frame and the BullFrog residual stepper.

Core physics for G3 -- architecture A3, independent sequential sub-box tiles.
Probe code, NOT package code: promotion is a V4 decision, exactly as this
file's sibling v2_g5_core.py says of the split kernels. The stepper is written
as a pure function with the full derivation below so promotion is a copy-paste.

READ FIRST -- G5's MEASURED TILING-ERROR SCALINGS DO NOT TRANSFER TO G3.
In A2 (G5) the tile solved the SHORT half of a split kernel, (1-S(k)) ik/k^2,
which vanishes as (k r_s)^2 at small k: the tile's largest modes carried almost
no force, and the tiling error scaled as ~2.0/P (kernel ringing). In A3 the tile
solves the FULL ik/k^2, whose amplitude DIVERGES as 1/k toward the tile
fundamental. The tile's largest modes carry the MOST force, and their
periodization is a wholesale substitution of the tile box's modes for the true
box's -- not a ringing correction. NOTHING in runs/v2/g5_kernel_findings.md
(the "error x P ~ 2.0" law, edge/interior ~ 1, "the buffer is not a knob, tile
size is") may be assumed here. G3 re-measures every scaling from scratch.

Consequence for the buffer, which is the practical difference: because the full
1/k^2 kernel has no exponential cut, A WIDER BUFFER DOES NOT RESTORE THE MISSING
LONG MODES -- only the frame does. What the buffer actually fixes is (a)
particles migrating across the core wall, which in the COLA frame is only the
small residual y, and (b) the short-range force on core particles near the wall,
which needs real neighbours rather than periodic images. So the buffer scale is
set by max|y| plus the residual force's correlation length, BOTH MEASURABLE. That
is the mechanism-level reason an sCOLA buffer can be far thinner than a full-PM
tiling buffer, and it is a testable claim rather than a citation. Stage 1
measures it; the measurement, not a literature quote, picks the (T, b) grid.

===========================================================================
THE DERIVATION
===========================================================================

The BullFrog DKD step this file rewrites (integrate.float_step_bullfrog:329),
with h = dD/2, alpha, and bcoef = beta/D_mid from bullfrog_float_coeffs:

    x_h = mod(x_0 + h v_0, L)
    g   = force(x_h)
    v_1 = alpha v_0 + bcoef g
    x_1 = mod(x_h + h v_1, L)

The LPT trajectory in this codebase's conventions (lpt.py:1-22): Psi is
+grad lap^-1 delta, so x = q + D1 Psi1 - D2 Psi2, and cosmology.growth_factor_2
returns EXACTLY -(3/7) D1^2 (cosmology.py:80-90 -- the EdS form, the LCDM
correction Omega_m(a)^(-1/143) deliberately dropped). Therefore

    x_LPT(D)      = q + D Psi1 + (3/7) D^2 Psi2
    v_LPT(D)      = Psi1 + (6/7) D Psi2      == exactly lpt_ics' returned v_D
    x_LPT''(D)    = (6/7) Psi2               == CONSTANT in D

x_LPT is a QUADRATIC POLYNOMIAL in the integration variable, so every Taylor
expansion below TERMINATES and the rewrite is EXACT, not an approximation. No
D''(a), no quadrature, no finite difference is needed anywhere -- the three COLA
coefficients are pure functions of BullFrogTable's existing columns, and one of
them is literally the existing `betas` column.

  NB the exactness is CONVENTION-LEVEL. It holds because growth_factor_2 is the
  EdS form IN CODE; it is not a claim about true LCDM second-order growth. Both
  G3 arms use the same convention so it cancels in the ratio. Do not cite this
  as a statement about D2(a).

Residual variables:   y = x - x_LPT(D),   u = v_D - v_LPT(D).

Generic half-drift from D_a to D_b = D_a + h, holding the velocity variable w
fixed (which is what a drift does):

    x_b        = x_a + h w
    x_LPT(D_b) = x_LPT(D_a) + h v_LPT(D_a) + (3/7) h^2 Psi2
    => y_b     = y_a + h (w - v_LPT(D_a)) - (3/7) h^2 Psi2

FIRST half-drift: w = v_0 and D_a = D_0, so w - v_LPT(D_a) = u_0 exactly:

    y_h = y_0 + h u_0 - (3/7) h^2 Psi2

SECOND half-drift: w = v_1, the velocity at D_1, but D_a = D_mid, so

    w - v_LPT(D_mid) = u_1 + [v_LPT(D_1) - v_LPT(D_mid)] = u_1 + (6/7) h Psi2
    => y_1 = y_h + h u_1 + (6/7) h^2 Psi2 - (3/7) h^2 Psi2
           = y_h + h u_1 + (3/7) h^2 Psi2

*** THE TWO HALF-DRIFT CORRECTIONS HAVE OPPOSITE SIGNS *** and cancel over a
full step. The asymmetry is entirely because the velocity variable is
re-referenced at the kick (v_1 lives at D_1, not at D_mid). Giving both drifts
the same constant -- the intuitive guess, and what a first pass at this algebra
produces -- is WRONG and yields a plausible but incorrect trajectory. y_h still
matters independently of the cancellation, because it sets the force evaluation
point. The A0 identity (cola_mono vs mono at f64 round-off, via
cola_vs_direct_residual below) is the check that catches exactly this.

KICK, with v_1 = alpha v_0 + bcoef g:

    u_1 = v_1 - v_LPT(D_1) = alpha u_0 + bcoef g + [alpha v_LPT(D_0) - v_LPT(D_1)]
    alpha v_LPT(D_0) - v_LPT(D_1) = (alpha - 1) Psi1 + (6/7)(alpha D_0 - D_1) Psi2
                                  = -beta Psi1 - (6/7)(D_1 - alpha D_0) Psi2

Collecting, per step k (nothing new from cosmology.py is required):

    h    = 0.5 * table.dD[k]
    c_d2 = (3/7) h^2                             applied -c_d2, then +c_d2
    c_k1 = -table.betas[k]                       an EXISTING BullFrogTable column
    c_k2 = -(6/7) (D_1 - alpha D_0)              D_i = table.D_steps[k], [k+1]

y IS NEVER WRAPPED. Only the reconstructed x = mod(y + x_LPT(D), L) is, and only
at the force call. This is why the uint16 codec question is genuinely orthogonal
to the seam question (D-007's wrap-never-clamp invariant governs INTEGER state;
y is f64 probe state) and why y stays small and well-conditioned: it is what is
left after the frame has carried the bulk of the displacement.

INITIAL CONDITION: y = u = 0 EXACTLY at a_init, because lpt_ics returns
precisely x_LPT(D_init), v_LPT(D_init). The caller ASSERTS this rather than
assuming it -- an assumption there would silently invalidate every tiled number
downstream, and asserting it also makes the ZA-frame ablation (Psi2 = 0 in the
frame, ICs unchanged) fall out for free.

ATTRIBUTION HAZARD, plumbed from the first run rather than discovered at the
end: BullFrog's affine kick is ITSELF 2LPT-exact by construction
(_bullfrog_weights docstring; Rampf, List & Hahn 2024 Eqs 2.3-2.4), so it
already carries an LPT-exactness mechanism that partly overlaps the COLA frame.
The diagnostic is per-step rms|u| / rms|v| (frame_activity below). If that is
~1, the frame is doing nothing useful in this integrator and the sCOLA premise
needs re-examining BEFORE any tiled number is read.

NOT USED, deliberately: Tassev's COLA time-stepping modification (the nLPT
exponent). It is a step-count accuracy device, not a frame; adopting it would
introduce a SECOND difference between the arms and destroy attribution. Both G3
arms run BullFrog unchanged, which also keeps the monolithic reference
bit-comparable to G5's.
"""

import numpy as np

# The frame's second-order coefficients, written once. x_LPT(D) carries
# +(3/7) D^2 because lpt.py's sign convention puts -D2 Psi2 in the position and
# growth_factor_2 is exactly -(3/7) D1^2; v_LPT(D) is its exact D-derivative.
_C_POS2 = 3.0 / 7.0
_C_VEL2 = 6.0 / 7.0


# ===========================================================================
# the frame
# ===========================================================================


def lpt_frame(delta0, box_size):
    """(q, Psi1, Psi2) as (N^3, 3) numpy f64, C-order -- the tile-sliceable frame.

    C-order matches lpt.lagrangian_grid, so a tile's Lagrangian block is a pure
    reshape + slice with no gather. Returned as host numpy because the frame is
    built once per (config, seed), written to disk, and then sliced per tile.

    delta0 MUST be the same linear density make_config_leg used, regenerated from
    the same jax.random.PRNGKey(seed) at CPU f64. The caller's job is to assert
    that x_lpt(q, Psi1, Psi2, D_init) reproduces the saved IC positions exactly;
    see the module docstring's INITIAL CONDITION note.
    """
    import jax.numpy as jnp

    from inexor import lpt

    n = int(delta0.shape[0])
    q = lpt.lagrangian_grid(n, box_size, jnp.float64)
    psi1 = lpt.zeldovich_displacement(delta0, box_size, jnp.float64)
    psi2 = lpt.second_order_displacement(delta0, box_size, jnp.float64)
    return (
        np.asarray(q, np.float64),
        np.asarray(psi1, np.float64),
        np.asarray(psi2, np.float64),
    )


def x_lpt(q, psi1, psi2, d_growth):
    """x_LPT(D) = q + D Psi1 + (3/7) D^2 Psi2.  UNWRAPPED -- the caller wraps.

    Exact in D because growth_factor_2 is the EdS form in code (module docstring).
    """
    return q + d_growth * psi1 + (_C_POS2 * d_growth * d_growth) * psi2


def v_lpt(psi1, psi2, d_growth):
    """v_LPT(D) = dx_LPT/dD = Psi1 + (6/7) D Psi2 == lpt_ics' returned v_D."""
    return psi1 + (_C_VEL2 * d_growth) * psi2


def frame_residual_at_init(x_ic, v_ic, q, psi1, psi2, d_init, box_size):
    """(y0, u0) and the max |y0| / |u0| the caller must assert are zero.

    y0 is min-imaged because the saved IC positions are wrapped while x_lpt is
    not; a particle whose unwrapped LPT position sits outside [0, L) would
    otherwise read as a full-box error rather than zero.
    """
    y0 = min_image(np.asarray(x_ic, np.float64) - x_lpt(q, psi1, psi2, d_init), box_size)
    u0 = np.asarray(v_ic, np.float64) - v_lpt(psi1, psi2, d_init)
    return y0, u0, float(np.abs(y0).max()), float(np.abs(u0).max())


# ===========================================================================
# the residual stepper
# ===========================================================================


def bullfrog_cola_coeffs(table):
    """(K, 7) f64 (h, alpha, bcoef, c_d2, c_k1, c_k2, D_mid) from a BullFrogTable.

    Pure function of BullFrogTable's existing columns -- see the module
    docstring's collected result. c_d2 is a MAGNITUDE: cola_step_bullfrog applies
    it as -c_d2 on the first half-drift and +c_d2 on the second. c_k1 is
    -table.betas, i.e. the negation of a column that already exists.

    The (h, alpha, bcoef) triple is identical to bullfrog_float_coeffs
    (integrate.py:350) by construction, so the direct and residual paths share
    their integrator weights exactly and any A0 discrepancy is float association
    only.
    """
    t = table
    h = 0.5 * t.dD
    alpha = t.alphas
    bcoef = t.betas / t.D_mid
    c_d2 = _C_POS2 * h * h
    c_k1 = -t.betas
    d_0 = t.D_steps[:-1]
    d_1 = t.D_steps[1:]
    c_k2 = -_C_VEL2 * (d_1 - alpha * d_0)
    return np.stack([h, alpha, bcoef, c_d2, c_k1, c_k2, t.D_mid], axis=1).astype(np.float64)


def cola_step_bullfrog(y, u, q, psi1, psi2, coeff, force_fn, box_size):
    """One BullFrog step on the RESIDUAL (y, u). Exact rewrite of
    integrate.float_step_bullfrog:329 -- see the module docstring's derivation.

    coeff = (h, alpha, bcoef, c_d2, c_k1, c_k2, D_mid) from bullfrog_cola_coeffs.
    force_fn takes WRAPPED physical positions, matching the direct path's contract.

    Note the two drift corrections carry OPPOSITE signs (-c_d2 then +c_d2). That
    is not a typo; see the derivation.
    """
    import jax.numpy as jnp

    h, alpha, bcoef, c_d2, c_k1, c_k2, d_mid = (float(c) for c in coeff)
    y = y + h * u - c_d2 * psi2
    x_h = jnp.mod(y + x_lpt(q, psi1, psi2, d_mid), box_size)
    g = force_fn(x_h)
    u = alpha * u + bcoef * g + c_k1 * psi1 + c_k2 * psi2
    y = y + h * u + c_d2 * psi2
    return y, u


def evolve_cola(y, u, q, psi1, psi2, coeffs, force_fn, box_size, d_final, monitor=False):
    """Run the residual path over a whole schedule; return (x_final, u, diag).

    x_final is the reconstructed WRAPPED position, directly comparable to the
    direct path's output. diag carries frame_activity (rms|u|/rms|v| per step,
    the attribution canary) when monitor=True.
    """
    import jax.numpy as jnp

    activity = []
    for k, c in enumerate(coeffs):
        y, u = cola_step_bullfrog(y, u, q, psi1, psi2, c, force_fn, box_size)
        if monitor:
            d_end = float(c[6]) + 0.5 * float(c[0])  # D_mid + h == D_{k+1}
            activity.append(frame_activity(u, psi1, psi2, d_end))
    x = jnp.mod(y + x_lpt(q, psi1, psi2, d_final), box_size)
    return x, u, dict(frame_activity=activity)


# ===========================================================================
# diagnostics
# ===========================================================================


def min_image(dx, box_size):
    """Minimum-image a coordinate difference into (-L/2, L/2]."""
    dx = np.asarray(dx, np.float64)
    return dx - box_size * np.round(dx / box_size)


def _rms3(a):
    """rms of a per-particle vector magnitude, (n, 3) -> scalar."""
    a = np.asarray(a, np.float64)
    return float(np.sqrt((a * a).sum(axis=1).mean()))


def frame_activity(u, psi1, psi2, d_growth):
    """rms|u| / rms|v|, the GLOBAL frame activity. Cheap per-step monitor only.

    *** DO NOT READ THIS AS THE ATTRIBUTION CANARY. *** It is dominated by
    small-scale virial motion, which the 2LPT frame was never going to carry and
    which a tile handles locally anyway, so it saturates at ~1 at late times even
    when the frame is working perfectly. MEASURED at smoke (32^3, L=32, a=1):
    global ratio 1.017 (rms|u| = 6.61 vs rms|v| = 6.50 -- the residual slightly
    EXCEEDS the total, because the perturbative frame over-predicts velocity in
    collapsed regions and u has to cancel it), while the same state's
    scale-resolved ratio is 0.29 at 16 Mpc/h blocks. Reading the global number
    alone would have condemned a healthy frame.

    Use frame_activity_by_scale for the diagnostic that decides anything.
    """
    u = np.asarray(u, np.float64)
    v = u + v_lpt(np.asarray(psi1, np.float64), np.asarray(psi2, np.float64), d_growth)
    rms_v, rms_u = _rms3(v), _rms3(u)
    return dict(rms_u=rms_u, rms_v=rms_v, ratio=(rms_u / rms_v if rms_v else None))


def lagrangian_block_mean(a, q, n_part, box_size, t_lag):
    """Mean of a per-particle vector over Lagrangian blocks of t_lag particle cells.

    Blocks are cut in LAGRANGIAN space (the initial grid), which is where G3's
    tile membership lives, so this is the bulk motion of a prospective tile core.
    """
    a = np.asarray(a, np.float64)
    spacing = box_size / n_part
    qi = np.rint(np.asarray(q, np.float64) / spacing).astype(np.int64) % n_part
    nb = n_part // t_lag
    bid = (qi[:, 0] // t_lag) * nb * nb + (qi[:, 1] // t_lag) * nb + (qi[:, 2] // t_lag)
    nblk = nb**3
    counts = np.bincount(bid, minlength=nblk)
    out = np.empty((nblk, 3), np.float64)
    for j in range(3):
        out[:, j] = np.bincount(bid, weights=a[:, j], minlength=nblk)
    return out / counts[:, None]


def frame_activity_by_scale(y, disp, u, v, q, n_part, box_size, blocks=(2, 4, 8, 16)):
    """THE ATTRIBUTION CANARY, scale-resolved. Stage 1's decisive diagnostic.

    For each candidate tile core size (in PARTICLE cells) compare the block-mean
    residual against the block-mean total, in both displacement and velocity. The
    frame is doing its job iff these ratios fall with block size -- that is the
    statement "the frame carries the LARGE-SCALE motion", which is the only part
    that couples across a tile boundary. A flat or rising profile means the frame
    is not carrying the coupling and the sCOLA premise fails.

    MEASURED at smoke (32^3, L=32, a=1): displacement ratio
    0.341 / 0.238 / 0.136 / 0.076 at 2 / 4 / 8 / 16 Mpc/h blocks, i.e. the frame
    carries 92% of the bulk displacement at 16 Mpc/h. Monotone, as required.

    `blocks` deliberately excludes n_part (one block = the whole box): the
    block-mean displacement there is the box's net momentum, which is exactly
    zero by construction, so the ratio is 0/0. That degenerate limit is reported
    separately by whole_box_momentum as a free identity rather than as a data row.
    """
    rows = []
    for t_lag in blocks:
        if n_part % t_lag or t_lag >= n_part:
            continue

        def bm(a, _t=t_lag):
            return lagrangian_block_mean(a, q, n_part, box_size, _t)

        r_disp = _rms3(bm(disp))
        r_vel = _rms3(bm(v))
        rows.append(
            dict(
                t_lag=int(t_lag),
                t_phys=float(t_lag * box_size / n_part),
                rms_block_y=_rms3(bm(y)),
                rms_block_disp=r_disp,
                ratio_disp=(_rms3(bm(y)) / r_disp if r_disp else None),
                rms_block_u=_rms3(bm(u)),
                rms_block_v=r_vel,
                ratio_vel=(_rms3(bm(u)) / r_vel if r_vel else None),
            )
        )
    return rows


def whole_box_momentum(disp, v):
    """Free identity: the whole-box mean displacement and velocity are exactly 0.

    Momentum conservation, so this is a plumbing check on the block machinery and
    on the force's k=0 nulling, not a physics result. MEASURED 0.0000 at smoke.
    """
    return dict(
        mean_abs_disp=float(np.abs(np.asarray(disp, np.float64).mean(axis=0)).max()),
        mean_abs_v=float(np.abs(np.asarray(v, np.float64).mean(axis=0)).max()),
    )


def residual_size(y, x_final, q, box_size, d_fine=None):
    """THE BUFFER SIZER (Stage 1's first deliverable).

    rms/p99/p999/max of |y| in Mpc/h, plus rms|y| / rms|x_final - q| -- the
    fraction of the total displacement the frame does NOT carry. The buffer must
    cover the residual excursion, so these numbers pick the (T, b) grid.

    x_final is the FINAL wrapped position, matching y's epoch. Passing the
    INITIAL position instead compares epochs rather than forming a fraction: the
    total displacement grows ~D, so at smoke the same y read 3.38 against the
    a=0.1 displacement and 0.421 against the a=1 displacement. The first number
    is meaningless and I generated it once; hence the explicit argument name.
    """
    y = np.asarray(y, np.float64)
    r = np.sqrt((y * y).sum(axis=1))
    rms_y = _rms3(y)
    rms_d = _rms3(min_image(np.asarray(x_final, np.float64) - q, box_size))
    out = dict(
        rms=rms_y,
        p99=float(np.percentile(r, 99.0)),
        p999=float(np.percentile(r, 99.9)),
        max=float(r.max()),
        rms_total_displacement=rms_d,
        rms_ratio=(rms_y / rms_d if rms_d else None),
    )
    if d_fine:
        out["cells"] = {k: out[k] / d_fine for k in ("rms", "p99", "p999", "max")}
        # Implied buffer for a few safety factors on the p999 excursion. The
        # buffer's job is particle migration plus the near-wall short-range
        # force, so p999 (not rms) is the right statistic to cover.
        out["implied_b_fine"] = {
            f"C{c}": int(np.ceil(c * out["p999"] / d_fine)) for c in (1, 2, 3)
        }
    return out


def cola_vs_direct_residual(x0, v0, q, psi1, psi2, table, force_fn, box_size, d_init, d_final):
    """THE A0 IDENTITY, factored so smoke and cdev run identical code.

    Runs the direct BullFrog path (integrate.float_step_bullfrog) and the
    residual path over the same schedule with the same force, and returns the max
    min-imaged position discrepancy plus the initial-residual check.

    The two paths differ ONLY by float association, so this must land at f64
    round-off (~1e-12 Mpc/h). It is the check that catches a wrong half-drift
    sign -- the one place this rewrite can plausibly go wrong -- and per the G5
    precedent NO other G3 number may be read until it passes.
    """
    import jax.numpy as jnp

    from inexor.integrate import bullfrog_float_coeffs, float_step_bullfrog

    y0, u0, max_y0, max_u0 = frame_residual_at_init(x0, v0, q, psi1, psi2, d_init, box_size)

    xd = jnp.asarray(x0, jnp.float64)
    vd = jnp.asarray(v0, jnp.float64)
    for c in bullfrog_float_coeffs(table):
        xd, vd = float_step_bullfrog(xd, vd, tuple(np.asarray(c, np.float64)), force_fn, box_size)

    xc, uc, diag = evolve_cola(
        jnp.asarray(y0, jnp.float64),
        jnp.asarray(u0, jnp.float64),
        jnp.asarray(q, jnp.float64),
        jnp.asarray(psi1, jnp.float64),
        jnp.asarray(psi2, jnp.float64),
        bullfrog_cola_coeffs(table),
        force_fn,
        box_size,
        d_final,
        monitor=True,
    )

    dx = min_image(np.asarray(xc, np.float64) - np.asarray(xd, np.float64), box_size)
    return dict(
        max_abs_dx=float(np.abs(dx).max()),
        rms_dx=float(np.sqrt((dx * dx).sum(axis=1).mean())),
        init_max_abs_y0=max_y0,
        init_max_abs_u0=max_u0,
        frame_activity=diag["frame_activity"],
        x_cola=np.asarray(xc, np.float64),
        x_direct=np.asarray(xd, np.float64),
        u_final=np.asarray(uc, np.float64),
    )
