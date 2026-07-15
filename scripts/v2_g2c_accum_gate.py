"""v2 gate G2c: ACCUMULATED (quantize-every-step) codec error at a V0 config.

The question (plan-plan Sec 5, seed V1): do the v2 state codecs survive
stored-state evolution -- encode/decode roundtrip on (x, v) after EVERY step,
the CUBE pattern -- at the D-v2-8 bars? This gates the storage CODEC riding a
float engine; the v1 int-ladder engine is a different mechanism (S1 fallback
tier, validated at D-014) and is NOT what runs here.

Codec tiers (kill-line ladder, plan-plan V1):
  t6lin / t6cdf : int8 cell-relative positions (quantum = fine_cell / 256)
                  + int8 velocity residual vs the LOCAL coarse-grid mean flow
                  (CUBE 1712.06121 Sec 2.2 mechanism), linear-c6 / Gaussian-CDF
                  variants.                                  -> 6 B/particle
  t9            : int8 cell-relative positions + int16 max-range velocity
                                                             -> 9 B/particle
  t12 (control) : int16-global positions (uint16 lattice) + int16 max-range
                  velocity -- the validated S1 class.        -> 12 B/particle

Floors first (m1_quant_gate discipline): step floor = ref(K) vs ref(2K);
mesh floor = ref fine-mesh vs half-mesh at the same particles/steps. The
D-v2-1 bar quotes codec error against the MESH floor for
k <= 0.2 k_Nyq(fine); the velocity gate is MULTIPOLE-GRADE (D-v2-8): P0 AND
P2 of the plane-parallel redshift-space field, quadrupole error normalized by
the monopole (dP2/P0 -- P2 crosses zero, a per-bin ratio there is ill-posed).

RSD convention (verified): s_z = x_z + f(a) D(a) v_D,z with v_D = dx/dD the
D-time velocity (lpt.py convention), from u_pec/(aH) = v_D dD/dt / H =
f D v_D (Kaiser 1987 plane-parallel mapping, z-axis LOS). The estimator is
validated in-run by the `kaiser` leg BEFORE any gate number is read: a
near-linear ZA field's binned P0s/P0r and P2s/P0r must match the per-mode
Kaiser prediction (1 + f mu^2)^2 binned over the SAME rfftn modes (phase-exact
expectation -- no cosmic-variance slop).

Honest accounting notes (printed into the json):
- The 3 B/p position claim assumes CUBE's sorted-by-cell layout; this gate
  measures REPRESENTATION error only (storage layout is a build decision).
- t6 linear clips residuals at +-c sigma = a SATURATING op. D-007 bans
  saturation on integer state in the ENGINE; a production t6 needs either an
  outlier side-channel (CUBE-style) or a D-007 exception ratified by JC. The
  outlier fraction and its side-channel byte cost are reported per run.
- The coarse mean-flow paint here uses f32 scatter-add (nondeterministic on
  GPU at ~1e-7 relative -- fine for a probe, far below the ~1e-5 codec
  errors); the engine version would use the deterministic int paint.

Orchestration: one arm per FRESH SUBPROCESS (peak_bytes_in_use is monotonic,
retrospective Sec 5); workers save final states to runs/v2/g2c_states/ and
the orchestrator computes all statistics pairwise against ref with ONE
estimator stack. Each worker reports the (peak B/p, wall/step) pair for the
cost-of-memory record (SU is n/a on deneb; it becomes real at V3).

Run (deneb via Slurm, scripts/v2_g2c_deneb.sbatch):
    pixi run -e gpu python scripts/v2_g2c_accum_gate.py --config cdev
CPU smoke (laptop):
    pixi run python scripts/v2_g2c_accum_gate.py --config smoke
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(REPO, "runs", "v2")
STATE_DIR = os.path.join(OUT_DIR, "g2c_states")

# (n_part, L, n_fine, n_coarse): C-dev per D-v2-8; cdev8 = same cell/spacing
# (codec quanta + k*cell mesh error preserved), 1/8 the volume; smoke = CPU.
CONFIGS = {
    "cdev": dict(n_part=256, L=128.0, n_fine=512, n_coarse=128),
    "cdev8": dict(n_part=128, L=64.0, n_fine=256, n_coarse=64),
    "smoke": dict(n_part=64, L=32.0, n_fine=128, n_coarse=32),
}
A_INIT, A_FINAL, SPACING = 0.1, 1.0, "log"
SEED = 0
CODEC_ARMS = ("t6lin", "t6cdf", "t9", "t12")
C_CLIP = 6.0  # residual clip range in sigma (linear tier); G2b convention


# ===========================================================================
# codecs (worker side; jnp, eager)
# ===========================================================================


def _rt_pos_lattice(x, L, n_levels):
    """Position roundtrip on a 2^m lattice spanning the box: quantum = L / n_levels.

    n_levels * quantum == L exactly, so the mod-L wrap is modular (never
    saturating). Covers int16-global (n_levels = 2^16) AND int8 cell-relative
    (n_levels = n_fine * 256: cell boundaries are multiples of the quantum).
    """
    import jax.numpy as jnp

    q = L / n_levels
    return jnp.mod(jnp.rint(x / q) * q, L)


def _rt_vel_int16_max(v):
    """int16 max-range velocity: quantum from the CURRENT max|v| (no clip --
    zero saturation by construction; the scale is a 4-byte side constant)."""
    import jax.numpy as jnp

    q = 2.0 * jnp.max(jnp.abs(v)) / 65536.0
    return jnp.rint(v / q) * q, 0.0


def _coarse_mean_flow(x, v, L, n_coarse):
    """Mass-weighted CIC mean-flow field on the coarse mesh + empty-cell count.

    Returns (vbar_mesh (3, Nc, Nc, Nc), n_empty). f32 scatter-add (probe-grade;
    see module docstring).
    """
    import jax.numpy as jnp

    from inexor.painting import _cic_pieces, _corner_flat_weight, _CORNERS

    base, frac = _cic_pieces(x, n_coarse, L)
    counts = jnp.zeros((n_coarse**3,), dtype=jnp.float32)
    mom = jnp.zeros((3, n_coarse**3), dtype=jnp.float32)
    for corner in _CORNERS:
        flat, w = _corner_flat_weight(base, frac, corner, n_coarse)
        counts = counts.at[flat].add(w, mode="promise_in_bounds")
        for c in range(3):
            mom = mom.at[c, flat].add(w * v[:, c], mode="promise_in_bounds")
    n_empty = jnp.sum(counts == 0.0)  # jnp scalar: this runs under jit
    vbar = mom / jnp.maximum(counts, 1e-12)[None, :]
    return vbar.reshape(3, n_coarse, n_coarse, n_coarse), n_empty


def _rt_vel_resid8(x, v, L, n_coarse, mode):
    """int8 velocity residual vs the local coarse-grid mean flow (CUBE Sec 2.2).

    mode="lin": uniform quantizer over +-C_CLIP sigma (CLIPS outliers -- the
    saturation caveat in the module docstring). mode="cdf": Gaussian-companded
    256-level quantizer at the same sigma. Returns (v_deq, outlier_frac).
    """
    import jax.numpy as jnp
    from jax.scipy.special import erf, erfinv

    from inexor.painting import cic_read_vector

    vbar_mesh, n_empty = _coarse_mean_flow(x, v, L, n_coarse)
    vbar = cic_read_vector(vbar_mesh[0], vbar_mesh[1], vbar_mesh[2], x, n_coarse, L)
    r = v - vbar
    sigma = jnp.sqrt(jnp.mean(r**2))  # pooled per-component sigma
    out_frac = jnp.mean(jnp.abs(r) > C_CLIP * sigma)  # jnp scalar (jit)
    if mode == "lin":
        q = 2.0 * C_CLIP * sigma / 256.0
        rq = jnp.rint(jnp.clip(r, -C_CLIP * sigma, C_CLIP * sigma) / q) * q
    elif mode == "cdf":
        u = 0.5 * (1.0 + erf(r / (jnp.sqrt(2.0) * sigma)))
        edge = 0.5 / 256.0
        uq = jnp.clip((jnp.floor(u * 256.0) + 0.5) / 256.0, edge, 1.0 - edge)
        rq = jnp.sqrt(2.0) * sigma * erfinv(2.0 * uq - 1.0)
    else:
        raise ValueError(f"mode must be 'lin' or 'cdf', got {mode!r}")
    return vbar + rq, out_frac, n_empty


def make_roundtrip(arm, L, n_fine, n_coarse):
    """Per-step (x, v) -> (x', v', diag) roundtrip for a codec arm.

    Pure jnp with a fixed diag structure per arm, so the whole step+roundtrip
    can run under ONE jitted, donated program (the job-27 lesson: the eager
    loop's per-op temporaries OOM the 6 GB card at configs the v1 per-step-jit
    discipline handled fine).
    """

    def rt(x, v):
        diag = {}
        if arm == "ref":
            return x, v, diag
        if arm in ("t6lin", "t6cdf", "t9"):
            xq = _rt_pos_lattice(x, L, n_fine * 256)
        elif arm == "t12":
            xq = _rt_pos_lattice(x, L, 2**16)
        else:
            raise ValueError(f"unknown arm {arm!r}")
        if arm in ("t6lin", "t6cdf"):
            vq, out_frac, n_empty = _rt_vel_resid8(xq, v, L, n_coarse, arm[2:])
            diag = {"outlier_frac": out_frac, "n_empty_coarse": n_empty}
        else:
            vq, _ = _rt_vel_int16_max(v)
        return xq, vq, diag

    return rt


# ===========================================================================
# worker: evolve one arm, save the final state (fresh subprocess per arm)
# ===========================================================================


def run_single(args):
    import jax

    import jax.numpy as jnp

    from inexor import config as icfg
    from inexor.ic import gaussian_delta
    from inexor.integrate import bullfrog_float_coeffs, bullfrog_table, a_grid
    from inexor.forces import make_force_fn
    from inexor.integrate import float_step_bullfrog
    from inexor.lpt import lpt_ics

    cfg = CONFIGS[args.config]
    n_part, L = cfg["n_part"], cfg["L"]
    n_mesh = args.mesh or cfg["n_fine"]
    cosmo = icfg.Cosmology()
    box = icfg.BoxConfig(n_mesh=n_mesh, box_size=L, n_particles=n_part)
    force_fn = make_force_fn(box, fdtype=jnp.float32, paint="int")

    key = jax.random.PRNGKey(SEED)
    delta0 = gaussian_delta(key, n_part, L, cosmo)
    x, v = lpt_ics(delta0, L, A_INIT, cosmo, order=2, fdtype=jnp.float32)

    a_steps = a_grid(A_INIT, A_FINAL, args.k, SPACING)
    coeffs = bullfrog_float_coeffs(bullfrog_table(a_steps, cosmo))
    rt = make_roundtrip(args.arm, L, cfg["n_fine"], cfg["n_coarse"])

    dev = jax.devices()[0]

    def peak():
        try:
            return (dev.memory_stats() or {}).get("peak_bytes_in_use")
        except Exception:
            return None

    # ONE jitted, donated step+roundtrip program reused across steps (the
    # run_perstep discipline; job 27's eager loop OOM'd where v1's jit fit).
    # Coefficients ride as a traced (3,) row, so all K steps share the program.
    def step_and_rt(x_, v_, c_):
        x_, v_ = float_step_bullfrog(x_, v_, (c_[0], c_[1], c_[2]), force_fn, L)
        return rt(x_, v_)

    step_jit = jax.jit(step_and_rt, donate_argnums=(0, 1))
    coeffs_dev = jnp.asarray(coeffs, jnp.float32)

    diag_last = {}
    jax.block_until_ready(x)
    t0 = time.perf_counter()
    for k in range(coeffs_dev.shape[0]):
        x, v, diag_last = step_jit(x, v, coeffs_dev[k])
    x, v = jax.block_until_ready(x), jax.block_until_ready(v)
    wall = time.perf_counter() - t0
    diag_last = {k: float(val) for k, val in diag_last.items()}

    os.makedirs(STATE_DIR, exist_ok=True)
    tag = f"{args.config}_{args.arm}_k{args.k}_m{n_mesh}"
    np.savez(
        os.path.join(STATE_DIR, f"{tag}.npz"),
        x=np.asarray(x, np.float32),
        v=np.asarray(v, np.float32),
    )
    rec = dict(
        arm=args.arm,
        k=args.k,
        n_mesh=n_mesh,
        config=args.config,
        platform=dev.platform,
        wall_s=wall,
        wall_per_step=wall / args.k,
        peak_bytes=peak(),
        peak_bytes_per_particle=(peak() / n_part**3) if peak() else None,
        **diag_last,
    )
    print("WORKER_JSON " + json.dumps(rec))


# ===========================================================================
# statistics (orchestrator side; numpy + package estimators)
# ===========================================================================


def _mu2_grid(n_mesh, L):
    """mu^2 = (kz/k)^2 on the rfftn half-grid (k=0 mode -> mu^2 = 0)."""
    from inexor.diagnostics import _k_grid

    k1, kz, kmag = _k_grid(n_mesh, L)
    KZ = np.broadcast_to(kz[None, None, :], kmag.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu2 = np.where(kmag > 0, (KZ / np.where(kmag > 0, kmag, 1.0)) ** 2, 0.0)
    return kmag, mu2


def multipoles(delta, L, dk=None):
    """Binned P0(k), P2(k) of a real mesh (plane-parallel, z-LOS).

    P_ell(k) = (2 ell + 1) < P(k, mu) L_ell(mu) >_bin over the rfftn half-grid
    (L2 even in mu, so the half-grid mu >= 0 restriction is unbiased); same
    fundamental-width bins as diagnostics.pk_estimator.
    """
    from inexor.diagnostics import _bin_edges

    delta = np.asarray(delta, np.float64)
    N = delta.shape[0]
    pm = (np.abs(np.fft.rfftn(delta)) ** 2 * (L**3 / N**6)).ravel()
    kmag, mu2 = _mu2_grid(N, L)
    km, mu2 = kmag.ravel(), mu2.ravel()
    leg2 = 0.5 * (3.0 * mu2 - 1.0)
    edges = _bin_edges(N, L, dk)
    counts, _ = np.histogram(km, bins=edges)
    s0, _ = np.histogram(km, bins=edges, weights=pm)
    s2, _ = np.histogram(km, bins=edges, weights=pm * leg2)
    centers = 0.5 * (edges[1:] + edges[:-1])
    good = counts > 0
    p0 = s0[good] / counts[good]
    p2 = 5.0 * s2[good] / counts[good]
    return centers[good], p0, p2


def rsd_positions(x, v, L, cosmo):
    """s_z = x_z + f(a_final) D(a_final) v_D,z (module-docstring convention)."""
    from inexor.cosmology import growth_factor_a, growth_rate_a

    fD = growth_rate_a(A_FINAL, cosmo) * growth_factor_a(A_FINAL, cosmo)
    s = x.copy()
    s[:, 2] = np.mod(x[:, 2] + fD * v[:, 2], L)
    return s


def paint_delta(pos, n_mesh, L):
    """Deterministic int paint -> f64 numpy delta (stats mesh = the fine mesh)."""
    import jax.numpy as jnp

    from inexor.painting import density_contrast

    d = density_contrast(jnp.asarray(pos), n_mesh, L, pos.shape[0], paint="int")
    return np.asarray(d, np.float64)


def compare_pair(ref_npz, arm_npz, n_stat, L, cosmo, k_gate):
    """All gate statistics for one (ref, arm) state pair.

    Real-space |dP/P| + 1-r (existing estimators) and redshift-space
    |dP0/P0|, |dP2/P0| (multipole-grade bar), each split at k <= k_gate.
    """
    from inexor.diagnostics import cross_r, pk_estimator

    ref, arm = np.load(ref_npz), np.load(arm_npz)
    out = {}
    d_ref = paint_delta(ref["x"], n_stat, L)
    d_arm = paint_delta(arm["x"], n_stat, L)
    k, p_ref, _ = pk_estimator(d_ref, L)
    _, p_arm, _ = pk_estimator(d_arm, L)
    _, r, _ = cross_r(d_arm, d_ref, L)
    band = k <= k_gate
    dpp = np.abs(p_arm / p_ref - 1.0)
    out["real"] = dict(
        max_dP_P_gate=float(np.nanmax(dpp[band])),
        max_dP_P_all=float(np.nanmax(dpp)),
        max_1mr_gate=float(np.nanmax(np.abs(1.0 - r[band]))),
    )
    s_ref = rsd_positions(ref["x"], ref["v"], L, cosmo)
    s_arm = rsd_positions(arm["x"], arm["v"], L, cosmo)
    ks, p0_ref, p2_ref = multipoles(paint_delta(s_ref, n_stat, L), L)
    _, p0_arm, p2_arm = multipoles(paint_delta(s_arm, n_stat, L), L)
    bands = ks <= k_gate
    dp0 = np.abs(p0_arm / p0_ref - 1.0)
    dp2 = np.abs((p2_arm - p2_ref) / p0_ref)  # quadrupole error over monopole
    out["rsd"] = dict(
        max_dP0_P0_gate=float(np.nanmax(dp0[bands])),
        max_dP2_P0_gate=float(np.nanmax(dp2[bands])),
        k=ks[bands].tolist(),
        dP0_P0=dp0[bands].tolist(),
        dP2_P0=dp2[bands].tolist(),
    )
    return out


def kaiser_check(cosmo):
    """Estimator validation floor: near-linear ZA field vs the PER-MODE Kaiser
    prediction (1 + f mu^2)^2, binned over the same rfftn modes -- phase-exact,
    so the tolerance is ZA nonlinearity + painting, not cosmic variance."""
    import jax

    from inexor.diagnostics import _bin_edges
    from inexor.ic import gaussian_delta
    from inexor.lpt import za_ics

    n, L, amp = 64, 128.0, 0.05
    key = jax.random.PRNGKey(SEED)
    delta0 = amp * np.asarray(gaussian_delta(key, n, L, cosmo), np.float64)
    x, v = za_ics(delta0, L, A_FINAL, cosmo)  # a = a_final: D = 1
    x, v = np.asarray(x, np.float64), np.asarray(v, np.float64)
    from inexor.cosmology import growth_rate_a

    f = growth_rate_a(A_FINAL, cosmo)
    s = rsd_positions(x, v, L, cosmo)
    d_r = paint_delta(x, n, L)
    d_s = paint_delta(s, n, L)
    _, p0_r, _ = multipoles(d_r, L)
    ks, p0_s, p2_s = multipoles(d_s, L)
    # phase-exact expectation: weight the REAL field's per-mode power
    pm = (np.abs(np.fft.rfftn(d_r)) ** 2).ravel()
    kmag, mu2 = _mu2_grid(n, L)
    km = kmag.ravel()
    kf = (1.0 + f * mu2.ravel()) ** 2
    leg2 = 0.5 * (3.0 * mu2.ravel() - 1.0)
    edges = _bin_edges(n, L)
    cts, _ = np.histogram(km, bins=edges)
    w0, _ = np.histogram(km, bins=edges, weights=pm * kf)
    wr, _ = np.histogram(km, bins=edges, weights=pm)
    w2, _ = np.histogram(km, bins=edges, weights=pm * kf * leg2)
    good = cts > 0
    exp_p0_ratio = w0[good] / wr[good]
    exp_p2_over_p0 = 5.0 * w2[good] / w0[good]
    nlow = 6  # low-k bins; ZA nonlinearity grows with k
    meas_ratio = p0_s[:nlow] / p0_r[:nlow]
    err0 = np.max(np.abs(meas_ratio / exp_p0_ratio[:nlow] - 1.0))
    err2 = np.max(np.abs(p2_s[:nlow] / p0_s[:nlow] - exp_p2_over_p0[:nlow]))
    ok = bool(err0 < 0.01 and err2 < 0.01)
    return dict(ok=ok, max_p0_ratio_err=float(err0), max_p2_over_p0_err=float(err2), f=float(f))


# ===========================================================================
# orchestration
# ===========================================================================


def spawn(config, arm, k, mesh=None):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--single",
        "--config",
        config,
        "--arm",
        arm,
        "--k",
        str(k),
    ]
    if mesh:
        cmd += ["--mesh", str(mesh)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        lines = (p.stderr or "").strip().splitlines()
        # persist the whole stderr; surface the real exception, not JAX's
        # trailing "For simplicity..." boilerplate (job 26/27 lesson)
        err_dir = os.path.join(OUT_DIR, "g2c_errs")
        os.makedirs(err_dir, exist_ok=True)
        err_path = os.path.join(err_dir, f"{config}_{arm}_k{k}_m{mesh or 0}.err")
        with open(err_path, "w") as fh:
            fh.write(p.stderr or "")
        exc = [t for t in lines if ("Error" in t or "Exception" in t) and "For simplicity" not in t]
        return dict(
            arm=arm,
            k=k,
            mesh=mesh,
            error=(exc[-1] if exc else (lines[-1] if lines else f"exit {p.returncode}")),
            error_file=err_path,
            oom=any("RESOURCE_EXHAUSTED" in t or "Out of memory" in t for t in lines),
        )
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("WORKER_JSON ")][-1]
    return json.loads(line[len("WORKER_JSON ") :])


def state_path(config, arm, k, mesh):
    return os.path.join(STATE_DIR, f"{config}_{arm}_k{k}_m{mesh}.npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="cdev", choices=sorted(CONFIGS))
    ap.add_argument("--ks", default="10,20,40", help="K sweep (accumulation axis)")
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--arm", default="ref", help=argparse.SUPPRESS)
    ap.add_argument("--k", type=int, default=10, help=argparse.SUPPRESS)
    ap.add_argument("--mesh", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.single:
        run_single(args)
        return

    from inexor.config import Cosmology

    cfg = CONFIGS[args.config]
    n_fine, L = cfg["n_fine"], cfg["L"]
    cosmo = Cosmology()
    k_gate = 0.2 * np.pi * n_fine / L  # 0.2 k_Nyq(fine); >= k_sci at C-dev
    ks = [int(s) for s in args.ks.split(",")]
    k_head = max(ks)

    print(f"=== G2c accumulated codec gate [{args.config}] ===")
    print(f"    fine mesh {n_fine}^3, L = {L}, gate band k <= {k_gate:.3f} h/Mpc")
    print("--- estimator validation (Kaiser, phase-exact) ---")
    kres = kaiser_check(cosmo)
    print(
        f"    P0 ratio err {kres['max_p0_ratio_err']:.2e}, "
        f"P2/P0 err {kres['max_p2_over_p0_err']:.2e} -> "
        f"{'OK' if kres['ok'] else 'FAIL (do NOT read the gate numbers)'}"
    )

    runs = {}
    legs = [("ref", k, None) for k in sorted(set(ks + [2 * k for k in ks]))]
    legs += [("ref", k_head, n_fine // 2)]  # mesh-floor arm
    legs += [(arm, k, None) for arm in CODEC_ARMS for k in ks]
    for arm, k, mesh in legs:
        m = mesh or n_fine
        print(f"[worker] arm={arm} K={k} mesh={m} ...", flush=True)
        r = spawn(args.config, arm, k, mesh)
        runs[(arm, k, m)] = r
        if r.get("error"):
            print(
                f"    WORKER {'OOM' if r.get('oom') else 'FAILED'}: {r['error'][:110]}", flush=True
            )

    results = dict(
        config=args.config,
        k_gate=k_gate,
        kaiser=kres,
        workers={f"{a}_k{k}_m{m}": r for (a, k, m), r in runs.items()},
        bytes_per_particle=dict(
            t6="6 + mean-flow field (12 B/coarse cell) + outlier side-channel"
            " (outlier_frac * 16 B/p equiv); position 3 B/p assumes"
            " sorted-by-cell layout",
            t9="9 + scale constants",
            t12="12 + scale constants",
        ),
        floors={},
        arms={},
    )

    def ref_path(k, m=None):
        return state_path(args.config, "ref", k, m or n_fine)

    print("\n--- floors (the bars codec error is judged against) ---")
    for k in ks:
        if not os.path.exists(ref_path(k)) or not os.path.exists(ref_path(2 * k)):
            continue
        c = compare_pair(ref_path(k), ref_path(2 * k), n_fine, L, cosmo, k_gate)
        results["floors"][f"step_k{k}"] = c
        print(
            f"    step floor K={k} vs {2 * k}: dP/P {c['real']['max_dP_P_gate']:.3e}  "
            f"dP0/P0 {c['rsd']['max_dP0_P0_gate']:.3e}  dP2/P0 {c['rsd']['max_dP2_P0_gate']:.3e}"
        )
    mesh_arm = state_path(args.config, "ref", k_head, n_fine // 2)
    if os.path.exists(mesh_arm) and os.path.exists(ref_path(k_head)):
        c = compare_pair(ref_path(k_head), mesh_arm, n_fine, L, cosmo, k_gate)
        results["floors"]["mesh"] = c
        print(
            f"    mesh floor {n_fine} vs {n_fine // 2} (K={k_head}): "
            f"dP/P {c['real']['max_dP_P_gate']:.3e}  dP0/P0 {c['rsd']['max_dP0_P0_gate']:.3e}  "
            f"dP2/P0 {c['rsd']['max_dP2_P0_gate']:.3e}"
        )

    print("\n--- codec arms vs ref (same K; gate band) ---")
    for arm in CODEC_ARMS:
        for k in ks:
            sp = state_path(args.config, arm, k, n_fine)
            if not os.path.exists(sp) or not os.path.exists(ref_path(k)):
                continue
            c = compare_pair(ref_path(k), sp, n_fine, L, cosmo, k_gate)
            results["arms"][f"{arm}_k{k}"] = c
            w = runs.get((arm, k, n_fine), {})
            extra = f"  outliers {w['outlier_frac']:.2e}" if "outlier_frac" in w else ""
            print(
                f"    {arm:6s} K={k:<3} dP/P {c['real']['max_dP_P_gate']:.3e}  "
                f"dP0/P0 {c['rsd']['max_dP0_P0_gate']:.3e}  "
                f"dP2/P0 {c['rsd']['max_dP2_P0_gate']:.3e}{extra}"
            )

    print("\nVerdict vs the mesh floor is JC's call (D-v2-8 bar); nothing self-ratified.")
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "g2c_results.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
