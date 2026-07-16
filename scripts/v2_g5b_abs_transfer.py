"""G5b: the C-dev fine mesh's ABSOLUTE transfer, on a ladder that is valid.

WHY THIS EXISTS. Job 39's mesh floor (`v2_g5_two_level_force.py`, evolve arm)
refined the MESH at FIXED particles: 256^3 particles seen by a 256 / 512 / 1024
mesh. Its (kc)^2 scaling check FAILED -- ratio 1.35 (force) / 0.76 (evolved)
against an expected ~4 -- and the reason is structural, not a bug:

    e(256 vs 512) = 0.050  <  e(512 vs 1024) = 0.066

the discrepancy GREW under refinement. Mesh truncation error cannot do that.
At n_mesh=1024 the cell is 0.125 against a particle spacing of 0.5, so that
rung puts four cells inside one interparticle gap and starts resolving two-body
scattering the 512 run smooths over. It measures the onset of discreteness, not
convergence. So the floor is not a floor, and D-v2-1's "below the mesh floor"
framing has no bar behind it (see runs/v2/g5_cdev_band.png).

WHAT THIS MEASURES INSTEAD. A ladder at FIXED mesh:particle = 2 -- C-dev's own
ratio -- so every rung sits in the same discreteness regime and only the
resolution changes:

    r128   128^3 particles / 256  mesh   cell 0.500
    r256   256^3 particles / 512  mesh   cell 0.250   <- C-dev
    r512   512^3 particles / 1024 mesh   cell 0.125   <- the reference

matched phase (below), evolved on the SAME BullFrog schedule, painted to ONE
common mesh so the CIC window cancels in every ratio. Then:

  * dP/P(r256 vs r512) = C-dev's error vs a 2x-better-resolved sim. This is
    the number JC asked for. The reference is NOT converged -- its own cell is
    0.125 -- so taken raw this is a LOWER BOUND on C-dev's absolute error.
  * the (kc)^2 check across the three rungs: e(r128 vs r256) / e(r256 vs r512)
    ~ 4 iff the error is mesh-truncation-dominated. THIS is the check job 39's
    ladder failed; if it passes here, the ladder is valid and Richardson turns
    the lower bound into an estimate: with e ~ (kc)^2, halving the cell cuts
    the error 4x, so e(C-dev) = (4/3) * e(r256 vs r512).
    If it FAILS here too, we report the lower bound and say so.

MATCHED PHASE IS NOT FREE. ic.gaussian_delta draws white noise as
jax.random.normal(key, (N,N,N)) -- the RNG stream depends on SHAPE, so the same
key at 128^3 and 512^3 gives unrelated fields. We draw ONE white-noise field at
the finest particle grid and band-limit it to each coarser grid, which makes the
shared Fourier modes of the coloured field bit-identical across rungs (see
_white_truncated for the (N_lo/N_hi)^{3/2} normalization and its derivation).
Cosmic variance then cancels in every ratio.

SHOT NOISE is the known limitation: the rungs carry different particle loads, so
they carry different discreteness power. Grid ICs are sub-Poisson, so the Poisson
level L^3/n_part^3 is an OVER-subtraction, not a correction. We report the ratio
BOTH raw and Poisson-subtracted; the truth is bracketed between them. If the
bracket is wide compared to the split error, the measurement does not support a
tolerance and says so rather than pretending.

Precision: every leg is f32. The 1024 rung cannot be f64 on deneb (~52 GB
against a 56 GB host); F2 in job 39 put f32-vs-f64 at 5.5e-7, six orders under
what we are measuring, and running ALL rungs f32 keeps it common-mode.

Usage:
    python scripts/v2_g5b_abs_transfer.py --config smoke --checks-only  # laptop
    python scripts/v2_g5b_abs_transfer.py --config smoke                # laptop
    sbatch scripts/v2_g5b_deneb.sbatch                                  # C-dev
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "runs", "v2")
STATE_DIR = os.path.join(OUT_DIR, "g5b_states")
ERR_DIR = os.path.join(OUT_DIR, "g5b_errs")

# Ladders at FIXED mesh:particle = 2. n_hi = the finest particle grid = the
# white-noise field every rung is band-limited from. n_paint = the ONE mesh all
# spectra are measured on (2x the coarsest particle grid, so k_gate sits well
# inside its Nyquist and the CIC window cancels in the ratios).
CONFIGS = {
    "cdev": dict(L=128.0, rungs=[(128, 256), (256, 512), (512, 1024)], n_paint=256),
    "cdev8": dict(L=64.0, rungs=[(64, 128), (128, 256), (256, 512)], n_paint=128),
    "smoke": dict(L=32.0, rungs=[(16, 32), (32, 64), (64, 128)], n_paint=32),
}

SEED = 0
A_INIT = 0.1  # G5's A_CONTROL: the schedule job 39's evolve arm used
A_FINAL = 1.0  # G5's A_PIVOT
N_STEPS = 20  # G5's N_STEPS
TARGET = 1  # index into rungs: the rung that IS C-dev
REF = 2  # index into rungs: the reference


def geometry(cfg):
    c = CONFIGS[cfg]
    L = c["L"]
    n_part_t, n_mesh_t = c["rungs"][TARGET]
    d_f = L / n_mesh_t
    return dict(
        L=L,
        rungs=c["rungs"],
        n_paint=c["n_paint"],
        n_hi=max(p for p, _ in c["rungs"]),
        target_cell=d_f,
        # D-v2-1's band, defined by the TARGET rung's fine mesh -- identical to
        # geometry()["k_gate"] in v2_g5_two_level_force.py, so the dP/P numbers
        # here and there are barred on the same band.
        k_gate=0.2 * np.pi / d_f,
        k_nyq_paint=np.pi * c["n_paint"] / L,
    )


def rung_name(n_part, n_mesh):
    return f"r{n_part}_m{n_mesh}"


# ===========================================================================
# matched-phase ICs
# ===========================================================================


def _white_truncated(white_hi, n_lo):
    """Band-limit a real unit-variance white-noise field n_hi^3 -> n_lo^3.

    The point is that the COLOURED field's shared modes come out bit-identical
    across rungs. With numpy's unnormalized fftn and the ic.gaussian_delta
    convention delta_k = fftn(white) * sqrt(P N^3/L^3):

        delta_lo(x) = sum_k W_lo(k) sqrt(P/L^3) N_lo^{-3/2} e^{ikx}
        delta_hi(x) = sum_k W_hi(k) sqrt(P/L^3) N_hi^{-3/2} e^{ikx}

    so the two real-space fields agree on shared k iff

        fftn(white_lo)[k] = fftn(white_hi)[k] * (N_lo/N_hi)^{3/2},

    which is what the trailing factor applies. (Equivalently: white noise has
    unit variance PER CELL, so a coarser grid needs its modes scaled to keep
    that; the exponent is 3/2 because variance goes as the mode amplitude
    squared.)

    The coarse Nyquist planes are ZEROED. A one-sided slice [-h, h) keeps
    k = -h but not +h, which breaks the Hermitian symmetry a real field needs;
    rather than fold them (they alias onto each other) we drop them. They sit AT
    the coarse Nyquist -- for C-dev's r128 rung that is k = 3.14 h/Mpc against a
    k_gate of 2.51, and the r128 rung only ever feeds the (kc)^2 check.
    """
    n_hi = white_hi.shape[0]
    if n_lo > n_hi:
        raise ValueError(f"cannot band-limit {n_hi}^3 up to {n_lo}^3")
    if n_lo == n_hi:
        return np.array(white_hi, dtype=np.float64, copy=True)
    W = np.fft.fftshift(np.fft.fftn(white_hi))
    c, h = n_hi // 2, n_lo // 2
    sl = slice(c - h, c + h)
    Wl = W[sl, sl, sl].copy()
    Wl[0, :, :] = 0.0
    Wl[:, 0, :] = 0.0
    Wl[:, :, 0] = 0.0
    white_lo = np.fft.ifftn(np.fft.ifftshift(Wl)).real
    return white_lo * (n_lo / n_hi) ** 1.5


def _colour(white, n_mesh, L, cosmo):
    """delta from a GIVEN white-noise field -- ic.gaussian_delta's convention.

    Mirrors ic.gaussian_delta (ic.py:34-62) exactly, host f64, with the white
    noise injected rather than drawn. check_identity() asserts it reproduces
    ic.gaussian_delta bit-for-bit at the same key, which is what licenses the
    mirror.
    """
    from inexor.cosmology import linear_power

    N = n_mesh
    dk = np.fft.rfftn(white)
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=L / N)
    kk = np.sqrt(kx.reshape(N, 1, 1) ** 2 + kx.reshape(1, N, 1) ** 2 + kz.reshape(1, 1, -1) ** 2)
    kk_safe = kk.copy()
    kk_safe[0, 0, 0] = kk.flat[1]
    colour = np.sqrt(linear_power(kk_safe.ravel(), cosmo).reshape(kk.shape) * N**3 / L**3)
    colour[0, 0, 0] = 0.0
    return np.fft.irfftn(dk * colour, s=(N, N, N))


def _white_hi(n_hi):
    """The ONE white-noise field every rung derives from (host f64)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    return np.asarray(jax.random.normal(jax.random.PRNGKey(SEED), (n_hi,) * 3, dtype="float64"))


def matched_delta(n_part, n_hi, L, cosmo):
    return _colour(_white_truncated(_white_hi(n_hi), n_part), n_part, L, cosmo)


# ===========================================================================
# checks (these HARD-FAIL; a check that does not run reads as one that passed)
# ===========================================================================


def check_identity(cfg):
    """Two properties, both asserted, before any physics leg is allowed to run.

    1. DEGENERATE LIMIT: at n_lo == n_hi the whole matched-phase path collapses
       to ic.gaussian_delta at the same key. If this drifts, _colour has
       silently forked from the package convention and every rung is measuring
       a different cosmology than the repo's other results.
    2. SHARED MODES: for n_lo < n_hi the coloured fields agree, mode by mode,
       strictly inside the coarse Nyquist. This is matched phase itself -- if it
       fails, the ratios carry cosmic variance and the whole measurement is
       noise.
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from inexor import ic
    from inexor.config import Cosmology

    g = geometry(cfg)
    L, n_hi, cosmo = g["L"], g["n_hi"], Cosmology()
    out = {}

    white = _white_hi(n_hi)
    mine = _colour(white, n_hi, L, cosmo)
    theirs = np.asarray(
        ic.gaussian_delta(jax.random.PRNGKey(SEED), n_hi, L, cosmo, fdtype=jnp.float64)
    )
    scale = float(np.abs(theirs).max())
    d_ident = float(np.abs(mine - theirs).max()) / scale
    out["degenerate_identity"] = dict(max_rel=d_ident, ok=bool(d_ident < 1e-12))

    # numpy's rfftn is UNNORMALIZED, so a coefficient carries a factor N^3 that
    # is pure grid convention, not physics: comparing raw coefficients across
    # resolutions fails by exactly (N_lo/N_hi)^3 even when the fields are
    # identical. Divide each by its own N^3 to compare physical amplitudes.
    d_hi = np.fft.rfftn(mine) / n_hi**3
    shared = {}
    for n_part, _ in g["rungs"]:
        if n_part == n_hi:
            continue
        d_lo = np.fft.rfftn(_colour(_white_truncated(white, n_part), n_part, L, cosmo)) / n_part**3
        # compare on the coarse grid's modes STRICTLY inside its Nyquist
        h = n_part // 2
        idx = np.concatenate([np.arange(1, h), np.arange(-h + 1, 0)])
        sub_lo = d_lo[np.ix_(idx, idx, np.arange(1, h))]
        sub_hi = d_hi[np.ix_(idx, idx, np.arange(1, h))]
        rel = float(np.abs(sub_lo - sub_hi).max() / np.abs(sub_hi).max())
        shared[f"r{n_part}"] = dict(max_rel=rel, ok=bool(rel < 1e-10))
    out["shared_modes"] = shared

    out["all_ok"] = bool(out["degenerate_identity"]["ok"] and all(v["ok"] for v in shared.values()))
    return out


# ===========================================================================
# legs
# ===========================================================================


def _state_path(cfg, n_part, n_mesh, when):
    return os.path.join(STATE_DIR, f"{cfg}_{rung_name(n_part, n_mesh)}_{when}.npz")


def config_leg(args):
    """Matched-phase 2LPT ICs for ONE rung, CPU f64, saved for the evolve leg."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from inexor import lpt
    from inexor.config import Cosmology

    g = geometry(args.config)
    cosmo = Cosmology()
    d0 = matched_delta(args.n_part, g["n_hi"], g["L"], cosmo)
    x, v = lpt.lpt_ics(jnp.asarray(d0), g["L"], A_INIT, cosmo, order=2, fdtype=jnp.float64)
    os.makedirs(STATE_DIR, exist_ok=True)
    path = _state_path(args.config, args.n_part, args.n_mesh, "ic")
    np.savez(path, x=np.asarray(x, np.float64), v=np.asarray(v, np.float64))
    return dict(kind="config", n_part=args.n_part, path=path, delta_std=float(d0.std()))


def evolve_leg(args):
    """Evolve ONE rung with the monolithic force at its own mesh."""
    import jax
    import jax.numpy as jnp

    from inexor.config import Cosmology
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table, float_step_bullfrog

    sys.path.insert(0, HERE)
    from v2_g5_core import force_global

    g = geometry(args.config)
    L = g["L"]
    n_total = args.n_part**3
    cosmo = Cosmology()

    with np.load(_state_path(args.config, args.n_part, args.n_mesh, "ic")) as f:
        x_np, v_np = f["x"], f["v"]

    def force_mono(pos):
        out, _ = force_global(pos, args.n_mesh, L, n_total, "mono", assign="cic")
        return jnp.asarray(out)

    a_steps = a_grid(A_INIT, A_FINAL, N_STEPS, "log")
    coeffs = bullfrog_float_coeffs(bullfrog_table(a_steps, cosmo))
    x = jnp.asarray(x_np, jnp.float32)
    v = jnp.asarray(v_np, jnp.float32)
    t0 = time.perf_counter()
    for c in coeffs:
        x, v = float_step_bullfrog(x, v, tuple(np.asarray(c, np.float64)), force_mono, L)
        x.block_until_ready()
    wall = time.perf_counter() - t0

    path = _state_path(args.config, args.n_part, args.n_mesh, "final")
    np.savez(path, x=np.asarray(x, np.float64))
    return dict(
        kind="evolve",
        n_part=args.n_part,
        n_mesh=args.n_mesh,
        wall_s=wall,
        wall_per_step=wall / N_STEPS,
        state_file=path,
        platform=jax.devices()[0].platform,
    )


# ===========================================================================
# spectra
# ===========================================================================


def _k_grid(n, L):
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=L / n)
    return np.sqrt(kx.reshape(n, 1, 1) ** 2 + kx.reshape(1, n, 1) ** 2 + kz.reshape(1, 1, -1) ** 2)


def _bin_edges(n, L):
    kf = 2.0 * np.pi / L
    return np.arange(0.5, n // 2 + 1) * kf


def pk_of(x, n, L):
    """P(k) of a particle set on the COMMON paint mesh, in (Mpc/h)^3.

    _m1_common.cic_paint is the estimator every prior parity claim in this repo
    used (D-013); it returns a density CONTRAST. The L^3/n^6 factor converts
    numpy's unnormalized rfftn of that contrast into a physical P(k) -- without
    it P carries an arbitrary grid normalization and the Poisson shot level
    (which is genuinely in (Mpc/h)^3) is not commensurate with it, so
    subtracting it silently does nothing.
    """
    sys.path.insert(0, HERE)
    import _m1_common as M

    kmag = _k_grid(n, L).ravel()
    edges = _bin_edges(n, L)
    counts = np.histogram(kmag, bins=edges)[0]
    p = np.histogram(
        kmag, bins=edges, weights=(np.abs(np.fft.rfftn(M.cic_paint(x, n, L))) ** 2).ravel()
    )[0]
    good = counts > 0
    kc = 0.5 * (edges[1:] + edges[:-1])
    return kc[good], p[good] / counts[good] * L**3 / n**6, counts[good]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="smoke", choices=sorted(CONFIGS))
    ap.add_argument("--kind", default=None, choices=["config", "evolve"])
    ap.add_argument("--n-part", type=int, default=None)
    ap.add_argument("--n-mesh", type=int, default=None)
    ap.add_argument("--checks-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.kind == "config":
        print(json.dumps(config_leg(args)))
        return
    if args.kind == "evolve":
        print(json.dumps(evolve_leg(args)))
        return

    g = geometry(args.config)
    L, rungs = g["L"], g["rungs"]
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ERR_DIR, exist_ok=True)

    print(f"=== G5b absolute transfer: config {args.config}", flush=True)
    print(
        f"    L {L}  n_hi {g['n_hi']}^3  paint {g['n_paint']}^3  "
        f"k_gate {g['k_gate']:.3f}  (k_nyq_paint {g['k_nyq_paint']:.2f})",
        flush=True,
    )
    for i, (np_, nm) in enumerate(rungs):
        tags = {TARGET: "  <- C-dev (target)", REF: "  <- reference"}.get(i, "")
        print(f"    rung {rung_name(np_, nm):12s} cell {L / nm:.4f}{tags}", flush=True)

    print("\n--- checks (hard-fail; nothing else runs if these do not pass) ---", flush=True)
    checks = check_identity(args.config)
    print(
        f"  degenerate identity (_colour == ic.gaussian_delta) : "
        f"{checks['degenerate_identity']['max_rel']:.3e}  "
        f"{'OK' if checks['degenerate_identity']['ok'] else 'FAIL'}",
        flush=True,
    )
    for k, v in checks["shared_modes"].items():
        print(
            f"  shared modes {k:6s} vs r{g['n_hi']:<6d}                    : "
            f"{v['max_rel']:.3e}  {'OK' if v['ok'] else 'FAIL'}",
            flush=True,
        )
    if not checks["all_ok"]:
        print("\nCHECKS FAILED -- matched phase is not established; refusing to run.", flush=True)
        sys.exit(1)
    if args.checks_only:
        return

    def spawn(kind, n_part, n_mesh):
        tag = f"{kind}_{rung_name(n_part, n_mesh)}"
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--config",
            args.config,
            "--kind",
            kind,
            "--n-part",
            str(n_part),
            "--n-mesh",
            str(n_mesh),
        ]
        env = dict(os.environ)
        env["JAX_PLATFORMS"] = "cpu"
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if p.returncode != 0:
            err = os.path.join(ERR_DIR, tag + ".err")
            with open(err, "w") as f:
                f.write(p.stdout + "\n===== STDERR =====\n" + p.stderr)
            print(f"  {tag}: FAILED (rc {p.returncode}) -> {err}", flush=True)
            tail = [ln for ln in p.stderr.strip().splitlines() if ln.strip()][-3:]
            for ln in tail:
                print(f"      {ln}", flush=True)
            sys.exit(1)
        rec = json.loads(p.stdout.strip().splitlines()[-1])
        print(f"  {tag}: {time.perf_counter() - t0:.1f}s", flush=True)
        return rec

    print("\n--- ICs (matched phase) ---", flush=True)
    for np_, nm in rungs:
        spawn("config", np_, nm)

    print("\n--- evolve (mono force at each rung's own mesh, f32) ---", flush=True)
    ev = {}
    for np_, nm in rungs:
        ev[rung_name(np_, nm)] = spawn("evolve", np_, nm)

    print("\n--- spectra (all painted to the SAME mesh) ---", flush=True)
    n_paint = g["n_paint"]
    pk, shot = {}, {}
    for np_, nm in rungs:
        nm_ = rung_name(np_, nm)
        with np.load(_state_path(args.config, np_, nm, "final")) as f:
            x = f["x"]
        kc, p, _ = pk_of(x, n_paint, L)
        pk[nm_] = p
        shot[nm_] = L**3 / np_**3
    band = kc <= g["k_gate"]

    def ratio(lo, hi, sub):
        a = pk[lo] - (shot[lo] if sub else 0.0)
        b = pk[hi] - (shot[hi] if sub else 0.0)
        r = a / b - 1.0
        return r, float(np.abs(r[band]).max())

    names = [rung_name(*r) for r in rungs]
    res = dict(
        config=args.config,
        geometry={k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in g.items()},
        checks=checks,
        evolve=ev,
        shot_poisson={k: float(v) for k, v in shot.items()},
        k=kc.tolist(),
        pk={k: v.tolist() for k, v in pk.items()},
    )

    print("\n  Poisson shot levels (Mpc/h)^3 -- an OVER-subtraction (grid ICs):", flush=True)
    for n in names:
        print(f"    {n:12s} {shot[n]:.4f}", flush=True)

    print(f"\n  in-band |dP/P| (k <= {g['k_gate']:.3f})            raw     shot-sub", flush=True)
    pairs = [
        ("coarse_vs_target", names[0], names[TARGET]),
        ("target_vs_ref", names[TARGET], names[REF]),
    ]
    for label, lo, hi in pairs:
        r_raw, m_raw = ratio(lo, hi, False)
        r_sub, m_sub = ratio(lo, hi, True)
        res[label] = dict(
            lo=lo,
            hi=hi,
            dP_over_P_raw=r_raw.tolist(),
            dP_over_P_shotsub=r_sub.tolist(),
            max_abs_gate_raw=m_raw,
            max_abs_gate_shotsub=m_sub,
        )
        print(f"    {label:18s} {lo:>11s} vs {hi:<11s} {m_raw:.3e}  {m_sub:.3e}", flush=True)

    e1_raw = res["coarse_vs_target"]["max_abs_gate_raw"]
    e2_raw = res["target_vs_ref"]["max_abs_gate_raw"]
    e1_sub = res["coarse_vs_target"]["max_abs_gate_shotsub"]
    e2_sub = res["target_vs_ref"]["max_abs_gate_shotsub"]
    sc = dict(
        ratio_raw=float(e1_raw / e2_raw) if e2_raw else None,
        ratio_shotsub=float(e1_sub / e2_sub) if e2_sub else None,
        expect="~4 if mesh-truncation-dominated ((kc)^2 law)",
        ok=bool(e2_raw and 2.5 <= e1_raw / e2_raw <= 6.0),
    )
    res["kc2_scaling_check"] = sc
    print(
        f"\n  (kc)^2 scaling check: ratio {sc['ratio_raw']:.2f} raw / "
        f"{sc['ratio_shotsub']:.2f} shot-sub  (expect ~4)  "
        f"{'OK -- ladder is valid' if sc['ok'] else 'FAIL -- ladder still contaminated'}",
        flush=True,
    )

    # Richardson: e ~ (kc)^2 => halving the cell cuts the error 4x, so the
    # measured target-vs-ref difference is (1 - 1/4) of the target's own error.
    if sc["ok"]:
        for tag, m in (("raw", e2_raw), ("shotsub", e2_sub)):
            res.setdefault("cdev_abs_error", {})[tag] = (4.0 / 3.0) * m
        print(
            f"  Richardson => C-dev's OWN in-band error ~ "
            f"{res['cdev_abs_error']['raw']:.3e} raw / "
            f"{res['cdev_abs_error']['shotsub']:.3e} shot-sub",
            flush=True,
        )
    else:
        res["cdev_abs_error"] = None
        print(
            f"  ladder invalid => target_vs_ref ({e2_raw:.3e}) stands only as a LOWER BOUND",
            flush=True,
        )

    out = args.out or os.path.join(OUT_DIR, f"g5b_abs_transfer_{args.config}.json")
    with open(out, "w") as f:
        json.dump(res, f)
    print(f"\nwrote {out}", flush=True)
    print("\nTolerance is JC's call; nothing self-ratified.", flush=True)


if __name__ == "__main__":
    main()
