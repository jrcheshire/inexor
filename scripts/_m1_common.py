"""Shared M1 parity-harness module: matched configs, npz schema, neutral estimator.

Imported by ALL THREE sides of the M1 cross-check (inexor / mbody / DISCO-DJ
harness scripts, each running in its own pixi env), so it must stay
dependency-light: NUMPY ONLY -- no jax, no mlx, no inexor imports. The same
pure-numpy CIC + P(k)/r(k) estimator measures every code's final particle
field, which makes the comparison estimator-neutral (any difference is the
field, not the measurement). Pattern ported from
disco-mocks/discomocks/xcheck.py (the validated mbody <-> DISCO-DJ
matched-phase cross-check).

npz exchange schema (all arrays float64, C-order):
  x       : (N^3, 3) positions, Mpc/h, wrapped into [0, L)
  v_d     : (N^3, 3) D-time velocities dx/dD, Mpc/h (inexor lpt convention;
            mbody momentum p = G_f(a) v_d; DISCO-DJ vel = v_d * Fplus(a))
  delta0  : (N, N, N) z=0 linear density (IC files only)
  a_steps : (K+1,) shared step boundaries (every code integrates on these)
  meta    : json string -- code, mode, git rev, timestamp, config, cosmology,
            G_f scalars (provenance; parsed back by load_state)
"""

import json
import os
import subprocess
import time as _time

import numpy as np

# ----- matched run configuration ---------------------------------------------

# Planck-2018-flavoured flat LCDM -- the shared default of inexor.Cosmology and
# mbody.Cosmology (field-identical), and the source dict for DISCO-DJ below.
COSMO = dict(Omega_m=0.31, Omega_b=0.049, h=0.677, n_s=0.965, sigma8=0.81)

# DISCO-DJ cosmo dict (same universe). DISCO-DJ's Cosmology is flat by
# construction -- Omega_de derives from closure (1 - Omega_c - Omega_b), so it
# must NOT be passed (xcheck.py convention, verified).
COSMO_DISCO = dict(
    Omega_c=COSMO["Omega_m"] - COSMO["Omega_b"],
    Omega_b=COSMO["Omega_b"],
    h=COSMO["h"],
    sigma8=COSMO["sigma8"],
    n_s=COSMO["n_s"],
    Omega_k=0.0,
    w0=-1.0,
    wa=0.0,
)

BOX_SIZE = 256.0  # Mpc/h, all M1 parity configs
A_INIT = 0.1
A_FINAL = 1.0

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.normpath(os.path.join(HERE, "..", "runs", "m1"))


def run_tag(n_mesh, n_steps, spacing="log", integrator="bullfrog", lpt_order=2, seed=0):
    """Canonical file tag for one matched configuration."""
    return f"n{n_mesh}k{n_steps}{spacing[:3]}_{integrator}_lpt{lpt_order}_s{seed}"


def a_grid(a_init=A_INIT, a_final=A_FINAL, n_steps=10, spacing="log"):
    """Shared step boundaries (numpy float64). Replicates inexor.integrate.a_grid
    BITWISE (np.geomspace, not exp-of-linspace -- they differ at the ulp level);
    the export script asserts the two agree exactly. The npz a_steps array is
    the single source of truth every code integrates on."""
    if spacing == "linear":
        return np.linspace(a_init, a_final, n_steps + 1)
    if spacing == "log":
        return np.geomspace(a_init, a_final, n_steps + 1)
    raise ValueError(f"spacing must be 'linear' or 'log', got {spacing!r}")


# ----- npz exchange -----------------------------------------------------------


def git_rev(repo_dir):
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def make_meta(code, mode, config, repo_dir, **extra):
    meta = dict(
        code=code,
        mode=mode,
        git=git_rev(repo_dir),
        created=_time.strftime("%Y-%m-%d %H:%M:%S"),
        config=config,
        cosmo=COSMO,
    )
    meta.update(extra)
    return meta


def save_state(path, x, v_d, meta, delta0=None, a_steps=None):
    """Write one exchange npz (see module docstring for the schema)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = dict(
        x=np.ascontiguousarray(np.asarray(x, dtype=np.float64)),
        v_d=np.ascontiguousarray(np.asarray(v_d, dtype=np.float64)),
        meta=np.array(json.dumps(meta)),
    )
    if delta0 is not None:
        arrays["delta0"] = np.ascontiguousarray(np.asarray(delta0, dtype=np.float64))
    if a_steps is not None:
        arrays["a_steps"] = np.asarray(a_steps, dtype=np.float64)
    np.savez_compressed(path, **arrays)
    return path


def load_state(path):
    """Read one exchange npz -> dict of arrays + parsed meta."""
    with np.load(path) as f:
        out = {k: f[k] for k in f.files if k != "meta"}
        out["meta"] = json.loads(str(f["meta"]))
    return out


# ----- neutral estimator (xcheck.py port, verbatim math) ----------------------


def cic_paint(pos, n, L):
    """Periodic CIC paint of (n_part, 3) positions (Mpc/h) -> overdensity.

    Real (n, n, n) float64 density contrast (mean 0); np.bincount over the 8
    corners; periodicity by index mod n (positions need not be pre-wrapped).
    """
    pos = np.asarray(pos, dtype=np.float64)
    npart = pos.shape[0]
    g = pos / (L / n)
    i0 = np.floor(g).astype(np.int64)
    frac = g - i0
    rho = np.zeros(n * n * n, dtype=np.float64)
    for dx in (0, 1):
        wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
        ix = (i0[:, 0] + dx) % n
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
            iy = (i0[:, 1] + dy) % n
            for dz in (0, 1):
                wz = frac[:, 2] if dz else 1.0 - frac[:, 2]
                iz = (i0[:, 2] + dz) % n
                w = wx * wy * wz
                flat = (ix * n + iy) * n + iz
                rho += np.bincount(flat, weights=w, minlength=n * n * n)
    rho *= n * n * n / npart
    return rho.reshape(n, n, n) - 1.0


def measure(delta_a, delta_b, n, L):
    """Shared P(k)/cross estimator on two real fields.

    Returns (k_centers, P_a, P_b, P_cross, r), binned in fundamental-width |k|
    shells from 0.5 k_f, k=0 excluded. r = P_x / sqrt(P_a P_b) is the primary
    (normalization-independent) cross-check metric.
    """
    dk_a = np.fft.rfftn(delta_a)
    dk_b = np.fft.rfftn(delta_b)
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=L / n)
    kmag = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2).ravel()
    norm = L**3 / n**6
    pa = (np.abs(dk_a) ** 2 * norm).ravel()
    pb = (np.abs(dk_b) ** 2 * norm).ravel()
    px = (np.real(dk_a * np.conj(dk_b)) * norm).ravel()

    kf = 2.0 * np.pi / L
    knyq = np.pi * n / L
    edges = np.arange(0.5 * kf, knyq + kf, kf)
    kc = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(kmag, edges) - 1
    nb = len(kc)
    sel = (idx >= 0) & (idx < nb)
    cnt = np.bincount(idx[sel], minlength=nb).astype(np.float64)
    cnt = np.maximum(cnt, 1.0)

    def binit(x):
        return np.bincount(idx[sel], weights=x[sel], minlength=nb) / cnt

    PA, PB, PX = binit(pa), binit(pb), binit(px)
    r = PX / np.sqrt(np.maximum(PA * PB, 1e-300))
    return kc, PA, PB, PX, r


def pk(delta, n, L):
    """Auto power of one field via the shared estimator: (k_centers, P)."""
    kc, PA, _, _, _ = measure(delta, delta, n, L)
    return kc, PA


def cic_window_modes(n, L):
    """Field-level CIC window on the rfft grid: W(k) = prod_i sinc^2(k_i d / (2 pi)).

    A single painted field deconvolves as delta_k / W (P(k) then gains 1/W^2);
    numpy sinc(x) = sin(pi x)/(pi x), so sinc(k d / (2 pi)) = sin(k d/2)/(k d/2)."""
    d = L / n
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=d)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=d)

    def s(k1d):
        return np.sinc(k1d * d / (2.0 * np.pi))

    return s(kx)[:, None, None] ** 2 * s(kx)[None, :, None] ** 2 * s(kz)[None, None, :] ** 2


def pk_deconvolved(delta, n, L):
    """Auto power with single-paint CIC deconvolution (delta_k / W)."""
    dk = np.fft.rfftn(delta) / cic_window_modes(n, L)
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=L / n)
    kmag = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2).ravel()
    norm = L**3 / n**6
    pa = (np.abs(dk) ** 2 * norm).ravel()
    kf = 2.0 * np.pi / L
    knyq = np.pi * n / L
    edges = np.arange(0.5 * kf, knyq + kf, kf)
    kc = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(kmag, edges) - 1
    nb = len(kc)
    sel = (idx >= 0) & (idx < nb)
    cnt = np.maximum(np.bincount(idx[sel], minlength=nb).astype(np.float64), 1.0)
    return kc, np.bincount(idx[sel], weights=pa[sel], minlength=nb) / cnt


# ----- comparison metrics -----------------------------------------------------


def min_image_dx(x_a, x_b, L):
    """Minimum-image displacement field between two position sets, (n, 3)."""
    d = np.asarray(x_a, np.float64) - np.asarray(x_b, np.float64)
    return d - L * np.round(d / L)


def compare_states(x_a, x_b, n_mesh, L):
    """Full parity readout between two final particle-position sets.

    Returns a json-ready dict: min-image displacement stats (Mpc/h and in
    cells), P(k) ratio stats, and the cross-correlation r(k), plus the binned
    curves for plotting.
    """
    d = min_image_dx(x_a, x_b, L)
    rms = float(np.sqrt(np.mean(d**2)))
    max_abs = float(np.max(np.abs(d)))
    cell = L / n_mesh
    da = cic_paint(x_a, n_mesh, L)
    db = cic_paint(x_b, n_mesh, L)
    kc, PA, PB, PX, r = measure(da, db, n_mesh, L)
    ratio = PB / np.maximum(PA, 1e-300)
    return dict(
        rms_mpch=rms,
        rms_cells=rms / cell,
        max_abs_mpch=max_abs,
        max_abs_cells=max_abs / cell,
        ratio_max_absdev=float(np.max(np.abs(ratio - 1.0))),
        r_min=float(np.min(r)),
        one_minus_r_max=float(np.max(1.0 - r)),
        k=kc.tolist(),
        p_ratio=ratio.tolist(),
        r=r.tolist(),
    )


def repro_floor(states, n_mesh, L):
    """Repro-floor readout: pairwise compare_states of runs 1.. against run 0.

    `states` is a list of (n, 3) position arrays from REPEATED identical-input
    runs of one code. Returns the per-pair dicts plus scalar worst-case
    summaries -- the code's own repeatability floor, measured BEFORE any parity
    gate is proposed (floor-first protocol, plan S5).
    """
    pairs = [compare_states(states[0], s, n_mesh, L) for s in states[1:]]
    if not pairs:
        return dict(n_repeats=len(states), pairs=[])
    return dict(
        n_repeats=len(states),
        worst_rms_cells=max(p["rms_cells"] for p in pairs),
        worst_max_abs_cells=max(p["max_abs_cells"] for p in pairs),
        worst_ratio_absdev=max(p["ratio_max_absdev"] for p in pairs),
        worst_one_minus_r=max(p["one_minus_r_max"] for p in pairs),
        pairs=pairs,
    )
