"""v2 gate G5: two-level (coarse long + fine tile short) PM force vs monolithic.

The question (plan-plan Sec 5, seed V2 / plan V2a): can a global COARSE mesh plus
a FINE mesh in streamed tiles reproduce the monolithic fine-mesh force well
enough to sit under the PM error floor? If yes the fine mesh leaves the memory
budget and halo-grade resolution becomes affordable -- M3's premise, A2's spine.
G5 also unblocks the C-dev G2c rerun (that gate fell back to C-dev/8 because the
MONOLITHIC 512^3 force does not fit deneb's 6 GB, and t9/t12 are degenerate
there), and feeds D-v2-8 item 4's open mesh:particle ratio.

Physics lives in v2_g5_core.py. Read runs/v2/g5_kernel_findings.md before
touching the arms -- the kernel study is already done and it inverted two design
assumptions:
  - the Gaussian family tiles as ~2.0/P: THE BUFFER IS NOT A KNOB, the tile size
    is. So beta = b/r_s is not a scan axis for it; P is.
  - kernel matching is FAMILY-SPECIFIC: mandatory for gauss, harmful for the
    windowed families.

THE LADDER (floors first; each rung differs from the one above by ONE mechanism):

    F0    probe vs make_force_fn                -> probe reimplementation error
    F1    long + short vs mono, on fine         -> the split kernel itself
    tileid one tile = whole box, b=0            -> the tile machinery itself
    F2    mono f32 vs mono f64                  -> float precision
    F3    int paint vs f64 paint                -> paint quantization
    F4    mono fine vs fine/2 vs fine/4         -> THE PM MESH FLOOR (the bar)
    C1-C5 coarse variants vs long_fine          -> coarse aliasing / window
    tile  short_tiled vs short_fine             -> tiling alone
    total  long_coarse + short_tiled vs mono    -> the headline
    evolve two-level vs mono-512 vs mono-1024   -> dP/P, which is what D-v2-1 BARS

F1 and tileid are this gate's kaiser-check: if either fails, NO other G5 number
may be read. They caught two real bugs during the kernel study (a brick
double-count, and a dropped cell layer that faked the ringing signature).

HONEST ACCOUNTING, all of it echoed into the json:
  - Physics legs are CPU f64. The monolithic f64 512^3 force is only ~7-8 GB
    (the ik_j are low-rank (N,1,1) broadcasts, not full arrays), trivial in
    deneb's 56 GB -- so f32/GPU noise never contaminates attribution.
  - JAX exposes no memory_stats on CPU: physics legs carry a correctness number,
    NOT a capacity number (peak_bytes null, with peak_valid_reason).
  - GPU legs exist ONLY for the (peak B/p, wall) triple. mono f32 at C-dev is
    EXPECTED to OOM the 6 GB card -- that is the capacity result ("monolithic
    does not fit; tiled does"), not a crash.
  - Host-side brick bucketing sits OUTSIDE the reported wall; bucketing in XLA
    is an M-v2-2 build item (the design study: cell bucketing fights XLA static
    shapes).
  - Force error spectra are CIC-painted: the window cancels in the RATIO only to
    first order.
  - SU is n/a on deneb; it becomes real at V3 on Vista (cost_of_memory.md).

Orchestration: one leg per FRESH SUBPROCESS (peak_bytes_in_use is monotonic with
no reset API -- retrospective Sec 5). Workers save a force/state to
runs/v2/g5_forces/ and exit; the CPU-pinned orchestrator loads pairs afterwards
and computes EVERY statistic with one estimator stack. No measured process ever
holds a reference (the V1 512:512 leg was declared "not a clean capacity
statement" for exactly that reason).

Run (deneb via Slurm, scripts/v2_g5_deneb.sbatch):
    pixi run -e gpu python scripts/v2_g5_two_level_force.py --config cdev
CPU smoke (laptop, seconds):
    pixi run python scripts/v2_g5_two_level_force.py --config smoke
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(REPO, "runs", "v2")
STATE_DIR = os.path.join(OUT_DIR, "g5_states")
FORCE_DIR = os.path.join(OUT_DIR, "g5_forces")
ERR_DIR = os.path.join(OUT_DIR, "g5_errs")

# (n_part, L, n_fine, n_coarse). C-dev per D-v2-8's config table; coarse = fine/4
# (PMFAST). cdev8 preserves fine cell AND spacing at 1/8 the volume -- the
# pre-agreed G2c fallback rule -- so every scan point (all in CELLS) transfers
# unchanged. smoke is the laptop path.
CONFIGS = {
    "cdev": dict(n_part=256, L=128.0, n_fine=512, n_coarse=128),
    "cdev8": dict(n_part=128, L=64.0, n_fine=256, n_coarse=64),
    "smoke": dict(n_part=32, L=32.0, n_fine=64, n_coarse=16),
}

SEED = 0
A_PIVOT = 1.0  # clustered + caustic-rich: the short arm actually carries force
A_CONTROL = 0.1  # quasi-linear control: the split error must be trivial here
N_STEPS = 20


def geometry(cfg):
    c = CONFIGS[cfg]
    d_f = c["L"] / c["n_fine"]
    d_c = c["L"] / c["n_coarse"]
    return dict(
        **c,
        fine_cell=d_f,
        coarse_cell=d_c,
        spacing=c["L"] / c["n_part"],
        k_nyq_fine=np.pi / d_f,
        k_nyq_coarse=np.pi / d_c,
        k_gate=0.2 * np.pi / d_f,  # D-v2-1: k <= 0.2 k_Nyq of the FINE mesh
    )


def tag_of(kind, cfg, **kw):
    parts = [kind, cfg]
    for k in (
        "family",
        "which",
        "nmesh",
        "alpha",
        "rout",
        "assign",
        "match",
        "tile",
        "buf",
        "platform",
        "fdtype",
        "paint",
        "a",
        "nsteps",
    ):
        v = kw.get(k)
        if v is not None:
            parts.append(f"{k}{v}")
    return "_".join(str(p) for p in parts).replace(".", "p")


# ===========================================================================
# worker (fresh subprocess per leg)
# ===========================================================================


def _peak_closure(jax):
    dev = jax.devices()[0]

    def peak():
        try:
            return (dev.memory_stats() or {}).get("peak_bytes_in_use")
        except Exception:
            return None

    return dev, peak


def _host_rss_bytes():
    """MIND THE UNIT: ru_maxrss is KiB on Linux, bytes on macOS."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def _load_config_state(cfg, a):
    path = os.path.join(STATE_DIR, f"config_{cfg}_a{a}".replace(".", "p") + ".npz")
    with np.load(path) as f:
        return f["x"], f["v"]


def make_config_leg(args):
    """Generate the 2LPT particle configuration ONCE, CPU f64, and save it.

    Every other leg LOADS this. Regenerating per worker from a PRNGKey -- the
    obvious move, and G2c's own pattern -- is WRONG here: CPU-f64 vs GPU-f32
    gaussian_delta + lpt_ics differ at ~1e-7 in position, exactly the size of the
    precision floor this gate is trying to measure. That contamination would look
    like a real result.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from inexor import ic, lpt
    from inexor.config import Cosmology

    g = geometry(args.config)
    cosmo = Cosmology()
    key = jax.random.PRNGKey(SEED)
    d0 = ic.linear_density(key, g["n_part"], g["L"], cosmo, f_NL=0.0, fdtype=jnp.float64)
    x, v = lpt.lpt_ics(d0, g["L"], args.a, cosmo, order=2, fdtype=jnp.float64)
    os.makedirs(STATE_DIR, exist_ok=True)
    path = os.path.join(STATE_DIR, f"config_{args.config}_a{args.a}".replace(".", "p") + ".npz")
    np.savez(path, x=np.asarray(x, np.float64), v=np.asarray(v, np.float64))
    return dict(kind="config", a=args.a, path=path, n_part=g["n_part"])


def force_leg(args):
    """One force field -> runs/v2/g5_forces/<tag>.npy, plus the triple."""
    import jax

    if args.fdtype == "f64":
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    sys.path.insert(0, HERE)
    from v2_g5_core import force_global, force_short_tiled

    g = geometry(args.config)
    dev, peak = _peak_closure(jax)
    x_np, _ = _load_config_state(args.config, args.a)
    d_f, d_c, L = g["fine_cell"], g["coarse_cell"], g["L"]
    n_total = g["n_part"] ** 3

    r_s = None if args.alpha is None else args.alpha * d_c
    r_out = None if args.rout is None else args.rout * d_c
    r_in = None if r_out is None else args.rin_frac * r_out
    kw = dict(family=args.family, r_s=r_s, r_in=r_in, r_out=r_out)

    t0 = time.perf_counter()
    if args.tile:
        pk = peak()
        g_out, diag = force_short_tiled(
            x_np, g["n_fine"], L, n_total, args.tile, args.buf, peak=peak, **kw
        )
        # force_short_tiled's diag carries its own `family`/`n_tile` keys, which
        # collide with the explicit ones below -> namespace it.
        rec_extra = {("diag_" + k if k in ("family", "n_tile") else k): v for k, v in diag.items()}
        _ = pk
    else:
        pos = jnp.asarray(x_np, dtype=jnp.float64 if args.fdtype == "f64" else jnp.float32)
        match = (d_c, d_f) if args.match else None
        order = 3 if args.assign == "tsc" else 2
        g_out, max_match = force_global(
            pos,
            args.nmesh,
            L,
            n_total,
            args.which,
            n_ref=g["n_fine"],
            match=match,
            clip=args.clip,
            assign=args.assign,
            **kw,
        )
        rec_extra = dict(max_match_applied=max_match, assign_order=order)
    wall = time.perf_counter() - t0

    os.makedirs(FORCE_DIR, exist_ok=True)
    path = os.path.join(FORCE_DIR, args.tag + ".npy")
    np.save(path, np.asarray(g_out, dtype=np.float64))
    return dict(
        kind="force",
        tag=args.tag,
        platform=dev.platform,
        fdtype=args.fdtype,
        family=args.family,
        which=args.which,
        nmesh=args.nmesh,
        alpha=args.alpha,
        rout=args.rout,
        assign=args.assign,
        match=bool(args.match),
        tile=args.tile,
        buf=args.buf,
        a=args.a,
        wall_s=wall,
        force_file=path,
        peak_bytes=peak(),
        peak_valid_reason=(None if dev.platform != "cpu" else "no memory_stats on CPU platform"),
        peak_bytes_per_particle=((peak() / n_total) if peak() else None),
        peak_host_rss_bytes=_host_rss_bytes(),
        rss_units="bytes",
        **rec_extra,
    )


def evolve_leg(args):
    """Evolve with a chosen force -> final state npz.

    THE ARM D-v2-1 ACTUALLY BARS. A single force eval cannot answer the gate: the
    bar is dP/P of an EVOLVED field, and the measured prior (0.33 (k c)^2 power
    loss accumulated over growth history; deconvolution recovers ~1/3) is exactly
    force error integrating into P(k) in a way one evaluation does not reveal.
    Force errors also partly cancel over steps.

    Reuses the package's BullFrog coefficients and float step -- only the force
    differs -- because evolve_float calls make_force_fn internally and has no
    injection point (integrate.py:481). Probe code, not a package change.
    """
    import jax

    # The 1024 mesh floor CANNOT run f64 on deneb: 1024^3 f64 needs ~52 GB
    # (delta 8.6 + dk 8.6 + kernel 8.6 + three g 25.8 + inv_k2 8.6) against a
    # 56 GB host. In f32 it is ~28 GB and fits. That is safe here because F2
    # measured f32-vs-f64 at ~3e-7 -- six orders below the ~1e-1 mesh floor this
    # leg exists to establish -- and the error is common-mode across the pair.
    if args.fdtype == "f64":
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from inexor.config import Cosmology
    from inexor.integrate import a_grid, bullfrog_float_coeffs, bullfrog_table, float_step_bullfrog

    sys.path.insert(0, HERE)
    from v2_g5_core import force_global, force_short_tiled

    g = geometry(args.config)
    L, d_f, d_c = g["L"], g["fine_cell"], g["coarse_cell"]
    n_total = g["n_part"] ** 3
    cosmo = Cosmology()
    x_np, v_np = _load_config_state(args.config, A_CONTROL)
    dev, peak = _peak_closure(jax)

    r_s = None if args.alpha is None else args.alpha * d_c
    r_out = None if args.rout is None else args.rout * d_c
    r_in = None if r_out is None else args.rin_frac * r_out

    def force_mono(pos):
        out, _ = force_global(pos, args.nmesh, L, n_total, "mono", assign="cic")
        return jnp.asarray(out)

    def force_two_level(pos):
        pn = np.asarray(pos, dtype=np.float64)
        gl, _ = force_global(
            jnp.asarray(pn),
            g["n_coarse"],
            L,
            n_total,
            "long",
            family=args.family,
            r_s=r_s,
            r_in=r_in,
            r_out=r_out,
            n_ref=g["n_fine"],
            match=((d_c, d_f) if args.match else None),
            clip=args.clip,
            assign=args.assign,
        )
        gs, _ = force_short_tiled(
            pn,
            g["n_fine"],
            L,
            n_total,
            args.tile,
            args.buf,
            family=args.family,
            r_s=r_s,
            r_in=r_in,
            r_out=r_out,
        )
        return jnp.asarray(gl + gs)

    # Dispatch on the two_level arm explicitly: `arm == "mono"` sent mono_half
    # (a MONO arm at a coarser mesh) down the two-level path, where tile=None
    # blew up as int(None). The mesh floor would have been silently missing.
    force_fn = force_two_level if args.arm == "two_level" else force_mono
    a_steps = a_grid(A_CONTROL, A_PIVOT, args.nsteps, "log")
    coeffs = bullfrog_float_coeffs(bullfrog_table(a_steps, cosmo))
    fdt = jnp.float64 if args.fdtype == "f64" else jnp.float32
    x = jnp.asarray(x_np, fdt)
    v = jnp.asarray(v_np, fdt)
    t0 = time.perf_counter()
    for c in coeffs:
        x, v = float_step_bullfrog(x, v, tuple(np.asarray(c, np.float64)), force_fn, L)
    wall = time.perf_counter() - t0

    os.makedirs(FORCE_DIR, exist_ok=True)
    path = os.path.join(FORCE_DIR, args.tag + ".npz")
    np.savez(path, x=np.asarray(x, np.float64), v=np.asarray(v, np.float64))
    return dict(
        kind="evolve",
        tag=args.tag,
        arm=args.arm,
        platform=dev.platform,
        fdtype=args.fdtype,
        nmesh=args.nmesh,
        nsteps=args.nsteps,
        family=args.family,
        alpha=args.alpha,
        tile=args.tile,
        buf=args.buf,
        wall_s=wall,
        wall_per_step=wall / args.nsteps,
        state_file=path,
        peak_bytes=peak(),
        peak_valid_reason=(None if dev.platform != "cpu" else "no memory_stats on CPU platform"),
        peak_host_rss_bytes=_host_rss_bytes(),
        rss_units="bytes",
    )


def run_single(args):
    if args.kind == "config":
        rec = make_config_leg(args)
    elif args.kind == "force":
        rec = force_leg(args)
    elif args.kind == "evolve":
        rec = evolve_leg(args)
    else:
        raise ValueError(f"unknown kind {args.kind!r}")
    print("WORKER_JSON " + json.dumps(rec))


# ===========================================================================
# statistics (orchestrator side; numpy only)
# ===========================================================================


def _k_grid(n, L):
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    kz = 2.0 * np.pi * np.fft.rfftfreq(n, d=L / n)
    kmag = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2)
    return kmag


def _bin_edges(n, L):
    kf = 2.0 * np.pi / L
    return np.arange(0.5 * kf, np.pi * n / L + kf, kf)


def cic_paint_vector(pos, vals, n, L):
    """Pure-numpy CIC paint of a per-particle vector -> (3, n, n, n).

    The estimator side is numpy-only by convention (_m1_common's rule), so the
    force fields are painted here rather than on device.

    np.bincount, NOT np.add.at: same 8-corner structure as _m1_common.cic_paint,
    and add.at is the classic numpy slow path (~10x here). At C-dev this runs
    8 corners x 3 components over 16.7M particles per call, on every coarse and
    tile leg -- add.at would have put the ESTIMATOR, not the physics, on the
    critical path of an overnight job.
    """
    d = L / n
    g = np.asarray(pos, dtype=np.float64) / d
    i0 = np.floor(g).astype(np.int64)
    frac = g - i0
    out = np.zeros((3, n * n * n), dtype=np.float64)
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
                for j in range(3):
                    out[j] += np.bincount(flat, weights=w * vals[:, j], minlength=n * n * n)
    return out.reshape(3, n, n, n)


def force_metrics(g_ref, g_arm):
    """D-013-class scalars, m1_force_probe convention.

    Percentiles rather than max: max is ONE outlier particle. And the rms is a
    DIAGNOSTIC, not the gate metric -- it is dominated by modes near the fine
    Nyquist where the gate does not care (see transfer_and_r).
    """
    rms_g = float(np.sqrt((g_ref**2).mean()))
    d = np.sqrt(((g_arm - g_ref) ** 2).sum(axis=1))
    return dict(
        rms_dg_over_rms_g=float(np.sqrt(((g_arm - g_ref) ** 2).mean()) / rms_g),
        p50=float(np.percentile(d, 50) / rms_g),
        p99=float(np.percentile(d, 99) / rms_g),
        p999=float(np.percentile(d, 99.9) / rms_g),
        rms_g=rms_g,
    )


def transfer_and_r(g_ref, g_arm, pos, n, L, k_gate):
    """THE GATE METRIC: T(k) = P_cross/P_ref and 1 - r(k), per k-shell.

    The deepest attribution available here, and it is actionable:
      a smooth T(k) != 1 is CORRECTABLE (window mismatch -- calibrate the kernel)
      a decorrelation 1-r > 0 is IRREDUCIBLE (aliasing -- no factor removes it)

    PROXY: both fields are CIC-painted, so the paint window cancels in the RATIO
    to first order -- but not exactly. Stated, not hidden.
    """
    fr = cic_paint_vector(pos, g_ref, n, L)
    fa = cic_paint_vector(pos, g_arm, n, L)
    kmag = _k_grid(n, L).ravel()
    edges = _bin_edges(n, L)
    saa = np.zeros(len(edges) - 1)
    sbb = np.zeros(len(edges) - 1)
    sab = np.zeros(len(edges) - 1)
    for j in range(3):
        ak = np.fft.rfftn(fr[j]).ravel()
        bk = np.fft.rfftn(fa[j]).ravel()
        saa += np.histogram(kmag, bins=edges, weights=(np.abs(ak) ** 2))[0]
        sbb += np.histogram(kmag, bins=edges, weights=(np.abs(bk) ** 2))[0]
        sab += np.histogram(kmag, bins=edges, weights=np.real(ak * np.conj(bk)))[0]
    counts = np.histogram(kmag, bins=edges)[0]
    good = (counts > 0) & (saa > 0)
    kc = 0.5 * (edges[1:] + edges[:-1])[good]
    T = sab[good] / saa[good]
    r = sab[good] / np.sqrt(saa[good] * sbb[good])
    band = kc <= k_gate
    return dict(
        k=kc.tolist(),
        T=T.tolist(),
        one_minus_r=(1.0 - r).tolist(),
        max_abs_T_minus_1_gate=float(np.abs(T[band] - 1.0).max()) if band.any() else None,
        max_one_minus_r_gate=float((1.0 - r)[band].max()) if band.any() else None,
    )


def seam_profile(g_ref, g_arm, pos, n_tile, cell, n_bins=8):
    """Error binned by distance to the nearest tile boundary, in fine cells.

    The DECISIVE mechanism signature, and a shape rather than a scalar:
    erfc truncation is EDGE-CONCENTRATED; kernel periodization/ringing is
    UNIFORM across the core. Measured 1.14 (uniform) for the gauss family during
    the kernel study -- which is how R7 was confirmed.
    """
    e = np.sqrt(((g_arm - g_ref) ** 2).sum(axis=1))
    rms = float(np.sqrt((g_ref**2).mean()))
    u = np.mod(pos / cell, n_tile)
    dist = np.minimum(u, n_tile - u).min(axis=1)
    edges = np.linspace(0, n_tile / 2.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dist >= lo) & (dist < hi)
        out.append(float(np.sqrt((e[m] ** 2).mean()) / rms) if m.any() else None)
    prof = [v for v in out if v is not None]
    return dict(
        dist_cells=(0.5 * (edges[1:] + edges[:-1])).tolist(),
        rms_dg_over_rms_g=out,
        edge_over_interior=(float(prof[0] / prof[-1]) if len(prof) > 1 and prof[-1] else None),
    )


def pk_ratio(x_a, x_b, n, L, k_gate):
    """|dP/P| between two evolved particle sets, on the D-v2-1 band.

    Uses _m1_common.cic_paint -- the estimator-neutral, numpy-only instrument
    every prior parity claim in this repo was measured with (m1-results, D-013),
    so the evolution arm's dP/P is directly comparable to them rather than to a
    fourth private paint.
    """
    sys.path.insert(0, HERE)
    import _m1_common as M

    kmag = _k_grid(n, L).ravel()
    edges = _bin_edges(n, L)
    counts = np.histogram(kmag, bins=edges)[0]
    pa = np.histogram(
        kmag, bins=edges, weights=(np.abs(np.fft.rfftn(M.cic_paint(x_a, n, L))) ** 2).ravel()
    )[0]
    pb = np.histogram(
        kmag, bins=edges, weights=(np.abs(np.fft.rfftn(M.cic_paint(x_b, n, L))) ** 2).ravel()
    )[0]
    good = (counts > 0) & (pa > 0)
    kc = 0.5 * (edges[1:] + edges[:-1])[good]
    ratio = pb[good] / pa[good] - 1.0
    band = kc <= k_gate
    return dict(
        k=kc.tolist(),
        dP_over_P=ratio.tolist(),
        max_abs_gate=float(np.abs(ratio[band]).max()) if band.any() else None,
    )


# ===========================================================================
# orchestration
# ===========================================================================


def spawn(kind, cfg, tag, platform="cpu", **kw):
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--single",
        "--kind",
        kind,
        "--config",
        cfg,
        "--tag",
        tag,
    ]
    for k, v in kw.items():
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        cmd += [flag] if v is True else [flag, str(v)]
    env = dict(os.environ)
    if platform == "cpu":
        # Physics legs are CPU f64. Set (not pop) JAX_PLATFORMS -- G2c's spawn
        # only pops it -- and give the FFT threads, since a single-threaded
        # 512^3 f64 FFT is ~4 min of pure wall. The reference leg's wall is not
        # a reported measurement, so this contaminates nothing.
        env["JAX_PLATFORMS"] = "cpu"
        env["OMP_NUM_THREADS"] = "8"
    else:
        env.pop("JAX_PLATFORMS", None)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        lines = (p.stderr or "").strip().splitlines()
        os.makedirs(ERR_DIR, exist_ok=True)
        err_path = os.path.join(ERR_DIR, tag + ".err")
        with open(err_path, "w") as fh:
            fh.write(p.stderr or "")
        # surface the real exception, not JAX's trailing "For simplicity"
        # boilerplate (V1 job 26/27 lesson)
        exc = [t for t in lines if ("Error" in t or "Exception" in t) and "For simplicity" not in t]
        return dict(
            tag=tag,
            error=(exc[-1] if exc else (lines[-1] if lines else f"exit {p.returncode}")),
            error_file=err_path,
            # An OOM is a RESULT (it maps the ceiling), not a crash to hide.
            oom=any("RESOURCE_EXHAUSTED" in t or "Out of memory" in t for t in lines),
        )
    out = [ln for ln in p.stdout.splitlines() if ln.startswith("WORKER_JSON ")]
    if not out:
        return dict(tag=tag, error="no WORKER_JSON line")
    return json.loads(out[-1][len("WORKER_JSON ") :])


def load_force(rec):
    return np.load(rec["force_file"]) if "force_file" in rec else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="cdev", choices=sorted(CONFIGS))
    ap.add_argument("--pivot-only", action="store_true", help="skip the scans")
    ap.add_argument("--skip-evolve", action="store_true")
    ap.add_argument("--skip-capacity", action="store_true", help="no GPU legs (laptop smoke)")
    ap.add_argument("--skip-1024", action="store_true", help="skip the converged 2n mesh floor")
    ap.add_argument("--keep-forces", action="store_true")
    # worker-private
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    for f, d in (
        ("--kind", None),
        ("--tag", None),
        ("--family", "gauss"),
        ("--which", "mono"),
        ("--assign", "cic"),
        ("--arm", "mono"),
    ):
        ap.add_argument(f, default=d, help=argparse.SUPPRESS)
    for f, t, d in (
        ("--nmesh", int, None),
        ("--tile", int, None),
        ("--buf", int, None),
        ("--alpha", float, None),
        ("--rout", float, None),
        ("--rin-frac", float, 0.6),
        ("--clip", float, 10.0),
        ("--a", float, A_PIVOT),
        ("--nsteps", int, N_STEPS),
    ):
        ap.add_argument(f, type=t, default=d, help=argparse.SUPPRESS)
    ap.add_argument("--fdtype", default="f64", choices=["f32", "f64"], help=argparse.SUPPRESS)
    ap.add_argument("--match", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.single:
        run_single(args)
        return

    # The ORCHESTRATOR stays off the GPU: its own jax work would preallocate
    # XLA_PYTHON_CLIENT_MEM_FRACTION of the card and starve every worker (V1
    # job 30: all 34 workers OOM'd on the 5% remainder). Must precede any jax
    # import -- and this process imports none at all.
    os.environ["JAX_PLATFORMS"] = "cpu"

    cfg = args.config
    g = geometry(cfg)
    n_f, n_c, L = g["n_fine"], g["n_coarse"], g["L"]
    k_gate = g["k_gate"]
    res = dict(
        config=cfg,
        geometry={k: (float(v) if isinstance(v, float) else v) for k, v in g.items()},
        pivot=dict(alpha=1.0, tile=n_f // 4, buf=None, family="gauss", assign="tsc", match=True),
        validation={},
        floors={},
        coarse={},
        tiles={},
        evolve={},
        workers={},
        notes={},
    )
    res["notes"] = dict(
        su="deneb rows carry wall only; SU becomes real at V3 on Vista (cost_of_memory.md).",
        peak_cpu="JAX exposes no memory_stats on CPU: physics legs carry a correctness "
        "number, not a capacity number.",
        pk_proxy="force spectra are CIC-painted; the window cancels in the ratio only to "
        "first order.",
        host_bucketing="brick bucketing is host-side numpy, OUTSIDE the reported wall; "
        "bucketing in XLA is an M-v2-2 build item.",
        kernel_study="runs/v2/g5_kernel_findings.md -- gauss tiles as 2.0/P (buffer is NOT "
        "a knob, tile size is); matching is family-specific.",
    )

    def W(rec):
        res["workers"][rec.get("tag", "?")] = rec
        return rec

    print(f"=== G5 two-level force: {cfg} ===")
    print(
        f"n_part {g['n_part']}^3  L {L}  fine {n_f}^3 (cell {g['fine_cell']:.3f})  "
        f"coarse {n_c}^3 (cell {g['coarse_cell']:.3f})"
    )
    print(f"k_Nyq,fine {g['k_nyq_fine']:.2f}  ->  GATE BAND k <= {k_gate:.2f} h/Mpc\n")

    # ---- configurations -----------------------------------------------------
    for a in (A_PIVOT, A_CONTROL):
        t = tag_of("config", cfg, a=a)
        W(spawn("config", cfg, t, a=a))
        print(f"[config] a={a} done")
    pos = _load_config_state(cfg, A_PIVOT)[0]

    F = {}

    def force(tag, **kw):
        platform = kw.pop("platform", "cpu")
        r = W(spawn("force", cfg, tag, platform=platform, a=A_PIVOT, **kw))
        if "error" in r:
            print(f"  !! {tag}: {r['error']}")
            return None
        F[tag] = np.load(r["force_file"])
        return r

    # ---- F0 / F1 / tile identity: the kaiser check --------------------------
    print("--- validation (if these fail, NO other G5 number may be read) ---")
    force("mono_f64", which="mono", nmesh=n_f, fdtype="f64")
    for fam, kw in (
        ("gauss", dict(alpha=1.0)),
        ("compact", dict(rout=4.0)),
        ("gauss_compact", dict(alpha=1.0, rout=4.0)),
    ):
        lt, st = f"long_fine_{fam}", f"short_fine_{fam}"
        force(lt, which="long", nmesh=n_f, family=fam, **kw)
        force(st, which="short", nmesh=n_f, family=fam, **kw)
        if lt in F and st in F and "mono_f64" in F:
            m = force_metrics(F["mono_f64"], F[lt] + F[st])
            res["validation"][f"F1_{fam}"] = dict(
                metric=m["rms_dg_over_rms_g"], ok=bool(m["rms_dg_over_rms_g"] < 1e-12)
            )
            print(
                f"  F1 {fam:14s} long+short vs mono = {m['rms_dg_over_rms_g']:.2e}  "
                f"{'OK' if m['rms_dg_over_rms_g'] < 1e-12 else 'FAIL'}"
            )

    tid = force(
        "tileid_gauss", which="short", nmesh=n_f, family="gauss", alpha=1.0, tile=n_f, buf=0
    )
    if tid and "short_fine_gauss" in F:
        m = force_metrics(F["short_fine_gauss"], F["tileid_gauss"])
        res["validation"]["tile_identity"] = dict(
            metric=m["rms_dg_over_rms_g"], ok=bool(m["rms_dg_over_rms_g"] < 1e-12)
        )
        print(
            f"  tile identity (one tile = whole box) = {m['rms_dg_over_rms_g']:.2e}  "
            f"{'OK' if m['rms_dg_over_rms_g'] < 1e-12 else 'FAIL'}"
        )
        res["validation"]["partition"] = dict(
            ok=bool(tid.get("partition_ok")),
            n_owned=tid.get("n_owned_total"),
            min_owner=tid.get("min_owner_count"),
            max_owner=tid.get("max_owner_count"),
        )

    # A missing check must FAIL, never silently drop out of the all(): the smoke
    # run had tileid crash, leaving all() over the surviving F1s -> a false "OK".
    # That is the "guard against no-op" class (retrospective Sec 5) and it is
    # exactly how a broken gate looks clean.
    required = ["F1_gauss", "F1_compact", "F1_gauss_compact", "tile_identity", "partition"]
    missing = [r for r in required if r not in res["validation"]]
    for r in missing:
        res["validation"][r] = dict(ok=False, error="leg did not run or crashed")
    ok = bool(not missing) and all(
        v.get("ok") for v in res["validation"].values() if isinstance(v, dict)
    )
    res["validation"]["all_ok"] = ok
    res["validation"]["missing"] = missing
    if missing:
        print(f"  !! MISSING required checks: {missing}")
    print(f"  -> validation {'OK' if ok else 'FAIL (do NOT read the gate numbers)'}")

    # ---- floors -------------------------------------------------------------
    print("\n--- floors (the bars everything else is judged against) ---")
    force("mono_f32", which="mono", nmesh=n_f, fdtype="f32")
    if "mono_f32" in F:
        m = force_metrics(F["mono_f64"], F["mono_f32"])
        res["floors"]["precision"] = m
        print(f"  F2 precision (f32 vs f64)     rms {m['rms_dg_over_rms_g']:.3e}")
    force("mono_half", which="mono", nmesh=n_f // 2, fdtype="f64")
    force("mono_quarter", which="mono", nmesh=n_f // 4, fdtype="f64")
    if "mono_half" in F:
        m = force_metrics(F["mono_f64"], F["mono_half"])
        res["floors"]["mesh"] = dict(
            **m, spectra=transfer_and_r(F["mono_f64"], F["mono_half"], pos, n_f, L, k_gate)
        )
        print(
            f"  F4 MESH FLOOR (fine vs fine/2) rms {m['rms_dg_over_rms_g']:.3e}  <- the D-v2-1 bar"
        )
    if "mono_half" in F and "mono_quarter" in F:
        e1 = force_metrics(F["mono_f64"], F["mono_half"])["rms_dg_over_rms_g"]
        e2 = force_metrics(F["mono_half"], F["mono_quarter"])["rms_dg_over_rms_g"]
        # (kc)^2 => halving the cell should quarter the error. If it does not,
        # the "floor" is discreteness-contaminated (risk R3) and the gate must be
        # re-framed with JC against an absolute tolerance, not this floor.
        res["floors"]["mesh_scaling_check"] = dict(
            e_fine_half=e1,
            e_half_quarter=e2,
            ratio=(e2 / e1 if e1 else None),
            expect="~4 if mesh-error-dominated ((kc)^2 law)",
            ok=bool(e1 and 2.0 <= e2 / e1 <= 8.0),
        )
        print(
            f"     mesh (kc)^2 scaling check: ratio {e2 / e1:.2f} (expect ~4)  "
            f"{'OK' if 2.0 <= e2 / e1 <= 8.0 else 'SUSPECT -> floor may be contaminated'}"
        )

    # ---- coarse arms --------------------------------------------------------
    print("\n--- coarse arm (vs each family's own long_fine; tiling excluded) ---")
    print(
        f"  {'family':>14} {'param':>12} {'assign':>7} {'match':>6} {'rms':>10} {'|T-1|':>9} "
        f"{'1-r':>9}"
    )
    coarse_specs = [
        ("gauss", dict(alpha=1.0)),
        ("compact", dict(rout=4.0)),
        ("gauss_compact", dict(alpha=1.0, rout=4.0)),
    ]
    if args.pivot_only:
        coarse_specs = coarse_specs[:1]
    for fam, kw in coarse_specs:
        ref = F.get(f"long_fine_{fam}")
        if ref is None:
            continue
        for assign in ("cic", "tsc"):
            for match in (False, True):
                t = tag_of("coarse", cfg, family=fam, assign=assign, match=int(match), **kw)
                if (
                    force(t, which="long", nmesh=n_c, family=fam, assign=assign, match=match, **kw)
                    is None
                ):
                    continue
                m = force_metrics(ref, F[t])
                sp = transfer_and_r(ref, F[t], pos, n_f, L, k_gate)
                res["coarse"][t] = dict(**m, spectra=sp)
                pname = f"a={kw.get('alpha')}" if "alpha" in kw else f"r={kw.get('rout')}c"
                print(
                    f"  {fam:>14} {pname:>12} {assign:>7} {str(match):>6} "
                    f"{m['rms_dg_over_rms_g']:10.3e} {sp['max_abs_T_minus_1_gate']:9.3e} "
                    f"{sp['max_one_minus_r_gate']:9.3e}"
                )

    # ---- tiled arms ---------------------------------------------------------
    print("\n--- tiled short arm (vs short_fine; the ONLY difference is tiling) ---")
    print(
        f"  {'family':>14} {'T':>5} {'b':>4} {'P':>5} {'work':>6} {'rms':>10} {'x P':>7} "
        f"{'edge/int':>9}"
    )
    tile_specs = []
    for T in [n_f // 4] if args.pivot_only else [n_f // 8, n_f // 4, n_f // 2]:
        for b in [n_f // 16] if args.pivot_only else [n_f // 32, n_f // 16, n_f // 8]:
            tile_specs.append(("gauss", dict(alpha=1.0), T, b))
            tile_specs.append(("compact", dict(rout=4.0), T, b))
    for fam, kw, T, b in tile_specs:
        ref = F.get(f"short_fine_{fam}")
        if ref is None:
            continue
        t = tag_of("tiled", cfg, family=fam, tile=T, buf=b, **kw)
        r = force(t, which="short", nmesh=n_f, family=fam, tile=T, buf=b, **kw)
        if r is None:
            continue
        m = force_metrics(ref, F[t])
        sm = seam_profile(ref, F[t], pos, T, g["fine_cell"])
        P = r.get("padded_P")
        res["tiles"][t] = dict(**m, seam=sm, diag=r, peak_x_P=(m["rms_dg_over_rms_g"] * P))
        print(
            f"  {fam:>14} {T:5d} {r['b_realized']:4d} {P:5d} {r['fft_work_ratio']:6.2f} "
            f"{m['rms_dg_over_rms_g']:10.3e} {m['rms_dg_over_rms_g'] * P:7.2f} "
            f"{(sm['edge_over_interior'] or float('nan')):9.2f}"
        )
    print(
        "  ('x P' ~ 2.0 => the ringing law; 'edge/int' ~ 1 => uniform => ringing, not truncation)"
    )

    # ---- evolution ----------------------------------------------------------
    if not args.skip_evolve:
        print("\n--- evolution (THE arm D-v2-1 bars: dP/P of an EVOLVED field) ---")
        ev = {}
        for arm, kw in (
            ("mono", dict(nmesh=n_f)),
            ("mono_half", dict(nmesh=n_f // 2)),
            (
                "two_level",
                dict(
                    nmesh=n_f,
                    family="gauss",
                    alpha=1.0,
                    assign="tsc",
                    match=True,
                    tile=n_f // 4,
                    buf=n_f // 16,
                ),
            ),
        ):
            t = tag_of("evolve", cfg, **{**kw, "a": A_PIVOT})
            r = W(spawn("evolve", cfg, t, platform="cpu", arm=arm, nsteps=N_STEPS, **kw))
            if "error" in r:
                print(f"  !! {arm}: {r['error']}")
                continue
            with np.load(r["state_file"]) as f:
                ev[arm] = f["x"]
            print(f"  [{arm}] wall {r['wall_s']:.1f}s ({r['wall_per_step']:.2f} s/step)")

        # The mono-2n floor (1024 at C-dev): the CONVERGED reference the D-v2-1
        # bar really wants. f32 by necessity -- 1024^3 f64 is ~52 GB against a
        # 56 GB host -- which is safe because F2 put f32-vs-f64 at ~3e-7, six
        # orders under the ~1e-1 floor, and the error is common-mode across the
        # pair. Skippable (--skip-1024) since it doubles the evolution wall.
        if not args.skip_1024:
            kw = dict(nmesh=2 * n_f)
            t = tag_of("evolve", cfg, **{**kw, "a": A_PIVOT, "fdtype": "f32"})
            r = W(
                spawn(
                    "evolve", cfg, t, platform="cpu", arm="mono", nsteps=N_STEPS, fdtype="f32", **kw
                )
            )
            if "error" in r:
                note = "OOM -> 1024 floor unavailable" if r.get("oom") else r["error"]
                print(f"  !! mono_double (n={2 * n_f}): {note}")
                res["evolve"]["mono_double_error"] = note
            else:
                with np.load(r["state_file"]) as f:
                    ev["mono_double"] = f["x"]
                print(
                    f"  [mono_double n={2 * n_f} f32] wall {r['wall_s']:.1f}s "
                    f"({r['wall_per_step']:.2f} s/step)"
                )

        for name, lo, hi in (
            ("mesh_floor_half", "mono_half", "mono"),
            ("mesh_floor_double", "mono", "mono_double"),
        ):
            if lo in ev and hi in ev:
                res["evolve"][name] = pk_ratio(ev[lo], ev[hi], n_f, L, k_gate)
                print(
                    f"  {name:18s} |dP/P| on the band = {res['evolve'][name]['max_abs_gate']:.3e}"
                )
        if "mono" in ev and "two_level" in ev:
            res["evolve"]["two_level"] = pk_ratio(ev["mono"], ev["two_level"], n_f, L, k_gate)
            print(
                f"  two-level          |dP/P| on the band = "
                f"{res['evolve']['two_level']['max_abs_gate']:.3e}"
            )

        # (kc)^2 validity check on the EVOLVED floor, the twin of the force-side
        # one: halving the cell should quarter the error. If it does not, the
        # 2n rung is discreteness-dominated rather than mesh-dominated (risk R3),
        # the "floor" is overestimated, the gate passes trivially, and the bar
        # must be re-framed with JC against an absolute tolerance instead.
        if "mesh_floor_half" in res["evolve"] and "mesh_floor_double" in res["evolve"]:
            e1 = res["evolve"]["mesh_floor_half"]["max_abs_gate"]
            e2 = res["evolve"]["mesh_floor_double"]["max_abs_gate"]
            ok = bool(e2 and 2.0 <= e1 / e2 <= 8.0)
            res["evolve"]["floor_scaling_check"] = dict(
                e_half_vs_fine=e1,
                e_fine_vs_double=e2,
                ratio=(e1 / e2 if e2 else None),
                expect="~4 if mesh-error-dominated ((kc)^2 law)",
                ok=ok,
            )
            print(
                f"  floor (kc)^2 check: ratio {e1 / e2:.2f} (expect ~4)  "
                f"{'OK' if ok else 'SUSPECT -> floor contaminated, re-frame the bar with JC'}"
            )

        # Prefer the CONVERGED (2n) floor; fall back to fine-vs-half and say so.
        floor_key = (
            "mesh_floor_double" if "mesh_floor_double" in res["evolve"] else "mesh_floor_half"
        )
        if floor_key in res["evolve"] and "two_level" in res["evolve"]:
            fl = res["evolve"][floor_key]["max_abs_gate"]
            tl = res["evolve"]["two_level"]["max_abs_gate"]
            res["evolve"]["headroom"] = dict(
                floor=fl,
                floor_from=floor_key,
                two_level=tl,
                headroom_ratio=(fl / tl if tl else None),
                note="headroom = floor / split error; JC picks the config off the Pareto "
                "curve, not a binary pass/fail (plan V2a, resolved 2026-07-15).",
            )
            print(f"  HEADROOM (floor / split) = {fl / tl:.1f}x   [floor from {floor_key}]")

    # ---- capacity: the ONLY GPU legs (D-v2-4's instrument) ------------------
    # Everything above is CPU f64 on purpose, so it measures physics and nothing
    # else. The memory claim needs the card, and it is a SEPARATE process so the
    # two never contaminate each other.
    if not args.skip_capacity:
        print("\n--- capacity: peak B/p on the GPU (all physics above was CPU f64) ---")
        print(f"  {'arm':>12} {'peak B/p':>10} {'peak MB':>9} {'wall s':>8}  note")
        cap = {}
        legs = [
            ("mono_gpu", dict(which="mono", nmesh=n_f, fdtype="f32")),
            (
                "tiled_gpu",
                dict(
                    which="short",
                    nmesh=n_f,
                    family="gauss",
                    alpha=1.0,
                    tile=n_f // 4,
                    buf=n_f // 16,
                    fdtype="f32",
                ),
            ),
        ]
        for name, kw in legs:
            r = W(spawn("force", cfg, tag_of("cap", cfg, **kw), platform="gpu", a=A_PIVOT, **kw))
            if "error" in r:
                # mono at C-dev is EXPECTED to OOM the 6 GB card, and that IS the
                # capacity result ("monolithic does not fit; tiled does"), not a
                # failure to hide.
                note = "OOM = the capacity RESULT" if r.get("oom") else r["error"][:40]
                cap[name] = dict(
                    oom=bool(r.get("oom")), error=r["error"], peak_bytes_per_particle=None
                )
                print(f"  {name:>12} {'-':>10} {'-':>9} {'-':>8}  {note}")
                continue
            bp, pk = r.get("peak_bytes_per_particle"), r.get("peak_bytes")
            cap[name] = dict(
                peak_bytes_per_particle=bp,
                peak_bytes=pk,
                wall_s=r.get("wall_s"),
                platform=r.get("platform"),
                padded_P=r.get("padded_P"),
                cap=r.get("cap"),
                peak_bytes_per_tile_particle=((pk / r["cap"]) if pk and r.get("cap") else None),
                peak_valid_reason=r.get("peak_valid_reason"),
            )
            print(
                f"  {name:>12} {(bp if bp else float('nan')):10.1f} "
                f"{((pk or 0) / 1e6):9.1f} {(r.get('wall_s') or 0):8.2f}  "
                f"{r.get('platform', '?')}"
            )
        cap["claim"] = (
            "The O(tile) claim is tested by INVARIANCE, not one number: run the same "
            "(tile, buf) at cdev AND cdev8 -- the tiled peak must be FLAT in box volume "
            "while mono's scales. peak_bytes_per_tile_particle is the headline."
        )
        res["capacity"] = cap

    print(
        "\nVerdict vs the mesh floor is JC's call (D-v2-1 band, plan-plan V2 exit); "
        "nothing self-ratified."
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"g5_results_{cfg}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {path}")
    if not args.keep_forces:
        n = 0
        for f in os.listdir(FORCE_DIR) if os.path.isdir(FORCE_DIR) else []:
            if f.endswith(".npy"):
                os.remove(os.path.join(FORCE_DIR, f))
                n += 1
        print(f"removed {n} force .npy ({FORCE_DIR}); --keep-forces to retain")


if __name__ == "__main__":
    main()
