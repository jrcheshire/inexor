"""Force from a point mass: inexor's two-level PM force against one global fine mesh and Newton
(paper App. A.1).

One unit-mass source is painted; the force field it generates is gathered at probe points
(log-spaced separations, random directions), so the probes carry no mass. Three forces:

  two-level   long arm: forces.force_global(which="long", assign="tsc", paint="int",
              match=cfg.coarse_match) on the coarse mesh, gathered at the probes
              (pos_gather); short arm: the body of forces.make_tile_force_fn's `one_tile`
              (tile_paint_int -> tile_delta_from_int -> split_prefactor("short") ->
              tile_gather_vector) with the source painted into every tile whose tile+buffer
              contains it and each probe read from the tile that owns it (forces.owning_tile).
              f64 meshes and integer paint on both arms. Also computed with f64 paint on both
              arms (the global mesh's paint), so its difference to the global mesh is the split
              alone, and with the coarse mesh in f32 (production) for four sources.
  global mesh forces.force_global(which="mono", assign="cic", paint="f64") on the n_fine^3
              mesh (same 0.25 Mpc/h cell), in a child process (~12.5 GB peak).
  Newton      Ewald sum (periodic, neutralizing background), the continuum limit of the
              k != 0 Poisson solve both PM forces approximate.

Gates, refused if they fail: the composed tile arm equals `one_tile` bit for bit at the painted
point; field superposition equals force_short_tiled + force_global on two-particle
configurations; the Ewald sum is independent of its screening parameter.

Geometry: the `cdev` preset through inexor.plan.engine_config (alpha = 1, match order 3).
40 sources (tile core / face / edge / corner), 54 separations x 64 directions, all drawn from
a fixed seed. CPU backend, float64. Writes force_accuracy.npz (all forces), force_curves.json
(the figure's medians and percentiles) and summary.txt to --out-dir:

    pixi run --frozen python scripts/validation/force_accuracy.py
    pixi run --frozen python scripts/validation/force_accuracy.py --from-cache   # reduce only
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from _common import maxrss as _maxrss
from _common import setup_jax

from inexor import forces as F
from inexor.plan import engine_config

PRESET = "cdev"
R_REPORT = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
N_DIR = 64           # probes per (source, r)
N_R_LOG = 48         # log-spaced separations, plus R_REPORT
SEED = 20261008
# sources per class: deep core, then within one buffer of a tile face / edge / corner
N_SOURCES = dict(core=12, face=10, edge=9, corner=9)


# ---------------------------------------------------------------------------
# geometry and sampling
# ---------------------------------------------------------------------------

def geometry():
    ec = engine_config(PRESET)
    ec.validate()
    P, b_real = F.padded_size(ec.n_tile, ec.b_fine, n_fine=ec.n_fine)
    return dict(ec=ec, L=ec.box_size, n_tot=ec.n_total, n_fine=ec.n_fine,
                n_coarse=ec.n_coarse, n_tile=ec.n_tile, b_real=b_real, P=P,
                cell=ec.fine_cell, r_s=ec.r_s, match=ec.coarse_match,
                tile_len=ec.n_tile * ec.fine_cell, buf_len=b_real * ec.fine_cell,
                n_side=ec.n_fine // ec.n_tile, frac_bits=ec.frac_bits)


def sample_sources(g, rng):
    """Sources by class. Per axis, the distance to the nearest tile-boundary plane is drawn
    in [0, buffer) for a 'near' axis and in [buffer, tile/2] for a core axis."""
    T, B = g["tile_len"], g["buf_len"]
    n_near = dict(core=0, face=1, edge=2, corner=3)
    out, cls = [], []
    for c, n in N_SOURCES.items():
        for _ in range(n):
            near = rng.permutation(3) < n_near[c]
            d = np.where(near, rng.uniform(0.0, B, 3), rng.uniform(B, 0.5 * T, 3))
            loc = np.where(rng.random(3) < 0.5, d, T - d)
            tile = rng.integers(0, g["n_side"], 3)
            out.append(np.mod(tile * T + loc, g["L"]))
            cls.append(c)
    return np.asarray(out), np.asarray(cls)


def r_grid(g):
    r_min, r_max = 0.1 * g["cell"], 40.0
    r = np.concatenate([np.geomspace(r_min, r_max, N_R_LOG), R_REPORT])
    return np.unique(np.round(r, 12))


def random_dirs(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Newton: Ewald sum
# ---------------------------------------------------------------------------

def ewald_field(d, L, a_ewald=2.5, n_img=2, m_max=6, chunk=4096):
    """Field E(d) of a unit point charge in a periodic cube with neutralizing background,
    div E = 4 pi (delta_D - 1/V): Ewald real-space sum over images |n_i| <= n_img plus the
    k-space sum over |m_i| <= m_max; screening a = a_ewald / L. f64 numpy, (n,3)."""
    from scipy.special import erfc

    a = a_ewald / L
    V = L**3
    rng_i = np.arange(-n_img, n_img + 1)
    imgs = np.stack(np.meshgrid(rng_i, rng_i, rng_i, indexing="ij"), -1).reshape(-1, 3) * L
    mm = np.arange(-m_max, m_max + 1)
    ms = np.stack(np.meshgrid(mm, mm, mm, indexing="ij"), -1).reshape(-1, 3)
    ms = ms[np.any(ms != 0, axis=1)]
    kv = 2.0 * np.pi * ms / L
    k2 = np.sum(kv**2, axis=1)
    kcoef = (4.0 * np.pi / V) * np.exp(-k2 / (4.0 * a * a)) / k2
    out = np.empty_like(d)
    for lo in range(0, d.shape[0], chunk):
        dd = d[lo:lo + chunk]
        rn = dd[:, None, :] + imgs[None, :, :]
        r = np.linalg.norm(rn, axis=2)
        s = (erfc(a * r) + 2.0 * a * r / math.sqrt(math.pi) * np.exp(-(a * r) ** 2)) / r**3
        e_real = np.sum(rn * s[:, :, None], axis=1)
        sinkr = np.sin(dd @ kv.T)
        e_k = (sinkr * kcoef[None, :]) @ kv
        out[lo:lo + chunk] = e_real + e_k
    return out


def newton_g(d, g):
    """Newtonian acceleration in the package's units (div g = -delta, delta = counts/mean):
    a unit particle sources (V/n_tot) delta_D, so g = -(V / (4 pi n_tot)) E."""
    return -(g["L"] ** 3 / (4.0 * math.pi * g["n_tot"])) * ewald_field(d, g["L"])


# ---------------------------------------------------------------------------
# the two-level force as a field (source painted, probes gathered)
# ---------------------------------------------------------------------------

class TileField:
    """The short arm of `make_tile_force_fn` split into paint+solve and gather, so the field
    of a painted source can be read at unpainted probes. Same calls in the same order as
    `one_tile` (forces.py): paint, decode, rfftn, three irfftn with the kernels
    `pref * ik_j` (`split_prefactor`) formed in the program, CIC gather; `paint` = "int"
    (engine) or "f64"."""

    def __init__(self, g, paint="int"):
        self.g, self.paint = g, paint
        P, cell = g["P"], g["cell"]
        self.shape = (P,) * 3
        self.mean = float(g["n_tot"]) / float(g["n_fine"]) ** 3
        pref, iks = F.split_prefactor(self.shape, cell, "short", r_s=g["r_s"],
                                      fdtype=np.float64)
        fb = g["frac_bits"]
        shape, mean = self.shape, self.mean

        def solve(u):
            live = jnp.ones((u.shape[0],), dtype=bool)
            if paint == "int":
                mesh_i, n_out = F.tile_paint_int(u, live, shape, cell, fb)
                delta = F.tile_delta_from_int(mesh_i, mean, fb, fdtype=jnp.float64)
            else:
                delta, n_out = F.tile_paint_f64(u, live, shape, cell, mean, fdtype=jnp.float64)
            dk = jnp.fft.rfftn(delta)
            return [jnp.fft.irfftn(dk * (pref * ik), s=shape) for ik in iks], n_out

        def gather(gx, gy, gz, u):
            live = jnp.ones((u.shape[0],), dtype=bool)
            return F.tile_gather_vector(gx, gy, gz, u, live, shape, cell)

        self.solve = jax.jit(solve)
        self.gather = jax.jit(gather)

    def tiles(self):
        n = self.g["n_side"]
        return [(i, j, k) for i in range(n) for j in range(n) for k in range(n)]

    def local(self, x, t):
        origin, extent = F.tile_origin_extent(t, self.g["n_tile"], self.g["b_real"],
                                              self.g["cell"])
        return np.asarray(F.tile_local_coords(jnp.asarray(x), origin, self.g["L"])), extent

    def force(self, src, probes):
        """Short force at `probes` (n,3) from a unit source at `src` (3,). Each probe reads its
        owning tile; a tile whose tile+buffer does not contain the source contributes 0."""
        g = self.g
        out = np.zeros_like(probes)
        own = F.owning_tile(probes, g["cell"], g["n_tile"], g["n_fine"])
        n_solves = 0
        for t in self.tiles():
            us, extent = self.local(src[None, :], t)
            if not np.all(us < extent):
                continue
            sel = np.all(own == np.asarray(t), axis=1)
            if not sel.any():
                continue
            fields, n_out = self.solve(jnp.asarray(us))
            n_solves += 1
            up, _ = self.local(probes[sel], t)
            res, n_out_g = self.gather(*fields, jnp.asarray(up))
            if int(n_out) or int(n_out_g):
                raise RuntimeError(f"tile {t}: rows outside the padded box")
            out[sel] = np.asarray(res)
        return out, n_solves


def long_force(src, probes, g, paint="int", fdtype=np.float64):
    out, _ = F.force_global(src[None, :], g["n_coarse"], g["L"], g["n_tot"], "long",
                            r_s=g["r_s"], match=g["match"], assign="tsc", paint=paint,
                            pos_gather=probes, fdtype=fdtype)
    return out


def mono_force(src, probes, g):
    out, _ = F.force_global(src[None, :], g["n_fine"], g["L"], g["n_tot"], "mono",
                            assign="cic", paint="f64", pos_gather=probes)
    return out


# ---------------------------------------------------------------------------
# gates on the composition (printed, and refused if they fail)
# ---------------------------------------------------------------------------

def gate_composition(g, tf):
    """(1) TileField read at the painted point == `one_tile` bitwise; (2) superposition of
    TileField + long-arm fields == force_short_tiled + force_global on painted 2-particle
    configurations (the float two-level reference path), including cross-tile pairs."""
    one_tile, geom = F.make_tile_force_fn(g["n_fine"], g["L"], g["n_tot"], g["n_tile"],
                                          g["ec"].b_fine, r_s=g["r_s"], paint="int",
                                          fdtype=jnp.float64)
    assert geom["P"] == g["P"] and geom["b_realized"] == g["b_real"]
    worst1 = 0.0
    for x in [np.array([17.3, 40.1, 9.7]), np.array([63.9, 1.2, 70.0])]:
        t = tuple(int(v) for v in F.owning_tile(x[None], g["cell"], g["n_tile"],
                                                g["n_fine"])[0])
        u, _ = tf.local(x[None], t)
        ref, _, _ = one_tile(jnp.asarray(u), jnp.asarray([True]), jnp.asarray([True]))
        fields, _ = tf.solve(jnp.asarray(u))
        mine, _ = tf.gather(*fields, jnp.asarray(u))
        worst1 = max(worst1, float(np.max(np.abs(np.asarray(mine) - np.asarray(ref)))))
    print(f"gate 1: TileField vs one_tile at the painted point, max |diff| = {worst1:.3e}")
    if worst1 != 0.0:
        raise SystemExit("gate 1 failed: the composed tile arm is not one_tile's")

    configs = [
        (np.array([20.0, 30.0, 40.0]), np.array([21.3, 29.2, 40.9])),   # same tile core
        (np.array([62.5, 30.0, 40.0]), np.array([66.0, 31.0, 39.0])),   # across a face, in buffer
        (np.array([60.0, 63.0, 1.0]), np.array([67.5, 66.0, 126.0])),   # across edge + wrap
        (np.array([50.0, 30.0, 40.0]), np.array([66.0, 30.5, 40.0])),   # other tile, beyond buffer
    ]
    worst2 = 0.0
    for s, p in configs:
        pos = np.stack([s, p])

        def member_fn(t, pos=pos):
            u, extent = tf.local(pos, t)
            return np.flatnonzero(np.all(u < extent, axis=1))

        sh_ref, diag = F.force_short_tiled(pos, g["n_fine"], g["L"], g["n_tot"], g["n_tile"],
                                           g["ec"].b_fine, member_fn, 2, r_s=g["r_s"],
                                           paint="int")
        assert diag["partition_ok"] and diag["n_overhang_total"] == 0
        lo_ref, _ = F.force_global(pos, g["n_coarse"], g["L"], g["n_tot"], "long",
                                   r_s=g["r_s"], match=g["match"], assign="tsc", paint="int")
        ref = sh_ref + lo_ref
        mine = np.zeros_like(pos)
        for x in pos:
            sh, _ = tf.force(x, pos)
            mine += sh + long_force(x, pos, g)
        rel = float(np.max(np.abs(mine - ref)) / np.max(np.abs(ref)))
        assert np.all(np.abs(ref) > 0), "degenerate configuration"
        worst2 = max(worst2, rel)
    print(f"gate 2: field superposition vs force_short_tiled + force_global on 2-particle "
          f"configs, max rel diff = {worst2:.3e}")
    if worst2 > 1e-10:
        raise SystemExit("gate 2 failed: the field route is not the engine's two-level force")
    return worst1, worst2


def gate_ewald(g, rng):
    """Ewald: independent of the screening parameter, and -> minimum image + background at
    small r (cubic symmetry cancels the images' linear tidal term)."""
    d = random_dirs(rng, 400) * rng.uniform(0.3, 40.0, 400)[:, None]
    e1 = ewald_field(d, g["L"])
    e2 = ewald_field(d, g["L"], a_ewald=3.5, n_img=2, m_max=10)
    rel_a = float(np.max(np.linalg.norm(e1 - e2, axis=1) / np.linalg.norm(e1, axis=1)))
    V = g["L"] ** 3
    r = np.linalg.norm(d, axis=1)
    e_mi = d / r[:, None] ** 3 - (4.0 * np.pi / (3.0 * V)) * d
    rel_mi = np.linalg.norm(e1 - e_mi, axis=1) / np.linalg.norm(e1, axis=1)
    print(f"gate 3: Ewald screening-independence max rel = {rel_a:.2e}; "
          f"Ewald vs (1/r^2 + background): max rel {rel_mi[r < 5].max():.1e} (r<5), "
          f"{rel_mi[r < 20].max():.1e} (r<20), {rel_mi.max():.1e} (r<40)")
    if rel_a > 1e-9:
        raise SystemExit("gate 3 failed: Ewald sum not converged")
    return rel_a


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------

def draw_inputs(g):
    """Sources, classes, separations and probe displacements, from SEED alone."""
    rng = np.random.default_rng(SEED)
    src, cls = sample_sources(g, rng)
    r = r_grid(g)
    D = np.stack([random_dirs(rng, len(r) * N_DIR).reshape(len(r), N_DIR, 3)
                  * r[:, None, None] for _ in src])
    return src, cls, r, D


def measure_mono(path_out):
    """The global fine-mesh force for every source (run in a child process: the 512^3 f64
    solve peaks at ~12.5 GB, released when the child exits)."""
    t0 = time.perf_counter()
    g = geometry()
    src, _, r, D = draw_inputs(g)
    GM = np.empty_like(D)
    for i, s in enumerate(src):
        probes = np.mod(s + D[i].reshape(-1, 3), g["L"])
        GM[i] = mono_force(s, probes, g).reshape(D.shape[1:])
    np.savez(path_out, GM=GM, wall=time.perf_counter() - t0, maxrss=_maxrss())
    print(f"global mesh: {len(src)} solves in {time.perf_counter() - t0:.1f}s, "
          f"peak RSS {_maxrss() / 1e9:.1f} GB", flush=True)


def measure(out_dir):
    t0 = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "force_accuracy.npz")
    mono_path = os.path.join(out_dir, "force_accuracy_mono.npz")
    subprocess.run([sys.executable, os.path.abspath(__file__), "--phase", "mono",
                    "--mono-out", mono_path], check=True)
    GM = np.load(mono_path)["GM"]

    g = geometry()
    print(f"{PRESET}: L={g['L']} n_fine={g['n_fine']} n_coarse={g['n_coarse']} "
          f"tile={g['n_tile']} buf={g['b_real']} (P={g['P']}) r_s={g['r_s']} "
          f"match={g['match']} n_side={g['n_side']}")
    tf = TileField(g, paint="int")
    gates = list(gate_composition(g, tf))
    gates.append(gate_ewald(g, np.random.default_rng(SEED + 1)))
    jax.clear_caches()   # drop the gate's one_tile executable and its embedded kernels

    src, cls, r, D = draw_inputs(g)
    n_src = len(src)
    G2, NT = np.empty_like(D), np.empty_like(D)
    n_solves = []
    for i, s in enumerate(src):
        probes = np.mod(s + D[i].reshape(-1, 3), g["L"])
        sh, ns = tf.force(s, probes)
        G2[i] = (sh + long_force(s, probes, g)).reshape(D.shape[1:])
        NT[i] = newton_g(D[i].reshape(-1, 3), g).reshape(D.shape[1:])
        n_solves.append(ns)
        print(f"  source {i + 1:2d}/{n_src} {cls[i]:6s} at ({s[0]:6.2f},{s[1]:6.2f},"
              f"{s[2]:6.2f}): {ns} tile solves, {time.perf_counter() - t0:6.1f}s", flush=True)
    del tf
    jax.clear_caches()

    # the same two-level force with f64 paint on both arms (the global mesh's paint), and
    # with the coarse mesh in f32 (the production dtype, plan.RATIFIED) on a subset
    tf64 = TileField(g, paint="f64")
    G2f = np.empty_like(D)
    for i, s in enumerate(src):
        probes = np.mod(s + D[i].reshape(-1, 3), g["L"])
        sh, _ = tf64.force(s, probes)
        G2f[i] = (sh + long_force(s, probes, g, paint="f64")).reshape(D.shape[1:])
    del tf64
    jax.clear_caches()
    tf = TileField(g, paint="int")
    sub = [int(np.flatnonzero(cls == c)[0]) for c in ("core", "face", "edge", "corner")]
    side32 = []
    for i in sub:
        probes = np.mod(src[i] + D[i].reshape(-1, 3), g["L"])
        sh, _ = tf.force(src[i], probes)
        g32 = sh + long_force(src[i], probes, g, fdtype=np.float32)
        side32.append(np.linalg.norm(g32 - G2[i].reshape(-1, 3), axis=1)
                      / np.linalg.norm(GM[i].reshape(-1, 3), axis=1))
    wall = time.perf_counter() - t0
    peak = max(_maxrss("self"), _maxrss("children"))
    np.savez(cache, src=src, cls=cls, r=r, D=D, G2=G2, G2f=G2f, GM=GM, GN=NT,
             n_solves=np.asarray(n_solves), side_coarse32=np.stack(side32),
             sub=np.asarray(sub), wall=wall, gates=np.asarray(gates),
             rss=np.asarray([_maxrss("self"), _maxrss("children")]))
    print(f"measured in {wall:.1f}s, peak RSS parent {_maxrss('self') / 1e9:.1f} GB, "
          f"child {_maxrss('children') / 1e9:.1f} GB (max {peak / 1e9:.1f}) -> {cache}")
    return dict(np.load(cache))


# ---------------------------------------------------------------------------
# reduction, summary, the figure's curves
# ---------------------------------------------------------------------------

def reduce(m, paint="int"):
    """Per-probe ratios for the two-level force with `paint` ("int": engine; "f64": the global
    mesh's paint, so the difference to it is the split alone)."""
    D, GM, GN = m["D"], m["GM"], m["GN"]
    G2 = m["G2"] if paint == "int" else m["G2f"]
    rhat = D / np.linalg.norm(D, axis=-1, keepdims=True)
    fr_n = -np.sum(GN * rhat, axis=-1)          # attractive radial component, Newton
    out = dict(r=m["r"], cls=m["cls"])
    for name, G in (("2l", G2), ("gm", GM), ("nt", GN)):
        fr = -np.sum(G * rhat, axis=-1)
        perp = G + fr[..., None] * rhat
        out[f"ratio_{name}"] = fr / fr_n
        out[f"tan_{name}"] = np.linalg.norm(perp, axis=-1) / np.linalg.norm(G, axis=-1)
    out["rel"] = np.linalg.norm(G2 - GM, axis=-1) / np.linalg.norm(GM, axis=-1)
    return out


def pct(x, q, axis):
    return np.percentile(x, q, axis=axis)


def _flat(a, sel, n_r):
    """(sources, r, dirs) -> (r, sources*dirs) for the selected sources."""
    return np.moveaxis(a[sel], 1, 0).reshape(n_r, -1)


def summary(m, out_dir):
    red_i, red_f = reduce(m, "int"), reduce(m, "f64")
    r, cls = red_i["r"], red_i["cls"]
    n_r = len(r)
    core = cls == "core"
    idx = [int(np.argmin(np.abs(r - rv))) for rv in R_REPORT]
    lines = []
    p = lines.append
    p(f"sources: {dict((str(k), int(v)) for k, v in zip(*np.unique(cls, return_counts=True)))}, "
      f"{n_r} separations x {m['D'].shape[2]} directions each")
    for red, what in ((red_i, "integer paint (engine; the figure)"),
                      (red_f, "f64 paint (split only)")):
        p(f"|F_2L - F_GM| / |F_GM|, two-level with {what}: median / 95th pct")
        p(f"{'r [Mpc/h]':>10} {'core med':>10} {'core p95':>10} {'bdry med':>10} "
          f"{'bdry p95':>10}")
        for j in idx:
            c, b = _flat(red["rel"], core, n_r)[j], _flat(red["rel"], ~core, n_r)[j]
            p(f"{r[j]:10.3g} {np.median(c):10.2e} {pct(c, 95, None):10.2e} "
              f"{np.median(b):10.2e} {pct(b, 95, None):10.2e}")
    p("F_r / F_Newton, median [16, 84 pct] over all sources and directions")
    p(f"{'r':>10} {'two-level (int paint)':>30} {'two-level (f64 paint)':>30} "
      f"{'global mesh':>30}")
    for j in idx:
        cells = []
        for red, k in ((red_i, "2l"), (red_f, "2l"), (red_i, "gm")):
            a = _flat(red[f"ratio_{k}"], slice(None), n_r)[j]
            cells.append(f"{np.median(a):.5f} [{pct(a, 16, None):.5f},{pct(a, 84, None):.5f}]")
        p(f"{r[j]:10.3g} {cells[0]:>30} {cells[1]:>30} {cells[2]:>30}")
    p("tangential fraction |F_perp| / |F|: median / 95th pct")
    p(f"{'r':>10} {'two-level (int)':>20} {'global mesh':>20} {'Newton (Ewald)':>20}")
    for j in idx:
        cells = []
        for k in ("2l", "gm", "nt"):
            a = _flat(red_i[f"tan_{k}"], slice(None), n_r)[j]
            cells.append(f"{np.median(a):.2e} / {pct(a, 95, None):.2e}")
        p(f"{r[j]:10.3g} {cells[0]:>20} {cells[1]:>20} {cells[2]:>20}")
    s32 = m["side_coarse32"].reshape(len(m["sub"]), n_r, -1)
    p(f"f32 vs f64 coarse mesh (int paint), {len(m['sub'])} sources, rel. to |F_GM|, "
      "median per r: " + ", ".join(f"r={r[j]:.3g}: {np.median(s32[:, j]):.1e}" for j in idx)
      + f"; at r={r[-1]:.3g}: {np.median(s32[:, -1]):.1e} (max {s32[:, -1].max():.1e})")
    g = m["gates"]
    p(f"gates: TileField vs one_tile max |diff| {g[0]:.1e}; superposition vs "
      f"force_short_tiled + force_global max rel {g[1]:.1e}; Ewald screening max rel "
      f"{g[2]:.1e}")
    ns = dict((int(k), int(v)) for k, v in zip(*np.unique(m["n_solves"], return_counts=True)))
    p(f"tile solves per source (count: sources): {ns}; measurement wall "
      f"{float(m['wall']):.0f}s; peak RSS parent {m['rss'][0] / 1e9:.1f} GB, "
      f"global-mesh child {m['rss'][1] / 1e9:.1f} GB")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
        fh.write(text + "\n")


def export_curves(m, path):
    """The figure's curves (integer paint, as the engine runs) as JSON: per r, the radial force
    over Newton (medians; two-level 16-84 pct) and |F_2L - F_GM| / |F_GM| (median and 95th
    pct) for sources in tile cores and near tile boundaries."""
    red = reduce(m, "int")
    r, cls = red["r"], red["cls"]
    n_r = len(r)
    core = cls == "core"
    two = _flat(red["ratio_2l"], slice(None), n_r)
    out = dict(r=r, ratio_gm_med=np.median(_flat(red["ratio_gm"], slice(None), n_r), axis=1),
               ratio_2l_med=np.median(two, axis=1), ratio_2l_p16=pct(two, 16, 1),
               ratio_2l_p84=pct(two, 84, 1))
    for sel, tag in ((core, "core"), (~core, "bdry")):
        a = _flat(red["rel"], sel, n_r)
        out[f"rel_{tag}_med"] = np.median(a, axis=1)
        out[f"rel_{tag}_p95"] = pct(a, 95, 1)
    with open(path, "w") as fh:
        json.dump({k: np.asarray(v, float).tolist() for k, v in out.items()}, fh)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=os.path.join("runs", "validation", "force_accuracy"),
                    help="where the npz, curves and summary go (default %(default)s)")
    ap.add_argument("--from-cache", action="store_true",
                    help="reduce the saved force_accuracy.npz in --out-dir; no measurement")
    ap.add_argument("--phase", choices=("all", "mono"), default="all",
                    help="'mono' is the global-mesh child process")
    ap.add_argument("--mono-out")
    args = ap.parse_args()
    setup_jax(x64=True)
    if jax.default_backend() != "cpu":
        raise SystemExit("run on the CPU backend")
    if args.phase == "mono":
        return measure_mono(args.mono_out)
    cache = os.path.join(args.out_dir, "force_accuracy.npz")
    m = dict(np.load(cache)) if args.from_cache else measure(args.out_dir)
    summary(m, args.out_dir)
    export_curves(m, os.path.join(args.out_dir, "force_curves.json"))


if __name__ == "__main__":
    main()
