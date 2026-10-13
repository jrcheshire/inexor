"""The two-level force in production precision: self-force and a clustered field
(paper App. A.1).

    self       isolated unit sources at tile core / face / edge / corner positions (the classes
               and `cdev` geometry of force_accuracy.py; --reps sets of 40): the two-level force
               on each source at its own position, integer paints, coarse mesh float32, fine
               tiles float64, in units of the global-mesh force from a unit source one fine cell
               away.
    clustered  every particle of an evolved snapshot (`--positions`, the (N, 3) x.npy of an
               export of a --config run): the two-level force in production precision against
               one global n_fine^3 CIC mesh (float64 paint), per particle: |dF|/|F|, the signed
               component of dF along F, split by distance to the nearest tile face (inside or
               beyond one buffer). At cgh64 (512^3) the global 1024^3 mesh peaks near 111 GB.

CPU backend, float64 where not stated:

    pixi run --frozen python scripts/validation/force_production.py self
    pixi run --frozen python scripts/validation/force_production.py clustered \
        --config cgh64 --positions RUN/export/x.npy
"""

import argparse
import json
import os
import time

import numpy as np
from _common import maxrss, setup_jax, write_json


def run_self(args):
    import force_accuracy as FA

    rng = np.random.default_rng(FA.SEED + 1)
    g = FA.geometry()
    sets = [FA.sample_sources(g, rng) for _ in range(max(1, args.reps))]
    src = np.concatenate([a for a, _ in sets])
    cls = np.concatenate([b for _, b in sets])
    tf = FA.TileField(g, paint="int")
    t0 = time.perf_counter()
    # the global-mesh force of a unit source one fine cell away (position-independent to a few
    # per cent; 8 samples)
    dirs = FA.random_dirs(rng, 8)
    scale = float(np.median([np.linalg.norm(FA.mono_force(
        src[i], np.mod(src[i] + g["cell"] * dirs[i], g["L"])[None, :], g)) for i in range(8)]))
    self_f = []
    for s in src:
        p = s[None, :]
        sh, _ = tf.force(s, p)
        self_f.append(np.linalg.norm(sh + FA.long_force(s, p, g, fdtype=np.float32)))
    rel = np.asarray(self_f) / scale
    out = dict(geometry=dict(L=g["L"], n_fine=g["n_fine"], n_coarse=g["n_coarse"],
                             n_tile=g["n_tile"], cell=g["cell"]),
               n_sources=int(len(src)), wall_s=time.perf_counter() - t0,
               cell_force_scale=scale,
               self_over_cell_force=dict(
                   all=dict(median=float(np.median(rel)), p95=float(np.percentile(rel, 95)),
                            max=float(rel.max())),
                   **{c: dict(median=float(np.median(rel[cls == c])), max=float(rel[cls == c].max()))
                      for c in ("core", "face", "edge", "corner")}))
    return out


def _brick_index(x, n_fine, n_brick, cell):
    """Particles sorted by brick -> (order, brick starts, bricks per side)."""
    nb = n_fine // n_brick
    b = np.mod(np.floor(x / (cell * n_brick)).astype(np.int64), nb)
    bid = (b[:, 0] * nb + b[:, 1]) * nb + b[:, 2]
    order = np.argsort(bid, kind="stable")
    return order, np.searchsorted(bid[order], np.arange(nb**3 + 1)), nb


def _tile_members(order, starts, nb, tijk, n_tile, b_fine, n_brick):
    """Particle indices of the brick union covering tile + buffer (refuses a wrapping span)."""
    pad = -(-b_fine // n_brick)
    span = n_tile // n_brick + 2 * pad
    if span > nb:
        raise ValueError(f"tile + buffer spans {span} bricks of {nb}: it would double-count")
    lo = np.asarray(tijk, np.int64) * (n_tile // n_brick) - pad
    idx = [order[starts[b]:starts[b + 1]]
           for i in range(span) for j in range(span) for k in range(span)
           for b in [(((lo[0] + i) % nb) * nb + (lo[1] + j) % nb) * nb + (lo[2] + k) % nb]]
    return np.concatenate(idx)


def run_clustered(args):
    import jax.numpy as jnp

    from inexor import forces as F
    from inexor.plan import engine_config

    if not args.positions:
        raise SystemExit("clustered needs --positions")
    ec = engine_config(args.config)
    ec.validate()
    L, n = ec.box_size, ec.n_total
    x = np.mod(np.load(args.positions).astype(np.float64).reshape(-1, 3), L)
    if x.shape[0] != n:
        raise SystemExit(f"{x.shape[0]} positions, expected {n}")
    t0 = time.perf_counter()
    g_long, _ = F.force_global(x, ec.n_coarse, L, n, "long", r_s=ec.r_s, match=ec.coarse_match,
                               assign="tsc", paint="int", fdtype=np.float32)
    t_long = time.perf_counter() - t0
    order, starts, nb = _brick_index(x, ec.n_fine, ec.n_brick, ec.fine_cell)

    def members(t):
        return _tile_members(order, starts, nb, t, ec.n_tile, ec._b_realized, ec.n_brick)

    cap = -(-max(len(members(t)) for t in ec.tiles) // 1024) * 1024
    g_short, _ = F.force_short_tiled(x, ec.n_fine, L, n, ec.n_tile, ec.b_fine, members, cap,
                                     r_s=ec.r_s, paint="int", fdtype=jnp.float64,
                                     max_accumulate_bytes=4 << 30)
    two = np.asarray(g_long, np.float64) + np.asarray(g_short)
    del g_long, g_short
    t_two = time.perf_counter() - t0
    mono, _ = F.force_global(x, ec.n_fine, L, n, "mono", assign="cic", paint="f64")
    mono = np.asarray(mono)
    t_all = time.perf_counter() - t0
    fmag = np.linalg.norm(mono, axis=1)
    d = two - mono
    rel = np.linalg.norm(d, axis=1) / fmag
    par = np.einsum("ij,ij->i", d, mono) / fmag**2
    # distance to the nearest tile face, in units of the buffer
    tile_len, buf_len = ec.n_tile * ec.fine_cell, ec._b_realized * ec.fine_cell
    u = np.mod(x, tile_len)
    face = np.minimum(u, tile_len - u).min(axis=1) / buf_len
    sel = dict(all=np.ones(n, bool), near_face=face < 1.0, core=face >= 1.0)
    out = dict(geometry=dict(L=L, n_part=ec.n_part, n_fine=ec.n_fine, n_coarse=ec.n_coarse,
                             n_tile=ec.n_tile, buf=ec._b_realized, mono_mesh=ec.n_fine),
               wall_s=dict(long=t_long, two_level=t_two, all=t_all))
    for name, m in sel.items():
        r, p = rel[m], par[m]
        if not m.any():     # e.g. no particle beyond one buffer of a face when tile <= 2 buf
            out[name] = dict(n=0)
            continue
        out[name] = dict(n=int(m.sum()), rel_median=float(np.median(r)),
                         rel_p95=float(np.percentile(r, 95)), rel_p99=float(np.percentile(r, 99)),
                         signed_parallel_mean=float(p.mean()),
                         signed_parallel_median=float(np.median(p)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("self", "clustered"))
    ap.add_argument("--out", default=None,
                    help="output JSON (default runs/validation/force_production/<mode>.json)")
    ap.add_argument("--reps", type=int, default=10, help="self: source sets of 40")
    ap.add_argument("--config", default="cgh64", help="clustered: the snapshot's preset")
    ap.add_argument("--positions", default=None, help="clustered: (N, 3) positions .npy")
    args = ap.parse_args()
    out_path = args.out or os.path.join("runs", "validation", "force_production",
                                        f"{args.mode}.json")
    setup_jax(x64=True)
    out = run_self(args) if args.mode == "self" else run_clustered(args)
    out["peak_rss_gb"] = maxrss() / 1e9
    write_json(out_path, out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
