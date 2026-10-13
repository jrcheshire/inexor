"""inexor against DISCO-DJ in velocity, from the same initial particles.

Field level, since an export carries no particle identity: density, momentum
p = sum_i v_i W(x - x_i) and its divergence theta = i k . p, each code painted with CIC on one
n^3 mesh. Per bin (width 2 k_f from k_f / 2 to the mesh Nyquist): the ratio of the two codes'
auto spectra and their cross-correlation r = Re<A B*> / sqrt(<|A|^2><|B|^2>). Velocities in
dx/dD (Mpc/h per unit growth factor): inexor's km/s export is divided by the
`peculiar_velocity_factor` its header records; DISCO-DJ's `v_d` already is.

Inputs: an inexor export directory (x.npy, v.npy, export.json) and the DISCO-DJ final state
from `scripts/compare/disco_crosscheck.py evolve` on the same ICs (its npz, keys x and v_d,
or a directory holding x.npy and v_d.npy).

    pixi run --frozen python scripts/validation/velocity_compare.py \\
        --inexor RUN/export --disco W/disco_out.npz --n-mesh 512

The paint is plain numpy (np.bincount over chunks of particles; each chunk's bincount starts
from the grid's current values, so every cell sums its contributions in particle order).
"""

import argparse
import json
import os
import time

import numpy as np
from _common import maxrss, write_json


def paint_cic_fields(x, v, box, n, chunk):
    """CIC-painted (count, count x v_x, count x v_y, count x v_z) on an n^3 mesh, flat."""
    cell = box / n
    grids = [np.zeros(n**3) for _ in range(4)]
    seed_idx = np.arange(n**3)
    two = np.arange(2)
    for s in range(0, x.shape[0], chunk):
        xc = np.asarray(x[s:s + chunk], np.float64) % box
        vc = np.asarray(v[s:s + chunk], np.float64)
        u = xc / cell - 0.5
        f = np.floor(u)
        d = u - f
        i0 = f.astype(np.int64) % n
        ii = [(i0[:, ax, None] + two) % n for ax in range(3)]
        ww = [np.stack([1.0 - d[:, ax], d[:, ax]], axis=-1) for ax in range(3)]
        idx = ((ii[0][:, :, None, None] * n + ii[1][:, None, :, None]) * n
               + ii[2][:, None, None, :]).ravel()
        w = ((ww[0][:, :, None, None] * ww[1][:, None, :, None])
             * ww[2][:, None, None, :]).reshape(-1, 8)
        del u, f, d, i0, ii, ww
        first = s == 0
        full_idx = idx if first else np.concatenate([seed_idx, idx])
        del idx
        for g, comp in zip(grids, (None, 0, 1, 2)):
            wc = (w if comp is None else w * vc[:, comp, None]).ravel()
            g[:] = np.bincount(full_idx, weights=wc if first else np.concatenate([g, wc]),
                               minlength=n**3)
            del wc
    return [g.reshape(n, n, n) for g in grids]


def fields(x, v, box, n, chunk):
    rho, px, py, pz = paint_cic_fields(x, v, box, n, chunk)
    cell = box / n
    mean = x.shape[0] / n**3
    k1 = 2 * np.pi * np.fft.fftfreq(n, d=cell)
    kz = 2 * np.pi * np.fft.rfftfreq(n, d=cell)
    d = np.fft.rfftn(rho / mean - 1.0)
    p = [np.fft.rfftn(f / mean) for f in (px, py, pz)]
    theta = 1j * (k1[:, None, None] * p[0] + k1[None, :, None] * p[1] + kz[None, None, :] * p[2])
    return d, p, theta


def load_disco(path):
    if os.path.isdir(path):
        return (np.load(os.path.join(path, "x.npy"), mmap_mode="r"),
                np.load(os.path.join(path, "v_d.npy"), mmap_mode="r"))
    with np.load(path) as f:
        return f["x"], f["v_d"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inexor", required=True, help="export directory (x.npy, v.npy, export.json)")
    ap.add_argument("--disco", required=True,
                    help="DISCO-DJ evolve npz (x, v_d), or a directory with x.npy and v_d.npy")
    ap.add_argument("--n-mesh", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=1 << 25, help="particles per paint chunk")
    ap.add_argument("--out", default=os.path.join("runs", "validation", "velocity_compare.json"),
                    help="output JSON (default %(default)s)")
    args = ap.parse_args()
    n = args.n_mesh
    with open(os.path.join(args.inexor, "export.json")) as fh:
        hdr = json.load(fh)
    box = float(hdr["box_size"])
    fac = 1.0 if hdr.get("velocity_is_dtime") else float(hdr["peculiar_velocity_factor"])
    t0 = time.perf_counter()
    xi = np.load(os.path.join(args.inexor, "x.npy"), mmap_mode="r")
    vi = np.load(os.path.join(args.inexor, "v.npy"), mmap_mode="r")
    A = fields(xi, np.asarray(vi, np.float64) / fac, box, n, args.chunk)
    xd, vd = load_disco(args.disco)
    if xd.shape != xi.shape:
        raise SystemExit(f"DISCO-DJ has {xd.shape[0]:,} particles, inexor {xi.shape[0]:,}")
    B = fields(xd, vd, box, n, args.chunk)
    cell = box / n
    k1 = 2 * np.pi * np.fft.fftfreq(n, d=cell)
    kz = 2 * np.pi * np.fft.rfftfreq(n, d=cell)
    kk = np.sqrt(k1[:, None, None]**2 + k1[None, :, None]**2 + kz[None, None, :]**2).ravel()
    wz = np.full(kz.size, 2.0)
    wz[0] = 1.0
    wz[-1] = 1.0
    w = np.broadcast_to(wz, (n, n, kz.size)).ravel()
    kf = 2 * np.pi / box
    edges = np.arange(0.5 * kf, np.pi / cell, 2 * kf)
    idx = np.digitize(kk, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    nb = len(edges) - 1

    def binsum(arr):
        return np.bincount(idx[ok], weights=(arr.ravel() * w)[ok], minlength=nb)[:nb]

    cnt = binsum(np.ones(kk.size))
    res = dict(box=box, n_mesh=n, velocity_unit="dx/dD (Mpc/h per unit D)", inexor_factor=fac,
               k=list(binsum(kk.reshape(n, n, -1)) / np.maximum(cnt, 1)), n_modes=list(cnt))
    for name, a, b in (("density", [A[0]], [B[0]]), ("momentum", A[1], B[1]),
                       ("divergence", [A[2]], [B[2]])):
        paa = sum(binsum(np.abs(x)**2) for x in a)
        pbb = sum(binsum(np.abs(x)**2) for x in b)
        pab = sum(binsum((x * np.conj(y)).real) for x, y in zip(a, b))
        res[name] = dict(ratio=list(paa / pbb), r=list(pab / np.sqrt(paa * pbb)))
    res["wall_s"] = time.perf_counter() - t0
    write_json(args.out, res, indent=None)
    k = np.asarray(res["k"])
    for name in ("density", "momentum", "divergence"):
        ra, rr = np.asarray(res[name]["ratio"]), np.asarray(res[name]["r"])
        for lo, hi in ((0.0, 0.5), (0.5, 1.57), (1.57, 3.0), (3.0, 5.0)):
            m = (k >= lo) & (k < hi)
            print(f"{name:10s} k {lo:4.2f}-{hi:4.2f}: ratio-1 in [{(ra[m] - 1).min():+.1e}, "
                  f"{(ra[m] - 1).max():+.1e}]  1-r max {(1 - rr[m]).max():.1e}")
    print(f"wall {res['wall_s']:.0f} s, peak RSS {maxrss() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
