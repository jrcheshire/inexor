"""Matter power spectrum to the mesh Nyquist, independent of inexor's own estimator (paper
App. A.2): TSC assignment with interlacing (two grids offset by half a cell, combined in
Fourier space), the TSC window divided out, shot noise V/N subtracted, linear bins.

    measure  positions (an (N, 3) .npy in [0, box), e.g. an export's x.npy, or an .npz with key
             `x`, e.g. a disco_crosscheck.py evolve output) -> one spectrum JSON. Default bins
             of width 2 k_f from k_f / 2; `--dk`, `--k0` set others (the card's bins at L = 256:
             --k0 0 --dk 0.02454369260617026).
    ratio    two spectra on the same bins -> max |P_a / P_b - 1| over 0.02 < k <= k_max for
             several k_max.

    pixi run --frozen python scripts/validation/pk_highk.py measure RUN/export/x.npy \\
        --box 256 --n-mesh 1024 --out runs/validation/pk_highk/inexor_n1024.json
    pixi run --frozen python scripts/validation/pk_highk.py ratio A.json B.json

The paint is plain numpy (np.bincount): particles sorted by x-plane, painted in blocks of 8
planes, even blocks first; each block's bincount starts from the grid's current values, so every
cell sums its contributions in a fixed order. At 1024^3 with 512^3 particles the peak is
~50 GB.
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.fft
from _common import maxrss, write_json

BLOCK = 8          # x-planes per paint block
K_MAX_REPORT = (1.57, 3.0, 5.0, 6.0, 8.0, 10.0, 12.0)


def load_positions(path):
    """(N, 3) positions from an .npy (memory-mapped) or an .npz with key `x`."""
    if path.endswith(".npz"):
        with np.load(path) as f:
            return f["x"]
    return np.load(path, mmap_mode="r")


def paint_tsc(x, box, n, shift):
    """TSC density (unit masses) on an n^3 mesh, grid points at (i - shift) cells."""
    cell = box / n
    # cell index of each particle along x after the shift; sort; blocks of BLOCK planes
    ix = np.floor(x[:, 0] / cell + shift + 0.5).astype(np.int64) % n
    order = np.argsort(ix, kind="stable")
    xs = np.ascontiguousarray(x[order], dtype=np.float64)
    ixs = ix[order]
    del ix, order
    starts = np.searchsorted(ixs, np.arange(0, n + BLOCK, BLOCK))
    del ixs
    grid = np.zeros((n, n, n), np.float64)
    nb = starts.shape[0] - 1
    span = BLOCK + 4            # the block's planes, +-1 for the stencil, +-1 for the sort key
    off = np.arange(3) - 1
    for parity in range(2):
        for b in range(parity, nb, 2):
            p = xs[starts[b]:starts[b + 1]]
            if p.shape[0] == 0:
                continue
            u = p / cell + shift
            c = np.floor(u + 0.5)
            d = u - c
            w = np.stack([0.5 * (0.5 - d) ** 2, 0.75 - d * d, 0.5 * (0.5 + d) ** 2], axis=-1)
            ci = c.astype(np.int64)
            base = b * BLOCK - 2
            lx = (ci[:, 0, None] + off - base) % n
            if lx.max() >= span:
                raise RuntimeError(f"block {b}: a particle paints outside its slab")
            ly = (ci[:, 1, None] + off) % n
            lz = (ci[:, 2, None] + off) % n
            idx = (lx[:, :, None, None] * n + ly[:, None, :, None]) * n + lz[:, None, None, :]
            wt = (w[:, 0, :, None, None] * w[:, 1, None, :, None]) * w[:, 2, None, None, :]
            planes = (base + np.arange(span)) % n
            seed = grid[planes].ravel()
            loc = np.bincount(np.concatenate([np.arange(seed.size), idx.ravel()]),
                              weights=np.concatenate([seed, wt.ravel()]), minlength=seed.size)
            grid[planes] = loc.reshape(span, n, n)
    return grid


def tsc_window(n, box):
    k1 = 2 * np.pi * np.fft.fftfreq(n, d=box / n)
    kz = 2 * np.pi * np.fft.rfftfreq(n, d=box / n)
    h = box / n / 2

    def s(k):
        return np.sinc(k * h / np.pi) ** 3   # np.sinc(x) = sin(pi x) / (pi x)

    return k1, kz, s(k1)[:, None, None] * s(k1)[None, :, None] * s(kz)[None, None, :]


def measure(a):
    t0 = time.time()
    x = load_positions(a.positions)
    N, n, L = x.shape[0], a.n_mesh, a.box
    nbar_cell = N / n**3
    g1 = paint_tsc(x, L, n, 0.0)
    if abs(g1.sum() - N) >= 1e-6 * N:
        raise RuntimeError(f"painted mass {g1.sum()} != {N}")
    d1 = scipy.fft.rfftn(g1 / nbar_cell - 1.0, workers=a.workers)
    del g1
    k1, kz, W = tsc_window(n, L)
    if not a.no_interlace:
        g2 = paint_tsc(x, L, n, 0.5)
        d2 = scipy.fft.rfftn(g2 / nbar_cell - 1.0, workers=a.workers)
        del g2
        h = L / n / 2  # the second grid samples the field shifted by -h along each axis
        d2 *= np.exp(1j * k1 * h)[:, None, None]
        d2 *= np.exp(1j * k1 * h)[None, :, None]
        d2 *= np.exp(1j * kz * h)[None, None, :]
        d1 += d2
        d1 *= 0.5
        del d2
    p3 = (np.abs(d1) ** 2) / W**2 * (L**3 / n**6)
    del d1
    kk = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2)
    # rfft half-space: modes with 0 < kz < Nyquist stand for two
    mult = np.full(kz.shape, 2.0)
    mult[0] = 1.0
    if n % 2 == 0:
        mult[-1] = 1.0
    mult = np.broadcast_to(mult[None, None, :], kk.shape)
    kf = 2 * np.pi / L
    dk = a.dk or 2 * kf
    kmax = np.pi * n / L  # Nyquist
    k0 = 0.5 * kf if a.k0 is None else a.k0
    edges = np.arange(k0, kmax + dk, dk)
    idx = np.digitize(kk.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    nb = len(edges) - 1
    cnt = np.bincount(idx[ok], weights=mult.ravel()[ok], minlength=nb)
    ksum = np.bincount(idx[ok], weights=(kk.ravel() * mult.ravel())[ok], minlength=nb)
    psum = np.bincount(idx[ok], weights=(p3.ravel() * mult.ravel())[ok], minlength=nb)
    good = cnt > 0
    shot = L**3 / N
    res = dict(k=(ksum[good] / cnt[good]).tolist(), p=(psum[good] / cnt[good] - shot).tolist(),
               p_raw=(psum[good] / cnt[good]).tolist(), n_modes=cnt[good].tolist(), shot=shot,
               box=L, n_mesh=n, n_part=int(N), interlaced=not a.no_interlace,
               k_nyquist=kmax, assignment="TSC", pos=a.positions)
    out = a.out or os.path.join(
        "runs", "validation", "pk_highk",
        f"{os.path.basename(os.path.dirname(os.path.abspath(a.positions)))}_n{n}"
        f"{'_nointerlace' if a.no_interlace else ''}.json")
    write_json(out, res, indent=None)
    print(f"{a.positions}: N={N:,} n_mesh={n} in {time.time() - t0:.0f} s, "
          f"peak RSS {maxrss() / 1e9:.1f} GB")


def ratio(a):
    with open(a.a) as fh:
        pa = json.load(fh)
    with open(a.b) as fh:
        pb = json.load(fh)
    k = np.asarray(pa["k"])
    if not np.allclose(k, pb["k"]):
        raise SystemExit("the two spectra are on different bins")
    r = np.asarray(pa["p"]) / np.asarray(pb["p"]) - 1.0
    rows = []
    for kk in K_MAX_REPORT:
        m = (k > 0.02) & (k <= kk)
        if m.any():
            rows.append(dict(k_max=kk, max_abs_ratio_minus_1=float(np.abs(r[m]).max())))
            print(f"  max |P_a/P_b - 1| for k <= {kk}: {rows[-1]['max_abs_ratio_minus_1']:.2e}")
    out = a.out or os.path.join("runs", "validation", "pk_highk", "ratio.json")
    write_json(out, dict(a=a.a, b=a.b, k=k.tolist(), ratio_minus_1=r.tolist(), max_abs=rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="P(k) of one set of positions")
    m.add_argument("positions", help="(N, 3) .npy, or .npz with key x")
    m.add_argument("--box", type=float, required=True, help="box side, Mpc/h")
    m.add_argument("--n-mesh", type=int, default=1024)
    m.add_argument("--out", default=None, help="output JSON (default under "
                   "runs/validation/pk_highk/, named after the positions' directory)")
    m.add_argument("--dk", type=float, default=None, help="bin width (default 2 k_f)")
    m.add_argument("--k0", type=float, default=None, help="first bin edge (default k_f/2)")
    m.add_argument("--no-interlace", action="store_true")
    m.add_argument("--workers", type=int, default=-1, help="scipy.fft workers (-1: all cores)")
    r = sub.add_parser("ratio", help="max |P_a/P_b - 1| of two measure outputs")
    r.add_argument("a")
    r.add_argument("b")
    r.add_argument("--out", default=None,
                   help="output JSON (default runs/validation/pk_highk/ratio.json)")
    a = ap.parse_args()
    measure(a) if a.cmd == "measure" else ratio(a)


if __name__ == "__main__":
    main()
