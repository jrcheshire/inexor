"""FoF halos of two evolved states from the same ICs, compared (inexor vs DISCO-DJ).

    find SRC --out CAT.npz     FoF (kdcount, periodic) on one state: halo particle counts,
                               minimum-image COMs and the matter CIC density mesh.
                               SRC: an inexor export dir (export.json + x.npy) or a
                               `disco_crosscheck.py evolve` npz (x + meta).
    compare A.npz B.npz        abundance per particle-count bin, mutual-nearest halo matching,
                               halo bias P_hm/P_mm per code and the halo-halo cross-correlation
                               between codes; tables on stdout, everything in --out JSON.

Both states go through the same finder, linking length and mesh, so differences are the
codes'. Masses are N x m_p (FoF, b in units of the mean interparticle spacing). Spectra are
CIC without window deconvolution: b = P_hm/P_mm and the cross-correlation coefficient cancel
the window. `--shift` finds on positions moved by L/2 and moves the outputs back, so
`compare` of a plain and a shifted catalog checks the periodic handling (identical).

kdcount is pip-only and builds a C extension, so this runs in a throwaway env:

    pixi exec --spec python=3.12 --spec numpy --spec scipy --spec pip --spec c-compiler -- \
        bash -c 'pip install -q kdcount && python -I scripts/compare/halo_compare.py ...'
"""

import argparse
import json
import os
import time

import numpy as np

RHO_CRIT = 2.77536627e11  # h^2 Msun / Mpc^3
N_EDGES = [20, 50, 150, 500, 1500, 10**9]


def load_state(src):
    """Positions (float64, Mpc/h in [0, L)), box, particles per side, Omega_m, source label."""
    if os.path.isdir(src):
        with open(os.path.join(src, "export.json")) as f:
            h = json.load(f)
        x = np.load(os.path.join(src, h["files"]["x"]))
        om = h["cosmology"]["Omega_m"]
        meta = dict(code="inexor", a=float(h["a"]), provenance=h.get("provenance"))
        box, n_part = float(h["box_size"]), int(h["n_part"])
    else:
        d = np.load(src)
        m = json.loads(str(d["meta"]))
        x = d["x"]
        c = m["disco_cosmo"]
        om = c["Omega_c"] + c["Omega_b"]
        meta = dict(code=m.get("code", "discodj"),
                    a=float(m.get("a_out", d["a_steps"][-1])), stop_step=m.get("stop_step"),
                    n_mesh=m.get("n_mesh"), precision=m.get("precision"))
        box, n_part = float(m["box_size"]), int(m["n_part"])
    x = np.mod(np.asarray(x, np.float64).reshape(-1, 3), box)
    assert x.shape[0] == n_part**3, (x.shape, n_part)
    return x, box, n_part, om, meta


def cic_mesh(pos, box, n, chunk=1 << 24):
    """CIC mass assignment of `pos` onto an n^3 periodic mesh; returns counts per cell."""
    rho = np.zeros(n**3, np.float64)
    for s in range(0, len(pos), chunk):
        p = pos[s:s + chunk] * (n / box)
        i0 = np.floor(p).astype(np.int64)
        f = p - i0
        i0 %= n
        i1 = (i0 + 1) % n
        for dx in (0, 1):
            ix, wx = (i1[:, 0], f[:, 0]) if dx else (i0[:, 0], 1 - f[:, 0])
            for dy in (0, 1):
                iy, wy = (i1[:, 1], f[:, 1]) if dy else (i0[:, 1], 1 - f[:, 1])
                for dz in (0, 1):
                    iz, wz = (i1[:, 2], f[:, 2]) if dz else (i0[:, 2], 1 - f[:, 2])
                    rho += np.bincount((ix * n + iy) * n + iz, wx * wy * wz, minlength=n**3)
    return rho.reshape(n, n, n)


def fof_catalog(x, box, n_part, b, n_floor):
    """kdcount FoF; returns (counts, minimum-image COMs) of groups with >= n_floor members,
    ordered by decreasing count."""
    from kdcount import cluster, models

    ll = b * box / n_part
    data = models.points(np.minimum(x, np.nextafter(box, 0)), boxsize=box)
    f = cluster.fof(data, linking_length=ll)
    keep = int(np.searchsorted(-f.length, -n_floor, side="right"))
    counts = np.asarray(f.length[:keep], np.int64)
    lab = f.labels
    sel = lab < keep
    ref = x[f.indices[f.offset[:keep]]]
    rel = x[sel] - ref[lab[sel]]
    rel -= box * np.round(rel / box)
    com = np.stack([np.bincount(lab[sel], rel[:, d], minlength=keep) for d in range(3)], 1)
    com = np.mod(ref + com / counts[:, None], box)
    return counts, com, ll


def cmd_find(a):
    t0 = time.time()
    x, box, n_part, om, meta = load_state(a.src)
    if a.shift:
        x = np.mod(x + 0.5 * box, box)
    t1 = time.time()
    counts, com, ll = fof_catalog(x, box, n_part, a.b, a.n_floor)
    t2 = time.time()
    rho = cic_mesh(x, box, a.n_mesh)
    if a.shift:
        com = np.mod(com - 0.5 * box, box)
        rho = np.roll(rho, -a.n_mesh // 2, axis=(0, 1, 2))
    delta = (rho / rho.mean() - 1).astype(np.float32)
    m_p = RHO_CRIT * om * (box / n_part) ** 3
    meta.update(src=os.path.abspath(a.src), box=box, n_part=n_part, omega_m=om, m_p=m_p,
                b=a.b, linking_length=ll, n_floor=a.n_floor, n_mesh=a.n_mesh, shift=a.shift,
                t_load=t1 - t0, t_fof=t2 - t1, t_mesh=time.time() - t2)
    np.savez(a.out, counts=counts, com=com, delta_m=delta, meta=json.dumps(meta))
    print(f"[find] {meta['code']} {n_part}^3 L={box:g}: {len(counts)} groups >= {a.n_floor} "
          f"(>= 20: {(counts >= 20).sum()}), largest {counts[0] if len(counts) else 0}; "
          f"ll={ll:.4f} m_p={m_p:.3e}; load {t1 - t0:.0f} s, fof {t2 - t1:.0f} s, "
          f"mesh {meta['t_mesh']:.0f} s -> {a.out}")


def load_cat(p):
    d = np.load(p)
    return dict(counts=d["counts"], com=d["com"], delta_m=d["delta_m"],
                meta=json.loads(str(d["meta"])))


def r200m(counts, box, n_part):
    """R_200m (Mpc/h) of a halo of `counts` particles."""
    return (box / n_part) * (3 * counts / (800 * np.pi)) ** (1 / 3)


def match(A, B, box, n_part, r_fac):
    """Mutual nearest COMs within r_fac x R_200m of the larger halo. Returns (ia, ib) pairs."""
    from scipy.spatial import cKDTree

    ta = cKDTree(np.minimum(A["com"], np.nextafter(box, 0)), boxsize=box)
    tb = cKDTree(np.minimum(B["com"], np.nextafter(box, 0)), boxsize=box)
    dab, jab = tb.query(A["com"])
    _, jba = ta.query(B["com"])
    ia = np.arange(len(A["counts"]))
    mutual = jba[jab] == ia
    rmax = r_fac * r200m(np.maximum(A["counts"], B["counts"][jab]), box, n_part)
    ok = mutual & (dab < rmax)
    return ia[ok], jab[ok], dab[ok]


def k_bins(n, box):
    kf = 2 * np.pi / box
    k1 = np.fft.fftfreq(n, 1 / n) * kf
    kz = np.fft.rfftfreq(n, 1 / n) * kf
    kk = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2)
    w = np.full(kz.shape, 2.0)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    return kk, np.broadcast_to(w, kk.shape), kf


def field_k(delta):
    return np.fft.rfftn(delta)


def binned(fa, fb, kk, w, edges, norm):
    p = (fa * np.conj(fb)).real * w
    idx = np.digitize(kk, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    num = np.bincount(idx[ok], p[ok], minlength=len(edges) - 1)
    den = np.bincount(idx[ok], w[ok], minlength=len(edges) - 1)
    kmean = np.bincount(idx[ok], (kk * w)[ok], minlength=len(edges) - 1) / den
    return kmean, num / den * norm, den


def cmd_compare(a):
    A, B = load_cat(a.a), load_cat(a.b)
    ma, mb = A["meta"], B["meta"]
    for key in ("box", "n_part", "b", "n_mesh"):
        if ma[key] != mb[key]:
            raise SystemExit(f"{key} differs: {ma[key]} vs {mb[key]}")
    if abs(ma["a"] / mb["a"] - 1) > 1e-6:
        raise SystemExit(f"the states are at different epochs: a = {ma['a']} vs {mb['a']}")
    box, n_part, n = ma["box"], ma["n_part"], ma["n_mesh"]
    m_p = ma["m_p"]
    la, lb = a.labels
    out = dict(a=ma, b=mb, labels=a.labels, n_edges=N_EDGES)

    print(f"\n{la} vs {lb}: {n_part}^3 in {box:g} Mpc/h, z={1 / ma['a'] - 1:.3f}, "
          f"FoF b={ma['b']}, m_p={m_p:.3e} Msun/h\n")
    print("Abundance (particle-count bins; sigma = Poisson, uncorrelated, an upper bound)")
    print(f"{'N bin':>14} {'M [Msun/h]':>19} {la:>9} {lb:>9} {'ratio':>7} {'sigma':>6}")
    rows = []
    for lo, hi in zip(N_EDGES[:-1], N_EDGES[1:]):
        na = int(((A["counts"] >= lo) & (A["counts"] < hi)).sum())
        nb = int(((B["counts"] >= lo) & (B["counts"] < hi)).sum())
        r = na / nb if nb else np.nan
        s = r * np.sqrt(1 / max(na, 1) + 1 / max(nb, 1))
        hi_s = "inf" if hi >= 10**9 else str(hi)
        print(f"{lo:>6}-{hi_s:<7} {lo * m_p:9.2e}-{min(hi, 10**7) * m_p:<9.2e} {na:9d} {nb:9d} "
              f"{r:7.4f} {s:6.4f}")
        rows.append(dict(lo=lo, hi=hi, n_a=na, n_b=nb, ratio=r, sigma=s))
    out["abundance"] = rows

    ia, ib, dist = match(A, B, box, n_part, a.match_r)
    print(f"\nMatching (mutual nearest COMs within {a.match_r} R_200m of the larger halo; "
          f"binned on {la}'s count)")
    print(f"{'N bin':>14} {'f_match ' + la:>14} {'f_match ' + lb:>14} {'med lnN_b/N_a':>14} "
          f"{'16-84%':>15} {'med d/R200m':>12}")
    rows = []
    ca, cb = A["counts"], B["counts"]
    for lo, hi in zip(N_EDGES[:-1], N_EDGES[1:]):
        sa = (ca >= lo) & (ca < hi)
        sb = (cb >= lo) & (cb < hi)
        mpair = sa[ia]
        fa = mpair.sum() / max(sa.sum(), 1)
        fb = sb[ib].sum() / max(sb.sum(), 1)
        lr = np.log(cb[ib[mpair]] / ca[ia[mpair]])
        dr = dist[mpair] / r200m(ca[ia[mpair]], box, n_part)
        q = np.percentile(lr, [16, 50, 84]) if len(lr) else [np.nan] * 3
        md = float(np.median(dr)) if len(dr) else np.nan
        hi_s = "inf" if hi >= 10**9 else str(hi)
        print(f"{lo:>6}-{hi_s:<7} {fa:14.4f} {fb:14.4f} {q[1]:14.4f} {q[0]:7.3f},{q[2]:<7.3f} "
              f"{md:12.3f}")
        rows.append(dict(lo=lo, hi=hi, f_match_a=fa, f_match_b=fb, lnratio_p16_50_84=list(q),
                         med_offset_r200m=md))
    out["matching"] = rows

    kk, w, kf = k_bins(n, box)
    edges = np.concatenate([kf * np.arange(1, 12, 2), [0.3, 0.4, 0.5]])
    edges = np.unique(edges[edges <= 0.5 + 1e-9])
    norm = box**3 / float(n) ** 6
    fma, fmb = field_k(A["delta_m"]), field_k(B["delta_m"])
    km, pmm_a, nmodes = binned(fma, fma, kk, w, edges, norm)
    _, pmm_b, _ = binned(fmb, fmb, kk, w, edges, norm)
    _, pmm_ab, _ = binned(fma, fmb, kk, w, edges, norm)
    spec = dict(k=km.tolist(), n_modes=nmodes.tolist(), pmm_ratio=(pmm_a / pmm_b).tolist(),
                r_mm=(pmm_ab / np.sqrt(pmm_a * pmm_b)).tolist(), bins=[])
    print(f"\nSpectra ({n}^3 CIC, k <= 0.5 h/Mpc). Matter: P_{la}/P_{lb} - 1 per k bin:")
    print("  k     " + " ".join(f"{k:7.3f}" for k in km))
    print("  dP/P  " + " ".join(f"{r - 1:+.0e}" for r in pmm_a / pmm_b))
    print(f"\nHalo bias b = P_hm/P_mm per code, b_{la}/b_{lb} - 1, and r_hh = cross-correlation "
          f"of the two codes' halo fields")
    for lo, hi in zip(N_EDGES[:-1], N_EDGES[1:]):
        sel = [(C["counts"] >= lo) & (C["counts"] < hi) for C in (A, B)]
        nha, nhb = (int(s.sum()) for s in sel)
        if min(nha, nhb) < 50:
            continue
        fha, fhb = (field_k((rho / rho.mean() - 1).astype(np.float32))
                    for rho in (cic_mesh(C["com"][s], box, n) for C, s in zip((A, B), sel)))
        _, phm_a, _ = binned(fha, fma, kk, w, edges, norm)
        _, phm_b, _ = binned(fhb, fmb, kk, w, edges, norm)
        _, phh_a, _ = binned(fha, fha, kk, w, edges, norm)
        _, phh_b, _ = binned(fhb, fhb, kk, w, edges, norm)
        _, phh_ab, _ = binned(fha, fhb, kk, w, edges, norm)
        ba, bb = phm_a / pmm_a, phm_b / pmm_b
        r_hh = phh_ab / np.sqrt(phh_a * phh_b)
        hi_s = "inf" if hi >= 10**9 else str(hi)
        print(f"  N {lo}-{hi_s} ({nha} / {nhb} halos)")
        print("    b_a    " + " ".join(f"{v:7.3f}" for v in ba))
        print("    db/b   " + " ".join(f"{v - 1:+7.4f}" for v in ba / bb))
        print("    r_hh   " + " ".join(f"{v:7.4f}" for v in r_hh))
        spec["bins"].append(dict(lo=lo, hi=hi, n_a=nha, n_b=nhb, b_a=ba.tolist(), b_b=bb.tolist(),
                                 r_hh=r_hh.tolist(),
                                 shot_a=box**3 / nha, shot_b=box**3 / nhb))
    out["spectra"] = spec
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print(f"\n-> {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("find")
    f.add_argument("src")
    f.add_argument("--out", required=True)
    f.add_argument("--b", type=float, default=0.2)
    f.add_argument("--n-floor", type=int, default=10, help="smallest group kept in the catalog")
    f.add_argument("--n-mesh", type=int, default=256)
    f.add_argument("--shift", action="store_true",
                   help="find on positions shifted by L/2 (periodicity self-test); outputs unshifted")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--labels", nargs=2, default=["inexor", "DISCO-DJ"])
    c.add_argument("--match-r", type=float, default=2.0)
    c.add_argument("--out")
    a = ap.parse_args()
    {"find": cmd_find, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
