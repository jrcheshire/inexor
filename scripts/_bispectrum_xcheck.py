"""Cross-implementation control for diagnostics.bispectrum, against DISCO-DJ.

WHY A SCRIPT AND NOT JUST A TEST. discodj lives in the disco-mocks pixi env, not
inexor's, so the two estimators cannot be imported into one process. The two
legs therefore run in their own envs and meet on disk -- the same reason
scripts/_m1_common.py exists. Field data crosses as .npy rather than as a seed,
so no RNG-portability question enters (JAX and discodj's env need not agree on
a random stream for this to be exact).

WHAT THIS CONTROLS FOR, AND WHAT IT DOES NOT. discodj's estimator is the same
Scoccimarro construction with the same conventions, verified by reading its
source rather than assuming:

  - shell indicator (k >= edge[n]) * (k < edge[n+1]) -- half-open, as ours
  - rfft half-grid (get_fourier_grid(full=False)), as ours
  - data branch  * L^(2d) / N^d ; norm branch * N^(2d); evaluate_bispectrum
    divides them, giving B = L^6 S / (N^9 norm) -- our alpha exactly

It is nonetheless NOT an oracle: it ships no tests of its own, and it labels
bins at arithmetic centres while computing a genuine bin average. The
deterministic tests in tests/test_bispectrum.py (plane-wave closed form,
brute-force triangle count) are what pin our side; this control adds an
independent implementation, so agreement is evidence and disagreement is a
question rather than a verdict. Measure discodj's own floor on the plane-wave
field before reading anything into a comparison on a random one.

BINNING. discodj takes contiguous bin EDGES, so shells are expressed as
edges = k_f * (0.5 + arange(nbins+1)); bin n is then [(n+0.5)k_f, (n+1.5)k_f)
with centre (n+1) k_f, which is this repo's shell convention at dk = k_f. Its
estimator asserts kmax < (2/3) k_Nyq, so nbins <= N/3 - 0.5.

Usage:
    # leg 1, inexor env
    pixi run python scripts/_bispectrum_xcheck.py emit
    # leg 2, disco-mocks env
    cd ~/spherex/disco-mocks && pixi run python \
        ~/spherex/inexor/scripts/_bispectrum_xcheck.py discodj
    # leg 3, inexor env -- prints the comparison
    pixi run python scripts/_bispectrum_xcheck.py compare
"""

import argparse
import json
import os

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "v2", "xcheck")
FIELD_NPY = os.path.join(OUT_DIR, "xcheck_field.npy")
SPEC_JSON = os.path.join(OUT_DIR, "xcheck_spec.json")
DISCO_JSON = os.path.join(OUT_DIR, "xcheck_discodj.json")

N_MESH = 32
L_BOX = 256.0
NBINS = 10  # <= N/3 - 0.5 = 10.17, discodj's 2/3-Nyquist assert
# (i, j, l) bin indices; centre (n+1) k_f. (7,7,1) is the squeezed 8-8-2 config
# the gate cares about; (4,3,2) is a generic closing triangle as a second point.
TRI_BINS = [(7, 7, 1), (4, 3, 2)]


def _edges(box_size, nbins):
    kf = 2.0 * np.pi / box_size
    return kf * (0.5 + np.arange(nbins + 1, dtype=np.float64))


def _plane_wave(n, m1, m2, m3, a):
    ax = (2.0 * np.pi / n) * np.arange(n)

    def cos(m):
        return np.cos(m[0] * ax[:, None, None] + m[1] * ax[None, :, None] + m[2] * ax[None, None, :])

    return a * (cos(m1) + cos(m2) + cos(m3))


def cmd_emit(_args):
    """Leg 1: write the field + spec, and this side's measurement."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from inexor.diagnostics import bispectrum

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(7)
    field = rng.normal(size=(N_MESH, N_MESH, N_MESH)).astype(np.float64)
    # A plane-wave field rides along in the same file set so the OTHER side's
    # own floor is measurable, per reference-oracle-parity-floor: a comparison
    # against a reference whose accuracy is unknown is a bound, not a check.
    pw = _plane_wave(N_MESH, (3, 0, 0), (0, 4, 0), (-3, -4, 0), 0.5)

    np.save(FIELD_NPY, np.stack([field, pw]))
    edges = _edges(L_BOX, NBINS)
    centers = 0.5 * (edges[1:] + edges[:-1])
    tris = [tuple(float(centers[i]) for i in t) for t in TRI_BINS]

    ours = {}
    for name, f in (("gaussian", field), ("plane_wave", pw)):
        b, n_tri = bispectrum(f, L_BOX, tris)
        ours[name] = dict(B=[float(v) for v in b], n_tri=[float(v) for v in n_tri])

    spec = dict(
        n_mesh=N_MESH, box_size=L_BOX, nbins=NBINS,
        edges=[float(e) for e in edges], tri_bins=[list(t) for t in TRI_BINS],
        tri_centers=[list(t) for t in tris], inexor=ours,
    )
    with open(SPEC_JSON, "w") as fh:
        json.dump(spec, fh, indent=2)
    print(f"wrote {FIELD_NPY}\nwrote {SPEC_JSON}")
    for name in ours:
        print(f"  inexor {name:11s} B = {ours[name]['B']}  n_tri = {ours[name]['n_tri']}")


def cmd_discodj(_args):
    """Leg 2: measure the same field with DISCO-DJ. Run in the disco-mocks env."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from discodj.core.summary_statistics import bispectrum as dj_bispectrum

    spec = json.load(open(SPEC_JSON))
    fields = np.load(FIELD_NPY)
    edges = tuple(spec["edges"])
    ell = spec["box_size"]

    # The normalization pass is field-independent; discodj recomputes it per
    # call inside evaluate_bispectrum, so it is done once here instead.
    # compute_norm replaces the field with 1. but reads res/shape off the
    # argument FIRST, so it still needs a correctly shaped array.
    n = spec["n_mesh"]
    norm = dj_bispectrum(jnp.zeros((n, n, n), jnp.float64), ell, bins=edges, compute_norm=True)
    out = {}
    for name, arr in (("gaussian", fields[0]), ("plane_wave", fields[1])):
        res = dj_bispectrum(jnp.asarray(arr, jnp.float64), ell, bins=edges, only_B=False)
        bk = np.asarray(res["Bk"]) / np.asarray(norm["Bk"])
        # discodj enumerates i>=j>=l; recover the row index of each wanted triple
        idx = []
        rows = []
        for i in range(spec["nbins"]):
            for j in range(i + 1):
                for ll in range(j + 1):
                    a, wa = (edges[i + 1] + edges[i]) / 2, (edges[i + 1] - edges[i]) / 2
                    b, wb = (edges[j + 1] + edges[j]) / 2, (edges[j + 1] - edges[j]) / 2
                    c, wc = (edges[ll + 1] + edges[ll]) / 2, (edges[ll + 1] - edges[ll]) / 2
                    if a - wa < (b + wb) + (c + wc):
                        rows.append((i, j, ll))
        for t in spec["tri_bins"]:
            idx.append(rows.index(tuple(sorted(t, reverse=True))))
        out[name] = dict(B=[float(bk[m]) for m in idx],
                         n_tri=[float(np.asarray(norm["Bk"])[m]) for m in idx])
    out["_row_index"] = idx
    with open(DISCO_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {DISCO_JSON}")
    for name in ("gaussian", "plane_wave"):
        print(f"  discodj {name:11s} B = {out[name]['B']}")


def cmd_compare(_args):
    """Leg 3: the comparison, with the reference's own floor stated first."""
    spec = json.load(open(SPEC_JSON))
    dj = json.load(open(DISCO_JSON))
    ell, n = spec["box_size"], spec["n_mesh"]

    print("=== discodj's OWN floor, against the plane-wave closed form ===")
    print("(a comparison against a reference of unknown accuracy is a bound, not a")
    print(" check, so the reference's own error is measured before anything else)")
    # bins (4,3,2) -> centres (5,4,3) k_f, i.e. the closed 3-4-5 triangle, for
    # which B = L^6 a^3 / (4 n_tri) exactly (a = 0.5, see cmd_emit).
    for i, t in enumerate(spec["tri_bins"]):
        if tuple(t) != (4, 3, 2):
            continue
        n_tri = spec["inexor"]["plane_wave"]["n_tri"][i]
        exact = ell**6 * 0.5**3 / (4.0 * n_tri)
        for who, val in (("inexor", spec["inexor"]["plane_wave"]["B"][i]),
                         ("discodj", dj["plane_wave"]["B"][i])):
            print(f"  tri {t} vs closed form {exact: .8e}: {who:8s} rel {abs(val / exact - 1):.3e}")

    print("\n=== B, both fields, both triangles ===")
    worst = 0.0
    for name in ("plane_wave", "gaussian"):
        for i, t in enumerate(spec["tri_bins"]):
            a = spec["inexor"][name]["B"][i]
            b = dj[name]["B"][i]
            rel = abs(b / a - 1.0) if a != 0 else float("nan")
            worst = max(worst, rel)
            print(f"  {name:11s} tri {t}: inexor {a: .8e}  discodj {b: .8e}  rel {rel:.3e}")
    print(f"\nworst relative disagreement: {worst:.3e}   (N={n}, L={ell})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("emit", cmd_emit), ("discodj", cmd_discodj), ("compare", cmd_compare)):
        sub.add_parser(name).set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
