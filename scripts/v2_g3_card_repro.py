"""Compare two Stage 5 cards on the quantities that must NOT have moved.

WHY. Adding the auto transfer changed shell_transfer's signature and added two
keys to stats(), but touched none of the existing computations. That is a
REGRESSION CLAIM, and a regression claim asserted in a commit message is worth
nothing -- this checks it against the two cards.

WHAT COUNTS AS AGREEMENT. Same seed, same config, same host, so the arithmetic
is identical and the only source of disagreement is XLA-CPU's own run-to-run
reproducibility. The default bound is therefore tight (1e-12 relative) rather
than a physics tolerance: this is a bitwise-ish check with room for the
reduction-order nondeterminism the project has already characterised, NOT a
statement about how much the science may drift. A failure here means the
refactor moved a number, which is exactly what it must not have done.

The new keys (rho_auto, A_prod, shell_A) are expected to be absent from the
older card and are reported, never compared.

Usage:
    pixi run python scripts/v2_g3_card_repro.py runs/v2/a.json runs/v2/b.json
"""

import argparse
import json
import sys

import numpy as np

# per-triangle quantities that the auto-transfer change must have left alone
CELL_KEYS = ("R_Q", "R_B", "rho", "W", "T_prod", "n_tri")
ROW_KEYS = ("shell_T", "shell_r", "k", "r", "wall")
NEW_KEYS = ("rho_auto", "A_prod")


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return None, f"shape {a.shape} vs {b.shape}"
    scale = np.maximum(np.abs(a), np.abs(b))
    scale[scale == 0.0] = 1.0
    return float(np.nanmax(np.abs(a - b) / scale)), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--tol", type=float, default=1e-12,
                    help="max relative deviation allowed on an unchanged quantity")
    args = ap.parse_args()

    old = json.load(open(args.old))
    new = json.load(open(args.new))

    print(f"OLD {args.old}\nNEW {args.new}\n")
    worst, failures, checked = 0.0, [], 0

    for ro in old["rows"]:
        rn = next((r for r in new["rows"] if r["arm"] == ro["arm"]), None)
        if rn is None:
            failures.append(f"{ro['arm']}: absent from the new card")
            continue
        for key in ROW_KEYS:
            if key not in ro or key not in rn:
                continue
            # wall time is a scalar and is EXPECTED to move; report, never gate
            if key == "wall":
                print(f"  {ro['arm']:16s} wall {ro[key]:7.1f} -> {rn[key]:7.1f} s "
                      f"({rn[key] / ro[key]:.2f}x, not gated)")
                continue
            dev, err = _rel(ro[key], rn[key])
            checked += 1
            if err or dev > args.tol:
                failures.append(f"{ro['arm']}/{key}: {err or f'rel dev {dev:.3e}'}")
            elif dev is not None:
                worst = max(worst, dev)

        for ks, co in ro["cells"].items():
            cn = rn["cells"].get(ks)
            if cn is None:
                failures.append(f"{ro['arm']}/k_short={ks}: cell absent")
                continue
            for key in CELL_KEYS:
                if key not in co or key not in cn:
                    continue
                dev, err = _rel(co[key], cn[key])
                checked += 1
                if err or dev > args.tol:
                    failures.append(
                        f"{ro['arm']}/ks={ks}/{key}: {err or f'rel dev {dev:.3e}'}")
                elif dev is not None:
                    worst = max(worst, dev)

    n_new = sum(1 for r in new["rows"] for c in r["cells"].values()
                for k in NEW_KEYS if k in c)
    print(f"\n  compared {checked} arrays across {len(old['rows'])} arms")
    print(f"  worst relative deviation: {worst:.3e}  (bound {args.tol:.0e})")
    print(f"  new keys present in the new card: {n_new} cell entries "
          f"({', '.join(NEW_KEYS)}) -- reported, not compared")

    if failures:
        print(f"\n  FAIL ({len(failures)}):")
        for f in failures[:20]:
            print(f"    {f}")
        return 1
    print("\n  PASS -- the auto-transfer change moved no existing quantity.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
