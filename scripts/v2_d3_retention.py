"""What does `drift_and_migrate_device` hold while its kernels run?

Vista 995602 read the device migrate's peak at 438 B per slab row at cgh64,
against ~114 (eject) and ~80 (insert) B per padded row for the kernels alone, so
most of the peak is arrays the driver keeps alive at once. This names them: just
before every eject, insert and write-back program call, it sums the bytes of
every live `jax.Array` the Python side references (a `gc` scan), grouped by
shape and dtype, and keeps the largest sample. XLA's own workspace inside a
kernel is NOT seen here -- R0 measured that per kernel -- so held + workspace is
the peak's decomposition.

Runs on any backend (the arrays are the same objects on CPU XLA). Reported per
slab row, the unit sec. 30 used.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))


def _probe():
    path = os.path.join(REPO, "scripts", "v2_d3_device_migrate.py")
    spec = importlib.util.spec_from_file_location("d3probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _held():
    import jax

    total, groups = 0, Counter()
    seen = set()
    for o in gc.get_objects():
        if isinstance(o, jax.Array) and id(o) not in seen:
            seen.add(id(o))
            try:
                nb = int(o.nbytes)
            except Exception:
                continue
            total += nb
            groups[(tuple(o.shape), str(o.dtype))] += nb
    return total, groups


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-part", type=int, default=256)
    ap.add_argument("--kind", default="real", choices=("real", "xback"))
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args(argv)

    import jax

    jax.config.update("jax_enable_x64", True)
    from inexor.device import migrate

    P = _probe()
    n = args.n_part
    nb = P._nb_of(n)
    heavy = args.kind == "xback"
    st = P._build_state(n, nb, seed=11 if heavy else 13, brick_slack=0.0 if heavy else 0.10,
                        arena_frac=0.30 if heavy else 0.01, with_ids=heavy)
    c = P._c_drift(st, 1.9 if heavy else 0.5)
    slab_rows = n**3 // nb

    worst = {}  # site -> (bytes, groups)

    def sample(site):
        gc.collect()
        b, g = _held()
        if b > worst.get(site, (0, None))[0]:
            worst[site] = (b, g)

    def wrap_factory(name, factory):
        def wrapped(*a, **k):
            fn = factory(*a, **k)

            def call(*ca, **ck):
                sample(name)
                return fn(*ca, **ck)

            return call

        return wrapped

    migrate._eject_kernel = wrap_factory("eject", migrate._eject_kernel)
    migrate._insert_kernel = wrap_factory("insert", migrate._insert_kernel)
    migrate._write_program = wrap_factory("write", migrate._write_program)

    walls = []
    for _ in range(args.steps):
        t = time.perf_counter()
        migrate.drift_and_migrate_device(st, c)
        walls.append(time.perf_counter() - t)

    card = dict(argv=sys.argv, n_part=n, nb=nb, kind=args.kind, slab_rows=slab_rows,
                walls_s=walls, sites={})
    for site, (b, g) in sorted(worst.items()):
        top = [dict(shape=list(s), dtype=d, bytes=v, b_per_slab_row=v / slab_rows)
               for (s, d), v in g.most_common(args.top)]
        card["sites"][site] = dict(bytes=b, b_per_slab_row=b / slab_rows, top=top)
        print(f"\n[{site}] held {b / 2**20:.1f} MiB = {b / slab_rows:.1f} B per slab row "
              f"({slab_rows:,} rows)", flush=True)
        for t in top:
            print(f"  {t['b_per_slab_row']:7.1f} B/slab row  {t['bytes'] / 2**20:8.1f} MiB  "
                  f"{t['dtype']:8s} {tuple(t['shape'])}", flush=True)
    out = os.path.join(REPO, "runs", "v2", f"d3_retention{args.out_suffix}.json")
    with open(out, "w") as f:
        json.dump(card, f, indent=1)
    print(f"\ncard: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
