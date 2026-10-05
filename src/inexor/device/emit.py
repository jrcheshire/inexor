"""T9 emission on the cards: the slab files `icgen._emit_t9_slabs` writes, computed on devices.

Card k owns a contiguous run of destination slabs and, one thread per card:

  1. per SOURCE slab (in plane chunks), one program turns displacements into position
     codes, the brick-major bucket key, the destination slab and the ring-window
     check, and keeps (key, off, v) on the card, with per-destination counts synced;
  2. per DESTINATION slab, one program gathers the rows bound for it from its window of
     sources in ascending source index (the host order, seam included), stable-sorts
     them by slab-relative key, and computes occupancy, per-brick scales and velocity
     codes;
  3. the host writes the slab with `icgen._write_t9_slab`.

Bitwise equal to the host emission (gated on the CPU backend): keys are
integers; the position quantum is an exact power of two, so `x / quantum` equals
`x * (1 / quantum)` (refused otherwise); every velocity division is by a full-shape
array computed in the program (CPU XLA turns a scalar or broadcast divisor into a
reciprocal multiply); rows are gathered in the host's concatenation order and sorted
stably; maxima are exact.

INPUTS. u_x lives on the cards as shards with `window` brick slabs of halo
(`card_slab_ranges`, filled by `ooc_fft.inverse_to_card_shards`); u_y, u_z are host
arrays and v is three readers (`read_slab`), each uploaded per plane chunk.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np

from .. import ooc_fft

#: planes per source-program call; bits do not depend on it (elementwise, concatenated
#: in order), card memory does
PLANES_PER_CALL = 4


def _padded(n):
    from ..forces import capacity_shape

    return int(capacity_shape(max(1, int(n)) + 1, rungs=12))


def card_slab_ranges(n, nb, devices, window, slabs=None):
    """Per card: destination slabs [lo, hi) and the u_x plane range it must hold.

    Returns `[{lo, hi, x0, nx, device}]`; planes `x0 .. x0 + nx - 1` (mod n) cover the
    card's slabs plus `window` brick slabs either side, the sources its slabs draw on.
    `slabs` = [lo, hi) splits one rank's destination slabs (default all of them).
    """
    n, nb, window = int(n), int(nb), int(window)
    s0, s1 = (0, nb) if slabs is None else (int(slabs[0]), int(slabs[1]))
    p = n // nb
    h = window * p
    return [dict(lo=s0 + a, hi=s0 + b, x0=(s0 + a) * p - h, nx=(b - a) * p + 2 * h, device=dev)
            for (a, b), dev in zip(ooc_fft.partition_units(s1 - s0, len(devices), 1), devices)]


def shards_from_host(field, ranges):
    """Upload a host (n, n, n) field as `card_slab_ranges` shards (for tests)."""
    import jax

    n = field.shape[0]
    out = []
    for r in ranges:
        idx = (int(r["x0"]) + np.arange(int(r["nx"]))) % n
        out.append(dict(r, delta=jax.device_put(np.ascontiguousarray(field[idx]), r["device"])))
    return out


def _source_program(nb, per, per3, window, n_levels, inv_q):
    import jax
    import jax.numpy as jnp

    nbp = nb * nb * per3

    def fn(ux, uy, uz, vx, vy, vz, qx, coords, box, s):
        qs = (qx[:, None, None], coords[None, :, None], coords[None, None, :])
        idx = []
        for q, u in zip(qs, (ux, uy, uz)):
            x = jnp.mod(q + u, box).reshape(-1)
            idx.append(jnp.mod(jnp.rint(x.astype(jnp.float64) * inv_q).astype(jnp.int64),
                               n_levels))
        i3 = jnp.stack(idx, axis=1)
        b = i3 // 256
        off = (i3 - b * 256).astype(jnp.uint8)
        brick = b // per
        within = b - brick * per
        bf = (brick[:, 0] * nb + brick[:, 1]) * nb + brick[:, 2]
        wf = (within[:, 0] * per + within[:, 1]) * per + within[:, 2]
        key = bf * per3 + wf
        dest = key // nbp
        ring = jnp.mod(dest - s, nb)
        bad = jnp.sum(jnp.logical_not((ring <= window) | (ring >= nb - window)))
        counts = jnp.zeros(nb, dtype=jnp.int64).at[dest].add(1)
        v = jnp.stack([vx.reshape(-1), vy.reshape(-1), vz.reshape(-1)], axis=1)
        return key, off, v, bad, counts

    return jax.jit(fn)


def _dest_program(cap, nb2, per3, nbp):
    import jax
    import jax.numpy as jnp

    big = int(np.iinfo(np.int64).max)

    def fn(keys, offs, vs, d, lo_bucket, n_real, div):
        key = jnp.concatenate(keys)
        off = jnp.concatenate(offs)
        v = jnp.concatenate(vs)
        sel = jnp.nonzero((key // nbp) == d, size=cap, fill_value=0)[0]
        real = jnp.arange(cap) < n_real
        rel = jnp.where(real, key[sel] - lo_bucket, big)
        order = jnp.argsort(rel, stable=True)
        rows = sel[order]
        real_s = real[order]
        key_s = jnp.where(real_s, rel[order], 0)
        off_s = off[rows]
        v_s = v[rows].astype(jnp.float64)
        occ = jnp.zeros(nb2 * per3, dtype=jnp.int64).at[key_s].add(real_s.astype(jnp.int64))
        brick = key_s // per3
        absv = jnp.max(jnp.abs(v_s), axis=1)
        vmax = jnp.zeros(nb2, dtype=jnp.float64).at[brick].max(jnp.where(real_s, absv, 0.0))
        s_b = vmax / div
        s_b = jnp.where(s_b > 0.0, s_b, 1.0)
        srow = jnp.where(real_s, s_b[brick], 1.0)
        wf = jnp.stack([jnp.rint(v_s[:, a] / srow) for a in range(3)], axis=1)
        abs_max = jnp.max(jnp.where(real_s[:, None], jnp.abs(wf), 0.0))
        return off_s, wf.astype(jnp.int16), occ, s_b, abs_max

    return jax.jit(fn)


def emit_t9_slabs_cards(workdir, ux_shards, uy, uz, v_ro, t9, n, box, nb, dt, window,
                        planes_per_call=PLANES_PER_CALL, timings=None, complete=True):
    """Emit every T9 slab file from card-side work; `icgen._emit_t9_slabs`' contract.

    `ux_shards` are `card_slab_ranges` dicts with `delta` (the card's u_x planes);
    `uy`, `uz` host (n, n, n) arrays (sliced by global plane); `v_ro` three readers.
    Returns `(written, n_total)` with `written` in slab order. `timings`, if a dict,
    receives summed per-card seconds: `upload_s`, `source_s`, `dest_s`, `write_s`.
    `complete=False` emits only the shards' own [lo, hi) slabs and skips the
    whole-box checks (the production-shape memory leg).
    """
    import jax
    import jax.numpy as jnp

    from ..eject_jax import require_x64
    from ..icgen import INT16_MAX, _write_t9_slab

    require_x64()
    n, nb, window = int(n), int(nb), int(window)
    dt = np.dtype(dt)
    if math.frexp(float(t9.quantum))[0] != 0.5:
        raise ValueError(
            f"position quantum {t9.quantum!r} is not a power of two; the card emission "
            "multiplies by its reciprocal and would round differently from the host. Use "
            "emission='host' for this geometry.")
    per = t9.n_buckets_side // nb
    per3 = per**3
    nb2, nbp = nb * nb, nb * nb * per3
    p = n // nb
    c = max(1, min(int(planes_per_call), p))
    if p % c:
        c = p
    coords = np.arange(n, dtype=dt) * dt.type(box / n)  # the host emission's exact values
    inv_q = 1.0 / float(t9.quantum)
    src_prog = ooc_fft._card_program(
        ("emit_src", nb, per, window, int(t9.n_levels), inv_q), lambda: _source_program(
            nb, per, per3, window, int(t9.n_levels), inv_q))
    lock = threading.Lock()
    written, totals = {}, {"n": 0}
    clock = dict(upload_s=0.0, source_s=0.0, dest_s=0.0, write_s=0.0)

    def _add(name, t0):
        with lock:
            clock[name] += time.perf_counter() - t0

    def card(k, _k1, dev):
        r = ux_shards[k]

        def put(a):
            # ascontiguousarray would promote a 0-d scalar to shape (1,)
            a = np.asarray(a)
            return jax.device_put(np.ascontiguousarray(a) if a.ndim else a, dev)

        coords_d, box_d = put(coords), put(np.asarray(box, dtype=dt))
        div = put(np.full(nb2, float(INT16_MAX), dtype=np.float64))
        cache = {}

        def source(s):
            keys, offs, vs = [], [], []
            counts = np.zeros(nb, dtype=np.int64)
            bad = 0
            for g0 in range(s * p, (s + 1) * p, c):
                g1 = g0 + c
                t0 = time.perf_counter()
                i0 = (g0 - int(r["x0"])) % n
                ux = jax.lax.dynamic_slice_in_dim(r["delta"], put(np.int64(i0)), c, axis=0)
                args = (ux, put(uy[g0:g1]), put(uz[g0:g1]),
                        *(put(v_ro[a].read_slab(g0, g1)) for a in range(3)),
                        put(coords[g0:g1]), coords_d, box_d, put(np.int64(s)))
                _add("upload_s", t0)
                t0 = time.perf_counter()
                key, off, v, nbad, cnt = src_prog(*args)
                counts += np.asarray(cnt)
                bad += int(nbad)
                keys.append(key)
                offs.append(off)
                vs.append(v)
                _add("source_s", t0)
            if bad:
                raise RuntimeError(
                    f"{bad} particles from source slab {s} routed outside the +-{window} "
                    "window (the displacement bound above should have caught this)")
            if len(keys) > 1:
                keys, offs, vs = ([jnp.concatenate(keys)], [jnp.concatenate(offs)],
                                  [jnp.concatenate(vs)])
            return keys[0], offs[0], vs[0], counts

        def sources_of(d):
            return sorted({(d + o) % nb for o in range(-window, window + 1)})

        lo, hi = int(r["lo"]), int(r["hi"])
        for d in range(lo, hi):
            srcs = sources_of(d)
            for s in srcs:
                if s not in cache:
                    cache[s] = source(s)
            n_real = int(sum(cache[s][3][d] for s in srcs))
            cap = _padded(n_real)
            prog = ooc_fft._card_program(("emit_dest", cap, nb2, per3, len(srcs)),
                                         lambda cap=cap: _dest_program(cap, nb2, per3, nbp))
            t0 = time.perf_counter()
            off_s, w, occ, s_b, abs_max = prog(
                tuple(cache[s][0] for s in srcs), tuple(cache[s][1] for s in srcs),
                tuple(cache[s][2] for s in srcs), put(np.int64(d)),
                put(np.int64(d * nbp)), put(np.int64(n_real)), div)
            off_h = np.asarray(off_s)[:n_real]
            w_h = np.asarray(w)[:n_real]
            occ_h, s_h, amax = np.asarray(occ), np.asarray(s_b), float(abs_max)
            _add("dest_s", t0)
            if amax > INT16_MAX:
                raise ValueError(f"velocity code {amax:.0f} escapes int16 in slab {d}; "
                                 "integer state is never clamped")
            t0 = time.perf_counter()
            name = _write_t9_slab(workdir, d, occ_h, off_h, w_h, s_h, d * nbp, d * nb2)
            _add("write_s", t0)
            with lock:
                written[d] = name
                totals["n"] += n_real
            needed = {s for dd in range(d + 1, hi) for s in sources_of(dd)}
            for s in [s for s in cache if s not in needed]:
                del cache[s]

    devs = [r["device"] for r in ux_shards]
    ooc_fft._run_parts(card, [(k, k + 1) for k in range(len(ux_shards))], devs)
    if complete and len(written) != nb:
        raise RuntimeError(f"wrote {len(written)} of {nb} slabs")
    if complete and totals["n"] != n**3:
        raise RuntimeError(f"emitted {totals['n']} particles, expected {n**3}")
    if timings is not None:
        timings.update(clock)
    return [written[d] for d in sorted(written)], totals["n"]
