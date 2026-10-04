"""The particle export's decode on the cards: `(slots, x, v)` per chunk, in brick order.

A chunk is a run of consecutive bricks of the state's own range (the coarse paint's chunk,
`paint.default_chunk_bricks`); its rows are staged at the fixed shapes of `paint.step_shapes`
(`paint.slab_window_fixed` with the velocity codes) and one compiled program per run decodes
positions (`decode.decode_core`), velocities `w * scale` (`* vfac` for km/s) and casts both to
the output dtype, the operations and order of the host export. Bitwise the host path
(`export._host_chunks`); gated on it.

Chunks go round-robin to the cards, each card on its own thread, with a bounded number in
flight; results come back in chunk order.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .paint import _KERNEL_LOCK, _on, chunk_rows, default_chunk_bricks, slab_window_fixed, \
    step_shapes

_EXPORT_KERNELS = {}


def _export_kernel(*, cap, p3, per, nb, arena_base, t9, dtype, kms):
    key = (cap, p3, per, nb, arena_base, float(t9.quantum), int(t9.n_buckets_side),
           np.dtype(dtype).str, bool(kms))
    with _KERNEL_LOCK:
        fn = _EXPORT_KERNELS.get(key)
    if fn is not None:
        return fn
    import jax
    import jax.numpy as jnp

    from .decode import decode_core

    out = jnp.dtype(dtype)

    def body(starts, occ, live_counts, arena_slots, row_offsets, bricks, off, w, arena_bucket,
             n_rows, scales, vfac):
        slots, x, _bor, bi, _live = decode_core(
            starts, occ, live_counts, arena_slots, row_offsets, bricks, off, arena_bucket,
            arena_base, n_rows, cap=cap, lift=cap + 1, p3=p3, per=per, bricks_per_side=nb,
            t9=t9, fdtype=jnp.float64)
        # the host's `w.astype(f64) * scale`, then `* vfac`, then the cast, in that order
        v = w[slots].astype(jnp.float64) * scales[bi][:, None]
        if kms:
            v = v * vfac
        return slots, x.astype(out), v.astype(out)

    with _KERNEL_LOCK:
        return _EXPORT_KERNELS.setdefault(key, jax.jit(body))


def export_chunks(st, devices=None, chunk_bricks=None, dtype=np.float32, vfac=None,
                  with_slots=False, depth=2, timings=None):
    """Yield `(slots, x, v)` per non-empty chunk of the state's own bricks, in brick order.

    `x`, `v` are host arrays of `dtype`, `v` multiplied by `vfac` when given (else the
    native D-time velocity); `slots` (state slots, for the export's ids) is None unless
    `with_slots`.
    `devices` is a sequence of jax devices (None: jax's default device); at most
    `depth` chunks per card are in flight. `timings`, if a dict, accumulates `card s`
    (decode, cast and copy back, summed over the cards' threads).
    """
    from ..eject_jax import require_x64
    from ..forces import capacity_shape

    require_x64()
    devs = [None] if devices is None else list(devices)
    nb = int(st.bricks_per_side)
    L = default_chunk_bricks(nb) if chunk_bricks is None else int(chunk_bricks)
    lo, hi = st.owned_bricks
    if L < 1 or int(st.n_bricks) % L or lo % L or hi % L:
        raise ValueError(f"chunk_bricks {L} does not tile the {int(st.n_bricks)} bricks and "
                         f"this state's range [{lo}, {hi}) into whole chunks")
    rows = chunk_rows(st, L)
    todo = [g for g in range(lo // L, hi // L) if rows[g]]
    if not todo:
        return
    cap = int(capacity_shape(int(rows[todo].max())))
    shapes = step_shapes(st, L, cap)
    fn = _export_kernel(cap=cap, p3=int(st.buckets_per_brick),
                        per=int(st.t9.n_buckets_side // nb), nb=nb,
                        arena_base=int(shapes["live_w"]), t9=st.t9, dtype=dtype,
                        kms=vfac is not None)
    vf = 1.0 if vfac is None else float(vfac)
    vs = np.asarray(st.vel_scale, dtype=np.float64)

    import time

    def run(g, dev):
        t0 = time.perf_counter()
        bricks = np.arange(g * L, (g + 1) * L, dtype=np.int64)
        w = slab_window_fixed(st, bricks, shapes, with_w=True)
        n = int(w["n_rows"])
        slots, x, v = fn(
            _on(w["starts"], dev), _on(w["occ"], dev), _on(w["live_counts"], dev),
            _on(w["arena_slots"], dev), _on(w["row_offsets"], dev), _on(w["bricks"], dev),
            _on(w["off"], dev), _on(w["w"], dev), _on(w["arena_bucket"], dev),
            _on(n, dev, np.int64), _on(vs[g * L:(g + 1) * L], dev),
            _on(vf, dev, np.float64))
        state_slots = None
        if with_slots:
            r = np.asarray(slots[:n])
            W = int(shapes["live_w"])
            state_slots = np.where(r < W, r + int(w["slot0"]),
                                   w["arena_rows"][np.clip(r - W, 0, None)]
                                   if len(w["arena_rows"]) else 0)
        out = (state_slots, np.asarray(x[:n]), np.asarray(v[:n]))
        if timings is not None:
            timings["card s"] = timings.get("card s", 0.0) + time.perf_counter() - t0
        return out

    pools = [ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"export-card{k}")
             for k in range(len(devs))]
    pending = []
    try:
        nxt = 0
        while nxt < len(todo) or pending:
            while nxt < len(todo) and len(pending) < depth * len(devs):
                k = nxt % len(devs)
                pending.append(pools[k].submit(run, todo[nxt], devs[k]))
                nxt += 1
            yield pending.pop(0).result()
    finally:
        for f in pending:
            f.cancel()
        for p in pools:
            p.shutdown(wait=True)
