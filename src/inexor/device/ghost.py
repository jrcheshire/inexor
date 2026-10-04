"""Ghost slabs: the neighbour ranks' brick x-slabs that one rank's tile windows read.

A tile plane's window reaches `pad` brick slabs past its core on each side
(`layout.brick_span`), so a rank's first and last tile planes read slabs its neighbours own.
`exchange_ghosts` copies those slabs from their owners once per step, and `SlabView` serves
the state's owned slabs and the ghost slabs through the reads the windowed tile loop makes
(`device.window`, `device.decode.tile_decode_plan`).

A ghost slab is a verbatim copy of the owner's slot range (`off` only), its bricks'
occupancy and starts relative to the slab, and its arena residents (`off`, bucket) in the
owner's per-brick ascending slot order, so its spans, and with them every window shape, are
the owner's. No `w`: buffer rows' velocities are masked out of the kick (`device/kick.py`),
so they stage as zeros. Ghost rows get ids past the state's rows (live runs slab by slab,
then residents), so every lookup by row id stays a sorted search. The sender sends views of
its state and the receiver keeps the arrays that arrive: two ghost slabs per side of host.
"""

from __future__ import annotations

import numpy as np

from ..state import tile_window_counts


def pack_slabs(st, slabs):
    """The arrays `GhostSlabs` rebuilds brick x-slabs `slabs` (owned by `st`) from, one set
    per slab under `"<s>:<name>"` keys. The slot range and occupancy are views of the state,
    so the sender holds no copy of them."""
    from .migrate import pass_arena_index

    nb, p3 = int(st.bricks_per_side), int(st.buckets_per_brick)
    nb2 = nb * nb
    ar_slots, ar_bricks = pass_arena_index(st)
    out = dict(slabs=np.asarray([int(x) for x in slabs], dtype=np.int64),
               p3=np.asarray([p3], dtype=np.int64))
    for s in (int(x) for x in slabs):
        lo_b, hi_b = st.slab_bricks(s)
        s0, s1 = int(st.brick_start[lo_b]), int(st.brick_start[hi_b])
        a0, a1 = np.searchsorted(ar_bricks, [lo_b, hi_b])
        rows = ar_slots[a0:a1]
        out.update({
            f"{s}:off": st.off[s0:s1],
            f"{s}:starts": np.asarray(st.brick_start[lo_b:hi_b], dtype=np.int64) - s0,
            f"{s}:occ": st._occ(lo_b, hi_b),
            f"{s}:res_off": st.off[rows],
            f"{s}:res_bucket": np.asarray(st.arena_bucket, dtype=np.int64)[
                rows - int(st.arena_base)],
            f"{s}:res_count": np.bincount(ar_bricks[a0:a1] - lo_b,
                                          minlength=nb2).astype(np.int64),
        })
    return out


class GhostSlabs:
    """Brick x-slabs another rank owns, rebuilt from `pack_slabs` parcels, ids from `base`.

    Read through `SlabView`. Each slab keeps the arrays it arrived in (no copy); its slot
    range takes ids in parcel order from `base`, and every slab's residents follow.
    """

    def __init__(self, parcels, base, bricks_per_side):
        self.base = int(base)
        self.nb = int(bricks_per_side)
        nb2 = self.nb ** 2
        self._slab = {}
        at = self.base
        for p in parcels:
            p3 = int(p["p3"][0])
            for s in p["slabs"].tolist():
                if s in self._slab:
                    raise ValueError(f"ghost slab {s} arrived twice")
                off = p[f"{s}:off"]
                cnt = p[f"{s}:res_count"]
                self._slab[s] = dict(
                    lo=at, span=int(off.shape[0]), off=off, starts=at + p[f"{s}:starts"],
                    occ=p[f"{s}:occ"].reshape(nb2, p3), res_count=cnt,
                    res_off=p[f"{s}:res_off"], res_bucket=p[f"{s}:res_bucket"],
                    res_first=np.concatenate(([0], np.cumsum(cnt))))
                at += int(off.shape[0])
        for s in sorted(self._slab):
            d = self._slab[s]
            d["res_lo"] = at
            at += int(d["res_count"].sum())
        self.n_rows = at - self.base
        self.nbytes = sum(int(v.nbytes) for d in self._slab.values()
                          for v in (d["off"], d["occ"], d["res_off"]))

    def slabs(self):
        return sorted(self._slab)

    def has(self, s):
        return int(s) in self._slab

    def _brick(self, b):
        s, i = divmod(int(b), self.nb ** 2)
        d = self._slab.get(s)
        if d is None:
            raise IndexError(f"brick {int(b)} (slab {s}) is neither owned nor a ghost")
        return d, i

    def resident_off(self, ids):
        """`off` rows of resident ids (each slab's residents are one id range)."""
        ids = np.asarray(ids, dtype=np.int64)
        out = np.empty((len(ids), 3), dtype=np.uint8)
        for d in self._slab.values():
            n = int(d["res_count"].sum())
            m = (ids >= d["res_lo"]) & (ids < d["res_lo"] + n)
            if m.any():
                out[m] = d["res_off"][ids[m] - d["res_lo"]]
        return out


class SlabView:
    """The reads the windowed tile loop makes, over a state's owned slabs plus `ghosts`.

    Owned slabs read the state; other slabs read the ghost copies (none on one rank, where
    every slab is owned). Row ids below `len(state.off)` are the state's slots; ghost rows
    sit above them.
    """

    def __init__(self, st, ghosts=None):
        self.state = st
        self.ghosts = ghosts
        self.t9 = st.t9
        self.bricks_per_side = int(st.bricks_per_side)
        self.buckets_per_brick = int(st.buckets_per_brick)
        self.n_bricks = int(st.n_bricks)
        self.ids = st.ids
        self._lo, self._hi = st.owned_slabs

    def owns(self, s):
        return self._lo <= int(s) < self._hi

    def _ghost(self, what):
        if self.ghosts is None:
            raise IndexError(f"{what} is outside the owned slabs and there are no ghosts")
        return self.ghosts

    # ---- per brick range of one slab (a run: `(s, lo_b, hi_b)`, global brick ordinals)
    def range_rows(self, s, lo_b, hi_b):
        """Row ids [lo, hi) of bricks [lo_b, hi_b) of slab `s`."""
        s = int(s)
        if self.owns(s):
            st = self.state
            return int(st.brick_start[int(lo_b)]), int(st.brick_start[int(hi_b)])
        d = self._ghost(f"slab {s}")._slab.get(s)
        if d is None:
            raise IndexError(f"slab {s} is neither owned nor a ghost")
        nb2 = self.bricks_per_side ** 2
        i, j = int(lo_b) - s * nb2, int(hi_b) - s * nb2
        end = d["lo"] + d["span"]
        return (int(d["starts"][i]) if i < nb2 else end), (int(d["starts"][j]) if j < nb2 else end)

    def range_chunk(self, name, s, lo_b, hi_b, L):
        """`L` rows of `off` or `w` from the first slot of bricks [lo_b, hi_b) of slab `s`: a
        view where the backing array holds them, else the range's rows zero-padded. Ghost
        `w` is zeros."""
        from . import window

        lo, hi = self.range_rows(s, lo_b, hi_b)
        if self.owns(s):
            src, at = getattr(self.state, name), lo
        elif name == "w":
            return np.zeros((L, 3), dtype=self.state.w.dtype)
        else:
            d = self.ghosts._slab[int(s)]
            src, at = d["off"], lo - d["lo"]
        if window._slab_fits(src.shape[0], at, L):
            return src[at:at + L]
        out = np.zeros((L,) + src.shape[1:], dtype=src.dtype)
        out[:hi - lo] = src[at:at + hi - lo]
        return out

    def range_fits(self, s, lo_b, hi_b, L):
        """True when `range_chunk` returns a view of the owned state (no copy)."""
        from . import window

        return self.owns(s) and window._slab_fits(int(self.state.off.shape[0]),
                                                  self.range_rows(s, lo_b, hi_b)[0], L)

    def residents(self, slabs, index=None):
        """Arena residents of the bricks in `slabs`: (ids ascending, bricks, buckets)."""
        nb2 = self.bricks_per_side ** 2
        return self.residents_of_runs([(s, int(s) * nb2, (int(s) + 1) * nb2) for s in slabs],
                                      index=index)

    def residents_of_runs(self, runs, index=None):
        """Arena residents of the bricks in `runs` (`(s, lo_b, hi_b)`, disjoint): (ids
        ascending, bricks, buckets). `index` is the state's `migrate.pass_arena_index`
        (computed here if None); pass it when calling per window, as the arena is fixed
        for the tile loop."""
        from .migrate import pass_arena_index

        st = self.state
        nb2 = self.bricks_per_side ** 2
        runs = [(int(s), int(a), int(b)) for s, a, b in runs]
        own = [r for r in runs if self.owns(r[0])]
        ids, bricks, buckets = [], [], []
        if own and st.n_arena:
            slots, by_brick = pass_arena_index(st) if index is None else index
            parts = [np.arange(*np.searchsorted(by_brick, [lo_b, hi_b])) for _s, lo_b, hi_b in own]
            k = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
            k = k[np.argsort(slots[k], kind="stable")]
            ids.append(slots[k])
            bricks.append(by_brick[k])
            buckets.append(np.asarray(st.arena_bucket, dtype=np.int64)[
                slots[k] - int(st.arena_base)])
        for s, lo_b, hi_b in sorted(r for r in runs if not self.owns(r[0])):
            d = self._ghost(f"slab {s}")._slab[s]
            i, j = lo_b - s * nb2, hi_b - s * nb2
            f0, f1 = int(d["res_first"][i]), int(d["res_first"][j])
            ids.append(d["res_lo"] + np.arange(f0, f1, dtype=np.int64))
            bricks.append(lo_b + np.repeat(np.arange(j - i, dtype=np.int64),
                                           d["res_count"][i:j]))
            buckets.append(np.asarray(d["res_bucket"][f0:f1], dtype=np.int64))
        if not ids:
            e = np.zeros(0, dtype=np.int64)
            return e, e, e
        return np.concatenate(ids), np.concatenate(bricks), np.concatenate(buckets)

    def brick_resident_counts(self):
        """Arena residents per brick over owned and ghost slabs (0 elsewhere), int64."""
        st = self.state
        out = np.zeros(self.n_bricks, dtype=np.int64)
        if st.n_arena:
            ab = np.asarray(st.arena_bucket, dtype=np.int64)
            b = ab[ab >= 0] // self.buckets_per_brick
            out += np.bincount(b, minlength=self.n_bricks)
        if self.ghosts is not None:
            nb2 = self.bricks_per_side ** 2
            for s in self.ghosts.slabs():
                out[s * nb2:(s + 1) * nb2] = self.ghosts._slab[s]["res_count"]
        return out

    def resident_rows(self, name, ids, out):
        """Rows of `off` or `w` for resident ids, written into `out` (ghost `w` is zeros).
        Owned ids come first (`residents` order), so each part is one slice."""
        st = self.state
        ids = np.asarray(ids, dtype=np.int64)
        k = int(np.searchsorted(ids, st.off.shape[0]))
        out[:k] = getattr(st, name)[ids[:k]]
        if k < len(ids):
            out[k:len(ids)] = self.ghosts.resident_off(ids[k:]) if name == "off" else 0
        return out

    # ---- per brick
    def brick_occ(self, bricks):
        b = np.asarray(bricks, dtype=np.int64)
        nb2 = self.bricks_per_side ** 2
        own = (b // nb2 >= self._lo) & (b // nb2 < self._hi)
        if own.all():
            return self.state.brick_occ(b)
        out = np.empty((len(b), self.buckets_per_brick), dtype=self.state.index_dtype)
        out[own] = self.state.brick_occ(b[own])
        g = self._ghost(f"brick {int(b[~own][0])}")
        for k in np.flatnonzero(~own):
            d, i = g._brick(b[k])
            out[k] = d["occ"][i]
        return out

    def brick_starts(self, bricks):
        b = np.asarray(bricks, dtype=np.int64)
        nb2 = self.bricks_per_side ** 2
        out = np.asarray(self.state.brick_start, dtype=np.int64)[
            np.clip(b, 0, self.n_bricks)]
        ghost = (b // nb2 < self._lo) | (b // nb2 >= self._hi)
        if ghost.any():
            g = self._ghost(f"brick {int(b[ghost][0])}")
            for k in np.flatnonzero(ghost):
                d, i = g._brick(b[k])
                out[k] = d["starts"][i]
        return out

    def arena_slots_of_brick(self, b):
        b = int(b)
        if self.owns(b // self.bricks_per_side ** 2):
            return self.state.arena_slots_of_brick(b)
        d, i = self._ghost(f"brick {b}")._brick(b)
        first = d["res_lo"] + int(d["res_first"][i])
        return np.arange(first, first + int(d["res_count"][i]), dtype=np.int64)

    def brick_member_counts(self):
        """Members per brick over owned and ghost slabs (0 elsewhere), int64 (n_bricks,)."""
        out = self.state.brick_member_counts()
        if self.ghosts is not None:
            nb2 = self.bricks_per_side ** 2
            for s in self.ghosts.slabs():
                d = self.ghosts._slab[s]
                out[s * nb2:(s + 1) * nb2] = (d["occ"].sum(axis=1, dtype=np.int64)
                                              + d["res_count"])
        return out

    def tile_member_counts(self, n_tile, b_fine, n_brick, n_fine, planes=None):
        nb = self.state._check_brick_grid(n_fine, n_brick)
        return tile_window_counts(self.brick_member_counts().reshape(nb, nb, nb), n_tile,
                                  b_fine, n_brick, planes=planes)


def as_view(st):
    """`st` if it is already a `SlabView`, else a view of `st` with no ghosts."""
    return st if isinstance(st, SlabView) else SlabView(st)


def ghost_slabs_needed(decomp, pad):
    """(left, right): the brick x-slabs past this rank's ends its tile windows read."""
    lo, hi = decomp.slabs
    nb = decomp.tiles_side * decomp.bricks_per_tile
    return ([(lo - pad + k) % nb for k in range(pad)], [(hi + k) % nb for k in range(pad)])


def exchange_ghosts(st, decomp, comm, pad):
    """This rank's `GhostSlabs`: `pad` slabs from each neighbour, or None on one rank.

    Every rank sends its first `pad` slabs left and its last `pad` slabs right
    (`comm.exchange_neighbours`). Returns `(ghosts, receipt)`; the receipt counts the rows
    and bytes received.
    """
    from ..comm import exchange_neighbours

    if comm is None or comm.size == 1:
        return None, dict(ghost_slabs=0, ghost_rows=0, ghost_bytes=0)
    lo, hi = st.owned_slabs
    if hi - lo < pad:
        raise ValueError(f"this rank holds {hi - lo} slab(s), fewer than the {pad} its "
                         "neighbours' tile windows read")
    from_left, from_right = exchange_neighbours(
        comm, pack_slabs(st, range(lo, lo + pad)), pack_slabs(st, range(hi - pad, hi)))
    want_l, want_r = ghost_slabs_needed(decomp, pad)
    for got, want, side in ((from_left, want_l, "left"), (from_right, want_r, "right")):
        if got["slabs"].tolist() != want:
            raise ValueError(f"the {side} neighbour sent slabs {got['slabs'].tolist()}, "
                             f"this rank's windows read {want}")
    ghosts = GhostSlabs([from_left, from_right], st.off.shape[0], st.bricks_per_side)
    nbytes = sum(int(v.nbytes) for d in (from_left, from_right) for v in d.values())
    return ghosts, dict(ghost_slabs=2 * pad, ghost_rows=int(ghosts.n_rows), ghost_bytes=nbytes)
