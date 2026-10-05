"""The IC stage's transforms across ranks, against their single-process twins.

`ooc_fft.forward_planes_to_pencils` and `inverse_pencils_to_planes` exchange planes and
pencils in batches over loopback ranks. Gathered over the ranks, their output must be the
single-process transform's byte for byte (card noise, card-shard and host-plane forwards;
inverse to card shards with wrapping halo ranges, to host planes, and the 2LPT square
accumulation), at 1-4 ranks x 1-2 cards and several batch sizes.
"""

import numpy as np
import pytest

pytest.importorskip("jax")

from inexor import ooc_fft  # noqa: E402
from inexor.comm import run_loopback  # noqa: E402
from tests.ranks_common import RANKS_MARKS, need_devices  # noqa: E402

pytestmark = RANKS_MARKS

N = 32
M = N // 2 + 1
DT = np.float32
K = ooc_fft.KSpaceKernel


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _devs(rank, cards):
    import jax

    need_devices(cards)
    d = jax.devices()
    return [d[(rank * cards + k) % len(d)] for k in range(cards)]


def _split(n_ranks):
    return (ooc_fft.partition_units(N, n_ranks, 1), ooc_fft.partition_units(N, n_ranks, 1))


def _cards(lo, hi, devs):
    return [(lo + a, lo + z, d) for (a, z), d in zip(ooc_fft.partition_units(hi - lo, len(devs), 1),
                                                     devs)]


def _gather(parts, y_parts):
    out = np.empty((N, N, M), dtype=parts[0].dtype)
    for p, (y0, y1) in zip(parts, y_parts):
        out[:, y0:y1] = p
    return out


def _key():
    import jax

    return jax.random.PRNGKey(7)


CASES = [(1, 1, 16), (1, 2, 3), (2, 1, 1), (2, 2, 5), (3, 1, 4), (4, 2, 64)]


@pytest.mark.parametrize("kernel", [None, "grad"])
@pytest.mark.parametrize("n_ranks,cards,batch", CASES)
def test_noise_pencils_are_the_single_process_spectrum(n_ranks, cards, batch, kernel):
    import jax

    kern = None if kernel is None else K.grad_invk2(1)
    want = ooc_fft.noise_forward_cards(_key(), N, [jax.devices()[0]], DT, kernel=kern,
                                       box_size=10.0)
    x_parts, y_parts = _split(n_ranks)
    draw = ooc_fft.noise_plane_program(N, DT)

    def rank(c):
        devs = _devs(c.rank, cards)
        lo, hi = x_parts[c.rank]
        return ooc_fft.forward_planes_to_pencils(
            lambda g, dev: draw(jax.device_put(_key(), dev), jax.device_put(np.uint32(g), dev)),
            N, _cards(lo, hi, devs), x_parts, y_parts, comm=c, dtype=DT, kernel=kern,
            box_size=10.0, batch_planes=batch)

    got = _gather(run_loopback(n_ranks, rank, timeout=120.0), y_parts)
    assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize("n_ranks,cards,batch", CASES)
def test_card_shard_and_host_plane_forwards_are_the_single_process_spectra(n_ranks, cards,
                                                                            batch):
    import jax
    import jax.numpy as jnp

    field = np.random.default_rng(3).standard_normal((N, N, N)).astype(DT)
    want_cards = ooc_fft.forward_from_card_planes(
        [dict(lo=0, hi=N, device=jax.devices()[0], delta=jax.device_put(field))], N)
    want_host = ooc_fft.forward_from_slabs_device(lambda lo, hi: field[lo:hi], N,
                                                  kernel=K.grad_invk2(2), box_size=5.0)
    x_parts, y_parts = _split(n_ranks)

    def rank(c):
        devs = _devs(c.rank, cards)
        lo, hi = x_parts[c.rank]
        cs = _cards(lo, hi, devs)
        shard = {a: jax.device_put(field[a:z], d) for a, z, d in cs}
        start = {d: a for a, _z, d in cs}

        def from_shard(g, dev):
            return jnp.fft.rfft2(shard[start[dev]][g - start[dev]:g - start[dev] + 1],
                                 axes=(-2, -1))

        def from_host(g, dev):
            return jnp.fft.rfft2(jax.device_put(field[g:g + 1], dev), axes=(-2, -1))

        a = ooc_fft.forward_planes_to_pencils(from_shard, N, cs, x_parts, y_parts, comm=c,
                                              dtype=DT, batch_planes=batch)
        b = ooc_fft.forward_planes_to_pencils(from_host, N, cs, x_parts, y_parts, comm=c,
                                              dtype=DT, kernel=K.grad_invk2(2), box_size=5.0,
                                              batch_planes=batch)
        return a, b

    out = run_loopback(n_ranks, rank, timeout=120.0)
    assert _gather([o[0] for o in out], y_parts).tobytes() == want_cards.tobytes()
    assert _gather([o[1] for o in out], y_parts).tobytes() == want_host.tobytes()


def _work():
    """A spectrum after the axis-0 inverse (the state `inverse_*` pass 1 reads)."""
    rng = np.random.default_rng(5)
    w = (rng.standard_normal((N, N, M)) + 1j * rng.standard_normal((N, N, M))).astype(np.complex64)
    return w


@pytest.mark.parametrize("n_ranks,cards,batch", CASES)
def test_inverse_planes_are_the_single_process_planes(n_ranks, cards, batch):
    """Card shards with halo ranges that wrap (U_x), host planes (V, U_y, U_z) and the 2LPT
    square accumulation, against `inverse_to_card_shards`, `inverse_to_slabs_device` and
    `inverse_accumulate_cards`' per-plane program."""
    import jax
    import jax.numpy as jnp

    work = _work()
    d0 = jax.devices()[0]
    halo = 3
    x_parts, y_parts = _split(n_ranks)
    # every rank's card ranges with halo either side, as U_x's (they wrap at the ends)
    ranges = {}
    for r, (lo, hi) in enumerate(x_parts):
        for k, (a, z) in enumerate(ooc_fft.partition_units(hi - lo, cards, 1)):
            ranges[(r, k)] = (lo + a - halo, z - a + 2 * halo)
    want_shards = {rk: np.asarray(a) for rk, a in zip(
        ranges, ooc_fft.inverse_to_card_shards(work.copy(), N, [(x0, nx, d0) for x0, nx in
                                                                ranges.values()], pass2=False))}
    want_planes = np.concatenate([s for _lo, s in ooc_fft.inverse_to_slabs_device(
        work.copy(), N, pass2=False)])
    acc0 = np.random.default_rng(9).standard_normal((N, N, N)).astype(DT)
    add = ooc_fft.acc_sq_program(N, DT)
    want_acc = jax.device_put(acc0, d0)
    for g in range(N):
        want_acc = add(want_acc, jax.device_put(np.int64(g), d0),
                       jax.device_put(work[g:g + 1], d0), jax.device_put(DT(-0.5), d0))
    want_acc = np.asarray(want_acc)

    def rank(c):
        devs = _devs(c.rank, cards)
        y0, y1 = y_parts[c.rank]
        mine = np.ascontiguousarray(work[:, y0:y1])
        lo, hi = x_parts[c.rank]
        shards = {}

        def to_shard(k, i, g, sp):
            shards[(k, i)] = np.asarray(jnp.fft.irfft2(sp, s=(N, N), axes=(-2, -1))[0])

        cs = [(*ranges[(c.rank, k)], devs[k]) for k in range(cards)]
        ooc_fft.inverse_pencils_to_planes(mine, N, cs, y_parts, to_shard, comm=c,
                                          batch_planes=batch)
        planes = np.empty((hi - lo, N, N), dtype=DT)

        def to_host(k, i, g, sp):
            planes[g - lo] = np.asarray(jnp.fft.irfft2(sp, s=(N, N), axes=(-2, -1)))[0]

        own = _cards(lo, hi, devs)
        ooc_fft.inverse_pencils_to_planes(mine, N, [(a, z - a, d) for a, z, d in own], y_parts,
                                          to_host, comm=c, batch_planes=batch)
        acc = {k: jax.device_put(acc0[a:z], d) for k, (a, z, d) in enumerate(own)}

        def accumulate(k, i, g, sp):
            d = own[k][2]
            acc[k] = add(acc[k], jax.device_put(np.int64(i), d), sp,
                         jax.device_put(DT(-0.5), d))

        ooc_fft.inverse_pencils_to_planes(mine, N, [(a, z - a, d) for a, z, d in own], y_parts,
                                          accumulate, comm=c, batch_planes=batch)
        return shards, planes, np.concatenate([np.asarray(acc[k]) for k in range(cards)])

    out = run_loopback(n_ranks, rank, timeout=120.0)
    for r, (shards, planes, acc) in enumerate(out):
        lo, hi = x_parts[r]
        for k in range(cards):
            got = np.stack([shards[(k, i)] for i in range(ranges[(r, k)][1])])
            assert got.tobytes() == want_shards[(r, k)].tobytes(), f"rank {r} card {k}"
        assert planes.tobytes() == want_planes[lo:hi].tobytes(), f"rank {r} planes"
        assert acc.tobytes() == want_acc[lo:hi].tobytes(), f"rank {r} accumulated source"


# --- the generator across ranks ---

GEN_N, GEN_NB, GEN_BOX = 64, 8, 32.0
# its largest velocity sits at plane 41, on a rank other than 0 at 2, 3 and 4 ranks, so a
# rank-local maximum (the manifest's vel_scale comes from rank 0) changes the bytes
GEN_SEED = 10
IGNORED = ("provenance", "stage_s", "emission_s", "n_devices", "n_ranks", "stage_cleanup")


def _generate(workdir, comm, devs, f_NL=0.0, window=1, batch=3, log=None):
    from inexor import icgen
    from inexor.config import Cosmology

    import jax

    return icgen.generate_t9_slabs_device(
        str(workdir), jax.random.PRNGKey(GEN_SEED), GEN_N, GEN_BOX, Cosmology(), 0.1, GEN_NB,
        f_NL=f_NL, window=window, devices=devs, comm=comm, batch_planes=batch, log=log)


def _ic_dir(d):
    """{file: bytes} of the slab files, and the manifest without its run-specific fields."""
    import json
    import os

    man = json.load(open(os.path.join(d, "manifest.json")))
    files = {f: open(os.path.join(d, f), "rb").read() for f in man["files"]}
    return files, {k: v for k, v in man.items() if k not in IGNORED}


@pytest.fixture(scope="module")
def one_rank_ics(tmp_path_factory):
    import jax

    jax.config.update("jax_enable_x64", True)
    out = {}
    for f_NL, window in ((0.0, 1), (0.5, 1), (0.0, 2)):
        d = tmp_path_factory.mktemp(f"one-{f_NL}-{window}")
        _generate(d, None, [jax.devices()[0]], f_NL=f_NL, window=window)
        out[(f_NL, window)] = _ic_dir(d)
    return out


@pytest.mark.parametrize("n_ranks,cards,batch,f_NL,window", [
    (1, 2, 3, 0.0, 1), (2, 1, 1, 0.0, 1), (2, 2, 3, 0.5, 1), (3, 1, 64, 0.5, 1),
    (4, 2, 5, 0.0, 1), (4, 1, 3, 0.5, 1), (2, 2, 3, 0.0, 2), (4, 2, 2, 0.0, 2)])
def test_rank_ics_are_the_one_rank_ics(one_rank_ics, tmp_path, n_ranks, cards, batch, f_NL,
                                       window):
    files, man = one_rank_ics[(f_NL, window)]
    assert len(files) == GEN_NB and man["n_particles"] == GEN_N**3

    def rank(c):
        return _generate(tmp_path, c, _devs(c.rank, cards), f_NL=f_NL, window=window,
                         batch=batch)

    mans = run_loopback(n_ranks, rank, timeout=300.0)
    assert all(m == mans[0] for m in mans)
    got_files, got_man = _ic_dir(tmp_path)
    assert got_man == man
    assert sorted(got_files) == sorted(files)
    for f in files:
        assert got_files[f] == files[f], f
    assert mans[0].get("n_ranks", 1) == n_ranks


def test_a_failing_rank_leaves_no_manifest(tmp_path):
    import os

    def rank(c):
        def log(line):
            if c.rank == 1 and "source" in line:
                raise RuntimeError("rank 1 fails after the 2LPT source, on purpose")
        return _generate(tmp_path, c, _devs(c.rank, 1), log=log)

    with pytest.raises(Exception):
        run_loopback(2, rank, timeout=300.0)
    assert not os.path.exists(os.path.join(tmp_path, "manifest.json"))


def test_the_host_lanes_are_refused_across_ranks(tmp_path):
    from inexor import icgen
    from inexor.config import Cosmology

    import jax

    def rank(c):
        with pytest.raises(ValueError, match="across ranks"):
            icgen.generate_t9_slabs_device(str(tmp_path), jax.random.PRNGKey(4), GEN_N, GEN_BOX,
                                           Cosmology(), 0.1, GEN_NB, devices=_devs(c.rank, 1),
                                           comm=c, noise="host")
        return True

    assert all(run_loopback(2, rank, timeout=60.0))
