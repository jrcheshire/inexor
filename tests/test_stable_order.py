"""`_stable_order` must return the SAME permutation as the wide `argsort`.

It exists only to put a small-range integer sort on numpy's radix path, so its
entire contract is that nothing observable changes. The encode downstream is
order-dependent through a float max, so an "equivalent" permutation that differs
on ties would move bits without losing a particle -- the hardest kind of defect
to notice.

The range check is tested in both directions, because numpy narrows MODULARLY:
a key of 65536 cast to uint16 stores as 0 and sorts first. D-v2-20 found exactly
that already live in `migrate` and `repack`, where the narrowing was bare.
"""

import numpy as np
import pytest

from inexor.state import _stable_order


def _same(key, n_values):
    got = _stable_order(key, n_values)
    want = np.argsort(np.asarray(key), kind="stable")
    return np.array_equal(got, want)


@pytest.mark.parametrize("n_values", [2, 256, 257, 512, 65536])
def test_matches_wide_argsort_on_random(n_values):
    rng = np.random.default_rng(11)
    key = rng.integers(0, n_values, size=4096).astype(np.int64)
    assert _same(key, n_values)


@pytest.mark.parametrize(
    "name,key",
    [
        ("empty", np.empty(0, dtype=np.int64)),
        ("single", np.array([7], dtype=np.int64)),
        ("all-equal", np.full(1024, 3, dtype=np.int64)),
        ("already-sorted", np.arange(512, dtype=np.int64)),
        ("reversed", np.arange(512, dtype=np.int64)[::-1].copy()),
        ("two-values", np.array([0, 511] * 512, dtype=np.int64)),
        ("zeros-then-max", np.concatenate([np.zeros(511, np.int64), [511]])),
        ("max-then-zeros", np.concatenate([[511], np.zeros(511, np.int64)])),
    ],
)
def test_matches_wide_argsort_on_adversarial(name, key):
    assert _same(key, 512), name


def test_ties_keep_original_order():
    """Stability is the load-bearing property, so assert it directly."""
    key = np.array([2, 0, 2, 0, 1, 2], dtype=np.int64)
    order = _stable_order(key, 3)
    assert list(order) == [1, 3, 4, 0, 2, 5]


def test_out_of_range_falls_back_instead_of_wrapping():
    """A key above the narrow type's range must NOT be cast."""
    key = np.array([65536, 1, 0], dtype=np.int64)
    # n_values claims it fits; the VALUES say otherwise, and the values win
    assert _same(key, 512)
    assert list(_stable_order(key, 512)) == [2, 1, 0]


def test_negative_keys_fall_back():
    key = np.array([-1, 5, 0], dtype=np.int64)
    assert _same(key, 512)
    assert list(_stable_order(key, 512)) == [0, 2, 1]


def test_the_narrow_path_is_actually_taken(monkeypatch):
    """Anti-vacuity: prove the cast happens, or every test above is a no-op.

    Without this, a `_stable_order` that simply forwarded to `np.argsort` on the
    wide key would pass all of the above -- the suite would be asserting that a
    function equals itself.
    """
    seen = []
    real = np.argsort

    def spy(a, *args, **kwargs):
        seen.append(np.asarray(a).dtype)
        return real(a, *args, **kwargs)

    monkeypatch.setattr(np, "argsort", spy)
    _stable_order(np.arange(512, dtype=np.int64)[::-1].copy(), 512)
    assert seen and seen[-1] == np.uint16, f"expected a uint16 sort, saw {seen}"

    seen.clear()
    _stable_order(np.arange(200, dtype=np.int64)[::-1].copy(), 256)
    assert seen and seen[-1] == np.uint8, f"expected a uint8 sort, saw {seen}"

    seen.clear()
    _stable_order(np.array([70000, 1], dtype=np.int64), 512)
    assert seen and seen[-1] == np.int64, f"expected the wide sort, saw {seen}"
