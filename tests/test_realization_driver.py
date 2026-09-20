"""The realization driver's reporting path, which only ever runs on a cluster.

gb 1010730 died in `cmd_card` on `float(card["k_nonlinear"])`: the card
documents that value as None when the linear Delta^2 never reaches 1 inside
the range `nonlinear_scale` scans, and at a = 0.1189 -- the cgh64 smoke's step
3 -- it does not. The leg that found it is the one guarding the hero legs, so
the cost was three minutes; the same line would have run at 4096^3 too.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))

import v2_m6_realization as rlz  # noqa: E402

K = np.array([0.05, 0.2, 0.5, 1.0, 2.0])
SCAN_HI = 10.0


def test_a_known_nonlinear_scale_splits_the_band():
    lin, txt = rlz._linear_band(K, 0.6, SCAN_HI)
    assert list(lin) == [True, True, True, False, False]
    assert txt == "0.6000"


def test_no_crossing_leaves_every_scanned_bin_linear():
    """The 1010730 case. No crossing anywhere scanned means linear theory
    applies across the band, not that the band is unjudgeable."""
    lin, txt = rlz._linear_band(K, None, SCAN_HI)
    assert lin.all()
    assert "none below" in txt and "10" in txt


def test_bins_past_the_scanned_range_are_not_claimed_either_way():
    """None says nothing about k the scan never looked at, so a bin above the
    ceiling is excluded rather than called linear."""
    k = np.array([1.0, SCAN_HI * 2])
    lin, _ = rlz._linear_band(k, None, SCAN_HI)
    assert list(lin) == [True, False]
    # and a card from before the scan range was recorded claims nothing at all
    lin, txt = rlz._linear_band(k, None, None)
    assert not lin.any() and "unknown" in txt


def test_the_helper_is_what_the_card_leg_calls():
    """Vacuity guard: these tests are worth nothing if `cmd_card` still does
    its own float()."""
    import inspect

    src = inspect.getsource(rlz.cmd_card)
    assert "_linear_band(" in src
    assert 'float(card["k_nonlinear"])' not in src
