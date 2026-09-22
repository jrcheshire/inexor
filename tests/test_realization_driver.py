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


# --- carding the ICs (`--ic-dir`) ------------------------------------------
#
# The step-40 card measures P(k) against LINEAR theory, so its z profile mixes
# the realization's own IC draw with whatever the 40 steps did. Carding the ICs
# through the same estimator separates them, and needs its own loader:
# `load_checkpoint` refuses an IC generation by design.


class _Args:
    def __init__(self, tmp, ic_dir):
        self.ic_dir = str(ic_dir)
        self.workdir = str(tmp)
        self.slack = 0.10
        self.alloc_margin = 0.10
        self.arena_frac = 0.01


def _manifest(d, kind):
    import json

    from inexor import icgen

    d.mkdir(parents=True, exist_ok=True)
    prov = {} if kind is None else {"kind": kind, "step": 40}
    (d / icgen.MANIFEST).write_text(json.dumps({"schema": "t9-slabs-2",
                                                "provenance": prov}))
    return d


def test_a_checkpoint_passed_as_ic_dir_is_refused(tmp_path):
    """A checkpoint loads fine through `load_slot_state`, so nothing downstream
    would notice; it would just be scored against the a_init oracle."""
    import pytest

    ck = _manifest(tmp_path / "gen1", "inexor-checkpoint")
    with pytest.raises(SystemExit) as e:
        rlz._ic_state(_Args(tmp_path, ck))
    assert "CHECKPOINT" in str(e.value)


def test_a_missing_manifest_is_refused_by_name(tmp_path):
    import pytest

    with pytest.raises(SystemExit) as e:
        rlz._ic_state(_Args(tmp_path, tmp_path / "nope"))
    assert "--ic-dir" in str(e.value)


def test_an_ic_generation_reaches_the_ic_loader(tmp_path, monkeypatch):
    """The IC branch must call `load_slot_state`, not `load_checkpoint`, and
    must pass the caller's capacity knobs through rather than the defaults."""
    from inexor import icgen

    ic = _manifest(tmp_path / "c-r0", None)
    seen = {}

    def fake(workdir, **kw):
        seen.update(workdir=workdir, **kw)
        return "STATE"

    monkeypatch.setattr(icgen, "load_slot_state", fake)
    args = _Args(tmp_path, ic)
    assert rlz._ic_state(args, alloc="ALLOC") == "STATE"
    assert seen["workdir"] == str(ic)
    assert seen["alloc"] == "ALLOC"
    assert seen["brick_slack"] == args.slack and seen["arena_frac"] == args.arena_frac


def test_the_card_leg_routes_ic_dir_to_step_zero(tmp_path):
    """Vacuity guard. `a_out` is indexed by `step`, so the IC branch returning
    anything but 0 scores the ICs at the wrong epoch -- silently."""
    import inspect

    src = inspect.getsource(rlz.cmd_card)
    assert "args.ic_dir" in src and "_ic_state(" in src
    assert "_ic_state(args, alloc=allocator), 0" in src


def test_an_ic_card_cannot_overwrite_the_checkpoint_card():
    """Both land in --workdir and they are the two epochs being compared."""
    import inspect

    src = inspect.getsource(rlz.cmd_card)
    assert 'tag=("_ics" if args.ic_dir else "")' in src
