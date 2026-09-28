"""Manual repair (cut/join) of the AUTOMATIC flow-/volume-based breath segmentation:
``core.compute.apply_segmentation_overrides``. Contrast
``core.analysis.segments``'s ``separators``/``whole_file``, which are the
EMG-only, no-flow-channel counterpart tested in ``test_segments_emg.py``.

All tests call the pure function directly with numpy-synthesised flow/volume arrays,
the same style ``test_segments_emg.py`` already uses for ``whole_file``/``separators`` —
no settings/TOML machinery needed at this level.
"""
import numpy as np
import pytest

from respmech.core.compute import (
    SegmentationOverrideNotice, _walk_insp_end, apply_segmentation_overrides,
    separateintobreathsbyflow,
)
from respmech.core.analysis.segments import remap_segment_number

FS = 100
# bufferwidth=1 collapses the buffered-mean criterion to a plain per-sample sign
# check, so a synthetic square-wave flow signal's zero crossings land EXACTLY where
# the test arithmetic below expects them, with no smoothing to reason about.
BUF = 1


def _square_breath(ti_samples, te_samples):
    """One breath: `ti_samples` of inspiration (flow < 0) then `te_samples` of
    expiration (flow > 0) -- the same square-wave shape `separateintobreathsbyflow`'s
    own zero-crossing walk (with BUF=1) segments unambiguously."""
    return np.concatenate([np.full(ti_samples, -1.0), np.full(te_samples, 1.0)])


def _make_recording(n_breaths=3, ti=100, te=100):
    flow = np.concatenate([_square_breath(ti, te) for _ in range(n_breaths)])
    n = len(flow)
    timecol = np.arange(n) / FS
    # `volume` is a channel INDEPENDENT of flow (as it is in a real recording: a
    # separate pneumotach/spirometer signal) -- a smooth ramp with no relation to the
    # flow wobble a later test injects, so a join/cut test can assert on it without
    # the wobble itself leaking into the comparison.
    volume = np.linspace(0.0, 1.0, n)
    empty = np.array([])
    return timecol, flow, volume, empty, empty, empty, empty, empty


def _auto_breaths(filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns):
    class _Mechanics:
        breathseparationbuffer = BUF
        excludebreaths = []
        breathtypes = []

    class _Processing:
        mechanics = _Mechanics

    class _Ns:
        processing = _Processing

    return separateintobreathsbyflow(
        filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, _Ns)


def test_walk_insp_end_matches_the_unbounded_walker_on_a_clean_segment():
    ti, te = 100, 100
    flow = _square_breath(ti, te)
    assert _walk_insp_end(flow, 0, len(flow), BUF) == ti


def test_walk_insp_end_ignores_a_wobble_after_the_real_transition():
    """The whole point of the bounded, single-transition walker: once inspiration has
    ended, any LATER sign change within the segment (the artefact a `join_s` exists to
    absorb) must not be found as a second transition."""
    ti, te = 100, 100
    flow = _square_breath(ti, te)
    flow[150:155] = -1.0    # a brief negative wobble mid-expiration
    assert _walk_insp_end(flow, 0, len(flow), BUF) == ti


def test_no_overrides_is_never_called_by_the_pipeline_guard():
    """Documents the contract `core.pipeline.segment_file` relies on for its
    empty-overrides byte-identity guarantee: this module exposes no "apply with empty
    lists" behaviour to test, because the caller never calls it in that case (see
    `apply_segmentation_overrides`'s own docstring). Nothing to assert here beyond the
    function existing with that documented contract -- a smoke check that the import
    above succeeds."""
    assert callable(apply_segmentation_overrides)


# -- join_s: undo an auto-detector over-split (flow wobble) ------------------------

def test_join_recovers_the_clean_breath_count_and_volume_1e9():
    """The literal acceptance criterion this function must meet: a flow wobble that
    splits one breath into two, joined back, gives the SAME breath count and the SAME
    per-breath volume arrays (1e-9) as the clean recording -- because `volume` here is a channel
    the wobble never touched (see `_make_recording`'s docstring), and `_walk_insp_end`
    finds the SAME inspiration end in both cases (the wobble sits entirely after it)."""
    ti, te = 100, 100
    timecol, clean_flow, volume, poes, pgas, pdi, ent, emg = _make_recording(3, ti, te)
    clean_breaths = _auto_breaths("clean.csv", timecol, clean_flow, volume, poes, pgas, pdi, ent, emg)
    assert list(clean_breaths.keys()) == [1, 2, 3]

    wobble_flow = clean_flow.copy()
    # breath #2 spans [200, 400) (insp [200,300), exp [300,400)); inject a brief
    # negative dip at [350, 355), well inside its expiration.
    wobble_flow[350:355] = -1.0
    wobble_breaths = _auto_breaths(
        "wobble.csv", timecol, wobble_flow, volume, poes, pgas, pdi, ent, emg)
    assert len(wobble_breaths) == 4, "the wobble must actually over-split for this test to mean anything"
    false_boundary_s = float(wobble_breaths[3]["time"][0])
    assert false_boundary_s == pytest.approx(3.5)

    joined, notices = apply_segmentation_overrides(
        "wobble.csv", wobble_breaths, cut_s=[], join_s=[false_boundary_s],
        timecol=timecol, flow=wobble_flow, volume=volume, poes=poes, pgas=pgas, pdi=pdi,
        entropycolumns=ent, emgcolumns=emg, fs=FS, bufferwidth=BUF,
        ignored_breaths=[], kinds={})
    assert notices == []
    assert list(joined.keys()) == [1, 2, 3]
    for n in (1, 2, 3):
        np.testing.assert_allclose(
            joined[n]["inspiration"]["volume"], clean_breaths[n]["inspiration"]["volume"],
            rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            joined[n]["expiration"]["volume"], clean_breaths[n]["expiration"]["volume"],
            rtol=0, atol=1e-9)


def test_join_with_no_nearby_automatic_boundary_is_a_soft_notice():
    timecol, flow, volume, poes, pgas, pdi, ent, emg = _make_recording(2, 100, 100)
    auto = _auto_breaths("f.csv", timecol, flow, volume, poes, pgas, pdi, ent, emg)
    result, notices = apply_segmentation_overrides(
        "f.csv", auto, cut_s=[], join_s=[1.0],   # 1.0s is mid-breath-1, not a boundary
        timecol=timecol, flow=flow, volume=volume, poes=poes, pgas=pgas, pdi=pdi,
        entropycolumns=ent, emgcolumns=emg, fs=FS, bufferwidth=BUF,
        ignored_breaths=[], kinds={})
    assert len(notices) == 1
    assert isinstance(notices[0], SegmentationOverrideNotice)
    assert "no automatic breath boundary" in notices[0].message
    assert list(result.keys()) == [1, 2]   # unaffected -- the bad join was ignored


def test_join_tolerance_matches_the_documented_max_2_over_fs_or_0_05_s():
    """`tolerance_s=None` defaults to max(2/fs, 0.05) -- at FS=100 that is 0.05 s. A
    join 0.05 s off the real boundary (200/FS=2.0s) still resolves; 0.06 s off does
    not."""
    timecol, flow, volume, poes, pgas, pdi, ent, emg = _make_recording(2, 100, 100)
    auto = _auto_breaths("f.csv", timecol, flow, volume, poes, pgas, pdi, ent, emg)
    _, notices_ok = apply_segmentation_overrides(
        "f.csv", auto, cut_s=[], join_s=[2.05],
        timecol=timecol, flow=flow, volume=volume, poes=poes, pgas=pgas, pdi=pdi,
        entropycolumns=ent, emgcolumns=emg, fs=FS, bufferwidth=BUF,
        ignored_breaths=[], kinds={})
    assert notices_ok == []
    _, notices_bad = apply_segmentation_overrides(
        "f.csv", auto, cut_s=[], join_s=[2.06],
        timecol=timecol, flow=flow, volume=volume, poes=poes, pgas=pgas, pdi=pdi,
        entropycolumns=ent, emgcolumns=emg, fs=FS, bufferwidth=BUF,
        ignored_breaths=[], kinds={})
    assert len(notices_bad) == 1


# -- cut_s: split one auto-detected breath into two ---------------------------------

def test_cut_mid_breath_conserves_total_ti_plus_te():
    """The literal acceptance criterion this function must meet: a cut in the middle of a breath
    gives two breaths whose Ti+Te sums to the original's."""
    ti, te = 100, 100
    timecol, flow, volume, poes, pgas, pdi, ent, emg = _make_recording(1, ti, te)
    auto = _auto_breaths("f.csv", timecol, flow, volume, poes, pgas, pdi, ent, emg)
    original_total = len(np.asarray(auto[1]["time"]).reshape(-1))

    cut_at_s = 1.5   # sample 150 -- mid-expiration (insp is [0,100), exp is [100,200))
    result, notices = apply_segmentation_overrides(
        "f.csv", auto, cut_s=[cut_at_s], join_s=[],
        timecol=timecol, flow=flow, volume=volume, poes=poes, pgas=pgas, pdi=pdi,
        entropycolumns=ent, emgcolumns=emg, fs=FS, bufferwidth=BUF,
        ignored_breaths=[], kinds={})
    assert notices == []
    assert list(result.keys()) == [1, 2]
    total_after = sum(
        len(np.asarray(result[n]["inspiration"]["time"]).reshape(-1))
        + len(np.asarray(result[n]["expiration"]["time"]).reshape(-1))
        for n in (1, 2))
    assert total_after == original_total
    # breath 1 keeps the real inspiration (minus the same one-sample transition drop
    # every OTHER breath boundary in this codebase already has, see
    # apply_segmentation_overrides's own comment on why) + the truncated expiration;
    # breath 2 is a degenerate zero-length-inspiration tail (a cut placed
    # mid-expiration never invents a second inspiration that was not in the data).
    assert len(np.asarray(result[1]["inspiration"]["time"]).reshape(-1)) == ti - 1
    assert len(np.asarray(result[2]["inspiration"]["time"]).reshape(-1)) == 0


def test_cut_mid_inspiration_also_conserves_total_ti_plus_te():
    """The mirror of the mid-expiration case above, and a real regression: an
    earlier version of ``apply_segmentation_overrides`` applied the transition '-1'
    to the FIRST resulting segment's own inspiration end unconditionally, even when
    that segment never found a real transition and simply ran into its own CUT
    boundary (the degenerate case this cut placement produces, since the cut sits
    before the breath's real insp->exp transition) — silently dropping one real
    sample that belongs to neither resulting breath. Fixed by only applying the '-1'
    when the segment's inspiration end is a genuine transition OR the segment's own
    end is a NATURAL boundary (matching how ``exend`` already treats natural vs. cut
    boundaries) -- never when both are false."""
    ti, te = 100, 100
    timecol, flow, volume, poes, pgas, pdi, ent, emg = _make_recording(1, ti, te)
    auto = _auto_breaths("f.csv", timecol, flow, volume, poes, pgas, pdi, ent, emg)
    original_total = len(np.asarray(auto[1]["time"]).reshape(-1))

    cut_at_s = 0.5   # sample 50 -- mid-INSPIRATION (insp is [0,100), exp is [100,200))
    result, notices = apply_segmentation_overrides(
        "f.csv", auto, cut_s=[cut_at_s], join_s=[],
        timecol=timecol, flow=flow, volume=volume, poes=poes, pgas=pgas, pdi=pdi,
        entropycolumns=ent, emgcolumns=emg, fs=FS, bufferwidth=BUF,
        ignored_breaths=[], kinds={})
    assert notices == []
    assert list(result.keys()) == [1, 2]
    total_after = sum(
        len(np.asarray(result[n]["inspiration"]["time"]).reshape(-1))
        + len(np.asarray(result[n]["expiration"]["time"]).reshape(-1))
        for n in (1, 2))
    assert total_after == original_total
    # breath 1 is a degenerate zero-length-expiration stub (pure inspiration up to
    # the cut, nothing dropped: the cut boundary itself keeps no '-1'); breath 2
    # keeps the real inspiration remainder (minus the one-sample transition drop, a
    # genuine transition this time) plus the expiration (itself minus the one-sample
    # drop at the file's own natural end, same as the uncut original breath's own
    # exp length, see the mid-expiration test above).
    assert len(np.asarray(result[1]["expiration"]["time"]).reshape(-1)) == 0
    assert len(np.asarray(result[1]["inspiration"]["time"]).reshape(-1)) == 50
    assert len(np.asarray(result[2]["expiration"]["time"]).reshape(-1)) == te - 1


def test_cut_outside_the_recording_is_a_soft_notice():
    timecol, flow, volume, poes, pgas, pdi, ent, emg = _make_recording(1, 100, 100)
    auto = _auto_breaths("f.csv", timecol, flow, volume, poes, pgas, pdi, ent, emg)
    result, notices = apply_segmentation_overrides(
        "f.csv", auto, cut_s=[999.0], join_s=[],
        timecol=timecol, flow=flow, volume=volume, poes=poes, pgas=pgas, pdi=pdi,
        entropycolumns=ent, emgcolumns=emg, fs=FS, bufferwidth=BUF,
        ignored_breaths=[], kinds={})
    assert len(notices) == 1
    assert "falls outside the recording" in notices[0].message
    assert list(result.keys()) == [1]


def test_cut_colliding_with_an_existing_boundary_is_a_soft_notice():
    timecol, flow, volume, poes, pgas, pdi, ent, emg = _make_recording(2, 100, 100)
    auto = _auto_breaths("f.csv", timecol, flow, volume, poes, pgas, pdi, ent, emg)
    result, notices = apply_segmentation_overrides(
        "f.csv", auto, cut_s=[2.0], join_s=[],   # exactly the existing boundary
        timecol=timecol, flow=flow, volume=volume, poes=poes, pgas=pgas, pdi=pdi,
        entropycolumns=ent, emgcolumns=emg, fs=FS, bufferwidth=BUF,
        ignored_breaths=[], kinds={})
    assert len(notices) == 1
    assert "coincides with an existing boundary" in notices[0].message
    assert list(result.keys()) == [1, 2]


# -- ignore/kind follow the NEW numbering, not the old one --------------------------

def test_ignored_breaths_and_kinds_apply_to_the_new_numbering():
    timecol, flow, volume, poes, pgas, pdi, ent, emg = _make_recording(3, 100, 100)
    auto = _auto_breaths("f.csv", timecol, flow, volume, poes, pgas, pdi, ent, emg)
    # cut breath 1 in two -> the OLD breath 2 (unaffected) is now breath 3, so marking
    # NEW breath 3 as ignored/typed must land on what WAS breath 2.
    result, _ = apply_segmentation_overrides(
        "f.csv", auto, cut_s=[1.5], join_s=[],
        timecol=timecol, flow=flow, volume=volume, poes=poes, pgas=pgas, pdi=pdi,
        entropycolumns=ent, emgcolumns=emg, fs=FS, bufferwidth=BUF,
        ignored_breaths=[3], kinds={4: "ic"})
    assert list(result.keys()) == [1, 2, 3, 4]
    assert result[3]["ignored"] is True
    assert result[1]["ignored"] is False and result[2]["ignored"] is False
    assert result[4]["kind"] == "ic"


# -- renumbering reuse: remap_segment_number generalises to overrides --------------

def test_remap_segment_number_reused_for_a_cut_before_an_excluded_breath():
    """A cut/join edit must renumber existing exclusions/types the same way placing
    or removing an EMG-only separator already does (old start time -> new index).
    Reproduces that acceptance criterion at the level `remap_segment_number` itself
    operates on -- a boundary list before/after an edit -- since that function is
    already generic over WHERE the boundary list came from (`separators` or, as
    here, an auto+override list); the UI wiring (`ui.screens.preview._mechanics`)
    calls it the same way `_segments.py`'s `_set_separators` already does for
    `processing.segmentation.separators`.

    3 breaths at [0,200), [200,400), [400,600); breath 3 (old numbering) is excluded.
    Cutting at t=1.0s (inside breath 1) inserts a new boundary before everything else,
    shifting every later breath's number up by one -- the exclusion must follow breath
    3 to its new number, 4."""
    old_bounds_s = [0.0, 2.0, 4.0]              # [0,200), [200,400), [400,600)
    new_bounds_s = sorted([0.0, 1.0, 2.0, 4.0])  # a cut at 1.0s adds one boundary
    assert remap_segment_number(old_bounds_s, new_bounds_s, 3) == 4
    # the untouched breaths keep their relative order, just shifted by the insertion:
    assert remap_segment_number(old_bounds_s, new_bounds_s, 1) == 1
    assert remap_segment_number(old_bounds_s, new_bounds_s, 2) == 3
