"""Unit tests for the K-035 boundary-truncation quality notice (ticket 20260906-1109).

Background: ``compute.trim`` discards only a leading partial expiration and a trailing
partial inspiration; it has no way to tell whether the breath it KEEPS at either
boundary is itself complete. A recording that begins already in inspiration, or ends
still in expiration, keeps that truncated boundary breath and analyses it as if
whole — and, with drift correction on (the default), that truncation also tilts the
volume baseline of every OTHER breath in the file, with no error and (until this
change) no warning anywhere. See K-035
(leverancer/2026-09-05-respmech-indholdsgennemgang/…-bilag.md, line ~488) and the
closed ticket 20260905-1156 for the full investigation and the originally measured
numbers (breath 1 Ti 1.161 s vs 1.661 s uncut, vol_endexp drift -0.269 to -0.684 L).

The detection compares the first/last breath's inspiratory/expiratory duration
against this file's own median for OTHER breaths, rather than testing the raw
boundary sample's sign: a first attempt at the sign-based version false-flagged the
built-in sample recording and the committed golden synthetic inputs, both of which
end (or begin) a hair's-breadth short of a full extra phase without being truncated
at all -- indistinguishable from real truncation at the single-sample level, but not
a real bug. ``test_committed_golden_synthetic_input_has_no_false_positive`` and
``test_builtin_sample_recording_has_no_false_positive`` below are the regression
tests for exactly that false positive.

The threshold itself (``min_relative_duration=0.8``) was corrected during self-review
from an initial 0.6, which — verified by replaying K-035's own exact reproduction
(0.5 s into the built-in sample's first inspiration and last expiration) through this
function — did NOT catch the motivating inspiratory case at all (ratio ~0.72).
``test_reproduces_k035_own_measured_case`` below pins that exact scenario so the
threshold can never regress back below it unnoticed.

Follow-up (ticket 20260906-1307): a review raised a well-founded, literature-backed
concern that 0.8 may over-flag ordinary high-variability breathing. A Monte Carlo
comparison against a MAD-based alternative (see ``trim_boundary_notices``'s own
docstring for the full reasoning and numbers) showed the alternative trades away
real detection power rather than being an unambiguous improvement, so the decision
was to keep 0.8/3 as documented defaults and expose both as a per-analysis SETTING
instead (``Settings.processing.segmentation.boundary_notice_min_relative_duration``/
``boundary_notice_min_other_breaths``). ``TestBoundaryNoticeSettings`` below covers
the new setting end to end (default, override via Settings, and validation).
"""
import numpy as np
import pytest

from respmech.core import compute, pipeline
from respmech.core.settings import ExcludeEntry, Settings
from respmech.core._legacy_ns import to_legacy_ns

from _helpers import requires_synth, synth_settings

FS = 200


# --- pure function, against a hand-built breaths dict -----------------------

def _breath(n_insp, n_exp, *, ignored=False):
    return {"inspiration": {"time": np.zeros(n_insp)}, "expiration": {"time": np.zeros(n_exp)},
            "ignored": ignored}


def _dummy_settings(*, correct_drift=True, fs=FS):
    s = Settings()
    s.input.format.sampling_frequency = fs
    s.processing.volume.correct_drift = correct_drift
    return to_legacy_ns(s)


class TestTrimBoundaryNoticesPure:
    def _breaths(self, n=6, insp=100, exp=150, *, first_insp=None, last_exp=None,
                first_ignored=False, last_ignored=False):
        breaths = {i: _breath(insp, exp) for i in range(1, n + 1)}
        if first_insp is not None:
            breaths[1] = _breath(first_insp, exp, ignored=first_ignored)
        elif first_ignored:
            breaths[1]["ignored"] = True
        if last_exp is not None:
            breaths[n] = _breath(insp, last_exp, ignored=last_ignored)
        elif last_ignored:
            breaths[n]["ignored"] = True
        return breaths

    def test_uniform_breaths_give_no_notice(self):
        assert compute.trim_boundary_notices(self._breaths(), _dummy_settings()) == []

    def test_single_breath_file_cannot_be_compared_so_no_notice(self):
        assert compute.trim_boundary_notices(self._breaths(n=1), _dummy_settings()) == []

    def test_too_few_other_breaths_skips_the_check(self):
        # n=4 -> 3 other breaths is exactly the min_other_breaths floor (checked ==
        # not <), so this one MUST still fire; n=3 -> 2 others must NOT.
        breaths3 = self._breaths(n=3, first_insp=10)
        assert compute.trim_boundary_notices(breaths3, _dummy_settings()) == []
        breaths4 = self._breaths(n=4, first_insp=10)
        assert len(compute.trim_boundary_notices(breaths4, _dummy_settings())) == 1

    def test_first_breath_much_shorter_inspiration_is_flagged(self):
        # 100-sample typical inspiration, first breath only 30 (30%, well under 80%)
        breaths = self._breaths(first_insp=30)
        notices = compute.trim_boundary_notices(breaths, _dummy_settings())
        assert len(notices) == 1
        assert "first breath" in notices[0] and "inspiration" in notices[0]

    def test_last_breath_much_shorter_expiration_is_flagged(self):
        breaths = self._breaths(last_exp=40)      # 150-sample typical, 40 = ~27%
        notices = compute.trim_boundary_notices(breaths, _dummy_settings())
        assert len(notices) == 1
        assert "last breath" in notices[0] and "expiration" in notices[0]

    def test_both_boundaries_truncated_gives_two_notices(self):
        breaths = self._breaths(first_insp=20, last_exp=30)
        notices = compute.trim_boundary_notices(breaths, _dummy_settings())
        assert len(notices) == 2

    def test_mild_natural_variation_is_not_flagged(self):
        # first breath's inspiration at 90% of typical -- ordinary variability, not truncation
        breaths = self._breaths(insp=100, first_insp=90)
        assert compute.trim_boundary_notices(breaths, _dummy_settings()) == []

    def test_a_ratio_just_under_the_threshold_is_flagged(self):
        # 79% of a 100-sample typical inspiration is just under the 80% threshold
        breaths = self._breaths(insp=100, first_insp=79)
        assert len(compute.trim_boundary_notices(breaths, _dummy_settings())) == 1

    def test_drift_correction_consequence_only_mentioned_when_it_applies(self):
        breaths = self._breaths(first_insp=20)
        on = compute.trim_boundary_notices(breaths, _dummy_settings(correct_drift=True))
        off = compute.trim_boundary_notices(breaths, _dummy_settings(correct_drift=False))
        assert "baseline" in on[0]
        assert "baseline" not in off[0]

    def test_a_middle_breath_being_short_is_not_a_boundary_notice(self):
        # short breath in the MIDDLE of the file (e.g. a sigh) must not trip this check --
        # it is not a boundary and trim() never touches it.
        breaths = self._breaths(n=6)
        breaths[3] = _breath(20, 20)
        assert compute.trim_boundary_notices(breaths, _dummy_settings()) == []

    def test_already_excluded_boundary_breath_with_drift_on_still_warns_differently(self):
        breaths = self._breaths(first_insp=20, first_ignored=True)
        notices = compute.trim_boundary_notices(breaths, _dummy_settings(correct_drift=True))
        assert len(notices) == 1
        assert "already excluded" in notices[0]
        # must not repeat advice the user has already followed, and must not claim the
        # excluded breath is itself still being analysed as complete
        assert "exclude the first breath" not in notices[0]
        assert "analysed as if it were complete" not in notices[0]
        assert "baseline" in notices[0]        # the reason it still matters at all

    def test_already_excluded_boundary_breath_with_drift_off_has_nothing_left_to_warn(self):
        breaths = self._breaths(first_insp=20, first_ignored=True)
        notices = compute.trim_boundary_notices(breaths, _dummy_settings(correct_drift=False))
        assert notices == []

    def test_reproduces_k035_own_measured_case(self):
        """Pin the exact scenario from K-035's own reproduction (leverancer/…-bilag.md):
        breath 1 Ti 1.161 s against a file median of ~1.6 s. This is the case an
        earlier, lower threshold (0.6) shipped during development did NOT catch."""
        insp_typ = round(1.6045 * FS)         # measured median from the built-in sample
        insp_first = round(1.161 * FS)        # K-035's own measured truncated Ti
        breaths = self._breaths(n=9, insp=insp_typ, first_insp=insp_first)
        notices = compute.trim_boundary_notices(breaths, _dummy_settings())
        assert len(notices) == 1
        assert "first breath" in notices[0]


# --- synthetic flow generator (adapted from test_no_breaths.py::_write) ----

def _write_recording(path, *, lead_cut_s=0.0, trail_cut_s=0.0, n_breaths=6, vt=0.6,
                     period_s=4.0, fs=FS):
    """A minimal LabChart-shaped CSV of ``n_breaths`` clean, correctly-cut breaths
    (a plain sinusoidal flow with no lead-in/lead-out), optionally shortened by
    ``lead_cut_s`` seconds at the start (simulating an export that started mid-way
    through the first inspiration) and/or ``trail_cut_s`` seconds at the end
    (simulating an export that stopped mid-way through the last expiration)."""
    per = int(period_s * fs)
    n = n_breaths * per + 1
    t = np.arange(n) / fs
    amp = vt * np.pi / period_s
    flow = -amp * np.sin(2 * np.pi * t / period_s)
    one = (1 - np.cos(2 * np.pi * (t % period_s) / period_s)) / 2
    volume = one * vt
    poes = -5.0 * np.sin(2 * np.pi * t / period_s) - 5.0
    pgas = 2.0 * np.sin(2 * np.pi * t / period_s) + 8.0

    lead_cut = int(lead_cut_s * fs)
    trail_cut = int(trail_cut_s * fs)
    if lead_cut:
        assert flow[lead_cut] < 0, "lead_cut_s must land inside an inspiration"
        t, flow, volume, poes, pgas = (a[lead_cut:] for a in (t, flow, volume, poes, pgas))
        t = t - t[0]
    if trail_cut:
        assert flow[-trail_cut - 1] >= 0, "trail_cut_s must land inside an expiration"
        t, flow, volume, poes, pgas = (a[:-trail_cut] for a in (t, flow, volume, poes, pgas))

    np.savetxt(path, np.column_stack(
        [t, t * 0, t * 0, t * 0, flow, volume, poes, pgas, pgas - poes]), delimiter=",")


def _settings(src, out, *, correct_drift=True, exclude=None):
    s = Settings()
    s.input.folder, s.input.files = str(src), "*.csv"
    s.input.format.sampling_frequency = FS
    c = s.input.channels
    c.flow, c.volume, c.poes, c.pgas, c.pdi = 5, 6, 7, 8, 9
    s.processing.volume.correct_drift = correct_drift
    if exclude is not None:
        s.processing.exclude_breaths = [ExcludeEntry(file=exclude[0], breaths=exclude[1])]
    s.output.folder = str(out)
    return s.validate()


def _dirs(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir(); out.mkdir()
    return src, out


class TestTrimBoundaryNoticesBatch:
    """End to end through ``pipeline.run_batch`` — the real per-file path a batch run
    (CLI or GUI) actually takes."""

    def test_correctly_cut_recording_has_no_boundary_notice(self, tmp_path):
        src, out = _dirs(tmp_path)
        _write_recording(src / "clean.csv")
        result = pipeline.run_batch(_settings(src, out))
        assert result.ok_files
        fr = result.ok_files["clean.csv"]
        assert fr.notices == []
        assert fr.breaths_table is not None and len(fr.breaths_table) >= 2

    def test_cut_mid_inspiration_at_start_is_flagged(self, tmp_path):
        # period_s=4 -> inspiration half-period 2.0 s; cutting 1.0 s leaves ~50% of it,
        # well under the 80% threshold.
        src, out = _dirs(tmp_path)
        _write_recording(src / "cut_start.csv", lead_cut_s=1.0)
        result = pipeline.run_batch(_settings(src, out))
        fr = result.ok_files["cut_start.csv"]
        assert any("inspiration" in n and "first breath" in n for n in fr.notices)
        assert any("baseline" in n for n in fr.notices)     # drift correction is on

    def test_cut_mid_expiration_at_end_is_flagged(self, tmp_path):
        src, out = _dirs(tmp_path)
        _write_recording(src / "cut_end.csv", trail_cut_s=1.0)
        result = pipeline.run_batch(_settings(src, out))
        fr = result.ok_files["cut_end.csv"]
        assert any("expiration" in n and "last breath" in n for n in fr.notices)

    def test_both_boundaries_truncated_via_run_batch(self, tmp_path):
        src, out = _dirs(tmp_path)
        _write_recording(src / "cut_both.csv", lead_cut_s=1.0, trail_cut_s=1.0)
        result = pipeline.run_batch(_settings(src, out))
        fr = result.ok_files["cut_both.csv"]
        assert len(fr.notices) == 2

    def test_notice_survives_into_the_run_report(self, tmp_path):
        from respmech.core.io.writers import write_batch
        src, out = _dirs(tmp_path)
        _write_recording(src / "cut_start.csv", lead_cut_s=1.0)
        s = _settings(src, out)
        result = pipeline.run_batch(s)
        write_batch(result, s, str(out))
        report = (out / "run-report.txt").read_text()
        assert "Quality notices" in report
        assert "    cut_start.csv: the first breath's inspiration" in report

    def test_notice_surfaces_live_as_a_warning_progress_event_naming_the_file(self, tmp_path):
        from respmech.core.pipeline import ProgressEvent
        src, out = _dirs(tmp_path)
        _write_recording(src / "cut_start.csv", lead_cut_s=1.0)
        events = []
        pipeline.run_batch(_settings(src, out), progress=events.append)
        warnings_seen = [e for e in events if isinstance(e, ProgressEvent) and e.kind == "warning"]
        assert any(e.message.startswith("cut_start.csv: ") and "inspiration" in e.message
                  for e in warnings_seen)

    def test_without_drift_correction_no_baseline_claim(self, tmp_path):
        src, out = _dirs(tmp_path)
        _write_recording(src / "cut_start.csv", lead_cut_s=1.0)
        result = pipeline.run_batch(_settings(src, out, correct_drift=False))
        fr = result.ok_files["cut_start.csv"]
        assert any("inspiration" in n for n in fr.notices)
        assert not any("baseline" in n for n in fr.notices)

    def test_excluding_the_truncated_breath_does_not_silence_a_still_valid_warning(self, tmp_path):
        """Following the notice's own remedy (exclude the breath) does not, by itself,
        undo the drift-correction baseline tilt, since correctdrift anchors on the raw
        signal regardless of exclusion — the notice must keep warning, with different
        wording (not re-suggesting an exclusion already done)."""
        src, out = _dirs(tmp_path)
        _write_recording(src / "cut_start.csv", lead_cut_s=1.0)
        result = pipeline.run_batch(_settings(src, out, exclude=("cut_start.csv", [1])))
        fr = result.ok_files["cut_start.csv"]
        assert any("already excluded" in n for n in fr.notices)
        assert not any("exclude the first breath" in n for n in fr.notices)

    @requires_synth()
    def test_committed_golden_synthetic_input_has_no_false_positive(self, tmp_path):
        """The synthetic recordings under tests/golden/input/ are known-good, correctly
        cut inputs the golden tests themselves rely on — they must not trip the new
        check (regression test for the sign-based first attempt's false positive)."""
        out = tmp_path / "out"
        out.mkdir()
        s = synth_settings(out)
        result = pipeline.run_batch(s)
        assert result.ok_files
        for fname, fr in result.ok_files.items():
            assert fr.notices == [], f"{fname}: unexpected boundary notice {fr.notices}"

    def test_builtin_sample_recording_has_no_false_positive(self, tmp_path):
        """The built-in 'Explore sample data' recording is documentation's own worked
        example of a correctly-cut file; it must not trip the new check either
        (regression test for the sign-based first attempt's false positive). Its own
        last breath's expiration measures ~88% of the file's median (the synthetic
        generator varies each breath's period by design) -- the tightest known margin
        against the 80% threshold, which is exactly why 80% and not something higher
        was chosen."""
        from respmech.core.sample import write_sample_recording, build_sample_settings
        out = tmp_path / "out"
        out.mkdir()
        desc = write_sample_recording(str(tmp_path))
        s = build_sample_settings(desc, str(out))
        result = pipeline.run_batch(s)
        assert result.ok_files
        for fname, fr in result.ok_files.items():
            assert fr.notices == [], f"{fname}: unexpected boundary notice {fr.notices}"


class TestTrimBoundaryNoticesPreview:
    """The Preview & QC live-tuning path (``workers.stage_mechanics_preview``), which
    a user sees BEFORE ever running a batch."""

    def test_correctly_cut_file_has_no_boundary_notice_in_preview(self, tmp_path):
        from respmech.ui.workers import stage_mechanics_preview
        src, out = _dirs(tmp_path)
        path = src / "clean.csv"
        _write_recording(path)
        data = stage_mechanics_preview(_settings(src, out), str(path))
        assert data.get("boundary_notices") == []

    def test_cut_mid_inspiration_flagged_in_preview(self, tmp_path):
        from respmech.ui.workers import stage_mechanics_preview
        src, out = _dirs(tmp_path)
        path = src / "cut_start.csv"
        _write_recording(path, lead_cut_s=1.0)
        data = stage_mechanics_preview(_settings(src, out), str(path))
        assert any("inspiration" in n for n in data.get("boundary_notices", []))


class TestShortBoundaryNote:
    """The compressed status-bar text (`_mechanics._short_boundary_note`), which
    exists because the full sentence from `trim_boundary_notices` can run past 300
    characters and the Qt status bar it is shown in does not wrap."""

    def test_empty_input_gives_empty_string(self):
        from respmech.ui.screens.preview._mechanics import _short_boundary_note
        assert _short_boundary_note([]) == ""

    def test_first_breath_notice_is_shortened_and_names_the_edge(self):
        from respmech.ui.screens.preview._mechanics import _short_boundary_note
        full = ("the first breath's inspiration (0.99 s) is much shorter than this "
                "file's typical inspiration (2.00 s) — the recording likely begins "
                "mid-inspiration, so the first breath is truncated and analysed as if "
                "it were complete. With drift correction on, this also tilts the "
                "volume baseline of every breath in the file, not just this one. "
                "Re-export the epoch so it starts in expiration, or exclude the "
                "first breath in Preview & QC.")
        short = _short_boundary_note([full])
        assert "first" in short
        assert len(short) < len(full)
        assert len(short) < 100

    def test_already_excluded_notice_gets_its_own_short_form(self):
        from respmech.ui.screens.preview._mechanics import _short_boundary_note
        full = ("the last breath's expiration (0.50 s) is much shorter than this "
                "file's typical expiration (1.60 s) — it is already excluded from "
                "the analysis, but drift correction anchors on the recording's raw "
                "first and last sample regardless of which breaths are excluded, so "
                "the volume baseline of the OTHER breaths in this file may still be "
                "tilted. Re-export the epoch so it ends in inspiration to fix this "
                "at the source.")
        short = _short_boundary_note([full])
        assert "excluded" in short
        assert "last" in short
        assert len(short) < 100

    def test_both_notices_shortened_together(self):
        from respmech.ui.screens.preview._mechanics import _short_boundary_note
        short = _short_boundary_note([
            "the first breath's inspiration (0.5 s) is much shorter ...",
            "the last breath's expiration (0.5 s) is much shorter ...",
        ])
        assert "first" in short and "last" in short


class TestBoundaryNoticeSettings:
    """Ticket 20260906-1307: the threshold/min-other-breaths pair is now a per-analysis
    Settings/TOML field, not a hardcoded default -- so a study with atypically high
    natural breath-to-breath variability can raise the threshold itself instead of
    living with the system-wide default's false-positive rate."""

    def test_settings_default_matches_previous_hardcoded_default(self):
        s = Settings()
        assert s.processing.segmentation.boundary_notice_min_relative_duration == 0.8
        assert s.processing.segmentation.boundary_notice_min_other_breaths == 3

    def test_legacy_ns_carries_the_settings_value(self):
        s = Settings()
        s.processing.segmentation.boundary_notice_min_relative_duration = 0.65
        s.processing.segmentation.boundary_notice_min_other_breaths = 4
        legacy = to_legacy_ns(s)
        assert legacy.processing.mechanics.boundarynoticeminrelativeduration == 0.65
        assert legacy.processing.mechanics.boundarynoticeminotherbreaths == 4

    def test_lower_settings_threshold_stops_flagging_a_ratio_the_default_would_flag(self):
        # 75% is flagged under the 0.8 default (test_a_ratio_just_under_the_threshold_
        # is_flagged pins 79%) but must NOT be flagged once a study lowers its own
        # threshold to 0.7 via Settings -- proving the override reaches the function
        # via `settings`, not just via the explicit kwarg.
        breaths = {i: _breath(100, 150) for i in range(1, 7)}
        breaths[1] = _breath(75, 150)
        s = Settings()
        s.input.format.sampling_frequency = FS
        legacy_default = to_legacy_ns(s)
        assert len(compute.trim_boundary_notices(breaths, legacy_default)) == 1

        s.processing.segmentation.boundary_notice_min_relative_duration = 0.7
        legacy_lowered = to_legacy_ns(s)
        assert compute.trim_boundary_notices(breaths, legacy_lowered) == []

    def test_raising_settings_min_other_breaths_can_suppress_a_notice(self):
        # 4 breaths (3 others) fires under the default floor of 3 (see
        # test_too_few_other_breaths_skips_the_check); raising the floor to 4 via
        # Settings must suppress it without touching the ratio at all.
        breaths = {i: _breath(100, 150) for i in range(1, 5)}
        breaths[1] = _breath(10, 150)
        s = Settings()
        s.input.format.sampling_frequency = FS
        s.processing.segmentation.boundary_notice_min_other_breaths = 4
        legacy = to_legacy_ns(s)
        assert compute.trim_boundary_notices(breaths, legacy) == []

    def test_explicit_kwarg_still_overrides_settings(self):
        # backward-compatible escape hatch: a caller (or a future test) that passes the
        # kwarg explicitly is not overridden by whatever Settings happens to carry.
        breaths = {i: _breath(100, 150) for i in range(1, 7)}
        breaths[1] = _breath(75, 150)
        settings = _dummy_settings()  # ships the 0.8 default
        assert compute.trim_boundary_notices(
            breaths, settings, min_relative_duration=0.5) == []

    @pytest.mark.parametrize("bad_value", [0.0, -0.1, 1.5])
    def test_validate_rejects_an_out_of_range_ratio(self, bad_value):
        s = Settings()
        s.input.format.sampling_frequency = FS
        s.input.channels.flow = 5
        s.input.channels.poes = 7
        s.input.channels.pgas = 8
        s.input.channels.pdi = 9
        s.input.channels.volume = 6
        s.processing.segmentation.boundary_notice_min_relative_duration = bad_value
        with pytest.raises(Exception):
            s.validate()

    def test_validate_rejects_a_non_positive_min_other_breaths(self):
        s = Settings()
        s.input.format.sampling_frequency = FS
        s.input.channels.flow = 5
        s.input.channels.poes = 7
        s.input.channels.pgas = 8
        s.input.channels.pdi = 9
        s.input.channels.volume = 6
        s.processing.segmentation.boundary_notice_min_other_breaths = 0
        with pytest.raises(Exception):
            s.validate()
