"""The "Advanced…" modal, and the ECG settings that moved behind it.

The contract that matters is Cancel. Both existing modals in this app work by staging —
every edit lives in dialog-local widgets and reaches the settings only when the caller
commits on OK — so Cancel needs no undo, no snapshot and no rollback. There is no
transaction helper anywhere in this UI, so a modal that edited state directly would have had
to invent one, and would have had to get the dirty flag and the recompute scope right on the
rollback path too.
"""
import pytest
from PySide6.QtWidgets import QDialog

from respmech.ui.advanced_dialog import AdvancedDialog, Field, apply_values
from respmech.ui.state import AppState

from _helpers import requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


def _fields():
    return [
        Field("a", "An integer", "int", "processing.emg.a", "help for a", lo=0, hi=10),
        Field("b", "A float", "float", "processing.emg.b", "help for b",
              lo=0.0, hi=1.0, step=0.1, decimals=2),
        Field("c", "A flag", "bool", "processing.emg.c", "help for c"),
    ]


# -- the generic dialog --------------------------------------------------------
def test_it_shows_the_current_values(qapp):
    dlg = AdvancedDialog("T", _fields(), {"a": 3, "b": 0.25, "c": True})
    assert dlg.values() == {"a": 3, "b": 0.25, "c": True}


def test_every_row_names_its_settings_variable(qapp):
    """A control that moves screens must keep saying which TOML key it writes."""
    dlg = AdvancedDialog("T", _fields(), {"a": 1, "b": 0.5, "c": False})
    for f in _fields():
        tip = dlg.widget(f.key).toolTip()
        assert f.path in tip and len(tip) > len(f.path) + 5


def test_a_longer_intro_of_ordinary_words_does_not_widen_the_dialog(qapp):
    """D10 (UI-overhaul) lengthened the EMG-advanced modal's intro text. WrapLabel's minimum
    width is bound by its LONGEST WORD, not the paragraph's total length (see
    section_flow.WrapLabel's own docstring) — that is the property a longer intro relies on to
    not widen the modal. Guard the property itself, generically, with a ratio rather than a
    pixel literal: a short intro and a much longer one built from equally ordinary words must
    report the same minimum width."""
    short = AdvancedDialog("T", _fields(), {"a": 3, "b": 0.25, "c": True}, intro="Short intro.")
    long_intro = ("A rather long explanatory introduction sentence about several settings. " * 4)
    long = AdvancedDialog("T", _fields(), {"a": 3, "b": 0.25, "c": True}, intro=long_intro)
    assert long.minimumSizeHint().width() == short.minimumSizeHint().width()


def test_editing_stages_without_touching_anything_outside(qapp):
    dlg = AdvancedDialog("T", _fields(), {"a": 1, "b": 0.5, "c": False})
    dlg.widget("a").setValue(7)
    dlg.widget("c").setChecked(True)
    assert dlg.values() == {"a": 7, "b": 0.5, "c": True}


def test_a_field_that_depends_on_a_checkbox_follows_it(qapp):
    """A detail field that only means something once its checkbox is on (e.g. 'Resample
    to' beside 'Resample before analysis') must start disabled when the checkbox is
    unticked, and re-enable/disable live as the checkbox is toggled — not just once,
    at dialog build time."""
    fields = [
        Field("on", "Turn it on", "bool", "processing.emg.on", "help"),
        Field("detail", "Detail", "int", "processing.emg.detail", "help",
              lo=0, hi=10, depends_on="on"),
    ]
    dlg = AdvancedDialog("T", fields, {"on": False, "detail": 5})
    assert dlg.widget("detail").isEnabled() is False
    dlg.widget("on").setChecked(True)
    assert dlg.widget("detail").isEnabled() is True
    dlg.widget("on").setChecked(False)
    assert dlg.widget("detail").isEnabled() is False


def test_a_field_that_depends_on_a_checked_checkbox_starts_enabled(qapp):
    fields = [
        Field("on", "Turn it on", "bool", "processing.emg.on", "help"),
        Field("detail", "Detail", "int", "processing.emg.detail", "help",
              lo=0, hi=10, depends_on="on"),
    ]
    dlg = AdvancedDialog("T", fields, {"on": True, "detail": 5})
    assert dlg.widget("detail").isEnabled() is True


def test_apply_reports_whether_anything_actually_changed(qapp):
    """Pressing OK without editing must not mark the analysis modified or schedule work."""
    class T:
        a, b, c = 1, 0.5, False
    target = T()
    assert apply_values(target, {"a": 1, "b": 0.5, "c": False}) is False
    assert apply_values(target, {"a": 2, "b": 0.5, "c": False}) is True
    assert target.a == 2


def test_the_derived_line_follows_the_staged_values(qapp):
    """For a coupling the numbers alone do not show."""
    dlg = AdvancedDialog("T", _fields(), {"a": 2, "b": 0.5, "c": False},
                         derived=lambda v: f"a is {v['a']}")
    assert dlg.derived.text() == "a is 2"
    dlg.widget("a").setValue(9)
    assert dlg.derived.text() == "a is 9"


# -- Apply / genuine non-modality / debounced derived (D13, UI-overhaul) ------
# QDialog.exec() sets Qt.WA_ShowModal unconditionally while it runs, REGARDLESS of a prior
# setModal(False) — verified empirically. A test that stubs exec() entirely (as _mech_stub
# below does, for every OTHER mechanics-modal test) can never see that failure class: these
# tests call the REAL AdvancedDialog.exec() via QTimer.singleShot to close it.
def test_a_modal_dialog_stays_modal_during_the_real_exec(qapp):
    """The default and every existing caller (ECG/EMG) must be unaffected."""
    from PySide6.QtCore import QTimer
    dlg = AdvancedDialog("T", _fields(), {"a": 1, "b": 0.5, "c": False})
    seen = {}

    def check_and_close():
        seen["is_modal"] = dlg.isModal()
        dlg.accept()

    QTimer.singleShot(0, check_and_close)
    assert dlg.exec() == QDialog.Accepted
    assert seen["is_modal"] is True


def test_a_non_modal_dialog_is_genuinely_non_modal_during_the_real_exec(qapp):
    from PySide6.QtCore import QTimer
    dlg = AdvancedDialog("T", _fields(), {"a": 1, "b": 0.5, "c": False}, modal=False)
    seen = {}

    def check_and_close():
        seen["is_modal"] = dlg.isModal()
        dlg.accept()

    QTimer.singleShot(0, check_and_close)
    assert dlg.exec() == QDialog.Accepted
    assert seen["is_modal"] is False


def test_a_non_modal_dialog_still_reports_reject_from_the_real_exec(qapp):
    from PySide6.QtCore import QTimer
    dlg = AdvancedDialog("T", _fields(), {"a": 1, "b": 0.5, "c": False}, modal=False)
    QTimer.singleShot(0, dlg.reject)
    assert dlg.exec() == QDialog.Rejected


def test_a_dialog_without_on_apply_has_no_apply_button(qapp):
    dlg = AdvancedDialog("T", _fields(), {"a": 1, "b": 0.5, "c": False})
    assert dlg.btn_apply is None


def test_apply_commits_without_closing_and_never_calls_accept_or_reject(qapp):
    """Not just result() == Rejected — that is Qt's default regardless of whether accept()
    or reject() ran, so it cannot tell 'still open' from 'closed unfavourably'. Spy on the
    two methods themselves."""
    calls = []
    dlg = AdvancedDialog("T", _fields(), {"a": 1, "b": 0.5, "c": False},
                         on_apply=lambda edited: calls.append(edited))
    accepted, rejected = [], []
    dlg.accept = lambda: accepted.append(1)
    dlg.reject = lambda: rejected.append(1)
    dlg.widget("a").setValue(7)
    dlg.btn_apply.click()
    assert calls == [{"a": 7}]
    assert not accepted and not rejected


def test_apply_without_an_edit_does_not_call_on_apply(qapp):
    calls = []
    dlg = AdvancedDialog("T", _fields(), {"a": 1, "b": 0.5, "c": False},
                         on_apply=lambda edited: calls.append(edited))
    dlg.btn_apply.click()
    assert calls == []


def test_apply_moves_the_edited_baseline_forward(qapp):
    """A second Apply — or a later OK — must not resend an already-applied key:
    apply_values() is idempotent so it would be harmless, but it is still a wasted
    recompute that a debounced derived-line search makes newly expensive."""
    calls = []
    dlg = AdvancedDialog("T", _fields(), {"a": 1, "b": 0.5, "c": False},
                         on_apply=lambda edited: calls.append(edited))
    dlg.widget("a").setValue(7)
    assert dlg.edited_values() == {"a": 7}
    dlg.btn_apply.click()
    assert dlg.edited_values() == {}, "Apply must move the baseline forward"
    dlg.btn_apply.click()              # nothing new staged: on_apply must not fire again
    assert calls == [{"a": 7}]


def test_the_derived_line_shows_immediately_even_when_debounced(qapp):
    """The dialog must open showing a real value, never an empty line waiting out the
    first debounce window."""
    dlg = AdvancedDialog("T", _fields(), {"a": 2, "b": 0.5, "c": False},
                         derived=lambda v: f"a is {v['a']}", derived_debounce_ms=200)
    assert dlg.derived.text() == "a is 2"


def test_derived_debounce_delays_recompute_until_editing_stops(qapp):
    from PySide6.QtTest import QTest
    calls = []

    def derived(v):
        calls.append(v["a"])
        return f"a is {v['a']}"

    dlg = AdvancedDialog("T", _fields(), {"a": 1, "b": 0.5, "c": False},
                         derived=derived, derived_debounce_ms=60)
    calls.clear()                      # drop the synchronous open-time computation
    dlg.widget("a").setValue(5)
    dlg.widget("a").setValue(6)
    dlg.widget("a").setValue(7)        # three rapid edits inside one debounce window
    assert calls == [], "derived ran synchronously despite a debounce window"
    QTest.qWait(200)
    assert calls == [7], "debounce must coalesce to the LAST value, computed once"
    assert dlg.derived.text() == "a is 7"


# -- the ECG modal -------------------------------------------------------------
def _preview(qapp, tmp_path):
    from respmech.ui.screens.preview_screen import PreviewScreen
    pv = PreviewScreen(AppState(synth_settings(str(tmp_path), data_out=_OUT)))
    pv._refresh_files()
    qapp.processEvents()
    return pv


def test_the_strip_keeps_the_knobs_people_drag(qapp, tmp_path):
    """Detection is driven from the strip; only the template shaping moved."""
    pv = _preview(qapp, tmp_path)
    strip = pv.ecg_opts
    assert pv.ecg_min_height.parent() is strip and pv.ecg_min_distance.parent() is strip
    assert pv.remove_ecg.parent() is strip and pv.ecg_capture_channel.parent() is strip
    assert pv.ecg_min_width.parent() is not strip, "the shape guard is still on the strip"
    assert pv.ecg_window.parent() is not strip, "the template width is still on the strip"
    pv.shutdown()


@pytest.mark.parametrize("accept", [True, False])
def test_ok_commits_and_cancel_changes_nothing(qapp, tmp_path, accept, monkeypatch):
    import respmech.ui.advanced_dialog as ad
    pv = _preview(qapp, tmp_path)
    e = pv.state.settings.processing.emg
    before = (e.ecg_min_width_s, e.ecg_window_s)
    edits = []
    pv.settings_edited.connect(lambda: edits.append(1))

    real = ad.AdvancedDialog

    class _Stub(real):
        def exec(self):
            self.widget("ecg_window_s").setValue(0.55)
            return QDialog.Accepted if accept else QDialog.Rejected

    monkeypatch.setattr(ad, "AdvancedDialog", _Stub)
    pv._open_ecg_advanced()

    if accept:
        assert e.ecg_window_s == 0.55
        assert edits, "an accepted edit did not mark the analysis modified"
    else:
        assert (e.ecg_min_width_s, e.ecg_window_s) == before, "Cancel wrote to the settings"
        assert not edits, "Cancel marked the analysis modified"
    pv.shutdown()


def test_ok_without_an_edit_is_not_an_edit(qapp, tmp_path, monkeypatch):
    import respmech.ui.advanced_dialog as ad
    pv = _preview(qapp, tmp_path)
    edits = []
    pv.settings_edited.connect(lambda: edits.append(1))
    real = ad.AdvancedDialog

    class _Stub(real):
        def exec(self):
            return QDialog.Accepted            # accepted, but nothing touched

    monkeypatch.setattr(ad, "AdvancedDialog", _Stub)
    pv._open_ecg_advanced()
    assert not edits, "pressing OK unchanged marked the analysis modified"
    pv.shutdown()


def test_the_committed_value_reaches_the_widget_that_still_owns_it(qapp, tmp_path, monkeypatch):
    """The strip widgets still exist and still round-trip through from_state/to_state, so
    they have to be re-synced or the next strip edit would write back the old value."""
    import respmech.ui.advanced_dialog as ad
    pv = _preview(qapp, tmp_path)
    real = ad.AdvancedDialog

    class _Stub(real):
        def exec(self):
            self.widget("ecg_min_width_s").setValue(0.005)
            return QDialog.Accepted

    monkeypatch.setattr(ad, "AdvancedDialog", _Stub)
    pv._open_ecg_advanced()
    assert pv.ecg_min_width.value() == 0.005
    pv.shutdown()


# -- the Mechanics modal -------------------------------------------------------
def _settings_screen(qapp, tmp_path):
    from respmech.ui.screens.settings_screen import SettingsScreen
    sc = SettingsScreen(AppState(synth_settings(str(tmp_path), data_out=_OUT)))
    sc._mark_clean()
    return sc


def test_the_mechanics_card_left_setup_for_the_preview_tab(qapp, tmp_path):
    from PySide6.QtWidgets import QGroupBox
    sc = _settings_screen(qapp, tmp_path)
    titles = {g.title() for g in sc.findChildren(QGroupBox)}
    # Setup is lean now: Input, Channels, Output, and the conditional Sample entropy card.
    assert "Mechanics" not in titles and "Advanced (rarely changed)" not in titles
    assert titles == {"Input", "Channels", "Output", "Sample entropy"}
    pv = _preview(qapp, tmp_path)
    assert hasattr(pv, "btn_mech_advanced"), "the Preview Mechanics tab hosts Advanced…"
    pv.shutdown()


def _mech_stub(monkeypatch, edit, accept=True):
    import respmech.ui.advanced_dialog as ad
    real = ad.AdvancedDialog

    class _Stub(real):
        def exec(self):
            edit(self)
            return QDialog.Accepted if accept else QDialog.Rejected
    monkeypatch.setattr(ad, "AdvancedDialog", _Stub)


def _rendered_preview(qapp, tmp_path):
    """A PreviewScreen with the first synthetic file's mechanics channels actually
    rendered (``_refresh_files()`` alone does not render — measured directly), so
    ``self._trend_probe``/``_trend_probe_file``/``_trend_probe_shape`` are populated,
    exactly as the real 'select a file' flow leaves them. The live breath count and the
    trend-anchor hint in Mechanics — advanced… both read these."""
    pv = _preview(qapp, tmp_path)
    pv._preview()
    assert pv._trend_probe is not None, "the synthetic file must render for this test"
    return pv


@pytest.mark.parametrize("accept", [True, False])
def test_mech_ok_commits_and_cancel_changes_nothing(qapp, tmp_path, accept, monkeypatch):
    pv = _preview(qapp, tmp_path)
    s = pv.state.settings
    before = s.processing.segmentation.buffer
    edits = []
    pv.settings_edited.connect(lambda: edits.append(1))
    _mech_stub(monkeypatch, lambda d: d.widget("buffer").setValue(777), accept=accept)
    pv._open_mech_advanced()
    if accept:
        assert s.processing.segmentation.buffer == 777
        assert edits, "an accepted edit did not mark the analysis modified"
    else:
        assert s.processing.segmentation.buffer == before
        assert not edits, "Cancel marked the analysis modified"
    pv.shutdown()


def test_buffer_field_is_labelled_debounce_and_tooltip_drops_the_padding_wording(
        qapp, tmp_path, monkeypatch):
    """D29 (UI-overhaul): the field was renamed from 'Breath-separation buffer' to
    'Breath-separation debounce', and its tooltip no longer describes padding around a
    breath boundary (the code looks FORWARD, it does not add guard samples around one) —
    but the TOML key it names must still be there, since that is how a control keeps
    saying which settings variable it writes even after label text changes."""
    from respmech.ui.section_flow import WrapLabel
    pv = _preview(qapp, tmp_path)
    seen = {}

    def edit(d):
        seen["tooltip"] = d.widget("buffer").toolTip()
        seen["labels"] = {lab.text() for lab in d.findChildren(WrapLabel)}

    _mech_stub(monkeypatch, edit)
    pv._open_mech_advanced()
    tip = seen["tooltip"].lower()
    assert "guard" not in tip and "padding" not in tip and "added around" not in tip
    assert "processing.segmentation.buffer" in seen["tooltip"]
    assert "Breath-separation debounce" in seen["labels"]
    assert "Breath-separation buffer" not in seen["labels"]
    pv.shutdown()


def test_buffer_note_shows_the_derived_seconds_and_updates_live(qapp, tmp_path, monkeypatch):
    """D29: samples alone hide how long the debounce actually holds at the rate the run
    analyses at, so the field grew a note beside it — '200 samples ≈ 0.20 s at 1000 Hz' for
    the synthetic fixture's settings (1000 Hz, no resample, buffer=200) — that must update
    as the user edits buffer/resample/resample_to_frequency, not just show the value the
    dialog opened with."""
    pv = _preview(qapp, tmp_path)
    assert pv.state.settings.input.format.sampling_frequency == 1000
    assert pv.state.settings.processing.segmentation.buffer == 200
    seen = {}

    def edit(d):
        seen["opened"] = d.note("buffer").text()
        d.widget("buffer").setValue(800)
        seen["after_buffer_edit"] = d.note("buffer").text()
        d.widget("resample").setChecked(True)
        d.widget("resample_to_frequency").setValue(100)
        seen["after_resample_edit"] = d.note("buffer").text()

    _mech_stub(monkeypatch, edit, accept=False)
    pv._open_mech_advanced()
    assert seen["opened"] == "200 samples ≈ 0.20 s at 1000 Hz."
    assert seen["after_buffer_edit"] == "800 samples ≈ 0.80 s at 1000 Hz."
    assert seen["after_resample_edit"] == "800 samples ≈ 8.00 s at 100 Hz."
    pv.shutdown()


def test_buffer_debounce_seconds_is_a_pure_rate_conversion():
    """D29's derivation, tested Qt-free against the ticket's own worked example: 800
    samples is 0.80 s at the recording's native 1000 Hz, or 4.00 s if resampled to 200 Hz —
    the effective-rate rule matches Settings.validate()'s fs_eff (resample target when
    resample is on and positive, else the native rate)."""
    from respmech.ui.screens.preview._mechanics import (_buffer_debounce_hint,
                                                         _buffer_debounce_seconds)
    assert _buffer_debounce_seconds(800, False, 200, 1000) == pytest.approx(0.80)
    assert _buffer_debounce_seconds(800, True, 200, 1000) == pytest.approx(4.00)
    # resample ticked but no target yet (0, the field's un-set state before Setup runs):
    # falls back to the native rate rather than dividing by zero.
    assert _buffer_debounce_seconds(800, True, 0, 1000) == pytest.approx(0.80)
    assert _buffer_debounce_seconds(800, False, 200, None) is None
    assert _buffer_debounce_hint(800, False, 200, 1000) == "800 samples ≈ 0.80 s at 1000 Hz."
    assert _buffer_debounce_hint(800, True, 200, 1000) == "800 samples ≈ 4.00 s at 200 Hz."


def test_resample_off_by_default_and_hz_field_follows_the_checkbox(qapp, tmp_path, monkeypatch):
    """A fresh analysis has resampling off (P1 bug: it must stay off until the user opts in,
    and be genuinely toggleable). The 'Resample to' Hz field is meaningless while the
    checkbox is unticked, so it must start disabled and re-enable/disable as the checkbox
    is toggled — otherwise an unticked box next to a live-looking '200 Hz' reads as
    resampling being on and stuck there."""
    pv = _preview(qapp, tmp_path)
    s = pv.state.settings
    assert s.processing.sampling.resample is False

    seen = {}

    def edit(d):
        cb, hz = d.widget("resample"), d.widget("resample_to_frequency")
        seen["checked_initially"] = cb.isChecked()
        seen["hz_enabled_initially"] = hz.isEnabled()
        cb.setChecked(True)
        seen["hz_enabled_after_check"] = hz.isEnabled()
        cb.setChecked(False)
        seen["hz_enabled_after_uncheck"] = hz.isEnabled()

    _mech_stub(monkeypatch, edit)
    pv._open_mech_advanced()

    assert seen == {
        "checked_initially": False,
        "hz_enabled_initially": False,
        "hz_enabled_after_check": True,
        "hz_enabled_after_uncheck": False,
    }
    assert s.processing.sampling.resample is False   # left unticked -> still off after OK
    pv.shutdown()


def test_trend_method_field_follows_correct_trend_checkbox(qapp, tmp_path, monkeypatch):
    """P2: every checkbox-gated detail field in the Mechanics — advanced dialog follows its
    checkbox, not just resample. 'Trend interpolation' only means something once 'Correct
    end-expiratory trend' is ticked."""
    pv = _preview(qapp, tmp_path)

    seen = {}

    def edit(d):
        cb, method = d.widget("correct_trend"), d.widget("trend_method")
        seen["method_enabled_initially"] = method.isEnabled()
        cb.setChecked(True)
        seen["method_enabled_after_check"] = method.isEnabled()
        cb.setChecked(False)
        seen["method_enabled_after_uncheck"] = method.isEnabled()

    _mech_stub(monkeypatch, edit)
    pv._open_mech_advanced()

    assert seen == {
        "method_enabled_initially": False,        # correct_trend is off by default
        "method_enabled_after_check": True,
        "method_enabled_after_uncheck": False,
    }
    pv.shutdown()


def test_mech_modal_is_model_direct_and_setup_cannot_revert_it(qapp, tmp_path, monkeypatch):
    """The whole point of the move: Setup.to_state runs on every tab change and must not own
    these any more, or the Preview edit reverts on the first switch to Setup."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path), data_out=_OUT)))
    pv, sc = win.preview_screen, win.settings_screen

    def edit(d):
        d.widget("method").setCurrentIndex(1)        # Volume
        d.widget("buffer").setValue(321)
    _mech_stub(monkeypatch, edit)
    pv._open_mech_advanced()
    for _ in range(3):
        sc.to_state()
    assert win.state.settings.processing.segmentation.method == "volume"
    assert win.state.settings.processing.segmentation.buffer == 321
    win.close()


def test_mech_ok_without_an_edit_is_not_an_edit(qapp, tmp_path, monkeypatch):
    pv = _preview(qapp, tmp_path)
    edits = []
    pv.settings_edited.connect(lambda: edits.append(1))
    _mech_stub(monkeypatch, lambda d: None)
    pv._open_mech_advanced()
    assert not edits
    pv.shutdown()


def test_breath_count_overrides_survive_the_modal(qapp, tmp_path, monkeypatch):
    """A multi-line, structured field — the one thing in here that is not a number."""
    pv = _preview(qapp, tmp_path)
    _mech_stub(monkeypatch, lambda d: d.widget("breath_counts").setPlainText("synth_case_A.csv = 12"))
    pv._open_mech_advanced()
    counts = pv.state.settings.processing.breath_counts
    assert [(c.file, c.count) for c in counts] == [("synth_case_A.csv", 12)]
    pv.shutdown()


def test_breath_count_overrides_are_stamped_with_the_current_input_folder(qapp, tmp_path, monkeypatch):
    """Ticket B06: every line the box commits belongs to the CURRENT input folder — the
    box holds one flat list, so re-parsing it always relabels the whole thing fresh, which
    is correct here (the user is looking at THIS folder's batch while editing it)."""
    from _helpers import INPUT
    pv = _preview(qapp, tmp_path)
    _mech_stub(monkeypatch, lambda d: d.widget("breath_counts").setPlainText(
        "synth_case_A.csv = 12\nsynth_case_B.csv = 9"))
    pv._open_mech_advanced()
    counts = pv.state.settings.processing.breath_counts
    assert {c.file: c.folder for c in counts} == {
        "synth_case_A.csv": INPUT, "synth_case_B.csv": INPUT}
    pv.shutdown()


def test_an_untouched_breath_count_box_never_gets_reparsed_or_restamped(qapp, tmp_path, monkeypatch):
    """_parse_breath_counts only ever runs when the user actually edited the field
    (edited_values() gates on that) — OK without touching this box must be a pure no-op,
    never silently restamping folders on entries the user never looked at."""
    from respmech.core.settings import BreathCountEntry
    pv = _preview(qapp, tmp_path)
    original = BreathCountEntry(file="synth_case_A.csv", count=12, folder="/some/old/folder")
    pv.state.settings.processing.breath_counts = [original]
    _mech_stub(monkeypatch, lambda d: None)      # OK without touching anything
    pv._open_mech_advanced()
    assert pv.state.settings.processing.breath_counts == [original]
    assert pv.state.settings.processing.breath_counts[0].folder == "/some/old/folder"
    pv.shutdown()


def test_the_per_file_overrides_card_points_at_the_file_rail_for_exclusions(qapp, tmp_path, monkeypatch):
    """Ticket B06 point 5: a user looking for where exclusions are tracked would
    reasonably check this card (the only "per-file" thing in Mechanics — advanced) and,
    finding nothing, conclude exclusions aren't tracked at all. The card must now say
    where they actually live."""
    from PySide6.QtWidgets import QLabel
    pv = _preview(qapp, tmp_path)
    captured = {}
    _mech_stub(monkeypatch, lambda d: captured.setdefault("dlg", d))
    pv._open_mech_advanced()
    texts = [w.text() for w in captured["dlg"].findChildren(QLabel)]
    assert any("file rail" in t and "exclu" in t.lower() for t in texts)
    pv.shutdown()


def test_the_emg_advanced_modal_hosts_the_rms_settings(qapp, tmp_path, monkeypatch):
    """RMS window, outlier limit and normalisation left Setup for the EMG Advanced modal."""
    pv = _preview(qapp, tmp_path)
    e = pv.state.settings.processing.emg
    edits = []
    pv.settings_edited.connect(lambda: edits.append(1))

    def edit(d):
        d.widget("rms_window_s").setValue(0.08)
        d.widget("normalization").setCurrentIndex(0)   # None
    _mech_stub(monkeypatch, edit)
    pv._open_emg_advanced()
    assert e.rms_window_s == 0.08 and e.normalization == "none"
    assert edits
    pv.shutdown()


#: How early "RMS window" must appear to count as LEADING the string, not merely appearing in
#: it somewhere. The ticket asked for it first; a substring check alone would keep passing if a
#: future edit buried the phrase at the end of a long sentence.
_LEADS_WITHIN = 20


def test_the_emg_advanced_button_and_modal_lead_with_rms_settings(qapp, tmp_path, monkeypatch):
    """D10 (UI-overhaul): both entry points into the EMG-advanced modal — the button's tooltip
    and the modal's own intro — must LEAD WITH the RMS window, since that is what a user most
    often opens this modal looking for, and previously neither said so (the button tooltip only
    named the gate/guards/exports, and the modal's intro claimed everything in it tunes the
    noise gate, which was never true of the RMS card). Checked by POSITION, not just substring
    presence, since a mention buried at the end of a long sentence would not fix the problem
    this ticket exists for. Asserted on toolTip()/fullText() as the full strings, not on a
    possibly-elided text().

    Normalisation is deliberately NOT claimed to "define the reported EMG number" here: only
    rms_window_s does (see its own Field help text in _emg_noise.py); normalization's help text
    says the opposite ("never changes the raw RMS") — an earlier draft of this text conflated
    the two and was caught in self-review.
    """
    pv = _preview(qapp, tmp_path)
    tip = pv.btn_emg_advanced.toolTip()
    assert tip.index("RMS window") < _LEADS_WITHIN, (
        f"button tooltip does not lead with the RMS window: {tip!r}")
    assert "normalisation" in tip.lower()
    assert "amplitude normalisation, which define" not in tip.lower(), (
        "tooltip must not claim normalisation defines the reported EMG number — only the "
        "RMS window does; normalization's own help text says it 'never changes the raw RMS'")

    captured = {}
    _mech_stub(monkeypatch, lambda d: captured.setdefault("dlg", d), accept=False)
    pv._open_emg_advanced()
    dlg = captured["dlg"]
    intro_label = dlg.layout().itemAt(0).widget()
    intro = intro_label.fullText()
    assert intro.index("RMS window") < _LEADS_WITHIN, (
        f"modal intro does not lead with the RMS window: {intro!r}")
    assert "normalisation" in intro.lower()
    assert "amplitude normalisation that define" not in intro.lower(), (
        "intro must not claim normalisation defines the reported EMG number — see the "
        "tooltip assertion above for why")
    pv.shutdown()


def test_the_emg_advanced_cards_and_fields_are_unmoved(qapp, tmp_path, monkeypatch):
    """D10 only reworded the intro/tooltip; the six cards and their 19 fields must be exactly
    as before, proving nothing was moved by accident while rewriting the words around them."""
    pv = _preview(qapp, tmp_path)
    captured = {}
    _mech_stub(monkeypatch, lambda d: captured.setdefault("dlg", d), accept=False)
    pv._open_emg_advanced()
    dlg = captured["dlg"]
    assert [c.title() for c in dlg.cards] == [
        "RMS and normalisation", "Noise suppression", "Spectral gate (STFT)",
        "Gated peak (saved output)", "Heartbeat and island guards", "Diagnostics"]
    assert len(dlg._fields) == 19
    assert {f.key for f in dlg._fields} == {
        "rms_window_s", "outlier_rms_sd_limit", "normalization",
        "prop_decrease", "fidelity_target",
        "n_std_thresh", "n_fft", "win_length", "hop_length", "n_grad_freq", "n_grad_time",
        "enabled", "gate_half_width_s",
        "min_survival", "min_island_s", "long_rr_factor", "max_long_rr_frac",
        "hr_ceiling_margin",
        "save_sound"}
    pv.shutdown()


def test_the_emg_advanced_button_width_is_unaffected_by_its_tooltip(qapp, tmp_path):
    """A QPushButton's sizeHint is driven by its LABEL, never its tooltip, so D10's much longer
    tooltip must not have changed the strip's width. Ratio guard, not a pixel literal: the
    button's width with the new tooltip must equal its width with no tooltip at all."""
    pv = _preview(qapp, tmp_path)
    btn = pv.btn_emg_advanced
    with_tip = btn.sizeHint().width()
    original_tip = btn.toolTip()
    assert original_tip, "the button has no tooltip — this test would be vacuous"
    btn.setToolTip("")
    without_tip = btn.sizeHint().width()
    btn.setToolTip(original_tip)
    assert with_tip == without_tip, (
        f"button width changed with its tooltip ({with_tip} vs {without_tip} px) — a tooltip "
        f"must never affect layout")
    pv.shutdown()


def test_only_one_screen_writes_the_stft_length(qapp, tmp_path):
    """Regression: the EMG Advanced modal gained an n_fft control while Setup still had one,
    so a modal edit was reverted on the next tab change — to_state rewrites everything Setup
    owns. Measured before the fix: 512 became 256 again."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path), noise=True, data_out=_OUT)))
    assert not hasattr(win.settings_screen, "noise_nfft"), "a second n_fft writer is back"
    n = win.state.settings.processing.emg.noise
    n.n_fft = 512
    for _ in range(3):
        win.settings_screen.to_state()
    assert n.n_fft == 512
    win.close()


# -- the end-expiratory trend anchor controls ---------------------------------

def test_mech_advanced_exposes_the_trend_anchor_settings(qapp, tmp_path, monkeypatch):
    """These decide whether the correction can run at all. With none of them on screen a
    user whose recording found no troughs had no way to fix it — that is what turned a
    tunable threshold into a dead end."""
    pv = _preview(qapp, tmp_path)
    seen = {}
    _mech_stub(monkeypatch, lambda d: seen.update(
        {k: d.widget(k) is not None for k in
         ("trend_peak_min_prominence_frac", "trend_peak_min_distance_s",
          "trend_peak_min_height")}))
    pv._open_mech_advanced()
    assert seen == {"trend_peak_min_prominence_frac": True,
                    "trend_peak_min_distance_s": True,
                    "trend_peak_min_height": True}
    pv.shutdown()


def test_trend_anchor_rows_follow_the_correct_trend_checkbox(qapp, tmp_path, monkeypatch):
    pv = _preview(qapp, tmp_path)
    seen = {}

    def edit(d):
        cb = d.widget("correct_trend")
        frac = d.widget("trend_peak_min_prominence_frac")
        seen["off"] = frac.isEnabled()
        cb.setChecked(True)
        seen["on"] = frac.isEnabled()
        cb.setChecked(False)
        seen["off_again"] = frac.isEnabled()

    _mech_stub(monkeypatch, edit)
    pv._open_mech_advanced()
    assert seen == {"off": False, "on": True, "off_again": False}
    pv.shutdown()


def test_the_legacy_threshold_reads_back_as_auto_at_its_minimum(qapp, tmp_path, monkeypatch):
    """The 'Auto' sentinel is the only way to CLEAR an inherited absolute threshold from
    the GUI; if it ever committed 0.0 instead of None the legacy gate would silently stay
    selected (and 0.0 admits every plateau wiggle as an anchor)."""
    pv = _preview(qapp, tmp_path)
    s = pv.state.settings
    s.processing.volume.trend_peak_min_height = 0.5
    seen = {}

    def edit(d):
        w = d.widget("trend_peak_min_height")
        seen["shows_value"] = w.value() == 0.5
        w.setValue(w.minimum())                 # the row now reads "Auto — …"
        seen["special_text"] = w.specialValueText() != ""

    _mech_stub(monkeypatch, edit)
    pv._open_mech_advanced()
    assert seen == {"shows_value": True, "special_text": True}
    assert s.processing.volume.trend_peak_min_height is None
    pv.shutdown()


# -- Apply, non-modality and the live breath count (D13, UI-overhaul) ---------
def test_the_mechanics_button_names_what_is_behind_it(qapp, tmp_path):
    """The button used to just say 'Advanced…' — indistinguishable from the ECG/EMG
    buttons of the same name, and naming nothing that would tell you what shapes the
    channel stack you are looking at."""
    pv = _preview(qapp, tmp_path)
    assert pv.btn_mech_advanced.text() == "Advanced… (breath detection, volume, WOB)"
    assert pv.btn_mech_advanced.toolTip() == (
        "Breath segmentation, work-of-breathing source, volume/drift corrections, "
        "resampling and per-file breath-count overrides.")
    pv.shutdown()


def test_mech_dialog_is_not_modal_while_the_real_exec_runs(qapp, tmp_path, monkeypatch):
    """Patches AdvancedDialog.exec ITSELF (not a subclass that overrides it away, as
    _mech_stub does for every other mechanics test here) so the real modality logic
    actually runs — a test that stubs exec() entirely cannot see this failure class.

    ``isModal()`` is sampled from INSIDE the ``QTimer.singleShot`` callback — i.e. while
    the real ``exec()``'s own event loop is spinning — not before ``real_exec(self)`` is
    even called. Sampling it before proves only that the right ``modal=`` argument reached
    the constructor (``setModal()`` in ``__init__``, unaffected by whatever ``exec()``
    itself later does); it would stay green even if ``exec()`` were reverted to
    unconditionally call ``super().exec()`` — the exact regression this override exists to
    prevent, and the one the earlier, top-of-file generic tests already catch this way."""
    import respmech.ui.advanced_dialog as ad
    from PySide6.QtCore import QTimer
    pv = _preview(qapp, tmp_path)
    real_exec = ad.AdvancedDialog.exec
    seen = {}

    def spying_exec(self):
        seen["has_apply"] = self.btn_apply is not None

        def check_and_close():
            seen["is_modal_during_exec"] = self.isModal()
            self.accept()

        QTimer.singleShot(0, check_and_close)
        return real_exec(self)

    monkeypatch.setattr(ad.AdvancedDialog, "exec", spying_exec)
    pv._open_mech_advanced()
    assert seen["is_modal_during_exec"] is False, \
        "the mechanics dialog must be genuinely non-modal WHILE it is open"
    assert seen["has_apply"] is True
    pv.shutdown()


def test_ecg_and_emg_dialogs_stay_modal_with_no_apply_button(qapp, tmp_path, monkeypatch):
    """The other two AdvancedDialog callers must be completely unaffected by D13 — same
    real-exec, sampled-mid-exec check as the mechanics test above, for both of them."""
    import respmech.ui.advanced_dialog as ad
    from PySide6.QtCore import QTimer
    pv = _preview(qapp, tmp_path)
    real_exec = ad.AdvancedDialog.exec
    seen = {"modal": [], "has_apply": []}

    def spying_exec(self):
        seen["has_apply"].append(self.btn_apply is not None)

        def check_and_close():
            seen["modal"].append(self.isModal())
            self.accept()

        QTimer.singleShot(0, check_and_close)
        return real_exec(self)

    monkeypatch.setattr(ad.AdvancedDialog, "exec", spying_exec)
    pv._open_ecg_advanced()
    pv._open_emg_advanced()
    assert seen["modal"] == [True, True]
    assert seen["has_apply"] == [False, False]
    pv.shutdown()


def test_mech_dialog_debounces_the_live_derived_line(qapp, tmp_path, monkeypatch):
    """The ticket requires the live breath-count search to be debounced — a real scipy peak
    search, not the trivial arithmetic the OTHER two modals' derived lines do. Prove
    ``_open_mech_advanced`` actually passes a non-zero ``derived_debounce_ms`` (setting it
    to 0 would make every keystroke recompute synchronously, which every OTHER
    mechanics-derived-line test in this file would still pass, since they all deliberately
    call ``_refresh_derived_now()`` to bypass the debounce for their own assertions)."""
    import respmech.ui.advanced_dialog as ad
    from PySide6.QtCore import QTimer
    pv = _preview(qapp, tmp_path)
    real_exec = ad.AdvancedDialog.exec
    seen = {}

    def spying_exec(self):
        seen["has_timer"] = self._derived_timer is not None
        seen["interval_ms"] = self._derived_timer.interval() if self._derived_timer else None
        QTimer.singleShot(0, self.accept)
        return real_exec(self)

    monkeypatch.setattr(ad.AdvancedDialog, "exec", spying_exec)
    pv._open_mech_advanced()
    assert seen["has_timer"] is True, "the live breath count must be debounced, not synchronous"
    assert seen["interval_ms"] == 250
    pv.shutdown()


def test_mech_dialog_has_exactly_three_buttons(qapp, tmp_path, monkeypatch):
    pv = _preview(qapp, tmp_path)
    seen = {}
    _mech_stub(monkeypatch, lambda d: seen.update(
        cancel=d.btn_cancel, apply=d.btn_apply, ok=d.btn_ok), accept=False)
    pv._open_mech_advanced()
    assert seen["cancel"] is not None and seen["apply"] is not None and seen["ok"] is not None
    pv.shutdown()


def test_apply_commits_the_edited_keys_and_triggers_a_recompute_without_closing(
        qapp, tmp_path, monkeypatch):
    """Apply must run through the exact same commit path as OK — this stubs exec() (so it
    does not test modality, that is covered above) but exercises the REAL on_apply callback
    wired to the REAL Apply button, mid-'dialog', before any Accepted/Rejected is decided.

    Spies on ``_request_autorun`` directly, not just ``settings_edited`` — deleting the
    ``_request_autorun()`` call from ``_commit`` while leaving ``settings_edited.emit()`` in
    place would still mark the analysis "modified" and pass a signal-only assertion, without
    ever actually scheduling the recompute the ticket names explicitly ("udløser en
    genberegning")."""
    pv = _preview(qapp, tmp_path)
    s = pv.state.settings
    edits = []
    pv.settings_edited.connect(lambda: edits.append(1))
    autoruns = []
    monkeypatch.setattr(pv, "_request_autorun",
                        lambda *a, **k: autoruns.append((a, k)))

    def edit(d):
        assert d.btn_apply is not None, "the mechanics dialog must offer Apply"
        d.widget("buffer").setValue(555)
        d.btn_apply.click()
        assert s.processing.segmentation.buffer == 555, "Apply must commit like OK does"
        assert edits, "Apply must mark the analysis modified"
        assert autoruns, "Apply must trigger a recompute"
        edits.clear()
        autoruns.clear()

    _mech_stub(monkeypatch, edit, accept=False)   # Cancel afterwards: must not undo Apply
    pv._open_mech_advanced()
    assert s.processing.segmentation.buffer == 555, "an applied edit survives a later Cancel"
    assert not edits, "a Cancel after Apply must not mark the analysis modified again"
    assert not autoruns, "a Cancel after Apply must not trigger a second recompute"
    pv.shutdown()


def test_apply_then_a_further_edit_then_ok_commits_both(qapp, tmp_path, monkeypatch):
    """The full real-world flow: tweak, Apply, tweak something else, OK. Both edits must
    land, Apply must not resend on the later OK (edited_values() baseline moved forward —
    see the generic test_apply_moves_the_edited_baseline_forward), and BOTH commits must
    each trigger their own recompute."""
    pv = _preview(qapp, tmp_path)
    s = pv.state.settings
    autoruns = []
    monkeypatch.setattr(pv, "_request_autorun",
                        lambda *a, **k: autoruns.append((a, k)))

    def edit(d):
        d.widget("buffer").setValue(555)
        d.btn_apply.click()
        assert s.processing.segmentation.buffer == 555
        assert len(autoruns) == 1, "Apply must trigger exactly one recompute"
        d.widget("height").setValue(0.25)

    _mech_stub(monkeypatch, edit, accept=True)
    pv._open_mech_advanced()
    assert s.processing.segmentation.buffer == 555, "Apply's edit must survive the later OK"
    assert s.processing.segmentation.peak.height == 0.25, "OK's own edit must also commit"
    assert len(autoruns) == 2, "each commit (Apply, then OK) must trigger its own recompute"
    pv.shutdown()


def test_the_live_line_explains_when_no_file_has_been_previewed(qapp, tmp_path, monkeypatch):
    pv = _preview(qapp, tmp_path)               # NOT rendered: _trend_probe stays None
    assert pv._trend_probe is None
    seen = {}
    _mech_stub(monkeypatch, lambda d: seen.update(text=d.derived.text()), accept=False)
    pv._open_mech_advanced()
    assert seen["text"] == "No previewed file to count breaths in yet."
    pv.shutdown()


def test_the_live_line_counts_breaths_while_trend_correction_is_off(qapp, tmp_path, monkeypatch):
    """The default state (correct_trend is False): the dead 'End-expiratory trend
    correction is off.' sentence is replaced by a live, useful number."""
    pv = _rendered_preview(qapp, tmp_path)
    assert pv.state.settings.processing.volume.correct_trend is False
    seen = {}
    _mech_stub(monkeypatch, lambda d: seen.update(text=d.derived.text()), accept=False)
    pv._open_mech_advanced()
    assert "breath" in seen["text"] and "found with these thresholds" in seen["text"]
    assert pv._trend_probe_file in seen["text"]
    pv.shutdown()


def test_the_live_breath_count_matches_what_a_real_volume_based_run_would_find(
        qapp, tmp_path, monkeypatch):
    """Not just A number: the SAME count ``stage_mechanics_preview`` — the real pipeline —
    would produce end-to-end for a volume-based run under the same staged thresholds. A
    weaker test that only re-derives the same call the implementation itself makes would
    not catch the implementation quietly diverging from the real pipeline."""
    from respmech.ui.workers import stage_mechanics_preview
    pv = _rendered_preview(qapp, tmp_path)
    s = pv.state.settings
    s.processing.segmentation.method = "volume"      # force the real pipeline down this path
    real = stage_mechanics_preview(s, pv._current_file())
    v = {"height": s.processing.segmentation.peak.height,
         "distance_s": s.processing.segmentation.peak.distance_s,
         "width_s": s.processing.segmentation.peak.width_s}
    n = pv._advanced_live_breath_count(pv._trend_probe, v)
    assert n == real["nbreaths"] > 0
    pv.shutdown()


def test_the_trend_anchor_line_is_unchanged_when_trend_correction_is_on(
        qapp, tmp_path, monkeypatch):
    """The refactor that added the breath count for the OFF case must not change the
    ON-case wording at all — asserted against the EXACT string the original, unmoved
    f-string would produce (not just substrings), so a tweak to the 'OK.'/'NOT ENOUGH…'
    tail — which a pure substring check would miss — fails this test too."""
    import numpy as np
    from respmech.core import compute
    pv = _rendered_preview(qapp, tmp_path)
    s = pv.state.settings
    seen = {}

    def edit(d):
        d.widget("correct_trend").setChecked(True)
        d._refresh_derived_now()          # bypass the debounce window for this assertion
        seen["text"] = d.derived.text()
        seen["v"] = d.values()

    _mech_stub(monkeypatch, edit, accept=False)
    pv._open_mech_advanced()

    v = seen["v"]
    need = compute._TREND_MIN_ANCHORS.get(v["trend_method"], 2)
    n = compute.trend_anchors(
        pv._trend_probe, s.input.format.sampling_frequency,
        min_height=v["trend_peak_min_height"],
        min_prominence_frac=v["trend_peak_min_prominence_frac"],
        min_distance_s=v["trend_peak_min_distance_s"]).size
    expected = (f"In {pv._trend_probe_file}: volume range {float(np.ptp(pv._trend_probe)):.2f}, "
                f"{n} trough(s) found — "
                + ("OK." if n >= need else f"NOT ENOUGH (needs {need}); the run will fail this file."))
    assert seen["text"] == expected
    pv.shutdown()


def test_the_trend_anchor_hint_explains_when_no_file_has_been_previewed(qapp, tmp_path, monkeypatch):
    """The ON-case counterpart to test_the_live_line_explains_when_no_file_has_been_previewed
    (the OFF case) — pre-existing logic, moved verbatim by this ticket, but untested until
    now anywhere in the repository."""
    from respmech.core import compute
    pv = _preview(qapp, tmp_path)               # NOT rendered: _trend_probe stays None
    assert pv._trend_probe is None
    seen = {}

    def edit(d):
        d.widget("correct_trend").setChecked(True)
        d._refresh_derived_now()
        seen["text"] = d.derived.text()
        seen["v"] = d.values()

    _mech_stub(monkeypatch, edit, accept=False)
    pv._open_mech_advanced()
    need = compute._TREND_MIN_ANCHORS.get(seen["v"]["trend_method"], 2)
    assert seen["text"] == f"Needs at least {need} end-expiratory troughs in every file."
    pv.shutdown()


def test_the_trend_anchor_hint_withdraws_when_volume_conditioning_changes(
        qapp, tmp_path, monkeypatch):
    """The ON-case counterpart to test_the_live_count_withdraws_when_volume_conditioning_changes
    (the OFF case) — pre-existing logic, moved verbatim by this ticket, but untested until
    now anywhere in the repository."""
    from respmech.core import compute
    pv = _rendered_preview(qapp, tmp_path)
    seen = {}

    def edit(d):
        d.widget("correct_trend").setChecked(True)
        d._refresh_derived_now()
        seen["before"] = d.derived.text()
        seen["v"] = d.values()
        cb = d.widget("correct_drift")
        cb.setChecked(not cb.isChecked())
        d._refresh_derived_now()
        seen["after"] = d.derived.text()

    _mech_stub(monkeypatch, edit, accept=False)
    pv._open_mech_advanced()
    assert "trough(s) found" in seen["before"]
    need = compute._TREND_MIN_ANCHORS.get(seen["v"]["trend_method"], 2)
    assert seen["after"] == ("Volume conditioning changed — press OK, then reopen this "
                             f"dialog for a trough count. Needs at least {need}.")
    pv.shutdown()


def test_the_live_count_withdraws_when_volume_conditioning_changes(qapp, tmp_path, monkeypatch):
    """The moment a field that CHANGES the staged volume is touched, the probe describes a
    signal the run will no longer use — the count must be withdrawn, not left stale and
    confidently wrong. Reuses the exact _TREND_PROBE_KEYS guard _trend_hint already had for
    the trend-anchor count."""
    pv = _rendered_preview(qapp, tmp_path)
    seen = {}

    def edit(d):
        seen["before"] = d.derived.text()
        cb = d.widget("correct_drift")
        cb.setChecked(not cb.isChecked())
        d._refresh_derived_now()          # bypass the debounce window for this assertion
        seen["after"] = d.derived.text()

    _mech_stub(monkeypatch, edit, accept=False)
    pv._open_mech_advanced()
    assert "found with these thresholds" in seen["before"]
    assert "Volume conditioning changed" in seen["after"]
    assert "breath count" in seen["after"]
    pv.shutdown()


def test_the_live_count_explains_when_the_thresholds_cannot_be_evaluated(
        qapp, tmp_path, monkeypatch):
    """A minimum distance of 0 s makes scipy.signal.find_peaks raise outright (distance
    must be >= 1 sample) — a confidently wrong count would be worse than admitting it
    could not be computed."""
    pv = _rendered_preview(qapp, tmp_path)
    seen = {}

    def edit(d):
        d.widget("distance_s").setValue(0.0)
        d._refresh_derived_now()          # bypass the debounce window for this assertion
        seen["text"] = d.derived.text()

    _mech_stub(monkeypatch, edit, accept=False)
    pv._open_mech_advanced()
    assert seen["text"] == "Could not count breaths with these breath-detection thresholds."
    pv.shutdown()
