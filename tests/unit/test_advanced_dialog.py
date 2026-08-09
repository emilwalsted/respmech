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
