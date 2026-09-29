"""Every interactive control the modular analysis added carries a tooltip.

Each control has a sentence in ``ui/help_text.py`` (or its own ``Field`` help) saying what
it does and what to enter. A control added without one is easy to miss, since nothing else
fails: this walks the screens and dialogs the modular analysis introduced and names any
input control (check box, radio button, combo, spin box, command button) without a tooltip.
Container titles and the dialogs' own Cancel/OK/Apply buttons do not carry one, by the same
convention as the rest of the app.
"""
import os

from PySide6.QtWidgets import (QAbstractSpinBox, QCheckBox, QComboBox, QDialog, QPushButton,
                               QRadioButton, QWidget)

_STANDARD_BUTTONS = {"Cancel", "OK", "Apply", "&Cancel", "&OK", "&Apply"}
_INPUTS = (QCheckBox, QRadioButton, QComboBox, QAbstractSpinBox, QPushButton)


def _missing(root):
    out = []
    for w in root.findChildren(QWidget):
        if not isinstance(w, _INPUTS) or w.toolTip():
            continue
        text = w.text() if hasattr(w, "text") and not isinstance(w, QAbstractSpinBox) else ""
        if isinstance(w, QPushButton) and text in _STANDARD_BUTTONS:
            continue
        out.append(f"{type(w).__name__} {text!r}")
    return out


def test_the_signal_set_pickers_controls_have_tooltips(qapp):
    from respmech.ui.signal_set_dialog import EmgRecordingContentDialog, SignalSetDialog
    for dlg in (SignalSetDialog(), EmgRecordingContentDialog()):
        assert _missing(dlg) == []
        dlg.close()


def test_the_reference_pickers_controls_have_tooltips(qapp, tmp_path):
    from respmech.core.sample import build_sample_settings, write_sample_recording
    from respmech.ui.reference_picker_dialog import ReferencePickerDialog
    desc = write_sample_recording(os.path.join(str(tmp_path), "input"))
    s = build_sample_settings(desc, os.path.join(str(tmp_path), "output"))
    dlg = ReferencePickerDialog(desc["filename"], s, [desc["filename"]])
    assert _missing(dlg) == []
    dlg.close()


def test_the_setup_screens_controls_have_tooltips_for_the_complete_and_the_emg_only_set(qapp):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    for emg_only in (False, True):
        win = MainWindow(AppState())
        if emg_only:
            win.state.settings.analysis.signals = ["emg"]
        assert win.settings_screen.open_sample_analysis(use_current_signals=emg_only)
        assert _missing(win.settings_screen) == [], "emg-only" if emg_only else "complete"
        win.settings_screen._mark_clean()
        win.close()


def test_the_advanced_dialogs_new_fields_have_tooltips(qapp, monkeypatch):
    """Mechanics ▸ Advanced… (with the lung-volume, flow-volume and per-file groups) and the
    EMG-only segmentation parameters, opened for real but never run modally."""
    from respmech.ui.advanced_dialog import AdvancedDialog
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState

    opened = []

    def fake_exec(self):
        opened.append(self)
        return QDialog.Rejected
    monkeypatch.setattr(AdvancedDialog, "exec", fake_exec)
    monkeypatch.setattr(QDialog, "exec", fake_exec)

    win = MainWindow(AppState())
    assert win.settings_screen.open_sample_analysis()
    win.preview_screen._open_mech_advanced()
    win.settings_screen._mark_clean()
    win.close()

    win = MainWindow(AppState())
    win.state.settings.analysis.signals = ["emg"]
    assert win.settings_screen.open_sample_analysis(use_current_signals=True)
    for method in ("fixed_windows", "emg_burst"):
        win.state.settings.processing.segmentation.method = method
        win.preview_screen._open_segmentation_advanced()
    win.settings_screen._mark_clean()
    win.close()

    assert len(opened) == 3
    for dlg in opened:
        assert _missing(dlg) == [], dlg.windowTitle()
