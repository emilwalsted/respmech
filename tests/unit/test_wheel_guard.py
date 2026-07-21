"""Scrolling must never edit the thing under the cursor.

Spin boxes, combo boxes and tab bars all consume the wheel to change themselves, so
scrolling across a form silently changed a setting. Measured before the guard existed:
Setup's "EMG RMS window" stepped 0.05 s -> 0.02 s (changing every reported EMG number) and
marked the analysis modified; a role dropdown in the channel dialog jumped to Flow, which —
Flow being mutually exclusive — moved the flow channel off the column it was on; and the
Preview strip's ECG parameters stepped and scheduled a recompute.

The per-screen tests below are EXHAUSTIVE rather than sampled: they walk every spin box and
combo actually present. A widget added later therefore fails here instead of quietly
reintroducing the bug, which the guard's install-time class discovery cannot prevent.
"""
import glob
import os

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QAbstractSpinBox, QComboBox, QScrollArea

from respmech.ui.state import AppState

from _helpers import INPUT, requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


def _wheel(qapp, target, notches=-3):
    qapp.sendEvent(target, QWheelEvent(
        QPointF(5, 5), target.mapToGlobal(QPoint(5, 5)), QPoint(0, 0),
        QPoint(0, notches * 120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    qapp.processEvents()


def _value(w):
    return w.currentIndex() if isinstance(w, QComboBox) else w.value()


def _stealers(root):
    return root.findChildren(QAbstractSpinBox) + root.findChildren(QComboBox)


def _assert_nothing_edits(qapp, root, label):
    """No wheel-stealing widget anywhere under `root` may change on a wheel."""
    widgets = [w for w in _stealers(root) if w.isVisible()]
    assert widgets, f"{label}: found no widgets to check — the walk is broken"
    before = [(w, _value(w)) for w in widgets]
    for w, _ in before:
        _wheel(qapp, w)
    changed = [w.objectName() or w.__class__.__name__
               for w, v0 in before if _value(w) != v0]
    assert not changed, f"{label}: the wheel edited {len(changed)} widget(s): {changed}"


# -- Setup screen --------------------------------------------------------------
def _settings_screen(qapp, tmp_path):
    from respmech.ui.screens.settings_screen import SettingsScreen
    sc = SettingsScreen(AppState(synth_settings(str(tmp_path), data_out=_OUT)))
    sc.resize(900, 700)
    sc.show()
    qapp.processEvents()
    sc._mark_clean()
    return sc


def test_no_setup_widget_is_edited_by_the_wheel(qapp, tmp_path):
    sc = _settings_screen(qapp, tmp_path)
    _assert_nothing_edits(qapp, sc, "Setup")
    assert not sc.is_dirty(), "scrolling marked the analysis modified"


def test_the_setup_form_scrolls_at_the_scroll_areas_own_speed(qapp, tmp_path):
    """The guard forwards to the viewport rather than moving the scrollbar itself, so
    scrolling over a widget must land in exactly the same place as scrolling beside it."""
    sc = _settings_screen(qapp, tmp_path)
    bar = sc.findChild(QScrollArea).verticalScrollBar()
    assert bar.maximum() > 0, "the form must be scrollable for this to mean anything"

    _wheel(qapp, sc.findChild(QScrollArea).viewport())
    native = bar.value()
    bar.setValue(0)
    _wheel(qapp, sc.emg_rms_window)
    assert bar.value() == native, "scrolling over a widget runs at a different speed"


# -- channel dialog ------------------------------------------------------------
def _dialog(qapp):
    from respmech.core.settings import Settings
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    from respmech.ui.workers import load_raw_matrix
    files = sorted(glob.glob(os.path.join(INPUT, "synth_case_*.csv")))
    dlg = ChannelSetupDialog(files, 1000, initial={"flow": 5, "poes": 7, "pgas": 8, "pdi": 9},
                             loader=lambda p: load_raw_matrix(Settings(), p))
    dlg.resize(940, 660)
    dlg.show()
    qapp.processEvents()
    return dlg


def test_no_channel_dialog_widget_is_edited_by_the_wheel(qapp):
    dlg = _dialog(qapp)
    before = dlg.selected_mapping()
    _assert_nothing_edits(qapp, dlg, "channel dialog")
    assert dlg.selected_mapping() == before, "the wheel re-assigned a channel"


def test_the_file_picker_above_the_list_does_nothing_at_all(qapp):
    """It sits outside the scroll area, so scrolling it must neither switch files nor
    move the unrelated column list underneath."""
    dlg = _dialog(qapp)
    bar = dlg._scroll.verticalScrollBar()
    idx, at = dlg.file_combo.currentIndex(), bar.value()
    _wheel(qapp, dlg.file_combo)
    assert dlg.file_combo.currentIndex() == idx, "the wheel switched the previewed file"
    assert bar.value() == at, "the wheel moved the column list from outside it"


# -- Preview screen ------------------------------------------------------------
def test_no_preview_widget_is_edited_by_the_wheel(qapp, tmp_path):
    """The strips sit right above wheel-zoomable plots, and every ECG/noise parameter
    here writes into the analysis and schedules a recompute."""
    from respmech.ui.screens.preview_screen import PreviewScreen
    pv = PreviewScreen(AppState(synth_settings(str(tmp_path), data_out=_OUT)))
    pv.resize(1100, 800)
    pv.show()
    qapp.processEvents()
    edits = []
    pv.settings_edited.connect(lambda: edits.append(1))
    _assert_nothing_edits(qapp, pv, "Preview")
    assert not edits, "scrolling emitted settings_edited"
    pv.shutdown()


# -- tab bars ------------------------------------------------------------------
@pytest.mark.parametrize("which", ["main", "preview"])
def test_the_wheel_does_not_change_tab(qapp, tmp_path, which):
    """A wheel that drifts up off the form onto the tab bar would land on Preview,
    which schedules work."""
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(str(tmp_path), data_out=_OUT)))
    win.resize(1200, 800)
    win.show()
    qapp.processEvents()
    tabs = win.tabs if which == "main" else win.preview_screen.subtabs
    before = tabs.currentIndex()
    _wheel(qapp, tabs.tabBar())
    assert tabs.currentIndex() == before, f"the wheel changed the {which} tab"
    win.close()


# -- a scrollable widget inside the form must not be a dead patch ---------------
def test_the_breath_count_box_scrolls_itself_then_hands_the_wheel_on(qapp, tmp_path):
    """Regression: it consumed the wheel even at its own end, so the form stopped
    scrolling under the cursor and never resumed."""
    sc = _settings_screen(qapp, tmp_path)
    box = sc.breath_counts_edit
    box.setPlainText("\n".join(f"file_{i}.csv = {i}" for i in range(60)))
    qapp.processEvents()
    inner = box.verticalScrollBar()
    assert inner.maximum() > 0, "the box must overflow for this to mean anything"

    form = sc.findChild(QScrollArea).verticalScrollBar()
    at = form.value()
    _wheel(qapp, box.viewport())                      # first: it scrolls itself
    assert inner.value() > 0
    assert form.value() == at, "the form moved while the box still had room"

    inner.setValue(inner.maximum())                   # now it is spent
    _wheel(qapp, box.viewport())
    assert form.value() > at, "the wheel died instead of scrolling the form on"
