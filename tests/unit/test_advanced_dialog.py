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


def test_editing_stages_without_touching_anything_outside(qapp):
    dlg = AdvancedDialog("T", _fields(), {"a": 1, "b": 0.5, "c": False})
    dlg.widget("a").setValue(7)
    dlg.widget("c").setChecked(True)
    assert dlg.values() == {"a": 7, "b": 0.5, "c": True}


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
