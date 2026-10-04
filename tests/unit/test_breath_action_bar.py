"""The right-click menu's choices as controls on the toolbar row next to Refresh.

A type selector, 'IC reference for' and 'Reference manoeuvres…' act on the MARKED breath,
reuse the menu's own funnels and are disabled until a breath is marked (or while a run
locks the write actions).
"""
import os

import pytest

from respmech.ui.state import AppState

from _helpers import INPUT, requires_synth, synth_settings

pytestmark = requires_synth()

NAME = "synth_case_A.csv"


def _screen(tmp_path):
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s))
    return win, win.preview_screen, s


def _render(pv, s):
    from respmech.ui.workers import stage_mechanics_preview
    pv._refresh_files(); pv.file_rail.select_filename(NAME)
    pv._render_preview(stage_mechanics_preview(s, os.path.join(INPUT, NAME)))


def _controls(pv):
    return (pv.breath_type_combo, pv.btn_breath_ic_ref, pv.btn_breath_refs)


def _kind_of(s, n):
    t = next((t for t in s.processing.breath_types if t.file == NAME and t.breath == n), None)
    return t.kind if t else None


def test_bar_sits_on_the_refresh_row_and_starts_disabled(qapp, tmp_path):
    win, pv, s = _screen(tmp_path)
    bar = pv.layout().itemAt(0).layout()
    widgets = [bar.itemAt(i).widget() for i in range(bar.count()) if bar.itemAt(i).widget()]
    for w in (pv.btn_refresh_all, *_controls(pv)):
        assert w in widgets
    _render(pv, s)
    assert not any(w.isEnabled() for w in _controls(pv))
    assert "Mark a breath first" in pv.breath_type_combo.toolTip()
    assert pv.breath_type_combo.currentIndex() == -1


def test_marking_enables_the_controls_and_unmarking_disables_them(qapp, tmp_path):
    win, pv, s = _screen(tmp_path)
    _render(pv, s)
    n = next(iter(pv._breath_spans))
    pv._select_breath(n)
    assert pv.breath_type_combo.isEnabled() and pv.btn_breath_refs.isEnabled()
    assert pv.breath_bar_label.text() == f"Breath {n}:"
    assert pv.breath_type_combo.currentData() == "tidal"
    assert not pv.btn_breath_ic_ref.isEnabled()                 # not an IC manoeuvre yet
    assert "IC manoeuvre" in pv.btn_breath_ic_ref.toolTip()
    pv._select_breath(n)                                        # click again: unmark
    assert not any(w.isEnabled() for w in _controls(pv))
    assert pv.breath_bar_label.text() == "Breath:"


def test_selector_offers_the_same_kinds_as_the_menu(qapp, tmp_path):
    win, pv, s = _screen(tmp_path)
    _render(pv, s)
    pv._select_breath(next(iter(pv._breath_spans)))
    items = [pv.breath_type_combo.itemData(i) for i in range(pv.breath_type_combo.count())]
    assert tuple(items) == pv._type_menu_kinds()
    assert "rest" not in items                                  # flow-bearing recording


def test_selector_types_and_excludes_the_marked_breath_only(qapp, tmp_path):
    win, pv, s = _screen(tmp_path)
    _render(pv, s)
    a, b = list(pv._breath_spans)[:2]
    pv._select_breath(a)
    combo = pv.breath_type_combo
    combo.activated.emit(combo.findData("ic"))
    assert _kind_of(s, a) == "ic" and _kind_of(s, b) is None
    assert pv.breath_type_combo.currentData() == "ic"
    assert pv.btn_breath_ic_ref.isEnabled()                     # now an IC manoeuvre
    combo.activated.emit(combo.findData("excluded"))
    assert _kind_of(s, a) is None
    excl = {x for e in s.processing.exclude_breaths if e.file == NAME for x in e.breaths}
    assert excl == {a}
    assert pv.breath_type_combo.currentData() == "excluded"
    assert not pv.btn_breath_ic_ref.isEnabled()
    combo.activated.emit(combo.findData("tidal"))
    assert not s.processing.exclude_breaths and not s.processing.breath_types
    assert pv._selected_breath == a                             # the mark survives


def test_the_selector_follows_a_type_set_from_the_right_click_menu(qapp, tmp_path):
    win, pv, s = _screen(tmp_path)
    _render(pv, s)
    n = next(iter(pv._breath_spans))
    pv._select_breath(n)
    pv._apply_breath_type_choice(n, "fvc")
    assert pv.breath_type_combo.currentData() == "fvc"


def test_ic_reference_menu_points_the_file_at_the_marked_breath(qapp, tmp_path):
    win, pv, s = _screen(tmp_path)
    _render(pv, s)
    n = next(iter(pv._breath_spans))
    pv._select_breath(n)
    pv._apply_breath_type_choice(n, "ic")
    actions = pv.btn_breath_ic_ref.menu().actions()
    assert [a.text() for a in actions][0] == "This file" and actions[-1].text() == "All files"
    actions[0].trigger()
    entry = next(e for e in s.processing.references if e.file == NAME)
    assert entry.ic.file == NAME and entry.ic.breaths == [n]


def test_reference_manoeuvres_button_opens_the_picker(qapp, tmp_path, monkeypatch):
    win, pv, s = _screen(tmp_path)
    _render(pv, s)
    pv._select_breath(next(iter(pv._breath_spans)))
    seen = []
    monkeypatch.setattr(type(pv), "_open_reference_picker", lambda self, f=None: seen.append(f))
    pv.btn_breath_refs.click()
    assert seen == [None]


def test_a_run_locks_the_controls_with_a_reason_and_releases_them(qapp, tmp_path):
    win, pv, s = _screen(tmp_path)
    _render(pv, s)
    n = next(iter(pv._breath_spans))
    pv._select_breath(n)
    pv._apply_breath_type_choice(n, "ic")
    pv.set_run_active(True)
    assert not any(w.isEnabled() for w in _controls(pv))
    assert "run is in progress" in pv.breath_type_combo.toolTip()
    pv.set_run_active(False)
    assert all(w.isEnabled() for w in _controls(pv))


def test_a_refused_write_resets_the_selector(qapp, tmp_path):
    win, pv, s = _screen(tmp_path)
    _render(pv, s)
    n = next(iter(pv._breath_spans))
    pv._select_breath(n)
    pv._run_active = True                                       # bypass the disabled state
    combo = pv.breath_type_combo
    combo.setCurrentIndex(combo.findData("ic"))
    pv._on_breath_type_combo(combo.currentIndex())
    assert _kind_of(s, n) is None
    pv._run_active = False
    pv._update_breath_bar()
    assert combo.currentData() == "tidal"


def test_switching_file_clears_the_mark_and_the_controls(qapp, tmp_path):
    win, pv, s = _screen(tmp_path)
    _render(pv, s)
    pv._select_breath(next(iter(pv._breath_spans)))
    pv._clear_breath_selection()
    assert not any(w.isEnabled() for w in _controls(pv))
