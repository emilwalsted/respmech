"""M-20: the breath-typing click primitive.

``BreathSpansItem``'s right-click/Ctrl+left-click handling is exercised directly with
lightweight fake pyqtgraph-style events (same style as
``test_gui_interactive.py::test_legend_click_does_not_toggle_breath``) — deterministic,
no scene geometry needed. The ``_set_breath_type`` funnel, the run-lock, the mutual
exclusivity of tidal/excluded/typed, and the minimal type menu are exercised through a
real ``MainWindow``/``PreviewScreen``, the same pattern as the rest of this test suite.
"""
import os

import pyqtgraph as pg
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QMenu

from respmech.ui.screens.preview._plot_helpers import BreathSpansItem
from respmech.ui.state import AppState

from _helpers import INPUT, requires_synth, synth_settings

pytestmark = requires_synth()


def _render_mech(pv, s, name="synth_case_A.csv"):
    from respmech.ui.workers import stage_mechanics_preview
    pv._refresh_files(); pv.file_rail.select_filename(name)
    pv._render_preview(stage_mechanics_preview(s, os.path.join(INPUT, name)))


# --------------------------------------------------------------------------- #
# The item-level primitive (BreathSpansItem.hoverEvent/mouseClickEvent)
# --------------------------------------------------------------------------- #
class _FakeHover:
    def __init__(self, x, exit_=False):
        self._x = x
        self._exit = exit_
        self.claimed = []

    def isExit(self):
        return self._exit

    def pos(self):
        return QPointF(self._x, 0.0)

    def acceptClicks(self, button):
        self.claimed.append(button)
        return True


class _FakeClick:
    def __init__(self, x, button, modifiers=Qt.NoModifier):
        self._x = x
        self._button = button
        self._modifiers = modifiers
        self._scene_pos = QPointF(x, 100.0)
        self.accepted = False
        self.ignored = False

    def pos(self):
        return QPointF(self._x, 0.0)

    def scenePos(self):
        return self._scene_pos

    def button(self):
        return self._button

    def modifiers(self):
        return self._modifiers

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


def _two_span_item():
    """A BreathSpansItem attached to a real (viewless) pyqtgraph scene — not a bare
    construction. mouseClickEvent bails out for an item with no scene at all (the
    same guard that protects against a stale, torn-down item — see
    test_a_click_on_an_item_detached_from_its_scene_is_ignored), which a bare item
    would trip for the wrong reason. Nothing else in this module holds a Python
    reference to the scene a plain QGraphicsScene().addItem(item) would create —
    QGraphicsScene owns its items, not the other way round, so an unreferenced
    scene is garbage-collected almost immediately and takes the (Python-wrapped)
    item down with it (measured: 'libshiboken: Internal C++ object already
    deleted' on the very next call). Stash the scene ON the item so it stays
    alive exactly as long as the item does."""
    item = BreathSpansItem()
    item.set_spans([(0.0, 1.0, pg.mkBrush(0, 0, 0), 1),
                    (2.0, 3.0, pg.mkBrush(0, 0, 0), 2)])   # a real gap between 1.0 and 2.0
    item._test_scene = pg.GraphicsScene()
    item._test_scene.addItem(item)
    return item


def test_hovering_a_span_claims_the_right_button_but_a_gap_does_not(qapp):
    item = _two_span_item()
    on_span = _FakeHover(0.5)
    item.hoverEvent(on_span)
    assert on_span.claimed == [Qt.RightButton]

    in_gap = _FakeHover(1.5)
    item.hoverEvent(in_gap)
    assert in_gap.claimed == [], "a gap must never claim the right button"

    on_exit = _FakeHover(0.5, exit_=True)
    item.hoverEvent(on_exit)
    assert on_exit.claimed == [], "isExit() must be a no-op, never a claim"


def test_right_click_on_a_span_accepts_and_emits_type_requested(qapp):
    item = _two_span_item()
    received = []
    item.typeRequested.connect(lambda n, pos: received.append((n, pos)))
    ev = _FakeClick(0.5, Qt.RightButton)
    item.mouseClickEvent(ev)
    assert ev.accepted is True and ev.ignored is False
    assert received == [(1, ev.scenePos())]


def test_right_click_in_a_gap_is_ignored_so_pyqtgraphs_own_menu_can_show(qapp):
    """Acceptance criterion: 'a right-click hitting no span reaches pyqtgraph's own
    menu' — this item must ev.ignore() rather than ev.accept() so ViewBox's own
    mouseClickEvent (menuEnabled() still True) gets its turn."""
    item = _two_span_item()
    received = []
    item.typeRequested.connect(lambda n, pos: received.append(n))
    ev = _FakeClick(1.5, Qt.RightButton)          # 1.5 is the gap between the two spans
    item.mouseClickEvent(ev)
    assert ev.accepted is False and ev.ignored is True
    assert received == []


def test_plain_left_click_is_ignored_by_the_item_itself(qapp):
    """A plain left click must flow to the existing scene-level toggle path
    unaffected — the item itself never claims or accepts it, exactly as before this
    ticket (it had no mouseClickEvent at all)."""
    item = _two_span_item()
    received = []
    item.typeRequested.connect(lambda n, pos: received.append(n))
    ev = _FakeClick(0.5, Qt.LeftButton)
    item.mouseClickEvent(ev)
    assert ev.accepted is False and ev.ignored is True
    assert received == []


def test_ctrl_left_click_on_a_span_accepts_and_emits_type_requested(qapp):
    item = _two_span_item()
    received = []
    item.typeRequested.connect(lambda n, pos: received.append(n))
    ev = _FakeClick(2.5, Qt.LeftButton, modifiers=Qt.ControlModifier)
    item.mouseClickEvent(ev)
    assert ev.accepted is True
    assert received == [2]


def test_ctrl_left_click_in_a_gap_is_ignored(qapp):
    item = _two_span_item()
    ev = _FakeClick(1.5, Qt.LeftButton, modifiers=Qt.ControlModifier)
    item.mouseClickEvent(ev)
    assert ev.accepted is False and ev.ignored is True


def test_a_click_on_an_item_detached_from_its_scene_is_ignored(qapp):
    """Self-review finding: pyqtgraph's acceptClicks() claim lives on the SCENE and
    only refreshes on a later mouse move, not on this item leaving the scene — a
    re-render that tears this exact item down between a hover and its following
    click can otherwise still deliver the click to now-detached, meaningless
    geometry. A real pg.PlotWidget is needed here: a bare BreathSpansItem never had
    a scene at all, which would pass this guard for the wrong reason."""
    pw = pg.PlotWidget()
    pi = pw.getPlotItem()
    item = BreathSpansItem()
    item.set_spans([(0.0, 1.0, pg.mkBrush(0, 0, 0), 1)])
    pi.addItem(item)
    assert item.scene() is not None
    pi.removeItem(item)
    assert item.scene() is None

    ev = _FakeClick(0.5, Qt.RightButton)
    item.mouseClickEvent(ev)
    assert ev.accepted is False and ev.ignored is True


# --------------------------------------------------------------------------- #
# The _set_breath_type funnel (run-lock, mutual exclusivity, t_onset_s)
# --------------------------------------------------------------------------- #
def test_set_breath_type_is_locked_while_a_run_is_active_and_says_why(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))

    pv.set_run_active(True)
    assert pv._set_breath_type(a_breath, "rest") is None
    assert s.processing.breath_types == []
    assert s.processing.exclude_breaths == []
    bar_msg = win.statusBar().currentMessage().lower()
    assert "locked" in bar_msg and "run" in bar_msg
    win.close()


def test_set_breath_type_is_a_no_op_for_a_breath_the_current_render_does_not_know(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    bogus = max(pv._breath_spans) + 1000
    assert pv._set_breath_type(bogus, "rest") is None
    assert s.processing.breath_types == []
    win.close()


def test_typing_a_breath_removes_any_existing_exclusion_and_stores_onset(qapp, tmp_path):
    """M-19's model: a typed breath is excluded from the tidal average via
    processing.breath_types, never processing.exclude_breaths — typing must therefore
    remove any prior manual exclusion of the SAME breath in the same write."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))

    assert pv._toggle_breath(a_breath) is True          # excluded first
    assert any(a_breath in e.breaths for e in s.processing.exclude_breaths)

    result = pv._set_breath_type(a_breath, "rest")
    assert result == "rest"
    assert all(a_breath not in e.breaths for e in s.processing.exclude_breaths)
    entry = next(t for t in s.processing.breath_types
                if t.file == pv.file_rail.current_filename() and t.breath == a_breath)
    assert entry.kind == "rest"
    # t_onset_s is stored in the RECORDING's own clock, not the trimmed window's —
    # self._breath_spans is zero-based at the trim start, so + _trim_offset_s
    # recovers the same absolute clock breath['time'][0]/the EMG views already use.
    t0, _t1 = pv._breath_spans[a_breath]
    assert entry.t_onset_s == pytest.approx(t0 + pv._trim_offset_s)
    win.close()


def test_typing_a_breath_updates_the_rails_typed_badge(qapp, tmp_path):
    """M-32: _set_breath_type calls the wide _sync_rail_breath_state (replacing the old,
    narrower _sync_excluded_badge) — this is the funnel's own regression test for that
    wiring, would fail if a future edit routed the write back through a narrower,
    exclusion-only sync that never touches typed_counts."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    name = pv.file_rail.current_filename()

    assert pv.file_rail.entry(name).typed_counts == {}
    assert pv._set_breath_type(a_breath, "rest") == "rest"
    assert pv.file_rail.entry(name).typed_counts == {"rest": 1}

    # tidal clears it back out again
    assert pv._set_breath_type(a_breath, "tidal") == "tidal"
    assert pv.file_rail.entry(name).typed_counts == {}
    win.close()


def test_excluding_a_typed_breath_removes_the_type_entry(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    name = pv.file_rail.current_filename()

    assert pv._set_breath_type(a_breath, "rest") == "rest"
    assert pv._set_breath_type(a_breath, "excluded") == "excluded"
    assert all(t.breath != a_breath for t in s.processing.breath_types if t.file == name)
    entry = next(e for e in s.processing.exclude_breaths if e.file == name)
    assert a_breath in entry.breaths
    win.close()


def test_setting_tidal_clears_both_exclusion_and_typing(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))

    assert pv._set_breath_type(a_breath, "rest") == "rest"
    result = pv._set_breath_type(a_breath, "tidal")
    assert result == "tidal"
    assert s.processing.breath_types == []
    assert s.processing.exclude_breaths == []
    win.close()


def test_a_breath_is_never_both_typed_and_excluded_across_every_transition(qapp, tmp_path):
    """settings.validate()'s 'both typed and excluded' invariant must never actually
    be reachable from this funnel, for every pairwise transition among the three
    states this ticket's minimal menu offers."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    name = pv.file_rail.current_filename()

    def _is_excluded():
        return any(a_breath in e.breaths for e in s.processing.exclude_breaths if e.file == name)

    def _is_typed():
        return any(t.breath == a_breath for t in s.processing.breath_types if t.file == name)

    for sequence in (("excluded", "rest", "tidal", "excluded"),
                     ("rest", "excluded", "rest", "tidal")):
        for kind in sequence:
            pv._set_breath_type(a_breath, kind)
            assert not (_is_excluded() and _is_typed()), (
                f"breath {a_breath} is both excluded and typed after setting {kind!r}")
    win.close()


def test_toggle_breath_still_returns_the_same_bool_none_contract(qapp, tmp_path):
    """_toggle_breath is reimplemented on top of _set_breath_type (M-20) but must keep
    its exact external contract: True/False/None, unaffected by the new three-way
    model — this is a regression guard, not new behaviour."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))

    assert pv._toggle_breath(a_breath) is True
    assert pv._toggle_breath(a_breath) is False
    assert pv._toggle_breath(max(pv._breath_spans) + 1000) is None
    win.close()


# --------------------------------------------------------------------------- #
# The minimal type menu (Tidal / Excluded / Rest) — M-20's own scope
# --------------------------------------------------------------------------- #
def test_build_type_menu_has_the_minimal_labels_and_no_lone_ampersand(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from _helpers import _lone_ampersands
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    pv._suggested_fvc = None   # deterministic: no hint competing with the minimal set

    from PySide6.QtWidgets import QMenu
    menu = pv._build_type_menu(a_breath, ("tidal", "excluded", "rest"))
    # M-31/M-37: the caller's kinds are still exactly Tidal/Excluded/Rest, but the menu
    # now always appends the reference actions behind a separator (text() == '').
    # 'Use as IC reference for' is a real submenu now (Qt draws its own arrow, so the
    # action's own text carries no manual '▸').
    assert [a.text() for a in menu.actions()] == [
        "Tidal", "Excluded", "Rest", "", "Use as IC reference for", "Reference manoeuvres…"]
    # Self-review finding: an unparented menu with these exact (ampersand-free) labels
    # would pass an "offenders is empty" check for the wrong reason. Prove the menu is
    # actually IN the tree _lone_ampersands walks, not just that its own text is clean.
    assert menu in win.findChildren(QMenu), "the menu must be parented under the window"
    assert not _lone_ampersands(win), "the type menu must be reachable from the window's own scan"
    menu.close()
    win.close()


def test_choosing_rest_from_the_menu_sets_the_type_and_status(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    name = pv.file_rail.current_filename()

    menu = pv._build_type_menu(a_breath, ("tidal", "excluded", "rest"))
    rest_action = next(a for a in menu.actions() if a.text() == "Rest")
    rest_action.trigger()
    entry = next(t for t in s.processing.breath_types if t.file == name and t.breath == a_breath)
    assert entry.kind == "rest"
    assert "rest" in pv.status.text().lower()
    win.close()


# --------------------------------------------------------------------------- #
# M-31: the full manoeuvre menu — IC/FVC/max/sniff kinds, the Suggested-FVC hint,
# the M-37 placeholders, and 'Rest' restricted to an EMG-only file.
# --------------------------------------------------------------------------- #
def test_build_type_menu_offers_the_full_manoeuvre_set_with_no_lone_ampersand(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.screens.preview._mechanics import _TYPE_MENU_KINDS
    from _helpers import _lone_ampersands
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    pv._suggested_fvc = None

    menu = pv._build_type_menu(a_breath, _TYPE_MENU_KINDS)
    texts = [a.text() for a in menu.actions()]
    assert texts == [
        "Tidal", "Excluded", "IC manoeuvre", "FVC manoeuvre", "IC + FVC",
        "Maximal inspiratory effort", "Sniff", "Rest", "Other…",
        "", "Use as IC reference for", "Reference manoeuvres…"]
    assert not _lone_ampersands(win)
    menu.close()
    win.close()


def test_the_reference_submenu_is_disabled_until_the_breath_is_typed_ic(qapp, tmp_path):
    """M-37: 'Use as IC reference for' only makes sense once THIS breath is already
    typed 'ic'/'ic_fvc' -- an untyped (or excluded/tidal) breath leaves it disabled,
    with a statusTip saying so. 'Reference manoeuvres…' (the full cross-file picker)
    has no such precondition and stays enabled regardless."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))

    menu = pv._build_type_menu(a_breath, ("tidal", "excluded"))
    ref_for = next(a for a in menu.actions() if a.text() == "Use as IC reference for")
    ref_manoeuvres = next(a for a in menu.actions() if a.text() == "Reference manoeuvres…")
    assert ref_for.isEnabled() is False
    assert ref_manoeuvres.isEnabled() is True
    win.close()


def test_the_reference_submenu_enables_once_typed_ic_and_sets_the_reference(qapp, tmp_path):
    """M-37 round trip (the ticket's own acceptance criterion): type a breath IC, then
    'Use as IC reference for ▸ This file' links THIS file to its own breath via
    processing.references, with folder stamped on the brand-new entry."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    name = pv.file_rail.current_filename()

    assert pv._set_breath_type(a_breath, "ic") == "ic"
    menu = pv._build_type_menu(a_breath, ("tidal", "excluded"))
    ref_for = next(a for a in menu.actions() if a.text() == "Use as IC reference for")
    assert ref_for.isEnabled() is True
    this_file = next(a for a in ref_for.menu().actions() if a.text() == "This file")
    this_file.trigger()

    entry = next(e for e in s.processing.references if e.file == name)
    assert entry.ic.file == name and entry.ic.breaths == [a_breath]
    assert entry.folder == s.input.folder
    assert "reference for this file" in pv.status.text().lower()
    win.close()


def test_choosing_ic_from_the_menu_sets_the_type_and_status(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    name = pv.file_rail.current_filename()

    menu = pv._build_type_menu(a_breath, ("tidal", "excluded", "ic", "fvc"))
    ic_action = next(a for a in menu.actions() if a.text() == "IC manoeuvre")
    ic_action.trigger()
    entry = next(t for t in s.processing.breath_types if t.file == name and t.breath == a_breath)
    assert entry.kind == "ic"
    assert "ic manoeuvre" in pv.status.text().lower()


# --------------------------------------------------------------------------- #
# M-37: _set_reference's three scopes, the run lock, and the reference chip
# --------------------------------------------------------------------------- #
def test_set_reference_group_scope_writes_a_group_reference_default(qapp, tmp_path):
    from respmech.core.summary import group_key
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    name = pv.file_rail.current_filename()
    group = group_key(name, s)

    assert pv._set_breath_type(a_breath, "ic_fvc") == "ic_fvc"
    assert pv._set_reference(name, a_breath, "group") == "group"
    entry = next(e for e in s.processing.reference_defaults if e.group == group)
    assert entry.ic.file == name and entry.ic.breaths == [a_breath]
    assert entry.folder == s.input.folder
    win.close()


def test_set_reference_all_scope_points_every_matched_file_at_the_breath(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.validation import matching_files
    import os as _os
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    name = pv.file_rail.current_filename()
    expected = {_os.path.basename(f) for f in matching_files(s.input.folder, s.input.files)}
    assert len(expected) > 1, "the synthetic input set must have more than one file"

    assert pv._set_breath_type(a_breath, "ic") == "ic"
    assert pv._set_reference(name, a_breath, "all") == "all"
    targets = {e.file for e in s.processing.references if e.ic is not None}
    assert targets == expected
    entries = [e for e in s.processing.references if e.ic is not None]
    for e in entries:
        assert e.ic.file == name and e.ic.breaths == [a_breath]
    # Self-review finding: an earlier version handed every entry the SAME BreathRef
    # instance (and its same mutable .breaths list), so mutating one file's reference
    # later silently corrupted every other file's too. Each entry's .ic (and its
    # .breaths list) must be its OWN object -- proving VALUES agree (above) is not
    # enough to catch aliasing, so this checks identity directly.
    ic_refs = [e.ic for e in entries]
    assert len({id(r) for r in ic_refs}) == len(ic_refs), "every entry shares one BreathRef"
    assert len({id(r.breaths) for r in ic_refs}) == len(ic_refs), "breaths lists are aliased"
    ic_refs[0].breaths.append(999)
    assert all(e.ic.breaths == [a_breath] for e in entries[1:]), (
        "mutating one file's reference breaths corrupted another file's")
    win.close()


def test_set_reference_and_reference_picker_are_blocked_while_a_run_is_active(
        qapp, tmp_path, monkeypatch):
    """The run-lock check must happen BEFORE any dialog work, not just before writing —
    proven by spying on ReferencePickerDialog itself, since 'write_action_blocked fired
    twice' alone would pass identically even if a dialog were built (just wastefully,
    or worse, incorrectly) ahead of the guard (self-review finding)."""
    from respmech.ui.main_window import MainWindow
    import respmech.ui.reference_picker_dialog as _rpd
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    name = pv.file_rail.current_filename()
    pv._set_breath_type(a_breath, "ic")

    built = []
    monkeypatch.setattr(_rpd.ReferencePickerDialog, "__init__",
                        lambda self, *a, **kw: built.append(a))
    blocked = []
    pv.write_action_blocked.connect(blocked.append)
    pv.set_run_active(True)
    assert pv._set_reference(name, a_breath, "file") is None
    assert not s.processing.references
    pv._open_reference_picker(name)          # must return without ever building a dialog
    assert len(blocked) == 2
    assert built == [], "ReferencePickerDialog was constructed despite the run lock"
    assert all("locked" in m.lower() for m in blocked)
    pv.set_run_active(False)
    win.close()


def test_open_reference_picker_never_overwrites_an_untouched_multi_breath_reference(
        qapp, tmp_path, monkeypatch):
    """Self-review finding (the original bug, independently confirmed by two review
    passes): an untouched slot must never be written back, even when a DIFFERENT slot
    in the SAME dialog is touched and accepted -- otherwise a legitimate multi-breath
    own-typed-breath IC reference (IcSettings.aggregate combines repeats) is silently
    truncated to the picker's single-breath selection the moment ANY other slot is
    edited and OK is clicked."""
    from PySide6.QtWidgets import QDialog
    from respmech.core.analysis.references import resolve_reference
    from respmech.core.settings import BreathTypeEntry
    from respmech.ui.main_window import MainWindow
    import respmech.ui.reference_picker_dialog as _rpd
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    name = pv.file_rail.current_filename()
    other = next(n for n in pv.file_rail.filenames() if n != name)

    # `name` has THREE typed IC breaths -- resolve_reference's own-typed fallback
    # aggregates all of them (IcSettings.aggregate); the picker's ic ROW can only ever
    # pre-select the first, since its combo is single-breath.
    s.processing.breath_types.extend([
        BreathTypeEntry(file=name, breath=1, kind="ic", t_onset_s=0.1),
        BreathTypeEntry(file=name, breath=2, kind="ic", t_onset_s=0.2),
        BreathTypeEntry(file=name, breath=3, kind="ic", t_onset_s=0.3),
        BreathTypeEntry(file=other, breath=5, kind="fvc", t_onset_s=0.5),
    ])
    assert resolve_reference(name, "ic", s).breaths == [1, 2, 3]

    def _touch_fvc_only_and_accept(self):
        row = self._rows["fvc"]
        row.file_combo.setCurrentIndex(row.file_combo.findData(other))
        row.breath_combo.setCurrentIndex(row.breath_combo.findData(5))
        return QDialog.Accepted

    monkeypatch.setattr(_rpd.ReferencePickerDialog, "exec", _touch_fvc_only_and_accept)
    pv._open_reference_picker(name)

    entry = next(e for e in s.processing.references if e.file == name)
    assert entry.ic is None, "an untouched slot must not gain an explicit entry"
    assert entry.fvc.file == other and entry.fvc.breaths == [5]
    ref = resolve_reference(name, "ic", s)
    assert ref.file == name and ref.breaths == [1, 2, 3], (
        "the untouched ic reference must still resolve to all three typed breaths, "
        "not be truncated by editing a different slot")
    win.close()


def test_the_mech_window_chip_shows_the_resolved_ic_reference(qapp, tmp_path):
    """The ticket's own acceptance criterion: 'chip viser den' after the round trip.
    Checked on BOTH ``fullText()`` (the ElidingLabel's own un-elided source, what the
    chip actually renders from) and ``toolTip()`` (what a hover recovers) — asserting
    only the tooltip would not prove the chip text was ever appended to the label's own
    full text at all, only that the tooltip string happens to contain the substring."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    name = pv.file_rail.current_filename()

    assert "ic ref" not in pv.mech_window_label.fullText().lower()
    assert "ic ref" not in pv.mech_window_label.toolTip().lower()
    assert pv._set_breath_type(a_breath, "ic") == "ic"
    assert pv._set_reference(name, a_breath, "file") == "file"
    expected = f"ic ref: this file (breath {a_breath})"
    assert expected in pv.mech_window_label.fullText().lower()
    assert expected in pv.mech_window_label.toolTip().lower()
    win.close()


def test_the_mech_window_chip_full_text_survives_under_windows_font_metrics(
        qapp, tmp_path, windows_metrics):
    """Acceptance criterion: '...chippen...består windows_metrics-ratiotests'. The
    combined trim-window + reference text can legitimately be too long to show in full
    at the Windows runner's wider advance — ElidingLabel's whole POINT is to shrink the
    DISPLAYED text rather than force the window wider (CLAUDE.md's 'never assert on a
    QLabel's rendered text() when elide shortened it'). The full sentence must still be
    intact and recoverable from fullText()/toolTip() regardless of how narrow the
    granted width ends up, and the label's minimumSizeHint must stay at its small,
    non-forcing floor even with the reference chip appended."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    name = pv.file_rail.current_filename()
    pv._set_breath_type(a_breath, "ic")
    pv._set_reference(name, a_breath, "file")

    full = pv.mech_window_label.fullText()
    assert f"ic ref: this file (breath {a_breath})" in full.lower()
    assert pv.mech_window_label.toolTip().lower().endswith(full.lower())
    # the floor this label reports must not grow just because the string got longer —
    # that is what lets a squeezing layout still give it room, on any font.
    assert pv.mech_window_label.minimumSizeHint().width() <= 24
    win.close()


def test_file_rails_references_requested_opens_the_picker_for_that_row(qapp, tmp_path, monkeypatch):
    """FileRail's own context-menu action (M-32, unconnected until this ticket) must
    reach PreviewScreen._open_reference_picker with the ROW's filename."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    other = next(n for n in pv.file_rail.filenames() if n != pv.file_rail.current_filename())

    seen = []
    monkeypatch.setattr(pv, "_open_reference_picker", lambda filename=None: seen.append(filename))
    pv.file_rail.referencesRequested.emit(other)
    assert seen == [other]
    win.close()
    win.close()


def test_suggested_fvc_hint_is_shown_disabled_for_the_suggested_breath_only(qapp, tmp_path):
    """The hint reads self._suggested_fvc directly — set here rather than depending on
    which synthetic breath actually has the longest expiration, so the test is
    deterministic regardless of the sample data's own shape."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    breaths = sorted(pv._breath_spans)
    suggested, other = breaths[0], breaths[1]
    pv._suggested_fvc = suggested

    menu_hit = pv._build_type_menu(suggested, ("tidal", "excluded"))
    actions = menu_hit.actions()
    hint = next(a for a in actions if a.text() == "Suggested: FVC")
    assert hint.isEnabled() is False
    assert actions[0].text() == "Suggested: FVC", "the hint leads the menu"
    assert actions[1].isSeparator(), "the hint is set off from the real choices below it"

    menu_other = pv._build_type_menu(other, ("tidal", "excluded"))
    assert not any(a.text() == "Suggested: FVC" for a in menu_other.actions())
    win.close()


def test_suggested_fvc_is_computed_from_the_staged_breaths(qapp, tmp_path):
    """stage_mechanics_preview computes the hint from the SAME raw breath dicts
    manoeuvres.suggest_fvc is documented against — end to end through a real render,
    not a hand-set attribute like the test above. Cross-checked against an independent
    computation from the file's own breaths, not just 'is not None'."""
    from respmech.core import compute
    from respmech.core._legacy_ns import to_legacy_ns
    from respmech.core.analysis.manoeuvres import suggest_fvc
    from respmech.ui.workers import stage_mechanics_preview
    s = synth_settings(str(tmp_path))
    data = stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv"))
    assert data["suggested_fvc"] is not None

    # independent expectation: re-derive the same breaths the worker staged, and
    # compute the expected suggestion straight from suggest_fvc's own documented
    # rule (longest untyped, non-ignored expiration) rather than trusting the
    # worker's own number circularly.
    ls = to_legacy_ns(s)
    breaths = compute.separateintobreaths(
        ls.processing.mechanics.separateby, "synth_case_A.csv", data["t"],
        data["series"]["flow"], data["series"]["volume"],
        data["series"].get("poes"), data["series"].get("pgas"), data["series"].get("pdi"),
        [], [], ls)
    assert data["suggested_fvc"] == suggest_fvc(breaths)

    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    assert pv._suggested_fvc == data["suggested_fvc"]
    win.close()


def test_suggested_fvc_hint_is_dropped_the_instant_its_own_breath_is_typed(qapp, tmp_path):
    """suggest_fvc's own contract is 'untyped, non-ignored' — the moment _set_breath_type
    types (or excludes) the SUGGESTED breath, showing the hint on it again would be a
    stale, self-contradicting suggestion until some later, unrelated re-stage happened
    to correct it. Cleared immediately instead (see _set_breath_type's own comment)."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    pv._suggested_fvc = a_breath

    pv._set_breath_type(a_breath, "fvc")
    assert pv._suggested_fvc is None

    menu = pv._build_type_menu(a_breath, ("tidal", "excluded"))
    assert not any(a.text() == "Suggested: FVC" for a in menu.actions())
    win.close()


def test_suggested_fvc_hint_survives_typing_a_DIFFERENT_breath(qapp, tmp_path):
    """The stale-hint fix must not over-clear: typing some OTHER breath leaves an
    existing suggestion for a still-untyped breath alone."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    breaths = sorted(pv._breath_spans)
    suggested, other = breaths[0], breaths[1]
    pv._suggested_fvc = suggested

    pv._set_breath_type(other, "excluded")
    assert pv._suggested_fvc == suggested
    win.close()


def test_handle_type_requested_offers_the_manoeuvre_kinds_for_a_flow_bearing_set(
        qapp, tmp_path, monkeypatch):
    """M-31: 'Rest' names a noise-reference SEGMENT — offering it on a flow-bearing file
    (where breaths, not segments, are the unit) would type a real tidal breath as a
    reference no resolver for that shape ever reads."""
    from respmech.ui.main_window import MainWindow
    from PySide6.QtCore import QPointF as _QPointF
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))

    captured = {}

    def _capture(breath_no, kinds):
        captured["kinds"] = kinds
        return QMenu(pv.plots)
    monkeypatch.setattr(pv, "_build_type_menu", _capture)
    pv._handle_type_requested(a_breath, _QPointF(0.0, 0.0))
    assert "rest" not in captured["kinds"], "flow-bearing set must not offer Rest"
    assert {"ic", "fvc", "ic_fvc", "max_insp", "sniff", "other"} <= set(captured["kinds"])
    win.close()


def test_handle_type_requested_offers_only_rest_and_other_for_an_emg_only_signal_set(
        qapp, tmp_path, monkeypatch):
    """M-31 self-review finding: an EMG-only segment has no inspiration/expiration split
    for `core.analysis.manoeuvres.extract` to read (`core.pipeline.run_batch` skips
    manoeuvre extraction entirely for `caps.mode == 'emg_only'`) — offering IC/FVC/max/
    sniff there would type a segment as, say, 'ic' with no Manoeuvres row ever appearing
    for it. Only Tidal/Excluded/Rest/Other make sense on that tab."""
    from respmech.ui.main_window import MainWindow
    from PySide6.QtCore import QPointF as _QPointF
    s = synth_settings(str(tmp_path), channels={
        "flow": None, "poes": None, "pgas": None, "pdi": None, "volume": None})
    win = MainWindow(AppState(s)); pv = win.preview_screen

    captured = {}

    def _capture(breath_no, kinds):
        captured["kinds"] = kinds
        return QMenu(pv.plots)
    monkeypatch.setattr(pv, "_build_type_menu", _capture)
    pv._handle_type_requested(1, _QPointF(0.0, 0.0))
    assert set(captured["kinds"]) == {"tidal", "excluded", "rest", "other"}
    win.close()


def test_handle_type_requested_pops_the_menu_without_a_view(qapp, tmp_path):
    """_handle_type_requested must not raise when the emitting item cannot be
    resolved to a view (self.sender() returns None outside a real signal emission) —
    it falls back to QCursor.pos() rather than crash."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    from PySide6.QtCore import QPointF as _QPointF
    pv._handle_type_requested(a_breath, _QPointF(0.0, 0.0))   # must not raise
    win.close()


# --------------------------------------------------------------------------- #
# End-to-end through a REAL pyqtgraph scene (self-review finding: the fake-event
# tests above prove the item's own logic, but not that pyqtgraph's real dispatch
# actually reaches it ahead of ViewBox, or that a gap click actually still reaches
# ViewBox's own menu).
# --------------------------------------------------------------------------- #
def _send_click(widget, view_pt, global_pt, button):
    press = QMouseEvent(QEvent.Type.MouseButtonPress, view_pt, global_pt,
                        button, button, Qt.NoModifier)
    QApplication.sendEvent(widget, press)
    release = QMouseEvent(QEvent.Type.MouseButtonRelease, view_pt, global_pt,
                          button, Qt.NoButton, Qt.NoModifier)
    QApplication.sendEvent(widget, release)


def _hover_then_click(pw, vb, x, button):
    """Move the mouse to the scene x-coordinate ``x`` (arranging the hover pyqtgraph's
    own click resolution depends on, see ui/CLAUDE.md), then click ``button`` there."""
    scene_pt = vb.mapViewToScene(pg.Point(x, vb.viewRect().center().y()))
    view_pt = pw.mapFromScene(scene_pt)
    global_pt = pw.mapToGlobal(view_pt)
    move = QMouseEvent(QEvent.Type.MouseMove, view_pt, global_pt,
                       Qt.NoButton, Qt.NoButton, Qt.NoModifier)
    QApplication.sendEvent(pw.viewport(), move)
    _send_click(pw.viewport(), view_pt, global_pt, button)


def test_a_real_right_click_on_a_span_reaches_the_item_and_viewbox_never_raises(qapp):
    """The mechanism ui/CLAUDE.md documents, proven end to end rather than only at
    the fake-event level: a real pyqtgraph PlotWidget, a real mouse move + click
    delivered through Qt, with ViewBox.raiseContextMenu instrumented to prove it
    was never called even though menuEnabled() (and hence the ordinary z-order
    race BreathSpansItem would otherwise lose) is left completely untouched."""
    pw = pg.PlotWidget()
    pw.resize(400, 300)
    pw.show()
    pi = pw.getPlotItem()
    vb = pi.getViewBox()
    assert vb.menuEnabled()
    item = BreathSpansItem()
    item.set_spans([(0.0, 1.0, pg.mkBrush(0, 0, 0), 7)])
    item.setZValue(-10)                    # the real, paint-order zValue — not raised for the test
    received = []
    item.typeRequested.connect(lambda n, pos: received.append(n))
    pi.addItem(item)
    vb.setXRange(0, 1); vb.setYRange(0, 1)
    qapp.processEvents()

    raised = []
    vb.raiseContextMenu = lambda ev: raised.append(ev)

    _hover_then_click(pw, vb, 0.5, Qt.RightButton)
    qapp.processEvents()

    assert received == [7], "the real scene must deliver the right-click to the item"
    assert raised == [], "ViewBox's own context menu must never raise for a hit span"
    assert vb.menuEnabled(), "the mechanism must not work by disabling the menu"
    pw.close()


def test_a_real_right_click_in_a_gap_still_reaches_viewboxs_own_menu(qapp):
    """Acceptance criterion, end to end: a right-click that hits no span must still
    open pyqtgraph's own context menu, unmodified — proving the hover claim really
    is conditional on hitting a span, through the real dispatch path."""
    pw = pg.PlotWidget()
    pw.resize(400, 300)
    pw.show()
    pi = pw.getPlotItem()
    vb = pi.getViewBox()
    item = BreathSpansItem()
    item.set_spans([(0.0, 0.3, pg.mkBrush(0, 0, 0), 1),
                    (0.7, 1.0, pg.mkBrush(0, 0, 0), 2)])   # a real gap at 0.3-0.7
    item.setZValue(-10)
    received = []
    item.typeRequested.connect(lambda n, pos: received.append(n))
    pi.addItem(item)
    vb.setXRange(0, 1); vb.setYRange(0, 1)
    qapp.processEvents()

    raised = []
    vb.raiseContextMenu = lambda ev: raised.append(ev)

    _hover_then_click(pw, vb, 0.5, Qt.RightButton)         # 0.5 is inside the gap
    qapp.processEvents()

    assert received == []
    assert len(raised) == 1, "a gap click must still reach ViewBox's own menu"
    pw.close()


def test_a_real_ctrl_left_click_on_a_span_reaches_the_item_without_any_hover_claim(qapp):
    """Ctrl+left-click needs no acceptClicks trick (ViewBox never accepts the left
    button regardless of modifiers), so it must work even with NO prior hover at
    all — a direct press at a fresh position."""
    pw = pg.PlotWidget()
    pw.resize(400, 300)
    pw.show()
    pi = pw.getPlotItem()
    vb = pi.getViewBox()
    item = BreathSpansItem()
    item.set_spans([(0.0, 1.0, pg.mkBrush(0, 0, 0), 3)])
    item.setZValue(-10)
    received = []
    item.typeRequested.connect(lambda n, pos: received.append(n))
    pi.addItem(item)
    vb.setXRange(0, 1); vb.setYRange(0, 1)
    qapp.processEvents()

    scene_pt = vb.mapViewToScene(pg.Point(0.5, vb.viewRect().center().y()))
    view_pt = pw.mapFromScene(scene_pt)
    global_pt = pw.mapToGlobal(view_pt)
    press = QMouseEvent(QEvent.Type.MouseButtonPress, view_pt, global_pt,
                        Qt.LeftButton, Qt.LeftButton, Qt.ControlModifier)
    QApplication.sendEvent(pw.viewport(), press)
    release = QMouseEvent(QEvent.Type.MouseButtonRelease, view_pt, global_pt,
                          Qt.LeftButton, Qt.NoButton, Qt.ControlModifier)
    QApplication.sendEvent(pw.viewport(), release)
    qapp.processEvents()

    assert received == [3]
    pw.close()


# --------------------------------------------------------------------------- #
# The EMG views' own overlay must also reflect a typed breath's kind, not just a
# plain exclusion (self-review finding: _paint_breaths' _breath_kind_now path was
# only reachable via the live _repaint_breath update, never via a fresh repaint).
# --------------------------------------------------------------------------- #
def test_emg_raw_view_colours_a_typed_breath_on_a_fresh_repaint(qapp, tmp_path):
    """_repaint_view_breaths -> _paint_breaths -> _breath_kind_now must resolve a
    typed breath's kind from settings on a FRESH repaint, not just carry forward
    whatever the live _set_breath_type repaint already set on the SAME item."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path), remove_ecg=True)
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    assert pv._bov.get("raw", {}).get("regions"), "fixture must have painted the raw EMG view"

    assert pv._set_breath_type(a_breath, "rest") == "rest"
    pv._repaint_view_breaths("raw")            # force a FRESH _paint_breaths, not the live update
    item, idx = pv._bov["raw"]["regions"][a_breath][0]
    assert item._spans[idx][2].color().getRgb() == pv._breath_brush("rest").color().getRgb()
    win.close()


def test_a_typed_breath_survives_a_fresh_mechanics_restage(qapp, tmp_path):
    """workers.stage_mechanics_preview's kind computation (b['kind'] or the
    'excluded' pseudo-kind or None) must actually reflect a persisted
    BreathTypeEntry when the file is re-staged from scratch — not just when the
    live overlay is incrementally repainted."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_mechanics_preview
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    assert pv._set_breath_type(a_breath, "rest") == "rest"

    fresh = stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv"))
    kind = next(k for (n, _t0, _t1, k) in fresh["spans"] if n == a_breath)
    assert kind == "rest"
    win.close()


# --------------------------------------------------------------------------- #
# _exclude_key's new breath_types half (self-review finding: only the
# pre-existing exclude_breaths half had a cache-key regression test).
# --------------------------------------------------------------------------- #
def test_exclude_key_changes_when_breath_types_changes(qapp, tmp_path):
    from respmech.core.settings import BreathTypeEntry
    from respmech.ui.screens._preview_cache import _exclude_key
    s = synth_settings(str(tmp_path))
    before = _exclude_key(s)
    s.processing.breath_types.append(BreathTypeEntry(file="a.csv", breath=1, kind="rest"))
    after_add = _exclude_key(s)
    assert after_add != before, "adding a breath_types entry must change the cache key"

    s.processing.breath_types[0].kind = "ic"
    after_kind_change = _exclude_key(s)
    assert after_kind_change != after_add, "changing a typed breath's kind must change the cache key"


# --------------------------------------------------------------------------- #
# The breath's type as text under its number
# --------------------------------------------------------------------------- #
def test_breath_label_text_carries_the_type_on_a_second_line():
    from respmech.ui.screens.preview._mechanics import _MechanicsMixin as M
    assert M._breath_label(7) == "#7"                      # plain tidal: unchanged
    assert M._breath_label(7, None) == "#7"
    assert M._breath_label(3, "fvc") == "#3\nFVC"
    assert M._breath_label(3, "excluded") == "#3\n(Excluded)"
    assert M._breath_label(3, "ic_fvc") == "#3\nIC+FVC"
    # compact stand-in, and an unknown kind falls back to 'other' rather than raising
    assert M._breath_label(3, "fvc", compact=True) == "#3\nF"
    assert M._breath_label(3, "excluded", compact=True) == "#3\n×"
    assert M._breath_label(3, "not-a-kind") == "#3\nOther"


def _label_text(pv, n):
    return pv._breath_texts[n].textItem.toPlainText()


def test_typing_and_excluding_a_breath_retags_its_label_live(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    n = next(iter(pv._breath_spans))
    assert _label_text(pv, n) == f"#{n}"

    assert pv._set_breath_type(n, "fvc") == "fvc"
    assert _label_text(pv, n).split("\n")[0] == f"#{n}"
    assert _label_text(pv, n).split("\n")[1] in ("FVC", "F")   # full text or compact, per zoom

    assert pv._toggle_breath(n) is True                        # excluded (a typed breath is retyped)
    assert _label_text(pv, n).split("\n")[1] in ("(Excluded)", "×")

    assert pv._set_breath_type(n, "tidal") == "tidal"
    assert _label_text(pv, n) == f"#{n}"
    win.close()


def test_label_swaps_to_the_marker_when_its_span_is_too_narrow_on_screen(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    n = next(iter(pv._breath_spans))
    pv._set_breath_type(n, "fvc")
    txt = pv._breath_texts[n]
    st = txt._rm_label
    vb = txt.getViewBox()

    # pretend a tiny span (as when zoomed far out) and a laid-out view
    st["span"] = (0.0, 0.001)
    st["full_px"] = 40.0
    vb.setXRange(0.0, 100.0, padding=0)
    pv._refresh_breath_label(txt)
    if vb.width() > 0:
        assert txt.textItem.toPlainText() == f"#{n}\nF"
        # wide span again -> full text
        st["span"] = (0.0, 100.0)
        pv._refresh_breath_label(txt)
        assert txt.textItem.toPlainText() == f"#{n}\nFVC"
    win.close()


def test_x_zoom_re_evaluates_labels_through_the_pin_slot(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    n = next(iter(pv._breath_spans))
    pv._set_breath_type(n, "ic")
    txt = pv._breath_texts[n]
    vb = txt.getViewBox()
    if vb.width() <= 0:
        pytest.skip("view has no laid-out width in this environment")
    t0, t1 = pv._breath_spans[n]
    st = txt._rm_label
    st["full_px"] = vb.width() * 0.5               # needs half the view to fit its text
    vb.setXRange(t0, t1, padding=0)                # zoomed in on the breath: span fills the view
    assert txt.textItem.toPlainText() == f"#{n}\nIC"
    vb.setXRange(t0 - 1000, t1 + 1000, padding=0)  # zoomed far out: span is a sliver
    assert txt.textItem.toPlainText() == f"#{n}\nI"
    win.close()
