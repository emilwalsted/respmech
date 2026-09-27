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
from PySide6.QtCore import QPointF, Qt

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
    item = BreathSpansItem()
    item.set_spans([(0.0, 1.0, pg.mkBrush(0, 0, 0), 1),
                    (2.0, 3.0, pg.mkBrush(0, 0, 0), 2)])   # a real gap between 1.0 and 2.0
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
    t0, _t1 = pv._breath_spans[a_breath]
    assert entry.t_onset_s == pytest.approx(t0)
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
# The minimal type menu (Tidal / Excluded / Rest)
# --------------------------------------------------------------------------- #
def test_build_type_menu_has_the_minimal_labels_and_no_lone_ampersand(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from _helpers import _lone_ampersands
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))

    menu = pv._build_type_menu(a_breath, ("tidal", "excluded", "rest"))
    assert [a.text() for a in menu.actions()] == ["Tidal", "Excluded", "Rest"]
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
