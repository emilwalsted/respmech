"""M-27: manual separators in Preview (placement, removal, renumbering, the run-lock
and the removal tolerance) — the EMG-only 'EMG – segments' tab's own new action, and
its shared click routing through ``_toggle_from_emg_click`` (``_emg_noise.py``).

``SeparatorLinesItem`` (``_plot_helpers.py``) is exercised directly, with no Qt scene
needed for its ``dataBounds`` contract. The button, the funnel and the renumbering are
exercised through a real ``MainWindow``/``PreviewScreen``, the same pattern
``test_breath_typing_ui.py``/``test_emg_only_preview.py`` already use for the sibling
click primitives.
"""
import os

import pytest
from PySide6.QtCore import QPointF, Qt

from respmech.core.settings import BreathTypeEntry, ExcludeEntry, SeparatorEntry
from respmech.ui.screens.preview._plot_helpers import SeparatorLinesItem
from respmech.ui.state import AppState
from respmech.ui.workers import stage_emg_segments_preview

from _helpers import INPUT, requires_synth, synth_settings

pytestmark = requires_synth()

FILENAME = "synth_case_A.csv"


def _emg_only_settings(tmp_path, method="separators", separator_times=None):
    s = synth_settings(tmp_path, channels={
        "flow": None, "poes": None, "pgas": None, "pdi": None, "volume": None})
    s.processing.segmentation.method = method
    if separator_times is not None:
        s.processing.segmentation.separators.append(
            SeparatorEntry(file=FILENAME, times_s=list(separator_times)))
    return s


def _path(settings):
    return os.path.join(settings.input.folder, FILENAME)


def _render(pv, s):
    data = stage_emg_segments_preview(pv.state.settings, _path(s))
    pv._render_segments_preview(data)


# --------------------------------------------------------------------------- #
# SeparatorLinesItem — the Qt-free-enough dataBounds contract
# --------------------------------------------------------------------------- #

def test_databounds_is_none_for_the_y_axis_regardless_of_content(qapp):
    item = SeparatorLinesItem()
    assert item.dataBounds(1) is None
    item.set_times([1.0, 2.5, 0.5])
    assert item.dataBounds(1) is None


def test_databounds_x_axis_tracks_the_times_or_is_none_when_empty(qapp):
    item = SeparatorLinesItem()
    assert item.dataBounds(0) is None
    item.set_times([2.0, 0.5, 1.0])
    assert item.dataBounds(0) == (0.5, 2.0)


def test_set_times_sorts_and_never_claims_mouse_buttons(qapp):
    item = SeparatorLinesItem()
    item.set_times([3.0, 1.0])
    assert item._times == [1.0, 3.0]
    assert item.acceptedMouseButtons() == Qt.NoButton


# --------------------------------------------------------------------------- #
# The 'Place separators' button: whole_file disables it with the ticket's own wording
# --------------------------------------------------------------------------- #

def test_button_is_disabled_with_the_full_sentence_in_whole_file_mode(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="whole_file")
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    assert pv.btn_place_separators.isEnabled() is False
    assert pv.btn_place_separators.toolTip() == (
        "Whole-file mode has one segment — choose Manual separators in Setup ▸ "
        "Signals ▸ Change… to place separators.")
    win.close()


def test_button_is_enabled_in_separators_mode(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0])
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    assert pv.btn_place_separators.isEnabled() is True
    win.close()


def test_switching_to_whole_file_unarms_and_disables_the_button(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0])
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv.btn_place_separators.setChecked(True)
    assert pv._separators_armed is True

    pv.state.settings.processing.segmentation.method = "whole_file"
    pv.sync_from_settings()
    assert pv.btn_place_separators.isEnabled() is False
    assert pv.btn_place_separators.isChecked() is False
    assert pv._separators_armed is False
    win.close()


# --------------------------------------------------------------------------- #
# _toggle_separator_at: add vs. remove, and the pixel-scaled tolerance
# --------------------------------------------------------------------------- #

class _FakeVB:
    """A fake ViewBox exposing only ``viewPixelSize()`` — the one thing the tolerance
    calculation reads (``vb.viewPixelSize()[0] * 6``), so the tolerance rule is
    unit-testable without any real Qt scene geometry."""

    def __init__(self, px_x=0.01):
        self._px_x = px_x

    def viewPixelSize(self):
        return (self._px_x, 1.0)


def test_a_click_away_from_any_existing_separator_places_a_new_one(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0])
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._toggle_separator_at(FILENAME, 2.5, _FakeVB())
    entry = next(e for e in pv.state.settings.processing.segmentation.separators
                if e.file == FILENAME)
    assert entry.times_s == [1.0, 2.5]
    assert "placed" in pv.status.text()
    win.close()


def test_placing_a_separator_updates_the_rails_segment_count(qapp, tmp_path):
    """M-32: _toggle_separator_at calls the wide _sync_rail_breath_state, which is the
    only place FileRailEntry.segments gets computed — 1 separator makes 2 segments."""
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators")
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    assert pv.file_rail.entry(FILENAME).segments == 1        # 0 separators -> 1 segment
    pv._toggle_separator_at(FILENAME, 1.0, _FakeVB())
    assert pv.file_rail.entry(FILENAME).segments == 2         # 1 separator -> 2 segments
    pv._toggle_separator_at(FILENAME, 2.5, _FakeVB())
    assert pv.file_rail.entry(FILENAME).segments == 3
    win.close()


def test_a_click_within_tolerance_of_an_existing_separator_removes_it(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0, 2.0])
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    # tol = 0.01 * 6 = 0.06; 1.0 + 0.03 is well within it
    pv._toggle_separator_at(FILENAME, 1.03, _FakeVB(px_x=0.01))
    entry = next(e for e in pv.state.settings.processing.segmentation.separators
                if e.file == FILENAME)
    assert entry.times_s == [2.0]
    assert "removed" in pv.status.text()
    win.close()


def test_a_click_just_outside_tolerance_places_instead_of_removing(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0])
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    # tol = 0.01 * 6 = 0.06; 1.1 is well outside it
    pv._toggle_separator_at(FILENAME, 1.1, _FakeVB(px_x=0.01))
    entry = next(e for e in pv.state.settings.processing.segmentation.separators
                if e.file == FILENAME)
    assert entry.times_s == [1.0, 1.1]
    assert "placed" in pv.status.text()
    win.close()


def test_creating_a_brand_new_separator_entry_stamps_the_folder_only_once(qapp, tmp_path):
    """Folder-stamp-only-on-creation (the same B06 rule ExcludeEntry/BreathTypeEntry
    already follow): a brand-new SeparatorEntry gets the current input folder; editing
    it again afterwards must never re-stamp it."""
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators")
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._toggle_separator_at(FILENAME, 1.0, _FakeVB())
    entry = next(e for e in pv.state.settings.processing.segmentation.separators
                if e.file == FILENAME)
    assert entry.folder == s.input.folder

    entry.folder = "/some/other/folder"          # simulate a carried-over entry
    pv._toggle_separator_at(FILENAME, 2.0, _FakeVB())
    assert entry.folder == "/some/other/folder", "a plain edit must never re-stamp folder"
    win.close()


# --------------------------------------------------------------------------- #
# _set_separators: the renumbering itself (add/remove/renumber, the ticket's own
# literal acceptance sequence, and BreathTypeEntry alongside ExcludeEntry)
# --------------------------------------------------------------------------- #

def test_exclusion_follows_a_segment_through_an_insertion_before_it(qapp, tmp_path):
    """The ticket's own literal acceptance sequence: 3 separators (4 segments) ->
    exclude segment 3 -> insert a separator before it -> the exclusion follows to the
    new segment 4."""
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0, 2.0, 3.5])
    s.processing.exclude_breaths.append(ExcludeEntry(file=FILENAME, breaths=[3]))
    win = MainWindow(AppState(s))
    pv = win.preview_screen

    pv._set_separators(FILENAME, [1.0, 1.5, 2.0, 3.5])   # inserted at 1.5, before segment 3
    entry = next(e for e in pv.state.settings.processing.exclude_breaths if e.file == FILENAME)
    assert entry.breaths == [4]
    win.close()


def test_exclusion_of_a_later_untouched_segment_is_unaffected_by_an_earlier_insertion(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0, 2.0])
    s.processing.exclude_breaths.append(ExcludeEntry(file=FILENAME, breaths=[1]))
    win = MainWindow(AppState(s))
    pv = win.preview_screen

    pv._set_separators(FILENAME, [1.0, 1.5, 2.0])
    entry = next(e for e in pv.state.settings.processing.exclude_breaths if e.file == FILENAME)
    assert entry.breaths == [1]
    win.close()


def test_removing_a_separator_merges_the_exclusion_into_the_preceding_segment(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0, 2.0, 3.5])
    s.processing.exclude_breaths.append(ExcludeEntry(file=FILENAME, breaths=[3]))
    win = MainWindow(AppState(s))
    pv = win.preview_screen

    pv._set_separators(FILENAME, [1.0, 3.5])             # the boundary at 2.0 removed
    entry = next(e for e in pv.state.settings.processing.exclude_breaths if e.file == FILENAME)
    assert entry.breaths == [2]                          # segment 3 folded into segment 2
    win.close()


def test_breath_type_entry_is_renumbered_and_its_onset_re_anchored(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0, 2.0, 3.5])
    s.processing.breath_types.append(
        BreathTypeEntry(file=FILENAME, breath=3, kind="rest", t_onset_s=2.0))
    win = MainWindow(AppState(s))
    pv = win.preview_screen

    pv._set_separators(FILENAME, [1.0, 1.5, 2.0, 3.5])
    entry = next(t for t in pv.state.settings.processing.breath_types if t.file == FILENAME)
    assert entry.breath == 4
    assert entry.kind == "rest"
    assert entry.t_onset_s == 2.0        # still the segment's own (unchanged) start
    win.close()


def test_a_merge_collision_between_two_typed_breaths_keeps_exactly_one_entry(qapp, tmp_path):
    """Settings.validate() forbids two BreathTypeEntry rows for the same (file, breath);
    a removal that merges two previously distinct typed segments must resolve that
    itself rather than leave an invalid pair behind."""
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0, 2.0, 3.5])
    s.processing.breath_types.append(
        BreathTypeEntry(file=FILENAME, breath=2, kind="ic", t_onset_s=1.0))
    s.processing.breath_types.append(
        BreathTypeEntry(file=FILENAME, breath=3, kind="rest", t_onset_s=2.0))
    win = MainWindow(AppState(s))
    pv = win.preview_screen

    pv._set_separators(FILENAME, [1.0, 3.5])             # 2.0 removed -> 2 and 3 merge
    entries = [t for t in pv.state.settings.processing.breath_types if t.file == FILENAME]
    assert len(entries) == 1
    assert entries[0].breath == 2
    assert entries[0].kind == "ic"        # the lower original breath number's entry wins
    pv.state.settings.validate()           # must not raise ("typed more than once")
    win.close()


def test_a_merge_collision_between_an_exclusion_and_a_typed_breath_keeps_the_typed_one(qapp, tmp_path):
    """The other half of Settings.validate()'s invariant ('both typed and excluded'):
    a removal that merges a previously EXCLUDED segment and a previously TYPED one onto
    the same new number must resolve that too, not just a typed-vs-typed collision.
    A typed breath already carries the same tidal-average exclusion a plain exclusion
    does (plus a kind), so the typed entry wins and the plain exclusion is dropped."""
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0, 2.0, 3.5])
    s.processing.exclude_breaths.append(ExcludeEntry(file=FILENAME, breaths=[2]))
    s.processing.breath_types.append(
        BreathTypeEntry(file=FILENAME, breath=3, kind="rest", t_onset_s=2.0))
    win = MainWindow(AppState(s))
    pv = win.preview_screen

    pv._set_separators(FILENAME, [1.0, 3.5])             # 2.0 removed -> 2 and 3 merge
    excl_entries = [e for e in pv.state.settings.processing.exclude_breaths if e.file == FILENAME]
    typed_entries = [t for t in pv.state.settings.processing.breath_types if t.file == FILENAME]
    assert excl_entries == [], "the plain exclusion must be dropped, not left colliding"
    assert len(typed_entries) == 1
    assert typed_entries[0].breath == 2
    assert typed_entries[0].kind == "rest"
    pv.state.settings.validate()          # must not raise ("both typed and excluded")
    win.close()


def test_an_exclusion_untouched_by_any_merge_survives_alongside_an_unrelated_typed_breath(qapp, tmp_path):
    """The new exclusion/typed collision guard must not over-trigger: an exclusion and a
    typed breath that do NOT collide (different segments throughout) must both survive
    a separator edit unchanged in kind, only renumbered as normal."""
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0, 2.0, 3.5])
    s.processing.exclude_breaths.append(ExcludeEntry(file=FILENAME, breaths=[1]))
    s.processing.breath_types.append(
        BreathTypeEntry(file=FILENAME, breath=4, kind="rest", t_onset_s=3.5))
    win = MainWindow(AppState(s))
    pv = win.preview_screen

    pv._set_separators(FILENAME, [1.0, 1.5, 2.0, 3.5])   # inserted at 1.5, no merge at all
    excl_entry = next(e for e in pv.state.settings.processing.exclude_breaths if e.file == FILENAME)
    typed_entry = next(t for t in pv.state.settings.processing.breath_types if t.file == FILENAME)
    assert excl_entry.breaths == [1]
    assert typed_entry.breath == 5 and typed_entry.kind == "rest"
    pv.state.settings.validate()
    win.close()


def test_set_separators_never_restamps_an_existing_entrys_folder(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0])
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    entry = next(e for e in pv.state.settings.processing.segmentation.separators
                if e.file == FILENAME)
    entry.folder = "/carried/over"
    pv._set_separators(FILENAME, [1.0, 2.0])
    assert entry.folder == "/carried/over"
    win.close()


# --------------------------------------------------------------------------- #
# The run-lock: a click during a run is refused with a full status-bar sentence
# --------------------------------------------------------------------------- #

def test_place_or_remove_separator_is_locked_while_a_run_is_active_and_says_why(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0])
    win = MainWindow(AppState(s))
    pv = win.preview_screen

    class _Ev:
        def isAccepted(self):
            return False

        def button(self):
            return Qt.LeftButton

        def scenePos(self):
            return QPointF(0.0, 0.0)

    pv.set_run_active(True)
    # The button itself must ALSO visually disable, not just the click-level guard below
    # (belt-and-braces): set_run_active routes through _update_separators_button, and a
    # regression that dropped that call would leave the button clickable-LOOKING.
    assert pv.btn_place_separators.isEnabled() is False
    pv._place_or_remove_separator(_Ev(), [], 0.0)
    assert s.processing.segmentation.separators[0].times_s == [1.0]   # unchanged
    bar_msg = win.statusBar().currentMessage().lower()
    assert "locked" in bar_msg and "run" in bar_msg
    win.close()


# --------------------------------------------------------------------------- #
# End to end: a real click on the segments stack, armed, places a separator instead
# of toggling exclusion — proving the shared _toggle_from_emg_click reroute actually
# wires up, not just _toggle_separator_at called directly.
# --------------------------------------------------------------------------- #

def test_an_armed_click_on_the_segments_stack_places_a_separator_not_an_exclusion(tmp_path):
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0, 2.0])
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename(FILENAME)
    _render(pv, s)
    num, t0, t1, _kind = pv._breaths[1]                # segment 2 of 3
    off = pv._trim_offset_s
    xc = (t0 + t1) / 2.0 + off

    class _Rect:
        def contains(self, _p):
            return True

    class _View:
        def sceneBoundingRect(self):
            return _Rect()

        def mapSceneToView(self, _pos):
            return QPointF(xc, 0.0)

        def viewPixelSize(self):
            return (0.01, 1.0)

    class _Plot:
        def getViewBox(self):
            return _View()

    class _Ev:
        def isAccepted(self):
            return False

        def button(self):
            return Qt.LeftButton

        def scenePos(self):
            return QPointF(0.0, 0.0)

    pv.btn_place_separators.setChecked(True)           # arm via the real button
    assert pv._separators_armed is True
    pv._segments_subplots = [_Plot()]
    pv._on_segments_clicked(_Ev())

    entry = next(e for e in pv.state.settings.processing.segmentation.separators
                if e.file == FILENAME)
    assert xc in entry.times_s
    assert pv.state.settings.processing.exclude_breaths == [], (
        "armed mode must place a separator, never fall through to the exclude toggle"
    )
    win.close()


# --------------------------------------------------------------------------- #
# _update_separator_lines / _render_segments_stack: the drawn markers themselves
# --------------------------------------------------------------------------- #

def test_a_real_render_draws_one_separator_line_item_per_subplot_with_the_right_times(tmp_path):
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0, 2.0])
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename(FILENAME)
    _render(pv, s)

    assert pv._separator_items, "no SeparatorLinesItem drawn at all"
    assert len(pv._separator_items) == len(pv._segments_subplots)
    for item in pv._separator_items:
        assert item._times == [1.0, 2.0]
    win.close()


def test_a_file_with_no_separator_entry_draws_empty_separator_lines(tmp_path):
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators")   # no SeparatorEntry at all
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename(FILENAME)
    _render(pv, s)

    assert pv._separator_items
    for item in pv._separator_items:
        assert item._times == []
    win.close()


def test_the_empty_emg_early_return_leaves_no_stale_separator_items(qapp, tmp_path):
    """_render_segments_stack's own early return (no/degenerate EMG data) must reset
    _separator_items rather than leave a previous render's now-torn-down items behind
    — segments_plots.clear() already destroys the underlying graphics items, but the
    Python-side bookkeeping list must not go on naming them."""
    import numpy as np
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0])
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename(FILENAME)
    _render(pv, s)
    assert pv._separator_items                     # a real render populated it first

    pv._render_segments_stack(np.array([]), 200.0, None, FILENAME)
    assert pv._separator_items == []
    win.close()


# --------------------------------------------------------------------------- #
# A file switch must unarm/disable immediately, synchronously — self-review finding:
# without this, a click during the (async) window before the new file's own 'segments'
# job completes could resolve _selected_filename() to the NEW file while still hit-
# testing the OLD file's now-torn-down plot geometry.
# --------------------------------------------------------------------------- #

def test_a_file_switch_unarms_and_disables_the_button_synchronously(tmp_path):
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0])
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv._refresh_files()
    pv.file_rail.select_filename(FILENAME)
    _render(pv, s)
    pv.btn_place_separators.setChecked(True)
    assert pv._separators_armed is True

    pv.file_rail.select_filename("synth_case_B.csv")   # _begin_file_switch runs synchronously
    assert pv._separators_armed is False, (
        "a file switch must unarm immediately, before the new file's own 'segments' "
        "job (if any) has even been dispatched, let alone completed"
    )
    assert pv.btn_place_separators.isChecked() is False
    assert pv.btn_place_separators.isEnabled() is False
    win.close()


def test_an_already_accepted_click_is_ignored_even_while_armed(qapp, tmp_path):
    """The Ctrl+left-click/right-click breath-typing primitive (M-20) accepts its click
    at ITEM level (BreathSpansItem.mouseClickEvent), before the scene-level funnel this
    ticket's own armed check lives in ever runs — so a right-click landing on an
    existing breath still opens the type menu while armed, never places a separator.
    _place_or_remove_separator mirrors _toggle_from_emg_click's own
    ``if ev.isAccepted(): return`` guard for exactly this reason; this pins it."""
    from respmech.ui.main_window import MainWindow

    s = _emg_only_settings(tmp_path, method="separators", separator_times=[1.0])
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    pv.btn_place_separators.setChecked(True)
    assert pv._separators_armed is True

    class _AcceptedEv:
        def isAccepted(self):
            return True          # already claimed by an item (e.g. the type menu)

        def button(self):
            return Qt.LeftButton

        def scenePos(self):
            return QPointF(0.0, 0.0)

    pv._place_or_remove_separator(_AcceptedEv(), [], 0.0)
    assert s.processing.segmentation.separators[0].times_s == [1.0]   # unchanged
    win.close()
