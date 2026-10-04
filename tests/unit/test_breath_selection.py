"""Single click MARKS a breath (it no longer toggles its exclusion).

The mark is one view-state number, ``PreviewScreen._selected_breath``, painted in one green
on every surface that shows that breath: its span on each plot, its number label, its row in
the result table (scrolled into view) and its loop plus legend entry in the Campbell /
flow-volume diagram. Excluding / including moved to the right-click menu (Tidal / Excluded).
"""
import os

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from respmech.ui.result_table import ResultTableModel
from respmech.ui.screens.preview._plot_helpers import BreathSpansItem
from respmech.ui.state import AppState
from respmech.ui.theme import SELECTED_BREATH_HEX, SELECTED_BREATH_RGB

from _helpers import INPUT, requires_synth, synth_settings

pytestmark = requires_synth()


def _screen(tmp_path):
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s))
    return win, win.preview_screen, s


def _render_mech(pv, s, name="synth_case_A.csv"):
    from respmech.ui.workers import stage_mechanics_preview
    pv._refresh_files(); pv.file_rail.select_filename(name)
    pv._render_preview(stage_mechanics_preview(s, os.path.join(INPUT, name)))


def _excluded(s, name="synth_case_A.csv"):
    return {b for e in s.processing.exclude_breaths if e.file == name for b in e.breaths}


def _span_items(pv):
    return {id(it): it for lst in pv._breath_regions.values() for it, _i in lst}.values()


# --------------------------------------------------------------------------- #
# the item: paints the mark, nothing else changes
# --------------------------------------------------------------------------- #
def test_item_set_selected_only_changes_the_mark(qapp):
    item = BreathSpansItem()
    item.set_spans([(0.0, 1.0, None, 1), (1.0, 2.0, None, 2)])
    assert item._selected is None
    item.set_selected(2)
    assert item._selected == 2
    item.set_selected(None)
    assert item._selected is None


def test_table_model_highlights_the_breath_row_and_keeps_it_across_a_refill(qapp):
    import pandas as pd
    m = ResultTableModel(pd.DataFrame({"breath_no": [1, 2, 4], "x": [0.1, 0.2, 0.3]}))
    assert m.set_highlight_breath(4) == 2
    bg = m.data(m.index(2, 1), role=8)                       # Qt.BackgroundRole
    assert bg is not None and bg.color().green() == SELECTED_BREATH_RGB[1]
    assert m.data(m.index(0, 1), role=8) is None
    # a recompute refills the table: the mark follows the breath, not the row position
    m.set_dataframe(pd.DataFrame({"breath_no": [2, 4], "x": [0.2, 0.3]}))
    assert m.data(m.index(1, 0), role=8) is not None
    assert m.set_highlight_breath(3) is None                 # an excluded breath has no row
    assert m.set_highlight_breath(None) is None


# --------------------------------------------------------------------------- #
# the screen: click -> mark, again -> unmark, other -> move; never excludes
# --------------------------------------------------------------------------- #
def test_click_marks_unmarks_and_moves_without_touching_exclusion(qapp, tmp_path):
    win, pv, s = _screen(tmp_path)
    _render_mech(pv, s)
    nums = list(pv._breath_spans)
    a, b = nums[0], nums[1]
    pv._select_breath(a)
    assert pv._selected_breath == a
    assert all(it._selected == a for it in _span_items(pv))
    pv._select_breath(b)                                     # another breath: the mark moves
    assert pv._selected_breath == b
    assert all(it._selected == b for it in _span_items(pv))
    pv._select_breath(b)                                     # the marked one again: unmark
    assert pv._selected_breath is None
    assert all(it._selected is None for it in _span_items(pv))
    assert _excluded(s) == set()                             # no click ever excluded anything
    win.close()


def test_the_real_scene_click_handler_marks_instead_of_toggling(qapp, tmp_path):
    """``_on_plot_clicked`` is the actual ``sigMouseClicked`` target: drive it with a fake
    event over the centre of a breath (same technique as the legend-click test)."""
    from PySide6.QtCore import QPointF, Qt
    win, pv, s = _screen(tmp_path)
    _render_mech(pv, s)
    num = next(iter(pv._breath_spans))
    t0, t1 = pv._breath_spans[num]
    xc = (t0 + t1) / 2.0

    class _Rect:
        def contains(self, _p): return True

    class _View:
        def sceneBoundingRect(self): return _Rect()
        def mapSceneToView(self, _pos): return QPointF(xc, 0.0)

    class _Plot:
        def getViewBox(self): return _View()

    class _Ev:
        def isAccepted(self): return False
        def button(self): return Qt.LeftButton
        def scenePos(self): return QPointF(0.0, 0.0)

    pv._channel_plots = [_Plot()]
    pv._on_plot_clicked(_Ev())
    assert pv._selected_breath == num and _excluded(s) == set()
    pv._on_plot_clicked(_Ev())
    assert pv._selected_breath is None and _excluded(s) == set()
    win.close()


def test_exclusion_still_works_through_the_right_click_funnel_and_keeps_the_mark(qapp, tmp_path):
    win, pv, s = _screen(tmp_path)
    _render_mech(pv, s)
    num = next(iter(pv._breath_spans))
    pv._select_breath(num)
    assert pv._set_breath_type(num, "excluded") == "excluded"   # what the menu's "Excluded" calls
    assert num in _excluded(s)
    assert pv._selected_breath == num                            # marking is independent of type
    win.close()


def test_the_mark_is_dropped_when_the_file_changes(qapp, tmp_path):
    win, pv, s = _screen(tmp_path)
    _render_mech(pv, s)
    pv._select_breath(next(iter(pv._breath_spans)))
    pv._reset_breath_state()                                    # what a file switch calls
    assert pv._selected_breath is None
    win.close()


def test_the_marked_label_is_green_and_bold_and_restores(qapp, tmp_path):
    win, pv, s = _screen(tmp_path)
    _render_mech(pv, s)
    num = next(iter(pv._breath_texts))
    txt = pv._breath_texts[num]
    pv._select_breath(num)
    assert txt.textItem.defaultTextColor().getRgb()[:3] == SELECTED_BREATH_RGB
    # a type change on the marked breath keeps the mark colour on the label
    pv._set_breath_type(num, "excluded")
    pv._retag_breath_label(txt, num, "excluded", pv._breath_texts)
    assert txt.textItem.defaultTextColor().getRgb()[:3] == SELECTED_BREATH_RGB
    pv._select_breath(num)
    assert txt.textItem.defaultTextColor().getRgb()[:3] != SELECTED_BREATH_RGB
    win.close()


# --------------------------------------------------------------------------- #
# tables and loops need a real batch result
# --------------------------------------------------------------------------- #
def _with_batch(pv, s):
    from respmech.core.pipeline import run_batch
    _render_mech(pv, s)
    pv._on_batch_result(run_batch(s, only_files=["synth_case_A.csv"]))


def test_marking_highlights_the_table_row_and_draws_the_loop_with_a_legend_entry(qapp, tmp_path):
    win, pv, s = _screen(tmp_path)
    _with_batch(pv, s)
    model = pv._table_model
    tidal = [int(n) for n in model._df["breath_no"]]
    assert tidal, "the synthetic file must give at least one tidal breath"
    num = tidal[-1]                                  # the last row, so scrolling has work to do
    scrolled = []
    real = pv.table.scrollTo
    pv.table.scrollTo = lambda idx, *hint: (scrolled.append(idx.row()), real(idx, *hint))
    pv._select_breath(num)
    row = model._highlight_row
    assert row == tidal.index(num)
    assert scrolled == [row]                         # the row is scrolled into view
    ax = pv.campbell.figure.axes[0]
    marked = [ln for ln in ax.lines if ln.get_label() == f"breath #{num}"]
    assert len(marked) == 1
    assert marked[0].get_color().lower() == SELECTED_BREATH_HEX.lower()
    # unmarking removes both the row highlight and the green loop
    pv._select_breath(num)
    assert model._highlight_row is None
    ax = pv.campbell.figure.axes[0]
    assert not [ln for ln in ax.lines if str(ln.get_label()).startswith("breath #")]
    win.close()


def test_the_flow_volume_loop_gets_a_legend_only_while_a_breath_is_marked(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch
    s = synth_settings(str(tmp_path), channels={"poes": None, "pgas": None, "pdi": None,
                                                "emg": [], "entropy": []})
    win = MainWindow(AppState(s))
    pv = win.preview_screen
    _render_mech(pv, s)
    res = run_batch(s, only_files=["synth_case_A.csv"])
    pv._on_batch_result(res)
    num = int(pv._table_model._df["breath_no"].iloc[0])
    ax = pv.campbell.figure.axes[0]
    assert ax.get_legend() is None
    pv._select_breath(num)
    ax = pv.campbell.figure.axes[0]
    assert [ln for ln in ax.lines if ln.get_label() == f"breath #{num}"]
    win.close()


def test_an_export_redraw_never_carries_the_screen_only_mark(qapp, tmp_path):
    win, pv, s = _screen(tmp_path)
    _with_batch(pv, s)
    num = int(pv._table_model._df["breath_no"].iloc[0])
    pv._select_breath(num)
    pv._loop_export = True                                  # what the export sets around its redraw
    pv._draw_campbell_or_loop(pv._campbell_breaths)
    ax = pv.campbell.figure.axes[0]
    assert not [ln for ln in ax.lines if str(ln.get_label()).startswith("breath #")]
    pv._loop_export = False
    win.close()
