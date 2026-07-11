"""Stacked channel graphs must share one left margin so their x-axes line up.

Without a pinned left-axis width, a panel whose y-ticks read "-100" sits further right
than one reading "0", and the stacked channels visibly step in and out. theme.align_left_axis
pins every panel to the same width; these tests assert the panels end up x-aligned.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")

from PySide6.QtWidgets import QApplication  # noqa: E402

from respmech.ui.state import AppState  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT = os.path.join(ROOT, "tests", "golden", "input")


@pytest.fixture(scope="module")
def qapp():
    from respmech.ui import theme
    app = QApplication.instance() or QApplication([])
    theme.apply_theme(app)
    yield app


def _viewbox_lefts(plots):
    QApplication.processEvents()
    QApplication.processEvents()
    return [p.getViewBox().sceneBoundingRect().left() for p in plots]


def test_channel_setup_previews_are_x_aligned(qapp):
    """The column previews use separate PlotWidgets (no shared grid) — the case that
    actually misaligned. Columns of very different magnitude must still line up."""
    from respmech.ui.channel_setup_dialog import ChannelSetupDialog
    n = 1500
    mat = np.column_stack([
        np.arange(n) / 500.0,                       # time: 0..3   (narrow ticks)
        0.02 * np.sin(np.linspace(0, 40, n)),       # EMG: ±0.02
        -80 + 30 * np.sin(np.linspace(0, 15, n)),   # Poes: -60..-110  (wide ticks)
        np.linspace(0, 1, n)])
    dlg = ChannelSetupDialog(["demo.csv"], 500, loader=lambda p: (mat, ["t", "e", "p", "v"]))
    dlg.resize(760, 560); dlg.show()
    widths = [p.getAxis("left").width() for p in dlg._plots]
    assert len(set(round(w) for w in widths)) == 1          # all one width
    lefts = _viewbox_lefts(dlg._plots)
    assert max(lefts) - min(lefts) < 1.0                    # plotting areas start together
    dlg.close()


@pytest.mark.skipif(not os.path.exists(os.path.join(INPUT, "synth_case_A.csv")),
                    reason="synthetic input absent")
def test_mechanics_channel_stack_is_x_aligned(qapp):
    from respmech.settingsio.migrate import migrate_dict
    from respmech.ui.workers import stage_mechanics_preview
    from respmech.ui.main_window import MainWindow
    legacy = {"input": {"inputfolder": INPUT, "files": "synth_case_A.csv",
                        "format": {"samplingfrequency": 1000},
                        "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 9,
                                 "column_volume": 6, "column_flow": 5, "columns_emg": [2, 3, 4],
                                 "columns_entropy": [10, 11, 12]}},
              "processing": {"mechanics": {"breathseparationbuffer": 200, "separateby": "flow",
                                           "avgresamplingobs": 300}, "emg": {}},
              "output": {"outputfolder": "", "data": {}}}
    s, _ = migrate_dict(legacy); s.input.folder = INPUT
    win = MainWindow(AppState(s)); pv = win.preview_screen
    pv._refresh_files(); pv.file_combo.setCurrentText("synth_case_A.csv")
    pv._render_preview(stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv")))
    pv.plots.setFixedSize(640, 540)
    lefts = _viewbox_lefts(pv._channel_plots)               # processes events → layout settles
    widths = [p.getAxis("left").width() for p in pv._channel_plots]
    assert len(set(round(w) for w in widths)) == 1          # every panel one width → aligned
    assert max(lefts) - min(lefts) < 1.0                    # plotting areas start together
    win.close()


def test_align_helper_is_safe_and_fixed_width(qapp):
    import pyqtgraph as pg
    from respmech.ui import theme
    p = pg.PlotWidget(); p.resize(320, 200); p.show()
    theme.align_left_axis(p)
    QApplication.processEvents(); QApplication.processEvents()
    assert abs(p.getAxis("left").width() - theme.PLOT_AXIS_WIDTH) <= 1   # pinned to the fixed width
    theme.align_left_axis(None)                              # never raises on a bad arg
    p.close()
