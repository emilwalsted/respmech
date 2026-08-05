"""The EMG working views have to stay readable at the height they now default to.

Emil asked for the noise tab to divide roughly into thirds — chrome and settings, the two
working views, the three reference panels (02-08-2026). That left each working view ~130 px,
and three separate things broke at that size, none of which any existing guard could see:

  1. pyqtgraph printed the y tick labels through each other. Its crowding rules
     (``textFillLimits``) only apply from the SECOND tick level down — ``generateDrawSpecs``
     says ``if i > 0: ## always draw top level`` — so the major labels are drawn however
     little room there is. Measured: six 13 px labels at 8.6 px pitch.
  2. the rotated axis label was clipped to "MG (a.u.)", because 81 px of label was being
     drawn into 63 px of axis. BOTH views were clipped; only one showed it, the other's
     overflow happening to fall into a taller x axis before it reached the widget edge.
  3. the in-plot legends covered the signal they were labelling — one 18 px legend row over
     a 43 px data area.

The measurements here therefore go through the DRAWING code (``generateDrawSpecs``), not
through widget geometry: every one of these defects lives inside the plot's own painting and
leaves widget geometry looking perfectly healthy.

They also set a REAL view range first. Driving ``refresh_files()`` alone leaves the EMG
panels empty at 0..1. Measured, that empty state is what hides the SI-prefix defect outright
(an empty range gives ``autoSIPrefixScale == 1.0``, so that assertion is vacuous without a
range); it also softens the tick and label checks, which then only bite at the shortest
parametrisations. Every one of these guards was first written without a range and passed on
code that was visibly broken in a screenshot, which is why the range is asserted, not assumed.
"""
import pytest

from respmech.ui.state import AppState

from _helpers import requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

# The range the bundled synthetic sample actually produces on the detail channel, measured
# with the reactive pipeline run to completion. Any range of this order does the job; what
# matters is that it is NOT the empty plot's 0..1.
_REAL_RANGE = (-0.42906, 0.63771)

_DATA_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


# A breath complete enough that _overlay_campbell_work draws its labelled artists — an
# unlabelled one would make any assertion about the legend pass for the wrong reason.
_BREATH = {
    "ignored": False,
    "poes": [0, -5, -10, -5, 0], "volume": [0, 0.3, 0.8, 0.5, 0],
    "poesavg": [0, -5, -10, -5, 0], "volumeavg": [0, 0.3, 0.8, 0.5, 0],
    "eelvavg": [0.0, 0.0], "eilvavg": [0.8, -10.0],
    "inspiration": {"volumeavg": [0, 0.3, 0.8], "poesavg": [0, -5, -10]},
    "wob": {"wobtotal": 9.2},
}


def _noise_tab(qapp, tmp_path, height=800):
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path), data_out=_DATA_OUT, remove_ecg=True, noise=True)
    win = MainWindow(AppState(s))
    win.resize(1280, height)
    win.show()
    win.preview_screen.refresh_files()
    win.tabs.setCurrentIndex(1)
    pv = win.preview_screen
    pv.subtabs.setCurrentIndex(pv.subtabs.count() - 1)          # the noise page
    for _ in range(12):
        qapp.processEvents()
    return win, pv


def _working_views(pv):
    return (("Detail channel", pv.emg_plots), ("Conditioned result", pv.emg_result_plots))


def _with_range(qapp, pv):
    """Give both working views a realistic y range and prove it took."""
    for _, pw in _working_views(pv):
        pw.getPlotItem().getViewBox().setYRange(*_REAL_RANGE, padding=0)
    for _ in range(6):
        qapp.processEvents()
    for name, pw in _working_views(pv):
        lo, hi = pw.getPlotItem().getViewBox().viewRange()[1]
        assert hi - lo > 0.5, (
            f"{name} is still on an empty plot's range ({lo}..{hi}) — every measurement "
            f"below would be taken on a plot that has no trouble fitting anything")


def _drawn_tick_labels(pw):
    """(label height, sorted tops) of the y tick text the axis will actually paint."""
    from PySide6.QtGui import QImage, QPainter
    ax = pw.getPlotItem().getAxis("left")
    img = QImage(64, 64, QImage.Format_ARGB32)
    img.fill(0)
    p = QPainter(img)
    try:
        ax.picture = None
        _, _, text_specs = ax.generateDrawSpecs(p)
    finally:
        p.end()
    if not text_specs:
        return 0, []
    return text_specs[0][0].height(), sorted(t[0].top() for t in text_specs)


@pytest.mark.parametrize("height", [700, 800, 900, 1100, 1400])
def test_y_tick_labels_never_overprint(qapp, tmp_path, height):
    """The defect Emil would see as an unreadable smear down the left edge."""
    win, pv = _noise_tab(qapp, tmp_path, height)
    _with_range(qapp, pv)
    for name, pw in _working_views(pv):
        h, tops = _drawn_tick_labels(pw)
        assert len(tops) >= 2, f"{name}: fewer than two tick labels — nothing to overlap"
        gaps = [b - a for a, b in zip(tops, tops[1:])]
        assert min(gaps) >= h, (
            f"{name} at a {height} px window: tick labels are {h:.0f} px tall but only "
            f"{min(gaps):.1f} px apart — they overprint")
    win.close()


@pytest.mark.parametrize("height", [700, 800, 900, 1100, 1400])
def test_the_axis_label_is_never_drawn_clipped(qapp, tmp_path, height):
    """Either the label fits the axis it is rotated into, or it is not shown at all.

    A label that does not fit is not harmlessly truncated — it loses its FIRST characters,
    which is how "EMG (a.u.)" came to read "MG (a.u.)".
    """
    win, pv = _noise_tab(qapp, tmp_path, height)
    _with_range(qapp, pv)
    checked = 0
    for name, pw in _working_views(pv):
        ax = pw.getPlotItem().getAxis("left")
        if not ax.label.isVisible():
            # A hidden label is the documented last resort, but it must be hidden because it
            # could NOT fit — not because the mechanism quietly gave up. Prove the shortest
            # wording really is too long for this axis.
            shortest = min((v for v in ax._variants if v), key=len)
            ax._apply_label(shortest)
            try:
                assert ax.label.boundingRect().width() > ax.height(), (
                    f"{name} at a {height} px window: the label is hidden although the "
                    f"shortest wording {shortest!r} would have fitted")
            finally:
                ax._pick_label()
            checked += 1
            continue
        need = ax.label.boundingRect().width()      # rotated: its WIDTH spans the axis height
        assert need <= ax.height(), (
            f"{name} at a {height} px window: the label {ax.labelText!r} needs {need:.0f} px "
            f"of the {ax.height():.0f} px axis and will be clipped")
        checked += 1
    assert checked == 2, f"only {checked} of the two working views were checked"
    win.close()


def test_the_axis_label_grows_back_when_there_is_room(qapp, tmp_path):
    """The shortening must be a response to the height, not a permanent downgrade."""
    win, pv = _noise_tab(qapp, tmp_path, 800)
    _with_range(qapp, pv)
    short = pv.emg_plots.getPlotItem().getAxis("left").labelText
    win.resize(1280, 1400)
    for _ in range(12):
        qapp.processEvents()
    _with_range(qapp, pv)
    tall = pv.emg_plots.getPlotItem().getAxis("left").labelText
    assert len(tall) > len(short), (
        f"the label stayed {tall!r} on a tall window after shortening to {short!r} on a "
        f"short one — it is not re-chosen on resize")
    assert "a.u." in tall, f"the unit never comes back: {tall!r}"
    win.close()


def test_the_working_views_carry_no_si_multiplier(qapp, tmp_path):
    """pyqtgraph states its ×N factor INSIDE the axis label. Since the label here is allowed
    to shorten — and, in the worst case, to disappear — a scaled axis could be left showing
    "500" for a value of 0.5 with nothing on screen saying so. Absolute ticks are what makes
    the shortening safe, and they cost nothing: this range reads -0.4 / 0.1 / 0.6 unscaled.
    """
    win, pv = _noise_tab(qapp, tmp_path, 800)
    _with_range(qapp, pv)
    for name, pw in _working_views(pv):
        ax = pw.getPlotItem().getAxis("left")
        assert ax.autoSIPrefixScale == 1.0, (
            f"{name}: tick values are scaled by {ax.autoSIPrefixScale:g} while the label "
            f"that would declare it is {'hidden' if not ax.label.isVisible() else 'free to hide'}")
    win.close()


def test_neither_working_view_has_an_in_plot_legend(qapp, tmp_path):
    """A legend is an item inside the ViewBox, so wherever it anchors it covers signal."""
    win, pv = _noise_tab(qapp, tmp_path)
    for name, pw in _working_views(pv):
        assert pw.getPlotItem().legend is None, (
            f"{name} has an in-plot legend again — at this panel height it hides the trace")
    win.close()


def test_the_detail_traces_are_still_named(qapp, tmp_path):
    """Dropping the legend must not cost the reader the stage names; they moved to the band.

    The conditioned view needs no equivalent — its corner picker names every channel in the
    very colour the channel is drawn in, which is why its legend was pure duplication.
    """
    win, pv = _noise_tab(qapp, tmp_path)
    pv._set_trace_key([("raw", (128, 128, 128)), ("ECG-removed", (0, 0, 255))])
    key = pv.emg_trace_key
    assert key.parent() is pv.emg_plots, "the trace key is not floating in the detail plot"
    assert "raw" in key.text() and "ECG-removed" in key.text()
    assert key.isVisible(), (
        "the key holds the right text but is hidden — _PlotTitleOverlay.fit hides it when it "
        "does not fit between the title and the corner, and the detail view has no legend to "
        "fall back on, so the traces would be unidentifiable")
    band = pv._plot_title_overlays[0]._band
    assert key.y() + key.height() <= band, (
        f"the key runs to y={key.y() + key.height()} past the {band} px reserved band, so it "
        f"is over the data it is meant to label")
    win.close()


def test_the_detail_key_stays_visible_on_windows_font_metrics(qapp, windows_metrics, tmp_path):
    """The key must survive the widest metrics we ship to, at the narrowest screen we target.

    This is the case that made hiding it unacceptable rather than merely a pity. Modelled at
    QFont.setStretch(145) in a 1280 px window — the narrowest this project supports — the
    one-line key measured 816 px against 815 px of room between the title and the channel
    combo, so it vanished by one pixel and the detail view was left with four unnamed traces
    and no legend to fall back on.

    It also must not buy its visibility with plot height. Wrapping is only free while it
    stays inside the band the channel combo already reserves; two failed attempts here ran to
    four and then six lines (band 45 px -> 78 px -> 112 px), spending more of the graph than
    the whole re-proportioning had won back.
    """
    win, pv = _noise_tab(qapp, tmp_path, 800)
    pv._set_trace_key([("raw", (150, 165, 180))])
    for _ in range(4):
        qapp.processEvents()
    one_line = pv.emg_trace_key.height()
    assert one_line > 0

    pv._set_trace_key([("raw", (150, 165, 180)), ("ECG-removed", (44, 110, 155)),
                       ("noise-reduced", (60, 140, 90)), ("RMS envelope", (180, 50, 42))])
    for _ in range(4):
        qapp.processEvents()
    key = pv.emg_trace_key
    assert key.isVisible(), (
        "the four-stage key is hidden at Windows metrics in a 1280 px window — the detail "
        "traces have nothing naming them")
    assert key.height() <= 2 * one_line, (
        f"the key wrapped to {key.height() / one_line:.0f} lines ({key.height()} px against a "
        f"{one_line} px line); it is meant to fit the band the combo already reserves")
    win.close()


def test_clearing_the_panels_clears_the_trace_key(qapp, tmp_path):
    """The key must go blank with the plot it labels.

    The in-plot legend it replaced was emptied for free by ``PlotItem.clear()``. A separate
    QLabel is not, so every path that blanks the detail plot — a file switch, a settings
    change that invalidates the panels, a full reset — left the key naming four traces on an
    empty graph.
    """
    win, pv = _noise_tab(qapp, tmp_path)
    pv._set_trace_key([("raw", (150, 165, 180)), ("ECG-removed", (44, 110, 155))])
    for _ in range(4):
        qapp.processEvents()
    assert pv.emg_trace_key.text(), "the key was never populated — the check below is vacuous"

    for clear in (pv._clear_file_panels, pv._clear_all_panels):
        pv._set_trace_key([("raw", (150, 165, 180)), ("ECG-removed", (44, 110, 155))])
        for _ in range(4):
            qapp.processEvents()
        clear()
        for _ in range(4):
            qapp.processEvents()
        assert not pv.emg_trace_key.text(), (
            f"{clear.__name__} blanked the detail plot but left the key naming its traces")
    win.close()


def test_the_tick_ladder_always_terminates():
    """``tickSpacing`` climbs a ladder, so it must be impossible for the climb not to end.

    ``_next_nice`` is exact only in the normal double range: below ~3.3e-311 its decade term
    underflows to 0.0, so it stops increasing (and at 5e-324 it divided by zero). No real
    signal has a span that small, but an unbounded loop over a function that can stop
    increasing is a hang rather than a wrong tick, so the bound is asserted, not argued.
    """
    import math
    from respmech.ui.screens.preview_screen import _FitAxis, _next_nice

    for e in range(-320, 300):                       # every normal decade
        for m in (1.0, 1.5, 2.0, 3.0, 4.999, 5.0, 7.0, 9.99):
            v = m * 10.0 ** e
            if v <= 0 or math.isinf(v):
                continue
            nxt = _next_nice(v)                      # must never raise...
            if v > 3.3e-311:
                assert nxt > v, f"_next_nice({v!r}) returned {nxt!r} — the ladder cannot climb"
    assert _next_nice(5e-324) == 5e-324, "a subnormal input must be returned, not raise"

    ax = _FitAxis(orientation="left")
    ax.setHeight(63)

    # Called straight on the MAIN thread, with no watchdog. An earlier version ran the
    # climb in a worker thread to time it out, which was unsafe: tickSpacing reaches
    # _min_pitch -> self.font(), a Qt call on a QGraphicsWidget, and Qt GUI objects may only
    # be touched from the thread that owns them. A watchdog buys nothing here anyway — the
    # climb is a bounded `for _ in range(64)`, so it cannot fail to terminate by construction,
    # and if that bound is ever removed this call hangs and CI's own job timeout says so.
    for lo, hi in ((-4.94e-323, 4.94e-323), (0, 1e-320), (5, 5), (0, 1e12), (-0.429, 0.638)):
        ax.tickSpacing(lo, hi, 63.0)            # must return, whatever it returns
    # ...and the bound must not have changed the answer for a real signal
    assert ax.tickSpacing(-0.429, 0.638, 63.0)[0][0] == 0.5


def test_the_diagnostic_figures_follow_their_panel_in_both_directions(qapp, tmp_path):
    """The two matplotlib diagnostics sit in a splitter, and matplotlib — unlike a pyqtgraph
    axis — never re-lays a figure out on resize. So their furniture has to be re-decided
    whenever the panel changes, and that re-decision must be IDEMPOTENT.

    Both halves were real. Deciding once at draw time meant a legend shed on a short panel
    never came back when the panel was dragged tall again. Fixing that by re-running
    ``tight_layout()`` on every resize then compounded — it shrinks the axes against the
    margins currently in force — and a few splitter drags collapsed both figures into a
    corner of their own panel, which is what a screenshot caught.
    """
    from respmech.ui.screens.preview_screen import refit_compact_figure
    # B03 (UI-overhaul) folded Run & results into this same tab as a collapsed drawer —
    # even minimised to a single toggle row, it costs Preview & QC's workspace ~60-90 px
    # of the window it used to have entirely to itself. At the default 800 px height that
    # left the "tall" [140, 140, 520] request below delivering only 200 px to the third
    # region (vs. 262 px before B03), under the threshold the legend needs to render at
    # all — not a defect in the idempotent-refit logic under test here, just less headroom
    # than this specific "tall" scenario needs. 900 px restores it comfortably (measured:
    # 300 px delivered, vs. the same 262 px baseline needed).
    win, pv = _noise_tab(qapp, tmp_path, height=900)
    canvases = [pv.fidelity_canvas, pv.emg_psd_canvas]

    # give each figure something to lay out, exactly as a render does
    for c in canvases:
        c.figure.clear()
        ax = c.figure.add_subplot(111)
        ax.plot([0, 1], [0, 1], label="a")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        from respmech.ui.screens.preview_screen import _fit_compact_figure
        _fit_compact_figure(c, ax, title="t", legend_kw={"fontsize": 7})
        c.draw()

    def box(c):
        c.draw()
        p = c.figure.axes[0].get_position()
        return tuple(float(v) for v in (p.x0, p.y0, p.width, p.height))

    # A tolerance, not equality: the compounding this guards against moved the axes by ~0.07
    # of the figure per few fits, while the layout solver lands on a rounding boundary and
    # can wobble in the third decimal (measured 0.714 vs 0.713 on the Windows runner, where
    # exact comparison went red for a difference 70x smaller than the defect).
    for c in canvases:
        before = box(c)
        for _ in range(20):
            refit_compact_figure(c)
        after = box(c)
        drift = max(abs(a - b) for a, b in zip(before, after))
        assert drift < 0.01, (
            f"the layout is not idempotent: 20 refits moved the axes from {before} to "
            f"{after} (drift {drift:.4f}) — repeated fits compound and collapse the figure")

    # ...and the furniture really does come back when the panel grows again
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QSplitter
    page = pv.subtabs.currentWidget().widget()
    v = [w for w in page.findChildren(QSplitter) if w.orientation() == Qt.Vertical][0]
    seen = []
    for sizes in ([140, 140, 520], [140, 140, 125], [140, 140, 520]):
        v.setSizes(sizes)
        for _ in range(8):
            qapp.processEvents()
        seen.append(pv.fidelity_canvas.figure.axes[0].get_legend() is not None)
    assert seen[0] and not seen[1] and seen[2], (
        f"legend present across tall/short/tall was {seen} — it must be shed when there is "
        f"no room and restored when there is")
    win.close()


def test_the_fidelity_x_label_never_runs_off_its_panel(qapp, tmp_path):
    """The x label must stay inside the canvas at any panel width, and the wording chosen
    must be a function of the width alone.

    matplotlib centres the x label on the AXES, not the figure, so "is the text narrower than
    the canvas?" answers yes while the label overhangs: at the default 345 px the full wording
    is 344 px (fits) but is drawn from x=25 to x=369 and loses its last characters.

    Getting the ladder to answer by width alone took three corrections, each of which this
    test would have caught: the fit ran from an event filter, which sees the resize BEFORE the
    canvas does, so it measured the old size; the renderer was captured before the settling
    draw, so extents came back in the previous figure's coordinates; and the font size was
    applied AFTER the wording was chosen, so a label measured at 8 pt was drawn at 10 pt.
    """
    from PySide6.QtWidgets import QSplitter
    win, pv = _noise_tab(qapp, tmp_path)
    pv._draw_fidelity({"frontier": {0.2: [0.9, 0.9, 0.9], 0.6: [0.8, 0.8, 0.8],
                                    1.0: [0.7, 0.7, 0.7]},
                       "prop_decrease": 0.6, "fidelity_target": 0.8})
    for _ in range(8):
        qapp.processEvents()
    c = pv.fidelity_canvas
    ax = c.figure.axes[0]
    assert ax.get_xlabel(), "no x label was drawn — the checks below would be vacuous"

    row = pv._emg_diag_row
    assert isinstance(row, QSplitter)
    seen = {}
    # narrow, wide, narrow again: the same width must give the same answer both times
    for sizes in ([420, 410, 410], [700, 270, 270], [300, 470, 470],
                  [420, 410, 410], [700, 270, 270]):
        row.setSizes(sizes)
        for _ in range(10):
            qapp.processEvents()
        c.draw()
        bb = ax.xaxis.label.get_window_extent(renderer=c.get_renderer())
        over = max(0.0, -bb.x0) + max(0.0, bb.x1 - c.width())
        assert over <= 1.0, (
            f"at a {c.width()} px panel the label {ax.get_xlabel()!r} is drawn "
            f"[{bb.x0:.0f},{bb.x1:.0f}] and runs {over:.0f} px outside it")
        prev = seen.setdefault(c.width(), ax.get_xlabel())
        assert prev == ax.get_xlabel(), (
            f"a {c.width()} px panel showed {prev!r} once and {ax.get_xlabel()!r} another "
            f"time — the wording depends on which width it came from, not on the width")
    win.close()


def test_fitting_never_resizes_the_figure_behind_qt(qapp, tmp_path):
    """A figure must never be a different size from the canvas widget that owns it.

    This shipped in v2.3.4 and Emil caught it in the live app: the fidelity panel showed a
    small plot composited on top of the previous, larger one, with the y label clipped to
    "Fideli". The cause was measuring the x-label ladder against the WIDGET width and forcing
    the figure to match with set_size_inches(forward=False). Qt paints the Agg buffer at its
    own size into the top-left of the widget and leaves the rest of the widget alone, so a
    figure even slightly smaller than its canvas leaves the previous frame showing around it.

    The figure's size belongs to Qt. Anything that needs a size must read one, not set one.
    """
    from respmech.ui.screens.preview_screen import _fit_compact_figure, refit_compact_figure
    win, pv = _noise_tab(qapp, tmp_path)
    pv._draw_fidelity({"frontier": {0.2: [0.9, 0.9, 0.9], 0.6: [0.8, 0.8, 0.8],
                                    1.0: [0.7, 0.7, 0.7]},
                       "prop_decrease": 0.6, "fidelity_target": 0.8})
    # the PSD panel is only drawn once a detail channel has been staged; give it the same
    # kind of figure directly, so both diagnostics are covered rather than one
    psd = pv.emg_psd_canvas
    psd.figure.clear()
    _ax = psd.figure.add_subplot(111)
    _ax.plot([0, 1], [0, 1], label="raw")
    _ax.set_xlabel("Frequency (Hz)"); _ax.set_ylabel("Power (dB)")
    _fit_compact_figure(psd, _ax, legend_kw={"fontsize": 7})
    for _ in range(8):
        qapp.processEvents()

    checked = 0
    for name, c in (("fidelity", pv.fidelity_canvas), ("PSD", pv.emg_psd_canvas)):
        if not c.figure.axes:
            continue
        # Put the figure DELIBERATELY out of step with its widget, which is the only state in
        # which the defect is reachable — in the app it happens when the panel is drawn before
        # the layout has settled. A fit taken here must READ that disagreement, never resolve
        # it by moving the figure: resolving it is precisely what painted the small frame into
        # the corner. (A plain c.resize() cannot set this up; matplotlib's canvas updates the
        # figure synchronously, so the two are never out of step that way.)
        dpi = c.figure.dpi or 100.0
        c.figure.set_size_inches((c.width() - 160) / dpi, (c.height() - 90) / dpi,
                                 forward=False)
        before = (float(c.figure.bbox.width), float(c.figure.bbox.height))
        assert abs(before[0] - c.width()) > 1, (
            f"{name}: the figure and the widget agree — this test would not exercise the "
            f"disagreement it exists for")
        refit_compact_figure(c)
        after = (float(c.figure.bbox.width), float(c.figure.bbox.height))
        assert after == before, (
            f"{name}: fitting resized the figure from {before} to {after} behind Qt's back. "
            f"Qt paints the Agg buffer at its own size into the widget's top-left corner and "
            f"leaves the rest showing the previous frame")
        checked += 1
    assert checked == 2, f"only {checked} of the two diagnostic canvases were checked"
    win.close()


def test_the_campbell_panel_is_readable_at_the_height_it_gets(qapp, tmp_path):
    """The Campbell diagram is the one Preview panel whose own figure title WAS its label.

    At the height the panel actually gets on a laptop — 130 px measured — matplotlib cannot
    fit that title: it was drawn with its top 10 px outside the figure, so it read as a
    half-cut "Campbell diagram", and the rotated y label came out as "olume above end-ex".
    Both were true of v2.3.3 as well; this is not a regression, it is a defect that a
    screenshot for the website finally made visible.

    The title now belongs to the panel header, which cannot clip, and the figure carries it
    ONLY for the export — a stand-alone figure for a report has no header around it.
    """
    from PySide6.QtWidgets import QLabel
    win, pv = _noise_tab(qapp, tmp_path, 800)
    pv.subtabs.setCurrentIndex(0)                      # the Mechanics page
    for _ in range(10):
        qapp.processEvents()
    pv._draw_campbell({1: _BREATH})
    for _ in range(6):
        qapp.processEvents()

    ax = pv.campbell.figure.axes[0]
    assert ax.get_title() == "", (
        "the figure sets its own title on screen again — at this panel height matplotlib "
        "draws it outside the figure and it is clipped")
    # ...and the panel names it, so nothing was lost by taking the title off the figure
    box = pv.campbell.parent()
    headers = [w.text() for w in box.findChildren(QLabel)] if box is not None else []
    assert any("Campbell" in h for h in headers), (
        f"no panel header names the Campbell diagram; found {headers}")
    win.close()


def test_the_campbell_labels_shorten_and_come_back(qapp, tmp_path):
    """The y label is rotated, so its length runs along the axes HEIGHT and it clips exactly
    as an x label clips against the width. It must shorten to fit and lengthen when it can."""
    win, pv = _noise_tab(qapp, tmp_path, 800)
    pv.subtabs.setCurrentIndex(0)
    for _ in range(10):
        qapp.processEvents()
    pv._draw_campbell({1: _BREATH})
    for _ in range(6):
        qapp.processEvents()

    def state():
        c = pv.campbell
        c.draw()
        r = c.get_renderer()
        ax = c.figure.axes[0]
        over = (ax.yaxis.label.get_window_extent(renderer=r).height
                - ax.get_window_extent(renderer=r).height)
        return ax.get_ylabel(), max(0.0, over), ax.get_legend() is not None

    short_label, over, _ = state()
    assert over <= 1.0, f"the y label {short_label!r} overruns the axes by {over:.0f} px"

    win.resize(1500, 1400)
    for _ in range(14):
        qapp.processEvents()
    tall_label, over_tall, legend_tall = state()
    assert over_tall <= 1.0, f"the y label {tall_label!r} overruns by {over_tall:.0f} px"
    assert len(tall_label) > len(short_label), (
        f"the y label stayed {tall_label!r} on a tall panel after shortening to "
        f"{short_label!r} — it is not re-chosen")
    assert legend_tall, "the legend never comes back on a panel with room for it"
    win.close()


def test_the_noise_tab_divides_into_thirds(qapp, tmp_path):
    """What Emil actually asked for: chrome+settings / the two working views / the three
    reference panels, each roughly a third of the window at the default size."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QSplitter
    win, pv = _noise_tab(qapp, tmp_path, 800)
    page = pv.subtabs.currentWidget().widget()
    splits = [w for w in page.findChildren(QSplitter) if w.orientation() == Qt.Vertical]
    assert splits, "no vertical splitter on the noise page — nothing to measure"
    sizes = splits[0].sizes()
    assert len(sizes) == 3, f"expected three stacked regions, got {len(sizes)}"
    total = win.height()
    shares = {
        "chrome+settings": (total - sum(sizes)) / total,
        "working views": (sizes[0] + sizes[1]) / total,
        "reference row": sizes[2] / total,
    }
    for name, share in shares.items():
        assert 0.25 <= share <= 0.42, (
            f"{name} takes {share:.0%} of the window, not roughly a third "
            f"(all three: { {k: f'{v:.0%}' for k, v in shares.items()} })")
    win.close()
