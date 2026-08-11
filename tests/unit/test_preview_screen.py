"""Preview screen as an analysis & QC surface: the Campbell diagram's recoil line +
shaded work + WOB (P12), the EMG RMS envelope (P13), the batch-QC overview (P16), the
crosshair + Campbell export (P17) and zero-reference baselines (P22). Previously
test_review_wave4.py; the preview tests from waves 1 (P2) and 6 (P27) are appended below."""
import os

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPointF, Qt

from respmech.ui.screens.preview._plot_helpers import BreathSpansItem
from respmech.ui.state import AppState

from _helpers import INPUT, requires_synth, synth_settings

pytestmark = requires_synth()


def _render_mech(pv, s):
    from respmech.ui.workers import stage_mechanics_preview
    pv._refresh_files(); pv.file_rail.select_filename("synth_case_A.csv")
    pv._render_preview(stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv")))


# --------------------------------------------------------------------------- #
# P12 — Campbell shows the work of breathing
# --------------------------------------------------------------------------- #
def test_campbell_has_recoil_line_and_shaded_work(qapp):
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    fr = run_batch(s).ok_files["synth_case_A.csv"]
    pv._draw_campbell(fr.breaths)
    ax = pv.campbell.figure.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert {"average breath", "elastic recoil", "inspiratory work"} <= set(labels)
    assert len(ax.collections) >= 1                      # the shaded work polygon
    assert any("WOB" in t.get_text() for t in ax.texts)  # the read-out
    win.close()


# --------------------------------------------------------------------------- #
# D12 (UI-overhaul) — the preview's Campbell diagram matches the written figure
# --------------------------------------------------------------------------- #
def test_campbell_preview_orientation_matches_the_written_figure(qapp):
    """Before D12 the preview plotted Poes on x / volume on y while the written Campbell PDF
    (core/plots._pv_average) plots volume on x (inverted) / Poes on y -- the same breaths
    came out as mirror images of each other. Emil's decision, 04-08-2026, was to fix the
    preview to match the writer's 1.x-inherited convention, not the other way around."""
    import inspect
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch
    from respmech.core import plots
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    fr = run_batch(s).ok_files["synth_case_A.csv"]
    pv._draw_campbell(fr.breaths)
    ax = pv.campbell.figure.axes[0]
    assert ax.xaxis_inverted()
    assert "EELV" in ax.get_xlabel() or "end-expiration" in ax.get_xlabel()
    kept = [b for b in fr.breaths.values() if not b["ignored"]]
    loop_line = ax.lines[0]   # the first ax.plot() in _draw_campbell, one call per kept breath
    np.testing.assert_array_equal(loop_line.get_xdata(), kept[0]["volume"])
    np.testing.assert_array_equal(loop_line.get_ydata(), kept[0]["poes"])
    # _pv_average writes straight to a PDF and returns no inspectable Axes, so the writer's
    # own orientation is verified structurally rather than by rendering it a second time.
    src = inspect.getsource(plots._pv_average)
    assert 'ax.plot(b["volume"], b["poes"]' in src
    assert "ax.invert_xaxis()" in src
    win.close()


def test_campbell_recoil_line_and_shaded_work_match_the_new_orientation(qapp):
    """test_campbell_preview_orientation_matches_the_written_figure only checks the bare
    per-breath loop line (the first ax.plot() call). _overlay_campbell_work draws three more
    data-derived pieces -- the average-breath line, the elastic recoil line, and the shaded
    inspiratory-work polygon (fill_between, swapped from fill_betweenx by D12) -- and a
    transposition bug in any of them would show the same defect this ticket exists to fix
    (recoil line / shaded work landing in the wrong place relative to the loop) without
    tripping the orientation test above. Render the real overlay and check each piece's own
    data against the breath dict, not just that the labels/collection count exist (the older
    P12 test above does only that, and would pass unchanged even mirror-imaged)."""
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    fr = run_batch(s).ok_files["synth_case_A.csv"]
    pv._draw_campbell(fr.breaths)
    ax = pv.campbell.figure.axes[0]
    kept = [b for b in fr.breaths.values() if not b["ignored"]]
    b = kept[0]
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    avg = by_label["average breath"]
    np.testing.assert_array_equal(avg.get_xdata(), b["volumeavg"])
    np.testing.assert_array_equal(avg.get_ydata(), b["poesavg"])

    eelv, eilv = b["eelvavg"], b["eilvavg"]   # each [volume, poes]
    recoil = by_label["elastic recoil"]
    np.testing.assert_array_equal(recoil.get_xdata(), [eelv[0], eilv[0]])
    np.testing.assert_array_equal(recoil.get_ydata(), [eelv[1], eilv[1]])

    # fill_between(x, y1, y2) fills VERTICALLY as a function of x, so the shaded polygon's
    # x-extent must span the inspiratory VOLUME values. Had this stayed fill_betweenx (or been
    # swapped back to it by mistake), the polygon's x-extent would instead span the Poes
    # values -- a different, wrong number range that this assertion catches.
    poly = next(c for c in ax.collections if c.get_label() == "inspiratory work")
    verts = poly.get_paths()[0].vertices
    iv = np.asarray(b["inspiration"]["volumeavg"], float)
    assert np.isclose(verts[:, 0].min(), iv.min())
    assert np.isclose(verts[:, 0].max(), iv.max())
    win.close()


def test_campbell_volume_axis_label_always_carries_the_datum():
    """Every candidate wording for the volume axis must name its datum (EELV or
    end-expiration). "Volume (L)" alone reads as absolute lung volume on a Campbell diagram,
    a different quantity from volume above EELV -- the ambiguity a short panel used to fall
    back to before this ticket."""
    from respmech.ui.screens.preview._mechanics import _CAMPBELL_XLABEL_VARIANTS
    assert len(_CAMPBELL_XLABEL_VARIANTS) >= 2   # more than one rung, or there is no ladder
    for variant in _CAMPBELL_XLABEL_VARIANTS:
        assert "EELV" in variant or "end-expiration" in variant, variant


def test_campbell_volume_label_fits_the_narrow_panel_it_actually_gets(qapp):
    """Measured on the target 1280x760 window: the Campbell panel is 295x130 px. Before D12
    the volume label lived on the Y axis there, with only 63.3 px of vertical room -- not
    enough for anything but the ambiguous "Volume (L)". After the axis swap it lives on X,
    where the same panel gives it 295 px. Assert the pick as a FIT (rendered width against
    the room actually measured this run), never as a hard-coded pixel number -- macOS and
    Windows measure text extents differently."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from respmech.ui.screens.preview._figure_fit import _pick_xlabel
    from respmech.ui.screens.preview._mechanics import _CAMPBELL_XLABEL_VARIANTS
    fig = Figure(figsize=(4, 4))
    canvas = FigureCanvasQTAgg(fig)
    canvas.resize(295, 130)
    ax = fig.add_subplot(111)
    chosen = _pick_xlabel(canvas, ax, _CAMPBELL_XLABEL_VARIANTS)
    assert "EELV" in chosen or "end-expiration" in chosen
    r = canvas.get_renderer()
    extent = ax.xaxis.label.get_window_extent(renderer=r)
    centre = (extent.x0 + extent.x1) / 2.0
    room = 2.0 * min(centre, float(canvas.figure.bbox.width) - centre)
    assert extent.width <= room + 1.0   # +1 px float/antialiasing slack, not a text-width literal


def test_wob_annotation_contrasts_with_its_background_and_the_loop_colour(qapp):
    """The WOB read-out used to be drawn in pal["mpl_loop"] -- the exact colour of the loops
    it sits on top of -- which measured 3.35:1 against white at 8 pt, under WCAG's 4.5:1
    body-text floor. It now uses pal["fg"] at 9 pt bold with a background box: a colour
    distinct from the loops in both themes, and >=4.5:1 against the figure background in
    light theme."""
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch
    from respmech.ui import theme as _theme
    import matplotlib.colors as mcolors

    def _rel_luminance(rgb):
        out = []
        for v in rgb[:3]:
            out.append(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
        return 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]

    def _contrast(a, b):
        la, lb = _rel_luminance(a) + 0.05, _rel_luminance(b) + 0.05
        return max(la, lb) / min(la, lb)

    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    fr = run_batch(s).ok_files["synth_case_A.csv"]
    for pal in (_theme._PLOT_LIGHT, _theme._PLOT_DARK):
        pv._draw_campbell(fr.breaths, pal=pal)
        ax = pv.campbell.figure.axes[0]
        wob_text = next(t for t in ax.texts if "WOB" in t.get_text())
        fg_rgb = mcolors.to_rgb(pal["fg"])
        loop_rgb = mcolors.to_rgb(pal["mpl_loop"])
        text_rgb = mcolors.to_rgb(wob_text.get_color())
        assert text_rgb == fg_rgb
        assert text_rgb != loop_rgb   # not the colour of the loops it is drawn over
        assert wob_text.get_bbox_patch() is not None   # background box keeps it legible
        if pal is _theme._PLOT_LIGHT:
            assert _contrast(text_rgb, mcolors.to_rgb(pal["mpl_bg"])) >= 4.5
    win.close()


def test_mechanics_splitter_gives_the_diagram_more_room_than_before(qapp):
    """The table used to outweigh the diagram 3:1 (setStretchFactor(0,3)/(1,1),
    setSizes([720, 240])) -- Emil measured a maximised window at 1265 px of table against
    406x249 of diagram, three times the weight in favour of the number dump. The new ~58/42
    split (stretch 7:5, setSizes([560, 400])) is asserted as the RATIO the splitter actually
    reaches, never as a pixel literal: Qt redistributes setSizes() against whatever width the
    splitter is given at construction time, so only the ratio survives across environments."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    # D22 (UI-overhaul) wrapped the table in a titled panel (like the Campbell diagram
    # beside it), so the splitter's direct child is now that panel, not the table itself.
    lower = pv._table_panel.parentWidget()
    table_w, diagram_w = lower.sizes()
    assert diagram_w > 0
    ratio = table_w / diagram_w
    assert 1.2 <= ratio <= 1.6, ratio   # 560/400 == 1.4; the OLD split measured 720/240 == 3.0
    win.close()


# --------------------------------------------------------------------------- #
# P13 — RMS envelope
# --------------------------------------------------------------------------- #
def test_rms_envelope_helper():
    from respmech.ui.screens.preview_screen import _rms_envelope
    x = np.array([3.0, 4.0, 0.0, 0.0])
    env = _rms_envelope(x, 2)
    assert env.shape == x.shape and np.isfinite(env).all()
    assert env[1] == pytest.approx(np.sqrt((9 + 16) / 2))   # window over samples 0..1
    # a constant signal's RMS envelope equals its amplitude
    assert _rms_envelope(np.full(50, 2.0), 5) == pytest.approx(np.full(50, 2.0))


def test_emg_detail_overlays_rms_envelope(qapp):
    from respmech.ui.main_window import MainWindow
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    t = np.linspace(0, 1, 1000)
    raw = np.sin(2 * np.pi * 50 * t)
    pv.render_emg_time({"t": t, "raw": raw, "ecg": raw * 0.9, "noise": raw * 0.8,
                        "noise_applied": True})
    names = [it.name() for it in pv.emg_plots.getPlotItem().listDataItems() if it.name()]
    assert "RMS envelope" in names
    win.close()


# --------------------------------------------------------------------------- #
# P16 — batch QC overview
# --------------------------------------------------------------------------- #
def test_qc_overview_ok_and_warn(qapp):
    from types import SimpleNamespace
    import pandas as pd
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    fr = run_batch(s).ok_files["synth_case_A.csv"]
    pv._update_qc_overview(fr)
    assert pv.qc_overview.property("status") == "ok"
    assert "breaths used" in pv.qc_overview.text() and "no flags" in pv.qc_overview.text()
    # a non-physiological breath (Vt ≤ 0) trips the warn flag
    bad = SimpleNamespace(breaths=fr.breaths,
                          breaths_table=pd.DataFrame({"vt": [0.5, -0.1], "wobtotal": [1.0, 2.0]}))
    pv._update_qc_overview(bad)
    assert pv.qc_overview.property("status") == "warn"
    assert "vt≤0" in pv.qc_overview.text()
    win.close()


def test_qc_overview_names_a_carried_over_exclusion(qapp, tmp_path):
    """Ticket B06: the QC line must say when the excluded count includes breaths carried
    over from a different recordings folder — not just report a bare number, which is
    exactly what made the original bug invisible."""
    from respmech.core.settings import ExcludeEntry
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    pv._toggle_breath(a_breath)
    fr = run_batch(s).ok_files["synth_case_A.csv"]
    pv._update_qc_overview(fr)
    assert "excluded" in pv.qc_overview.text() and "carried over" not in pv.qc_overview.text()
    # restamp the SAME entry to a different folder, without touching its breaths — the
    # "switched folders, never revisited this file" case
    entry = next(e for e in s.processing.exclude_breaths if e.file == "synth_case_A.csv")
    entry.folder = str(tmp_path / "a-different-folder")
    pv._update_qc_overview(fr)
    assert "carried over from a previous recordings folder" in pv.qc_overview.text()
    win.close()


def test_breath_brush_hatches_a_carried_exclusion_not_an_ordinary_one(qapp):
    """The overlay's own visual distinction (ticket B06 point 3): an ordinary exclusion is
    solid, a carried-over one is hatched — never the same brush style, so the two are never
    visually indistinguishable, and an included breath ignores `carried` entirely (there is
    nothing to distinguish when nothing is excluded)."""
    from PySide6.QtCore import Qt
    from respmech.ui.main_window import MainWindow
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    assert pv._breath_brush(False, carried=True).style() == pv._breath_brush(False, carried=False).style()
    solid = pv._breath_brush(True, carried=False)
    hatched = pv._breath_brush(True, carried=True)
    assert solid.style() == Qt.SolidPattern
    assert hatched.style() != Qt.SolidPattern
    assert solid.style() != hatched.style()
    win.close()


# --------------------------------------------------------------------------- #
# B02 — the file rail's per-file state + the QC chip/'Process & write' reset discipline
# --------------------------------------------------------------------------- #
def test_switching_to_an_unloadable_file_leaves_qc_chip_and_process_button_reset(qapp, tmp_path, monkeypatch):
    """A file switch (_begin_file_switch -> _clear_file_panels) resets the QC chip to
    neutral and disables 'Process & write this file' BEFORE any job has had a chance to
    say whether the new file even loads — only a successful mechanics render
    (_render_preview) may re-enable it. A file that fails to load must therefore leave
    both exactly where the switch left them: this is B02's nulstillingsdiscipline,
    extended from _forget_campbell's existing export-button pattern."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.screens.preview import _mechanics
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    pv._refresh_files(); pv.file_rail.select_index(0)
    pv._preview()
    assert pv.btn_process_file.isEnabled() is True     # a valid render enabled it

    monkeypatch.setattr(_mechanics, "stage_mechanics_preview",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("cannot read file")))
    pv.file_rail.select_index(1)                        # -> _begin_file_switch -> _clear_file_panels
    assert pv.qc_overview.text() == "QC:  —"
    assert pv.qc_overview.property("status") == "muted"
    assert pv.btn_process_file.isEnabled() is False
    pv._preview()                                        # fails; never reaches _render_preview
    assert pv.qc_overview.text() == "QC:  —"
    assert pv.qc_overview.property("status") == "muted"
    assert pv.btn_process_file.isEnabled() is False
    win.close()


def test_a_failed_test_run_marks_the_qc_chip_not_assessed_and_the_rail_row_failed(qapp, tmp_path):
    """The 'batch' job's own failure (a worker-level exception, not a soft per-file
    precondition) must present as 'QC:  not assessed — …' with status 'warn' — never
    'ok' — and must mark the previewed file's rail row failed. Purely presentational: no
    computed value is involved (ticket requirement)."""
    from PySide6.QtCore import QThread
    from respmech.ui.main_window import MainWindow
    from respmech.ui.screens.preview_screen import _Job
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    pv._refresh_files(); pv.file_rail.select_index(0)
    pv._preview()
    name = pv._selected_filename()
    job = _Job("batch", pv._tokens["batch"], QThread(), object())
    job.error = "OSError: disk read error"
    pv._jobs["batch"] = job
    pv._on_job_done(job, None)
    assert pv.qc_overview.property("status") == "warn"
    assert "not assessed" in pv.qc_overview.text().lower()
    entry = pv.file_rail.entry(name)
    assert entry.verdict == "failed" and "disk read error" in entry.error
    win.close()


def test_qc_chip_resets_on_a_rendering_bug_in_the_batch_render(qapp, tmp_path, monkeypatch):
    """Self-review finding: a genuine rendering bug inside _on_batch_result (NOT a
    per-file analysis error, which _FileRunError already covers) must not leave the QC
    chip showing a stale prior 'ok' verdict under a 'display error' card."""
    from PySide6.QtCore import QThread
    from respmech.ui.main_window import MainWindow
    from respmech.ui.screens.preview_screen import _Job
    from respmech.core.pipeline import run_batch
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    result = run_batch(s, only_files=["synth_case_A.csv"])
    pv._on_batch_result(result)
    assert pv.qc_overview.property("status") == "ok"    # a real prior verdict exists

    def _boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(pv, "_fill_table", _boom)
    job = _Job("batch", pv._tokens["batch"], QThread(), object())
    pv._jobs["batch"] = job
    pv._on_job_done(job, result)
    assert pv.qc_overview.property("status") == "warn"
    assert "not assessed" in pv.qc_overview.text().lower()
    win.close()


def test_toggling_a_breath_updates_the_rails_exclusion_badge(qapp, tmp_path):
    name = "synth_case_A.csv"
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    pv._toggle_breath(a_breath)
    assert pv.file_rail.entry(name).excluded_count == 1
    pv._toggle_breath(a_breath)                          # toggle back off
    assert pv.file_rail.entry(name).excluded_count == 0
    win.close()


def test_exclusions_key_on_the_exact_basename_and_survive_a_save_reload_round_trip(qapp, tmp_path):
    """B02 moved the previewed file's identity off file_combo.currentText() onto
    FileRail.current_filename() (via PreviewScreen._selected_filename()) — the
    exclude_breaths key must still be exactly the file's basename, unchanged, and a
    saved analysis must reload with its exclusions intact and visible on the rail
    without the file having to be re-selected first (ticket requirement)."""
    from respmech.ui.main_window import MainWindow
    from respmech.settingsio.toml_io import load_toml, save_toml
    name = "synth_case_A.csv"
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    pv._toggle_breath(a_breath)
    entry = next(e for e in s.processing.exclude_breaths if e.file == name)
    assert entry.breaths == [a_breath]

    toml_path = tmp_path / "analysis.toml"
    save_toml(s, toml_path)
    reloaded = load_toml(toml_path)
    entry2 = next(e for e in reloaded.processing.exclude_breaths if e.file == name)
    assert entry2.breaths == [a_breath]

    win2 = MainWindow(AppState(reloaded)); pv2 = win2.preview_screen
    pv2.refresh_files()
    assert pv2.file_rail.entry(name).excluded_count == 1     # visible WITHOUT selecting it first
    win2.close()
    win.close()


def test_sync_rail_exclusions_clears_a_stale_badge_from_a_previous_analysis(qapp, tmp_path):
    """Self-review finding: _sync_rail_exclusions only ever ADDED/updated a badge — it
    never zeroed one for a file that persists in the rail (set_manifest preserves state
    across a rebuild) but the CURRENT analysis names no exclusion for. Opening a second
    analysis over the same folder/mask, with no exclusion where the first analysis had
    one, must not leave the first analysis's stale count on screen."""
    from respmech.ui.main_window import MainWindow
    name = "synth_case_A.csv"
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    pv._toggle_breath(a_breath)
    assert pv.file_rail.entry(name).excluded_count == 1

    # a second analysis over the SAME folder/mask (so the rail's rows persist across the
    # rebuild), with NO exclusions at all
    s2 = synth_settings(str(tmp_path))
    win.state.settings = s2
    pv.sync_from_settings()
    assert pv.file_rail.entry(name).excluded_count == 0, (
        "the previous analysis's exclusion badge must not survive switching analyses")
    win.close()


# --------------------------------------------------------------------------- #
# B06 — carried-over exclusions are named, not silently reapplied
# --------------------------------------------------------------------------- #
def test_switching_folders_names_the_carried_exclusion_and_clear_actually_changes_compute(
        qapp, tmp_path):
    """The ticket's own reproduction: two folders, each with a file of the same name,
    breaths excluded in the first. Switching to the second must not apply them invisibly
    — but MUST still apply them (unchanged numeric behaviour, matching the pre-ticket
    result) until the user explicitly clears them, at which point compute's own output —
    not just what the file rail/overlay draw — demonstrably changes. Asserts on
    run_batch's actual breath count in both branches, per the ticket's own instruction not
    to trust the interface alone."""
    import shutil
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch
    from respmech.core.settings import clear_carried_over, is_carried_folder

    name = "sample_recording.csv"
    s01 = tmp_path / "S01"; s01.mkdir()
    s02 = tmp_path / "S02"; s02.mkdir()
    shutil.copy(os.path.join(INPUT, "synth_case_A.csv"), s01 / name)
    shutil.copy(os.path.join(INPUT, "synth_case_A.csv"), s02 / name)

    s = synth_settings(str(tmp_path / "out"))
    s.input.folder = str(s01)
    s.input.files = name
    win = MainWindow(AppState(s)); pv = win.preview_screen
    pv._refresh_files(); pv.file_rail.select_filename(name)
    from respmech.ui.workers import stage_mechanics_preview
    pv._render_preview(stage_mechanics_preview(s, str(s01 / name)))
    a_breath = next(iter(pv._breath_spans))
    pv._toggle_breath(a_breath)                      # excluded in S01
    entry = next(e for e in s.processing.exclude_breaths if e.file == name)
    assert entry.folder == str(s01)                  # stamped with the folder it was made in

    total_breaths = len(pv._breath_spans)
    baseline = run_batch(s).ok_files[name]
    assert len(baseline.breaths_table) == total_breaths - 1   # the exclusion took effect

    # switch to S02 — same filename, no user interaction with THIS file's exclusions
    s.input.folder = str(s02)
    pv.refresh_files()                                # what MainWindow wires inputs_changed to
    assert is_carried_folder(entry.folder, s.input.folder) is True
    pv.file_rail.select_filename(name)
    assert pv._exclusion_carried_for(name) is True
    assert pv.file_rail.entry(name).excluded_carried is True

    # named, not silently applied — but NOT silently dropped either: the same list still
    # gives the same number until the user explicitly says otherwise (no golden change).
    still_applied = run_batch(s).ok_files[name]
    assert len(still_applied.breaths_table) == total_breaths - 1

    clear_carried_over(s)                             # the banner's "Clear" action
    assert not any(e.file == name for e in s.processing.exclude_breaths)
    cleared = run_batch(s).ok_files[name]
    assert len(cleared.breaths_table) == total_breaths   # compute itself now excludes nothing
    win.close()


def test_toggling_one_breath_does_not_launder_the_rest_of_a_carried_entry(qapp, tmp_path):
    """Self-review finding: entry.folder is ONE tag for the whole file, but a click only
    ever decides ONE breath. Un-excluding breath #1 of a carried entry that ALSO still
    excludes breath #2 (which this click never touched) must not silently promote #2 to
    'confirmed for this folder' too — only a fresh entry (created here) or the Setup
    banner's explicit Clear may change what an entry's folder tag says."""
    from respmech.core.settings import ExcludeEntry, is_carried_folder
    from respmech.ui.main_window import MainWindow
    other_folder = str(tmp_path / "a-different-folder")     # never actually loaded/scanned
    s = synth_settings("")                                   # s.input.folder stays INPUT (real)
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    spans = sorted(pv._breath_spans)
    b1, b2 = spans[0], spans[1]
    s.processing.exclude_breaths.append(
        ExcludeEntry(file="synth_case_A.csv", breaths=[b1, b2], folder=other_folder))  # carried

    pv._toggle_breath(b1)                              # un-exclude b1 only
    entry = next(e for e in s.processing.exclude_breaths if e.file == "synth_case_A.csv")
    assert entry.breaths == [b2]
    assert entry.folder == other_folder, (
        "an EXISTING entry's folder must not be restamped by a plain toggle")
    assert is_carried_folder(entry.folder, s.input.folder) is True   # b2 still reads carried
    win.close()


def test_a_brand_new_entry_is_still_stamped_fresh(qapp, tmp_path):
    """The other half of the same fix: a file with NO prior exclusion at all must still get
    a fresh, current-folder entry on its very first toggle — only an EXISTING entry is left
    alone."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))
    pv._toggle_breath(a_breath)
    entry = next(e for e in s.processing.exclude_breaths if e.file == "synth_case_A.csv")
    assert entry.folder == s.input.folder
    win.close()


def test_qc_overview_checks_the_result_files_own_name_not_the_selected_one(qapp, tmp_path):
    """Self-review finding: _on_batch_result's fallback (result.files.get(cur) or
    next(iter(...))) can hand _update_qc_overview a DIFFERENT file's result than the one
    currently selected. The carried-over check must follow fr's OWN filename, or a stale/
    mismatched result could blame the wrong file's exclusions."""
    from types import SimpleNamespace
    import pandas as pd
    from respmech.core.settings import ExcludeEntry
    from respmech.ui.main_window import MainWindow
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    # "selected" file has no carried exclusion; a DIFFERENT file (never selected) does
    s.processing.exclude_breaths.append(
        ExcludeEntry(file="synth_case_B.csv", breaths=[1], folder="/a/different/folder"))
    fr_for_other_file = SimpleNamespace(
        file="synth_case_B.csv", breaths=[1, 2],
        breaths_table=pd.DataFrame({"vt": [0.5]}))
    pv._update_qc_overview(fr_for_other_file)
    assert "carried over from a previous recordings folder" in pv.qc_overview.text()
    win.close()


# --------------------------------------------------------------------------- #
# P17 — crosshair + export
# --------------------------------------------------------------------------- #
def test_crosshair_and_export(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    assert len(pv._crosshair_lines) == len(pv._channel_plots)
    assert all(not ln.isVisible() for ln in pv._crosshair_lines)   # hidden until hover
    p0 = pv._channel_plots[0]
    pv._on_mech_mouse_moved([QPointF(p0.sceneBoundingRect().center())])
    assert pv._crosshair_lines[0].isVisible()
    assert "t = " in pv.crosshair_label.text()
    # export is gated until a Campbell exists, then writes a real file
    assert not pv.btn_export_fig.isEnabled()
    pv._draw_campbell(run_batch(s).ok_files["synth_case_A.csv"].breaths)
    assert pv.btn_export_fig.isEnabled()
    out = str(tmp_path / "campbell.png")
    pv.campbell.figure.savefig(out)
    assert os.path.getsize(out) > 0
    win.close()


# --------------------------------------------------------------------------- #
# P22 — zero-reference baselines
# --------------------------------------------------------------------------- #
def test_zero_baseline_on_every_channel(qapp):
    from respmech.ui.main_window import MainWindow
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    for p in pv._channel_plots:
        horiz = [it for it in p.items
                 if isinstance(it, pg.InfiniteLine) and float(it.angle) % 180 == 0]
        assert horiz, "channel missing its zero baseline"
    win.close()


# ---------------------------------------------------------------------------
# Keyboard file-stepping (P27) — from wave 6
# ---------------------------------------------------------------------------
def test_file_stepping(qapp):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.manifest import manifest_from_filenames
    win = MainWindow(AppState()); pv = win.preview_screen
    pv.file_rail.set_manifest(manifest_from_filenames("", ["f1.csv", "f2.csv", "f3.csv"]))
    pv.file_rail.select_index(0)
    pv._step_file(+1); assert pv.file_rail.current_filename() == "f2.csv"
    pv._step_file(+1); pv._step_file(+1)                 # clamps at the end
    assert pv.file_rail.current_filename() == "f3.csv"
    pv._step_file(-1); assert pv.file_rail.current_filename() == "f2.csv"
    win.close()


# ---------------------------------------------------------------------------
# Keyboard, focus and accessible names (C02) — a focused widget must see its own
# keys before a window-wide shortcut steals them, and icon-only controls need a
# real accessible name.
# ---------------------------------------------------------------------------
def test_pagedown_scrolls_the_focused_table_not_the_file_selection(qapp):
    """Before this ticket: PageUp/PageDown were registered on the whole PreviewScreen
    with Qt's default WindowShortcut context, so they fired regardless of which widget
    actually had focus. A user scrolling the breath table with PageDown got the
    recording switched out from under them instead — measured as
    ``table hasFocus: True``, ``vbar 0 -> 0, step calls: [1]``."""
    import pandas as pd
    from PySide6.QtTest import QTest
    from respmech.ui.main_window import MainWindow

    win = MainWindow(AppState())
    pv = win.preview_screen
    win.resize(1000, 700)
    win.show()
    win.activateWindow()
    # QShortcut's WindowShortcut/WidgetWithChildrenShortcut matching requires the
    # widget chain to actually be VISIBLE, not merely embedded — with Setup as the
    # still-current tab, ``pv``/``pv.table`` report ``isVisible() == False`` even
    # though the offscreen platform happily lets a hidden widget claim ``hasFocus()``
    # (found while writing this test: without this line the bug this test targets
    # cannot reproduce at all, fixed or not, and the test is worthless either way).
    win.tabs.setCurrentWidget(pv)
    for _ in range(5):
        qapp.processEvents()
    assert pv.isVisible() and pv.table.isVisible()

    # Hundreds of rows, matching a real subject's export — enough to overflow whatever
    # vertical room the table actually gets, regardless of window/splitter sizing.
    pv._table_model.set_dataframe(pd.DataFrame({"breath_no": range(1, 301),
                                                "value": range(1, 301)}))
    for _ in range(3):
        qapp.processEvents()

    steps = []
    pv.file_rail.step = lambda delta: steps.append(delta)   # spy: the shortcut calls this
    # directly (see the fix — no wrapping through _step_file), so this is what stealing
    # the key would actually invoke.

    pv.table.setCurrentIndex(pv._table_model.index(0, 0))   # a user has clicked a cell
    pv.table.setFocus(Qt.FocusReason.OtherFocusReason)
    qapp.processEvents()
    assert pv.table.hasFocus(), "the breath table did not actually take focus"
    row_before = pv.table.currentIndex().row()
    assert row_before == 0

    QTest.keyClick(pv.table, Qt.Key.Key_PageDown)
    qapp.processEvents()

    # QAbstractItemView's own PageDown handling moves the CURRENT ROW by a page — the
    # observable, reliable signal that the table itself consumed the key. (Its vertical
    # scrollbar's raw ``value()`` is a rendering detail that does not reliably update
    # under a headless/offscreen platform even when the current row visibly changes.)
    assert pv.table.currentIndex().row() > row_before, (
        "PageDown did not move the focused table's current row")
    assert steps == [], f"PageDown incorrectly changed the file selection: {steps}"
    win.close()


def test_file_nav_shortcuts_registered_with_native_tooltips(qapp):
    """Ctrl+[ / Ctrl+] (drawn as the Safari/Xcode ⌘[/⌘] pair) and Alt+Left/Alt+Right are
    the new primary file-nav shortcuts. PageUp/PageDown are kept for continuity but
    rescoped to the file rail (WidgetWithChildrenShortcut — see the test above), and the
    tooltips no longer hardcode 'PgUp'/'PgDn', keys no Mac laptop has."""
    from PySide6.QtGui import QShortcut, QKeySequence
    from respmech.ui.main_window import MainWindow

    win = MainWindow(AppState())
    pv = win.preview_screen
    shortcuts = pv.findChildren(QShortcut)

    def _has(seq):
        return any(sc.key() == QKeySequence(seq) for sc in shortcuts)

    assert _has("Ctrl+["), "Ctrl+[ (previous file) is not registered"
    assert _has("Ctrl+]"), "Ctrl+] (next file) is not registered"
    assert _has("Alt+Left"), "Alt+Left alias is not registered"
    assert _has("Alt+Right"), "Alt+Right alias is not registered"

    pgup = next(sc for sc in shortcuts if sc.key() == QKeySequence(Qt.Key.Key_PageUp))
    pgdn = next(sc for sc in shortcuts if sc.key() == QKeySequence(Qt.Key.Key_PageDown))
    assert pgup.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut
    assert pgdn.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut

    native = QKeySequence("Ctrl+[").toString(QKeySequence.SequenceFormat.NativeText)
    assert native in pv.btn_prev_file.toolTip()
    assert "PgUp" not in pv.btn_prev_file.toolTip()
    assert "PgDn" not in pv.btn_next_file.toolTip()
    win.close()


def test_accessible_names_on_icon_steppers_file_rail_and_plot_containers(qapp):
    """A QAccessible query on the built Preview screen used to find: the two icon
    steppers announced by their glyph ('◀'/'▶'), the (now-removed) file combo
    announcing only its current VALUE with no label, and every plot container entirely
    nameless (role Client/Table, empty name). ``accessibleName()`` is what QAccessible's
    NameRole falls back to, so asserting it directly is equivalent without needing a
    live accessibility backend in a headless run."""
    from respmech.ui.main_window import MainWindow

    win = MainWindow(AppState())
    pv = win.preview_screen
    assert pv.btn_prev_file.accessibleName() == "Previous file"
    assert pv.btn_next_file.accessibleName() == "Next file"
    assert pv.file_rail.view.accessibleName(), "the file rail's list has no accessible name"
    assert pv.file_rail.filter_edit.accessibleName() == "Filter files"
    assert pv.plots.accessibleName() == "Mechanics signals"
    assert pv.table.accessibleName() == "Per-breath results"
    assert pv.campbell.accessibleName() == "Campbell diagram"
    assert pv.emg_raw_plots.accessibleName() == "Raw EMG channels"
    win.close()


# --------------------------------------------------------------------------- #
# P28 — detected-format read-out
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(os.path.join(INPUT, "synth_case_A.csv")),
                    reason="synthetic input absent")


# ---------------------------------------------------------------------------
# Live recompute on breath exclusion (P2) — from wave 1
# ---------------------------------------------------------------------------
def test_toggle_breath_requests_recompute(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_mechanics_preview
    s = synth_settings(tmp_path)
    win = MainWindow(AppState(s)); pv = win.preview_screen
    pv._refresh_files(); pv.file_rail.select_filename("synth_case_A.csv")
    pv._render_preview(stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv")))
    a_breath = next(iter(pv._breath_spans))
    pv._toggle_breath(a_breath)
    assert pv._batch_recompute_pending is True            # the average is being recomputed…
    assert "recomputing" in pv.status.text().lower()      # …not deferred to a manual run
    win.close()


def test_mech_caption_survives_render_and_toggle(qapp, tmp_path):
    """The Mechanics tab's persistent caption (A03) names the breath count, the exclusion
    count and the click-to-exclude instruction after BOTH _render_preview and
    _toggle_breath — unlike the status line it used to only live in, which an EMG job
    landing a moment later would silently erase (see test_gui_reactive.py for that half)."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_mechanics_preview
    s = synth_settings(tmp_path)
    win = MainWindow(AppState(s)); pv = win.preview_screen
    pv._refresh_files(); pv.file_rail.select_filename("synth_case_A.csv")
    pv._render_preview(stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv")))
    cap = pv.mech_caption.fullText().lower()
    assert "breath" in cap and "click a shaded breath to include/exclude" in cap
    assert ", 0 excluded" not in cap                      # nothing excluded yet -> no count clause
    a_breath = next(iter(pv._breath_spans))
    pv._toggle_breath(a_breath)
    cap = pv.mech_caption.fullText().lower()
    assert ", 1 excluded" in cap
    assert "click a shaded breath to include/exclude" in cap   # the instruction still there
    win.close()


# ---------------------------------------------------------------------------
# D23 (UI-overhaul) — the mechanics time axis names which clock it shows, and the
# analysis window sits at a persistent spot beside the QC chip
# ---------------------------------------------------------------------------
def test_mechanics_axis_names_the_trimmed_window_not_the_raw_file_clock(qapp, tmp_path):
    """The Mechanics stack's x-axis is zero-based on the TRIMMED analysis window
    (``t = data["t"]`` in ``_render_preview_stage1``), while the EMG raw stack is
    zero-based on the file's own untrimmed start (``_render_raw_stack``) — both used
    to be labelled the plain, ambiguous 'Time (s)'. This asserts the mechanics stack
    no longer claims that label, so a user cannot land on the wrong breath after
    reading a time on one stack and looking for it on the other."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_mechanics_preview
    s = synth_settings(tmp_path)
    win = MainWindow(AppState(s)); pv = win.preview_screen
    data = stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv"))
    assert data["startix"] > 0                        # confirms this fixture actually trims
    pv._render_preview(data)
    mech_label = pv._channel_plots[-1].getAxis("bottom").labelText
    assert mech_label != "Time (s)"
    assert "analysis start" in mech_label.lower()
    win.close()


def test_mechanics_axis_relabel_leaves_the_emg_raw_clock_alone(qapp, tmp_path):
    """The EMG raw stack plots the file's own untrimmed clock, so its bottom axis is
    already correct — only the Mechanics stack needed relabelling."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(tmp_path)
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    assert pv._emg_raw_subplots, "synth_case_A must carry EMG channels for this assertion"
    assert pv._emg_raw_subplots[-1].getAxis("bottom").labelText == "Time (s)"
    win.close()


def test_analysis_window_label_names_the_files_actual_trim(qapp, tmp_path):
    """The persistent label beside the QC chip must report THIS file's real trim
    (startix/endix/fs from the same staging dict the stack is drawn from), not a
    placeholder — matched against independently recomputed values, never a literal."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_mechanics_preview
    s = synth_settings(tmp_path)
    win = MainWindow(AppState(s)); pv = win.preview_screen
    data = stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv"))
    pv._render_preview(data)
    fs = data["fs"]
    start_s, end_s = data["startix"] / fs, data["endix"] / fs
    text = pv.mech_window_label.fullText()
    assert "Analysis window" in text
    assert f"{start_s:.2f}" in text
    assert f"{end_s:.2f}" in text
    assert start_s > 0 and "trimmed)" in text          # this fixture trims from the start
    win.close()


def test_analysis_window_label_does_not_claim_a_trim_that_failed(qapp, tmp_path):
    """stage_mechanics_preview's TrimError fallback hands back startix=0, endix=len(flow)
    — the WHOLE untrimmed file, not a real window — because no breath could be segmented
    (self-review of this ticket: feeding those into the normal formula reads as a
    successful 0.00-<total> s trim, while the QC chip is simultaneously reporting the
    failure right next to it). The label must say the trim failed, not silently omit the
    '(… trimmed)' clause and look like an ordinary, successful, un-trimmed recording."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_mechanics_preview
    s = synth_settings(tmp_path)
    win = MainWindow(AppState(s)); pv = win.preview_screen
    data = dict(stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv")))
    total_s = len(data["emg_flow"]) / data["fs"]
    data["startix"], data["endix"], data["trim_error"] = 0, len(data["emg_flow"]), "no breaths"
    pv._render_preview_stage1(data)
    text = pv.mech_window_label.fullText()
    assert "could not trim" in text.lower()
    assert "analysis window" not in text.lower()        # must not read as an ordinary success
    assert f"{total_s:.2f}" in text
    win.close()


def test_analysis_window_label_survives_a_status_message_from_another_screen(qapp, tmp_path):
    """The label lives in its own widget beside the QC chip, not the shared status bar
    every screen's ``status_changed`` feeds (main_window.py) — unlike the trim window's
    old only home, a transient status line the ticket found overwritten mid-session by
    Setup's own message. A later status update, from Setup or from Mechanics itself,
    must not erase it."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(tmp_path)
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    before = pv.mech_window_label.fullText()
    assert before
    win.settings_screen._set_status("Unrelated Setup-screen message")
    pv._set_status("Unrelated Mechanics status update")
    assert pv.mech_window_label.fullText() == before
    win.close()


def test_switching_to_an_unloadable_file_also_clears_the_analysis_window_label(qapp, tmp_path, monkeypatch):
    """Same nulstillingsdiscipline as the QC chip (see
    test_switching_to_an_unloadable_file_leaves_qc_chip_and_process_button_reset):
    a file switch clears the analysis-window label before any job says whether the
    new file even loads, and a load failure must leave it cleared."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.screens.preview import _mechanics
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    pv._refresh_files(); pv.file_rail.select_index(0)
    pv._preview()
    assert pv.mech_window_label.fullText()              # a valid render populated it

    monkeypatch.setattr(_mechanics, "stage_mechanics_preview",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("cannot read file")))
    pv.file_rail.select_index(1)
    assert pv.mech_window_label.fullText() == ""
    pv._preview()                                        # fails; never reaches _render_preview
    assert pv.mech_window_label.fullText() == ""
    win.close()


# ---------------------------------------------------------------------------
# D22 (UI-overhaul) — the per-breath table names its WOB source, only when it matters
# ---------------------------------------------------------------------------
def test_wob_table_note_names_the_average_source_and_clears_for_individual(qapp, tmp_path):
    """When 'Work of breathing from' is Average (the default) every wobtotal-family
    column in the per-breath table repeats one whole-file value; before this ticket
    nothing on screen said so. The table's own titled-panel header (D22) now names it,
    and the note disappears the moment the setting is switched to Individual, where the
    numbers genuinely vary breath to breath."""
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)

    assert s.processing.wob.calc_from == "average"        # the default this ticket covers
    result = run_batch(s, only_files=["synth_case_A.csv"])
    pv._on_batch_result(result)
    title = pv._table_panel._title_label.fullText().lower()
    assert "work of breathing" in title
    assert "averaged breath" in title
    assert "advanced" in title                            # points at where the setting lives

    s.processing.wob.calc_from = "individual"
    result2 = run_batch(s, only_files=["synth_case_A.csv"])
    pv._on_batch_result(result2)
    title2 = pv._table_panel._title_label.fullText()
    assert title2 == "Per-breath results"                  # base title, no WOB qualification
    win.close()


def test_wob_table_note_clears_on_a_soft_precondition_error(qapp, tmp_path):
    """A soft per-file error (e.g. NoBreathsError) clears the table via the REAL
    _on_batch_result path (not by calling _set_wob_table_note directly, which would
    pass even if the wiring at the _SOFT_FILE_ERRORS branch in _mechanics.py were ever
    deleted) — the header note must clear with it, or a stale 'averaged breath'
    qualification would describe a table that no longer has any rows. Self-review
    finding, 10-08-2026: the first version of this test only checked the helper in
    isolation."""
    from respmech.ui.main_window import MainWindow
    from respmech.core.pipeline import run_batch, BatchResult, FileResult
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    result = run_batch(s, only_files=["synth_case_A.csv"])
    pv._on_batch_result(result)
    assert "work of breathing" in pv._table_panel._title_label.fullText().lower()

    soft = BatchResult(files={"synth_case_A.csv": FileResult(
        file="synth_case_A.csv", error="NoBreathsError: no breaths found",
        error_kind="NoBreathsError")})
    pv._on_batch_result(soft)
    assert pv.table.model().rowCount() == 0                # the table really did clear
    assert pv._table_panel._title_label.fullText() == "Per-breath results"
    win.close()


def test_wob_table_note_survives_an_empty_but_column_bearing_table(qapp, tmp_path):
    """All breaths in a file can end up excluded without raising NoBreathsError (that
    fires earlier, at breath separation) — the breaths table can still come back with
    zero rows but the wobtotal column present. ``df['wobtotal'].iloc[0]`` must not
    raise on that; self-review finding, 10-08-2026 (the defensive except-branch existed
    but nothing exercised it)."""
    import pandas as pd
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    empty = pd.DataFrame({"wobtotal": pd.Series([], dtype=float)})
    pv._set_wob_table_note(empty)                            # must not raise
    text = pv._table_panel._title_label.fullText().lower()
    assert "work of breathing" in text and "advanced" in text
    win.close()


def test_wob_table_note_shows_a_nan_wobtotal_plainly(qapp, tmp_path):
    """The core deliberately writes NaN into wob* columns when a detector is
    unreliable rather than an inflated number (see result_table.py's
    _format_display docstring) — the note must render that plainly, not crash."""
    import pandas as pd
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    df = pd.DataFrame({"wobtotal": [float("nan")]})
    pv._set_wob_table_note(df)                               # must not raise
    assert "work of breathing" in pv._table_panel._title_label.fullText().lower()
    win.close()


# ---------------------------------------------------------------------------
# The Campbell export must never write a diagram that is no longer on screen
# ---------------------------------------------------------------------------
def test_clearing_the_campbell_also_disarms_its_export(qapp, tmp_path):
    """"Export Campbell…" re-renders from a cached copy of the breaths (so a dark-mode
    export comes out light), which makes that cache and the drawn figure the same fact. If
    only the figure is cleared, the button stays live over stale data and writes a diagram
    for a file the user is no longer looking at — under that file's name.

    So the invariant is: the export is enabled exactly when there is something to export.
    """
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState(synth_settings(tmp_path)))
    pv = win.preview_screen

    def armed():
        return pv.btn_export_fig.isEnabled(), getattr(pv, "_campbell_breaths", None) is not None

    assert armed() == (False, False), "the export starts armed with nothing drawn"
    for clear in ("_clear_file_panels", "_clear_all_panels"):
        # stand in for a completed render: both halves of "a diagram exists" are set
        pv._campbell_breaths = {"stand-in": 1}
        pv.btn_export_fig.setEnabled(True)
        assert armed() == (True, True)
        getattr(pv, clear)()
        assert armed() == (False, False), (
            f"{clear} left the export armed over a cleared figure")
    win.close()


# ---------------------------------------------------------------------------
# D15 (UI-overhaul) — the mechanics render must not freeze the GUI thread
# ---------------------------------------------------------------------------
def _total_items(plots):
    return sum(len(p.items) for p in plots)


def test_breath_overlay_item_count_grows_by_one_label_per_breath_not_eleven(qapp):
    """The regression this ticket exists to fix: before D15, every breath added 5
    pg.LinearRegionItems + 5 boundary lines + 1 label to the 5-channel mechanics stack
    (11 items/breath). Redrawing with 20 more breaths must now add exactly 20 items
    (one label each) — the aggregate BreathSpansItem count per plot stays fixed at 1,
    however many breaths it carries."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    from respmech.ui.workers import stage_mechanics_preview
    data = stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv"))

    def spans_of(n):
        return [(i + 1, float(i), float(i) + 0.5, False) for i in range(n)]

    pv._render_preview_stage1(data)                # builds the 5 channel plots
    assert len(pv._channel_plots) == 5
    pv._draw_breath_overlays(spans_of(3), label_y=0.0)
    for p in pv._channel_plots:
        assert sum(isinstance(it, BreathSpansItem) for it in p.items) == 1
    base = _total_items(pv._channel_plots)

    pv._render_preview_stage1(data)                # plots.clear() wipes the first draw
    pv._draw_breath_overlays(spans_of(3 + 20), label_y=0.0)
    for p in pv._channel_plots:
        # still exactly ONE aggregate item per plot — not 21
        assert sum(isinstance(it, BreathSpansItem) for it in p.items) == 1
    grown = _total_items(pv._channel_plots)

    assert grown - base == 20, "20 more breaths must add 20 items (labels), not 20*11"
    win.close()


def test_raw_view_breath_overlays_also_use_one_aggregate_item_per_plot(qapp):
    """The same fix applies to _paint_breaths, shared by the raw/detail/result EMG
    views — _render_raw_stack's overlay call was the ticket's other measured cost, and
    the raw view can have more channels (hence more items/breath) than the mechanics
    stack's fixed 5."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    assert pv._emg_raw_subplots, "the fixture must have EMG channels for this test to mean anything"
    for p in pv._emg_raw_subplots:
        assert sum(isinstance(it, BreathSpansItem) for it in p.items) == 1
    win.close()


def test_toggle_breath_recolours_the_shared_span_item_through_the_new_drawing_path(qapp):
    """Click-to-toggle must still change a breath's colour, with the same colours
    _breath_brush already defines — now through BreathSpansItem.set_brush() instead of
    pg.LinearRegionItem.setBrush() on a removed per-breath item."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings("")
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    plot = pv._channel_plots[0]
    span_items = [it for it in plot.items if isinstance(it, BreathSpansItem)]
    assert len(span_items) == 1
    item = span_items[0]

    breath_no = next(iter(pv._breath_spans))
    item_idx = next(idx for (it, idx) in pv._breath_regions[breath_no] if it is item)
    incl_rgb = item._spans[item_idx][2].color().getRgb()
    assert incl_rgb == pv._breath_brush(False).color().getRgb()

    pv._toggle_breath(breath_no)                    # exclude it
    excl_rgb = item._spans[item_idx][2].color().getRgb()
    assert excl_rgb == pv._breath_brush(True).color().getRgb()
    assert excl_rgb != incl_rgb

    pv._toggle_breath(breath_no)                    # re-include it
    back_rgb = item._spans[item_idx][2].color().getRgb()
    assert back_rgb == incl_rgb
    win.close()


def test_busy_overlay_clear_error_keeps_a_legitimately_busy_spinner_alive(qapp):
    """The bug behind the frozen-looking spinner: _clear_panel_overlays()'s stop()
    unconditionally hides the overlay, busy or not. _render_preview_stage1 used to call
    it first thing — which, for the reactive job path, meant the SAME overlay _launch()
    had just shown as busy was force-hidden before a single pixel of the (then still
    fully synchronous) render had been drawn. Because Qt only actually repaints a
    hide() once the event loop next turns, the widget's LAST rendered frame just sat
    there, unchanged, for the whole freeze — indistinguishable from a live but frozen
    spinner. clear_error() is the fix: it must leave a busy overlay exactly as it was,
    and only actually hide something if there was an error card to dismiss."""
    from respmech.ui.screens.preview._busy_overlay import BusyOverlay
    from PySide6.QtWidgets import QWidget
    panel = QWidget()
    ov = BusyOverlay(panel)

    ov.start("Loading…")
    assert ov.is_busy() and ov.error is None
    ov.clear_error()
    assert ov.is_busy(), "clear_error() must not stop a legitimately busy overlay"

    ov.show_error("Channel preview failed", "boom")
    assert not ov.is_busy() and ov.error is not None
    ov.clear_error()
    assert not ov.is_busy() and ov.error is None, "clear_error() must dismiss a stale error card"
    panel.deleteLater()


def test_async_mech_render_defers_overlays_and_raw_stack_across_the_event_loop(qapp, tmp_path):
    """The reactive 'mech' job (_render_preview_async, what _RENDER['mech'] actually
    calls) must not draw the breath overlays and the raw EMG stack in the same
    synchronous call as the channel curves: each is posted with
    QTimer.singleShot(0, …) so the event loop turns between them — this is what
    keeps the busy spinner animated and the GUI responsive during the remaining
    render time, rather than freezing for the whole thing in one Python call.
    Verified structurally (item counts appearing over successive processEvents()
    calls, matching the empirically-confirmed one-processEvents()-per-singleShot(0)
    firing order) — a real native block-time measurement is not possible in this
    offscreen sandbox, see the ticket's own note on that."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_mechanics_preview
    s = synth_settings(tmp_path)
    win = MainWindow(AppState(s)); pv = win.preview_screen
    data = stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv"))

    pv._overlays["channels"].start("Loading channels…")
    pv._overlays["raw"].start("Loading channels…")
    pv._render_preview_async(data)
    assert len(pv._channel_plots) == 5              # stage1 ran synchronously
    assert pv._breath_spans == {}                   # stage2 (overlays) has NOT run yet
    assert pv._overlays["channels"].is_busy()        # nobody has stopped it — still drawing

    qapp.processEvents()                             # fires stage2's singleShot(0)
    assert pv._breath_spans                          # stage2 has now run
    assert not pv._emg_raw_subplots                  # stage3 (raw stack) has NOT run yet
    assert pv._overlays["channels"].is_busy()
    assert pv._overlays["raw"].is_busy()

    qapp.processEvents()                             # fires stage3's singleShot(0)
    assert pv._emg_raw_subplots                      # stage3 has now run
    assert not pv._overlays["channels"].is_busy(), "the chain must stop its own overlay when done"
    assert not pv._overlays["raw"].is_busy()
    win.close()


def test_async_mech_render_abandons_a_render_superseded_mid_chain(qapp, tmp_path):
    """If the user steps to another file before a deferred stage of
    _render_preview_async fires, that stage must recognise (via the generation
    counter _reset_breath_state bumps on a file switch) that it no longer owns the
    panels and do nothing — not repopulate stale breath state for a file that is no
    longer on screen."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_mechanics_preview
    s = synth_settings(tmp_path)
    win = MainWindow(AppState(s)); pv = win.preview_screen
    data = stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv"))

    pv._render_preview_async(data)                  # stage1 runs; stage2 is queued
    pv._reset_breath_state()                        # simulate a file switch mid-chain
    assert pv._breath_spans == {}

    qapp.processEvents()                             # would-be stage2 for the OLD render
    qapp.processEvents()                             # would-be stage3 for the OLD render
    assert pv._breath_spans == {}, "a superseded render must not repopulate stale breath state"
    win.close()


def test_a_click_during_the_async_stage1_to_stage2_gap_cannot_toggle_a_stale_breath(qapp, tmp_path):
    """Found in self-review of this ticket: a SAME-FILE settings edit re-dispatches
    'mech' without going through a file switch, so _reset_breath_state (which used
    to be the only thing clearing _breath_spans) never runs. Without stage1 ALSO
    clearing _breath_spans/_breath_regions/_breath_texts itself, a click landing in
    the stage1->stage2 gap (the async path's one QTimer.singleShot(0, ...) turn)
    would hit-test against the PRE-edit breath spans — still non-empty, so
    _on_plot_clicked's guard would not catch it — and could silently toggle an
    exclusion for a breath number that no longer matches what stage2 is about to
    draw. This did not exist before D15: the old _render_preview was one
    synchronous call with no gap for a click to land in."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.workers import stage_mechanics_preview
    s = synth_settings(tmp_path)
    win = MainWindow(AppState(s)); pv = win.preview_screen
    data = stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv"))

    pv._render_preview_async(data)                  # first render, run to completion
    qapp.processEvents(); qapp.processEvents()
    assert pv._breath_spans, "the fixture must have detected breaths for this test to mean anything"

    # a same-file settings-edit re-dispatch: NO file switch, so _reset_breath_state
    # is deliberately NOT called here (that would defeat the point of this test)
    pv._render_preview_async(data)
    assert pv._breath_spans == {}, "stage1 must clear the stale click-map immediately, not wait for stage2"
    a_breath_no = data["spans"][0][0]
    assert pv._toggle_breath(a_breath_no) is None, "a click in the gap must be a no-op, never a stale toggle"
    excl = {b for e in s.processing.exclude_breaths
            if e.file == "synth_case_A.csv" for b in e.breaths}
    assert a_breath_no not in excl, "no exclusion may have been written from a stale-window click"


# --------------------------------------------------------------------------- #
# D24 (UI-overhaul) — the run lock reaches Preview & QC's own write actions.
# Before this ticket only Setup (+ the header Analysis menu) locked on run_started; a
# breath-exclusion click or "Process & write this file" during a batch looked like it
# worked while the running batch (already holding a frozen deepcopy of the settings taken
# at RunScreen._start) never saw it. Action-level, not a whole-surface lock — see B04's
# reversal of exactly that pattern, referenced in PreviewScreen.set_run_active's docstring.
# --------------------------------------------------------------------------- #
def test_run_lock_disables_process_button_without_disabling_the_whole_screen(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    assert pv.btn_process_file.isEnabled() is True        # a valid render enabled it

    pv.set_run_active(True)
    assert pv.btn_process_file.isEnabled() is False
    assert pv.isEnabled() is True, "the surface itself must stay reachable — B04"
    assert pv.file_rail.isEnabled() is True

    pv.set_run_active(False)
    assert pv.btn_process_file.isEnabled() is True, "a diagram is still loaded — restored, not left off"
    win.close()


def test_run_lock_survives_a_file_switch_mid_run(qapp, tmp_path):
    """A file switch mid-run is allowed (graphs/rail stay live) and re-renders the
    Mechanics stack via the same path that normally re-enables 'Process & write this
    file' for the newly loaded file. That render must not silently defeat the lock."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    pv.set_run_active(True)
    assert pv.btn_process_file.isEnabled() is False

    _render_mech(pv, s)                                    # simulates the live re-render a file switch triggers
    assert pv.btn_process_file.isEnabled() is False, "the run lock must survive a mid-run render"
    win.close()


def test_toggle_breath_is_locked_while_a_run_is_active_and_says_why(qapp, tmp_path):
    """Self-review finding: PreviewScreen.status is a HIDDEN label (screen.py), and while
    a run is active MainWindow._on_screen_status suppresses every screen's status_changed
    except run_screen's own — so a click rejection that only called _set_status would be
    genuinely invisible to a real user (the exact symptom this ticket exists to fix), even
    though the underlying exclude_breaths write was correctly blocked. Assert on the actual
    visible surface (the window status bar), not the internal label, so a regression back
    to "technically blocked, silently so" fails this test."""
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)
    a_breath = next(iter(pv._breath_spans))

    pv.set_run_active(True)
    assert pv._toggle_breath(a_breath) is None
    assert s.processing.exclude_breaths == [], "a locked click must never rewrite exclude_breaths"
    bar_msg = win.statusBar().currentMessage().lower()
    assert "locked" in bar_msg and "run" in bar_msg, "the rejection must reach the VISIBLE status bar"

    pv.set_run_active(False)
    pv._toggle_breath(a_breath)                            # ordinary toggle works again
    assert len(s.processing.exclude_breaths) == 1
    win.close()


# --------------------------------------------------------------------------- #
# Point 6 (respmech CI ticket 20260811-0910) — suite-scaling: PlotItem/ViewBox menus are
# now closed at PreviewScreen shutdown, via plot_perf.close_plots().
# --------------------------------------------------------------------------- #
def test_closing_the_window_closes_the_mechanics_stacks_current_plot_items(qapp, tmp_path):
    """Point 6 (respmech CI ticket 20260811-0910, suite scaling). ``PlotItem``/``ViewBox``
    build their own context menus EAGERLY at construction (one ``ctrlMenu`` + six
    submenus per ``PlotItem``, one ``ViewBoxMenu`` per ``ViewBox``) -- and the GUI suite
    never deletes a closed ``MainWindow`` (deleting one segfaults on Python 3.11, see
    ``conftest.py``), so those menus used to accumulate for the life of the test session:
    RESPMECH_NET_CENSUS measured 6,137 surviving QMenus after just 57 GUI-heavy tests.
    ``PreviewScreen.shutdown()`` (called from ``MainWindow.closeEvent``) now closes every
    ``PlotItem`` CURRENTLY held by ``pv.plots`` via pyqtgraph's OWN cleanup API --
    ``ViewBox.setMenuEnabled(False)`` + ``PlotItem.close()`` -- not a sweep of
    already-closed windows from outside (the earlier approach that segfaulted
    nondeterministically; see ``plot_perf.close_plots()``).

    This is a white-box test of the mechanism (does ``.close()`` actually run on the
    plots this container references RIGHT NOW), not an end-to-end census: a re-render
    mid-session (``self.plots.clear()`` followed by fresh ``addPlot()`` calls, which
    several code paths do routinely) discards the PREVIOUS render's ``PlotItem``s from
    ``pv.plots.ci.items`` without closing them first -- a second, larger, NOT YET fixed
    leak source the ticket documents separately. An end-to-end app-wide menu count would
    conflate the two and either flake on render count or silently stop testing the
    mechanism this ticket actually landed.
    """
    from respmech.ui.main_window import MainWindow

    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s)); pv = win.preview_screen
    _render_mech(pv, s)                                    # populates pv.plots with real PlotItems

    plot_items = list(pv.plots.ci.items.keys())
    assert plot_items, "the mechanics stack should have built at least one PlotItem"
    assert all(p.ctrlMenu is not None for p in plot_items), (
        "every freshly rendered PlotItem should still own its context menu before close()"
    )

    win.close()

    assert all(p.ctrlMenu is None for p in plot_items), (
        "PreviewScreen.shutdown() must call PlotItem.close() on every plot the mechanics "
        "stack currently holds, dropping its ctrlMenu -- a survivor here means the window "
        "closed without releasing menus it was still directly responsible for"
    )
    win.close()                                            # idempotent: a second close must not raise


def test_closing_the_window_closes_setups_channel_summary_plots(qapp, tmp_path):
    """Point 6 continued (ticket 20260811-0910): the dominant remaining leak source once
    the mechanics stack's own shutdown-time close landed. Setup's read-only channel
    summary (ChannelSummary -> ColumnStack) builds its own PlotWidgets -- one ctrlMenu per
    assigned channel -- OUTSIDE PreviewScreen entirely, and MainWindow.closeEvent used to
    call shutdown() only on preview_screen/run_screen. That ColumnStack is built once at
    MainWindow construction (Setup renders the initial channel mapping eagerly) and is
    never rebuilt unless the mapping changes, so it stays alive -- and reachable, not
    garbage -- for the window's whole life. Measured directly: a bare, freshly constructed
    MainWindow that renders nothing else still left 83 surviving QMenus after close()
    before this fix, 6 after -- the window's own File/Help menu-bar menus, unrelated to
    pyqtgraph (RESPMECH_NET_CENSUS; see the ticket for the full investigation and the two
    isolated-experiment measurements that misattributed and then correctly attributed this
    source).

    The PlotItem references are captured BEFORE win.close(), not re-fetched via
    ``PlotWidget.getPlotItem()`` afterwards -- see test_column_stack.py's
    ``test_close_plots_closes_every_embedded_plotitem_and_empties_the_list`` for why a
    closed PlotWidget's own ``getPlotItem()`` returns ``None``, not a readable PlotItem.
    """
    from respmech.ui.main_window import MainWindow

    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s))
    summary = win.settings_screen.channel_summary
    assert summary.stack is not None, (
        "synth_settings() assigns real channels, so Setup's initial refresh should have "
        "built a real ColumnStack -- if this assert ever fails, the test below is vacuous"
    )
    plot_items = [p.getPlotItem() for p in summary.stack.plots]
    assert plot_items and all(pi.ctrlMenu is not None for pi in plot_items)

    win.close()

    assert all(pi.ctrlMenu is None for pi in plot_items), (
        "MainWindow.closeEvent must also release Setup's channel summary plots, not just "
        "preview_screen/run_screen's -- a survivor here is exactly the leak source that "
        "PreviewScreen.shutdown() alone could never reach"
    )
    win.close()                                            # idempotent: a second close must not raise
