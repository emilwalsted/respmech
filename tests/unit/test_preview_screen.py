"""Preview screen as an analysis & QC surface: the Campbell diagram's recoil line +
shaded work + WOB (P12), the EMG RMS envelope (P13), the batch-QC overview (P16), the
crosshair + Campbell export (P17) and zero-reference baselines (P22). Previously
test_review_wave4.py; the preview tests from waves 1 (P2) and 6 (P27) are appended below."""
import os

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPointF

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
