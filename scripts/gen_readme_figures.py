#!/usr/bin/env python
"""Regenerate every README graphic from the synthetic sample recording.

One source of truth: the app's own onboarding sample (``core.sample`` — realistic
mechanics with an open Campbell loop, band-limited diaphragm EMG, and a heartbeat/ECG
artefact) analysed through the full pipeline (``build_sample_settings`` — ECG removal +
spectral noise reduction). The four feature figures are drawn by the core diagnostic
plot writers, so they match the app's own output; the three UI screenshots are grabbed
from the offscreen app. No patient data.

    python scripts/gen_readme_figures.py        # writes docs/img/*.png

Deterministic (the sample uses a fixed seed). Run from the repo root.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # clean widget grabs, no black bars
os.environ.setdefault("RESPMECH_THEME", "light")        # docs are light-theme

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "docs", "img")
WIN_W, WIN_H = 1400, 880


def _build(work):
    from respmech.core.sample import write_sample_recording, build_sample_settings
    from respmech.core.pipeline import run_batch
    desc = write_sample_recording(os.path.join(work, "input"))
    s = build_sample_settings(desc, os.path.join(work, "output"))
    d = s.output.diagnostics
    d.save_pv_average = d.save_drift = d.save_emg = True
    result = run_batch(s)
    fr = result.files[desc["filename"]]
    if getattr(fr, "error", None):
        raise SystemExit(f"sample analysis failed: {fr.error}")
    return s, result, fr, desc


# --------------------------------------------------------------------------- #
# feature figures (matplotlib, via the core diagnostic writers + two customs)
# --------------------------------------------------------------------------- #
def _feature_figures(fr):
    from respmech.core import plots, plot_style
    sig = fr.signals or {}
    with plot_style.light_rc_context():
        plots._pv_average(fr, "sample_recording.csv", f"{OUT}/campbell.png")
        plots._volume_correction(fr, "sample_recording.csv", f"{OUT}/drift.png")
        _emg_stages(sig, f"{OUT}/emg-stages.png")
        _breath_exclusion(fr, sig, f"{OUT}/breath-exclusion.png", excluded=4)
    print("feature figures: campbell, drift, emg-stages, breath-exclusion")


def _emg_stages(sig, path):
    """One EMG channel across the conditioning steps (raw with detected R-peaks ▼ ->
    ECG removed -> noise reduced), zoomed so both the artefact and the cleanup are legible."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from respmech.core.plots import _BRAND, _CAPTURE, _MUTED
    from respmech.core.sample import DETECT_CHANNEL

    t = np.asarray(sig["time"], float)
    st = sig["emg_stages"]
    peaks = np.asarray(sig.get("emg_peaks", []), float)
    ch = DETECT_CHANNEL
    raw = np.asarray(st["raw"], float)[:, ch]
    t0 = t[0] + 3.0
    m = (t >= t0) & (t <= t0 + 7.0)
    pk = peaks[(peaks >= t0) & (peaks <= t0 + 7.0)]
    ymax = 1.05 * np.max(np.abs(raw[m]))

    fig = Figure(figsize=(9.2, 5.2), dpi=140)
    FigureCanvasAgg(fig)
    rows = [("1 · Raw EMG — heartbeat (ECG) artefact, detected R-peaks ▼", "raw", True),
            ("2 · ECG removed", "ecg_removed", False),
            ("3 · ECG removed + spectral noise reduced", "noise_reduced", False)]
    for i, (label, key, mark) in enumerate(rows):
        y = np.asarray(st[key], float)[:, ch]
        ax = fig.add_subplot(3, 1, i + 1)
        ax.plot(t[m], y[m], color=_BRAND, lw=0.7)
        ax.set_ylim(-ymax, ymax); ax.set_xlim(t0, t0 + 7.0)
        ax.set_ylabel("EMG (a.u.)", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(True, color=_MUTED, alpha=0.2)
        ax.text(0.010, 0.94, label, transform=ax.transAxes, fontsize=9.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=_MUTED, alpha=0.85))
        if mark and pk.size:
            ax.scatter(pk, np.full(pk.size, ymax * 0.92), s=24, c=_CAPTURE, marker="v",
                       zorder=6, clip_on=False)
        if i < 2:
            ax.set_xticklabels([])
    fig.axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Diaphragm EMG conditioning — one channel, step by step", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, bbox_inches="tight", facecolor="white")


def _breath_exclusion(fr, sig, path, excluded=4):
    """Flow + volume with the segmented breaths numbered and one shaded red — the
    click-to-exclude QC feature."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from respmech.core.plots import _BRAND, _MUTED, _IGNORE

    t = np.asarray(sig["time"], float)
    flow = np.asarray(sig["flow"], float)
    vol = np.asarray(sig.get("vol_final") if sig.get("vol_final") is not None
                     else sig.get("vol_drift"), float).squeeze()
    spans = [(int(b["number"]), float(np.asarray(b["time"]).flat[0]),
              float(np.asarray(b["time"]).flat[-1])) for b in fr.breaths.values()]

    fig = Figure(figsize=(11.0, 3.6), dpi=140)
    FigureCanvasAgg(fig)
    for r, (y, lbl) in enumerate([(flow, "Flow (L/s)"), (vol, "Volume (L)")]):
        ax = fig.add_subplot(2, 1, r + 1)
        ax.plot(t, y, color=_BRAND, lw=0.7)
        ax.set_xlim(t[0], t[-1]); ax.set_ylabel(lbl, fontsize=9)
        ax.tick_params(labelsize=7); ax.grid(True, color=_MUTED, alpha=0.2)
        ytop = ax.get_ylim()[1]
        for num, a, b in spans:
            ax.axvline(a, color="k", lw=0.4, ls="--", alpha=0.35)
            if num == excluded:
                ax.axvspan(a, b, color=_IGNORE, alpha=0.16)
            if r == 0:
                red = (num == excluded)
                ax.text((a + b) / 2, ytop, f"#{num}", ha="center", va="bottom", fontsize=8.5,
                        color=("#c0392b" if red else _MUTED), fontweight="bold" if red else "normal")
        if r == 0:
            ax.set_xticklabels([])
    fig.axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Breath segmentation & exclusion — click a breath to drop it "
                 f"(#{excluded} excluded, shaded red)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, bbox_inches="tight", facecolor="white")


# --------------------------------------------------------------------------- #
# UI screenshots (offscreen Qt)
# --------------------------------------------------------------------------- #
def _screenshots(settings, result, filename):
    from PySide6.QtWidgets import QApplication, QScrollArea
    from respmech.ui import theme
    from respmech.ui.state import AppState
    from respmech.ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    theme.apply_theme(app)
    win = MainWindow(AppState(settings))
    win.resize(WIN_W, WIN_H); win.show()
    pump = lambda n=6: [app.processEvents() for _ in range(n)]
    pump()

    # Setup
    win.tabs.setCurrentIndex(0)
    win.settings_screen.from_state(); pump()
    win.settings_screen.findChild(QScrollArea).verticalScrollBar().setValue(0); pump()
    win.grab().save(f"{OUT}/setup.png")

    # Preview & QC (Mechanics) with the Campbell + per-breath table populated
    win.tabs.setCurrentIndex(1)
    pv = win.preview_screen
    pv.refresh_files(); pv.file_combo.setCurrentText(filename)
    pv._preview(); pump()
    pv._on_batch_result(result)
    for i in range(pv.subtabs.count()):
        if pv.subtabs.tabText(i).lower().startswith("mechanics"):
            pv.subtabs.setCurrentIndex(i); break
    pump(8)
    win.grab().save(f"{OUT}/preview-mechanics.png")

    # Run & results, populated from the batch
    win.tabs.setCurrentIndex(2)
    rs = win.run_screen
    rs.refresh_actions()
    rs.log.appendPlainText("Batch run — output written to the chosen folder.\n")
    rs.log.appendPlainText(f"  processed  {filename}")
    rs._on_finished(result)
    rs._set_status("Batch complete — results written to the output folder.")
    pump(8)
    win.grab().save(f"{OUT}/run.png")
    print("screenshots: setup, preview-mechanics, run")


def main():
    sys.path.insert(0, os.path.join(REPO, "src"))
    os.makedirs(OUT, exist_ok=True)
    # a unique temp dir we own — never a fixed home path, so a re-run can never delete a
    # user's data. (The onboarding writes its sample to a temp dir too, so the Setup
    # screenshot's path is faithful to what a first-time user actually sees.)
    work = tempfile.mkdtemp(prefix="respmech-readme-")
    try:
        settings, result, fr, desc = _build(work)
        print(f"sample: {len(fr.breaths or {})} breaths, "
              f"{np.asarray((fr.signals or {}).get('emg_peaks', [])).size} R-peaks")
        _feature_figures(fr)
        _screenshots(settings, result, desc["filename"])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("done — 7 graphics written to docs/img/")
    sys.stdout.flush()
    os._exit(0)          # Qt worker threads can otherwise keep the process alive at exit


if __name__ == "__main__":
    main()
