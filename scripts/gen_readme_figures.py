#!/usr/bin/env python
"""Regenerate every README graphic from the synthetic sample recording.

One source of truth: the app's own onboarding sample (``core.sample`` — realistic
mechanics with an open Campbell loop, band-limited diaphragm EMG, and a heartbeat/ECG
artefact) analysed through the full pipeline (``build_sample_settings`` — ECG removal +
spectral noise reduction). The four feature figures are drawn by the core diagnostic
plot writers, so they match the app's own output; the UI screenshots (Setup, Preview & QC,
Run & results, the signal-set picker, a breath typed as a manoeuvre and the EMG-only
segments tab) are grabbed from the offscreen app. No patient data.

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
    ecgr = np.asarray(st["ecg_removed"], float)[:, ch]
    noi = np.asarray(st["noise_reduced"], float)[:, ch]
    t0 = t[0] + 3.0
    m = (t >= t0) & (t <= t0 + 7.0)
    pk = peaks[(peaks >= t0) & (peaks <= t0 + 7.0)]
    # one shared y-scale at the EMG level so the three rows are directly comparable; the
    # R-waves are several times bigger and simply run off the top of the raw row (their
    # positions are marked ▼) — they don't need their full height to make the point
    ymax = 1.05 * max(np.max(np.abs(ecgr[m])), np.max(np.abs(noi[m])))

    def rms(y):                                  # the 50 ms RMS envelope the analysis measures
        w = int(0.05 * 1000); k = np.ones(w) / w
        return np.sqrt(np.convolve(y ** 2, k, mode="same"))

    _RMS = "#E08A2E"                             # amber — distinct from the blue signal and red ▼
    fig = Figure(figsize=(9.2, 5.6), dpi=140)
    FigureCanvasAgg(fig)
    rows = [("1 · Raw EMG — heartbeat (ECG) R-waves run off-scale (detected R-peaks ▼)", raw, True),
            ("2 · ECG removed", ecgr, False),
            ("3 · ECG removed + spectral noise reduced", noi, False)]
    for i, (label, y, mark) in enumerate(rows):
        ax = fig.add_subplot(3, 1, i + 1)
        ax.plot(t[m], y[m], color=_BRAND, lw=0.5, alpha=0.45,
                label="signal" if i == 0 else None)
        r = rms(y)[m]
        ax.plot(t[m], r, color=_RMS, lw=1.5, label="RMS envelope" if i == 0 else None)
        ax.plot(t[m], -r, color=_RMS, lw=1.5)
        ax.set_ylim(-ymax, ymax); ax.set_xlim(t0, t0 + 7.0)
        ax.set_ylabel("EMG (a.u.)", fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(True, color=_MUTED, alpha=0.2)
        ax.text(0.010, 0.95, label, transform=ax.transAxes, fontsize=9.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=_MUTED, alpha=0.9))
        if mark and pk.size:
            ax.scatter(pk, np.full(pk.size, ymax * 0.90), s=24, c=_CAPTURE, marker="v",
                       zorder=6, clip_on=False)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7.5, frameon=True, framealpha=0.9, ncol=2)
        if i < 2:
            ax.set_xticklabels([])
    fig.axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Diaphragm EMG conditioning — signal and its RMS envelope, step by step",
                 fontsize=12)
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
    """Grab the three UI screenshots against the CURRENT app shape.

    UI-overhaul rewrote the surface this function drives (tickets B01-B03): the file combo
    that used to sit on Preview & QC is gone (replaced by ``FileRail``, which silently
    adopts the sample's one file as the selection on its first ``set_manifest()``, so no
    explicit selection call is needed here any more — see ``FileRail.set_manifest``'s own
    docstring), and Run & results is no longer a third tab: it is a drawer embedded in
    Preview & QC, expanded automatically when a run starts (``RunScreen._set_running``).
    Driving a real dry run (rather than hand-feeding ``result`` into ``_on_finished``) means
    this stays correct through any future change to what a run does on completion, one
    fewer place to keep in sync with ``run_screen.py`` by hand — and mirrors
    ``tools/capture_screens.py``'s proven approach (ticket Z01), which captures the same
    drawer this way.
    """
    import time

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

    # Preview & QC (Mechanics), with the Campbell + per-breath table populated.
    win.tabs.setCurrentIndex(1)
    pv = win.preview_screen
    pv.refresh_files()
    pv._preview(); pump()
    pv._on_batch_result(result)
    for i in range(pv.subtabs.count()):
        if pv.subtabs.tabText(i).lower().startswith("mechanics"):
            pv.subtabs.setCurrentIndex(i); break
    pump(8)
    win.grab().save(f"{OUT}/preview-mechanics.png")

    # Run & results: the drawer folded under Preview & QC (ticket B03), captured via a
    # real dry run against the same sample so the drawer's auto-expand fires and the run
    # log/results carry genuine content rather than a hand-assembled stand-in.
    rs = win.run_screen
    finished = []
    rs.run_finished.connect(lambda: finished.append(1))
    rs._start(write=False)
    end = time.monotonic() + 60
    while not finished and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.05)
    pump(8)
    win.grab().save(f"{OUT}/run.png")
    print(f"screenshots: setup, preview-mechanics, run (dry run finished: {bool(finished)})")
    win.close(); pump()
    return app, pump


def _settle(app, pv, timeout=120):
    """Pump until Preview & QC has no job in flight (the reactive jobs have delivered)."""
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents(); time.sleep(0.05)
        busy = [n for n, ov in pv._overlays.items() if ov.isVisible() and getattr(ov, "busy", False)]
        if not busy and not pv._jobs and not pv._launch_queue:
            t = time.monotonic() + 0.5           # one more beat for paints scheduled by the delivery
            while time.monotonic() < t:
                app.processEvents(); time.sleep(0.02)
            if not pv._jobs and not pv._launch_queue:
                return True
    return False


def _modular_screenshots(app, pump):
    """The three screens the modular analysis added: the signal-set picker a new analysis
    starts with, a breath typed as an IC manoeuvre (Preview & QC, with its Manoeuvres table),
    and the EMG-only analysis's own first tab, 'EMG – segments'. Each is driven through the
    same offscreen ``MainWindow`` as the screenshots above, from a fresh window so nothing
    carries over (an empty recent-analyses list and no local folder name reach an image)."""
    from respmech.core.settings import Settings
    from respmech.ui.state import AppState
    from respmech.ui.main_window import MainWindow
    from respmech.ui.signal_set_dialog import SignalSetDialog

    # 1 · the signal-set picker
    dlg = SignalSetDialog()
    dlg.show(); pump(8)
    dlg.grab().save(f"{OUT}/signal-set.png")
    dlg.close(); pump()

    # 2 · a breath typed as an IC manoeuvre, on the full sample (the same door a first-time
    # user takes: 'Explore with sample data')
    win = MainWindow(AppState())
    win.resize(WIN_W, 1080); win.show(); pump()     # taller, so the Manoeuvres table under the per-breath table shows
    if not win.settings_screen.open_sample_analysis():
        raise SystemExit("the sample analysis did not open")
    win.tabs.setCurrentIndex(1)
    pv = win.preview_screen
    pv.refresh_files(); pv._preview()
    if not _settle(app, pv):
        raise SystemExit("the typed-breath preview did not settle")
    for i in range(pv.subtabs.count()):
        if pv.subtabs.tabText(i).lower().startswith("mechanics"):
            pv.subtabs.setCurrentIndex(i); break
    if pv._set_breath_type(6, "ic") is None:
        raise SystemExit("could not type breath 6 as an IC manoeuvre")
    if not _settle(app, pv):
        raise SystemExit("the typed-breath re-run did not settle")
    pump(8)
    win.grab().save(f"{OUT}/breath-types.png")
    win.settings_screen._mark_clean()        # typing a breath dirtied it; close() would ask to save
    win.close(); pump()

    # 3 · the EMG-only analysis: no flow channel, segments placed by separators
    st = Settings(); st.analysis.signals = ["emg"]
    win = MainWindow(AppState(st))
    win.resize(WIN_W, WIN_H); win.show(); pump()
    if not win.settings_screen.open_sample_analysis(use_current_signals=True):
        raise SystemExit("the EMG-only sample did not open")
    win.tabs.setCurrentIndex(1)
    pv = win.preview_screen
    pv.refresh_files(); pv.subtabs.setCurrentIndex(0); pv._preview()
    if not _settle(app, pv):
        raise SystemExit("the EMG-only preview did not settle")
    pump(8)
    win.grab().save(f"{OUT}/emg-only.png")
    win.settings_screen._mark_clean()
    win.close(); pump()
    print("screenshots: signal-set, breath-types, emg-only")


def main():
    sys.path.insert(0, os.path.join(REPO, "src"))
    os.makedirs(OUT, exist_ok=True)
    # a unique temp dir we own — never a fixed home path, so a re-run can never delete a
    # user's data. (The onboarding writes its sample to a temp dir too, so the Setup
    # screenshot's path is faithful to what a first-time user actually sees.)
    work = tempfile.mkdtemp(prefix="respmech-readme-")
    code = 1
    try:
        settings, result, fr, desc = _build(work)
        print(f"sample: {len(fr.breaths or {})} breaths, "
              f"{np.asarray((fr.signals or {}).get('emg_peaks', [])).size} R-peaks")
        _feature_figures(fr)
        app, pump = _screenshots(settings, result, desc["filename"])
        _modular_screenshots(app, pump)
        print("done — 10 graphics written to docs/img/")
        code = 0
    finally:
        shutil.rmtree(work, ignore_errors=True)
        sys.stdout.flush()
        os._exit(code)   # Qt worker threads can otherwise keep the process alive at exit, also after an error


if __name__ == "__main__":
    main()
