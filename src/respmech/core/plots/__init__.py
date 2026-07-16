"""Diagnostic figures + audio for a batch (feature P11, restored/upgraded by the
old-vs-new audit).

The core computes but never plots; this is the optional plotting *consumer* the
core's docstrings point at. It turns finished results into publication-style **PDF**
figures (vector, paginated) under ``<out>/diagnostics/``, driven by the
``output.diagnostics`` flags:

* ``save_pv_average``   → per-file Campbell (Poes–Volume) loop: every breath overlaid,
  the average breath drawn bold, with the elastic-recoil line + WOB polygon.
* ``save_pv_individual``→ a paginated grid of one Campbell loop per breath (recoil line,
  WOB polygon, shared axes, inverted x-axis, ignored breaths crossed out). A single
  cohort "all-files average" Campbell grid is written alongside it.
* ``save_raw``          → the FULL untrimmed signals (flow, volume, Poes, Pgas, Pdi) with
  breath boundaries + ignored-breath shading.
* ``save_trimmed``      → the trimmed, analysed signals (breaths concatenated).
* ``save_drift``        → the staged volume-correction figure (uncorrected → zeroed →
  drift-corrected → trend-adjusted), the trend-adjustment diagnostic (when trend
  correction is on), and the end-expiratory/end-inspiratory endpoint trend check.
* ``save_emg``          → per-channel EMG overviews at each conditioning stage (raw /
  ECG-removed / noise-reduced) with the flow reference, R-peak capture markers and
  breath boundaries; ``processing.emg.plot_yscale`` sets the y-range.

``processing.emg.save_sound`` additionally exports each EMG channel/stage as a WAV.

Rendering uses matplotlib's object API (``Figure`` + Agg canvas) rather than ``pyplot``
so it holds no global state — safe to call from a worker thread and it never disturbs
the GUI's Qt backend. Every figure is wrapped so a plotting failure degrades to a
skipped figure (reported in the returned list), never a failed run.
"""
from __future__ import annotations

import os

import numpy as np

_BRAND = "#2C6E9B"
_ACCENT = "#5CA9DD"
_MUTED = "#8894A0"
_CAPTURE = "#D33A3A"
_IGNORE = "#E45757"


def _canvas(figsize):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=figsize, dpi=140)
    FigureCanvasAgg(fig)
    return fig


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    return path


def _ordered(fr):
    """All breaths of a file result, in order (ignored included)."""
    return list(fr.breaths.values()) if fr.breaths else []


def _breaths(fr):
    """Non-ignored breaths of a file result, in order."""
    return [b for b in _ordered(fr) if not b.get("ignored")]


def _recoil_and_polygon(ax, eilv, eelv, alpha_line=1.0, alpha_fill=0.5):
    """Draw the elastic-recoil line (EILV↔EELV) and the shaded elastic-WOB triangle,
    exactly as the legacy Campbell diagrams did: Polygon(eelv, eilv, [eilv_x, eelv_y])."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
    try:
        lx, ly = zip(eilv, eelv)
    except (TypeError, ValueError):
        return
    ax.add_line(Line2D(lx, ly, linewidth=2, alpha=alpha_line, color=_BRAND))
    tri = [[eelv[0], eelv[1]], [eilv[0], eilv[1]], [eilv[0], eelv[1]]]
    ax.add_patch(Polygon(tri, alpha=alpha_fill, color="#999999", fill=True))


def _pv_limits(breaths, vkey, pkey):
    """Common, padded (×1.1) axis limits over the non-ignored breaths — so every panel
    in a grid shares one scale and breaths are visually comparable (legacy behaviour)."""
    maxx = maxy = -np.inf
    minx = miny = np.inf
    for b in breaths:
        if b.get("ignored"):
            continue
        v, p = np.asarray(b[vkey], float), np.asarray(b[pkey], float)
        if not v.size:
            continue
        maxx, maxy = max(maxx, v.max(), 0), max(maxy, p.max(), 0)
        minx, miny = min(minx, v.min(), 0), min(miny, p.min(), 0)
    if not np.isfinite([minx, maxx, miny, maxy]).all():
        return None
    return minx * 1.1, maxx * 1.1, miny * 1.1, maxy * 1.1


# --------------------------------------------------------------------------- #
# Campbell / PV loops
# --------------------------------------------------------------------------- #
def _pv_average(fr, fname, path):
    bs = _breaths(fr)
    if not bs:
        return None
    fig = _canvas((5.6, 5.8))
    ax = fig.add_subplot(111)
    for b in bs:
        ax.plot(b["volume"], b["poes"], color=_ACCENT, alpha=0.28, lw=0.8)
    b0 = bs[0]
    if b0.get("volumeavg") is not None and b0.get("poesavg") is not None \
            and len(b0["volumeavg"]) and len(b0["poesavg"]):
        ax.plot(b0["volumeavg"], b0["poesavg"], color=_BRAND, lw=2.4, label="average breath")
        if b0.get("eilvavg") is not None and b0.get("eelvavg") is not None:
            _recoil_and_polygon(ax, b0["eilvavg"], b0["eelvavg"])
        ax.legend(loc="best", frameon=False)
    ax.set_xlabel("Volume (L)")
    ax.set_ylabel("Oesophageal pressure (cmH₂O)")
    ax.set_title(f"{fname} — Campbell / PV loop ({len(bs)} breaths)")
    ax.grid(True, color=_MUTED, alpha=0.2)
    ax.invert_xaxis()
    return _save(fig, path)


def _pv_grid(breaths, title_prefix, path, cols, rows, vkey, pkey, ekey_i, ekey_e, titler):
    """Paginated Campbell grid (one loop per breath), shared axes, inverted x-axis,
    recoil line + WOB polygon, ignored breaths crossed out. Multi-page PDF."""
    from matplotlib.backends.backend_pdf import PdfPages
    ordered = list(breaths)
    if not ordered:
        return None
    lim = _pv_limits(ordered, vkey, pkey)
    if lim is None:
        return None
    minx, maxx, miny, maxy = lim
    cols = max(1, int(cols)); rows = max(1, int(rows))
    per_page = cols * rows
    npages = -(-len(ordered) // per_page)
    with PdfPages(path) as pdf:
        for pg in range(npages):
            page = ordered[pg * per_page:(pg + 1) * per_page]
            fig = _canvas((3.0 * cols, 2.9 * rows))
            for i, b in enumerate(page):
                ax = fig.add_subplot(rows, cols, i + 1)
                ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy); ax.invert_xaxis()
                ignored = b.get("ignored")
                ax.plot(b[vkey], b[pkey], "-k", lw=1.4, alpha=0.2 if ignored else 1.0)
                if ignored:
                    ax.plot([minx, maxx], [miny, maxy], "-r", lw=1)
                    ax.plot([minx, maxx], [maxy, miny], "-r", lw=1)
                elif b.get(ekey_i) is not None and b.get(ekey_e) is not None:
                    _recoil_and_polygon(ax, b[ekey_i], b[ekey_e])
                ax.set_title(titler(b), fontsize=9)
                ax.tick_params(labelsize=7)
                ax.grid(True, color=_MUTED, alpha=0.2)
            fig.suptitle(f"{title_prefix} (page {pg + 1} of {npages})", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
    return path


def _pv_individual(fr, fname, path, cols, rows):
    bs = _ordered(fr)
    if not bs:
        return None
    return _pv_grid(bs, f"{fname} — Campbell diagrams", path, cols, rows,
                    "volume", "poes", "eilv", "eelv",
                    lambda b: f"Breath #{b.get('number', '?')}")


def _pv_cohort(result, path, cols, rows):
    """One panel per file: that file's MEAN Campbell loop — the cross-subject overview
    the old 'All files – average Campbell.pdf' provided."""
    reps = []
    for fname, fr in result.ok_files.items():
        bs = _breaths(fr)
        if bs and bs[0].get("volumeavg") is not None and len(bs[0]["volumeavg"]):
            reps.append(bs[0])
    if not reps:
        return None
    return _pv_grid(reps, "All files — average Campbell", path, cols, rows,
                    "volumeavg", "poesavg", "eilvavg", "eelvavg",
                    lambda b: str(b.get("filename", "?")))


# --------------------------------------------------------------------------- #
# time-domain signal figures
# --------------------------------------------------------------------------- #
def _signals_trimmed(fr, fname, path):
    bs = _breaths(fr)
    if not bs:
        return None

    def cat(key):
        return np.concatenate([np.asarray(b[key], float) for b in bs])
    panels = [("Flow (L/s)", "flow"), ("Volume (L)", "volume"), ("Poes (cmH₂O)", "poes"),
              ("Pgas (cmH₂O)", "pgas"), ("Pdi (cmH₂O)", "pdi")]
    panels = [(lbl, k) for lbl, k in panels if k in bs[0]]
    # cumulative breath boundaries in the concatenated sample axis
    bounds = np.cumsum([0] + [len(np.asarray(b["flow"], float)) for b in bs])
    fig = _canvas((11.0, 1.8 * len(panels)))
    for i, (lbl, key) in enumerate(panels):
        ax = fig.add_subplot(len(panels), 1, i + 1)
        ax.plot(cat(key), color=_BRAND, lw=0.6)
        for x in bounds[1:-1]:
            ax.axvline(x, color="k", lw=0.3, ls="--", alpha=0.35)
        ax.set_ylabel(lbl, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, color=_MUTED, alpha=0.2)
        if i < len(panels) - 1:
            ax.set_xticklabels([])
    fig.axes[-1].set_xlabel("Sample (trimmed, breaths concatenated)")
    fig.suptitle(f"{fname} — analysed (trimmed) signals", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, path)


def _signals_raw(fr, fname, path):
    """The FULL, untrimmed recording (a genuine pre-trim QC view, distinct from the
    trimmed figure), with breath boundaries and ignored-breath shading in real time."""
    sig = fr.signals or {}
    t = sig.get("raw_time")
    if t is None or not len(t):
        return None
    panels = [("Flow (L/s)", "raw_flow"), ("Volume (L)", "raw_volume"),
              ("Poes (cmH₂O)", "raw_poes"), ("Pgas (cmH₂O)", "raw_pgas"), ("Pdi (cmH₂O)", "raw_pdi")]
    panels = [(lbl, k) for lbl, k in panels if sig.get(k) is not None and len(sig[k])]
    fig = _canvas((11.0, 1.8 * len(panels)))
    for i, (lbl, key) in enumerate(panels):
        ax = fig.add_subplot(len(panels), 1, i + 1)
        ax.plot(t, sig[key], color=_BRAND, lw=0.5)
        _mark_breaths(ax, _ordered(fr))
        ax.set_ylabel(lbl, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, color=_MUTED, alpha=0.2)
        if i < len(panels) - 1:
            ax.set_xticklabels([])
    fig.axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{fname} — raw (untrimmed) signals", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, path)


def _mark_breaths(ax, breaths):
    """Breath-start dashed lines + red shading over ignored breaths, keyed on the breath
    time (absolute seconds — matches the raw/staged time axes)."""
    yl = ax.get_ylim()
    for b in breaths:
        bt = np.asarray(b.get("time", []), float)
        if not bt.size:
            continue
        ax.axvline(bt[0], color="k", lw=0.4, ls="--", alpha=0.4)
        if b.get("ignored"):
            ax.axvspan(bt[0], bt[-1], color=_IGNORE, alpha=0.10)
    ax.set_ylim(yl)


# --------------------------------------------------------------------------- #
# volume-correction diagnostics
# --------------------------------------------------------------------------- #
def _volume_correction(fr, fname, path):
    """Staged volume-correction overview: flow, then the volume at each correction stage
    (uncorrected → zeroed → drift-corrected → trend-adjusted). Restores the old
    'Volume correction' figure so each stage can be verified."""
    sig = fr.signals or {}
    t = sig.get("time")
    if t is None or not len(t):
        return None
    rows = [("Flow (L/s)", sig.get("flow")),
            ("Uncorrected volume (L)", sig.get("vol_uncorrected")),
            ("Zeroed volume (L)", sig.get("vol_zeroed"))]
    if sig.get("drift_on"):        # off ⇒ vol_drift == vol_zeroed; don't imply a stage that didn't run
        rows.append(("Linear drift-corrected (L)", sig.get("vol_drift")))
    if sig.get("trend_on"):
        rows.append(("Trend-adjusted volume (L)", sig.get("vol_final")))
    rows = [(lbl, y) for lbl, y in rows if y is not None and len(y)]
    fig = _canvas((11.0, 1.7 * len(rows)))
    for i, (lbl, y) in enumerate(rows):
        ax = fig.add_subplot(len(rows), 1, i + 1)
        ax.plot(t, y, color=_BRAND, lw=0.7)
        ax.set_ylabel(lbl, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, color=_MUTED, alpha=0.2)
        if i < len(rows) - 1:
            ax.set_xticklabels([])
    fig.axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{fname} — volume correction stages", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, path)


def _trend(fr, fname, path, settings):
    """Trend-adjustment diagnostic: the drift-corrected volume with the detected
    end-expiratory peaks and the fitted trend envelope that gets subtracted."""
    sig = fr.signals or {}
    if not sig.get("trend_on"):
        return None
    vol = np.asarray(sig.get("vol_drift", []), float).squeeze()
    fs = sig.get("fs")
    if vol.size < 3 or not fs:
        return None
    from scipy import signal as _sig
    from scipy.interpolate import interp1d
    v = settings.processing.volume
    peaks = _sig.find_peaks((vol * -1) + vol.max(), height=v.trend_peak_min_height,
                            distance=max(1, int(v.trend_peak_min_distance_s * fs)))[0]
    if peaks.size < 2:
        return None
    f = interp1d(peaks, vol[peaks], v.trend_method, fill_value="extrapolate")
    envelope = f(np.linspace(0, vol.size - 1, vol.size))
    t = np.asarray(sig.get("time"), float)
    if t.size != vol.size:
        t = np.arange(vol.size) / fs
    fig = _canvas((11.0, 5.0))
    ax = fig.add_subplot(211)
    ax.plot(t, vol, color=_BRAND, lw=0.7, label="drift-corrected volume")
    ax.plot(t[peaks], vol[peaks], "o", color=_CAPTURE, ms=3, label="detected end-expiratory peaks")
    ax.plot(t, envelope, color=_ACCENT, lw=1.2, label=f"fitted trend ({v.trend_method})")
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.set_ylabel("Volume (L)", fontsize=9)
    ax.grid(True, color=_MUTED, alpha=0.2)
    ax2 = fig.add_subplot(212, sharex=ax)
    ax2.plot(t, vol - envelope, color=_BRAND, lw=0.7)
    ax2.set_ylabel("Trend-adjusted (L)", fontsize=9)
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, color=_MUTED, alpha=0.2)
    fig.suptitle(f"{fname} — volume trend adjustment", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, path)


def _drift(fr, fname, path):
    bs = _breaths(fr)
    if not bs:
        return None

    def _vol(b, key):
        v = b.get(key)
        return float(v[0]) if isinstance(v, (list, tuple, np.ndarray)) and len(v) else np.nan
    x = [b.get("number", i + 1) for i, b in enumerate(bs)]
    eelv = [_vol(b, "eelv") for b in bs]
    eilv = [_vol(b, "eilv") for b in bs]
    if not (np.any(np.isfinite(eelv)) or np.any(np.isfinite(eilv))):
        return None
    fig = _canvas((7.0, 4.0))
    ax = fig.add_subplot(111)
    ax.plot(x, eilv, "o-", color=_BRAND, label="end-inspiratory volume")
    ax.plot(x, eelv, "o-", color=_ACCENT, label="end-expiratory volume")
    ax.set_xlabel("Breath #")
    ax.set_ylabel("Lung volume (L)")
    ax.set_title(f"{fname} — volume endpoints across breaths (drift check)")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, color=_MUTED, alpha=0.2)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# EMG overviews + audio
# --------------------------------------------------------------------------- #
def _emg_overview(fr, fname, path, ylim, stage_key, stage_label):
    """One stacked panel per EMG channel: the conditioned EMG, the flow reference on a
    twin axis, R-peak capture markers (red ▼ + faint line), breath boundaries and
    ignored-breath shading. ``ylim`` = processing.emg.plot_yscale."""
    sig = fr.signals or {}
    stages = sig.get("emg_stages") or {}
    data = stages.get(stage_key)
    if data is None:
        return None
    data = np.asarray(data, float)
    if data.size == 0:
        return None
    if data.ndim == 1:
        data = data[:, None]
    t = np.asarray(sig.get("time"), float)
    flow = np.asarray(sig.get("flow"), float)
    peaks = np.asarray(sig.get("emg_peaks", []), float)
    cols = sig.get("emg_cols") or list(range(1, data.shape[1] + 1))
    nch = data.shape[1]
    use_ylim = bool(ylim and len(ylim) == 2 and ylim[0] < ylim[1])
    fig = _canvas((11.7, 2.1 * nch + 0.6))
    for i in range(nch):
        ax = fig.add_subplot(nch, 1, i + 1)
        ax.plot(t, data[:, i], color=_BRAND, lw=0.4, label=f"EMG col {cols[i] if i < len(cols) else i + 1}")
        if len(flow) == len(t):
            ax2 = ax.twinx()
            ax2.plot(t, flow, color=_ACCENT, lw=0.6, alpha=0.7)
            ax2.set_yticks([])
        if use_ylim:
            ax.set_ylim(ylim)
        if t.size:
            ax.set_xlim(t[0], t[-1])
        top = ax.get_ylim()[1]
        if peaks.size:
            ax.scatter(peaks, np.full(peaks.size, top), s=14, c=_CAPTURE, marker="v", zorder=5, clip_on=False)
            for pk in peaks:
                ax.axvline(pk, color=_CAPTURE, lw=0.3, alpha=0.3)
        _mark_breaths(ax, _ordered(fr))
        ax.set_ylabel(f"col {cols[i] if i < len(cols) else i + 1}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, color=_MUTED, alpha=0.2)
        ax.legend(loc="upper left", frameon=False, fontsize=7)
        if i < nch - 1:
            ax.set_xticklabels([])
    fig.axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{fname} — {stage_label}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return _save(fig, path)


_EMG_STAGES = [("raw", "Raw EMG"),
               ("ecg_removed", "EMG (ECG removed)"),
               ("noise_reduced", "EMG (ECG removed + noise reduced)")]


def _write_emg_audio(fr, fname, fs, figdir):
    """Export each EMG channel/stage as a normalised int16 WAV at the analysis rate."""
    from scipy.io.wavfile import write as _wavwrite
    sig = fr.signals or {}
    stages = sig.get("emg_stages") or {}
    cols = sig.get("emg_cols") or []
    rate = int(round(fs)) if fs else 2000
    written = []
    for key, label in _EMG_STAGES:
        data = stages.get(key)
        if data is None:
            continue
        data = np.asarray(data, float)
        if data.ndim == 1:
            data = data[:, None]
        for i in range(data.shape[1]):
            ch = data[:, i]
            peak = np.max(np.abs(ch))
            scaled = np.int16(ch / peak * 32767) if peak > 0 else np.zeros(ch.size, np.int16)
            col = cols[i] if i < len(cols) else i + 1
            wav = os.path.join(figdir, f"{fname} – EMG col {col} ({label}).wav")
            _wavwrite(wav, rate, scaled)
            written.append(wav)
    return written


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def write_figures(result, settings, outputfolder: str) -> tuple[list, list]:
    """Write every enabled diagnostic figure (and WAV export). Returns (written_paths,
    failures) where failures is a list of ``(name, error)`` — a bad figure is skipped,
    not fatal, so a run always finishes even if one plot cannot be drawn."""
    dg = settings.output.diagnostics
    emg = settings.processing.emg
    cols, rows = dg.pv_columns, dg.pv_rows
    ylim = list(getattr(emg, "plot_yscale", []) or [])

    # per-file jobs: (label, callable(fr, fname, path) -> path or None, filename-suffix)
    jobs = []
    if dg.save_pv_average:
        jobs.append(("PV average", _pv_average, "Campbell (average).pdf"))
    if dg.save_pv_individual:
        jobs.append(("PV individual", lambda fr, fn, p: _pv_individual(fr, fn, p, cols, rows),
                     "Campbell (breaths).pdf"))
    if dg.save_raw:
        jobs.append(("raw signals", _signals_raw, "signals (raw).pdf"))
    if dg.save_trimmed:
        jobs.append(("trimmed signals", _signals_trimmed, "signals (trimmed).pdf"))
    if dg.save_drift:
        jobs.append(("volume correction", _volume_correction, "volume correction.pdf"))
        jobs.append(("trend", lambda fr, fn, p: _trend(fr, fn, p, settings), "volume trend.pdf"))
        jobs.append(("drift", _drift, "volume endpoints.pdf"))

    figdir = os.path.join(outputfolder, "diagnostics")
    written, failures = [], []
    have = bool(jobs) or getattr(dg, "save_emg", False) or emg.save_sound
    if not have:
        return [], []
    os.makedirs(figdir, exist_ok=True)

    for fname, fr in result.ok_files.items():
        for label, fn, suffix in jobs:
            path = os.path.join(figdir, f"{fname} – {suffix}")
            try:
                p = fn(fr, fname, path)
                if p:
                    written.append(p)
            except Exception as e:              # noqa: BLE001 - a plot must never fail a run
                failures.append((f"{fname}/{label}", str(e)))
        # EMG overviews (one figure per conditioning stage present)
        if getattr(dg, "save_emg", False) and fr.signals and fr.signals.get("emg_stages"):
            for key, lbl in _EMG_STAGES:
                if fr.signals["emg_stages"].get(key) is None:
                    continue
                path = os.path.join(figdir, f"{fname} – {lbl}.pdf")
                try:
                    p = _emg_overview(fr, fname, path, ylim, key, lbl)
                    if p:
                        written.append(p)
                except Exception as e:          # noqa: BLE001
                    failures.append((f"{fname}/EMG {key}", str(e)))
        # EMG audio
        if emg.save_sound and fr.signals and fr.signals.get("emg_stages"):
            try:
                written.extend(_write_emg_audio(fr, fname, (fr.signals or {}).get("fs"), figdir))
            except Exception as e:              # noqa: BLE001
                failures.append((f"{fname}/EMG audio", str(e)))

    # one cohort "all-files average" Campbell across the whole batch
    if dg.save_pv_individual and len(result.ok_files) > 1:
        path = os.path.join(figdir, "All files – Campbell (average).pdf")
        try:
            p = _pv_cohort(result, path, cols, rows)
            if p:
                written.append(p)
        except Exception as e:                  # noqa: BLE001
            failures.append(("cohort/PV average", str(e)))
    return written, failures
