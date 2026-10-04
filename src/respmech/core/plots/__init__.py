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
* ``save_flow_volume``  → the file's tidal flow-volume loops placed inside its own maximal
  flow-volume loop (MFVL), when a breath is typed as a forced vital capacity.
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

from respmech.core import plot_style
from respmech.core.analysis import mfvl as mfvllib
from respmech.core.analysis import normal_range
from respmech.core.analysis.pressure import mean_peepi_rectangle_height, peepi_rectangle_height  # noqa: F401 (re-exported for the Preview panel)
from respmech.core.analysis.signals import Capabilities

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


def draw_peepi_rectangle(ax, eilv, eelv, height, *, color="#D9822B", alpha=0.35, zorder=None):
    """Hatched PEEPi rectangle of the modified Campbell diagram: it spans the tidal volume
    (EELV to EILV) and rises ``height`` cmH2O above the end-expiratory Poes, so its area is the
    threshold work the elastic-recoil polygon does not hold. Draw it BEFORE the polygon.
    A no-op without a positive finite height or a usable EELV/EILV pair, which is what keeps a
    figure without the feature identical to what it always was."""
    from matplotlib.patches import Rectangle
    if height is None or not np.isfinite(height) or height <= 0:
        return
    try:
        x0, y0 = float(eelv[0]), float(eelv[1])
        x1 = float(eilv[0])
    except (TypeError, ValueError, IndexError):
        return
    if not np.isfinite([x0, y0, x1]).all():
        return
    kw = {} if zorder is None else {"zorder": zorder}
    ax.add_patch(Rectangle((min(x0, x1), y0), abs(x1 - x0), float(height), facecolor=color,
                           edgecolor=color, alpha=alpha, hatch="////", fill=True, lw=0.6,
                           label="PEEPi", **kw))


def _recoil_and_polygon(ax, eilv, eelv, alpha_line=1.0, alpha_fill=0.5, peepi_height=None):
    """Draw the elastic-recoil line (EILV↔EELV) and the shaded elastic-WOB triangle,
    exactly as the legacy Campbell diagrams did: Polygon(eelv, eilv, [eilv_x, eelv_y])."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
    try:
        lx, ly = zip(eilv, eelv)
    except (TypeError, ValueError):
        return
    draw_peepi_rectangle(ax, eilv, eelv, peepi_height)
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
    if not bs or not len(bs[0].get("poes", [])):
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
            _recoil_and_polygon(ax, b0["eilvavg"], b0["eelvavg"],
                                peepi_height=mean_peepi_rectangle_height(bs))
        ax.legend(loc="best", frameon=False)
    ax.set_xlabel("Volume (L)")
    ax.set_ylabel("Oesophageal pressure (cmH₂O)")
    ax.set_title(f"{fname} — Campbell / PV loop ({len(bs)} breaths)")
    ax.grid(True, color=_MUTED, alpha=0.2)
    ax.invert_xaxis()
    return _save(fig, path)


def _pv_grid(breaths, title_prefix, path, cols, rows, vkey, pkey, ekey_i, ekey_e, titler,
             peepi_height=peepi_rectangle_height):
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
                    _recoil_and_polygon(ax, b[ekey_i], b[ekey_e], peepi_height=peepi_height(b))
                ax.set_title(titler(b), fontsize=9)
                ax.tick_params(labelsize=7)
                ax.grid(True, color=_MUTED, alpha=0.2)
            fig.suptitle(f"{title_prefix} (page {pg + 1} of {npages})", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
    return path


def _pv_individual(fr, fname, path, cols, rows):
    bs = _ordered(fr)
    if not bs or not len(bs[0].get("poes", [])):
        return None
    return _pv_grid(bs, f"{fname} — Campbell diagrams", path, cols, rows,
                    "volume", "poes", "eilv", "eelv",
                    lambda b: f"#{b.get('number', '?')}")


def _pv_cohort(result, path, cols, rows):
    """One panel per file: that file's MEAN Campbell loop — the cross-subject overview
    the old 'All files – average Campbell.pdf' provided."""
    reps = []
    heights = {}
    for fname, fr in result.ok_files.items():
        bs = _breaths(fr)
        if (bs and bs[0].get("volumeavg") is not None and len(bs[0]["volumeavg"])
                and len(bs[0].get("poes", []))):
            reps.append(bs[0])
            heights[id(bs[0])] = mean_peepi_rectangle_height(bs)
    if not reps:
        return None
    return _pv_grid(reps, "All files — average Campbell", path, cols, rows,
                    "volumeavg", "poesavg", "eilvavg", "eelvavg",
                    lambda b: str(b.get("filename", "?")),
                    peepi_height=lambda b: heights.get(id(b)))


# --------------------------------------------------------------------------- #
# Flow-volume: tidal loops inside the MFVL
# --------------------------------------------------------------------------- #
def draw_normal_band(ax, band, *, color=_MUTED, label=_MUTED):
    """Draw a ``core.analysis.normal_range.NormalBand`` behind a flow-volume figure: a
    shaded band between the curves at the lower and upper limits of normal, a dashed
    expected curve and a discreet reference note in the top-left corner. ``None`` draws
    nothing. Same axis as the MFVL itself (volume below TLC, expiratory flow upward)."""
    if band is None:
        return
    ax.fill_between(band.v_band, band.flow_lower, band.flow_upper, color=color, alpha=0.16,
                    lw=0, zorder=0, label=f"normal range ({band.reference_label})")
    ax.plot(band.v_expected, band.flow_expected, color=color, lw=1.0, ls="--", alpha=0.7,
            zorder=0)
    ax.text(0.01, 0.99, band.note(), transform=ax.transAxes, ha="left", va="top",
            fontsize=6, color=label, alpha=0.8)


def draw_flow_volume_mfvl(ax, placed, *, loop=_MUTED, mean=_BRAND, envelope="black",
                          marker=_MUTED, label=_MUTED, normal=None):
    """Draw ``mfvl.placed_tidal_loops``'s result onto ``ax``: grey tidal loops, a bold
    mean loop, the MFVL envelope (expiratory curve plus the FVC manoeuvre's own
    inspiratory limb) and dotted EELV/EILV markers. Volume runs from TLC on the
    left. The colours are parameters so the Preview panel can draw the same picture in its
    own theme; the defaults are the light-theme colours the PDF uses. ``normal`` is an
    optional ``normal_range.NormalBand`` drawn behind everything."""
    draw_normal_band(ax, normal, color=marker, label=label)
    for x, flow in placed["loops"]:
        ax.plot(x, flow, color=loop, alpha=0.35, lw=0.8, zorder=1)
    if placed["mean"] is not None:
        ax.plot(placed["mean"][0], placed["mean"][1], color=mean, lw=2.4, zorder=3,
                label="average tidal breath")
    ax.plot(placed["mefv_v"], placed["mefv_flow"], color=envelope, lw=1.8, zorder=2,
            label="MFVL")
    if placed.get("insp_v") is not None:
        # the inhalation to TLC that precedes the forced expiration: same colour and no
        # legend entry of its own, since together the two limbs are ONE manoeuvre's loop
        ax.plot(placed["insp_v"], placed["insp_flow"], color=envelope, lw=1.8, zorder=2)
    if placed["eelv"] is not None:
        ax.axvline(placed["eelv"], color=marker, ls=":", lw=1.0, zorder=0)
        ax.text(placed["eelv"], 0.02, " EELV", transform=ax.get_xaxis_transform(),
                va="bottom", ha="left", fontsize=8, color=label)
    if placed["eilv"] is not None:
        ax.axvline(placed["eilv"], color=marker, ls=":", lw=1.0, zorder=0)
        ax.text(placed["eilv"], 0.02, "EILV ", transform=ax.get_xaxis_transform(),
                va="bottom", ha="right", fontsize=8, color=label)
    ax.axhline(0, color=marker, lw=0.8, zorder=0)
    note = None
    if placed["ic_op"] is None:
        note = "No inspiratory-capacity reference: tidal loops cannot be placed"
    elif placed.get("in_domain_pct") is not None and placed["in_domain_pct"] < 99.5:
        note = (f"{100.0 - placed['in_domain_pct']:.0f}% of the tidal loop lies outside the "
                "MFVL's volume range")
    if note:
        ax.text(0.5, 0.04, note, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7, color=label)


def _flow_volume_mfvl(fr, fname, path, settings):
    """Tidal flow-volume loops placed inside the file's own maximal flow-volume loop
    (MFVL). ``None`` for a file without a resolved FVC reference; with an FVC but no IC
    reference only the envelope is drawn (the loops cannot be anchored to the TLC axis),
    and loops that fall outside the envelope's volume range are flagged, both by a note
    on the figure (see ``draw_flow_volume_mfvl``)."""
    placed = mfvllib.placed_tidal_loops(
        fr.breaths, getattr(fr, "manoeuvres", None), settings.processing.mfvl,
        settings.processing.lung_volume.ic)
    if placed is None:
        return None
    fig = _canvas((6.4, 5.4))
    ax = fig.add_subplot(111)
    draw_flow_volume_mfvl(ax, placed,
                          normal=normal_range.normal_band_for(settings, fname))
    ax.set_xlabel("Volume below TLC (L)")
    ax.set_ylabel("Flow (L/s)")
    ax.grid(True, color=_MUTED, alpha=0.2)
    ax.set_title(f"{fname} — tidal breathing in the MFVL")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    return _save(fig, path)


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
    # A channel absent from the declared signal set is still a KEY (compute.py leaves it as
    # an empty array, never removes it) — `k in bs[0]` alone no longer tells present from
    # absent, so this checks LENGTH instead.
    panels = [(lbl, k) for lbl, k in panels if len(bs[0].get(k, []))]
    # cumulative breath boundaries in the concatenated sample axis, taken from `time` (always
    # non-empty for a phased breath) rather than `flow` (not guaranteed once a future signal
    # set can omit it, e.g. an EMG-only whole-file segment)
    bounds = np.cumsum([0] + [len(np.asarray(b["time"], float)) for b in bs])
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
    from scipy.interpolate import interp1d
    from respmech.core.compute import trend_anchors, _TREND_MIN_ANCHORS
    v = settings.processing.volume
    # ONE detector shared with compute.correcttrend — this used to re-implement it and
    # could therefore draw anchors that were never subtracted.
    peaks = trend_anchors(vol, fs, min_height=v.trend_peak_min_height,
                          min_prominence_frac=v.trend_peak_min_prominence_frac,
                          min_distance_s=v.trend_peak_min_distance_s)
    if peaks.size < _TREND_MIN_ANCHORS.get(v.trend_method, 2):
        return None
    f = interp1d(peaks, vol[peaks], v.trend_method, fill_value="extrapolate")
    envelope = f(np.linspace(0, vol.size - 1, vol.size))
    t = np.asarray(sig.get("time"), float)
    if t.size != vol.size:
        t = np.arange(vol.size) / fs
    fig = _canvas((11.0, 5.0))
    ax = fig.add_subplot(211)
    ax.plot(t, vol, color=_BRAND, lw=0.7, label="drift-corrected volume")
    ax.plot(t[peaks], vol[peaks], "o", color=_CAPTURE, ms=3,
            label=f"detected end-expiratory troughs (n={peaks.size})")
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
    ax.set_xlabel("Breath number")
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
    # Forward-compat guard: today `sig["flow"]` is always an array (core/pipeline.py sets it
    # unconditionally), but if a future signal set without flow ever leaves it None, guard here
    # before np.asarray(None) turns it into an unsized 0-d array and `len(flow)` below raises.
    flow_raw = sig.get("flow")
    flow = np.asarray(flow_raw, float) if flow_raw is not None else np.asarray([])
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
def write_figures(result, settings, outputfolder: str, progress=None,
                  cohort_outputs: bool = True) -> tuple[list, list]:
    """Write every enabled diagnostic figure (and WAV export). Returns (written_paths,
    failures) where failures is a list of ``(name, error)`` — a bad figure is skipped,
    not fatal, so a run always finishes even if one plot cannot be drawn.

    Figures are written in the light style whatever theme the GUI is wearing: the GUI's
    dark theme installs its colours into global matplotlib rcParams, which the figures
    below would otherwise inherit at Figure()/add_subplot()/savefig() time.

    ``progress`` is an optional ``callable(fname)`` fired before each file's figures — the
    slowest part of a batch — so the GUI can show which file is being drawn.

    ``cohort_outputs`` (default True) gates the one figure built across the WHOLE batch
    rather than per file — the "All files – Campbell (average)" cohort figure. A subset/
    re-run (ticket A05) passes False so that figure, which the writer already treats as
    cohort-level, is never silently rebuilt from a fraction of the study."""
    with plot_style.light_rc_context():
        return _write_figures_impl(result, settings, outputfolder, progress=progress,
                                   cohort_outputs=cohort_outputs)


def per_file_figure_jobs(settings):
    """The per-file diagnostic-figure jobs ``output.diagnostics`` currently enables:
    ``[(label, callable(fr, fname, path) -> path or None, filename-suffix), ...]``.

    This is the single list ``_write_figures_impl`` (below) draws on to actually write
    figures, and the plan the Run screen shows before a run (``core.io.plan.plan_outputs``)
    draws on to describe what a run WOULD write — one place that knows which jobs exist, so
    a figure type can never appear in one without the other (ticket A06). A job's callable
    can still return ``None`` for a given file (e.g. "trend" when trend correction has no
    detectable anchors) — the JOB existing is settings-driven and static; whether it
    actually produces a file for a particular recording is data-driven and is not decided
    here. Callers that need a ceiling, not a promise, must treat this list's length as an
    upper bound per file, never an exact count.

    Channel-aware since this ticket: the Campbell jobs need Poes (``Capabilities.poes``) and
    the volume-correction/trend/drift jobs need a volume trace (``Capabilities.volume``) —
    a signal set without one of those never gets the job added at all, so it never shows up
    in ``core.io.plan.plan_outputs``' ceiling either (the two read this exact list).

    Uses ``from_settings_or_none``: this is the exact function backing Setup's 'You will
    get' preview (``diagnostic_figure_type_count``), which — per ``SettingsScreen.
    _update_save_preview``'s own docstring — resolves one event-loop turn into
    ``MainWindow``'s real startup, still before ``Settings.validate()`` ever runs. A
    malformed, hand-edited ``analysis.signals`` must degrade the same way the UI's other
    render paths do (no capability-gated jobs added) rather than throw out of a deferred
    Qt callback."""
    dg = settings.output.diagnostics
    cols, rows = dg.pv_columns, dg.pv_rows
    caps = Capabilities.from_settings_or_none(settings)
    poes = caps is not None and caps.poes
    volume = caps is not None and caps.volume
    # _signals_raw/_signals_trimmed draw ONLY flow/volume/poes/pgas/pdi panels (never EMG);
    # an EMG-only signal set (mode "emg_only") has none of those, so `panels` would be empty
    # and `fig.axes[-1]` would raise IndexError on a figure with zero subplots. Gated the
    # same way the Poes/volume jobs already are, rather than letting the per-job try/except
    # in _write_figures_impl silently absorb it as an unexplained failure on every run.
    any_pressure_or_flow = caps is not None and (
        caps.flow or caps.volume or caps.poes or caps.pgas or caps.pdi)
    jobs = []
    if dg.save_pv_average and poes:
        jobs.append(("PV average", _pv_average, "Campbell (average).pdf"))
    if dg.save_pv_individual and poes:
        jobs.append(("PV individual", lambda fr, fn, p: _pv_individual(fr, fn, p, cols, rows),
                     "Campbell (breaths).pdf"))
    if dg.save_raw and any_pressure_or_flow:
        jobs.append(("raw signals", _signals_raw, "signals (raw).pdf"))
    if dg.save_trimmed and any_pressure_or_flow:
        jobs.append(("trimmed signals", _signals_trimmed, "signals (trimmed).pdf"))
    # the MFVL figure needs a flow trace and a typed FVC breath (settings-only ceiling; a
    # file with no resolvable FVC still returns None from the job itself)
    if (getattr(dg, "save_flow_volume", True) and caps is not None and caps.flow
            and caps.volume and mfvllib.fvc_typed_in_settings(settings)):
        jobs.append(("flow-volume MFVL",
                     lambda fr, fn, p: _flow_volume_mfvl(fr, fn, p, settings),
                     "flow-volume (tidal in MFVL).pdf"))
    if dg.save_drift and volume:
        jobs.append(("volume correction", _volume_correction, "volume correction.pdf"))
        jobs.append(("trend", lambda fr, fn, p: _trend(fr, fn, p, settings), "volume trend.pdf"))
        jobs.append(("drift", _drift, "volume endpoints.pdf"))
    return jobs


def _emg_stage_candidates(settings):
    """Which of the three EMG conditioning stages (raw / ECG-removed / noise-reduced) a
    file COULD carry, given ``settings`` alone — before any data has been loaded. "Raw"
    exists whenever EMG channels are configured at all; the other two follow directly from
    the matching settings flags. A stage being a candidate here does not guarantee a
    particular file's computed ``signals['emg_stages']`` actually has it (see
    ``per_file_figure_jobs``'s docstring on jobs vs data) — this is the settings-only
    ceiling ``core.io.plan`` uses, never the data-driven truth the writer itself checks."""
    if not (settings.input.channels.emg or []):
        return []
    cands = [("raw", "Raw EMG")]
    if settings.processing.emg.remove_ecg:
        cands.append(("ecg_removed", "EMG (ECG removed)"))
    if settings.processing.emg.noise.enabled:
        cands.append(("noise_reduced", "EMG (ECG removed + noise reduced)"))
    return cands


def emg_overview_candidates(settings):
    """The EMG-overview-figure stage candidates a file could produce — see
    ``_emg_stage_candidates``. Kept as its own name (today identical to
    ``emg_audio_candidates``) because the overview and audio exports are independent
    features that could diverge later; sharing one private helper keeps them in lock-step
    until they do."""
    return _emg_stage_candidates(settings)


def emg_audio_candidates(settings):
    """The EMG-audio (WAV) stage candidates a file could produce — see
    ``emg_overview_candidates``."""
    return _emg_stage_candidates(settings)


def diagnostic_figure_type_count(settings):
    """How many DISTINCT diagnostic-figure *types* the current settings would produce for
    one file — the count Setup's "You will get" line shows (ticket A06). Not a per-file
    file count (that needs a file list; see ``core.io.plan.plan_outputs`` for that): a
    single ``save_drift`` tick is worth 3 figure types (volume correction, trend, volume
    endpoints), not 1, and this is the one place both Setup and Run read that number from,
    so they can never again disagree about what a set of ticked boxes means."""
    dg = settings.output.diagnostics
    n = len(per_file_figure_jobs(settings))
    if getattr(dg, "save_emg", False):
        n += len(emg_overview_candidates(settings))
    return n


def _write_figures_impl(result, settings, outputfolder: str, progress=None,
                        cohort_outputs: bool = True) -> tuple[list, list]:
    dg = settings.output.diagnostics
    emg = settings.processing.emg
    ylim = list(getattr(emg, "plot_yscale", []) or [])

    jobs = per_file_figure_jobs(settings)
    cols, rows = dg.pv_columns, dg.pv_rows
    caps = Capabilities.from_settings(settings)

    figdir = os.path.join(outputfolder, "diagnostics")
    written, failures = [], []
    have = bool(jobs) or getattr(dg, "save_emg", False) or emg.save_sound
    if not have:
        return [], []
    os.makedirs(figdir, exist_ok=True)

    for fname, fr in result.ok_files.items():
        if progress is not None:
            try:
                progress(fname)
            except Exception:               # noqa: BLE001 — progress is cosmetic, never fatal
                pass
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

    # one cohort "all-files average" Campbell across the whole batch — never built from a
    # subset (A05): a 2+-file re-run/single-file write is still not the whole study. Gated
    # on Capabilities.poes too (this ticket): a signal set without Poes has no Campbell loop
    # to average across files, per-file or cohort.
    if dg.save_pv_individual and len(result.ok_files) > 1 and cohort_outputs and caps.poes:
        path = os.path.join(figdir, "All files – Campbell (average).pdf")
        try:
            p = _pv_cohort(result, path, cols, rows)
            if p:
                written.append(p)
        except Exception as e:                  # noqa: BLE001
            failures.append(("cohort/PV average", str(e)))
    return written, failures
