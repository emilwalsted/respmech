"""Diagnostic figures for a batch (feature P11).

The core computes but never plots; this is the optional plotting *consumer* the
core's docstrings point at. It turns finished results into publication-style PNGs
under ``<out>/diagnostics/``, driven by the ``output.diagnostics`` flags:

* ``save_pv_average``  → Poes–Volume (Campbell) loop: every breath overlaid, the
  resampled average breath drawn bold. This is the figure work-of-breathing is read
  from, so it is the flagship.
* ``save_pv_individual`` → a small-multiples grid, one Poes–Volume loop per breath.
* ``save_raw`` / ``save_trimmed`` → a stacked time-series of the analysed signals
  (flow, volume, Poes, Pdi). The core keeps the trimmed per-breath signal, so both
  flags draw from it; the panel is labelled accordingly.
* ``save_drift`` → end-expiratory / end-inspiratory lung volume across breaths, to
  show whether drift correction left any residual trend.

Rendering uses matplotlib's object API (``Figure`` + Agg canvas) rather than
``pyplot`` so it holds no global state — safe to call from a worker thread and it
never disturbs the GUI's Qt backend. Every figure is wrapped so a plotting failure
degrades to a skipped figure (reported in the returned list), never a failed run.
"""
from __future__ import annotations

import os

import numpy as np

_BRAND = "#2C6E9B"
_ACCENT = "#5CA9DD"
_MUTED = "#8894A0"


def _canvas(figsize):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=figsize, dpi=140)
    FigureCanvasAgg(fig)
    return fig


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    return path


def _breaths(fr):
    """Non-ignored breaths of a file result, in order."""
    if not fr.breaths:
        return []
    return [b for b in fr.breaths.values() if not b.get("ignored")]


def _pv_average(fr, fname, path):
    bs = _breaths(fr)
    if not bs:
        return None
    fig = _canvas((5.2, 5.6))
    ax = fig.add_subplot(111)
    for b in bs:
        ax.plot(b["volume"], b["poes"], color=_ACCENT, alpha=0.28, lw=0.8)
    b0 = bs[0]
    if b0.get("volumeavg") is not None and b0.get("poesavg") is not None \
            and len(b0["volumeavg"]) and len(b0["poesavg"]):
        ax.plot(b0["volumeavg"], b0["poesavg"], color=_BRAND, lw=2.4, label="average breath")
        ax.legend(loc="best", frameon=False)
    ax.set_xlabel("Volume (L)")
    ax.set_ylabel("Oesophageal pressure (cmH₂O)")
    ax.set_title(f"{fname} — Campbell / PV loop ({len(bs)} breaths)")
    ax.grid(True, color=_MUTED, alpha=0.2)
    return _save(fig, path)


def _pv_individual(fr, fname, path):
    bs = _breaths(fr)
    if not bs:
        return None
    n = len(bs)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig = _canvas((3.0 * cols, 2.8 * rows))
    for i, b in enumerate(bs):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.plot(b["volume"], b["poes"], color=_BRAND, lw=1.1)
        ax.set_title(f"Breath #{b.get('number', i + 1)}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, color=_MUTED, alpha=0.2)
    fig.suptitle(f"{fname} — per-breath PV loops", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, path)


def _signals(fr, fname, path):
    bs = _breaths(fr)
    if not bs:
        return None
    # concatenate the trimmed per-breath signals into one continuous trace
    def cat(key):
        return np.concatenate([np.asarray(b[key], dtype=float) for b in bs])
    panels = [("Flow (L/s)", "flow"), ("Volume (L)", "volume"),
              ("Poes (cmH₂O)", "poes"), ("Pdi (cmH₂O)", "pdi")]
    panels = [(lbl, k) for lbl, k in panels if k in bs[0]]
    fig = _canvas((10.0, 1.9 * len(panels)))
    for i, (lbl, key) in enumerate(panels):
        ax = fig.add_subplot(len(panels), 1, i + 1)
        ax.plot(cat(key), color=_BRAND, lw=0.6)
        ax.set_ylabel(lbl, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, color=_MUTED, alpha=0.2)
        if i < len(panels) - 1:
            ax.set_xticklabels([])
    fig.axes[-1].set_xlabel("Sample (trimmed, breaths concatenated)")
    fig.suptitle(f"{fname} — analysed signals", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, path)


def _drift(fr, fname, path):
    bs = _breaths(fr)
    if not bs:
        return None
    # eelv/eilv are stored as [volume, pressure] pairs (compute.py); the drift check
    # is on the VOLUME endpoint, so take component 0.
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


def write_figures(result, settings, outputfolder: str) -> tuple[list, list]:
    """Write every enabled diagnostic figure. Returns (written_paths, failures)
    where failures is a list of ``(name, error)`` — a bad figure is skipped, not
    fatal, so a run always finishes even if one plot cannot be drawn."""
    dg = settings.output.diagnostics
    jobs = []
    if dg.save_pv_average:
        jobs.append(("PV average", _pv_average, "PV loops (average)"))
    if dg.save_pv_individual:
        jobs.append(("PV individual", _pv_individual, "PV loops (breaths)"))
    if dg.save_raw or dg.save_trimmed:
        jobs.append(("signals", _signals, "signals"))
    if dg.save_drift:
        jobs.append(("drift", _drift, "drift"))
    if not jobs:
        return [], []

    figdir = os.path.join(outputfolder, "diagnostics")
    os.makedirs(figdir, exist_ok=True)
    written, failures = [], []
    for fname, fr in result.ok_files.items():
        for label, fn, suffix in jobs:
            path = os.path.join(figdir, f"{fname} – {suffix}.png")
            try:
                p = fn(fr, fname, path)
                if p:
                    written.append(p)
            except Exception as e:              # noqa: BLE001 - a plot must never fail a run
                failures.append((f"{fname}/{label}", str(e)))
    return written, failures
