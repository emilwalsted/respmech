"""The style of the figures the core WRITES.

An output figure is a document that leaves the app, not a view of it, so it is pinned to
this light style however the analysis was launched and whatever theme the GUI is wearing.

This matters because the GUI's dark theme installs its colours into process-global
matplotlib ``rcParams`` (``ui.theme.style_matplotlib``), and ``core.plots`` reads that same
global at ``Figure()`` / ``add_subplot()`` / ``savefig()`` time. Avoiding ``pyplot`` avoids
the global *figure registry*, not the global *rcParams* — so without pinning, a dark UI
bleeds a near-black ground and light type into the written PDFs.

The values mirror the light table of ``ui.theme`` (both aim at the same calm scientific
look), but they live here so ``core`` stays UI-free: nothing under ``core/`` imports the
GUI. ``test_output_pdfs_are_always_light`` pins the two tables in step.
"""
from __future__ import annotations

import contextlib

# Every key ui.theme.style_matplotlib() writes, at its LIGHT value. The key set must stay
# complete: a key the dark theme sets but this table omits would leak straight through the
# context below.
_LIGHT_RC = {
    "figure.facecolor": "#FFFFFF",
    "figure.edgecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF",
    "figure.autolayout": True,
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#B5BEC8",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.titlesize": 11,
    # "bold", not "600": nothing sets font.family, so matplotlib uses DejaVu Sans, which
    # ships only weights 400 and 700. A request for 600 has always failed and fallen back
    # to 700 — printing "findfont: Failed to find font weight 600, now using 700." on every
    # figure — so this pins the weight that was already being rendered. Verified pixel-
    # identical; changing it would break the byte-for-byte written-figure guarantee.
    "axes.titleweight": "bold",
    "axes.titlepad": 8.0,
    "axes.labelsize": 10,
    "axes.labelcolor": "#33404D",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "#DCE2E8",
    "grid.alpha": 0.7,
    "grid.linewidth": 0.6,
    "font.size": 10,
    "text.color": "#33404D",
    "xtick.color": "#5C6B7A",
    "ytick.color": "#5C6B7A",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
    "lines.solid_capstyle": "round",
}

_LIGHT_CYCLE = ["#2C6E9B", "#B4322A", "#1F7A4D", "#B7791F",
                "#7D5BA6", "#0E7C7B", "#B4507A", "#5C6B7A"]


def light_rc() -> dict:
    """The full light rcParams for written figures, prop cycle included."""
    rc = dict(_LIGHT_RC)
    try:
        import matplotlib as mpl  # noqa: PLC0415
        rc["axes.prop_cycle"] = mpl.cycler(color=_LIGHT_CYCLE)
    except Exception:                       # pragma: no cover - matplotlib absent
        pass
    return rc


def light_rc_context():
    """Render inside the light style, then restore whatever the caller had.

    Restoring matters: the GUI draws its own (possibly dark) canvases from the same
    process, so writing figures must not leave the app's plots light. Degrades to a no-op
    context if matplotlib is absent, like the rest of the plotting stack.
    """
    try:
        import matplotlib as mpl  # noqa: PLC0415
    except Exception:                       # pragma: no cover - matplotlib absent
        return contextlib.nullcontext()
    return mpl.rc_context(rc=light_rc())
