"""Backwards-compatible facade for the split Preview & QC screen (ticket A02).

``preview_screen.py`` used to be a single 4273-line module holding every part
of Screen 2 (mechanics, ECG reduction, EMG + noise reduction, the busy
overlay, the reactive job scheduler, and various plot/figure-fit helpers). It
is now split into the ``respmech.ui.screens.preview`` package, organised one
module per concern (``_plot_helpers``, ``_figure_fit``, ``_busy_overlay``,
``_jobs``, ``_mechanics``, ``_ecg``, ``_emg_noise``, ``screen``). This module
re-exports everything the old module exposed at top level, unchanged, so
every existing import of ``respmech.ui.screens.preview_screen`` keeps
working without modification.

Prefer importing from ``respmech.ui.screens.preview`` (or its submodules)
directly in new code; this module exists purely for backwards compatibility.
"""
from respmech.ui.screens.preview import (
    # plot helpers
    _SUP, SciAxis, _CHANNELS, _EMG_PENS, _pen, _rms_envelope, _FALLBACK_PAL,
    _plot_pal, _CHECK_ICON_PATH, _check_icon_url, _tick_colour, _FitAxis,
    _next_nice, _restrict_body_wheel_to_x,
    # compact figure fitting
    _pick_xlabel, _pick_ylabel, _fit_compact_figure, refit_compact_figure,
    _CompactFigureFitter, _PlotTitleOverlay,
    # busy overlay
    BusyOverlay,
    # reactive-job bookkeeping
    _TAB_MECH, _TAB_ECG, _TAB_NOISE, _PANELS, _SPIN_TEXT, _KIND_LABEL,
    _AUTO_KINDS, _FILE_KINDS, _kinds_for_settings_path, _changed_settings_paths,
    _Job, _ORPHANED_THREADS, _MAX_ACTIVE, _FileRunError,
    # mechanics sub-tab
    _parse_breath_counts, _SOFT_FILE_ERRORS, _TREND_PROBE_KEYS,
    # EMG + noise sub-tab
    NEEDS_ECG_HINT, NEEDS_ECG_GATE_HINT,
    # the screen itself
    PreviewScreen, AUTO_BATCH_HINT,
)

__all__ = [
    "_SUP", "SciAxis", "_CHANNELS", "_EMG_PENS", "_pen", "_rms_envelope",
    "_FALLBACK_PAL", "_plot_pal", "_CHECK_ICON_PATH", "_check_icon_url",
    "_tick_colour", "_FitAxis", "_next_nice", "_restrict_body_wheel_to_x",
    "_pick_xlabel", "_pick_ylabel", "_fit_compact_figure", "refit_compact_figure",
    "_CompactFigureFitter", "_PlotTitleOverlay",
    "BusyOverlay",
    "_TAB_MECH", "_TAB_ECG", "_TAB_NOISE", "_PANELS", "_SPIN_TEXT", "_KIND_LABEL",
    "_AUTO_KINDS", "_FILE_KINDS", "_kinds_for_settings_path", "_changed_settings_paths",
    "_Job", "_ORPHANED_THREADS", "_MAX_ACTIVE", "_FileRunError",
    "_parse_breath_counts", "_SOFT_FILE_ERRORS", "_TREND_PROBE_KEYS",
    "NEEDS_ECG_HINT", "NEEDS_ECG_GATE_HINT",
    "PreviewScreen", "AUTO_BATCH_HINT",
]
