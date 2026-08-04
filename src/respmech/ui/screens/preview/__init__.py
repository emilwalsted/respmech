"""Re-export surface for the split preview_screen.py package (ticket A02).

Every name the old single-file module exposed at top level is re-exported
here unchanged, so `from respmech.ui.screens.preview_screen import X` (or
`from respmech.ui.screens.preview import X`) keeps working for every X that
worked before the split. `preview_screen.py` itself is now a thin facade
over this package.
"""

from ._plot_helpers import (_SUP, SciAxis, _CHANNELS, _EMG_PENS, _pen, _rms_envelope, _FALLBACK_PAL, _plot_pal, _CHECK_ICON_PATH, _check_icon_url, _tick_colour, _FitAxis, _next_nice, _restrict_body_wheel_to_x)
from ._figure_fit import (_pick_xlabel, _pick_ylabel, _fit_compact_figure, refit_compact_figure, _CompactFigureFitter, _PlotTitleOverlay)
from ._busy_overlay import (BusyOverlay)
from ._jobs import (_TAB_MECH, _TAB_ECG, _TAB_NOISE, _PANELS, _SPIN_TEXT, _KIND_LABEL, _AUTO_KINDS, _FILE_KINDS, _kinds_for_settings_path, _changed_settings_paths, _Job, _ORPHANED_THREADS, _MAX_ACTIVE, _FileRunError)
from ._mechanics import (_parse_breath_counts, _SOFT_FILE_ERRORS, _TREND_PROBE_KEYS)
from ._emg_noise import (NEEDS_ECG_HINT, NEEDS_ECG_GATE_HINT)
from .screen import PreviewScreen, AUTO_BATCH_HINT
