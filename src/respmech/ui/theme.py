"""Visual theme for the RespMech GUI.

A single place that defines the app's look: a Fusion-based ``QPalette`` plus a
comprehensive Qt style sheet, and matching styling for the two plotting stacks
(``pyqtgraph`` for the live channel plots, ``matplotlib`` for the Campbell and
noise-fidelity figures). The aim is a calm, trustworthy *scientific instrument*
feel: white cards on a soft neutral canvas, one restrained steel-blue accent,
generous spacing, legible type, and quiet status colours.

Design goals / robustness contract:

* This module is **import-safe without Qt** — every Qt/pyqtgraph/matplotlib
  import is lazy and guarded, so ``import respmech.ui.theme`` never fails and the
  colour constants below can be reused by headless code.
* Nothing here ever raises: a missing colour-scheme API, an absent optional
  dependency, or an old Qt just degrades gracefully.
* Light theme is preferred; a dark variant is selected automatically when the OS
  reports a dark colour scheme (or via the ``RESPMECH_THEME`` env override).

Public API:
    apply_theme(app)        -> str   # applies palette + QSS; returns "light"/"dark"
    style_pyqtgraph()               # global pyqtgraph config (no-op if absent)
    style_matplotlib()              # scientific rcParams (no-op if absent)
    PALETTE, ACCENT, OK, WARN, ERROR, MUTED, INFO   # semantic colour constants
"""
from __future__ import annotations

import os
from string import Template

# --------------------------------------------------------------------------- #
# Semantic colour constants (theme-independent, safe on white plot backgrounds).
# These are the values widget/plot code should reach for when it needs a colour
# by meaning rather than by Qt role.
# --------------------------------------------------------------------------- #
ACCENT = "#2C6E9B"   # restrained clinical steel-blue — the single accent
OK = "#1F7A4D"       # success / valid
WARN = "#B7791F"     # caution
ERROR = "#B4322A"    # failure / invalid
MUTED = "#5C6B7A"    # secondary / de-emphasised text
INFO = ACCENT        # informational == accent

PALETTE = {
    "accent": ACCENT,
    "ok": OK,
    "warn": WARN,
    "error": ERROR,
    "muted": MUTED,
    "info": INFO,
}

# A calm, colour-blind-friendly cycle used by matplotlib and available to plots.
CYCLE = [ACCENT, "#B4322A", "#1F7A4D", "#B7791F", "#7D5BA6",
         "#0E7C7B", "#B4507A", "#5C6B7A"]

# --------------------------------------------------------------------------- #
# Plot palettes. Neither plotting stack inherits Qt's QSS/palette, so each needs
# its own light/dark colour table. The dark values are deliberately brightened so
# traces, breath shading and figure text stay legible on a near-black plot ground
# — a dark app whose plots stayed white would not read as dark at all. Screens
# pull the active table via ``plot_palette()`` at render time. The light table
# reproduces the historical hard-coded colours exactly, so light mode is unchanged.
# --------------------------------------------------------------------------- #
_PLOT_LIGHT = {
    "bg": "#FCFDFE", "fg": "#33404D", "grid_alpha": 0.15,
    # channel traces keyed by physiological meaning (mechanics stack)
    "channels": {"flow": (44, 110, 155), "volume": (31, 122, 77),
                 "poes": (180, 50, 42), "pgas": (183, 121, 31),
                 "pdi": (125, 91, 166)},
    # distinct pens cycled for the EMG raw/result/picker views
    "emg_cycle": [(44, 110, 155), (180, 50, 42), (31, 122, 77), (183, 121, 31),
                  (125, 91, 166), (14, 124, 123), (180, 80, 122), (92, 107, 122)],
    "breath_incl_brush": (44, 110, 155, 32), "breath_excl_brush": (180, 50, 42, 70),
    "breath_incl_label": (90, 107, 122), "breath_excl_label": (180, 50, 42),
    "separator": (150, 165, 180), "noise_region": (44, 110, 155, 45),
    "raw_trace": (150, 165, 180), "noise_trace": (90, 150, 200),
    # legend backing is only painted in dark mode (see _style_legend); the light
    # value is transparent+unused so light-mode legends stay exactly as before.
    "legend_bg": (255, 255, 255, 0),
    # matplotlib semantic line/figure colours (light values are the exact historical
    # ones so light-mode figures stay byte-identical: pure-white ground, grey lines)
    "mpl_bg": "#FFFFFF",
    "mpl_accent": "#2C6E9B", "mpl_ok": "#1F7A4D", "mpl_warn": "#B7791F",
    "mpl_error": "#B4322A", "mpl_muted": "0.6",
    "mpl_loop": "0.55", "mpl_zeroline": "0.85", "mpl_target": "0.35",
    "mpl_edge": "#B5BEC8", "mpl_grid": "#DCE2E8", "mpl_tick": "#5C6B7A",
    "mpl_cycle": list(CYCLE),
}
_PLOT_DARK = {
    "bg": "#14181D", "fg": "#B4BFCB", "grid_alpha": 0.18,
    "channels": {"flow": (96, 172, 226), "volume": (94, 200, 142),
                 "poes": (232, 118, 106), "pgas": (226, 178, 92),
                 "pdi": (180, 152, 226)},
    "emg_cycle": [(96, 172, 226), (232, 118, 106), (94, 200, 142), (226, 178, 92),
                  (180, 152, 226), (74, 200, 198), (230, 140, 182), (150, 166, 182)],
    "breath_incl_brush": (110, 172, 224, 42), "breath_excl_brush": (214, 92, 82, 62),
    "breath_incl_label": (150, 166, 182), "breath_excl_label": (234, 122, 112),
    "separator": (98, 112, 128), "noise_region": (110, 172, 224, 55),
    "raw_trace": (128, 140, 156), "noise_trace": (118, 176, 224),
    "legend_bg": (22, 27, 33, 220),   # near-opaque dark backing so legend text stays legible over fills
    "mpl_bg": "#14181D",
    "mpl_accent": "#5CA9DD", "mpl_ok": "#5FBE88", "mpl_warn": "#E0B357",
    "mpl_error": "#E8897F", "mpl_muted": "#7E8A97",
    "mpl_loop": "#9AA6B2", "mpl_zeroline": "#333B45", "mpl_target": "#8A97A4",
    "mpl_edge": "#3B4550", "mpl_grid": "#2A323C", "mpl_tick": "#8A97A4",
    "mpl_cycle": ["#5CA9DD", "#E8897F", "#5FBE88", "#E0B357", "#B39DE0",
                  "#54C8C6", "#E68CB6", "#9AA6B2"],
}

# Whether the most-recently-applied theme is the dark variant (drives the plots).
_IS_DARK = False


def plot_palette() -> dict:
    """Return the plot colour table for the currently-applied theme (a copy).
    Never raises; defaults to the light table before any theme is applied."""
    try:
        return dict(_PLOT_DARK if _IS_DARK else _PLOT_LIGHT)
    except Exception:
        return dict(_PLOT_LIGHT)


def is_dark() -> bool:
    """Whether the most-recently-applied theme is the dark variant. Never raises."""
    return bool(_IS_DARK)


# Fixed left-axis width (px) for vertically-stacked channel graphs. Wide enough for a
# rotated axis label plus physiological tick values (e.g. "-140.0"); pinning it keeps
# every panel's plotting area starting at the same x so the stacked x-axes line up.
PLOT_AXIS_WIDTH = 68


def align_left_axis(plot, width: int = PLOT_AXIS_WIDTH) -> None:
    """Pin a plot's left-axis to a fixed width so vertically-stacked channel graphs share
    the same left margin and their x-axes align — regardless of y-tick-label width. Without
    this, a panel whose ticks read "-140" sits further right than one reading "0", and the
    stacked channels visibly step in and out. Never raises (cosmetic-only)."""
    try:
        plot.getAxis("left").setWidth(int(width))
    except Exception:                       # pragma: no cover - never break a render over this
        pass

# --------------------------------------------------------------------------- #
# Full theme token tables. The QSS template and QPalette are built from these.
# --------------------------------------------------------------------------- #
_LIGHT = {
    "window": "#EEF1F5", "surface": "#FFFFFF", "surface_alt": "#F5F8FA",
    "base": "#FFFFFF", "text": "#1F2933", "text_muted": "#5C6B7A",
    "border": "#D8DEE6", "border_strong": "#C4CCD6", "grid": "#E6EBF0",
    "header_bg": "#F1F4F8",
    "accent": "#2C6E9B", "accent_hover": "#2A6591", "accent_pressed": "#24597F",
    "accent_fg": "#FFFFFF", "accent_soft": "#E6F0F7",
    "disabled_bg": "#F0F2F5", "disabled_fg": "#A7B0BB",
    "st_info_fg": "#215C86", "st_info_bg": "#E7F0F7", "st_info_bd": "#C4DBEC",
    "st_ok_fg": "#1F6B48", "st_ok_bg": "#E6F3EC", "st_ok_bd": "#BFE0CD",
    "st_warn_fg": "#8A5A12", "st_warn_bg": "#FBF1DD", "st_warn_bd": "#EBD6A8",
    "st_error_fg": "#A02B22", "st_error_bg": "#FBE9E7", "st_error_bd": "#F0C4BF",
}

_DARK = {
    "window": "#171B21", "surface": "#21272F", "surface_alt": "#1D232B",
    "base": "#1B2128", "text": "#E4E9EF", "text_muted": "#97A3B0",
    "border": "#2F3843", "border_strong": "#3B4550", "grid": "#2A323C",
    "header_bg": "#232B34",
    "accent": "#4B9CD3", "accent_hover": "#5CA9DD", "accent_pressed": "#3E8AC0",
    "accent_fg": "#0C1116", "accent_soft": "#263643",
    "disabled_bg": "#21272F", "disabled_fg": "#58636F",
    "st_info_fg": "#8FC4E8", "st_info_bg": "#1E2E3A", "st_info_bd": "#2C4457",
    "st_ok_fg": "#7FCBA4", "st_ok_bg": "#172A20", "st_ok_bd": "#2A4636",
    "st_warn_fg": "#E0B357", "st_warn_bg": "#2C2413", "st_warn_bd": "#4A3D1D",
    "st_error_fg": "#E8897F", "st_error_bg": "#2E1B19", "st_error_bd": "#4C2A26",
}

# Remembers the tokens of the most recently applied theme (for introspection).
_ACTIVE = dict(_LIGHT)


def active_theme() -> dict:
    """Return a copy of the token table for the theme currently applied."""
    return dict(_ACTIVE)


_CHECK_ICON_PATH = None


def _check_icon_path() -> str:
    """A white checkmark PNG (generated once, cached) for the ``:checked`` state of the
    checkbox indicator — Qt QSS ``url()`` needs a real file, not a data URI. Returns "" if
    it cannot be generated (the indicator then just fills with the accent colour)."""
    global _CHECK_ICON_PATH
    if _CHECK_ICON_PATH is None:
        try:
            import os as _os
            import tempfile
            from PySide6.QtGui import QImage, QPainter, QPen, QColor
            from PySide6.QtCore import QPointF, Qt as _Qt
            img = QImage(28, 28, QImage.Format_ARGB32)
            img.fill(_Qt.transparent)
            p = QPainter(img)
            p.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor("white"))
            pen.setWidth(3)
            pen.setCapStyle(_Qt.RoundCap)
            pen.setJoinStyle(_Qt.RoundJoin)
            p.setPen(pen)
            p.drawPolyline([QPointF(6, 14), QPointF(12, 21), QPointF(22, 8)])
            p.end()
            path = _os.path.join(tempfile.gettempdir(), "respmech_qss_check.png")
            img.save(path)
            _CHECK_ICON_PATH = path.replace("\\", "/")
        except Exception:                       # pragma: no cover - cosmetic
            _CHECK_ICON_PATH = ""
    return _CHECK_ICON_PATH


# --------------------------------------------------------------------------- #
# Style sheet template. Only ``$token`` placeholders are special (``string.
# Template``); the many literal ``{ }`` of QSS pass through untouched. All
# selectors below are valid Qt Style Sheet syntax.
# --------------------------------------------------------------------------- #
_QSS = Template(r"""
/* ---- canvas & generic containers ------------------------------------- */
QMainWindow, QDialog { background-color: $window; }

/* Plain containers are transparent so titled cards read cleanly on the
   canvas; every widget that must own its background re-asserts one below. */
QWidget { background: transparent; color: $text; }

QToolTip {
    background-color: $surface;
    color: $text;
    border: 1px solid $border;
    padding: 6px 9px;
    border-radius: 6px;
}

/* ---- card-style group boxes with titled headers ---------------------- */
QGroupBox {
    background-color: $surface;
    border: 1px solid $border;
    border-radius: 10px;
    margin-top: 20px;
    padding: 16px 16px 14px 16px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 2px;
    padding: 0 6px;
    color: $text_muted;
    letter-spacing: 1px;
}

/* ---- buttons: default + a 'primary' accent variant ------------------- */
QPushButton {
    background-color: $surface;
    color: $text;
    border: 1px solid $border_strong;
    border-radius: 8px;
    padding: 7px 16px;
    min-height: 18px;
}
QPushButton:hover { background-color: $surface_alt; border-color: $accent; }
QPushButton:pressed { background-color: $accent_soft; }
QPushButton:focus { outline: none; border-color: $accent; }
QPushButton:default { border-color: $accent; }
QPushButton:disabled {
    color: $disabled_fg;
    background-color: $disabled_bg;
    border-color: $border;
}
/* opt in with setObjectName("primary") OR setProperty("primary", True) */
QPushButton#primary, QPushButton[primary="true"] {
    background-color: $accent;
    color: $accent_fg;
    border: 1px solid $accent;
    font-weight: 600;
}
QPushButton#primary:hover, QPushButton[primary="true"]:hover {
    background-color: $accent_hover; border-color: $accent_hover;
}
QPushButton#primary:pressed, QPushButton[primary="true"]:pressed {
    background-color: $accent_pressed; border-color: $accent_pressed;
}
QPushButton#primary:disabled, QPushButton[primary="true"]:disabled {
    background-color: $disabled_bg; color: $disabled_fg; border-color: $border;
}

/* ---- tabs: clean underline style ------------------------------------- */
QTabWidget::pane {
    border: none;
    border-top: 1px solid $border;
    top: -1px;
    background: transparent;
}
QTabBar { qproperty-drawBase: 0; background: transparent; }
QTabBar::tab {
    background: transparent;
    color: $text_muted;
    padding: 9px 18px;
    margin: 0 2px;
    border: none;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:hover { color: $text; }
QTabBar::tab:selected {
    color: $accent;
    border-bottom: 2px solid $accent;
    font-weight: 600;
}
QTabBar:focus { outline: none; }

/* ---- text entry, combos, spin boxes: subtle border + focus ring ------ */
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QAbstractSpinBox {
    background-color: $base;
    color: $text;
    border: 1px solid $border;
    border-radius: 7px;
    padding: 7px 8px;
    min-height: 18px;                 /* match the Browse/action buttons' height in picker rows */
    selection-background-color: $accent;
    selection-color: $accent_fg;
}
QLineEdit:hover, QComboBox:hover, QAbstractSpinBox:hover { border-color: $border_strong; }
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QComboBox:focus, QAbstractSpinBox:focus {
    border: 1px solid $accent;   /* width unchanged -> no layout shift */
}
QLineEdit:disabled, QPlainTextEdit:disabled, QTextEdit:disabled,
QComboBox:disabled, QAbstractSpinBox:disabled {
    background-color: $disabled_bg; color: $disabled_fg; border-color: $border;
}
QComboBox QAbstractItemView {
    background-color: $surface;
    color: $text;
    border: 1px solid $border;
    selection-background-color: $accent_soft;
    selection-color: $text;
    outline: none;
    padding: 2px;
}

/* ---- check boxes / radios: ALWAYS draw the box, so a deselected option reads
       as an empty checkbox, not plain text ----------------------------- */
QCheckBox, QRadioButton { spacing: 8px; background: transparent; color: $text; }
QCheckBox:disabled, QRadioButton:disabled { color: $disabled_fg; }
QCheckBox::indicator, QRadioButton::indicator {
    width: 16px; height: 16px; border: 1px solid $border_strong; background: $base;
}
QCheckBox::indicator { border-radius: 4px; }
QRadioButton::indicator { border-radius: 9px; }
QCheckBox::indicator:hover, QRadioButton::indicator:hover { border: 1px solid $accent; }
QCheckBox::indicator:checked {
    border: 1px solid $accent; background: $accent; image: url("$check_icon");
}
QRadioButton::indicator:checked { border: 1px solid $accent; background: $accent; }
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {
    border: 1px solid $border; background: $disabled_bg;
}

/* ---- tables: light header, zebra rows, quiet grid -------------------- */
QTableView, QTableWidget {
    background-color: $surface;
    alternate-background-color: $surface_alt;
    color: $text;
    gridline-color: $grid;
    border: 1px solid $border;
    border-radius: 8px;
    selection-background-color: $accent_soft;
    selection-color: $text;
    qproperty-alternatingRowColors: true;   /* zebra without touching screens */
}
QTableView::item, QTableWidget::item { padding: 3px 6px; }
QHeaderView { background: transparent; }
QHeaderView::section {
    background-color: $header_bg;
    color: $text_muted;
    padding: 6px 8px;
    border: none;
    border-right: 1px solid $grid;
    border-bottom: 1px solid $border;
    font-weight: 600;
}
QHeaderView::section:last { border-right: none; }
QTableCornerButton::section {
    background-color: $header_bg;
    border: none;
    border-bottom: 1px solid $border;
    border-right: 1px solid $grid;
}

/* ---- progress bar ---------------------------------------------------- */
QProgressBar {
    background-color: $surface_alt;
    border: 1px solid $border;
    border-radius: 7px;
    text-align: center;
    color: $text;
    min-height: 18px;
}
QProgressBar::chunk {
    background-color: $accent;
    border-radius: 6px;
    margin: 1px;
}

/* ---- splitters (used heavily on the preview screen) ------------------ */
QSplitter::handle { background-color: $border; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:vertical { height: 1px; }
QSplitter::handle:hover { background-color: $accent; }

/* ---- scrollbars: slim & unobtrusive ---------------------------------- */
QScrollBar:vertical { background: transparent; width: 12px; margin: 0; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 0; }
QScrollBar::handle:vertical {
    background: $border_strong; border-radius: 6px; min-height: 28px; margin: 2px;
}
QScrollBar::handle:horizontal {
    background: $border_strong; border-radius: 6px; min-width: 28px; margin: 2px;
}
QScrollBar::handle:hover { background: $text_muted; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; background: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ---- menus (context menus, combo/dialog menus) ----------------------- */
QMenu {
    background-color: $surface;
    color: $text;
    border: 1px solid $border;
    border-radius: 8px;
    padding: 4px;
}
QMenu::item { padding: 6px 22px; border-radius: 5px; }
QMenu::item:selected { background-color: $accent_soft; color: $text; }
QMenu::separator { height: 1px; background: $border; margin: 4px 6px; }

/* ---- labels: base + status variants via a dynamic 'status' property --- */
QLabel { background: transparent; color: $text; }
QLabel[role="heading"] { font-size: 15px; font-weight: 600; color: $text; }
QLabel[status="muted"] { color: $text_muted; }
/* a disabled label (e.g. a caption inside a disabled group) greys with its controls */
QLabel:disabled, QLabel[status="muted"]:disabled { color: $disabled_fg; }
/* A banner's BOX (padding/border) hangs off the never-changing 'banner' property, not off
   'status': Qt bakes a QLabel's QSS box into its contentsMargins at the widget's FIRST
   polish and never recomputes it, so a box carried by a 'status' rule is lost forever on
   any banner whose first status has no box (e.g. "muted", which every banner starts empty
   in). 'status' therefore only ever swaps COLOURS, which re-polishing does honour. */
QLabel[banner="true"] {
    border-style: solid; border-width: 1px; border-color: transparent;
    border-radius: 6px; padding: 5px 10px;
}
QLabel[status="info"] {
    color: $st_info_fg; background-color: $st_info_bg; border-color: $st_info_bd;
}
QLabel[status="ok"] {
    color: $st_ok_fg; background-color: $st_ok_bg; border-color: $st_ok_bd;
}
QLabel[status="warn"] {
    color: $st_warn_fg; background-color: $st_warn_bg; border-color: $st_warn_bd;
}
QLabel[status="error"] {
    color: $st_error_fg; background-color: $st_error_bg; border-color: $st_error_bd;
}

/* ---- application header bar ------------------------------------------- */
QFrame#appHeader {
    background-color: $surface;
    border: none;
    border-bottom: 1px solid $border;
}
QLabel#appTitle { font-size: 17px; font-weight: 700; color: $accent; }
QLabel#appSubtitle { color: $text_muted; font-size: 11px; }

/* ---- the header's Analysis menu -------------------------------------
   right-padding leaves room for the drop-down indicator: styling the button
   without it makes Qt draw the arrow ON TOP of the last letter. */
QToolButton#appMenuButton {
    padding: 5px 24px 5px 10px;
    border: 1px solid $border;
    border-radius: 4px;
    background-color: transparent;
    color: $text;
    font-weight: 600;
}
QToolButton#appMenuButton:hover { background-color: $surface_alt; border-color: $border_strong; }
QToolButton#appMenuButton:pressed,
QToolButton#appMenuButton:open { background-color: $accent_soft; border-color: $accent; color: $accent; }
QToolButton#appMenuButton:disabled { color: $disabled_fg; border-color: $border; }
QToolButton#appMenuButton::menu-indicator {
    subcontrol-origin: padding;
    subcontrol-position: center right;
    right: 8px;
}

/* ---- compact icon-ish buttons (file pickers, small actions) ---------- */
QPushButton[compact="true"] {
    padding: 6px 12px;
    min-width: 0;
    font-weight: 600;
}

/* ---- glyph-only stepper buttons (◀ ▶): the default 16px side padding is
       wider than the glyph, which clips the arrow away entirely; 6px keeps the
       button compact while giving the arrow air. min-width:0 means the rendered
       width simply tracks sizeHint (this QSS minimum overwrites any code-side
       setFixedWidth minimum at polish), so the glyph can never be clipped —
       but do not push the padding past ~9px without re-checking the buttons'
       34px maximum in preview_screen. ---- */
QPushButton[nav="true"] {
    padding: 6px 6px;
    min-width: 0;
}

/* ---- form fields: keep them tidy, not full-width ---------------------
   Fusion's QFormLayout grows every uncapped field to the full form width
   (fieldGrowthPolicy defaults to AllNonFixedFieldsGrow), so a field with no cap
   stretches across the whole window while its spin-box siblings sit at ~150px.
   These are MAXIMA, not fixed widths: the field still shrinks with the window.
   Opt in via setProperty("formField", ...) so combos elsewhere (Preview, the
   channel/noise dialogs) keep sizing themselves.
   The tokens describe how much room the CONTENT needs, not the widget type:
   "compact" shares the spin boxes' column; "wide" is for fields whose longest
   content would elide there (the EMG-normalisation choices, a 'filename = count'
   line). Check a new field's sizeHint against the cap before picking. */
QAbstractSpinBox { max-width: 150px; }
*[formField="compact"] { max-width: 150px; }   /* same column as the spin boxes */
*[formField="wide"]    { max-width: 320px; }   /* content that will not fit the column */

/* ---- inner (sub-)tab bars read a touch smaller ----------------------- */
QTabBar::tab { min-width: 40px; }
""")


# --------------------------------------------------------------------------- #
# Theme selection
# --------------------------------------------------------------------------- #
def _prefers_dark(app) -> bool:
    """Best-effort OS dark-mode detection. Never raises; defaults to light."""
    env = os.environ.get("RESPMECH_THEME", "").strip().lower()
    if env in ("dark", "light"):
        return env == "dark"
    try:
        from PySide6.QtCore import Qt  # noqa: PLC0415
        cs = app.styleHints().colorScheme()
        return cs == Qt.ColorScheme.Dark
    except Exception:
        return False


def _build_palette(t: dict):
    """Construct a Fusion-friendly QPalette from a token table."""
    from PySide6.QtGui import QColor, QPalette  # noqa: PLC0415

    def c(key):
        return QColor(t[key])

    p = QPalette()
    p.setColor(QPalette.Window, c("window"))
    p.setColor(QPalette.WindowText, c("text"))
    p.setColor(QPalette.Base, c("base"))
    p.setColor(QPalette.AlternateBase, c("surface_alt"))
    p.setColor(QPalette.ToolTipBase, c("surface"))
    p.setColor(QPalette.ToolTipText, c("text"))
    p.setColor(QPalette.Text, c("text"))
    p.setColor(QPalette.Button, c("surface"))
    p.setColor(QPalette.ButtonText, c("text"))
    p.setColor(QPalette.BrightText, c("st_error_fg"))
    p.setColor(QPalette.Highlight, c("accent"))
    p.setColor(QPalette.HighlightedText, c("accent_fg"))
    p.setColor(QPalette.Link, c("accent"))
    p.setColor(QPalette.LinkVisited, c("accent_pressed"))
    # PlaceholderText exists on Qt6; guard just in case.
    try:
        p.setColor(QPalette.PlaceholderText, c("text_muted"))
    except Exception:
        pass
    # Disabled group.
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, c("disabled_fg"))
    p.setColor(QPalette.Disabled, QPalette.Highlight, c("disabled_bg"))
    p.setColor(QPalette.Disabled, QPalette.HighlightedText, c("disabled_fg"))
    return p


def apply_theme(app) -> str:
    """Apply the Fusion base style, palette, QSS and a comfortable base font to
    the given ``QApplication``. Also configures the plotting stacks. Chooses the
    dark variant when the OS reports a dark colour scheme, else light. Robust:
    logs nothing and raises nothing on failure. Returns "light"/"dark"."""
    global _ACTIVE, _IS_DARK
    try:
        dark = _prefers_dark(app)
        _IS_DARK = dark
        tokens = _DARK if dark else _LIGHT
        _ACTIVE = dict(tokens)

        try:
            app.setStyle("Fusion")
        except Exception:
            pass

        # Comfortable base font size without forcing a (possibly missing) family.
        try:
            f = app.font()
            if f.pointSizeF() > 0:
                f.setPointSizeF(max(f.pointSizeF(), 10.5))
            else:
                f.setPixelSize(14)
            app.setFont(f)
        except Exception:
            pass

        try:
            app.setPalette(_build_palette(tokens))
        except Exception:
            pass

        try:
            app.setStyleSheet(_QSS.safe_substitute(dict(tokens, check_icon=_check_icon_path())))
        except Exception:
            pass

        # Match the plots to the UI (each is independently guarded).
        style_pyqtgraph()
        style_matplotlib()
        return "dark" if dark else "light"
    except Exception:
        # Absolute belt-and-braces: theming must never take the app down.
        return "light"


# --------------------------------------------------------------------------- #
# Plot styling
# --------------------------------------------------------------------------- #
def style_pyqtgraph() -> None:
    """Set global pyqtgraph config to match the active theme: a light near-white
    ground with dark traces in light mode, a near-black ground with brightened
    traces in dark mode, antialiasing throughout. No-op if pyqtgraph is absent."""
    try:
        import pyqtgraph as pg  # noqa: PLC0415
    except Exception:
        return
    pal = _PLOT_DARK if _IS_DARK else _PLOT_LIGHT
    bg, fg = pal["bg"], pal["fg"]
    try:
        pg.setConfigOptions(antialias=True, background=bg, foreground=fg)
    except Exception:
        # Fall back to setting options one at a time.
        for key, val in (("antialias", True), ("background", bg), ("foreground", fg)):
            try:
                pg.setConfigOption(key, val)
            except Exception:
                pass


def style_matplotlib() -> None:
    """Apply clean scientific ``rcParams`` matched to the active theme for the
    Campbell/PSD/fidelity figures: modest type, top/right spines off, a soft grid,
    tight layout, a colour-blind-friendly cycle — light figures in light mode, a
    near-black ground with light type in dark mode. No-op if matplotlib is absent."""
    try:
        import matplotlib as mpl  # noqa: PLC0415
    except Exception:
        return
    pal = _PLOT_DARK if _IS_DARK else _PLOT_LIGHT
    bg = pal.get("mpl_bg", pal["bg"])
    try:
        rc = {
            "figure.facecolor": bg,
            "figure.edgecolor": bg,
            "savefig.facecolor": bg,
            "figure.autolayout": True,
            "axes.facecolor": bg,
            "axes.edgecolor": pal["mpl_edge"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.titlesize": 11,
            "axes.titleweight": "600",
            "axes.titlepad": 8.0,
            "axes.labelsize": 10,
            "axes.labelcolor": pal["fg"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": pal["mpl_grid"],
            "grid.alpha": 0.7,
            "grid.linewidth": 0.6,
            "font.size": 10,
            "text.color": pal["fg"],
            "xtick.color": pal["mpl_tick"],
            "ytick.color": pal["mpl_tick"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "lines.solid_capstyle": "round",
        }
        for key, val in rc.items():
            try:
                mpl.rcParams[key] = val
            except Exception:
                pass
        try:
            mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=pal["mpl_cycle"])
        except Exception:
            pass
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Small widget helpers
# --------------------------------------------------------------------------- #
def _repolish(widget) -> None:
    try:
        st = widget.style()
        if st is not None:
            st.unpolish(widget)
            st.polish(widget)
        widget.update()
    except Exception:
        pass



def make_primary(button) -> None:
    """Mark a ``QPushButton`` as the primary/accent action and re-polish. Works
    whether or not an object name is set. Never raises."""
    try:
        button.setProperty("primary", True)
        _repolish(button)
    except Exception:
        pass
