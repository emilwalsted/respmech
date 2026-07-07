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
    set_status(label, kind)         # toggle a QLabel status variant + repolish
    apply_plot_style(plot_item)     # optional per-plot pyqtgraph grid/axis helper
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

# Backgrounds/foregrounds for the plotting stacks (kept light on purpose so the
# figures read like journal paper panels regardless of the app's light/dark UI).
PG_BACKGROUND = "#FCFDFE"
PG_FOREGROUND = "#33404D"

# A calm, colour-blind-friendly cycle used by matplotlib and available to plots.
CYCLE = [ACCENT, "#B4322A", "#1F7A4D", "#B7791F", "#7D5BA6",
         "#0E7C7B", "#B4507A", "#5C6B7A"]

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
    padding: 6px 8px;
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

/* ---- check boxes / radios (native indicator, tidy spacing) ----------- */
QCheckBox, QRadioButton { spacing: 8px; background: transparent; color: $text; }
QCheckBox:disabled, QRadioButton:disabled { color: $disabled_fg; }

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
QLabel[status="info"] {
    color: $st_info_fg; background-color: $st_info_bg;
    border: 1px solid $st_info_bd; border-radius: 6px; padding: 5px 10px;
}
QLabel[status="ok"] {
    color: $st_ok_fg; background-color: $st_ok_bg;
    border: 1px solid $st_ok_bd; border-radius: 6px; padding: 5px 10px;
}
QLabel[status="warn"] {
    color: $st_warn_fg; background-color: $st_warn_bg;
    border: 1px solid $st_warn_bd; border-radius: 6px; padding: 5px 10px;
}
QLabel[status="error"] {
    color: $st_error_fg; background-color: $st_error_bg;
    border: 1px solid $st_error_bd; border-radius: 6px; padding: 5px 10px;
}
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
    global _ACTIVE
    try:
        dark = _prefers_dark(app)
        tokens = _DARK if dark else _LIGHT
        _ACTIVE = dict(tokens)

        try:
            app.setStyle("Fusion")
        except Exception:
            pass

        # Comfortable base font size without forcing a (possibly missing) family.
        try:
            from PySide6.QtGui import QFont  # noqa: F401,PLC0415
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
            app.setStyleSheet(_QSS.safe_substitute(tokens))
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
    """Set global pyqtgraph config for a scientific look: near-white background,
    dark foreground, antialiasing. No-op if pyqtgraph is unavailable."""
    try:
        import pyqtgraph as pg  # noqa: PLC0415
    except Exception:
        return
    try:
        pg.setConfigOptions(
            antialias=True,
            background=PG_BACKGROUND,
            foreground=PG_FOREGROUND,
        )
    except Exception:
        # Fall back to setting options one at a time.
        for key, val in (("antialias", True),
                         ("background", PG_BACKGROUND),
                         ("foreground", PG_FOREGROUND)):
            try:
                pg.setConfigOption(key, val)
            except Exception:
                pass


def apply_plot_style(plot_item, grid_alpha: float = 0.12) -> None:
    """Optional helper: give a pyqtgraph ``PlotItem`` a subtle grid and quiet
    axes consistent with the theme. Screens may call this per plot. Never raises."""
    try:
        plot_item.showGrid(x=True, y=True, alpha=grid_alpha)
        for edge in ("left", "bottom", "right", "top"):
            try:
                ax = plot_item.getAxis(edge)
                if ax is not None:
                    ax.setPen(PG_FOREGROUND)
                    ax.setTextPen(PG_FOREGROUND)
            except Exception:
                pass
    except Exception:
        pass


def plot_pen(width: int = 1):
    """Return a pyqtgraph pen in the accent colour (or ``None`` if pyqtgraph is
    unavailable) so channel traces share the theme's accent."""
    try:
        import pyqtgraph as pg  # noqa: PLC0415
        return pg.mkPen(ACCENT, width=width)
    except Exception:
        return None


def style_matplotlib() -> None:
    """Apply clean, whitegrid-ish scientific ``rcParams`` for the Campbell and
    fidelity figures: modest type, top/right spines off, a soft grid, tight
    layout, and a colour-blind-friendly cycle. Figures stay light on purpose so
    they read like journal panels. No-op if matplotlib is unavailable."""
    try:
        import matplotlib as mpl  # noqa: PLC0415
    except Exception:
        return
    try:
        rc = {
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
            "axes.titleweight": "600",
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
        for key, val in rc.items():
            try:
                mpl.rcParams[key] = val
            except Exception:
                pass
        try:
            mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=CYCLE)
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


def set_status(label, kind: str = "muted") -> None:
    """Set a ``QLabel``'s status variant (``"info"``/``"ok"``/``"warn"``/
    ``"error"``/``"muted"``) and re-polish so the new QSS takes effect. Passing
    ``None`` or ``""`` clears the variant back to a plain label. Never raises."""
    try:
        label.setProperty("status", kind or "")
        _repolish(label)
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
