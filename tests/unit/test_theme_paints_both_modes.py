"""Every window must PAINT itself, and every fixed colour must be legible, in BOTH themes.

Emil's Windows screenshot, 30-07-2026: a light-mode session with a BLACK "Mechanics —
advanced" modal over a light application. Measured cause, and it was not a colour at all —
the window was painting no background whatsoever. In ``theme._QSS``,

    QMainWindow, QDialog { background-color: $window; }     came BEFORE
    QWidget            { background: transparent; }

and both are single type selectors, so they carry EQUAL CSS specificity and the LATER rule
wins the tie-break. Every top-level window in the app therefore had a transparent background,
with nothing behind it to show through: measured 74% of the mechanics modal, 49% of the
startup chooser and 26% of the main window left unfilled. macOS hides this (the window server
fills an unpainted surface with the system window colour); Windows paints it black.

The test is a real render with a magenta sentinel underneath: anything still magenta was never
painted. It is the only formulation that catches this — a palette or stylesheet assertion
passes happily while the pixels stay unpainted.

The contrast guards use WCAG relative luminance so the numbers mean something; the floor is
3:1, WCAG's minimum for a graphical element or large text.
"""
import pytest

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QDialog

from respmech.ui.state import AppState

from _helpers import requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

_SENTINEL = QColor(255, 0, 255)
#: Fraction of a window's surface allowed to remain unpainted. Not zero: antialiased corners
#: on a rounded card can leave a sampled pixel showing through, and the defect this catches is
#: 26-74%, so any small number separates them by a mile.
_UNPAINTED_TOLERANCE = 0.02
#: WCAG floor for a graphical element / large text.
_CONTRAST_FLOOR = 3.0


def _unpainted_fraction(widget):
    """Render ``widget`` over a magenta ground and report how much of it stayed magenta."""
    img = QImage(widget.size(), QImage.Format_ARGB32)
    img.fill(_SENTINEL)
    widget.render(img)
    xs = range(0, img.width(), max(1, img.width() // 40))
    ys = range(0, img.height(), max(1, img.height() // 40))
    total = hit = 0
    for y in ys:
        for x in xs:
            c = img.pixelColor(x, y)
            total += 1
            if c.red() > 250 and c.green() < 5 and c.blue() > 250:
                hit += 1
    return hit / max(1, total)


def _relative_luminance(colour):
    c = QColor(colour) if isinstance(colour, str) else QColor(*[int(v) for v in colour[:3]])
    out = []
    for v in (c.red(), c.green(), c.blue()):
        s = v / 255.0
        out.append(s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4)
    return 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]


def contrast(a, b):
    la, lb = _relative_luminance(a) + 0.05, _relative_luminance(b) + 0.05
    return max(la, lb) / min(la, lb)


def _all_windows(qapp, tmp_path):
    """One of every top-level window the user can meet, built and shown."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.dialogs import TextViewerDialog
    from respmech.ui.startup_dialog import StartupDialog

    s = synth_settings(str(tmp_path), data_out={"saveaveragedata": True},
                       remove_ecg=True, noise=True)
    win = MainWindow(AppState(s))
    win.resize(1100, 760)
    win.show()
    for _ in range(8):
        qapp.processEvents()

    captured = {}
    # Patch AdvancedDialog.exec ITSELF, not QDialog.exec (D13, UI-overhaul): the mechanics
    # modal is now non-modal and its exec() override never calls QDialog's own exec() at
    # all (see advanced_dialog.py), so a patch on the base class would silently miss it —
    # the dialog would actually show() and block on a real QEventLoop, forever, since
    # nothing here ever accepts/rejects/closes it.
    from respmech.ui.advanced_dialog import AdvancedDialog
    original = AdvancedDialog.exec

    def _capture(self):
        captured[self.windowTitle()] = self
        return QDialog.Rejected

    AdvancedDialog.exec = _capture
    try:
        for opener in ("_open_mech_advanced", "_open_emg_advanced", "_open_ecg_advanced"):
            getattr(win.preview_screen, opener)()
    finally:
        AdvancedDialog.exec = original

    windows = {"MainWindow": win}
    windows.update(captured)
    windows["StartupDialog"] = StartupDialog()
    windows["TextViewerDialog"] = TextViewerDialog("Error log", "trace\n" * 20)
    return windows


@pytest.mark.parametrize("dark", [False, True])
def test_every_window_paints_its_own_background(qapp, tmp_path, dark, request):
    """The regression: a top-level window with a transparent background shows the platform's
    uninitialised surface — black on Windows — regardless of which theme is selected."""
    if dark:
        request.getfixturevalue("dark_app")
    from respmech.ui import theme
    expected = QColor(theme.active_theme()["window"])

    windows = _all_windows(qapp, tmp_path)
    for name, w in windows.items():
        w.show()
        for _ in range(6):
            qapp.processEvents()
        bare = _unpainted_fraction(w)
        assert bare <= _UNPAINTED_TOLERANCE, (
            f"{name} left {bare:.0%} of its surface unpainted in "
            f"{'dark' if dark else 'light'} mode — nothing sits behind a top-level window, "
            f"so the platform decides what shows there (black on Windows)")
    for w in windows.values():
        w.close()
    assert expected.isValid()


@pytest.mark.parametrize("dark", [False, True])
def test_theme_tokens_are_legible_against_their_own_backgrounds(qapp, dark, request):
    """Each semantic token must clear the contrast floor against the surface it is used on,
    in whichever theme is active. A token tuned for one ground and used on the other is the
    defect class here — the preview error card carried dark-tuned literals that scored
    2.3:1 on the light theme's card."""
    if dark:
        request.getfixturevalue("dark_app")
    from respmech.ui import theme
    t = theme.active_theme()
    pairs = [
        ("body text", t["text"], t["window"]),
        ("body text on a card", t["text"], t["surface"]),
        ("muted text", t["text_muted"], t["surface"]),
        ("muted text on canvas", t["text_muted"], t["window"]),
        ("accent foreground", t["accent_fg"], t["accent"]),
        ("info", t["st_info_fg"], t["st_info_bg"]),
        ("ok", t["st_ok_fg"], t["st_ok_bg"]),
        ("warn", t["st_warn_fg"], t["st_warn_bg"]),
        ("error", t["st_error_fg"], t["st_error_bg"]),
        ("warn on canvas", t["st_warn_fg"], t["window"]),
    ]
    bad = [(n, fg, bg, contrast(fg, bg)) for n, fg, bg in pairs
           if contrast(fg, bg) < _CONTRAST_FLOOR]
    assert not bad, (
        f"{'dark' if dark else 'light'} mode: "
        + "; ".join(f"{n} {fg} on {bg} is {c:.2f}:1" for n, fg, bg, c in bad))


@pytest.mark.parametrize("dark", [False, True])
def test_the_plot_palette_is_legible_on_its_own_plot_ground(qapp, dark, request):
    """Neither plotting stack inherits Qt's QSS, so each trace colour is chosen by hand —
    and a colour left at its light-palette value disappears against the dark plot ground."""
    if dark:
        request.getfixturevalue("dark_app")
    from respmech.ui import theme
    pal = theme.plot_palette()
    ground = pal["bg"]
    faint = [(role, rgb, contrast(rgb, ground))
             for role, rgb in pal["channels"].items()
             if contrast(rgb, ground) < _CONTRAST_FLOOR]
    faint += [(f"emg[{i}]", rgb, contrast(rgb, ground))
              for i, rgb in enumerate(pal["emg_cycle"])
              if contrast(rgb, ground) < _CONTRAST_FLOOR]
    faint += [(k, pal[k], contrast(pal[k], pal["mpl_bg"]))
              for k in ("mpl_accent", "mpl_ok", "mpl_warn", "mpl_error")
              if contrast(pal[k], pal["mpl_bg"]) < _CONTRAST_FLOOR]
    assert not faint, (
        f"{'dark' if dark else 'light'} plot ground {ground}: "
        + "; ".join(f"{r} {v} is {c:.2f}:1" for r, v, c in faint))


@pytest.mark.parametrize("dark", [False, True])
def test_the_channel_tick_contrasts_with_every_channel_fill(qapp, dark, request):
    """The result-picker tick is drawn inside the channel's own coloured box, and the dark
    theme brightens every channel colour — so a fixed white tick that reads in light mode
    becomes a smear on dark. It is chosen by measuring, and this pins that it stays chosen
    by measuring."""
    if dark:
        request.getfixturevalue("dark_app")
    from respmech.ui import theme
    from respmech.ui.screens.preview_screen import _tick_colour
    weak = []
    for i, fill in enumerate(theme.plot_palette()["emg_cycle"]):
        c = contrast(_tick_colour(fill), fill)
        if c < _CONTRAST_FLOOR:
            weak.append((i, fill, _tick_colour(fill), c))
    assert not weak, (
        f"{'dark' if dark else 'light'}: "
        + "; ".join(f"channel {i} tick {t} on {f} is {c:.2f}:1" for i, f, t, c in weak))


# --------------------------------------------------------------------------- #
# C02 — disabled-field contrast, and a visible keyboard focus ring on
# checkboxes/radio buttons that does not shift the row it sits in.
# --------------------------------------------------------------------------- #
def test_disabled_tokens_meet_the_contrast_floor_in_both_themes():
    """Before this ticket: disabled_fg/disabled_bg measured 1.96:1 in light mode and
    2.46:1 in dark — you could tell a disabled field was THERE (e.g. Advanced/EMG
    noise's Auto-derived prop_decrease), not read the value it held. Dark mode's
    disabled_bg was also byte-identical to surface, so a disabled field's flat
    dissolved into the card it sat on and only a 1.27:1 border said a control existed
    at all. Reads the tokens directly, no Qt/rendering involved (theme.py is designed
    to be import-safe without Qt)."""
    from respmech.ui import theme
    for name, tokens in (("light", theme._LIGHT), ("dark", theme._DARK)):
        c = contrast(tokens["disabled_fg"], tokens["disabled_bg"])
        assert c >= _CONTRAST_FLOOR, (
            f"{name} mode: disabled_fg {tokens['disabled_fg']} on disabled_bg "
            f"{tokens['disabled_bg']} is only {c:.2f}:1")
    assert theme._DARK["disabled_bg"] != theme._DARK["surface"], (
        "dark disabled_bg is still byte-identical to surface — a disabled field's flat "
        "would dissolve into the card it sits on")


def _render_image(widget):
    img = QImage(widget.size(), QImage.Format_ARGB32)
    img.fill(0)
    widget.render(img)
    return img


def _diff_bbox(img_a, img_b):
    """Bounding box (xmin, ymin, xmax, ymax) of pixels that differ between two
    same-sized renders, or ``None`` if they are pixel-identical — mirrors
    ``PIL.ImageChops.difference(...).getbbox()``, which is how this ticket's own
    investigation first measured "zero pixels changed"."""
    assert img_a.size() == img_b.size()
    w, h = img_a.width(), img_a.height()
    xmin = ymin = xmax = ymax = None
    for y in range(h):
        for x in range(w):
            if img_a.pixelColor(x, y) != img_b.pixelColor(x, y):
                xmin = x if xmin is None else min(xmin, x)
                xmax = x if xmax is None else max(xmax, x)
                ymin = y if ymin is None else min(ymin, y)
                ymax = y if ymax is None else max(ymax, y)
    return None if xmin is None else (xmin, ymin, xmax, ymax)


def test_checkbox_and_radio_focus_is_visible_without_shifting_the_row(qapp):
    """C02: neither QCheckBox nor QRadioButton had any ``:focus`` rule — a keyboard
    user tabbing through a checklist (e.g. Output's ten save-format boxes) could not
    see which box currently had focus. Measured before the fix: two renders with
    focus on different boxes were pixel-identical (bbox ``None``). The fix widens the
    indicator's border only on focus and shrinks its content box by the same amount
    (16px + 1px border == 14px + 2px border == an 18px outer footprint either way), so
    the row's layout must not move either."""
    from PySide6.QtCore import Qt as _Qt
    from PySide6.QtWidgets import QCheckBox, QVBoxLayout, QWidget
    from respmech.ui import theme

    theme.apply_theme(qapp)
    w = QWidget()
    lay = QVBoxLayout(w)
    a = QCheckBox("Option A")
    b = QCheckBox("Option B")
    lay.addWidget(a)
    lay.addWidget(b)
    w.resize(220, 70)
    w.show()
    w.activateWindow()   # offscreen assigns focus only once a window is active
    for _ in range(3):
        qapp.processEvents()

    geo_a_before, geo_b_before = a.geometry(), b.geometry()

    a.setFocus(_Qt.FocusReason.OtherFocusReason)
    qapp.processEvents()
    assert a.hasFocus()
    img_a_focused = _render_image(w)
    a.clearFocus()

    b.setFocus(_Qt.FocusReason.OtherFocusReason)
    qapp.processEvents()
    assert b.hasFocus()
    img_b_focused = _render_image(w)
    b.clearFocus()
    qapp.processEvents()

    assert _diff_bbox(img_a_focused, img_b_focused) is not None, (
        "moving focus from one checkbox to another produced zero visible difference")
    assert a.geometry() == geo_a_before, "focusing a checkbox moved its own row"
    assert b.geometry() == geo_b_before, "focusing a checkbox moved a sibling row"
    w.close()
