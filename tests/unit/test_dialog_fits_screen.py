"""Every dialog must fit the smallest screen we ship to, with its buttons reachable.

Emil's report on Windows, 30-07-2026: the "Mechanics — advanced" modal opened taller than the
screen, OK and Cancel sat below the bottom edge, and the window could not be resized smaller.
Measured cause: ~20 QFormLayout rows in a plain QVBoxLayout with no scroll area, so the
layout's minimum (636x904 on Windows font metrics) propagated to the window — and a window can
never be resized below its minimum, so nothing could recover the footer.

That is the same failure ``test_window_fits_screen.py`` already pins for the main WINDOW, and
these are its dialog-shaped siblings. Everything here is stated against a named SCREEN budget
or as a ratio, never as a pixel literal: a pixel literal is a measurement of the developer's
fonts and goes red on Windows, which is the mistake the sibling file is about.

The ``windows_metrics`` fixture models the wider Windows advance so these run on both metrics
locally; CI runs the unit suite on windows-latest and macos-latest, so both are covered for
real as well.
"""
import pytest

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QAbstractSpinBox, QDialog, QPlainTextEdit, QPushButton

from respmech.ui.state import AppState

from _helpers import requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

# The screens the app must be usable on, as (width, height) of the whole display.
# The tightest is a 1080p Windows laptop at 150% scaling; a 13" MacBook is 1280x800.
_SCREENS = {"mac 1280x800": (1280, 800),
            "win 1080p@125%": (1536, 864),
            "win 1080p@150%": (1280, 720)}
#: Title bar + taskbar/dock. Generous on purpose — a budget that only just fits is not a budget.
_CHROME = 70
#: The tightest usable height any dialog must fit inside.
_SHORTEST_USABLE_H = min(h for _w, h in _SCREENS.values()) - _CHROME
_NARROWEST_W = min(w for w, _h in _SCREENS.values())

#: A dialog that can be squeezed reports a minimum well under its natural size. A dialog that
#: cannot reports minimum == natural, i.e. exactly 1.0. Mirrors _WRAPPED in the sibling file.
_SHRINKABLE = 0.75

_DATA_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


def _advanced_dialogs(qapp, tmp_path):
    """Build the three Advanced modals without blocking on exec()."""
    from respmech.ui.advanced_dialog import AdvancedDialog
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path), data_out=_DATA_OUT, remove_ecg=True, noise=True)
    win = MainWindow(AppState(s))
    win.resize(1200, 800)
    win.show()
    for _ in range(6):
        qapp.processEvents()
    captured = {}
    # Patch AdvancedDialog.exec ITSELF, not QDialog.exec (D13, UI-overhaul): the mechanics
    # modal is now non-modal and its exec() override never calls QDialog's own exec() at
    # all (see advanced_dialog.py), so a patch on the base class would silently miss it —
    # the dialog would actually show() and block on a real QEventLoop, forever, since
    # nothing here ever accepts/rejects/closes it.
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
    assert captured, "no Advanced dialog was built — every assertion below would be vacuous"
    return captured, win


def _bottom_of(widget, dialog):
    """``widget``'s bottom edge in ``dialog`` coordinates."""
    return widget.mapTo(dialog, QPoint(0, 0)).y() + widget.height()


@pytest.mark.parametrize("metrics", ["native", "windows"])
def test_every_advanced_modal_fits_the_shortest_screen(qapp, tmp_path, metrics, request):
    """The regression itself: the dialog's own minimum must fit the tightest screen.

    minimumSizeHint, not sizeHint, is the number that decides reachability — it is the floor
    a window can be dragged to, so a minimum taller than the screen means the bottom of the
    dialog is unreachable by any gesture the user has.
    """
    if metrics == "windows":
        request.getfixturevalue("windows_metrics")
    dialogs, win = _advanced_dialogs(qapp, tmp_path)
    for title, dlg in dialogs.items():
        got = dlg.minimumSizeHint()
        assert got.height() <= _SHORTEST_USABLE_H, (
            f"{title!r} demands {got.height()} px of minimum height, more than the "
            f"{_SHORTEST_USABLE_H} px usable on the shortest screen we ship to — its footer "
            f"cannot be reached and no resize can recover it")
        assert got.width() <= _NARROWEST_W, (
            f"{title!r} demands {got.width()} px of minimum width against a "
            f"{_NARROWEST_W} px screen")
    win.close()


@pytest.mark.parametrize("screen", list(_SCREENS))
def test_ok_and_cancel_stay_on_screen_when_clamped(qapp, tmp_path, screen, windows_metrics):
    """Clamp each modal to a simulated screen and assert BOTH footer buttons are inside it.

    Run on the Windows metrics, which is where this broke. The precondition assert matters:
    without it the test would pass trivially on a machine whose fonts make every dialog small.
    """
    from respmech.ui import screen_fit
    width, height = _SCREENS[screen]
    avail = QRect(0, 0, width, height - _CHROME)
    dialogs, win = _advanced_dialogs(qapp, tmp_path)

    biggest = max(d.sizeHint().height() for d in dialogs.values())
    assert biggest > avail.height(), (
        "no modal's natural height exceeds the simulated screen, so this test would prove "
        "nothing — the fixture or the screen budget is wrong")

    for title, dlg in dialogs.items():
        dlg.show()
        screen_fit.clamp_to_screen(dlg, avail=avail)
        for _ in range(4):
            qapp.processEvents()
        assert dlg.height() <= avail.height() and dlg.width() <= avail.width(), (
            f"{title!r} is {dlg.width()}x{dlg.height()} on a "
            f"{avail.width()}x{avail.height()} work area")
        for button in (dlg.btn_ok, dlg.btn_cancel):
            assert _bottom_of(button, dlg) <= dlg.height(), (
                f"{title!r}: {button.text()!r} is below the bottom of the dialog on {screen}")
    win.close()


def test_a_modal_can_be_squeezed_far_below_its_natural_size(qapp, tmp_path, windows_metrics):
    """The invariant behind the budget, stated without pixels.

    A form laid out directly in a QVBoxLayout has minimum == natural height: it cannot be
    squeezed at all and whatever it sums to is forced on the window. A form inside a scroll
    area asks for its natural size but falls back to a small fraction of it.
    """
    dialogs, win = _advanced_dialogs(qapp, tmp_path)
    for title, dlg in dialogs.items():
        natural, floor = dlg.sizeHint().height(), dlg.minimumSizeHint().height()
        assert natural > 200, f"{title!r} reports a {natural} px natural height — never laid out"
        assert floor < natural * _SHRINKABLE, (
            f"{title!r} demands {floor} px of its {natural} px natural height — it cannot be "
            f"squeezed, so its whole height is forced on the window")
    win.close()


def test_the_footer_is_outside_the_scrolled_body(qapp, tmp_path):
    """Reachability by construction rather than by measurement: whatever the content does,
    the commit row is not part of what scrolls."""
    dialogs, win = _advanced_dialogs(qapp, tmp_path)
    for title, dlg in dialogs.items():
        inside = dlg.scroll.findChildren(QPushButton)
        assert dlg.btn_ok not in inside and dlg.btn_cancel not in inside, (
            f"{title!r} has its footer inside the scroll area, so it can scroll out of view")
    win.close()


def test_enter_commits_and_does_not_discard(qapp, tmp_path):
    """Enter must accept. Qt promotes the FIRST autoDefault button in the focus chain, and
    Cancel is added first so it reads left of OK — so without an explicit default, the one
    keyboard action available on a stuck dialog threw every staged edit away."""
    dialogs, win = _advanced_dialogs(qapp, tmp_path)
    for title, dlg in dialogs.items():
        defaults = [b.text() for b in dlg.findChildren(QPushButton) if b.isDefault()]
        assert defaults == [dlg.btn_ok.text()], (
            f"{title!r}: the default button is {defaults} — Enter must trigger "
            f"{dlg.btn_ok.text()!r}, not discard the dialog")
    win.close()


def test_the_wheel_over_a_field_scrolls_the_form_instead_of_editing_it(qapp, tmp_path):
    """A spin box inside a scroll area steps ITSELF on a wheel event, so scrolling past a
    control silently rewrites a scientific parameter. Also asserts the guard is held as an
    attribute — an event filter that is garbage-collected stops filtering, and every
    construction-time test stays green while the bug is back."""
    dialogs, win = _advanced_dialogs(qapp, tmp_path)
    dlg = dialogs["Mechanics — advanced"]
    dlg.show()
    dlg.resize(dlg.minimumSizeHint())
    for _ in range(4):
        qapp.processEvents()
    assert getattr(dlg, "_wheel_guard", None) is not None, "the wheel guard is not held"

    spins = [w for w in dlg.findChildren(QAbstractSpinBox) if w.isEnabled()]
    assert spins, "no spin box found — the assertion below would be vacuous"
    target = spins[0]
    before = target.value()
    ev = QWheelEvent(target.rect().center().toPointF(),
                     dlg.mapToGlobal(target.rect().center()).toPointF(),
                     QPoint(0, -120), QPoint(0, -120), Qt.NoButton, Qt.NoModifier,
                     Qt.NoScrollPhase, False)
    qapp.sendEvent(target, ev)
    for _ in range(4):
        qapp.processEvents()
    assert target.value() == before, (
        "a wheel over a spin box changed an analysis parameter instead of scrolling")
    win.close()


def test_a_scrollable_field_is_not_a_wheel_dead_patch(qapp, tmp_path):
    """A scrollable field inside the scrollable form must hand the wheel on once it has
    nothing left to scroll, or the form stops moving under the cursor and never resumes."""
    dialogs, win = _advanced_dialogs(qapp, tmp_path)
    dlg = dialogs["Mechanics — advanced"]
    dlg.show()
    dlg.resize(dlg.minimumSizeHint())
    for _ in range(6):
        qapp.processEvents()
    boxes = [w for w in dlg._widgets.values() if isinstance(w, QPlainTextEdit)]
    assert boxes, "no multi-line field found — this dialog is expected to have one"
    bar = dlg.scroll.verticalScrollBar()
    assert bar.maximum() > 0, "the form does not scroll at its own minimum size"
    bar.setValue(0)
    viewport = boxes[0].viewport()
    ev = QWheelEvent(viewport.rect().center().toPointF(),
                     viewport.mapToGlobal(viewport.rect().center()).toPointF(),
                     QPoint(0, -120), QPoint(0, -120), Qt.NoButton, Qt.NoModifier,
                     Qt.NoScrollPhase, False)
    qapp.sendEvent(viewport, ev)
    for _ in range(4):
        qapp.processEvents()
    assert bar.value() > 0, (
        "the wheel over the multi-line field went nowhere — it is a dead patch in the form")
    win.close()


def test_a_spin_box_never_clips_its_own_caption(qapp, tmp_path, windows_metrics):
    """A spin box showing ``specialValueText`` is showing a WORD, not a number, and the
    theme's 150 px cap clipped it mid-sentence on Windows metrics."""
    from respmech.ui import screen_fit
    from PySide6.QtWidgets import QLineEdit
    dialogs, win = _advanced_dialogs(qapp, tmp_path)
    dlg = dialogs["Mechanics — advanced"]
    dlg.show()
    screen_fit.clamp_to_screen(dlg, avail=QRect(0, 0, 1280, 720 - _CHROME))
    for _ in range(6):
        qapp.processEvents()
    captioned = [w for w in dlg.findChildren(QAbstractSpinBox) if w.specialValueText()]
    assert captioned, "no spin box carries a specialValueText — nothing to check"
    for box in captioned:
        need = box.fontMetrics().horizontalAdvance(box.specialValueText())
        line = box.findChild(QLineEdit)
        have = line.width() if line is not None else box.width()
        assert have >= need, (
            f"{box.specialValueText()!r} needs {need} px but its field is {have} px — the "
            f"caption is clipped, so the control does not say what it is doing")
    win.close()


def test_clamping_does_not_cap_a_dialog_to_the_screen_it_opened_on(qapp, tmp_path):
    """The clamp sets an opening SIZE, never a permanent maximum.

    An earlier cut pinned ``maximumSize`` to the work area so a later showMaximized could not
    push the footer off. But that cap outlives the screen that set it: a modal opened on a
    laptop and dragged to a docked monitor stayed stuck at the laptop's work area, with the
    OS maximise button neutered too. The structural fix makes the cap unnecessary — no dialog
    demands more minimum height than the screen any more.
    """
    from respmech.ui import screen_fit
    dialogs, win = _advanced_dialogs(qapp, tmp_path)
    small = QRect(0, 0, 1280, 720 - _CHROME)
    for title, dlg in dialogs.items():
        dlg.show()
        screen_fit.clamp_to_screen(dlg, avail=small)
        for _ in range(3):
            qapp.processEvents()
        assert dlg.width() <= small.width() and dlg.height() <= small.height(), (
            f"{title!r} did not open inside the simulated screen")
        # now the same dialog on a much larger display
        dlg.resize(small.width() + 900, small.height() + 550)
        for _ in range(3):
            qapp.processEvents()
        assert dlg.width() > small.width(), (
            f"{title!r} cannot grow past the screen it first opened on — its maximum has "
            f"been pinned to {dlg.maximumWidth()}x{dlg.maximumHeight()}")
    win.close()


def test_a_modal_opens_with_the_caret_on_its_first_setting(qapp, tmp_path):
    """A QScrollArea is focusable and sits before its contents in the tab order, so adding
    one silently moved the initial focus off the first field: typing did nothing until the
    user tabbed or clicked."""
    from PySide6.QtWidgets import QScrollArea
    dialogs, win = _advanced_dialogs(qapp, tmp_path)
    for title, dlg in dialogs.items():
        dlg.show()
        dlg.activateWindow()        # offscreen assigns focus only once a window is active
        for _ in range(4):
            qapp.processEvents()
        focused = qapp.focusWidget()
        assert focused is not None and not isinstance(focused, QScrollArea), (
            f"{title!r} opens with focus on {type(focused).__name__} rather than a setting")
        assert focused in dlg._widgets.values(), (
            f"{title!r} opens with focus on {type(focused).__name__}, which is not one of "
            f"its fields")
    win.close()


def test_no_caption_is_painted_outside_its_own_card(qapp, tmp_path, windows_metrics):
    """Qt does not CLIP a child laid out wider than its parent — it paints it past the edge.
    So a card that under-reports its minimum loses glyphs with no scrollbar to recover them,
    which is the same trap flow_layout.py documents one level down."""
    dialogs, win = _advanced_dialogs(qapp, tmp_path)
    for title, dlg in dialogs.items():
        dlg.show()
        dlg.resize(dlg.minimumSizeHint())       # the size the dialog itself calls legal
        for _ in range(6):
            qapp.processEvents()
        for card in dlg.cards:
            for caption in card._captions:
                right = caption.geometry().x() + caption.geometry().width()
                assert right <= card.width() + 1, (
                    f"{title!r}: caption {caption.fullText()[:40]!r} extends {right - card.width()} "
                    f"px past its card — those glyphs are painted outside and unreachable")
    win.close()


def test_every_setting_lands_on_a_named_card(qapp, tmp_path):
    """No setting may sit on the "Other" card, and no card may be empty.

    The grouping is a declarative table in ``preview_screen``, separate from the field list
    it groups, so the two can drift: a field added without a table entry, or a table entry
    naming a key that no longer exists. Neither is allowed to be silent — a setting the user
    cannot find is as good as missing, and it is still committed on OK.

    "Other" is the safety net, not a location: it exists so a drifted field is DISPLAYED
    rather than lost, and this asserts the net is never actually needed.
    """
    dialogs, win = _advanced_dialogs(qapp, tmp_path)
    for title, dlg in dialogs.items():
        names = [c.title() for c in dlg.cards]
        assert "Other" not in names, (
            f"{title!r} has settings on the 'Other' card — a field was added without a "
            f"section entry: {[c.title() for c in dlg.cards if c.title() == 'Other']}")
        empty = [c.title() for c in dlg.cards if not c._fields]
        assert not empty, f"{title!r} has empty cards {empty} — a section names no live key"
    win.close()


def test_no_setting_is_built_twice(qapp, tmp_path):
    """A key listed on two cards used to build two controls while ``_widgets`` kept only the
    second, so edits to the first were read back from the wrong widget and silently lost.
    The field list and the widget map must agree exactly."""
    dialogs, win = _advanced_dialogs(qapp, tmp_path)
    for title, dlg in dialogs.items():
        keys = [f.key for f in dlg._fields]
        assert len(keys) == len(set(keys)), (
            f"{title!r} builds a duplicate setting: "
            f"{sorted(k for k in set(keys) if keys.count(k) > 1)}")
        assert len(dlg._widgets) == len(keys), (
            f"{title!r} built {len(keys)} fields but kept {len(dlg._widgets)} widgets — "
            f"one control is orphaned and its edits go nowhere")
        on_cards = sum(len(c._fields) for c in dlg.cards)
        assert on_cards == len(keys), (
            f"{title!r} shows {on_cards} controls for {len(keys)} settings")
    win.close()


@pytest.mark.parametrize("channels", [1, 4, 12, 16])
def test_the_noise_profile_dialog_keeps_every_trace_readable(qapp, channels):
    """Its plots share one pane, so without a per-row minimum the traces collapse as the
    channel count grows — measured at zero pixels of data area for a 12-channel rig, on the
    very dialog whose whole job is to let the user SEE a quiet span."""
    import numpy as np
    from respmech.ui.noise_profile_dialog import NoiseProfileDialog
    t = np.arange(2000) / 500.0
    raw = [0.02 * np.sin(2 * np.pi * 60 * t)
           + 0.004 * np.random.default_rng(i).standard_normal(2000) for i in range(channels)]
    dlg = NoiseProfileDialog(raw, t, 500, list(range(2, 2 + channels)), file_name="d.csv")
    dlg.show()
    for _ in range(6):
        qapp.processEvents()
    heights = [p.getViewBox().geometry().height() for p in dlg._plots]
    assert len(heights) == channels
    # A ratio of the row budget, not a pixel figure: what matters is that a trace keeps most
    # of the room its row was promised, however the fonts measure.
    from respmech.ui.noise_profile_dialog import _ROW_MIN_H
    assert min(heights) > _ROW_MIN_H * 0.4, (
        f"{channels} channels leaves {min(heights):.0f} px of data area per trace")
    assert dlg.minimumSizeHint().height() <= _SHORTEST_USABLE_H, (
        f"{channels} channels forces a {dlg.minimumSizeHint().height()} px minimum height")
    dlg.close()
