"""Keep every top-level window inside the screen it opens on.

A window can never be resized below its layout's minimum, so a window whose content demands
more room than the display has is not merely awkward — its bottom edge, and whatever lives
there, is unreachable by any gesture the user has. That is what happened to the
"Mechanics — advanced" modal: 904 px of demanded minimum height against 650 px of usable
height on a 1080p Windows laptop at 150% scaling, with OK and Cancel in the missing 254 px.

This module is the second half of the answer, and only the second half. The first half is
structural — a scrollable body whose minimum does not grow with its content (see
``advanced_dialog``) — and no amount of clamping substitutes for it: ``resize()`` cannot take
a window below its minimum, so clamping a window that demands too much silently does nothing.
Fix the minimum first; use this to make sure the window also OPENS at a sensible size and can
never be pushed off the display afterwards.

Everything here is best-effort and never raises: a window that fails to be clamped is worse
positioned, not broken, and this must never be the reason the app cannot start.
"""
from __future__ import annotations

#: How much of the work area a window may occupy when its content wants more. Not 1.0: a
#: window flush with the work area looks maximised without being maximised, and on Windows it
#: hides the resize handles at the edges.
DEFAULT_FRACTION = 0.94


def available_for(widget=None):
    """The work area (screen minus taskbar/dock/menu bar) of the screen ``widget`` is on.

    Falls back to the primary screen, and returns ``None`` when no usable geometry is
    reported — an unusual headless setup, where the caller should simply leave the window
    alone rather than guess a size.
    """
    try:
        from PySide6.QtWidgets import QApplication  # noqa: PLC0415

        screen = None
        if widget is not None:
            # the screen the window is actually on, so a dialog opened on a second monitor is
            # clamped to THAT monitor rather than to the primary one
            try:
                handle = widget.windowHandle()
                screen = handle.screen() if handle is not None else None
                if screen is None:
                    screen = widget.screen()
            except Exception:
                screen = None
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen is None:
            return None
        avail = screen.availableGeometry()
        if avail is None or avail.width() <= 0 or avail.height() <= 0:
            return None
        return avail
    except Exception:                       # pragma: no cover - defensive
        return None


def clamp_to_screen(win, fraction: float = DEFAULT_FRACTION, avail=None, prefer=None) -> None:
    """Size ``win`` to its natural size but never past the work area, and move it fully on.

    ``avail`` overrides the detected work area, which is how a test simulates a 1280x720
    laptop without one: geometry is the whole subject here, so a guard that can only measure
    the developer's own display measures the wrong thing.

    ``prefer`` overrides ``sizeHint()`` as the size to aim for. A window whose content is a
    ``QScrollArea`` needs it: ``QScrollArea.sizeHint`` is clamped to 36x24 em regardless of
    what is inside, so a dialog that is meant to open showing a dozen channel previews would
    otherwise open showing two.

    Call it when the window is about to be shown, NOT from a constructor — a layout that has
    not been built yet reports a zero minimum, and clamping against zero is a no-op that
    looks like it worked. ``MainWindow._fit_to_screen`` did exactly that for three releases.
    """
    try:
        if avail is None:
            avail = available_for(win)
        if avail is None:
            return

        # Deliberately NO setMaximumSize. An earlier cut pinned the window's maximum to this
        # work area to stop a later showMaximized() pushing the footer off — but the cap
        # outlives the screen that set it: a modal opened on a laptop and then dragged to a
        # docked 4K monitor stayed stuck at the laptop's work area for the rest of its life,
        # measured resize(2200,1200) -> 1280x650, with the OS maximise button neutered too.
        # The cap was also solving a problem that no longer exists: it guarded a window whose
        # MINIMUM exceeded the screen, and the scrollable body means no dialog demands that
        # any more. A maximised window puts its footer at the bottom of the work area, which
        # is on screen by definition.
        hint = prefer if prefer is not None else win.sizeHint()
        want_w = min(hint.width(), int(avail.width() * fraction))
        want_h = min(hint.height(), int(avail.height() * fraction))
        # Never fight the window's own minimum: resize() below it is silently ignored, and
        # pretending otherwise is how a "fitted" window ends up off-screen anyway. If the
        # minimum genuinely exceeds the work area the structural fix is missing upstream —
        # position the window so its TOP-LEFT is visible, which at least keeps the title bar
        # and the first controls reachable.
        floor = win.minimumSizeHint()
        want_w = max(want_w, min(floor.width(), avail.width()))
        want_h = max(want_h, min(floor.height(), avail.height()))
        win.resize(want_w, want_h)

        frame = win.frameGeometry()
        frame.moveCenter(avail.center())
        top_left = frame.topLeft()
        top_left.setX(max(avail.left(), min(top_left.x(), avail.right() - frame.width() + 1)))
        top_left.setY(max(avail.top(), min(top_left.y(), avail.bottom() - frame.height() + 1)))
        # A window taller or wider than the work area would be centred OFF BOTH edges by the
        # moveCenter above; pinning to the work area's top-left keeps the half the user can
        # act on rather than losing symmetric slices of both ends.
        win.move(top_left)
    except Exception:                       # pragma: no cover - positioning is never fatal
        pass
