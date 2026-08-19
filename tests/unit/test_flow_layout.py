"""``ui.flow_layout``: the wrapping strip used by the Preview control chips.

The wrapping itself is covered where it matters (test_window_fits_screen.py measures the
window minimum it exists to keep down). This file covers the placement WITHIN a row.
"""
def test_items_are_centred_vertically_in_their_row(qapp):
    """A row's items must share a vertical CENTRE, not a top edge.

    Before this, ``_lay`` placed every item at the row's top, each with its own natural
    height. On the ECG strip that put the "Capture channel"/"Min height"/"Min gap" captions
    and the two checkboxes visibly higher than the combo, the spin boxes and the buttons
    beside them (Emil, 30-07-2026: "kontrollerne skal alignes vertikalt"). The offset is a
    few pixels per item, which is exactly the size that reads as sloppy rather than broken.

    Centres are compared with a 1 px tolerance because centring an odd height difference in
    integer pixels cannot be exact."""
    from PySide6.QtWidgets import QWidget
    from respmech.ui.flow_layout import install_flow

    host = QWidget()
    lay = install_flow(host, h=10, v=4, margins=(0, 0, 0, 0))
    hoejder = (18, 26, 20, 32)
    for i, h in enumerate(hoejder):
        w = QWidget()
        w.setFixedSize(60, h)                      # deliberately mismatched heights
        lay.addWidget(w)
    host.resize(400, 60)                            # wide enough for one row
    host.show()
    lay.activate()

    geoms = [lay.itemAt(i).geometry() for i in range(lay.count())]
    assert len({g.y() for g in geoms}) > 1, (
        "all items share a top edge — they are being top-aligned, which is the bug")
    centres = [g.y() + g.height() / 2 for g in geoms]
    assert max(centres) - min(centres) <= 1.0, (
        f"items are not vertically centred in the row: centres {centres}")
    # and the row is still exactly as tall as its tallest item
    assert max(g.y() + g.height() for g in geoms) - min(g.y() for g in geoms) == max(hoejder)


def test_eliding_widgets_hint_from_the_full_text_so_a_squeeze_can_settle(qapp):
    """Eliding must never feed back into the layout that decided the width.

    Both eliding widgets shorten their caption to whatever width they are granted. If the
    shortened caption also shortened their ``sizeHint``, the strip could never settle: a
    squeezing ``QHBoxLayout`` distributes its deficit between ``sizeHint`` and
    ``minimumSizeHint``, so a smaller hint means a smaller deficit, a smaller deficit means
    this widget is granted MORE width, more width un-elides it, and the hint grows straight
    back. ``ElidingLabel`` was written with the override that breaks this; ``ElidingCheckBox``
    was not, and its home is exactly the squeezing row (_emg_noise.py's strip) rather than a
    placing FlowLayout. Measured 19-08-2026: the real strip cycled endlessly through five
    states at every width from 360 to 520 px, captions flicking between the full string and a
    bare ellipsis — Emil's "the row of buttons resizes and shuffles the controls around in an
    endless loop" on the EMG tab.

    The invariant is the fix: shrink the widget, and the RENDERED text may change all it
    likes, but the HINT may not."""
    from respmech.ui.flow_layout import ElidingCheckBox, ElidingLabel

    for widget in (ElidingCheckBox("Reduce EMG noise", floor_chars=8),
                   ElidingCheckBox("Auto strength", floor_chars=6),
                   ElidingLabel("Reduce EMG noise")):
        widget.resize(400, 24); widget.show()
        qapp.processEvents()
        hint = widget.sizeHint().width()
        elided_somewhere = False
        for width in (400, 200, 120, 80, 60, 40):
            widget.resize(width, 24)
            qapp.processEvents()
            assert widget.sizeHint().width() == hint, (
                f"{type(widget).__name__} moved its sizeHint {hint} -> "
                f"{widget.sizeHint().width()} when squeezed to {width} px: eliding is "
                f"feeding back into the layout, which is what makes the strip oscillate")
            elided_somewhere |= widget.text() != widget.fullText()
        # ... and the elide itself still happens — the HINT is pinned, not the rendering.
        # (Checked across the sweep, not after the narrowest step: below the chrome width
        # there is no room to elide INTO, and both classes then deliberately fall back to
        # showing the full string rather than rendering a bare ellipsis.)
        assert elided_somewhere, (
            f"{type(widget).__name__} never elided at any width — pinning the hint must not "
            f"disable the shortening itself")


def test_the_emg_noise_strip_settles_instead_of_oscillating(qapp):
    """The end-to-end guard for the same bug, on the real strip's own members.

    A pinned sizeHint is the mechanism; "the row stops moving" is the promise. Lay the strip
    out repeatedly at a fixed width and require the rendered captions and granted widths to
    reach a fixed point — before the fix this cycled through five distinct states forever, at
    every width tried."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget
    from respmech.ui.flow_layout import ElidingCheckBox, ElidingLabel

    host = QWidget()
    row = QHBoxLayout(host); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(10)
    noise_enabled = ElidingCheckBox("Reduce EMG noise", floor_chars=8)
    readout = ElidingLabel("", mode=Qt.ElideMiddle, floor=90)
    readout.setMaximumWidth(150)
    readout.setFullText("rest reference: COPD_01E_Baseline.txt  1.0-5.0 s")
    noise_auto = ElidingCheckBox("Auto strength", floor_chars=6)
    for w in (noise_enabled, QPushButton("Set noise profile"), readout,
              noise_auto, QPushButton("Advanced...")):
        row.addWidget(w)
    row.addStretch(1)
    host.show()

    for width in (600, 520, 480, 440, 400, 360, 320, 280):
        host.resize(width, 40)
        states = []
        for _ in range(12):
            qapp.processEvents()
            host.layout().activate()
            states.append((noise_enabled.text(), noise_auto.text(),
                           noise_enabled.width(), noise_auto.width()))
        settled = set(states[4:])           # allow the first passes to converge
        assert len(settled) == 1, (
            f"the strip never settles at {width} px — it cycles through {len(settled)} "
            f"states: {sorted(settled)}")
