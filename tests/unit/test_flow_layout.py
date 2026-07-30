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
