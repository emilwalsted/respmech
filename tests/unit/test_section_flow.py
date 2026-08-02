"""The section-column layout's contracts, stated without pixel literals.

``section_flow`` is what keeps a twenty-setting modal on a laptop screen. Its two
load-bearing promises are that a column of cards never demands the SUM of its cards (so the
dialog can always be squeezed to one column) and that it demands no minimum HEIGHT at all (so
the scroll area, not the window, absorbs the content). Everything else is arithmetic.

The balancer is a pure function over a list of heights, so most of this file needs no Qt at
all and says the same thing on every platform's fonts.
"""
import pytest

from respmech.ui.section_flow import SectionCard, WrapLabel, partition_columns

pytest.importorskip("PySide6")


def _tallest(heights, runs, vgap):
    return max(sum(heights[i:j]) + vgap * (j - i - 1) for i, j in runs)


def _equal_count(heights, k):
    """The naive split the balancer must never be worse than: equal CARD COUNT per column."""
    n = len(heights)
    size, runs, start = max(1, -(-n // k)), [], 0
    while start < n:
        runs.append((start, min(n, start + size)))
        start += size
    return runs


# heights measured off the real mechanics sections at Windows metrics, plus the two shapes a
# contiguous balancer is most likely to get wrong
_CASES = [
    [289, 151, 224, 674, 150, 110, 160],        # the real mechanics dialog
    [487, 300, 210, 190],                       # the real EMG dialog
    [900, 100, 100, 100, 100],                  # one tall section first
    [100, 100, 100, 100, 900],                  # one tall section last
    [200, 200, 200, 200, 200, 200],             # uniform
    [50],                                       # degenerate
]


@pytest.mark.parametrize("heights", _CASES)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_the_balanced_split_is_never_worse_than_an_equal_count_split(heights, k):
    runs = partition_columns(heights, k, vgap=12)
    assert runs, "the balancer returned no columns"
    # contiguous and complete: every card placed exactly once, in order
    assert runs[0][0] == 0 and runs[-1][1] == len(heights)
    assert all(a[1] == b[0] for a, b in zip(runs, runs[1:]))
    assert _tallest(heights, runs, 12) <= _tallest(heights, _equal_count(heights, k), 12)


def test_the_balancer_strictly_beats_the_naive_split_somewhere():
    """Guards against a vacuous suite: the inequality above would hold for a balancer that
    simply returned the naive split."""
    wins = 0
    for heights in _CASES:
        for k in (2, 3):
            runs = partition_columns(heights, k, vgap=12)
            if _tallest(heights, runs, 12) < _tallest(heights, _equal_count(heights, k), 12):
                wins += 1
    assert wins, "the balancer never beat an equal-count split — it is doing nothing"


def _card(qapp, title, rows, note=None):
    card = SectionCard(title)
    from PySide6.QtWidgets import QSpinBox
    for caption in rows:
        card.add_row(WrapLabel(caption), QSpinBox())
    if note:
        card.add_note(WrapLabel(note))
    card.show()
    for _ in range(3):
        qapp.processEvents()
    return card


def test_a_note_never_sets_a_cards_minimum_width(qapp):
    """Prose is the most compressible thing on the screen and must be the LAST thing to
    constrain a layout. Stated as a ratio, so it says the same thing on any fonts."""
    short = _card(qapp, "Trend", ["Minimum spacing", "Minimum depth"], note="Short note.")
    long = _card(qapp, "Trend", ["Minimum spacing", "Minimum depth"],
                 note="A considerably longer explanation " * 12)
    ratio = long.minimumSizeHint().width() / max(1, short.minimumSizeHint().width())
    assert ratio < 1.02, (
        f"a note ten times longer widened the card's minimum by {ratio:.2f}x — prose is "
        f"driving the layout")


def test_a_cards_minimum_is_its_widest_row_not_the_sum_of_a_row(qapp):
    """``WrapLongRows`` stacks the field under its caption when the row will not fit, so the
    honest minimum of a row is max(caption, field). This is the assertion that goes red if
    someone flips the row-wrap policy for aesthetic reasons — at which point the reported
    minimum becomes a lie and cards paint past their column edge."""
    card = _card(qapp, "Breath detection",
                 ["A deliberately long caption for this setting", "Short"])
    floor, natural = card.minimumSizeHint().width(), card.sizeHint().width()
    assert natural > 0
    assert floor < natural, (
        f"the card cannot be squeezed at all ({floor} of {natural} px)")


def test_the_layout_minimum_is_the_widest_card_and_has_no_height(qapp):
    """Both halves of the contract at once. The width half is what lets the dialog be
    squeezed to one column; the height half is what stops the cards demanding their whole
    stack back from the window, which is the original defect."""
    from PySide6.QtWidgets import QWidget
    from respmech.ui.section_flow import install_sections
    host = QWidget()
    columns = install_sections(host)
    cards = [_card(qapp, f"Card {i}", ["Caption here", "Another caption"]) for i in range(4)]
    for c in cards:
        columns.addWidget(c)
    host.show()
    for _ in range(4):
        qapp.processEvents()

    widest = max(c.minimumSizeHint().width() for c in cards)
    total = sum(c.minimumSizeHint().width() for c in cards)
    assert columns.minimumSize().width() <= widest + 4, (
        "the layout demands more than its widest single card")
    assert columns.minimumSize().width() < total / 2, (
        "the layout's minimum is closer to the SUM of its cards than to the widest one")
    assert columns.minimumSize().height() == 0, (
        "the layout demands a minimum height — the window will inherit it and the scroll "
        "area becomes pointless")


def test_the_column_count_follows_the_width_and_never_oscillates(qapp):
    """Expressed entirely in terms of the layout's own measured column target, never a pixel
    literal, so it means the same thing on macOS and Windows."""
    from PySide6.QtWidgets import QWidget
    from respmech.ui.section_flow import install_sections
    host = QWidget()
    columns = install_sections(host, max_columns=3)
    for i in range(6):
        columns.addWidget(_card(qapp, f"Card {i}", ["Caption here", "Another caption"]))
    host.show()
    for _ in range(4):
        qapp.processEvents()

    target, gap = columns.column_target(), columns._hgap
    assert target > 0
    assert columns.columns_for(target - 1) == 1
    assert columns.columns_for(2 * target + gap) >= 2
    assert columns.columns_for(3 * target + 2 * gap) == 3
    assert columns.columns_for(10 * target) == 3, "max_columns is not respected"

    seen = [columns.columns_for(w) for w in range(target // 2, 6 * target, max(1, target // 8))]
    assert seen == sorted(seen), f"the column count is not monotone in width: {seen}"


def test_more_columns_never_means_more_height(qapp):
    """The entire point of the columns: spending width buys back height."""
    from PySide6.QtWidgets import QWidget
    from respmech.ui.section_flow import install_sections
    host = QWidget()
    columns = install_sections(host, max_columns=3)
    for i in range(6):
        columns.addWidget(_card(qapp, f"Card {i}", ["Caption here", "Another caption"]))
    host.show()
    for _ in range(4):
        qapp.processEvents()
    target, gap = columns.column_target(), columns._hgap
    one = columns.heightForWidth(target)
    three = columns.heightForWidth(3 * target + 2 * gap)
    assert three < one, (
        f"three columns ({three} px) is not shorter than one ({one} px)")
