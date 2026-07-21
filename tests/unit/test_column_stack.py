"""The shared stacked column preview (ui/column_stack.py).

Extracted from the channel-assignment dialog so the Setup screen's read-only channel
summary can draw the same thing. The dialog's own behaviour is covered by
test_channel_setup.py; what matters here is the generality the extraction was FOR — showing
a subset of columns, in a chosen order, with a caller-supplied header — because nothing
else exercises that until the summary exists.
"""
import numpy as np
import pytest
from PySide6.QtWidgets import QLabel

from respmech.ui.column_stack import (ASSIGNABLE, ColumnStack, ROLE_NAMES, ROLES,
                                      name_suffix, role_color)


def _matrix(n=200, cols=6):
    rng = np.random.RandomState(0)
    return rng.randn(n, cols), [f"ch{i}" for i in range(cols)]


def test_shows_every_column_by_default(qapp):
    m, names = _matrix()
    st = ColumnStack(1000).build(m, names)
    assert len(st.plots) == 6 and len(st.curves) == 6 and len(st.headers) == 6
    assert st.headers[0].text().startswith("Column 1")


def test_shows_only_the_chosen_columns_in_the_given_order(qapp):
    """The summary lists assigned roles in pipeline order, not column order."""
    m, names = _matrix()
    st = ColumnStack(1000, columns=[4, 1, 2]).build(m, names)
    assert len(st.plots) == 3
    assert [h.text().split()[1] for h in st.headers] == ["5", "2", "3"]


def test_the_time_axis_is_labelled_on_the_last_row_only(qapp):
    m, names = _matrix()
    st = ColumnStack(1000, columns=[3, 0]).build(m, names)
    assert st.plots[0].getAxis("bottom").style["showValues"] is False
    assert st.plots[1].getAxis("bottom").style["showValues"] is True
    assert st.plots[1].getAxis("bottom").labelText == "Time (s)"


def test_rows_share_one_time_axis(qapp):
    m, names = _matrix()
    st = ColumnStack(1000, columns=[0, 2, 4]).build(m, names)
    assert st.plots[0].getViewBox().linkedView(0) is None      # the chain root
    assert st.plots[1].getViewBox().linkedView(0) is not None
    assert st.plots[2].getViewBox().linkedView(0) is not None


def test_previews_are_inert(qapp):
    """The y-scale is the information — it must not be pannable or zoomable."""
    m, names = _matrix()
    st = ColumnStack(1000, columns=[1]).build(m, names)
    assert st.plots[0].getViewBox().state["mouseEnabled"] == [False, False]


def test_traces_are_coloured_by_role(qapp):
    m, names = _matrix()
    st = ColumnStack(1000, columns=[1, 2]).build(m, names, roles={1: "flow", 2: "emg"})
    flow = st.curves[0].opts["pen"].color()
    assert (flow.red(), flow.green(), flow.blue()) == tuple(role_color(st.pal, "flow"))
    st.set_role(1, "poes")                                     # re-role recolours in place
    poes = st.curves[0].opts["pen"].color()
    assert (poes.red(), poes.green(), poes.blue()) == tuple(role_color(st.pal, "poes"))


def test_the_header_factory_receives_each_shown_column(qapp):
    m, names = _matrix()
    seen = []

    def factory(i, head):
        seen.append(i)
        head.addWidget(QLabel(f"role for {i}"))

    ColumnStack(1000, columns=[5, 0], header_factory=factory).build(m, names)
    assert seen == [5, 0]                       # called in display order, with column indices


def test_set_data_replots_and_relabels_without_rebuilding(qapp):
    m, names = _matrix(n=200)
    st = ColumnStack(1000, columns=[0, 1]).build(m, names)
    plots_before = list(st.plots)
    m2, _ = _matrix(n=350)
    st.set_data(m2, ["newA", "newB", "c", "d", "e", "f"])
    assert st.plots == plots_before, "rows were rebuilt instead of re-plotted"
    assert len(st.curves[0].getData()[0]) == 350
    assert st.headers[0].text().endswith("newA")


def test_a_column_absent_from_the_new_file_is_blanked_not_stale(qapp):
    """Switching to a narrower file must not leave the previous file's trace on screen."""
    m, names = _matrix(cols=6)
    st = ColumnStack(1000, columns=[5]).build(m, names)
    st.set_data(*_matrix(n=200, cols=3))
    assert np.all(np.isnan(st.curves[0].getData()[1]))


def test_one_column_is_accepted(qapp):
    """A single-channel recording arrives as a 1-D array from some paths."""
    st = ColumnStack(1000, columns=[0]).build(np.arange(50.0), ["only"])
    assert len(st.plots) == 1 and len(st.curves[0].getData()[0]) == 50


# -- the vocabulary the two views must agree on --------------------------------
def test_every_assignable_role_has_a_summary_name_and_a_colour(qapp):
    """The summary names a role in prose where the dialog names it in a menu; a role
    present in one and missing from the other would render blank on the Setup screen."""
    pal = ColumnStack(1000).pal
    for key in ASSIGNABLE:
        assert key in ROLE_NAMES, f"{key} has no summary name"
        assert role_color(pal, key) is not None


def test_entropy_is_not_a_dropdown_role(qapp):
    """It is non-exclusive, so it gets a per-column checkbox instead — a dropdown cannot
    say "this column is both flow and entropy", and pretending it could deleted data."""
    assert "entropy" not in [k for k, _l in ROLES]
    assert "entropy" in ASSIGNABLE


@pytest.mark.parametrize("names, i, expect", [
    (["flow"], 0, "  ·  flow"),
    (["  "], 0, ""),                       # blank
    (["__index"], 0, ""),                  # pandas artefacts stay hidden
    (["Unnamed: 3"], 0, ""),
    ([], 0, ""),                           # past the end
])
def test_name_suffix_hides_the_unhelpful(qapp, names, i, expect):
    assert name_suffix(names, i) == expect
