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
                                      infer_role_from_name, infer_roles_from_names,
                                      name_suffix, role_color)
from respmech.ui.plot_axis import MinPitchAxis


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


def test_a_column_the_file_does_not_have_is_blank_not_a_crash(qapp):
    """A saved mapping can outlive the file it was made against — a re-export with fewer
    channels. The row must still say which column it points at."""
    m, names = _matrix(cols=5)
    st = ColumnStack(1000, columns=[0, 6]).build(m, names)
    assert len(st.plots) == 2
    assert np.all(np.isnan(st.curves[1].getData()[1]))
    assert st.headers[1].text().startswith("Column 7")


def test_one_column_is_accepted(qapp):
    """A single-channel recording arrives as a 1-D array from some paths."""
    st = ColumnStack(1000, columns=[0]).build(np.arange(50.0), ["only"])
    assert len(st.plots) == 1 and len(st.curves[0].getData()[0]) == 50


# -- B05: MinPitchAxis + the compact sparkline mode -----------------------------
def test_rows_use_min_pitch_axis_by_default(qapp):
    """Even the dialog's rows (74 px) are short enough that pyqtgraph's top tick level can
    overlap itself; every ColumnStack row gets the thinning axis, not just Preview's."""
    m, names = _matrix()
    st = ColumnStack(1000, columns=[0]).build(m, names)
    assert isinstance(st.plots[0].getAxis("left"), MinPitchAxis)


def test_sparkline_mode_hides_the_axes_and_is_uniformly_short(qapp):
    m, names = _matrix()
    st = ColumnStack(1000, columns=[0, 1, 2], row_height=20, sparkline=True).build(m, names)
    for plot in st.plots:
        assert plot.getAxis("left").isVisible() is False
        assert plot.getAxis("bottom").isVisible() is False
        assert plot.minimumHeight() == 20              # no extra height for a hidden time axis
    assert st.plots[-1].getAxis("bottom").labelText == ""


def test_non_sparkline_mode_keeps_the_last_row_time_axis(qapp):
    """The channel-assignment dialog is unaffected by the sparkline addition: only the
    last row shows tick values, and it still carries the 'Time (s)' label."""
    m, names = _matrix()
    st = ColumnStack(1000, columns=[0, 1]).build(m, names)
    assert st.plots[0].getAxis("bottom").style["showValues"] is False
    assert st.plots[1].getAxis("bottom").style["showValues"] is True
    assert st.plots[1].getAxis("bottom").labelText == "Time (s)"
    assert st.plots[0].getAxis("left").isVisible() is True


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


# -- D27: seeding the channel-assignment dialog from the file's own column names ------
@pytest.mark.parametrize("name, expect", [
    ("flow", "flow"), ("Flow (L/s)", "flow"),                    # case-insensitive, extra text
    ("volume", "volume"), ("Vol", "volume"), ("VOLUME (L)", "volume"),
    ("poes", "poes"), ("Pes", "poes"), ("oesophageal", "poes"),
    ("pgas", "pgas"), ("pga", "pgas"), ("Gastric pressure", "pgas"),
    ("pdi", "pdi"), ("di", "pdi"),
    ("emg1", "emg"), ("EMG2", "emg"), ("EMG3", "emg"),
    ("edi", "emg"),               # a recognised diaphragm-EMG alias, NOT pdi's "di" it contains
    ("time", ""), ("ENT1", ""),   # no keyword matches at all
    ("", ""), ("   ", ""), ("__index", ""), ("Unnamed: 3", ""),   # blank/placeholder
])
def test_infer_role_from_name(qapp, name, expect):
    assert infer_role_from_name(name) == expect


@pytest.mark.parametrize("name", [
    "0", "12", "-3.5", "3,5",       # a bare number, '.' or ',' decimal
    "0,0000", "1,2500",             # header-less EU export: comma-decimal data row as header
])
def test_infer_role_from_name_rejects_numbers(qapp, name):
    """The bug report this ticket fixes: a header-less export gives pandas' own
    number-like fragments as column 'names', which must never be treated as real ones."""
    assert infer_role_from_name(name) == ""


@pytest.mark.parametrize("name", [
    "flowpoes",          # "flow" (4, flow) and "poes" (4, poes) -- an equally long tie
    "pdi_edi",           # "pdi" (3, pdi) and "edi" (3, emg) -- also tied, not resolved by length
])
def test_infer_role_from_name_is_ambiguous_not_guessed(qapp, name):
    assert infer_role_from_name(name) == ""


def test_infer_roles_from_names_matches_the_tickets_own_example(qapp):
    """The exact reproduction from the ticket: a 9-column CSV whose header names every
    channel. Column 0 (time) is never assignable and must be skipped."""
    names = "time,flow,volume,poes,pgas,pdi,emg1,emg2,emg3".split(",")
    assert infer_roles_from_names(names) == {
        1: "flow", 2: "volume", 3: "poes", 4: "pgas", 5: "pdi",
        6: "emg", 7: "emg", 8: "emg",
    }


def test_infer_roles_from_names_matches_the_real_synth_fixture_header(qapp):
    """tests/golden/input/synth_case_*.csv's actual header — the fixture every dialog test
    in test_channel_setup.py loads. ENT1-3 (entropy-only columns) suggest nothing, since
    entropy is never seeded from a name."""
    names = "time,EMG1,EMG2,EMG3,flow,volume,poes,pgas,pdi,ENT1,ENT2,ENT3".split(",")
    assert infer_roles_from_names(names) == {
        1: "emg", 2: "emg", 3: "emg", 4: "flow", 5: "volume", 6: "poes", 7: "pgas", 8: "pdi",
    }


def test_infer_roles_from_names_skips_a_headerless_export(qapp):
    """No real header row: pandas' own fragments of the first data row become the column
    'names' (comma-decimal, ';'-separated EU export) — none of them may seed anything."""
    names = ["0,0000", "0,2000", "0,3000", "0,4000"]
    assert infer_roles_from_names(names) == {}


def test_infer_roles_from_names_never_returns_column_zero(qapp):
    """Column 0 is the time axis and is never assignable in the dialog, even in the
    unlikely case its own name happened to look like a role."""
    names = ["flow", "flow"]           # column 0 named "flow" too
    assert infer_roles_from_names(names) == {1: "flow"}
