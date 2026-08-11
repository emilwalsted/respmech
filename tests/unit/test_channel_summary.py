"""The read-only channel readout that replaced the seven editable column fields.

Assignment happens only in the channel dialog now, so Setup shows what was chosen instead of
asking for it. What matters here: only assigned roles appear, a column carrying two roles is
still drawn once, the derived-volume case says so rather than showing a column nothing reads,
and each row keeps the settings-path tooltip its deleted field used to carry.
"""
import numpy as np
import pytest

from respmech.core.settings import Settings
from respmech.ui.channel_summary import (ChannelSummary, EMPTY_TEXT, ORDER, ROLE_HELP,
                                         assigned_columns, describe)
from respmech.ui.column_stack import ASSIGNABLE

from _helpers import INPUT, requires_synth, synth_settings  # noqa: F401


def _channels(**kw):
    ch = Settings().input.channels
    for k, v in kw.items():
        setattr(ch, k, v)
    return ch


def _matrix(n=200, cols=12):
    return np.random.RandomState(0).randn(n, cols), [f"ch{i}" for i in range(cols)]


# -- which rows appear ---------------------------------------------------------
def test_only_assigned_roles_get_a_row(qapp):
    ch = _channels(flow=5, poes=7, pgas=8, pdi=9)      # no volume, no emg, no entropy
    texts = ChannelSummary().show_mapping(ch).texts()
    assert not any("EMG" in t or "Entropy" in t for t in texts)


def test_unassigned_volume_still_gets_a_row_naming_the_state(qapp):
    """Ticket D02: Volume is the one role the summary never stays silent about, because a
    flow-only rig with no volume channel is a supported setup, not an omission — unlike
    EMG/entropy (genuinely absent above) the readout must say so rather than say nothing."""
    ch = _channels(flow=5, poes=7, pgas=8, pdi=9)      # no volume, no emg, no entropy
    texts = ChannelSummary().show_mapping(ch).texts()
    assert len(texts) == 5
    assert texts[0] == "Flow signal: Column #5"
    assert "Volume: not assigned" in texts


def test_rows_read_in_analysis_order(qapp):
    ch = _channels(flow=5, volume=6, poes=7, pgas=8, pdi=9, emg=[2], entropy=[3])
    texts = ChannelSummary().show_mapping(ch).texts()
    assert [t.split(":")[0] for t in texts] == [
        "Flow signal", "Volume", "Oesophageal pressure (Poes)",
        "Gastric pressure (Pgas)", "Transdiaphragmatic pressure (Pdi)", "EMG", "Entropy"]


def test_nothing_assigned_says_so(qapp):
    su = ChannelSummary().show_mapping(_channels())
    assert su.texts() == [] and su.stack is None
    assert "Assign channels" in EMPTY_TEXT


def test_multi_column_roles_list_every_column(qapp):
    ch = _channels(flow=5, emg=[2, 3, 4])
    assert "EMG: Columns #2, #3, #4" in ChannelSummary().show_mapping(ch).texts()


def test_a_single_column_role_is_not_pluralised(qapp):
    assert describe(_channels(emg=[7]), "emg") == "EMG: Column #7"


# -- the two cases that would otherwise read as quietly wrong -------------------
def test_derived_volume_says_so_instead_of_naming_an_ignored_column(qapp):
    """With 'Calculate volume from flow' on, the volume column is ignored. Showing its
    number would be a number nothing reads."""
    ch = _channels(flow=5, volume=6)
    assert describe(ch, "volume", integrate_from_flow=True) == "Volume: derived from flow"
    assert describe(ch, "volume", integrate_from_flow=False) == "Volume: Column #6"


def test_neither_volume_source_says_not_assigned_instead_of_nothing(qapp):
    """Ticket D02: unlike every other role, Volume never returns None from describe() —
    a flow-only rig is supported, so the summary must be able to say "not assigned" the
    way the other roles quietly disappear instead."""
    ch = _channels(flow=5)                             # no volume column, no integration
    assert describe(ch, "volume", integrate_from_flow=False) == "Volume: not assigned"
    assert describe(ch, "flow", integrate_from_flow=False) is not None
    assert describe(ch, "poes", integrate_from_flow=False) is None      # unaffected role


def test_a_column_carrying_two_roles_is_drawn_once_and_names_both(qapp):
    """Entropy is non-exclusive, so column 5 can be both the flow signal and an entropy
    channel. One trace, one header, both roles named."""
    ch = _channels(flow=5, poes=7, entropy=[5, 11])
    m, names = _matrix()
    su = ChannelSummary().show_mapping(ch, matrix=m, names=names)
    assert assigned_columns(ch) == [5, 7, 11]
    assert len(su.stack.plots) == 3
    assert su.texts()[0].startswith("Flow signal + Entropy  ·  Column 5")
    assert su.texts()[2].startswith("Entropy  ·  Column 11")


def test_there_is_no_separate_legend(qapp):
    """Each graph names itself, so a list above repeating the same facts would only be a
    second place to keep in sync. Volume is assigned here (unlike most fixtures in this
    file) specifically so the ticket D02 trailing note below does not fire — that note is
    the ONE deliberate exception, covered on its own by the tests below, not this one."""
    ch = _channels(flow=5, volume=6, poes=7)
    m, names = _matrix()
    su = ChannelSummary().show_mapping(ch, matrix=m, names=names)
    assert su.rows == su.stack.headers
    assert not any(t.startswith("Flow signal:") for t in su.texts())


def test_unassigned_volume_gets_a_trailing_note_beside_the_graphs(qapp):
    """Ticket D02: the graphs can only draw a column, and Volume may have none — a label
    is appended after the stack instead, the one deliberate exception to 'every row is a
    graph header' that test_there_is_no_separate_legend otherwise enforces."""
    ch = _channels(flow=5, poes=7)                     # no volume column, no integration
    m, names = _matrix()
    su = ChannelSummary().show_mapping(ch, matrix=m, names=names)
    assert len(su.rows) == len(su.stack.headers) + 1
    assert su.rows[-1] not in su.stack.headers
    assert su.texts()[-1] == "Volume: not assigned"


def test_derived_volume_gets_a_trailing_note_and_no_misleading_column_tag(qapp):
    """Ticket D02: with 'derive from flow' on, an assigned Volume column is ignored by the
    core loader at run time — the column's own graph row must not still say 'Volume' (it
    would claim a signal is being read that is not), and the trailing note must say the
    volume is derived from flow instead of naming the ignored column."""
    ch = _channels(flow=5, volume=6, poes=7)
    m, names = _matrix()
    su = ChannelSummary().show_mapping(ch, matrix=m, names=names, integrate_from_flow=True)
    assert not any("Volume" in h.text() for h in su.stack.headers)
    assert su.texts()[-1] == "Volume: derived from flow"


def test_a_two_role_header_carries_both_settings_paths(qapp):
    ch = _channels(flow=5, entropy=[5])
    m, names = _matrix()
    su = ChannelSummary().show_mapping(ch, matrix=m, names=names)
    tip = su.stack.headers[0].toolTip()
    assert "input.channels.flow" in tip and "input.channels.entropy" in tip


def test_the_role_wins_the_colour_of_a_shared_column(qapp):
    from respmech.ui.column_stack import role_color
    ch = _channels(flow=5, entropy=[5])
    m, names = _matrix()
    su = ChannelSummary().show_mapping(ch, matrix=m, names=names)
    pen = su.stack.curves[0].opts["pen"].color()
    assert (pen.red(), pen.green(), pen.blue()) == tuple(role_color(su.stack.pal, "flow"))


# -- the graphs are optional ---------------------------------------------------
def test_rows_render_without_a_readable_file(qapp):
    """Setup shows the mapping as soon as it exists; the traces need a file, the rows do not."""
    su = ChannelSummary().show_mapping(_channels(flow=5, poes=7))
    assert su.texts() and su.stack is None


def test_rebuilding_replaces_rather_than_accumulates(qapp):
    su = ChannelSummary()
    su.show_mapping(_channels(flow=5, poes=7))
    su.show_mapping(_channels(flow=5))
    assert su.texts() == ["Flow signal: Column #5", "Volume: not assigned"]
    su.show_mapping(_channels())
    assert su.texts() == []


# -- the tooltip contract inherited from the deleted fields --------------------
def test_every_row_names_its_settings_variable(qapp):
    """test_gui_hardening asserted this on sc.col_flow / sc.cols_emg; the rows are its new
    home, so hovering still tells the user which TOML key they are looking at."""
    ch = _channels(flow=5, volume=6, poes=7, pgas=8, pdi=9, emg=[2], entropy=[3])
    su = ChannelSummary().show_mapping(ch)          # no file -> the plain fallback list
    for role, lab in zip(ORDER, su.rows):
        var = ROLE_HELP[role][0]
        assert var in lab.toolTip(), f"{var} missing from {lab.toolTip()!r}"
        assert len(lab.toolTip()) > len(var) + 15, "no description"


@pytest.mark.parametrize("role", ASSIGNABLE)
def test_every_assignable_role_can_be_described_and_helped(qapp, role):
    """A role the dialog can assign but the summary cannot render would vanish from Setup."""
    assert role in ROLE_HELP
    value = [3] if role in ("emg", "entropy") else 3
    assert describe(_channels(**{role: value}), role) is not None


# -- the readout names itself -------------------------------------------------
def test_each_graph_says_what_it_is_before_where_it_sits(qapp):
    """Reading a trace should not require carrying the legend above in your head, so the
    role comes first and the column number second."""
    ch = _channels(flow=5, poes=7, emg=[2])
    m, names = _matrix()
    su = ChannelSummary().show_mapping(ch, matrix=m, names=names)
    heads = [h.text() for h in su.stack.headers]
    assert heads[0].startswith("EMG  ·  Column 2")
    assert heads[1].startswith("Flow signal  ·  Column 5")
    assert heads[2].startswith("Oesophageal pressure (Poes)  ·  Column 7")


def test_a_column_with_two_roles_names_both(qapp):
    ch = _channels(flow=5, entropy=[5])
    m, names = _matrix()
    su = ChannelSummary().show_mapping(ch, matrix=m, names=names)
    assert su.stack.headers[0].text().startswith("Flow signal + Entropy  ·  Column 5")


def test_the_previews_are_shorter_here_than_in_the_dialog(qapp):
    """A readout, not a working surface: a low sparkline (ticket B05) with no axis
    apparatus at all, so unlike the dialog's full ColumnStack there is no axis text left to
    clip and every row — including the last — is the same, uniformly short, height."""
    from respmech.ui.column_stack import ROW_HEIGHT
    from respmech.ui.channel_summary import SUMMARY_ROW_HEIGHT
    assert SUMMARY_ROW_HEIGHT < ROW_HEIGHT
    ch = _channels(flow=5, poes=7)
    m, names = _matrix()
    su = ChannelSummary().show_mapping(ch, matrix=m, names=names)
    assert su.stack.plots[0].minimumHeight() == SUMMARY_ROW_HEIGHT
    # unlike the dialog, no row reserves extra height for a time axis — it is hidden
    assert su.stack.plots[-1].minimumHeight() == SUMMARY_ROW_HEIGHT


def test_the_summary_has_no_visible_axis(qapp):
    """The compact readout shows the SHAPE of a signal, not values one could read off it —
    the header text already names the role/column/source-name, so a tick-labelled axis
    would only repeat that while costing the vertical space the summary exists to save."""
    ch = _channels(flow=5, poes=7)
    m, names = _matrix()
    su = ChannelSummary().show_mapping(ch, matrix=m, names=names)
    for plot in su.stack.plots:
        assert plot.getAxis("left").isVisible() is False
        assert plot.getAxis("bottom").isVisible() is False


# -- plot cleanup (point 6, ticket 20260811-0910) -----------------------------------------
def test_rebuilding_the_mapping_closes_the_previous_stacks_plots(qapp):
    """Every show_mapping() rebuild used to just setParent(None) the old ColumnStack and
    drop the reference — never closing its PlotWidgets' own context menus. _clear() now
    closes the outgoing stack's plots before the new one replaces it.

    The PlotItem references are captured BEFORE the rebuild, not re-fetched via
    ``PlotWidget.getPlotItem()`` afterwards — see test_column_stack.py's
    ``test_close_plots_closes_every_embedded_plotitem_and_empties_the_list`` for why a
    closed PlotWidget's own ``getPlotItem()`` returns ``None``, not a readable PlotItem."""
    ch = _channels(flow=5, poes=7)
    m, names = _matrix()
    su = ChannelSummary().show_mapping(ch, matrix=m, names=names)
    old_plots = list(su.stack.plots)
    old_plot_items = [p.getPlotItem() for p in old_plots]
    assert old_plot_items and all(pi.ctrlMenu is not None for pi in old_plot_items)
    su.show_mapping(ch, matrix=m, names=names)   # rebuilds regardless of the mapping's own signature
    assert all(pi.ctrlMenu is None for pi in old_plot_items)
    assert su.stack is not None and su.stack.plots is not old_plots


def test_close_plots_closes_the_current_stack_without_rebuilding(qapp):
    """The counterpart to Preview & QC's shutdown(): releases the LAST-built ColumnStack's
    plots (the one still alive when the owning window closes), called from
    MainWindow.closeEvent — unlike _clear()/show_mapping(), this does not restore the
    empty-state label, because the caller is discarding the whole screen, not redrawing it."""
    ch = _channels(flow=5, poes=7)
    m, names = _matrix()
    su = ChannelSummary().show_mapping(ch, matrix=m, names=names)
    plot_items = [p.getPlotItem() for p in su.stack.plots]
    su.close_plots()
    assert all(pi.ctrlMenu is None for pi in plot_items)


def test_close_plots_never_raises_with_nothing_assigned(qapp):
    """An unpopulated summary (self.stack is None) has nothing to close."""
    ChannelSummary().close_plots()
