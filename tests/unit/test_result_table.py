"""The shared DataFrame result-table model (ticket A04, UI-overhaul): header content
(identifier + unit, full-length tooltip), 6-sig-fig display with full-precision
tooltip/UserRole, alignment, and the table-scoped copy shortcut. Preview's per-breath
table and Run's averaged-metrics table both build on this model; GUI-integration
coverage for those two screens lives in test_preview_screen.py / test_run_screen.py."""
import pandas as pd
import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableView

from respmech.ui.result_table import (ResultTableModel, configure_result_table,
                                      install_copy_shortcut, resize_result_table)


def _df():
    return pd.DataFrame({
        "ptp_oesinsp": [169.96689123456, 55.5],
        "poes_tidal_swing": [9.1083368, 12.2],
        "breath_no": [1, 2],
        "file": ["synth_case_A.csv", "synth_case_B.csv"],
    })


# --------------------------------------------------------------------------- #
# header: identifier + unit, full tooltip
# --------------------------------------------------------------------------- #
def test_header_carries_identifier_then_unit_on_two_lines():
    model = ResultTableModel(_df())
    cols = {model.column_id(i): i for i in range(model.columnCount())}
    text = model.headerData(cols["ptp_oesinsp"], Qt.Orientation.Horizontal,
                            Qt.ItemDataRole.DisplayRole)
    assert text == "ptp_oesinsp\ncmH₂O·s·min⁻¹"


def test_header_keeps_a_blank_second_line_when_the_unit_is_unknown():
    model = ResultTableModel(_df())
    cols = {model.column_id(i): i for i in range(model.columnCount())}
    # breath_no deliberately resolves to no unit (respmech.core.quantities.unit_for)
    text = model.headerData(cols["breath_no"], Qt.Orientation.Horizontal,
                            Qt.ItemDataRole.DisplayRole)
    assert text == "breath_no\n"


def test_header_tooltip_carries_both_parts_in_full_even_when_the_label_would_elide():
    model = ResultTableModel(_df())
    cols = {model.column_id(i): i for i in range(model.columnCount())}
    tip = model.headerData(cols["ptp_oesinsp"], Qt.Orientation.Horizontal,
                           Qt.ItemDataRole.ToolTipRole)
    assert tip == "ptp_oesinsp (cmH₂O·s·min⁻¹)"
    tip_no_unit = model.headerData(cols["breath_no"], Qt.Orientation.Horizontal,
                                   Qt.ItemDataRole.ToolTipRole)
    assert tip_no_unit == "breath_no"          # no dangling empty parens


# --------------------------------------------------------------------------- #
# numeric formatting: 6 sig figs on screen, full precision in tooltip + UserRole
# --------------------------------------------------------------------------- #
def test_numeric_cells_show_six_significant_figures():
    model = ResultTableModel(_df())
    cols = {model.column_id(i): i for i in range(model.columnCount())}
    idx = model.index(0, cols["ptp_oesinsp"])
    # 169.96689123456 -> 6 sig figs, NOT .4g (which would round away digits a
    # researcher could be citing: 170, not 169.967)
    assert model.data(idx, Qt.ItemDataRole.DisplayRole) == "169.967"
    idx2 = model.index(0, cols["poes_tidal_swing"])
    assert model.data(idx2, Qt.ItemDataRole.DisplayRole) == "9.10834"


def test_numeric_cell_tooltip_and_userrole_keep_full_precision():
    model = ResultTableModel(_df())
    cols = {model.column_id(i): i for i in range(model.columnCount())}
    idx = model.index(0, cols["ptp_oesinsp"])
    raw = 169.96689123456
    assert model.data(idx, Qt.ItemDataRole.ToolTipRole) == str(raw)
    assert model.data(idx, Qt.ItemDataRole.UserRole) == raw
    # the truncated DisplayRole must not have replaced the underlying value
    assert model.data(idx, Qt.ItemDataRole.DisplayRole) != str(raw)


def test_file_column_strings_are_never_reformatted():
    model = ResultTableModel(_df())
    cols = {model.column_id(i): i for i in range(model.columnCount())}
    idx = model.index(0, cols["file"])
    assert model.data(idx, Qt.ItemDataRole.DisplayRole) == "synth_case_A.csv"
    assert model.data(idx, Qt.ItemDataRole.UserRole) == "synth_case_A.csv"


def test_numeric_cells_right_aligned_string_cells_left_aligned():
    model = ResultTableModel(_df())
    cols = {model.column_id(i): i for i in range(model.columnCount())}
    numeric = model.data(model.index(0, cols["ptp_oesinsp"]), Qt.ItemDataRole.TextAlignmentRole)
    stringy = model.data(model.index(0, cols["file"]), Qt.ItemDataRole.TextAlignmentRole)
    assert numeric & Qt.AlignmentFlag.AlignRight
    assert stringy & Qt.AlignmentFlag.AlignLeft


def test_integer_breath_number_is_still_treated_as_numeric():
    """breath_no is an int column; it must format/align like a number, not fall
    through the isinstance guard some other way."""
    model = ResultTableModel(_df())
    cols = {model.column_id(i): i for i in range(model.columnCount())}
    idx = model.index(0, cols["breath_no"])
    assert model.data(idx, Qt.ItemDataRole.DisplayRole) == "1"
    align = model.data(idx, Qt.ItemDataRole.TextAlignmentRole)
    assert align & Qt.AlignmentFlag.AlignRight


# --------------------------------------------------------------------------- #
# set_dataframe(None) clears the table (old setRowCount(0); setColumnCount(0))
# --------------------------------------------------------------------------- #
def test_set_dataframe_none_clears_rows_and_columns():
    model = ResultTableModel(_df())
    assert model.rowCount() > 0 and model.columnCount() > 0
    model.set_dataframe(None)
    assert model.rowCount() == 0 and model.columnCount() == 0


def test_set_dataframe_with_no_columns_also_clears():
    model = ResultTableModel(_df())
    model.set_dataframe(pd.DataFrame())
    assert model.rowCount() == 0 and model.columnCount() == 0


# --------------------------------------------------------------------------- #
# configure_result_table: read-only, selectable, header sized for two lines
# --------------------------------------------------------------------------- #
def test_configure_result_table_is_read_only_and_selectable(qapp):
    from PySide6.QtWidgets import QAbstractItemView
    view = QTableView()
    view.setModel(ResultTableModel(_df()))
    configure_result_table(view)
    assert view.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    assert view.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection


def test_configure_result_table_sizes_the_header_for_two_lines(qapp):
    view = QTableView()
    view.setModel(ResultTableModel(_df()))
    configure_result_table(view)
    header = view.horizontalHeader()
    fm = header.fontMetrics()
    # never a pixel literal (macOS/Windows font metrics differ): the header must be
    # tall enough for two lines of ITS OWN measured font, not a guessed constant
    assert header.height() >= 2 * fm.height()


def test_configure_result_table_caps_a_single_wide_column(qapp):
    view = QTableView()
    view.setModel(ResultTableModel(pd.DataFrame({"file": ["x" * 500]})))
    configure_result_table(view)
    assert view.horizontalHeader().maximumSectionSize() <= 220


# --------------------------------------------------------------------------- #
# copy shortcut: rectangular selection -> TSV on the clipboard
# --------------------------------------------------------------------------- #
def test_copy_shortcut_is_scoped_to_the_table_not_a_future_menu(qapp):
    """The copy slot must exist on the table itself, independent of any menu bar
    (see result_table.py's module docstring): a QShortcut with the platform Copy key
    sequence, scoped so it only fires while the table (or a child of it) has focus."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeySequence
    view = QTableView()
    view.setModel(ResultTableModel(_df()))
    sc = install_copy_shortcut(view)
    assert sc.key() == QKeySequence(QKeySequence.StandardKey.Copy)
    assert sc.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut
    assert sc.parent() is view


def test_copy_selection_writes_tsv_with_header_when_whole_columns_selected(qapp):
    view = QTableView()
    model = ResultTableModel(_df())
    view.setModel(model)
    configure_result_table(view)
    cols = {model.column_id(i): i for i in range(model.columnCount())}
    sel = view.selectionModel()
    top = model.index(0, cols["ptp_oesinsp"])
    bottom = model.index(1, cols["poes_tidal_swing"])
    from PySide6.QtCore import QItemSelection
    sel.select(QItemSelection(top, bottom), sel.SelectionFlag.ClearAndSelect)
    from respmech.ui.result_table import _copy_selection_as_tsv
    _copy_selection_as_tsv(view)
    from PySide6.QtGui import QGuiApplication
    text = QGuiApplication.clipboard().text()
    lines = text.split("\n")
    assert lines[0] == "ptp_oesinsp (cmH₂O·s·min⁻¹)\tpoes_tidal_swing (cmH₂O)"
    assert lines[1] == f"{169.96689123456}\t{9.1083368}"
    assert lines[2] == f"{55.5}\t{12.2}"


def test_copy_selection_omits_header_for_a_partial_row_range(qapp):
    view = QTableView()
    model = ResultTableModel(_df())
    view.setModel(model)
    configure_result_table(view)
    cols = {model.column_id(i): i for i in range(model.columnCount())}
    sel = view.selectionModel()
    # row 0 only, out of 2 -- not a whole column, so no header line
    only_cell = model.index(0, cols["ptp_oesinsp"])
    sel.select(only_cell, sel.SelectionFlag.ClearAndSelect)
    from respmech.ui.result_table import _copy_selection_as_tsv
    _copy_selection_as_tsv(view)
    from PySide6.QtGui import QGuiApplication
    text = QGuiApplication.clipboard().text()
    assert text == str(169.96689123456)


def test_copy_with_no_selection_does_not_touch_the_clipboard(qapp):
    from PySide6.QtGui import QGuiApplication
    QGuiApplication.clipboard().setText("sentinel")
    view = QTableView()
    view.setModel(ResultTableModel(_df()))
    configure_result_table(view)
    from respmech.ui.result_table import _copy_selection_as_tsv
    _copy_selection_as_tsv(view)               # no selection made
    assert QGuiApplication.clipboard().text() == "sentinel"


# --------------------------------------------------------------------------- #
# elision: assert on the RELATIONSHIP between measured text and actual column
# width, never on a pixel literal (macOS/Windows font metrics differ — see
# CLAUDE.md's layout-test notes)
# --------------------------------------------------------------------------- #
def test_no_header_line_or_numeric_cell_is_elided_at_the_columns_actual_width(qapp):
    df = pd.DataFrame({
        "poes_tidal_swing": [169.96689123456, 12.2],
        "pgas_tidal_swing": [1.234567, 3.4],
        "ptp_oesinsp": [55.555555, 88.1],
    })
    view = QTableView()
    model = ResultTableModel(df)
    view.setModel(model)
    configure_result_table(view)
    resize_result_table(view)
    header = view.horizontalHeader()
    fm_header = header.fontMetrics()
    fm_cell = view.fontMetrics()
    for c in range(model.columnCount()):
        width = header.sectionSize(c)
        id_line, unit_line = model.headerData(
            c, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole).split("\n", 1)
        # resizeColumnsToContents sizes >= the content it measured, modulo the
        # maximumSectionSize cap — none of this fixture's columns are anywhere near
        # that cap, so the section must already be wide enough for its own text
        assert fm_header.horizontalAdvance(id_line) <= width
        if unit_line:
            assert fm_header.horizontalAdvance(unit_line) <= width
        for r in range(model.rowCount()):
            text = model.data(model.index(r, c), Qt.ItemDataRole.DisplayRole)
            assert fm_cell.horizontalAdvance(text) <= width


# --------------------------------------------------------------------------- #
# self-review follow-ups: non-contiguous selection, missing values, negative
# numbers, the .6g scientific-notation boundary, reshape safety, no-model guard
# --------------------------------------------------------------------------- #
def test_copy_of_a_non_contiguous_selection_never_includes_unselected_cells(qapp):
    """A Ctrl-click selection of two diagonal cells is NOT a filled rectangle.
    Walking the bounding box and copying everything in it would silently paste
    values the user never selected — found in self-review, confirmed by
    reproducing it against the real model/view before this test was written."""
    from PySide6.QtCore import QItemSelectionModel
    from PySide6.QtGui import QGuiApplication
    from respmech.ui.result_table import _copy_selection_as_tsv
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    view = QTableView()
    model = ResultTableModel(df)
    view.setModel(model)
    configure_result_table(view)
    sel = view.selectionModel()
    # the two diagonal corners of the 3x3 grid — NOT a rectangle
    sel.select(model.index(0, 0), QItemSelectionModel.SelectionFlag.Select)
    sel.select(model.index(2, 2), QItemSelectionModel.SelectionFlag.Select)
    _copy_selection_as_tsv(view)
    text = QGuiApplication.clipboard().text()
    lines = text.split("\n")
    assert lines == ["1\t\t", "\t\t", "\t\t9"]
    # the untouched middle cells never leaked in
    for leaked in ("2", "3", "4", "5", "6", "7", "8"):
        assert leaked not in text


def test_copy_of_two_columns_with_an_unselected_column_between_them(qapp):
    """Selecting whole columns a and c but not b (a real, ordinary interaction —
    Ctrl-click two column headers) must not paste column b's values."""
    from PySide6.QtCore import QItemSelectionModel
    from PySide6.QtGui import QGuiApplication
    from respmech.ui.result_table import _copy_selection_as_tsv
    df = pd.DataFrame({"a": [1, 2], "b": [99, 98], "c": [3, 4]})
    view = QTableView()
    model = ResultTableModel(df)
    view.setModel(model)
    configure_result_table(view)
    sel = view.selectionModel()
    sel.select(model.index(0, 0), QItemSelectionModel.SelectionFlag.Select)
    sel.select(model.index(1, 0), QItemSelectionModel.SelectionFlag.Select)
    sel.select(model.index(0, 2), QItemSelectionModel.SelectionFlag.Select)
    sel.select(model.index(1, 2), QItemSelectionModel.SelectionFlag.Select)
    _copy_selection_as_tsv(view)
    text = QGuiApplication.clipboard().text()
    assert "99" not in text and "98" not in text
    lines = text.split("\n")
    assert lines[1:] == ["1\t\t3", "2\t\t4"]


def test_none_renders_blank_but_nan_stays_visible():
    """None looks like a missing value (blank), not the literal word 'None'. NaN
    stays visible as 'nan': the core deliberately writes NaN into some EMG columns
    when a detector is unreliable, and hiding it would hide that signal.

    A numeric-dtype column silently upgrades a stored ``None`` to NaN, and on
    this pandas version so does a plain ``[None, "a string"]`` list or a
    ``np.array(..., dtype=object)`` fed straight into ``pd.DataFrame`` — pandas
    infers its own string extension dtype and normalises ``None`` to NaN as
    part of that. An explicitly ``dtype=object`` ``pd.Series`` is what actually
    keeps a real ``None`` around to test against."""
    df = pd.DataFrame({"x": pd.Series([None, "not a number"], dtype=object)})
    model = ResultTableModel(df)
    none_idx = model.index(0, 0)
    assert model.data(none_idx, Qt.ItemDataRole.DisplayRole) == ""
    assert model.data(none_idx, Qt.ItemDataRole.ToolTipRole) == ""

    nan_model = ResultTableModel(pd.DataFrame({"x": [float("nan"), 1.5]}))
    nan_idx = nan_model.index(0, 0)
    assert nan_model.data(nan_idx, Qt.ItemDataRole.DisplayRole) == "nan"
    assert nan_model.data(nan_idx, Qt.ItemDataRole.ToolTipRole) == "nan"


def test_copy_renders_none_blank_and_nan_as_text(qapp):
    from PySide6.QtCore import QItemSelectionModel
    from PySide6.QtGui import QGuiApplication
    from respmech.ui.result_table import _copy_selection_as_tsv
    # a second row so selecting only row 0 is NOT a whole-column selection —
    # keeps this test focused on cell rendering, not the header-inclusion rule.
    # An explicitly dtype=object Series (see test_none_renders_blank_but_nan_
    # stays_visible for why) so the None actually survives into the model.
    df = pd.DataFrame({"note": pd.Series([None, "other"], dtype=object),
                       "value": [float("nan"), 2.0]})
    view = QTableView()
    model = ResultTableModel(df)
    view.setModel(model)
    configure_result_table(view)
    sel = view.selectionModel()
    from PySide6.QtCore import QItemSelection
    sel.select(QItemSelection(model.index(0, 0), model.index(0, 1)),
              QItemSelectionModel.SelectionFlag.ClearAndSelect)
    _copy_selection_as_tsv(view)
    assert QGuiApplication.clipboard().text() == "\tnan"


def test_negative_numbers_format_and_align_like_any_other_number():
    df = pd.DataFrame({"poes_mininsp": [-12.345678, -0.001]})
    model = ResultTableModel(df)
    idx = model.index(0, 0)
    assert model.data(idx, Qt.ItemDataRole.DisplayRole) == "-12.3457"
    align = model.data(idx, Qt.ItemDataRole.TextAlignmentRole)
    assert align & Qt.AlignmentFlag.AlignRight


def test_six_sig_fig_formatting_across_the_scientific_notation_boundary():
    """Pin the exact .6g behaviour at the boundaries the ticket's format choice
    implies, rather than leaving it to accident (self-review finding)."""
    df = pd.DataFrame({"x": [9999999.0, 1234567.89, 0.00001234, 0.0]})
    model = ResultTableModel(df)
    texts = [model.data(model.index(r, 0), Qt.ItemDataRole.DisplayRole) for r in range(4)]
    assert texts == ["1e+07", "1.23457e+06", "1.234e-05", "0"]


def test_reshaping_the_dataframe_clears_a_live_selection(qapp):
    """The two screens re-fill the SAME table with a different file's (differently
    shaped) result each time. beginResetModel()/endResetModel() must leave no
    stale, out-of-range selection behind for the next copy to trip over."""
    view = QTableView()
    model = ResultTableModel(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    view.setModel(model)
    configure_result_table(view)
    sel = view.selectionModel()
    sel.select(model.index(2, 1), sel.SelectionFlag.ClearAndSelect)
    assert sel.selectedIndexes()
    # a smaller shape: the previously-selected row/column no longer exist
    model.set_dataframe(pd.DataFrame({"a": [1]}))
    assert sel.selectedIndexes() == []
    # a copy right after a reshape must not raise even though the old selection
    # pointed past the new bounds
    from respmech.ui.result_table import _copy_selection_as_tsv
    _copy_selection_as_tsv(view)               # no selection survived -> no-op, no crash


def test_copy_shortcut_before_a_model_is_set_does_not_crash(qapp):
    from respmech.ui.result_table import _copy_selection_as_tsv
    view = QTableView()                         # setModel() never called
    _copy_selection_as_tsv(view)                # must return quietly, not raise


def test_a_real_pandas_boolean_column_is_not_formatted_as_a_number():
    """df.iat[...] on a bool-dtype column hands back numpy.bool_, not a literal
    Python bool — confirm the REAL DataFrame path (not just a Python literal)
    stays out of the numeric formatting/alignment branch."""
    df = pd.DataFrame({"excluded": [True, False]})
    model = ResultTableModel(df)
    idx = model.index(0, 0)
    assert model.data(idx, Qt.ItemDataRole.DisplayRole) == "True"
    align = model.data(idx, Qt.ItemDataRole.TextAlignmentRole)
    assert align & Qt.AlignmentFlag.AlignLeft
