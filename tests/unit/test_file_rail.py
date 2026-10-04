"""The file rail (ticket B02, UI-overhaul): one row per manifest file with its own
state, replacing Preview & QC's plain ``file_combo`` and Run & results' ``files_table``.
Pure widget-level tests against :class:`respmech.ui.file_rail.FileRail` — no
``MainWindow``/screen needed, since the widget carries no dependency on either."""
import shiboken6
from PySide6.QtCore import Qt

from respmech.ui.file_rail import FileRail
from respmech.ui.manifest import FileEntry, Manifest, manifest_from_filenames


def _manifest(names, *, folder="", outliers=(), freq_mismatch=()):
    """A manifest with some files INCLUDED and some marked as outliers/frequency
    mismatches, for exercising the rail's caveat rendering."""
    settings_fs = 1000 if freq_mismatch else None
    entries = []
    for n in names:
        if n in outliers:
            entries.append(FileEntry(path=n, filename=n, ext=".csv", columns=3,
                                     included=False, exclude_reason="3 columns (majority is 9)"))
        elif n in freq_mismatch:
            entries.append(FileEntry(path=n, filename=n, ext=".csv", columns=9,
                                     detected_fs=500, included=True))
        else:
            entries.append(FileEntry(path=n, filename=n, ext=".csv", columns=9,
                                     detected_fs=settings_fs or 1000, included=True))
    return Manifest(folder=folder, mask="*.csv", settings_fs=settings_fs, files=tuple(entries))


# --------------------------------------------------------------------------- #
# rows / manifest population
# --------------------------------------------------------------------------- #
def test_set_manifest_populates_rows_for_included_and_outlier_files(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv", "b.csv", "odd.csv"], outliers=["odd.csv"]))
    assert rail.filenames() == ["a.csv", "b.csv", "odd.csv"]
    assert rail.entry("odd.csv").caveat == "3 columns (majority is 9)"
    assert rail.entry("a.csv").caveat is None


def test_frequency_mismatch_is_a_caveat_on_an_included_file(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv", "b.csv"], freq_mismatch=["b.csv"]))
    assert rail.entry("b.csv").caveat is not None
    assert "500" in rail.entry("b.csv").caveat and "1000" in rail.entry("b.csv").caveat
    assert rail.entry("a.csv").caveat is None


def test_state_survives_a_manifest_rebuild_for_persisting_filenames(qapp):
    """A Setup edit that merely re-scans the same folder (e.g. widening the mask) must
    not forget what has already been previewed/run for a file that is still there."""
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv", "b.csv"]))
    rail.mark_result("a.csv", ok=True, breaths=7)
    rail.mark_seen("b.csv")
    rail.set_excluded_count("a.csv", 2)
    rail.set_manifest(_manifest(["a.csv", "b.csv", "c.csv"]))   # rebuild, one new file
    a = rail.entry("a.csv")
    assert a.verdict == "ok" and a.breaths == 7 and a.excluded_count == 2
    assert rail.entry("b.csv").seen is True
    assert rail.entry("c.csv").verdict == "unknown" and rail.entry("c.csv").seen is False


def test_manifest_none_clears_the_rail(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv"]))
    assert rail.count() == 1
    rail.set_manifest(None)
    assert rail.count() == 0
    assert rail.filenames() == []


# --------------------------------------------------------------------------- #
# identity / selection — mirrors QComboBox.currentTextChanged semantics
# --------------------------------------------------------------------------- #
def test_select_filename_emits_only_on_an_actual_change(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv", "b.csv"]))   # quietly adopts "a.csv" — no emit yet
    seen = []
    rail.selectionChanged.connect(seen.append)
    rail.select_filename("a.csv")           # same identity as the quiet adoption -> no emit
    assert seen == []
    rail.select_filename("b.csv")
    assert seen == ["b.csv"]
    rail.select_filename("b.csv")           # same identity again -> no second emit
    assert seen == ["b.csv"]


def test_set_manifest_quietly_adopts_the_first_file_without_emitting(qapp):
    """Mirrors the old file_combo: populating it auto-selected index 0 as a bare Qt side
    effect, never through currentTextChanged (its own populate ran under blockSignals) —
    so nothing downstream ever reacted to a freshly built screen's very first file list.
    A caller that DOES need to react to 'first file of a fresh rail' reads
    current_filename() itself; selectionChanged only ever reports an actual switch."""
    rail = FileRail()
    seen = []
    rail.selectionChanged.connect(seen.append)
    rail.set_manifest(_manifest(["a.csv", "b.csv"]))
    assert rail.current_filename() == "a.csv"
    assert seen == []
    # a SUBSEQUENT rebuild that still contains the current file must not re-adopt or emit
    rail.set_manifest(_manifest(["a.csv", "b.csv", "c.csv"]))
    assert rail.current_filename() == "a.csv"
    assert seen == []


def test_select_filename_works_even_when_the_row_does_not_exist(qapp):
    """Ticket requirement: the rail's identity is not gated on a matching manifest row —
    a caller (Setup's dirty-toggle test, or a cross-screen jump before the rail has been
    populated) can still set/read an identity that has no row yet."""
    rail = FileRail()
    rail.select_filename("ghost.csv")
    assert rail.current_filename() == "ghost.csv"


def test_step_clamps_at_both_ends(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv", "b.csv", "c.csv"]))
    rail.select_index(0)
    rail.step(+1); assert rail.current_filename() == "b.csv"
    rail.step(+1); rail.step(+1)                    # clamps at the end
    assert rail.current_filename() == "c.csv"
    rail.step(-1); assert rail.current_filename() == "b.csv"


def test_select_index_out_of_range_is_a_no_op(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv"]))
    rail.select_index(0)
    rail.select_index(5)
    assert rail.current_filename() == "a.csv"


# --------------------------------------------------------------------------- #
# filter — must never touch the current selection
# --------------------------------------------------------------------------- #
def test_filter_hides_rows_without_changing_the_selection(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["alpha.csv", "beta.csv", "gamma.csv"]))
    rail.select_filename("alpha.csv")
    seen = []
    rail.selectionChanged.connect(seen.append)
    rail.filter_edit.setText("zzz-no-match")
    assert rail.visible_filenames() == []
    assert rail.current_filename() == "alpha.csv"     # identity untouched
    assert seen == []                                 # a partial/no-match filter never fires
    rail.filter_edit.setText("beta")
    assert rail.visible_filenames() == ["beta.csv"]
    assert rail.current_filename() == "alpha.csv"      # still untouched — filtering never selects
    assert seen == []


def test_filter_is_case_insensitive_contains_match(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["P08_r.csv", "P12_r.csv"]))
    rail.filter_edit.setText("08_R")
    assert rail.visible_filenames() == ["P08_r.csv"]


# --------------------------------------------------------------------------- #
# per-file state -> visible without selecting the file first
# --------------------------------------------------------------------------- #
def test_exclusion_badge_visible_without_selecting_the_file(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv", "b.csv"]))
    rail.set_excluded_count("b.csv", 3)
    assert rail.current_filename() != "b.csv"          # never selected
    assert rail.entry("b.csv").excluded_count == 3
    text = rail._model.data(rail._model.index(1))     # the row's DisplayRole text
    assert "3" in text and "excl" in text


def test_carried_over_exclusion_gets_its_own_glyph_and_tooltip_note(qapp):
    """Ticket B06: a filename's exclusion recorded against a DIFFERENT recordings folder
    reads differently from an ordinary one — the row text carries a distinct marker (not
    just the same '[N excl]') and the tooltip names why."""
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv", "b.csv"]))
    rail.set_excluded_count("a.csv", 2, carried=False)
    rail.set_excluded_count("b.csv", 2, carried=True)
    plain = rail._model.data(rail._model.index(0))
    carried = rail._model.data(rail._model.index(1))
    assert "2 excl" in plain and "2 excl" in carried
    assert plain != carried                              # visibly distinguishable
    tip_plain = rail._model.data(rail._model.index(0), role=Qt.ToolTipRole)
    tip_carried = rail._model.data(rail._model.index(1), role=Qt.ToolTipRole)
    assert "carried over" not in tip_plain
    assert "carried over" in tip_carried


def test_carried_flag_survives_a_manifest_rebuild_for_persisting_filenames(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv"]))
    rail.set_excluded_count("a.csv", 2, carried=True)
    rail.set_manifest(_manifest(["a.csv", "b.csv"]))
    assert rail.entry("a.csv").excluded_carried is True


def test_carried_defaults_to_false_and_a_bare_count_update_does_not_reset_it():
    """set_excluded_count is called from more than one place (a toggle, a rail sync);
    a caller that only wants to update the count and does not pass carried should not
    silently CLEAR a carried flag another caller just set for the same file."""
    from respmech.ui.file_rail import FileRailModel
    model = FileRailModel()
    model.set_manifest(_manifest(["a.csv"]))
    model.set_excluded_count("a.csv", 2, carried=True)
    assert model.entry("a.csv").excluded_carried is True
    # explicit re-affirmation, same as _sync_rail_exclusions does every time it runs
    model.set_excluded_count("a.csv", 2, carried=True)
    assert model.entry("a.csv").excluded_carried is True
    # an explicit carried=False (a confirming click — see _toggle_breath) does reset it
    model.set_excluded_count("a.csv", 2, carried=False)
    assert model.entry("a.csv").excluded_carried is False


def test_mark_result_updates_verdict_and_breaths(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv"]))
    rail.mark_result("a.csv", ok=True, breaths=9)
    e = rail.entry("a.csv")
    assert e.verdict == "ok" and e.breaths == 9 and e.error is None
    rail.mark_result("a.csv", ok=False, error="TrimError: boom")
    e = rail.entry("a.csv")
    assert e.verdict == "failed" and e.breaths is None and "boom" in e.error


def test_mark_result_on_a_missing_filename_is_a_silent_no_op(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv"]))
    rail.mark_result("does_not_exist.csv", ok=True, breaths=1)   # must not raise
    assert rail.entry("does_not_exist.csv") is None


# --------------------------------------------------------------------------- #
# failed-first sort — reach the one failure in a large batch at a glance
# --------------------------------------------------------------------------- #
def test_sort_failed_first_brings_failures_to_the_top(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest([f"f{i}.csv" for i in range(1, 6)]))
    rail.mark_result("f4.csv", ok=False, error="boom")
    rail.sort_failed_first(True)
    assert rail.visible_filenames()[0] == "f4.csv"
    rail.sort_failed_first(False)
    assert rail.visible_filenames() == [f"f{i}.csv" for i in range(1, 6)]   # back to manifest order


def test_sort_failed_first_composes_with_the_filter(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a1.csv", "a2.csv", "b1.csv"]))
    rail.mark_result("a2.csv", ok=False, error="boom")
    rail.sort_failed_first(True)
    rail.filter_edit.setText("a")
    assert rail.visible_filenames() == ["a2.csv", "a1.csv"]


# --------------------------------------------------------------------------- #
# double-click activation — distinct from a plain selection
# --------------------------------------------------------------------------- #
def test_double_click_emits_file_activated_not_selection_changed_alone(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv", "b.csv"]))
    activated = []
    rail.fileActivated.connect(activated.append)
    idx = rail._find_proxy_row("b.csv")
    rail._on_double_clicked(rail._proxy.index(idx, 0))
    assert activated == ["b.csv"]


# --------------------------------------------------------------------------- #
# manifest_from_filenames — the caveat-free manifest RunScreen builds
# --------------------------------------------------------------------------- #
def test_manifest_from_filenames_has_no_caveats():
    m = manifest_from_filenames("/data", ["/data/a.csv", "/data/b.csv"])
    assert [f.filename for f in m.files] == ["a.csv", "b.csv"]
    assert m.outliers == () and m.freq_mismatches == ()
    assert all(f.included for f in m.files)


# --------------------------------------------------------------------------- #
# typed manoeuvres / reference / segments / role (M-32)
# --------------------------------------------------------------------------- #
def test_typed_glyph_and_tooltip_breakdown(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv", "b.csv"]))
    rail.set_typed_state("b.csv", {"ic": 2, "fvc": 1})
    text = rail._model.data(rail._model.index(1))
    assert "◆" in text
    plain_text = rail._model.data(rail._model.index(0))
    assert "◆" not in plain_text                        # untyped file gets no glyph
    tip = rail._model.data(rail._model.index(1), role=Qt.ToolTipRole)
    assert "3 breaths typed as a manoeuvre" in tip
    assert "ic ×2" in tip and "fvc ×1" in tip


def test_typed_carried_flag_gets_its_own_tooltip_note(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv", "b.csv"]))
    rail.set_typed_state("a.csv", {"ic": 1}, carried=False)
    rail.set_typed_state("b.csv", {"ic": 1}, carried=True)
    tip_plain = rail._model.data(rail._model.index(0), role=Qt.ToolTipRole)
    tip_carried = rail._model.data(rail._model.index(1), role=Qt.ToolTipRole)
    assert "carried over" not in tip_plain
    assert "carried over" in tip_carried


def test_typed_state_survives_a_manifest_rebuild_for_persisting_filenames(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv"]))
    rail.set_typed_state("a.csv", {"ic": 2}, carried=True)
    rail.set_manifest(_manifest(["a.csv", "b.csv"]))
    e = rail.entry("a.csv")
    assert e.typed_counts == {"ic": 2} and e.typed_carried is True


def test_typed_state_can_be_cleared_back_to_empty(qapp):
    """A second analysis over the same folder/mask that types nothing for a file the
    FIRST analysis had typed breaths in must not leave the first analysis's stale glyph
    on screen — same zeroing requirement B06 established for exclusions."""
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv"]))
    rail.set_typed_state("a.csv", {"ic": 1})
    assert rail.entry("a.csv").typed_counts
    rail.set_typed_state("a.csv", {})
    assert rail.entry("a.csv").typed_counts == {}
    assert "◆" not in rail._model.data(rail._model.index(0))


def test_reference_glyphs_linked_and_missing(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv", "b.csv", "c.csv"]))
    rail.set_reference("a.csv", "linked")
    rail.set_reference("b.csv", "missing")
    text_a = rail._model.data(rail._model.index(0))
    text_b = rail._model.data(rail._model.index(1))
    text_c = rail._model.data(rail._model.index(2))
    assert "⇢" in text_a and "⇢?" not in text_a
    assert "⇢?" in text_b
    assert "⇢" not in text_c and "⇢?" not in text_c   # unresolved (None) shows nothing
    tip_b = rail._model.data(rail._model.index(1), role=Qt.ToolTipRole)
    assert "not resolved" in tip_b


def test_segments_count_shown_in_tooltip_but_not_as_a_row_glyph(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv"]))
    rail.set_segments("a.csv", 3)
    text = rail._model.data(rail._model.index(0))
    assert "3" not in text                               # no dedicated row badge (ticket scope)
    tip = rail._model.data(rail._model.index(0), role=Qt.ToolTipRole)
    assert "3 EMG segments" in tip


def test_segments_singular_grammar_at_exactly_one(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv"]))
    rail.set_segments("a.csv", 1)
    tip = rail._model.data(rail._model.index(0), role=Qt.ToolTipRole)
    assert "1 EMG segment" in tip and "1 EMG segments" not in tip


def test_tooltip_combines_typed_excluded_and_caveat_all_at_once(qapp):
    """The real-world worst case for row formatting: a file that is simultaneously
    typed, excluded and manifest-flagged. Every line must be present in the tooltip
    regardless of the others."""
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv"], outliers=["a.csv"]))
    rail.set_typed_state("a.csv", {"ic": 1})
    rail.set_excluded_count("a.csv", 2)
    tip = rail._model.data(rail._model.index(0), role=Qt.ToolTipRole)
    assert "1 breath typed as a manoeuvre (ic ×1)" in tip
    assert "2 breaths manually excluded" in tip
    assert "⚠" in tip and rail.entry("a.csv").caveat in tip


def test_role_is_set_by_mark_result_and_cleared_on_failure(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv"]))
    rail.mark_result("a.csv", ok=True, breaths=0, role="reference")
    e = rail.entry("a.csv")
    assert e.role == "reference"
    tip = rail._model.data(rail._model.index(0), role=Qt.ToolTipRole)
    assert "Reference manoeuvres only" in tip
    rail.mark_result("a.csv", ok=False, error="boom")
    assert rail.entry("a.csv").role is None              # cleared like breaths is on failure


def test_typed_reference_segments_role_default_to_empty_none(qapp):
    """A plain, untouched row must not accidentally render any M-32 badge."""
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv"]))
    e = rail.entry("a.csv")
    assert e.typed_counts == {} and e.typed_carried is False
    assert e.reference is None and e.segments is None and e.role is None
    text = rail._model.data(rail._model.index(0))
    assert "◆" not in text and "⇢" not in text


# --------------------------------------------------------------------------- #
# eliding delegate — a long filename must never push a state glyph off-screen
# --------------------------------------------------------------------------- #
def test_a_long_filename_with_every_badge_fits_the_rail_on_windows_metrics(windows_metrics):
    """M-32's own acceptance criterion: a long filename with every badge active either
    fits the 280 px rail or is elided, but the state glyphs/badges are never the part
    that gets cut. Modelled on the Windows runner's wider font metrics (macOS is the
    friendliest platform we ship to — see tests/CLAUDE.md)."""
    from PySide6.QtGui import QFontMetrics

    from respmech.ui.file_rail import (_RAIL_ITEM_PADDING_PX, FileRailEntry,
                                       _elided_row_text, _row_text)

    long_name = "a_very_long_synthetic_recording_name_32c.csv"   # 32+ characters
    assert len(long_name) >= 32
    e = FileRailEntry(filename=long_name, verdict="ok", typed_counts={"ic": 2},
                      reference="missing", excluded_count=3, excluded_carried=True,
                      caveat="detected 500 Hz sampling — settings say 1000 Hz")
    fm = QFontMetrics(windows_metrics.font())
    avail = 280 - _RAIL_ITEM_PADDING_PX
    full = _row_text(e)
    elided = _elided_row_text(e, fm, avail)

    # every glyph/badge is present, whether or not the filename needed shortening
    for badge in ("◆", "⇢?", "[3 excl ↺]", "⚠"):
        assert badge in elided, f"{badge!r} missing from {elided!r}"
    # the filename itself was actually shortened under this budget (fixed segments alone
    # eat most of a 280 px rail once every badge is active)
    assert fm.horizontalAdvance(elided) < fm.horizontalAdvance(full)
    assert fm.horizontalAdvance(elided) <= avail + 2       # +2px rounding slack


def test_a_short_filename_with_no_badges_is_not_elided(qapp):
    """The common case (a short name, nothing active) must round-trip unchanged — the
    delegate must not shorten a filename that already fits."""
    from PySide6.QtGui import QFontMetrics

    from respmech.ui.file_rail import FileRailEntry, _elided_row_text, _row_text

    e = FileRailEntry(filename="a.csv", verdict="ok")
    fm = QFontMetrics(qapp.font())
    elided = _elided_row_text(e, fm, 280)
    assert elided == _row_text(e)


def test_existing_rail_badges_are_unaffected_by_the_m32_fields(qapp):
    """Acceptance criterion: the pre-M-32 exclusion badge is unchanged in shape."""
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv"]))
    rail.set_excluded_count("a.csv", 2, carried=True)
    text = rail._model.data(rail._model.index(0))
    assert "[2 excl ↺]" in text


# --------------------------------------------------------------------------- #
# row context menu — plumbing for M-37 (M-32)
# --------------------------------------------------------------------------- #
def test_row_context_menu_emits_references_requested_for_the_right_row(qapp):
    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv", "b.csv"]))
    requested = []
    rail.referencesRequested.connect(requested.append)
    menu = rail._build_row_context_menu("b.csv")
    actions = menu.actions()
    assert len(actions) == 1
    assert "Reference manoeuvres" in actions[0].text()
    actions[0].trigger()
    assert requested == ["b.csv"]
    # QMenu is a Window (Qt::Popup) even when parented, so it is its own entry in
    # QApplication.topLevelWidgets(). A plain close() (the pattern the sibling
    # _build_type_menu tests use, test_breath_typing_ui.py) schedules a deleteLater()
    # that depends on the event loop draining it before the NEXT test's window-reaping
    # teardown scans topLevelWidgets() — here the triggered action's own lambda closure
    # over `rail` (needed to bind the right filename) keeps the whole rail alive via a
    # genuine Python reference cycle (view -> connected bound methods -> rail), so the
    # cycle is only broken by gc's cyclic collector rather than plain refcounting, and
    # collecting a still-parented top-level QObject mid-cycle is exactly the known py3.11
    # "mid-destruction pointer" hazard conftest.py's own comment documents. Deleting the
    # C++ object immediately and deterministically (never queued, no GC-timing
    # dependency) sidesteps it entirely.
    shiboken6.delete(menu)


def test_delegate_prepare_option_disables_style_elision_and_protects_badges(qapp):
    """Exercises the REAL delegate code path (``_RailItemDelegate._prepare_option``, the
    method ``paint()`` itself calls), not just the standalone ``_elided_row_text`` pure
    function the other elision tests use — proves the delegate forces
    ``Qt.TextElideMode.ElideNone`` so the STYLE can never re-elide (and potentially eat a
    badge) on top of the already-correct manual eliding."""
    from PySide6.QtWidgets import QStyleOptionViewItem

    rail = FileRail()
    rail.set_manifest(_manifest(["a_very_long_synthetic_recording_name_32c.csv"]))
    rail.set_typed_state("a_very_long_synthetic_recording_name_32c.csv", {"ic": 2})
    rail.set_excluded_count("a_very_long_synthetic_recording_name_32c.csv", 3, carried=True)
    index = rail._proxy.index(0, 0)

    base = QStyleOptionViewItem()
    base.rect.setWidth(280)
    opt = rail._delegate._prepare_option(base, index)

    assert opt.textElideMode == Qt.TextElideMode.ElideNone
    assert "◆" in opt.text                          # ◆ typed glyph
    assert "[3 excl ↺]" in opt.text                  # exclusion badge + carried mark


def test_context_menu_at_an_invalid_position_never_emits(qapp):
    """A right-click below the last row (or on an empty rail) must not pop up a menu at
    all — _on_context_menu's own indexAt()/name guards, not just the pure menu-builder
    the test above exercises."""
    from PySide6.QtCore import QPoint

    rail = FileRail()
    rail.set_manifest(_manifest(["a.csv"]))
    requested = []
    rail.referencesRequested.connect(requested.append)
    rail._on_context_menu(QPoint(5, 10_000))    # far below any real row
    assert requested == []
