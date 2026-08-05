"""The file rail (ticket B02, UI-overhaul): one row per manifest file with its own
state, replacing Preview & QC's plain ``file_combo`` and Run & results' ``files_table``.
Pure widget-level tests against :class:`respmech.ui.file_rail.FileRail` — no
``MainWindow``/screen needed, since the widget carries no dependency on either."""
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
