"""MigrationReportDialog (ticket D16) — structured presentation of a MigrationReport:
word wrap on, proportional font, real headings instead of literal '#'/'##', the
behaviour-changing 'Normalised' items promoted next to the intro, the pure renames
('Mapped'/'Dropped') behind a disclosure, and a 'Save report…' button that writes the
raw, machine-readable MigrationReport.text() untouched.
"""
from respmech.settingsio.migrate import MigrationReport
from respmech.ui.migration_report_dialog import MigrationReportDialog, open_migration_report


def _report(mapped=None, normalised=None, dropped=None):
    return MigrationReport(mapped=mapped or [], normalised=normalised or [],
                           dropped=dropped or [])


def test_intro_is_plain_prose_not_markdown(qapp):
    dlg = MigrationReportDialog(_report(normalised=["emg.remove_noise -> processing.emg.noise "
                                                    "(fixed n_fft=256, expiration-based reference)"]))
    assert "#" not in dlg.windowTitle()
    # the intro paragraph is a QLabel, not a monospace view — no literal markdown heading
    assert dlg.findChild(type(dlg.save_btn)) is not None       # sanity: dialog actually built


def test_intro_states_the_consequence_not_just_the_mechanics(qapp):
    """Ticket D16's own acceptance example: the intro must say results will NOT exactly
    reproduce 1.x numbers when something was normalised (drift-fixed) — not merely
    describe what technically changed. A 'Normalised' entry is BY DEFINITION a
    behaviour fix (migrate.py's own report header: 'Normalised (drift fixed)'), so this
    holds for any non-empty normalised list, not only the EMG-noise example."""
    dlg = MigrationReportDialog(_report(normalised=["processing.volume.trend_peak_min_height "
                                                    "unset -> troughs detected differently"]))
    intro = dlg.layout().itemAt(0).widget().text()
    assert "not exactly reproduce" in intro or "will not" in intro
    assert "1.x" in intro
    # a purely-renamed import (nothing normalised) gets the opposite, equally explicit
    # claim — reproduces exactly — not silence on the question
    dlg2 = MigrationReportDialog(_report(mapped=["a.b -> c.d"]))
    intro2 = dlg2.layout().itemAt(0).widget().text()
    assert "reproduces" in intro2 and "1.x" in intro2


def test_normalised_items_are_shown_upfront_not_behind_the_disclosure(qapp):
    dlg = MigrationReportDialog(_report(
        mapped=["a.b -> c.d"],
        normalised=["emg.remove_noise -> processing.emg.noise (fixed n_fft=256, "
                    "expiration-based reference for a stable estimate)"],
        dropped=["x.y (never read by the code)"]))
    # collect all currently-visible label text (i.e. NOT inside the collapsed scroll area)
    from PySide6.QtWidgets import QLabel
    visible_text = " ".join(l.text() for l in dlg.findChildren(QLabel) if l.isVisibleTo(dlg))
    assert "fixed n_fft=256" in visible_text
    assert "expiration-based reference" in visible_text
    # the pure renames are NOT part of the upfront visible text — they are inside the
    # collapsed scroll area, which starts hidden
    assert "a.b -> c.d" not in visible_text
    assert "never read by the code" not in visible_text
    assert dlg._scroll.isVisibleTo(dlg) is False


def test_toggle_reveals_the_full_field_by_field_mapping(qapp):
    """isVisibleTo(dlg), not isVisible(): the dialog is never shown() in this headless
    test, so isVisible() (which also asks whether the whole ancestor chain is on
    screen) would read False regardless of the toggle — see TextViewerDialog's own
    collapsed-detail test for the same convention."""
    dlg = MigrationReportDialog(_report(mapped=["a.b -> c.d"], dropped=["x.y (unused)"]))
    assert dlg._scroll.isVisibleTo(dlg) is False
    dlg._toggle_btn.setChecked(True)
    assert dlg._scroll.isVisibleTo(dlg) is True
    from PySide6.QtWidgets import QLabel
    full_text = " ".join(l.text() for l in dlg._scroll.widget().findChildren(QLabel))
    assert "a.b -> c.d" in full_text
    assert "x.y (unused)" in full_text
    dlg._toggle_btn.setChecked(False)
    assert dlg._scroll.isVisibleTo(dlg) is False


def _heading_texts(dlg):
    """Headings use theme.py's role="heading" QSS property (not hand-rolled bold, which
    breaks on a platform where QFont.pointSize() returns -1 — see migration_report_dialog
    _heading()'s own comment) — so headings are found by that property, matching how the
    rest of the codebase (about_dialog.py, run_screen.py, ...) marks a heading."""
    from PySide6.QtWidgets import QLabel
    return [l.text() for l in dlg.findChildren(QLabel) if l.property("role") == "heading"]


def test_three_sections_render_as_real_headings_not_literal_hashes(qapp):
    dlg = MigrationReportDialog(_report(mapped=["a"], normalised=["b"], dropped=["c"]))
    dlg._toggle_btn.setChecked(True)
    headings = _heading_texts(dlg)
    assert "Normalised (drift fixed)" in headings
    assert "Mapped (renamed/moved)" in headings
    assert "Dropped (not used by the code)" in headings
    assert not any(h.startswith("#") for h in headings)   # no literal markdown


def test_empty_normalised_section_is_not_shown_and_intro_says_all_renames(qapp):
    dlg = MigrationReportDialog(_report(mapped=["a.b -> c.d"]))
    assert "Normalised (drift fixed)" not in _heading_texts(dlg)
    assert "straightforward rename" in dlg.layout().itemAt(0).widget().text()


def test_save_report_writes_the_raw_machine_readable_text(qapp, tmp_path, monkeypatch):
    """MigrationReport.text() itself is untouched — it is the machine-readable listing
    the ticket explicitly says must stay as-is; the dialog's Save button writes exactly
    that, not a re-rendering of the dialog's own prose."""
    from PySide6.QtWidgets import QFileDialog
    report = _report(mapped=["a.b -> c.d"], normalised=["e.f changed"], dropped=["g.h"])
    dlg = MigrationReportDialog(report)
    target = tmp_path / "saved-report.txt"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    dlg._save()
    assert target.read_text(encoding="utf-8") == report.text()
    assert target.read_text(encoding="utf-8").startswith("# Settings migration report")


def test_save_report_remembers_its_own_sticky_folder_separately_from_analysis(
        qapp, tmp_path, isolated_prefs, monkeypatch):
    """Ticket D16 warns against blending this Save with the ANALYSIS Save-as… default —
    this dialog gets its OWN P26-style sticky folder ('migration_report'), which must
    stay independent of the 'analysis' bucket every other Open/Save dialog uses.

    prefs.last_folder() only honours a stored path that still IS a real directory
    (see prefs.py), so both directories are created for real, not just named."""
    from PySide6.QtWidgets import QFileDialog
    unrelated = tmp_path / "unrelated_analysis_dir"
    unrelated.mkdir()
    picked = tmp_path / "picked"
    picked.mkdir()
    isolated_prefs.set_last_folder("analysis", str(unrelated))
    seen = {}

    def _fake_save(*a, **k):
        seen["start"] = a[2] if len(a) > 2 else k.get("dir")
        return (str(picked / "report.txt"), "")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(_fake_save))
    dlg = MigrationReportDialog(_report(mapped=["a.b -> c.d"]))
    dlg._save()
    assert str(unrelated) not in seen["start"]
    assert isolated_prefs.last_folder("migration_report", "") == str(picked)
    assert isolated_prefs.last_folder("analysis", "") == str(unrelated)


def test_headings_and_bullets_are_selectable_for_copying_one_line(qapp):
    """Minor polish (self-review finding): about_dialog.py's convention is that any text
    a user might want to copy without using the dedicated Save/Copy button is mouse-
    selectable. The report's headings/bullets/intro should be too."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel
    dlg = MigrationReportDialog(_report(mapped=["a.b -> c.d"], normalised=["e.f changed"]))
    dlg._toggle_btn.setChecked(True)
    for lab in dlg.findChildren(QLabel):
        if lab.text().strip():
            assert lab.textInteractionFlags() & Qt.TextSelectableByMouse


def test_save_report_cancelled_writes_nothing(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog
    dlg = MigrationReportDialog(_report(mapped=["a.b -> c.d"]))
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    dlg._save()   # must not raise, must not create a file anywhere new
    assert list(tmp_path.iterdir()) == []


def test_open_migration_report_replaces_a_prior_dialog_not_accumulate(qapp):
    """Same 'replace, don't accumulate windows' convention as dialogs.open_error_dialog."""
    first = open_migration_report(None, _report(mapped=["a"]))
    second = open_migration_report(None, _report(mapped=["b"]), prior=first)
    assert first is not second
    assert not first.isVisible()          # the old one was closed


def test_import_legacy_returns_a_migration_report_object(qapp, tmp_path):
    """Ticket D16: AppState.import_legacy used to return report.text() (a flat string);
    the UI now needs the structured mapped/normalised/dropped lists to build its own
    dialog, so it returns the MigrationReport object itself. report.text() is still
    reachable (and unchanged) via the object's own .text() method."""
    from respmech.ui.state import AppState
    legacy = tmp_path / "legacy_setup.py"
    legacy.write_text(
        "settings = {\n"
        "  'input': {'inputfolder': 'in', 'files': '*.csv',\n"
        "    'format': {'samplingfrequency': 1000},\n"
        "    'data': {'column_poes':7,'column_pgas':8,'column_pdi':9,'column_volume':6,\n"
        "             'column_flow':5,'columns_emg':[2,3,4],'columns_entropy':[]}},\n"
        "  'processing': {'mechanics': {'breathseparationbuffer':200,'separateby':'flow'},\n"
        "                 'emg': {'remove_ecg': False, 'remove_noise': False}},\n"
        "  'output': {'outputfolder': 'out',\n"
        "    'data': {'saveaveragedata': True, 'savebreathbybreathdata': True}}\n}\n")
    st = AppState()
    report = st.import_legacy(str(legacy))
    assert isinstance(report, MigrationReport)
    assert report.mapped                                  # at least the renames happened
    assert report.text().startswith("# Settings migration report")
