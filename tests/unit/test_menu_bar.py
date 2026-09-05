"""The desktop menu bar (C01, UI-overhaul): a real ``QMenuBar`` (File/View/Help)
alongside the header's existing Analysis button, sharing the SAME QAction objects so
neither door can drift out of sync with the other, plus the offline About box.

QTest cannot fire a keyboard shortcut in this (offscreen) environment — a probe action
with Ctrl+S bound zero times whether added only to a menu, via ``win.addAction``, or in
a real ``QMenuBar`` (measured while building this ticket). These tests therefore assert
REGISTRATION (the right ``QAction`` exists, carries the right ``shortcut()``, and is
literally the same object the header already uses) rather than delivery; the actual
keypress behaviour is verified by hand on macOS and noted in the ticket's Worklog.

NB when introspecting a QMenuBar/QMenu from a test: bind ``menu.actions()`` to a local
before indexing/chaining (``acts = menu.actions(); acts[0]…``). Chaining
``menu.actions()[0].menu()`` in one expression was observed to raise
"Internal C++ object (QMenu) already deleted" — a PySide6/shiboken6 wrapper-lifetime
quirk over the temporary list, not a defect in the menu itself (confirmed valid via
``shiboken6.isValid`` either way). Production code never does this chaining; it holds
``self._file_menu``/``self._view_menu``/``self._help_menu`` as real attributes.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel

from respmech.ui.state import AppState
from respmech.ui.main_window import MainWindow
from respmech.ui.about_dialog import AboutDialog
from respmech import __version__

from _helpers import INPUT, requires_synth, synth_settings  # noqa: F401


def _make_win():
    return MainWindow(AppState())


def test_menu_bar_has_file_view_help_with_expected_shortcuts(qapp):
    win = _make_win()
    top = [a.text() for a in win.menuBar().actions()]
    assert top == ["&File", "&View", "&Help"]

    file_labels = [a.text() for a in win._file_menu.actions()]
    assert "New analysis" in file_labels
    assert "Open analysis…" in file_labels
    assert "Open Recent" in file_labels
    assert "Save" in file_labels
    assert "Save as…" in file_labels
    assert "Open output folder" in file_labels
    assert "Close Window" in file_labels

    assert win._act_new.shortcut().toString() == "Ctrl+N"
    assert win._act_open.shortcut().toString() == "Ctrl+O"
    assert win._act_save.shortcut().toString() == "Ctrl+S"
    assert win._act_save_as.shortcut().toString() == "Ctrl+Shift+S"
    assert win._act_close_window.shortcut().toString() == "Ctrl+W"

    view_labels = [a.text() for a in win._view_menu.actions()]
    # Run & results stopped being its own tab in B03 (folded into a Preview & QC
    # drawer) — the menu is built from the ACTUAL tabs, so there are two, not three.
    assert view_labels == ["Setup", "Preview && QC"]
    assert win._view_actions[0].shortcut().toString() == "Ctrl+1"
    assert win._view_actions[1].shortcut().toString() == "Ctrl+2"

    help_labels = [a.text() for a in win._help_menu.actions()]
    assert help_labels == ["RespMech documentation", "RespMech website",
                           "Report an issue", "", "About RespMech…"]
    win.close()


def test_menu_bar_and_header_share_the_same_action_objects(qapp):
    """The header's Analysis menu and the File menu must never be able to disagree —
    they are the exact same QAction, added to two containers."""
    win = _make_win()
    file_acts = win._file_menu.actions()
    header_acts = win.analysis_btn.menu().actions()
    for act in (win._act_new, win._act_open, win._act_save, win._act_save_as):
        assert act in file_acts
        assert act in header_acts
    win.close()


def test_save_enabled_state_is_identical_on_both_menus(qapp, tmp_path):
    """Save's enabled state is driven by the shared SettingsScreen logic — disabled
    with nothing to save, enabled for a new-but-valid analysis (Save falls through to
    Save as…), and disabled again once an actually-saved analysis is clean. Both menus
    read the same action, so this only needs checking once."""
    win = _make_win()
    win._refresh_analysis_menu()
    assert win._act_save.isEnabled() is False   # a fresh, empty analysis has nothing to save

    sc = win.settings_screen
    sc.state.settings = synth_settings(str(tmp_path))
    sc.from_state()
    sc._mark_dirty()
    win._refresh_analysis_menu()
    # never-saved (no settings_path yet) -> enabled regardless of dirty, per
    # _refresh_analysis_menu's own docstring ("Save is offered when the analysis is NEW…")
    assert win._act_save.isEnabled() is True
    assert win._act_save in win._file_menu.actions()
    assert win._act_save in win.analysis_btn.menu().actions()

    p = str(tmp_path / "saved.toml")
    assert sc._write_analysis(p) is True        # gives the analysis a real settings_path
    win._refresh_analysis_menu()
    assert win._act_save.isEnabled() is False    # saved AND clean -> nothing to save
    win.close()


def test_recent_analyses_populate_both_menus_identically(qapp, tmp_path, isolated_prefs):
    from respmech.ui import prefs
    win = _make_win()
    p = str(tmp_path / "analysis.toml")
    open(p, "w").close()
    prefs.add_recent_analysis(p)

    win._rebuild_recent_analyses()
    header_recents = [a.text() for a in win._recent_actions]
    file_recents = [a.text() for a in win._file_recent_menu.actions()]
    assert header_recents == file_recents
    assert any("analysis.toml" in t for t in header_recents)
    assert win._file_recent_menu.isEnabled() is True

    # two menu-opens in a row keep the same set (identity of the underlying recents list,
    # not necessarily the same Python objects — the header's own recents are rebuilt the
    # same way on every open, see _rebuild_recent_analyses's docstring)
    win._rebuild_recent_analyses()
    assert [a.text() for a in win._recent_actions] == header_recents
    assert [a.text() for a in win._file_recent_menu.actions()] == file_recents
    win.close()


def test_no_recents_disables_open_recent_and_hides_header_separator(qapp, isolated_prefs):
    win = _make_win()
    win._rebuild_recent_analyses()
    assert win._file_recent_menu.isEnabled() is False
    assert win._recent_sep.isVisible() is False
    win.close()


def test_file_menus_shared_actions_lock_during_a_run(qapp, tmp_path, isolated_prefs):
    """K-094: only ``analysis_btn`` (the header's own QToolButton) used to be locked
    during a run — the File menu carries the EXACT SAME QAction objects (see
    ``test_menu_bar_and_header_share_the_same_action_objects`` above) and their
    shortcuts, so 'File > New analysis' (or Ctrl+N) could still swap the running
    settings out from under the worker with the header button greyed out. Recents are
    rebuilt fresh on every menu-open (_rebuild_recent_analyses), so unlike New/Open/
    Get started/Sample/Duplicate — locked once by _on_run_started below — a recent's
    enabled state must be re-derived from ``_run_active`` every time, which is what
    ``win._rebuild_recent_analyses()``/``win._refresh_analysis_menu()`` do here."""
    from respmech.ui import prefs
    win = _make_win()
    p = str(tmp_path / "analysis.toml")
    open(p, "w").close()
    prefs.add_recent_analysis(p)
    win._rebuild_recent_analyses()
    assert win.analysis_btn.isEnabled() is True
    assert win._act_new.isEnabled() is True
    assert win._recent_actions[0].isEnabled() is True

    win._on_run_started()
    assert win.analysis_btn.isEnabled() is False
    for act in (win._act_new, win._act_open, win._act_get_started,
               win._act_sample, win._act_duplicate):
        assert act.isEnabled() is False
    # Save/Save as are recomputed by _refresh_analysis_menu on every menu-open (unlike
    # the five above, locked once): assert the LOCK survives a live-state recompute
    # that would otherwise re-enable a savable new analysis.
    sc = win.settings_screen
    sc.state.settings = synth_settings(str(tmp_path))
    sc.from_state()
    sc._mark_dirty()
    win._refresh_analysis_menu()
    assert win._act_save.isEnabled() is False
    assert win._act_save_as.isEnabled() is False
    # Recents are rebuilt (not just re-flagged) on every open — reopen and check the
    # freshly-built action, and the "Open Recent" submenu itself.
    win._rebuild_recent_analyses()
    assert win._recent_actions[0].isEnabled() is False
    assert win._file_recent_menu.isEnabled() is False

    win._on_run_finished()
    assert win.analysis_btn.isEnabled() is True
    for act in (win._act_new, win._act_open, win._act_get_started,
               win._act_sample, win._act_duplicate):
        assert act.isEnabled() is True
    win._refresh_analysis_menu()
    assert win._act_save.isEnabled() is True        # dirty new analysis -> savable again
    assert win._recent_actions[0].isEnabled() is True
    assert win._file_recent_menu.isEnabled() is True
    win.close()


def test_view_menu_reflects_a_locked_tab(qapp):
    win = _make_win()
    win.tabs.setTabEnabled(win._i_preview, False)
    win._refresh_view_menu()
    assert win._view_actions[win._i_preview].isEnabled() is False
    assert win._view_actions[win._i_settings].isEnabled() is True

    win.tabs.setTabEnabled(win._i_preview, True)
    win._refresh_view_menu()
    assert win._view_actions[win._i_preview].isEnabled() is True
    win.close()


def test_view_menu_action_switches_tabs(qapp):
    win = _make_win()
    win._view_actions[win._i_preview].trigger()
    assert win.tabs.currentIndex() == win._i_preview
    win._view_actions[win._i_settings].trigger()
    assert win.tabs.currentIndex() == win._i_settings
    win.close()


def test_open_output_folder_action_uses_the_run_screens_own_slot(qapp, monkeypatch):
    win = _make_win()
    called = []
    monkeypatch.setattr(win.run_screen, "_open_output_folder", lambda: called.append(True))
    win._act_open_output.trigger()
    assert called == [True]
    win.close()


def test_about_dialog_is_self_bearing_offline(qapp):
    dlg = AboutDialog()
    text = " ".join(lab.text() for lab in dlg.findChildren(QLabel))
    assert __version__ in text
    assert "General Public License" in text
    assert "respmech.dk" in text
    assert "github.com/emilwalsted/respmech" in text
    # a version string embedded verbatim, not just "a version somewhere"
    assert f"RespMech {__version__}" in text
    dlg.close()


def test_about_dialog_does_not_leak_on_repeated_opens(qapp):
    """Self-review finding: accept()/close() only HIDES a QDialog, it does not destroy
    it — Help > About RespMech… used to leave one more QDialog parented under MainWindow
    per open, forever, since a bare ``AboutDialog(self).exec()`` never called
    deleteLater(). Fixed via Qt.WA_DeleteOnClose on the dialog itself, verified here by
    actually opening/closing it five times rather than just asserting the attribute."""
    win = _make_win()
    before = len(win.findChildren(QDialog))
    for _ in range(5):
        dlg = AboutDialog(win)
        assert dlg.testAttribute(Qt.WA_DeleteOnClose) is True
        dlg.show()
        dlg.accept()
        qapp.processEvents()      # let the queued deleteLater() actually run
    after = len(win.findChildren(QDialog))
    assert after == before, f"About dialog leaked: {before} -> {after} QDialog children"
    win.close()


def test_about_action_has_the_about_role(qapp):
    """QAction.MenuRole.AboutRole lets macOS move the item into the app menu — assert the
    role rather than relying on a platform we cannot run interactively here."""
    from PySide6.QtGui import QAction
    win = _make_win()
    about = next(a for a in win._help_menu.actions() if a.text() == "About RespMech…")
    assert about.menuRole() == QAction.MenuRole.AboutRole
    win.close()
