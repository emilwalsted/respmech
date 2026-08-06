"""Opening a ``.toml``/``.py`` analysis by platform means other than the Open dialog
(ticket C04, UI-overhaul): dropping it onto the main window, double-clicking it in
Finder/Explorer (packaging's ``document_type``), and macOS's native FileOpen launch event.

The ticket's own description claims ``settings_screen.open_analysis`` "already carries the
guard against discarding unsaved changes" — reading the code shows that is not so: every
existing caller (``_open_recent``, ``open_analysis_dialog``, the startup chooser) calls
``confirm_discard_changes`` itself, first. The drop/FileOpen handlers below follow that same
established pattern rather than the ticket's (incorrect) description of it.
"""
import types

from PySide6.QtCore import QMimeData, QPoint, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from respmech.ui.app import resolve_startup_path
from respmech.ui.qapp import _capture_or_forward
from respmech.ui.state import AppState


# --------------------------------------------------------------------------- #
# resolve_startup_path — Qt-free, no qapp needed
# --------------------------------------------------------------------------- #
def test_resolve_startup_path_prefers_the_native_open_event():
    assert resolve_startup_path(["respmech-gui"], "/native/opened.toml") == "/native/opened.toml"


def test_resolve_startup_path_falls_back_to_argv():
    assert resolve_startup_path(["respmech-gui", "/cli/passed.toml"], None) == "/cli/passed.toml"


def test_resolve_startup_path_native_wins_over_argv_when_both_present():
    got = resolve_startup_path(["respmech-gui", "/cli/one.toml"], "/native/two.toml")
    assert got == "/native/two.toml"


def test_resolve_startup_path_is_case_insensitive_on_the_extension():
    assert resolve_startup_path(["respmech-gui", "SUBJECT07.TOML"]) == "SUBJECT07.TOML"
    assert resolve_startup_path(["respmech-gui"], "Analysis.Toml") == "Analysis.Toml"


def test_resolve_startup_path_ignores_a_legacy_py_and_nothing_at_all():
    # .py at startup is deliberately NOT recognised here (only via drag-and-drop onto an
    # already-open window, through settings_screen.open_analysis's own extension routing).
    assert resolve_startup_path(["respmech-gui", "/some/legacy.py"]) is None
    assert resolve_startup_path(["respmech-gui"], "/some/legacy.py") is None
    assert resolve_startup_path(["respmech-gui"], None) is None
    assert resolve_startup_path([], None) is None


# --------------------------------------------------------------------------- #
# RespMechApplication's FileOpen capture-or-forward logic — Qt permits only one
# QApplication per process, so this is tested against a plain stand-in, not a second app.
# --------------------------------------------------------------------------- #
def test_capture_or_forward_remembers_the_first_path_with_no_handler_yet():
    target = types.SimpleNamespace(opened_path=None, on_file_open=None)
    _capture_or_forward(target, "/a.toml")
    assert target.opened_path == "/a.toml"
    # a second one before a handler exists does not overwrite the first
    _capture_or_forward(target, "/b.toml")
    assert target.opened_path == "/a.toml"


def test_capture_or_forward_routes_live_once_a_handler_is_set():
    calls = []
    target = types.SimpleNamespace(opened_path=None, on_file_open=calls.append)
    _capture_or_forward(target, "/later.toml")
    assert calls == ["/later.toml"]
    assert target.opened_path is None   # never captured — it went straight to the handler


# --------------------------------------------------------------------------- #
# MainWindow: dropping a .toml/.py, guarded like every other way of opening
# --------------------------------------------------------------------------- #
def _url_mime(*paths):
    m = QMimeData()
    m.setUrls([QUrl.fromLocalFile(p) for p in paths])
    return m


def _drag_enter(widget, mime):
    from PySide6.QtWidgets import QApplication
    ev = QDragEnterEvent(QPoint(0, 0), Qt.DropAction.CopyAction, mime,
                         Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, ev)
    return ev


def _drop(widget, mime):
    from PySide6.QtWidgets import QApplication
    ev = QDropEvent(QPoint(0, 0), Qt.DropAction.CopyAction, mime,
                    Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, ev)
    return ev


def test_mainwindow_accepts_a_single_local_toml_or_py_drag(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    assert _drag_enter(win, _url_mime(str(tmp_path / "a.toml"))).isAccepted() is True
    assert _drag_enter(win, _url_mime(str(tmp_path / "legacy.py"))).isAccepted() is True
    win.close()


def test_mainwindow_rejects_multi_file_remote_and_unrecognised_extension_drags(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    multi = _url_mime(str(tmp_path / "a.toml"), str(tmp_path / "b.toml"))
    assert _drag_enter(win, multi).isAccepted() is False

    remote = QMimeData(); remote.setUrls([QUrl("https://example.com/a.toml")])
    assert _drag_enter(win, remote).isAccepted() is False

    wrong_ext = _url_mime(str(tmp_path / "recording.csv"))
    assert _drag_enter(win, wrong_ext).isAccepted() is False
    win.close()


def test_dropping_a_toml_calls_open_analysis_when_the_analysis_is_clean(qapp, tmp_path):
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    calls = []
    win.settings_screen.open_analysis = lambda p: (calls.append(p), True)[1]
    p = str(tmp_path / "dropped.toml")

    _drag_enter(win, _url_mime(p))    # a real drag session always sends this first
    drop = _drop(win, _url_mime(p))
    assert drop.isAccepted() is True
    assert calls == [p]
    win.close()


def test_dropping_over_unsaved_edits_asks_and_a_cancel_blocks_the_open(qapp, tmp_path, monkeypatch):
    """The ticket's own claim that open_analysis 'already carries' the discard-changes guard
    does not hold up against the code (see module docstring): every real caller applies
    confirm_discard_changes itself first. This is the guard actually firing for a drop."""
    from respmech.ui.screens import settings_screen as ss
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc._mark_dirty()
    calls = []
    sc.open_analysis = lambda p: (calls.append(p), True)[1]

    monkeypatch.setattr(ss.QMessageBox, "question",
                        staticmethod(lambda *a, **k: ss.QMessageBox.Cancel))
    dropped = str(tmp_path / "dropped.toml")
    _drag_enter(win, _url_mime(dropped))
    drop = _drop(win, _url_mime(dropped))
    assert drop.isAccepted() is True     # the DRAG is still a recognised .toml…
    assert calls == []                   # …but the guard aborted the actual open
    win.close()


def test_dropping_over_unsaved_edits_a_discard_lets_the_open_through(qapp, tmp_path, monkeypatch):
    from respmech.ui.screens import settings_screen as ss
    from respmech.ui.main_window import MainWindow
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc._mark_dirty()
    calls = []
    sc.open_analysis = lambda p: (calls.append(p), True)[1]

    monkeypatch.setattr(ss.QMessageBox, "question",
                        staticmethod(lambda *a, **k: ss.QMessageBox.Discard))
    p = str(tmp_path / "dropped.toml")
    _drag_enter(win, _url_mime(p))
    _drop(win, _url_mime(p))
    assert calls == [p]
    win.close()


# --------------------------------------------------------------------------- #
# Packaging metadata: *.toml is a registered document type on BOTH platforms briefcase
# builds (macOS dmg / Windows MSI). Parsed directly with tomllib — no briefcase import
# needed, so this runs in the ordinary unit suite, not just a packaging build.
# --------------------------------------------------------------------------- #
def test_pyproject_declares_toml_as_a_document_type():
    import os
    try:
        import tomllib
    except ModuleNotFoundError:                # pragma: no cover - py<3.11 fallback
        import tomli as tomllib
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    with open(os.path.join(root, "pyproject.toml"), "rb") as f:
        data = tomllib.load(f)
    doc = data["tool"]["briefcase"]["app"]["respmech"]["document_type"]["toml"]
    assert doc["extension"] == "toml"
    assert doc["description"]
    assert doc["icon"]
    assert doc["url"].startswith(("http://", "https://"))
    # NOT re-declared as separate copies under [...respmech.macOS]/[...respmech.windows]
    # (see pyproject.toml's own comment): briefcase's merge_config only overrides a
    # base-level table when the platform section itself redefines the SAME key, so this
    # single, app-level table already reaches both platforms' build configuration without
    # being duplicated under either — verified directly against briefcase 0.4.4's own
    # parse_config()/validate_document_type_config() during development of this ticket.
    resp = data["tool"]["briefcase"]["app"]["respmech"]
    assert "document_type" not in resp.get("macOS", {})
    assert "document_type" not in resp.get("windows", {})
