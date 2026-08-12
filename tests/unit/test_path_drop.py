"""Drag-and-drop onto a path field (ticket C04, UI-overhaul).

Measured on the live Setup screen before this existed: dragging a folder from Finder onto
"Recordings folder" gave ``dragEnter accepted: False``/``drop accepted: False`` and left the
field's text byte-unchanged — Finder's file/folder drag carries only ``text/uri-list``, and
``QLineEdit``'s own drop handling only recognises ``text/plain``. These tests reproduce that
exact scenario against ``install_path_drop`` directly, then against the real "Recordings
folder"/"Output folder" fields on a constructed ``SettingsScreen``.
"""
from PySide6.QtCore import QMimeData, QPoint, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QApplication, QLineEdit

from respmech.ui.path_drop import install_path_drop
from respmech.ui.state import AppState


def _drag_enter(widget, mime):
    ev = QDragEnterEvent(QPoint(0, 0), Qt.DropAction.CopyAction, mime,
                         Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, ev)
    return ev


def _drop(widget, mime):
    ev = QDropEvent(QPoint(0, 0), Qt.DropAction.CopyAction, mime,
                    Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, ev)
    return ev


def _url_mime(*paths):
    m = QMimeData()
    m.setUrls([QUrl.fromLocalFile(p) for p in paths])
    return m


def test_single_local_folder_drag_is_accepted_and_fills_the_field(qapp, tmp_path):
    line = QLineEdit()
    install_path_drop(line)
    mime = _url_mime(str(tmp_path))
    assert mime.hasFormat("text/plain") is False   # the real-world signature that broke this

    enter = _drag_enter(line, mime)
    assert enter.isAccepted() is True

    drop = _drop(line, _url_mime(str(tmp_path)))
    assert drop.isAccepted() is True
    assert line.text() == str(tmp_path)


def test_multiple_urls_are_rejected_and_field_stays_untouched(qapp, tmp_path):
    line = QLineEdit()
    line.setText("kept")
    install_path_drop(line)
    a, b = tmp_path / "a", tmp_path / "b"
    mime = _url_mime(str(a), str(b))

    assert _drag_enter(line, mime).isAccepted() is False
    assert _drop(line, _url_mime(str(a), str(b))).isAccepted() is False
    assert line.text() == "kept"


def test_non_local_url_is_rejected_and_field_stays_untouched(qapp):
    line = QLineEdit()
    line.setText("kept")
    install_path_drop(line)
    mime = QMimeData()
    mime.setUrls([QUrl("https://example.com/analysis.toml")])

    assert _drag_enter(line, mime).isAccepted() is False
    mime2 = QMimeData()
    mime2.setUrls([QUrl("https://example.com/analysis.toml")])
    assert _drop(line, mime2).isAccepted() is False
    assert line.text() == "kept"


def test_plain_text_drag_still_works_unaffected(qapp):
    """Not our gesture: no URLs at all (a text selection dragged from an editor) is left
    for QLineEdit's own, pre-existing drag-to-insert handling — the ticket is explicit that
    this must keep working exactly as before."""
    line = QLineEdit()
    install_path_drop(line)
    mime = QMimeData()
    mime.setText("some text")
    assert _drag_enter(line, mime).isAccepted() is True   # QLineEdit's native behaviour


def test_recordings_folder_field_accepts_a_dropped_folder(qapp, tmp_path):
    from respmech.ui.screens.settings_screen import SettingsScreen
    sc = SettingsScreen(AppState())
    mime = _url_mime(str(tmp_path))
    enter = _drag_enter(sc.in_folder, mime)
    assert enter.isAccepted() is True
    drop = _drop(sc.in_folder, _url_mime(str(tmp_path)))
    assert drop.isAccepted() is True
    assert sc.in_folder.text() == str(tmp_path)
    sc.close()


def test_output_folder_field_also_accepts_a_dropped_folder(qapp, tmp_path):
    from respmech.ui.screens.settings_screen import SettingsScreen
    sc = SettingsScreen(AppState())
    # a real drag session always sends dragEnter before drop; skipping straight to drop
    # is not just unrealistic here but crashes this sandbox's offscreen Qt outright.
    _drag_enter(sc.out_folder, _url_mime(str(tmp_path)))
    drop = _drop(sc.out_folder, _url_mime(str(tmp_path)))
    assert drop.isAccepted() is True
    assert sc.out_folder.text() == str(tmp_path)
    sc.close()


# --------------------------------------------------------------------------- #
# folder=True rejects an existing FILE (self-review finding): the Browse… button on
# these two fields only ever offers a folder picker, so a drag accepting a file would be
# a real inconsistency — and, on the real window, would silently pre-empt MainWindow's
# own "drop this .toml/.py to open it" handling for a drop that lands squarely on the field.
# --------------------------------------------------------------------------- #
def test_folder_field_rejects_a_dropped_file(qapp, tmp_path):
    f = tmp_path / "analysis.toml"
    f.write_text("")
    line = QLineEdit()
    line.setText("kept")
    install_path_drop(line, folder=True)

    assert _drag_enter(line, _url_mime(str(f))).isAccepted() is False
    assert _drop(line, _url_mime(str(f))).isAccepted() is False
    assert line.text() == "kept"


def test_folder_field_still_accepts_a_dropped_folder(qapp, tmp_path):
    line = QLineEdit()
    install_path_drop(line, folder=True)
    _drag_enter(line, _url_mime(str(tmp_path)))
    assert _drop(line, _url_mime(str(tmp_path))).isAccepted() is True
    assert line.text() == str(tmp_path)


def test_non_folder_field_still_accepts_a_dropped_file(qapp, tmp_path):
    """folder=False (the default) keeps accepting a file, as documented — there is no live
    file-picker path field yet, but a future one should not need a second mechanism."""
    f = tmp_path / "analysis.toml"
    f.write_text("")
    line = QLineEdit()
    install_path_drop(line)
    _drag_enter(line, _url_mime(str(f)))
    assert _drop(line, _url_mime(str(f))).isAccepted() is True
    assert line.text() == str(f)


def test_recordings_and_output_folder_fields_reject_a_dropped_analysis_file(qapp, tmp_path):
    """Closes the loop on the exact self-review finding: dropping a .toml/.py precisely
    onto Recordings/Output folder no longer silently swallows it into the folder field —
    MainWindow's own drop-to-open handling (test_file_association.py) is meant to get it."""
    from respmech.ui.screens.settings_screen import SettingsScreen
    sc = SettingsScreen(AppState())
    f = tmp_path / "analysis.toml"
    f.write_text("")
    for field, kept in ((sc.in_folder, "input"), (sc.out_folder, "output")):
        _drag_enter(field, _url_mime(str(f)))
        drop = _drop(field, _url_mime(str(f)))
        assert drop.isAccepted() is False
        assert field.text() == kept   # the defaults from_state() filled in, untouched
    sc.close()
