"""Source-level wording guards for the status/dom-bus overhaul (A03).

Mostly Qt-free — these scan the ``ui`` package's own source text rather than construct
any widget, so they run everywhere (no display, no qapp) and catch a regression the
moment it is typed, regardless of which screen it lands on. The one exception is the
lone-ampersand guard below: a caption bug only exists once Qt has actually rendered the
text, so it constructs a real window (``qapp``) instead of scanning source strings.
"""
import ast
import glob
import os

from _helpers import ROOT

_UI_FILES = sorted(
    p for p in glob.glob(os.path.join(ROOT, "src", "respmech", "ui", "**", "*.py"),
                         recursive=True)
    if "__pycache__" not in p)


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_no_file_s_plural_left_in_the_ui_package():
    """'N file(s)' reads as broken English at both n=1 and n>1; every site was rewritten
    to a real singular/plural (A03 point 4). A new one creeping back in is a regression,
    not a style nit — it is exactly the pattern this test exists to catch."""
    offenders = [os.path.relpath(p, ROOT) for p in _UI_FILES if "file(s)" in _read(p)]
    assert offenders == [], f"'file(s)' left in: {offenders}"


def test_no_prose_sends_the_user_to_a_settings_tab():
    """The tab is named 'Setup', not 'Settings' — user-facing prose that says otherwise
    sends the user hunting for a tab that does not exist (the ticket's concrete case: a
    noise-reference picker that was described as living '(Settings)' but has never lived
    anywhere but the Preview graph). This intentionally does not flag the
    SettingsScreen class name, ``core.settings.Settings``, or comments that talk about
    the settings OBJECT/MODEL — only prose steering the user to a screen by name.

    Ticket D21 (UI-overhaul) went further than this test's original scope and retired the
    two status-bar sentences that used to read 'Settings incomplete: …' / 'Settings valid
    ✓' as well (they now read 'Setup incomplete: …' / 'Setup valid ✓'), on the grounds
    that a status message living on a tab literally named 'Setup' has no good reason to
    call itself something else — so those two phrases are guarded against here too,
    rather than only documented as an exception that no longer applies."""
    offenders = []
    for p in _UI_FILES:
        text = _read(p)
        for needle in ("in Settings.", "(Settings)", "on the Settings tab",
                       "on the Settings screen", "Settings incomplete", "Settings valid"):
            if needle in text:
                offenders.append((os.path.relpath(p, ROOT), needle))
    assert offenders == [], f"prose still names a 'Settings' tab: {offenders}"


def _tooltip_and_intro_strings(path):
    """Every string literal passed as ``setToolTip(...)``'s first argument or as an
    ``intro=`` keyword, parsed from the AST rather than grepped — so a docstring or a
    code comment that happens to use the same words (this file is full of them, by
    design: they document the widget internals) can never be mistaken for user-facing
    text. Adjacent string literals (``"a" "b"``) are already one ``ast.Constant`` by the
    time the parser sees them, so a wrapped multi-line message is captured whole.

    Known gap: this only sees a PLAIN string literal in that argument position. A
    ``setToolTip(tip)``/``setToolTip(f"...")``/``intro=_help_tip(...)`` call — several of
    which exist in the package — is invisible to this scan, since none of those is an
    ``ast.Constant``. Harmless today (checked by hand: none of those sites currently say
    'the strip'), but a future offender built that way would slip past this test."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr == "setToolTip" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out.append((node.lineno, arg.value))
        for kw in node.keywords:
            if kw.arg == "intro" and isinstance(kw.value, ast.Constant) \
                    and isinstance(kw.value.value, str):
                out.append((node.lineno, kw.value.value))
    return out


def test_no_caption_anywhere_turns_an_ampersand_into_a_mnemonic(qapp, tmp_path):
    """A lone ``&`` in Qt text is a mnemonic marker, not an ampersand — and that is true
    of far more than just push buttons.

    Qt eats the ``&`` and underlines the character after it. When that character is a
    SPACE the caption silently loses the word: "Run & results ▸" renders as
    "Run _results ▸", which is what shipped in v2.4.0 and what the documentation
    screenshots taken from it show. Guard the whole window rather than one widget type:
    every caption in this app that wants a literal ampersand already doubles it
    ("Preview && QC", "Process && write this file"), so a lone ``&`` is either this bug
    or a deliberate mnemonic — and a deliberate mnemonic is never on a space.

    This generalises the original, narrower button-only guard to group-box titles,
    every menu/menu-bar action, tab captions and buddy labels too — later screens
    register their own windows/menus in the same ``_lone_ampersands(root)`` scan rather
    than growing a second, independent copy of it."""
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState

    from _helpers import _lone_ampersands, synth_settings

    win = MainWindow(AppState(synth_settings(tmp_path)))
    offenders = _lone_ampersands(win)
    win.close()
    assert not offenders, (
        "these captions carry a lone '&' that Qt will swallow — double it to '&&': "
        f"{offenders}")


def test_lone_ampersand_scan_actually_catches_an_injected_regression(qapp):
    """The guard above passing on a clean window is not proof it would catch a real
    regression — confirm it flags the literal v2.4.0 caption on a bare probe widget,
    covering the button/group-box/tab-bar/action paths in one pass."""
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QGroupBox, QMenuBar, QPushButton, QTabBar, QWidget

    from _helpers import _lone_ampersands

    probe = QWidget()
    QPushButton("Run & results ▸", probe)
    QGroupBox("RMS & normalisation", probe)
    tabs = QTabBar(probe)
    tabs.addTab("Notes & figures")
    bar = QMenuBar(probe)
    menu = bar.addMenu("Reports")
    menu.addAction(QAction("Export & share", probe))

    offenders = _lone_ampersands(probe)
    offending_texts = {text for _kind, text in offenders}
    assert offending_texts == {
        "Run & results ▸", "RMS & normalisation", "Notes & figures", "Export & share"}


def test_no_tooltip_or_advanced_dialog_intro_says_the_strip():
    """Ticket D21 (UI-overhaul): three Advanced-dialog intros and a tooltip described a
    row of controls as 'the strip' or 'the strip' plus the checkbox they gate — a word
    that never appears anywhere in the actual interface, so a user reading it has nothing
    on screen to match it against. They now say 'the toolbar above the plots', which
    names something the user can actually see and point at."""
    offenders = []
    for p in _UI_FILES:
        for lineno, s in _tooltip_and_intro_strings(p):
            if "the strip" in s:
                offenders.append((os.path.relpath(p, ROOT), lineno, s))
    assert offenders == [], f"a tooltip/intro still says 'the strip': {offenders}"
