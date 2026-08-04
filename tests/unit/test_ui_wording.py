"""Source-level wording guards for the status/dom-bus overhaul (A03).

Qt-free — these scan the ``ui`` package's own source text rather than construct any
widget, so they run everywhere (no display, no qapp) and catch a regression the moment
it is typed, regardless of which screen it lands on.
"""
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
    anywhere but the Preview graph). This intentionally does NOT flag 'Settings' used as
    the settings OBJECT/MODEL ('Settings incomplete: …', 'Settings valid ✓', the
    SettingsScreen class/comments) — only prose steering the user to a screen by name."""
    offenders = []
    for p in _UI_FILES:
        text = _read(p)
        for needle in ("in Settings.", "(Settings)", "on the Settings tab",
                       "on the Settings screen"):
            if needle in text:
                offenders.append((os.path.relpath(p, ROOT), needle))
    assert offenders == [], f"prose still names a 'Settings' tab: {offenders}"
