"""Qt-free helper for 'Duplicate for another recordings folder…' (ticket C03 point 5):
deriving a suggested output folder for a duplicated analysis from the SAME relative
naming the current analysis already has between its input and output folders.

Kept separate from ``duplicate_dialog.py`` (the Qt dialog) and ``settings_screen.py``
(the orchestration) so the naming logic itself is trivially unit-testable without a
QApplication.
"""
from __future__ import annotations

import os


def derive_sibling_output(old_input: str, old_output: str, new_input: str) -> str | None:
    """Suggest a new output folder for ``new_input``, reusing the naming pattern between
    ``old_input`` and ``old_output`` — but ONLY when ``old_output`` is a SIBLING of
    ``old_input`` (same parent folder), the one case common enough (and unambiguous
    enough) to guess safely. Anything else (output nested inside input, on a different
    drive, or either folder blank) returns ``None`` so the caller falls back to asking —
    per the ticket's own instruction, this is a SUGGESTION shown for confirmation, never
    applied silently.

    When the old input folder's basename appears in the old output folder's basename
    (the common per-subject convention, e.g. input "S01" / output "S01-output"), the new
    suggestion substitutes the new input's basename for it (-> "S02-output" for a new
    input "S02"). Otherwise the old output's basename is reused unchanged next to the new
    input (still useful, just not subject-specific).
    """
    if not old_input or not old_output or not new_input:
        return None
    old_input_abs = os.path.abspath(old_input)
    old_output_abs = os.path.abspath(old_output)
    new_input_abs = os.path.abspath(new_input)
    if os.path.dirname(old_output_abs) != os.path.dirname(old_input_abs):
        return None                    # not a sibling — caller must ask instead of guess
    old_base = os.path.basename(old_input_abs)
    out_base = os.path.basename(old_output_abs)
    new_base = os.path.basename(new_input_abs)
    if old_base and old_base in out_base:
        out_base = out_base.replace(old_base, new_base)
    return os.path.join(os.path.dirname(new_input_abs), out_base)
