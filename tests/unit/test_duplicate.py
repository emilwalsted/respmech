"""Qt-free tests for ui/duplicate.py (ticket C03 point 5) — no QApplication needed."""
import os

from respmech.ui.duplicate import derive_sibling_output


def test_sibling_output_substitutes_the_subject_token(tmp_path):
    """The common per-subject convention: output "S01-output" next to input "S01" ->
    a NEW input "S02" suggests "S02-output", not the old subject's name."""
    study = tmp_path / "Study"
    old_input = study / "S01"; old_output = study / "S01-output"
    new_input = study / "S02"
    old_input.mkdir(parents=True); old_output.mkdir(); new_input.mkdir()
    got = derive_sibling_output(str(old_input), str(old_output), str(new_input))
    assert got == str(study / "S02-output")


def test_sibling_output_without_a_shared_token_keeps_the_basename(tmp_path):
    """No subject token to substitute -> the old output's own name is reused unchanged
    next to the new input (still a usable suggestion, just not subject-specific)."""
    study = tmp_path / "Study"
    old_input = study / "S01"; old_output = study / "results"
    new_input = study / "S02"
    old_input.mkdir(parents=True); old_output.mkdir(); new_input.mkdir()
    got = derive_sibling_output(str(old_input), str(old_output), str(new_input))
    assert got == str(study / "results")


def test_non_sibling_output_returns_none():
    """Output is NOT a sibling of input (nested inside it) -> the ticket's own instruction
    is to fall back to asking rather than guess."""
    got = derive_sibling_output(
        os.path.join("study", "S01"),
        os.path.join("study", "S01", "output"),      # nested, not a sibling
        os.path.join("study", "S02"))
    assert got is None


def test_blank_inputs_return_none():
    assert derive_sibling_output("", "/study/S01-output", "/study/S02") is None
    assert derive_sibling_output("/study/S01", "", "/study/S02") is None
    assert derive_sibling_output("/study/S01", "/study/S01-output", "") is None
