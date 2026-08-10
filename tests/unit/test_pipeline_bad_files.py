"""D18 (UI-overhaul): a file the pipeline cannot read must land as one failed row in
the batch result, never as an exception that aborts the whole run -- regardless of
where in the batch the bad file sorts, and regardless of whether shared-profile EMG
noise reduction (with ``auto_prop``) is on.

Before this fix, a per-file failure during the noise-profile-build pre-phase
(``_build_noise_set``'s ``auto_prop`` collection loop) or the single-file ECG
auto-detect pre-phase (``_auto_detect_ecg_settings``) was not guarded the way the main
per-file loop in ``run_batch`` is: it bubbled out of ``run_batch`` as a raw exception
before a single file was processed, and whether that happened depended on the bad
file's position in the sorted file list (auto_prop's 40000-sample collection loop could
finish -- and never reach the bad file -- before hitting it, or could hit it first and
abort). See ticket 20260804-0935 for the full analysis.

The one case that must NOT be silently skipped is the noise ``reference_file`` itself:
if that specific file cannot be read, the shared profile cannot be built at all, and
the failure must say so in plain English naming the reference file -- not bubble up
whatever raw exception the underlying loader happened to raise.
"""
import os
import shutil
from unittest.mock import patch

import pandas as pd
import pytest

from respmech.core.io.loaders import DataValidationError
from respmech.core.pipeline import run_batch
from respmech.settingsio.migrate import migrate_dict
from _helpers import INPUT, requires_synth, synth_legacy_dict, write_delim

pytestmark = requires_synth()


def _settings(indir, outdir, *, noise_enabled=True, reference_file=None, auto_prop=True):
    legacy = synth_legacy_dict(str(outdir), remove_ecg=True, remove_noise=noise_enabled)
    legacy["input"]["files"] = "*.csv"
    settings, _ = migrate_dict(legacy)
    settings.input.folder = str(indir)
    settings.output.folder = str(outdir)
    if noise_enabled:
        n = settings.processing.emg.noise
        n.enabled = True
        n.reference_file = reference_file
        n.use_expiration = True
        n.auto_prop = auto_prop
    return settings


def _make_batch(tmp_path, *, extra_bad_name=None, n_good=3):
    """``n_good`` copies of the committed golden recording (so breath segmentation
    behaves exactly as elsewhere in the suite), named so they sort in the middle of the
    alphabet, plus an optional too-narrow (4-column, vs. the 12 the settings require)
    file at ``extra_bad_name`` -- readable as a file, but ``load()`` raises
    ``DataValidationError`` once it reaches a column index past column 4."""
    indir = tmp_path / "in"
    indir.mkdir()
    good_names = []
    for i in range(n_good):
        name = f"case{i}.csv"
        shutil.copyfile(os.path.join(INPUT, "synth_case_A.csv"), indir / name)
        good_names.append(name)
    if extra_bad_name:
        write_delim(indir / extra_bad_name, ncols=4)
    return str(indir), good_names


@pytest.mark.parametrize("bad_name", ["aaa_bad.csv", "zzz_bad.csv"])
def test_a_bad_file_fails_its_own_row_regardless_of_order(tmp_path, bad_name):
    """D18 point 2: with an unreadable file first (sorts before the good files) or last
    (sorts after them, so auto_prop's 40000-sample loop can reach its cap first on the
    old code), the batch result is the same three OK rows and one failed row -- never an
    exception."""
    indir, good_names = _make_batch(tmp_path, extra_bad_name=bad_name)
    settings = _settings(indir, tmp_path / "out", reference_file=good_names[0])

    result = run_batch(settings)

    assert set(result.ok_files) == set(good_names)
    assert set(result.failed_files) == {bad_name}
    assert result.files[bad_name].error_kind == "DataValidationError"
    assert bad_name in result.files[bad_name].error


def test_a_bad_file_is_a_normal_failed_row_without_noise_reduction_too(tmp_path):
    """Same shape, noise reduction off entirely -- the main per-file loop already
    handled this case; this is the control that proves the two order-dependent tests
    above are exercising the noise-build pre-phase, not the main loop."""
    indir, good_names = _make_batch(tmp_path, extra_bad_name="bad.csv")
    settings = _settings(indir, tmp_path / "out", noise_enabled=False)

    result = run_batch(settings)

    assert set(result.ok_files) == set(good_names)
    assert set(result.failed_files) == {"bad.csv"}


@pytest.mark.parametrize("bad_ref_kind,expected_type", [
    ("narrow", DataValidationError),   # readable, but too few columns
    ("empty", pd.errors.EmptyDataError),   # not readable as a table at all
])
def test_a_bad_reference_file_raises_a_clear_error_naming_it_and_keeps_its_type(
        tmp_path, bad_ref_kind, expected_type):
    """D18 point 3: unlike the collection loop, ``_reference_noise_clip`` must not
    silently skip a bad file -- if the reference itself cannot be read, the shared
    profile cannot be built at all. The failure must name the reference file in plain
    English (the ``empty`` case's underlying pandas error, ``EmptyDataError: No columns
    to parse from file``, does NOT name it on its own -- proving the wrap adds real
    information). Tightened after self-review (10-08-2026) to also assert the ORIGINAL
    exception TYPE survives the wrap (not just ``pytest.raises(Exception)``, which
    cannot tell a helpful ``DataValidationError`` from a bare ``ValueError``):
    ``ui/workers.py``'s ``stage_noise_fidelity`` and ``ui/screens/run_screen.py``'s
    fix-hint lookup both key their handling off the exception CLASS, and a version of
    this fix that flattened everything to ``ValueError`` broke both silently -- the
    suite stayed green because neither is exercised by this file."""
    indir, good_names = _make_batch(tmp_path, n_good=3)
    if bad_ref_kind == "narrow":
        write_delim(tmp_path / "in" / "ref_bad.csv", ncols=4)
    else:
        (tmp_path / "in" / "ref_bad.csv").write_text("")
    settings = _settings(indir, tmp_path / "out", reference_file="ref_bad.csv")

    with pytest.raises(expected_type) as exc_info:
        run_batch(settings)

    assert "ref_bad.csv" in str(exc_info.value)


def test_every_file_failing_in_auto_prop_raises_a_clear_error(tmp_path):
    """Self-review finding (10-08-2026): if EVERY file fails during the auto_prop
    collection loop, falling through used to hand an empty list to ``np.concatenate``,
    which raises a bare 'need at least one array to concatenate' naming no file and no
    cause -- worse than the batch-aborting bug D18 exists to fix in the first place.
    ``_emg_segmented`` (what the collection loop calls per file) is patched to always
    fail, so every batch file hits that path, while the shared reference clip is built
    via the (unpatched) explicit-``reference_intervals`` branch of
    ``_reference_noise_clip``, which does not call ``_emg_segmented`` at all -- this
    isolates "the collection loop itself has nothing to work with" from "the reference
    file is unreadable" (already covered above), matching how this can happen for real
    (e.g. a mis-assigned/inverted flow channel fails every file's breath segmentation
    while the reference clip, built more permissively, still succeeds)."""
    indir, good_names = _make_batch(tmp_path, n_good=3)
    settings = _settings(indir, tmp_path / "out", reference_file=good_names[0])
    settings.processing.emg.noise.use_expiration = False
    settings.processing.emg.noise.reference_intervals = [[0.0, 1.0]]

    with patch("respmech.core.pipeline._emg_segmented", side_effect=RuntimeError("boom")):
        with pytest.raises(ValueError, match="auto_prop"):
            run_batch(settings)


def test_ecg_auto_detect_falls_back_when_its_reference_file_cannot_be_read(tmp_path):
    """D18 point 4: ``_auto_detect_ecg_settings`` reads a single file. If that read
    fails (the file exists -- unlike the missing-file case already covered by
    test_ecg_auto_detect.py -- but cannot be parsed as data), the whole batch must not
    abort: fall back to the configured manual ECG settings, with a warning, and process
    every good file normally."""
    indir, good_names = _make_batch(tmp_path, n_good=3)
    (tmp_path / "in" / "ecgref_bad.csv").write_text("")
    settings = _settings(indir, tmp_path / "out", noise_enabled=False)
    settings.processing.emg.ecg_auto_detect = True
    settings.processing.emg.ecg_reference_file = "ecgref_bad.csv"
    # Deliberately-distinctive manual settings, so a batch that completes successfully
    # proves they were actually the ones applied (not silently overwritten by a partial
    # auto-detect result). Field names corrected after self-review (10-08-2026): these
    # are the ``Settings`` dataclass's own field names (``detect_channel``,
    # ``ecg_min_height``) -- the ORIGINAL test wrote ``column_detect``/``minheight``,
    # which are the LEGACY-namespace names used internally by ``to_legacy_ns``/
    # ``pipeline.py``, not attributes of this object. Being a plain dataclass (no
    # ``__slots__``), setting those created two harmless, unread stray attributes, and
    # the final assertion below read back exactly the value the test itself had just
    # set -- a tautology that could never fail, verifying nothing.
    settings.processing.emg.detect_channel = 0
    settings.processing.emg.ecg_min_height = 999.0

    with pytest.warns(UserWarning, match="ecgref_bad.csv"):
        result = run_batch(settings)

    assert result.ecg_auto_report is None
    assert set(result.ok_files) == set(good_names)
    # ecgref_bad.csv also matches the batch's own "*.csv" mask (deliberately -- it is a
    # real, if empty, file in the input folder) and correctly fails as its own ordinary
    # row too, via the main loop's pre-existing per-file guard: falling back for the
    # auto-detect PRE-phase does not also have to make it a valid recording.
    assert set(result.failed_files) == {"ecgref_bad.csv"}
    assert settings.processing.emg.ecg_min_height == 999.0
