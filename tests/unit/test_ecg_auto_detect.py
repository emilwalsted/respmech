"""processing.emg.ecg_auto_detect: expose core.emg.suggest_ecg_settings (the GUI ECG tab's
"Auto-suggest" analysis) to the batch pipeline / CLI, so it can be driven from a settings.toml
without ever opening the GUI — the CLI counterpart to processing.emg.noise.auto_prop.
Computed ONCE per test from a reference file and applied identically to every file, mirroring
auto_prop's "selected once, applied identically, never re-tuned per file" model.
"""
import os

import numpy as np
import pandas as pd
import pytest

from respmech.core.pipeline import run_batch
from respmech.settingsio.migrate import migrate_dict
from _helpers import INPUT, requires_synth, synth_legacy_dict  # noqa: F401

pytestmark = requires_synth()

_EMG_COLS = ("EMG1", "EMG2", "EMG3")


def _ecg_spike(w):
    """A short biphasic spike ~ an R-wave (same shape as tests/unit/test_ecg.py)."""
    t = np.linspace(-1, 1, w)
    return np.exp(-(t ** 2) / 0.02) * np.sin(2 * np.pi * 1.5 * t)


def _contaminate(src_path, out_path, *, fs=1000, hr_period_s=0.8, amp=1.5, seed=0):
    """Copy a golden synthetic CSV, keeping its flow/pressure columns (so breath
    segmentation behaves exactly as elsewhere in the suite) but replacing EMG1..3 with
    baseline noise plus a clear, ECG-like periodic contamination (amplitude decaying
    across channels, as on a real electrode montage) so a channel/parameter choice is
    actually meaningful to detect. Returns the number of beats injected."""
    df = pd.read_csv(src_path)
    n = len(df)
    rng = np.random.default_rng(seed)
    for col in _EMG_COLS:
        df[col] = rng.normal(0, 0.02, n)
    w = int(0.06 * fs)
    spike = _ecg_spike(w) * amp
    p = int(0.5 * fs)
    n_beats = 0
    while p + w < n:
        for ci, col in enumerate(_EMG_COLS):
            df.loc[p:p + w - 1, col] += spike * (1.0 - 0.15 * ci)
        p += int(hr_period_s * fs)
        n_beats += 1
    df.to_csv(out_path, index=False)
    return n_beats


@pytest.fixture
def ecg_input(tmp_path):
    """A 2-file batch (borrowed flow/pressure from the committed synthetic golden input)
    with a known, injected ECG contamination on every EMG channel. Returns (folder, beats)
    where ``beats`` maps filename -> number of beats injected."""
    indir = tmp_path / "in"
    indir.mkdir()
    beats = {}
    for name in ("synth_case_A.csv", "synth_case_B.csv"):
        beats[name] = _contaminate(os.path.join(INPUT, name), indir / name, seed=hash(name) % 1000)
    return str(indir), beats


def _settings(indir, outdir, **emg_overrides):
    legacy = synth_legacy_dict(str(outdir), remove_ecg=True)
    settings, _ = migrate_dict(legacy)
    settings.input.folder = str(indir)
    for key, val in emg_overrides.items():
        setattr(settings.processing.emg, key, val)
    return settings


def test_auto_detect_overrides_unusable_manual_settings_for_every_file(ecg_input, tmp_path):
    indir, beats = ecg_input
    # A manual height far above the injected spike amplitude detects nothing on its own —
    # proves the auto-detected value (not the manual one) is what actually ran.
    settings = _settings(indir, tmp_path / "out", ecg_min_height=50.0, ecg_auto_detect=True)
    result = run_batch(settings)

    assert not result.failed_files, result.failed_files
    assert result.ecg_auto_report is not None
    assert result.ecg_auto_report["reference_file"] == "synth_case_A.csv"   # first matched file
    assert result.ecg_auto_report["_diagnostics"]["confidence"] == "high"
    # Which of the 3 contaminated channels scores best is suggest_ecg_settings' own
    # decision (covered by tests/unit/test_emg_suggest.py) — here we only need a valid,
    # in-range channel whose derived parameters actually work on every file below.
    assert result.ecg_auto_report["detect_channel"] in (0, 1, 2)

    for fname, n_beats in beats.items():
        fr = result.ok_files[fname]
        assert fr.ecg is not None
        # detected most of the injected beats on EVERY file, i.e. the auto-detected
        # parameters (not just the reference file's own) were applied to each of them.
        assert fr.ecg["n_peaks"] >= 0.5 * n_beats, f"{fname}: only {fr.ecg['n_peaks']}/{n_beats}"
        assert fr.ecg["suppression"] > 0.2


def test_auto_detect_off_leaves_the_unusable_manual_settings_in_effect(ecg_input, tmp_path):
    indir, _beats = ecg_input
    settings = _settings(indir, tmp_path / "out", ecg_min_height=50.0, ecg_auto_detect=False)
    result = run_batch(settings)

    assert not result.failed_files, result.failed_files
    assert result.ecg_auto_report is None
    for fr in result.ok_files.values():
        assert fr.ecg["n_peaks"] == 0             # confirms the height=50 setting truly ran


def test_auto_detect_requires_remove_ecg(ecg_input, tmp_path):
    indir, _beats = ecg_input
    legacy = synth_legacy_dict(str(tmp_path / "out"), remove_ecg=False)
    settings, _ = migrate_dict(legacy)
    settings.input.folder = str(indir)
    settings.processing.emg.ecg_auto_detect = True

    with pytest.raises(ValueError, match="remove_ecg"):
        run_batch(settings)


def test_auto_detect_reference_file_is_selectable(ecg_input, tmp_path):
    indir, _beats = ecg_input
    settings = _settings(indir, tmp_path / "out", ecg_min_height=50.0, ecg_auto_detect=True,
                         ecg_reference_file="synth_case_B.csv")
    result = run_batch(settings)

    assert not result.failed_files, result.failed_files
    assert result.ecg_auto_report["reference_file"] == "synth_case_B.csv"
    for fr in result.ok_files.values():
        assert fr.ecg["n_peaks"] > 0


def test_auto_detect_is_golden_neutral_when_off_by_default(ecg_input, tmp_path):
    """ecg_auto_detect defaults to False -> BatchResult.ecg_auto_report stays None and no
    extra file is touched, i.e. existing settings.toml files behave exactly as before."""
    indir, _beats = ecg_input
    settings = _settings(indir, tmp_path / "out")
    assert settings.processing.emg.ecg_auto_detect is False
    result = run_batch(settings)
    assert result.ecg_auto_report is None


def test_auto_detect_reports_a_clear_error_for_a_missing_reference_file(ecg_input, tmp_path):
    indir, _beats = ecg_input
    settings = _settings(indir, tmp_path / "out", ecg_auto_detect=True,
                         ecg_reference_file="does_not_exist.csv")
    with pytest.raises(ValueError, match="ecg_reference_file"):
        run_batch(settings)


def test_auto_detect_cache_priming_matches_the_uncached_path_on_integer_dtype_emg(tmp_path):
    """Regression test: _auto_detect_ecg_settings primes the per-run cache with
    _ecg_remove(s, raw_emg) -- the SAME native-dtype array the main loop's cache-miss path
    (_load_and_ecg) would pass -- not the float-cast copy suggest_ecg_settings needs. If the
    two ever diverge, an integer-dtype raw EMG column (a plausible ADC-export CSV) gets
    silently HIGHER precision on the reference file than on every other file in the batch
    (core.emg.subtractecg mutates its input array in place, truncating float results back to
    int), breaking the "applied identically to every file" guarantee for that one file."""
    from respmech.core import pipeline as P

    fs = 1000
    n = 4000
    rng = np.random.default_rng(0)
    emg_float = rng.normal(0, 5, (n, 2))
    w = int(0.06 * fs)
    spike = _ecg_spike(w) * 40.0
    p = int(0.3 * fs)
    while p + w < n:
        emg_float[p:p + w, 0] += spike
        p += int(0.5 * fs)
    emg_int = emg_float.astype(np.int64)          # a plausible integer ADC-count export

    load_result = (np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
                   np.zeros((n, 0)), emg_int)

    class _Fmt:
        samplingfrequency = fs
    class _Input:
        inputfolder = str(tmp_path)
        format = _Fmt()
    class _Emg:
        column_detect = 0
        minheight = 0.0005
        mindistance = 0.5
        minwidth = 0.001
        windowsize = 0.4
        remove_ecg = True
    class _Processing:
        emg = _Emg()
    class S:
        input = _Input()
        processing = _Processing()
    s = S()

    class _EmgCfg:
        ecg_reference_file = None
    class _SettingsProcessing:
        emg = _EmgCfg()
    class Settings:
        processing = _SettingsProcessing()

    import respmech.core.pipeline as pipeline_mod
    monkeypatched = pipeline_mod._load
    pipeline_mod._load = lambda path, s: load_result
    try:
        cache = {}
        report = P._auto_detect_ecg_settings(Settings(), s, ["any.csv"], cache=cache)
        assert report["reference_file"] == "any.csv"
        cached_path = os.path.abspath("any.csv")
        assert cached_path in cache
        _load_result_cached, emgcols_ecg_cached, ecg_diag_cached = cache[cached_path]

        # The uncached ("cache miss") path a sibling file would take: fresh _ecg_remove on
        # the SAME raw (int-dtype) array, with the now auto-detected settings applied.
        emgcols_ecg_fresh, ecg_diag_fresh = P._ecg_remove(s, emg_int)
    finally:
        pipeline_mod._load = monkeypatched

    assert np.array_equal(np.asarray(emgcols_ecg_cached), np.asarray(emgcols_ecg_fresh)), (
        "cached (reference-file) ECG removal diverges from the uncached path a sibling "
        "file would take -- the shared parameters were NOT applied identically")
    assert ecg_diag_cached["n_peaks"] == ecg_diag_fresh["n_peaks"]


def test_auto_detect_is_surfaced_in_cli_stdout_and_run_report(ecg_input, tmp_path, capsys):
    """Emil's request was specifically to run large batches UNSUPERVISED; that only works if
    the auto-detected settings (and their confidence) are visible somewhere a CLI-only user
    actually sees, not just buried in a Python dict (BatchResult.ecg_auto_report)."""
    from respmech.cli.__main__ import main as cli_main
    from respmech.settingsio.toml_io import save_toml

    indir, _beats = ecg_input
    outdir = tmp_path / "out"
    settings = _settings(indir, outdir, ecg_auto_detect=True)
    toml_path = tmp_path / "s.toml"
    save_toml(settings, toml_path)

    rc = cli_main(["run", str(toml_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ECG auto-detect" in out
    assert "synth_case_A.csv" in out          # the reference file actually used

    report = (outdir / "run-report.txt").read_text()
    assert "ECG auto-detect" in report
    assert "synth_case_A.csv" in report


def test_auto_detect_warns_per_file_when_shared_settings_do_not_fit(tmp_path):
    """A slow reference file (~50 bpm) derives a wide refractory floor; applied to a much
    faster sibling file (~200 bpm) most real beats fall inside that floor and are missed --
    exactly the failure mode detection_quality exists to catch. This must warn, not fail,
    the run (an unsupervised batch keeps going but leaves a visible trail)."""
    indir = tmp_path / "in"
    indir.mkdir()
    _contaminate(os.path.join(INPUT, "synth_case_A.csv"), indir / "synth_case_A.csv",
                 hr_period_s=1.2, amp=2.0, seed=1)     # slow reference (~50 bpm)
    _contaminate(os.path.join(INPUT, "synth_case_B.csv"), indir / "synth_case_B.csv",
                 hr_period_s=0.3, amp=2.0, seed=2)     # fast sibling (~200 bpm)

    settings = _settings(str(indir), tmp_path / "out", ecg_auto_detect=True)
    with pytest.warns(UserWarning, match="quality check failed|no R-peaks"):
        result = run_batch(settings)
    assert not result.failed_files, result.failed_files
