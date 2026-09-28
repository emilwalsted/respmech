"""End-to-end-ish tests for the pipeline, writers and CLI on the committed
synthetic data (no external data needed)."""
import os

import pandas as pd
import pytest

from respmech.cli.__main__ import main as cli_main
from respmech.core._legacy_ns import to_legacy_ns
from respmech.core.io.writers import write_batch
from respmech.core.pipeline import run_batch, segment_file
from respmech.settingsio.migrate import migrate_dict
from _helpers import INPUT, requires_synth, synth_legacy_dict, synth_settings  # noqa: F401


pytestmark = requires_synth()


def _legacy(outdir):
    return synth_legacy_dict(outdir, calcwobfromaverage=True, data_out={
        "saveaveragedata": True, "savebreathbybreathdata": True,
        "saveprocesseddata": True, "includeignoredbreaths": False})


def test_run_batch_and_write(tmp_path):
    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    events = []
    result = run_batch(settings, progress=events.append)
    assert set(result.ok_files) == {"synth_case_A.csv", "synth_case_B.csv"}
    assert not result.failed_files
    # progress events emitted
    assert any(e.kind == "file_done" for e in events)
    assert any(e.kind == "finished" for e in events)

    written = write_batch(result, settings, str(tmp_path))
    # core data outputs: 2 breathdata + 2 processed + 1 average
    for name in ("data/synth_case_A.csv.breathdata.xlsx", "data/synth_case_B.csv.breathdata.xlsx",
                 "data/synth_case_A.csv – Processed data.csv", "data/Average breathdata.xlsx",
                 "data/Cohort summary.xlsx",              # P8/P15 cohort aggregation
                 "analysis-used.toml", "run-report.txt"):  # P7 provenance
        assert os.path.isfile(os.path.join(tmp_path, name)), f"missing {name}"
    # P11 diagnostic figures land under diagnostics/ (vector PDF)
    assert any(p.endswith(".pdf") and os.sep + "diagnostics" + os.sep in p for p in written)
    avg = pd.read_excel(os.path.join(tmp_path, "data", "Average breathdata.xlsx"), sheet_name="Data")
    assert list(avg["file"]) == ["synth_case_A.csv", "synth_case_B.csv"]
    assert "wobtotal" in avg.columns
    # the raw result table is unchanged — the extras live in separate sheets/files
    assert "rms_col_2_pct" not in avg.columns


def test_segment_file_equals_run_batch_breath_keys(tmp_path):
    """``segment_file()``, called directly on one file, must build the exact same
    breaths run_batch's main loop builds for that file at the SAME point in the
    pipeline (right after segmentation, before calculatemechanics runs and adds its
    own keys such as 'mechanics'/'wob'/'rms') — the extraction (M-06) is supposed
    to be a pure move, never a behaviour change. ``ref_breaths`` here is
    ``result.ok_files[...].breaths``: calculatemechanics mutates those SAME dicts
    in place (adding keys, never touching 'time'/'flow'/'emgcols'), so the base
    _make_breath key set is a SUBSET of ref's keys, and the shared columns'
    values must still be identical."""
    import numpy as np

    base_keys = {"number", "name", "expiration", "inspiration", "time", "flow",
                 "volume", "poes", "pgas", "pdi", "breathcnt", "ignored", "kind",
                 "has_phases", "entcols", "emgcols", "filename"}

    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    result = run_batch(settings)
    ref_breaths = result.ok_files["synth_case_A.csv"].breaths

    s = to_legacy_ns(settings)
    path = os.path.join(settings.input.folder, "synth_case_A.csv")
    breaths, trimmed = segment_file(settings, s, path, cache={}, cancel_check=None)

    assert set(breaths) == set(ref_breaths), "breath numbers differ"
    for no, ref in ref_breaths.items():
        got = breaths[no]
        assert set(got) == base_keys, f"breath #{no}: unexpected segment_file key set {set(got)}"
        assert base_keys <= set(ref), f"breath #{no}: run_batch dropped a base key"
        for phase in ("inspiration", "expiration"):
            assert set(got[phase]) <= set(ref[phase]), f"breath #{no} {phase}: key mismatch"
        np.testing.assert_array_equal(got["time"], ref["time"])
        np.testing.assert_array_equal(got["flow"], ref["flow"])
        np.testing.assert_array_equal(np.asarray(got["emgcols"]), np.asarray(ref["emgcols"]))
        assert got["ignored"] == ref["ignored"]
        assert got["kind"] == ref["kind"]
        assert got["has_phases"] == ref["has_phases"]

    # segment_file's own returned Trimmed matches the file this batch actually ran.
    assert trimmed.startix >= 0 and trimmed.endix > trimmed.startix
    assert len(trimmed.flow) == trimmed.endix - trimmed.startix


def test_synth_settings_channels_none_drops_the_role(tmp_path):
    """synth_settings(channels={...}) sets the given roles on the migrated Settings
    without touching anything else — the fase-0 helper feature keyed by this
    ticket's acceptance criteria (a settings object with pgas/pdi absent, standard
    behaviour otherwise unaffected)."""
    default = synth_settings(str(tmp_path))
    assert default.input.channels.pgas is not None
    assert default.input.channels.pdi is not None

    s = synth_settings(str(tmp_path), channels={"pgas": None, "pdi": None})
    assert s.input.channels.pgas is None
    assert s.input.channels.pdi is None
    # untouched roles keep the canonical synthetic-input assignment
    assert s.input.channels.flow == default.input.channels.flow
    assert s.input.channels.poes == default.input.channels.poes
    assert s.input.channels.emg == default.input.channels.emg


def test_cli_migrate_and_validate(tmp_path):
    legacy_py = tmp_path / "legacy.py"
    legacy_py.write_text(
        "settings = {'input': {'inputfolder': %r, 'files': 'synth_case_*.csv',"
        "'format': {'samplingfrequency': 1000},"
        "'data': {'column_poes':7,'column_pgas':8,'column_pdi':9,'column_volume':6,"
        "'column_flow':5,'columns_emg':[],'columns_entropy':[]}}}" % INPUT)
    toml = tmp_path / "s.toml"
    assert cli_main(["migrate", str(legacy_py), "-o", str(toml)]) == 0
    assert toml.exists()
    assert cli_main(["validate", str(toml)]) == 0


def test_examples_settings_toml_validates_and_dry_runs(capsys):
    """K-046 (05-09-2026 content review, finding 7.4): a CLI-only user had no
    settings.toml to copy anywhere but a web page. examples/settings.toml ships in the
    repository, already wired to the committed sample_recording.csv, so both commands the
    manual tells a CLI user to run first actually work, unedited, from a fresh checkout."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    example = os.path.join(repo_root, "examples", "settings.toml")
    assert os.path.isfile(example)
    assert cli_main(["validate", example]) == 0
    out = capsys.readouterr().out
    assert "matches 1 file(s)" in out
    assert cli_main(["run", example, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    example_output = os.path.join(repo_root, "examples", "output")
    assert not os.path.isdir(example_output)   # a dry run never writes anything


def test_cli_run_dry_run(tmp_path, capsys):
    # write a minimal TOML via migrate, then dry-run
    from respmech.settingsio.toml_io import save_toml
    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)
    rc = cli_main(["run", str(toml), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert not os.path.exists(os.path.join(tmp_path, "data"))  # nothing written


def test_cli_run_dry_run_shows_the_output_plan(tmp_path, capsys):
    """K-108/A06: a CLI dry run used to print only per-file breath counts, a different
    (smaller) promise than the GUI's Dry run, which shows core.io.plan.plan_outputs'
    full ceiling (data/, diagnostics/, provenance). Both now build the SAME plan."""
    from respmech.settingsio.toml_io import save_toml
    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)
    rc = cli_main(["run", str(toml), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Output plan:" in out
    assert "Run report and analysis snapshot" in out   # a Plan group every run always has
    assert "Total:" in out
    assert str(tmp_path) in out


def test_cli_run_dry_run_says_segments_for_an_emg_only_set(tmp_path, capsys):
    """EMG-only segmentation (whole_file/separators) has no breaths to count -- the
    per-file dry-run line says 'N segments' instead."""
    from respmech.settingsio.toml_io import save_toml

    s = synth_settings(tmp_path, channels={
        "flow": None, "poes": None, "pgas": None, "pdi": None, "volume": None})
    s.processing.segmentation.method = "whole_file"
    toml = tmp_path / "s.toml"
    save_toml(s, toml)
    rc = cli_main(["run", str(toml), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "synth_case_A.csv: 1 segments" in out
    assert "synth_case_A.csv: 1 breaths" not in out


def test_cli_run_wrote_message_names_the_output_root(tmp_path, capsys):
    """K-278: '... to <output>/data' undercounted what a run actually writes —
    diagnostics/ figures and the two root-level provenance files are all part of the
    count too (a 65-file run measured only 8 of them under data/)."""
    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    toml = tmp_path / "s.toml"
    from respmech.settingsio.toml_io import save_toml
    save_toml(settings, toml)
    rc = cli_main(["run", str(toml)])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"Wrote " in out and f"file(s) to {tmp_path}" in out
    assert f"to {tmp_path}/data" not in out


def test_cli_validate_reports_unknown_keys(tmp_path, capsys):
    """K-113: a misspelled key was collected in Settings.unknown but read nowhere —
    respmech validate must report it (and fail, so a script that checks the exit code
    also catches it)."""
    from respmech.settingsio.toml_io import save_toml
    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)
    # append an unrecognised key by hand, under a brand-new table (a TOML file can only
    # declare a given table once, and save_toml already wrote [processing.volume])
    with open(toml, "a", encoding="utf-8") as f:
        f.write("\n[processing.made_up_section]\ncorect_drift = false\n")
    rc = cli_main(["validate", str(toml)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "unrecognised setting" in err
    assert "processing.made_up_section" in err
    assert "corect_drift" in err


def test_cli_validate_probes_the_output_folder(tmp_path, capsys, monkeypatch):
    """K-098: the same real write probe the GUI's Dry run already performs
    (core.io.plan.probe_write_folder) — never os.access, unreliable against Windows
    ACLs — so a read-only or missing output folder is caught here instead of after an
    entire batch has been computed. Monkeypatched rather than chmod'd (a real
    permission-bit test is unreliable under root, which ignores them — see
    test_gui_hardening.py's own os.access skip-guard for the same probe)."""
    from respmech.settingsio.toml_io import save_toml
    from respmech.core.io import plan as plan_mod
    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)
    monkeypatch.setattr(plan_mod, "probe_write_folder",
                        lambda folder: plan_mod.WriteProbe(False, "disk full"))
    rc = cli_main(["validate", str(toml)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not writable" in err
    assert "disk full" in err


def test_cli_validate_accepts_a_writable_output_folder(tmp_path, capsys):
    """The probe itself is real (not monkeypatched here): an ordinary, writable tmp_path
    output folder must not be flagged."""
    from respmech.settingsio.toml_io import save_toml
    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)
    rc = cli_main(["validate", str(toml)])
    assert rc == 0
    assert "not writable" not in capsys.readouterr().err


def _write_layout_csv(path, n, *, time_col, extra_cols=None):
    """A minimal CSV matching the synthetic layout's column order (time, flow, volume,
    poes, pgas, pdi) so ``respmech validate``'s new merged-block/constant-channel probes
    (13-09-2026) have something plausible to read column indices from — the
    values themselves are not physiologically meaningful, only the two properties each
    test below cares about (column 0's timestamps, or one column's variance)."""
    import numpy as np
    cols = {"time": time_col, "flow": np.sin(np.linspace(0, 10, n)),
            "volume": np.linspace(0, 1, n), "poes": np.linspace(-5, -3, n),
            "pgas": np.linspace(6, 8, n), "pdi": np.linspace(11, 13, n)}
    if extra_cols:
        cols.update(extra_cols)
    pd.DataFrame(cols).to_csv(path, index=False)


def _validate_settings_toml(tmp_path, folder, *, flow=2, volume=3, poes=4, pgas=5, pdi=6):
    from respmech.settingsio.toml_io import save_toml
    legacy = {"input": {"inputfolder": str(folder), "files": "*.csv",
                        "format": {"samplingfrequency": 1000},
                        "data": {"column_flow": flow, "column_volume": volume,
                                 "column_poes": poes, "column_pgas": pgas,
                                 "column_pdi": pdi, "columns_emg": [], "columns_entropy": []}},
             "output": {"outputfolder": str(tmp_path / "out")}}
    settings, _ = migrate_dict(legacy)
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)
    return toml


def test_cli_validate_warns_about_merged_time_blocks(tmp_path, capsys):
    """Acceptance criterion 1 (13-09-2026): a CSV whose column 0 looks like
    two recordings merged by timestamp (duplicated/decreasing steps over an otherwise
    regular ~1000 Hz axis) must be flagged by `respmech validate`."""
    import numpy as np
    n = 3000
    t_clean = np.arange(n) / 1000.0
    t_merged = np.sort(np.concatenate([t_clean[:600], t_clean]))   # first 600 samples duplicated
    _write_layout_csv(tmp_path / "merged.csv", len(t_merged), time_col=t_merged)
    toml = _validate_settings_toml(tmp_path, tmp_path)
    rc = cli_main(["validate", str(toml)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "merged.csv" in err
    assert "merged row-by-row" in err


def test_cli_validate_does_not_warn_on_a_normal_time_column(tmp_path, capsys):
    """A plain, regular time axis (no duplicates) must not trip the merged-block check —
    the negative case for the test above."""
    import numpy as np
    n = 3000
    _write_layout_csv(tmp_path / "clean.csv", n, time_col=np.arange(n) / 1000.0)
    toml = _validate_settings_toml(tmp_path, tmp_path)
    rc = cli_main(["validate", str(toml)])
    assert rc == 0
    assert "merged row-by-row" not in capsys.readouterr().err


def test_cli_validate_warns_about_a_constant_assigned_channel(tmp_path, capsys):
    """Acceptance criterion 2: an assigned channel that never varies (here Pdi, wired
    to a genuinely all-zero column — the reported bug's own failure mode) must be
    flagged by name. A non-flow constant channel is advisory only, though (a
    permanently unused pressure port is a legitimate real setup) — reported, but does
    NOT fail validate (see the constant-FLOW test below for the case that does)."""
    import numpy as np
    n = 2000
    _write_layout_csv(tmp_path / "flatpdi.csv", n, time_col=np.arange(n) / 1000.0,
                      extra_cols={"pdi": np.zeros(n)})
    toml = _validate_settings_toml(tmp_path, tmp_path)
    rc = cli_main(["validate", str(toml)])
    assert rc == 0
    err = capsys.readouterr().err
    assert "flatpdi.csv" in err
    assert "Pdi" in err
    assert "constant channel" in err


def test_cli_validate_fails_on_a_constant_flow_channel(tmp_path, capsys):
    """Unlike a non-flow constant channel above, a constant FLOW channel really does
    fail the run (`ConstantFlowError`), so `respmech validate` fails on it too."""
    import numpy as np
    n = 2000
    _write_layout_csv(tmp_path / "flatflow.csv", n, time_col=np.arange(n) / 1000.0,
                      extra_cols={"flow": np.zeros(n)})
    toml = _validate_settings_toml(tmp_path, tmp_path)
    rc = cli_main(["validate", str(toml)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "flatflow.csv" in err
    assert "Flow" in err


def _emg_only_settings_toml(tmp_path, folder, filename, *, method="whole_file"):
    """A minimal, standalone EMG-only Settings object (never the committed synth_case_*
    golden inputs, which have real flow/pressure channels and must stay untouched) —
    saved to a TOML the CLI can load, matching ``_validate_settings_toml``'s own
    'build a minimal settings file, never hand-edit the committed golden' convention."""
    from respmech.core.settings import Settings
    from respmech.settingsio.toml_io import save_toml
    s = Settings()
    s.input.folder = str(folder)
    s.input.files = filename
    s.input.format.sampling_frequency = 1000
    s.input.channels.emg = [2, 3, 4]
    s.analysis.signals = ["emg"]
    s.processing.segmentation.method = method
    s.output.folder = str(tmp_path / "out")
    toml = tmp_path / "s.toml"
    save_toml(s, toml)
    return toml


def test_cli_validate_warns_about_a_constant_emg_channel_on_a_flow_bearing_set(tmp_path, capsys):
    """The existing, unaffected case: a constant EMG channel on an ORDINARY (flow-
    declared) set is advisory only — same as a constant Pdi above — never fails
    validate. Negative case for the EMG-only test below, which DOES fail on this."""
    import numpy as np
    n = 2000
    _write_layout_csv(tmp_path / "flatemg.csv", n, time_col=np.arange(n) / 1000.0,
                      extra_cols={"emg1": np.zeros(n)})   # column 7 -- a constant EMG channel
    from respmech.settingsio.toml_io import save_toml
    legacy = {"input": {"inputfolder": str(tmp_path), "files": "*.csv",
                        "format": {"samplingfrequency": 1000},
                        "data": {"column_flow": 2, "column_volume": 3,
                                 "column_poes": 4, "column_pgas": 5, "column_pdi": 6,
                                 "columns_emg": [7], "columns_entropy": []}},
             "output": {"outputfolder": str(tmp_path / "out")}}
    settings, _ = migrate_dict(legacy)
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)
    rc = cli_main(["validate", str(toml)])
    assert rc == 0
    err = capsys.readouterr().err
    assert "EMG #1" in err
    assert "constant channel" in err


def test_cli_validate_fails_on_a_constant_emg_channel_for_an_emg_only_set(tmp_path, capsys):
    """Acceptance criterion: with NO flow declared at all (EMG-only), a constant
    EMG channel is the hard failure — there is no flow channel for the ordinary rule to
    even look at."""
    import numpy as np
    n = 2000
    t = np.arange(n) / 1000.0
    pd.DataFrame({"time": t, "EMG1": np.zeros(n),
                 "EMG2": np.sin(t), "EMG3": np.cos(t)}).to_csv(
        tmp_path / "flatemgonly.csv", index=False)
    toml = _emg_only_settings_toml(tmp_path, tmp_path, "flatemgonly.csv")
    rc = cli_main(["validate", str(toml)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "flatemgonly.csv" in err
    assert "EMG #1" in err


def test_cli_validate_passes_an_emg_only_set_with_no_constant_channel(tmp_path, capsys):
    """The negative case: a well-behaved EMG-only recording (no constant channel) must
    still validate cleanly, on either segmentation method."""
    import numpy as np
    n = 2000
    t = np.arange(n) / 1000.0
    pd.DataFrame({"time": t, "EMG1": np.sin(t), "EMG2": np.sin(t + 0.5),
                 "EMG3": np.cos(t)}).to_csv(tmp_path / "okemgonly.csv", index=False)
    toml = _emg_only_settings_toml(tmp_path, tmp_path, "okemgonly.csv", method="separators")
    rc = cli_main(["validate", str(toml)])
    assert rc == 0
    err = capsys.readouterr().err
    assert "constant channel" not in err


def test_cli_validate_the_golden_synthetic_files_have_no_new_caveats(tmp_path, capsys):
    """Acceptance criterion 1's negative case, against the committed golden input rather
    than a hand-built file: the real synth_case_*.csv recordings (with their real
    channel assignment) must trip neither new check."""
    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    from respmech.settingsio.toml_io import save_toml
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)
    rc = cli_main(["validate", str(toml)])
    assert rc == 0
    err = capsys.readouterr().err
    assert "merged row-by-row" not in err
    assert "constant channel" not in err


def test_cli_validate_prints_the_signal_set_summary_line(capsys):
    """M-13 acceptance criterion 1, against examples/settings.toml (the full pressure
    family + EMG + a 3-column entropy set, all DERIVED -- no [analysis] table in that
    file): 'respmech validate' prints a Signals/Entropy/Analyses line and exits 0.
    Nothing is off (every core signal is declared), so there is no '· off:' segment."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    example = os.path.join(repo_root, "examples", "settings.toml")
    rc = cli_main(["validate", example])
    assert rc == 0
    out = capsys.readouterr().out
    assert ("Signals: flow, poes, pgas, pdi, emg (derived) · Entropy: 3 columns · "
           "Analyses: Breath timing, Work of breathing, Gastric pressure, "
           "Transdiaphragmatic pressure, Ventilatory muscle ratio, EMG, Sample entropy"
           in out)
    assert "off:" not in out


def test_cli_validate_names_off_signals_for_a_reduced_set(tmp_path, capsys):
    """A signal set that excludes Pgas/Pdi reports them as 'off ... (not in signal set)'
    -- the wording `respmech validate` uses is distinct from the Provenance sheet's
    own 'absent by signal set' (see test_core_outputs.py's sibling test)."""
    from respmech.settingsio.toml_io import save_toml
    s = synth_settings(tmp_path, channels={"pgas": None, "pdi": None})
    toml = tmp_path / "s.toml"
    save_toml(s, toml)
    rc = cli_main(["validate", str(toml)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Signals: flow, poes, emg (derived)" in out
    assert "· off: Pgas/Pdi (not in signal set)" in out
    assert "absent by signal set" not in out


def test_cli_run_dry_run_prints_the_analyses_line(tmp_path, capsys):
    """M-13: `respmech run --dry-run` prints the same 'Analyses: ...' line the GUI's
    commitment sheet shows for the identical settings (run_screen.py's
    `_update_commitment`), right after the output plan's Total line."""
    from respmech.settingsio.toml_io import save_toml
    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)
    rc = cli_main(["run", str(toml), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert ("Analyses: Breath timing, Work of breathing, Gastric pressure, "
           "Transdiaphragmatic pressure, Ventilatory muscle ratio, EMG, Sample entropy"
           in out)
    # Between the output plan's Total line and the blank line before per-file counts.
    total_idx = out.index("Total:")
    analyses_idx = out.index("Analyses:")
    assert total_idx < analyses_idx


def test_cli_init_writes_a_template_that_validates_immediately_with_all_flags(tmp_path):
    """With --folder/--files/--fs all given, the written file needs no further editing
    at all -- load_toml + validate() succeed unedited."""
    from respmech.settingsio.toml_io import load_toml
    out = tmp_path / "new.toml"
    rc = cli_main(["init", str(out), "--signals", "flow,poes,emg",
                  "--folder", str(INPUT), "--files", "synth_case_*.csv", "--fs", "1000"])
    assert rc == 0
    assert out.exists()
    s = load_toml(str(out))
    s.validate()
    assert s.analysis.signals == ["flow", "poes", "emg"]
    assert s.input.channels.pgas is None and s.input.channels.pdi is None
    assert s.input.channels.emg == [4]


def test_cli_init_writes_a_windows_style_folder_as_valid_toml(tmp_path):
    """A `--folder` containing backslashes (a raw Windows path, e.g. a user pasting
    `C:\\Users\\Emil\\data`) must round-trip through the generated TOML file unchanged
    -- a naive f-string into a double-quoted TOML string would leave `\\U`/`\\d`/etc,
    which `tomllib` rejects outright as an invalid escape. Parsed directly with
    `tomllib` here (not `load_toml`, which rebases a RELATIVE folder against the
    file's own directory using `os.path.isabs` -- true for a Windows path only on
    Windows itself, so that rebase step would be a second, unrelated variable on this
    Linux test runner)."""
    import tomllib
    out = tmp_path / "win.toml"
    win_folder = r"C:\Users\Emil\data"
    rc = cli_main(["init", str(out), "--signals", "flow",
                  "--folder", win_folder, "--files", "*.csv", "--fs", "1000"])
    assert rc == 0
    with open(out, "rb") as f:
        data = tomllib.load(f)       # would raise TOMLDecodeError before the fix
    assert data["input"]["folder"] == win_folder


def test_toml_string_falls_back_to_an_escaped_basic_string_for_an_embedded_quote():
    """`_toml_string`'s own unit-level contract: a literal (single-quoted) string for
    the common case (no escaping needed, works unchanged for a Windows `\\` path), a
    properly escaped basic (double-quoted) string as the fallback for a value that
    itself contains a literal `'` (which a literal string has no escape for)."""
    import tomllib
    from respmech.cli.__main__ import _toml_string

    win_path = r"C:\Users\Emil\data"
    quoted_win = _toml_string(win_path)
    assert quoted_win == r"'C:\Users\Emil\data'"
    assert tomllib.loads(f"x = {quoted_win}")["x"] == win_path

    tricky = "O'Brien's folder"
    quoted = _toml_string(tricky)
    assert quoted.startswith('"') and quoted.endswith('"')
    assert tomllib.loads(f"x = {quoted}")["x"] == tricky


def test_cli_init_writes_a_template_that_validates_after_filling_fs_and_folder(tmp_path):
    """Acceptance criterion 2, verbatim: with NO --folder/--files/--fs, the written file
    is not yet runnable (Settings.validate() requires sampling_frequency) -- but becomes
    so, unedited otherwise, once fs and the folder/files are filled in. The channel
    column numbers and [analysis] signals the template already wrote need no editing."""
    from respmech.settingsio.toml_io import load_toml
    from respmech.core.settings import SettingsError
    out = tmp_path / "new.toml"
    rc = cli_main(["init", str(out), "--signals", "poes,flow"])
    assert rc == 0
    s = load_toml(str(out))
    with pytest.raises(SettingsError, match="sampling_frequency"):
        s.validate()

    s.input.folder = str(INPUT)
    s.input.files = "synth_case_*.csv"
    s.input.format.sampling_frequency = 1000
    s.validate()                              # now accepts it, unedited otherwise
    assert sorted(s.analysis.signals) == ["flow", "poes"]


def test_cli_init_writes_only_the_relevant_channel_entries(tmp_path):
    """'kun relevante sektioner' (only relevant sections): an EMG-only template carries
    no flow/poes/pgas/pdi channel keys at all, and validates with the whole_file
    segmentation method it wrote for the flow-less case."""
    from respmech.settingsio.toml_io import load_toml
    out = tmp_path / "emg_only.toml"
    rc = cli_main(["init", str(out), "--signals", "emg",
                  "--folder", str(INPUT), "--files", "*.csv", "--fs", "1000"])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "flow" not in text.split("[input.channels]")[1].split("[processing")[0]
    s = load_toml(str(out))
    s.validate()
    assert s.processing.segmentation.method == "whole_file"
    assert s.input.channels.flow is None


def test_cli_init_rejects_an_unknown_signal(tmp_path, capsys):
    rc = cli_main(["init", str(tmp_path / "x.toml"), "--signals", "flow,made_up"])
    assert rc == 2
    assert "unknown signal" in capsys.readouterr().err
    assert not (tmp_path / "x.toml").exists()


def test_cli_init_rejects_an_empty_signal_set(tmp_path, capsys):
    """An all-comma, entirely-empty --signals is the ONE case that hits `cmd_init`'s
    FIRST rule (mirrors Settings.validate()'s own `not declared` check exactly --
    core/settings.py) -- not any non-empty set, however incomplete, since a non-empty
    --signals always yields a non-empty `declared` set (see the next test)."""
    rc = cli_main(["init", str(tmp_path / "x.toml"), "--signals", ",,"])
    assert rc == 2
    assert capsys.readouterr().err == (
        "error: --signals must name at least one of 'flow' or 'emg'\n")
    assert not (tmp_path / "x.toml").exists()


def test_cli_init_rejects_a_pressure_signal_without_flow(tmp_path, capsys):
    """'poes' alone is a NON-empty declared set ({'poes'}), so this hits `cmd_init`'s
    SECOND rule (poes/pgas/pdi require flow), word for word what
    `Settings.validate()` itself raises for the equivalent saved TOML -- NOT the
    first, 'at least one of flow or emg' rule, which only ever fires for a truly
    empty set (see the sibling test above)."""
    rc = cli_main(["init", str(tmp_path / "x.toml"), "--signals", "poes"])
    assert rc == 2
    assert capsys.readouterr().err == (
        "error: --signals: 'poes', 'pgas' and 'pdi' require 'flow'\n")
    assert not (tmp_path / "x.toml").exists()


# --- `respmech breaths`, check_links in validate, IC/FVC ref lines in dry-run ---------

def _breath_numbers(out: str) -> set:
    import re
    return {int(n) for n in re.findall(r"(?:Breath|Segment) #(\d+):", out)}


def test_cli_breaths_numbers_equal_a_real_runs_breath_keys(tmp_path, capsys):
    """The numbers `respmech breaths` prints for a file are EXACTLY
    `FileResult.breaths`' own keys for a real `run_batch` of the same settings --
    both build breaths through `core.pipeline.segment_file` (`run_batch`'s main loop
    calls it directly), so this pins that the CLI command never drifts from what an
    actual run would number."""
    from respmech.settingsio.toml_io import save_toml
    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)

    result = run_batch(settings)
    ref_keys = set(result.ok_files["synth_case_A.csv"].breaths)

    rc = cli_main(["breaths", str(toml), "synth_case_A.csv"])
    assert rc == 0
    out = capsys.readouterr().out
    assert _breath_numbers(out) == ref_keys
    assert ref_keys, "the comparison isn't vacuous"
    # only the requested file is listed
    assert "synth_case_B.csv" not in out


def test_cli_breaths_with_no_file_argument_lists_every_matched_file(tmp_path, capsys):
    from respmech.settingsio.toml_io import save_toml
    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)
    rc = cli_main(["breaths", str(toml)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "synth_case_A.csv:" in out
    assert "synth_case_B.csv:" in out


def test_cli_breaths_reports_onset_duration_kind_and_exclusion(tmp_path, capsys):
    """A breath typed 'ic' (every typed kind is unioned into excludebreaths) shows up
    with its own kind and as excluded; an ordinary breath shows kind=tidal and no
    'excluded' marker."""
    from respmech.settingsio.toml_io import save_toml
    from respmech.core.settings import BreathTypeEntry

    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    settings.processing.breath_types.append(
        BreathTypeEntry(file="synth_case_A.csv", breath=4, kind="ic"))
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)

    rc = cli_main(["breaths", str(toml), "synth_case_A.csv"])
    assert rc == 0
    out = capsys.readouterr().out
    line4 = next(l for l in out.splitlines() if l.strip().startswith("Breath #4:"))
    assert "kind=ic" in line4 and "excluded" in line4
    line1 = next(l for l in out.splitlines() if l.strip().startswith("Breath #1:"))
    assert "kind=tidal" in line1 and "excluded" not in line1
    assert "onset" in line1 and "duration" in line1


def test_cli_breaths_suggested_fvc_hint_drives_a_ready_to_paste_snippet(tmp_path, capsys):
    """`core.analysis.manoeuvres.suggest_fvc` (the longest untyped expiration) marks
    one breath 'suggested FVC' and the printed TOML snippet uses that SAME breath
    number, filled in (not a placeholder), plus `folder` -- ready to paste unedited."""
    from respmech.settingsio.toml_io import save_toml
    from respmech.core.analysis.manoeuvres import suggest_fvc
    from respmech.core.pipeline import segment_file
    from respmech.core._legacy_ns import to_legacy_ns
    import os as _os

    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)

    s = to_legacy_ns(settings)
    path = _os.path.join(settings.input.folder, "synth_case_A.csv")
    breaths, _trimmed = segment_file(settings, s, path)
    expected_hint = suggest_fvc(breaths)
    assert expected_hint is not None, "the comparison isn't vacuous"

    rc = cli_main(["breaths", str(toml), "synth_case_A.csv"])
    assert rc == 0
    out = capsys.readouterr().out
    hint_line = next(l for l in out.splitlines()
                     if l.strip().startswith(f"Breath #{expected_hint}:"))
    assert "suggested FVC" in hint_line
    assert "[[processing.breath_types]]" in out
    assert 'file = "synth_case_A.csv"' in out
    assert f"breath = {expected_hint}" in out
    assert 'kind = "fvc"' in out
    assert f"folder = '{settings.input.folder}'" in out


def test_cli_breaths_rejects_an_unmatched_file_argument(tmp_path, capsys):
    from respmech.settingsio.toml_io import save_toml
    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)
    rc = cli_main(["breaths", str(toml), "does_not_exist.csv"])
    assert rc == 2
    assert "does_not_exist.csv" in capsys.readouterr().err


def test_cli_validate_prints_check_links_cautions_as_warnings_but_stays_ok(tmp_path, capsys):
    """A reference source not among the analysed files is a soft check_links caution
    (require_references is off by default) -- printed, but `validate` still exits 0,
    per LungVolumeSettings.require_references' own doctrine: "an unresolved link is a
    caution plus NaN and a notice; it is a blocker only when this flag is set"."""
    from respmech.settingsio.toml_io import save_toml
    from respmech.core.settings import ReferenceEntry, BreathRef

    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    settings.processing.references.append(ReferenceEntry(
        file="synth_case_A.csv", ic=BreathRef(file="missing_source.csv", breaths=[2])))
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)

    rc = cli_main(["validate", str(toml)])
    assert rc == 0
    err = capsys.readouterr().err
    assert "WARNING: processing.references[synth_case_A.csv]: ic source " \
          "'missing_source.csv' is not among the analysed files" in err


def test_cli_validate_fails_when_a_required_reference_source_is_missing(tmp_path, capsys):
    """The SAME missing-source caution as above, but with require_references on --
    the one condition that fails `validate`'s own exit code, mirroring
    ui.validation.path_problem's existing hard block exactly (the same
    missing_reference_sources() check, the same require_references gate)."""
    from respmech.settingsio.toml_io import save_toml
    from respmech.core.settings import ReferenceEntry, BreathRef

    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    settings.processing.references.append(ReferenceEntry(
        file="synth_case_A.csv", ic=BreathRef(file="missing_source.csv", breaths=[2])))
    settings.processing.lung_volume.require_references = True
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)

    rc = cli_main(["validate", str(toml)])
    assert rc == 1


def test_cli_validate_a_caution_with_no_missing_source_never_fails_even_when_required(
        tmp_path, capsys):
    """require_references only turns a MISSING source into a hard failure -- a
    caution of a different kind (here: an excluded linked breath, the source file
    itself present and fine) never fails validate, required or not."""
    from respmech.settingsio.toml_io import save_toml
    from respmech.core.settings import ReferenceEntry, BreathRef, ExcludeEntry

    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    settings.processing.references.append(ReferenceEntry(
        file="synth_case_A.csv", ic=BreathRef(file="synth_case_B.csv", breaths=[2])))
    settings.processing.exclude_breaths.append(
        ExcludeEntry(file="synth_case_B.csv", breaths=[2]))
    settings.processing.lung_volume.require_references = True
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)

    rc = cli_main(["validate", str(toml)])
    err = capsys.readouterr().err
    assert "is excluded" in err
    assert rc == 0


def test_cli_run_dry_run_shows_the_ic_and_fvc_reference_line_per_file(tmp_path, capsys):
    """cmd_run's own acceptance example: 'IC ref: ... · FVC ref: ...' for a file whose
    IC reference resolved (here: the file's own typed IC/FVC breaths -- 'own', not the
    filename repeated back), and 'no IC reference — lung-volume columns NaN' for a
    file in the SAME analysis (the IC family IS present) whose own IC never resolves."""
    from respmech.settingsio.toml_io import save_toml
    from respmech.core.settings import BreathTypeEntry

    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    settings.processing.breath_types.append(
        BreathTypeEntry(file="synth_case_A.csv", breath=4, kind="ic"))
    settings.processing.breath_types.append(
        BreathTypeEntry(file="synth_case_A.csv", breath=7, kind="fvc"))
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)

    rc = cli_main(["run", str(toml), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "IC ref: own #4 · FVC ref: own #7" in out
    assert "no IC reference — lung-volume columns NaN" in out


def test_cli_run_dry_run_omits_the_reference_line_when_no_ic_family_exists(tmp_path, capsys):
    """The ordinary case (no cross-file/own-typed IC reference anywhere in this
    analysis) must not print a spurious 'no IC reference' line for every file --
    `_legacy` (this file's own default fixture) types no breath at all."""
    from respmech.settingsio.toml_io import save_toml
    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)
    rc = cli_main(["run", str(toml), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "IC ref:" not in out
    assert "no IC reference" not in out


def test_cli_run_dry_run_shows_a_cross_file_reference_with_one_hash_per_breath(tmp_path, capsys):
    """_ref_label's multi-breath format is '#2,#3' (one '#' per breath number), not
    '#2,3' -- and for a genuinely CROSS-file link (the source is a different file
    than the one being reported on), the label names that file, not 'own'."""
    from respmech.settingsio.toml_io import save_toml
    from respmech.core.settings import BreathTypeEntry, ReferenceEntry, BreathRef

    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    settings.processing.breath_types.append(
        BreathTypeEntry(file="synth_case_B.csv", breath=2, kind="ic"))
    settings.processing.breath_types.append(
        BreathTypeEntry(file="synth_case_B.csv", breath=3, kind="ic"))
    settings.processing.references.append(ReferenceEntry(
        file="synth_case_A.csv", ic=BreathRef(file="synth_case_B.csv", breaths=[2, 3])))
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)

    rc = cli_main(["run", str(toml), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    # synth_case_A.csv's OWN reference line names the source file, not 'own' -- its
    # IC comes from breaths typed in B, not from a breath typed in A itself.
    # (synth_case_B.csv gets its OWN, separate reference line further down that DOES
    # say 'own' -- B's IC breaths are B's own typed breaths, a different, equally
    # valid resolution for a different file; this test only pins A's line.)
    a_idx = next(i for i, l in enumerate(lines) if l.strip() == "synth_case_A.csv: 8 breaths")
    assert lines[a_idx + 1].strip() == "IC ref: synth_case_B.csv #2,#3 · FVC ref: none"


def test_cli_validate_passes_cleanly_when_required_references_all_resolve(tmp_path, capsys):
    """require_references only fails validate on a MISSING source -- a fully
    resolving reference set (here: own-typed IC/FVC breaths) must validate clean,
    with no cautions at all, even with the flag on."""
    from respmech.settingsio.toml_io import save_toml
    from respmech.core.settings import BreathTypeEntry

    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    settings.processing.breath_types.append(
        BreathTypeEntry(file="synth_case_A.csv", breath=4, kind="ic"))
    settings.processing.lung_volume.require_references = True
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)

    rc = cli_main(["validate", str(toml)])
    assert rc == 0
    assert capsys.readouterr().err == ""


def test_cli_breaths_names_an_emg_only_analysiss_entries_segment_not_breath(tmp_path, capsys):
    """An EMG-only signal set has no inspiration/expiration split
    (core.analysis.segments._make_segment's own 'Segment #N' naming, has_phases=False)
    -- `respmech breaths` must print exactly what the breath dict itself is named,
    never a hardcoded 'Breath #N'. `suggest_fvc` has nothing eligible on such a set
    (it always requires has_phases), so this also exercises the '(no suggested FVC
    breath)' fallback line."""
    import numpy as np
    from respmech.core.settings import Settings
    from respmech.settingsio.toml_io import save_toml

    n = 2000
    t = np.arange(n) / 1000.0
    pd.DataFrame({"time": t, "EMG1": np.sin(t), "EMG2": np.sin(t + 0.5),
                 "EMG3": np.cos(t)}).to_csv(tmp_path / "okemgonly.csv", index=False)
    s = Settings()
    s.input.folder = str(tmp_path)
    s.input.files = "okemgonly.csv"
    s.input.format.sampling_frequency = 1000
    s.input.channels.emg = [2, 3, 4]
    s.analysis.signals = ["emg"]
    s.processing.segmentation.method = "whole_file"
    s.output.folder = str(tmp_path / "out")
    toml = tmp_path / "s.toml"
    save_toml(s, toml)

    rc = cli_main(["breaths", str(toml)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Segment #1:" in out
    assert "Breath #1:" not in out
    assert "(no suggested FVC breath)" in out


def test_cli_breaths_reports_an_error_for_one_file_and_still_lists_the_next(tmp_path, capsys):
    """A file that fails to segment (here: a constant flow channel, the same
    ConstantFlowError `respmech validate` already fails on) prints an ERROR line for
    THAT file and a non-zero exit code, but does not abort the whole command -- the
    next matched file is still listed in full."""
    import numpy as np
    n = 2000
    _write_layout_csv(tmp_path / "aaa_flatflow.csv", n, time_col=np.arange(n) / 1000.0,
                      extra_cols={"flow": np.zeros(n)})
    _write_layout_csv(tmp_path / "zzz_clean.csv", n, time_col=np.arange(n) / 1000.0)
    toml = _validate_settings_toml(tmp_path, tmp_path)

    rc = cli_main(["breaths", str(toml)])
    assert rc == 1
    captured = capsys.readouterr()
    out, err = captured.out, captured.err
    assert "aaa_flatflow.csv:" in out
    assert "zzz_clean.csv:" in out
    assert "ERROR: ConstantFlowError" in err
    # the good file after the bad one is still listed with real breath lines
    assert "onset" in out.split("zzz_clean.csv:", 1)[1]


def test_cli_breaths_matches_a_resampled_runs_breath_count_and_duration(tmp_path, capsys):
    """A pre-analysis resample must change what `respmech breaths` segments too --
    without applying the same resample override `run_batch` applies before its own
    `segment_file` call, this command would segment the file's NATIVE rate while a
    real run segments the RESAMPLED one, printing different sample counts/durations
    for the exact same settings."""
    from respmech.settingsio.toml_io import save_toml

    settings, _ = migrate_dict(_legacy(str(tmp_path)))
    settings.processing.sampling.resample = True
    settings.processing.sampling.resample_to_frequency = 500   # native is 1000 Hz
    toml = tmp_path / "s.toml"
    save_toml(settings, toml)

    result = run_batch(settings)
    ref_breaths = result.ok_files["synth_case_A.csv"].breaths
    ref_keys = set(ref_breaths)
    ref_samples_breath1 = len(ref_breaths[1]["time"])

    rc = cli_main(["breaths", str(toml), "synth_case_A.csv"])
    assert rc == 0
    out = capsys.readouterr().out
    assert _breath_numbers(out) == ref_keys
    # onset/duration for breath #1 must reflect the RESAMPLED rate (500 Hz), not the
    # native 1000 Hz -- duration = sample count / fs, so a wrong fs would print a
    # duration exactly twice (or half) what the resampled run actually has.
    import re
    m = re.search(r"Breath #1:.*duration ([\d.]+)s", out)
    assert m is not None
    printed_duration = float(m.group(1))
    expected_duration = ref_samples_breath1 / 500
    assert abs(printed_duration - expected_duration) < 0.01
