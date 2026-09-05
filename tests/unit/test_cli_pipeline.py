"""End-to-end-ish tests for the pipeline, writers and CLI on the committed
synthetic data (no external data needed)."""
import os

import pandas as pd
import pytest

from respmech.cli.__main__ import main as cli_main
from respmech.core.io.writers import write_batch
from respmech.core.pipeline import run_batch
from respmech.settingsio.migrate import migrate_dict
from _helpers import INPUT, requires_synth, synth_legacy_dict  # noqa: F401


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
