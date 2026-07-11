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
    # P11 diagnostic figures land under diagnostics/
    assert any(p.endswith(".png") for p in written)
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
