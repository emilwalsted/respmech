"""Cross-platform (macOS ↔ Windows) portability guards.

These lock the fixes that keep a run reproducible regardless of the OS the operator
uses: encoding tolerance (Excel "Unicode Text" UTF-16 / European cp1252 exports),
case-insensitive + metacharacter-safe file matching (macOS glob is case-sensitive,
Windows is not — so a mixed-case batch would otherwise process a *different* file set,
and the shared EMG noise profile would diverge numerically between the two platforms),
the configured decimal separator, uppercase file extensions, and rebasing an analysis
file's relative folders against its own location rather than the process CWD (a frozen
app's CWD is System32 / the install dir). Qt-free except the prefs test.
"""
import os

import numpy as np
import pandas as pd
import pytest

from respmech.core.io.loaders import _read_table, _checkcolumn, DataValidationError
from respmech.core.pipeline import match_input_files


# --- encoding tolerance -----------------------------------------------------
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1252", "utf-16"])
def test_read_table_reads_every_common_encoding(tmp_path, encoding):
    """A header carrying non-ASCII characters (µ, ø) only decodes under the right codec;
    the tolerant reader must handle all four the field throws at us without raising."""
    df_out = pd.DataFrame({"Poøs [µV]": [1.0, 2.0, 3.0], "Flow [L/s]": [0.1, -0.2, 0.3]})
    p = tmp_path / "data.csv"
    p.write_bytes(df_out.to_csv(index=False).encode(encoding))
    df = _read_table(str(p))
    # positional numeric access (exactly what core.load does) survives any header codec
    assert df.shape == (3, 2)
    assert df.iloc[:, 1].tolist() == [0.1, -0.2, 0.3]


def test_read_table_never_raises_on_undecodable_bytes(tmp_path):
    """latin-1 maps every byte, so a file that is valid under no strict codec still reads
    (header mojibake is harmless — the numeric columns are read positionally)."""
    p = tmp_path / "weird.csv"
    # bytes 0x81/0x8D/0x8F are undefined in cp1252 and invalid utf-8 — placed in the header
    # so only latin-1 can decode the file; the numeric rows stay clean.
    p.write_bytes(bytes([0x81]) + b"a," + bytes([0x8D]) + b"b\n1,2\n3,4\n")
    df = _read_table(str(p))
    assert df.iloc[0, 0] == 1 and df.iloc[0, 1] == 2


def test_read_table_honours_semicolon_and_comma_decimal(tmp_path):
    """European Excel CSV: ';'-separated with a ',' decimal — must parse as real floats,
    not a single mis-split text column."""
    p = tmp_path / "euro.csv"
    p.write_bytes("Flow;Poes\n0,1;-4,5\n0,2;-4,7\n".encode("cp1252"))
    df = _read_table(str(p), sep=";", decimal=",")
    assert df.shape == (2, 2)
    assert df.iloc[:, 0].tolist() == [0.1, 0.2]
    assert df.iloc[:, 1].tolist() == [-4.5, -4.7]


# --- validation: text column must not crash with TypeError -------------------
def test_checkcolumn_reports_text_before_isnan_typeerror():
    """A non-numeric (object/str) column must raise the friendly 'must be numeric' error,
    NOT the TypeError np.isnan throws on an object array."""
    with pytest.raises(DataValidationError, match="text values"):
        _checkcolumn("Flow column", np.array(["a", "b", "c"], dtype=object))
    with pytest.raises(DataValidationError, match="text values"):
        _checkcolumn("Flow column", np.array(["x", "y"]))          # unicode dtype


def test_checkcolumn_flags_nan_and_passes_clean():
    with pytest.raises(DataValidationError, match="NaN"):
        _checkcolumn("Flow column", np.array([1.0, np.nan, 3.0]))
    _checkcolumn("Flow column", np.array([1.0, 2.0, 3.0]))         # clean numeric -> no raise


# --- file matching: case-insensitive, dotfile-aware, metacharacter-safe ------
def test_match_input_files_is_case_insensitive(tmp_path):
    for name in ("Case_A.CSV", "case_b.csv", "notes.txt"):
        (tmp_path / name).write_text("x")
    got = {os.path.basename(p) for p in match_input_files(str(tmp_path), "*.csv")}
    assert got == {"Case_A.CSV", "case_b.csv"}          # both, despite mixed extension case


def test_match_input_files_excludes_dotfiles_unless_pattern_is_dotted(tmp_path):
    (tmp_path / "real.csv").write_text("x")
    (tmp_path / ".hidden.csv").write_text("x")
    assert {os.path.basename(p) for p in match_input_files(str(tmp_path), "*.csv")} == {"real.csv"}
    assert {os.path.basename(p) for p in match_input_files(str(tmp_path), ".*")} == {".hidden.csv"}


def test_match_input_files_safe_against_folder_metacharacters(tmp_path):
    """A folder whose NAME contains glob metacharacters ('Study [2024]') makes glob.glob
    match nothing; the os.listdir-based matcher is immune."""
    weird = tmp_path / "Study [2024]"
    weird.mkdir()
    (weird / "a.csv").write_text("x")
    (weird / "b.csv").write_text("x")
    got = {os.path.basename(p) for p in match_input_files(str(weird), "*.csv")}
    assert got == {"a.csv", "b.csv"}


def test_match_input_files_handles_a_subdirectory_pattern(tmp_path):
    """A path-bearing mask ('raw/*.csv', or a Windows 'raw\\*.csv') must resolve the subdir
    against the folder and match the filename part — parity with the old glob.glob(join())."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.csv").write_text("x")
    (raw / "b.csv").write_text("x")
    (tmp_path / "top.csv").write_text("x")   # NOT under raw/ -> excluded
    got = {os.path.basename(p) for p in match_input_files(str(tmp_path), "raw/*.csv")}
    assert got == {"a.csv", "b.csv"}
    # both separators accepted (a mask authored on Windows still works on macOS)
    got_bs = {os.path.basename(p) for p in match_input_files(str(tmp_path), "raw\\*.csv")}
    assert got_bs == {"a.csv", "b.csv"}


def test_match_input_files_returns_sorted_full_paths(tmp_path):
    (tmp_path / "b.csv").write_text("x")
    (tmp_path / "a.csv").write_text("x")
    out = match_input_files(str(tmp_path), "*.csv")
    assert out == sorted(out)                            # deterministic order
    assert all(os.path.isfile(p) for p in out)           # full paths (a drop-in for glob.glob)
    assert os.path.basename(out[0]) == "a.csv"


# --- analysis file: relative folders rebase against the file, not the CWD ----
def test_load_toml_rebases_relative_folders_against_file_dir(tmp_path):
    from respmech.settingsio.toml_io import load_toml, save_toml
    from respmech.core.settings import Settings
    sub = tmp_path / "study"
    sub.mkdir()
    s = Settings()
    s.input.folder = "input"        # the shipped defaults — relative
    s.output.folder = "output"
    save_toml(s, str(sub / "analysis.toml"))
    loaded = load_toml(str(sub / "analysis.toml"))
    assert loaded.input.folder == os.path.normpath(str(sub / "input"))
    assert loaded.output.folder == os.path.normpath(str(sub / "output"))


def test_load_toml_leaves_external_absolute_folders_untouched(tmp_path):
    """An absolute folder OUTSIDE the analysis file's directory is an explicit external
    location and must survive a save→load round trip unchanged (not relativized)."""
    from respmech.settingsio.toml_io import load_toml, save_toml
    from respmech.core.settings import Settings
    ext = tmp_path / "external"
    ext.mkdir()
    abs_in = os.path.normpath(str(ext / "in"))
    sub = tmp_path / "study"
    sub.mkdir()
    s = Settings()
    s.input.folder = abs_in                          # outside sub/ (the .toml's dir)
    s.output.folder = os.path.normpath(str(ext / "out"))
    save_toml(s, str(sub / "a.toml"))
    loaded = load_toml(str(sub / "a.toml"))
    assert loaded.input.folder == abs_in             # external absolute path preserved


def test_toml_open_save_roundtrip_keeps_folders_relative(tmp_path):
    """The portability guard: opening a shared analysis rebases its relative folders to
    absolute (for the run), but saving it back must write them RELATIVE again — otherwise
    the first Open→Save on one machine bakes in machine-local absolute paths and the shared
    file breaks for everyone else."""
    import tomllib
    from respmech.settingsio.toml_io import load_toml, save_toml
    from respmech.core.settings import Settings
    sub = tmp_path / "study"
    sub.mkdir()
    p = str(sub / "analysis.toml")
    s = Settings()
    s.input.folder = "input"                         # the shipped relative defaults
    s.output.folder = "output"
    save_toml(s, p)
    loaded = load_toml(p)                             # rebases to absolute for the run
    assert os.path.isabs(loaded.input.folder) and os.path.isabs(loaded.output.folder)
    save_toml(loaded, p)                             # …and Save must re-relativize
    with open(p, "rb") as f:
        raw = tomllib.load(f)
    assert raw["input"]["folder"] == "input"         # stored value is portable again
    assert raw["output"]["folder"] == "output"


# --- recent-analyses: case-insensitive dedup on a case-insensitive FS --------
def test_recent_analyses_dedups_case_insensitively(qapp, isolated_prefs, monkeypatch):
    """On Windows (case-insensitive FS) the same analysis opened under different casing
    must collapse to one entry — deduped on os.path.normcase, which is a no-op on macOS."""
    monkeypatch.setattr(os.path, "normcase", str.lower)   # simulate the Windows fold
    monkeypatch.setattr(os.path, "isfile", lambda p: True)  # bypass the on-disk existence filter
    prefs = isolated_prefs
    prefs.add_recent_analysis("/study/Analysis.toml")
    prefs.add_recent_analysis("/study/analysis.TOML")     # same file, different case
    assert len(prefs.recent_analyses()) == 1
