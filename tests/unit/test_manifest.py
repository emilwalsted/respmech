"""The batch manifest (ticket B01): FileEntry/Manifest/build_manifest/narrow_mask, all
Qt-free — no ``qapp`` fixture anywhere in this file, by design (the whole point of the
ticket is that this model can be built and asserted on without a QApplication)."""
import os

import pytest

from respmech.settingsio.migrate import migrate_dict
from respmech.core.settings import Settings
from respmech.ui.manifest import Manifest, build_manifest, group_readout, narrow_mask

from _helpers import write_delim as _write_delim, write_xlsx as _write_xlsx


def _settings(folder, mask, fs=1000, decimal="."):
    # NB the legacy/migration field is "decimalcharacter", not "decimal" — migrate_dict
    # silently ignores an unrecognised key and falls back to "." rather than raising, so a
    # typo here would fail quietly (caught while writing the cache-key regression test
    # below, which needs a real comma-decimal Settings to be meaningful).
    legacy = {"input": {"inputfolder": folder, "files": mask,
                        "format": {"samplingfrequency": fs, "decimalcharacter": decimal}},
             "output": {"outputfolder": os.path.join(folder, "out")}}
    s, _ = migrate_dict(legacy)
    return s


# --------------------------------------------------------------------------- #
# narrow_mask
# --------------------------------------------------------------------------- #
def test_narrow_mask_is_a_noop_for_a_single_pattern(tmp_path):
    _write_delim(tmp_path / "a.csv", 9)
    narrowed, from_, dropped = narrow_mask(str(tmp_path), "*.csv")
    assert narrowed == "*.csv" and from_ is None and dropped == {}


def test_narrow_mask_picks_the_dominant_extension(tmp_path):
    for n in ("a", "b", "c"):
        _write_delim(tmp_path / f"{n}.csv", 9)
    _write_delim(tmp_path / "d.txt", 9, sep="\t")
    narrowed, from_, dropped = narrow_mask(str(tmp_path), "*.csv; *.txt")
    assert narrowed == "*.csv"
    assert from_ == "*.csv; *.txt"
    assert dropped == {".txt": 1}


def test_narrow_mask_is_a_noop_without_a_folder():
    narrowed, from_, dropped = narrow_mask("", "*.csv; *.txt")
    assert narrowed == "*.csv; *.txt" and from_ is None and dropped == {}


# --------------------------------------------------------------------------- #
# build_manifest — uniform / narrower outlier / wider outlier / mixed extensions
# --------------------------------------------------------------------------- #
def test_manifest_uniform_folder_has_no_caveats(tmp_path):
    for n in ("a", "b", "c"):
        _write_delim(tmp_path / f"{n}.csv", 9)
    s = _settings(str(tmp_path), "*.csv")
    m = build_manifest(str(tmp_path), "*.csv", s)
    assert isinstance(m, Manifest)
    assert len(m.files) == 3
    assert m.majority_columns == 9
    assert m.outliers == ()
    assert m.freq_mismatches == ()
    assert m.mask_narrowed_from is None
    assert m.is_clean


def test_manifest_flags_a_narrower_outlier_file(tmp_path):
    for n in ("a", "b", "c"):
        _write_delim(tmp_path / f"{n}.csv", 9)
    _write_delim(tmp_path / "d.csv", 8)          # narrower than the majority
    s = _settings(str(tmp_path), "*.csv")
    m = build_manifest(str(tmp_path), "*.csv", s)
    assert m.majority_columns == 9
    names = {f.filename for f in m.outliers}
    assert names == {"d.csv"}
    assert not m.is_clean
    outlier = m.outliers[0]
    assert outlier.columns == 8 and not outlier.included
    assert "8" in outlier.exclude_reason and "9" in outlier.exclude_reason
    # the outlier is invisible to the included set — B02/B03 and the channel dialog must
    # never see it as part of the batch
    assert "d.csv" not in {f.filename for f in m.included_files}


def test_manifest_flags_a_wider_outlier_file(tmp_path):
    for n in ("a", "b", "c"):
        _write_delim(tmp_path / f"{n}.csv", 9)
    _write_delim(tmp_path / "d.csv", 12)         # wider than the majority
    s = _settings(str(tmp_path), "*.csv")
    m = build_manifest(str(tmp_path), "*.csv", s)
    assert m.majority_columns == 9
    assert {f.filename for f in m.outliers} == {"d.csv"}


def test_manifest_tie_break_prefers_the_widest_layout(tmp_path):
    """Mirrors _valid_input_files's existing tie-break rule exactly: an even split gives
    the WIDEST layout the majority, not the narrowest and not the alphabetically-first."""
    _write_delim(tmp_path / "a.csv", 8)
    _write_delim(tmp_path / "b.csv", 9)
    s = _settings(str(tmp_path), "*.csv")
    m = build_manifest(str(tmp_path), "*.csv", s)
    assert m.majority_columns == 9
    assert {f.filename for f in m.outliers} == {"a.csv"}


def test_manifest_degenerate_one_column_files_never_win_the_majority_vote(tmp_path):
    """Self-review regression (05-08-2026): _valid_input_files filters `columns >= 2`
    BEFORE voting on a majority (a 1-column file carries no signal beyond the time axis);
    build_manifest originally did not, so a folder where degenerate files outnumbered the
    real recordings could crown 1-column THE majority and mark the genuine data files as
    outliers — the opposite of what _valid_input_files (and thus the channel-assignment
    dialog's own file list) would say. The two must never disagree about which files are
    'the batch'."""
    for n in ("a", "b", "c", "d"):                  # 4 degenerate files outvote 3 real ones
        _write_delim(tmp_path / f"junk{n}.csv", 1)
    for n in ("x", "y", "z"):
        _write_delim(tmp_path / f"rec{n}.csv", 9)
    s = _settings(str(tmp_path), "*.csv")
    m = build_manifest(str(tmp_path), "*.csv", s)
    assert m.majority_columns == 9
    included = {f.filename for f in m.included_files}
    assert included == {"recx.csv", "recy.csv", "recz.csv"}
    outliers = {f.filename for f in m.outliers}
    assert outliers == {"junka.csv", "junkb.csv", "junkc.csv", "junkd.csv"}


def test_manifest_all_files_degenerate_has_no_majority(tmp_path):
    """Mirrors _valid_input_files's `if not probed: return []` for the all-degenerate case:
    no file can win a majority no candidate is eligible for."""
    for n in ("a", "b"):
        _write_delim(tmp_path / f"{n}.csv", 1)
    s = _settings(str(tmp_path), "*.csv")
    m = build_manifest(str(tmp_path), "*.csv", s)
    assert m.majority_columns is None
    assert len(m.outliers) == 2


def test_manifest_cache_key_separates_decimal_separators(tmp_path):
    """Self-review regression (05-08-2026): the probe cache is keyed on file identity
    (path, mtime, size) alone; a decimal-separator change (auto-detected mid-session, or
    edited by hand) changes how peek_columns/probe_sampling_frequency PARSE the same
    unchanged file, so the decimal separator must be part of the cache key too, or a
    rebuild after such a change could silently serve back a value probed under the OLD
    separator."""
    p = tmp_path / "a.csv"
    # a semicolon-delimited, comma-decimal file: 3 columns under decimal=',' (';'-split),
    # but reads as 1 run-on column under decimal='.' (','-split treats the whole line as
    # one field once the header has no commas of its own)
    p.write_text("time;a;b\n0,000;1,5;2,5\n0,010;1,6;2,6\n0,020;1,7;2,7\n"
                 * 10, encoding="utf-8")
    cache = {}
    s_comma = _settings(str(tmp_path), "*.csv", decimal=",")
    m1 = build_manifest(str(tmp_path), "*.csv", s_comma, cache=cache)
    assert m1.files[0].columns == 3
    s_dot = _settings(str(tmp_path), "*.csv", decimal=".")
    m2 = build_manifest(str(tmp_path), "*.csv", s_dot, cache=cache)     # SAME cache dict
    assert m2.files[0].columns == 1                # not the stale, comma-decimal 3


def test_manifest_never_probes_xlsx_sampling_frequency(tmp_path):
    """Self-review regression (05-08-2026): probe_sampling_frequency's nrows=5000 cap does
    NOT bound an .xlsx read's cost the way it does for the CSV/TXT C-engine (openpyxl
    parses the whole sheet before pandas applies the cap) — measured up to ~300 ms/file,
    scaling with true row count, which a hundred-file batch could turn into tens of seconds
    of a frozen GUI on every input edit. Excel input therefore never gets a frequency
    caution — column-count/outlier detection is unaffected, only the (secondary)
    frequency-mismatch note is skipped for this format."""
    pytest.importorskip("openpyxl")
    from respmech.ui.workers import probe_sampling_frequency
    _write_xlsx(tmp_path / "a.xlsx", 9, fs=1000.0)
    s = _settings(str(tmp_path), "*.xlsx", fs=1000)
    assert probe_sampling_frequency(s, str(tmp_path / "a.xlsx")) is None
    m = build_manifest(str(tmp_path), "*.xlsx", s)
    assert m.files[0].detected_fs is None
    assert m.freq_mismatches == ()          # never flagged either way for this format


def test_manifest_mixed_extensions_narrows_and_records_the_drop(tmp_path):
    for n in ("a", "b", "c"):
        _write_delim(tmp_path / f"{n}.csv", 9)
    _write_delim(tmp_path / "d.txt", 9, sep="\t")
    _write_delim(tmp_path / "e.txt", 9, sep="\t")
    s = _settings(str(tmp_path), "*.csv; *.txt")
    m = build_manifest(str(tmp_path), "*.csv; *.txt", s)
    assert m.mask == "*.csv"
    assert m.mask_narrowed_from == "*.csv; *.txt"
    assert m.narrowed_out_exts == (".txt",)
    assert m.narrowed_out_count == 2
    assert len(m.files) == 3                      # the .txt files never even get probed
    assert not m.is_clean


def test_manifest_unreadable_file_is_excluded_not_crashed_on(tmp_path):
    for n in ("a", "b"):
        _write_delim(tmp_path / f"{n}.csv", 9)
    (tmp_path / "broken.csv").write_bytes(b"\x00\x01\x02not,a,real,csv\xff\xfe")
    s = _settings(str(tmp_path), "*.csv")
    m = build_manifest(str(tmp_path), "*.csv", s)
    assert m.majority_columns == 9
    broken = next(f for f in m.files if f.filename == "broken.csv")
    # a byte-peek always returns SOME column split for garbage bytes (it just isn't the
    # majority) rather than raising — either way it must never be counted as included
    assert not broken.included


# --------------------------------------------------------------------------- #
# build_manifest — sampling-frequency mismatch
# --------------------------------------------------------------------------- #
def test_manifest_flags_a_frequency_mismatch(tmp_path):
    _write_delim(tmp_path / "at1000.csv", 9, fs=1000.0)
    _write_delim(tmp_path / "at2000.csv", 9, fs=2000.0)
    s = _settings(str(tmp_path), "*.csv", fs=1000)
    m = build_manifest(str(tmp_path), "*.csv", s)
    assert m.majority_columns == 9                # both files still agree on columns
    assert len(m.included_files) == 2
    mismatched = {f.filename for f in m.freq_mismatches}
    assert mismatched == {"at2000.csv"}
    assert not m.is_clean
    ok = next(f for f in m.files if f.filename == "at1000.csv")
    assert ok.detected_fs == 1000
    bad = next(f for f in m.files if f.filename == "at2000.csv")
    assert bad.detected_fs == 2000


def test_manifest_outlier_files_are_never_frequency_probed(tmp_path):
    """Frequency is only useful information for files that are actually going to run —
    an outlier is already excluded on column count alone."""
    for n in ("a", "b", "c"):
        _write_delim(tmp_path / f"{n}.csv", 9, fs=1000.0)
    _write_delim(tmp_path / "d.csv", 8, fs=2000.0)     # narrower AND a different rate
    s = _settings(str(tmp_path), "*.csv", fs=1000)
    m = build_manifest(str(tmp_path), "*.csv", s)
    outlier = next(f for f in m.files if f.filename == "d.csv")
    assert not outlier.included
    assert outlier.detected_fs is None             # never probed — would be a wasted read


# --------------------------------------------------------------------------- #
# .xlsx: no longer misreported via a meaningless byte-peek
# --------------------------------------------------------------------------- #
def test_manifest_reads_real_xlsx_column_count(tmp_path):
    pytest.importorskip("openpyxl")
    _write_xlsx(tmp_path / "a.xlsx", 9)
    _write_xlsx(tmp_path / "b.xlsx", 9)
    s = _settings(str(tmp_path), "*.xlsx")
    m = build_manifest(str(tmp_path), "*.xlsx", s)
    assert m.majority_columns == 9
    assert m.outliers == ()


# --------------------------------------------------------------------------- #
# performance instrument: the byte-peek must be what counts columns for .csv/.txt
# --------------------------------------------------------------------------- #
def test_manifest_never_calls_probe_data_columns_for_a_csv_folder(tmp_path):
    """The byte-peek (peek_columns) is the instrument for .csv/.txt — probe_data_columns
    (46 ms/file) must stay cold on the path that runs on every input edit."""
    from respmech.ui import workers
    for n in ("a", "b", "c"):
        _write_delim(tmp_path / f"{n}.csv", 9)
    s = _settings(str(tmp_path), "*.csv")
    calls = {"n": 0}
    real = workers.probe_data_columns

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    workers.probe_data_columns = counting
    try:
        m = build_manifest(str(tmp_path), "*.csv", s)   # uses the real default probers
    finally:
        workers.probe_data_columns = real
    assert calls["n"] == 0
    assert m.majority_columns == 9


def test_manifest_rebuild_over_an_unchanged_folder_reprobes_nothing(tmp_path):
    from respmech.ui import workers
    for n in ("a", "b", "c"):
        _write_delim(tmp_path / f"{n}.csv", 9)
    s = _settings(str(tmp_path), "*.csv")

    col_calls, fs_calls = {"n": 0}, {"n": 0}
    real_peek, real_fs = workers.peek_columns, workers.probe_sampling_frequency

    def counting_peek(*a, **k):
        col_calls["n"] += 1
        return real_peek(*a, **k)

    def counting_fs(*a, **k):
        fs_calls["n"] += 1
        return real_fs(*a, **k)

    cache = {}
    build_manifest(str(tmp_path), "*.csv", s, columns_prober=counting_peek,
                   freq_prober=counting_fs, cache=cache)
    first_cols, first_fs = col_calls["n"], fs_calls["n"]
    assert first_cols == 3 and first_fs == 3

    build_manifest(str(tmp_path), "*.csv", s, columns_prober=counting_peek,
                   freq_prober=counting_fs, cache=cache)
    assert col_calls["n"] == first_cols            # not re-probed
    assert fs_calls["n"] == first_fs                # not re-probed


def test_manifest_cache_is_invalidated_when_a_file_changes(tmp_path):
    """The memo is keyed on (path, mtime, size) — editing a file after a first build must
    still be picked up by a second build over the same cache, not silently stay stale."""
    p = tmp_path / "a.csv"
    _write_delim(p, 9)
    _write_delim(tmp_path / "b.csv", 9)
    _write_delim(tmp_path / "c.csv", 9)
    s = _settings(str(tmp_path), "*.csv")
    cache = {}
    m1 = build_manifest(str(tmp_path), "*.csv", s, cache=cache)
    assert m1.majority_columns == 9
    _write_delim(p, 7)                              # rewrite a.csv with fewer columns
    m2 = build_manifest(str(tmp_path), "*.csv", s, cache=cache)
    a = next(f for f in m2.files if f.filename == "a.csv")
    assert a.columns == 7 and not a.included


# --------------------------------------------------------------------------- #
# header-block detection (ticket D01): a LabChart-style export's preamble
# (Interval=/ExcelDateTime=/ChannelTitle=/Range= lines) can sniff as a LOW column count
# that is nonetheless the MAJORITY across a batch sharing the same instrument export
# shape — outliers alone can never catch this, because there is no minority to exclude.
# --------------------------------------------------------------------------- #
def test_manifest_flags_a_majority_consistent_header_block_as_a_warning(tmp_path):
    header = "Interval=\t0.001 s\nChannelTitle=\tFlow\tPoes\tPgas\tPdi\n"
    rows = "\n".join(f"{i * 0.001:.3f}\t{i}\t{i + 1}\t{i + 2}\t{i + 3}" for i in range(10))
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text(header + rows + "\n")
    s = _settings(str(tmp_path), "*.txt")
    m = build_manifest(str(tmp_path), "*.txt", s)
    assert m.majority_columns == 2                   # both files' first line has 2 fields
    assert m.outliers == ()                           # no minority to exclude — both agree
    assert {f.filename for f in m.header_warnings} == {"a.txt", "b.txt"}
    assert not m.is_clean


def test_manifest_a_clean_folder_has_no_header_warnings(tmp_path):
    for n in ("a", "b", "c"):
        _write_delim(tmp_path / f"{n}.csv", 9)
    s = _settings(str(tmp_path), "*.csv")
    m = build_manifest(str(tmp_path), "*.csv", s)
    assert m.header_warnings == ()
    assert m.is_clean


def test_manifest_header_prober_is_injectable_and_skipped_for_excluded_files(tmp_path):
    """Mirrors the existing columns_prober/freq_prober injection tests: a caller can assert
    on/count calls without touching the real byte-peek, and — like freq_prober — the header
    probe never runs for a file the majority vote already excluded (its own column count,
    not its head, is why it failed, so its head is not worth reading)."""
    for n in ("a", "b", "c"):
        _write_delim(tmp_path / f"{n}.csv", 9)
    _write_delim(tmp_path / "d.csv", 8)               # column-count outlier, excluded
    s = _settings(str(tmp_path), "*.csv")
    seen = []

    def fake_header_prober(_settings, path):
        seen.append(os.path.basename(path))
        return None

    m = build_manifest(str(tmp_path), "*.csv", s, header_prober=fake_header_prober)
    assert sorted(seen) == ["a.csv", "b.csv", "c.csv"]   # d.csv (excluded) never probed
    assert m.header_warnings == ()


def test_manifest_header_prober_result_is_cached(tmp_path):
    _write_delim(tmp_path / "a.csv", 9)
    s = _settings(str(tmp_path), "*.csv")
    calls = {"n": 0}

    def counting(_settings, _path):
        calls["n"] += 1
        return None

    cache = {}
    build_manifest(str(tmp_path), "*.csv", s, header_prober=counting, cache=cache)
    build_manifest(str(tmp_path), "*.csv", s, header_prober=counting, cache=cache)
    assert calls["n"] == 1                             # second build served from cache


def test_manifest_does_not_false_flag_a_clean_wide_file(tmp_path):
    """Self-review regression: peek_header_warning only sniffs 8 KB, and a WIDE recording's
    own row can exceed that on its own (measured: a 500-column row is ~5.4 KB) — the second
    row then gets truncated mid-line inside the byte-peek, which used to read as a false
    'the first two rows disagree on the number of fields' on a perfectly uniform file."""
    ncols = 500
    header = "\t".join(["time"] + [f"c{i}" for i in range(2, ncols + 1)])
    row = "\t".join(f"{i:.3f}" for i in range(ncols))
    (tmp_path / "wide.txt").write_text(header + "\n" + (row + "\n") * 5)
    s = _settings(str(tmp_path), "*.txt")
    m = build_manifest(str(tmp_path), "*.txt", s)
    assert m.majority_columns == ncols
    assert m.header_warnings == ()
    assert m.is_clean


def test_manifest_supplying_all_three_probers_never_imports_workers(tmp_path, monkeypatch):
    """The docstring's promise ('inject your own probers to test this function without
    importing workers at all') must still hold now that there are THREE probers, not two —
    self-review finding after ``header_prober`` was added: the import guard must stay
    per-argument, not regress to 'import if ANY default is used' once a caller supplies
    every one of them."""
    import sys
    _write_delim(tmp_path / "a.csv", 9)
    s = _settings(str(tmp_path), "*.csv")
    monkeypatch.delitem(sys.modules, "respmech.ui.workers", raising=False)
    m = build_manifest(str(tmp_path), "*.csv", s,
                       columns_prober=lambda _s, _p: 9,
                       freq_prober=lambda _s, _p: None,
                       header_prober=lambda _s, _p: None)
    assert "respmech.ui.workers" not in sys.modules
    assert m.majority_columns == 9


# --------------------------------------------------------------------------- #
# empty / missing folder
# --------------------------------------------------------------------------- #
def test_manifest_over_a_missing_folder_is_empty_not_an_error(tmp_path):
    s = _settings(str(tmp_path), "*.csv")
    m = build_manifest(str(tmp_path / "does-not-exist"), "*.csv", s)
    assert m.files == () and m.majority_columns is None and m.is_clean


def test_manifest_with_no_matching_files_is_empty(tmp_path):
    s = _settings(str(tmp_path), "*.nope")
    m = build_manifest(str(tmp_path), "*.nope", s)
    assert m.files == () and m.majority_columns is None


# --------------------------------------------------------------------------- #
# group_readout (ticket D19: a live pre-run read-out of what "Group files by" does)
# --------------------------------------------------------------------------- #
_GROUP_FILES = ["P01_rest.csv", "P01_60W.csv", "P02_120W.csv"]


def test_group_readout_is_empty_and_muted_for_no_files():
    status, text = group_readout([], Settings())
    assert status == "muted" and text == ""


def test_group_readout_with_blank_pattern_uses_the_leading_token_default():
    """No ``output.group_regex`` set: falls back to ``core.summary.group_key``'s own
    default (the leading filename token), which the ticket's own worked example gives
    as P01, P01, P02 -- two groups."""
    status, text = group_readout(_GROUP_FILES, Settings())
    assert status == "info"
    assert "3 files" in text and "2 groups" in text
    assert "P01 (2)" in text and "P02 (1)" in text


def test_group_readout_warns_and_names_the_files_a_partial_pattern_pools_as_all():
    """The ticket's own worked example: ``_([a-z]+)\\.csv`` matches P01_rest.csv (key
    'rest') but neither P01_60W.csv nor P02_120W.csv (digits/uppercase aren't [a-z]) --
    both pool into "(all)", and the read-out must say so and name them, not just quietly
    report '2 groups' as if nothing were wrong."""
    s = Settings()
    s.output.group_regex = r"_([a-z]+)\.csv"
    status, text = group_readout(_GROUP_FILES, s)
    assert status == "warn"
    assert "(all)" in text
    assert "P01_60W.csv" in text and "P02_120W.csv" in text
    assert "P01_rest.csv" not in text   # the one file that DID match must not be named as a straggler
    assert "2 files do not match" in text   # plural


def test_group_readout_warns_with_correct_singular_grammar_for_one_straggler():
    """Self-review fix (10-08-2026): the info branch already got singular/plural right
    ('1 file -> 1 group'); the warn branch's message did not ('1 file do not match')."""
    s = Settings()
    s.output.group_regex = r"_([a-z]+)\.csv"
    status, text = group_readout(["P01_rest.csv", "P02_60W.csv"], s)
    assert status == "warn"
    assert "1 file does not match" in text


def test_group_readout_does_not_warn_when_a_files_own_leading_token_is_the_all_sentinel():
    """Self-review fix (10-08-2026): under the DEFAULT (blank-pattern) grouping,
    core.summary.group_key's leading-token fallback returns the token verbatim and never
    itself produces the literal string "(all)" -- that string is only ever a SENTINEL
    from the regex branch's own "nothing matched" fallback. A file coincidentally named
    so its leading token happens to spell "(all)" must therefore be reported as a normal,
    successfully-grouped file, not as a straggler from a pattern that (with no regex set)
    does not even exist."""
    status, text = group_readout(["(all)_rest.csv", "(all)_60W.csv"], Settings())
    assert status == "info"
    assert "do not match" not in text
    assert "(all) (2)" in text   # the group name IS the string "(all)", correctly reported


def test_group_readout_reports_a_clear_error_for_an_invalid_pattern_without_crashing():
    """An unparsable regex must not silently fall back to pooling everything as "(all)"
    the way core.summary.group_key itself does (that fallback exists so a bad pattern
    never crashes a RUN, and stays untouched) -- the read-out re-validates independently
    so it can show the user re.error's own message instead."""
    s = Settings()
    s.output.group_regex = "[unclosed"
    status, text = group_readout(_GROUP_FILES, s)
    assert status == "error"
    assert text and "unterminated" in text.lower()   # re.error's own wording for this pattern


def test_group_readout_reports_an_invalid_pattern_even_with_no_files_matched_yet():
    """Self-review fix (10-08-2026): the pattern is now validated BEFORE the no-files
    check. An unparsable regex is knowable without any files at all, and a user who has
    typed one deserves to be told, rather than getting silence indistinguishable from
    'nothing scanned yet' -- the original ordering reported "muted"/"" here."""
    s = Settings()
    s.output.group_regex = "[unclosed"
    status, text = group_readout([], s)
    assert status == "error"
    assert "unterminated" in text.lower()


def test_group_readout_truncates_a_large_group_count_like_named_list_does():
    names = [f"P{n:02d}_rest.csv" for n in range(1, 9)]   # 8 distinct leading tokens
    status, text = group_readout(names, Settings(), group_limit=6)
    assert status == "info"
    assert "8 files" in text and "8 groups" in text
    assert "+2 more" in text


def test_group_readout_truncates_the_straggler_list_separately_from_the_group_list():
    """file_limit (stragglers, a filename list) and group_limit (group summaries, short
    'NAME (n)' tokens) are independent -- a batch that produces few groups but many
    straggler filenames should truncate the filenames at file_limit regardless of
    group_limit."""
    names = [f"P{n:02d}.dat" for n in range(1, 6)]   # none of these end in .csv -> all "(all)"
    s = Settings()
    s.output.group_regex = r"\.csv$"
    status, text = group_readout(names, s, file_limit=2)
    assert status == "warn"
    assert "5 files do not match" in text
    assert "+3 more" in text
