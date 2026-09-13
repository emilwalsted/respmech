"""Unit tests for core/quality.py's two data-quality predicates (ticket 20260913-2054)
and the core.io.loaders probes that feed them from a file on disk.

Background: a user's LabChart CSV was actually three merged blocks (duplicated
timestamps) with two constant assigned channels (Pdi, and an unused column) — respmech
reported only a misleading "incomplete breath at the start or end" for one downstream
breath, and it took a manual data investigation to find the real cause. These two
checks exist to name the real cause up front instead. The real, local-only data file
never reaches this repo (see CLAUDE.md); everything here is synthetic.
"""
import numpy as np
import pandas as pd
import pytest

from respmech.core.quality import detect_constant_channel, detect_merged_time_blocks
from respmech.core.settings import Settings
from respmech.core.io.loaders import probe_constant_channels, probe_merged_time_blocks


# --------------------------------------------------------------------------- #
# detect_merged_time_blocks — pure predicate
# --------------------------------------------------------------------------- #
def test_detects_a_typical_merged_block_signature():
    """Models the reported case: three LabChart blocks (700s/103.6s/33.25s at 1000 Hz)
    merged row-by-row by timestamp. Simplified to two blocks here — the mechanism (a
    stretch of duplicated timestamps at the head of an otherwise regular axis) is the
    same regardless of how many blocks overlap."""
    n = 5000
    t_clean = np.arange(n) / 1000.0
    t_merged = np.sort(np.concatenate([t_clean[:1000], t_clean]))   # first second tripled... 2x
    msg = detect_merged_time_blocks(t_merged)
    assert msg is not None
    assert "merged row-by-row" in msg
    assert "1000 Hz" in msg


def test_a_clean_regular_time_axis_is_not_flagged():
    t = np.arange(5000) / 1000.0
    assert detect_merged_time_blocks(t) is None


def test_a_sample_index_column_is_not_mistaken_for_merged_time():
    """An integer step >= 1.0 reads as a sample-index column, not seconds — mirrors
    ui.workers.detect_sampling_frequency's own guard exactly, so the two never disagree
    about what counts as 'looks like time' in the first place."""
    idx = np.arange(5000, dtype=float)     # step = 1.0, an index, not seconds
    assert detect_merged_time_blocks(idx) is None


def test_irregular_spacing_is_not_mistaken_for_merged_time():
    """A column that is not a clean sample clock in the first place (e.g. an actual
    signal channel, not a time axis) must not be flagged just because it happens to
    have some repeated/decreasing samples — there is no 'regular time axis' claim to
    contradict here."""
    rng = np.random.RandomState(0)
    noisy = np.cumsum(rng.uniform(-1, 2, 5000))   # wildly irregular steps, some negative
    assert detect_merged_time_blocks(noisy) is None


def test_a_single_duplicate_sample_in_a_huge_file_is_not_flagged():
    """Neither threshold alone is enough — a lone rounding-glitch duplicate must not
    trip this (min_duplicates), even though it would clear the fraction threshold on
    its own in a large enough file only if that file were tiny; conversely a huge file
    with exactly one duplicate never clears the fraction floor either. Confirms both
    guards are load-bearing, not just one."""
    t = np.arange(20000) / 1000.0
    t = t.copy()
    t[500] = t[499]                       # exactly one duplicated timestamp
    assert detect_merged_time_blocks(t) is None


def test_a_couple_of_duplicates_in_a_tiny_file_is_not_flagged():
    t = np.arange(25) / 1000.0
    t = t.copy()
    t[10] = t[9]
    t[11] = t[9]                          # 2 duplicates in a 25-sample file (>2% fraction)
    assert detect_merged_time_blocks(t) is None   # below the absolute min_duplicates floor


def test_golden_synthetic_inputs_are_never_flagged():
    """The committed golden characterisation-test inputs (real, physiologically-modelled
    synthetic recordings, not hand-built edge cases) must never trip this — the negative
    case that matters most, since these files feed the golden reference suite."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(here, "..", "golden", "input")
    for name in ("synth_case_A.csv", "synth_case_B.csv"):
        path = os.path.join(input_dir, name)
        if not os.path.exists(path):
            pytest.skip("golden input absent")
        t = pd.read_csv(path).iloc[:, 0].to_numpy(dtype=float)
        assert detect_merged_time_blocks(t) is None, f"{name} false-flagged"


# --------------------------------------------------------------------------- #
# detect_constant_channel — pure predicate
# --------------------------------------------------------------------------- #
def test_an_all_zero_channel_is_constant():
    assert detect_constant_channel(np.zeros(500)) is True


def test_an_all_nonzero_constant_channel_is_constant():
    """Not a relative check — a channel that only ever reads a fixed non-zero value is
    just as much a mis-assigned/dead channel as one reading zero."""
    assert detect_constant_channel(np.full(500, 5.0)) is True


def test_a_varying_channel_is_not_constant():
    assert detect_constant_channel(np.sin(np.linspace(0, 10, 500))) is False


def test_nan_samples_are_ignored_not_treated_as_variation():
    values = np.full(500, 2.0)
    values[::10] = np.nan
    assert detect_constant_channel(values) is True


def test_an_entirely_nan_channel_is_not_reported_as_constant():
    """Nothing to compare — a genuinely unreadable/empty channel is a different,
    already-reported problem (core.io.loaders.validatedata's NaN check)."""
    assert detect_constant_channel(np.full(500, np.nan)) is False


def test_golden_synthetic_inputs_have_no_constant_channels():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(here, "..", "golden", "input")
    for name in ("synth_case_A.csv", "synth_case_B.csv"):
        path = os.path.join(input_dir, name)
        if not os.path.exists(path):
            pytest.skip("golden input absent")
        df = pd.read_csv(path)
        for col in df.columns[1:]:                 # skip time
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            assert not detect_constant_channel(values), f"{name}:{col} false-flagged"


# --------------------------------------------------------------------------- #
# core.io.loaders probes — file-reading wrappers around the two predicates above
# --------------------------------------------------------------------------- #
def _settings(folder, *, flow=2, poes=3, pgas=4, pdi=5, volume=None, emg=None, entropy=None, fs=1000):
    s = Settings()
    s.input.folder = folder
    s.input.files = "*.csv"
    s.input.format.sampling_frequency = fs
    ch = s.input.channels
    ch.flow, ch.poes, ch.pgas, ch.pdi = flow, poes, pgas, pdi
    ch.volume = volume
    ch.emg = emg or []
    ch.entropy = entropy or []
    return s


def test_probe_merged_time_blocks_reads_column_zero(tmp_path):
    n = 5000
    t_clean = np.arange(n) / 1000.0
    t_merged = np.sort(np.concatenate([t_clean[:1000], t_clean]))
    path = tmp_path / "merged.csv"
    pd.DataFrame({"time": t_merged, "flow": np.zeros(len(t_merged))}).to_csv(path, index=False)
    s = _settings(str(tmp_path))
    msg = probe_merged_time_blocks(s, str(path))
    assert msg is not None and "merged row-by-row" in msg


def test_probe_merged_time_blocks_none_for_xlsx(tmp_path):
    """No cheap capped read is available for .xlsx (see the probe's own docstring) —
    it must return None rather than raise."""
    path = tmp_path / "f.xlsx"
    n = 100
    pd.DataFrame({"time": np.arange(n) / 1000.0, "flow": np.zeros(n)}).to_excel(path, index=False)
    s = _settings(str(tmp_path))
    assert probe_merged_time_blocks(s, str(path)) is None


def test_probe_constant_channels_names_channel_and_column(tmp_path):
    n = 2000
    path = tmp_path / "flat.csv"
    pd.DataFrame({
        "time": np.arange(n) / 1000.0,
        "flow": np.sin(np.linspace(0, 10, n)),
        "poes": np.linspace(-5, -3, n),
        "pgas": np.linspace(6, 8, n),
        "pdi": np.zeros(n),
    }).to_csv(path, index=False)
    s = _settings(str(tmp_path), pdi=5)
    out = probe_constant_channels(s, str(path))
    assert out == ("Pdi (column 5)",)


def test_probe_constant_channels_checks_every_assigned_role(tmp_path):
    """flow/poes/pgas/pdi/EMG/entropy are all checked -- not just the four singles."""
    n = 500
    path = tmp_path / "many_flat.csv"
    pd.DataFrame({
        "time": np.arange(n) / 1000.0,
        "flow": np.sin(np.linspace(0, 10, n)),
        "poes": np.linspace(-5, -3, n),
        "pgas": np.linspace(6, 8, n),
        "pdi": np.linspace(11, 13, n),
        "emg1": np.zeros(n),                # constant EMG channel
        "ent1": np.full(n, 3.0),            # constant entropy channel
    }).to_csv(path, index=False)
    s = _settings(str(tmp_path), emg=[6], entropy=[7])
    out = probe_constant_channels(s, str(path))
    assert out == ("EMG #1 (column 6)", "Entropy #1 (column 7)")


def test_probe_constant_channels_empty_when_nothing_assigned(tmp_path):
    n = 100
    path = tmp_path / "f.csv"
    pd.DataFrame({"time": np.arange(n) / 1000.0, "a": np.zeros(n)}).to_csv(path, index=False)
    s = Settings()
    s.input.format.sampling_frequency = 1000
    assert probe_constant_channels(s, str(path)) == ()


def test_probe_constant_channels_ignores_an_out_of_range_column(tmp_path):
    """A channel assigned past the file's actual column count is a real mismatch,
    reported elsewhere (core.io.loaders._column at run time) -- this probe must not
    raise on it, just skip it (rather than, say, wrapping around to the last column)."""
    n = 100
    path = tmp_path / "f.csv"
    pd.DataFrame({"time": np.arange(n) / 1000.0,
                 "flow": np.sin(np.linspace(0, 10, n))}).to_csv(path, index=False)
    s = _settings(str(tmp_path), flow=2, poes=99, pgas=None, pdi=None)
    out = probe_constant_channels(s, str(path))
    assert out == ()
