"""Unit tests for core/quality.py's two data-quality predicates (13-09-2026)
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


def test_a_duplicated_sample_index_column_is_also_not_flagged():
    """The bare index above (no duplicates at all) would pass even without the
    integer-step guard, simply because it never reaches the duplicate-counting logic —
    this is the guard's actual load-bearing case: an index column that ALSO has a few
    duplicated/restarting values (e.g. a re-started counter across merged blocks) must
    still read as 'not a time axis' rather than being flagged."""
    n = 100000
    idx = np.arange(n, dtype=float)
    for i in (1000, 50000, 90000):
        idx[i] = idx[i - 1]
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
    """A lone rounding-glitch duplicate must not trip this on its own -- both the
    absolute floor (1 < min_duplicates) and the fraction floor (1/19999 << 2%) reject
    it independently here, so this alone does not prove either threshold is load-
    bearing (see the two isolating tests below for that)."""
    t = np.arange(20000) / 1000.0
    t = t.copy()
    t[500] = t[499]                       # exactly one duplicated timestamp
    assert detect_merged_time_blocks(t) is None


def test_min_duplicates_floor_is_load_bearing_in_isolation():
    """Isolates ONLY the absolute floor: 2 duplicates, spread out so the surviving
    positive steps stay uniform (ratio well under the 0.25 irregularity guard) and
    ``min_duplicate_fraction`` is passed as 0 so the fraction test can never be what
    rejects this on its own. If ``min_duplicates`` were removed or miscompared, this
    would flag -- confirmed by deleting the check and re-running by hand."""
    n = 1000
    t = np.arange(n) / 1000.0
    t = t.copy()
    t[500] = t[499]
    t[700] = t[699]                        # 2 duplicates, below min_duplicates=3
    assert detect_merged_time_blocks(t, min_duplicate_fraction=0) is None


def test_min_duplicate_fraction_floor_is_load_bearing_in_isolation():
    """Isolates ONLY the fraction floor: 3 duplicates (clears the default
    ``min_duplicates``) spread across a 100,000-sample file so the fraction
    (3/99999 ≈ 0.003%) stays far below the 2% default -- with ``min_duplicates``
    passed as 1 so the absolute floor can never be what rejects this on its own."""
    n = 100000
    t = np.arange(n) / 1000.0
    t = t.copy()
    for i in (1000, 50000, 90000):
        t[i] = t[i - 1]
    assert detect_merged_time_blocks(t, min_duplicates=1) is None


def test_a_couple_of_duplicates_in_a_tiny_file_is_not_flagged():
    t = np.arange(25) / 1000.0
    t = t.copy()
    t[10] = t[9]
    t[11] = t[9]
    assert detect_merged_time_blocks(t) is None


def test_a_coarse_time_column_uniform_end_to_end_is_not_flagged():
    """A time column printed at coarser precision than its true sampling interval (a
    real, plain export quirk -- e.g. 2000 Hz rounded to 3 decimal places) gives a high,
    but UNIFORM, rate of zero-diffs end to end. This must not read as a merged block:
    a genuine merge's duplicate run ends once only one recording is left, so its own
    tail is clean, which this column's is not (measured: ~50% non-increasing steps
    throughout, tail included -- the exact false positive this check exists to avoid)."""
    fs = 2000
    n = 60000
    t = np.round(np.arange(n) / fs, 3)
    assert detect_merged_time_blocks(t) is None


def test_a_genuine_merge_with_a_clean_tail_is_still_flagged_despite_the_coarse_check():
    """The tail-cleanliness requirement added for the case above must not itself
    swallow the reported shape: a real interleaved merge (duplicates concentrated
    early, clean for the rest of the file) still gets flagged."""
    n = 5000
    t_clean = np.arange(n) / 1000.0
    t_merged = np.sort(np.concatenate([t_clean[:1000], t_clean]))
    msg = detect_merged_time_blocks(t_merged)
    assert msg is not None and "merged row-by-row" in msg


def test_concatenated_blocks_each_restarting_their_own_time_base_are_flagged():
    """The OTHER real shape a multi-block export can take: blocks simply appended one
    after another (LabChart's ordinary 'export all blocks' behaviour), each restarting
    near t=0 -- a single large BACKWARD JUMP per boundary, not a run of duplicates.
    Confirmed this shape slips past the duplicate-based check alone (2-4 jumps in a
    multi-thousand-sample file clear neither the count nor the fraction floor) before
    this dedicated jump check was added."""
    block1 = np.arange(700000) / 1000.0
    block2 = np.arange(103600) / 1000.0
    block3 = np.arange(33250) / 1000.0
    t = np.concatenate([block1, block2, block3])
    msg = detect_merged_time_blocks(t)
    assert msg is not None
    assert "restarting its own time base" in msg
    assert "sample 700000" in msg
    assert "700" in msg and "s" in msg          # jump magnitude (~700 s, the first block's length)


def test_a_small_forward_gap_is_not_mistaken_for_a_restarted_block():
    """A legitimate PAUSE in acquisition (a forward jump -- missing data, never
    analysed as breaths either way) must not trip the backward-jump check, which only
    ever looks at NEGATIVE steps."""
    t = np.concatenate([np.arange(5000) / 1000.0, 100.0 + np.arange(5000) / 1000.0])
    assert detect_merged_time_blocks(t) is None


def test_an_implausible_inferred_rate_is_not_flagged_either_way():
    """Consistency with ui.workers.detect_sampling_frequency's own plausibility band
    (10-200000 Hz): a column that otherwise looks regular but implies a rate outside
    that band is not trustworthy as 'looks like time' in the first place, so neither
    check fires on it -- the two functions must never disagree about this. A step of
    0.5 s (2 Hz, below the 10 Hz floor) is deliberately non-integer, so this is not
    accidentally caught by the earlier sample-index guard instead (which would make
    this pass for the wrong reason)."""
    t = np.arange(5000) * 0.5                # step 0.5 s -> 2 Hz, below the 10 Hz floor
    assert detect_merged_time_blocks(t) is None


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


def test_probe_constant_channels_does_not_crash_on_a_nan_channel_setting(tmp_path):
    """A channel setting can carry ``float('nan')`` for "not really assigned" rather
    than ``None`` (mirrors ``core.io.loaders._column``'s own guard, which handles both)
    -- a hand-edited or migrated TOML is the realistic source. Before the NaN guard,
    ``if col`` (NaN is truthy) let this straight through to a crashing
    ``df.iloc[:, nan - 1]``."""
    n = 100
    path = tmp_path / "f.csv"
    pd.DataFrame({"time": np.arange(n) / 1000.0,
                 "flow": np.sin(np.linspace(0, 10, n))}).to_csv(path, index=False)
    s = _settings(str(tmp_path), flow=2, poes=float("nan"), pgas=None, pdi=None)
    assert probe_constant_channels(s, str(path)) == ()


def test_probe_constant_channels_does_not_crash_on_a_float_channel_setting(tmp_path):
    """A plain (non-integer) float column setting -- e.g. ``flow = 2.0`` surviving a
    hand-written TOML -- must be treated as column 2, not crash or silently mismatch."""
    n = 100
    path = tmp_path / "f.csv"
    pd.DataFrame({"time": np.arange(n) / 1000.0,
                 "flow": np.zeros(n)}).to_csv(path, index=False)
    s = _settings(str(tmp_path), flow=2.0, poes=None, pgas=None, pdi=None)
    assert probe_constant_channels(s, str(path)) == ("Flow (column 2)",)
