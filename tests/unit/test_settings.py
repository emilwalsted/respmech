import pytest

from respmech.core.settings import (
    SCHEMA_VERSION, BreathCountEntry, CarriedOverState, ExcludeEntry, Settings,
    SettingsError, carried_over_state, clear_carried_over, is_carried_folder,
)


def _minimal():
    return {
        "input": {
            "format": {"sampling_frequency": 2000, "matlab_variant": "mac"},
            "channels": {"poes": 7, "pgas": 8, "pdi": 10, "flow": 13, "volume": 14},
        },
    }


def test_defaults_and_parse():
    s = Settings.from_dict(_minimal()).validate()
    assert s.processing.segmentation.method == "flow"
    assert s.processing.wob.calc_from == "average"
    assert s.processing.emg.rms_window_s == 0.050


def test_unknown_keys_are_captured_not_fatal():
    d = _minimal()
    d["processing"] = {"sampling": {"resample": True}, "totally_new_section": {"x": 1}}
    s = Settings.from_dict(d).validate()
    # legacy bug #6: an unknown nested subsection must not crash the loader
    assert "processing.totally_new_section" in s.unknown
    assert s.processing.sampling.resample is True


def test_missing_required_raises():
    with pytest.raises(SettingsError):
        Settings.from_dict({"input": {"format": {"matlab_variant": "mac"}}}).validate()


def test_volume_required_unless_integrated():
    d = _minimal()
    d["input"]["channels"].pop("volume")
    with pytest.raises(SettingsError):
        Settings.from_dict(d).validate()
    d["processing"] = {"volume": {"integrate_from_flow": True}}
    Settings.from_dict(d).validate()  # now OK


def test_enum_validation():
    d = _minimal()
    d["processing"] = {"segmentation": {"method": "sideways"}}
    with pytest.raises(SettingsError):
        Settings.from_dict(d).validate()


def test_ecg_auto_detect_requires_remove_ecg():
    d = _minimal()
    d["processing"] = {"emg": {"ecg_auto_detect": True, "remove_ecg": False}}
    d["input"]["channels"]["emg"] = [1]
    with pytest.raises(SettingsError, match="remove_ecg"):
        Settings.from_dict(d).validate()
    d["processing"]["emg"]["remove_ecg"] = True
    Settings.from_dict(d).validate()  # now OK


def test_ecg_auto_detect_requires_emg_channels():
    d = _minimal()
    d["processing"] = {"emg": {"ecg_auto_detect": True, "remove_ecg": True}}
    with pytest.raises(SettingsError, match="channels.emg"):
        Settings.from_dict(d).validate()
    d["input"]["channels"]["emg"] = [1, 2]
    Settings.from_dict(d).validate()  # now OK


def test_ecg_auto_detect_off_skips_the_cross_check():
    # Off by default: an otherwise-invalid combination (remove_ecg False, no EMG channels)
    # must not be rejected when auto-detect itself is not requested.
    d = _minimal()
    Settings.from_dict(d).validate()


# -- K-222: detect_channel is a 0-based index into input.channels.emg, and the GUI
# already clamps it (Preview & QC) — a hand-written or migrated settings file only
# reaches this validate() call, so core.pipeline used to crash mid-batch with a raw
# IndexError instead.
def test_detect_channel_out_of_range_is_rejected():
    d = _minimal()
    d["input"]["channels"]["emg"] = [1, 2]
    d["processing"] = {"emg": {"detect_channel": 2}}   # valid indices are 0, 1
    with pytest.raises(SettingsError, match="detect_channel"):
        Settings.from_dict(d).validate()


def test_detect_channel_negative_is_rejected():
    d = _minimal()
    d["input"]["channels"]["emg"] = [1, 2]
    d["processing"] = {"emg": {"detect_channel": -1}}
    with pytest.raises(SettingsError, match="detect_channel"):
        Settings.from_dict(d).validate()


def test_detect_channel_in_range_is_accepted():
    d = _minimal()
    d["input"]["channels"]["emg"] = [1, 2]
    d["processing"] = {"emg": {"detect_channel": 1}}
    Settings.from_dict(d).validate()


def test_detect_channel_default_is_fine_with_no_emg_channels():
    # The default (0) with an empty EMG channel list must not be rejected — detect_channel
    # only matters once something actually reads it (remove_ecg, an EMG figure job).
    d = _minimal()
    Settings.from_dict(d).validate()


def test_detect_channel_out_of_range_is_fine_with_no_emg_channels():
    # Even a nonsensical value is allowed with an empty list: the range check is gated on
    # `ch.emg` being truthy, so a negative or huge detect_channel means nothing yet.
    d = _minimal()
    d["processing"] = {"emg": {"detect_channel": -5}}
    Settings.from_dict(d).validate()


# -- K-225: noise reduction requiring ECG removal was only a GUI activation gate
# (screen.py's noise_enabled.setEnabled(has_emg and ecg_on)) — a hand-written settings
# file with noise.enabled = true and remove_ecg = false ran the profile against a
# signal that still contained heartbeats, modelling the cardiac artefact as steady
# background noise. BREAKING per Emil's decision 05-09-2026 — flagged in CHANGELOG.md.
def test_noise_enabled_requires_remove_ecg():
    d = _minimal()
    d["input"]["channels"]["emg"] = [1]
    d["processing"] = {"emg": {"remove_ecg": False,
                               "noise": {"enabled": True, "reference_file": "rest.csv"}}}
    with pytest.raises(SettingsError, match="remove_ecg"):
        Settings.from_dict(d).validate()
    d["processing"]["emg"]["remove_ecg"] = True
    Settings.from_dict(d).validate()  # now OK


def test_noise_disabled_skips_the_remove_ecg_cross_check():
    d = _minimal()
    d["processing"] = {"emg": {"remove_ecg": False, "noise": {"enabled": False}}}
    Settings.from_dict(d).validate()


def test_round_trip():
    d = _minimal()
    d["processing"] = {"exclude_breaths": [{"file": "a.txt", "breaths": [1, 2]}]}
    s = Settings.from_dict(d).validate()
    assert s.processing.exclude_breaths[0].file == "a.txt"
    assert Settings.from_dict(s.to_dict()).to_dict() == s.to_dict()


# -- the end-expiratory trend anchor rule -------------------------------------

def test_trend_threshold_defaults_to_auto():
    """The retired default was an ABSOLUTE 0.8 measured below the recording's global
    maximum, which no ordinary tidal recording can reach; unset means the scale-free
    per-trough rule (see compute.trend_anchors)."""
    v = Settings().processing.volume
    assert v.trend_peak_min_height is None
    assert v.trend_peak_min_prominence_frac == 0.05


def test_auto_threshold_round_trips_as_an_absent_key():
    from respmech.settingsio.toml_io import dumps_toml
    assert "trend_peak_min_height" not in dumps_toml(Settings())
    s = Settings()
    s.processing.volume.trend_peak_min_height = 0.5
    assert "trend_peak_min_height = 0.5" in dumps_toml(s)
    assert Settings.from_dict(s.to_dict()).processing.volume.trend_peak_min_height == 0.5


def test_segmentation_buffer_still_writes_as_samples():
    """D29 (UI-overhaul) renamed the GUI label/tooltip to 'Breath-separation debounce' and
    added a derived seconds hint, but explicitly did NOT touch storage: the TOML key stays
    ``buffer``, and the value stays samples, not seconds — an unedited analysis must write
    the exact same number it always has."""
    from respmech.settingsio.toml_io import dumps_toml
    assert "buffer = 800" in dumps_toml(Settings())
    s = Settings()
    s.processing.segmentation.buffer = 200
    assert "buffer = 200" in dumps_toml(s)
    assert Settings.from_dict(s.to_dict()).processing.segmentation.buffer == 200


def _trend_settings(**vol):
    d = _minimal()
    d["schema_version"] = SCHEMA_VERSION
    d["processing"] = {"volume": {"correct_trend": True, **vol}}
    return d


def test_schema_1_upgrades_the_retired_trend_default_and_says_so():
    d = _trend_settings(trend_peak_min_height=0.8)
    d["schema_version"] = 1
    s = Settings.from_dict(d)
    assert s.processing.volume.trend_peak_min_height is None
    assert s.schema_version == SCHEMA_VERSION
    assert len(s.notices) == 1 and "trend_peak_min_height" in s.notices[0]


def test_schema_1_never_reinterprets_a_deliberate_threshold():
    d = _trend_settings(trend_peak_min_height=0.5)
    d["schema_version"] = 1
    s = Settings.from_dict(d)
    assert s.processing.volume.trend_peak_min_height == 0.5   # the production value
    assert s.notices == []


def test_notices_are_not_written_back_into_the_analysis():
    d = _trend_settings(trend_peak_min_height=0.8)
    d["schema_version"] = 1
    assert "notices" not in Settings.from_dict(d).to_dict()


def test_trend_prominence_fraction_is_range_checked():
    for bad in (0.0, 1.0, -0.1, 5.0):
        with pytest.raises(SettingsError):
            Settings.from_dict(_trend_settings(trend_peak_min_prominence_frac=bad)).validate()
    Settings.from_dict(_trend_settings(trend_peak_min_prominence_frac=0.5)).validate()


def test_trend_checks_are_inert_while_the_correction_is_off():
    d = _minimal()
    d["processing"] = {"volume": {"correct_trend": False,
                                  "trend_peak_min_prominence_frac": 99.0}}
    Settings.from_dict(d).validate()      # a value that cannot bite must not be fatal


def test_trough_spacing_is_checked_against_the_ANALYSIS_rate_not_the_file_rate():
    """The pre-analysis resample replaces the sampling rate AFTER validate() runs, so a
    spacing that is fine at the file's rate can be under one sample at the analysis rate
    — which used to reach scipy as a bare 'distance must be greater or equal to 1'."""
    d = _trend_settings(trend_peak_min_distance_s=0.004)      # 8 samples at 2000 Hz
    Settings.from_dict(d).validate()
    d["processing"]["sampling"] = {"resample": True, "resample_to_frequency": 100}
    with pytest.raises(SettingsError):                        # 0.4 samples at 100 Hz
        Settings.from_dict(d).validate()


def test_a_legacy_zero_trend_threshold_is_not_rejected():
    """0 is a legal legacy value meaning 'no absolute gate' — find_peaks(height=0) keeps
    every trough. It is NOT equivalent to omitting the key (which selects the scale-free
    rule and gives a different envelope), so refusing it would kill the whole run for an
    analysis that used to work, with no exact-reproduction path left."""
    Settings.from_dict(_trend_settings(trend_peak_min_height=0)).validate()
    Settings.from_dict(_trend_settings(trend_peak_min_height=0.0)).validate()
    with pytest.raises(SettingsError):
        Settings.from_dict(_trend_settings(trend_peak_min_height=-0.5)).validate()


# -- carried-over per-folder state (ticket B06) -------------------------------
# exclude_breaths/breath_counts/the noise reference key on the bare filename, which is
# ambiguous the moment two recordings folders share a filename (the common multi-subject
# workflow). ExcludeEntry/BreathCountEntry.folder and NoiseSettings.reference_folder let
# the UI tell the two situations apart; is_carried_folder/carried_over_state/
# clear_carried_over are the one place that decides it, reused by the Setup banner, the
# Preview overlay and the file rail so none of them can disagree.

def test_a_file_written_before_this_field_existed_loads_with_folder_none():
    """Backward compat: an exclude_breaths entry with no 'folder' key (every analysis
    written before this ticket) must load unchanged, with folder defaulting to None."""
    d = _minimal()
    d["processing"] = {"exclude_breaths": [{"file": "a.txt", "breaths": [1, 2]}]}
    s = Settings.from_dict(d).validate()
    assert s.processing.exclude_breaths[0].folder is None
    assert s.processing.exclude_breaths[0].breaths == [1, 2]     # nothing else is lost


def test_an_unrecognised_field_on_an_exclude_entry_is_captured_not_fatal():
    """Forward compat: a FUTURE field this version doesn't know about, on one list entry,
    must not crash the loader or drop the entry's own known fields (the same tolerance
    Settings.from_dict already gives top-level unknown sections, exercised here at the
    per-list-item level added by this ticket)."""
    d = _minimal()
    d["processing"] = {"exclude_breaths": [
        {"file": "a.txt", "breaths": [1], "folder": "input", "some_future_field": "x"}]}
    s = Settings.from_dict(d).validate()
    assert s.processing.exclude_breaths[0].file == "a.txt"
    assert s.processing.exclude_breaths[0].folder == "input"
    assert "processing.exclude_breaths.[0].some_future_field" in s.unknown


def test_is_carried_folder_true_for_mismatch_or_either_side_unknown():
    assert is_carried_folder("/data/S01", "/data/S02") is True
    assert is_carried_folder(None, "/data/S02") is True          # unrecorded -> unproven
    assert is_carried_folder("/data/S01", None) is True          # no current folder yet
    assert is_carried_folder(None, None) is True
    assert is_carried_folder("/data/S01", "/data/S01") is False
    assert is_carried_folder("/data/S01/", "/data/S01") is False  # normpath-equal


def test_is_carried_folder_uses_normcase_like_the_rest_of_the_codebase():
    """Self-review finding: the SAME real folder re-typed or re-browsed with different case
    must not read as carried-over on a case-insensitive filesystem (Windows) — matches the
    normcase dedup ui.prefs already relies on for the same reason. normcase is a no-op on
    case-sensitive macOS/Linux, so this test derives its expectation from what normcase
    ACTUALLY does on the platform running it, rather than hard-coding one OS's answer —
    meaningful, and correct, on every CI runner."""
    import os
    a, b = "/data/S01", "/DATA/S01"
    same_on_this_platform = os.path.normcase(a) == os.path.normcase(b)
    assert is_carried_folder(a, b) is (not same_on_this_platform)


def test_carried_over_state_is_empty_with_no_current_folder():
    """Nothing can be 'carried over' relative to a folder that isn't set yet (a fresh
    guided analysis) — every entry would trivially mismatch an empty string, which is
    noise, not a real warning."""
    s = Settings()
    s.input.folder = ""          # e.g. settings_screen.enter_new_mode's guided-flow blank
    s.processing.exclude_breaths.append(ExcludeEntry(file="a.txt", breaths=[1], folder=None))
    st = carried_over_state(s)
    assert not st
    assert st == CarriedOverState()


def test_carried_over_state_names_every_kind_of_carried_state():
    s = Settings()
    s.input.folder = "/data/S02"
    s.processing.exclude_breaths.append(
        ExcludeEntry(file="a.txt", breaths=[1, 2], folder="/data/S01"))
    s.processing.exclude_breaths.append(
        ExcludeEntry(file="b.txt", breaths=[3], folder="/data/S02"))    # matches -> not carried
    s.processing.breath_counts.append(
        BreathCountEntry(file="a.txt", count=9, folder="/data/S01"))
    s.processing.emg.noise.reference_file = "a.txt"
    s.processing.emg.noise.reference_intervals = [[0.0, 1.0]]
    s.processing.emg.noise.reference_folder = "/data/S01"
    st = carried_over_state(s)
    assert st.exclude_files == ["a.txt"]
    assert st.breath_count_files == ["a.txt"]
    assert st.noise_reference is True
    assert bool(st) is True


def test_carried_over_state_ignores_an_empty_exclusion_and_an_unset_noise_reference():
    """An ExcludeEntry with no breaths left (the include-everything state _toggle_breath
    leaves behind — see preview/_mechanics.py) and a NoiseSettings with no reference set at
    all must never be reported as carried: there is nothing there to warn about."""
    s = Settings()
    s.input.folder = "/data/S02"
    s.processing.exclude_breaths.append(ExcludeEntry(file="a.txt", breaths=[], folder="/data/S01"))
    st = carried_over_state(s)
    assert st.exclude_files == []
    assert st.noise_reference is False


def test_clear_carried_over_drops_only_the_mismatched_entries():
    s = Settings()
    s.input.folder = "/data/S02"
    kept = ExcludeEntry(file="b.txt", breaths=[3], folder="/data/S02")
    dropped = ExcludeEntry(file="a.txt", breaths=[1], folder="/data/S01")
    s.processing.exclude_breaths.extend([dropped, kept])
    s.processing.breath_counts.append(BreathCountEntry(file="a.txt", count=9, folder="/data/S01"))
    s.processing.emg.noise.reference_file = "a.txt"
    s.processing.emg.noise.reference_intervals = [[0.0, 1.0]]
    s.processing.emg.noise.reference_folder = "/data/S01"
    clear_carried_over(s)
    assert s.processing.exclude_breaths == [kept]
    assert s.processing.breath_counts == []
    assert s.processing.emg.noise.reference_file is None
    assert s.processing.emg.noise.reference_intervals == []
    assert not carried_over_state(s)


def test_clear_carried_over_is_a_no_op_when_nothing_is_carried():
    """The ordinary case — nothing has ever pointed at a different folder — must be left
    completely untouched, not merely end up equal."""
    s = Settings()
    s.input.folder = "/data/S02"
    entry = ExcludeEntry(file="b.txt", breaths=[3], folder="/data/S02")
    s.processing.exclude_breaths.append(entry)
    clear_carried_over(s)
    assert s.processing.exclude_breaths == [entry]
    assert s.processing.exclude_breaths[0] is entry      # same object, never touched
