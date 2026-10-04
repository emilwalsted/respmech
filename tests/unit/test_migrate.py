import copy
import textwrap

import pytest

from respmech.settingsio.migrate import (extract_legacy_dict, migrate_dict,
                                         migrate_file)


LEGACY = {
    "input": {
        "inputfolder": "/abs/in",
        "files": "*.txt",
        "format": {"samplingfrequency": 2000, "matlabfileformat": 2, "decimalcharacter": "."},
        "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 10,
                 "column_volume": 14, "column_flow": 13,
                 "columns_emg": [2, 3, 4, 5, 6], "columns_entropy": []},
    },
    "processing": {
        "sampling": {"resample": False, "resampletofrequency": 200},
        "mechanics": {
            "breathseparationbuffer": 800, "separateby": "flow",
            "integratevolumefromflow": True, "correctvolumetrend": True,
            "volumetrendpeakminwidth": 0.005,           # never read -> dropped
            "calcwobfromaverage": True,                  # drift -> wob.calc_from
            "avgresamplingobs": 500,                     # drift -> wob.avg_resampling_obs
            "excludebreaths": [["RIU_H5_IC.txt", [1, 2, 4]]],
            "breathcounts": [["x.txt", 6]],
        },
        "emg": {"rms_s": 0.05, "remove_ecg": True, "column_detect": 0,
                "minheight": 0.15, "outlierrmssdlimit": 3, "remove_noise": True,
                "emgplotyscale": [-0.5, 0.5], "noise_profile": [["a.txt", "", [1.0, 1.1]]]},
        "entropy": {"entropy_epochs": 2, "entropy_tolerance": 0.1},
    },
    "output": {
        "outputfolder": "/abs/out",
        "data": {"saveaveragedata": True, "saveprocesseddata": True},
        "diagnostics": {"savepvaverage": True, "savepvoverview": True,
                        "savepvindividualworkload": True, "pvcolumns": 2, "pvrows": 2},
    },
}


def _deep(d):
    """A copy that may be edited without leaking into the shared LEGACY fixture."""
    return copy.deepcopy(d)


def test_migrate_maps_and_normalises():
    s, r = migrate_dict(LEGACY)
    s.validate()
    assert s.input.channels.flow == 13
    assert s.input.format.matlab_variant == "mac"           # 2 -> mac
    assert s.processing.segmentation.method == "flow"
    assert s.processing.volume.integrate_from_flow is True
    # drift normalised
    assert s.processing.wob.calc_from == "average"
    assert s.processing.wob.avg_resampling_obs == 500
    assert any("calcwobfromaverage" in n for n in r.normalised)
    # dropped keys reported
    assert any("volumetrendpeakminwidth" in d for d in r.dropped)
    assert any("savepvoverview" in d for d in r.dropped)
    # exclude/breath counts converted to typed entries
    assert s.processing.exclude_breaths[0].file == "RIU_H5_IC.txt"
    assert s.processing.exclude_breaths[0].breaths == [1, 2, 4]
    assert s.processing.breath_counts[0].count == 6
    assert s.processing.emg.outlier_rms_sd_limit == 3
    # the legacy per-file noise_profile is consumed by the migrator to build the shared
    # noise table (below) and no longer kept as an inert field on Settings
    assert not hasattr(s.processing.emg, "noise_profile")


def test_noise_mapping_and_toml_round_trip(tmp_path):
    from respmech.settingsio.toml_io import save_toml, load_toml
    legacy = dict(LEGACY)
    legacy["processing"] = dict(LEGACY["processing"])
    legacy["processing"]["emg"] = {
        "remove_ecg": True, "remove_noise": True, "column_detect": 4,
        "noise_profile": [["RIU_H5_40W.txt", "/abs/RIU_H5_Baseline.txt", [20.5, 20.55]],
                          ["RIU_H5_60W.txt", "/abs/RIU_H5_Baseline.txt", [20.5, 20.55]]],
    }
    s, r = migrate_dict(legacy)
    n = s.processing.emg.noise
    assert n.enabled is True
    assert n.reference_file == "RIU_H5_Baseline.txt"        # consolidated shared source
    assert n.reference_intervals == [[20.5, 20.55]]
    assert n.n_fft == 256 and n.auto_prop is True           # bug fix + fidelity gate defaults
    assert any("n_fft=len(noise)**2" in x for x in r.normalised)
    # survives a TOML round-trip (declarative, no code)
    p = tmp_path / "s.toml"
    save_toml(s, p)
    assert load_toml(p).to_dict() == s.to_dict()


def test_matlab_windows_variant():
    legacy = {"input": {"format": {"samplingfrequency": 1, "matlabfileformat": 1},
                        "data": {"column_poes": 1, "column_pgas": 2, "column_pdi": 3,
                                 "column_flow": 4, "column_volume": 5}}}
    s, _ = migrate_dict(legacy)
    assert s.input.format.matlab_variant == "windows"


def test_extract_refuses_non_literal(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("settings = {'x': open('/etc/passwd').read()}\n")
    with pytest.raises(ValueError):
        extract_legacy_dict(p)


def test_extract_reads_literal_without_executing(tmp_path):
    p = tmp_path / "s.py"
    p.write_text(textwrap.dedent("""
        import os  # would fail if executed in a weird env; must NOT run
        raise SystemExit('this file must never be executed')
        settings = {'input': {'format': {'samplingfrequency': 2000}}}
    """))
    d = extract_legacy_dict(p)
    assert d["input"]["format"]["samplingfrequency"] == 2000


# -- the end-expiratory trend anchor rule -------------------------------------

def test_v1_without_a_trend_threshold_migrates_to_the_scale_free_rule():
    """0.8 was v1's DEFAULT, not a user choice — and it is an absolute depth below the
    recording's global maximum, so it matches no trough on ordinary tidal breathing.
    Carrying it forward would migrate the bug."""
    s, r = migrate_dict(LEGACY)                       # sets correctvolumetrend, no height
    assert s.processing.volume.trend_peak_min_height is None
    assert any("volumetrendpeakminheight" in n for n in r.normalised)


def test_v1_with_a_deliberate_trend_threshold_keeps_the_absolute_gate():
    legacy = _deep(LEGACY)
    legacy["processing"]["mechanics"]["volumetrendpeakminheight"] = 0.5
    s, _r = migrate_dict(legacy)
    assert s.processing.volume.trend_peak_min_height == 0.5


def test_v1_carrying_the_old_default_is_upgraded_and_reported():
    legacy = _deep(LEGACY)
    legacy["processing"]["mechanics"]["volumetrendpeakminheight"] = 0.8
    s, r = migrate_dict(legacy)
    assert s.processing.volume.trend_peak_min_height is None
    assert any("retired default" in n for n in r.normalised)


# -- MigrationReport.text() prints Defaulted; absent pgas/pdi; derived signals --------

def test_report_text_includes_the_defaulted_section():
    """MigrationReport.defaulted existed as a dataclass field but text() never
    iterated it and nothing ever appended to it -- both are fixed together here:
    LEGACY carries neither an absent pgas/pdi nor anything else that skips the
    Defaulted section, so this alone proves the section heading itself now always
    appears (a regression found along the way: the field was dead code)."""
    s, r = migrate_dict(LEGACY)
    text = r.text()
    assert "## Defaulted" in text
    # analysis.signals is unconditionally reported (a legacy file never has one)
    assert any("analysis.signals" in d for d in r.defaulted)
    assert any("analysis.signals" in line for line in text.splitlines())


def test_missing_column_pgas_and_pdi_migrate_to_none_validate_and_are_reported():
    """Acceptance criterion: migrating a legacy file with no column_pgas/pdi at all
    gives a valid TOML (Settings.validate() already accepted an absent pressure role
    before this -- see the two channels going to None below); what is new here is
    that the migration REPORT now says so explicitly instead of the fact being
    invisible."""
    legacy = _deep(LEGACY)
    del legacy["input"]["data"]["column_pgas"]
    del legacy["input"]["data"]["column_pdi"]
    s, r = migrate_dict(legacy)
    assert s.input.channels.pgas is None
    assert s.input.channels.pdi is None
    s.validate()                                    # does not raise
    assert any("input.channels.pgas" in d and "not present" in d for d in r.defaulted)
    assert any("input.channels.pdi" in d and "not present" in d for d in r.defaulted)


def test_present_column_pgas_and_pdi_are_not_reported_as_defaulted():
    """The converse of the test above: LEGACY's own column_pgas/column_pdi ARE
    present, so neither channel gets a 'not present in the legacy file' line --
    only the always-unconditional analysis.signals entry is in `r.defaulted`."""
    s, r = migrate_dict(LEGACY)
    assert not any("input.channels.pgas" in d for d in r.defaulted)
    assert not any("input.channels.pdi" in d for d in r.defaulted)
    assert len(r.defaulted) == 1
    assert "analysis.signals" in r.defaulted[0]


def test_analysis_signals_is_derived_from_the_migrated_channels_and_reported():
    """LEGACY names poes/pgas/pdi/flow/volume/emg columns but no entropy channels --
    the derived signal set (core.analysis.signals.derived_signals) must list exactly
    the single-role signals actually assigned, reported with the real list, not a
    vague 'derived' placeholder."""
    s, r = migrate_dict(LEGACY)
    from respmech.core.analysis.signals import derived_signals
    derived = sorted(derived_signals(s.input.channels))
    assert derived == sorted(["flow", "poes", "pgas", "pdi", "emg"])
    line = next(d for d in r.defaulted if "analysis.signals" in d)
    for sig in derived:
        assert sig in line
    # this changes no actual behaviour: an empty analysis.signals still means
    # "derive it" (AnalysisSettings' own docstring) -- the report is informational.
    assert s.analysis.signals == []
