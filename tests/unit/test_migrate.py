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
    assert s.processing.emg.noise_profile == [["a.txt", "", [1.0, 1.1]]]


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
