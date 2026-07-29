import pytest

from respmech.core.settings import Settings, SettingsError


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


def test_round_trip():
    d = _minimal()
    d["processing"] = {"exclude_breaths": [{"file": "a.txt", "breaths": [1, 2]}]}
    s = Settings.from_dict(d).validate()
    assert s.processing.exclude_breaths[0].file == "a.txt"
    assert Settings.from_dict(s.to_dict()).to_dict() == s.to_dict()
