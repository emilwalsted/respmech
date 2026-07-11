"""User-preferences store (ui.prefs / QSettings): sticky folders + the remembered
channel rig (P25) and the recent-analyses list (P26). Uses the isolated_prefs fixture
(conftest) so the real user preferences are never touched. Previously test_review_wave6.py."""
import pytest


def test_prefs_sticky_folder_and_rig(qapp, isolated_prefs, tmp_path):
    prefs = isolated_prefs
    prefs.set_last_folder("analysis", str(tmp_path))
    assert prefs.last_folder("analysis") == str(tmp_path)
    assert prefs.last_folder("missing", "fallback") == "fallback"

    from respmech.core.settings import Settings
    s = Settings()
    s.input.channels.poes = 7; s.input.channels.emg = [2, 3, 4]
    s.input.format.sampling_frequency = 500
    prefs.save_rig(s)
    rig = prefs.last_rig()
    assert rig["poes"] == 7 and rig["emg"] == [2, 3, 4] and rig["sampling_frequency"] == 500
    s2 = Settings()
    prefs.apply_rig(s2, rig)
    assert s2.input.channels.poes == 7 and s2.input.channels.emg == [2, 3, 4]
    assert s2.input.format.sampling_frequency == 500


def test_prefs_recent_analyses(qapp, isolated_prefs, tmp_path):
    prefs = isolated_prefs
    a = tmp_path / "a.toml"; a.write_text("x")
    b = tmp_path / "b.toml"; b.write_text("y")
    prefs.add_recent_analysis(str(a)); prefs.add_recent_analysis(str(b))
    prefs.add_recent_analysis(str(a))                    # re-adding moves it to the front
    assert prefs.recent_analyses()[0] == str(a)
    assert set(prefs.recent_analyses()) == {str(a), str(b)}
    a.unlink()
    assert str(a) not in prefs.recent_analyses()         # vanished files are dropped
