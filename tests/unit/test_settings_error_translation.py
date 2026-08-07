"""``ui.validation.friendly_settings_error`` (ticket D02): translates a core
``Settings.validate()`` exception into a sentence naming the UI control to fix, instead
of the raw dotted TOML path the core layer writes for a settings file. Three surfaces used
to each show that raw text verbatim — Setup's status bar (``_validation_status``), its
"cannot save" message (``_save_blocker``), and the ONE shared list the QC strip and the Run
screen's commitment sheet both read (``ui.validation.blockers``, ticket B07) — so this is
tested both as a pure function against every real message ``validate()`` can raise, and at
each of those three call sites.
"""
import re

import pytest

from respmech.core.settings import Settings, SettingsError
from respmech.ui.validation import blockers, friendly_settings_error

#: what every SettingsError message's technical key looks like — the same pattern
#: friendly_settings_error's own fallback strips. A translation "leaking" one of these is
#: exactly the bug this ticket closes.
_DOTTED_KEY = re.compile(r"\b[a-z][a-z_]*(?:\.[a-z][a-z_]*){2,}\b")


def _valid_settings():
    """A ``Settings()`` that passes ``validate()`` outright — every test below mutates
    exactly ONE field off this baseline, so a raised error can only be the one under test."""
    s = Settings()
    s.input.format.sampling_frequency = 1000
    ch = s.input.channels
    ch.flow, ch.poes, ch.pgas, ch.pdi, ch.volume = 2, 3, 4, 5, 6
    return s


def test_baseline_settings_validate_cleanly(qapp):
    _valid_settings().validate()                     # must not raise


# --------------------------------------------------------------------------- #
# every real message Settings.validate() can raise, translated
# --------------------------------------------------------------------------- #
def _raised(mutate):
    s = _valid_settings()
    mutate(s)
    with pytest.raises(SettingsError) as exc:
        s.validate()
    return exc.value


_CASES = [
    ("sampling frequency unset",
     lambda s: setattr(s.input.format, "sampling_frequency", None),
     ["sampling frequency"]),
    ("sampling frequency not an int",
     lambda s: setattr(s.input.format, "sampling_frequency", 1000.5),
     ["sampling frequency"]),
    ("matlab variant invalid",
     lambda s: setattr(s.input.format, "matlab_variant", "linux"),
     ["matlab"]),
    ("volume channel missing, not derived from flow",
     lambda s: setattr(s.input.channels, "volume", None),
     ["volume"]),
    ("flow channel missing",                          # dead path via blockers() (channel_
     lambda s: setattr(s.input.channels, "flow", None),  # collision intercepts first), but
     ["flow"]),                                        # _validation_status() calls validate()
    ("segmentation method invalid",                     # directly and has no such gate
     lambda s: setattr(s.processing.segmentation, "method", "pressure"),
     ["breath", "segment"]),
    ("segmentation buffer not an int",
     lambda s: setattr(s.processing.segmentation, "buffer", 12.5),
     ["breath", "buffer", "segment"]),
    ("wob calc_from invalid",
     lambda s: setattr(s.processing.wob, "calc_from", "median"),
     ["work of breathing"]),
    ("wob avg_resampling_obs not an int",
     lambda s: setattr(s.processing.wob, "avg_resampling_obs", 12.5),
     ["resampling", "work of breathing"]),
    ("volume trend method invalid",
     lambda s: setattr(s.processing.volume, "trend_method", "spline9"),
     ["trend", "interpolation"]),
    ("ecg auto-detect without remove_ecg",
     lambda s: setattr(s.processing.emg, "ecg_auto_detect", True),
     ["ecg"]),
    ("ecg auto-detect without an emg channel",
     lambda s: (setattr(s.processing.emg, "remove_ecg", True),
               setattr(s.processing.emg, "ecg_auto_detect", True)),
     ["ecg", "emg"]),
    ("trend prominence out of range",
     lambda s: (setattr(s.processing.volume, "correct_trend", True),
               setattr(s.processing.volume, "trend_peak_min_prominence_frac", 1.5)),
     ["trend"]),
    ("trend absolute threshold negative",
     lambda s: (setattr(s.processing.volume, "correct_trend", True),
               setattr(s.processing.volume, "trend_peak_min_height", -5.0)),
     ["trend"]),
    ("trend anchor spacing too small for the analysis rate",
     lambda s: (setattr(s.processing.volume, "correct_trend", True),
               setattr(s.processing.volume, "trend_peak_min_distance_s", 0.0001)),
     ["trend"]),
]


@pytest.mark.parametrize("label,mutate,expect_words", _CASES, ids=[c[0] for c in _CASES])
def test_every_validate_message_translates_without_a_dotted_key(qapp, label, mutate, expect_words):
    exc = _raised(mutate)
    friendly = friendly_settings_error(exc)
    assert not _DOTTED_KEY.search(friendly), (
        f"{label}: translated text still leaks a raw settings key: {friendly!r}")
    assert friendly != str(exc)                      # actually translated, not passed through
    low = friendly.lower()
    assert any(w in low for w in expect_words), (
        f"{label}: {friendly!r} does not name the control the ticket expects "
        f"(looked for one of {expect_words})")


def test_the_headline_volume_message_names_both_ways_to_fix_it(qapp):
    """The exact scenario the ticket exists for — spelled out because the parametrized
    'contains volume' check above is intentionally loose."""
    exc = _raised(lambda s: setattr(s.input.channels, "volume", None))
    friendly = friendly_settings_error(exc)
    assert "input.channels.volume" not in friendly
    assert "volume" in friendly.lower()
    assert "assign" in friendly.lower() and "derive" in friendly.lower()


def test_unknown_message_falls_back_to_stripping_the_dotted_key_generically(qapp):
    """A future core message this table has not been updated for must still never leak a
    raw settings path — the whole point of a fallback, not just the mapped cases."""
    friendly = friendly_settings_error(
        SettingsError("processing.some.brand.new_field is not a recognised option"))
    assert not _DOTTED_KEY.search(friendly)
    assert "a setting" in friendly


def test_empty_message_falls_back_to_the_exception_class_name(qapp):
    assert friendly_settings_error(SettingsError("")) == "SettingsError"


# --------------------------------------------------------------------------- #
# blockers() — the QC strip / Run screen's shared source (ticket B07 + D02)
# --------------------------------------------------------------------------- #
def test_blockers_translates_a_settings_error(qapp):
    s = _valid_settings()
    s.input.channels.volume = None
    top = blockers(s)[0]
    assert "input.channels.volume" not in top
    assert "volume" in top.lower()


# --------------------------------------------------------------------------- #
# the two settings_screen.py call sites the ticket itself did not name
# --------------------------------------------------------------------------- #
def test_validation_status_never_shows_a_raw_dotted_key(qapp):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.state.settings = _valid_settings()
    sc.state.settings.input.channels.volume = None
    text = sc._validation_status()
    assert text.startswith("Invalid:")
    assert not _DOTTED_KEY.search(text)
    assert "volume" in text.lower()
    win.close()


def test_save_blocker_never_shows_a_raw_dotted_key(qapp):
    from respmech.ui.main_window import MainWindow
    from respmech.ui.state import AppState
    win = MainWindow(AppState())
    sc = win.settings_screen
    sc.state.settings = _valid_settings()
    sc.state.settings.input.channels.volume = None
    blocker = sc._save_blocker()
    assert blocker is not None
    assert not _DOTTED_KEY.search(blocker)
    assert "volume" in blocker.lower()
    assert sc.can_save() is False
    win.close()
