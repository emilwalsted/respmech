"""Signal-set derivation (``core/analysis/signals.py``): per-channel-combination
coverage of ``derived_signals``/``effective_signals``, and R8 (sample entropy is
never a function of the signal set)."""
from types import SimpleNamespace

import pytest

from respmech.core.analysis.signals import (
    Capabilities,
    SINGLE_SIGNALS,
    derived_signals,
    effective_signals,
)


def _ch(**kw):
    base = dict(flow=None, poes=None, pgas=None, pdi=None, volume=None, emg=[], entropy=[])
    base.update(kw)
    return SimpleNamespace(**base)


def _settings(ch, *, analysis_signals=None, integrate_from_flow=False):
    kw = {"input": SimpleNamespace(channels=ch)}
    if analysis_signals is not None:
        kw["analysis"] = SimpleNamespace(signals=analysis_signals)
    kw["processing"] = SimpleNamespace(volume=SimpleNamespace(integrate_from_flow=integrate_from_flow))
    return SimpleNamespace(**kw)


def test_single_signals_is_the_four_pressure_and_flow_roles():
    assert SINGLE_SIGNALS == ("flow", "poes", "pgas", "pdi")


@pytest.mark.parametrize("assigned, expected", [
    ({}, frozenset()),
    ({"flow": 5}, frozenset({"flow"})),
    ({"flow": 5, "poes": 7}, frozenset({"flow", "poes"})),
    ({"poes": 7}, frozenset({"poes"})),
    ({"flow": 5, "poes": 7, "pgas": 8, "pdi": 9}, frozenset({"flow", "poes", "pgas", "pdi"})),
    ({"emg": [2, 3]}, frozenset({"emg"})),
    ({"flow": 5, "emg": [2, 3]}, frozenset({"flow", "emg"})),
    ({"flow": 5, "poes": 7, "pgas": 8, "pdi": 9, "emg": [2, 3]},
     frozenset({"flow", "poes", "pgas", "pdi", "emg"})),
])
def test_derived_signals_per_channel_combination(assigned, expected):
    assert derived_signals(_ch(**assigned)) == expected


def test_derived_signals_ignores_volume_and_entropy():
    # volume/entropy are not SINGLE_SIGNALS roles; assigning them alone derives nothing.
    assert derived_signals(_ch(volume=6, entropy=[10, 11])) == frozenset()


def test_derived_signals_empty_emg_list_does_not_add_emg():
    assert derived_signals(_ch(flow=5, emg=[])) == frozenset({"flow"})


def test_effective_signals_prefers_explicit_over_derived():
    settings = _settings(_ch(flow=5, poes=7), analysis_signals=["flow"])
    assert effective_signals(settings) == frozenset({"flow"})


def test_effective_signals_falls_back_to_derived_when_analysis_missing():
    # Settings has no `.analysis` attribute at all as of this ticket.
    settings = _settings(_ch(flow=5, poes=7))
    assert not hasattr(settings, "analysis")
    assert effective_signals(settings) == frozenset({"flow", "poes"})


def test_effective_signals_falls_back_to_derived_when_explicit_is_empty():
    settings = _settings(_ch(flow=5), analysis_signals=[])
    assert effective_signals(settings) == frozenset({"flow"})


def test_effective_signals_rejects_a_bare_string_instead_of_silently_splitting_it():
    """`frozenset("flow")` would silently give {'f','l','o','w'}, not {'flow'} — an
    easy `signals = "flow"` vs. `signals = ["flow"]` typo once a real,
    hand-editable ``AnalysisSettings.signals`` field exists. Must raise, not
    corrupt the signal set in silence."""
    settings = _settings(_ch(flow=5), analysis_signals="flow")
    with pytest.raises(TypeError):
        effective_signals(settings)


@pytest.mark.parametrize("assigned, expected_mode", [
    ({"flow": 5, "poes": 7, "pgas": 8, "pdi": 9}, "full"),
    ({"flow": 5, "poes": 7, "pgas": 8, "pdi": 9, "emg": [2]}, "full"),
    ({"flow": 5, "poes": 7}, "poes_only"),
    ({"flow": 5, "poes": 7, "emg": [2]}, "poes_only"),
    ({"flow": 5}, "flow_only"),
    ({"flow": 5, "emg": [2]}, "flow_only"),
    ({"emg": [2]}, "emg_only"),
    ({"flow": 5, "pgas": 8}, "custom"),
    ({}, "custom"),
])
def test_capabilities_mode_per_channel_combination(assigned, expected_mode):
    caps = Capabilities.from_settings(_settings(_ch(**assigned)))
    assert caps.mode == expected_mode


def test_entropy_is_independent_of_the_signal_set():
    """R8: Capabilities.entropy is bool(ch.entropy), never a function of `declared`."""
    # entropy assigned, nothing else at all -> entropy True, everything else False/custom.
    caps = Capabilities.from_settings(_settings(_ch(entropy=[10, 11, 12])))
    assert caps.entropy is True
    assert caps.emg is False
    assert caps.flow is False

    # a full flow/poes/pgas/pdi set with EMG explicitly unassigned still keeps entropy.
    caps_full = Capabilities.from_settings(_settings(
        _ch(flow=5, poes=7, pgas=8, pdi=9, emg=[], entropy=[10, 11, 12])
    ))
    assert caps_full.mode == "full"
    assert caps_full.emg is False
    assert caps_full.entropy is True

    # emg-only, with entropy assigned too.
    caps_emg_only = Capabilities.from_settings(_settings(_ch(emg=[2, 3], entropy=[10])))
    assert caps_emg_only.mode == "emg_only"
    assert caps_emg_only.entropy is True

    # no entropy channel assigned anywhere -> False, regardless of everything else.
    caps_no_entropy = Capabilities.from_settings(_settings(
        _ch(flow=5, poes=7, pgas=8, pdi=9, emg=[2, 3])
    ))
    assert caps_no_entropy.entropy is False


def test_volume_requires_flow_and_either_a_channel_or_integrate_from_flow():
    # No flow at all -> volume is always False, even with a Volume channel assigned
    # (there is nothing to segment breaths on, so there is no per-breath volume either).
    caps = Capabilities.from_settings(_settings(_ch(volume=6)))
    assert caps.volume is False

    # Flow + an assigned Volume channel.
    caps = Capabilities.from_settings(_settings(_ch(flow=5, volume=6)))
    assert caps.volume is True

    # Flow, no Volume channel, integrate_from_flow off -> no volume.
    caps = Capabilities.from_settings(_settings(_ch(flow=5)))
    assert caps.volume is False

    # Flow, no Volume channel, integrate_from_flow on -> volume derived instead.
    caps = Capabilities.from_settings(_settings(_ch(flow=5), integrate_from_flow=True))
    assert caps.volume is True


def test_from_settings_tolerates_a_settings_double_with_no_processing_attribute():
    """Several existing test doubles elsewhere in the suite build a bare
    `SimpleNamespace(input=..., processing=SimpleNamespace(emg=...))` for
    unrelated purposes, without a `.processing.volume`. `from_settings` must read
    that as "integrate_from_flow is off", not raise -- consistent with
    `effective_signals`'s own defensive `settings.analysis` lookup."""
    settings = SimpleNamespace(input=SimpleNamespace(channels=_ch(flow=5)))
    caps = Capabilities.from_settings(settings)
    assert caps.flow is True
    assert caps.volume is False


def test_required_roles_excludes_entropy_and_includes_volume_when_applicable():
    caps = Capabilities.from_settings(_settings(_ch(flow=5, volume=6, emg=[2], entropy=[10])))
    assert caps.required_roles() == frozenset({"flow", "volume", "emg"})


def test_analyses_labels_grow_with_capabilities():
    minimal = Capabilities.from_settings(_settings(_ch(flow=5)))
    assert minimal.analyses() == ("Breath timing",)

    full = Capabilities.from_settings(_settings(
        _ch(flow=5, poes=7, pgas=8, pdi=9, emg=[2], entropy=[10])
    ))
    assert full.analyses() == (
        "Breath timing", "Work of breathing", "Gastric pressure",
        "Transdiaphragmatic pressure", "Ventilatory muscle ratio", "EMG", "Sample entropy",
    )


def test_analyses_pressure_labels_require_flow_like_registry_does():
    """A pgas/pdi channel assigned with no flow channel is a state `validate()`
    would reject once it exists, but `Capabilities.from_settings` itself does not
    validate — it only derives. `analyses()` must not claim "Gastric pressure" /
    "Transdiaphragmatic pressure" / "Ventilatory muscle ratio" for such a state,
    since registry.py's own pressures_pgas/pressures_pdi/vmr rows all require
    flow too (segmentation itself needs it) and would resolve zero columns."""
    caps = Capabilities.from_settings(_settings(_ch(pgas=8, pdi=9, poes=7)))
    assert caps.flow is False
    assert caps.analyses() == ()
