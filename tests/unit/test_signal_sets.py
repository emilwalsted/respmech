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


# --------------------------------------------------------------------------------- #
# AnalysisSettings on the real Settings model: every Settings.validate() rule, the
# empty-list/save-omission semantics, and reconciliation being from_dict-only.
# --------------------------------------------------------------------------------- #
from respmech.core.settings import Settings, SettingsError  # noqa: E402


def _full_settings() -> Settings:
    """A ``Settings()`` with every legacy role assigned (poes/pgas/pdi/flow/volume) and
    an empty ``analysis.signals`` — the "everything derives, nothing declared" baseline
    every test below mutates exactly one thing off."""
    s = Settings()
    s.input.format.sampling_frequency = 1000
    ch = s.input.channels
    ch.flow, ch.poes, ch.pgas, ch.pdi, ch.volume = 5, 7, 8, 9, 6
    return s


def test_derived_signal_set_still_validates_with_every_role_assigned():
    _full_settings().validate()          # must not raise


def test_empty_effective_set_is_rejected():
    s = _full_settings()
    for role in ("flow", "poes", "pgas", "pdi"):
        setattr(s.input.channels, role, None)
    with pytest.raises(SettingsError, match=r"must name at least one of 'flow' or 'emg'"):
        s.validate()


def test_pressure_without_flow_is_rejected_derived():
    s = _full_settings()
    s.input.channels.flow = None
    with pytest.raises(SettingsError,
                        match=r"'poes', 'pgas' and 'pdi' require 'flow'"):
        s.validate()


def test_unknown_signal_name_is_rejected():
    s = _full_settings()
    s.analysis.signals = ["flow", "not-a-real-signal"]
    with pytest.raises(SettingsError, match=r"unknown signal 'not-a-real-signal'"):
        s.validate()


def test_channel_required_by_explicit_signal_set():
    s = _full_settings()
    s.input.channels.pdi = None
    s.analysis.signals = ["flow", "pdi"]
    with pytest.raises(SettingsError, match=r"input\.channels\.pdi is required by analysis\.signals"):
        s.validate()


def test_emg_declared_requires_at_least_one_emg_column():
    s = _full_settings()
    s.analysis.signals = ["flow", "emg"]
    with pytest.raises(SettingsError,
                        match=r"input\.channels\.emg must name at least one column"):
        s.validate()


def test_volume_rule_only_applies_when_flow_is_declared():
    # flow present, volume missing, not integrated -> still rejected (D02, unchanged).
    s = _full_settings()
    s.input.channels.volume = None
    with pytest.raises(SettingsError, match=r"input\.channels\.volume is required"):
        s.validate()
    # flow ABSENT (declared set falls back to poes/pgas/pdi... but that itself needs
    # flow, so drop pressures too, leaving a pure EMG-only set) -> volume rule never
    # fires at all, regardless of the channel being unset.
    s2 = _full_settings()
    s2.input.channels.flow = None
    s2.input.channels.poes = None
    s2.input.channels.pgas = None
    s2.input.channels.pdi = None
    s2.input.channels.volume = None
    s2.input.channels.emg = [2, 3]
    s2.processing.segmentation.method = "whole_file"   # EMG-only method, matches the set
    s2.validate()                        # must not raise


def test_segmentation_method_enum_includes_the_new_emg_only_methods():
    s = _full_settings()
    s.processing.segmentation.method = "not-a-method"
    with pytest.raises(SettingsError,
                        match=r"'flow', 'volume', 'whole_file', 'separators', "
                              r"'fixed_windows' or 'emg_burst'"):
        s.validate()


def test_segmentation_method_requires_flow_for_the_flow_family():
    s = _full_settings()
    s.input.channels.emg = [2, 3]
    s.analysis.signals = ["emg"]          # EMG-only: no flow in the effective set
    s.processing.segmentation.method = "volume"
    with pytest.raises(SettingsError,
                        match=r"segmentation\.method 'volume' requires 'flow'"):
        s.validate()


def test_segmentation_method_emg_only_rejected_when_flow_is_declared():
    s = _full_settings()
    s.processing.segmentation.method = "fixed_windows"
    with pytest.raises(SettingsError,
                        match=r"segmentation\.method 'fixed_windows' is for an "
                              r"EMG-only signal set"):
        s.validate()


def test_entropy_is_never_checked_against_the_signal_set():
    """R8: an entropy-only channel assignment (no flow/poes/pgas/pdi/emg at all) fails
    validate() for having NO analysable signal -- never for anything entropy-shaped,
    since entropy is independent of the signal set entirely."""
    s = _full_settings()
    for role in ("flow", "poes", "pgas", "pdi"):
        setattr(s.input.channels, role, None)
    s.input.channels.entropy = [10, 11]
    with pytest.raises(SettingsError, match=r"must name at least one of 'flow' or 'emg'"):
        s.validate()


# -- empty-list / save-omission semantics ("gem-udeladelse") -----------------------

def test_from_dict_to_dict_is_idempotent_for_an_empty_signal_set():
    s = Settings()
    assert s.analysis.signals == []
    assert Settings.from_dict(s.to_dict()).analysis.signals == []


def test_from_dict_to_dict_is_idempotent_for_an_explicit_signal_set():
    s = Settings()
    s.analysis.signals = ["flow", "poes"]
    assert Settings.from_dict(s.to_dict()).analysis.signals == ["flow", "poes"]


def test_save_toml_omits_analysis_table_for_an_empty_signal_set(tmp_path):
    from respmech.settingsio.toml_io import save_toml
    s = _full_settings()
    path = tmp_path / "a.toml"
    save_toml(s, path)
    assert "[analysis]" not in path.read_text()
    assert "signals" not in path.read_text()


def test_save_toml_omits_analysis_table_when_explicit_equals_derived(tmp_path):
    from respmech.settingsio.toml_io import save_toml
    s = _full_settings()
    s.analysis.signals = ["flow", "poes", "pgas", "pdi"]     # == derived_signals(ch)
    path = tmp_path / "a.toml"
    save_toml(s, path)
    assert "[analysis]" not in path.read_text()


def test_save_toml_keeps_analysis_table_when_it_diverges_from_derived(tmp_path):
    """A genuinely persistent divergence names a role whose channel is NOT (yet)
    assigned -- e.g. a preset chosen before channels are mapped (5.3's own reason
    ``[analysis]`` exists at all). A role narrower than what IS assigned would instead
    be re-added by from_dict's own reconciliation on the very next load (tested
    separately below), so it would not actually demonstrate persistence."""
    from respmech.settingsio.toml_io import save_toml, load_toml
    s = Settings()
    s.input.format.sampling_frequency = 1000
    s.input.channels.flow = 5                     # only flow is actually assigned
    s.analysis.signals = ["flow", "poes"]          # poes declared ahead of assignment
    path = tmp_path / "a.toml"
    save_toml(s, path)
    text = path.read_text()
    assert "[analysis]" in text
    assert "poes" in text
    s2 = load_toml(path)
    assert s2.analysis.signals == ["flow", "poes"]
    assert s2.notices == []          # nothing to reconcile: poes was never assigned


def test_dumps_toml_always_writes_the_resolved_effective_set_sorted():
    import tomllib

    from respmech.settingsio.toml_io import dumps_toml
    s = _full_settings()                          # analysis.signals == [] (derived)
    text = dumps_toml(s)
    assert "[analysis]" in text
    assert tomllib.loads(text)["analysis"]["signals"] == ["flow", "pdi", "pgas", "poes"]


# -- reconciliation is from_dict-ONLY, never any other write path ------------------

def test_reconciliation_adds_an_assigned_but_undeclared_channel_with_exactly_one_notice():
    d = {
        "input": {
            "format": {"sampling_frequency": 1000},
            "channels": {"flow": 5, "poes": 7, "pgas": 8},
        },
        "analysis": {"signals": ["flow", "poes"]},   # pgas assigned but not declared
    }
    s = Settings.from_dict(d)
    assert s.analysis.signals == ["flow", "poes", "pgas"]
    assert len(s.notices) == 1
    assert "input.channels.pgas is assigned but 'pgas' was not in analysis.signals" in s.notices[0]


def test_reconciliation_is_silent_without_an_explicit_analysis_table():
    d = {
        "input": {
            "format": {"sampling_frequency": 1000},
            "channels": {"flow": 5, "poes": 7, "pgas": 8},
        },
    }
    s = Settings.from_dict(d)
    assert s.analysis.signals == []
    assert s.notices == []


def test_reconciliation_mutating_settings_after_load_never_retroactively_reconciles():
    """Reconciliation runs exactly once, inside from_dict — assigning a channel to an
    ALREADY-BUILT Settings object (the in-app write path, once it exists) must not
    somehow re-trigger it; nothing calls _reconcile_signals except from_dict."""
    d = {
        "input": {
            "format": {"sampling_frequency": 1000},
            "channels": {"flow": 5},
        },
        "analysis": {"signals": ["flow"]},
    }
    s = Settings.from_dict(d)
    assert s.notices == []
    s.input.channels.poes = 7           # assigned after load, outside from_dict
    assert s.analysis.signals == ["flow"]   # unchanged -- no live reconciliation
    assert s.notices == []


# -- malformed analysis.signals must never crash from_dict (self-review finding) --

def _minimal_dict(**channels):
    return {
        "input": {"format": {"sampling_frequency": 1000}, "channels": channels},
    }


def test_reconciliation_tolerates_a_bare_string_signals_value_without_crashing():
    """A hand-edited TOML forgetting the brackets (`signals = "flow"` instead of
    `signals = ["flow"]`) must not crash from_dict with a raw AttributeError from
    list.append(), nor crash validate() with a raw, uncategorised TypeError -- it
    should read as an ordinary, actionable SettingsError like everything else
    validate() rejects."""
    d = _minimal_dict(flow=5)
    d["analysis"] = {"signals": "flow"}
    s = Settings.from_dict(d)             # must not raise
    assert s.analysis.signals == "flow"   # left untouched for validate() to report
    assert s.notices == []
    with pytest.raises(SettingsError, match=r"analysis\.signals must be a list"):
        s.validate()


def test_validate_rejects_a_non_list_non_string_signals_value_cleanly():
    d = _minimal_dict(flow=5)
    d["analysis"] = {"signals": 5}
    s = Settings.from_dict(d)             # must not raise (reconciliation skips it too)
    assert s.notices == []
    with pytest.raises(SettingsError, match=r"analysis\.signals must be a list"):
        s.validate()


def test_reconciliation_tolerates_a_none_signals_value_without_crashing():
    """Only reachable via a dict-based caller (TOML has no null), but from_dict is a
    general tolerant-ingestion API -- must not crash, and should read as "derive it"
    exactly like an omitted [analysis] table, matching effective_signals' own
    `raw = ... or ()` fallback."""
    d = _minimal_dict(flow=5, poes=7, volume=6)
    d["analysis"] = {"signals": None}
    s = Settings.from_dict(d)             # must not raise
    assert s.notices == []
    s.validate()                          # must not raise either -- derives {flow, poes}


def test_reconciliation_never_mutates_the_callers_own_signals_list():
    """`_build`/`_coerce` do not copy a plain list field, so `signals` could otherwise
    still be the exact list object a caller's dict handed to from_dict -- appending a
    reconciled role in place would silently grow that caller's own object too."""
    signals_list = ["poes"]
    d = _minimal_dict(flow=5, poes=7)
    d["analysis"] = {"signals": signals_list}
    s = Settings.from_dict(d)
    assert s.analysis.signals == ["poes", "flow"]
    assert signals_list == ["poes"]       # the caller's own list is untouched
    assert s.analysis.signals is not signals_list


def test_dumps_toml_never_crashes_on_a_malformed_signals_value():
    """Self-review finding: dumps_toml (the run-manifest writer) called effective_signals
    directly, which raises a bare TypeError for a non-list analysis.signals -- normally
    unreachable (every real call path writes the manifest only after validate() already
    rejected such a value cleanly), but a direct/unvalidated call must still not crash,
    matching save_toml's existing tolerance for the same malformed input."""
    from respmech.settingsio.toml_io import dumps_toml
    s = Settings()
    s.input.format.sampling_frequency = 1000
    s.analysis.signals = "flow"           # malformed: bare string, not a list
    text = dumps_toml(s)                  # must not raise
    assert 'signals = "flow"' in text
