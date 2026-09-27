"""``resolve_noise_reference_mode`` (core.settings) and its two pipeline consumers:
``_reference_noise_clip`` (the reference clip itself) and ``_emg_segmented``'s
``exclude_typed_from_expiration`` guard.

Three concerns, matching the ticket's own acceptance criteria:

* the resolution chain itself (a pure function of settings, no I/O) -- 'auto' with a
  flow channel reproduces the pre-existing ``use_expiration``/``reference_intervals``
  rule exactly; 'auto' without one resolves via a rest-typed reference segment;
* ``_reference_noise_clip`` builds BIT-IDENTICAL clips to the pre-ticket code for every
  flow-bearing settings shape already in use (synthetic, the built-in sample, a
  migrated legacy dict) -- proven against an inline reimplementation of the OLD
  algorithm, not merely re-run through the new one;
* the new EMG-only ``rest_segments`` branch, and the expiration-mask guard (decision
  11) that excludes a typed breath's expiration from the flow-bearing reference clip,
  is a no-op absent typed breaths and does exclude them when present.
"""
import os

import numpy as np
import pytest

from _helpers import INPUT, requires_synth, synth_settings

from respmech.core._legacy_ns import to_legacy_ns
from respmech.core.pipeline import (_emg_segmented, _load_and_ecg,
                                    _reference_noise_clip)
from respmech.core.settings import (BreathTypeEntry, SeparatorEntry, Settings,
                                    SettingsError, resolve_noise_reference_mode,
                                    resolve_noise_reference_mode_or_none)


# --------------------------------------------------------------------------------- #
# resolution chain -- pure, no I/O
# --------------------------------------------------------------------------------- #

def _flow_settings(*, use_expiration=True, reference_intervals=None, reference_mode="auto"):
    s = Settings()
    s.input.format.sampling_frequency = 1000
    ch = s.input.channels
    ch.flow, ch.volume, ch.emg = 2, 6, [3]
    n = s.processing.emg.noise
    n.reference_file = "ref.csv"
    n.use_expiration = use_expiration
    n.reference_intervals = reference_intervals or []
    n.reference_mode = reference_mode
    return s


def _emg_only_settings(*, method="whole_file", reference_mode="auto",
                       reference_intervals=None, rest_breath=None):
    s = Settings()
    s.input.format.sampling_frequency = 1000
    ch = s.input.channels
    ch.emg = [2]
    s.analysis.signals = ["emg"]
    s.processing.segmentation.method = method
    n = s.processing.emg.noise
    n.reference_file = "ref.csv"
    n.reference_intervals = reference_intervals or []
    n.reference_mode = reference_mode
    if rest_breath is not None:
        s.processing.breath_types.append(
            BreathTypeEntry(file="ref.csv", breath=rest_breath, kind="rest"))
    return s


def test_auto_with_flow_and_use_expiration_resolves_expiration():
    """The precise, pre-existing predicate: use_expiration True (the default) ->
    'expiration', regardless of whether reference_intervals also happen to be set."""
    s = _flow_settings(use_expiration=True, reference_intervals=[[1.0, 2.0]])
    assert resolve_noise_reference_mode(s) == "expiration"


def test_auto_with_flow_and_no_intervals_resolves_expiration():
    """use_expiration False but no intervals set: falls back to expiration (`not
    reference_intervals` half of the predicate), exactly as before this ticket."""
    s = _flow_settings(use_expiration=False, reference_intervals=[])
    assert resolve_noise_reference_mode(s) == "expiration"


def test_auto_with_flow_and_explicit_intervals_resolves_intervals():
    s = _flow_settings(use_expiration=False, reference_intervals=[[1.0, 5.0]])
    assert resolve_noise_reference_mode(s) == "intervals"


def test_auto_without_flow_and_a_rest_segment_resolves_rest_segments():
    s = _emg_only_settings(rest_breath=1)
    assert resolve_noise_reference_mode(s) == "rest_segments"


def test_auto_without_flow_and_no_rest_segment_falls_back_to_intervals():
    s = _emg_only_settings(reference_intervals=[[0.1, 0.9]])
    assert resolve_noise_reference_mode(s) == "intervals"


def test_auto_without_flow_and_neither_is_unresolved():
    s = _emg_only_settings()
    assert resolve_noise_reference_mode(s) == "unresolved"


def test_rest_typed_breath_in_a_DIFFERENT_file_does_not_count():
    """The lookup is keyed on the reference file specifically -- a 'rest' segment
    typed in some other file in the analysis must not make an unrelated reference
    file's mode resolve as though it had one of its own."""
    s = _emg_only_settings()
    s.processing.breath_types.append(
        BreathTypeEntry(file="other.csv", breath=1, kind="rest"))
    assert resolve_noise_reference_mode(s) == "unresolved"


@pytest.mark.parametrize("mode", ["rest_segments", "interburst"])
def test_explicit_emg_only_modes_are_returned_as_is(mode):
    s = _emg_only_settings(reference_mode=mode)
    assert resolve_noise_reference_mode(s) == mode


def test_never_raises_even_for_a_combination_validate_would_reject():
    """resolve_noise_reference_mode is a pure classifier -- it is Settings.validate()'s
    job to reject an explicit EMG-only mode requested while a flow channel is declared,
    not this function's. (Real callers always validate first; this only pins that the
    resolver itself does not duplicate that enforcement.)"""
    s = _flow_settings(reference_mode="rest_segments")
    assert resolve_noise_reference_mode(s) == "rest_segments"


# --------------------------------------------------------------------------------- #
# Settings.validate() -- the messages this ticket adds/replaces
# --------------------------------------------------------------------------------- #

def test_validate_rejects_unresolved_emg_only_noise():
    s = _emg_only_settings()
    s.processing.emg.remove_ecg = True
    s.processing.emg.noise.enabled = True
    s.processing.emg.noise.auto_prop = False
    with pytest.raises(SettingsError, match="no usable rest reference"):
        s.validate()


def test_validate_accepts_a_resolved_emg_only_rest_reference():
    s = _emg_only_settings(rest_breath=1)
    s.processing.emg.remove_ecg = True
    s.processing.emg.noise.enabled = True
    s.processing.emg.noise.auto_prop = False
    s.validate()                                       # must not raise


def test_validate_rejects_interburst_while_noise_enabled():
    s = _emg_only_settings(reference_mode="interburst")
    s.processing.emg.remove_ecg = True
    s.processing.emg.noise.enabled = True
    s.processing.emg.noise.auto_prop = False
    with pytest.raises(SettingsError, match="not yet implemented"):
        s.validate()


def test_validate_rejects_auto_prop_for_an_emg_only_set():
    """The crash this replaces: _build_noise_set's auto_prop gather is flow-only, so
    without this guard an EMG-only analysis with noise reduction on (auto_prop
    defaults to True) would reach it and kill the whole batch."""
    s = _emg_only_settings(rest_breath=1)
    s.processing.emg.remove_ecg = True
    s.processing.emg.noise.enabled = True
    assert s.processing.emg.noise.auto_prop is True     # the default this guards against
    with pytest.raises(SettingsError, match="auto_prop"):
        s.validate()


def test_validate_rejects_an_invalid_reference_mode_value():
    s = _flow_settings()
    s.processing.emg.noise.reference_mode = "bogus"
    with pytest.raises(SettingsError, match="reference_mode"):
        s.validate()


def test_validate_rejects_explicit_emg_only_mode_with_flow_declared():
    s = _flow_settings(reference_mode="rest_segments")
    with pytest.raises(SettingsError, match="only valid for an EMG-only signal set"):
        s.validate()


def test_validate_ignores_reference_mode_shape_checks_when_noise_disabled_but_not_the_form():
    """The enum/EMG-only-only checks are unconditional (cheap, shape-level, like
    segmentation.method's own checks) -- they fire even with noise reduction off."""
    s = _flow_settings(reference_mode="interburst")
    s.processing.emg.noise.enabled = False
    with pytest.raises(SettingsError, match="only valid for an EMG-only signal set"):
        s.validate()


# --------------------------------------------------------------------------------- #
# _reference_noise_clip: bit-identical to the pre-ticket algorithm
# --------------------------------------------------------------------------------- #

def _old_style_clip(settings, s):
    """The exact pre-M-22 algorithm, reimplemented here (not imported) so a change to
    the production function cannot silently make this comparison meaningless."""
    ns_cfg = settings.processing.emg.noise
    ref = ns_cfg.reference_file
    path = os.path.join(s.input.inputfolder, ref)
    fs = s.input.format.samplingfrequency
    if ns_cfg.use_expiration or not ns_cfg.reference_intervals:
        emg_full, ins, ex = _emg_segmented(path, s)
        return emg_full[ex]
    _load_result, emg_ecg, _diag = _load_and_ecg(path, s)
    parts = [emg_ecg[int(t0 * fs):int(t1 * fs)] for t0, t1 in ns_cfg.reference_intervals]
    return np.concatenate(parts, axis=0)


@requires_synth()
def test_reference_noise_clip_bit_identical_for_synth_settings_noise_true(tmp_path):
    """synth_settings(noise=True): use_expiration=False + explicit reference_intervals
    -> 'intervals' mode, unchanged from before this ticket."""
    s = synth_settings(tmp_path, noise=True)
    assert resolve_noise_reference_mode(s) == "intervals"
    legacy = to_legacy_ns(s)
    old = _old_style_clip(s, legacy)
    new = _reference_noise_clip(s, legacy)
    assert new.shape == old.shape
    assert np.array_equal(new, old)


@requires_synth()
def test_reference_noise_clip_bit_identical_for_expiration_mode(tmp_path):
    """The other half of 'auto': use_expiration=True (the schema default) -> the
    reference file's own expiration -- also unchanged."""
    s = synth_settings(tmp_path)
    n = s.processing.emg.noise
    n.enabled = True
    n.reference_file = "synth_case_A.csv"
    assert n.use_expiration is True
    assert resolve_noise_reference_mode(s) == "expiration"
    legacy = to_legacy_ns(s)
    old = _old_style_clip(s, legacy)
    new = _reference_noise_clip(s, legacy)
    assert new.shape == old.shape
    assert np.array_equal(new, old)


@requires_synth()
def test_reference_noise_clip_bit_identical_for_the_sample_recording(tmp_path):
    from respmech.core.sample import build_sample_settings, write_sample_recording

    desc = write_sample_recording(str(tmp_path))
    s = build_sample_settings(desc, str(tmp_path / "out"))
    assert resolve_noise_reference_mode(s) == "intervals"   # sample uses explicit intervals
    legacy = to_legacy_ns(s)
    old = _old_style_clip(s, legacy)
    new = _reference_noise_clip(s, legacy)
    assert new.shape == old.shape
    assert np.array_equal(new, old)


@requires_synth()
def test_reference_noise_clip_bit_identical_for_a_migrated_v1_dict(tmp_path):
    """A migrated legacy noise_profile always sets use_expiration=True (migrate.py) --
    'expiration' mode, the same as the hand-built flow-bearing case above, but reached
    through migrate_dict instead of a hand-built Settings()."""
    from respmech.settingsio.migrate import migrate_dict

    legacy_dict = {
        "input": {"inputfolder": INPUT, "files": "synth_case_*.csv",
                  "format": {"samplingfrequency": 1000},
                  "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 9,
                           "column_volume": 6, "column_flow": 5,
                           "columns_emg": [2, 3, 4], "columns_entropy": []}},
        "processing": {
            "mechanics": {"breathseparationbuffer": 200, "separateby": "flow"},
            "emg": {"remove_ecg": True, "remove_noise": True,
                    "noise_profile": [["synth_case_A.csv", "synth_case_A.csv", [0.1, 0.9]]]},
        },
        "output": {"outputfolder": str(tmp_path)},
    }
    s, _report = migrate_dict(legacy_dict)
    assert resolve_noise_reference_mode(s) == "expiration"
    legacy = to_legacy_ns(s)
    old = _old_style_clip(s, legacy)
    new = _reference_noise_clip(s, legacy)
    assert new.shape == old.shape
    assert np.array_equal(new, old)


# --------------------------------------------------------------------------------- #
# decision 11: expiration mask excludes typed breaths, only for the reference call,
# and only when a typed breath actually exists
# --------------------------------------------------------------------------------- #

@requires_synth()
def test_expiration_mask_unchanged_without_any_typed_breath(tmp_path):
    s = synth_settings(tmp_path)
    legacy = to_legacy_ns(s)
    path = os.path.join(INPUT, "synth_case_A.csv")
    _emg, ins_off, ex_off = _emg_segmented(path, legacy, exclude_typed_from_expiration=False)
    _emg2, ins_on, ex_on = _emg_segmented(path, legacy, exclude_typed_from_expiration=True)
    assert np.array_equal(ins_off, ins_on)
    assert np.array_equal(ex_off, ex_on)


@requires_synth()
def test_expiration_mask_excludes_a_typed_breaths_expiration_when_the_flag_is_set(tmp_path):
    s = synth_settings(tmp_path)
    s.processing.breath_types.append(
        BreathTypeEntry(file="synth_case_A.csv", breath=1, kind="ic"))
    legacy = to_legacy_ns(s)
    path = os.path.join(INPUT, "synth_case_A.csv")
    _emg_off, _ins_off, ex_off = _emg_segmented(path, legacy, exclude_typed_from_expiration=False)
    _emg_on, _ins_on, ex_on = _emg_segmented(path, legacy, exclude_typed_from_expiration=True)
    # Filtering can only ever REMOVE True entries relative to the unfiltered mask, and
    # with a real typed breath present it must remove at least one.
    assert np.array_equal(ex_on & ex_off, ex_on)          # ex_on is a subset of ex_off
    assert ex_on.sum() < ex_off.sum()


# --------------------------------------------------------------------------------- #
# the new EMG-only 'rest_segments' clip -- concatenation of ONLY the rest-typed
# segments, using the analysis's own configured segmentation method
# --------------------------------------------------------------------------------- #

def _emg_only_synth_settings(tmp_path):
    return synth_settings(tmp_path, channels={
        "flow": None, "poes": None, "pgas": None, "pdi": None, "volume": None})


@requires_synth()
def test_rest_segments_clip_is_the_concatenation_of_only_the_rest_typed_segments(tmp_path):
    s = _emg_only_synth_settings(tmp_path)
    s.analysis.signals = ["emg"]
    s.processing.segmentation.method = "separators"
    s.processing.segmentation.separators = [
        SeparatorEntry(file="synth_case_A.csv", times_s=[2.0, 4.0])]
    s.processing.breath_types.append(
        BreathTypeEntry(file="synth_case_A.csv", breath=2, kind="rest"))
    s.processing.emg.remove_ecg = True
    s.processing.emg.noise.enabled = True
    s.processing.emg.noise.reference_file = "synth_case_A.csv"
    s.processing.emg.noise.auto_prop = False
    s.validate()                                        # must resolve cleanly
    assert resolve_noise_reference_mode(s) == "rest_segments"

    legacy = to_legacy_ns(s)
    clip = _reference_noise_clip(s, legacy)

    from respmech.core.pipeline import segment_file
    breaths, _trimmed = segment_file(s, legacy, os.path.join(INPUT, "synth_case_A.csv"))
    assert [b["kind"] for b in breaths.values()] == [None, "rest", None]
    expected = np.asarray(breaths[2]["emgcols"])
    assert clip.shape == expected.shape
    assert np.array_equal(clip, expected)


@requires_synth()
def test_rest_segments_clip_concatenates_multiple_rest_segments_in_order(tmp_path):
    """Edge case: more than one 'rest'-typed segment in the same file -- the clip must
    be the concatenation of ALL of them, in segment-number order, not just the first
    match (`_rest_segments_clip` builds its list from `breaths.values()`, which is
    already ordered by segment number -- this pins that it stays that way)."""
    s = _emg_only_synth_settings(tmp_path)
    s.analysis.signals = ["emg"]
    s.processing.segmentation.method = "separators"
    s.processing.segmentation.separators = [
        SeparatorEntry(file="synth_case_A.csv", times_s=[2.0, 4.0])]   # -> 3 segments
    s.processing.breath_types.append(
        BreathTypeEntry(file="synth_case_A.csv", breath=1, kind="rest"))
    s.processing.breath_types.append(
        BreathTypeEntry(file="synth_case_A.csv", breath=3, kind="rest"))
    s.processing.emg.remove_ecg = True
    s.processing.emg.noise.enabled = True
    s.processing.emg.noise.reference_file = "synth_case_A.csv"
    s.processing.emg.noise.auto_prop = False
    s.validate()
    assert resolve_noise_reference_mode(s) == "rest_segments"

    legacy = to_legacy_ns(s)
    clip = _reference_noise_clip(s, legacy)

    from respmech.core.pipeline import segment_file
    breaths, _trimmed = segment_file(s, legacy, os.path.join(INPUT, "synth_case_A.csv"))
    assert [b["kind"] for b in breaths.values()] == ["rest", None, "rest"]
    expected = np.concatenate(
        [np.asarray(breaths[1]["emgcols"]), np.asarray(breaths[3]["emgcols"])], axis=0)
    assert clip.shape == expected.shape
    assert np.array_equal(clip, expected)


@requires_synth()
def test_rest_segments_clip_names_the_file_when_no_rest_segment_survives_at_runtime(tmp_path):
    """resolve_noise_reference_mode only checks the SETTINGS shape (does a
    BreathTypeEntry name this file with kind 'rest'?), not how many segments the file
    actually produces once loaded. A typed breath number outside the range the
    configured segmentation actually reaches (here: 'separators' with one separator ->
    2 segments, but the typed entry names segment 5) resolves 'rest_segments' at
    validate() time yet has zero matching segments at runtime -- building the clip
    must fail with a clear, named error instead of a bare empty-concatenate crash."""
    s = _emg_only_synth_settings(tmp_path)
    s.analysis.signals = ["emg"]
    s.processing.segmentation.method = "separators"
    s.processing.segmentation.separators = [
        SeparatorEntry(file="synth_case_A.csv", times_s=[2.0])]   # -> 2 segments
    s.processing.breath_types.append(
        BreathTypeEntry(file="synth_case_A.csv", breath=5, kind="rest"))   # out of range
    s.processing.emg.remove_ecg = True
    s.processing.emg.noise.enabled = True
    s.processing.emg.noise.reference_file = "synth_case_A.csv"
    s.processing.emg.noise.auto_prop = False
    s.validate()                                        # resolves cleanly at settings level
    assert resolve_noise_reference_mode(s) == "rest_segments"
    legacy = to_legacy_ns(s)
    with pytest.raises(ValueError, match="no 'rest'-typed segment"):
        _reference_noise_clip(s, legacy)


# --------------------------------------------------------------------------------- #
# resolve_noise_reference_mode_or_none (M-24) -- the UI-render-safe pairing, exactly
# like Capabilities.from_settings_or_none, for the same malformed-analysis.signals hazard.
# --------------------------------------------------------------------------------- #

def test_or_none_matches_the_raw_resolver_on_every_ordinary_settings_shape():
    """No behaviour change for anything that doesn't crash the raw function."""
    assert (resolve_noise_reference_mode_or_none(_flow_settings())
           == resolve_noise_reference_mode(_flow_settings()))
    assert (resolve_noise_reference_mode_or_none(_emg_only_settings(rest_breath=1))
           == resolve_noise_reference_mode(_emg_only_settings(rest_breath=1)))
    assert (resolve_noise_reference_mode_or_none(_emg_only_settings())
           == resolve_noise_reference_mode(_emg_only_settings()) == "unresolved")


def test_or_none_degrades_to_none_instead_of_raising_on_a_malformed_signals_field():
    """The actual bug this function was written to fix (found in self-review of M-24):
    a hand-edited bare-string analysis.signals crashed PreviewScreen's construction path
    (_refresh_noise_readout/_refresh_noise_reference_band call the raw resolver on every
    open, before Settings.validate() ever runs). Confirm both that the raw function still
    raises (so Settings.validate()'s own error reporting is unaffected) and that the _or_none
    pairing degrades cleanly instead."""
    s = Settings()
    s.analysis.signals = "flow"          # malformed: a bare string, not a list
    s.processing.emg.noise.reference_file = "ref.csv"   # a real UI render would have this set
    with pytest.raises(TypeError):
        resolve_noise_reference_mode(s)
    assert resolve_noise_reference_mode_or_none(s) is None
