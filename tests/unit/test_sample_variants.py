"""Sample variants ('full', 'flow', 'flow_poes', 'emg') — a column subset of the SAME
deterministic signals, so 'Explore with sample data' can follow whichever signal set the
current analysis declares (Setup > Signals, or the new-analysis picker) instead of
always opening the complete demo recording. 'emg' is EMG-only: no flow/pressure
columns at all, split into segments via manual separators at the generator's own
breath-to-breath boundaries.

The 'full' variant is the pre-existing sample: its bytes are pinned by SHA-256 here so a
future change to ``_signals()``/``write_sample_recording`` cannot silently perturb it —
every other RespMech test that runs against the sample (ECG/noise-reduction figures,
CHANGELOG screenshots, the README-figure generator) assumes this exact recording.
"""
import hashlib
import os

import pytest

from respmech.core.analysis.signals import Capabilities
from respmech.core.sample import (
    FILENAME, VARIANT_FILENAMES, VARIANT_SIGNALS, build_sample_settings,
    sample_variant_for_mode, write_sample_recording,
)

# Locks the exact bytes of the 'full' sample recording as it existed before sample
# variants were introduced (verified against a checkout of the prior core/sample.py
# during development). A deliberate change to the physiological model or CSV formatting
# updates this constant in the SAME commit as the change, with the reason recorded in
# the commit/CHANGELOG — never "fixed" to make the test pass without also explaining
# why the recording changed.
_FULL_SHA256 = "871514694152ef122484e7c9e19ae942514df50c1f94fe4f30d69907077d8fa6"


def test_full_variant_is_the_default_and_hash_pinned(tmp_path):
    desc = write_sample_recording(str(tmp_path))
    assert desc["variant"] == "full"
    assert desc["filename"] == FILENAME
    digest = hashlib.sha256(open(desc["path"], "rb").read()).hexdigest()
    assert digest == _FULL_SHA256, (
        "the 'full' sample recording's bytes changed -- if this is deliberate, update "
        "_FULL_SHA256 here and say why in the commit/CHANGELOG, since other tests and "
        "screenshots assume this exact recording")


@pytest.mark.parametrize("variant, expected_filename, expected_roles", [
    ("flow", "sample_recording_flow.csv", {"flow", "volume"}),
    ("flow_poes", "sample_recording_flow_poes.csv", {"flow", "volume", "poes"}),
])
def test_each_variant_writes_its_own_filename_with_only_its_declared_columns(
        tmp_path, variant, expected_filename, expected_roles):
    desc = write_sample_recording(str(tmp_path), variant=variant)
    assert desc["variant"] == variant
    assert desc["filename"] == expected_filename == VARIANT_FILENAMES[variant]
    mapping = desc["mapping"]
    present = {role for role in ("flow", "volume", "poes", "pgas", "pdi")
               if mapping.get(role) is not None}
    assert present == expected_roles
    assert mapping["emg"] == [] and mapping["entropy"] == []
    header = open(desc["path"], encoding="utf-8").readline().strip().lstrip("#").split(",")
    assert header == ["time", *sorted(expected_roles, key=lambda r: mapping[r])]


def test_variants_write_to_separate_files_so_one_never_overwrites_another(tmp_path):
    """Acceptance criterion: each variant is analysable in its own folder without
    overwriting another -- writing all four into the SAME folder must still leave four
    distinct, independently-readable files (the filenames alone guarantee this)."""
    paths = {v: write_sample_recording(str(tmp_path), variant=v)["path"]
             for v in ("full", "flow", "flow_poes", "emg")}
    assert len(set(paths.values())) == 4
    for p in paths.values():
        assert os.path.isfile(p)


def test_unknown_variant_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown sample variant"):
        write_sample_recording(str(tmp_path), variant="bogus")


@pytest.mark.parametrize("variant", ["full", "flow", "flow_poes", "emg"])
def test_build_sample_settings_declares_the_matching_signal_set(tmp_path, variant):
    desc = write_sample_recording(str(tmp_path / "input"), variant=variant)
    s = build_sample_settings(desc, str(tmp_path / "output"), variant=variant)
    s.validate()
    assert s.analysis.signals == VARIANT_SIGNALS[variant]
    caps = Capabilities.from_settings(s)
    assert caps.mode == {"full": "full", "flow": "flow_only", "flow_poes": "poes_only",
                         "emg": "emg_only"}[variant]


@pytest.mark.parametrize("variant", ["flow", "flow_poes"])
def test_non_full_variants_skip_the_emg_pipeline(tmp_path, variant):
    """The flow/flow_poes variants have no EMG channel at all -- ECG removal and noise
    reduction (meaningless without one) must stay at their Settings() default rather
    than being switched on over nothing."""
    from respmech.core.settings import Settings
    desc = write_sample_recording(str(tmp_path / "input"), variant=variant)
    s = build_sample_settings(desc, str(tmp_path / "output"), variant=variant)
    default = Settings().processing.emg
    assert s.processing.emg.remove_ecg == default.remove_ecg is False
    assert s.processing.emg.noise.enabled == default.noise.enabled is False
    assert s.input.channels.emg == []


@pytest.mark.parametrize("variant", ["full", "flow", "flow_poes"])
def test_each_variant_is_analysable_end_to_end(tmp_path, variant):
    """Every flow-bearing variant runs the real pipeline (not just Settings.validate())
    and detects the same 9 breaths as the full recording -- the underlying breathing
    trace is identical across variants, only the exported columns differ. 'emg' has no
    flow to segment by breath at all -- see its own dedicated tests below for the
    equivalent (segment) check."""
    from respmech.core.pipeline import run_batch
    desc = write_sample_recording(str(tmp_path / "input"), variant=variant)
    s = build_sample_settings(desc, str(tmp_path / "output"), variant=variant)
    result = run_batch(s)
    assert not result.failed_files, result.failed_files
    fr = next(iter(result.ok_files.values()))
    assert len([b for b in fr.breaths.values() if not b["ignored"]]) == 9


def test_sample_variant_for_mode_maps_the_two_dedicated_shapes_and_falls_back_to_full():
    assert sample_variant_for_mode("flow_only") == "flow"
    assert sample_variant_for_mode("poes_only") == "flow_poes"
    assert sample_variant_for_mode("full") == "full"
    # This pure, mode-string-only mapping deliberately has no 'emg_only' entry: that
    # mode DOES have its own dedicated sample variant ('emg'), but reaching
    # it needs a check ordered BEFORE settings_screen.open_sample_analysis's separate
    # "EMG in the declared set always wins to 'full'" rule (caps.emg is ALSO true for a
    # genuine EMG-only set — it is the only signal declared), so that screen checks
    # Capabilities.mode directly instead of calling this function for that one case —
    # see its own docstring. An unclassifiable 'custom' set still has no variant either.
    assert sample_variant_for_mode("emg_only") == "full"
    assert sample_variant_for_mode("custom") == "full"


def test_emg_variant_writes_only_emg_columns_with_its_own_filename(tmp_path):
    desc = write_sample_recording(str(tmp_path), variant="emg")
    assert desc["filename"] == "sample_recording_emg.csv" == VARIANT_FILENAMES["emg"]
    mapping = desc["mapping"]
    assert mapping["emg"] == [2, 3, 4]
    for role in ("flow", "volume", "poes", "pgas", "pdi"):
        assert mapping[role] is None
    header = open(desc["path"], encoding="utf-8").readline().strip().lstrip("#").split(",")
    assert header == ["time", "EMG1", "EMG2", "EMG3"]


def test_emg_variant_places_a_separator_at_each_internal_breath_boundary(tmp_path):
    """8 breath-to-breath boundaries (N_BREATHS - 1 = 9 - 1) -- the lead-in/breath-0
    boundary is deliberately excluded (see _signals()'s own comment), so the recording
    splits into exactly 9 segments, matching the generator's own breath count."""
    desc = write_sample_recording(str(tmp_path), variant="emg")
    onsets = desc["breath_onsets_s"]
    assert len(onsets) == 8
    assert onsets == sorted(onsets)
    assert all(onsets[i] < onsets[i + 1] for i in range(len(onsets) - 1))


def test_emg_variant_declares_separators_segmentation_with_an_intervals_noise_reference(tmp_path):
    from respmech.core.settings import resolve_noise_reference_mode
    desc = write_sample_recording(str(tmp_path / "input"), variant="emg")
    s = build_sample_settings(desc, str(tmp_path / "output"), variant="emg")
    s.validate()
    assert s.processing.segmentation.method == "separators"
    assert len(s.processing.segmentation.separators) == 1
    entry = s.processing.segmentation.separators[0]
    assert entry.file == desc["filename"]
    assert entry.times_s == desc["breath_onsets_s"]
    assert s.processing.emg.noise.enabled is True
    assert s.processing.emg.noise.auto_prop is False
    assert resolve_noise_reference_mode(s) == "intervals"


def test_emg_variant_is_analysable_end_to_end_as_nine_segments(tmp_path):
    """The real pipeline (not just Settings.validate()) splits the EMG-only recording
    into exactly 9 segments -- the generator's own breath count -- via the manual
    separators build_sample_settings wires in."""
    from respmech.core.pipeline import run_batch
    desc = write_sample_recording(str(tmp_path / "input"), variant="emg")
    s = build_sample_settings(desc, str(tmp_path / "output"), variant="emg")
    result = run_batch(s)
    assert not result.failed_files, result.failed_files
    fr = next(iter(result.ok_files.values()))
    assert len(fr.breaths) == 9
    assert fr.breaths_table is not None and len(fr.breaths_table) == 9
