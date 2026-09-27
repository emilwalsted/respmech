"""``_kinds_for_settings_path`` (``ui.screens.preview._jobs``), capability-aware (M-24).

``caps=None`` — every call site that predates this ticket — must reproduce the old,
flow-only rule byte-for-byte: 'the mechanics test run strips EMG' holds for a flow-bearing
signal set (the test run never touches EMG there), but NOT for an EMG-only one, where
``core.pipeline.run_batch``'s own S2 branch (M-21) builds the test run's mechanics FROM the
EMG channels — so a real EMG-only ``Capabilities`` must widen the EMG-related buckets to
include 'batch' (and the not-yet-existing 'segments' preview job M-25 adds — inert today,
since ``_schedule_all`` only ever dispatches a kind that is a member of ``_AUTO_KINDS``).
"""
from respmech.core.analysis.signals import Capabilities
from respmech.core.settings import Settings
from respmech.ui.screens.preview_screen import _AUTO_KINDS, _kinds_for_settings_path


def _flow_caps():
    s = Settings()
    ch = s.input.channels
    ch.flow, ch.volume, ch.emg = 1, 2, [3]
    return Capabilities.from_settings(s)


def _emg_only_caps():
    s = Settings()
    s.input.channels.emg = [1]
    s.analysis.signals = ["emg"]
    return Capabilities.from_settings(s)


def test_caps_defaults_to_none_and_reproduces_the_old_flow_only_rule():
    for path in ("input.channels.emg", "processing.emg.remove_ecg", "processing.emg.rms_window_s"):
        without = _kinds_for_settings_path(path)
        assert without == _kinds_for_settings_path(path, None)
        assert "batch" not in without
        assert "segments" not in without


def test_flow_bearing_capabilities_leave_the_old_rule_unchanged():
    caps = _flow_caps()
    assert caps.mode != "emg_only"
    for path in ("input.channels.emg", "processing.emg.remove_ecg", "processing.emg.rms_window_s"):
        kinds = _kinds_for_settings_path(path, caps)
        assert kinds == _kinds_for_settings_path(path)
        assert "batch" not in kinds
        assert "segments" not in kinds


def test_emg_only_capabilities_widen_the_channel_set_edit():
    caps = _emg_only_caps()
    assert caps.mode == "emg_only"
    kinds = _kinds_for_settings_path("input.channels.emg", caps)
    assert kinds == frozenset(("mech", "ecg", "emg_all", "emg_detail", "noise",
                              "batch", "segments"))


def test_emg_only_capabilities_widen_an_emg_processing_edit():
    caps = _emg_only_caps()
    kinds = _kinds_for_settings_path("processing.emg.rms_window_s", caps)
    assert {"ecg", "emg_all", "emg_detail", "noise", "batch", "segments"} == kinds


def test_emg_only_capabilities_do_not_widen_the_narrow_or_gated_emg_rules():
    """The EMG-only widening only touches the two wide EMG buckets above — the writes-only
    robust_peak gate, the batch-only outlier flag and the display-only trio keep their own
    narrower shape regardless of the signal set."""
    caps = _emg_only_caps()
    assert _kinds_for_settings_path("processing.emg.robust_peak.enabled", caps) == frozenset()
    assert _kinds_for_settings_path("processing.emg.robust_peak.enabled") == frozenset()
    assert (_kinds_for_settings_path("processing.emg.outlier_rms_sd_limit", caps)
           == frozenset(("batch",)))
    assert (_kinds_for_settings_path("processing.emg.normalization", caps)
           == frozenset(("emg_all", "emg_detail")))


def test_breath_types_and_separators_fall_through_to_every_auto_kind():
    """M-24's own omfang note: both paths are deliberately left on the wide catch-all — a
    typed breath changes the noise reference mask itself (decision 11), so it must NOT get
    the narrower processing.exclude_breaths treatment; a separator edit feeds the same
    ref_clip_key rest_segments branch a segmentation.buffer edit already does."""
    assert _kinds_for_settings_path("processing.breath_types") == frozenset(_AUTO_KINDS)
    assert _kinds_for_settings_path("processing.segmentation.separators") == frozenset(_AUTO_KINDS)
    caps = _emg_only_caps()
    assert _kinds_for_settings_path("processing.breath_types", caps) == frozenset(_AUTO_KINDS)
    assert (_kinds_for_settings_path("processing.segmentation.separators", caps)
           == frozenset(_AUTO_KINDS))
