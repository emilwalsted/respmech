"""``ui.stft_frames`` is the single arithmetic both resting-span guards share.

Ticket D09: the dialog used to warn on a fixed 0.5 s (about processing time),
the tab judged the same span in STFT frames, and at 1000 Hz no span satisfied
both. These tests pin the shared arithmetic to the ticket's own measured
numbers, and pin the module's mirrored constants to the core values they must
never drift from (the module cannot import the core, because the preview screen
imports it at startup and the core carries the heavy DSP stack).
"""
from __future__ import annotations

from respmech.ui.stft_frames import (DEFAULT_HOP_LENGTH, DEFAULT_WIN_LENGTH,
                                     MIN_STABLE_FRAMES, min_seconds_for_frames,
                                     stft_frame_count)


def test_the_tickets_measured_numbers_at_1000_hz():
    # 0.5 s at 1000 Hz: 500 samples -> 4 frames, below stability
    assert stft_frame_count(500, 256, 64) == 4
    # 0.45 s -> 4 frames too (the trimmed-back-on-advice case from the ticket)
    assert stft_frame_count(450, 256, 64) == 4
    # 0.704 s -> exactly 8 frames, the stability threshold
    assert stft_frame_count(704, 256, 64) == 8
    assert stft_frame_count(704, 256, 64) >= MIN_STABLE_FRAMES


def test_the_tickets_counterexample_at_2000_hz():
    # The real analysis interval [5.1799, 5.6595] s at 2000 Hz: 0.4797 s ≈ 959
    # samples -> 11 frames. Good, and the old fixed 0.5 s rule stayed silent
    # only by luck; the frame count is what actually made it good.
    span = int(round(0.4797 * 2000))
    assert stft_frame_count(span, 256, 64) == 11


def test_min_seconds_is_the_inverse_of_the_frame_count():
    # 8 frames need 704 samples with the defaults: 0.704 s at 1000 Hz,
    # 0.352 s at 2000 Hz — the frequency dependence that broke the fixed rule.
    assert min_seconds_for_frames(8, 1000, 256, 64) == 0.704
    assert min_seconds_for_frames(8, 2000, 256, 64) == 0.352
    # and it really is the minimum: one sample less loses a frame
    for fs in (1000.0, 2000.0):
        s = min_seconds_for_frames(8, fs, 256, 64)
        assert stft_frame_count(int(round(s * fs)), 256, 64) == 8
        assert stft_frame_count(int(round(s * fs)) - 1, 256, 64) == 7


def test_a_span_shorter_than_the_window_counts_one_frame_not_negative():
    assert stft_frame_count(100, 256, 64) == 1
    assert stft_frame_count(0, 256, 64) == 1


def test_the_mirrored_constants_match_the_core():
    """The module mirrors the core's values instead of importing them (startup
    import weight). This test is the tripwire for drift: it MAY import the core."""
    from respmech.core import noise as core_noise

    assert DEFAULT_WIN_LENGTH == core_noise.DEFAULT_WIN
    assert DEFAULT_HOP_LENGTH == core_noise.DEFAULT_HOP
    # MIN_STABLE_FRAMES mirrors the hardcoded 8 in NoiseProfile.from_clip. The
    # core states it in a warning string rather than a constant, so assert
    # against the source text: if the core's threshold moves, this names both files.
    import inspect

    src = inspect.getsource(core_noise.NoiseProfile.from_clip)
    assert f"< {MIN_STABLE_FRAMES}" in src.replace("n_frames <", "<"), (
        "core.noise.NoiseProfile.from_clip no longer tests 'n_frames < 8' — update "
        "MIN_STABLE_FRAMES in ui/stft_frames.py to match, and these tests with it")


def test_the_dialog_and_tab_share_the_arithmetic_structurally():
    """The two guards must be one statement. That is guaranteed by both calling
    THIS function; here we pin that the modules actually do."""
    import inspect

    from respmech.ui import noise_profile_dialog
    from respmech.ui.screens.preview import _emg_noise

    assert "stft_frame_count" in inspect.getsource(noise_profile_dialog)
    assert "stft_frame_count" in inspect.getsource(_emg_noise)
