"""The one piece of arithmetic both resting-span guards must share.

A noise reference is judged by how many STFT frames it yields, because the
per-frequency std that ``core.noise.NoiseProfile.from_clip`` builds is only
stable with enough frames behind it. Two places in the UI reason about that
number for the SAME span: the noise-profile dialog while the user drags, and
the EMG tab when the span is applied. Until ticket D09 they measured different
things — the dialog warned on a fixed 0.5 s width (about processing time, which
the reference's width does not even drive), while the tab counted frames — and
at 1000 Hz the two guards excluded each other: every span the dialog accepted
in silence, the tab called "short", and the only spans the tab called "good"
drew the dialog's warning. A user could follow the app's advice precisely and
end up with a statistically unstable profile.

So the arithmetic lives here once, pure and import-light, and both guards call
it. Deliberately NO import of :mod:`respmech.core.noise`: this module is pulled
in by the preview screen at startup, and the core module carries the heavy DSP
imports that ``test_gui_startup_does_not_import_the_compute_core`` exists to
keep out of startup. The two constants below therefore MIRROR the core's values
instead of importing them, and ``tests/unit/test_stft_frames.py`` pins the
mirror: if either side changes, that test names this file.
"""
from __future__ import annotations

#: Frames below this make the per-frequency std unstable. Mirrors the hardcoded
#: ``8`` in ``core.noise.NoiseProfile.from_clip`` (which this ticket must not
#: touch); the test suite asserts the two agree.
MIN_STABLE_FRAMES = 8

#: Mirror of ``core.noise.DEFAULT_WIN`` / ``DEFAULT_HOP`` — the fallback when a
#: caller has no settings object at hand. The live call sites thread the real
#: values from ``settings.processing.emg.noise``, so these only ever matter in
#: bare test construction.
DEFAULT_WIN_LENGTH = 256
DEFAULT_HOP_LENGTH = 64


def stft_frame_count(span_samples: int, win_length: int, hop_length: int) -> int:
    """How many STFT frames a reference span of ``span_samples`` yields.

    The same formula the EMG tab has used all along (``_apply_noise_reference``):
    one frame once the window fits, plus one per hop. Guarded with ``max(0, …)``
    so a span shorter than the window counts as the single partial frame the
    STFT still produces, rather than going negative.
    """
    return 1 + max(0, int(span_samples) - int(win_length)) // max(1, int(hop_length))


def min_seconds_for_frames(frames: int, fs: float,
                           win_length: int, hop_length: int) -> float:
    """The shortest span, in seconds, that yields at least ``frames`` frames.

    This is what turns the guard into advice: instead of "too short", the dialog
    can say how far to drag. At 1000 Hz with the 256/64 defaults, 8 frames need
    ``256 + 7·64 = 704`` samples, i.e. 0.704 s; at 2000 Hz the same 704 samples
    are 0.352 s — the frequency dependence that made the fixed 0.5 s rule wrong
    in both directions.
    """
    if fs <= 0:
        return float("inf")
    samples = int(win_length) + (max(1, int(frames)) - 1) * max(1, int(hop_length))
    return samples / float(fs)
