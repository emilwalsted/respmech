"""Cooperative cancellation (core._cancel): a supplied cancel_check() that returns True
aborts a heavy compute early with Cancelled; the default (None) path never raises and is
the byte-identical code the golden pins."""
import numpy as np
import pytest

from respmech.core._cancel import Cancelled, check

from _helpers import requires_synth, synth_settings  # noqa: F401


def test_check_helper_only_raises_when_cancelled():
    check(None)                 # no checker -> no-op
    check(lambda: False)        # not cancelled -> no-op
    with pytest.raises(Cancelled):
        check(lambda: True)
    assert issubclass(Cancelled, BaseException) and not issubclass(Cancelled, Exception)


def _ecg_signal():
    sig = np.zeros((1200, 2), dtype=float)
    sig[100::200, :] = 1.0      # regular peaks on both channels
    return sig


def test_remove_ecg_aborts_on_cancel_but_not_by_default():
    from respmech.core.emg import remove_ecg
    sig = _ecg_signal()
    kw = dict(samplingfrequency=1000, ecgminheight=0.5, ecgmindistance=0.05,
              ecgminwidth=0.0, windowsize=0.4)
    out, _w, _p = remove_ecg(np.array(sig), sig[:, 0], cancel_check=None, **kw)   # no raise
    assert out is not None
    with pytest.raises(Cancelled):
        remove_ecg(np.array(sig), sig[:, 0], cancel_check=lambda: True, **kw)


def test_noise_build_and_prop_selection_abort_on_cancel():
    from respmech.core.noise import build_profiles, select_prop_decrease
    clip = np.random.RandomState(0).standard_normal((2048, 2))
    profs = build_profiles(clip, 1000)                       # baseline, no checker
    with pytest.raises(Cancelled):
        build_profiles(clip, 1000, cancel_check=lambda: True)
    act = np.random.RandomState(1).standard_normal((2048, 2))
    with pytest.raises(Cancelled):
        select_prop_decrease(profs, act, act, 1000, cancel_check=lambda: True)
    # not cancelled -> completes and returns a (prop, report)
    prop, report = select_prop_decrease(profs, act, act, 1000, cancel_check=lambda: False)
    assert 0.0 < prop <= 1.0 and "channels" in report


@requires_synth()
def test_run_batch_aborts_inside_the_mechanics_loop(tmp_path):
    """The heavy per-breath mechanics/entropy compute is the part a single-file preview
    "batch" job spends its time in, and it used to be UNCANCELLABLE (run_batch only checked
    cancel_check between files) — so a superseded/settings-changed/shut-down preview batch ran
    the whole file to completion on its worker thread, leaking a GIL-bound thread that (on the
    2-core headless Windows CI) starved the reactive tests into a timeout cascade. It is now
    cooperatively cancellable: a checker that flips True only AFTER run_batch has entered the
    file aborts with Cancelled from inside compute.calculatemechanics/sample_entropy instead of
    finishing the file.

    Every guard run_batch checks BEFORE the file (the noise-profile build, the cross-file
    reference forepass and its post-forepass check, the per-file guard) silently returns an
    (empty-batch) result on True rather than raising, exactly like the between-files guard
    always has, so ``cc()`` must answer False through all of them to reach a real in-file
    checkpoint at all."""
    from respmech.core.pipeline import run_batch
    s = synth_settings(str(tmp_path),
                       data_out={"saveaveragedata": True, "savebreathbybreathdata": True})
    # The checker flips True only once run_batch has actually ENTERED the file (its own
    # ``file_start`` progress event), never on a call count. A count has to be re-tuned
    # every time a pre-file guard is added (it already had to be once, for the forepass),
    # and until it is, the flip lands on one of those guards and the run ends with a quiet
    # "cancelled" return instead of the in-file Cancelled this test is about.
    entered = {"file": False}
    calls = {"after_entry": 0}

    def on_progress(ev):
        if ev.kind == "file_start":
            entered["file"] = True

    def cc():
        if entered["file"]:
            calls["after_entry"] += 1
        return entered["file"]

    with pytest.raises(Cancelled):
        run_batch(s, progress=on_progress, cancel_check=cc, only_files=["synth_case_A.csv"])
    # proves the abort came from an IN-FILE checkpoint (segmentation / the per-breath loop),
    # not a pre-file guard: every check answered False until the file was entered, and at
    # least one check ran after that -- the one that raised.
    assert entered["file"] and calls["after_entry"] >= 1

    # default path (no checker) still runs the file to completion -> byte-identical golden path
    result = run_batch(s, cancel_check=None, only_files=["synth_case_A.csv"])
    assert result.files and "synth_case_A.csv" in result.files
