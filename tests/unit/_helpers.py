"""Pure test helpers shared across the unit suite (no fixtures — import these directly).

The one canonical synthetic-input settings builder lives here so the channel mapping
(poes 7 · pgas 8 · pdi 9 · volume 6 · flow 5 · EMG 2,3,4 · entropy 10,11,12) is defined
once instead of copy-pasted — previously under six names across a dozen files, free to
drift. ``synth_legacy_dict`` returns the pre-migration dict; ``synth_settings`` returns a
migrated ``Settings`` with folders resolved (and optional shared-profile noise wired).
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT = os.path.join(ROOT, "tests", "golden", "input")


def requires_synth():
    """A ``pytest.mark.skipif`` for tests that need the committed synthetic recordings."""
    import pytest
    return pytest.mark.skipif(
        not os.path.exists(os.path.join(INPUT, "synth_case_A.csv")),
        reason="synthetic input absent")


def synth_legacy_dict(out="", *, samplingfrequency=1000, remove_ecg=False, remove_noise=False,
                      avgresamplingobs=300, calcwobfromaverage=None, data_out=None):
    """The canonical synthetic-input legacy settings dict (for ``migrate_dict``)."""
    mech = {"breathseparationbuffer": 200, "separateby": "flow", "avgresamplingobs": avgresamplingobs}
    if calcwobfromaverage is not None:
        mech["calcwobfromaverage"] = calcwobfromaverage
    return {
        "input": {"inputfolder": INPUT, "files": "synth_case_*.csv",
                  "format": {"samplingfrequency": samplingfrequency},
                  "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 9,
                           "column_volume": 6, "column_flow": 5,
                           "columns_emg": [2, 3, 4], "columns_entropy": [10, 11, 12]}},
        "processing": {"mechanics": mech,
                       "emg": {"remove_ecg": remove_ecg, "remove_noise": remove_noise}},
        "output": {"outputfolder": str(out), "data": data_out if data_out is not None else {}},
    }


def synth_settings(out="", *, noise=False, **kw):
    """Migrated ``Settings`` over the canonical synthetic input, folders resolved.
    ``noise=True`` wires the shared-profile EMG noise reduction (reference synth_case_A,
    expiration window) exactly as the reactive-preview tests expect."""
    from respmech.settingsio.migrate import migrate_dict
    if noise:
        kw.setdefault("remove_ecg", True)
    s, _ = migrate_dict(synth_legacy_dict(out, remove_noise=noise, **kw))
    s.input.folder = INPUT
    s.output.folder = str(out)
    if noise:
        n = s.processing.emg.noise
        n.enabled = True
        n.reference_file = "synth_case_A.csv"
        n.use_expiration = False
        n.reference_intervals = [[1.0, 5.0]]
        n.auto_prop = True
    return s


def is_dark_hex(h):
    """True when a #RRGGBB colour reads as a dark surface (perceived luminance)."""
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (0.299 * r + 0.587 * g + 0.114 * b) < 110
