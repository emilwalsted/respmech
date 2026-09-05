"""K-204: ``outlier_rms_sd_limit`` filters EMG RMS outliers (``processoutliers`` compares
each breath's ``rms_max``/``poes_mininsp`` ratio against the others'), but with no EMG
channels configured there is no ``rms_max`` column to filter at all — ``build_breath_table``
never adds it (``if len(emgcols) > 0:``). A study-wide setting left on for an analysis that
happens to have no EMG channels used to KeyError every single file instead of simply having
nothing to do.
"""
from respmech.core import pipeline
from respmech.core.results import build_breath_table
from respmech.core._legacy_ns import to_legacy_ns
from respmech.core.settings import Settings

from _helpers import requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()


def test_outlier_limit_with_no_emg_channels_does_not_crash(tmp_path):
    s = synth_settings(tmp_path)
    s.input.channels.emg = []                          # no EMG -> no rms_max column
    s.processing.emg.outlier_rms_sd_limit = 3.0         # would have KeyError'd before the fix
    s.validate()
    result = pipeline.run_batch(s)
    assert result.failed_files == {}
    assert result.ok_files                              # at least one file actually processed


def test_outlier_limit_with_emg_channels_is_unaffected(tmp_path):
    """The guard must not change behaviour for the ordinary case (EMG channels present) —
    same file set, same limit, still runs clean."""
    s = synth_settings(tmp_path)
    s.processing.emg.outlier_rms_sd_limit = 3.0
    s.validate()
    result = pipeline.run_batch(s)
    assert result.failed_files == {}
    assert result.ok_files


def test_build_breath_table_guard_directly():
    """Unit-level: an empty EMG column list with the limit on must not reach
    ``processoutliers`` at all (which would KeyError on the absent ``rms_max``)."""
    s = Settings()
    s.input.format.sampling_frequency = 1000
    s.input.channels.flow = 5
    s.input.channels.volume = 6
    s.input.channels.poes = 7
    s.input.channels.pgas = 8
    s.input.channels.pdi = 9
    s.processing.emg.outlier_rms_sd_limit = 3.0
    ns = to_legacy_ns(s)
    breaths = {
        1: {"number": 1, "ignored": False, "mechanics": {"m": 1.0}, "wob": {"w": 1.0}},
        2: {"number": 2, "ignored": False, "mechanics": {"m": 2.0}, "wob": {"w": 2.0}},
        3: {"number": 3, "ignored": False, "mechanics": {"m": 3.0}, "wob": {"w": 3.0}},
    }
    per_breath, average_row = build_breath_table("x.csv", breaths, ns)
    assert len(per_breath) == 3
    assert average_row["file"].iloc[0] == "x.csv"
