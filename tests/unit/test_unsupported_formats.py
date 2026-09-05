"""``.xls`` (the old binary Excel format) is deliberately unsupported: xlrd is not a
dependency, so a real ``.xls`` file fails with an ImportError inside pandas rather than
loading. Only ``.xlsx`` is registered as the Excel loader now (ticket: installation,
platforms and formats, 12.3/K-051).

Every one of the three places that gate which files a run or the UI will touch on the
extension alone must agree, or a user could open/preview a ``.xls`` file in the channel
picker and then have the actual run reject it with an unrelated-looking error. None of
these paths need to read real file content for the extension check itself, so a
non-existent path is enough."""
from respmech.core.io.loaders import DataValidationError, load as core_load
from _helpers import requires_synth, synth_settings

pytestmark = requires_synth()


def test_core_load_rejects_xls(tmp_path):
    from respmech.core._legacy_ns import to_legacy_ns
    s = synth_settings(str(tmp_path))
    try:
        core_load(str(tmp_path / "recording.xls"), to_legacy_ns(s))
        assert False, "expected DataValidationError"
    except DataValidationError as e:
        assert "Unsupported input file type: .xls" in str(e)


def test_core_load_still_accepts_xlsx_extension(tmp_path):
    """Not a full round trip (no real .xlsx content) -- just proves the extension itself
    still resolves to the Excel loader instead of also being rejected."""
    from respmech.core._legacy_ns import to_legacy_ns
    s = synth_settings(str(tmp_path))
    try:
        core_load(str(tmp_path / "recording.xlsx"), to_legacy_ns(s))
    except DataValidationError as e:
        assert "Unsupported input file type" not in str(e)
    except Exception:
        pass  # any other failure is about the (nonexistent) file content, not the extension


def test_load_raw_matrix_rejects_xls(tmp_path):
    from respmech.ui.workers import load_raw_matrix
    s = synth_settings(str(tmp_path))
    try:
        load_raw_matrix(s, str(tmp_path / "recording.xls"))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "Unsupported input file type: .xls" in str(e)


def test_probe_data_columns_excludes_xls(tmp_path):
    """A .xls file must not be offered as a selectable data file: the manifest scanner
    treats a None column count as 'not a loadable data file' and drops it from the list."""
    from respmech.ui.workers import probe_data_columns
    s = synth_settings(str(tmp_path))
    assert probe_data_columns(s, str(tmp_path / "recording.xls")) is None
