"""Unit tests for core/'s user-facing error messages (ticket D25).

Background: three of the errors a real recording can raise pointed at settings that
do not exist in v2 (``inverseflow``/``avgresamplingobs`` are v1-only names; the v2
TOML keys are ``processing.volume.inverse_flow``/``processing.wob.avg_resampling_obs``,
surfaced in the GUI as 'Invert the flow signal'/'Average-breath resampling points').
A fourth reported a raw numpy ``ValueError`` about mismatched array dimensions
verbatim as the file's result, instead of a RespMech message.
"""
import ast
import re
from pathlib import Path

import numpy as np
import pytest

import respmech
from respmech.core import compute, pipeline
from respmech.core.compute import DegenerateBreathError, TrimError
from respmech.core.settings import Settings
from respmech.core._legacy_ns import to_legacy_ns

FS = 200


def _dummy_settings():
    s = Settings()
    s.input.format.sampling_frequency = FS
    return to_legacy_ns(s)


def test_trim_error_never_crossing_zero_names_v2_controls():
    n = 50
    flow = np.ones(n)                       # never <= 0: "below" is empty
    zeros = np.zeros(n)
    emgcolumns = np.zeros((n, 0))
    with pytest.raises(TrimError) as exc_info:
        compute.trim(zeros, flow, zeros, zeros, zeros, zeros, emgcolumns, _dummy_settings())
    msg = str(exc_info.value)
    assert "inverseflow" not in msg
    assert "Setup" in msg
    assert "Invert the flow signal" in msg
    assert "Preview & QC" in msg


def test_trim_error_empty_range_names_v2_controls():
    # above (flow>=0) has only the first sample; below (flow<=0) has the rest, so
    # endix (index 0) <= startix (index 1): the "empty range" branch.
    flow = np.array([1.0, -1.0, -1.0, -1.0])
    zeros = np.zeros(4)
    emgcolumns = np.zeros((4, 0))
    with pytest.raises(TrimError) as exc_info:
        compute.trim(zeros, flow, zeros, zeros, zeros, zeros, emgcolumns, _dummy_settings())
    msg = str(exc_info.value)
    assert "inverseflow" not in msg
    assert "Setup" in msg
    assert "Invert the flow signal" in msg


def test_resample_error_names_v2_controls_not_legacy_keys():
    breaths = {
        1: {"ignored": False, "number": 1,
            "inspiration": {"volume": np.array([1.0]), "poes": np.array([1.0])},
            "expiration": {"volume": np.array([1.0]), "poes": np.array([1.0])}},
    }
    s = _dummy_settings()
    with pytest.raises(ValueError) as exc_info:
        compute.calculateaveragebreaths(breaths, s)
    msg = str(exc_info.value)
    assert "avgresamplingobs" not in msg
    assert "Could not resample breath #1" in msg
    assert "Breath detection" in msg
    assert "Preview & QC" in msg
    assert "exclude this breath" in msg


def test_degenerate_breath_gives_a_respmech_error_not_numpys():
    """A phase whose single sample was squeezed to a 0-d array cannot be concatenated
    with the other, longer (1-d) phase — numpy's own message is about dimensions, not
    breaths, and used to reach the user verbatim (compute.py:284, ticket D25)."""
    insp = {"time": np.array([0.0, 0.01, 0.02]), "flow": np.array([0.0, 0.1, 0.2]),
            "volume": np.array([0.0, 0.1, 0.2]), "poes": np.array([0.0, 0.1, 0.2]),
            "pgas": np.array([0.0, 0.1, 0.2]), "pdi": np.array([0.0, 0.1, 0.2])}
    exp = {"time": np.array([0.03]).squeeze(), "flow": np.array([0.05]).squeeze(),
           "volume": np.array([0.25]).squeeze(), "poes": np.array([0.1]).squeeze(),
           "pgas": np.array([1.0]).squeeze(), "pdi": np.array([0.9]).squeeze()}
    with pytest.raises(DegenerateBreathError) as exc_info:
        compute._make_breath(7, exp, insp, False, [], [], "rec.csv")
    msg = str(exc_info.value)
    assert "Breath #7" in msg
    assert "rec.csv" in msg
    assert "dimension" not in msg.lower()   # the raw numpy wording is gone
    assert "exclude it" in msg or "exclude" in msg


def _write_recording(path, cut_offset, n_breaths=3, period_s=4.0, vt=0.6):
    """A synthetic sinusoidal-flow recording truncated a few samples after the last
    inspiration->expiration crossing — the same shape (an incomplete final breath) a
    real recording produced in a noise sweep (ticket D25). ``cut_offset`` controls how
    many samples of the final expiration survive the truncation."""
    per = int(period_s * FS)
    t = np.arange(n_breaths * per + 1) / FS
    amp = vt * np.pi / period_s
    flow = -amp * np.sin(2 * np.pi * t / period_s)
    zero_up = np.where((flow[:-1] < 0) & (flow[1:] >= 0))[0]
    cut = zero_up[-1] + cut_offset
    t, flow = t[:cut], flow[:cut]
    one = (1 - np.cos(2 * np.pi * (t % period_s) / period_s)) / 2
    volume = one * vt
    poes = -5.0 * np.sin(2 * np.pi * t / period_s) - 5.0
    pgas = 2.0 * np.sin(2 * np.pi * t / period_s) + 8.0
    np.savetxt(path, np.column_stack(
        [t, t * 0, t * 0, t * 0, flow, volume, poes, pgas, pgas - poes]), delimiter=",")


def _batch_settings(src, out):
    s = Settings()
    s.input.folder, s.input.files = str(src), "*.csv"
    s.input.format.sampling_frequency = FS
    c = s.input.channels
    c.flow, c.volume, c.poes, c.pgas, c.pdi = 5, 6, 7, 8, 9
    s.output.folder = str(out)
    return s.validate()


def test_degenerate_breath_end_to_end_batch_continues(tmp_path):
    """The exact trim()+segmentation path (not a hand-crafted phase dict) reaches
    DegenerateBreathError for one file, and the rest of the batch still completes."""
    src, out = tmp_path / "in", tmp_path / "out"
    src.mkdir(); out.mkdir()
    _write_recording(src / "bad.csv", cut_offset=4)
    _write_recording(src / "good.csv", cut_offset=1, n_breaths=6)
    result = pipeline.run_batch(_batch_settings(src, out))
    fr = result.failed_files["bad.csv"]
    assert fr.error_kind == "DegenerateBreathError"
    assert "numpy" not in fr.error.lower()
    assert "dimension" not in fr.error.lower()
    assert "bad.csv" in result.failed_files
    assert "good.csv" in result.ok_files
    assert len(result.ok_files["good.csv"].breaths_table) > 0


# --- no core/ message may leak a v1-only settings key --------------------------

_PKG_DIR = Path(respmech.__file__).parent
_MIGRATE_PY = _PKG_DIR / "settingsio" / "migrate.py"
_SETTINGS_PY = _PKG_DIR / "core" / "settings.py"
_CORE_DIR = _PKG_DIR / "core"
_KEY_RE = re.compile(r"[a-z][a-z0-9_]{4,}")


def _dataclass_field_names(path):
    """Every field name of every ``@dataclass`` in ``path`` — the v2 vocabulary a
    message is always allowed to use."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        is_dataclass = any(
            (isinstance(d, ast.Name) and d.id == "dataclass") or
            (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "dataclass")
            for d in node.decorator_list)
        if not is_dataclass:
            continue
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                names.add(item.target.id)
    return names


def _legacy_dict_keys(path):
    """Every literal key ``migrate.py`` reads FROM the legacy dict — ``d.get("key",
    ...)``, ``"key" in d`` and ``d["key"]`` — regardless of which legacy section ``d``
    is, since the names themselves (not their dict) are what a v2 message must not
    reuse."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            keys.add(node.args[0].value)
        elif isinstance(node, ast.Compare) and isinstance(node.left, ast.Constant) \
                and isinstance(node.left.value, str) \
                and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            keys.add(node.left.value)
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, str):
            keys.add(node.slice.value)
    return keys


def _legacy_only_keys():
    """Legacy dict keys that are NOT also a valid v2 field name — i.e. genuinely
    renamed/retired vocabulary (``inverseflow``, ``avgresamplingobs``, ...), not the
    identity-mapped ones (``resample``, ``files``, ``remove_ecg``, ...) that are both
    a legacy key AND a current v2 name and so are legitimate in a message."""
    legacy = _legacy_dict_keys(_MIGRATE_PY)
    v2 = _dataclass_field_names(_SETTINGS_PY)
    return {k for k in legacy if k not in v2 and _KEY_RE.fullmatch(k)}


def _raised_message_texts(path):
    """Static string fragments of every ``raise ...(...)`` and ``warnings.warn(...)``
    in ``path`` — the text a user can actually see, not docstrings or internal dict
    keys (which legitimately keep legacy-shaped names in ``_legacy_ns``/``compute``,
    the module docstring explains why)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    texts = []

    def collect(subtree):
        for n in ast.walk(subtree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) > 3:
                texts.append(n.value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            collect(node.exc)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "warn":
            for a in node.args:
                collect(a)
    return texts


def test_core_messages_never_leak_a_legacy_settings_key():
    legacy_only = _legacy_only_keys()
    assert "inverseflow" in legacy_only and "avgresamplingobs" in legacy_only  # sanity
    offenders = []
    for path in sorted(_CORE_DIR.glob("*.py")):
        for text in _raised_message_texts(path):
            for key in legacy_only:
                if key in text:
                    offenders.append((path.name, key, text))
    assert not offenders, (
        "legacy (pre-v2) settings key(s) leaked into a raised core/ message: "
        f"{offenders}")
