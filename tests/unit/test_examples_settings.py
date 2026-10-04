"""`examples/settings.toml`'s poes/pgas/pdi channels used to be commented
`# REQUIRED`, which stopped being true once a signal set could omit them (flow-only,
flow+poes — see `core.analysis.signals`). This locks the corrected wording and that the
file still validates unedited, so a future edit cannot silently reintroduce a REQUIRED
claim `Settings.validate()` no longer makes."""
import os

from respmech.cli.__main__ import main as cli_main

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "examples", "settings.toml")


def test_example_settings_still_validate(capsys):
    assert os.path.isfile(_EXAMPLE)
    assert cli_main(["validate", _EXAMPLE]) == 0
    out = capsys.readouterr().out
    assert "matches 1 file(s)" in out


def test_poes_pgas_pdi_are_documented_as_optional_not_required():
    text = open(_EXAMPLE, encoding="utf-8").read()
    for role in ("poes", "pgas", "pdi"):
        line = next(ln for ln in text.splitlines() if ln.strip().startswith(f"{role} "))
        assert "REQUIRED" not in line, f"{role} line still claims REQUIRED: {line!r}"
        assert "optional" in line, f"{role} line does not say optional: {line!r}"
    # flow/volume are genuinely required (flow unconditionally; volume unless
    # integrate_from_flow) and must keep saying so -- this test only relaxes the three
    # pressure channels, never flow/volume.
    flow_line = next(ln for ln in text.splitlines() if ln.strip().startswith("flow "))
    assert "REQUIRED" in flow_line


# --- the commented example tables ---------------------------------------------------------
# examples/settings.toml carries commented-out examples of the tables an analysis can add
# (signal set, typed breaths, references, subjects, lung volumes, separators ...). They are
# only useful if they are still true: each block, uncommented, must be read by this version
# without a single unrecognised key, and must still validate.

import re
import tomllib

import pytest

from respmech.core.settings import Settings

_HEADER = re.compile(r"^# \[\[?[a-z_.]+\]\]?\s*$")


def _commented_blocks():
    """Every run of commented lines (ended by a blank or uncommented line) from its first
    TOML table header on, with the leading '# ' removed."""
    lines = open(_EXAMPLE, encoding="utf-8").read().split("\n") + [""]
    blocks, cur = [], []
    for ln in lines:
        if ln.startswith("# ") or ln == "#":
            cur.append(ln)
        else:
            if cur:
                blocks.append(cur)
            cur = []
    out = []
    for b in blocks:
        heads = [i for i, ln in enumerate(b) if _HEADER.match(ln)]
        if heads:
            body = "\n".join(ln[2:] if ln.startswith("# ") else "" for ln in b[heads[0]:])
            out.append((b[heads[0]][2:].strip(), body))
    return out


def _merge(base: dict, extra: dict) -> dict:
    """Merge ``extra`` over ``base``: tables merge, lists of tables append."""
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        elif isinstance(v, list) and isinstance(out.get(k), list):
            out[k] = out[k] + v
        else:
            out[k] = v
    return out


_BLOCKS = _commented_blocks()


def _base() -> dict:
    with open(_EXAMPLE, "rb") as fh:
        return tomllib.load(fh)


def test_the_example_has_a_commented_block_for_every_table_it_documents():
    heads = " ".join(h for h, _ in _BLOCKS)
    for table in ("analysis", "processing.breath_types", "processing.references",
                  "processing.reference_defaults", "input.subjects", "processing.lung_volume",
                  "processing.segmentation.overrides", "processing.segmentation"):
        assert f"[{table}]" in heads or f"[[{table}]]" in heads, table


@pytest.mark.parametrize("head,body", _BLOCKS, ids=[h for h, _ in _BLOCKS])
def test_each_commented_example_is_read_without_an_unrecognised_key(head, body):
    d = _merge(_base(), tomllib.loads(body))
    s = Settings.from_dict(d)
    assert s.unknown == {}, f"{head}: keys this version does not recognise: {s.unknown}"


@pytest.mark.parametrize("head,body", [b for b in _BLOCKS if "segmentation]" not in b[0]],
                         ids=[h for h, _ in _BLOCKS if "segmentation]" not in h])
def test_each_flow_bearing_commented_example_still_validates(head, body):
    d = _merge(_base(), tomllib.loads(body))
    Settings.from_dict(d).validate()


def test_the_emg_only_example_validates_on_an_emg_only_analysis():
    """The separators and the segmentation method belong to an analysis that declares EMG
    only, with no flow channel: assemble that around them."""
    emg_only = {"schema_version": 3,
                "input": {"folder": "input", "files": "sample_recording_emg.csv",
                          "format": {"sampling_frequency": 1000},
                          "channels": {"emg": [2, 3, 4], "entropy": [2, 3, 4]}},
                "analysis": {"signals": ["emg"]},
                "processing": {"emg": {"detect_channel": 0}},
                "output": {"folder": "output"}}
    seg = [b for h, b in _BLOCKS if h == "[processing.segmentation]"]
    assert seg, "the EMG-only example must be there"
    assert "separators" in seg[0] and "times_s" in seg[0]   # the method AND its boundaries
    d = _merge(emg_only, tomllib.loads(seg[0]))
    s = Settings.from_dict(d)
    assert s.unknown == {}
    s.validate()
