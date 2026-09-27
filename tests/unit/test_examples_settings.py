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
