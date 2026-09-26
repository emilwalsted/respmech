#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic golden test — verifies the NEW core (src/respmech) reproduces
golden_reference.json.

The golden reference is the newest code's output (Emil's ground-truth decision):
mechanics/WOB/EMG-RMS are byte-identical to the pre-refactor legacy output; the
entropy columns reflect the deliberate bug #2 fix (entropy computed on trimmed,
aligned data). See golden_newcore.py.

Run with (pinned env — see requirements-golden.txt):
    pytest tests/golden/test_golden.py -v
"""
import math
import os

import pytest

import golden_newcore as gc

RTOL = 1e-9
ATOL = 1e-12


@pytest.fixture(scope="module")
def golden():
    import json
    with open(gc.GOLDEN) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def current():
    return gc.build_all()


def _close(path, a, b):
    if isinstance(a, str) or isinstance(b, str) or a is None or b is None or \
            isinstance(a, bool) or isinstance(b, bool):
        assert a == b, f"{path}: {a!r} != {b!r}"
        return
    if math.isnan(a) and math.isnan(b):
        return
    assert math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL), f"{path}: {a!r} != {b!r}"


def _cmp(path, a, b):
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: key mismatch {set(a) ^ set(b)}"
        for k in a:
            _cmp(f"{path}.{k}", a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _cmp(f"{path}[{i}]", x, y)
    else:
        _close(path, a, b)


@pytest.mark.parametrize("scenario", sorted(gc.mg.SCENARIOS))
def test_scenario_matches_golden(scenario, golden, current):
    assert scenario in golden
    _cmp(scenario, golden[scenario], current[scenario])


def test_legacy_scenarios_have_only_the_three_top_level_keys(golden):
    """Every LEGACY_SCENARIOS entry in golden_reference.json has precisely
    {average_breathdata, per_file, processed} at its top level — never a
    manoeuvres/per_file_scalars key a future v2 scenario might add. Guards the
    LEGACY_SCENARIOS/V2_SCENARIOS split: the frozen v1 oracle (make_golden.py's
    run_all) has no way to produce those newer keys, so a legacy scenario gaining
    one would mean golden_newcore.py wrote over an oracle-checked entry."""
    expected = {"average_breathdata", "per_file", "processed"}
    for name in gc.mg.LEGACY_SCENARIOS:
        assert name in golden, f"{name}: missing from golden_reference.json"
        extra = set(golden[name]) - expected
        assert not extra, f"{name}: unexpected top-level key(s) {extra}"


def test_v2_scenarios_never_reach_the_legacy_oracle(monkeypatch):
    """run_all() (the frozen v1 oracle, legacy/respmech.py) must iterate
    LEGACY_SCENARIOS only, never the full SCENARIOS union. ``load_respmech`` and
    ``collect_outputs`` are stubbed out (the real oracle needs a scipy API removed
    upstream for one of the five legacy scenarios, and there is nothing to collect
    from a stub run anyway) so this stays a fast, structural check of run_all()'s
    OWN iteration, not a re-run of the real analysis. A SCENARIOS entry that is not
    in LEGACY_SCENARIOS has no legacy-dict shape at all (a plain object here, not an
    override dict) — it would raise inside ``deep_update`` the moment run_all() so
    much as looked at it, so completing with exactly len(LEGACY_SCENARIOS) stub
    calls is the proof that it never does."""
    calls = []
    fake_module = type("FakeRespmech", (), {"analyse": staticmethod(lambda settings: calls.append(settings))})()
    monkeypatch.setattr(gc.mg, "load_respmech", lambda: fake_module)
    monkeypatch.setattr(gc.mg, "collect_outputs",
                         lambda outdir: {"average_breathdata": None, "per_file": {}, "processed": {}})
    fake_scenarios = dict(gc.mg.SCENARIOS)
    fake_scenarios["fake_v2_only_scenario"] = object()   # would crash deep_update if ever reached
    monkeypatch.setattr(gc.mg, "SCENARIOS", fake_scenarios)

    results = gc.mg.run_all()

    assert len(calls) == len(gc.mg.LEGACY_SCENARIOS)
    assert set(results) == set(gc.mg.LEGACY_SCENARIOS)
