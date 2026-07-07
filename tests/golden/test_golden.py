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
