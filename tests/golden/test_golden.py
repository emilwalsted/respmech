#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Golden / characterisation test for RespMech.

Re-runs the current respmech.py over the committed synthetic input and asserts
that every numeric output matches golden_reference.json within a documented
tolerance. This is the safety net for the production refactor: the refactored
calculation core must reproduce these numbers exactly (within tolerance) before
any behavioural change is accepted.

Tolerance
---------
Values are compared with a relative tolerance of 1e-9 and an absolute tolerance
of 1e-12 (see GOLDEN_RTOL / GOLDEN_ATOL). These are tight: the refactor is a
restructuring, not a numerical change, so outputs should be bit-for-bit identical
apart from floating-point reassociation. If a legitimate, explained numerical
change is introduced, regenerate the reference with:

    python make_golden.py --write

and justify the diff in the commit message / PR.

Run with:
    pytest tests/golden/test_golden.py -v
(using an interpreter whose scipy still provides integrate.cumtrapz and accepts a
positional x in integrate.simpson — see tests/golden/README.md).
"""
import os
import json
import math

import pytest

import make_golden as mg

GOLDEN_RTOL = 1e-9
GOLDEN_ATOL = 1e-12

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_JSON = os.path.join(HERE, "golden_reference.json")


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN_JSON) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def current():
    return mg.run_all()


def _compare_value(path, a, b):
    # String / NaN-sentinel values must match exactly.
    if isinstance(a, str) or isinstance(b, str):
        assert a == b, f"{path}: {a!r} != {b!r}"
        return
    if a is None or b is None:
        assert a == b, f"{path}: {a!r} != {b!r}"
        return
    if isinstance(a, bool) or isinstance(b, bool):
        assert a == b, f"{path}: {a!r} != {b!r}"
        return
    # Numeric
    if math.isnan(a) and math.isnan(b):
        return
    assert math.isclose(a, b, rel_tol=GOLDEN_RTOL, abs_tol=GOLDEN_ATOL), \
        f"{path}: {a!r} != {b!r} (rtol={GOLDEN_RTOL}, atol={GOLDEN_ATOL})"


def _compare(path, a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict), f"{path}: type mismatch dict vs {type(b)}"
        assert set(a) == set(b), f"{path}: key mismatch {set(a) ^ set(b)}"
        for k in a:
            _compare(f"{path}.{k}", a[k], b[k])
    elif isinstance(a, list):
        assert isinstance(b, list), f"{path}: type mismatch list vs {type(b)}"
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _compare(f"{path}[{i}]", x, y)
    else:
        _compare_value(path, a, b)


def test_scenarios_present(golden, current):
    assert set(golden) == set(current), \
        f"scenario set differs: {set(golden) ^ set(current)}"


@pytest.mark.parametrize("scenario", sorted(mg.SCENARIOS))
def test_scenario_matches_golden(scenario, golden, current):
    assert scenario in golden, f"no golden reference for scenario {scenario}"
    _compare(scenario, golden[scenario], current[scenario])
