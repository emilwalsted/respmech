#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production golden regression test.

Unlike the synthetic golden, the production golden is built from Emil's real
validation recordings, which are **gitignored** (public repo). So this test:

  * SKIPS entirely when the raw data is not present locally, and
  * otherwise re-runs the current code over the fast (non-EMG) scenarios and
    asserts the captured numbers still match the committed production_golden.json.

The EMG scenarios are slow (ECG removal + spectral noise reduction + forced
plots) and are not re-run here by default; verify them with
`python build_production_golden.py emg_h5_ecg_noise_outlier` when needed.

Run with (pinned env — see requirements-golden.txt):
    pytest tests/golden/test_production_golden.py -v
"""
import os
import json
import math

import pytest

import build_production_golden as bpg

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "production_golden.json")

# Only the fast, deterministic, non-EMG scenarios are re-run in the test.
FAST_SCENARIOS = {"zeros_debugging", "trimming_debugging"}

RTOL = 1e-9
ATOL = 1e-12


def _has_production_data():
    for sc in bpg.SCENARIOS:
        if sc["name"] in FAST_SCENARIOS:
            d = os.path.join(bpg.PROD, sc["input_dir"])
            if os.path.isdir(d) and any(f.endswith(".txt") for f in os.listdir(d)):
                return True
    return False


pytestmark = pytest.mark.skipif(
    not os.path.exists(GOLDEN) or not _has_production_data(),
    reason="production raw data not present locally (gitignored); nothing to verify",
)


@pytest.fixture(scope="module")
def committed():
    with open(GOLDEN) as f:
        return json.load(f)


def _close(a, b):
    if isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        return a == b
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)


@pytest.mark.parametrize("scenario", sorted(FAST_SCENARIOS))
def test_fast_scenario_reproduces_golden(scenario, committed):
    assert scenario in committed, f"{scenario} missing from committed golden"
    sc = next(s for s in bpg.SCENARIOS if s["name"] == scenario)
    fresh = bpg.run_scenario(sc, manifest={})

    for fname, exp in committed[scenario]["files"].items():
        got = fresh["files"].get(fname)
        assert got is not None, f"{scenario}/{fname} not produced now"
        assert got["status"] == exp["status"], \
            f"{scenario}/{fname} status changed: {got['status']} != {exp['status']}"
        if exp["status"] != "ok":
            continue
        eg, gg = exp["golden"], got["golden"]
        assert set(eg) == set(gg), f"{scenario}/{fname} columns changed"
        for col in eg:
            ev, gv = eg[col], gg[col]
            assert len(ev) == len(gv), f"{scenario}/{fname}/{col} length changed"
            for i, (a, b) in enumerate(zip(ev, gv)):
                assert _close(a, b), f"{scenario}/{fname}/{col}[{i}]: {a} != {b}"
