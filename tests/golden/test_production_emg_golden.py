#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production EMG golden — the NEW canonical core must reproduce the regenerated
H5/H6 EMG golden (shared-profile noise reduction + ECG removal + window-baseline PTP).

Raw data is gitignored (public repo); this SKIPS when it is absent. It is the
regression lock for the canonical EMG pipeline when the data is present.
"""
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

import regen_production_emg_golden as R

GOLDEN = R.GOLDEN
RTOL, ATOL = 1e-9, 1e-12


def _has_data():
    d = os.path.join(R.bpg.PROD, "EMG processing fix test")
    return os.path.isdir(d) and any(f.endswith(".txt") for f in os.listdir(d))


pytestmark = pytest.mark.skipif(
    not os.path.exists(GOLDEN) or not _has_data(),
    reason="production EMG data not present locally (gitignored)")


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN) as f:
        return json.load(f)


@pytest.mark.parametrize("scenario", R.EMG_SCENARIOS)
def test_emg_scenario_reproduces_golden(scenario, golden):
    res = R.run_scenario(scenario)
    gfiles = golden[scenario]["files"]
    for fname, fr in res.ok_files.items():
        assert fname in gfiles and gfiles[fname]["status"] == "ok", f"{fname} missing/erroring in golden"
        gcols = gfiles[fname]["golden"]
        cur = fr.breaths_table.reset_index(drop=True)
        assert set(gcols) == set(map(str, cur.columns)), f"{fname}: column set changed"
        for col in gcols:
            a = pd.to_numeric(pd.Series(gcols[col]), errors="coerce").to_numpy(float)
            b = pd.to_numeric(cur[col], errors="coerce").to_numpy(float)
            assert len(a) == len(b), f"{fname}/{col}: length changed"
            for i, (x, y) in enumerate(zip(a, b)):
                if math.isnan(x) and math.isnan(y):
                    continue
                assert math.isclose(x, y, rel_tol=RTOL, abs_tol=ATOL), \
                    f"{fname}/{col}[{i}]: {x} != {y}"
