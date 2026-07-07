#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production golden — non-EMG scenarios (Zeros, Trimming) reproduced by the NEW
canonical core (window-baseline PTP included). EMG-carrying scenarios (Resampling,
H5, H6) are locked by test_production_emg_golden.py.

Raw data is gitignored (public repo); this SKIPS when it is absent.
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
SCENARIOS = ["zeros_debugging", "trimming_debugging"]


def _has_data():
    for name in ("Zeros debugging/input", "Trimming debugging"):
        d = os.path.join(R.bpg.PROD, name)
        if os.path.isdir(d) and any(f.endswith(".txt") for f in os.listdir(d)):
            return True
    return False


pytestmark = pytest.mark.skipif(
    not os.path.exists(GOLDEN) or not _has_data(),
    reason="production data not present locally (gitignored)")


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN) as f:
        return json.load(f)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_reproduces_golden(scenario, golden):
    res = R.run_scenario(scenario)
    gfiles = golden[scenario]["files"]
    # every golden 'ok' file must reproduce; every golden 'error' file must still error
    for fname, gentry in gfiles.items():
        if gentry.get("status") == "error":
            assert fname in res.failed_files, f"{fname} expected to error but did not"
            continue
        assert fname in res.ok_files, f"{fname} expected ok but errored/absent"
        cur = res.ok_files[fname].breaths_table.reset_index(drop=True)
        gcols = gentry["golden"]
        assert set(gcols) == set(map(str, cur.columns)), f"{fname}: columns changed"
        for col in gcols:
            a = pd.to_numeric(pd.Series(gcols[col]), errors="coerce").to_numpy(float)
            b = pd.to_numeric(cur[col], errors="coerce").to_numpy(float)
            assert len(a) == len(b), f"{fname}/{col}: length changed"
            for i, (x, y) in enumerate(zip(a, b)):
                if math.isnan(x) and math.isnan(y):
                    continue
                assert math.isclose(x, y, rel_tol=RTOL, abs_tol=ATOL), \
                    f"{fname}/{col}[{i}]: {x} != {y}"
