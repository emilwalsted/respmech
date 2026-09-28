#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production golden — non-EMG scenarios (Zeros, Trimming) reproduced by the NEW
canonical core (window-baseline PTP included). EMG-carrying scenarios (Resampling,
H5, H6) are locked by test_production_emg_golden.py. The v2-only typed_ic_h5 scenario
(M-41 — typed IC + cross-file operating lung volumes, no legacy-dict shape at all) has
its own, independently-skipped test at the bottom of this file.

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


# --------------------------------------------------------------------------------- #
# typed_ic_h5 (M-41) — v2-only, no legacy-dict shape; skipped independently of the
# module-level SCENARIOS above (whose own skip is what fires in this sandbox, since
# no production data of any kind is present here at all; this scenario gets its own
# reason so a machine that HAS the Zeros/Trimming data but not yet RIU_H5_typed_ic.toml
# reports precisely what is still missing, instead of the generic module reason).
# --------------------------------------------------------------------------------- #

def _typed_ic_h5_skip_reason():
    if not os.path.exists(GOLDEN):
        return "typed_ic_h5: production_golden.json missing"
    d = os.path.join(R.bpg.PROD, "EMG processing fix test")
    if not (os.path.isdir(d) and any(f.endswith(".txt") for f in os.listdir(d))):
        return ("typed_ic_h5: 'EMG processing fix test' production recordings not "
                "present locally (gitignored)")
    if not os.path.exists(os.path.join(d, "RIU_H5_IC.txt")):
        return "typed_ic_h5: RIU_H5_IC.txt (the typed-IC recording) not present locally"
    if not os.path.exists(R.TYPED_IC_TOML["typed_ic_h5"]):
        return ("typed_ic_h5: RIU_H5_typed_ic.toml not authored locally yet — see "
                "tests/golden/README.md's production-scenario table")
    with open(GOLDEN) as f:
        if "typed_ic_h5" not in json.load(f):
            return ("typed_ic_h5: no frozen golden entry yet — run "
                    "'python tests/golden/regen_production_emg_golden.py typed_ic_h5 --write' "
                    "locally first")
    return None


_TYPED_IC_H5_SKIP_REASON = _typed_ic_h5_skip_reason()


@pytest.mark.skipif(_TYPED_IC_H5_SKIP_REASON is not None,
                    reason=_TYPED_IC_H5_SKIP_REASON or "")
def test_typed_ic_h5_scenario_reproduces_golden(golden):
    res = R.run_scenario("typed_ic_h5")
    gfiles = golden["typed_ic_h5"]["files"]
    # golden-driven, like test_scenario_reproduces_golden above: every golden 'ok' file
    # must still reproduce and every golden 'error' file must still error — walking
    # res.ok_files instead would let a file that regressed from ok to erroring quietly
    # vanish from the comparison instead of failing it.
    for fname, gentry in gfiles.items():
        if gentry.get("status") == "error":
            assert fname in res.failed_files, f"{fname} expected to error but did not"
            continue
        assert fname in res.ok_files, f"{fname} expected ok but errored/absent"
        fr = res.ok_files[fname]
        if fr.breaths_table is None:
            continue  # reference-only file (RIU_H5_IC.txt itself) — nothing to compare
        cur = fr.breaths_table.reset_index(drop=True)
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
