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


def test_typed_ic_fvc_same_file_vol_ic_matches_analytical_value(current):
    """This scenario's own acceptance criterion: 'typed_ic_fvc_same_file' breath #4 (typed
    'ic') reports vol_ic within 1e-9 of the analytical target, not merely whatever the
    code happens to compute this run — see generate_data.py's
    make_manoeuvre_file()/_manoeuvre_breath() docstring for why the generator's design
    makes an EXACT match (not just a close one) possible: VT_IC = 3.0 L is a bare
    Python float that survives the breath-separation sign-walk's one-sample-per-phase
    drop via linspace's own exact phase-boundary endpoints, and every other step
    between the CSV and this assertion (trim/zero/drift-correct, the JSON round-trip)
    is a documented no-op on that value. ic_eelv_pre is asserted too: it is the OTHER
    half of vol_ic's
    formula (`max(volume) - ic_eelv_pre`), and this scenario's design makes it exactly
    0.0 (the mean of all three preceding tidal breaths' own end-expiratory volume) —
    an unasserted vol_ic alone would not catch a regression that moved EITHER term by
    the same amount."""
    row = current["typed_ic_fvc_same_file"]["manoeuvres"]["synth_manoeuvre_A.csv"]["4"]
    assert row["kind"] == "ic"
    assert row["vol_ic"] == pytest.approx(3.0, abs=1e-9)
    assert row["ic_eelv_pre"] == pytest.approx(0.0, abs=1e-9)
    assert row["ic_eelv_pre_n"] == 3
    assert row["quality"] == []


def test_typed_ic_crossfile_reference_only_file_is_skipped_from_per_file(current):
    """M-40's own scope statement made mechanical: the reference-only
    ``synth_crossfile_ic.csv`` (every breath typed, no tidal breathing at all —
    ``core.pipeline.run_batch``'s M-30 detection) never gets a ``per_file`` /
    breathdata entry (``golden_newcore.run_scenario``'s own
    ``if fr.breaths_table is not None`` guard, already built by an earlier ticket in
    this programme) — only the tidal ``synth_crossfile_stage.csv`` does."""
    per_file = current["typed_ic_crossfile"]["per_file"]
    assert "synth_crossfile_stage.csv.breathdata.xlsx" in per_file
    assert "synth_crossfile_ic.csv.breathdata.xlsx" not in per_file


def test_typed_ic_crossfile_vol_ic_matches_analytical_value(current):
    """The reference file's own typed IC breath reports ``vol_ic`` == 3.0 exactly
    (same ``_MANOEUVRE_IC`` trapezoid recipe and the same reasoning as
    ``test_typed_ic_fvc_same_file_vol_ic_matches_analytical_value`` above — see that
    test's docstring and ``generate_data.py``'s ``_manoeuvre_breath`` docstring for
    why the match is exact rather than merely close).

    Unlike ``typed_ic_fvc_same_file``'s breath #4 (which has 3 preceding tidal
    breaths in the SAME file to average ``ic_eelv_pre`` over), this file has NO
    tidal breaths at all: ``ic_eelv_pre`` falls back to the breath's own immediate
    pre-inspiratory sample (``ic_eelv_pre_n == 1``, exactly 0.0 -- the trapezoid's own
    ``vol_base``), and the breath is simultaneously the min- and max-numbered breath
    among an EMPTY tidal set, so ``quality`` carries ``BOUNDARY`` (documented,
    expected behaviour for a reference-only file's typed breath -- see
    ``core.pipeline.run_batch``'s own comment on this: 'flags every manoeuvre
    BOUNDARY -- there is no tidal context to judge low effort or eelv stability
    against, which is the honest answer, not a bug'), NOT the empty list
    `typed_ic_fvc_same_file`'s breath #4 gets."""
    row = current["typed_ic_crossfile"]["manoeuvres"]["synth_crossfile_ic.csv"]["1"]
    assert row["kind"] == "ic"
    assert row["vol_ic"] == pytest.approx(3.0, abs=1e-9)
    assert row["ic_eelv_pre"] == pytest.approx(0.0, abs=1e-9)
    assert row["ic_eelv_pre_n"] == 1
    assert row["quality"] == ["BOUNDARY"]


def test_typed_ic_crossfile_stage_file_resolves_the_crossfile_reference(current):
    """The tidal ``synth_crossfile_stage.csv`` resolves its explicit
    ``processing.references`` entry to the reference file's own ``vol_ic`` (3.0):
    every breath's ``vol_ic_ref``/``ic_ref_n``/``ic_ref_source`` (``core.analysis.
    references.attach``, M-35) names the reference file, one accepted attempt.

    ``core.analysis.lungvol.attach`` (M-36) then derives the operating-lung-volume
    family from that constant reference. ``processing.lung_volume.ic.eelv_tracking``
    is left at its default ``'none'`` here -- the only valid choice for a CROSS-file
    reference (``'within_file'`` needs a SAME-file IC and NaNs otherwise, see
    ``lungvol.py``'s own module docstring and this scenario's own TOML comment) -- so
    ``ic_op`` is the SAME resolved reference IC for every breath in the file, which
    in turn makes the VC-anchored ``vol_eelv = vc - ic_op`` an EXACT CONSTANT across
    every breath too (it does not depend on ``vol_endexp`` at all under 'none'
    tracking -- only 'within_file' tracking's ``d_eelv`` does). This is why this
    test pins ``vol_eelv``'s exact, constant value analytically (``vc(5.0) -
    ic_op(3.0) == 2.0``) rather than asserting it 'moves in the same direction as
    vol_endexp' -- a phrase that describes 'within_file' tracking's own per-breath
    drift term (already pinned by ``tests/unit/test_lungvol.py``'s analytical test
    for THAT mode) and cannot be demonstrated by a scenario whose whole point is a
    cross-file reference. ``vol_eelv_abs``/``tlc``/``delta_ic`` stay NaN (no TLC, no
    baseline configured) -- exercising that side of the family too."""
    pf = current["typed_ic_crossfile"]["per_file"]["synth_crossfile_stage.csv.breathdata.xlsx"]
    n = len(pf["vol_ic_ref"])
    assert n > 0
    for i in range(n):
        assert pf["vol_ic_ref"][i] == pytest.approx(3.0, abs=1e-9)
        assert pf["ic_ref_n"][i] == pytest.approx(1.0, abs=1e-9)
        assert pf["ic_ref_source"][i] == "synth_crossfile_ic.csv"
        assert pf["ic_op"][i] == pytest.approx(3.0, abs=1e-9)
        assert pf["vc"][i] == pytest.approx(5.0, abs=1e-9)
        assert pf["vol_eelv"][i] == pytest.approx(2.0, abs=1e-9)
        assert pf["tlc"][i] == "NaN"
        assert pf["vol_eelv_abs"][i] == "NaN"
        assert pf["delta_ic"][i] == "NaN"
