"""Registry-lite (``core/analysis/registry.py``): the legacy mechanics key order is
pinned against the real compute output, and ``resolve()`` is exercised against a
capability-dependent case (entropy present, EMG absent)."""
import json
import os

from _helpers import ROOT, requires_synth, synth_settings

from respmech.core.analysis.registry import LEGACY_MECHANICS_ORDER, resolve
from respmech.core.analysis.signals import Capabilities

# A snapshot of quantities.unit_for()'s answer for every real Data-sheet column name
# that appears anywhere in tests/golden/golden_reference.json (average_breathdata +
# each file's own breath table, across every scenario), taken before quantities.py's
# generic _pct/t_/_frac/peepi/tt_/_cv/_db rules and registry fallback were added. A
# mismatch here means one of those deliberately generic rules accidentally
# reclassified a column that is already shipping — see
# test_unit_for_is_unchanged_for_every_current_golden_column.
_GOLDEN_UNIT_SNAPSHOT = {
    # emg_burst neural timing and per-file burst QC (core.analysis.segments), golden
    # scenario emg_only_burst -- explicit registry units, none reachable through _RULES
    'ti_emg': 's',
    'te_emg': 's',
    'ttot_emg': 's',
    'ti_ttot_emg': '—',
    'bf_emg': 'min⁻¹',
    'emg_seg_n_bursts': '',
    'emg_seg_burst_frac': '—',
    'emg_seg_contrast': '—',
    # opt-in PEEPi columns (core.analysis.pressure), golden scenario flow_peepi_on
    'peepi_dyn': 'cmH₂O',
    'peepi_pgas_drop': 'cmH₂O',
    'peepi_corr': 'cmH₂O',
    'peepi_lag': 's',
    'int_oes_preflow': 'cmH₂O·s',
    'ptp_oes_preflow': 'cmH₂O·s·min⁻¹',
    'wob_in_thr': 'J·min⁻¹',
    'wob_in_total_thr': 'J·min⁻¹',
    'wobtotal_thr': 'J·min⁻¹',
    'int_oesinsp_peepi': 'cmH₂O·s',
    'ptp_oesinsp_peepi': 'cmH₂O·s·min⁻¹',
    'int_pdiinsp_peepi': 'cmH₂O·s',
    'ptp_pdiinsp_peepi': 'cmH₂O·s·min⁻¹',
    'bf': 'min⁻¹',
    'breath_no': '',
    # M-36: operating lung volumes (core.analysis.lungvol.attach), added to
    # golden_reference.json's typed_ic_fvc_same_file scenario once the SAME
    # own-typed IC reference M-35 already resolves is consumed (no new settings
    # entry needed; d_eelv itself is NOT in this scenario -- default
    # eelv_tracking='none'). ic_op/delta_ic/delta_eelv/tlc/vc resolve via the
    # registry (unit='L', none matches the generic vol_/vt rule); every
    # vol_*/vt_pct_ic/delta_ic_pct/*_pct_vc/*_pct_tlc column already resolves via
    # the pre-existing vol_/_pct generic rules.
    'delta_eelv': 'L',
    'delta_ic': 'L',
    'delta_ic_pct': '%',
    'eelv_pct_tlc': '%',
    'eelv_pct_vc': '%',
    'eilv_pct_tlc': '%',
    'eilv_pct_vc': '%',
    'ex_flow_midvol': 'L·s⁻¹',
    'exp_pgas_rise': 'cmH₂O',
    'file': '',
    'flow_midvolexp': 'L·s⁻¹',
    'flow_midvolinsp': 'L·s⁻¹',
    # M-35: cross-file IC references (core.analysis.references.attach), added to
    # golden_reference.json's typed_ic_fvc_same_file scenario when its own typed
    # breath resolves as its own reference (the own-typed fallback, no new settings
    # entry needed). ic_ref_n/ic_ref_source resolve via the registry (unit=""), not
    # _RULES; vol_ic_ref via the pre-existing vol_ prefix rule.
    'ic_op': 'L',
    'ic_ref_n': '',
    'ic_ref_source': '',
    'in_flow_midvol': 'L·s⁻¹',
    'insp_pdi_rise': 'cmH₂O',
    'irv_pct_tlc': '%',
    'irv_pct_vc': '%',
    'int_oesinsp': 'cmH₂O·s',
    'int_pdiinsp': 'cmH₂O·s',
    'int_pgasexp': 'cmH₂O·s',
    'integral_emg_col_2': 'a.u.·s',
    'integral_emg_col_3': 'a.u.·s',
    'integral_emg_col_4': 'a.u.·s',
    'integral_emg_exp_col_2': 'a.u.·s',
    'integral_emg_exp_col_3': 'a.u.·s',
    'integral_emg_exp_col_4': 'a.u.·s',
    'integral_emg_insp_col_2': 'a.u.·s',
    'integral_emg_insp_col_3': 'a.u.·s',
    'integral_emg_insp_col_4': 'a.u.·s',
    'integralemg_exp_max': 'a.u.·s',
    'integralemg_exp_mean': 'a.u.·s',
    'integralemg_insp_max': 'a.u.·s',
    'integralemg_insp_mean': 'a.u.·s',
    'integralemg_max': 'a.u.·s',
    'integralemg_mean': 'a.u.·s',
    'max_ex_flow': 'L·s⁻¹',
    'max_in_flow': 'L·s⁻¹',
    # M-42: placement of a tidal breath against the file's own resolved MEFV
    # envelope (core.analysis.mfvl.attach/tidal_mfvl_ext), added to
    # golden_reference.json's typed_ic_fvc_same_file scenario once the SAME
    # own-typed FVC reference (already resolved for lungvol's VC fallback) gives
    # this module a curve to place breaths against. efl_present/fvc_eofe_ok
    # resolve via the registry (unit=""); ve_cap/mvv_est via the registry
    # (unit='L·min⁻¹'); every *_pct* column already resolves via the pre-existing
    # generic suffix rule.
    'br_mvv_pct': '%',
    'efl_coverage_pct': '%',
    'efl_pct': '%',
    'efl_present': '',
    'fev1_source': '',
    'ex_flow_pct_mfvl_max': '%',
    'in_flow_pct_mfvl_max': '%',
    'max_ex_flow_pct_mfvl_peak': '%',
    'max_in_flow_pct_mfvl_peak': '%',
    'mvv_est': 'L·min⁻¹',
    'te_min_mfvl': 's',
    've_cap': 'L·min⁻¹',
    've_pct_cap': '%',
    've_pct_mvv': '%',
    've_reserve_pct': '%',
    'pdi_endexp': 'cmH₂O',
    'pdi_endinsp': 'cmH₂O',
    'pdi_maxinsp': 'cmH₂O',
    'pdi_minexp': 'cmH₂O',
    'pdi_tidal_swing': 'cmH₂O',
    'pgas_endexp': 'cmH₂O',
    'pgas_endinsp': 'cmH₂O',
    'pgas_maxexp': 'cmH₂O',
    'pgas_minexp': 'cmH₂O',
    'pgas_tidal_swing': 'cmH₂O',
    'poes_endexp': 'cmH₂O',
    'poes_endinsp': 'cmH₂O',
    'poes_maxexp': 'cmH₂O',
    'poes_midvolexp': 'cmH₂O',
    'poes_midvolinsp': 'cmH₂O',
    'poes_mininsp': 'cmH₂O',
    'poes_tidal_swing': 'cmH₂O',
    'ptp_oesinsp': 'cmH₂O·s·min⁻¹',
    'ptp_pdiinsp': 'cmH₂O·s·min⁻¹',
    'ptp_pgasexp': 'cmH₂O·s·min⁻¹',
    'rms_col_2': 'a.u.',
    'rms_col_3': 'a.u.',
    'rms_col_4': 'a.u.',
    'rms_exp_col_2': 'a.u.',
    'rms_exp_col_3': 'a.u.',
    'rms_exp_col_4': 'a.u.',
    'rms_exp_max': 'a.u.',
    'rms_exp_mean': 'a.u.',
    'rms_insp_col_2': 'a.u.',
    'rms_insp_col_3': 'a.u.',
    'rms_insp_col_4': 'a.u.',
    'rms_insp_max': 'a.u.',
    'rms_insp_mean': 'a.u.',
    'rms_max': 'a.u.',
    'rms_mean': 'a.u.',
    # EMG-only whole_file diagnostics (core.analysis.segments): where the peak EMG
    # activity fell in the recording and how it compares to the top-3 highest values
    # in the whole-file envelope (registry.py's _SEGMENT_EMG). Pre-existing golden
    # columns this snapshot had not caught up to yet.
    'rms_file_max_col_2': 'a.u.',
    'rms_file_max_col_3': 'a.u.',
    'rms_file_max_col_4': 'a.u.',
    'rms_file_top3_col_2': 'a.u.',
    'rms_file_top3_col_3': 'a.u.',
    'rms_file_top3_col_4': 'a.u.',
    't_rms_file_max_col_2': 's',
    't_rms_file_max_col_3': 's',
    't_rms_file_max_col_4': 's',
    'sample_entropy_col_10': '—',
    'sample_entropy_col_11': '—',
    'sample_entropy_col_12': '—',
    'sample_entropy_exp_col_10': '—',
    'sample_entropy_exp_col_11': '—',
    'sample_entropy_exp_col_12': '—',
    'sample_entropy_exp_max': '—',
    'sample_entropy_exp_mean': '—',
    'sample_entropy_insp_max': '—',
    'sample_entropy_insp_col_10': '—',
    'sample_entropy_insp_col_11': '—',
    'sample_entropy_insp_col_12': '—',
    'sample_entropy_insp_mean': '—',
    'sample_entropy_insp_min': '—',
    'sample_entropy_exp_min': '—',
    'sample_entropy_max': '—',
    'sample_entropy_mean': '—',
    'sample_entropy_min': '—',
    # EMG-only segmentation's own timing group (core.analysis.segments), replacing
    # the flow-derived insp/exp timing group a phase-less segment has none of.
    # Another pre-existing golden column set this snapshot had not caught up to yet.
    'seg_start_s': 's',
    'seg_end_s': 's',
    'seg_duration_s': 's',
    'te': 's',
    'ti': 's',
    'ti_ttot': '—',
    'tlc': 'L',
    'tlr_insp': '',
    'ttot': 's',
    'vc': 'L',
    've': 'L·min⁻¹',
    'vmr': '',
    'vol_eelv': 'L',
    'vol_eelv_abs': 'L',
    'vol_eilv': 'L',
    'vol_eilv_abs': 'L',
    'vol_endexp': 'L',
    'vol_endinsp': 'L',
    'vol_ic_ref': 'L',
    'vol_irv': 'L',
    'vt': 'L',
    'vt_pct_ic': '%',
    'wob_ex_total': 'J·min⁻¹',
    'wob_in_ela': 'J·min⁻¹',
    'wob_in_res': 'J·min⁻¹',
    'wob_in_total': 'J·min⁻¹',
    'wobtotal': 'J·min⁻¹',
}


def test_unit_for_is_unchanged_for_every_current_golden_column():
    """Snapshot test: every real column name in tests/golden/golden_reference.json
    (average_breathdata + each file's own per-file breath table, across every scenario)
    still resolves to exactly the unit it did before quantities.py's new generic rules
    and registry fallback were added. Also asserts the set of golden columns matches the
    pinned snapshot's keys, so a future scenario adding a genuinely new column name is
    caught here rather than silently skipped."""
    from respmech.core import quantities

    with open(os.path.join(ROOT, "tests", "golden", "golden_reference.json")) as fh:
        ref = json.load(fh)
    cols = set()
    for scenario in ref.values():
        cols.update(scenario.get("average_breathdata", {}).keys())
        for per_file in scenario.get("per_file", {}).values():
            if isinstance(per_file, dict):
                cols.update(per_file.keys())

    assert cols == set(_GOLDEN_UNIT_SNAPSHOT), (
        f"golden column set drifted from the pinned snapshot: "
        f"new={cols - set(_GOLDEN_UNIT_SNAPSHOT)}, gone={set(_GOLDEN_UNIT_SNAPSHOT) - cols}"
    )
    mismatches = {
        c: (quantities.unit_for(c), expected)
        for c, expected in _GOLDEN_UNIT_SNAPSHOT.items()
        if quantities.unit_for(c) != expected
    }
    assert not mismatches, f"unit_for changed for existing golden columns (got, expected): {mismatches}"


def test_capabilities_from_settings_against_real_settings_is_full_with_entropy():
    """The literal acceptance criterion: `Capabilities.from_settings(synth_settings())`
    gives mode 'full' and entropy=True, against a REAL, migrated `Settings` object —
    not the hand-rolled `SimpleNamespace` doubles `test_signal_sets.py` uses to unit-test
    the pure derivation logic on its own. Guards against a future `Settings`/`Channels`
    attribute rename breaking `Capabilities.from_settings` for real settings while those
    doubles keep passing regardless."""
    settings = synth_settings()
    caps = Capabilities.from_settings(settings)
    assert caps.mode == "full"
    assert caps.entropy is True

    settings.input.channels.emg = []
    settings.input.channels.entropy = [10, 11, 12]
    caps2 = Capabilities.from_settings(settings)
    assert caps2.emg is False
    assert caps2.entropy is True


@requires_synth()
def test_full_channel_key_list_equals_legacy_mechanics_order(tmp_path):
    """`LEGACY_MECHANICS_ORDER`'s names must be the exact, ordered key list
    `calculatemechanics` actually produces (compute.py:988-1009) — verified here
    against a real run, never assumed at import time (see registry.py's import
    budget)."""
    from respmech.core.pipeline import run_batch

    s = synth_settings(tmp_path)
    s.input.files = "synth_case_A.csv"
    result = run_batch(s)
    fr = result.ok_files["synth_case_A.csv"]
    breath = next(iter(fr.breaths.values()))
    assert list(breath["mechanics"]) == [c.name for c in LEGACY_MECHANICS_ORDER]


@requires_synth()
def test_entropy_without_emg_keeps_every_golden_column(tmp_path):
    """R8, at the compute/results level this ticket does not touch: with no EMG
    channel assigned but 3 entropy channels assigned (synth_settings' default),
    all 18 sample_entropy_* columns still appear (3 variants -- plain/insp/exp --
    x (3 per-channel columns + 3 max/min/mean columns) = 18), and no rms_* (EMG)
    column does."""
    from respmech.core.pipeline import run_batch

    s = synth_settings(tmp_path)
    s.input.files = "synth_case_A.csv"
    s.input.channels.emg = []
    result = run_batch(s)
    fr = result.ok_files["synth_case_A.csv"]
    cols = list(fr.breaths_table.columns)
    entropy_cols = [c for c in cols if c.startswith("sample_entropy")]
    assert len(entropy_cols) == 18
    assert not any(c.startswith("rms_") for c in cols)


def test_resolve_gives_entropy_module_without_emg_module():
    """The `flow_exclude_noemg` scenario named in the ticket's acceptance
    criteria: flow + entropy declared, no EMG. `resolve()` must include the
    entropy module and must not include the EMG module."""
    caps = Capabilities(
        flow=True, volume=True, poes=False, pgas=False, pdi=False, emg=False,
        entropy=True, declared=frozenset({"flow"}), mode="flow_only",
    )
    modules = resolve(caps)
    assert "entropy" in modules
    assert "emg" not in modules
    assert "timing" in modules            # the flow-only legacy timing group does resolve


def test_resolve_full_capabilities_includes_every_legacy_module():
    modules = resolve(Capabilities.FULL)
    assert {"timing", "pressures_poes", "pressures_pgas", "pressures_pdi", "vmr"} <= modules
    assert "emg" in modules


def test_seg_star_s_and_whole_file_rms_columns_resolve_to_units():
    """EMG-only segmentation's own columns (core.analysis.segments): seg_start_s/
    seg_end_s/seg_duration_s in seconds, whole_file's t_rms_file_max_col_/
    rms_file_max_col_/rms_file_top3_col_ in seconds/a.u. -- and resolve() names the
    segment_emg module for an emg_only capability set, distinct from the plain emg
    module a full-channel breath's rms_* columns resolve to."""
    from _helpers import assert_units

    assert_units({
        "seg_start_s": "s", "seg_end_s": "s", "seg_duration_s": "s",
        "t_rms_file_max_col_2": "s",
        "rms_file_max_col_2": "a.u.", "rms_file_top3_col_2": "a.u.",
    })
    caps = Capabilities(
        flow=False, volume=False, poes=False, pgas=False, pdi=False, emg=True,
        entropy=False, declared=frozenset({"emg"}), mode="emg_only",
    )
    modules = resolve(caps)
    assert "segment_emg" in modules
    assert "timing" not in modules             # no flow -> the legacy timing group never resolves


def test_capabilities_full_constant_has_entropy_true_and_resolves_entropy_module():
    assert Capabilities.FULL.entropy is True
    assert "entropy" in resolve(Capabilities.FULL)


def test_column_spec_needs_exactly_one_of_name_or_prefix():
    import pytest
    from respmech.core.analysis.registry import ColumnSpec

    with pytest.raises(ValueError):
        ColumnSpec(requires=frozenset(), module="x")            # neither
    with pytest.raises(ValueError):
        ColumnSpec(requires=frozenset(), module="x", name="a", prefix="b")  # both


def test_legacy_mechanics_order_has_42_entries_all_unit_none():
    assert len(LEGACY_MECHANICS_ORDER) == 42
    assert all(c.unit is None for c in LEGACY_MECHANICS_ORDER)
    assert all(c.name is not None and c.prefix is None for c in LEGACY_MECHANICS_ORDER)
