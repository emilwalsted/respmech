"""Registry-lite (``core/analysis/registry.py``): the legacy mechanics key order is
pinned against the real compute output, and ``resolve()`` is exercised against a
capability-dependent case (entropy present, EMG absent)."""
from _helpers import requires_synth, synth_settings

from respmech.core.analysis.registry import LEGACY_MECHANICS_ORDER, resolve
from respmech.core.analysis.signals import Capabilities


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
