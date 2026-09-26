"""Registry-lite: output columns declared once, resolved against :class:`Capabilities`.

Qt-free and numeric-stack-free, same import budget as ``signals.py`` (see that
module's docstring) — verified by ``tests/unit/test_startup_imports.py``. This
module does not import ``signals``: nothing here needs a ``Capabilities``
instance at import time, only at call time (``resolve()`` takes one as an
argument), so there is no cycle to worry about either way.

This is deliberately "registry-lite": a flat tuple of small, immutable
``ColumnSpec`` records, no metaclasses, no decorators that mutate global state
at import time, no name-based lookup magic. Nothing in compute/pipeline
consults ``REGISTRY``/``resolve()`` yet — a later ticket (the compute-guards
one) wires ``LEGACY_MECHANICS_ORDER`` into ``calculatemechanics`` itself. This
ticket only establishes the shape and pins the one existing behaviour that
must never silently drift: the legacy mechanics key order.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnSpec:
    """One output-table column, or a whole column *family* via ``prefix``.

    Exactly one of ``name``/``prefix`` is set (``prefix`` for a skabelonnavn
    like ``"sample_entropy_"``, matching every column that starts with it).
    ``requires`` names the :class:`~respmech.core.analysis.signals.Capabilities`
    boolean fields (``"flow"``, ``"poes"``, ..., ``"entropy"``) that must all be
    true for the column to be computed at all. ``module`` is an informational
    label naming the pipeline stage that would produce it — nothing dispatches
    on it yet, that arrives with the modules themselves in later tickets.
    ``unit``/``level``/``sheet``/``trigger`` describe how the column is
    presented once it exists; ``unit=None`` means "not classified here" — the
    42 legacy mechanics columns keep getting their units from
    ``core/quantities.py`` exactly as today, unchanged by this ticket.
    """

    requires: frozenset
    module: str
    unit: str | None = None
    level: str = "breath"
    sheet: str = "Data"
    trigger: str = "always"
    name: str | None = None
    prefix: str | None = None

    def __post_init__(self):
        if (self.name is None) == (self.prefix is None):
            raise ValueError("ColumnSpec needs exactly one of name= or prefix=")

    @property
    def key(self) -> str:
        return self.name if self.name is not None else self.prefix


# The four capability combinations the legacy mechanics block's own key groups
# need, per the "column -> required capability -> module" mapping (each row
# beyond `flow` alone, which is implicit: this whole block only exists once a
# recording has been segmented into breaths, and that itself needs flow).
_TIMING = frozenset({"flow"})
_PRESSURES_POES = frozenset({"flow", "poes"})
_PRESSURES_PGAS = frozenset({"flow", "pgas"})
_PRESSURES_PDI = frozenset({"flow", "pdi"})
_VMR = frozenset({"flow", "poes", "pgas"})

# The 42 keys of `calculatemechanics`'s OrderedDict (compute.py:988-1009), in
# order, each paired with the capabilities it needs and its module name.
# Deliberately a plain, hand-written tuple: the order is pinned BY the test
# (tests/unit/test_analysis_registry.py), never read back out of compute at
# import time — see this package's import budget above.
_LEGACY_MECHANICS = (
    ("poes_maxexp", _PRESSURES_POES, "pressures_poes"),
    ("poes_mininsp", _PRESSURES_POES, "pressures_poes"),
    ("poes_endinsp", _PRESSURES_POES, "pressures_poes"),
    ("poes_endexp", _PRESSURES_POES, "pressures_poes"),
    ("poes_midvolexp", _PRESSURES_POES, "pressures_poes"),
    ("poes_midvolinsp", _PRESSURES_POES, "pressures_poes"),
    ("int_oesinsp", _PRESSURES_POES, "pressures_poes"),
    ("ptp_oesinsp", _PRESSURES_POES, "pressures_poes"),
    ("poes_tidal_swing", _PRESSURES_POES, "pressures_poes"),
    ("pgas_endinsp", _PRESSURES_PGAS, "pressures_pgas"),
    ("pgas_endexp", _PRESSURES_PGAS, "pressures_pgas"),
    ("pgas_maxexp", _PRESSURES_PGAS, "pressures_pgas"),
    ("pgas_minexp", _PRESSURES_PGAS, "pressures_pgas"),
    ("exp_pgas_rise", _PRESSURES_PGAS, "pressures_pgas"),
    ("int_pgasexp", _PRESSURES_PGAS, "pressures_pgas"),
    ("ptp_pgasexp", _PRESSURES_PGAS, "pressures_pgas"),
    ("pgas_tidal_swing", _PRESSURES_PGAS, "pressures_pgas"),
    ("int_pdiinsp", _PRESSURES_PDI, "pressures_pdi"),
    ("ptp_pdiinsp", _PRESSURES_PDI, "pressures_pdi"),
    ("pdi_minexp", _PRESSURES_PDI, "pressures_pdi"),
    ("pdi_maxinsp", _PRESSURES_PDI, "pressures_pdi"),
    ("pdi_endinsp", _PRESSURES_PDI, "pressures_pdi"),
    ("pdi_endexp", _PRESSURES_PDI, "pressures_pdi"),
    ("insp_pdi_rise", _PRESSURES_PDI, "pressures_pdi"),
    ("pdi_tidal_swing", _PRESSURES_PDI, "pressures_pdi"),
    ("flow_midvolexp", _TIMING, "timing"),
    ("flow_midvolinsp", _TIMING, "timing"),
    ("vol_endinsp", _TIMING, "timing"),
    ("vol_endexp", _TIMING, "timing"),
    ("max_in_flow", _TIMING, "timing"),
    ("max_ex_flow", _TIMING, "timing"),
    ("in_flow_midvol", _TIMING, "timing"),
    ("ex_flow_midvol", _TIMING, "timing"),
    ("ti", _TIMING, "timing"),
    ("te", _TIMING, "timing"),
    ("ttot", _TIMING, "timing"),
    ("ti_ttot", _TIMING, "timing"),
    ("vt", _TIMING, "timing"),
    ("bf", _TIMING, "timing"),
    ("ve", _TIMING, "timing"),
    ("vmr", _VMR, "vmr"),
    # tlr_insp's formula (compute.py, abs((poes_midvolexp - poes_midvolinsp) /
    # (flow_midvolexp - flow_midvolinsp))) reads poes and the flow-derived midvol
    # indices: flow is the block's shared baseline, poes is the one extra it needs.
    ("tlr_insp", _PRESSURES_POES, "pressures_poes"),
)

#: Literal, pinned order of the legacy `calculatemechanics` OrderedDict's 42
#: keys, each wrapped in a :class:`ColumnSpec` with ``unit=None`` (units for
#: this block still come from ``core/quantities.py``, unchanged).
LEGACY_MECHANICS_ORDER = tuple(
    ColumnSpec(name=key, requires=requires, module=module, unit=None)
    for key, requires, module in _LEGACY_MECHANICS
)

# Two rows outside the legacy mechanics block, added now (rather than left for
# a later ticket) because this ticket's own acceptance test needs a
# capability-dependent difference to resolve against: with entropy declared
# and no EMG channel assigned, `resolve()` must name the entropy module and
# must not name the EMG one. Later tickets add the remaining families
# (segments, manoeuvres, IC/FVC/PEEPi, ...); this is not meant to be exhaustive.
_ENTROPY = ColumnSpec(
    prefix="sample_entropy_", requires=frozenset({"entropy"}), module="entropy",
    unit=None, sheet="Extra",
)
_EMG = ColumnSpec(
    prefix="rms_", requires=frozenset({"emg"}), module="emg", unit=None,
)

#: Every column/family this skeleton knows about. Later tickets append to this,
#: never remove from or reorder ``LEGACY_MECHANICS_ORDER`` within it.
REGISTRY = LEGACY_MECHANICS_ORDER + (_ENTROPY, _EMG)

# The Capabilities boolean fields resolve() is willing to read. Kept as an
# explicit tuple (rather than e.g. dataclasses.fields(caps)) so a caller could
# hand in any object exposing these names as booleans, not only a real
# Capabilities instance.
_CAPABILITY_NAMES = ("flow", "volume", "poes", "pgas", "pdi", "emg", "entropy")


def resolve(caps, settings=None) -> frozenset:
    """The module names that would run for ``caps``.

    Informational only for now — see the module docstring, nothing dispatches
    on this yet. ``settings`` is accepted (and unused by every row in this
    skeleton) so that a later ticket's family-level rows, which decide from
    settings-wide state (an IC reference existing anywhere in the matched
    files, say) rather than from ``caps`` alone, can be added without changing
    this function's signature or its callers.
    """
    available = frozenset(name for name in _CAPABILITY_NAMES if getattr(caps, name))
    return frozenset(spec.module for spec in REGISTRY if spec.requires <= available)
