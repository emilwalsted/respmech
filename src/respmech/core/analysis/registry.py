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

# A plain string constant, no heavier than anything already declared in this
# module — quantities.py itself only ever imports THIS module lazily (inside a
# function body, to avoid the reverse cycle), so importing its constant here at
# module level is safe.
from respmech.core.quantities import DIMLESS


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

# EMG-only segmentation (core.analysis.segments, whole_file/separators): every column
# here requires ONLY `emg` (never `flow` -- these columns exist precisely because flow
# is absent). `seg_start_s`/`seg_end_s`/`seg_duration_s` (set directly on a phase-less
# segment's own `mechanics` dict, replacing the timing group `LEGACY_MECHANICS_ORDER`
# computes from insp/exp) are the only ones `core.quantities`'s generic `_RULES` cannot
# already classify on its own (no rule matches a bare `_s` suffix that is not `t_`-
# prefixed), so they are the only entries here whose `unit=` is actually load-bearing;
# `t_rms_file_max_col_`/`rms_file_max_col_`/`rms_file_top3_col_` (whole_file-only:
# core.analysis.segments._attach_whole_file_rms_diagnostics) are already resolved by
# _RULES' generic `t_`/`rms` prefixes before this registry is ever consulted -- their
# `unit=` here is documentation, not the resolving path -- but are still named as their
# own family (module="segment_emg") so `resolve()` can report this module distinctly
# from the plain per-breath `emg` module above.
_SEGMENT_EMG = (
    ColumnSpec(name="seg_start_s", requires=frozenset({"emg"}), module="segment_emg", unit="s"),
    ColumnSpec(name="seg_end_s", requires=frozenset({"emg"}), module="segment_emg", unit="s"),
    ColumnSpec(name="seg_duration_s", requires=frozenset({"emg"}), module="segment_emg", unit="s"),
    ColumnSpec(prefix="t_rms_file_max_col_", requires=frozenset({"emg"}), module="segment_emg", unit="s"),
    ColumnSpec(prefix="rms_file_max_col_", requires=frozenset({"emg"}), module="segment_emg", unit="a.u."),
    ColumnSpec(prefix="rms_file_top3_col_", requires=frozenset({"emg"}), module="segment_emg", unit="a.u."),
)

# Manoeuvre extraction (M-29, core.analysis.manoeuvres): the Manoeuvres-sheet columns
# `core.quantities._RULES`' generic prefix/suffix conventions cannot already classify
# on their own. `vol_ic` (`vol_` prefix -> L), `ic_peak_in_flow` (contains "flow" -> a
# rate), and every `poes_ic_*`/`pdi_ic_*`/`pgas_ic_*`/`poes_max_ref`/`pdi_max_ref`
# (poes/pgas/pdi prefix -> cmH2O) and `rms_max_ref` (`rms` prefix -> a.u.) are ALREADY
# resolved by _RULES before this registry is ever consulted -- listed here anyway
# (unit= is then documentation, not the resolving path, same precedent as
# _SEGMENT_EMG's t_rms_file_max_col_/rms_file_max_col_/rms_file_top3_col_ rows) so a
# reader of REGISTRY sees the whole Manoeuvres column family in one place.
_MANOEUVRES = (
    ColumnSpec(name="ic_eelv_pre", requires=_TIMING, module="manoeuvres", unit="L"),
    ColumnSpec(name="ic_eelv_pre_sd", requires=_TIMING, module="manoeuvres", unit="L"),
    ColumnSpec(name="ic_eelv_pre_n", requires=_TIMING, module="manoeuvres", unit=""),
    ColumnSpec(name="ic_ti", requires=_TIMING, module="manoeuvres", unit="s"),
    ColumnSpec(name="ic_plateau_s", requires=_TIMING, module="manoeuvres", unit="s"),
    ColumnSpec(name="quality", requires=_TIMING, module="manoeuvres", unit=""),
    ColumnSpec(name="vol_ic", requires=_TIMING, module="manoeuvres", unit="L"),
    ColumnSpec(name="ic_peak_in_flow", requires=_TIMING, module="manoeuvres", unit="L·s⁻¹"),
    ColumnSpec(name="poes_ic_min", requires=_PRESSURES_POES, module="manoeuvres", unit="cmH₂O"),
    ColumnSpec(name="poes_ic_eelv", requires=_PRESSURES_POES, module="manoeuvres", unit="cmH₂O"),
    ColumnSpec(name="poes_ic_swing", requires=_PRESSURES_POES, module="manoeuvres", unit="cmH₂O"),
    ColumnSpec(name="poes_ic_peakvol", requires=_PRESSURES_POES, module="manoeuvres", unit="cmH₂O"),
    ColumnSpec(name="pdi_ic_max", requires=_PRESSURES_PDI, module="manoeuvres", unit="cmH₂O"),
    ColumnSpec(name="pdi_ic_swing", requires=_PRESSURES_PDI, module="manoeuvres", unit="cmH₂O"),
    ColumnSpec(name="pgas_ic_peakvol", requires=_PRESSURES_PGAS, module="manoeuvres", unit="cmH₂O"),
    ColumnSpec(name="poes_max_ref", requires=_PRESSURES_POES, module="manoeuvres", unit="cmH₂O"),
    ColumnSpec(name="pdi_max_ref", requires=_PRESSURES_PDI, module="manoeuvres", unit="cmH₂O"),
    ColumnSpec(name="rms_max_ref", requires=frozenset({"flow", "emg"}), module="manoeuvres", unit="a.u."),
)

# Cross-file IC references (core.analysis.references.attach, wired into
# core.pipeline.run_batch's post-loop pass): `vol_ic_ref` already resolves via
# `core.quantities._RULES`' generic `vol_` prefix (-> L) before this registry is ever
# consulted, listed here anyway for documentation completeness, same precedent as
# _MANOEUVRES above. `ic_ref_n`/`ic_ref_source` are neither a volume nor any other
# _RULES-matched shape (a bare count and a filename), so their `unit=""` here IS the
# resolving path -- exactly the same "registry explicitly blank" role
# `ic_eelv_pre_n` already plays in _MANOEUVRES, not merely documentation.
_REFERENCE_MANOEUVRES = (
    ColumnSpec(name="vol_ic_ref", requires=_TIMING, module="references", unit="L"),
    ColumnSpec(name="ic_ref_n", requires=_TIMING, module="references", unit=""),
    ColumnSpec(name="ic_ref_source", requires=_TIMING, module="references", unit=""),
)

# Operating lung volumes (M-36, core.analysis.lungvol, wired into core.pipeline.
# run_batch right after references.attach): `d_eelv`, `delta_ic`, `delta_eelv`, `tlc`,
# `vc`, `ic_op` are the six new columns `core.quantities._RULES`' generic `vol_`/`vt`
# convention does NOT already classify (none starts with `vol_` and none is `vt`) --
# their `unit="L"` here IS the resolving path, not documentation. Every other new
# column this ticket adds (`vol_eelv`, `vol_eilv`, `vol_eelv_abs`, `vol_eilv_abs`,
# `vol_irv` via the `vol_` prefix; `vt_pct_ic`, `delta_ic_pct`, `eelv_pct_vc`,
# `eilv_pct_vc`, `irv_pct_vc`, `eelv_pct_tlc`, `eilv_pct_tlc`, `irv_pct_tlc` via the
# generic `_pct`/`_pct_` rule) is already resolved by `_RULES` before this registry is
# ever consulted -- listed here anyway for documentation completeness, same precedent
# as `_MANOEUVRES`/`_REFERENCE_MANOEUVRES` above.
_LUNG_VOLUMES = (
    ColumnSpec(name="d_eelv", requires=_TIMING, module="lungvol", unit="L"),
    ColumnSpec(name="ic_op", requires=_TIMING, module="lungvol", unit="L"),
    ColumnSpec(name="delta_ic", requires=_TIMING, module="lungvol", unit="L"),
    ColumnSpec(name="delta_eelv", requires=_TIMING, module="lungvol", unit="L"),
    ColumnSpec(name="tlc", requires=_TIMING, module="lungvol", unit="L"),
    ColumnSpec(name="vc", requires=_TIMING, module="lungvol", unit="L"),
    ColumnSpec(name="vol_irv", requires=_TIMING, module="lungvol", unit="L"),
    ColumnSpec(name="vol_eelv", requires=_TIMING, module="lungvol", unit="L"),
    ColumnSpec(name="vol_eilv", requires=_TIMING, module="lungvol", unit="L"),
    ColumnSpec(name="vol_eelv_abs", requires=_TIMING, module="lungvol", unit="L"),
    ColumnSpec(name="vol_eilv_abs", requires=_TIMING, module="lungvol", unit="L"),
    ColumnSpec(name="vt_pct_ic", requires=_TIMING, module="lungvol", unit="%"),
    ColumnSpec(name="delta_ic_pct", requires=_TIMING, module="lungvol", unit="%"),
    ColumnSpec(name="eelv_pct_vc", requires=_TIMING, module="lungvol", unit="%"),
    ColumnSpec(name="eilv_pct_vc", requires=_TIMING, module="lungvol", unit="%"),
    ColumnSpec(name="irv_pct_vc", requires=_TIMING, module="lungvol", unit="%"),
    ColumnSpec(name="eelv_pct_tlc", requires=_TIMING, module="lungvol", unit="%"),
    ColumnSpec(name="eilv_pct_tlc", requires=_TIMING, module="lungvol", unit="%"),
    ColumnSpec(name="irv_pct_tlc", requires=_TIMING, module="lungvol", unit="%"),
)

# FVC/MFVL/EFL/VEcap (M-42, core.analysis.mfvl). Manoeuvres-sheet fields
# (fvc/fev1/fvc_bev/fvc_fet/mfvl_tlc_consistency) and mfvl_ext fields
# (te_min_mfvl/ve_cap/mvv_est) that _RULES' generic conventions do not already
# classify -- their unit= here IS the resolving path. `mfvl_peak_ex_flow`,
# `mfvl_peak_in_flow` (contain "flow" -> L/s) and every `*_pct*` column
# (efl_pct/efl_coverage_pct/ex_flow_pct_mfvl_max/in_flow_pct_mfvl_max/
# max_ex_flow_pct_mfvl_peak/max_in_flow_pct_mfvl_peak/ve_pct_cap/ve_reserve_pct/
# ve_pct_mvv/br_mvv_pct) are already resolved by _RULES before this registry is
# ever consulted -- listed nowhere here, same "documented by the generic rule
# alone" precedent as _LUNG_VOLUMES' own vol_/_pct columns.
_MFVL = (
    ColumnSpec(name="fvc", requires=_TIMING, module="mfvl", unit="L"),
    ColumnSpec(name="fev1", requires=_TIMING, module="mfvl", unit="L"),
    ColumnSpec(name="fvc_bev", requires=_TIMING, module="mfvl", unit="L"),
    ColumnSpec(name="mfvl_tlc_consistency", requires=_TIMING, module="mfvl", unit="L"),
    ColumnSpec(name="fev1_fvc", requires=_TIMING, module="mfvl", unit=DIMLESS),
    ColumnSpec(name="fvc_fet", requires=_TIMING, module="mfvl", unit="s"),
    ColumnSpec(name="te_min_mfvl", requires=_TIMING, module="mfvl", unit="s"),
    ColumnSpec(name="fvc_eofe_ok", requires=_TIMING, module="mfvl", unit=""),
    ColumnSpec(name="efl_present", requires=_TIMING, module="mfvl", unit=""),
    ColumnSpec(name="fev1_source", requires=_TIMING, module="mfvl", unit=""),
    ColumnSpec(name="ve_cap", requires=_TIMING, module="mfvl", unit="L·min⁻¹"),
    ColumnSpec(name="mvv_est", requires=_TIMING, module="mfvl", unit="L·min⁻¹"),
)

# Opt-in PEEPi / modified Campbell diagram (core.analysis.pressure, trigger
# `processing.pressure.peepi.enabled`). Every column is resolved by `_RULES`' generic
# conventions already (peepi* -> cmH2O, peepi_lag -> s, int_/ptp_ -> integrals/rates,
# wob* -> J/min) before this registry is consulted: the `unit=` values here are
# documentation, same precedent as _MANOEUVRES above. The gastric-corrected columns need
# Pgas; the diaphragm ones need Pgas AND Pdi (the correction is what feeds them).
_PEEPI_POES = frozenset({"flow", "poes"})
_PEEPI_PGAS = frozenset({"flow", "poes", "pgas"})
_PEEPI_PDI = frozenset({"flow", "poes", "pgas", "pdi"})
_PEEPI = (
    ColumnSpec(name="peepi_dyn", requires=_PEEPI_POES, module="peepi", unit="cmH₂O", trigger="peepi"),
    ColumnSpec(name="peepi_pgas_drop", requires=_PEEPI_PGAS, module="peepi", unit="cmH₂O", trigger="peepi"),
    ColumnSpec(name="peepi_corr", requires=_PEEPI_PGAS, module="peepi", unit="cmH₂O", trigger="peepi"),
    ColumnSpec(name="peepi_lag", requires=_PEEPI_POES, module="peepi", unit="s", trigger="peepi"),
    ColumnSpec(name="int_oes_preflow", requires=_PEEPI_POES, module="peepi", unit="cmH₂O·s", trigger="peepi"),
    ColumnSpec(name="ptp_oes_preflow", requires=_PEEPI_POES, module="peepi", unit="cmH₂O·s·min⁻¹", trigger="peepi"),
    ColumnSpec(name="wob_in_thr", requires=_PEEPI_POES, module="peepi", unit="J·min⁻¹", trigger="peepi"),
    ColumnSpec(name="wob_in_total_thr", requires=_PEEPI_POES, module="peepi", unit="J·min⁻¹", trigger="peepi"),
    ColumnSpec(name="wobtotal_thr", requires=_PEEPI_POES, module="peepi", unit="J·min⁻¹", trigger="peepi"),
    ColumnSpec(name="int_oesinsp_peepi", requires=_PEEPI_POES, module="peepi", unit="cmH₂O·s", trigger="peepi"),
    ColumnSpec(name="ptp_oesinsp_peepi", requires=_PEEPI_POES, module="peepi", unit="cmH₂O·s·min⁻¹", trigger="peepi"),
    ColumnSpec(name="int_pdiinsp_peepi", requires=_PEEPI_PDI, module="peepi", unit="cmH₂O·s", trigger="peepi"),
    ColumnSpec(name="ptp_pdiinsp_peepi", requires=_PEEPI_PDI, module="peepi", unit="cmH₂O·s·min⁻¹", trigger="peepi"),
)

# Normalisation to a maximal manoeuvre (core.analysis.normalisation, opt-in via
# `processing.pressure.normalization.enabled`): the "Pressure normalised" sheet. Every
# column is resolved by `_RULES` before this registry is consulted (`poes*`/`pdi*` ->
# cmH2O, `*_pct*` -> %, `tt_*` -> dimensionless, `rms*` -> a.u.) except `nrdi`, which
# is deliberately blank: an EMG percentage times a breath rate is an arbitrary-unit index
# (Murphy et al. 2011), so its `unit=""` here IS the resolving path. The sheet also repeats
# `poes_max_ref`/`pdi_max_ref`/`rms_max_ref` (the reference each row was normalised to), which
# `_MANOEUVRES` above already registers under the same names.
_PRESSURE_NORMALISATION = (
    ColumnSpec(name="poes_insp_swing", requires=_PRESSURES_POES, module="pressure_normalisation",
               unit="cmH₂O", sheet="Pressure normalised", trigger="pressure_normalisation"),
    ColumnSpec(name="poes_insp_swing_pct", requires=_PRESSURES_POES, module="pressure_normalisation",
               unit="%", sheet="Pressure normalised", trigger="pressure_normalisation"),
    ColumnSpec(name="poes_mean_insp", requires=_PRESSURES_POES, module="pressure_normalisation",
               unit="cmH₂O", sheet="Pressure normalised", trigger="pressure_normalisation"),
    ColumnSpec(name="poes_mean_insp_pct", requires=_PRESSURES_POES, module="pressure_normalisation",
               unit="%", sheet="Pressure normalised", trigger="pressure_normalisation"),
    ColumnSpec(name="tt_es", requires=_PRESSURES_POES, module="pressure_normalisation",
               unit=DIMLESS, sheet="Pressure normalised", trigger="pressure_normalisation"),
    ColumnSpec(name="pdi_insp_swing", requires=_PRESSURES_PDI, module="pressure_normalisation",
               unit="cmH₂O", sheet="Pressure normalised", trigger="pressure_normalisation"),
    ColumnSpec(name="pdi_insp_swing_pct", requires=_PRESSURES_PDI, module="pressure_normalisation",
               unit="%", sheet="Pressure normalised", trigger="pressure_normalisation"),
    ColumnSpec(name="pdi_mean_insp", requires=_PRESSURES_PDI, module="pressure_normalisation",
               unit="cmH₂O", sheet="Pressure normalised", trigger="pressure_normalisation"),
    ColumnSpec(name="pdi_mean_insp_pct", requires=_PRESSURES_PDI, module="pressure_normalisation",
               unit="%", sheet="Pressure normalised", trigger="pressure_normalisation"),
    ColumnSpec(name="tt_di", requires=_PRESSURES_PDI, module="pressure_normalisation",
               unit=DIMLESS, sheet="Pressure normalised", trigger="pressure_normalisation"),
    ColumnSpec(name="rms_insp_max_pct", requires=frozenset({"flow", "emg"}),
               module="pressure_normalisation", unit="%", sheet="Pressure normalised",
               trigger="pressure_normalisation"),
    ColumnSpec(name="nrdi", requires=frozenset({"flow", "emg"}), module="pressure_normalisation",
               unit="", sheet="Pressure normalised", trigger="pressure_normalisation"),
)

#: Every column/family this skeleton knows about. Later tickets append to this,
#: never remove from or reorder ``LEGACY_MECHANICS_ORDER`` within it.
REGISTRY = (LEGACY_MECHANICS_ORDER + (_ENTROPY, _EMG) + _SEGMENT_EMG + _MANOEUVRES
           + _REFERENCE_MANOEUVRES + _LUNG_VOLUMES + _MFVL + _PEEPI
           + _PRESSURE_NORMALISATION)

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
