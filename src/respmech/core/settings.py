"""Typed, validated settings model for RespMech.

Replaces the legacy approach (an executable Python file holding a nested dict that
was JSON-merged onto defaults via ``SimpleNamespace``). Problems fixed here:

* settings are **data**, not executable code (loaded from TOML — see
  ``respmech.settingsio``);
* every field has a real default and validation, so a missing nested subsection can
  never ``KeyError`` the way the legacy ``applysettings`` did (legacy bug #6);
* the ``sampling`` (resample) section from the ``resampling-options`` line is a
  first-class, typed field.

The dataclasses mirror the TOML schema (``schema_version = 1``). ``from_dict`` is
tolerant (unknown keys are collected, not fatal) and ``validate`` raises
``SettingsError`` with a clear message.
"""
import math
import os
import types
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, Optional, Union, get_args, get_origin, get_type_hints

from respmech.core.analysis.signals import SINGLE_SIGNALS, effective_signals


class SettingsError(ValueError):
    """Raised when settings are missing/invalid, with an actionable message."""


SCHEMA_VERSION = 2

# schema 1 wrote processing.volume.trend_peak_min_height into every analysis, including
# the ones that never touched trend correction — the GUI had no control for it, so the
# value carried no user intent. It was inherited from legacy v1 and is an ABSOLUTE depth
# below the recording's global volume maximum, so on ordinary tidal breathing (range <
# 0.8) it matches no trough at all and the correction cannot run. Reading exactly this
# value from a schema-1 analysis therefore means "the old default", and is upgraded to
# the scale-free rule (recorded in Settings.notices, and reported wherever it is loaded).
RETIRED_TREND_PEAK_MIN_HEIGHT = 0.8


# --- sub-sections -----------------------------------------------------------

@dataclass
class InputFormat:
    sampling_frequency: int | None = None
    matlab_variant: str = "mac"          # "windows" | "mac"  (legacy 1 | 2)
    decimal: str = "."


@dataclass
class Channels:
    # 1-based column numbers (kept for familiarity with LabChart exports).
    poes: int | None = None
    pgas: int | None = None
    pdi: int | None = None
    volume: int | None = None
    flow: int | None = None
    emg: list[int] = field(default_factory=list)
    entropy: list[int] = field(default_factory=list)


@dataclass
class SubjectEntry:
    """One participant's spirometry-derived lung volumes, keyed on
    ``core.summary.group_key(file, settings)`` -- the SAME leading-filename-token (or
    ``output.group_regex``) key the cohort summary already groups files by, so a
    subject's TLC/VC/RV/FEV1/MVV apply to every file ``group_key`` assigns to their key,
    with no separate per-subject file list to keep in sync. ``fev1_l`` is a spirometry
    value, preferred over any FEV1 this codebase might derive from a recorded FVC
    manoeuvre later -- a formal spirometer reading is the more reliable number.
    ``mvv_lpm`` (maximum voluntary ventilation, L/min) is likewise a spirometry value,
    for comparison against a future derived MVV estimate.

    Never sent into ``core.compute``/``core._legacy_ns`` -- only
    ``core.analysis.references`` and a future lung-volumes analysis module read this
    table, well after the per-breath mechanics loop this never touches has already run.

    ``folder`` is the same carried-over-state provenance tag as every other tagged kind
    -- see ``_CARRIED_KINDS`` below."""
    key: str
    tlc_l: float | None = None
    vc_l: float | None = None
    rv_l: float | None = None
    fev1_l: float | None = None
    mvv_lpm: float | None = None
    folder: str | None = None


@dataclass
class InputSettings:
    folder: str = "input"
    files: str = "*.*"
    format: InputFormat = field(default_factory=InputFormat)
    channels: Channels = field(default_factory=Channels)
    # per-participant lung volumes (TLC/VC/RV/FEV1/MVV), keyed on group_key -- see
    # SubjectEntry's own docstring.
    subjects: list[SubjectEntry] = field(default_factory=list)


@dataclass
class AnalysisSettings:
    """The signal set this analysis declares (R7). Empty (the default) means
    "derive it from whichever channels are assigned" — see
    ``core.analysis.signals.effective_signals``. Only written to TOML while it
    diverges from that derived set (``settingsio.toml_io.save_toml``); the run
    manifest (``dumps_toml``) always writes the resolved, effective set instead,
    so a saved analysis file and a run's own provenance never disagree about
    what "signals" means for that file.
    """
    signals: list[str] = field(default_factory=list)


@dataclass
class SamplingSettings:
    """Optional pre-processing resample (from the resampling-options line)."""
    resample: bool = False
    resample_to_frequency: int = 200


@dataclass
class PeakSettings:
    height: float = 0.1
    distance_s: float = 0.1
    width_s: float = 0.5


@dataclass
class SeparatorEntry:
    """One file's manual segment boundaries for the ``separators`` EMG-only
    segmentation method: N times split the recording into N+1 segments, numbered
    from 1 (0 separators is the same as ``whole_file``, just reached via this
    explicit method instead). ``times_s`` are absolute seconds into the recording,
    measured from its own start (the same clock ``breath['time']`` already uses
    elsewhere), and must be strictly increasing and non-negative — enforced by
    ``Settings.validate()``, not here, so a malformed entry is reported the same
    way every other settings error is (a ``SettingsError``, never a raw exception).

    ``folder`` is the same carried-over-state provenance tag as ``ExcludeEntry.
    folder``/``BreathTypeEntry.folder`` -- see ``_CARRIED_KINDS`` below, which is
    what actually wires rebase/relativize/carried/clear for this field.
    """
    file: str
    times_s: list[float] = field(default_factory=list)
    folder: str | None = None


@dataclass
class SegmentationOverrideEntry:
    """One file's manual repair of the AUTOMATIC flow-/volume-based breath
    segmentation — reparation of a mis-detected boundary, never a new
    segmentation method. Contrast :class:`SeparatorEntry`, which REPLACES the whole
    boundary list for an EMG-only ``separators`` analysis: this entry only ADJUSTS the
    boundary list the automatic flow-/volume-based detector already produced.

    ``cut_s`` inserts a new breath boundary at a given time (splitting one detected
    breath into two, when a flat/leaky expiration was under-split into one breath that
    is really two). ``join_s`` names a time near an EXISTING automatic boundary and
    removes the nearest one within tolerance (merging the two breaths on either side
    into one, when a flow wobble mid-breath was over-split into two). Both are absolute
    seconds into the recording, the same clock ``breath['time']``/``SeparatorEntry.
    times_s`` already use. See ``core.compute.apply_segmentation_overrides`` for
    exactly how the two combine and are re-split into phases, and
    ``Settings.validate()`` for the (non-negative, strictly increasing within EACH
    list) form check.

    ``folder`` is the same carried-over-state provenance tag as every other tagged
    kind — see ``_CARRIED_KINDS`` below; stamped only when the entry is first created,
    exactly like ``SeparatorEntry.folder``/``ExcludeEntry.folder``.
    """
    file: str
    cut_s: list[float] = field(default_factory=list)
    join_s: list[float] = field(default_factory=list)
    folder: str | None = None


@dataclass
class SegmentationSettings:
    method: str = "flow"                 # "flow" | "volume" | "whole_file" | "separators"
    buffer: int = 800
    peak: PeakSettings = field(default_factory=PeakSettings)
    # Manual segment boundaries for the "separators" EMG-only method (one entry per
    # file; a file with no entry here under "separators" is treated as whole_file --
    # 0 separators is a legal, explicit way to say "one segment").
    separators: list[SeparatorEntry] = field(default_factory=list)
    # Manual repair (cut/join) of the AUTOMATIC flow-/volume-based segmentation for a
    # flow-bearing analysis — only ever consulted when `method` is "flow"/"volume";
    # `separators` above is the EMG-only counterpart (no automatic detection to
    # repair there), consulted only for "whole_file"/"separators". An entry present
    # for a file not currently using the method it applies to is simply never read,
    # the same "dormant, unconsulted" relationship `separators` already has with the
    # other methods.
    overrides: list[SegmentationOverrideEntry] = field(default_factory=list)
    # K-035 boundary-truncation quality notice (compute.trim_boundary_notices): how much
    # shorter than the file's own median a boundary breath's phase must be before it is
    # flagged as likely truncated by trim(). 0.8 was measured, not guessed (see that
    # function's docstring); exposed here (06-09-2026 review) so a recording with
    # atypically high natural breath-to-breath variability can be re-tuned per file/study
    # without a code change, without this codebase silently guessing a system-wide
    # replacement statistic that was NOT demonstrated to be an unambiguous improvement.
    boundary_notice_min_relative_duration: float = 0.8
    # minimum number of OTHER breaths required before the comparison is trusted (an
    # unstable median on a very short recording can otherwise flag, or hide, truncation).
    boundary_notice_min_other_breaths: int = 3


@dataclass
class VolumeSettings:
    inverse_flow: bool = False
    integrate_from_flow: bool = False
    inverse_volume: bool = False
    correct_drift: bool = True
    correct_trend: bool = False
    trend_method: str = "linear"
    # Scale-free trough criterion: the fraction of the recording's own volume range that
    # must flank a trough for it to anchor the trend envelope (see compute.trend_anchors).
    # Used unless trend_peak_min_height is set.
    trend_peak_min_prominence_frac: float = 0.05
    # Legacy absolute gate, in volume units, measured DOWNWARD from the recording's global
    # volume maximum. None (the default) = use the scale-free rule above. An explicit value
    # reproduces a pre-2.3.3 analysis exactly, and is never reinterpreted.
    trend_peak_min_height: Optional[float] = None
    trend_peak_min_distance_s: float = 0.4


@dataclass
class WobSettings:
    calc_from: str = "average"           # "average" | "individual"
    avg_resampling_obs: int = 500


@dataclass
class NoiseSettings:
    """Shared-profile EMG noise reduction. ONE profile + ONE parameter set built from
    a rest reference and applied identically to every file in a test (never re-tuned
    per file). See docs/NOISE_ECG_OPTIMIZATION.md."""
    enabled: bool = False
    # EMG-free rest reference the noise profile is built from (shared across the test).
    reference_file: str | None = None
    # explicit EMG-free windows [[t0, t1], ...] in reference_file; if empty and
    # use_expiration is True, the profile is built from the reference's expiration.
    reference_intervals: list[Any] = field(default_factory=list)
    # the recordings folder that was active when reference_file/reference_intervals were
    # captured (rebased/relativized the same way as input.folder/output.folder — see
    # settingsio.toml_io). None means "unrecorded" (an analysis from before this field
    # existed) and is always treated as unproven, the same as a genuine mismatch — see
    # is_carried_folder(). Lets the GUI warn when a reference chosen against one input
    # folder is still active after the folder changes, instead of silently reusing it.
    reference_folder: str | None = None
    use_expiration: bool = True
    # Which noise reference SOURCE to build the profile from. 'auto' (the default) keeps
    # `use_expiration`/`reference_intervals` as the two saved choices a flow-bearing
    # analysis has always had -- this field adds nothing new for that case, it only NAMES
    # the two EMG-only alternatives an analysis with no flow channel can reach: 'rest_segments'
    # (a segment typed 'rest' in the reference file -- see BREATH_KINDS) and 'interburst' (not
    # yet implemented -- see resolve_noise_reference_mode()). 'rest_segments'/'interburst' are
    # reachable only explicitly and only for an EMG-only signal set; Settings.validate() rejects
    # either one while a flow channel is declared, since 'auto' already covers that case fully.
    reference_mode: str = "auto"
    # fixed STFT parameters (decoupled from the noise-clip length — the legacy bug).
    n_fft: int = 256
    hop_length: int = 64
    win_length: int = 256
    n_std_thresh: float = 1.0
    n_grad_freq: int = 0
    n_grad_time: int = 4
    # prop_decrease is chosen ONCE per test: auto (highest value keeping worst-channel
    # fidelity >= target) or a fixed manual value.
    prop_decrease: float = 0.6
    auto_prop: bool = True
    fidelity_target: float = 0.8


@dataclass
class RobustPeakSettings:
    """Opt-in cardiac-gated peak EMG.

    The shipped per-breath EMG number is the MAX of the rolling RMS. On strongly
    cardiac-coupled recordings that maximum usually sits on a residual heartbeat rather
    than on diaphragm activity, because averaged-template subtraction leaves a sharp
    beat-to-beat residual that a maximum statistic seeks out. Gating blanks a window
    around every detected R-peak and takes the maximum of what survives, so the number is
    read from cardiac-free signal.

    Losing samples is not a cost here: EMG is reported relative to the patient's own
    maximum, so a transformation applied identically to every breath (and to the
    normalising maximum) cancels in the ratio. What does cost is incomplete R-peak
    detection — an undetected beat is neither subtracted nor blanked, so the gate discards
    the honest samples and leaves the missed beat exposed. Hence the quality guards below;
    when any of them trips the gated columns are NaN rather than quietly wrong.

    Off by default: enabling it ADDS columns and never changes the existing ones.
    """
    enabled: bool = False
    # Half-width of the blanked window around each R-peak. The cardiac excursion in the
    # rolling RMS is about (rms_window_s + QRS duration) wide; 120 ms covers it with margin.
    gate_half_width_s: float = 0.120
    # Quality guards. A phase whose surviving fraction or longest cardiac-free island falls
    # below these is not measurable and reports NaN.
    min_survival: float = 0.40
    min_island_s: float = 0.20
    # Detection guards, applied per file. An RR interval longer than long_rr_factor x the
    # median means a beat was missed; more than max_long_rr_frac of such intervals makes the
    # peak set too incomplete to gate on. hr_ceiling_margin flags a heart rate approaching
    # the detector's own refractory ceiling (60 / ecg_min_distance_s), where misses begin.
    long_rr_factor: float = 1.6
    max_long_rr_frac: float = 0.02
    hr_ceiling_margin: float = 0.10


@dataclass
class EmgSettings:
    rms_window_s: float = 0.050
    remove_ecg: bool = False
    detect_channel: int = 0
    ecg_min_height: float = 0.0005
    ecg_min_distance_s: float = 0.5
    ecg_min_width_s: float = 0.001
    ecg_window_s: float = 0.4
    # Auto-detect the 5 fields above ONCE per test from a reference file's raw EMG
    # (core.emg.suggest_ecg_settings — the same analysis the GUI's ECG "Auto-suggest"
    # button runs) and apply the result identically to every file, mirroring
    # noise.auto_prop. Off by default -> existing behaviour/golden output unchanged.
    # ecg_reference_file: file (relative to input.folder) to analyse; None -> the
    # first file the batch's input pattern matches.
    ecg_auto_detect: bool = False
    ecg_reference_file: str | None = None
    # the recordings folder active when ecg_reference_file was last written — same
    # provenance tag as NoiseSettings.reference_folder/ExcludeEntry.folder (see
    # is_carried_folder()/carried_over_state() below, and _CARRIED_KINDS). None means
    # "unrecorded" and is always treated as unproven, never guessed.
    ecg_reference_folder: str | None = None
    outlier_rms_sd_limit: float = 0.0
    # EMG amplitude normalisation for the OUTPUT tables (P14): each file's RMS columns
    # are also reported as a % of a reference so amplitudes compare across
    # subjects/electrodes. "none" | "per_file_max" | "per_file_mean". This adds a
    # normalised sheet to the output; it never changes the raw computed RMS. By
    # default the reference is each file's OWN max/mean breath, per column — which
    # makes every file's peak reach 100% by construction and does NOT make amplitudes
    # comparable across files/subjects (documented on the website).
    # Set normalization_reference_file to a filename already in this batch
    # (relative to input.folder, e.g. a maximal inspiratory/expiratory manoeuvre
    # recorded once per subject) to normalise every file against THAT file's
    # max/mean instead of its own — see core.summary.reference_values_for_batch.
    normalization: str = "per_file_max"
    normalization_reference_file: str | None = None
    # same provenance tag as ecg_reference_folder above, for the SAME reason.
    normalization_reference_folder: str | None = None
    save_sound: bool = False
    plot_yscale: list[float] = field(default_factory=lambda: [-0.1, 0.1])
    # legacy filename-keyed noise-profile intervals (kept for migration):
    # [[file, source_file_or_empty, [t0,t1]], ...]
    # new shared-profile noise reduction (canonical):
    noise: NoiseSettings = field(default_factory=NoiseSettings)
    # opt-in cardiac-gated peak EMG (adds columns; default off -> existing output unchanged)
    robust_peak: RobustPeakSettings = field(default_factory=RobustPeakSettings)


@dataclass
class EntropySettings:
    # `epochs` is the template length passed to the vendored pyEntropy routine, i.e. m + 1
    # (see core/entropy.py's docstring). Changed 2 -> 3 (05-09-2026, content review): the
    # previous default of 2 reported m = 1, not the m = 2 that is the near-universal
    # convention in the sample-entropy literature (Richman & Moorman 2000; Yentes et al.
    # 2013). Set 2 to reproduce the m = 1 values RespMech reported before this change.
    epochs: int = 3
    tolerance: float = 0.1


@dataclass
class PtpSettings:
    # Pressure-time product baseline = mean over a short window at the phase start
    # (end-expiratory for inspiration, end-inspiratory for expiration). A window
    # (vs a single sample) is robust to boundary noise. 0.05 s is a good default.
    baseline_window_s: float = 0.05


@dataclass
class PeepiSettings:
    """Opt-in PEEPi detection and the modified Campbell diagram's threshold work
    (``core.analysis.pressure``). Off by default: enabling it only ADDS columns, so no
    existing table changes with it off. (Separately, an EMG normalisation reference file with
    typed maximal breaths is read at those breaths whatever this setting says.)

    The four numeric fields are literature-informed STARTING values, not measured ones,
    and are provisional until they have been measured on real recordings (the same
    lesson as ``segmentation.boundary_notice_min_relative_duration``):

    ``search_window_s``: how far back into the preceding breath's expiration the onset
    of the pre-flow Poes deflection is searched for.
    ``smooth_s``: moving-average width applied to the search window before its slope is
    read (the pressures actually READ at the onset and at the start of flow stay raw).
    ``onset_slope_frac``: walking back from the start of inspiratory flow, the deflection
    lasts while the smoothed Poes is still falling faster than this fraction of the
    steepest fall in the search window.
    ``min_deflection`` (cmH2O): a smaller deflection than this is reported as 0.
    """
    enabled: bool = False
    search_window_s: float = 1.0
    smooth_s: float = 0.05
    onset_slope_frac: float = 0.1
    min_deflection: float = 0.5


@dataclass
class PressureNormalizationSettings:
    """Opt-in normalisation of the inspiratory pressures and EMG to a maximal
    manoeuvre (``core.analysis.normalisation``). Off by default: enabling it only
    ADDS a "Pressure normalised" sheet to each file's workbook, so every existing
    output is unchanged with it off.

    The reference is the file's ``max_insp``/``sniff`` breath, resolved exactly like the
    other manoeuvre references (an explicit ``processing.references`` entry, then the
    group default, then the file's own typed breath); there is nothing to configure here
    beyond switching the analysis on."""
    enabled: bool = False


@dataclass
class PressureSettings:
    """Pressure-derived analyses that build on the ordinary per-breath mechanics
    (``core.analysis.pressure``, ``core.analysis.normalisation``): today PEEPi and the
    modified Campbell diagram, and the normalisation to a maximal manoeuvre."""
    peepi: PeepiSettings = field(default_factory=PeepiSettings)
    normalization: PressureNormalizationSettings = field(
        default_factory=PressureNormalizationSettings)


@dataclass
class IcSettings:
    """Inspiratory-capacity manoeuvre extraction (M-29, ``core.analysis.manoeuvres``).

    ``preceding_breaths``/``min_preceding_breaths`` govern how ``ic_eelv_pre`` (the
    end-expiratory-lung-volume baseline an IC's own volume swing is measured against)
    is estimated: the mean end-expiratory volume of up to ``preceding_breaths`` tidal
    breaths immediately before the manoeuvre, falling back to the manoeuvre breath's
    OWN pre-inspiratory volume sample when fewer than ``min_preceding_breaths`` tidal
    breaths precede it in the file (too little context to average over). All the
    ``†``-marked fields below are DELIBERATE placeholders (the plan's own starting
    values, not measured ones): ``core.analysis.manoeuvres`` has no production IC
    recording available in this sandbox to calibrate against (K-035's lesson — a
    threshold picked without a real recording in hand is a guess, not a fact), so the
    formulas are implemented and pinned by synthetic/analytical tests, but the actual
    cut-offs need a pass against real recordings before they are trusted clinically.
    See ``docs/beslutninger.md`` for what to measure and where to record it.
    """
    preceding_breaths: int = 3
    min_preceding_breaths: int = 2
    # EELV_UNSTABLE fires when the preceding breaths' own EELV spread (SD) exceeds
    # this fraction of the MANOEUVRE'S OWN vol_ic -- never of ic_eelv_pre itself
    # (self-review finding): with the default zero-referenced + drift-corrected
    # volume signal, ic_eelv_pre routinely sits within a few mL of 0 L, and dividing
    # by a near-zero baseline would make the flag fire on ordinary breath-to-breath
    # noise in almost every real recording. vol_ic is always a real, non-trivial size.
    eelv_tolerance_frac: float = 0.2            # † EELV_UNSTABLE threshold
    plateau_flow_lps: float = 0.1               # † inspiratory-plateau flow ceiling
    min_plateau_s: float = 0.0                  # † NO_PLATEAU threshold (0 = never fires yet)
    repeatability_frac: float = 0.10            # † NOT_REPEATABLE threshold
    low_effort_frac: float = 0.5                # † LOW_EFFORT threshold (vs tidal median)
    aggregate: str = "mean"                     # "mean" | "median" — how repeat ICs are combined
    reject_flags: list[str] = field(default_factory=lambda: ["LOW_EFFORT"])
    # How a tidal breath's operating-lung-volume baseline tracks the reference IC. This
    # field only declares and validates the enum; the actual arithmetic is a later
    # ticket's scope, so the settings model is complete before the pipeline wiring
    # lands. "none" (the default): ic_op is simply the resolved reference IC, unchanged
    # across the file. "within_file": ic_op additionally tracks each tidal breath's own
    # end-expiratory drift relative to the reference IC's own end-expiratory level,
    # using ONLY same-file IC/EELV data (a cross-file end-expiratory comparison is not
    # meaningful -- two different recordings rarely share a common volume zero).
    eelv_tracking: str = "none"                  # "none" | "within_file"


@dataclass
class LungVolumeSettings:
    """Container for the operating-lung-volumes family (the IC-extraction settings
    already read by ``core.analysis.manoeuvres``; per-tidal-breath EELV/EILV tracking
    is a later ticket's own arithmetic, alongside this).

    ``require_references``: off by default -- a reference source named in
    ``processing.references``/``reference_defaults`` but absent from the batch's own
    matched files (``core.pipeline.match_input_files``) is only a soft
    ``core.analysis.references.check_links`` caution while this is False; turning it on
    makes such a missing source a hard ``ui.validation.path_problem`` blocker instead.
    One policy throughout: an unresolved link is a caution plus NaN and a notice; it is
    a blocker only when this flag is set.

    ``baseline_pattern``: a case-insensitive regex (default matches a filename
    containing 'baseline' or 'rest') used -- by a later ticket's own baseline-file
    heuristics, never here -- to suggest which file in a participant's own set is their
    resting/baseline recording when ``ReferenceEntry.baseline_ic``/
    ``GroupReferenceEntry.baseline_ic`` is not set explicitly. Stored and round-tripped
    already; not yet read by any code path.
    """
    require_references: bool = False
    baseline_pattern: str = r"(?i)baseline|rest"
    ic: IcSettings = field(default_factory=IcSettings)


@dataclass
class MfvlSettings:
    """FVC/MFVL/EFL/VEcap and ventilatory-capacity extraction (M-42,
    ``core.analysis.mfvl``).

    ``source``: how the maximal expiratory flow-volume (MEFV) envelope a tidal
    breath is placed against is built from a file's resolved ``fvc`` reference
    breath(s) (``core.analysis.references.resolve_reference(..., "fvc", ...)``, the
    SAME slot ``ic``/``baseline_ic``/``max_insp`` already share). ``"single"`` (the
    default) uses the ONE attempt with the largest ``fvc`` among the resolved
    breaths (ATS/ERS 2019: report the largest FVC/FEV1 across acceptable attempts,
    not necessarily from the same manoeuvre — reused here for which single curve to
    pick). ``"envelope"`` takes the per-volume MAXIMUM flow across every resolved
    attempt (Johnson 1999's own composite-MEFV construction) — identical to
    "single" by construction whenever only one attempt resolved.

    ``efl_rel_tol``/``efl_abs_tol_lps``: a tidal sample counts as flow-LIMITED
    (Johnson 1999) when ``flow >= mefv(v)*(1-efl_rel_tol) - efl_abs_tol_lps``. Both
    default to 0.0 (a sample must literally reach the envelope, the strictest
    reading) — the SAME ``†`` provenance as ``IcSettings``' own placeholders: this
    sandbox has no production FVC recording to calibrate a tolerance against
    (K-035's lesson), so the formula is pinned by analytical/synthetic tests, but
    the tolerance wants a pass against real recordings before it is trusted.

    ``efl_present_min_pct``: ``efl_present`` (a boolean summary flag) is True only
    once ``efl_pct`` clears this floor — a single grazing sample rounding to a
    fraction of a percent should not, on its own, read as "this patient shows
    expiratory flow limitation" on a summary display.

    ``mvv_fev1_multiplier``: the classic FEV1 x 40 MVV estimate (ATS/ACCP 2003),
    the fallback when neither ``input.subjects.mvv_lpm`` nor a resolved FVC-derived
    FEV1 applies.
    """
    source: str = "single"                       # "single" | "envelope"
    efl_rel_tol: float = 0.0                      # † EFL flow-limitation tolerance (relative)
    efl_abs_tol_lps: float = 0.0                  # † EFL flow-limitation tolerance (absolute, L/s)
    efl_present_min_pct: float = 5.0              # † efl_present floor
    mvv_fev1_multiplier: float = 40.0             # ATS/ACCP 2003 MVV = FEV1 x 40 fallback


@dataclass
class ExcludeEntry:
    file: str
    breaths: list[int] = field(default_factory=list)
    # the recordings folder (input.folder) active when this entry was last written —
    # rebased/relativized like input.folder/output.folder (settingsio.toml_io). None means
    # "unrecorded" (an analysis written before this field existed): from_dict/_toml_clean
    # already tolerate an unknown/missing key on either side of the schema change, so an
    # older analysis loads unchanged and a newer one opened by older code just files the
    # key under Settings.unknown. See is_carried_folder()/carried_over_state() — this field
    # never reaches core.compute (core/_legacy_ns.py deliberately drops it), it exists
    # purely so the UI can tell a same-name exclusion from a DIFFERENT folder apart from
    # one made in the folder currently loaded.
    folder: str | None = None


@dataclass
class BreathCountEntry:
    file: str
    count: int
    folder: str | None = None


#: the closed set of manoeuvre labels a single breath can be typed as (M-19). ``ic``/
#: ``fvc``/``ic_fvc`` are inspiratory-capacity/forced-vital-capacity manoeuvres (M-29
#: extracts their values); ``max_insp``/``sniff`` are maximal-effort references (read by
#: ``core.analysis.normalisation``); ``rest`` marks a quiet SEGMENT usable as an EMG-only noise reference
#: (M-22's ``resolve_noise_reference_mode``/``rest_segments`` -- only reachable for a
#: signal set with no flow channel at all, since a flow-bearing set already has a
#: well-defined quiet period in every breath's own expiration). On a flow-bearing file,
#: typing a BREATH ``rest`` does not make it INTO the reference either: like every other
#: kind, it is instead excluded from the automatic expiration-based reference clip
#: (decision 11) -- there is no mechanism today that lets a flow-bearing analysis pick
#: an explicit breath as its reference by typing it; ``processing.emg.noise.
#: reference_intervals`` is the only explicit override that signal set has.
#: ``other`` is any manoeuvre breath none of the above name. A typed breath is
#: excluded from the tidal average the same way a manually excluded one is -- see
#: ``core._legacy_ns.to_legacy_ns``'s union of ``exclude_breaths``/``breath_types``.
BREATH_KINDS = ("ic", "fvc", "ic_fvc", "max_insp", "sniff", "rest", "other")


@dataclass
class BreathTypeEntry:
    """One breath in one file, typed as a named manoeuvre rather than tidal breathing.

    Flat, per-breath entries (never a ``breath_no -> kind`` dict: TOML table keys are
    strings, so a dict keyed on an int would not survive ``from_dict(to_dict())`` --
    see the "TOML-tabelnøgler er strenge" finding this mirrors ``ExcludeEntry`` for).
    ``t_onset_s`` is a purely advisory time anchor (written when a breath is typed in
    the UI, a later ticket) -- breath numbers are re-segmentation-relative, so nothing
    here re-resolves it; it exists only so a future notice can flag a mismatch.
    ``folder`` is the same carried-over-state provenance tag as ``ExcludeEntry.folder``
    -- see ``_CARRIED_KINDS`` below, which is what actually wires rebase/relativize/
    carried/clear for this field; adding the row there was the whole point of M-07.
    """
    file: str
    breath: int
    kind: str
    label: str = ""
    t_onset_s: float | None = None
    folder: str | None = None


@dataclass
class BreathRef:
    """One or more breaths in one file, named as a reference source for a manoeuvre
    kind -- e.g. ``ic = { file = "P03_IC.txt", breaths = [2, 3, 4] }`` under a
    :class:`ReferenceEntry`/:class:`GroupReferenceEntry`. Plural ``breaths`` because a
    file can carry more than one repeat of the same manoeuvre (``IcSettings.aggregate``
    combines them once a later pipeline pass resolves the link); a single-breath
    reference is simply a one-element list.

    Nested inside an ``X | None`` field (``ReferenceEntry.ic`` etc.), which needs the
    earlier PEP 604 fix to ``_unwrap_optional`` to round-trip through TOML at all --
    without it ``_coerce`` never reaches the ``is_dataclass(typ)`` branch for a
    ``BreathRef | None`` annotation and silently hands back the raw dict instead of a
    ``BreathRef`` instance."""
    file: str
    breaths: list[int] = field(default_factory=list)


@dataclass
class ReferenceEntry:
    """Per-file reference manoeuvres: which breaths (in which file, possibly a
    different one) an ANALYSED file's inspiratory-capacity/forced-vital-capacity/
    maximal-effort values are read from. Takes precedence over any matching
    :class:`GroupReferenceEntry` for the same file's group, which in turn takes
    precedence over the file's OWN typed breaths -- see
    ``core.analysis.references.resolve_reference`` for the exact order, applied
    independently per slot (a file can have an explicit ``ic`` but fall through to the
    group default for ``fvc``).

    ``baseline_ic`` names the resting/baseline recording an operating-lung-volume
    calculation's ``delta_ic``/``delta_eelv`` (a later ticket's own arithmetic) is
    measured relative to -- a DIFFERENT recording by nature (a baseline is measured
    before/after, never inferred from the manoeuvre breath itself), so unlike
    ``ic``/``fvc``/``max_insp`` it has no "own typed breath" fallback (see
    ``resolve_reference``'s docstring).

    Referenced breaths are never sent into ``core.compute``/``core._legacy_ns``: this
    table is read only by ``core.analysis.references`` and later lung-volume/MFVL
    analysis modules, all downstream of the ordinary per-breath mechanics loop.

    ``folder`` is the same carried-over-state provenance tag as every other tagged kind
    -- see ``_CARRIED_KINDS`` below."""
    file: str
    folder: str | None = None
    ic: BreathRef | None = None
    fvc: BreathRef | None = None
    baseline_ic: BreathRef | None = None
    max_insp: BreathRef | None = None


@dataclass
class GroupReferenceEntry:
    """Same shape as :class:`ReferenceEntry`, but applied to every file in a GROUP
    (``core.summary.group_key``) rather than to one named file -- the common
    multi-file-per-participant case, where every file from the same participant shares
    one IC/FVC/baseline recording instead of repeating the same
    :class:`ReferenceEntry` under every one of that participant's files. A file-level
    ``ReferenceEntry`` for the same slot always wins over its group's default -- see
    :class:`ReferenceEntry`'s docstring and ``core.analysis.references.
    resolve_reference``.

    ``folder`` is the same carried-over-state provenance tag as every other tagged kind
    -- see ``_CARRIED_KINDS`` below."""
    group: str
    folder: str | None = None
    ic: BreathRef | None = None
    fvc: BreathRef | None = None
    baseline_ic: BreathRef | None = None
    max_insp: BreathRef | None = None


@dataclass
class ProcessingSettings:
    sampling: SamplingSettings = field(default_factory=SamplingSettings)
    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)
    volume: VolumeSettings = field(default_factory=VolumeSettings)
    wob: WobSettings = field(default_factory=WobSettings)
    emg: EmgSettings = field(default_factory=EmgSettings)
    entropy: EntropySettings = field(default_factory=EntropySettings)
    ptp: PtpSettings = field(default_factory=PtpSettings)
    lung_volume: LungVolumeSettings = field(default_factory=LungVolumeSettings)
    mfvl: MfvlSettings = field(default_factory=MfvlSettings)
    pressure: PressureSettings = field(default_factory=PressureSettings)
    exclude_breaths: list[ExcludeEntry] = field(default_factory=list)
    breath_counts: list[BreathCountEntry] = field(default_factory=list)
    breath_types: list[BreathTypeEntry] = field(default_factory=list)
    # per-file and per-group reference manoeuvres -- see ReferenceEntry/
    # GroupReferenceEntry's own docstrings and core.analysis.references.
    references: list[ReferenceEntry] = field(default_factory=list)
    reference_defaults: list[GroupReferenceEntry] = field(default_factory=list)


@dataclass
class DataOutput:
    save_average: bool = True
    save_breath_by_breath: bool = True
    save_processed: bool = False
    # Only affects the per-file processed-signal CSV (results.py); excluded breaths are
    # ALWAYS left out of the averages/workbooks. Legacy default was off.
    include_ignored_breaths: bool = False


@dataclass
class DiagnosticsOutput:
    save_pv_average: bool = True
    save_pv_individual: bool = True
    pv_columns: int = 3
    pv_rows: int = 4
    save_raw: bool = True
    save_trimmed: bool = True
    save_drift: bool = True
    save_emg: bool = True                # per-channel EMG overviews (raw / ECG-removed / noise-reduced)
    # tidal loops inside the file's own MFVL; only planned when a breath is typed fvc/ic_fvc
    save_flow_volume: bool = True


@dataclass
class OutputSettings:
    folder: str = "output"
    data: DataOutput = field(default_factory=DataOutput)
    diagnostics: DiagnosticsOutput = field(default_factory=DiagnosticsOutput)
    # cohort grouping for the summary (P15): a regex whose first capture group is the
    # subject/condition key; None → the leading filename token (e.g. "P03_120W" → "P03").
    group_regex: Optional[str] = None


@dataclass
class Settings:
    schema_version: int = SCHEMA_VERSION
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    input: InputSettings = field(default_factory=InputSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    # keys we did not recognise while parsing (kept for a warning, never fatal).
    unknown: dict = field(default_factory=dict)
    # schema upgrades applied while loading, in plain English. Surfaced by the GUI and
    # written into the run report so an upgraded setting is never applied silently.
    notices: list[str] = field(default_factory=list)

    # -- (de)serialisation --------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        unknown: dict = {}
        d = d or {}
        obj = _build(cls, d, unknown, path="")
        obj.unknown = unknown
        obj.notices = _upgrade(obj, d)
        obj.notices.extend(_reconcile_signals(obj))
        return obj

    def to_dict(self) -> dict:
        return _to_dict(self, drop={"unknown", "notices"})

    # -- validation ---------------------------------------------------------
    def validate(self) -> "Settings":
        f = self.input.format
        if f.sampling_frequency is None:
            raise SettingsError("input.format.sampling_frequency is required")
        if not isinstance(f.sampling_frequency, int):
            raise SettingsError("input.format.sampling_frequency must be an integer")
        if f.matlab_variant not in ("windows", "mac"):
            raise SettingsError("input.format.matlab_variant must be 'windows' or 'mac'")

        ch = self.input.channels

        # R7: an analysis names its signal set either explicitly (analysis.signals) or
        # implicitly (whichever channels are assigned) -- see core.analysis.signals.
        # effective_signals()/Capabilities.from_settings is the ONE function every other
        # consumer of "declared" (channel_collision, plan_outputs, the CLI/run-report --
        # wired in by later tickets) will read the same way from, so this method's own
        # notion of the declared set can never drift from theirs.
        raw_signals = self.analysis.signals
        if isinstance(raw_signals, str):
            raise SettingsError(
                "analysis.signals must be a list of signal names, not a bare string: "
                f"{raw_signals!r}")
        if raw_signals is None:
            raw_signals = []            # only reachable via a dict-based caller (TOML
        elif not isinstance(raw_signals, list):           # has no null) -- reads as "derive it"
            raise SettingsError(
                "analysis.signals must be a list of signal names, not "
                f"{type(raw_signals).__name__}")

        _known_signals = frozenset(SINGLE_SIGNALS) | {"emg"}
        for sig in raw_signals:
            if sig not in _known_signals:
                raise SettingsError(f"analysis.signals contains an unknown signal '{sig}'")

        declared = effective_signals(self)
        explicit = frozenset(raw_signals)
        if not declared:
            raise SettingsError("analysis.signals must name at least one of 'flow' or 'emg'")
        if (declared & {"poes", "pgas", "pdi"}) and "flow" not in declared:
            raise SettingsError("analysis.signals: 'poes', 'pgas' and 'pdi' require 'flow'")

        for name in SINGLE_SIGNALS:
            if name in declared and getattr(ch, name) is None:
                if explicit:
                    raise SettingsError(f"input.channels.{name} is required by analysis.signals")
                raise SettingsError(f"input.channels.{name} is required")
        if "emg" in declared and not ch.emg:
            raise SettingsError(
                "input.channels.emg must name at least one column when "
                "'emg' is in analysis.signals")

        if ("flow" in declared and not self.processing.volume.integrate_from_flow
                and ch.volume is None):
            raise SettingsError(
                "input.channels.volume is required unless "
                "processing.volume.integrate_from_flow is true")

        seg = self.processing.segmentation
        _SEGMENTATION_METHODS = (
            "flow", "volume", "whole_file", "separators", "fixed_windows", "emg_burst")
        if seg.method not in _SEGMENTATION_METHODS:
            raise SettingsError(
                "processing.segmentation.method must be 'flow', 'volume', 'whole_file', "
                "'separators', 'fixed_windows' or 'emg_burst'")
        if seg.method in ("flow", "volume") and "flow" not in declared:
            raise SettingsError(
                f"processing.segmentation.method '{seg.method}' requires 'flow' in "
                "analysis.signals")
        if (seg.method in ("whole_file", "separators", "fixed_windows", "emg_burst")
                and "flow" in declared):
            raise SettingsError(
                f"processing.segmentation.method '{seg.method}' is for an EMG-only signal set")
        if not isinstance(seg.buffer, int):
            raise SettingsError("processing.segmentation.buffer must be an integer")
        if not 0.0 < seg.boundary_notice_min_relative_duration <= 1.0:
            raise SettingsError(
                "processing.segmentation.boundary_notice_min_relative_duration must be "
                "between 0 (exclusive) and 1 (inclusive)")
        if not isinstance(seg.boundary_notice_min_other_breaths, int) or \
                seg.boundary_notice_min_other_breaths < 1:
            raise SettingsError(
                "processing.segmentation.boundary_notice_min_other_breaths must be a "
                "positive integer")

        # SeparatorEntry (EMG-only "separators" segmentation): form first (every time
        # non-negative and strictly increasing along the recording), THEN the one
        # cross-entry conflict (two entries for the same file) -- same two-pass shape
        # as the breath_types checks below, for the same reason: a malformed entry is
        # reported on its own terms rather than tripping a conflict check that assumes
        # well-formed data.
        for i, se in enumerate(seg.separators):
            if not isinstance(se, SeparatorEntry):
                raise SettingsError(
                    f"processing.segmentation.separators[{i}] must be a table with "
                    "file and times_s")
            times = se.times_s
            # math.isfinite rejects NaN/Inf too -- both ARE instances of float (so the
            # type check alone lets them through) but neither compares meaningfully
            # against 0 or a neighbour (NaN compares False to everything; Inf reads as
            # "increasing" relative to any finite predecessor), so without this a
            # malformed TOML `times_s = [1.0, nan, 3.0]` (TOML has nan/inf literals)
            # silently passed validate() and only surfaced later as a bare, untranslated
            # ValueError/OverflowError from int(round(t * fs)) in core.analysis.segments.
            if not isinstance(times, list) or any(
                    isinstance(t, bool) or not isinstance(t, (int, float))
                    or not math.isfinite(t) for t in times):
                raise SettingsError(
                    f"processing.segmentation.separators[{i}].times_s must be a list "
                    "of numbers")
            if any(t < 0 for t in times) or any(b <= a for a, b in zip(times, times[1:])):
                raise SettingsError(
                    f"processing.segmentation.separators[{i}].times_s must be "
                    "non-negative and strictly increasing")
        seen_separator_files: set[str] = set()
        for se in seg.separators:
            if se.file in seen_separator_files:
                raise SettingsError(
                    f"processing.segmentation.separators: {se.file} has more than one "
                    "entry")
            seen_separator_files.add(se.file)

        # SegmentationOverrideEntry (flow-/volume-bearing segmentation repair):
        # same two-pass shape (form, then the one cross-entry conflict) as separators
        # above, checked independently for `cut_s` and `join_s` -- each is its own
        # sorted list, not one combined timeline the way SeparatorEntry.times_s is.
        for i, oe in enumerate(seg.overrides):
            if not isinstance(oe, SegmentationOverrideEntry):
                raise SettingsError(
                    f"processing.segmentation.overrides[{i}] must be a table with "
                    "file, cut_s and join_s")
            for field_name in ("cut_s", "join_s"):
                times = getattr(oe, field_name)
                if not isinstance(times, list) or any(
                        isinstance(t, bool) or not isinstance(t, (int, float))
                        or not math.isfinite(t) for t in times):
                    raise SettingsError(
                        f"processing.segmentation.overrides[{i}].{field_name} must be "
                        "a list of numbers")
                if any(t < 0 for t in times) or any(b <= a for a, b in zip(times, times[1:])):
                    raise SettingsError(
                        f"processing.segmentation.overrides[{i}].{field_name} must be "
                        "non-negative and strictly increasing")
        seen_override_files: set[str] = set()
        for oe in seg.overrides:
            if oe.file in seen_override_files:
                raise SettingsError(
                    f"processing.segmentation.overrides: {oe.file} has more than one "
                    "entry")
            seen_override_files.add(oe.file)

        if self.processing.wob.calc_from not in ("average", "individual"):
            raise SettingsError("processing.wob.calc_from must be 'average' or 'individual'")
        if not isinstance(self.processing.wob.avg_resampling_obs, int):
            raise SettingsError("processing.wob.avg_resampling_obs must be an integer")

        if self.processing.volume.trend_method not in (
                "linear", "nearest", "nearest-up", "zero", "slinear",
                "quadratic", "cubic", "previous", "next"):
            raise SettingsError(
                "processing.volume.trend_method must be a valid scipy interp1d kind")

        # Mirrors the runtime guard in core.pipeline.run_batch, so both the CLI
        # ('respmech validate'/'run') and the GUI (Preview's _settings_ok, which
        # greys out Run on failure) catch a misconfigured auto-detect BEFORE a batch
        # starts, instead of only via run_batch's own raise mid-run.
        emg = self.processing.emg
        if emg.ecg_auto_detect:
            if not emg.remove_ecg:
                raise SettingsError(
                    "processing.emg.ecg_auto_detect requires processing.emg.remove_ecg "
                    "to be enabled")
            if not ch.emg:
                raise SettingsError(
                    "processing.emg.ecg_auto_detect requires input.channels.emg "
                    "to be configured")

        # detect_channel is a 0-based INDEX into input.channels.emg (never a column
        # number), so a value outside that list crashes core.pipeline mid-batch with a
        # raw IndexError (K-222). The GUI already clamps it in Preview & QC (see
        # ui/screens/preview/_ecg.py's channel_collision path); a hand-written or
        # migrated settings file only ever reaches this validate() call, so the same
        # bound is enforced here regardless of caller. An empty emg list is allowed:
        # detect_channel only matters once remove_ecg (or an EMG-figure job) actually
        # reads it, and the emg-required checks above already cover ecg_auto_detect.
        if ch.emg and not (0 <= emg.detect_channel < len(ch.emg)):
            raise SettingsError(
                f"processing.emg.detect_channel must be a 0-based index into "
                f"input.channels.emg (0..{len(ch.emg) - 1}), got {emg.detect_channel}")

        # K-225: only the GUI's noise_enabled checkbox requires 'Remove ECG' first
        # (screen.py's self.noise_enabled.setEnabled(has_emg and ecg_on)) — a
        # hand-written or migrated settings.toml with noise.enabled = true and
        # remove_ecg = false runs the profile against a signal that still contains
        # heartbeats, modelling the cardiac artefact as steady background noise.
        # Enforced here so the CLI and the GUI refuse the same configurations.
        # BREAKING: rejects any existing analysis file that already combines these
        # two flags — flagged in CHANGELOG.md.
        if emg.noise.enabled and not emg.remove_ecg:
            raise SettingsError(
                "processing.emg.noise.enabled requires processing.emg.remove_ecg "
                "to be enabled (noise reduction would otherwise treat the heartbeat "
                "as steady background noise)")

        # reference_mode's own shape: a closed enum, and 'rest_segments'/'interburst' are
        # reachable only explicitly and only for an EMG-only signal set -- 'auto' already
        # covers a flow-bearing analysis fully (resolve_noise_reference_mode's own docstring).
        if emg.noise.reference_mode not in ("auto", "rest_segments", "interburst"):
            raise SettingsError(
                "processing.emg.noise.reference_mode must be 'auto', 'rest_segments' "
                "or 'interburst'")
        if emg.noise.reference_mode != "auto" and "flow" in declared:
            raise SettingsError(
                f"processing.emg.noise.reference_mode={emg.noise.reference_mode!r} is "
                "only valid for an EMG-only signal set")

        # EMG-only noise reduction now HAS a reference-resolution rule
        # (resolve_noise_reference_mode, M-22) -- replacing the old blanket "not yet
        # supported" guard below with two narrower ones for what still is not built:
        #  1. no usable reference at all ('unresolved') -- the same "fail before a
        #     single file is processed" guarantee the old guard gave, just narrower
        #     now that a real usable case (a rest-typed reference segment, or explicit
        #     reference_intervals) exists;
        #  2. 'interburst' (no clip-building implementation yet -- a later ticket);
        #  3. 'auto_prop' (choose prop_decrease from active/quiet EMG pooled across the
        #     whole test) has no EMG-only implementation either -- it is built on
        #     inspiration/expiration PHASES (core.pipeline._emg_segmented, flow-only),
        #     and a genuine EMG-only active/quiet split (rest-typed segments against
        #     the rest, or bursts) is later tickets' scope. Left unguarded here, an
        #     EMG-only analysis with noise reduction on (auto_prop defaults to True)
        #     would reach _build_noise_set's flow-only gather loop and crash the WHOLE
        #     BATCH with an unguarded TrimError -- exactly the failure mode the old
        #     blanket guard existed to prevent, now reachable again for a case this
        #     ticket newly permits unless it is closed here too.
        if emg.noise.enabled:
            mode = resolve_noise_reference_mode(self)
            if mode == "unresolved":
                raise SettingsError(
                    "processing.emg.noise: no usable rest reference for an EMG-only "
                    "signal set")
            if mode == "interburst":
                raise SettingsError(
                    "processing.emg.noise.reference_mode='interburst' is not yet "
                    "implemented")
            if "flow" not in declared and "emg" in declared and emg.noise.auto_prop:
                raise SettingsError(
                    "processing.emg.noise.auto_prop is not yet supported for an "
                    "EMG-only signal set -- set processing.emg.noise.prop_decrease "
                    "manually and turn auto_prop off")

        v = self.processing.volume
        if v.correct_trend:
            if not 0.0 < v.trend_peak_min_prominence_frac < 1.0:
                raise SettingsError("processing.volume.trend_peak_min_prominence_frac "
                                    "must be between 0 and 1 (exclusive)")
            # 0 is legal and meaningful: find_peaks(height=0) on max(vol) - vol keeps
            # every trough, i.e. "no absolute gate". A v1 analysis may carry it, and it
            # is NOT equivalent to omitting the key (which selects the scale-free rule and
            # gives a different envelope), so it must keep running.
            if v.trend_peak_min_height is not None and v.trend_peak_min_height < 0:
                raise SettingsError("processing.volume.trend_peak_min_height cannot be "
                                    "negative (omit it to scale to each recording)")
            # The trough spacing is consumed at the ANALYSIS rate, which the pre-analysis
            # resample replaces after validate() runs — check the rate the code will
            # actually see, not the file's.
            samp = self.processing.sampling
            fs_eff = (samp.resample_to_frequency
                      if samp.resample and samp.resample_to_frequency > 0
                      else f.sampling_frequency)
            if v.trend_peak_min_distance_s * fs_eff < 1:
                raise SettingsError(
                    "processing.volume.trend_peak_min_distance_s must be at least one "
                    f"sample at the analysis rate ({fs_eff} Hz)")

        # M-19: a typed breath is a per-breath override, so it needs the same kind of
        # form/conflict checks exclude_breaths/breath_counts never got (nothing else in
        # this method touches either of those today). Form first (kind/breath shape),
        # THEN cross-entry conflicts, so a malformed entry is reported on its own terms
        # rather than tripping a conflict check that assumes well-formed data.
        for i, bt in enumerate(self.processing.breath_types):
            # `_coerce` only builds a BreathTypeEntry from a dict-shaped TOML table
            # element; a malformed one (e.g. a hand-edited `breath_types = [1, 2]`, a
            # bare list of numbers instead of tables) passes the raw value through
            # unchanged, and this is the FIRST code to actually read `.kind`/`.breath`
            # off it -- self-review finding: without this guard that raises a raw
            # AttributeError from validate(), not a SettingsError, which every caller of
            # validate() assumes is the only exception it can raise. A hand-edited TOML
            # is the only way to write this table at all today (no UI yet), so this is a
            # front-line failure mode, not a hypothetical one.
            if not isinstance(bt, BreathTypeEntry):
                raise SettingsError(
                    f"processing.breath_types[{i}] must be a table with file, breath "
                    "and kind")
            if bt.kind not in BREATH_KINDS:
                raise SettingsError(
                    f"processing.breath_types[{i}].kind must be one of "
                    + ", ".join(repr(k) for k in BREATH_KINDS))
            if isinstance(bt.breath, bool) or not isinstance(bt.breath, int) or bt.breath < 1:
                raise SettingsError(
                    f"processing.breath_types[{i}].breath must be a positive integer")

        seen_typed: set[tuple[str, int]] = set()
        for bt in self.processing.breath_types:
            key = (bt.file, bt.breath)
            if key in seen_typed:
                raise SettingsError(
                    f"processing.breath_types: breath {bt.breath} of {bt.file} is typed "
                    "more than once")
            seen_typed.add(key)

        excluded_by_file: dict[str, set[int]] = {}
        for e in self.processing.exclude_breaths:
            excluded_by_file.setdefault(e.file, set()).update(e.breaths)
        for bt in self.processing.breath_types:
            if bt.breath in excluded_by_file.get(bt.file, ()):
                raise SettingsError(
                    f"processing.breath_types: breath {bt.breath} of {bt.file} is both "
                    "typed and excluded")

        # whole_file segmentation (M-21, EMG-only) produces exactly one segment, always
        # numbered 1 -- a typed breath number above that can never correspond to
        # anything, regardless of which file it names.
        if seg.method == "whole_file":
            for bt in self.processing.breath_types:
                if bt.breath > 1:
                    raise SettingsError(
                        f"processing.breath_types: breath {bt.breath} > 1 is impossible "
                        "under whole_file segmentation")

        # M-29 IcSettings: self-review finding -- `min_preceding_breaths < 1` would let
        # `_ic_eelv_pre`'s mean-branch run over ZERO preceding breaths (an empty-array
        # mean is NaN, silently poisoning `vol_ic`), and an `aggregate` typo (e.g. a
        # hand-edited "medain") would silently fall through to the "mean" branch in
        # `apply_repeatability` instead of erroring -- both front-line failure modes
        # for a hand-edited TOML (no UI writes this table yet), same reasoning as the
        # breath_types guards above.
        ic = self.processing.lung_volume.ic
        if ic.min_preceding_breaths < 1:
            raise SettingsError(
                "processing.lung_volume.ic.min_preceding_breaths must be at least 1")
        if ic.preceding_breaths < ic.min_preceding_breaths:
            raise SettingsError(
                "processing.lung_volume.ic.preceding_breaths must be >= min_preceding_breaths")
        if ic.aggregate not in ("mean", "median"):
            raise SettingsError(
                'processing.lung_volume.ic.aggregate must be "mean" or "median"')
        for name in ("eelv_tolerance_frac", "plateau_flow_lps", "min_plateau_s",
                    "repeatability_frac", "low_effort_frac"):
            if getattr(ic, name) < 0:
                raise SettingsError(f"processing.lung_volume.ic.{name} must not be negative")
        if ic.eelv_tracking not in ("none", "within_file"):
            raise SettingsError(
                'processing.lung_volume.ic.eelv_tracking must be "none" or "within_file"')

        # M-42 MfvlSettings: same front-line-failure-mode reasoning as IcSettings
        # above (a hand-edited TOML with no UI writing this table yet).
        mfvl = self.processing.mfvl
        if mfvl.source not in ("single", "envelope"):
            raise SettingsError('processing.mfvl.source must be "single" or "envelope"')
        for name in ("efl_rel_tol", "efl_abs_tol_lps", "efl_present_min_pct",
                    "mvv_fev1_multiplier"):
            if getattr(mfvl, name) < 0:
                raise SettingsError(f"processing.mfvl.{name} must not be negative")

        # PeepiSettings: same front-line-failure-mode reasoning as MfvlSettings above.
        peepi = self.processing.pressure.peepi
        for name in ("search_window_s", "smooth_s", "onset_slope_frac", "min_deflection"):
            v = getattr(peepi, name)
            if not math.isfinite(v):
                raise SettingsError(f"processing.pressure.peepi.{name} must be a finite number")
            if v < 0:
                raise SettingsError(
                    f"processing.pressure.peepi.{name} must not be negative")
        if peepi.search_window_s <= 0:
            raise SettingsError("processing.pressure.peepi.search_window_s must be positive")
        if not 0 < peepi.onset_slope_frac <= 1:
            raise SettingsError(
                "processing.pressure.peepi.onset_slope_frac must be above 0 and at most 1")

        # references/reference_defaults/subjects -- FORM only (validate() never resolves
        # a link or touches a file on disk; that is core.analysis.references' job). Same
        # two-pass shape as breath_types/separators above: a malformed entry (hand-edited
        # TOML; no UI writes these tables yet) is reported on its own terms first, THEN
        # the one cross-entry conflict that assumes well-formed data.
        for i, r in enumerate(self.processing.references):
            if not isinstance(r, ReferenceEntry):
                raise SettingsError(
                    f"processing.references[{i}] must be a table with file")
        seen_ref_files: set[str] = set()
        for r in self.processing.references:
            if r.file in seen_ref_files:
                raise SettingsError(
                    f"processing.references: {r.file} appears more than once")
            seen_ref_files.add(r.file)

        for i, g in enumerate(self.processing.reference_defaults):
            if not isinstance(g, GroupReferenceEntry):
                raise SettingsError(
                    f"processing.reference_defaults[{i}] must be a table with group")
        seen_ref_groups: set[str] = set()
        for g in self.processing.reference_defaults:
            if g.group in seen_ref_groups:
                raise SettingsError(
                    f"processing.reference_defaults: group {g.group} appears more than once")
            seen_ref_groups.add(g.group)

        for i, subj in enumerate(self.input.subjects):
            if not isinstance(subj, SubjectEntry):
                raise SettingsError(f"input.subjects[{i}] must be a table with key")
        seen_subject_keys: set[str] = set()
        for i, subj in enumerate(self.input.subjects):
            if subj.key in seen_subject_keys:
                raise SettingsError(f"input.subjects: key {subj.key} must be unique")
            seen_subject_keys.add(subj.key)
            if subj.tlc_l is not None and not (0.0 <= subj.tlc_l <= 15.0):
                raise SettingsError(
                    f"input.subjects[{i}].tlc_l must be between 0 and 15 L")
            if (subj.rv_l is not None and subj.tlc_l is not None
                    and not (subj.rv_l < subj.tlc_l)):
                raise SettingsError(f"input.subjects[{i}].rv_l must be below tlc_l")

        return self


def _reference_file_has_rest_segment(settings: "Settings", reference_file: str | None) -> bool:
    """True if ``reference_file`` carries at least one segment typed ``'rest'`` in
    ``processing.breath_types`` -- a pure settings-level lookup (the same flat,
    per-file/per-breath-number entries a flow-bearing analysis's typed breaths already
    use, see :class:`BreathTypeEntry`). For an EMG-only signal set "breath number" IS
    the segment number ``whole_file``/``separators`` assign, so this needs no knowledge
    of the file's actual samples or segmentation to answer -- unlike building the clip
    itself (:func:`respmech.core.pipeline._reference_noise_clip`'s ``rest_segments``
    branch), which does have to load and segment the file."""
    if not reference_file:
        return False
    return any(bt.file == reference_file and bt.kind == "rest"
              for bt in settings.processing.breath_types)


def resolve_noise_reference_mode(settings: "Settings") -> str:
    """Which source :func:`respmech.core.pipeline._reference_noise_clip` builds the
    shared EMG noise-reduction profile from, for THIS settings object right now.

    Returns one of ``'expiration'`` | ``'intervals'`` | ``'rest_segments'`` |
    ``'interburst'`` | ``'unresolved'``. Pure and Qt-free: reasons only about the
    ``NoiseSettings``/``breath_types``/declared-signal-set shape, never touches a file
    on disk (that is pipeline's job, once a mode is resolved).

    ``processing.emg.noise.reference_mode`` is ``'auto'`` by default, which is where
    almost every analysis stays:

    * With a flow channel declared, ``auto`` reproduces EXACTLY the rule this codebase
      has always used (``use_expiration or not reference_intervals`` -- see
      ``_reference_noise_clip``'s docstring/history): ``'expiration'`` when true,
      ``'intervals'`` otherwise. ``use_expiration`` is simply ignored (never read,
      never rejected) once no flow channel is declared -- an EMG-only analysis has no
      inspiration/expiration phases for it to mean anything about.
    * Without a flow channel (an EMG-only signal set), ``auto`` looks instead at
      whether the reference file has a segment typed ``'rest'``
      (:func:`_reference_file_has_rest_segment`): ``'rest_segments'`` if so,
      ``'intervals'`` if explicit ``reference_intervals`` are set instead, otherwise
      ``'unresolved'`` -- the state ``Settings.validate()`` rejects with a clear
      message when noise reduction is enabled, rather than letting an un-buildable
      profile reach the pipeline.

    An explicit, non-``'auto'`` ``reference_mode`` (``'rest_segments'``/
    ``'interburst'``) is returned as-is, unresolved further -- these are only ever
    meaningful for an EMG-only set, and ``Settings.validate()`` is what actually
    enforces that (rejecting either one while a flow channel is declared) and rejects
    ``'interburst'`` outright while noise reduction is enabled (no clip-building
    implementation exists for it yet -- a later ticket's scope). This function itself
    never raises: it classifies whatever settings it is handed, the same way
    ``core.analysis.signals._mode_for`` classifies a signal set without validating it.
    """
    noise = settings.processing.emg.noise
    mode = noise.reference_mode
    if mode != "auto":
        return mode
    has_flow = "flow" in effective_signals(settings)
    if has_flow:
        return "expiration" if (noise.use_expiration or not noise.reference_intervals) else "intervals"
    if _reference_file_has_rest_segment(settings, noise.reference_file):
        return "rest_segments"
    if noise.reference_intervals:
        return "intervals"
    return "unresolved"


def resolve_noise_reference_mode_or_none(settings: "Settings") -> str | None:
    """``resolve_noise_reference_mode``, tolerant of a malformed ``analysis.signals`` --
    the same defensive pairing as :meth:`respmech.core.analysis.signals.Capabilities.
    from_settings_or_none`, for the identical reason (M-24): several UI render paths call
    the resolver on every edit/open -- including ``MainWindow``'s own construction on the
    command-line/drag-drop open path, which has no surrounding try/except at all -- so
    letting a hand-edited bare-string ``signals = "flow"`` raise there (via
    ``effective_signals``) crashes the whole window instead of leaving the report to
    ``Settings.validate()``. Returns ``None`` (never a real mode string) to mean "nothing
    safe to resolve"; callers degrade their render accordingly (the same way a caller of
    ``from_settings_or_none`` treats a ``None`` Capabilities)."""
    try:
        return resolve_noise_reference_mode(settings)
    except TypeError:
        return None


# --- carried-over per-folder state ------------------------------------------
#
# exclude_breaths/breath_counts key on the bare filename, and the noise reference is a
# filename too — neither carries which recordings folder it was set against. Point the
# same analysis at a DIFFERENT folder that happens to share a filename (the common
# multi-subject workflow: one LabChart export named the same in every subject folder) and
# the old entries silently keep applying, because nothing about the key changed. The
# folder fields above let the UI tell the two situations apart; these pure, Qt-free
# helpers are the one place that decides "does this recorded folder still match" so the
# comparison can't drift between the banner, the Preview overlay and the file rail.

def _norm_folder(p: str | None) -> str | None:
    # normcase folds drive-letter/separator case on Windows (a no-op on case-sensitive
    # macOS/Linux — see ui.prefs.set_last_folder's recent-analyses dedup for the same
    # pattern already established elsewhere in this codebase); without it, a folder
    # re-typed or re-browsed with different case than how it was first recorded (the same
    # real folder, unchanged) would falsely compare as carried-over on a case-insensitive
    # filesystem.
    return os.path.normcase(os.path.normpath(p)) if p else None


def is_carried_folder(entry_folder: str | None, current_folder: str | None) -> bool:
    """True if ``entry_folder`` does not demonstrably match ``current_folder``.

    An unrecorded folder (None on either side — an older analysis, or no input folder
    chosen yet) can never be PROVEN current, so it counts as carried too rather than being
    silently trusted. This is a deliberate choice, not a migration gap to fill in later:
    guessing a folder for old data would risk the opposite mistake (a real cross-folder
    exclusion applied without ever being shown), so an unrecorded folder is always the
    cautious answer. See Settings.processing.exclude_breaths / .breath_counts and
    ProcessingSettings.emg.noise.reference_folder.
    """
    a, b = _norm_folder(entry_folder), _norm_folder(current_folder)
    return not (a and b and a == b)


def _walk(settings: "Settings", dotted_path: str) -> tuple[Any, str]:
    """Resolve all but the LAST segment of ``dotted_path`` against the live settings
    object, returning ``(container, last_segment)`` so a caller can ``getattr``/
    ``setattr`` generically. E.g. ``"processing.emg.noise.reference_folder"`` ->
    ``(settings.processing.emg.noise, "reference_folder")``; ``"processing.exclude_breaths"``
    -> ``(settings.processing, "exclude_breaths")``. Shared by every consumer of
    :data:`_CARRIED_KINDS` below, and re-exported (unchanged) for
    ``settingsio.toml_io``'s own rebase/relativize, so the two can never disagree about
    what a row's path means."""
    obj: Any = settings
    segments = dotted_path.split(".")
    for seg in segments[:-1]:
        obj = getattr(obj, seg)
    return obj, segments[-1]


def _clear_noise_reference(noise: "NoiseSettings") -> None:
    noise.reference_file = None
    noise.reference_intervals = []


def _clear_ecg_reference(emg: "EmgSettings") -> None:
    emg.ecg_reference_file = None


def _clear_normalization_reference(emg: "EmgSettings") -> None:
    emg.normalization_reference_file = None


# Each row names ONE piece of per-folder-tagged batch state, generalizing what used to be
# six hand-spread call sites (CarriedOverState, carried_over_state, clear_carried_over,
# toml_io._rebase_folders, toml_io.save_toml, plus the Setup banner/rail/overlay reading
# their result) into one table those call sites all iterate — a future tagged kind
# (M-19 breath_types, M-21 separators, M-34 references/reference_defaults/subjects) adds
# ONE row here instead of touching all six again (the exact B06 recurrence this ticket
# closes off).
#
#   path      -- a dotted path (see _walk) that resolves to EITHER a LIST of entries each
#                carrying their own ``.folder`` (exclude_breaths, breath_counts) OR
#                DIRECTLY to a scalar ``*_folder`` field (noise/ecg/normalization
#                references). Which one it is is decided at runtime by the resolved
#                value's type (a list, vs a str/None) — never declared twice.
#   kind      -- the CarriedOverState dict key / rail-badge "kind name".
#   name_of   -- list mode: called with ONE ENTRY, returns its filename if it names
#                anything worth reporting (e.g. ExcludeEntry needs breaths != []), else
#                None. Scalar mode: called with the CONTAINER holding the folder field
#                (e.g. NoiseSettings, or EmgSettings for the two EMG references), so it
#                can look at a SIBLING field (reference_file, ecg_reference_file, …);
#                returns that sibling's value, or None/"" if nothing is set.
#   clear_fn  -- scalar mode only (None for list mode, where clearing means dropping the
#                whole entry instead): resets the kind's OWN sibling field(s) in place.
#                The folder field itself (`attr`) is cleared generically by the caller.
_CARRIED_KINDS: tuple[tuple[str, str, Callable[[Any], Any], Callable[[Any], None] | None], ...] = (
    ("processing.exclude_breaths", "exclude_files",
     lambda e: e.file if e.breaths else None, None),
    ("processing.breath_counts", "breath_count_files",
     lambda e: e.file, None),
    ("processing.emg.noise.reference_folder", "noise_reference",
     # Reproduce the OLD presence check `(reference_file or reference_intervals)` exactly —
     # NOT `reference_file` alone. `name_of`'s return value is tested for truthiness by
     # both callers below, so a bare `n.reference_file if (n.reference_file or
     # n.reference_intervals) else None` would return the (falsy) reference_file itself
     # when only `reference_intervals` is set, silently losing presence — a real
     # divergence a self-review pass caught by comparing against the pre-generalization
     # behaviour directly. The label falls back to a generic word only in that
     # intervals-only case, which every live write path (preview/_emg_noise.py's two
     # apply methods) always avoids by setting both fields together.
     lambda n: (n.reference_file or "reference") if (n.reference_file or n.reference_intervals)
     else None,
     _clear_noise_reference),
    ("processing.emg.ecg_reference_folder", "ecg_reference",
     lambda emg: emg.ecg_reference_file, _clear_ecg_reference),
    ("processing.emg.normalization_reference_folder", "normalization_reference",
     lambda emg: emg.normalization_reference_file, _clear_normalization_reference),
    # M-19: a typed breath always names a kind once the entry exists at all (kind is a
    # required field, never optional the way ExcludeEntry.breaths can be empty), so
    # unlike exclude_files' `if e.breaths` guard, every entry here is worth reporting.
    ("processing.breath_types", "breath_type_files",
     lambda e: e.file, None),
    # A separator entry always names a file worth reporting once it exists at all --
    # even zero times_s is a deliberate, explicit "one segment" choice via this
    # method (unlike exclude_files' `if e.breaths` guard for an entry that could
    # otherwise be a no-op left behind by the UI).
    ("processing.segmentation.separators", "separator_files",
     lambda e: e.file, None),
    # An override entry with both lists still empty is a no-op left behind by the
    # UI (same guard as exclude_files' `if e.breaths`) — an entry only names its file
    # once it actually adjusts something.
    ("processing.segmentation.overrides", "segmentation_override_files",
     lambda e: e.file if (e.cut_s or e.join_s) else None, None),
    # a reference entry always names a file worth reporting once it exists at
    # all, same reasoning as breath_types/separators above (an entry with every slot
    # still unset is still a deliberate placeholder the user created, not a no-op).
    ("processing.references", "reference_files",
     lambda e: e.file, None),
    ("processing.reference_defaults", "group_reference_groups",
     lambda e: e.group, None),
    ("input.subjects", "subject_keys",
     lambda e: e.key, None),
)


@dataclass
class CarriedOverState:
    """What in ``settings.processing`` still names a DIFFERENT (or unrecorded) recordings
    folder than ``settings.input.folder`` right now. Built by :func:`carried_over_state`;
    consumed by the Setup banner, Preview & QC's overlay/QC line, and the file rail badge —
    every one of them must agree on the same set, so they all go through this.

    Backed by one dict keyed on `_CARRIED_KINDS`' ``kind`` names (in table order), never
    constructed with positional args outside this module. ``exclude_files``/
    ``breath_count_files``/``noise_reference`` stay properties with their original shapes
    (a list of filenames; a bool) so every EXISTING caller is unaffected; new kinds
    (``ecg_reference``, ``normalization_reference``) follow ``noise_reference``'s bool
    shape, since like it they name a single batch-wide reference, not a per-file list."""
    _by_kind: dict[str, list[str]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return any(self._by_kind.values())

    @property
    def exclude_files(self) -> list[str]:
        return self._by_kind.get("exclude_files", [])

    @property
    def breath_count_files(self) -> list[str]:
        return self._by_kind.get("breath_count_files", [])

    @property
    def noise_reference(self) -> bool:
        return bool(self._by_kind.get("noise_reference"))

    @property
    def ecg_reference(self) -> bool:
        return bool(self._by_kind.get("ecg_reference"))

    @property
    def normalization_reference(self) -> bool:
        return bool(self._by_kind.get("normalization_reference"))

    @property
    def breath_type_files(self) -> list[str]:
        return self._by_kind.get("breath_type_files", [])

    @property
    def separator_files(self) -> list[str]:
        return self._by_kind.get("separator_files", [])

    @property
    def segmentation_override_files(self) -> list[str]:
        return self._by_kind.get("segmentation_override_files", [])

    @property
    def reference_files(self) -> list[str]:
        return self._by_kind.get("reference_files", [])

    @property
    def group_reference_groups(self) -> list[str]:
        return self._by_kind.get("group_reference_groups", [])

    @property
    def subject_keys(self) -> list[str]:
        return self._by_kind.get("subject_keys", [])

    def kinds_present(self) -> list[tuple[str, list[str]]]:
        """``(kind, names)`` for every kind that IS carried, in `_CARRIED_KINDS` table
        order. The one thing a caller that wants to name EVERY carried kind (the Setup
        banner) needs, without hard-coding each kind's own attribute name — a future row
        added to the table reaches such a caller by adding its own phrase there, not by
        re-touching an if/elif chain."""
        return list(self._by_kind.items())


def carried_over_state(settings: "Settings") -> CarriedOverState:
    """Nothing can be "carried over" relative to an input folder that isn't set yet (a
    fresh/guided analysis with no folder chosen) — every entry would trivially mismatch an
    empty string, which would just flag stale state that was never applied against
    anything. Return the empty (falsy) state in that case."""
    current = settings.input.folder
    if not current:
        return CarriedOverState()
    by_kind: dict[str, list[str]] = {}
    for path, kind, name_of, _clear_fn in _CARRIED_KINDS:
        container, attr = _walk(settings, path)
        val = getattr(container, attr)
        if isinstance(val, list):
            names = sorted({n for e in val if (n := name_of(e)) and is_carried_folder(e.folder, current)})
        else:
            name = name_of(container)
            names = [name] if name and is_carried_folder(val, current) else []
        if names:
            by_kind[kind] = names
    return CarriedOverState(by_kind)


def clear_carried_over(settings: "Settings") -> None:
    """The banner's "Clear" action: drop exclude_breaths/breath_counts entries, and every
    scalar reference (noise/ECG/normalisation), whose recorded folder does not match
    ``settings.input.folder`` right now. Mutates in place. Entries/references that already
    match the current folder — the ordinary case, nothing changed — are left completely
    untouched; this only ever removes state :func:`carried_over_state` would also flag."""
    current = settings.input.folder
    for path, _kind, name_of, clear_fn in _CARRIED_KINDS:
        container, attr = _walk(settings, path)
        val = getattr(container, attr)
        if isinstance(val, list):
            setattr(container, attr,
                     [e for e in val if not (name_of(e) and is_carried_folder(e.folder, current))])
        elif name_of(container) and is_carried_folder(val, current):
            clear_fn(container)
            setattr(container, attr, None)


# --- signal-set reconciliation ----------------------------------------------

def _reconcile_signals(obj: "Settings") -> list[str]:
    """Reconcile a channel assigned but not named in an EXPLICIT ``analysis.signals``
    list -- the only situation a hand-edited (or older-tool-written) TOML file can create,
    since every in-app write path goes through ONE tragte (``settings_screen.
    apply_signal_set``, a later ticket) that keeps the two in lockstep. Runs only in
    :meth:`Settings.from_dict`, never in any other write site: a file with no ``[analysis]``
    table at all (an empty list, "derive it") gets no notice, because there is nothing to
    reconcile against -- the derived set already equals whatever is assigned, by
    definition. Replaces ``obj.analysis.signals`` with a NEW list carrying any added
    roles (never reordering or removing an existing entry) and returns one
    plain-English note per role added, for ``Settings.notices``.
    """
    notices: list[str] = []
    signals = obj.analysis.signals
    # Defensive, mirroring core.analysis.signals.effective_signals' own tolerance: a
    # malformed value (anything but a real list -- a hand-edited `signals = "flow"` or
    # `signals = 5`, or a dict-based caller passing `signals = None`) is left for
    # validate()/effective_signals to report properly (a clear TypeError for the
    # bare-string case; None reads as "derive it"), never crashed on here with a raw
    # AttributeError/TypeError from .append()/set() before that reporting ever runs.
    if not isinstance(signals, list) or not signals:
        return notices
    # Copy rather than mutate the incoming list in place: `_build`/`_coerce` do not
    # copy a plain (non-dataclass) list field, so `signals` may still be the exact
    # object a caller's own dict handed to `from_dict` -- appending to it in place
    # would silently grow THAT object too if it were ever reused across more than one
    # `from_dict()` call (nothing in this repo does that today, but nothing should
    # have to rely on it staying that way either).
    signals = list(signals)
    have = set(signals)
    ch = obj.input.channels
    assigned = [role for role in SINGLE_SIGNALS if getattr(ch, role) is not None]
    if ch.emg:
        assigned.append("emg")
    for role in assigned:
        if role not in have:
            signals.append(role)
            have.add(role)
            notices.append(
                f"input.channels.{role} is assigned but '{role}' was not in "
                "analysis.signals; it has been added")
    obj.analysis.signals = signals
    return notices


# --- schema upgrades -------------------------------------------------------

def _upgrade(obj: "Settings", raw: dict) -> list[str]:
    """Apply schema upgrades in place; return a plain-English note for each one.

    ``raw`` is the source dict, so an omitted ``schema_version`` (a hand-written
    analysis) is read as the CURRENT schema rather than as the oldest one — such a file
    also omits the retired key, so it lands on the new default either way.
    """
    notices: list[str] = []
    version = raw.get("schema_version", SCHEMA_VERSION)
    if not isinstance(version, int):
        return notices
    if version < 2:
        v = obj.processing.volume
        if v.trend_peak_min_height == RETIRED_TREND_PEAK_MIN_HEIGHT:
            v.trend_peak_min_height = None
            notices.append(
                "processing.volume.trend_peak_min_height was the retired default "
                f"({RETIRED_TREND_PEAK_MIN_HEIGHT}) — an absolute depth below the "
                "recording's highest volume, which matches no trough at all on ordinary "
                "tidal breathing. It is now unset, so end-expiratory troughs are detected "
                "relative to each recording's own volume range. Set it explicitly under "
                "Mechanics — Advanced… to reproduce an older analysis exactly.")
    obj.schema_version = SCHEMA_VERSION
    return notices


# --- generic dataclass <-> dict helpers ------------------------------------

def _unwrap_optional(t):
    """Return the non-None type of Optional[T]/Union[T, None], else t.

    A PEP 604 ``T | None`` annotation has origin ``types.UnionType``, not
    ``typing.Union`` -- ``get_origin(t) is Union`` alone misses it, so a nested
    ``ref: Ref | None = None`` field never reached the ``is_dataclass(typ)`` branch in
    ``_coerce`` below and ``_build`` silently returned the raw dict instead of a ``Ref``
    instance. ``typing.Optional[T]`` still normalises to ``typing.Union[T, None]`` at
    runtime, so both spellings are covered by the same check.
    """
    if get_origin(t) in (Union, types.UnionType):
        args = [a for a in get_args(t) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return t


def _build(cls, data: dict, unknown: dict, path: str):
    if not isinstance(data, dict):
        raise SettingsError(f"{path or '<root>'}: expected a table/dict, got {type(data).__name__}")
    hints = get_type_hints(cls)
    field_names = {f.name for f in fields(cls)}
    kwargs = {}
    for key, val in data.items():
        if key not in field_names or key == "unknown":
            unknown[f"{path}{key}"] = val
            continue
        kwargs[key] = _coerce(hints[key], val, unknown, f"{path}{key}.")
    return cls(**kwargs)


def _coerce(ftype, val, unknown, path):
    typ = _unwrap_optional(ftype)
    if is_dataclass(typ) and isinstance(val, dict):
        return _build(typ, val, unknown, path)
    if get_origin(typ) in (list, "list") and isinstance(val, list):
        args = get_args(typ)
        if args and is_dataclass(args[0]):
            return [_build(args[0], v, unknown, f"{path}[{i}].") if isinstance(v, dict) else v
                    for i, v in enumerate(val)]
    return val


def _to_dict(obj, drop=frozenset()):
    if is_dataclass(obj):
        out = {}
        for f in fields(obj):
            if f.name in drop:
                continue
            out[f.name] = _to_dict(getattr(obj, f.name))
        return out
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    return obj
