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
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints


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
class InputSettings:
    folder: str = "input"
    files: str = "*.*"
    format: InputFormat = field(default_factory=InputFormat)
    channels: Channels = field(default_factory=Channels)


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
class SegmentationSettings:
    method: str = "flow"                 # "flow" | "volume"
    buffer: int = 800
    peak: PeakSettings = field(default_factory=PeakSettings)


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
    outlier_rms_sd_limit: float = 0.0
    # EMG amplitude normalisation for the OUTPUT tables (P14): each file's RMS columns
    # are also reported as a % of a reference so amplitudes compare across
    # subjects/electrodes. "none" | "per_file_max" | "per_file_mean". This adds a
    # normalised sheet to the output; it never changes the raw computed RMS. By
    # default the reference is each file's OWN max/mean breath, per column — which
    # makes every file's peak reach 100% by construction and does NOT make amplitudes
    # comparable across files/subjects (documented on the website, ticket 5.1/K-155/
    # K-158). Set normalization_reference_file to a filename already in this batch
    # (relative to input.folder, e.g. a maximal inspiratory/expiratory manoeuvre
    # recorded once per subject) to normalise every file against THAT file's
    # max/mean instead of its own — see core.summary.reference_values_for_batch.
    normalization: str = "per_file_max"
    normalization_reference_file: str | None = None
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


@dataclass
class ProcessingSettings:
    sampling: SamplingSettings = field(default_factory=SamplingSettings)
    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)
    volume: VolumeSettings = field(default_factory=VolumeSettings)
    wob: WobSettings = field(default_factory=WobSettings)
    emg: EmgSettings = field(default_factory=EmgSettings)
    entropy: EntropySettings = field(default_factory=EntropySettings)
    ptp: PtpSettings = field(default_factory=PtpSettings)
    exclude_breaths: list[ExcludeEntry] = field(default_factory=list)
    breath_counts: list[BreathCountEntry] = field(default_factory=list)


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
        for name in ("flow", "poes", "pgas", "pdi"):
            if getattr(ch, name) is None:
                raise SettingsError(f"input.channels.{name} is required")
        if not self.processing.volume.integrate_from_flow and ch.volume is None:
            raise SettingsError(
                "input.channels.volume is required unless "
                "processing.volume.integrate_from_flow is true")

        seg = self.processing.segmentation
        if seg.method not in ("flow", "volume"):
            raise SettingsError("processing.segmentation.method must be 'flow' or 'volume'")
        if not isinstance(seg.buffer, int):
            raise SettingsError("processing.segmentation.buffer must be an integer")

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
        return self


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


@dataclass
class CarriedOverState:
    """What in ``settings.processing`` still names a DIFFERENT (or unrecorded) recordings
    folder than ``settings.input.folder`` right now. Built by :func:`carried_over_state`;
    consumed by the Setup banner, Preview & QC's overlay/QC line, and the file rail badge —
    every one of them must agree on the same set, so they all go through this."""
    exclude_files: list[str] = field(default_factory=list)
    breath_count_files: list[str] = field(default_factory=list)
    noise_reference: bool = False

    def __bool__(self) -> bool:
        return bool(self.exclude_files or self.breath_count_files or self.noise_reference)


def carried_over_state(settings: "Settings") -> CarriedOverState:
    """Nothing can be "carried over" relative to an input folder that isn't set yet (a
    fresh/guided analysis with no folder chosen) — every entry would trivially mismatch an
    empty string, which would just flag stale state that was never applied against
    anything. Return the empty (falsy) state in that case."""
    current = settings.input.folder
    if not current:
        return CarriedOverState()
    # an entry with an empty breaths list (every breath re-included — the state
    # _toggle_breath leaves an ExcludeEntry in before it removes it entirely) or a
    # breath-count override of nothing names no actual state to warn about.
    exclude_files = sorted({e.file for e in settings.processing.exclude_breaths
                            if e.breaths and is_carried_folder(e.folder, current)})
    breath_count_files = sorted({e.file for e in settings.processing.breath_counts
                                 if is_carried_folder(e.folder, current)})
    noise = settings.processing.emg.noise
    noise_reference = bool(
        (noise.reference_file or noise.reference_intervals)
        and is_carried_folder(noise.reference_folder, current))
    return CarriedOverState(exclude_files, breath_count_files, noise_reference)


def clear_carried_over(settings: "Settings") -> None:
    """The banner's "Clear" action: drop exclude_breaths/breath_counts entries, and the
    noise reference, whose recorded folder does not match ``settings.input.folder`` right
    now. Mutates in place. Entries that already match the current folder — the ordinary
    case, nothing changed — are left completely untouched; this only ever removes state
    :func:`carried_over_state` would also flag."""
    current = settings.input.folder
    settings.processing.exclude_breaths = [
        e for e in settings.processing.exclude_breaths
        if not (e.breaths and is_carried_folder(e.folder, current))]
    settings.processing.breath_counts = [
        e for e in settings.processing.breath_counts
        if not is_carried_folder(e.folder, current)]
    noise = settings.processing.emg.noise
    if is_carried_folder(noise.reference_folder, current):
        noise.reference_file = None
        noise.reference_intervals = []
        noise.reference_folder = None


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
    """Return the non-None type of Optional[T]/Union[T, None], else t."""
    if get_origin(t) is Union:
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
