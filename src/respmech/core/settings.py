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
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints


class SettingsError(ValueError):
    """Raised when settings are missing/invalid, with an actionable message."""


SCHEMA_VERSION = 1


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
    trend_peak_min_height: float = 0.8
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
class EmgSettings:
    rms_window_s: float = 0.050
    remove_ecg: bool = False
    detect_channel: int = 0
    ecg_min_height: float = 0.0005
    ecg_min_distance_s: float = 0.5
    ecg_min_width_s: float = 0.001
    ecg_window_s: float = 0.4
    remove_noise: bool = False
    outlier_rms_sd_limit: float = 0.0
    save_sound: bool = False
    plot_yscale: list[float] = field(default_factory=lambda: [-0.1, 0.1])
    # legacy filename-keyed noise-profile intervals (kept for migration):
    # [[file, source_file_or_empty, [t0,t1]], ...]
    noise_profile: list[Any] = field(default_factory=list)
    # new shared-profile noise reduction (canonical):
    noise: NoiseSettings = field(default_factory=NoiseSettings)


@dataclass
class EntropySettings:
    epochs: int = 2
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


@dataclass
class BreathCountEntry:
    file: str
    count: int


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
    include_ignored_breaths: bool = True


@dataclass
class DiagnosticsOutput:
    save_pv_average: bool = True
    save_pv_individual: bool = True
    pv_columns: int = 3
    pv_rows: int = 4
    save_raw: bool = True
    save_trimmed: bool = True
    save_drift: bool = True


@dataclass
class OutputSettings:
    folder: str = "output"
    data: DataOutput = field(default_factory=DataOutput)
    diagnostics: DiagnosticsOutput = field(default_factory=DiagnosticsOutput)


@dataclass
class Settings:
    schema_version: int = SCHEMA_VERSION
    input: InputSettings = field(default_factory=InputSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    # keys we did not recognise while parsing (kept for a warning, never fatal).
    unknown: dict = field(default_factory=dict)

    # -- (de)serialisation --------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        unknown: dict = {}
        obj = _build(cls, d or {}, unknown, path="")
        obj.unknown = unknown
        return obj

    def to_dict(self) -> dict:
        return _to_dict(self, drop={"unknown"})

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
        return self


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
