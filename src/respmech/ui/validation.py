"""Filesystem-level validation shared by the Settings and Run screens.

The core ``Settings.validate()`` is deliberately filesystem-agnostic; this adds the
path checks (folders exist, files match, noise reference present) that both the
Settings screen's live validation and the Run screen need, so the two stay consistent.
Qt-free.
"""
from __future__ import annotations

import os
import re

# NB ``match_input_files`` is imported lazily inside ``matching_files`` below. It needs
# nothing but os/fnmatch, but importing it at module level drags the whole compute core
# -- scipy.interpolate, pandas, scipy.signal -- into GUI startup, which cost 1.4 s of the
# 2.0 s it took to open a window. See tests/unit/test_startup_imports.py.

# Ticket D02: ``Settings.validate()``'s messages are written for a TOML file (dotted
# settings paths, e.g. ``input.channels.volume``), not a screen. Three call sites used to
# each catch ``SettingsError`` and show ``str(e)`` verbatim: ``blockers()`` below (read by
# both the Setup QC strip and the Run screen's commitment sheet, ticket B07), and Setup's
# own ``_validation_status()`` and ``_save_blocker()``. Translating once, here, means a
# fix covers all three instead of three call sites independently getting it wrong the
# same way. Literal messages first (exact match, cheapest and least likely to
# mis-translate); ``_FRIENDLY_PREFIXES`` for the few whose text carries a dynamic suffix
# (e.g. the analysis rate in Hz).
_FRIENDLY_SETTINGS_ERRORS = {
    "input.channels.volume is required unless processing.volume.integrate_from_flow is true":
        "Volume channel not assigned — assign one, or tick 'No volume channel — derive "
        "volume by integrating flow' in 'Assign channels from data…'",
    "input.format.sampling_frequency is required":
        "Sampling frequency is not set (Setup ▸ Input)",
    "input.format.sampling_frequency must be an integer":
        "Sampling frequency must be a whole number (Setup ▸ Input)",
    "input.format.matlab_variant must be 'windows' or 'mac'":
        "MATLAB file variant must be Windows or Mac (Setup ▸ Input)",
    "processing.segmentation.method must be 'flow' or 'volume'":
        "Breath-splitting signal must be Flow or Volume (Preview & QC ▸ Mechanics ▸ "
        "Advanced… ▸ Breath detection)",
    "processing.segmentation.buffer must be an integer":
        "Breath-separation buffer must be a whole number of samples (Preview & QC ▸ "
        "Mechanics ▸ Advanced… ▸ Breath detection)",
    "processing.wob.calc_from must be 'average' or 'individual'":
        "Work of breathing source must be Average or Individual (Preview & QC ▸ "
        "Mechanics ▸ Advanced… ▸ Work of breathing)",
    "processing.wob.avg_resampling_obs must be an integer":
        "Average-breath resampling points must be a whole number (Preview & QC ▸ "
        "Mechanics ▸ Advanced… ▸ Work of breathing)",
    "processing.volume.trend_method must be a valid scipy interp1d kind":
        "Trend interpolation method is not valid (Preview & QC ▸ Mechanics ▸ Advanced… "
        "▸ End-expiratory trend)",
    "processing.emg.ecg_auto_detect requires processing.emg.remove_ecg to be enabled":
        "Auto-detect ECG needs Remove ECG turned on (Preview & QC ▸ ECG)",
    "processing.emg.ecg_auto_detect requires input.channels.emg to be configured":
        "Auto-detect ECG needs an EMG channel assigned ('Assign channels from data…')",
    "processing.volume.trend_peak_min_prominence_frac must be between 0 and 1 (exclusive)":
        "Trend anchor — minimum breath depth must be between 0 and 1 (Preview & QC ▸ "
        "Mechanics ▸ Advanced… ▸ End-expiratory trend)",
    "processing.volume.trend_peak_min_height cannot be negative (omit it to scale to "
    "each recording)":
        "Trend anchor — absolute threshold cannot be negative (Preview & QC ▸ Mechanics "
        "▸ Advanced… ▸ End-expiratory trend)",
}
#: messages whose text carries a dynamic suffix (e.g. "... at the analysis rate (500 Hz)")
#: — matched by prefix, so the friendly text stands alone rather than gluing raw TOML
#: notation onto a translated sentence.
_FRIENDLY_PREFIXES = (
    ("processing.volume.trend_peak_min_distance_s must be at least one sample",
     "Trend anchor — minimum spacing is smaller than one sample at the analysis rate "
     "(Preview & QC ▸ Mechanics ▸ Advanced… ▸ End-expiratory trend)"),
)
#: a single-channel-role "is required" message — reachable directly through
#: ``Settings.validate()`` (``_validation_status()``/``_save_blocker()`` call it without
#: ``channel_collision()``'s prior gate, unlike ``blockers()`` below), so this still needs
#: translating even though ``channel_collision`` already names flow/poes/pgas/pdi more
#: specifically for the callers that check it first.
_CHANNEL_REQUIRED_RE = re.compile(r"^input\.channels\.(\w+) is required$")
_CHANNEL_LABELS = {"flow": "Flow", "poes": "Poes", "pgas": "Pgas", "pdi": "Pdi",
                   "volume": "Volume"}
#: last-resort fallback for a validate() message this table does not (yet) recognise —
#: never let an unmapped future message leak a dotted settings path onto a screen the way
#: this whole function exists to stop. Three-or-more lowercase/underscore segments joined
#: by dots is what every ``SettingsError`` message's technical key looks like.
_DOTTED_KEY_RE = re.compile(r"\b[a-z][a-z_]*(?:\.[a-z][a-z_]*){2,}\b")


def friendly_settings_error(exc) -> str:
    """A human sentence for an exception raised by ``Settings.validate()`` — naming the UI
    control to fix instead of the raw dotted settings path the core layer writes for a
    TOML file. Falls back to a generic, path-free rendering of an unrecognised message so
    a future core validation added without a matching translation still cannot reintroduce
    this exact bug (ticket D02, point 5) — it just reads a little less specifically."""
    msg = str(exc).strip()
    if not msg:
        return exc.__class__.__name__
    friendly = _FRIENDLY_SETTINGS_ERRORS.get(msg)
    if friendly:
        return friendly
    for prefix, text in _FRIENDLY_PREFIXES:
        if msg.startswith(prefix):
            return text
    m = _CHANNEL_REQUIRED_RE.match(msg)
    if m:
        label = _CHANNEL_LABELS.get(m.group(1), m.group(1).title())
        return f"{label} channel not assigned — click 'Assign channels from data…'"
    return _DOTTED_KEY_RE.sub("a setting", msg)


def matching_files(folder: str, mask: str) -> list:
    """Files under ``folder`` matching a possibly multi-pattern ``mask`` — patterns split on
    ';' or ',' so a mask like '*.csv; *.txt' works in the UI. Delegates to the core matcher
    (``match_input_files``) so the file list the UI shows is exactly the set the batch will
    process — case-insensitive and folder-metacharacter-safe on both platforms."""
    from respmech.core.pipeline import match_input_files
    patterns = [p.strip() for p in (mask or "*.*").replace(";", ",").split(",") if p.strip()]
    out = set()
    for pat in (patterns or ["*.*"]):
        out.update(match_input_files(folder, pat))
    return sorted(out)


def path_problem(settings, probe_write: bool = False, matches: list | None = None) -> str | None:
    """Return a human message for the first filesystem problem, or None if all paths
    are usable for a run.

    ``probe_write`` (default False) additionally tries to actually create/write/clean up a
    file in the output folder (``core.io.plan.probe_write_folder`` — a real write, never
    ``os.access``, which is unreliable against Windows ACLs). Leave it False for Settings'
    live, every-keystroke validation: this function is called from there on every field
    edit (``ui/screens/settings_screen.py``'s ``_path_problem``), and a disk round-trip per
    keystroke is worst on exactly the kind of network/removable drive this whole check
    exists for. Only ``RunScreen._start`` (covers both Run and Dry run) and the output
    folder picker pass ``probe_write=True`` — see their call sites.

    ``matches`` (default None) lets a caller that has ALREADY globbed the input folder
    hand the result in, instead of this function globbing again — ``RunScreen`` (ticket
    B04) keeps its own glob memoised on (folder, mask, folder mtime) and passes it here so
    its every-keystroke enablement check stays cheap without this function growing its own,
    second cache. Omit it (the default) to glob fresh, as Settings' live validation does."""
    s = settings
    folder = (s.input.folder or "").strip()
    if not folder or not os.path.isdir(folder):
        return f"input folder does not exist: {folder or '(unset)'}"
    if matches is None:
        matches = matching_files(folder, s.input.files)
    if not matches:
        return f"no files match '{s.input.files}' in the input folder"
    out = (s.output.folder or "").strip()
    if not out:
        return "output folder is not set"
    parent = out if os.path.isdir(out) else os.path.dirname(os.path.abspath(out))
    if not os.path.isdir(parent):
        return f"output folder's location does not exist: {out}"
    n = s.processing.emg.noise
    if n.enabled and n.reference_file:
        ref = n.reference_file
        if not os.path.isabs(ref):
            ref = os.path.join(folder, ref)
        if not os.path.isfile(ref):
            return f"noise reference file not found: {n.reference_file}"
    if probe_write:
        from respmech.core.io.plan import probe_write_folder
        probe = probe_write_folder(out)
        if not probe.ok:
            return probe.message
    return None


def blockers(settings, matches: list | None = None) -> list:
    """Every reason a run cannot start yet, worst first, as full sentences naming the
    control to fix — channel mapping, then core validation, then filesystem paths. The
    ONE shared list Setup's QC strip (``settings_screen.SettingsScreen._qc_verdict``, ticket
    B07) and the Run screen's commitment sheet (``run_screen.RunScreen._blockers``, ticket
    B04) both read, so the two can never name a different top blocker for the identical
    settings — short-circuits to the first hard blocker found, matching both screens'
    pre-existing behaviour. A core ``SettingsError`` is passed through
    :func:`friendly_settings_error` (ticket D02) rather than shown verbatim, so neither
    reader ever has to display a raw dotted settings path.

    ``matches`` is passed straight through to :func:`path_problem` — see its own docstring;
    omit it to glob fresh, as Setup's live validation does."""
    c = channel_collision(settings)
    if c:
        return [c]
    try:
        settings.validate()
    except Exception as e:                      # noqa: BLE001 — any invalidity blocks a run
        return [friendly_settings_error(e)]
    p = path_problem(settings, matches=matches)
    return [p] if p else []


def channel_collision(settings) -> str | None:
    """A HARD channel-mapping error (message, else None): a required channel
    (flow/poes/pgas/pdi) not assigned at all, one pointing at column 1 — the time axis —
    or two of them sharing a column.

    Moved here from the Settings screen (ticket B04) so the Run screen's always-visible
    commitment sheet can name exactly the same blocker Setup's live QC strip already
    names for the identical mapping — the two must never disagree about why a run is
    blocked, the same reasoning that keeps ``path_problem`` above shared rather than
    duplicated per screen."""
    ch = settings.input.channels
    req = [("flow", ch.flow), ("poes", ch.poes), ("pgas", ch.pgas), ("pdi", ch.pdi)]
    unset = [n for n, c in req if c is None]
    if unset:
        return (f"{', '.join(unset)} not assigned — "
                "click 'Assign channels from data…'")
    on_time = [n for n, c in req if c == 1]
    if on_time:
        return (f"{', '.join(on_time)} point at column 1 (the time axis) — "
                "click 'Assign channels from data…'")
    cols = [c for _n, c in req if c]
    dup = sorted({c for c in cols if cols.count(c) > 1})
    if dup:
        names = [n for n, c in req if c in dup]
        return f"{', '.join(names)} are mapped to the same column"
    return None
