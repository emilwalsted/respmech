"""PEEPi and the modified Campbell diagram's threshold work (opt-in,
``processing.pressure.peepi.enabled``).

A breath that starts inspiratory flow against a positive end-expiratory recoil pressure
(dynamic intrinsic PEEP) must first bring alveolar pressure down to atmospheric before any
flow starts; the oesophageal pressure deflection over that interval is the threshold load.
This module measures it and reports the work it adds to the Campbell diagram in SEPARATE
columns. Nothing here changes an existing column: ``calculatewob`` and its five columns,
``calcptp`` and its baseline, and every golden-locked value are untouched, and the columns
are only added when the feature is on (docs/beslutninger.md, "Modified Campbell diagram /
PEEPi construction"; docs/REVERSE_ENGINEERING.md §5.16).

Everything is computed per breath, from the breath's own raw signals plus the tail of the
PRECEDING breath's expiration (the deflection usually starts before this breath's segment
does), and is stored in ``breath["pressure_ext"]`` for ``core.results.build_breath_table`` to
join after ``breath["wob"]``.

The zero crossing
-----------------
``separateintobreathsbyflow`` ends an expiration only when ``flow > 0`` OR the mean of the next
``breathseparationbuffer`` samples is ``> 0``. During an end-expiratory pause (flow at zero) that
forward mean turns negative up to ``buffer`` samples before the real onset of inspiratory
flow, so the breath boundary is NOT where flow starts. ``t_flow`` is therefore the start of the inspiratory flow proper: the last ``flow >= 0`` to
``flow < 0`` crossing before the phase's peak inspiratory flow (fallback: the phase start),
which makes the deflection independent of the segmentation buffer and robust to flow noise
inside the pause.

What the existing columns already contain depends on where the segmenter put the breath
boundary. ``calcptp`` integrates against the mean Poes of the first ``ptp_baseline_window_s`` of
the inspiratory phase, and ``calculatewob`` measures its elastic and resistive areas against
the Poes at the phase start/end of expiration. When the boundary lands in the pause BEFORE
the deflection, both references are the pre-deflection level and the existing PTP and Campbell
polygon already contain (part of) the threshold; when it lands after the deflection, they do
not. The ``*_peepi`` and ``*_thr`` columns therefore add only the part the existing reference
does not already contain (``shift_*`` below), never the full PEEPi, so the same recording gives
the same answer wherever the boundary falls.

Two quantities are deliberately not merged: ``int_oes_preflow`` (the area of the deflection,
referenced to the Poes at its onset) is reported separately and never added to any ``*_peepi``
column, because the ordinary PTP is already referenced to its own end-expiratory baseline and
adding the pre-flow area would subtract that baseline twice (docs/PTP_INVESTIGATION.md).

The four thresholds in :class:`respmech.core.settings.PeepiSettings` are literature-informed
STARTING values, not measured ones.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from respmech.core._compat import trapezoid
from respmech.core.analysis.signals import Capabilities

# cmH2O·L -> J, the same factor calculatewob uses.
WOBUNITCHANGEFACTOR = 98.0638 / 1000

#: A gap larger than this many samples between the end of the preceding breath's
#: expiration and the start of this breath's inspiration means the two are not neighbours in
#: the recording (segment_file's own phase slices leave a one-sample gap at every boundary).
_MAX_BOUNDARY_GAP_SAMPLES = 3.5


def peepi_source(caps) -> str:
    """Which PEEPi value feeds the threshold work: ``"corrected"`` (the dynamic deflection
    minus the gastric-pressure change over the same interval, Zakynthinos 1997/1999) when a
    Pgas channel exists, otherwise ``"dynamic"`` (Haluszka 1990)."""
    return "corrected" if getattr(caps, "pgas", False) else "dynamic"


def true_flow_start(flow) -> int:
    """Index, within an inspiratory phase, of the true start of inspiratory flow: the sample
    after the LAST sample with ``flow >= 0`` that lies before the phase's peak inspiratory
    flow (i.e. the last ``>= 0`` to ``< 0`` crossing leading into the real inspiration).
    Choosing the last such crossing, rather than the first, keeps a stray sub-zero sample of
    flow noise inside an end-expiratory pause from being mistaken for the start of flow.
    A phase whose peak flow is its first sample, or that never crosses, falls back to index 0
    (the phase start)."""
    f = np.asarray(flow, dtype=float).reshape(-1)
    if f.size == 0 or not np.any(f < 0):
        return 0
    peak = int(np.nanargmin(f))
    before = np.flatnonzero(f[:peak] >= 0)
    return int(before[-1]) + 1 if before.size else 0


def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average over ``n`` samples, the same length as ``x``; the ends are
    averaged over the samples that exist there rather than padded."""
    if n <= 1:
        return x.astype(float, copy=True)
    kernel = np.ones(n)
    return np.convolve(x, kernel, mode="same") / np.convolve(np.ones_like(x), kernel, mode="same")


def detect_peepi_onset(poes, t_flow: int, smooth_n: int, onset_slope_frac: float):
    """Index (in ``poes``' own coordinates) of the onset of the pre-flow Poes deflection that
    ends at ``t_flow``, or ``None`` when it cannot be located.

    ``poes`` is the search window: the tail of the preceding breath's expiration followed by
    this breath's samples up to ``t_flow``. The window is smoothed (``smooth_n`` samples),
    and the walk goes backwards from ``t_flow`` for as long as the smoothed Poes is still
    falling faster than ``onset_slope_frac`` of the steepest fall anywhere in the window.
    A window with no falling step at all has no deflection: the onset is ``t_flow`` itself
    (a deflection of exactly zero, not an unlocatable one). ``None`` is returned only for a
    window too short to hold a slope or one containing non-finite samples."""
    x = np.asarray(poes, dtype=float).reshape(-1)[: t_flow + 1]
    if x.size < 3 or not np.all(np.isfinite(x)):
        return None
    d = np.diff(_moving_average(x, smooth_n))
    steepest = float(d.min())
    if not steepest < 0:
        return int(t_flow)
    threshold = onset_slope_frac * steepest
    j = int(t_flow)
    while j > 0 and d[j - 1] < threshold:
        j -= 1
    return j


def _added_or_none(breath):
    """The breath's stored rectangle height as a finite float (0 included), else ``None``."""
    try:
        h = float(breath.get("peepi_added"))
    except (TypeError, ValueError):
        return None
    return h if np.isfinite(h) else None


def peepi_rectangle_height(breath):
    """Height (cmH2O) of the PEEPi rectangle the modified Campbell diagram adds for ONE breath,
    or ``None`` when the breath has none to draw (PEEPi off, unlocatable, or nothing added).
    The rectangle spans the breath's tidal volume and sits above the end-expiratory Poes, so
    its area is the threshold work ``wob_in_thr`` (before the ``bcnt``/``vefactor`` scaling)."""
    h = _added_or_none(breath)
    return h if h is not None and h > 0 else None


def mean_peepi_rectangle_height(breaths):
    """Mean rectangle height over the breaths that PEEPi was computed for, a breath with nothing
    to add counting as 0 (so the average loop's rectangle matches the mean of ``wob_in_thr``
    rather than only the breaths that happen to have PEEPi). ``None`` when no breath was
    computed at all, e.g. the feature is off."""
    hs = [h for h in (_added_or_none(b) for b in breaths) if h is not None]
    return float(np.mean(hs)) if hs else None


def _nan_columns(caps, has_pdi_columns: bool) -> "OrderedDict[str, float]":
    names = ["peepi_dyn"]
    if getattr(caps, "pgas", False):
        names += ["peepi_pgas_drop", "peepi_corr"]
    names += ["peepi_lag", "int_oes_preflow", "ptp_oes_preflow", "wob_in_thr",
              "wob_in_total_thr", "wobtotal_thr", "int_oesinsp_peepi", "ptp_oesinsp_peepi"]
    if has_pdi_columns:
        names += ["int_pdiinsp_peepi", "ptp_pdiinsp_peepi"]
    return OrderedDict((n, float("nan")) for n in names)


def attach(breath, prev_breath, bcnt, vefactor, settings):
    """Compute the PEEPi columns for ONE breath and store them in ``breath["pressure_ext"]``.

    Must run after ``compute.calculatemechanics`` has filled ``breath["mechanics"]`` (and, with
    Poes, ``breath["wob"]``): the threshold work is expressed in the same units and scaling
    (× ``bcnt`` × ``vefactor``) as the existing WOB. ``prev_breath`` is the breath immediately
    before this one in the recording, ignored or not, or ``None`` for the first breath. Returns
    a notice string when the breath's values are NaN and the reader should be told why, else
    ``None``. Never raises for a data problem: an unlocatable onset is NaN plus a notice."""
    caps = getattr(settings, "capabilities", Capabilities.FULL)
    cfg = settings.processing.pressure.peepi
    fs = float(settings.input.format.samplingfrequency)
    has_pdi_columns = bool(getattr(caps, "pgas", False) and getattr(caps, "pdi", False))
    cols = _nan_columns(caps, has_pdi_columns)

    breath.pop("peepi_added", None)

    def _store(notice):
        breath["pressure_ext"] = cols
        return notice

    insp = breath["inspiration"]
    t_flow = true_flow_start(insp["flow"])

    if prev_breath is None or not prev_breath.get("has_phases", True):
        return _store("no preceding breath to search for the pre-flow deflection in")
    prev_exp = prev_breath["expiration"]
    prev_t, this_t = np.asarray(prev_exp["time"]).reshape(-1), np.asarray(insp["time"]).reshape(-1)
    if (prev_t.size == 0 or this_t.size == 0
            or this_t[0] - prev_t[-1] > _MAX_BOUNDARY_GAP_SAMPLES / fs):
        return _store("the preceding breath does not directly precede this one in the recording")

    n_window = max(1, int(round(cfg.search_window_s * fs)))
    smooth_n = max(1, int(round(cfg.smooth_s * fs)))

    def _window(channel):
        tail = np.asarray(prev_exp[channel], dtype=float).reshape(-1)[-n_window:]
        head = np.asarray(insp[channel], dtype=float).reshape(-1)[: t_flow + 1]
        return np.concatenate([tail, head]), tail.size + t_flow

    poes_w, t_flow_w = _window("poes")
    onset = detect_peepi_onset(poes_w, t_flow_w, smooth_n, cfg.onset_slope_frac)
    if onset is None:
        return _store("the pre-flow Poes deflection could not be located "
                      "(search window too short or not finite)")

    deflection = max(float(poes_w[onset] - poes_w[t_flow_w]), 0.0)
    detected = deflection >= cfg.min_deflection
    peepi_dyn = deflection if detected else 0.0
    if detected:
        lag = (t_flow_w - onset) / fs
        pre = poes_w[onset] - poes_w[onset: t_flow_w + 1]
        int_preflow = float(trapezoid(pre, dx=1.0 / fs))
    else:
        lag = 0.0
        int_preflow = 0.0

    insp_poes = np.asarray(insp["poes"], dtype=float).reshape(-1)
    ptp_bw = max(1, int(round(float(getattr(getattr(settings.processing, "mechanics", None),
                                          "ptp_baseline_window_s", 0.05)) * fs)))
    base_ptp = float(np.mean(insp_poes[:ptp_bw]))
    base_wob = float(insp_poes[0])
    if detected:
        # the part of the deflection the existing references do NOT already contain
        shift_ptp = min(max(float(poes_w[onset]) - base_ptp, 0.0), peepi_dyn)
        shift_wob = min(max(float(poes_w[onset]) - base_wob, 0.0), peepi_dyn)
    else:
        shift_ptp = shift_wob = 0.0
    already_in_polygon = peepi_dyn - shift_wob

    values = OrderedDict()
    values["peepi_dyn"] = peepi_dyn
    peepi_src = peepi_dyn
    peepi_corr = float("nan")
    if getattr(caps, "pgas", False):
        pgas_w, _ = _window("pgas")
        pgas_drop = max(float(pgas_w[onset] - pgas_w[t_flow_w]), 0.0) if detected else 0.0
        peepi_corr = max(peepi_dyn - pgas_drop, 0.0)
        peepi_src = peepi_corr
        values["peepi_pgas_drop"] = pgas_drop
        values["peepi_corr"] = peepi_corr
    values["peepi_lag"] = lag
    values["int_oes_preflow"] = int_preflow
    values["ptp_oes_preflow"] = int_preflow * bcnt * vefactor

    mech = breath["mechanics"]
    wob = breath.get("wob", {})
    # Threshold work added ON TOP of the existing polygon: the source value minus what the
    # polygon already holds (never negative: a corrected PEEPi smaller than the polygon's own
    # share adds nothing rather than subtracting).
    added = max(peepi_src - already_in_polygon, 0.0)
    # the height of the rectangle the modified Campbell diagram draws over the tidal volume
    # (its area is ``wob_in_thr``, up to the unit change and the per-minute scaling)
    breath["peepi_added"] = float(added)
    wob_in_thr = added * mech["vt"] * WOBUNITCHANGEFACTOR * bcnt * vefactor
    values["wob_in_thr"] = wob_in_thr
    values["wob_in_total_thr"] = wob["wob_in_total"] + wob_in_thr
    values["wobtotal_thr"] = wob["wobtotal"] + wob_in_thr
    int_oes = mech["int_oesinsp"] + shift_ptp * mech["ti"]
    values["int_oesinsp_peepi"] = int_oes
    values["ptp_oesinsp_peepi"] = int_oes * bcnt * vefactor
    if has_pdi_columns:
        pdi_w = pgas_w - poes_w
        base_pdi = float(np.mean(np.asarray(insp["pdi"], dtype=float).reshape(-1)[:ptp_bw]))
        shift_pdi = min(max(base_pdi - float(pdi_w[onset]), 0.0), peepi_corr) if detected else 0.0
        int_pdi = mech["int_pdiinsp"] + shift_pdi * mech["ti"]
        values["int_pdiinsp_peepi"] = int_pdi
        values["ptp_pdiinsp_peepi"] = int_pdi * bcnt * vefactor

    cols.update(values)
    return _store(None)
