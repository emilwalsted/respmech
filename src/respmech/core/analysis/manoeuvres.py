"""Manoeuvre extraction (M-29): pull the values a single breath TYPED as a named
manoeuvre (inspiratory capacity, forced vital capacity, a maximal inspiratory/sniff
effort — see ``BreathTypeEntry``/``BREATH_KINDS`` in ``core.settings``) is captured
for, straight from its raw breath dict — never from ``calculatemechanics``' output,
which a typed breath never even reaches (M-19 unions every typed kind into
``excludebreaths`` on a flow-bearing signal set, so the ordinary mechanics loop skips
it exactly like a manually excluded breath).

Qt-free, numpy-only. ``extract()`` is deliberately a pure function of ONE breath dict
plus the file's OWN tidal (non-typed, non-excluded) breaths — it never reaches into
another file, another run, or a cohort-level aggregate, so a subset run and a full
batch run compute an identical value for the same file (the acceptance criterion this
module is built to satisfy: "LOW_EFFORT-flaget udregnes identisk fra en
delmængdekørsel og en fuld kørsel").

FVC/MFVL numerics (the actual FVC/FEV1/PEF/flow-volume curve) are M-42's scope, not
this one — ``validate_fvc_manoeuvre``/``suggest_fvc`` here are quality/UI helpers
only, deliberately empty of any spirometric arithmetic.

Threshold provenance: every field marked with a trailing ``†`` in
``core.settings.IcSettings`` is a PLACEHOLDER (the plan's own starting value), not one
measured against a real IC recording — this sandbox has no production data to
calibrate against (K-035's lesson: a threshold set without a real recording in hand is
a guess, not a fact). The formulas below are implemented and pinned by analytical/
synthetic tests; the cut-offs themselves want a pass against real recordings before
they are trusted. See this ticket's closing report / ``docs/beslutninger.md`` for what
to measure.
"""
from __future__ import annotations

import numpy as np

#: kinds `extract()` treats as an inspiratory-capacity manoeuvre — `ic_fvc` is a
#: single breath captured as BOTH an IC (its inspiration) and an FVC (its expiration),
#: so it gets the full IC field set exactly like a pure `ic` breath.
_IC_KINDS = frozenset({"ic", "ic_fvc"})
#: kinds that carry an expiratory (forced-vital-capacity) limb worth a quality check —
#: `fvc` alone, or the expiratory half of a combined `ic_fvc` manoeuvre.
_FVC_KINDS = frozenset({"fvc", "ic_fvc"})
_MAX_EFFORT_KINDS = frozenset({"max_insp", "sniff"})

#: minimum expiratory duration (s) below which `validate_fvc_manoeuvre` flags
#: `FVC_TOO_SHORT` — a placeholder in the same sense as `IcSettings`' `†` fields (no
#: production FVC recording to calibrate against here); a forced expiration lasting
#: under a second is not a plausible full exhalation regardless of that calibration.
_FVC_MIN_DURATION_S = 1.0


def _arr(x) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=float))


def _peak_volume_index(breath) -> int:
    return int(np.argmax(_arr(breath["volume"])))


def _preceding_eelvs(breath, tidal_breaths, n: int) -> list[float]:
    """End-expiratory volume (a tidal breath's own last, expiratory, volume sample)
    of the up-to-``n`` tidal breaths immediately preceding ``breath`` in the
    recording, ordered oldest-to-newest. Breath NUMBERS (not list position — the
    caller may hand ``tidal_breaths`` in any order) decide "preceding": this
    codebase numbers breaths sequentially as it segments a recording, so a lower
    number is always earlier in time."""
    if n <= 0:
        return []
    this_no = breath["number"]
    prior = sorted(
        (b for b in tidal_breaths if b["number"] < this_no),
        key=lambda b: b["number"],
    )
    prior = prior[-n:]
    return [float(_arr(b["volume"])[-1]) for b in prior]


def _ic_eelv_pre(breath, insp, tidal_breaths, ic_cfg) -> tuple[float, float, int]:
    """``(ic_eelv_pre, ic_eelv_pre_sd, ic_eelv_pre_n)`` — the mean (+ SD, + count) of
    up to ``preceding_breaths`` tidal breaths' own end-expiratory volume, falling back
    to this breath's OWN immediate pre-inspiratory volume sample (``n=1``, ``sd=0.0``)
    when fewer than ``min_preceding_breaths`` tidal breaths precede it (too little
    context in the file to average over — the manoeuvre near the very start of a
    recording, say)."""
    eelvs = _preceding_eelvs(breath, tidal_breaths, ic_cfg.preceding_breaths)
    if len(eelvs) >= ic_cfg.min_preceding_breaths:
        arr = np.asarray(eelvs, dtype=float)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=0)) if arr.size >= 2 else 0.0
        return mean, sd, len(eelvs)
    return float(insp["volume"][0]), 0.0, 1


def _trailing_plateau_s(flow, fs: float, threshold: float) -> float:
    """How long, counting back from the very end of ``flow`` (the tail of the
    inspiration — a breath held at TLC just before the manoeuvre ends), the signal
    stays below ``threshold`` in magnitude without interruption. 0.0 when the last
    sample already exceeds the threshold.

    Caveat for `plateau_flow_lps` calibration: an inspiration that ends AT its own
    peak flow (no deliberate breath-hold at all) still decelerates continuously
    through the crossing on its way into expiration, so the last few samples before
    that crossing are always inside a small `threshold` window — every breath reports
    a SPURIOUS few-tens-of-milliseconds "plateau" from this alone, not a real held
    breath. The default `min_plateau_s=0.0` never rejects on this (documented as
    never-fires-yet in `IcSettings`), so it costs nothing today, but a future
    calibration raising `min_plateau_s` above zero must set it above that spurious
    floor, not at the literal length of a genuine hold."""
    flow = _arr(flow)
    count = 0
    for v in flow[::-1]:
        if abs(v) < threshold:
            count += 1
        else:
            break
    return count / fs


def _low_effort(ic_peak_in_flow: float, poes_ic_swing: float | None, tidal_breaths,
                caps, ic_cfg) -> bool:
    """Compares THIS manoeuvre's own peak inspiratory flow (and, when Poes is
    available, its Poes swing) against the median of the file's own tidal breaths —
    never against another file, another run, or the mechanics table (see the module
    docstring's acceptance criterion). Not evaluable (returns ``False``, never
    guessed) when the file has no tidal breaths to compare against at all.

    A caveat worth knowing before recalibrating `low_effort_frac`: an inspiratory
    capacity manoeuvre only has to be COMPLETE, not fast — a slow but genuinely full
    IC can have a smaller peak flow than an ordinary brisk tidal breath. The Poes
    limb (when the channel exists) is a better effort signal for exactly that reason;
    with Poes absent (`caps.poes=False`) this flow-only check is the sole LOW_EFFORT
    signal and can misjudge a slow, complete manoeuvre."""
    flagged = False
    tidal_with_insp = [b for b in tidal_breaths if len(_arr(b["inspiration"]["flow"])) > 0]
    if tidal_with_insp:
        tidal_peaks = [-float(_arr(b["inspiration"]["flow"]).min()) for b in tidal_with_insp]
        median_flow = float(np.median(tidal_peaks))
        if median_flow > 0 and ic_peak_in_flow < ic_cfg.low_effort_frac * median_flow:
            flagged = True
    if caps.poes and poes_ic_swing is not None:
        tidal_with_poes = [b for b in tidal_breaths if len(_arr(b["inspiration"]["poes"])) > 0]
        if tidal_with_poes:
            tidal_swings = [
                float(_arr(b["inspiration"]["poes"])[0] - _arr(b["inspiration"]["poes"]).min())
                for b in tidal_with_poes
            ]
            median_poes = float(np.median(tidal_swings))
            if median_poes > 0 and poes_ic_swing < ic_cfg.low_effort_frac * median_poes:
                flagged = True
    return flagged


def _boundary(breath, tidal_breaths) -> bool:
    """True when ``breath`` is the first- or last-numbered breath among ``extract``'s
    OWN two inputs — this breath plus the tidal breaths it was given (never a
    manually excluded or differently-typed breath that may sit outside that set, since
    neither is visible here) — a manoeuvre this close to either edge of what IS visible
    carries the same "was this cut off?" risk K-035 raised for an ordinary boundary
    breath, so it is worth a visible flag rather than silent trust."""
    numbers = {b["number"] for b in tidal_breaths}
    numbers.add(breath["number"])
    return breath["number"] == min(numbers) or breath["number"] == max(numbers)


def ic_acceptance(breath, tidal_breaths, caps, ic_cfg, *, vol_ic: float,
                  ic_eelv_pre_sd: float, ic_eelv_pre_n: int, ic_peak_in_flow: float,
                  ic_plateau_s: float, poes_ic_swing: float | None) -> list[str]:
    """The ``quality`` flags for an IC/IC+FVC manoeuvre, EXCLUDING ``NOT_REPEATABLE``
    (which needs this file's OTHER typed IC breaths — not available from a single
    breath's own data — and is applied afterwards by :func:`apply_repeatability`).
    Takes the already-computed ``vol_ic``/``ic_eelv_pre_sd``/``ic_peak_in_flow``/
    ``ic_plateau_s`` values (:func:`_ic_fields` computes them once, for the row
    itself) rather than recomputing them a second time here."""
    flags: list[str] = []
    # EELV_UNSTABLE: the preceding breaths' own end-expiratory-volume spread, scaled
    # by THIS breath's own vol_ic (self-review finding) -- NOT by ic_eelv_pre itself.
    # This codebase zero-references AND drift-corrects volume by default
    # (processing.volume.correct_drift=True), so a real ic_eelv_pre routinely sits
    # within a few millilitres of 0 L; dividing the SD by a near-zero baseline
    # explodes the ratio for perfectly ordinary breath-to-breath noise, making the
    # flag fire on almost every real recording. vol_ic is always a real, physically
    # meaningful, non-trivial volume for an actual manoeuvre, so "the pre-manoeuvre
    # baseline wandered by more than eelv_tolerance_frac of the manoeuvre's OWN size"
    # is both scale-appropriate and well-behaved near zero.
    if ic_eelv_pre_n >= 2 and ic_eelv_pre_sd > 0 and vol_ic > 0:
        if (ic_eelv_pre_sd / vol_ic) > ic_cfg.eelv_tolerance_frac:
            flags.append("EELV_UNSTABLE")
    if _low_effort(ic_peak_in_flow, poes_ic_swing, tidal_breaths, caps, ic_cfg):
        flags.append("LOW_EFFORT")
    if ic_plateau_s < ic_cfg.min_plateau_s:
        flags.append("NO_PLATEAU")
    if _boundary(breath, tidal_breaths):
        flags.append("BOUNDARY")
    return flags


def _ic_fields(breath, kind: str, tidal_breaths, caps, s) -> dict:
    insp = breath["inspiration"]
    fs = float(s.input.format.samplingfrequency)
    ic_cfg = s.processing.lung_volume.ic

    ic_eelv_pre, ic_eelv_pre_sd, ic_eelv_pre_n = _ic_eelv_pre(breath, insp, tidal_breaths, ic_cfg)
    vol_ic = float(_arr(breath["volume"]).max()) - ic_eelv_pre
    ic_ti = len(_arr(insp["time"])) / fs
    ic_peak_in_flow = -float(_arr(insp["flow"]).min())
    ic_plateau_s = _trailing_plateau_s(insp["flow"], fs, ic_cfg.plateau_flow_lps)

    out: dict = {
        "kind": kind,
        "vol_ic": vol_ic,
        "ic_eelv_pre": ic_eelv_pre,
        "ic_eelv_pre_sd": ic_eelv_pre_sd,
        "ic_eelv_pre_n": ic_eelv_pre_n,
        "ic_ti": ic_ti,
        "ic_peak_in_flow": ic_peak_in_flow,
        "ic_plateau_s": ic_plateau_s,
    }

    poes_ic_swing = None
    if caps.poes:
        poes_ic_min = float(_arr(insp["poes"]).min())
        poes_ic_eelv = float(_arr(insp["poes"])[0])
        poes_ic_swing = poes_ic_eelv - poes_ic_min
        peak_ix = _peak_volume_index(breath)
        out.update({
            "poes_ic_min": poes_ic_min,
            "poes_ic_eelv": poes_ic_eelv,
            "poes_ic_swing": poes_ic_swing,
            "poes_ic_peakvol": float(_arr(breath["poes"])[peak_ix]),
        })
    if caps.pdi:
        pdi_ic_max = float(_arr(insp["pdi"]).max())
        pdi_eelv = float(_arr(insp["pdi"])[0])
        out.update({
            "pdi_ic_max": pdi_ic_max,
            "pdi_ic_swing": pdi_ic_max - pdi_eelv,
        })
    if caps.pgas:
        peak_ix = _peak_volume_index(breath)
        out["pgas_ic_peakvol"] = float(_arr(breath["pgas"])[peak_ix])

    quality = ic_acceptance(breath, tidal_breaths, caps, ic_cfg,
                            vol_ic=vol_ic, ic_eelv_pre_sd=ic_eelv_pre_sd,
                            ic_eelv_pre_n=ic_eelv_pre_n, ic_peak_in_flow=ic_peak_in_flow,
                            ic_plateau_s=ic_plateau_s, poes_ic_swing=poes_ic_swing)
    if kind in _FVC_KINDS:
        quality.extend(f for f in validate_fvc_manoeuvre(breath) if f not in quality)
    out["quality"] = quality
    return out


def max_effort_from_breath(breath, caps, s) -> dict:
    """Reference values for a maximal-effort manoeuvre breath (``max_insp``/
    ``sniff``): the peak inspiratory Poes/Pdi SWING and the peak EMG RMS reached
    anywhere in the breath — each present only with its own channel(s). These are
    plain reference NUMBERS for a later ticket's normalisation (M-47), not a quality
    judgement of their own — a maximal-effort breath carries no ``quality`` flags
    here (the IC-specific acceptance checks above do not apply to it).

    ``poes_max_ref``/``pdi_max_ref`` are swings from this breath's OWN immediate
    pre-inspiratory baseline (``insp[...][0]``) — the SAME convention
    ``poes_ic_swing``/``pdi_ic_swing`` use above (self-review finding: a max-effort
    breath and an IC's own swing must be on the same baseline-subtracted footing, or
    a later "this breath's swing as a % of the max reference" (M-47) ratio would
    divide a baseline-subtracted number by one that still carries the channel's
    absolute resting offset — never physiologically meaningful).

    ``rms_max_ref`` is computed here rather than read off the breath dict because a
    typed breath is `ignored=True` (M-19's union rule) and so never reaches the
    ordinary per-breath ``compute_segment_emg`` call in the mechanics loop — nothing
    upstream has computed an RMS envelope for it yet.
    """
    out: dict = {}
    insp = breath["inspiration"]
    if caps.poes:
        out["poes_max_ref"] = float(_arr(insp["poes"])[0]) - float(_arr(insp["poes"]).min())
    if caps.pdi:
        out["pdi_max_ref"] = float(_arr(insp["pdi"]).max()) - float(_arr(insp["pdi"])[0])
    if caps.emg and np.size(breath.get("emgcols", [])) > 0:
        from respmech.core import emg as emglib
        fs = float(s.input.format.samplingfrequency)
        rms_s = float(s.processing.emg.rms_s)
        cols = np.asarray(breath["emgcols"])
        peaks = []
        for ch in range(cols.shape[1]):
            values, _starts = emglib.rolling_rms(cols[:, ch], rms_s, fs)
            if values.size and not np.all(np.isnan(values)):
                peaks.append(float(np.nanmax(values)))
        if peaks:
            out["rms_max_ref"] = max(peaks)
    return out


def validate_fvc_manoeuvre(breath) -> list[str]:
    """Minimal, non-numeric quality flags for a breath's EXPIRATORY limb as a
    candidate forced-vital-capacity manoeuvre. Deliberately does not compute FVC,
    FEV1, PEF or any other spirometric value — that arithmetic is M-42's
    (``core.analysis.mfvl``) scope; this only answers "is this expiration even a
    plausible forced manoeuvre to look at", for the Manoeuvres sheet and the UI's
    quality display before M-42 exists.
    """
    exp = breath.get("expiration")
    if not exp or len(_arr(exp.get("time", []))) < 2:
        return ["FVC_TOO_SHORT"]
    duration = _arr(exp["time"])[-1] - _arr(exp["time"])[0]
    return ["FVC_TOO_SHORT"] if duration < _FVC_MIN_DURATION_S else []


def suggest_fvc(breaths: dict) -> int | None:
    """A plain, deterministic heuristic hint for the UI (M-31's "Suggested-hint"):
    the untyped, non-ignored breath with the LONGEST expiration in the file — a
    forced vital capacity manoeuvre is, by definition, a deliberately prolonged
    forced exhalation, so the longest expiration in an otherwise ordinary tidal
    recording is the most plausible untyped candidate. Returns ``None`` for a file
    with no eligible breath (already-typed and ignored breaths are skipped) rather
    than guessing one.

    Purely advisory: nothing here types the breath or excludes any other
    candidate — the user picks, this only points.
    """
    best_no, best_duration = None, -1.0
    for no, breath in breaths.items():
        if breath.get("ignored") or not breath.get("has_phases", True):
            continue
        exp = breath.get("expiration")
        if not exp:
            continue
        t = _arr(exp.get("time", []))
        if t.size < 2:
            continue
        duration = float(t[-1] - t[0])
        if duration > best_duration:
            best_no, best_duration = no, duration
    return best_no


def extract(breath, kind: str, tidal_breaths, caps, s) -> dict:
    """The Manoeuvres-sheet row for one typed breath, as a REN function of raw breath
    dicts (never of ``calculatemechanics`` output — see the module docstring).
    ``tidal_breaths`` is the file's own non-ignored (untyped) breaths, in whatever
    order the caller holds them in — every helper here re-sorts by breath NUMBER
    itself, never assuming list order. Returns ``{'kind': kind}`` alone for a kind
    this ticket does not extract numeric values for yet (``other`` — see the M-29
    ticket's scope; ``rest`` is never passed here at all, the caller skips it, since
    it names a noise-reference segment, not a manoeuvre).
    """
    if kind in _IC_KINDS:
        return _ic_fields(breath, kind, tidal_breaths, caps, s)
    if kind in _MAX_EFFORT_KINDS:
        out = {"kind": kind, "quality": []}
        out.update(max_effort_from_breath(breath, caps, s))
        return out
    if kind == "fvc":
        return {"kind": kind, "quality": validate_fvc_manoeuvre(breath)}
    return {"kind": kind, "quality": []}


def apply_repeatability(manoeuvres: dict, ic_cfg) -> None:
    """Second pass, run once per file AFTER every typed breath's own :func:`extract`
    result is in hand (``NOT_REPEATABLE`` needs to compare an IC against its OWN
    file's OTHER IC measurements, which a single call to ``extract`` never sees —
    see the module docstring). Mutates each IC/IC+FVC entry's ``quality`` list
    in place.

    Each row is compared against the ``ic_cfg.aggregate`` (mean or median) of its
    file's OTHER eligible ICs — leave-one-out, NEVER including the row's own value in
    what it is measured against (self-review finding: an inclusive centre both halves
    the effective tolerance for exactly two attempts — each one sits at half their
    pairwise difference from a shared mean — and, worse, lets one bad attempt drag the
    centre far enough to also implicate a genuinely agreeing pair; leave-one-out avoids
    both). A breath already carrying one of ``ic_cfg.reject_flags`` (``LOW_EFFORT`` by
    default) is excluded from the comparison GROUP entirely (on either side: it is
    never compared, and never compared against) — a low-effort attempt should not be
    able to drag a genuinely repeatable pair of good attempts into looking
    unrepeatable, or vice versa mask its own unreliability by "agreeing" with another
    bad attempt. A single eligible IC (no sibling to compare against) is never
    flagged: repeatability is a property of a PAIR, not of one measurement alone.

    ``aggregate='mean'`` (the default) is still an outlier-SENSITIVE centre once there
    are at least FOUR total eligible attempts (three "others" to average — with only
    three total, "the other two" is too few for mean and median to ever disagree): a
    genuinely bad fourth attempt drags the leave-one-out mean each GOOD attempt is
    compared against too, and can flag them as well. This is the well known
    mean-vs-median trade-off, not a bug — ``aggregate='median'`` is the outlier-robust
    alternative for a study that expects four or more repeat attempts per manoeuvre;
    see docs/beslutninger.md.
    """
    ic_entries = [row for row in manoeuvres.values() if row.get("kind") in _IC_KINDS]
    eligible = [row for row in ic_entries
               if not (set(row.get("quality", [])) & set(ic_cfg.reject_flags))]
    if len(eligible) < 2:
        return
    for i, row in enumerate(eligible):
        others = [eligible[j]["vol_ic"] for j in range(len(eligible)) if j != i]
        centre = float(np.median(others)) if ic_cfg.aggregate == "median" else float(np.mean(others))
        if centre == 0:
            continue
        if abs(row["vol_ic"] - centre) / abs(centre) > ic_cfg.repeatability_frac:
            if "NOT_REPEATABLE" not in row["quality"]:
                row["quality"].append("NOT_REPEATABLE")
