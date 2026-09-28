"""FVC/MFVL/EFL/VEcap and ventilatory capacity (M-42).

Two independent pieces, from a single file's already-resolved ``fvc`` reference
(``core.analysis.references.resolve_reference``, the SAME slot ic/max_insp/
baseline_ic already share):

* :func:`fvc_metrics` -- a REN function of ONE typed ``fvc``/``ic_fvc`` breath dict
  (never of ``calculatemechanics`` output -- a typed breath never reaches that loop,
  M-19's exclude-union): ``V_TLC = insp['volume'][-1]``, the maximal expiratory
  flow-volume (MEFV) envelope ``v = V_TLC - exp['volume']`` (forced non-decreasing,
  ``np.maximum.accumulate``), PEF (over a >=10 ms window, so a single-sample spike
  never wins), back-extrapolated ``t0``/BEV (ATS/ERS 2019), FEV1/FEV1%FVC, the
  end-of-forced-expiration criterion, and (from the file's OWN following breath, if
  its excursion is near-maximal) a peak inspiratory-flow reference. This is
  ``core.analysis.manoeuvres.extract``'s own explicitly-deferred arithmetic
  ("M-42's scope, not this one" -- see that module's docstring): kept in a SEPARATE
  module rather than folded into ``manoeuvres.py`` so that boundary stays real, not
  just a comment -- ``core.pipeline`` merges this function's output into a typed
  FVC/IC+FVC breath's Manoeuvres-sheet row itself (:func:`apply_to_row`), and
  ``manoeuvres.extract`` is never touched by this ticket.
* :func:`attach` -- placement of a file's TIDAL breaths against that MEFV envelope,
  on a TLC-anchored axis (Johnson 1999), giving each one an expiratory-flow-
  limitation percentage, the minimal expiratory time the envelope would allow at its
  own tidal excursion, and the ventilatory capacity/breathing reserve that implies
  (Johnson 1995/1999; ATS/ACCP 2003 for the MVV x40 fallback).

**Why :func:`attach` is wired inline in ``core.pipeline.run_batch``'s per-file loop,
NOT as a `references.attach`/`lungvol.attach`-style POST-loop pass**: the ticket's
own instruction is that these columns live in ``breath['mfvl_ext']``, joined by
``core.results.build_breath_table`` exactly like ``breath['wob']`` already is --
and that join happens INSIDE the per-file loop, before ``run_batch`` moves on to the
next file. A `references.attach`-style pass, by contrast, only runs after EVERY
file's own loop iteration has finished (deliberately, so a reference can resolve to
a file processed later in the batch). Reconciling "the column lives in the raw
breath dict, joined before the loop moves on" with "the IC operating point a cross-
file reference resolves may not exist yet" is not attempted here: **this ticket's
``attach`` only resolves a SAME-FILE ``fvc``/``ic``/``ic_fvc`` reference** (a file's
own typed breaths), which is already fully available mid-loop, and uses
``eelv_tracking='none'`` semantics for its own IC operating point (``ic_op =
vol_ic_ref``, held constant across the file) rather than depending on
``core.analysis.lungvol.attach``'s full per-breath EELV-tracking arithmetic (a
POST-loop pass that has not run yet at this point). A cross-file ``fvc``/``ic``
reference, or within-file EELV tracking, is a documented, deliberate gap -- see
this ticket's closing report / ``docs/beslutninger.md`` -- left for a future ticket
that reconciles the two loop-timing models properly, not guessed at here.

Qt-free, numpy-only. Threshold provenance: every ``†``-marked field in
``core.settings.MfvlSettings`` is a placeholder (the plan's own starting value, not
measured against a real FVC recording -- K-035's lesson: this sandbox has none to
calibrate against). The formulas are pinned by analytical/synthetic tests; the
cut-offs want a pass against real recordings before they are trusted.
"""
from __future__ import annotations

import numpy as np

from respmech.core.summary import group_key


def _arr(x) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=float))


def _finite(x) -> bool:
    """``True`` only for a real, non-NaN number -- self-review finding: the
    ``x == x`` idiom used throughout this module to mean "not NaN" is silently
    wrong for a MISSING dict key, where ``.get(...)`` hands back ``None`` and
    ``None == None`` is ``True``. A row an upstream failure left half-built (an
    ``fvc``/``ic_fvc`` row whose :func:`apply_to_row` call raised, so it never got
    its numeric fields) must read as "nothing usable here", not crash a later
    ``row["fev1"]`` lookup or arithmetic op several breaths later."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x


def mefv_curve(breath) -> tuple[np.ndarray, np.ndarray, float]:
    """``(v, flow, v_tlc)`` for ONE breath's own forced expiration. ``v_tlc =
    insp['volume'][-1]`` (the peak volume reached just before this breath's forced
    exhalation). ``v = v_tlc - exp['volume']`` is "volume below TLC" -- 0 at the
    very start of the forced expiration (at TLC), rising toward FVC as the subject
    exhales toward RV -- forced non-decreasing with ``np.maximum.accumulate`` so a
    tiny numerical/noise dip never makes ``v`` locally decrease (which would break
    every ``np.interp`` lookup against it, since interp requires an increasing x).
    ``flow`` is ``exp['flow']`` unchanged: this codebase's own sign convention has
    expiratory flow positive (``core.analysis.manoeuvres``' insp/exp fixtures;
    inspiratory flow is negative), so no sign flip is needed here."""
    insp = breath["inspiration"]
    exp = breath["expiration"]
    v_tlc = float(_arr(insp["volume"])[-1])
    v = np.maximum.accumulate(v_tlc - _arr(exp["volume"]))
    flow = _arr(exp["flow"])
    return v, flow, v_tlc


def _pef(flow: np.ndarray, fs: float) -> tuple[float, int]:
    """``(PEF, index)`` over a window of at least 10 ms (a single-sample spike must
    not win) -- a centred moving average of the flow trace, the max of THAT. The
    returned PEF value is the SMOOTHED maximum itself (self-review finding: an
    earlier version returned ``flow[ix]``, the raw sample at the window's centre,
    which could still equal a spike if the spike happened to sit there -- the whole
    point of smoothing first). ``index`` is the nearest REAL sample to that centre,
    for the back-extrapolation tangent below to anchor its point on a real
    ``(t, v)`` sample while using the smoothed value as its slope."""
    win = max(1, int(round(0.01 * fs))) if fs > 0 else 1
    if win <= 1 or flow.size < win:
        ix = int(np.argmax(flow))
        return float(flow[ix]), ix
    kernel = np.ones(win) / win
    smoothed = np.convolve(flow, kernel, mode="valid")
    smoothed_ix = int(np.argmax(smoothed))
    ix = min(smoothed_ix + win // 2, flow.size - 1)
    return float(smoothed[smoothed_ix]), ix


def _back_extrapolate(t: np.ndarray, v: np.ndarray, pef: float, pef_ix: int) -> tuple[float, float]:
    """``(t0, BEV)`` -- ATS/ERS 2019's back-extrapolation: the tangent line through
    the PEF sample (slope ``pef``, point ``(t[pef_ix], v[pef_ix])``) extrapolated
    back to where it crosses ``v=0``; BEV is ``v`` interpolated on the REAL
    (measured) volume-time curve at ``t0`` (not the tangent), since ``v`` is 0 at the
    true start of the recorded expiration by construction. ``t0`` never precedes the
    array's own first sample."""
    if pef <= 0:
        return float(t[0]), 0.0
    t0 = float(t[pef_ix] - v[pef_ix] / pef)
    t0 = max(t0, float(t[0]))
    bev = float(np.interp(t0, t, v))
    return t0, bev


def _fev1(t: np.ndarray, v: np.ndarray, t0: float, fvc: float) -> float:
    """Volume exhaled in the first second after ``t0``. A manoeuvre that exhausts
    its whole FVC before 1 s has elapsed clips to ``fvc`` rather than extrapolating
    past the last real sample."""
    t_rel = t - t0
    if t_rel[-1] < 1.0:
        return fvc
    return float(np.interp(1.0, t_rel, v))


def _eofe_ok(t: np.ndarray, v: np.ndarray, fet: float) -> bool:
    """ATS/ERS 2019's end-of-forced-expiration criterion: FET >= 15 s, OR less than
    25 mL of volume change over the LAST second of the curve."""
    if fet >= 15.0:
        return True
    mask = t >= (t[-1] - 1.0)
    if not mask.any():
        return False
    window = v[mask]
    return float(window.max() - window.min()) < 0.025


def _peak_in_flow_after(breath, tidal_breaths) -> float | None:
    """Peak inspiratory flow of the file's own NEXT (lowest-numbered, higher than
    this breath's own number) tidal breath -- the "subsequent inspiration" the
    ticket describes, a rapid near-maximal re-inflation some FVC/MVV protocols
    record right after the forced exhalation. ``None`` (never guessed) unless that
    breath's own volume excursion reaches at least 90% of ``fvc`` (the caller's own
    threshold check, done by the caller since it also needs ``fvc``) -- here we
    simply hand back the CANDIDATE breath's own excursion and peak flow; the caller
    applies the 0.9*fvc gate."""
    following = sorted((b for b in tidal_breaths if b["number"] > breath["number"]),
                       key=lambda b: b["number"])
    if not following:
        return None
    return following[0]


def fvc_metrics(breath, tidal_breaths, fs: float) -> dict:
    """The Manoeuvres-sheet numeric fields for ONE typed ``fvc``/``ic_fvc`` breath.
    A pure function of this ONE breath plus the file's OWN tidal breaths (for the
    subsequent-inspiration peak flow only) -- never of another file, run, or
    cohort-level aggregate, matching ``manoeuvres.extract``'s own acceptance
    criterion for the same reason.

    Returns ``fvc``, ``fev1``, ``fev1_fvc``, ``fvc_bev``, ``fvc_fet``,
    ``fvc_eofe_ok``, ``mfvl_peak_ex_flow`` (PEF), ``mfvl_peak_in_flow`` (``nan``
    when no eligible following breath exists), ``quality`` (``BEV_HIGH`` when BEV
    exceeds ``max(0.1 L, 5% of FVC)``, plus whatever :func:`manoeuvres.
    validate_fvc_manoeuvre` already flagged -- merged by the caller, not here, so
    this function never has to import ``manoeuvres`` back), and two INTERNAL,
    underscore-prefixed fields (``_v_tlc``, popped before the row ever reaches the
    written Manoeuvres sheet) that :func:`apply_tlc_consistency` reads."""
    exp = breath["expiration"]
    t = _arr(exp["time"])
    v, flow, v_tlc = mefv_curve(breath)
    fvc = float(v[-1])

    if fvc <= 0 or t.size < 2:
        return {
            "fvc": float("nan"), "fev1": float("nan"), "fev1_fvc": float("nan"),
            "fvc_bev": float("nan"), "fvc_fet": float("nan"), "fvc_eofe_ok": False,
            "mfvl_peak_ex_flow": float("nan"), "mfvl_peak_in_flow": float("nan"),
            "quality": [], "_v_tlc": v_tlc,
        }

    fs_curve = fs if fs and fs > 0 else (t.size - 1) / (t[-1] - t[0]) if t[-1] > t[0] else 1.0
    pef, pef_ix = _pef(flow, fs_curve)
    t0, bev = _back_extrapolate(t, v, pef, pef_ix)
    fev1 = _fev1(t, v, t0, fvc)
    fev1_fvc = fev1 / fvc if fvc > 0 else float("nan")
    fvc_fet = float(t[-1] - t0)
    eofe_ok = _eofe_ok(t, v, fvc_fet)

    quality: list[str] = []
    bev_threshold = max(0.1, 0.05 * fvc)
    if bev > bev_threshold:
        quality.append("BEV_HIGH")

    peak_in_flow = float("nan")
    following = _peak_in_flow_after(breath, tidal_breaths)
    if following is not None:
        insp = following["inspiration"]
        excursion = float(_arr(insp["volume"])[-1] - _arr(insp["volume"])[0])
        if excursion >= 0.9 * fvc:
            peak_in_flow = -float(_arr(insp["flow"]).min())

    return {
        "fvc": fvc, "fev1": fev1, "fev1_fvc": fev1_fvc, "fvc_bev": bev,
        "fvc_fet": fvc_fet, "fvc_eofe_ok": bool(eofe_ok),
        "mfvl_peak_ex_flow": pef, "mfvl_peak_in_flow": peak_in_flow,
        "quality": quality, "_v_tlc": v_tlc,
    }


def apply_to_row(row: dict, breath, tidal_breaths, fs: float) -> None:
    """Mutates an already-extracted ``manoeuvres.extract()`` row IN PLACE for a
    ``fvc``/``ic_fvc`` kind: merges :func:`fvc_metrics`'s numeric fields in, and
    extends (never replaces) ``row['quality']`` with any NEW flag
    :func:`fvc_metrics` raised -- the SAME "extend, skip duplicates" pattern
    ``manoeuvres._ic_fields`` already uses for ``validate_fvc_manoeuvre``'s own
    flags. Called from ``core.pipeline.run_batch``'s main loop AND its external-
    reference forepass, right after ``manoeuvreslib.extract`` -- both call sites
    need the SAME merge, so it lives here once rather than being duplicated at each
    site (the same "one shared function, not two drifting copies" precedent
    ``core.analysis.references.lookup_manoeuvre``'s own docstring already states)."""
    if row.get("kind") not in ("fvc", "ic_fvc"):
        return
    extra = fvc_metrics(breath, tidal_breaths, fs)
    extra_quality = extra.pop("quality", [])
    quality = row.setdefault("quality", [])
    quality.extend(f for f in extra_quality if f not in quality)
    row.update(extra)


def apply_tlc_consistency(manoeuvres: dict, ic_cfg=None, tol: float = 0.15) -> None:
    """Second pass (mirrors ``manoeuvres.apply_repeatability``'s own timing: run
    once per file AFTER every typed breath's own row is in hand). For every
    ``fvc``/``ic_fvc`` row that carries an internal ``_v_tlc`` (:func:`fvc_metrics`)
    and this SAME file's own ``ic``/``ic_fvc`` breaths (never a cross-file IC
    reference -- comparing two DIFFERENT recordings' TLC estimates would conflate a
    real physiological disagreement with the two recordings simply not sharing a
    volume zero), sets ``mfvl_tlc_consistency = V_TLC_fvc - V_TLC_ic`` (the IC
    breaths' own ``vol_ic + ic_eelv_pre``, mean over however many exist) and flags
    ``not_from_tlc`` when the magnitude exceeds ``tol`` (default 0.15 L -- the
    ticket's own placeholder, same K-035 provenance as every other threshold here).

    ``ic_cfg`` (``None`` tolerated for a caller with no IC settings at all -- every
    IC breath then simply lacks a ``quality`` list to check and is never excluded)
    is used for TWO self-review findings: (1) a ``reject_flags``-flagged IC breath
    (``LOW_EFFORT`` by default) is excluded from the average -- the SAME
    disqualifying rule :func:`resolve_same_file_ic_op` already applies for its own
    ic_op resolution, so a rejected attempt cannot drag this comparison's baseline
    any more than it can drag that one; (2) a non-finite ``vol_ic``/``ic_eelv_pre``
    on any ONE IC row no longer silently poisons ``np.mean`` into NaN for every FVC
    row in the file (:func:`_finite` filters each contributing value, not just the
    dict-key presence check the previous version used).

    Always pops the internal ``_v_tlc`` field from every row before returning
    (whether or not a comparison was possible) -- it must never reach the written
    Manoeuvres sheet (``core.results.build_manoeuvre_table`` copies every key of a
    row verbatim)."""
    reject_flags = set(getattr(ic_cfg, "reject_flags", ()) or ())
    ic_vtlcs = [row["vol_ic"] + row["ic_eelv_pre"] for row in manoeuvres.values()
               if row.get("kind") in ("ic", "ic_fvc")
               and _finite(row.get("vol_ic")) and _finite(row.get("ic_eelv_pre"))
               and not (set(row.get("quality", [])) & reject_flags)]
    ic_vtlc = float(np.mean(ic_vtlcs)) if ic_vtlcs else None

    for row in manoeuvres.values():
        v_tlc_fvc = row.pop("_v_tlc", None)
        if row.get("kind") not in ("fvc", "ic_fvc") or v_tlc_fvc is None or ic_vtlc is None:
            continue
        diff = v_tlc_fvc - ic_vtlc
        row["mfvl_tlc_consistency"] = diff
        if abs(diff) > tol:
            quality = row.setdefault("quality", [])
            if "not_from_tlc" not in quality:
                quality.append("not_from_tlc")


#: keys attach() always fills, IC-independent (a single PEF/peak-in-flow ratio,
#: never placed against the MEFV's own volume axis) -- the ticket's own acceptance
#: criterion: "uden IC-reference er max_*_pct_mfvl_peak udfyldt, resten NaN".
_ALWAYS_KEYS = ("max_ex_flow_pct_mfvl_peak", "max_in_flow_pct_mfvl_peak")
#: keys requiring a resolved IC operating point AND full coverage of this breath's
#: own operating range by the MEFV curve's own domain -- NaN'd together the moment
#: EITHER condition fails (a poor placement is exactly as unusable as no placement).
_PLACEMENT_KEYS = ("efl_pct", "efl_present", "ex_flow_pct_mfvl_max", "in_flow_pct_mfvl_max",
                  "te_min_mfvl", "ve_cap", "ve_pct_cap", "ve_reserve_pct")
#: keys requiring a resolved IC operating point but NOT MEFV-domain coverage (MVV
#: is independent of WHERE on the curve this breath sits).
_IC_ONLY_KEYS = ("mvv_est", "ve_pct_mvv", "br_mvv_pct")


def tidal_mfvl_ext(tidal_breath, *, mefv_v: np.ndarray | None, mefv_flow: np.ndarray | None,
                   ic_op: float | None, pef: float, peak_in_flow: float,
                   fev1_used: float | None, mfvl_cfg, mvv_override: float | None = None,
                   coverage_floor: float = 100.0) -> dict:
    """The ``breath['mfvl_ext']`` dict for ONE tidal breath. ``mefv_v``/``mefv_flow``
    (``None`` when no ``fvc`` reference resolved for this file at all -- distinct
    from a resolved-but-narrow-domain curve) and ``ic_op`` (``None`` when no ``ic``
    reference resolved) are the file-level, already-resolved pieces :func:`attach`
    hands every tidal breath identically; this function does no reference
    resolution of its own, matching ``core.analysis.lungvol.compute_breath_olv``'s
    own "the pure per-breath formula" precedent.

    ``coverage_floor`` (default 100.0, i.e. every column requires FULL coverage --
    the ticket's own wording, "NaN + one notice when efl_coverage_pct < 100") is a
    parameter rather than a hard-coded literal purely so a test can probe the
    boundary without constructing a breath that hits it exactly.

    Self-review finding: ``fev1_source`` (the ticket's own text column) does NOT
    live here, even though it is conceptually part of this dict. Every key
    returned by this function is joined into ``breaths_table`` by
    ``core.results.build_breath_table`` BEFORE that function's own
    ``mechs.mean()`` reduction (the same join point ``breath['wob']`` already
    uses) -- a STRING column there breaks that reduction outright
    (``TypeError: Cannot perform reduction 'mean' with string dtype``) for the
    whole file, not just this column. :func:`attach` instead sets
    ``fev1_source`` directly on ``breaths_table``/``average_row`` AFTER
    ``build_breath_table`` has already run, the same POST-hoc column-assignment
    pattern ``core.analysis.references.attach``'s own ``ic_ref_source`` (also
    text) already uses for exactly this reason."""
    out: dict = {}
    exp = tidal_breath["expiration"]
    insp = tidal_breath["inspiration"]
    mech = tidal_breath.get("mechanics", {})

    ex_flow = _arr(exp["flow"])
    ex_peak = float(ex_flow.max()) if ex_flow.size else float("nan")
    in_flow = _arr(insp["flow"])
    in_peak = -float(in_flow.min()) if in_flow.size else float("nan")

    out["max_ex_flow_pct_mfvl_peak"] = (
        100.0 * ex_peak / pef if pef == pef and pef > 0 else float("nan"))
    out["max_in_flow_pct_mfvl_peak"] = (
        100.0 * in_peak / peak_in_flow if peak_in_flow == peak_in_flow and peak_in_flow > 0
        else float("nan"))

    if ic_op is None or mefv_v is None or mefv_v.size == 0:
        for key in _PLACEMENT_KEYS + _IC_ONLY_KEYS:
            out[key] = float("nan")
        out["efl_coverage_pct"] = float("nan")
        return out

    v_exp = _arr(exp["volume"])
    vol_endexp_b = float(v_exp[-1])
    v_below_tlc = ic_op - (v_exp - vol_endexp_b)
    # Self-review finding: mech['vt'] (the whole BREATH's own max-min, inspiration
    # included) previously fed both the efl_pct denominator and the te_min/ve_cap
    # integration range -- with volume drift, or plain sample-to-sample noise, that
    # can differ from the EXPIRATION's own excursion this function actually
    # measures everything else against (v_below_tlc/dv/coverage are ALL built from
    # v_exp alone). Using the measured expiratory excursion here instead keeps
    # every quantity in this function self-consistent with what was actually
    # covered/measured, rather than mixing in a value from elsewhere in the breath.
    vt_exp = float(v_exp.max() - v_exp.min())

    domain_lo, domain_hi = float(mefv_v[0]), float(mefv_v[-1])
    dv = np.abs(np.diff(v_below_tlc))
    in_domain = (v_below_tlc[:-1] >= domain_lo) & (v_below_tlc[:-1] <= domain_hi)
    covered = float(np.sum(dv[in_domain]))
    total = float(np.sum(dv))
    efl_coverage_pct = 100.0 * covered / total if total > 0 else 0.0
    out["efl_coverage_pct"] = efl_coverage_pct

    if efl_coverage_pct < coverage_floor:
        for key in _PLACEMENT_KEYS:
            out[key] = float("nan")
    else:
        mefv_at = np.interp(v_below_tlc, mefv_v, mefv_flow, left=mefv_flow[0], right=mefv_flow[-1])
        limited = ex_flow >= mefv_at * (1.0 - mfvl_cfg.efl_rel_tol) - mfvl_cfg.efl_abs_tol_lps
        # Self-review finding: normalising by mech['vt'] (a net excursion) against a
        # numerator built from |dv| (a PATH length -- sample-to-sample volume noise,
        # e.g. cardiogenic oscillation, adds to it without adding to vt) could push
        # efl_pct over 100%. Normalising by `total` (the SAME path length the
        # numerator and efl_coverage_pct are already built from) keeps efl_pct
        # bounded to [0, 100] by construction (limited*dv <= dv always).
        efl_pct = 100.0 * float(np.sum(limited[:-1].astype(float) * dv) / total) if total > 0 else float("nan")
        out["efl_pct"] = efl_pct
        out["efl_present"] = bool(efl_pct == efl_pct and efl_pct >= mfvl_cfg.efl_present_min_pct)

        range_mask = (mefv_v >= v_below_tlc.min()) & (mefv_v <= v_below_tlc.max())
        local_max = float(mefv_flow[range_mask].max()) if range_mask.any() else float("nan")
        out["ex_flow_pct_mfvl_max"] = (
            100.0 * ex_peak / local_max if local_max == local_max and local_max > 0 else float("nan"))
        out["in_flow_pct_mfvl_max"] = out["max_in_flow_pct_mfvl_peak"]

        vol_irv_b = ic_op - vt_exp
        te_min = _te_min(mefv_v, mefv_flow, vol_irv_b, ic_op)
        out["te_min_mfvl"] = te_min
        ti_ttot = float(mech.get("ti_ttot", float("nan")))
        ve = float(mech.get("ve", float("nan")))
        if te_min == te_min and te_min > 0 and ti_ttot == ti_ttot:
            ve_cap = 60.0 * vt_exp * (1.0 - ti_ttot) / te_min
            out["ve_cap"] = ve_cap
            out["ve_pct_cap"] = 100.0 * ve / ve_cap if ve_cap > 0 else float("nan")
            out["ve_reserve_pct"] = 100.0 * (ve_cap - ve) / ve_cap if ve_cap > 0 else float("nan")
        else:
            out["ve_cap"] = out["ve_pct_cap"] = out["ve_reserve_pct"] = float("nan")

    ve = float(mech.get("ve", float("nan")))
    if mvv_override is not None and mvv_override > 0:
        mvv_est = mvv_override
    elif fev1_used is not None and fev1_used == fev1_used and fev1_used > 0:
        mvv_est = fev1_used * mfvl_cfg.mvv_fev1_multiplier
    else:
        mvv_est = None
    if mvv_est is not None and mvv_est > 0:
        out["mvv_est"] = mvv_est
        out["ve_pct_mvv"] = 100.0 * ve / mvv_est if ve == ve else float("nan")
        out["br_mvv_pct"] = 100.0 * (1.0 - ve / mvv_est) if ve == ve else float("nan")
    else:
        out["mvv_est"] = out["ve_pct_mvv"] = out["br_mvv_pct"] = float("nan")

    return out


def _te_min(mefv_v: np.ndarray, mefv_flow: np.ndarray, lo: float, hi: float,
           floor: float = 0.05) -> float:
    """``∫ dv / mefv(v)`` over ``[lo, hi]`` (Johnson 1995/1999's minimal expiratory
    time this envelope would allow at exactly this breath's own tidal excursion) --
    ``nan`` when the range is empty, is not FULLY inside the curve's own domain (not
    merely clipped to whatever part does fall inside it), or the envelope drops
    below ``floor`` (0.05 L/s -- division by a near-zero flow would blow the
    integral up to a physically meaningless value) anywhere inside ``[lo, hi]``."""
    if hi <= lo:
        return float("nan")
    # Self-review finding: an earlier version silently CLIPPED [lo, hi] to the
    # curve's own domain instead of refusing outright -- so a range that only
    # PARTLY fell outside the curve still returned a number, one integrated over
    # less than the breath's own actual excursion, while the caller's separate
    # efl_coverage_pct check (built from a slightly different quantity, the
    # breath's own last-minus-first sample rather than max-minus-min) could still
    # read 100%. NaN outright the moment [lo, hi] is not FULLY inside the domain,
    # rather than trusting the two checks to always agree.
    if lo < float(mefv_v[0]) or hi > float(mefv_v[-1]):
        return float("nan")
    grid = np.linspace(lo, hi, 200)
    flow_grid = np.interp(grid, mefv_v, mefv_flow)
    if np.any(flow_grid < floor):
        return float("nan")
    # Trapezoidal rule, hand-rolled rather than np.trapz/np.trapezoid: the former
    # is removed as of numpy 2.0, the latter does not exist before it, and
    # pyproject.toml supports numpy>=1.24 -- neither name is safe to call across
    # that whole range. `grid` is uniform (np.linspace), so a constant step works.
    inv = 1.0 / flow_grid
    dx = (hi - lo) / (grid.size - 1)
    return float(np.sum((inv[:-1] + inv[1:]) / 2.0) * dx)


def _envelope(curves: list[tuple[np.ndarray, np.ndarray, float]]) -> tuple[np.ndarray, np.ndarray, float]:
    """``source='envelope'``: the per-volume MAXIMUM flow across every resolved FVC
    attempt (each interpolated onto the union of every curve's own volume samples
    first) -- Johnson 1999's own composite-MEFV construction. ``v_tlc`` is the
    LARGEST of the attempts' own (a subject reaching a higher TLC on one attempt
    means that attempt's own axis covers the others', never the reverse)."""
    v_tlc = max(c[2] for c in curves)
    v_grid = np.unique(np.concatenate([c[0] for c in curves]))
    flow_grid = np.full_like(v_grid, np.nan)
    for v, flow, _ in curves:
        # self-review finding: extrapolating a SHORTER attempt flat past its own
        # end (the previous `right=flow[-1]`) let its near-RV tail value win the
        # per-volume max over a volume range it never actually reached, biasing
        # the composite curve upward there. NaN outside a curve's own domain
        # instead -- np.fmax treats NaN as "this curve has no opinion here" and
        # simply ignores it, so the max is only ever taken among attempts that
        # genuinely cover that volume.
        flow_grid = np.fmax(flow_grid, np.interp(v_grid, v, flow, left=np.nan, right=np.nan))
    return v_grid, flow_grid, v_tlc


def resolve_same_file_curve(fr_manoeuvres: dict, breaths: dict, mfvl_cfg
                            ) -> tuple[np.ndarray, np.ndarray, float, list[dict]] | None:
    """This file's OWN MEFV envelope, built from its typed ``fvc``/``ic_fvc``
    breaths (see the module docstring for why this is SAME-FILE only). ``"single"``
    (the default) picks the attempt with the LARGEST ``fvc`` -- ATS/ERS 2019's own
    "report the largest FVC/FEV1 across acceptable attempts" convention, reused
    here for which SINGLE curve to compare tidal breaths against.
    ``"envelope"`` (only when more than one attempt resolved -- with exactly one,
    it is identical to "single" by construction) takes the per-volume maximum
    across all of them (:func:`_envelope`). Returns ``None`` when this file has no
    usable typed FVC breath at all.

    The fourth element is the list of Manoeuvres row(s) that fed the returned
    curve (one for "single", every resolved attempt for "envelope") -- the
    caller (:func:`attach`) reads their OWN already-smoothed ``mfvl_peak_ex_flow``
    from this list rather than re-deriving a PEF from the curve's raw samples
    (self-review finding: ``mefv_flow.max()`` is a single-SAMPLE maximum of a
    curve built by ``np.interp``/``np.fmax``, exactly the kind of spike
    :func:`_pef`'s own smoothing exists to reject -- using it here silently
    disagreed with the smoothed PEF already written to the Manoeuvres sheet)."""
    candidates = [(no, row) for no, row in fr_manoeuvres.items()
                 if row.get("kind") in ("fvc", "ic_fvc") and _finite(row.get("fvc"))]
    if not candidates:
        return None
    if mfvl_cfg.source == "envelope" and len(candidates) > 1:
        curves = [mefv_curve(breaths[no]) for no, _ in candidates]
        v, flow, v_tlc = _envelope(curves)
        return v, flow, v_tlc, [row for _no, row in candidates]
    best_no, best_row = max(candidates, key=lambda item: item[1]["fvc"])
    v, flow, v_tlc = mefv_curve(breaths[best_no])
    return v, flow, v_tlc, [best_row]


def resolve_same_file_ic_op(fr_manoeuvres: dict, ic_cfg) -> float | None:
    """This file's own resolved IC operating point, using ``eelv_tracking='none'``
    semantics (``ic_op = vol_ic_ref``, held constant across the file -- see the
    module docstring for why this ticket does not attempt
    ``core.analysis.lungvol``'s full within-file EELV-tracking arithmetic here):
    the ``ic_cfg.aggregate`` (mean/median) of ``vol_ic`` over this file's OWN
    ``ic``/``ic_fvc`` breaths that are not flagged with one of
    ``ic_cfg.reject_flags`` -- the SAME disqualifying rule
    ``manoeuvres.apply_repeatability``'s own leave-one-out group already applies.
    ``None`` when no eligible IC breath exists in this file."""
    vols = [row["vol_ic"] for row in fr_manoeuvres.values()
           if row.get("kind") in ("ic", "ic_fvc") and _finite(row.get("vol_ic"))
           and not (set(row.get("quality", [])) & set(ic_cfg.reject_flags))]
    if not vols:
        return None
    return float(np.median(vols)) if ic_cfg.aggregate == "median" else float(np.mean(vols))


def _subject_fev1(settings, filename: str) -> float | None:
    group = group_key(filename, settings)
    for subj in settings.input.subjects:
        if subj.key == group and subj.fev1_l is not None:
            return float(subj.fev1_l)
    return None


def _subject_mvv(settings, filename: str) -> float | None:
    group = group_key(filename, settings)
    for subj in settings.input.subjects:
        if subj.key == group and subj.mvv_lpm is not None:
            return float(subj.mvv_lpm)
    return None


def attach(*, fr_manoeuvres: dict, breaths: dict, tidal_breaths: list, filename: str,
          settings, s) -> tuple[str | None, str | None]:
    """Stamps ``breath['mfvl_ext']`` on every tidal (non-ignored) breath of ONE
    file, for ``core.results.build_breath_table`` to join in exactly like
    ``breath['wob']``. Called from ``core.pipeline.run_batch``'s main loop, AFTER
    ``manoeuvreslib.apply_repeatability``/:func:`apply_tlc_consistency` and BEFORE
    ``build_breath_table`` is called for this same file (see the module docstring
    for why this cannot be a `references.attach`-style post-loop pass).

    Returns ``(notice, fev1_source)``. ``notice`` is a single, per-FILE string
    (never per-breath) when this file has a resolvable ``fvc`` reference but no
    ``ic`` reference (only the two IC-independent peak-vs-peak ratios are filled
    -- the ticket's own acceptance criterion), or ``None`` when either no ``fvc``
    reference resolves at all (no ``mfvl_ext`` is set on any breath -- there is
    nothing to report on a file that never uses this family) or an IC reference
    DID resolve (nothing to warn about). ``fev1_source`` (``'spirometry'`` |
    ``'recorded'`` | ``None``) is the CALLER's job to write onto
    ``breaths_table``/``average_row`` itself, AFTER ``build_breath_table`` runs
    (see :func:`tidal_mfvl_ext`'s own docstring for why a text column cannot go
    through that function's dict)."""
    mfvl_cfg = s.processing.mfvl
    ic_cfg = s.processing.lung_volume.ic

    curve = resolve_same_file_curve(fr_manoeuvres, breaths, mfvl_cfg)
    if curve is None:
        return None, None
    mefv_v, mefv_flow, _v_tlc, curve_rows = curve
    # Self-review finding: mefv_flow.max() is a single-SAMPLE maximum of the
    # (interpolated) composite curve and can equal a spike _pef's own smoothing
    # exists to reject -- use the participating attempt(s)' OWN already-smoothed
    # mfvl_peak_ex_flow instead, so this agrees with what the Manoeuvres sheet
    # itself reports (the same value for "single"; the max across attempts, a
    # composite PEF, for "envelope").
    pef_candidates = [row.get("mfvl_peak_ex_flow") for row in curve_rows
                      if _finite(row.get("mfvl_peak_ex_flow"))]
    pef = max(pef_candidates) if pef_candidates else float("nan")
    peak_in_flow_candidates = [row.get("mfvl_peak_in_flow") for row in fr_manoeuvres.values()
                               if row.get("kind") in ("fvc", "ic_fvc")
                               and _finite(row.get("mfvl_peak_in_flow"))]
    peak_in_flow = max(peak_in_flow_candidates) if peak_in_flow_candidates else float("nan")

    ic_op = resolve_same_file_ic_op(fr_manoeuvres, ic_cfg)

    # Beslutning 26-09-2026 (plan §11): spirometry FEV1 is preferred over the
    # derived one whenever a subject entry supplies it; the derived value (the
    # largest across this file's own resolved FVC attempts, ATS/ERS 2019's own
    # "report the largest" convention) is only the fallback. fev1_source records
    # WHICH one was actually used (the ticket's own explicit acceptance criterion).
    fev1_used = _subject_fev1(settings, filename)
    fev1_source = "spirometry" if fev1_used is not None else None
    if fev1_used is None:
        for row in fr_manoeuvres.values():
            if row.get("kind") in ("fvc", "ic_fvc") and _finite(row.get("fev1")):
                fev1_used = row["fev1"] if fev1_used is None else max(fev1_used, row["fev1"])
        if fev1_used is not None:
            fev1_source = "recorded"
    mvv_override = _subject_mvv(settings, filename)

    notice = None
    if ic_op is None:
        notice = ("no IC reference resolves for this file -- only "
                 "max_ex_flow_pct_mfvl_peak/max_in_flow_pct_mfvl_peak are filled, "
                 "every placement-dependent MFVL column is NaN")

    partial_coverage = False
    for breath in tidal_breaths:
        breath["mfvl_ext"] = tidal_mfvl_ext(
            breath, mefv_v=mefv_v, mefv_flow=mefv_flow, ic_op=ic_op, pef=pef,
            peak_in_flow=peak_in_flow, fev1_used=fev1_used, mvv_override=mvv_override,
            mfvl_cfg=mfvl_cfg)
        coverage = breath["mfvl_ext"].get("efl_coverage_pct")
        if coverage is not None and coverage == coverage and coverage < 100.0:
            partial_coverage = True

    # Ticket's own acceptance criterion: "placement-dependent columns NaN + ONE
    # notice when efl_coverage_pct < 100" -- once per FILE (never per breath, the
    # same convention every other notice in this codebase follows), and only when
    # an IC reference DID resolve (the no-IC notice above already explains why
    # those same columns are NaN; a second, overlapping notice would be redundant).
    if notice is None and partial_coverage:
        notice = ("at least one tidal breath's own operating range falls partly "
                 "outside this file's MEFV envelope (efl_coverage_pct < 100) -- "
                 "efl_pct/efl_present/ex_flow_pct_mfvl_max/in_flow_pct_mfvl_max/"
                 "te_min_mfvl/ve_cap/ve_pct_cap/ve_reserve_pct are NaN for that breath")

    return notice, fev1_source


# --------------------------------------------------------------------------- #
# Tidal loops placed inside the MFVL (figure geometry; no new calculation)
# --------------------------------------------------------------------------- #
#: ``processing.breath_types`` kinds that carry a forced vital capacity.
_FVC_KINDS = ("fvc", "ic_fvc")
#: Points a breath is resampled to for the average tidal loop.
_MEAN_LOOP_POINTS = 200


def fvc_typed_in_settings(settings) -> bool:
    """``True`` when ``settings`` declares at least one breath typed ``fvc``/``ic_fvc``:
    the settings-only test for "this analysis uses the FVC family", so a figure that needs
    an MEFV curve can be planned before any data is loaded. It is a ceiling, not a promise
    -- a file without such a breath still gets no figure (:func:`placed_tidal_loops`
    returns ``None`` for it). Only breaths typed in the file that is analysed count,
    because :func:`attach` builds the curve from the file's own typed breaths."""
    return any(getattr(e, "kind", None) in _FVC_KINDS
               for e in getattr(settings.processing, "breath_types", ()))


def _mean_loop(loops: list[tuple[np.ndarray, np.ndarray]]):
    """Point-wise mean of several ``(x, flow)`` loops, each resampled onto
    ``_MEAN_LOOP_POINTS`` points by its own sample index (breath fraction), so breaths of
    different length average without a common time base. A display construct only."""
    if not loops:
        return None
    grid = np.linspace(0.0, 1.0, _MEAN_LOOP_POINTS)
    xs, fs_ = [], []
    for x, flow in loops:
        if x.size < 2:
            continue
        frac = np.linspace(0.0, 1.0, x.size)
        xs.append(np.interp(grid, frac, x))
        fs_.append(np.interp(grid, frac, flow))
    if not xs:
        return None
    return np.mean(xs, axis=0), np.mean(fs_, axis=0)


def placed_tidal_loops(breaths, manoeuvres, mfvl_cfg, ic_cfg) -> dict | None:
    """Everything a flow-volume figure needs to draw this file's tidal loops inside its
    own MFVL, or ``None`` when the file has no usable typed FVC breath (or no tidal
    breath to draw against it).

    Reuses the resolved pieces :func:`attach` already builds the numbers from --
    :func:`resolve_same_file_curve` (which curve, ``source='single'|'envelope'``) and
    :func:`resolve_same_file_ic_op` -- so the picture can never place a loop differently
    from the ``efl_pct`` column. The x axis is "volume below TLC" (0 at TLC, the MEFV
    curve's own axis): ``x(t) = ic_op - (V(t) - vol_endexp)`` for a tidal breath, so the
    end of expiration (EELV) sits at ``ic_op`` and the end of inspiration (EILV) at
    ``ic_op - vt``.

    Returns a dict: ``mefv_v``/``mefv_flow`` (the expiratory envelope), ``v_tlc``,
    ``ic_op`` (``None`` without an IC reference: loops cannot be anchored to the TLC axis,
    so ``loops`` is then empty and only the envelope is drawn), ``loops`` (one
    ``(x, flow)`` per non-ignored tidal breath), ``mean`` (``(x, flow)`` or ``None``),
    ``eelv``/``eilv`` (mean end-expiratory/end-inspiratory position on the x axis, or
    ``None``)."""
    if not manoeuvres or not breaths:
        return None
    curve = resolve_same_file_curve(manoeuvres, breaths, mfvl_cfg)
    if curve is None:
        return None
    mefv_v, mefv_flow, v_tlc, _rows = curve
    tidal = [b for b in breaths.values() if not b.get("ignored")]
    if not tidal:
        return None
    ic_op = resolve_same_file_ic_op(manoeuvres, ic_cfg)
    loops: list[tuple[np.ndarray, np.ndarray]] = []
    eelv = eilv = None
    if ic_op is not None:
        for b in tidal:
            vol = _arr(b["volume"])
            flow = _arr(b["flow"])
            if vol.size < 2 or vol.size != flow.size:
                continue
            vol_endexp = float(_arr(b["expiration"]["volume"])[-1])
            loops.append((ic_op - (vol - vol_endexp), flow))
        if loops:
            eelv = float(ic_op)
            eilv = float(np.mean([x.min() for x, _f in loops]))
    return {"mefv_v": mefv_v, "mefv_flow": mefv_flow, "v_tlc": float(v_tlc),
            "ic_op": ic_op, "loops": loops, "mean": _mean_loop(loops),
            "eelv": eelv, "eilv": eilv}
