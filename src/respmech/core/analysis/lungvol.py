"""Operating lung volumes (M-36): per-tidal-breath EELV/EILV/IRV, derived from a
file's already-resolved IC reference (``core.analysis.references.attach``, M-35), and
per-file summary volumes (TLC, VC, the change in IC/EELV against a baseline file).

Qt-free, numpy/pandas only. Sign convention throughout: a RISE in end-expiratory lung
volume (EELV) means the lungs are hyperinflating between breaths -- so it SHRINKS the
inspiratory capacity actually available for that breath (``ic_op``), which is why
``ic_op = vol_ic_ref - d_eelv``, never ``+`` (an earlier draft of this ticket had the
sign backwards; the analytical test in ``tests/unit/test_lungvol.py`` pins the
direction).

Two EELV DATA are reported side by side, never merged into one column family
(Emil's decision 26-09-2026, see ``docs/beslutninger.md``):

* **RV-anchored, primary**: ``vol_eelv = vc_src - ic_op`` -- the volume above residual
  volume at end-expiration, i.e. ERV by definition (there is no separate ``vol_erv``
  column). ``vc_src`` is ``input.subjects``' own ``vc_l`` for this file's group, else
  the linked FVC reference's own ``fvc`` value (not yet ever populated --
  ``core.analysis.manoeuvres.extract`` computes no numeric FVC value until M-42's
  ``mfvl.py`` lands; this module already reads whichever key is there, so the moment
  M-42 adds it, the fallback starts working with no change here). This is the primary
  family because it needs no plethysmography (a spirometry-derived VC is far more often
  available than a measured TLC) -- a deliberate, PRACTICAL reason, not merely
  Emil's preference.
* **TLC-anchored, absolute, alongside**: ``vol_eelv_abs = tlc - ic_op``, present only
  when ``input.subjects`` names a TLC for this file's group. Missing TLC/VC is the
  ordinary case (most studies never measure either) and produces silent NaN in the
  corresponding family -- never a notice; a notice is reserved for the three
  physiologically-implausible-value checks below and for a within-file EELV-tracking
  request that could not actually track (see ``attach``'s own docstring).

``eelv_tracking`` (``processing.lung_volume.ic.eelv_tracking``, already declared and
validated by M-29's ``IcSettings``) decides whether ``ic_op`` is simply the file's
resolved reference IC unchanged across the whole file (``"none"``, the default -- the
ordinary "IC measured once, assumed constant" convention) or tracks each tidal
breath's own end-expiratory drift AWAY from the reference IC's own end-expiratory
level (``"within_file"`` -- same-file IC reference only; a cross-file comparison has
no shared volume zero to compare against, so it NaNs with one per-file notice instead
of silently reporting a meaningless number).
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from respmech.core.analysis import references as referenceslib
from respmech.core.summary import group_key

#: the eelv_tracking value that measures each tidal breath's own end-expiratory
#: level against the reference IC's own end-expiratory level, from the SAME file only.
_WITHIN_FILE = "within_file"


def _aggregate_field(result, ref, ic_cfg, field: str) -> tuple[float | None, int]:
    """``(aggregate, n)`` of ``field`` over ``ref.breaths`` that resolve (via
    ``references.lookup_manoeuvre``) and are not flagged with one of ``ic_cfg.reject_flags``
    -- the SAME two-part filter ``references._aggregate_ic`` applies to ``vol_ic``,
    generalised to any manoeuvre field (``ic_eelv_pre`` for within-file tracking,
    ``vol_ic`` for a ``baseline_ic`` link, ``fvc`` once M-42 populates it). A breath
    that resolves but simply has no value for ``field`` yet (``fvc`` today) is skipped
    like a missing one -- this is what lets the VC fallback below silently do nothing
    until M-42 exists, rather than raising or NaNing the whole file over it.

    Returns ``(None, 0)`` when nothing usable resolved -- the caller treats that
    exactly like an unresolved reference (NaN, and a notice only when the link was
    explicitly configured -- see ``attach``'s baseline/VC handling)."""
    vals: list[float] = []
    for b in dict.fromkeys(ref.breaths):
        row = referenceslib.lookup_manoeuvre(result, ref.file, b)
        if row is None:
            continue
        quality = row.get("quality") or []
        if set(quality) & set(ic_cfg.reject_flags):
            continue
        v = row.get(field)
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None, 0
    agg = float(np.median(vals)) if ic_cfg.aggregate == "median" else float(np.mean(vals))
    return agg, len(vals)


def _subject_for(settings, filename: str):
    """The ``core.settings.SubjectEntry`` for ``filename``'s group, or ``None`` --
    the SAME ``group_key`` a subject table is already keyed on (``input.subjects``'
    own docstring)."""
    group = group_key(filename, settings)
    for s in settings.input.subjects:
        if s.key == group:
            return s
    return None


def _resolve_tlc(subject) -> float | None:
    return float(subject.tlc_l) if subject is not None and subject.tlc_l is not None else None


def _resolve_vc(result, settings, filename: str, subject, ic_cfg) -> float | None:
    """``subjects.vc_l`` first (a formal spirometry value); else the linked ``fvc``
    reference's own ``fvc`` field, aggregated the same way an IC reference is (mean/
    median over accepted breaths) -- ``None`` for both today whenever no subject VC is
    entered, since ``manoeuvres.extract`` does not populate ``fvc`` until M-42."""
    if subject is not None and subject.vc_l is not None:
        return float(subject.vc_l)
    ref = referenceslib.resolve_reference(filename, "fvc", settings)
    if ref is None:
        return None
    agg, _n = _aggregate_field(result, ref, ic_cfg, "fvc")
    return agg


def _resolve_baseline_ic_ref(
        result, settings, filename: str, ic_cfg, all_names: list[str]
        ) -> tuple[float | None, bool, str | None]:
    """``(baseline vol_ic_ref, was_explicitly_configured, ambiguity_notice)`` for
    ``delta_ic``.

    Resolution order (``docs/REVERSE_ENGINEERING.md`` §5.14): an explicit/group
    ``baseline_ic`` link names specific breaths in some file, aggregated exactly like
    an ``ic`` reference (mean/median of ``vol_ic`` over accepted breaths -- ``ic``'s
    own aggregate rule, since a baseline is itself just another IC measurement). With
    no such link, this file's OWN ``group_key`` siblings are searched for a filename
    matching ``lung_volume.baseline_pattern``.

    **The CANDIDATE is chosen from ``all_names`` (the full matched set), never from
    ``result.ok_files`` (this run's own subset)** -- deliberately the same
    "decide from settings across the full set, not this run's subset" rule the
    column-family gate above already follows, so a subset run (a GUI single-file
    test, say) picks the SAME baseline file a full run would; only whether that
    file's `vol_ic_ref` is actually AVAILABLE (i.e. whether it was itself processed
    in THIS run) can differ, which is an honest "not resolved in this run" NaN, not a
    different decision about WHICH file is the baseline. The alphabetically first
    match's ALREADY-RESOLVED ``vol_ic_ref`` (M-35's own per-file scalar) is used
    directly, rather than re-aggregating specific breaths -- deliberately simpler
    than the explicit-link path, since a whole file matched by name is naturally
    "this file's own IC reference is the baseline", not a hand-picked breath subset.
    More than one sibling matching the pattern is reported via the third return value
    (never silently, unlike the ordinary "no match at all" case).

    The second element is ``True`` only when ``baseline_ic`` was explicitly named
    (settings-configured) -- an unresolved EXPLICIT link is a caution the caller turns
    into a notice; an ABSENT one (the common case: most single-session studies never
    set this at all) is silently ``None``, matching the "missing TLC/VC produces no
    notice" convention above."""
    ref = referenceslib.resolve_reference(filename, "baseline_ic", settings)
    if ref is not None:
        agg, _n = _aggregate_field(result, ref, ic_cfg, "vol_ic")
        return agg, True, None

    group = group_key(filename, settings)
    pattern = settings.processing.lung_volume.baseline_pattern
    try:
        rx = re.compile(pattern)
    except re.error:
        return None, False, None
    candidates = sorted(
        n for n in all_names
        if n != filename and group_key(n, settings) == group and rx.search(n))
    if not candidates:
        return None, False, None
    ambiguity = (
        f"{filename}: baseline_pattern {pattern!r} matches more than one sibling "
        f"({', '.join(candidates)}) — using {candidates[0]!r} (alphabetically first)"
        if len(candidates) > 1 else None)
    base_fr = result.ok_files.get(candidates[0])
    if (base_fr is None or base_fr.average_row is None
            or "vol_ic_ref" not in base_fr.average_row.columns):
        return None, False, ambiguity
    val = base_fr.average_row["vol_ic_ref"].iloc[0]
    return (float(val) if val == val else None), False, ambiguity


def compute_breath_olv(*, vol_ic_ref: float, vt: float, vol_endexp: float,
                       eelv_tracking: str = "none", ic_eelv_pre: float | None = None,
                       correct_trend: bool = False, tlc: float | None = None,
                       vc: float | None = None) -> dict:
    """Pure per-tidal-breath operating-lung-volume arithmetic -- the ONE formula this
    whole ticket exists to pin (see the module docstring for the sign convention and
    the two EELV datums).

    ``d_eelv``/``ic_op`` under ``eelv_tracking="within_file"``: ``ic_eelv_pre=None``
    means the resolved IC reference is cross-file (this module's caller only ever
    passes a real number here for a SAME-file reference -- see ``attach``) or simply
    unresolved; either way ``d_eelv``/``ic_op`` come back NaN, matching the ticket's
    "cross-file -> NaN + notice" rule (the notice itself is the caller's job, once per
    file, not this pure function's). ``correct_trend=True`` also forces NaN even with
    a same-file reference: the trend-correction pass (``core.compute``) subtracts the
    trough envelope so every ``vol_endexp`` sample sits at the SAME level by
    construction, which would silently report ``d_eelv`` as flat zero rather than the
    real (now-removed) EELV drift the filter cancelled -- reporting a definite zero
    for a quantity the processing has actively erased is worse than reporting that it
    cannot be known.

    Returns every key documented in the module docstring; the VC-anchored triple and
    the TLC-anchored triple are independently NaN'd when ``vc``/``tlc`` is ``None``.
    """
    out: dict = {}
    if eelv_tracking == _WITHIN_FILE:
        d_eelv = float("nan") if (correct_trend or ic_eelv_pre is None) else vol_endexp - ic_eelv_pre
        out["d_eelv"] = d_eelv
        ic_op = vol_ic_ref - d_eelv
    else:
        ic_op = vol_ic_ref
    out["ic_op"] = ic_op
    vol_irv = ic_op - vt
    out["vol_irv"] = vol_irv
    out["vt_pct_ic"] = 100.0 * vt / ic_op if ic_op else float("nan")

    if vc is not None:
        vol_eelv = vc - ic_op
        vol_eilv = vol_eelv + vt
        out["vol_eelv"] = vol_eelv
        out["vol_eilv"] = vol_eilv
        out["eelv_pct_vc"] = 100.0 * vol_eelv / vc if vc else float("nan")
        out["eilv_pct_vc"] = 100.0 * vol_eilv / vc if vc else float("nan")
        out["irv_pct_vc"] = 100.0 * vol_irv / vc if vc else float("nan")
    else:
        out["vol_eelv"] = out["vol_eilv"] = float("nan")
        out["eelv_pct_vc"] = out["eilv_pct_vc"] = out["irv_pct_vc"] = float("nan")

    if tlc is not None:
        vol_eelv_abs = tlc - ic_op
        vol_eilv_abs = vol_eelv_abs + vt
        out["vol_eelv_abs"] = vol_eelv_abs
        out["vol_eilv_abs"] = vol_eilv_abs
        out["eelv_pct_tlc"] = 100.0 * vol_eelv_abs / tlc if tlc else float("nan")
        out["eilv_pct_tlc"] = 100.0 * vol_eilv_abs / tlc if tlc else float("nan")
        out["irv_pct_tlc"] = 100.0 * vol_irv / tlc if tlc else float("nan")
    else:
        out["vol_eelv_abs"] = out["vol_eilv_abs"] = float("nan")
        out["eelv_pct_tlc"] = out["eilv_pct_tlc"] = out["irv_pct_tlc"] = float("nan")

    return out


def attach(result, settings, allfiles: list[str]) -> None:
    """Post-loop pass, run AFTER ``core.analysis.references.attach`` (which resolves
    ``vol_ic_ref`` per file -- this module only ever CONSUMES that column, it never
    recomputes it) but before ``average_table`` is concatenated, exactly like that
    function's own docstring describes its place in ``core.pipeline.run_batch``.

    **Column family**: decided from SETTINGS across ``allfiles`` (never just this
    run's own subset) via the SAME ``resolve_reference(..., "ic", ...)`` check
    ``references.attach`` already uses for its own family -- operating lung volumes
    are meaningless without a resolvable IC reference, so this module's family is
    exactly that one. When absent, this function does nothing at all: no columns, on
    any file, so a subset run and a full run write the identical column set either
    way (this ticket's own acceptance criterion).

    Once the family is present, EVERY olv column is added to every OK tidal file's
    ``breaths_table``/``average_row`` (never conditionally per-file) -- the VC/TLC-
    anchored triples are simply NaN on a file with no resolvable VC/TLC, which is what
    lets "column set identical in subset vs full run" hold even when only SOME files'
    groups have a subject VC/TLC entered.
    """
    ic_cfg = settings.processing.lung_volume.ic
    all_names = [os.path.basename(f) for f in allfiles]
    family = any(referenceslib.resolve_reference(n, "ic", settings) is not None
                for n in all_names)
    plan: dict = {"family": family, "active_files": []}
    result.analysis_plan["lung_volume"] = plan
    if not family:
        return

    for filename, fr in list(result.files.items()):
        if fr.error is not None or getattr(fr, "role", "tidal") != "tidal":
            continue
        if fr.average_row is None or "vol_ic_ref" not in fr.average_row.columns:
            continue
        # A per-file bug (an unanticipated exception, not one of the ordinary soft-
        # unresolved cases already handled below) must NaN/skip only THIS file's olv
        # columns, never abort the whole batch -- the same per-file isolation the main
        # run_batch loop already gives every other failure mode. A file that raises
        # here keeps its vol_ic_ref/references.attach columns untouched and simply
        # never gets the olv family; the exception text becomes a notice instead of a
        # crash.
        try:
            _attach_one_file(result, settings, filename, fr, ic_cfg, all_names)
        except Exception as e:                                    # pragma: no cover - defensive
            fr.notices.append(
                f"{filename}: operating lung volumes could not be computed "
                f"({type(e).__name__}: {e})")
            continue
        plan["active_files"].append(filename)


def _attach_one_file(result, settings, filename, fr, ic_cfg, all_names) -> None:
    """One file's worth of :func:`attach` -- pulled out so the caller can wrap it in a
    per-file try/except without duplicating the loop body."""
    vol_ic_ref = fr.average_row["vol_ic_ref"].iloc[0]
    subject = _subject_for(settings, filename)
    tlc = _resolve_tlc(subject)
    vc = _resolve_vc(result, settings, filename, subject, ic_cfg)

    # Same-file vs cross-file IC reference, for within_file tracking's own NaN rule.
    # `used is None` covers TWO distinct situations that must not share one notice:
    # references.attach() never resolved anything for this file at all (it already
    # reported that with its own notice) vs. this file's IC genuinely resolving to a
    # DIFFERENT file (the real cross-file case). Only the latter is a "cross-file"
    # notice; the former would just repeat a wrong reason on top of the right one.
    ic_eelv_pre = None
    if ic_cfg.eelv_tracking == _WITHIN_FILE:
        used = fr.references_used.get("ic")
        if used is not None and used["source"] == filename:
            ic_eelv_pre, _n = _aggregate_field(
                result, referenceslib.resolve_reference(filename, "ic", settings),
                ic_cfg, "ic_eelv_pre")
            if ic_eelv_pre is None:
                fr.notices.append(
                    f"{filename}: EELV tracking is 'within_file' but the resolved "
                    "IC reference's own end-expiratory level could not be "
                    "recovered — d_eelv/ic_op are NaN for this file")
        elif used is not None:
            fr.notices.append(
                f"{filename}: EELV tracking is 'within_file' but the resolved IC "
                "reference is cross-file — a cross-file end-expiratory comparison "
                "has no shared volume zero, so d_eelv/ic_op are NaN for this file")
        # used is None: this file's IC never resolved at all -- references.attach()
        # already notified that; nothing further to say here.

    baseline_ic_ref, baseline_configured, baseline_ambiguity = _resolve_baseline_ic_ref(
        result, settings, filename, ic_cfg, all_names)
    if baseline_ambiguity:
        fr.notices.append(baseline_ambiguity)
    if baseline_configured and baseline_ic_ref is None:
        fr.notices.append(
            f"{filename}: baseline_ic is configured but could not be resolved to "
            "a value — delta_ic/delta_eelv/delta_ic_pct are NaN for this file")
    if baseline_ic_ref is not None and vol_ic_ref == vol_ic_ref:
        delta_ic = float(vol_ic_ref) - baseline_ic_ref
        delta_ic_pct = 100.0 * delta_ic / baseline_ic_ref if baseline_ic_ref else float("nan")
    else:
        delta_ic = float("nan")
        delta_ic_pct = float("nan")
    delta_eelv = -delta_ic

    correct_trend = bool(settings.processing.volume.correct_trend)

    # The expected key set is fixed by settings alone (eelv_tracking, and whether
    # tlc/vc resolved), independent of any breath's own values -- computed once so an
    # EMPTY breaths_table (0 rows) still gets every column (as an empty/NaN Series)
    # instead of silently keeping none at all: DataFrame.apply(axis=1) cannot infer
    # its own output shape from zero rows and would otherwise just return the input
    # frame's columns unchanged, which downstream code (the implausible-value checks
    # below) assumes are always present once the family is active.
    olv_keys = compute_breath_olv(
        vol_ic_ref=0.0, vt=0.0, vol_endexp=0.0, eelv_tracking=ic_cfg.eelv_tracking,
        ic_eelv_pre=ic_eelv_pre, correct_trend=correct_trend, tlc=tlc, vc=vc).keys()

    if len(fr.breaths_table) == 0:
        for col in olv_keys:
            fr.breaths_table[col] = pd.Series(dtype=float)
            fr.average_row[col] = float("nan")
    else:
        def _olv_row(row):
            return pd.Series(compute_breath_olv(
                vol_ic_ref=float(vol_ic_ref), vt=float(row["vt"]),
                vol_endexp=float(row["vol_endexp"]), eelv_tracking=ic_cfg.eelv_tracking,
                ic_eelv_pre=ic_eelv_pre, correct_trend=correct_trend, tlc=tlc, vc=vc))

        olv = fr.breaths_table.apply(_olv_row, axis=1)
        for col in olv.columns:
            fr.breaths_table[col] = olv[col].to_numpy()
            fr.average_row[col] = float(fr.breaths_table[col].mean())

    fr.breaths_table["tlc"] = tlc if tlc is not None else float("nan")
    fr.breaths_table["vc"] = vc if vc is not None else float("nan")
    fr.breaths_table["delta_ic"] = delta_ic
    fr.breaths_table["delta_eelv"] = delta_eelv
    fr.breaths_table["delta_ic_pct"] = delta_ic_pct
    fr.average_row["tlc"] = tlc if tlc is not None else float("nan")
    fr.average_row["vc"] = vc if vc is not None else float("nan")
    fr.average_row["delta_ic"] = delta_ic
    fr.average_row["delta_eelv"] = delta_eelv
    fr.average_row["delta_ic_pct"] = delta_ic_pct

    # Physiologically-implausible-value notices — once per file, never per breath.
    if (fr.breaths_table["vol_irv"] < 0).any():
        fr.notices.append(f"{filename}: at least one breath's vol_irv is negative "
                          "(the operating IC is smaller than tidal volume)")
    if vc is not None and (fr.breaths_table["vol_eelv"] < 0).any():
        fr.notices.append(f"{filename}: at least one breath's vol_eelv is negative "
                          "(the operating IC exceeds VC)")
    if tlc is not None and (fr.breaths_table["vol_eelv_abs"] < 0).any():
        fr.notices.append(f"{filename}: at least one breath's vol_eelv_abs is "
                          "negative (the operating IC exceeds TLC)")
