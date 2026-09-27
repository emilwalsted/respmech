"""Reference-manoeuvre resolution across files.

An analysed file's inspiratory-capacity/forced-vital-capacity/baseline/maximal-effort
reference values do not have to come from the file itself -- a participant's IC might
be captured once, in a separate recording, and apply to every one of their exercise
files. ``resolve_reference`` is the ONE place that decides, for a given file and a
given reference slot, where that value comes from; ``check_links`` is the ONE place
that reports every reason such a link might not actually work once the batch's real
file list and breath typing are known.

``resolve_reference``/``check_links``/``missing_reference_sources``/
``external_reference_sources`` are pure and Qt-free (no file I/O, no
``core.compute``): they reason only about ``Settings`` and the plain filename list a
caller hands them, exactly like ``core.settings.resolve_noise_reference_mode`` reasons
about settings shape alone. ``attach`` is the one function here that is NOT pure -- it
mutates a ``core.pipeline.BatchResult`` a completed batch has already built, using the
manoeuvre values ``core.pipeline.run_batch``'s main loop (and its own out-of-batch
forepass) already extracted; it does no I/O of its own.

Referenced breaths are never sent into ``core.compute`` -- see ``ReferenceEntry``'s own
docstring in ``core.settings``.
"""
from __future__ import annotations

import os

import numpy as np

from respmech.core.settings import BreathRef, Settings
from respmech.core.summary import group_key


class ReferenceLinkError(ValueError):
    """A file's declared IC reference could not be resolved to an actual value at RUN
    time -- either nothing names a source for it even though this analysis uses IC
    references elsewhere (``attach``'s family rule), or the named source/breaths could
    not be found (excluded, mistyped, or the whole source file failed to load/segment
    in the forepass).

    Soft by default: ``attach`` catches it per file, NaNs that file's
    ``vol_ic_ref``/``ic_ref_n``/``ic_ref_source`` and records a notice, and every other
    file keeps going -- "an unresolved link is a caution plus NaN and a notice"
    (``check_links``'s own doctrine). It escapes and fails just that ONE file (via
    ``core.pipeline.run_batch``'s ordinary per-file error handling) only when
    ``processing.lung_volume.require_references`` is set -- the same flag that already
    turns a MISSING source file into a hard ``ui.validation.path_problem`` blocker
    before a run even starts (see ``LungVolumeSettings.require_references``'s own
    docstring); this is that same policy's runtime counterpart, for a failure
    ``Settings.validate()`` cannot see ahead of time (a source that IS matched by the
    file glob but fails to load, say).

    Registered in ``ui.screens.preview._mechanics._SOFT_FILE_ERRORS`` and
    ``ui.screens.run_screen._FIX_HINTS`` alongside ``TrimError``/``VolumeTrendError``/
    ``NoBreathsError``/``EmgSegmentationError`` -- a precondition failure of one
    recording's reference, not a crash -- for when it DOES escape and become a
    ``FileResult.error_kind``."""

#: reference slot name -> the BreathTypeEntry.kind values that count as "this file's own
#: typed breath" for that slot (core.settings.BREATH_KINDS). ic_fvc is both an IC and an
#: FVC manoeuvre in one breath (core.analysis.manoeuvres._IC_KINDS/_FVC_KINDS mirror this
#: exact split), and max_insp/sniff are both maximal-effort references (manoeuvres.
#: _MAX_EFFORT_KINDS). baseline_ic has NO entry here -- see resolve_reference's docstring
#: for why a baseline reference never falls back to a file's own typed breath.
_OWN_TYPED_KINDS = {
    "ic": frozenset({"ic", "ic_fvc"}),
    "fvc": frozenset({"fvc", "ic_fvc"}),
    "max_insp": frozenset({"max_insp", "sniff"}),
}

#: the four reference slots every ReferenceEntry/GroupReferenceEntry carries, in the
#: order check_links reports them (matches ReferenceEntry's own field order).
REFERENCE_SLOTS = ("ic", "fvc", "baseline_ic", "max_insp")


def resolve_reference(file: str, slot: str, settings: Settings) -> BreathRef | None:
    """The :class:`~respmech.core.settings.BreathRef` that names ``file``'s reference
    source for ``slot`` (one of :data:`REFERENCE_SLOTS`), or ``None`` if nothing
    resolves it.

    Resolution order, applied independently per slot (an explicit ``ic`` on ``file``'s
    own :class:`~respmech.core.settings.ReferenceEntry` does not force its ``fvc`` to
    also come from an explicit entry -- an unset slot always falls through):

    1. **explicit** -- ``file``'s own ``processing.references`` entry names this slot.
    2. **group default** -- ``processing.reference_defaults`` entry whose ``group``
       equals ``core.summary.group_key(file, settings)`` names this slot.
    3. **own typed** -- ``ic``/``fvc``/``max_insp`` only: ``file`` itself has one or
       more breaths in ``processing.breath_types`` typed with a matching kind (see
       :data:`_OWN_TYPED_KINDS`), so the file is treated as its own reference source
       (e.g. a peak-exercise file that itself contains a pre-exercise IC breath needs no
       separate reference file). ``baseline_ic`` has no such fallback: a baseline is by
       nature a DIFFERENT recording (measured before/after the manoeuvre, not inferred
       from it), so an unlinked baseline is simply unresolved rather than guessed at.
    4. **none** -- ``None``.
    """
    if slot not in REFERENCE_SLOTS:
        raise ValueError(f"unknown reference slot {slot!r}, expected one of {REFERENCE_SLOTS}")

    for r in settings.processing.references:
        if r.file == file:
            val = getattr(r, slot)
            if val is not None:
                return val
            break                      # this file HAS an entry; only this slot is unset

    group = group_key(file, settings)
    for g in settings.processing.reference_defaults:
        if g.group == group:
            val = getattr(g, slot)
            if val is not None:
                return val
            break

    own_kinds = _OWN_TYPED_KINDS.get(slot)
    if own_kinds:
        breaths = sorted(
            bt.breath for bt in settings.processing.breath_types
            if bt.file == file and bt.kind in own_kinds)
        if breaths:
            return BreathRef(file=file, breaths=breaths)

    return None


def _typed_kinds(settings: Settings, file: str, breath: int) -> set[str]:
    return {bt.kind for bt in settings.processing.breath_types
            if bt.file == file and bt.breath == breath}


def _is_excluded(settings: Settings, file: str, breath: int) -> bool:
    return any(e.file == file and breath in e.breaths
               for e in settings.processing.exclude_breaths)


def _check_one_ref(cautions: list[str], label: str, slot: str, ref: BreathRef | None,
                    names: set[str], settings: Settings) -> None:
    if ref is None:
        return
    if ref.file not in names:
        cautions.append(
            f"{label}: {slot} source {ref.file!r} is not among the analysed files")
        return
    if not ref.breaths:
        cautions.append(f"{label}: {slot} names no breaths in {ref.file!r}")
        return
    own_kinds = _OWN_TYPED_KINDS.get(slot, frozenset())
    for b in ref.breaths:
        if _is_excluded(settings, ref.file, b):
            cautions.append(
                f"{label}: {slot} breath {b} of {ref.file!r} is excluded")
        elif own_kinds and not (_typed_kinds(settings, ref.file, b) & own_kinds):
            cautions.append(
                f"{label}: {slot} breath {b} of {ref.file!r} is linked as {slot} but not "
                f"typed {slot}")


def check_links(settings: Settings, filenames: list[str]) -> list[str]:
    """Every reason a reference/subject link in ``settings`` might not work once the
    batch's real file list is known -- one plain-English caution per unresolved link,
    never an exception (a caution is advisory; ``Settings.validate()`` is the only thing
    that ever blocks a run, and only for ``lung_volume.require_references`` — see
    ``LungVolumeSettings``'s own docstring).

    ``filenames`` is whatever the caller's OWN file list is (``core.pipeline.
    match_input_files``'s result for a real run) -- basenames are compared, since every
    settings entry names a bare filename relative to ``input.folder``, while
    ``match_input_files`` returns full paths. Deliberately NOT a manifest's
    majority-column-count subset (``ui.manifest.Manifest.included_files``): a reference
    source can be a differently-shaped file (e.g. a short manoeuvre-only recording) that
    a column-count vote would exclude from the "main" batch display while ``run_batch``
    still processes it.

    Four caution kinds:

    * a reference source (``ic``/``fvc``/``baseline_ic``/``max_insp`` on either a
      ``ReferenceEntry`` or a ``GroupReferenceEntry``) not among ``filenames``;
    * a linked breath not typed the matching kind in ``processing.breath_types``
      (``ic``/``fvc``/``max_insp`` only -- ``baseline_ic`` names no breath kind of its
      own, see :data:`_OWN_TYPED_KINDS`);
    * a linked breath that is excluded (``processing.exclude_breaths``);
    * a ``reference_defaults``/``input.subjects`` group key matching no analysed file
      (via ``core.summary.group_key``).
    """
    names = {os.path.basename(f) for f in filenames}
    cautions: list[str] = []

    for r in settings.processing.references:
        label = f"processing.references[{r.file}]"
        for slot in REFERENCE_SLOTS:
            _check_one_ref(cautions, label, slot, getattr(r, slot), names, settings)

    seen_groups = {group_key(n, settings) for n in names}

    for g in settings.processing.reference_defaults:
        if g.group not in seen_groups:
            cautions.append(
                f"processing.reference_defaults: group {g.group!r} matches no analysed file")
            continue
        label = f"processing.reference_defaults[{g.group}]"
        for slot in REFERENCE_SLOTS:
            _check_one_ref(cautions, label, slot, getattr(g, slot), names, settings)

    for subj in settings.input.subjects:
        if subj.key not in seen_groups:
            cautions.append(
                f"input.subjects: key {subj.key!r} matches no analysed file")

    return cautions


def missing_reference_sources(settings: Settings, filenames: list[str]) -> list[str]:
    """Reference-source filenames (named in ``processing.references``/
    ``reference_defaults``) that are NOT among ``filenames`` -- sorted, deduplicated.
    Basenames are compared exactly like :func:`check_links`.

    This is the ONE piece of :func:`check_links`'s advisory caution that
    ``ui.validation.path_problem`` also needs as a HARD blocker, but only when
    ``processing.lung_volume.require_references`` is set (see that field's own
    docstring in ``core.settings``): a caution and a blocker must never independently
    decide whether a source is "missing", so both read this same function rather than
    ``path_problem`` re-deriving its own notion of "missing" from ``check_links``'s
    free-text caution strings.
    """
    names = {os.path.basename(f) for f in filenames}
    missing: set[str] = set()
    for r in settings.processing.references:
        for slot in REFERENCE_SLOTS:
            ref = getattr(r, slot)
            if ref is not None and ref.file not in names:
                missing.add(ref.file)
    for g in settings.processing.reference_defaults:
        for slot in REFERENCE_SLOTS:
            ref = getattr(g, slot)
            if ref is not None and ref.file not in names:
                missing.add(ref.file)
    return sorted(missing)


def external_reference_sources(settings: Settings, files: list[str]) -> list[str]:
    """Reference-source basenames (named anywhere in ``processing.references``/
    ``reference_defaults``, any of the four :data:`REFERENCE_SLOTS`) that are NOT
    already among ``files`` -- sorted, deduplicated. Basenames are compared exactly
    like :func:`check_links`.

    This is ``core.pipeline.run_batch``'s own list of files -- pass ``files``, the
    subset this particular run is actually processing (``only_files``-restricted or
    not), never ``allfiles`` (the full matched set): a source already IN this run's own
    file list gets its manoeuvres from the ordinary main loop, exactly like an in-batch
    reference-only file already does (M-30); only a source OUTSIDE this run's own list
    -- whether because it sits outside ``only_files`` this particular run, or because it
    is a dedicated reference recording ``input.files`` never matches at all -- needs the
    batch's forepass to load and segment it separately.
    """
    names = {os.path.basename(f) for f in files}
    sources: set[str] = set()
    for r in settings.processing.references:
        for slot in REFERENCE_SLOTS:
            ref = getattr(r, slot)
            if ref is not None:
                sources.add(ref.file)
    for g in settings.processing.reference_defaults:
        for slot in REFERENCE_SLOTS:
            ref = getattr(g, slot)
            if ref is not None:
                sources.add(ref.file)
    return sorted(sources - names)


def _lookup_manoeuvre(result, filename: str, breath_no: int) -> dict | None:
    """The ``core.analysis.manoeuvres.extract()`` result dict for ``(filename,
    breath_no)``, from wherever the batch actually put it -- an in-batch
    ``FileResult.manoeuvres`` (the ordinary main loop, or M-30's reference-only-file
    handling; either way the SAME dict shape) or the forepass's out-of-batch
    ``BatchResult.references``. ``None`` when neither has it: the source file itself
    failed (forepass or main-loop error), the breath was never typed at all, or the
    breath number simply does not exist in that file -- all three look identical from
    here, and all three are the same "could not be resolved" outcome to a caller."""
    fr = result.files.get(filename)
    if fr is not None and fr.error is None:
        row = (fr.manoeuvres or {}).get(breath_no)
        if row is not None:
            return row
    ext = result.references.get(filename)
    if ext is not None:
        return ext.get(breath_no)
    return None


def _aggregate_ic(result, ref: BreathRef, ic_cfg) -> tuple[float | None, int, list[str]]:
    """``(vol_ic_ref, ic_ref_n, notices)`` for one resolved IC :class:`BreathRef` --
    the ``ic_cfg.aggregate`` ('mean'/'median') of ``vol_ic`` over ``ref.breaths`` that
    both RESOLVE (:func:`_lookup_manoeuvre` finds them) and are not flagged with one of
    ``ic_cfg.reject_flags`` (``LOW_EFFORT`` by default -- the SAME disqualifying rule
    ``manoeuvres.apply_repeatability``'s own leave-one-out group already uses, so a
    rejected attempt never contributes to a reference value any more than it
    contributes to its own file's repeatability check).

    Returns ``(None, 0, notices)`` when nothing usable resolved (every breath missing,
    or every resolved breath rejected) -- the caller (:func:`attach`) turns that into a
    :class:`ReferenceLinkError`, exactly like a wholly unresolved reference.

    ``ref.breaths`` is deduplicated first (``dict.fromkeys``, order-preserving):
    nothing validates a hand-authored ``BreathRef`` for a repeated breath number the
    way ``Settings.validate()`` already does for a duplicate file/group entry, and
    without this a repeated number would silently double that breath's weight in the
    aggregate and inflate ``ic_ref_n`` past the number of attempts actually made
    (self-review finding)."""
    vols: list[float] = []
    notices: list[str] = []
    for b in dict.fromkeys(ref.breaths):
        row = _lookup_manoeuvre(result, ref.file, b)
        if row is None:
            notices.append(
                f"IC reference breath {b} of {ref.file!r} could not be resolved "
                "(the source file failed, or that breath was not typed as an IC "
                "manoeuvre there)")
            continue
        quality = row.get("quality") or []
        if set(quality) & set(ic_cfg.reject_flags):
            continue
        vol = row.get("vol_ic")
        if vol is not None:
            vols.append(float(vol))
    if not vols:
        return None, 0, notices
    agg = float(np.median(vols)) if ic_cfg.aggregate == "median" else float(np.mean(vols))
    return agg, len(vols), notices


def attach(result, settings: Settings, allfiles: list[str]) -> None:
    """Post-loop pass (the batch's "efterpas"): resolves every OK tidal file's IC
    reference (:func:`resolve_reference`'s explicit > group default > own typed > none
    order), adds the per-breath columns ``vol_ic_ref``/``ic_ref_n``/``ic_ref_source``
    to ``FileResult.breaths_table``/``average_row`` (appended -- never inserted before
    an existing column), records ``FileResult.references_used`` and
    ``BatchResult.analysis_plan``, and mutates ``result.files`` in place.

    Called AFTER ``run_batch``'s main per-file loop has finished (every ``FileResult``
    in ``result.files`` already has its final ``manoeuvres``/``breaths_table``/
    ``average_row``) but BEFORE ``average_table`` is concatenated from the individual
    ``average_row``s, so the new columns are already present when that concatenation
    runs (``pd.concat``'s default outer join fills a file with no reference columns
    with NaN there, needing no special-case code of its own).

    **The column family** (whether ``vol_ic_ref``/``ic_ref_n``/``ic_ref_source`` exist
    AT ALL for this analysis) is decided from SETTINGS across ``allfiles`` -- the FULL
    matched set, never just the subset this particular run happens to process -- so a
    subset run's written columns are identical to a full run's (this is the whole
    reason ``allfiles`` is a separate parameter from ``result.ok_files``). A file with
    the family present but no resolution of its OWN gets NaN'd columns and a
    :class:`ReferenceLinkError`-derived notice, exactly like a resolved-but-failed
    link: "an unresolved link is a caution plus NaN and a notice" applies to the whole
    family, not only to an explicitly configured one.

    ``processing.lung_volume.require_references``: when set, a file whose IC reference
    does not resolve is DEMOTED to a failed file (``fr.error``/``fr.error_kind`` set on
    the SAME ``FileResult`` instance already in ``result.files`` -- no new object is
    constructed, avoiding a pipeline/references import cycle) instead of being NaN'd;
    the caller is expected to rebuild its own ``average_rows``/written-file list from
    ``result.ok_files`` AFTER this call, exactly as ``run_batch`` already does, so a
    demoted file is excluded from the cohort the same way an ordinary main-loop
    failure already is.

    A file with no ``breaths_table``/``average_row`` at all (``fr.role != "tidal"`` --
    a reference-only file, M-30) is skipped: there is no per-breath table to add a
    column to. Such a file can still itself be a reference SOURCE for others (that is
    exactly what ``FileResult.manoeuvres``/the forepass are for) -- it just never gets
    a ``vol_ic_ref`` column of its own.
    """
    ic_cfg = settings.processing.lung_volume.ic
    require = settings.processing.lung_volume.require_references
    all_names = [os.path.basename(f) for f in allfiles]
    family = any(resolve_reference(n, "ic", settings) is not None for n in all_names)

    plan: dict = {"ic": {"family": family, "resolved": {}, "unresolved": []}}
    if not family:
        result.analysis_plan = plan
        return

    for filename, fr in list(result.files.items()):
        if fr.error is not None or getattr(fr, "role", "tidal") != "tidal":
            continue
        ref = resolve_reference(filename, "ic", settings)
        link_notices: list[str] = []
        try:
            if ref is None:
                raise ReferenceLinkError(
                    f"{filename}: no IC reference resolves for this file, although "
                    "this analysis names an IC reference elsewhere (processing."
                    "references / reference_defaults / a file's own typed IC breaths)")
            agg, n, link_notices = _aggregate_ic(result, ref, ic_cfg)
            if agg is None:
                reason = "; ".join(link_notices) if link_notices else "all rejected or missing"
                raise ReferenceLinkError(
                    f"{filename}: IC reference {ref.file!r} breaths {list(ref.breaths)} "
                    f"resolved to no usable value ({reason})")
        except ReferenceLinkError as e:
            if require:
                fr.error = str(e)
                fr.error_kind = "ReferenceLinkError"
                plan["ic"]["unresolved"].append(filename)
                continue
            for msg in link_notices:
                fr.notices.append(msg)
            fr.notices.append(str(e))
            fr.breaths_table["vol_ic_ref"] = np.nan
            fr.breaths_table["ic_ref_n"] = np.nan
            fr.breaths_table["ic_ref_source"] = None
            fr.average_row["vol_ic_ref"] = np.nan
            fr.average_row["ic_ref_n"] = np.nan
            fr.average_row["ic_ref_source"] = None
            plan["ic"]["unresolved"].append(filename)
            continue

        for msg in link_notices:
            fr.notices.append(msg)
        fr.breaths_table["vol_ic_ref"] = agg
        fr.breaths_table["ic_ref_n"] = float(n)
        fr.breaths_table["ic_ref_source"] = ref.file
        fr.average_row["vol_ic_ref"] = float(fr.breaths_table["vol_ic_ref"].mean())
        fr.average_row["ic_ref_n"] = float(n)
        fr.average_row["ic_ref_source"] = ref.file
        used = {"source": ref.file, "breaths": list(ref.breaths), "n": n, "value": agg}
        fr.references_used["ic"] = used
        plan["ic"]["resolved"][filename] = dict(used)

    result.analysis_plan = plan
