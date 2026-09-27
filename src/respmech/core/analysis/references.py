"""Reference-manoeuvre resolution across files (M-34).

An analysed file's inspiratory-capacity/forced-vital-capacity/baseline/maximal-effort
reference values do not have to come from the file itself -- a participant's IC might
be captured once, in a separate recording, and apply to every one of their exercise
files. ``resolve_reference`` is the ONE place that decides, for a given file and a
given reference slot, where that value comes from; ``check_links`` is the ONE place
that reports every reason such a link might not actually work once the batch's real
file list and breath typing are known.

Both are pure and Qt-free (no file I/O, no ``core.compute``): this module reasons only
about ``Settings`` and the plain filename list a caller hands it, exactly like
``core.settings.resolve_noise_reference_mode`` reasons about settings shape alone.
Resolving the ACTUAL breath values (loading the referenced file, running
``core.analysis.manoeuvres.extract`` on it, attaching the result to the referencING
file's per-breath columns) is M-35's pipeline-pass scope, not this module's.

Referenced breaths are never sent into ``core.compute`` -- see ``ReferenceEntry``'s own
docstring in ``core.settings``.
"""
from __future__ import annotations

import os

from respmech.core.settings import BreathRef, Settings
from respmech.core.summary import group_key

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
    that ever blocks a run, and only for ``lung_volumes.require_references`` — see
    ``LungVolumeSettings``'s own docstring).

    ``filenames`` is whatever the caller's OWN file list is (``core.pipeline.
    match_input_files``'s result for a real run) -- basenames are compared, since every
    settings entry names a bare filename relative to ``input.folder``, while
    ``match_input_files`` returns full paths. Deliberately NOT a manifest's
    majority-column-count subset (``ui.manifest.Manifest.included_files``): a reference
    source can be a differently-shaped file (e.g. a short manoeuvre-only recording) that
    a column-count vote would exclude from the "main" batch display while ``run_batch``
    still processes it.

    Four caution kinds, matching M-34's own scope:

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

    seen_groups = {group_key(f, settings) for f in filenames}

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
