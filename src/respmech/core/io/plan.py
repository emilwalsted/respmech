"""The plan a dry run shows and a real run's writer is measured against (ticket A06).

Before this, the Run screen's pre-flight list and ``write_batch`` each carried their own,
independent idea of what a run would write — and they had drifted apart badly (measured:
a dry run said 4 files where a real run wrote 14; on a bigger batch, 7 vs 45, 37 of them in
``diagnostics/``, a folder the old plan never looked at at all). This module is now the
ONE place that answers "what could a run with these settings, over these files, write?" —
built from the same settings flags and the same shared figure-job list
(``core.plots.per_file_figure_jobs`` / ``emg_overview_candidates`` / ``emg_audio_candidates``)
that ``core.io.writers.write_batch`` and ``core.plots.write_figures`` themselves read, so a
figure type can never appear in what gets written without also appearing here.

Qt-free by design (``core/`` may not import Qt) — a future CLI ``--dry-run`` can reuse this
exact function, not just the GUI.

**A plan is a ceiling, not a promise.** Some counts cannot be known without actually
processing the files: a file can fail outright (no outputs at all for it), a figure job can
decline to draw anything for a particular file's data (e.g. "trend" when no anchors are
found), and which EMG conditioning stages a file's *data* really has depends on what
conditioning produced, not just on which settings are ticked. Every :class:`OutputGroup`
whose count is only an upper bound sets ``is_cap=True`` — callers must render it as "up to
N", never "N". The invariant this buys: for the SAME settings and the SAME file list, every
path a real run actually writes is a member of :meth:`Plan.all_paths` — proven by
``tests/unit/test_core_outputs.py::test_plan_contains_every_path_a_real_run_writes``.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field


@dataclass
class OutputGroup:
    """One row of the plan: a category of output, how many files it covers, where they
    land (relative to the output folder), and — for the "show all files" expansion —
    every relative path in the group. ``examples`` is a short slice of ``paths`` for the
    compact, over-~10-files summary view."""
    category: str
    count: int
    is_cap: bool
    target: str                        # relative subfolder ("data/", "diagnostics/", "")
    paths: list = field(default_factory=list)

    @property
    def examples(self):
        return self.paths[:3]


@dataclass
class WriteProbe:
    """The result of actually trying to write into a folder (``probe_write_folder``) —
    never a promise inferred from permission bits alone."""
    ok: bool
    message: str = ""


@dataclass
class Plan:
    groups: list = field(default_factory=list)
    outputfolder: str = ""
    # Which mode this plan was built for (A05's cohort_outputs) — carried on the plan so a
    # later write (``core.io.writers.write_planned``) can execute it without the caller
    # having to remember and re-pass a flag that might drift from what was actually shown.
    cohort_outputs: bool = True
    probe: WriteProbe | None = None

    @property
    def total_count(self) -> int:
        return sum(g.count for g in self.groups)

    @property
    def is_cap(self) -> bool:
        """True if ANY group's count is an upper bound rather than an exact promise —
        the flag a renderer uses to decide whether to write "up to N" or "N"."""
        return any(g.is_cap for g in self.groups)

    def all_paths(self) -> list:
        out = []
        for g in self.groups:
            out.extend(g.paths)
        return out


def plan_outputs(settings, filenames, cohort_outputs: bool = True) -> Plan:
    """Build the plan for ``settings`` over ``filenames`` (an iterable of input paths or
    basenames — only the basename is used).

    ``filenames`` means "the files this run will attempt", not "the files that will
    succeed": pass the full matching-input list for a pre-flight/dry-run plan (nothing has
    run yet, so success/failure is unknown), or ``list(result.files)`` — BOTH ok and failed
    — for a plan built after a run has finished, so a failed file still costs it nothing
    beyond being listed as part of the ceiling it never subtracts from. Never pass only
    ``result.ok_files``: that would make the plan describe less than what a DIFFERENT run
    of the same settings could write, and the whole point of a ceiling is that it does not
    depend on which files happen to succeed.
    """
    from respmech.core.plots import (emg_audio_candidates, emg_overview_candidates,
                                     per_file_figure_jobs)

    d = settings.output.data
    dg = settings.output.diagnostics
    emg = settings.processing.emg
    names = [os.path.basename(f) for f in filenames]
    n = len(names)
    groups = []

    if d.save_breath_by_breath:
        # is_cap only when n>0 (a file among them COULD fail): with zero files, the exact
        # count is zero, full stop, no uncertainty to signal — same fix as the two groups
        # below (self-review: found there first, applies here identically).
        groups.append(OutputGroup(
            "Breath-by-breath workbook, one per recording", n, n > 0, "data/",
            [f"data/{nm}.breathdata.xlsx" for nm in names]))
    if d.save_processed:
        groups.append(OutputGroup(
            "Processed-signal CSV, one per recording", n, n > 0, "data/",
            [f"data/{nm} – Processed data.csv" for nm in names]))
    if d.save_average and cohort_outputs:
        groups.append(OutputGroup(
            "Average breath-data workbook", 1 if n else 0, n > 0, "data/",
            ["data/Average breathdata.xlsx"] if n else []))
        # Gated exactly like the average workbook (writers._write_cohort_summary is called
        # right alongside it), but n==0 must still report 0/False like its sibling above —
        # found in self-review: an unconditional is_cap=True here made an EMPTY file list
        # with save_average on report "up to 2 files" instead of the true, exact 2 (the
        # manifest + run report), because is_cap is a per-plan OR across every group.
        groups.append(OutputGroup(
            "Cohort summary workbook", 1 if n else 0, n > 0, "data/",
            ["data/Cohort summary.xlsx"] if n else []))

    fig_paths = []
    for nm in names:
        for _label, _fn, suffix in per_file_figure_jobs(settings):
            fig_paths.append(f"diagnostics/{nm} – {suffix}")
    if getattr(dg, "save_emg", False):
        for nm in names:
            for _key, label in emg_overview_candidates(settings):
                fig_paths.append(f"diagnostics/{nm} – {label}.pdf")
    if dg.save_pv_individual and cohort_outputs and n > 1:
        fig_paths.append("diagnostics/All files – Campbell (average).pdf")
    if fig_paths:
        groups.append(OutputGroup("Diagnostic figures", len(fig_paths), True,
                                  "diagnostics/", fig_paths))

    if emg.save_sound:
        # dict.fromkeys, not a plain list: a channel number configured twice in
        # ``input.channels.emg`` (a user data-entry slip, not validated against) must not
        # double every WAV path in the plan.
        channels = list(dict.fromkeys(settings.input.channels.emg or []))
        stages = emg_audio_candidates(settings)
        wav_paths = [f"diagnostics/{nm} – EMG col {ch} ({label}).wav"
                    for nm in names for ch in channels for _key, label in stages]
        if wav_paths:
            groups.append(OutputGroup("EMG audio export (WAV)", len(wav_paths), True,
                                      "diagnostics/", wav_paths))

    # Provenance. For a full run this is exact: analysis-used.toml + run-report.txt, always.
    #
    # For a subset (cohort_outputs=False, A05) the writer's OWN names differ, and this plan
    # must say so rather than confidently name files the write will never touch (found in
    # self-review: the first version of this listed "run-report.txt" verbatim for a subset
    # write, even though writers._write_run_report ALWAYS gives a subset its own report a
    # distinct, timestamped "run-report (partial, <time>).txt" name — never the full run's).
    # The manifest is more nuanced still: writers._write_manifest keeps the normal name if
    # no full-run manifest exists yet, leaves an EXISTING one untouched if this run's
    # settings are identical to it, and only timestamps a partial name if one exists AND
    # differs — a read-and-compare this plan does not duplicate. Both names are therefore
    # placeholders, not literal paths, for a subset plan.
    if cohort_outputs:
        groups.append(OutputGroup("Run report and analysis snapshot", 2, False, "",
                                  ["run-report.txt", "analysis-used.toml"]))
    else:
        groups.append(OutputGroup(
            "Run report and analysis snapshot (this write's OWN partial record — the full "
            "run's is left alone)", 2, True, "",
            ["run-report (partial, <time written>).txt",
             "analysis-used.toml, unless a full run's manifest already differs — then "
             "analysis-used (partial, <time written>).toml"]))

    return Plan(groups=groups, outputfolder=settings.output.folder, cohort_outputs=cohort_outputs)


def probe_write_folder(folder: str) -> WriteProbe:
    """Actually create (if needed), write into, and clean up after a temp file in
    ``folder`` — proof RespMech can write results there, not an ``os.access`` guess, which
    is unreliable against Windows ACLs (ticket A06 point 6).

    Creates ``folder`` and any missing parent directories if necessary, then removes every
    directory THIS call created — deepest first, so a dry run's probe (or the output-folder
    picker, which must not leave a trace either way) never leaves an empty folder behind on
    disk. A pre-existing folder, or one this call could not fully clean up afterwards
    (another process added a file to it in the meantime), is left exactly as found; the
    probe result still reports whether writing itself succeeded.
    """
    folder = os.path.abspath(folder)
    created = []
    d, prev = folder, None
    while d and d != prev and not os.path.isdir(d):
        created.append(d)
        prev, d = d, os.path.dirname(d)

    # ONE try/finally around both steps, not two separate ones: makedirs() creates parent
    # directories one at a time, so a failure on the LAST component (a too-long name, a
    # permission wall partway down) used to return before the cleanup below ever ran,
    # leaving every ancestor it had just created behind — found in self-review, reproduced
    # with a path component long enough to hit ENAMETOOLONG.
    ok, message = True, ""
    try:
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as e:
            ok, message = False, f"cannot create the output folder: {e}"
        else:
            try:
                fd, path = tempfile.mkstemp(prefix=".respmech-write-probe-", dir=folder)
                os.close(fd)
            except OSError as e:
                ok, message = False, f"cannot write to the output folder: {e}"
            else:
                # A delete failure here does NOT mean the probe failed — the write that
                # matters already succeeded — so it must never flip `ok` to False (self-
                # review: doing so inside a single shared try/except was a false negative).
                # The leftover probe file is harmless; it only means `created`'s rmdir for
                # its directory will find it non-empty and skip it below, same as any other
                # file that landed there in the meantime.
                try:
                    os.remove(path)
                except OSError:
                    pass
    finally:
        for d in created:              # deepest-first, exactly the order they were recorded in
            try:
                os.rmdir(d)
            except OSError:
                pass                    # not empty (a real file landed there meanwhile) — leave it
    return WriteProbe(ok, message)
