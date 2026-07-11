"""Write batch results to disk (Excel data + processed CSV + summary + figures).

Consumers of the core's results — the core itself writes nothing. Output layout
matches the legacy tool: ``<out>/data/Average breathdata.xlsx``,
``<out>/data/<file>.breathdata.xlsx`` and ``<out>/data/<file> – Processed data.csv``.

On top of that the writer adds the pieces a *publishable* analysis needs, all built
from the finished result at write time so the golden-pinned result tables are never
touched:

* each workbook carries a **Units** sheet (P10/P21) and a **Provenance** sheet
  (P21) alongside its Data sheet;
* the breath workbook carries a **normalised-EMG** sheet when normalisation is on
  (P14);
* a **Cohort summary** workbook reports mean ± SD / n / CV% across files, and by
  group (P8/P15);
* **diagnostic figures** are drawn into ``<out>/diagnostics/`` (P11);
* two provenance files — ``analysis-used.toml`` and ``run-report.txt`` — are dropped
  in ``<out>/`` so a result is never orphaned from its settings (P7).
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from respmech import __version__
from respmech.core import units as _units
from respmech.core.summary import build_cohort_summary, normalize_emg_table

_CREATED = f"Created with RespMech v{__version__} (github.com/emilwalsted/respmech)"


def _version_df():
    return pd.DataFrame(
        {"Created": _CREATED, "Website": "https://github.com/emilwalsted/respmech"},
        index=[0]).T.rename(columns={0: "Version info"})


def _units_df(columns):
    um = _units.units_map(columns)
    return pd.DataFrame({"Column": list(um.keys()), "Unit": list(um.values())})


def _provenance_rows(settings, when):
    ts = (when or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    ip = settings.input
    return pd.DataFrame(
        [("RespMech version", __version__),
         ("Generated", ts),
         ("Input folder", ip.folder),
         ("Input pattern", ip.files),
         ("Sampling frequency (Hz)", ip.format.sampling_frequency),
         ("Breath separation", f"{settings.processing.segmentation.method}, "
                               f"buffer {settings.processing.segmentation.buffer}"),
         ("Drift correction", settings.processing.volume.correct_drift),
         ("EMG normalisation", settings.processing.emg.normalization),
         ("Settings snapshot", "analysis-used.toml")],
        columns=["Key", "Value"])


def _write_xlsx(df: pd.DataFrame, path: str, settings=None, when=None, extra_sheets=None):
    """Write a Data sheet plus Units, any extra sheets, Provenance and Version.

    Only the Data sheet content is load-bearing (the golden suite pins the DataFrame,
    not the workbook); the extra sheets are additive context for a reader."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Data", index=False)
        _units_df(df.columns).to_excel(writer, sheet_name="Units", index=False)
        for name, edf in (extra_sheets or {}).items():
            edf.to_excel(writer, sheet_name=name, index=False)
        if settings is not None:
            _provenance_rows(settings, when).to_excel(writer, sheet_name="Provenance", index=False)
        _version_df().to_excel(writer, sheet_name="Version", index=False)


def write_batch(result, settings, outputfolder: str, when: datetime | None = None) -> list[str]:
    """Write all enabled outputs plus summary, figures and provenance; returns the
    list of files written."""
    datadir = os.path.join(outputfolder, "data")
    os.makedirs(datadir, exist_ok=True)
    written = []

    if settings.output.data.save_breath_by_breath:
        for fname, fr in result.ok_files.items():
            p = os.path.join(datadir, f"{fname}.breathdata.xlsx")
            extra = {}
            norm = normalize_emg_table(fr.breaths_table, settings)   # P14
            if norm is not None and len(norm):
                extra["EMG normalised"] = norm
            _write_xlsx(fr.breaths_table, p, settings=settings, when=when, extra_sheets=extra)
            written.append(p)

    if settings.output.data.save_processed:
        for fname, fr in result.ok_files.items():
            if fr.processed is not None:
                p = os.path.join(datadir, f"{fname} – Processed data.csv")
                fr.processed.to_csv(p, index=False)
                written.append(p)

    if settings.output.data.save_average and result.average_table is not None:
        p = os.path.join(datadir, "Average breathdata.xlsx")
        _write_xlsx(result.average_table, p, settings=settings, when=when)
        written.append(p)
        written += _write_cohort_summary(result, settings, datadir, when)   # P8/P15

    fig_written, fig_failures = _write_figures(result, settings, outputfolder)  # P11
    written += fig_written

    # provenance — always written, so a folder of results carries its own recipe (P7)
    written.append(_write_manifest(settings, outputfolder))
    written.append(_write_run_report(result, settings, outputfolder, written, when, fig_failures))
    return written


# --------------------------------------------------------------------------- #
# cohort summary (P8/P15)
# --------------------------------------------------------------------------- #
def _write_cohort_summary(result, settings, datadir, when) -> list[str]:
    agg = build_cohort_summary(result, settings)
    summary = agg.get("summary")
    if summary is None or len(summary) == 0:
        return []
    path = os.path.join(datadir, "Cohort summary.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        if agg.get("by_group") is not None:
            agg["by_group"].to_excel(writer, sheet_name="By group", index=False)
        _provenance_rows(settings, when).to_excel(writer, sheet_name="Provenance", index=False)
        _version_df().to_excel(writer, sheet_name="Version", index=False)
    return [path]


# --------------------------------------------------------------------------- #
# diagnostic figures (P11)
# --------------------------------------------------------------------------- #
def _write_figures(result, settings, outputfolder):
    """Returns (written_paths, failures). Failures — including matplotlib being
    absent — are surfaced in the run report, never silently dropped."""
    try:
        from respmech.core import plots
    except Exception as e:                       # pragma: no cover - plotting optional
        return [], [("figures", f"plotting unavailable: {e}")]
    return plots.write_figures(result, settings, outputfolder)


# --------------------------------------------------------------------------- #
# provenance files (P7)
# --------------------------------------------------------------------------- #
def _write_manifest(settings, outputfolder: str) -> str:
    """Drop the exact settings as a reloadable ``analysis-used.toml``."""
    from respmech.settingsio.toml_io import dumps_toml
    path = os.path.join(outputfolder, "analysis-used.toml")
    header = (f"# RespMech v{__version__} — settings used for this run.\n"
              f"# Reload with: File ▸ Load analysis (or point RespMech at this file).\n\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + dumps_toml(settings))
    return path


def _breath_counts(fr) -> tuple[int, int]:
    """(total, excluded) breath counts for a file result."""
    if not fr.breaths:
        return 0, 0
    total = len(fr.breaths)
    excluded = sum(1 for b in fr.breaths.values() if b.get("ignored"))
    return total, excluded


def _yn(flag) -> str:
    return "yes" if flag else "no"


def _write_run_report(result, settings, outputfolder: str,
                      written: list[str], when: datetime | None,
                      fig_failures=None) -> str:
    """A plain-text provenance log of what was read, kept, excluded and written."""
    ts = (when or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    ip, vol, samp = settings.input, settings.processing.volume, settings.processing.sampling
    seg, emg = settings.processing.segmentation, settings.processing.emg
    L: list[str] = []
    L.append(f"RespMech v{__version__} — run report")
    L.append(f"Generated: {ts}")
    L.append("")
    L.append("INPUT")
    L.append(f"  Folder:   {ip.folder}")
    L.append(f"  Pattern:  {ip.files}")
    L.append(f"  Sampling: {ip.format.sampling_frequency} Hz")
    L.append("")

    ok, failed = result.ok_files, result.failed_files
    L.append(f"FILES ({len(ok)} processed, {len(failed)} failed)")
    for fname, fr in ok.items():
        total, excl = _breath_counts(fr)
        used = total - excl
        note = f" ({excl} excluded → {used} used)" if excl else ""
        L.append(f"  [ok]   {fname}   {total} breaths{note}")
    for fname, fr in failed.items():
        L.append(f"  [FAIL] {fname}   ERROR: {fr.error}")
    L.append("")

    L.append("PROCESSING")
    L.append(f"  Integrate flow → volume: {_yn(vol.integrate_from_flow)}")
    L.append(f"  Invert flow / volume:    {_yn(vol.inverse_flow)} / {_yn(vol.inverse_volume)}")
    L.append(f"  Drift correction:        {_yn(vol.correct_drift)}")
    L.append(f"  Trend correction:        {_yn(vol.correct_trend)}"
             + (f" ({vol.trend_method})" if vol.correct_trend else ""))
    L.append(f"  Resample:                {_yn(samp.resample)}"
             + (f" (→ {samp.resample_to_frequency} Hz)" if samp.resample else ""))
    L.append(f"  Breath separation:       by {seg.method}, buffer {seg.buffer}")
    L.append(f"  ECG removal:             {_yn(emg.remove_ecg)}")
    L.append(f"  EMG noise removal:       {_yn(emg.remove_noise)}")
    L.append(f"  EMG normalisation:       {emg.normalization}")
    L.append("")

    L.append(f"OUTPUTS WRITTEN ({len(written) + 1} files)")
    for p in written:
        L.append(f"  {os.path.relpath(p, outputfolder)}")
    L.append("  run-report.txt")
    L.append("")

    if fig_failures:
        L.append(f"FIGURES SKIPPED ({len(fig_failures)})")
        for name, err in fig_failures:
            L.append(f"  {name}: {err}")
        L.append("")

    L.append("Settings snapshot: analysis-used.toml (reload to reproduce this run).")

    path = os.path.join(outputfolder, "run-report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return path
