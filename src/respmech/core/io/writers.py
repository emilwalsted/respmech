"""Write batch results to disk (Excel data + processed CSV + a run report).

Consumers of the core's results — the core itself writes nothing. Output layout
matches the legacy tool: ``<out>/data/Average breathdata.xlsx``,
``<out>/data/<file>.breathdata.xlsx`` and ``<out>/data/<file> – Processed data.csv``.

Every run also drops two provenance files in ``<out>/`` so a result is never
orphaned from the settings that produced it (feature P7):

* ``analysis-used.toml`` — the exact, reloadable settings snapshot;
* ``run-report.txt`` — a human-readable log of what was read, kept, excluded and
  written, with the RespMech version and a timestamp.
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from respmech import __version__

_CREATED = f"Created with RespMech v{__version__} (github.com/emilwalsted/respmech)"


def _version_df():
    return pd.DataFrame(
        {"Created": _CREATED, "Website": "https://github.com/emilwalsted/respmech"},
        index=[0]).T.rename(columns={0: "Version info"})


def _write_xlsx(df: pd.DataFrame, path: str):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Data", index=False)
        _version_df().to_excel(writer, sheet_name="Version", index=False)


def write_batch(result, settings, outputfolder: str, when: datetime | None = None) -> list[str]:
    """Write all enabled outputs plus the provenance files; returns the list of
    files written (data outputs first, then the two provenance files)."""
    datadir = os.path.join(outputfolder, "data")
    os.makedirs(datadir, exist_ok=True)
    written = []

    if settings.output.data.save_breath_by_breath:
        for fname, fr in result.ok_files.items():
            p = os.path.join(datadir, f"{fname}.breathdata.xlsx")
            _write_xlsx(fr.breaths_table, p)
            written.append(p)

    if settings.output.data.save_processed:
        for fname, fr in result.ok_files.items():
            if fr.processed is not None:
                p = os.path.join(datadir, f"{fname} – Processed data.csv")
                fr.processed.to_csv(p, index=False)
                written.append(p)

    if settings.output.data.save_average and result.average_table is not None:
        p = os.path.join(datadir, "Average breathdata.xlsx")
        _write_xlsx(result.average_table, p)
        written.append(p)

    # provenance — always written, so a folder of results carries its own recipe
    written.append(_write_manifest(settings, outputfolder))
    written.append(_write_run_report(result, settings, outputfolder, written, when))
    return written


# --------------------------------------------------------------------------- #
# provenance (P7)
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
                      written: list[str], when: datetime | None) -> str:
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
    L.append("")

    L.append(f"OUTPUTS WRITTEN ({len(written) + 1} files)")
    for p in written:
        L.append(f"  {os.path.relpath(p, outputfolder)}")
    L.append("  run-report.txt")
    L.append("")
    L.append("Settings snapshot: analysis-used.toml (reload to reproduce this run).")

    path = os.path.join(outputfolder, "run-report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return path
