"""Write batch results to disk (Excel data + processed CSV).

Consumers of the core's results — the core itself writes nothing. Output layout
matches the legacy tool: ``<out>/data/Average breathdata.xlsx``,
``<out>/data/<file>.breathdata.xlsx`` and ``<out>/data/<file> – Processed data.csv``.
"""
from __future__ import annotations

import os

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


def write_batch(result, settings, outputfolder: str) -> list[str]:
    """Write all enabled outputs; returns the list of files written."""
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

    return written
