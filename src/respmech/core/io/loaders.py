"""Input loaders: MATLAB / Excel / CSV / tab-text -> raw signal arrays.

Faithful port of the legacy ``load()``. One fix (legacy bug #4): volume integration
from flow uses ``scipy.integrate.cumulative_trapezoid`` (``cumtrapz`` was removed in
SciPy >= 1.14) — numerically identical.
"""
import os
from collections import OrderedDict

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.integrate
import pandas as pd


class DataValidationError(ValueError):
    """Raised when an input column is missing, non-numeric, NaN, or mismatched."""


def _column(value, name, ncols, filepath):
    """Resolve a 1-based column setting to a 0-based index, or say exactly what is wrong.

    This is the only place that knows both the setting and the file, so it is the only place
    the range check can be right. Without it an unassigned channel reached pandas as ``None``
    and surfaced as "unsupported operand type(s) for -: 'NoneType' and 'int'", and — worse —
    column 0 became ``iloc[:, -1]``, silently analysing the LAST column of the recording
    instead of reporting anything at all."""
    # NaN reaches here only from a hand-edited settings file (the model uses None, and
    # _legacy_ns maps only the optional volume column to NaN, which callers test first).
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise DataValidationError(
            f"{name} is not assigned. Pick a column with 'Assign channels from data…' in Setup.")
    v = int(value)
    if v < 1:
        raise DataValidationError(
            f"{name} is set to column {v}, but columns are numbered from 1.")
    if v > ncols:
        raise DataValidationError(
            f"{name} is set to column {v}, but {os.path.basename(filepath)} has "
            f"only {ncols} column{'s' if ncols != 1 else ''}.")
    return v - 1


def _read_table(f, **kw):
    """Read a CSV/TSV tolerantly across encodings. Excel's "Unicode Text" export is
    UTF-16 (BOM-prefixed); European instrument exports are often cp1252/latin-1 — the
    dev machine's plain UTF-8 files are only a subset. Downstream column access is
    positional (``df.iloc``), so header mojibake under a fallback never touches the
    numeric data."""
    with open(f, "rb") as fh:
        if fh.read(2) in (b"\xff\xfe", b"\xfe\xff"):
            return pd.read_csv(f, encoding="utf-16", **kw)
    for enc in ("utf-8-sig", "cp1252"):        # utf-8-sig also strips a stray UTF-8 BOM
        try:
            return pd.read_csv(f, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(f, encoding="latin-1", **kw)   # latin-1 maps every byte, never raises


def _checkcolumn(text, data):
    arr = np.asarray(data)
    # a non-numeric column reads as object/str dtype; check that BEFORE np.isnan, which
    # raises TypeError on an object array (so a text column would crash instead of
    # reporting the friendly "must be numeric" error).
    if arr.dtype.kind in ("U", "S", "O"):
        raise DataValidationError(text + " contains text values – all values must be numeric.")
    if np.isnan(arr).any():
        raise DataValidationError(text + " contains NaN values.")


def _alleq(iterable):
    from itertools import groupby
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def validatedata(flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings):
    _checkcolumn("Flow column", flow)
    _checkcolumn("Volume column", volume)
    _checkcolumn("Oesophageal pressure column", poes)
    _checkcolumn("Gastric pressure column", pgas)
    _checkcolumn("Trans-diaphragmatic pressure column", pdi)

    collens = [len(flow), len(volume), len(poes), len(pgas), len(pdi)]
    coltitles = ["Flow", "Volume", "Poes", "Pgas", "Pdi"]
    if len(entropycolumns) > 0:
        for i in range(0, entropycolumns.shape[1]):
            _checkcolumn("Entropy column #" + str(i + 1), entropycolumns[:, i])
            collens.append(len(entropycolumns[:, i]))
            coltitles.append("Entropy #" + str(i + 1))
    if len(emgcolumns) > 0:
        for i in range(0, emgcolumns.shape[1]):
            _checkcolumn("EMG column #" + str(i + 1), emgcolumns[:, i])
            collens.append(len(emgcolumns[:, i]))
            coltitles.append("EMG #" + str(i + 1))
    if not _alleq(collens):
        cols = "Column lengths:\n" + "".join(
            f"{coltitles[s]}: {collens[s]} observations.\n" for s in range(len(collens)))
        raise DataValidationError(
            "Data column lengths differ. All columns must have the same number of "
            "observations.\n" + cols)


def load(filepath, settings):
    _, fext = os.path.splitext(filepath)
    fext = fext.lower()                        # DATA.CSV / export.TXT (common on Windows)
    d = settings.input.data

    def _cols_from_df(df):
        n = df.shape[1]
        col = lambda v, name: _column(v, name, n, filepath)          # noqa: E731
        flow = df.iloc[:, col(d.column_flow, "Flow channel")].to_numpy()
        volume = ([] if np.isnan(d.column_volume)
                  else df.iloc[:, col(d.column_volume, "Volume channel")].to_numpy())
        poes = df.iloc[:, col(d.column_poes, "Oesophageal pressure channel")].to_numpy()
        pgas = df.iloc[:, col(d.column_pgas, "Gastric pressure channel")].to_numpy()
        pdi = df.iloc[:, col(d.column_pdi, "Trans-diaphragmatic pressure channel")].to_numpy()
        # NO .squeeze() on these two: entropy/EMG are (samples, channels) matrices and every
        # consumer indexes them 2-D. Squeezing collapsed a single-channel selection to 1-D,
        # which made validatedata's .shape[1] raise "IndexError: tuple index out of range"
        # before the recording had even finished loading. The .mat branch below uses
        # np.column_stack and was always correct; this keeps the two paths honest. For >= 2
        # channels the squeeze was a no-op, so existing analyses are unaffected.
        ent_ix = [col(c, "Entropy channel") for c in d.columns_entropy]
        emg_ix = [col(c, "EMG channel") for c in d.columns_emg]
        ent = [] if not ent_ix else df.iloc[:, ent_ix].to_numpy()
        emg = [] if not emg_ix else df.iloc[:, emg_ix].to_numpy()
        return flow, volume, poes, pgas, pdi, ent, emg

    def loadxls(f):
        return _cols_from_df(pd.read_excel(f))

    def loadcsv(f):
        # honour the configured decimal separator (loadtxt already did): European Excel
        # CSVs are ';'-separated with a ',' decimal, and would otherwise mis-split.
        dec = settings.input.format.decimalcharacter
        return _cols_from_df(_read_table(f, sep=";" if dec == "," else ",", decimal=dec))

    def loadtxt(f):
        return _cols_from_df(_read_table(f, sep='\t', decimal=settings.input.format.decimalcharacter))

    def loadmat(f):
        try:
            data = OrderedDict(sio.loadmat(f))
            if settings.input.format.matlabfileformat == 2:
                cols = list(data.items())
                get = lambda i: cols[i - 1][1].squeeze()
            else:
                cols = list(data["data_block1"])
                get = lambda i: cols[i - 1]
        except Exception as e:
            raise ImportError(
                "Cannot load MATLAB file – verify the MATLAB file format setting "
                "(windows/mac). Only simple LabChart exports are supported; otherwise "
                "export to CSV.") from e
        # same resolution as the DataFrame path: a negative index would otherwise wrap
        # round to the end of the list and analyse a wholly unrelated channel.
        n = len(cols)
        get1 = lambda v, name: get(_column(v, name, n, f) + 1)       # noqa: E731
        flow = get1(d.column_flow, "Flow channel")
        volume = [] if np.isnan(d.column_volume) else get1(d.column_volume, "Volume channel")
        poes = get1(d.column_poes, "Oesophageal pressure channel")
        pgas = get1(d.column_pgas, "Gastric pressure channel")
        pdi = get1(d.column_pdi, "Trans-diaphragmatic pressure channel")
        ent = ([] if len(d.columns_entropy) == 0 else
               np.column_stack([get1(c, "Entropy channel") for c in d.columns_entropy]))
        emg = ([] if len(d.columns_emg) == 0 else
               np.column_stack([get1(c, "EMG channel") for c in d.columns_emg]))
        return flow, volume, poes, pgas, pdi, ent, emg

    loaders = {'.xls': loadxls, '.xlsx': loadxls, '.csv': loadcsv, '.mat': loadmat, '.txt': loadtxt}
    if fext not in loaders:
        raise DataValidationError(f"Unsupported input file type: {fext}")
    flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns = loaders[fext](filepath)

    flow = flow.squeeze()
    if len(volume) > 0:
        volume = volume.squeeze()
    poes, pgas, pdi = poes.squeeze(), pgas.squeeze(), pdi.squeeze()

    if settings.processing.mechanics.inverseflow:
        flow = -flow
    if settings.processing.mechanics.integratevolumefromflow:
        xval = np.linspace(0, len(flow) / settings.input.format.samplingfrequency, len(flow))
        volume = np.concatenate([-sp.integrate.cumulative_trapezoid(flow, xval), [0.0000001]])
    if settings.processing.mechanics.inversevolume:
        volume = -volume

    validatedata(flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings)
    return flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns
