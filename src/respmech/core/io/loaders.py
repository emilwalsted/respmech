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

from respmech.core.quality import detect_constant_channel, detect_merged_time_blocks


class DataValidationError(ValueError):
    """Raised when an input column is missing, non-numeric, NaN, or mismatched."""


def _column(value, name, ncols, filepath):
    """Resolve a 1-based column setting to a 0-based index, or say exactly what is wrong.

    This is the only place that knows both the setting and the file, so it is the only place
    the range check can be right. Without it an unassigned channel reached pandas as ``None``
    and surfaced as "unsupported operand type(s) for -: 'NoneType' and 'int'", and — worse —
    column 0 became ``iloc[:, -1]``, silently analysing the LAST column of the recording
    instead of reporting anything at all."""
    # NaN reaches here only from a hand-edited settings file, or a caller (loaders.py's
    # own optional-channel branches below) that already tested for absence itself; the
    # model uses None, and _legacy_ns.to_legacy_ns maps every optional column (volume,
    # poes, pgas, pdi; flow forward-compatibly, see its own comment there) to NaN.
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


def _absent(value):
    """True when a column setting means 'not assigned' — the same test ``_column`` uses,
    exposed so a caller can skip resolving the column entirely and return an empty
    channel instead of raising. Volume already did this (``np.isnan(column_volume)``,
    the model's only channel optional today); poes/pgas/pdi now do the same, and flow's
    resolution deliberately still goes straight through ``_column`` unconditionally
    below, unchanged, because ``Settings.validate()`` still requires it (M-08) and
    ``tests/unit/test_unassigned_channels.py`` pins that an unassigned flow raises."""
    return value is None or (isinstance(value, float) and np.isnan(value))


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
    # (validation text, array, length-check title) triples. A channel that is legitimately
    # ABSENT (an empty array — volume today, poes/pgas/pdi from M-08 onward) is dropped
    # here, before _checkcolumn and before it can ever enter the length-consistency check
    # below: an absent channel has nothing to validate, and its length of 0 is not a
    # mismatch against the recording's real sample count, it is simply not part of this
    # recording at all.
    triples = [
        ("Flow column", flow, "Flow"),
        ("Volume column", volume, "Volume"),
        ("Oesophageal pressure column", poes, "Poes"),
        ("Gastric pressure column", pgas, "Pgas"),
        ("Trans-diaphragmatic pressure column", pdi, "Pdi"),
    ]
    if len(entropycolumns) > 0:
        for i in range(0, entropycolumns.shape[1]):
            triples.append(("Entropy column #" + str(i + 1), entropycolumns[:, i],
                            "Entropy #" + str(i + 1)))
    if len(emgcolumns) > 0:
        for i in range(0, emgcolumns.shape[1]):
            triples.append(("EMG column #" + str(i + 1), emgcolumns[:, i],
                            "EMG #" + str(i + 1)))

    present = [(text, arr, title) for text, arr, title in triples if len(arr) > 0]
    for text, arr, _ in present:
        _checkcolumn(text, arr)

    collens = [len(arr) for _, arr, _ in present]
    coltitles = [title for _, _, title in present]
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
        # flow always resolves through _column unconditionally (see _absent's docstring):
        # unassigned flow must still raise today, unchanged from before this function grew
        # optional-channel support for its three siblings below.
        flow = df.iloc[:, col(d.column_flow, "Flow channel")].to_numpy()
        opt = lambda v, name: (np.asarray([], dtype=float) if _absent(v)          # noqa: E731
                               else df.iloc[:, col(v, name)].to_numpy())
        volume = opt(d.column_volume, "Volume channel")
        poes = opt(d.column_poes, "Oesophageal pressure channel")
        pgas = opt(d.column_pgas, "Gastric pressure channel")
        pdi = opt(d.column_pdi, "Trans-diaphragmatic pressure channel")
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
        flow = get1(d.column_flow, "Flow channel")          # unconditional, see _absent's docstring
        opt1 = lambda v, name: (np.asarray([], dtype=float) if _absent(v)          # noqa: E731
                                else get1(v, name))
        volume = opt1(d.column_volume, "Volume channel")
        poes = opt1(d.column_poes, "Oesophageal pressure channel")
        pgas = opt1(d.column_pgas, "Gastric pressure channel")
        pdi = opt1(d.column_pdi, "Trans-diaphragmatic pressure channel")
        ent = ([] if len(d.columns_entropy) == 0 else
               np.column_stack([get1(c, "Entropy channel") for c in d.columns_entropy]))
        emg = ([] if len(d.columns_emg) == 0 else
               np.column_stack([get1(c, "EMG channel") for c in d.columns_emg]))
        return flow, volume, poes, pgas, pdi, ent, emg

    loaders = {'.xlsx': loadxls, '.csv': loadcsv, '.mat': loadmat, '.txt': loadtxt}
    if fext not in loaders:
        raise DataValidationError(f"Unsupported input file type: {fext}")
    flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns = loaders[fext](filepath)

    # guarded on length, not just for volume: an absent channel is an empty ndarray now
    # (never a bare Python list), so .squeeze() would not crash even unguarded, but an
    # absent channel has nothing to squeeze either — same defensive shape for all five.
    if len(flow) > 0:
        flow = flow.squeeze()
    if len(volume) > 0:
        volume = volume.squeeze()
    if len(poes) > 0:
        poes = poes.squeeze()
    if len(pgas) > 0:
        pgas = pgas.squeeze()
    if len(pdi) > 0:
        pdi = pdi.squeeze()

    # flow is always present today (Settings.validate requires it; M-08 introduces
    # flow-less EMG-only sets), but guarding on len(flow) now means this section already
    # behaves correctly once that lands, instead of dividing by a zero-length flow's
    # sampling count or feeding cumulative_trapezoid an empty pair.
    if settings.processing.mechanics.inverseflow and len(flow) > 0:
        flow = -flow
    if settings.processing.mechanics.integratevolumefromflow and len(flow) > 0:
        xval = np.linspace(0, len(flow) / settings.input.format.samplingfrequency, len(flow))
        volume = np.concatenate([-sp.integrate.cumulative_trapezoid(flow, xval), [0.0000001]])
    if settings.processing.mechanics.inversevolume:
        volume = -volume

    # validatedata() must see the RAW (possibly int64, possibly still-object) arrays:
    # _checkcolumn's text-column check (above) only works before a cast to float, since
    # casting a genuinely non-numeric column first would turn its friendly
    # DataValidationError ("... contains text values...") into a bare, uncaught
    # "could not convert string to float" ValueError instead.
    validatedata(flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings)

    # Downstream arithmetic (VMR's np.divide, PTP baselines, ...) assumes float64
    # throughout. A column that is all-integer in the source file (a constant-zero
    # "dummy" channel wired to an unused pressure port, or a genuinely integer-valued
    # instrument export) otherwise reaches pandas/scipy as int64, and an int64
    # zeros_like()/divide() combination then raises UFuncTypeError instead of producing
    # a result. Cast every channel here, once validation has confirmed it is safe to, so
    # nothing downstream needs to care about the input file's original dtype.
    # np.asarray(..., dtype=float) is a no-op for the already-float golden fixtures and
    # handles an unassigned channel's bare ``[]`` the same way as a populated array, so
    # no length guard is needed (matches the idiom used throughout core/ for this).
    flow = np.asarray(flow, dtype=float)
    volume = np.asarray(volume, dtype=float)
    poes = np.asarray(poes, dtype=float)
    pgas = np.asarray(pgas, dtype=float)
    pdi = np.asarray(pdi, dtype=float)
    entropycolumns = np.asarray(entropycolumns, dtype=float)
    emgcolumns = np.asarray(emgcolumns, dtype=float)

    return flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns


# --- cheap pre-run quality probes --------------------------------------------
#
# These read a NEW-style ``core.settings.Settings`` (not the legacy namespace ``load()``
# above uses) and a file path -- the same ``(settings, file_path) -> ...`` shape as
# ``ui.workers``'s manifest probers (``peek_columns``/``probe_sampling_frequency``/
# ``peek_header_warning``), so ``ui.manifest.build_manifest`` can default to them
# directly. They live here (not in ``core.quality``, which stays pandas-free) because
# they need the same tolerant CSV/TSV reader as ``load()`` above; ``respmech validate``
# (``cli/__main__.py``) calls them directly, without ever importing ``ui.manifest`` or
# Qt.

def _delimited_ext(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    return ext if ext in (".csv", ".txt") else None


def _delimiter_for(ext, decimal):
    """The same '.txt is always tab; .csv is comma, or semicolon under a comma decimal'
    pairing used throughout this module (``load``'s ``loadcsv``/``loadtxt``,
    ``probe_data_columns``) and in ``ui.workers`` (``peek_columns``,
    ``probe_sampling_frequency``, ``detect_decimal``) -- pulled out here so the two new
    probes below don't carry a FOURTH/FIFTH copy of the same three-way branch (self-
    review finding)."""
    return "\t" if ext == ".txt" else (";" if decimal == "," else ",")


def probe_merged_time_blocks(settings, file_path, *, max_rows=5000):
    """Cheap per-file probe for :func:`respmech.core.quality.detect_merged_time_blocks`:
    reads only column 0 (the time axis), capped at 5000 rows for the same reason
    ``ui.workers.probe_sampling_frequency`` caps its own column-0 read -- a live Setup
    scan re-probes a whole folder on every input edit. A merge that starts within the
    capped window (as the reported case did -- three LabChart blocks overlapping from
    the very first sample) is still caught; one that only begins later in a very long
    recording is not -- the same documented trade-off ``peek_header_warning`` makes for
    its own 8 KB head-only sniff. Returns ``None`` for .xlsx/.mat (no cheap capped read
    available) or any unreadable/short file.

    KNOWN, ACCEPTED COST (not fixed here): this reads column 0 with its own
    ``_read_table`` call, separate from ``ui.workers.probe_sampling_frequency``'s
    near-identical column-0 read on the same path in the same ``build_manifest`` scan
    -- measured at roughly 2x the per-file read cost for the pair (~10 ms -> ~20-30 ms
    on an 18 MB / 60k-row file), paid once per scan thanks to ``build_manifest``'s own
    cache. Unifying the two into one shared read would mean moving
    ``detect_sampling_frequency`` down out of ``ui.workers`` (Qt-adjacent purely by
    file location, not by need -- it is already pure numpy) into this module or
    ``core.quality``, which is a real, worthwhile refactor but a larger one than this
    probe's own addition -- left for a future ticket rather than risked here."""
    ext = _delimited_ext(file_path)
    if ext is None:
        return None
    fmt = settings.input.format
    dec = getattr(fmt, "decimal", ".") or "."
    sep = _delimiter_for(ext, dec)
    try:
        df = _read_table(file_path, sep=sep, decimal=dec, usecols=[0], nrows=max_rows)
    except Exception:                       # noqa: BLE001 — best-effort, never blocks the scan
        return None
    if df.shape[1] == 0:
        return None
    col = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    return detect_merged_time_blocks(col)


#: (label, ``Channels`` attribute) for the single-column roles checked by
#: :func:`probe_constant_channels` — deliberately excludes ``volume`` (often DERIVED via
#: ``processing.mechanics.integrate_from_flow`` rather than a real recorded channel; not
#: named in the ticket this probe was built for) and mirrors ``core.io.loaders.
#: validatedata``'s own channel labels where they overlap.
_CONSTANT_CHECK_SINGLE = (("Flow", "flow"), ("Poes", "poes"), ("Pgas", "pgas"), ("Pdi", "pdi"))


def _assigned_column(value):
    """A channel setting as a plain positive int, or ``None`` if it is not really
    assigned yet -- mirrors ``_column``'s own ``None``/NaN guard (a hand-edited TOML,
    or a schema field that is optional elsewhere, can carry ``float('nan')`` rather than
    ``None`` for "unset"; a bare NaN is truthy and compares False against every bound,
    so an unguarded ``if col`` / ``col < 1`` here let one straight through to a crashing
    ``df.iloc[:, nan - 1]`` — self-review finding). Never raises on a stray non-numeric
    value either (a hand-edited TOML could carry a string)."""
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def probe_constant_channels(settings, file_path, *, max_rows=5000):
    """Cheap per-file probe for :func:`respmech.core.quality.detect_constant_channel`,
    run over every ASSIGNED channel (flow, Poes, Pgas, Pdi, each EMG, each entropy
    column) — returns a tuple of ``"<name> (column <n>)"`` strings, one per channel
    found constant, or ``()`` when none are (including when no channel is assigned yet,
    or the file cannot be read this cheaply). Capped at the same 5000 rows as
    :func:`probe_merged_time_blocks`: a channel that is constant for the whole
    recording — the reported case, and the one that matters for the hard block in
    ``core.compute.separateintobreathsbyflow`` — reads as constant in the first 5000
    rows too; a channel that only goes flat later is not caught here (same class of
    trade-off, not a new one)."""
    ext = _delimited_ext(file_path)
    if ext is None:
        return ()
    ch = settings.input.channels
    named = [(label, getattr(ch, attr, None)) for label, attr in _CONSTANT_CHECK_SINGLE]
    for i, c in enumerate(ch.emg or []):
        named.append((f"EMG #{i + 1}", c))
    for i, c in enumerate(ch.entropy or []):
        named.append((f"Entropy #{i + 1}", c))
    assigned = [(name, col) for name, col in
               ((name, _assigned_column(raw)) for name, raw in named) if col is not None]
    if not assigned:
        return ()
    fmt = settings.input.format
    dec = getattr(fmt, "decimal", ".") or "."
    sep = _delimiter_for(ext, dec)
    try:
        df = _read_table(file_path, sep=sep, decimal=dec, nrows=max_rows)
    except Exception:                       # noqa: BLE001 — best-effort, never blocks the scan
        return ()
    ncols = df.shape[1]
    out = []
    for name, col in assigned:
        if col < 1 or col > ncols:
            continue                        # out of range: a real mismatch, reported at run time
        series = pd.to_numeric(df.iloc[:, col - 1], errors="coerce").to_numpy(dtype=float)
        if detect_constant_channel(series):
            out.append(f"{name} (column {col})")
    return tuple(out)
