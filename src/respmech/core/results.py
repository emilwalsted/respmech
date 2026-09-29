"""Assemble per-breath / per-file / processed result tables from computed breaths.

Faithful port of the legacy table-building logic (``savedataindividual``,
``getbreathdata``, ``processoutliers``, ``getprocesseddata``) but returning
DataFrames instead of writing files. One fix (legacy bug #3): the processed-data
export builds EMG column names from the actual channel count instead of a hardcoded
``EMG1..EMG5``.
"""
import numpy as np
import pandas as pd

from respmech.core.compute import NoBreathsError


def _getbreathdata(breath, datacol, colsprefix, appendcols, colsettings):
    df = pd.DataFrame(breath[datacol]).transpose()
    cols = [colsprefix + str(colsettings[x]) for x in range(0, len(colsettings))]
    cols = np.append(cols, appendcols)
    df.columns = cols
    return df


def processoutliers(data, settings):
    """Replace EMG RMS outliers (|rms/poes_mininsp| beyond N SD of the other
    breaths) with the mean of the other breaths' RMS."""
    ret = data
    ret["rms_poes"] = ret["rms_max"] / ret["poes_mininsp"]
    for _, row in data.iterrows():
        otherrows = data.loc[data["breath_no"] != row["breath_no"]]
        othermean = otherrows["rms_poes"].mean()
        othersd = otherrows["rms_poes"].std()
        sdmultiplier = settings.processing.emg.outlierrmssdlimit
        minval = othermean - othersd * sdmultiplier
        maxval = othermean + othersd * sdmultiplier
        if (row["rms_poes"] < minval) | (row["rms_poes"] > maxval):
            ret.loc[ret["breath_no"] == row["breath_no"], "rms_max"] = otherrows["rms_max"].mean()
            ret.loc[ret["breath_no"] == row["breath_no"], "rms_mean"] = otherrows["rms_mean"].mean()
    return ret.drop(columns="rms_poes")


def build_breath_table(file, breaths, settings):
    """Return (per_breath_df, average_row_df) for one file. ``per_breath_df`` is the
    '<file>.breathdata' Data sheet; ``average_row_df`` is that file's row in the
    merged 'Average breathdata'."""
    emgcols = settings.input.data.columns_emg
    entcols = settings.input.data.columns_entropy

    mechs = []
    for breathno in breaths:
        breath = breaths[breathno]
        if breath["ignored"]:
            continue
        dfmech = pd.DataFrame(breath["mechanics"], index=[0])
        dfmech.insert(loc=0, column="breath_no", value=breath["number"])
        if "wob" in breath:
            dfwob = pd.DataFrame(breath["wob"], index=[0])
            dfmech = dfmech.join(dfwob, how="outer", sort=False)
        # M-42: core.analysis.mfvl.attach stamps this dict on every tidal breath of a
        # file with a resolved (same-file) fvc reference — same join shape as wob.
        if "mfvl_ext" in breath:
            dfmfvl = pd.DataFrame(breath["mfvl_ext"], index=[0])
            dfmech = dfmech.join(dfmfvl, how="outer", sort=False)
        # core.analysis.pressure.attach (opt-in PEEPi): threshold-work columns stamped on
        # every tidal breath, joined after wob like the block above.
        if "pressure_ext" in breath:
            dfpress = pd.DataFrame(breath["pressure_ext"], index=[0])
            dfmech = dfmech.join(dfpress, how="outer", sort=False)
        # core.analysis.breathing_pattern.attach (opt-in): flow/volume-only pattern columns.
        if "breathing_pattern_ext" in breath:
            dfbp = pd.DataFrame(breath["breathing_pattern_ext"], index=[0])
            dfmech = dfmech.join(dfbp, how="outer", sort=False)

        has_phases = breath.get("has_phases", True)

        if len(emgcols) > 0:
            dfmech = dfmech.join(_getbreathdata(breath, "rms", "rms_col_", ['rms_max', 'rms_mean'], emgcols), how="outer", sort=False)
            if has_phases:
                dfmech = dfmech.join(_getbreathdata(breath, "rms_insp", "rms_insp_col_", ['rms_insp_max', 'rms_insp_mean'], emgcols), how="outer", sort=False)
                dfmech = dfmech.join(_getbreathdata(breath, "rms_exp", "rms_exp_col_", ['rms_exp_max', 'rms_exp_mean'], emgcols), how="outer", sort=False)
            dfmech = dfmech.join(_getbreathdata(breath, "intemg", "integral_emg_col_", ['integralemg_max', 'integralemg_mean'], emgcols), how="outer", sort=False)
            if has_phases:
                dfmech = dfmech.join(_getbreathdata(breath, "intemg_insp", "integral_emg_insp_col_", ['integralemg_insp_max', 'integralemg_insp_mean'], emgcols), how="outer", sort=False)
                dfmech = dfmech.join(_getbreathdata(breath, "intemg_exp", "integral_emg_exp_col_", ['integralemg_exp_max', 'integralemg_exp_mean'], emgcols), how="outer", sort=False)

            # Opt-in cardiac-gated peak EMG: extra columns alongside the existing ones, never
            # in place of them. Guarded on the setting, so with the feature off the sheet is
            # byte-identical to before.
            if getattr(settings.processing.emg, "robust_peak", None) and settings.processing.emg.robust_peak.enabled:
                dfmech = dfmech.join(_getbreathdata(breath, "rms_gated", "rms_gated_col_", ['rms_gated_max', 'rms_gated_mean'], emgcols), how="outer", sort=False)
                if has_phases:
                    dfmech = dfmech.join(_getbreathdata(breath, "rms_gated_insp", "rms_gated_insp_col_", ['rms_gated_insp_max', 'rms_gated_insp_mean'], emgcols), how="outer", sort=False)
                    dfmech = dfmech.join(_getbreathdata(breath, "rms_gated_exp", "rms_gated_exp_col_", ['rms_gated_exp_max', 'rms_gated_exp_mean'], emgcols), how="outer", sort=False)

            # whole_file-only diagnostics (core.analysis.segments): where in the recording
            # the peak EMG activity fell, and how it compares to the top-3 highest values in
            # the whole-file envelope. Never present for a real breath or a `separators`
            # segment (only `segments.whole_file` sets these three keys), so this join is a
            # pure no-op for every existing (has_phases=True) analysis.
            if 'rms_file_max' in breath:
                dfmech = dfmech.join(_getbreathdata(breath, "rms_file_max", "rms_file_max_col_", [], emgcols), how="outer", sort=False)
                dfmech = dfmech.join(_getbreathdata(breath, "t_rms_file_max", "t_rms_file_max_col_", [], emgcols), how="outer", sort=False)
                dfmech = dfmech.join(_getbreathdata(breath, "rms_file_top3", "rms_file_top3_col_", [], emgcols), how="outer", sort=False)

        if len(entcols) > 0:
            dfmech = dfmech.join(_getbreathdata(breath, "entropy", "sample_entropy_col_", ['sample_entropy_max', 'sample_entropy_min', 'sample_entropy_mean'], entcols), how="outer", sort=False)
            if has_phases:
                dfmech = dfmech.join(_getbreathdata(breath, "entropy_insp", "sample_entropy_insp_col_", ['sample_entropy_insp_max', 'sample_entropy_insp_min', 'sample_entropy_insp_mean'], entcols), how="outer", sort=False)
                dfmech = dfmech.join(_getbreathdata(breath, "entropy_exp", "sample_entropy_exp_col_", ['sample_entropy_exp_max', 'sample_entropy_exp_min', 'sample_entropy_exp_mean'], entcols), how="outer", sort=False)

        if "entropy_derived" in breath:
            derived = list(settings.input.data.entropy_derived)
            for key, prefix in (("entropy_derived", "sample_entropy_col_"),
                                ("entropy_derived_insp", "sample_entropy_insp_col_"),
                                ("entropy_derived_exp", "sample_entropy_exp_col_")):
                if key in breath:
                    dfmech = dfmech.join(_getbreathdata(breath, key, prefix, [], derived), how="outer", sort=False)

        mechs = dfmech if len(mechs) == 0 else pd.concat([mechs, dfmech], sort=False)

    if len(mechs) == 0:
        # No row was built, so there is no table and no average to take. The pipeline
        # already refuses such a file by name (compute.check_breaths); this keeps the
        # promise for any other caller instead of failing as 'list' has no attribute
        # 'mean' further down, or inside processoutliers.
        raise NoBreathsError(
            f"No breaths to build a result table from in {file} — every detected breath "
            f"is excluded, or none was detected.")

    # K-204: outlier_rms_sd_limit filters EMG RMS outliers (rms_max/poes_mininsp), so
    # without EMG channels there is no rms_max column to filter — processoutliers would
    # KeyError on it. A study-wide setting left on for an analysis that happens to have
    # no EMG channels (or, in the future, no Poes channel) must not fail every file; it
    # simply has nothing to do here. This is the sole owner of this guard.
    if settings.processing.emg.outlierrmssdlimit > 0 and len(emgcols) > 0 and 'poes_mininsp' in mechs.columns:
        mechs = processoutliers(mechs, settings)

    ret = pd.DataFrame(mechs.mean()).T
    ret = ret.drop(columns="breath_no")
    ret.insert(loc=0, column="file", value=file)
    return mechs, ret


def build_manoeuvre_table(manoeuvres: dict):
    """The 'Manoeuvres' sheet: one row per typed breath, from
    ``core.pipeline.FileResult.manoeuvres`` (``{breath_no: core.analysis.manoeuvres
    .extract(...) result}``). Returns ``None`` when ``manoeuvres`` is empty -- "the
    Manoeuvres sheet exists only when a typed breath is present" (M-29's own
    acceptance criterion) -- so a caller can join it in exactly like ``EMG
    normalised`` (``if df is not None and len(df):``).

    ``quality`` (a list of flag strings from ``core.analysis.manoeuvres.extract``) is
    joined into a single comma-separated cell -- a spreadsheet has no native list
    type, and an empty list becomes "" (a blank cell), never the literal text "[]"."""
    if not manoeuvres:
        return None
    rows = []
    for breath_no, fields in manoeuvres.items():
        row = {"breath_no": breath_no, "kind": fields.get("kind")}
        for key, value in fields.items():
            if key in ("kind",):
                continue
            row[key] = ", ".join(value) if key == "quality" else value
        rows.append(row)
    return pd.DataFrame(rows)


def build_processed_data(breaths, settings):
    """Return the trimmed per-breath processed signals as a DataFrame. Faithful to
    legacy ``getprocesseddata`` (merge-on-index + dropna) but with EMG column names
    following the actual channel count (fixes bug #3, which hardcoded EMG1..EMG5)."""
    emgcols = settings.input.data.columns_emg
    fs = settings.input.format.samplingfrequency
    # M-19: a "Breathkind" column is only ever ADDED when at least one breath ACTUALLY
    # WRITTEN to this table is typed -- an unconditional column (even one filled with ""
    # for every row) would be a new column present on EVERY analysis, including the
    # thousands that never use breath types, which is exactly the kind of always-present
    # new column tests/golden/test_golden.py's set-equality check would flag (see
    # docs/beslutninger.md's "bcnt/vefactor uændret med typede vejrtrækninger" entry for
    # the sibling reasoning). The check mirrors the per-breath emission predicate below
    # (self-review finding): on a flow-bearing signal set every typed breath is ALSO
    # `ignored=True` (the union with exclude_breaths), so with the default
    # `include_ignored_breaths=False` those breaths are never emitted at all -- computing
    # `has_typed` over every breath in the file (rather than just the ones this table will
    # actually contain) added a column that was present but blank on every row, which is
    # worse than not adding it. Checked ONCE for the whole file, not per breath, so the
    # column is either present on every emitted row or absent entirely.
    has_typed = any(
        breaths[no].get("kind") for no in breaths
        if settings.output.data.includeignoredbreaths or not breaths[no]["ignored"])
    processeddata = []
    for breathno in breaths:
        breath = breaths[breathno]
        if settings.output.data.includeignoredbreaths or not breath["ignored"]:
            # 'time' (not 'flow') is the row-count source: flow is the channel most likely
            # to become the ABSENT one in a future signal set (e.g. a Poes-only analysis),
            # while time is present on every breath regardless of which channels it carries.
            n = len(np.atleast_1d(breath["time"])) - 1
            times = np.arange(0, n, dtype=int) / fs
            df = pd.DataFrame(times, columns=["Time"])
            bnos = np.arange(0, n, dtype=int) * 0 + breathno
            df = pd.merge(df, pd.DataFrame(bnos, columns=["Breathno"]), how="outer", left_index=True, right_index=True)
            if has_typed:
                # An untyped breath in a file that has SOME typed breaths reports "" for
                # this column (not NaN) -- NaN would fall to the final .dropna() below
                # and silently delete every sample of every untyped breath in the file.
                kinds = np.full(n, breath.get("kind") or "", dtype=object)
                df = pd.merge(df, pd.DataFrame(kinds, columns=["Breathkind"]), how="outer",
                              left_index=True, right_index=True)
            for name, key in (("Flow", "flow"), ("Volume", "volume"), ("Poes", "poes"),
                              ("Pgas", "pgas"), ("Pdi", "pdi")):
                if len(breath[key]) == 0:
                    continue                        # absent channel (empty-array convention): no column
                df = pd.merge(df, pd.DataFrame(breath[key], columns=[name]), how="outer", left_index=True, right_index=True)
            if len(emgcols) > 0:
                emg = np.asarray(breath["emgcols"])
                names = [f"EMG{emgcols[i]}" for i in range(emg.shape[1])]
                df = pd.merge(df, pd.DataFrame(emg, columns=names), how="outer", left_index=True, right_index=True)
            processeddata = df if len(processeddata) == 0 else pd.concat([processeddata, df], sort=False)
    return processeddata.dropna() if len(processeddata) else pd.DataFrame()
