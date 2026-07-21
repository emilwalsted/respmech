"""Assemble per-breath / per-file / processed result tables from computed breaths.

Faithful port of the legacy table-building logic (``savedataindividual``,
``getbreathdata``, ``processoutliers``, ``getprocesseddata``) but returning
DataFrames instead of writing files. One fix (legacy bug #3): the processed-data
export builds EMG column names from the actual channel count instead of a hardcoded
``EMG1..EMG5``.
"""
import numpy as np
import pandas as pd


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
        dfwob = pd.DataFrame(breath["wob"], index=[0])
        dfmech.insert(loc=0, column="breath_no", value=breath["number"])
        dfmech = dfmech.join(dfwob, how="outer", sort=False)

        if len(emgcols) > 0:
            dfmech = dfmech.join(_getbreathdata(breath, "rms", "rms_col_", ['rms_max', 'rms_mean'], emgcols), how="outer", sort=False)
            dfmech = dfmech.join(_getbreathdata(breath, "rms_insp", "rms_insp_col_", ['rms_insp_max', 'rms_insp_mean'], emgcols), how="outer", sort=False)
            dfmech = dfmech.join(_getbreathdata(breath, "rms_exp", "rms_exp_col_", ['rms_exp_max', 'rms_exp_mean'], emgcols), how="outer", sort=False)
            dfmech = dfmech.join(_getbreathdata(breath, "intemg", "integral_emg_col_", ['integralemg_max', 'integralemg_mean'], emgcols), how="outer", sort=False)
            dfmech = dfmech.join(_getbreathdata(breath, "intemg_insp", "integral_emg_insp_col_", ['integralemg_insp_max', 'integralemg_insp_mean'], emgcols), how="outer", sort=False)
            dfmech = dfmech.join(_getbreathdata(breath, "intemg_exp", "integral_emg_exp_col_", ['integralemg_exp_max', 'integralemg_exp_mean'], emgcols), how="outer", sort=False)

            # Opt-in cardiac-gated peak EMG: extra columns alongside the existing ones, never
            # in place of them. Guarded on the setting, so with the feature off the sheet is
            # byte-identical to before.
            if getattr(settings.processing.emg, "robust_peak", None) and settings.processing.emg.robust_peak.enabled:
                dfmech = dfmech.join(_getbreathdata(breath, "rms_gated", "rms_gated_col_", ['rms_gated_max', 'rms_gated_mean'], emgcols), how="outer", sort=False)
                dfmech = dfmech.join(_getbreathdata(breath, "rms_gated_insp", "rms_gated_insp_col_", ['rms_gated_insp_max', 'rms_gated_insp_mean'], emgcols), how="outer", sort=False)
                dfmech = dfmech.join(_getbreathdata(breath, "rms_gated_exp", "rms_gated_exp_col_", ['rms_gated_exp_max', 'rms_gated_exp_mean'], emgcols), how="outer", sort=False)

        if len(entcols) > 0:
            dfmech = dfmech.join(_getbreathdata(breath, "entropy", "sample_entropy_col_", ['sample_entropy_max', 'sample_entropy_min', 'sample_entropy_mean'], entcols), how="outer", sort=False)
            dfmech = dfmech.join(_getbreathdata(breath, "entropy_insp", "sample_entropy_insp_col_", ['sample_entropy_insp_max', 'sample_entropy_insp_min', 'sample_entropy_insp_mean'], entcols), how="outer", sort=False)
            dfmech = dfmech.join(_getbreathdata(breath, "entropy_exp", "sample_entropy_exp_col_", ['sample_entropy_exp_max', 'sample_entropy_exp_min', 'sample_entropy_exp_mean'], entcols), how="outer", sort=False)

        mechs = dfmech if len(mechs) == 0 else pd.concat([mechs, dfmech], sort=False)

    if settings.processing.emg.outlierrmssdlimit > 0:
        mechs = processoutliers(mechs, settings)

    ret = pd.DataFrame(mechs.mean()).T
    ret = ret.drop(columns="breath_no")
    ret.insert(loc=0, column="file", value=file)
    return mechs, ret


def build_processed_data(breaths, settings):
    """Return the trimmed per-breath processed signals as a DataFrame. Faithful to
    legacy ``getprocesseddata`` (merge-on-index + dropna) but with EMG column names
    following the actual channel count (fixes bug #3, which hardcoded EMG1..EMG5)."""
    emgcols = settings.input.data.columns_emg
    fs = settings.input.format.samplingfrequency
    processeddata = []
    for breathno in breaths:
        breath = breaths[breathno]
        if settings.output.data.includeignoredbreaths or not breath["ignored"]:
            times = np.arange(0, len(breath["flow"]) - 1, dtype=int) / fs
            df = pd.DataFrame(times, columns=["Time"])
            bnos = np.arange(0, len(breath["flow"]) - 1, dtype=int) * 0 + breathno
            df = pd.merge(df, pd.DataFrame(bnos, columns=["Breathno"]), how="outer", left_index=True, right_index=True)
            for name, key in (("Flow", "flow"), ("Volume", "volume"), ("Poes", "poes"),
                              ("Pgas", "pgas"), ("Pdi", "pdi")):
                df = pd.merge(df, pd.DataFrame(breath[key], columns=[name]), how="outer", left_index=True, right_index=True)
            if len(emgcols) > 0:
                emg = np.asarray(breath["emgcols"])
                names = [f"EMG{emgcols[i]}" for i in range(emg.shape[1])]
                df = pd.merge(df, pd.DataFrame(emg, columns=names), how="outer", left_index=True, right_index=True)
            processeddata = df if len(processeddata) == 0 else pd.concat([processeddata, df], sort=False)
    return processeddata.dropna() if len(processeddata) else pd.DataFrame()
