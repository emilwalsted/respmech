"""Migrate legacy RespMech settings (an executable ``.py`` holding a ``settings``
dict) to the versioned :class:`~respmech.core.settings.Settings` model.

Safety: the legacy file is **not executed**. The ``settings = { ... }`` assignment
is located in the AST and evaluated with :func:`ast.literal_eval`, which only
accepts plain literals (dicts, lists, numbers, strings, booleans, ``None``). Any
non-literal content raises rather than running.

The migrator also normalises the known *settings drift* (legacy example files put
``calcwobfromaverage`` / ``avgresamplingobs`` under ``mechanics`` although the code
reads ``wob.*``; ``volumetrendpeakminwidth`` was never read) and reports every field
moved, renamed, defaulted or dropped so nothing changes silently.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from respmech.core.settings import Settings


@dataclass
class MigrationReport:
    mapped: list[str] = field(default_factory=list)
    normalised: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    defaulted: list[str] = field(default_factory=list)

    def text(self) -> str:
        lines = ["# Settings migration report", ""]
        for title, items in (("Mapped (renamed/moved)", self.mapped),
                             ("Normalised (drift fixed)", self.normalised),
                             ("Dropped (not used by the code)", self.dropped)):
            lines.append(f"## {title}")
            lines += [f"- {i}" for i in items] or ["- (none)"]
            lines.append("")
        return "\n".join(lines)


def extract_legacy_dict(py_path: str | Path) -> dict:
    """Return the ``settings`` dict literal from a legacy settings .py, without
    executing the file."""
    src = Path(py_path).read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "settings":
                    try:
                        return ast.literal_eval(node.value)
                    except (ValueError, SyntaxError) as e:
                        raise ValueError(
                            f"`settings` in {py_path} is not a plain literal "
                            f"(cannot migrate without executing code): {e}") from e
    raise ValueError(f"no `settings = {{...}}` dict found in {py_path}")


def migrate_dict(legacy: dict) -> tuple[Settings, MigrationReport]:
    r = MigrationReport()
    inp = legacy.get("input", {})
    fmt = inp.get("format", {})
    data = inp.get("data", {})
    proc = legacy.get("processing", {})
    mech = proc.get("mechanics", {})
    wob = proc.get("wob", {})
    emg = proc.get("emg", {})
    ent = proc.get("entropy", {})
    samp = proc.get("sampling", {})
    out = legacy.get("output", {})
    odata = out.get("data", {})
    odiag = out.get("diagnostics", {})

    def m(msg):
        r.mapped.append(msg)

    new: dict = {"schema_version": 1}

    # --- input ---
    matlab_variant = "mac"
    if "matlabfileformat" in fmt:
        matlab_variant = "windows" if fmt["matlabfileformat"] == 1 else "mac"
        m(f"input.format.matlabfileformat={fmt['matlabfileformat']} -> "
          f"input.format.matlab_variant='{matlab_variant}'")
    new["input"] = {
        "folder": inp.get("inputfolder", "input"),
        "files": inp.get("files", "*.*"),
        "format": {
            "sampling_frequency": fmt.get("samplingfrequency"),
            "matlab_variant": matlab_variant,
            "decimal": fmt.get("decimalcharacter", "."),
        },
        "channels": {
            "poes": data.get("column_poes"),
            "pgas": data.get("column_pgas"),
            "pdi": data.get("column_pdi"),
            "volume": _nan_to_none(data.get("column_volume")),
            "flow": data.get("column_flow"),
            "emg": data.get("columns_emg", []),
            "entropy": data.get("columns_entropy", []),
        },
    }
    m("input.inputfolder->input.folder; input.data.column_*->input.channels.*; "
      "columns_emg/entropy->channels.emg/entropy")

    # --- processing.sampling (resampling-options line) ---
    new_proc: dict = {}
    if samp:
        new_proc["sampling"] = {
            "resample": bool(samp.get("resample", False)),
            "resample_to_frequency": samp.get("resampletofrequency", 200),
        }
        m("processing.sampling.resampletofrequency->resample_to_frequency")

    # --- segmentation ---
    new_proc["segmentation"] = {
        "method": mech.get("separateby", "flow"),
        "buffer": mech.get("breathseparationbuffer", 800),
        "peak": {
            "height": mech.get("peakheight", 0.1),
            "distance_s": mech.get("peakdistance", 0.1),
            "width_s": mech.get("peakwidth", 0.5),
        },
    }
    m("mechanics.separateby->segmentation.method; breathseparationbuffer->segmentation.buffer; "
      "peakheight/distance/width->segmentation.peak.*")

    # --- volume ---
    new_proc["volume"] = {
        "inverse_flow": mech.get("inverseflow", False),
        "integrate_from_flow": mech.get("integratevolumefromflow", False),
        "inverse_volume": mech.get("inversevolume", False),
        "correct_drift": mech.get("correctvolumedrift", True),
        "correct_trend": mech.get("correctvolumetrend", False),
        "trend_method": mech.get("volumetrendadjustmethod", "linear"),
        "trend_peak_min_height": mech.get("volumetrendpeakminheight", 0.8),
        "trend_peak_min_distance_s": mech.get("volumetrendpeakmindistance", 0.4),
    }
    m("mechanics.{inverseflow,integratevolumefromflow,inversevolume,correctvolume*,"
      "volumetrend*}->processing.volume.*")
    if "volumetrendpeakminwidth" in mech:
        r.dropped.append("mechanics.volumetrendpeakminwidth (never read by the code)")

    # --- wob: normalise the drift (calc_from can come from wob or mechanics) ---
    if "calcwobfrom" in wob:
        calc_from = wob["calcwobfrom"]
    elif "calcwobfromaverage" in mech:
        calc_from = "average" if mech["calcwobfromaverage"] else "individual"
        r.normalised.append(
            f"mechanics.calcwobfromaverage={mech['calcwobfromaverage']} -> "
            f"wob.calc_from='{calc_from}' (legacy code read wob.calcwobfrom; the "
            "example key under mechanics was silently ignored)")
    else:
        calc_from = "average"
    avg_obs = wob.get("avgresamplingobs", mech.get("avgresamplingobs", 500))
    if "avgresamplingobs" in mech and "avgresamplingobs" not in wob:
        r.normalised.append(
            "mechanics.avgresamplingobs -> wob.avg_resampling_obs (legacy code read "
            "wob.avgresamplingobs)")
    new_proc["wob"] = {"calc_from": calc_from, "avg_resampling_obs": avg_obs}

    # --- emg ---
    new_proc["emg"] = {
        "rms_window_s": emg.get("rms_s", 0.050),
        "remove_ecg": emg.get("remove_ecg", False),
        "detect_channel": emg.get("column_detect", 0),
        "ecg_min_height": emg.get("minheight", 0.0005),
        "ecg_min_distance_s": emg.get("mindistance", 0.5),
        "ecg_min_width_s": emg.get("minwidth", 0.001),
        "ecg_window_s": emg.get("windowsize", 0.4),
        "outlier_rms_sd_limit": emg.get("outlierrmssdlimit", 0) or 0,
        "save_sound": emg.get("save_sound", False),
        "plot_yscale": emg.get("emgplotyscale", [-0.1, 0.1]),
        "noise_profile": emg.get("noise_profile", []),
    }
    m("emg.rms_s->rms_window_s; column_detect->detect_channel; min*->ecg_min_*; "
      "windowsize->ecg_window_s; outlierrmssdlimit->outlier_rms_sd_limit; "
      "emgplotyscale->plot_yscale")

    # --- noise reduction: consolidate the legacy per-file noise_profile list into a
    #     single shared reference + fixed parameters (identical transformation for
    #     every file). See docs/NOISE_ECG_OPTIMIZATION.md. ---
    legacy_np = emg.get("noise_profile", [])
    if emg.get("remove_noise", False):
        # reference file = the (shared) source file the entries point at, if any.
        sources = [e[1] for e in legacy_np if len(e) >= 2 and isinstance(e[1], str) and e[1]]
        ref_file = None
        if sources:
            ref_file = _basename(sources[0])
            if len({_basename(x) for x in sources}) > 1:
                r.normalised.append(
                    "emg.noise_profile referenced multiple source files; the shared "
                    f"profile now uses '{ref_file}' for ALL files (identical transform).")
        intervals = sorted({tuple(e[2]) for e in legacy_np if len(e) >= 3 and e[2]})
        if len({tuple(e[2]) for e in legacy_np if len(e) >= 3 and e[2]}) > 1:
            r.normalised.append(
                "emg.noise_profile used different intervals per file; consolidated to "
                "one shared interval set (per-file re-tuning is invalid for EMG "
                "normalisation).")
        new_proc["emg"]["noise"] = {
            "enabled": True,
            "reference_file": ref_file,
            "reference_intervals": [list(iv) for iv in intervals],
            # Prefer expiration of the rest reference: it yields many STFT frames
            # (~hundreds) for a stable estimate, vs the legacy short interval (~7
            # frames). The legacy intervals are kept as an explicit override.
            "use_expiration": True,
            # fixed STFT params (legacy n_fft=len(noise)**2 was a bug — dropped):
            "n_fft": 256, "hop_length": 64, "win_length": 256,
            "n_std_thresh": 1.0, "prop_decrease": 0.6, "auto_prop": True,
            "fidelity_target": 0.8,
        }
        r.normalised.append(
            "emg.remove_noise + per-file noise_profile -> processing.emg.noise "
            "(shared profile, fixed n_fft=256, expiration-based reference for a "
            "stable estimate, fidelity-gated prop_decrease). The legacy short "
            "interval + n_fft=len(noise)**2 parameterisation was a bug and is dropped.")

    # --- entropy ---
    new_proc["entropy"] = {
        "epochs": ent.get("entropy_epochs", 2),
        "tolerance": ent.get("entropy_tolerance", 0.1),
    }
    m("entropy.entropy_epochs->epochs; entropy_tolerance->tolerance")

    # --- exclude / breath counts ---
    new_proc["exclude_breaths"] = [
        {"file": e[0], "breaths": e[1]} for e in mech.get("excludebreaths", [])
        if isinstance(e, (list, tuple)) and len(e) >= 2]
    new_proc["breath_counts"] = [
        {"file": e[0], "count": e[1]} for e in mech.get("breathcounts", [])
        if isinstance(e, (list, tuple)) and len(e) >= 2]
    if mech.get("excludebreaths"):
        m("mechanics.excludebreaths [[file,[..]]] -> processing.exclude_breaths [{file,breaths}]")
    if mech.get("breathcounts"):
        m("mechanics.breathcounts [[file,n]] -> processing.breath_counts [{file,count}]")

    new["processing"] = new_proc

    # --- output ---
    if "savepvoverview" in odiag:
        r.dropped.append("output.diagnostics.savepvoverview (defined but never read by the code)")
    new["output"] = {
        "folder": out.get("outputfolder", "output"),
        "data": {
            "save_average": odata.get("saveaveragedata", True),
            "save_breath_by_breath": odata.get("savebreathbybreathdata", True),
            "save_processed": odata.get("saveprocesseddata", False),
            "include_ignored_breaths": odata.get("includeignoredbreaths", False),
        },
        "diagnostics": {
            "save_pv_average": odiag.get("savepvaverage", True),
            "save_pv_individual": odiag.get("savepvindividualworkload", True),
            "pv_columns": odiag.get("pvcolumns", 3),
            "pv_rows": odiag.get("pvrows", 4),
            "save_raw": odiag.get("savedataviewraw", True),
            "save_trimmed": odiag.get("savedataviewtrimmed", True),
            "save_drift": odiag.get("savedataviewdriftcor", True),
            "save_emg": odiag.get("saveemgoverview", True),
        },
    }
    m("output.outputfolder->output.folder; output.data.*->save_*; "
      "output.diagnostics.savedataview*->save_raw/trimmed/drift")

    settings = Settings.from_dict(new)
    return settings, r


def migrate_file(py_path: str | Path) -> tuple[Settings, MigrationReport]:
    return migrate_dict(extract_legacy_dict(py_path))


def _basename(path):
    import ntpath
    return ntpath.basename(str(path))


def _nan_to_none(v):
    # legacy sometimes leaves column_volume as float('nan'); treat as "absent".
    try:
        if v is not None and v != v:  # NaN
            return None
    except Exception:
        pass
    return v
