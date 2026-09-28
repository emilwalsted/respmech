"""Normalisation to a maximal manoeuvre (opt-in, ``processing.pressure.normalization``):
the inspiratory Poes/Pdi and the EMG of every tidal breath expressed against the same
person's own maximal inspiratory effort, and the tension-time indices built on it.

The reference is the ``max_insp``/``sniff`` breath a file resolves through
:func:`respmech.core.analysis.references.resolve_reference` (an explicit
``processing.references`` entry, then the group default, then the file's own typed
breath). What a maximal breath contributes (``poes_max_ref``, ``pdi_max_ref``,
``rms_max_ref``) was already measured by
:func:`respmech.core.analysis.manoeuvres.max_effort_from_breath`; this module only reads
those numbers back, per file, after the batch's main loop, and never recomputes them.
Nothing here touches ``core.compute``: the whole analysis is a post-pass over the
finished per-breath tables and the raw breath dicts, so no existing column changes with
it on or off, and it adds one sheet, "Pressure normalised", to each file's workbook.

Formulas (per tidal breath; ``ref`` is the file's reference, all pressures in cmH2O):

* ``poes_insp_swing = mean(insp.poes[:n_bw]) - poes_mininsp`` and
  ``pdi_insp_swing = pdi_maxinsp - mean(insp.pdi[:n_bw])``: how far the inspiratory
  pressure moved from the end-expiratory baseline, the baseline being the mean over the
  same short window at the phase start that ``calcptp`` uses (``n_bw``).
* ``poes_mean_insp = int_oesinsp / ti`` and ``pdi_mean_insp = int_pdiinsp / ti``: the mean
  baseline-referenced pressure over inspiration.
* ``<x>_pct = 100 * x / ref`` for the swing and the mean pressure.
* ``tt_es = int_oesinsp / (ttot * poes_max_ref)`` and
  ``tt_di = int_pdiinsp / (ttot * pdi_max_ref)``. This is the tension-time index
  ``(P_mean / P_max) * (Ti / Ttot)`` (Bellemare & Grassino 1982; Ramonatxo et al. 1995 for
  the oesophageal form), written with the pressure-time integral so ``Ti`` cancels. The
  oesophageal and the diaphragmatic index are different quantities with different
  thresholds (Ramonatxo: TTmus is about 2.1 times TTdi) and are never merged.
* ``rms_insp_max_pct = 100 * rms_insp_max / rms_max_ref`` and
  ``nrdi = rms_insp_max_pct * bf`` (neural respiratory drive index: EMG as a percentage
  of maximum times breathing rate, arbitrary units; Murphy et al. 2011). ``bf`` is the
  file's breath rate, constant within a file, so the file mean of ``nrdi`` equals
  ``mean(rms_insp_max_pct) * bf``.

Every quantity is NaN, with one notice per file, when it cannot be computed: no
reference resolves, the reference lacks that channel, or the reference is not a positive
number. The column set is decided by the signal set alone, never by whether a reference
resolved, so a file without one still writes the same columns (as NaN).

A nasal sniff has no mouth flow to segment on: the user types the breath that CONTAINS the
sniff, and the extractor takes the largest baseline-referenced swing over that whole typed
breath. Sniff Pdi and PImax Pdi are different numbers (Miller et al. 1985), so the kinds
behind a reference are recorded, and a reference that mixes them says so in a notice.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from respmech.core.analysis import manoeuvres as manoeuvreslib
from respmech.core.analysis import references as referenceslib
from respmech.core.analysis.signals import Capabilities
from respmech.core.settings import BreathRef

#: kinds whose Manoeuvres row carries the ``*_max_ref`` numbers this module reads
#: (``manoeuvres._MAX_EFFORT_KINDS``).
_MAX_EFFORT_KINDS = manoeuvreslib._MAX_EFFORT_KINDS

_SCALAR_KEYS = ("poes_max_ref", "pdi_max_ref", "rms_max_ref")
#: prefix of the per-channel EMG references ``max_effort_from_breath`` also writes.
_CHANNEL_PREFIX = "rms_max_ref_col_"


def _finite_positive(x) -> bool:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(x) and x > 0)


def _pct(x, ref) -> float:
    """``100 * x / ref``, NaN unless ``ref`` is a positive finite number."""
    if not _finite_positive(ref):
        return float("nan")
    return 100.0 * float(x) / float(ref)


def tension_time(pressure_integral, ttot, reference) -> float:
    """``pressure_integral / (ttot * reference)``: the tension-time index for one breath
    (see the module docstring); NaN unless ``ttot`` and ``reference`` are positive."""
    if not (_finite_positive(ttot) and _finite_positive(reference)):
        return float("nan")
    return float(pressure_integral) / (float(ttot) * float(reference))


def resolve_max_effort_values(result, ref) -> "tuple[dict, list, list, list]":
    """The maximal-effort reference numbers behind one resolved ``max_insp`` BreathRef.

    Returns ``(values, kinds, used_breaths, notices)``. ``values`` holds, for each of
    ``poes_max_ref``/``pdi_max_ref``/``rms_max_ref`` and each per-channel
    ``rms_max_ref_col_<label>`` that at least one usable breath carries, the LARGEST
    finite value among the usable breaths (the best of repeated maximal efforts is the
    reference). A breath is usable when its Manoeuvres row exists and is typed
    ``max_insp``/``sniff``. ``kinds`` is the sorted set of kinds among the usable breaths
    and ``used_breaths`` their numbers."""
    rows = []
    notices: list = []
    for b in dict.fromkeys(ref.breaths):
        row = referenceslib.lookup_manoeuvre(result, ref.file, b)
        if row is None:
            notices.append(
                f"maximal-effort reference breath {b} of {ref.file!r} could not be resolved "
                "(the source file failed, or that breath was not typed there)")
            continue
        if row.get("kind") not in _MAX_EFFORT_KINDS:
            notices.append(
                f"maximal-effort reference breath {b} of {ref.file!r} is typed "
                f"{row.get('kind')!r}, not max_insp or sniff, and is not used")
            continue
        rows.append((b, row))
    values: dict = {}
    for _b, row in rows:
        for key, v in row.items():
            if not (key in _SCALAR_KEYS or key.startswith(_CHANNEL_PREFIX)):
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and (key not in values or v > values[key]):
                values[key] = v
    kinds = sorted({row["kind"] for _b, row in rows})
    if len(kinds) > 1:
        notices.append(
            f"the maximal-effort reference of {ref.file!r} mixes {' and '.join(kinds)} "
            "breaths; sniff and maximal-inspiration pressures are different quantities, "
            "so the largest value of each quantity across both kinds is used, but check "
            "that this is intended")
    return values, kinds, [b for b, _row in rows], notices


def typed_max_effort_breaths(settings, filename) -> list:
    """Breath numbers ``filename`` itself has typed ``max_insp`` or ``sniff``, sorted."""
    # getattr: resolve_emg_reference also serves lightweight settings stand-ins with no
    # breath-type table at all, which simply have no typed breath
    return sorted(bt.breath for bt in getattr(settings.processing, "breath_types", ())
                  if bt.file == filename and bt.kind in _MAX_EFFORT_KINDS)


def emg_reference_from_max_effort(result, settings, filename) -> "dict | None":
    """The EMG reference for the EMG-normalised sheet, read from the ``max_insp``/``sniff``
    breath(s) ``filename`` has typed, as ``{rms column: reference}`` for every RMS column
    of the batch's own breath tables; ``None`` when ``filename`` has no such breath or none
    of them carries an EMG reference (the caller then falls back to the file's own
    maximum, as before).

    A column ending ``_col_<label>`` is a single channel and is normalised to THAT channel's
    peak in the maximal breath (NaN when that channel has none: never another channel's
    peak). A mean-across-channels column (``rms_mean``, ``rms_insp_mean``, ...) uses the mean
    of the channel peaks, and every other summary (``rms_max``, ...) the largest one, so a
    maximal effort reads 100 % in each. Only for ``normalization = "per_file_max"``, the
    caller's business: a mean-based mode keeps reading the file's own column means."""
    breaths = typed_max_effort_breaths(settings, filename)
    if not breaths:
        return None
    ref = BreathRef(file=filename, breaths=breaths)
    values, _kinds, _used, _notices = resolve_max_effort_values(result, ref)
    scalar = values.get("rms_max_ref")
    if scalar is None:
        return None
    channel_refs = [v for k, v in values.items() if k.startswith(_CHANNEL_PREFIX)]
    mean_ref = float(np.mean(channel_refs)) if channel_refs else scalar
    columns: list = []
    for fr in getattr(result, "ok_files", {}).values():
        table = getattr(fr, "breaths_table", None)
        if table is None:
            continue
        for c in table.columns:
            if str(c).lower().startswith("rms") and c not in columns:
                columns.append(c)
    out: dict = {}
    for c in columns:
        name = str(c)
        if "_col_" in name:
            # a channel with no reference of its own is NaN, never another channel's peak
            out[c] = values.get(_CHANNEL_PREFIX + name.rsplit("_col_", 1)[1], float("nan"))
        elif "_mean" in name:
            out[c] = mean_ref            # a mean across channels against the mean channel peak
        else:
            out[c] = scalar
    return out


def _baseline_mean(channel, n_bw: int) -> float:
    return float(np.mean(np.asarray(channel, dtype=float).reshape(-1)[:n_bw]))


def _breath_row(breath_no, row, breath, caps, ref: dict, has_emg: bool) -> dict:
    """One tidal breath's normalised values (see the module docstring)."""
    out: dict = {"breath_no": breath_no}
    ti, ttot = float(row["ti"]), float(row["ttot"])
    insp = breath["inspiration"]
    # the analysis rate the breath was actually cut at (a resampled run differs from the
    # file's own), recovered exactly from how compute set ti = len(insp flow) / fs
    fs = round(len(np.atleast_1d(insp["flow"])) / ti, 6) if ti > 0 else float("nan")

    def _n_bw(window_s):
        return int(max(1, round(window_s * fs))) if np.isfinite(fs) else 1

    window_s = ref["ptp_baseline_window_s"]
    n_bw = _n_bw(window_s)
    if caps.poes:
        out["poes_max_ref"] = ref.get("poes_max_ref", float("nan"))
        swing = _baseline_mean(insp["poes"], n_bw) - float(row["poes_mininsp"])
        mean_p = float(row["int_oesinsp"]) / ti if ti > 0 else float("nan")
        out["poes_insp_swing"] = swing
        out["poes_insp_swing_pct"] = _pct(swing, out["poes_max_ref"])
        out["poes_mean_insp"] = mean_p
        out["poes_mean_insp_pct"] = _pct(mean_p, out["poes_max_ref"])
        out["tt_es"] = tension_time(row["int_oesinsp"], ttot, out["poes_max_ref"])
    if caps.pdi:
        out["pdi_max_ref"] = ref.get("pdi_max_ref", float("nan"))
        swing = float(row["pdi_maxinsp"]) - _baseline_mean(insp["pdi"], n_bw)
        mean_p = float(row["int_pdiinsp"]) / ti if ti > 0 else float("nan")
        out["pdi_insp_swing"] = swing
        out["pdi_insp_swing_pct"] = _pct(swing, out["pdi_max_ref"])
        out["pdi_mean_insp"] = mean_p
        out["pdi_mean_insp_pct"] = _pct(mean_p, out["pdi_max_ref"])
        out["tt_di"] = tension_time(row["int_pdiinsp"], ttot, out["pdi_max_ref"])
    if has_emg:
        out["rms_max_ref"] = ref.get("rms_max_ref", float("nan"))
        pct = _pct(row["rms_insp_max"], out["rms_max_ref"])
        out["rms_insp_max_pct"] = pct
        out["nrdi"] = pct * float(row["bf"])
    return out


def _needed_columns(caps, has_emg: bool) -> list:
    need = ["ti", "ttot"]
    if caps.poes:
        need += ["poes_mininsp", "int_oesinsp"]
    if caps.pdi:
        need += ["pdi_maxinsp", "int_pdiinsp"]
    if has_emg:
        need += ["rms_insp_max", "bf"]
    return need


def attach(result, settings) -> None:
    """Post-loop pass, run after ``core.analysis.lungvol.attach``: for every OK tidal file,
    resolve its maximal-effort reference and store the "Pressure normalised" table on
    ``FileResult.pressure_normalised`` (``None`` when nothing can be normalised: no
    Poes, Pdi or EMG summary). Does nothing at all unless
    ``processing.pressure.normalization.enabled``.

    The decision is recorded in ``result.analysis_plan['pressure_normalisation']``
    (``enabled``, ``resolved``: file to the reference behind it, ``unresolved``: files
    that got NaN), the one place the writers read it back from. A file that is not a tidal
    file, or that failed, is skipped; an unexpected error in one file becomes a notice on
    that file and never aborts the batch."""
    cfg = settings.processing.pressure.normalization
    plan: dict = {"enabled": bool(cfg.enabled), "resolved": {}, "unresolved": []}
    result.analysis_plan["pressure_normalisation"] = plan
    if not cfg.enabled:
        return
    for filename, fr in list(result.files.items()):
        if fr.error is not None or getattr(fr, "role", "tidal") != "tidal":
            continue
        if fr.breaths_table is None or len(fr.breaths_table) == 0:
            continue
        try:
            _attach_one_file(result, settings, filename, fr, plan)
        except Exception as e:                                    # pragma: no cover - defensive
            fr.notices.append(
                f"{filename}: pressure normalisation could not be computed "
                f"({type(e).__name__}: {e})")


def _attach_one_file(result, settings, filename, fr, plan) -> None:
    caps = Capabilities.from_settings(settings)
    table = fr.breaths_table
    has_emg = bool(caps.emg) and "rms_insp_max" in table.columns
    if not (caps.poes or caps.pdi or has_emg):
        return
    missing = [c for c in _needed_columns(caps, has_emg) if c not in table.columns]
    if missing:
        plan.setdefault("skipped", []).append(filename)
        fr.notices.append(
            f"{filename}: pressure normalisation skipped, the breath table has no "
            f"{', '.join(missing)}")
        return

    ref_entry = referenceslib.resolve_reference(os.path.basename(filename), "max_insp", settings)
    values: dict = {}
    if ref_entry is None:
        fr.notices.append(
            f"{filename}: pressure normalisation is on but no maximal-effort reference "
            "resolves for this file (a max_insp or sniff breath named in processing."
            "references or reference_defaults, or typed in the file itself); the "
            "normalised columns are NaN")
        plan["unresolved"].append(filename)
    else:
        values, kinds, used, link_notices = resolve_max_effort_values(result, ref_entry)
        for msg in link_notices:
            fr.notices.append(f"{filename}: {msg}")
        if not used:
            fr.notices.append(
                f"{filename}: the maximal-effort reference {ref_entry.file!r} breaths "
                f"{list(ref_entry.breaths)} resolved to no usable breath; the normalised "
                "columns are NaN")
            plan["unresolved"].append(filename)
            values = {}
        else:
            for key, wanted, label in (("poes_max_ref", caps.poes, "oesophageal pressure"),
                                       ("pdi_max_ref", caps.pdi, "transdiaphragmatic pressure"),
                                       ("rms_max_ref", has_emg, "EMG")):
                if wanted and not _finite_positive(values.get(key)):
                    fr.notices.append(
                        f"{filename}: the maximal-effort reference has no positive {label} "
                        f"value ({key}); the columns built on it are NaN")
            used_entry = {"source": ref_entry.file, "breaths": used, "n": len(used),
                          "kinds": kinds}
            for key in _SCALAR_KEYS:
                if key in values:
                    used_entry[key] = values[key]
            fr.references_used["max_insp"] = used_entry
            plan["resolved"][filename] = dict(used_entry)

    ref = dict(values)
    ref["ptp_baseline_window_s"] = float(settings.processing.ptp.baseline_window_s)
    rows = []
    for _i, row in table.iterrows():
        no = int(row["breath_no"])
        breath = (fr.breaths or {}).get(no)
        if breath is None:
            continue
        rows.append(_breath_row(no, row, breath, caps, ref, has_emg))
    fr.pressure_normalised = pd.DataFrame(rows)
