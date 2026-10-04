"""Adapter: typed :class:`~respmech.core.settings.Settings` -> the attribute shape
the ported calculation functions read.

The calculation core in this package is a faithful port of the validated legacy
code (locked by the golden tests). To keep the numerics byte-identical, the ported
functions read settings via the *same* attribute paths the legacy code used
(``s.processing.mechanics.breathseparationbuffer`` …). This adapter reconstructs
that shape from the new typed model so the public API is TOML/typed while the
compute stays verbatim. It is an internal detail; later phases can inline it.
"""
from __future__ import annotations

import math
from collections import defaultdict
from types import SimpleNamespace

from respmech.core.analysis.signals import Capabilities
from respmech.core.settings import Settings


def _merged_exclude_breaths(s: Settings, flow_bearing: bool) -> list:
    """M-19: a typed breath is excluded from the tidal average exactly like a manually
    excluded one -- union the two per file into the ONE list the legacy segmenterers
    read (``compute.ignorebreaths``), so two separate entries for the same file never
    silently lose one (the last-writer-wins failure a naive "append both lists" would
    risk if a caller ever built ``excludebreaths`` from more than one source list).

    For a flow-bearing signal set every typed kind counts (an IC/FVC/etc. breath is a
    manoeuvre, never tidal breathing, regardless of which kind it is typed as). For an
    EMG-only set (M-21's ``separators`` segmentation -- not yet dispatched here, but
    ``breath_types`` is shared state the same union rule must already hold for) only
    ``rest``-typed segments are folded in: the OTHER kinds there name a manoeuvre effort
    itself (M-29 extracts its value), not a segment to discard, so excluding it here
    would silently throw the manoeuvre away before it is ever read.
    """
    merged: dict[str, set[int]] = defaultdict(set)
    for e in s.processing.exclude_breaths:
        merged[e.file].update(e.breaths)
    for t in s.processing.breath_types:
        if flow_bearing or t.kind == "rest":
            merged[t.file].add(t.breath)
    return [[f, sorted(bs)] for f, bs in merged.items()]


def _separators_passthrough(s: Settings) -> list:
    """``[[file, [t0, t1, …]], …]`` from ``processing.segmentation.separators`` --
    read back by ``compute._separator_times_for`` for the EMG-only ``separators``
    segmentation method. Unlike ``exclude_breaths``/``breath_types``, there is no
    OTHER source to merge (a separator entry only ever comes from this one list),
    so this is a plain shape transform, not a union."""
    return [[e.file, list(e.times_s)] for e in s.processing.segmentation.separators]


def _breath_types_passthrough(s: Settings) -> list:
    """v2-only passthrough (no legacy counterpart, like ``ptp_baseline_window_s``):
    every typed breath, grouped by file, as ``[breath_no, kind, t_onset_s]`` triples --
    read back by ``compute.breathkinds`` to set ``breath['kind']`` alongside
    ``ignorebreaths``' exclusion. Kept separate from the merged ``excludebreaths`` above
    (which only carries breath NUMBERS, never which kind) so a segmenter can set both
    ``ignored`` and ``kind`` from one lookup each, mirroring the existing
    ``ignorebreaths``/``excludebreaths`` split rather than inventing a third shape.
    """
    by_file: dict[str, list] = defaultdict(list)
    for t in s.processing.breath_types:
        by_file[t.file].append([t.breath, t.kind, t.t_onset_s])
    return [[f, entries] for f, entries in by_file.items()]


def to_legacy_ns(s: Settings) -> SimpleNamespace:
    ch = s.input.channels
    seg = s.processing.segmentation
    vol = s.processing.volume
    wob = s.processing.wob
    emg = s.processing.emg
    ent = s.processing.entropy
    od = s.output.data
    odg = s.output.diagnostics
    # Resolved once and reused below (the exclude-breaths union needs to know whether
    # flow is declared) -- same object `capabilities=` already passes through as-is.
    capabilities = Capabilities.from_settings(s)

    return SimpleNamespace(
        # Passed through as the dataclass, same precedent as processing.emg.robust_peak
        # below: compute reads it with getattr(settings, "capabilities", Capabilities.FULL)
        # (the boundarynotice_* idiom), because hand-built SimpleNamespace settings reach
        # the segmenterers from several call sites and need not carry this attribute at all.
        capabilities=capabilities,
        input=SimpleNamespace(
            inputfolder=s.input.folder,
            files=s.input.files,
            format=SimpleNamespace(
                samplingfrequency=s.input.format.sampling_frequency,
                matlabfileformat=1 if s.input.format.matlab_variant == "windows" else 2,
                decimalcharacter=s.input.format.decimal,
            ),
            data=SimpleNamespace(
                # legacy code uses np.isnan(column_X) to mean "absent" (loaders.py); flow
                # keeps raising via _column today regardless of None vs NaN (Settings.validate
                # still requires it), so this is forward-compat plumbing for a future change
                # that relaxes that requirement, not yet observable behaviour for flow.
                column_poes=ch.poes if ch.poes is not None else math.nan,
                column_pgas=ch.pgas if ch.pgas is not None else math.nan,
                column_pdi=ch.pdi if ch.pdi is not None else math.nan,
                column_volume=ch.volume if ch.volume is not None else math.nan,
                column_flow=ch.flow if ch.flow is not None else math.nan,
                columns_entropy=list(ch.entropy),
                entropy_derived=list(ch.entropy_derived),
                columns_emg=list(ch.emg),
            ),
        ),
        processing=SimpleNamespace(
            sampling=SimpleNamespace(
                resample=s.processing.sampling.resample,
                resampletofrequency=s.processing.sampling.resample_to_frequency,
            ),
            mechanics=SimpleNamespace(
                breathseparationbuffer=seg.buffer,
                separateby=seg.method,
                peakheight=seg.peak.height,
                peakdistance=seg.peak.distance_s,
                peakwidth=seg.peak.width_s,
                boundarynoticeminrelativeduration=seg.boundary_notice_min_relative_duration,
                boundarynoticeminotherbreaths=seg.boundary_notice_min_other_breaths,
                inverseflow=vol.inverse_flow,
                integratevolumefromflow=vol.integrate_from_flow,
                inversevolume=vol.inverse_volume,
                correctvolumedrift=vol.correct_drift,
                correctvolumetrend=vol.correct_trend,
                volumetrendadjustmethod=vol.trend_method,
                volumetrendpeakminheight=vol.trend_peak_min_height,
                volumetrendpeakmindistance=vol.trend_peak_min_distance_s,
                # v2-named passthrough (no legacy counterpart), like ptp_baseline_window_s
                trend_peak_min_prominence_frac=vol.trend_peak_min_prominence_frac,
                excludebreaths=_merged_exclude_breaths(s, capabilities.flow),
                breathcounts=[[e.file, e.count] for e in s.processing.breath_counts],
                breathtypes=_breath_types_passthrough(s),
                separators=_separators_passthrough(s),
                # v2-named passthrough: parameters of the automatic EMG-only methods
                # (fixed_windows / emg_burst), read by compute.separateintobreaths.
                emgsegmentation=SimpleNamespace(
                    window_s=seg.emg.window_s, hop_s=seg.emg.hop_s,
                    burst_threshold_frac=seg.emg.burst_threshold_frac,
                    burst_min_s=seg.emg.burst_min_s,
                    burst_smooth_s=seg.emg.burst_smooth_s,
                    burst_min_contrast=seg.emg.burst_min_contrast),
                ptp_baseline_window_s=s.processing.ptp.baseline_window_s,
            ),
            # v2-only passthrough (no legacy counterpart), like ptp_baseline_window_s
            # above — read by core.analysis.manoeuvres.extract via
            # s.processing.lung_volume.ic. Passed through as the dataclass, same
            # precedent as processing.emg.robust_peak above.
            lung_volume=SimpleNamespace(ic=s.processing.lung_volume.ic),
            # v2-only passthrough (no legacy counterpart) -- read by
            # core.analysis.mfvl.attach/tidal_mfvl_ext via s.processing.mfvl. Passed
            # through as the dataclass itself, same precedent as lung_volume.ic above.
            mfvl=s.processing.mfvl,
            # v2-only passthrough (no legacy counterpart) -- read by
            # core.analysis.pressure.attach via s.processing.pressure.peepi.
            pressure=s.processing.pressure,
            # v2-only passthrough -- read by core.analysis.breathing_pattern via
            # s.processing.breathing_pattern.
            breathing_pattern=s.processing.breathing_pattern,
            wob=SimpleNamespace(
                calcwobfrom=wob.calc_from,
                avgresamplingobs=wob.avg_resampling_obs,
            ),
            emg=SimpleNamespace(
                rms_s=emg.rms_window_s,
                remove_ecg=emg.remove_ecg,
                column_detect=emg.detect_channel,
                minheight=emg.ecg_min_height,
                mindistance=emg.ecg_min_distance_s,
                minwidth=emg.ecg_min_width_s,
                windowsize=emg.ecg_window_s,
                outlierrmssdlimit=emg.outlier_rms_sd_limit,
                save_sound=emg.save_sound,
                emgplotyscale=emg.plot_yscale,
                # passed through as the dataclass — read-only, and keeping one name avoids
                # inventing a second vocabulary for a setting that has no v1 ancestor
                robust_peak=emg.robust_peak,
            ),
            entropy=SimpleNamespace(
                entropy_epochs=ent.epochs,
                entropy_tolerance=ent.tolerance,
            ),
        ),
        output=SimpleNamespace(
            outputfolder=s.output.folder,
            data=SimpleNamespace(
                saveaveragedata=od.save_average,
                savebreathbybreathdata=od.save_breath_by_breath,
                saveprocesseddata=od.save_processed,
                includeignoredbreaths=od.include_ignored_breaths,
            ),
            diagnostics=SimpleNamespace(
                savepvaverage=odg.save_pv_average,
                savepvindividualworkload=odg.save_pv_individual,
                pvcolumns=odg.pv_columns,
                pvrows=odg.pv_rows,
                savedataviewraw=odg.save_raw,
                savedataviewtrimmed=odg.save_trimmed,
                savedataviewdriftcor=odg.save_drift,
            ),
        ),
    )
