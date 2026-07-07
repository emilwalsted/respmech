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
from types import SimpleNamespace

from respmech.core.settings import Settings


def to_legacy_ns(s: Settings) -> SimpleNamespace:
    ch = s.input.channels
    seg = s.processing.segmentation
    vol = s.processing.volume
    wob = s.processing.wob
    emg = s.processing.emg
    ent = s.processing.entropy
    od = s.output.data
    odg = s.output.diagnostics

    return SimpleNamespace(
        input=SimpleNamespace(
            inputfolder=s.input.folder,
            files=s.input.files,
            format=SimpleNamespace(
                samplingfrequency=s.input.format.sampling_frequency,
                matlabfileformat=1 if s.input.format.matlab_variant == "windows" else 2,
                decimalcharacter=s.input.format.decimal,
            ),
            data=SimpleNamespace(
                column_poes=ch.poes,
                column_pgas=ch.pgas,
                column_pdi=ch.pdi,
                # legacy code uses np.isnan(column_volume) to mean "absent"
                column_volume=ch.volume if ch.volume is not None else math.nan,
                column_flow=ch.flow,
                columns_entropy=list(ch.entropy),
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
                inverseflow=vol.inverse_flow,
                integratevolumefromflow=vol.integrate_from_flow,
                inversevolume=vol.inverse_volume,
                correctvolumedrift=vol.correct_drift,
                correctvolumetrend=vol.correct_trend,
                volumetrendadjustmethod=vol.trend_method,
                volumetrendpeakminheight=vol.trend_peak_min_height,
                volumetrendpeakmindistance=vol.trend_peak_min_distance_s,
                excludebreaths=[[e.file, list(e.breaths)] for e in s.processing.exclude_breaths],
                breathcounts=[[e.file, e.count] for e in s.processing.breath_counts],
            ),
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
                remove_noise=emg.remove_noise,
                outlierrmssdlimit=emg.outlier_rms_sd_limit,
                noise_profile=emg.noise_profile,
                save_sound=emg.save_sound,
                emgplotyscale=emg.plot_yscale,
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
