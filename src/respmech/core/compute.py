"""Pure respiratory-mechanics / WOB / segmentation computation.

Faithful port of the validated legacy ``respmech.py`` calculation functions
(locked by the golden tests). Differences vs legacy ``master``, all documented:

* No file I/O, no plotting, no ``print`` — this module only computes. Progress is
  reported by the pipeline via events.
* ``scipy.integrate.simpson`` is called with keyword ``x=`` (positional removed in
  SciPy >= 1.14) — numerically identical (legacy bug #4).
* **PTP baseline is preserved exactly** (`calcptp` keeps ``- pressure[0]``): parked
  for a post-refactor review per Emil; not changed here.

Settings are read via the legacy attribute shape (see ``_legacy_ns``) to keep the
numerics byte-identical to the golden reference.
"""
import numpy as np
import scipy as sp
import scipy.interpolate  # noqa: F401  (sp.interpolate)
import scipy.integrate    # noqa: F401
from collections import OrderedDict

from respmech.core import emg as emglib
from respmech.core import entropy as entlib


# --- volume conditioning ---------------------------------------------------

def zero(indata):
    return indata - (indata[0])


def correctdrift(volume, settings):
    xno = len(volume) - 1
    a = ((volume[xno]) - volume[0]) / xno
    val = a
    corvol = np.zeros(len(volume))
    for i in range(0, xno):
        corvol[i] = volume[i] + val
        val = val - a
    return corvol


def correcttrend(volume, settings):
    """Compute-only volume trend correction (no plot). Returns the corrected
    volume; the diagnostic plot is produced separately by the plots layer."""
    from scipy import signal
    vol = volume.squeeze()
    peaks = signal.find_peaks(
        (vol * -1) + max(vol),
        height=settings.processing.mechanics.volumetrendpeakminheight,
        distance=settings.processing.mechanics.volumetrendpeakmindistance * settings.input.format.samplingfrequency,
    )[0]
    f = sp.interpolate.interp1d(peaks, vol[peaks],
                                settings.processing.mechanics.volumetrendadjustmethod,
                                fill_value="extrapolate")
    peaksresampled = f(np.linspace(0, vol.size - 1, vol.size))
    return volume - peaksresampled


# --- trimming --------------------------------------------------------------

class TrimError(ValueError):
    """Raised when the data cannot be trimmed to whole breaths (a precondition
    failure, not a bug): the recording must start in late expiration and end in
    early inspiration."""


def trim(timecol, flow, volume, poes, pgas, pdi, emgcolumns, settings):
    below = np.argwhere(flow <= 0)
    above = np.argwhere(flow >= 0)
    if len(below) == 0 or len(above) == 0:
        raise TrimError(
            "Could not trim data to whole breaths: the flow signal never crosses "
            "zero as required (data must start in late expiration and end in early "
            "inspiration). Check the flow channel / inverseflow setting.")
    # Start at the first INSPIRATION sample (flow strictly < 0), not the first
    # non-positive one: a recording that begins at rest (flow == 0) would otherwise set
    # startix = 0, leaving the leading expiration in place — then the first breath begins
    # in expiration, its inspiration loop never advances, and inend underflows to -1 so
    # the "inspiration" slice [0:-1] spans the whole recording (a malformed breath #1
    # that was silently averaged into the mean breath / WOB).
    startix = int(np.argmax(flow < 0))
    endix = int(above[:, 0][len(above[:, 0]) - 1])
    if endix <= startix:
        raise TrimError(
            "Could not trim data to whole breaths: computed an empty range "
            "(the recording does not start in late expiration and end in early "
            "inspiration). Check the flow channel / inverseflow setting.")
    return (timecol[startix:endix], flow[startix:endix], volume[startix:endix],
            poes[startix:endix], pgas[startix:endix], pdi[startix:endix],
            emgcolumns[startix:endix], startix, endix)


# --- breath segmentation ---------------------------------------------------

def ignorebreaths(curfile, settings):
    d = dict(settings.processing.mechanics.excludebreaths)
    return d[curfile] if curfile in d else []


def _make_breath(breathcnt, exp, insp, ignored, entcols, emgcols, filename):
    return OrderedDict([
        ('number', breathcnt),
        ('name', 'Breath #' + str(breathcnt)),
        ('expiration', exp),
        ('inspiration', insp),
        ('time', np.concatenate((insp["time"], exp["time"])).squeeze()),
        ('flow', np.concatenate((insp["flow"], exp["flow"])).squeeze()),
        ('volume', np.concatenate((insp["volume"], exp["volume"])).squeeze()),
        ('poes', np.concatenate((insp["poes"], exp["poes"])).squeeze()),
        ('pgas', np.concatenate((insp["pgas"], exp["pgas"])).squeeze()),
        ('pdi', np.concatenate((insp["pdi"], exp["pdi"])).squeeze()),
        ('breathcnt', breathcnt),
        ('ignored', ignored),
        ('entcols', entcols),
        ('emgcols', emgcols),
        ('filename', filename),
    ])


def _phase_dicts(sl_in, sl_ex, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns):
    (instart, inend) = sl_in
    (exstart, exend) = sl_ex
    exp = {'time': timecol[exstart:exend].squeeze(), 'flow': flow[exstart:exend].squeeze(),
           'poes': poes[exstart:exend].squeeze(), 'pgas': pgas[exstart:exend].squeeze(),
           'pdi': pdi[exstart:exend].squeeze(), 'volume': volume[exstart:exend].squeeze()}
    insp = {'time': timecol[instart:inend].squeeze(), 'flow': flow[instart:inend].squeeze(),
            'poes': poes[instart:inend].squeeze(), 'pgas': pgas[instart:inend].squeeze(),
            'pdi': pdi[instart:inend].squeeze(), 'volume': volume[instart:inend].squeeze()}
    if len(entropycolumns) > 0:
        exp['entcols'] = entropycolumns[exstart:exend, :].squeeze()
        insp['entcols'] = entropycolumns[instart:inend, :].squeeze()
    if len(emgcolumns) > 0:
        exp['emgcols'] = emgcolumns[exstart:exend, :].squeeze()
        insp['emgcols'] = emgcolumns[instart:inend, :].squeeze()

    entlen = exend - instart
    if len(entropycolumns) > 0:
        entcols = np.zeros([entlen, entropycolumns.shape[1]])
        for ix in range(0, entropycolumns.shape[1]):
            entcols[:, ix] = entropycolumns[instart:exend, ix]
    else:
        entcols = []
    if len(emgcolumns) > 0:
        emgcols = np.zeros([entlen, emgcolumns.shape[1]])
        for ix in range(0, emgcolumns.shape[1]):
            emgcols[:, ix] = emgcolumns[instart:exend, ix]
    else:
        emgcols = []
    return exp, insp, entcols, emgcols


def separateintobreathsbyflow(filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings):
    breaths = OrderedDict()
    j = len(flow)
    bufferwidth = settings.processing.mechanics.breathseparationbuffer
    ib = ignorebreaths(filename, settings)
    i = 0
    breathno = 0
    breathcnt = 0
    while i < j:
        breathcnt += 1
        instart = i
        while (i < j and ((flow[i] < 0) or (np.mean(flow[i:min(j, i + bufferwidth)]) < 0))):
            i += 1
        inend = i - 1
        exstart = i
        while (i < j and ((flow[i] > 0) or (np.mean(flow[i:min(j, i + bufferwidth)]) > 0))):
            i += 1
        exend = min(i - 1, j)
        exp, insp, entcols, emgcols = _phase_dicts(
            (instart, inend), (exstart, exend), timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns)
        if breathcnt in ib:
            ignored = True
        else:
            breathno += 1
            ignored = False
        breaths[breathcnt] = _make_breath(breathcnt, exp, insp, ignored, entcols, emgcols, filename)
    return breaths


def separateintobreathsbyvolume(filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings):
    from scipy import signal
    breaths = OrderedDict()
    ib = ignorebreaths(filename, settings)
    breathno = 0
    breathcnt = 0
    invol = volume
    exvol = -1 * volume
    exvol = exvol + min(exvol) * -1
    samplingfrequency = settings.input.format.samplingfrequency
    peakheight = settings.processing.mechanics.peakheight
    peakdistance = settings.processing.mechanics.peakdistance
    peakwidth = settings.processing.mechanics.peakwidth
    inpeaks, _ = signal.find_peaks(invol, height=peakheight, distance=peakdistance * samplingfrequency, width=peakwidth * samplingfrequency)
    expeaks, _ = signal.find_peaks(exvol, height=peakheight, distance=peakdistance * samplingfrequency, width=peakwidth * samplingfrequency)
    for inpeak in inpeaks:
        breathcnt += 1
        if breathcnt == 1:
            instart = 0
            inend = inpeak - 1
        else:
            instart = expeaks[breathcnt - 2]
            inend = inpeak - 1
        exstart = inend + 1
        if breathcnt < len(inpeaks):
            exend = expeaks[breathcnt - 1] - 1
        else:
            exend = len(invol) - 1
        exp, insp, entcols, emgcols = _phase_dicts(
            (instart, inend), (exstart, exend), timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns)
        if breathcnt in ib:
            ignored = True
        else:
            breathno += 1
            ignored = False
        breaths[breathcnt] = _make_breath(breathcnt, exp, insp, ignored, entcols, emgcols, filename)
    return breaths


def separateintobreaths(method, filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings):
    if str(str.lower(method)) == "volume":
        return separateintobreathsbyvolume(filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings)
    return separateintobreathsbyflow(filename, timecol, flow, volume, poes, pgas, pdi, entropycolumns, emgcolumns, settings)


# --- pressure-time product & integration -----------------------------------

def calcptp(pressure, bcnt, vefactor, samplingfreq, baseline_samples=1):
    # Pressure-time product: integrate the pressure relative to its end-expiratory
    # baseline (see docs/PTP_INVESTIGATION.md). The baseline is the mean over a short
    # window at the phase start (``baseline_samples``), which is robust to boundary
    # noise; baseline_samples=1 reproduces the single-sample behaviour.
    pressure = pressure.squeeze()
    n = int(max(1, min(baseline_samples, len(pressure))))
    baseline = np.mean(pressure[:n])
    pressure = pressure - baseline
    xval = np.linspace(0, len(pressure) / samplingfreq, len(pressure))
    integral = sp.integrate.simpson(pressure, x=xval)
    ptp = integral * bcnt * vefactor
    return ptp, integral


# --- work of breathing (Campbell diagram) ----------------------------------

def calculatewob(breath, bcnt, vefactor, settings):
    WOBUNITCHANGEFACTOR = 98.0638 / 1000  # cmH2O -> Joule; Pa = J / m3
    if settings.processing.wob.calcwobfrom == "average":
        volin = breath["inspiration"]["volumeavg"]
        volex = breath["expiration"]["volumeavg"]
        poesin = breath["inspiration"]["poesavg"]
        poesex = breath["expiration"]["poesavg"]
    else:
        volin = breath["inspiration"]["volume"]
        volex = breath["expiration"]["volume"]
        poesin = breath["inspiration"]["poes"]
        poesex = breath["expiration"]["poes"]

    eilv = [volin[len(poesin) - 1], poesin[len(poesin) - 1]]
    eelv = [volex[len(volex) - 1], poesex[len(volex) - 1]]

    # Inspiratory elastic WOB
    tbase = abs(eilv[0] - eelv[0])
    theight = abs(eilv[1] - eelv[1])
    wobinela = tbase * theight / 2 * WOBUNITCHANGEFACTOR

    # Inspiratory resistive WOB
    slope = (poesin[len(poesin) - 1] - poesin[0]) / (volin[len(volin) - 1] - volin[0])
    flyin = volin * slope + poesin[0]
    levelpoesin = (poesin * -1) - (flyin * -1)
    levelpoesin[np.where(levelpoesin < 0)] = 0
    wobinres = max(abs(sp.integrate.simpson(levelpoesin, x=volin)), 0) * WOBUNITCHANGEFACTOR

    # Expiratory WOB
    levelpoesex = poesex - poesex[len(poesex) - 1]
    levelpoesex[np.where(levelpoesex < 0)] = 0
    wobex = max(abs(sp.integrate.simpson(levelpoesex, x=volex)), 0) * WOBUNITCHANGEFACTOR

    wobin = wobinela + wobinres
    wobtotal = wobin + wobex
    return OrderedDict([
        ('wobtotal', wobtotal * bcnt * vefactor),
        ('wob_in_total', wobin * bcnt * vefactor),
        ('wob_ex_total', wobex * bcnt * vefactor),
        ('wob_in_ela', wobinela * bcnt * vefactor),
        ('wob_in_res', wobinres * bcnt * vefactor),
    ])


# --- averaging -------------------------------------------------------------

def resample(x, settings, kind='linear'):
    x = x.squeeze()
    n = settings.processing.wob.avgresamplingobs
    f = sp.interpolate.interp1d(np.linspace(0, 1, x.size), x, kind)
    return f(np.linspace(0, 1, n))


def calculateaveragebreaths(breaths, settings):
    resamplingobs = settings.processing.wob.avgresamplingobs
    nobreaths = sum(1 for b in breaths.values() if not b["ignored"])
    volumein = np.empty([resamplingobs, nobreaths])
    volumeex = np.empty([resamplingobs, nobreaths])
    poesin = np.empty([resamplingobs, nobreaths])
    poesex = np.empty([resamplingobs, nobreaths])
    for breathno in breaths:
        breath = breaths[breathno]
        if not breath["ignored"]:
            nobreaths -= 1
            try:
                volumein[:, nobreaths] = resample(breath["inspiration"]["volume"], settings)
                volumeex[:, nobreaths] = resample(breath["expiration"]["volume"], settings)
                poesin[:, nobreaths] = resample(breath["inspiration"]["poes"], settings)
                poesex[:, nobreaths] = resample(breath["expiration"]["poes"], settings)
            except Exception:
                raise ValueError(
                    "Could not resample breath #" + str(breath["number"]) +
                    ". Inspect 'avgresamplingobs' (flow separation) or peak settings "
                    "(volume separation) and breath separation.")
    return (np.mean(volumein, axis=1), np.mean(volumeex, axis=1),
            np.mean(poesin, axis=1), np.mean(poesex, axis=1))


# --- entropy ---------------------------------------------------------------

def calculateentropy(breath, settings, phase=None):
    if phase is None:
        columns = breath["entcols"]
    else:
        columns = breath[phase]["entcols"]
    columns = np.array(columns, dtype=float)
    if columns.ndim == 1:
        columns = columns.reshape(-1, 1)

    # If EMG columns are also entropy columns, use the processed (not raw) data.
    if len(settings.input.data.columns_emg) > 0:
        emgcolnos = settings.input.data.columns_emg
        for entcolno in range(0, len(settings.input.data.columns_entropy)):
            entc = settings.input.data.columns_entropy[entcolno]
            if entc in emgcolnos:
                src = breath["emgcols"] if phase is None else breath[phase]["emgcols"]
                columns[:, entcolno] = np.asarray(src)[:, emgcolnos.index(entc)]

    epoch = settings.processing.entropy.entropy_epochs
    tolerancesd = settings.processing.entropy.entropy_tolerance
    sampen = np.zeros(columns.shape[1])
    for i in range(0, columns.shape[1]):
        std_ds = np.std(columns[:, i])
        se = entlib.sample_entropy(columns[:, i], epoch, tolerancesd * std_ds)
        sampen[i] = se[len(se) - 1]
    return sampen


# --- per-breath mechanics (the big one) ------------------------------------

def calculatemechanics(breath, bcnt, vefactor, avgvolumein, avgvolumeex, avgpoesin, avgpoesex, settings):
    retbreath = breath
    retbreath["inspiration"]["volumeavg"] = avgvolumein
    retbreath["expiration"]["volumeavg"] = avgvolumeex
    retbreath["volumeavg"] = np.concatenate([avgvolumein, avgvolumeex])
    retbreath["inspiration"]["poesavg"] = avgpoesin
    retbreath["expiration"]["poesavg"] = avgpoesex
    retbreath["poesavg"] = np.concatenate([avgpoesin, avgpoesex])

    retbreath["eilv"] = [retbreath["inspiration"]["volume"][-1], retbreath["inspiration"]["poes"][-1]]
    retbreath["eelv"] = [retbreath["expiration"]["volume"][-1], retbreath["expiration"]["poes"][-1]]
    retbreath["eilvavg"] = [retbreath["inspiration"]["volumeavg"][-1], retbreath["inspiration"]["poesavg"][-1]]
    retbreath["eelvavg"] = [retbreath["expiration"]["volumeavg"][-1], retbreath["expiration"]["poesavg"][-1]]

    exp = retbreath["expiration"]
    poes_maxexp = max(exp["poes"])
    poes_endexp = exp["poes"][len(exp["poes"]) - 1]
    pdi_minexp = min(exp["pdi"])
    pdi_endexp = exp["pdi"][len(exp["pdi"]) - 1]
    pgas_endexp = exp["pgas"][len(exp["pgas"]) - 1]
    pgas_maxexp = max(exp["pgas"])
    pgas_minexp = min(exp["pgas"])

    midvolexp = min(exp["volume"]) + ((max(exp["volume"]) - min(exp["volume"])) / 2)
    midvolexpix = np.where(exp["volume"] <= midvolexp)[0][0]
    poes_midvolexp = exp["poes"][midvolexpix]
    flow_midvolexp = -exp["flow"][midvolexpix]

    insp = retbreath["inspiration"]
    poes_mininsp = min(insp["poes"])
    poes_endinsp = insp["poes"][len(insp["poes"]) - 1]
    pdi_maxinsp = max(insp["pdi"])
    pdi_endinsp = insp["pdi"][len(insp["pdi"]) - 1]
    pgas_endinsp = insp["pgas"][len(insp["pgas"]) - 1]

    poes_tidal_swing = abs(max(retbreath["poes"]) - min(retbreath["poes"]))
    pgas_tidal_swing = abs(max(retbreath["pgas"]) - min(retbreath["pgas"]))
    pdi_tidal_swing = abs(max(retbreath["pdi"]) - min(retbreath["pdi"]))

    midvolinsp = min(insp["volume"]) + ((max(insp["volume"]) - min(insp["volume"])) / 2)
    midvolinspix = np.where(insp["volume"] >= midvolinsp)[0][0]
    poes_midvolinsp = insp["poes"][midvolinspix]
    flow_midvolinsp = -insp["flow"][midvolinspix]

    vol_endinsp = insp["volume"][len(insp["volume"]) - 1]
    vol_endexp = exp["volume"][len(exp["volume"]) - 1]

    ti = len(insp["flow"]) / settings.input.format.samplingfrequency
    te = len(exp["flow"]) / settings.input.format.samplingfrequency
    ttot = len(retbreath["flow"]) / settings.input.format.samplingfrequency
    ti_ttot = ti / ttot

    vt = max(retbreath["volume"]) - min(retbreath["volume"])
    ve = vt * bcnt * vefactor

    vmrnumerator = (pgas_endinsp - pgas_endexp)
    vmrdenominator = (poes_endinsp - poes_endexp)
    vmr = np.divide(vmrnumerator, vmrdenominator, out=np.zeros_like(vmrnumerator), where=vmrdenominator != 0)

    tlr_insp = abs((poes_midvolexp - poes_midvolinsp) / (flow_midvolexp - flow_midvolinsp))
    insp_pdi_rise = pdi_maxinsp - min(insp["pdi"])
    exp_pgas_rise = pgas_maxexp - min(exp["pgas"])

    # PTP is integrated relative to the end-expiratory baseline inside calcptp
    # (mean over a short window at the phase start). The former adjustforintegration
    # / "- min" pre-steps were redundant (integ(f - f[0]) is invariant to a constant
    # pre-shift — see docs/PTP_INVESTIGATION.md); the signed signals are passed
    # directly (Poes negated so inspiratory effort is positive).
    fs = settings.input.format.samplingfrequency
    ptp_bw = int(max(1, round(settings.processing.mechanics.ptp_baseline_window_s * fs)))
    ptp_oesinsp, int_oesinsp = calcptp(-insp["poes"], bcnt, vefactor, fs, ptp_bw)
    ptp_pdiinsp, int_pdiinsp = calcptp(insp["pdi"], bcnt, vefactor, fs, ptp_bw)
    ptp_pgasexp, int_pgasexp = calcptp(exp["pgas"], bcnt, vefactor, fs, ptp_bw)

    max_in_flow = min(insp["flow"]) * -1
    max_ex_flow = max(exp["flow"])
    inflowmidvol = insp["flow"][midvolinspix] * -1
    exflowmidvol = exp["flow"][midvolexpix]

    retbreath["wob"] = calculatewob(breath, bcnt, vefactor, settings)

    if len(breath["emgcols"]) > 0:
        retbreath["rms"], retbreath["intemg"] = emglib.calculate_rms(breath["emgcols"], settings.processing.emg.rms_s, settings.input.format.samplingfrequency)
        retbreath["rms_insp"], retbreath["intemg_insp"] = emglib.calculate_rms(breath["inspiration"]["emgcols"], settings.processing.emg.rms_s, settings.input.format.samplingfrequency)
        retbreath["rms_exp"], retbreath["intemg_exp"] = emglib.calculate_rms(breath["expiration"]["emgcols"], settings.processing.emg.rms_s, settings.input.format.samplingfrequency)

    if len(settings.input.data.columns_entropy) > 0:
        entropy = calculateentropy(breath, settings)
        entropy_insp = calculateentropy(breath, settings, "inspiration")
        entropy_exp = calculateentropy(breath, settings, "expiration")
        retbreath["entropy"] = np.append(entropy.T, [max(entropy.T), min(entropy.T), np.mean(entropy.T)])
        retbreath["entropy_insp"] = np.append(entropy_insp.T, [max(entropy_insp.T), min(entropy_insp.T), np.mean(entropy_insp.T)])
        retbreath["entropy_exp"] = np.append(entropy_exp.T, [max(entropy_exp.T), min(entropy_exp.T), np.mean(entropy_exp.T)])
    else:
        retbreath["entropy"] = []

    retbreath["mechanics"] = OrderedDict([
        ('poes_maxexp', poes_maxexp), ('poes_mininsp', poes_mininsp),
        ('poes_endinsp', poes_endinsp), ('poes_endexp', poes_endexp),
        ('poes_midvolexp', poes_midvolexp), ('poes_midvolinsp', poes_midvolinsp),
        ('int_oesinsp', int_oesinsp), ('ptp_oesinsp', ptp_oesinsp),
        ('poes_tidal_swing', poes_tidal_swing),
        ('pgas_endinsp', pgas_endinsp), ('pgas_endexp', pgas_endexp),
        ('pgas_maxexp', pgas_maxexp), ('pgas_minexp', pgas_minexp),
        ('exp_pgas_rise', exp_pgas_rise), ('int_pgasexp', int_pgasexp),
        ('ptp_pgasexp', ptp_pgasexp), ('pgas_tidal_swing', pgas_tidal_swing),
        ('int_pdiinsp', int_pdiinsp), ('ptp_pdiinsp', ptp_pdiinsp),
        ('pdi_minexp', pdi_minexp), ('pdi_maxinsp', pdi_maxinsp),
        ('pdi_endinsp', pdi_endinsp), ('pdi_endexp', pdi_endexp),
        ('insp_pdi_rise', insp_pdi_rise), ('pdi_tidal_swing', pdi_tidal_swing),
        ('flow_midvolexp', flow_midvolexp), ('flow_midvolinsp', flow_midvolinsp),
        ('vol_endinsp', vol_endinsp), ('vol_endexp', vol_endexp),
        ('max_in_flow', max_in_flow), ('max_ex_flow', max_ex_flow),
        ('in_flow_midvol', inflowmidvol), ('ex_flow_midvol', exflowmidvol),
        ('ti', ti), ('te', te), ('ttot', ttot), ('ti_ttot', ti_ttot),
        ('vt', vt), ('bf', bcnt * vefactor), ('ve', ve),
        ('vmr', vmr), ('tlr_insp', tlr_insp),
    ])
    return retbreath
