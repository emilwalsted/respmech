"""Reference (normal) values for the forced flow-volume loop, and the shape-scaled normal
range the MFVL figure draws behind a participant's own curve.

Two separate things live here, deliberately kept apart:

* a **reference set** answers "what are the expected FVC and FEV1, and their lower and
  upper limits of normal, for this sex, age and height?". Only that is a published
  equation; the shipped set is GLI 2022 (race-neutral, Bowerman et al. 2023), which covers
  FEV1, FVC and FEV1/FVC and nothing else (no flow-volume curve, no PEF).
* a **curve shape** turns those numbers into a drawable normal flow-volume curve. GLI 2022
  has none, so the shape comes from a second source: the ECCS 1993 reference equations for
  PEF and the flows at 25, 50 and 75 % of FVC (Quanjer et al., Eur Respir J 1993;6 Suppl
  16). Only the *ratios* between those flows are used, as a typical normal shape; the size
  is then fixed by GLI 2022, so the curve's FVC is the GLI FVC and its FEV1 (the volume
  exhaled after one second) is the GLI FEV1. The curve is therefore indicative of shape,
  not itself a published reference.

The registry (:data:`REGISTRY`, :func:`get_reference`) is the seam for adding further
reference materials later. Choosing between them from a setting is NOT implemented yet: the
figure code asks for :data:`DEFAULT_REFERENCE`.

Pure numpy, no Qt, no file handles; the participant's sex, age and height come from the
optional ``sex``/``age_years``/``height_cm`` fields of ``input.subjects`` (see
``core.settings.SubjectEntry``) and without them nothing is drawn.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from respmech.core.analysis import _gli2022_tables as _t

#: z-score of the 5th / 95th percentile (the usual lower / upper limit of normal).
Z_LLN = -1.645
Z_ULN = 1.645

DEFAULT_REFERENCE = "gli_2022"

#: Sex spellings accepted from a settings file, mapped to the canonical ``"male"``/``"female"``.
_SEX_SPELLINGS = {"m": "male", "male": "male", "man": "male",
                  "f": "female", "female": "female", "woman": "female"}


def normalise_sex(value) -> str | None:
    """``"male"``/``"female"`` for the accepted spellings (case-insensitive), else ``None``."""
    if not isinstance(value, str):
        return None
    return _SEX_SPELLINGS.get(value.strip().lower())


@dataclass(frozen=True)
class SpirometryPrediction:
    """Expected value and limits of normal (litres) for FVC and FEV1, from one reference."""
    fvc: float
    fvc_lln: float
    fvc_uln: float
    fev1: float
    fev1_lln: float
    fev1_uln: float


class ReferenceSet(Protocol):
    """What the figure needs from a reference material. ``predict`` returns ``None`` outside
    the set's own validity range (never an extrapolation)."""
    key: str
    label: str          # short, shown on the figure, e.g. "GLI 2022"
    citation: str       # full, for documentation and tooltips

    def predict(self, sex: str, age_years: float, height_cm: float
                ) -> SpirometryPrediction | None: ...


# --------------------------------------------------------------------------- GLI 2022
def _lms_fvc_fev1(param: str, sex: str, age: float, height_cm: float
                  ) -> tuple[float, float, float]:
    """(L, M, S) for FEV1 or FVC, following the GLI Global convention: the spline lookup is
    done at the age rounded to the nearest quarter year, while the equation itself uses the
    exact age."""
    coef = _t.COEFFS[(param, sex)]
    ages = _t.AGES
    lookup_age = round(age * 4.0) / 4.0
    ix = bisect.bisect_left(ages, lookup_age)
    if ix >= len(ages) or abs(ages[ix] - lookup_age) > 1e-9:
        raise ValueError(f"age {age} is outside the GLI 2022 table")
    m_spl, s_spl, _l_spl = (t[ix] for t in _t.SPLINES[(param, sex)])
    m = math.exp(coef["a0"] + coef["a1"] * math.log(height_cm)
                 + coef["a2"] * math.log(age) + m_spl)
    s = math.exp(coef["p0"] + coef["p1"] * math.log(age) + s_spl)
    return coef["q0"], m, s


def _limit(l_: float, m: float, s: float, z: float) -> float:
    """Value at z-score ``z`` of an LMS distribution (Cole's transform)."""
    if abs(l_) < 1e-12:
        return m * math.exp(s * z)
    return m * (1.0 + l_ * s * z) ** (1.0 / l_)


class Gli2022:
    """GLI 2022 race-neutral spirometry (Bowerman et al., Am J Respir Crit Care Med
    2023;207(6):768-774). Valid for 3-95 years; the height is not range-checked by the
    paper's tables, only by the plausibility guard below."""
    key = "gli_2022"
    label = "GLI 2022"
    citation = ("Global Lung Function Initiative 2022, race-neutral spirometry reference "
                "equations (Bowerman et al., Am J Respir Crit Care Med 2023;207:768-774)")
    age_range = (3.0, 95.0)
    height_range_cm = (80.0, 230.0)         # plausibility guard, not a published limit

    def predict(self, sex: str, age_years: float, height_cm: float
                ) -> SpirometryPrediction | None:
        sex = normalise_sex(sex)
        if sex is None:
            return None
        try:
            age = float(age_years)
            ht = float(height_cm)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(age) and math.isfinite(ht)):
            return None
        if not (self.age_range[0] <= age <= self.age_range[1]):
            return None
        if not (self.height_range_cm[0] <= ht <= self.height_range_cm[1]):
            return None
        out = {}
        for param in ("FVC", "FEV1"):
            l_, m, s = _lms_fvc_fev1(param, sex, age, ht)
            out[param] = (m, _limit(l_, m, s, Z_LLN), _limit(l_, m, s, Z_ULN))
        return SpirometryPrediction(
            fvc=out["FVC"][0], fvc_lln=out["FVC"][1], fvc_uln=out["FVC"][2],
            fev1=out["FEV1"][0], fev1_lln=out["FEV1"][1], fev1_uln=out["FEV1"][2])


#: Registered reference materials, by key. Add a new set here (and nowhere else in the
#: figure code) when another reference is wanted; a later release can then expose the
#: choice as a setting.
REGISTRY: dict[str, ReferenceSet] = {Gli2022.key: Gli2022()}


def register(reference: ReferenceSet) -> None:
    """Add (or replace) a reference material in :data:`REGISTRY`."""
    REGISTRY[reference.key] = reference


def get_reference(key: str | None = None) -> ReferenceSet:
    """The reference registered under ``key`` (the default one when ``None``)."""
    k = DEFAULT_REFERENCE if key is None else key
    try:
        return REGISTRY[k]
    except KeyError:
        raise KeyError(f"unknown reference {k!r}; registered: {sorted(REGISTRY)}") from None


# --------------------------------------------------------------------------- curve shape
# ECCS 1993 (Quanjer et al.): linear equations a0 + a_ht * height_cm + a_age * age_years, in
# L/s. Only used for the RATIOS between the flows (typical shape of the expiratory limb),
# never for an absolute size. Per the source, for adults under 25 the flow equations use
# age 25 (men: PEF; women: all), and the equations are published for ages 18-70 and heights
# 155-195 cm (men) / 145-180 cm (women); outside that the demographics are clamped.
_ECCS = {
    ("male", "PEF"): (0.15, 0.0614, -0.043),
    ("male", "FEF25"): (-0.47, 0.0546, -0.029),
    ("male", "FEF50"): (-0.35, 0.0379, -0.031),
    ("male", "FEF75"): (-1.34, 0.0261, -0.026),
    ("female", "PEF"): (-1.11, 0.055, -0.03),
    ("female", "FEF25"): (1.6, 0.0322, -0.025),
    ("female", "FEF50"): (1.16, 0.0245, -0.025),
    ("female", "FEF75"): (1.11, 0.0105, -0.025),
}
_ECCS_AGE = (18.0, 70.0)
_ECCS_HEIGHT = {"male": (155.0, 195.0), "female": (145.0, 180.0)}
_ECCS_AGE_FLOOR_FLOWS = {"male": {"PEF", "FEF75"}, "female": {"PEF", "FEF25", "FEF50", "FEF75"}}
_ECCS_CITATION = ("ECCS/ERS 1993 reference equations for PEF and flows at 25, 50 and 75 % of "
                  "FVC (Quanjer et al., Eur Respir J 1993;6 Suppl 16), used for the curve's "
                  "shape only")

#: Volume position (fraction of FVC exhaled) at which the model puts the peak flow.
PEF_VOLUME_FRACTION = 0.10


def _eccs_flow(sex: str, name: str, age: float, height: float) -> float:
    a0, a_ht, a_age = _ECCS[(sex, name)]
    lo, hi = _ECCS_HEIGHT[sex]
    ht = min(max(height, lo), hi)
    a = min(max(age, _ECCS_AGE[0]), _ECCS_AGE[1])
    if name in _ECCS_AGE_FLOOR_FLOWS[sex]:
        a = max(a, 25.0)
    return a0 + a_ht * ht + a_age * a


def curve_shape(sex: str, age_years: float, height_cm: float) -> tuple[np.ndarray, np.ndarray]:
    """The typical expiratory flow shape as ``(x, g)``: ``x`` the fraction of FVC exhaled
    (0..1) and ``g`` the flow as a fraction of the peak flow (0..1), anchored at the ECCS
    1993 PEF and flows at 25, 50 and 75 % of FVC exhaled, zero at the start and the end.
    Between the anchors the curve is a shape-preserving (monotone) cubic; before the peak
    it follows the constant-acceleration start of a forced manoeuvre,
    ``g = sqrt(x / x_peak)``, so the time to exhale the first litre stays finite."""
    sex = normalise_sex(sex)
    if sex is None:
        raise ValueError("sex must be 'male' or 'female'")
    pef = _eccs_flow(sex, "PEF", age_years, height_cm)
    gx = [PEF_VOLUME_FRACTION, 0.25, 0.50, 0.75, 1.0]
    gy = [1.0] + [_eccs_flow(sex, n, age_years, height_cm) / pef
                  for n in ("FEF25", "FEF50", "FEF75")] + [0.0]
    # a source whose ratios stop falling would not be a plausible expiratory limb
    if not all(np.diff(gy) < 0) or gy[-2] <= 0:
        raise ValueError("ECCS shape is not monotonically falling for these inputs")
    x_tail = np.linspace(PEF_VOLUME_FRACTION, 1.0, 400)
    g_tail = _pchip(np.array(gx), np.array(gy), x_tail)
    x_head = np.linspace(0.0, PEF_VOLUME_FRACTION, 100, endpoint=False)
    g_head = np.sqrt(x_head / PEF_VOLUME_FRACTION)
    return np.concatenate([x_head, x_tail]), np.concatenate([g_head, g_tail])


def _pchip(x: np.ndarray, y: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone cubic interpolation (no scipy dependency in the core)."""
    h = np.diff(x)
    delta = np.diff(y) / h
    n = len(x)
    d = np.zeros(n)
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] > 0:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
    d[0] = _end_slope(h[0], h[1], delta[0], delta[1])
    d[-1] = _end_slope(h[-1], h[-2], delta[-1], delta[-2])
    idx = np.clip(np.searchsorted(x, xi, side="right") - 1, 0, n - 2)
    t = (xi - x[idx]) / h[idx]
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t ** 2 * (3 - 2 * t)
    h11 = t ** 2 * (t - 1)
    return h00 * y[idx] + h10 * h[idx] * d[idx] + h01 * y[idx + 1] + h11 * h[idx] * d[idx + 1]


def _end_slope(h0: float, h1: float, d0: float, d1: float) -> float:
    s = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if s * d0 <= 0:
        return 0.0
    if d0 * d1 <= 0 and abs(s) > 3 * abs(d0):
        return 3 * d0
    return s


def scaled_curve(x: np.ndarray, g: np.ndarray, fvc: float, fev1: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """The shape ``(x, g)`` sized to ``fvc`` litres and to a flow scale at which the volume
    exhaled after one second is ``fev1``. Returns ``(volume_l, flow_lps)``, volume from 0
    (TLC) to ``fvc``."""
    if not (0.0 < fev1 < fvc):
        raise ValueError("need 0 < FEV1 < FVC")
    # time to exhale volume v is  t(v) = (1/k) * integral_0^v dv' / g(v'/fvc);  FEV1 is the
    # v with t(v) = 1 s, so k = integral_0^FEV1 dv / g.  Integrate in x (= v / fvc) with the
    # singular start (g ~ sqrt(x), integrable) done analytically up to the peak.
    xf = fev1 / fvc
    xp = PEF_VOLUME_FRACTION
    head = 2.0 * xp                       # integral_0^xp dx / sqrt(x / xp)
    mask = (x >= xp) & (x <= xf)
    xs = x[mask]
    gs = g[mask]
    if xs.size == 0 or xs[-1] < xf:       # close the interval exactly at xf
        gf = float(np.interp(xf, x, g))
        xs = np.append(xs, xf)
        gs = np.append(gs, gf)
    body = float(np.sum((xs[1:] - xs[:-1]) * 0.5 * (1.0 / gs[1:] + 1.0 / gs[:-1])))
    k = fvc * (head + body)               # L/s per unit of g
    return x * fvc, g * k


# --------------------------------------------------------------------------- the band
@dataclass(frozen=True)
class NormalBand:
    """What the figure draws: the expected curve and the band between the curves for the
    lower and upper limits of normal. All volumes are litres below TLC, flows L/s."""
    v_expected: np.ndarray
    flow_expected: np.ndarray
    v_band: np.ndarray            # common grid for the band edges
    flow_lower: np.ndarray        # curve scaled to FVC/FEV1 at LLN (0 beyond its own FVC)
    flow_upper: np.ndarray        # curve scaled to FVC/FEV1 at ULN
    prediction: SpirometryPrediction
    reference_label: str
    shape_citation: str

    def note(self) -> str:
        return f"Normal range: {self.reference_label}; shape ECCS 1993 (indicative)"


def normal_band(sex, age_years, height_cm, reference: str | None = None
                ) -> NormalBand | None:
    """The normal flow-volume range for one participant, or ``None`` when the inputs are
    missing or outside the reference's validity range (nothing is then drawn)."""
    sex_n = normalise_sex(sex)
    if sex_n is None or age_years is None or height_cm is None:
        return None
    ref = get_reference(reference)
    pred = ref.predict(sex_n, age_years, height_cm)
    if pred is None:
        return None
    try:
        x, g = curve_shape(sex_n, float(age_years), float(height_cm))
        v_e, f_e = scaled_curve(x, g, pred.fvc, pred.fev1)
        v_lo, f_lo = scaled_curve(x, g, pred.fvc_lln, pred.fev1_lln)
        v_hi, f_hi = scaled_curve(x, g, pred.fvc_uln, pred.fev1_uln)
    except ValueError:
        return None
    grid = np.linspace(0.0, float(v_hi[-1]), 500)
    lower = np.interp(grid, v_lo, f_lo, right=0.0)
    upper = np.interp(grid, v_hi, f_hi, right=0.0)
    return NormalBand(v_expected=v_e, flow_expected=f_e, v_band=grid,
                      flow_lower=lower, flow_upper=upper, prediction=pred,
                      reference_label=ref.label, shape_citation=_ECCS_CITATION)


def subject_demographics(settings, filename: str) -> tuple[str, float, float] | None:
    """``(sex, age_years, height_cm)`` of the participant ``filename`` belongs to, from
    ``input.subjects``, or ``None`` when the participant or any of the three is missing."""
    from respmech.core.summary import group_key        # noqa: PLC0415
    group = group_key(filename, settings)
    for subj in settings.input.subjects:
        if subj.key != group:
            continue
        sex = normalise_sex(getattr(subj, "sex", None))
        age = getattr(subj, "age_years", None)
        ht = getattr(subj, "height_cm", None)
        if sex is None or age is None or ht is None:
            return None
        return sex, float(age), float(ht)
    return None


def normal_band_for(settings, filename: str, reference: str | None = None
                    ) -> NormalBand | None:
    """:func:`normal_band` for the participant a file belongs to; ``None`` without the
    participant's demographics. Never raises: a figure must degrade to no band."""
    try:
        demo = subject_demographics(settings, filename)
        if demo is None:
            return None
        return normal_band(*demo, reference=reference)
    except Exception:                                   # noqa: BLE001
        return None
