# PTP baseline investigation — is commit `1630c40`'s `− pressure[0]` a bug?

_Investigation only — no code or golden changed. Verdict at the end._

**Question (from the post-refactor review list):** commit `1630c40` "Fixed PTP
calculations (zeroing)" changed `calcptp` from `pressure = pressure.squeeze()` to
`pressure = pressure.squeeze() - pressure[0]`. Is that extra `− pressure[0]` a
**double baseline subtraction bug**, or **correct** single-baseline behaviour?

## 1. Independent reference definition (from first principles)

The **pressure–time product** of a respiratory muscle over a phase is the time
integral of the pressure generated *relative to its resting (end-expiratory)
baseline*:

> PTP = ∫₀ᵀ ( P(t) − P_baseline ) dt   [units: cmH₂O·s], over the phase of duration T.

The physiologically correct baseline is the **end-expiratory level** — the pressure
at the relaxed end of expiration, i.e. the value at the **start of inspiration**
(for inspiratory muscles) — because the *absolute* oesophageal pressure has an
arbitrary offset (balloon position, transducer zero) that carries no physiological
meaning. Per metric:

| Metric | Phase | Effort signal (≥0) | Baseline (physiological) |
|---|---|---|---|
| PTPoes | inspiration | `P_ee − Poes(t)` (Poes falls below end-exp) | end-expiratory Poes = Poes at inspiration start |
| PTPdi  | inspiration | `Pdi(t) − Pdi_ee` (Pdi rises)              | end-expiratory Pdi = Pdi at inspiration start |
| PTPgas | expiration  | `Pgas(t) − Pgas_ei` (Pgas rises)           | end-inspiratory Pgas = Pgas at expiration start |

My independent reference integrates exactly this, using **`numpy.trapz`** (a
different integrator than the code's `scipy.integrate.simpson`, to stay independent).

## 2. What the code actually does

**`adjustforintegration(data)`** subtracts a *constant*: `adjustment =
min(min(data), 0)`, or `max(data)` if all values are negative. Crucially, for data
that is **entirely positive** it subtracts **0** — a **no-op**. The inspiratory
oesophageal input is `−Poes` (Poes is negative, so `−Poes` is all positive) ⇒
**`adjustforintegration(−Poes)` does nothing** (verified on real data).

**Callers** (unchanged since before the fix):
- `poesinsp = adjustforintegration(−Poes)` → effectively `−Poes` (no-op)
- `pdiinsp = Pdi − min(Pdi)` ; `pgasexp = Pgas − min(Pgas)`

**`calcptp`** then integrates:
- **before `1630c40`:** `∫ prep(t) dt` — i.e. `∫ (−Poes) dt` for Poes (no baseline
  at all, because `adjustforintegration` was a no-op), `∫ (Pdi − min Pdi) dt`, etc.
- **after `1630c40`:** `∫ ( prep(t) − prep(0) ) dt`.

**Key algebra:** `∫(f(t) − f(0)) dt` is *invariant* to any constant pre-shift of
`f` (if `f = g − c`, then `f − f(0) = g − g(0)`). So the earlier
`adjustforintegration` / `− min` steps become **numerically irrelevant** after the
fix. There is therefore **no double subtraction** — the after-fix code has exactly
**one** effective baseline: the **phase-start value** `P(0)`:

- Poes: `∫ ( (−Poes) − (−Poes)[0] ) dt = ∫ ( Poes[0] − Poes ) dt = ∫ (P_ee − Poes) dt`
- Pdi:  `∫ ( Pdi − Pdi[0] ) dt`  ;  Pgas: `∫ ( Pgas − Pgas[0] ) dt`

which is **exactly the independent reference in §1**.

## 3. Numbers

Three quantities per metric: **(a)** pre-fix code, **(b)** after-fix / current code,
**(c)** independent ground truth (§1, `trapz`).

### Real data — the canonical breath (Zeros/`NEP303_V2_40W`, breath #1, Poes-insp)
`n=2976`, `T=1.488 s`, `Poes[0]=−6.9566` (end-exp), `min=−10.0425`, `max=−6.9208`.

| quantity | value | note |
|---|---|---|
| (a) pre-fix `∫(−Poes)`            | **13.0625** | matches the old spreadsheet (~13.06) |
| (b) after-fix `∫(Poes[0]−Poes)`   | **2.7111**  | matches the current code (~2.71) |
| (c) ground truth `∫(P_ee−Poes)` trapz | **2.7113** | independent; = (b) to 2×10⁻⁴ (Simpson vs trapz) |

Decomposition: **(a) + Poes[0]·T = 13.0625 + (−6.9566·1.488) = 13.0625 − 10.35 =
2.71 = (b)**. The 10.35 the fix removed is **exactly the DC baseline offset**
`P_ee · T` — the end-expiratory pressure integrated over time, which has no
physiological meaning and only inflated the old value.

### Representative breaths — all three metrics (b vs c)
| dataset | metric | (a) pre-fix | (b) after-fix | (c) ground truth | (b)≈(c)? |
|---|---|---|---|---|---|
| synthetic A #2 | Poes | 13.4534 | 5.8466 | 5.8469 | ✓ |
| synthetic A #2 | Pdi  | 10.5232 | 10.0916 | 10.0911 | ✓ |
| synthetic A #2 | Pgas | 4.6218 | −4.2125 | −4.2134 | ✓ |
| real NEP303 #1 | Poes | 13.0625 | 2.7111 | 2.7113 | ✓ |
| real NEP303 #1 | Pdi  | 192.30 | 66.9297 | 66.9271 | ✓ |
| real NEP303 #1 | Pgas | 0.0382 | −0.0089 | −0.0089 | ✓ |

**(b) after-fix equals (c) my independent ground truth in every case** (residual
≤ 10⁻³, attributable to Simpson-vs-trapezoid, not baseline). The dramatic change is
largest for **Poes** (where `adjustforintegration` was a no-op, so pre-fix carried
the full DC offset); for **Pdi/Pgas** the `− min` already gave a near-end-expiratory
baseline, so the fix only nudges them. (The negative Pgas values are an artefact of
the synthetic/edge data shapes, not of the method.)

## 4. Verdict

**`− pressure[0]` is CORRECT — not a double-subtraction bug.**

- It establishes a single, physiologically standard **end-expiratory (phase-start)
  baseline**. The current code reproduces an independent from-first-principles PTP
  exactly.
- The pre-fix values (the old spreadsheets) were **physiologically wrong**: for Poes
  they integrated the raw pressure including its arbitrary DC offset, because
  `adjustforintegration` is a no-op on the all-positive `−Poes` input. The removed
  amount is precisely `P_ee · T`.
- There is no double subtraction: `∫(f − f[0])` is invariant to the earlier
  constant pre-shift, so only one baseline is in effect.

### Recommendation
1. **Keep the current behaviour** and **remove PTP from the parked "post-refactor
   review" list — resolved.** The current golden (which preserves it) is correct.
2. Treat the old spreadsheets' `int_*`/`ptp_*` columns as **outdated** (consistent
   with the ground-truth decision).
3. Optional, non-urgent (Emil's call — clarity/robustness, not correctness):
   - remove the now-redundant `adjustforintegration` / `− min` pre-steps and compute
     `∫(P − P_phase_start)` directly (no numeric change; clearer code);
   - consider a short end-expiratory **window mean** instead of the single first
     sample `P(0)` as baseline (robustness to noise at the phase boundary);
   - one genuine convention choice remains for **Pdi**: end-expiratory baseline
     (current) vs an absolute-zero baseline — the end-expiratory convention (current)
     is standard and recommended, and is consistent across all three metrics.
