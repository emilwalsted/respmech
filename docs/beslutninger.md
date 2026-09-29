# Decisions — settled, don't reopen without new evidence

Dated project decisions and "considered and rejected" points, newest first. Each
entry: date, decision, reasoning, source (a conversation, a review, or "author's
decision <date>" — never an internal ticket reference; this repo is public).

---

**29-09-2026 — The modular analysis, in one place: seven choices that are easy to undo
by accident (author's decisions, 26-09-2026).** The individual features each have their
own entry here or a section in `docs/REVERSE_ENGINEERING.md` (§5.11-5.17); these seven
cut across them.

1. **A reduced signal set has absent columns, not NaN columns.** An analysis that does not
   use Pgas, say, does not write `pgas_*` columns at all, in the breath table, the
   workbooks or the cohort summary. NaN-filled columns for signals nobody recorded would
   read as "measured, and missing". The price is that a batch mixing files of different
   signal sets would give a ragged cohort table, which is why point 2 exists.
2. **One channel layout per analysis.** `input.channels` applies to every file the
   analysis matches. An EMG-only manoeuvre recording and a full-channel exercise recording
   are therefore two analyses, linked by references that name the other file by path (a
   reference source need not be among the files being analysed). Analysing files of
   different layouts in one batch (a `file_groups` idea) was considered and left out: it
   is the largest architectural change in the list and nothing needs it yet.
3. **`analysis.signals = []` means "derived", not "none".** An empty list is what an
   analysis written before signal sets existed already looks like, so it must keep meaning
   whatever channels are assigned. An explicit list is only stored while it differs from
   the derived one, and a channel assigned but not named in an explicit list is reconciled
   upward when the file is loaded (with a notice), never dropped.
4. **An unresolved reference is a caution, never an error.** A reference that names a
   file or breath that does not exist gives blank values, a note in the run report and a
   caution in `respmech validate`. Only `processing.lung_volume.require_references` turns
   a missing reference *file* into a blocker. A batch of forty participants should not stop
   because one IC recording was mislabelled, but the missing value must be visible.
5. **A subset run writes the same columns as the full run.** Whether a lung-volume column
   exists is decided from the settings across all matched files, never from the subset
   being run, and a reference source outside the subset is read only for its manoeuvres
   (nothing is written for it). Otherwise a test run on two files and the real batch would
   disagree about the shape of the table.
6. **A spirometry FEV₁ wins over one derived from a typed FVC breath.** When
   `input.subjects` supplies `fev1_l`, the placement of tidal breaths against the maximal
   flow-volume curve and the ventilatory capacity use it (`fev1_source` records which one
   was used); the value derived from the recording is the fallback, flagged as such. A
   formal spirometer reading is the more trustworthy number.
7. **References are typed breaths only.** A reference names breaths, not time intervals: a
   nasal sniff has no mouth flow to build an interval-based rule on, and a typed breath
   carries its own quality flags. An interval form can be added later without changing the
   settings that exist now.

**Compatibility.** Every new field is empty or off by default, and an analysis that does
not use a new table is read, run and written exactly as before (the golden scenarios pin
this). Settings keys a version does not recognise now survive a save, but only from 2.5:
an analysis that uses the new tables must not be opened and saved by an older version,
which drops them, and that cannot be fixed after the fact. The changelog, `docs/INSTALL.md`
and the website say so, and the desktop app shows the unknown keys once when a file that
carries them is opened.

---

**29-09-2026 — Sample entropy on a volume column is computed on the conditioned volume
(author's decision 26-09-2026); deliberate numerical change, re-baked.** Four points:

1. **The rule.** An entropy column that coincides with the volume column is computed on the
   volume RespMech analyses (zeroed, drift- and trend-corrected as configured), exactly like
   an EMG column that is also an entropy column is computed on the processed EMG. The
   alternative, keeping the raw file column and offering the conditioned volume only as a
   separate derived column, was recommended and rejected by the author: entropy of "the"
   volume RespMech reports is what a reader expects. Flow, Poes, Pgas and Pdi are unchanged
   (raw trimmed column = the mechanics' own array).
2. **It is a numerical change for existing analyses** that list the volume column under
   `entropy`; nothing else moves (the five committed golden scenarios and every scenario
   without entropy on the volume column are byte-identical). It is re-baked in one new golden
   scenario, `entropy_on_volume`, with drift correction on. Countermeasures: the changelog
   says it in plain words, the settings schema is bumped to 3 so a saved older analysis gets
   one plain-English notice when it is opened (and not again after it is saved), and the
   Provenance sheet names the rule.
3. **The window.** The conditioning is applied to the entropy input matrix before breath
   segmentation, not per breath from the breath's own volume: a breath's own `volume` is the
   inspiration and expiration joined, one sample shorter than the entropy window (the sample
   between the two phases), so a per-breath substitution would have changed the window of
   this one column relative to the others.
4. **`input.channels.entropy_derived = ["volume"]`** covers a volume with no column of its
   own (`integrate_from_flow`); its three columns equal what a volume column would give and
   stay out of `sample_entropy_max/min/mean`.

---

**28-09-2026 — Normalisation to a maximal manoeuvre: a post-pass over the finished
tables, one sheet, reference read from the manoeuvre extraction (author's decision
28-09-2026).** Seven points that are easy to undo by accident:

1. **It is a post-pass, not a change to `compute`.** `core.analysis.normalisation.attach`
   runs after the operating-lung-volume pass, reads the per-file tables and the raw breath
   dicts, and stores its own table on `FileResult.pressure_normalised`. It never adds a
   column to `breaths_table` or `average_row`, so no golden value, the cohort summary or the
   `Average breathdata` workbook can move (pinned by a test that runs the same batch with the
   analysis on and off and compares the tables). Off by default:
   `processing.pressure.normalization.enabled`.
2. **The reference is the existing `max_insp`/`sniff` resolution, with the LARGEST value of
   each quantity taken over the breaths a link names.** The best of repeated maximal
   efforts is the reference; the kind(s) behind it are recorded in Provenance and the run
   report because sniff Pdi exceeds Pdi during a maximal inspiration (Miller et al. 1985).
   A link that mixes the two kinds is allowed, with a notice, rather than refused.
3. **Tension-time is written with the pressure-time integral:
   `tt_es = int_oesinsp / (ttot * poes_max_ref)`.** That equals `(Pmean/Pmax)*(Ti/Ttot)`
   (Bellemare & Grassino 1982) with `Ti` cancelled, so nothing new is measured. `tt_es` and
   `tt_di` stay separate columns: TTmus is about 2.1 times TTdi (Ramonatxo et al. 1995), so
   they cannot share a threshold.
4. **Each EMG channel is normalised to its OWN peak in the maximal breath.** The extraction
   now also writes `rms_max_ref_col_<channel>` beside the existing scalar `rms_max_ref` (the
   largest channel peak). Summary columns across channels (`rms_max`, `rms_insp_max`, and the
   `nrdi` input `rms_insp_max_pct`) use the scalar; a `_col_<channel>` column uses that
   channel's own value. Dividing one channel by another channel's peak would not be a
   percentage of its own maximum.
5. **`nrdi = rms_insp_max_pct * bf`, per breath.** The breath rate is constant within a file,
   so the file mean equals `mean(rms_insp_max_pct) * bf` (Murphy et al. 2011). It needs a
   maximal-effort reference: against a file's own maximum every file would read 100 % by
   construction, which is why the analysis has no per-file fallback.
6. **Two existing outputs change regardless of the setting.** The Manoeuvres sheet gains
   `rms_max_ref_col_<channel>`, and a `sniff` breath's `poes_max_ref`/`pdi_max_ref` are taken
   over the whole typed breath (no mouth flow means the phase boundary inside it is arbitrary).
   The EMG-normalised sheet also reads a named reference file at its typed maximal breath
   (`per_file_max` only). None of these touches a golden value.
7. **A known asymmetry, left as it is.** The reference swing is measured from the maximal
   breath's first inspiratory sample; the tidal swing from the mean over the same short
   baseline window `calcptp` uses. On noise-level pressure differences this moves a
   percentage by well under a percent, and changing the extraction would move the
   Manoeuvres sheet, so it was not touched.

**28-09-2026 — Modified Campbell diagram / PEEPi construction: built from the
published literature, with five fixed design points, before any code is written
(author's decision 28-09-2026).** No external code or correspondence is a
prerequisite; a later cross-check against other groups' implementations is a
possible follow-up, not a condition. The construction is:

1. **`t_flow` is the true zero-crossing, not the segment boundary.**
   `separateintobreathsbyflow` (compute.py) ends the expiratory run only when
   `flow > 0` OR the forward mean over `breathseparationbuffer` samples is `> 0`,
   so during an end-expiratory pause (flow at zero) the forward mean turns
   negative first and the boundary lands up to `buffer` samples before the real
   onset of inspiratory flow. `t_flow` is therefore the first sample of the
   inspiratory phase with `flow < 0` whose predecessor has `flow >= 0` (fallback:
   the phase start). It is a separate quantity from the golden-locked `calcptp`
   baseline (mean of the first `ptp_baseline_window_s` of the phase), which stays
   unchanged. Consequence for testing: the PEEPi drop must be independent of
   `breathseparationbuffer` (a synthetic breath with a 0.4 s zero-flow pause is
   the pinned case).
2. **The PEEPi rectangle is laid on top of the golden-locked Campbell polygon.**
   `calculatewob`'s five columns and its V = 0 crossing (decision 05-09-2026) are
   untouched, and `wobtotal` never absorbs the threshold work `wob_in_thr`. The
   extra area is reported in its own columns (`wob_in_thr`, `wob_in_total_thr`,
   `wobtotal_thr`), `wob_in_thr = peepi_source_value * vt * (98.0638/1000) * bcnt *
   vefactor` (cmH2O·L to J, the same factor as the existing WOB). This follows the
   precedent of extending the diagram with a separately reported area instead of
   redefining its components (Banner 1994).
3. **`peepi_source = 'corrected'` when Pgas exists, otherwise `'dynamic'`.**
   Dynamic PEEPi is the negative Poes deflection from the onset of inspiratory
   effort to `t_flow` (Haluszka 1990; Milic-Emili 1990), clamped to >= 0, set to
   0 below `min_deflection`, and NaN with a notice when no onset is found or the
   breath has no predecessor (the preceding breath is used even when it is an
   ignored breath). With Pgas, the expiratory-muscle contribution is removed by
   subtracting the Pgas rise over the SAME interval `[t_onset, t_flow]`
   (`peepi_pgas_drop`, `peepi_corr = max(peepi_dyn - peepi_pgas_drop, 0)`,
   equal to the pre-flow Pdi rise; Zakynthinos 1997, 1999). Which source fed
   the threshold work is written to Provenance.
4. **`int_oes_preflow` is reported separately** (with `ptp_oes_preflow`) and is
   never added to any `*_peepi` column: the single-baseline rule from
   `docs/PTP_INVESTIGATION.md` holds, so adding the pre-flow area to a PTP that
   is already referenced to its own baseline would subtract that baseline twice.
   The `*_peepi` PTP/integral columns are `int_oesinsp + peepi_dyn * ti` and, for
   the diaphragm, the same with `peepi_corr` (Appendini 1996).
5. **The starting thresholds are provisional until measured.** `search_window_s`
   (1.0 s), `smooth_s` (0.05 s), `onset_slope_frac` (0.1) and `min_deflection`
   (0.5 cmH2O) are literature-informed starting values, not measured ones. They
   are pinned on the built-in sample recording and synthetic data first and are
   frozen here only after being measured on real recordings (the lesson from the
   boundary-breath notice: a threshold chosen on feel can silently trade
   detection power away).
   Refinements made when the module was implemented (28-09-2026), to points 1, 2
   and 3 above: `t_flow` is the last `flow >= 0` to `< 0` crossing before peak
   inspiratory flow rather than the first one (the first is defeated by flow noise
   inside the pause); the threshold rectangle and the `*_peepi` PTP columns add only
   the part of the deflection the existing polygon/PTP references do not already
   contain, because the segmenter's boundary usually lands in the pause before the
   deflection and the existing columns then already hold it (otherwise it would be
   counted twice, and the answer would depend on where the boundary fell); and a
   window with no falling step at all reports a deflection of 0 rather than NaN
   (NaN is kept for an unlocatable window or a missing predecessor). The Pgas term of
   point 3 is the Pgas FALL over the interval (Pgas[t_onset] minus Pgas[t_flow]).
   First measurements (28-09-2026, synthetic data only, pinned in
   `tests/unit/test_peepi.py`): the analytical pause case is exact with the
   segment boundary 400 samples before flow (an exact-zero pause puts the boundary at the
   pause start for any `breathseparationbuffer` above the pause); on the built-in sample recording, which
   has no PEEPi but a Poes cardiac ripple and wander of 1-2 cmH2O, the starting
   `min_deflection` of 0.5 cmH2O reports a deflection in 3 of 8 measurable breaths,
   a wider smoothing window does not remove them, and 2.5 cmH2O reports zero. The
   values stay provisional until measured on real recordings.

The feature is opt-in (`processing.pressure.peepi.enabled = false`) and adds
columns only, so every existing golden scenario stays byte-identical with it off.
Sources: Haluszka J et al., Am Rev Respir Dis 1990;141:1194-7 (PMID 2111105);
Milic-Emili J, Recenti Prog Med 1990;81:733-7 (PMID 2126881); Zakynthinos SG et
al., Eur Respir J 1997;10:522-9 (PMID 9072979) and Am J Respir Crit Care Med
1999;160:785-90 (PMID 10471597); Appendini L et al., Am J Respir Crit Care Med
1996;154:1301-9 (PMID 8912740); Banner MJ et al., Crit Care Med 1994;22:515-23
(PMID 8125004).

**28-09-2026 — The MFVL/EFL/VEcap placement of a tidal breath against a file's own
forced-vital-capacity curve resolves the FVC and IC references SAME-FILE ONLY,
and uses a constant ("`eelv_tracking='none'`") IC operating point of its own
rather than the within-file EELV-tracking arithmetic the operating-lung-volumes
module already has (self-review, not fixed in the same change).** The design asked
for the placement columns to live in `breath['mfvl_ext']`, joined into the
per-breath table exactly like `breath['wob']` already is — which happens INSIDE
the main per-file processing loop, before that file's own table is built. The
existing pattern for a resolved CROSS-file reference (the IC/baseline machinery)
is a POST-loop pass, run only after every file in the batch has already been
through the main loop once, specifically so a reference naming a file processed
later is still resolvable. Reconciling "the column must exist before the loop
moves to the next file" with "a cross-file reference may not resolve until the
loop is done" was not attempted here: only a same-file forced-vital-capacity
breath's curve is used, and only that same file's own resolved inspiratory
capacity (held constant across the file, never per-breath EELV-tracked). A
cross-file reference or full EELV tracking for this specific feature is a
deliberate, documented gap for a later change, not a silent limitation — flagged
here so it is not mistaken for "not yet noticed" rather than "not yet done".

**28-09-2026 — Two near-identical MFVL ratio columns (a tidal breath's own peak
flow against the LOCAL ceiling its own operating range implies, versus against
the GLOBAL peak of the whole curve) are both computed, with the LOCAL one treated
as placement-dependent and the global one not — a best-effort reading of a terse
specification, not independently confirmed (self-review, flagged for
confirmation, not settled).** The inspiratory side has no separate curve at all
(only a single peak-flow reference from a near-maximal breath, when one exists),
so its own "local" column is computed with the SAME formula as its "global"
counterpart today — correct by construction, but only because there is nothing
else it could mean yet. Revisit if a dedicated maximal inspiratory flow-volume
curve is ever added.

---

**28-09-2026 — Manual segmentation overrides repair the automatic detector, they
never replace it; applied after automatic segmentation, before trim (boundary
notices), breath numbering and phase re-splitting; an empty override list is
byte-identical to today (author's decision, 26-09-2026).**
`processing.segmentation.overrides` (`SegmentationOverrideEntry`: `cut_s`/`join_s`,
one entry per file) exists to repair a mis-detected breath boundary on a
flow-/volume-bearing recording — a flow wobble that over-splits one real breath, or a
flat/leaky expiration that under-splits two real breaths into one — never to
introduce a second, competing segmentation method. This is the reason it is a small
adjustment list layered on top of the existing automatic detector
(`compute.apply_segmentation_overrides`, called from `segment_file` right after
`separateintobreathsbyflow`/`...byvolume`), not a parallel mechanism the way the
EMG-only `separators` method is (that method has no automatic detection at all to
repair, since an EMG-only signal set has no flow channel to detect boundaries on).
Considered and rejected: exposing overrides as a third `processing.segmentation.method`
value alongside `flow`/`volume`/`whole_file`/`separators` — rejected because a
repair is meaningless without the automatic detection it repairs, so coupling it to
whichever of `flow`/`volume` is already selected (rather than letting it stand alone)
is the only shape that keeps "overrides describe adjustments to THIS file's automatic
result" true by construction. Applying it AFTER automatic segmentation but BEFORE
`trim_boundary_notices` (the K-035 truncation check), breath numbering and
ignore/kind assignment means every downstream consumer sees one finished boundary
list — a cut near a file's own edge legitimately CAN trigger a boundary-truncation
notice, and that is correct, not a bug to special-case around. An empty
`cut_s`/`join_s` (or no entry for a file at all) is never even passed to
`apply_segmentation_overrides` — `segment_file` calls it only when at least one list
is non-empty — so every existing analysis stays byte-identical by construction,
never by the repair function happening to be a no-op on empty input. Source: author's
decision, 26-09-2026.

---

**27-09-2026 — Cross-file `ic` reference aggregation excludes `reject_flags`-flagged
breaths; the column family is settings-wide, and an unresolved link once the family
exists is a caution on EVERY non-resolving file, not only an explicitly linked one
(author's decision, 27-09-2026).** `references.attach`'s aggregate over a resolved
`BreathRef`'s breaths drops any breath flagged with one of `IcSettings.reject_flags`
(`LOW_EFFORT` by default) before averaging — the same disqualifying rule
`apply_repeatability`'s own leave-one-out group already applies, so a rejected
attempt never silently drags a reference value the way it is already kept from
dragging a repeatability check. Chosen over aggregating every linked breath
unconditionally, which would have made `ic_ref_n` (the accepted-breath count) a
trivial echo of the settings' own breath list rather than a meaningful quality
signal. Separately: whether the three reference columns exist at all is decided from
settings across the FULL matched file set, never from what a given run's own subset
happened to resolve (so a subset run's written columns always match a full run's);
once that family exists, a file with no reference of its own gets the three columns
NaN'd plus a notice — read literally from the design's own "an unresolved link is a
caution plus NaN and a notice" policy applied to the whole family, not narrowed to
only a file that names an explicit link. This can be noisier than some study designs
want (every non-participating file in a large batch gets a notice once ANY file in it
uses an IC reference) — reopen if that proves a poor default in practice.

---

**26-09-2026 — Operating lung volumes: RV-anchored EELV is primary, TLC-anchored is
reported alongside as an absolute value; `eelv_tracking` defaults to `"none"`; a
within-file tracking request against a cross-file IC reference NaNs rather than
guesses (author's decision, 26-09-2026).** `vol_eelv = vc_src - ic_op` (volume above
residual volume at end-expiration — ERV by definition; there is no separate
`vol_erv` column) is the PRIMARY EELV datum, against the recommendation reviewed at
the time (a TLC-anchored primary with the RV-anchored value as the secondary
`vol_erv`). Chosen because a spirometry-derived VC (`input.subjects.vc_l`, or in
future a linked FVC manoeuvre's own value) is far more often available in practice
than a measured TLC (which needs plethysmography or an equivalent) — the primary
family is therefore usable without any lung-volume measurement beyond ordinary
spirometry, a practical reason, not merely a stylistic preference. The TLC-anchored
absolute value (`vol_eelv_abs = tlc - ic_op`) is kept alongside, never merged into
one column family, so a study that DOES have a measured TLC is never forced to
choose between the two. `ic_op = vol_ic_ref - d_eelv` (not `+`): a RISE in EELV
between breaths SHRINKS the inspiratory capacity actually available for the next
breath, so the reference IC must be reduced by the same amount the end-expiratory
level has risen, never increased. `eelv_tracking = "none"` (the reference IC held
constant across the whole file) is the default, matching the traditional "IC
measured once, assumed constant" convention; `"within_file"` is opt-in, and is
restricted to a same-file IC reference only — two different recordings rarely share
a common volume zero, so a cross-file end-expiratory comparison would be numerically
well-defined but physiologically meaningless, and NaNs with a notice instead of
silently reporting a number nobody should trust. See
`docs/REVERSE_ENGINEERING.md` §5.14 for the full arithmetic.

**Known gap, not fixed in the same change (self-review, 27-09-2026):** the RV-anchored
family's whole rationale (`vc - ic_op` equals `eelv - rv`) depends on a subject's
entered `tlc_l`/`vc_l`/`rv_l` being mutually consistent (`vc_l ≈ tlc_l - rv_l`).
`Settings.validate()` only checks `rv_l < tlc_l`; nothing cross-checks `vc_l`
against the other two, and `rv_l` is not consumed by any code path outside that one
check today. An internally inconsistent subject entry therefore produces a
silently wrong split between the two EELV families, with no notice — the module's
own implausible-value checks (`vol_eelv < 0`, `vol_eelv_abs < 0`) cannot catch it,
since both families can individually be non-negative and still disagree by exactly
the inconsistency. Left open, to either validate
`abs(tlc_l - vc_l - rv_l)` against a tolerance, or document `rv_l` as presently
decorative.

---

**27-09-2026 — Manoeuvre-extraction quality thresholds (`IcSettings`) ship as
documented placeholders, not measured values; `NOT_REPEATABLE` is a second pass over
the whole file, not part of `extract()` itself (author's decision, 27-09-2026).**
`core.analysis.manoeuvres.extract()` pulls IC/FVC/max-effort values straight from a
typed breath's raw dict (it never reaches `calculatemechanics`, see the "typed breath"
decision below). Its quality flags (`EELV_UNSTABLE`, `LOW_EFFORT`, `NO_PLATEAU`,
`NOT_REPEATABLE`) each need a numeric cut-off — `eelv_tolerance_frac`,
`plateau_flow_lps`/`min_plateau_s`, `low_effort_frac`, `repeatability_frac` — and no
real IC recording was available in the sandbox that wrote this code to calibrate any
of them against. Every one is a starting value only, pinned by analytical/synthetic
tests (the formula is trusted, the cut-off is not) — measuring them against real
recordings and freezing the result here is still owed. `NOT_REPEATABLE` specifically
cannot be decided by `extract()` at all: a single call sees one breath, never its
file's OTHER typed IC attempts. `apply_repeatability(manoeuvres, ic_cfg)` runs once
per file, after every typed breath's own `extract()` result is in hand, comparing each
eligible IC's `vol_ic` against the mean/median of the file's other eligible ones — an
IC already carrying a `reject_flags` flag (`LOW_EFFORT` by default) is excluded from
that comparison GROUP entirely, and a lone IC with no sibling is never flagged. Chosen
because repeatability is inherently a property of a pair, and a bad attempt should not
be able to either drag a genuinely repeatable pair down or "agree" with another bad
attempt to look falsely repeatable.

**27-09-2026 — `ic_eelv_pre` averages up to N preceding TIDAL breaths' own end-
expiratory volume, falling back to this breath's own immediate pre-inspiratory sample
only when too few precede it; `EELV_UNSTABLE` scales that preceding set's own spread
by the MANOEUVRE'S OWN `vol_ic`, never by `ic_eelv_pre` itself (author's decision,
27-09-2026, corrected same day by self-review).** The alternative to averaging —
always using the manoeuvre breath's own single `inspiration['volume'][0]` sample as
the baseline — is noisier and was set aside in favour of averaging over
`IcSettings.preceding_breaths` (default 3, minimum 2) tidal breaths, which the module
falls back to that single sample for only when fewer than `min_preceding_breaths`
tidal breaths exist before it in the file (too little context to average, e.g. a
manoeuvre near the very start of a recording). `EELV_UNSTABLE`'s first cut, a plain
coefficient of variation of the preceding breaths' own EELVs (SD ÷ their own mean),
was found wrong at self-review before merge: this codebase zero-references AND
drift-corrects volume by default (`processing.volume.correct_drift=True`), so a real
`ic_eelv_pre` routinely sits within a few millilitres of 0 L, and dividing by a
near-zero baseline made the ratio explode on ordinary breath-to-breath noise — the
flag would have fired on almost every real recording, not just genuinely unstable
ones. Scaling the SAME spread by `vol_ic` instead (the manoeuvre's own, always
non-trivial, volume) keeps the flag well-behaved near zero and physiologically
scale-appropriate: "the pre-manoeuvre baseline wandered by more than
`eelv_tolerance_frac` of the manoeuvre's own size" is the actual question a reader
needs answered to trust `ic_eelv_pre`, not a self-referential ratio of the baseline to
itself.

**27-09-2026 — `poes_max_ref`/`pdi_max_ref` (max_insp/sniff) are SWINGS from the
breath's own immediate pre-inspiratory baseline, not the raw absolute pressure
(author's decision, 27-09-2026, corrected same day by self-review).** The first cut
reported `poes_max_ref = −min(inspiration['poes'])`/`pdi_max_ref =
max(inspiration['pdi'])` — the absolute peak pressure, unlike `poes_ic_swing`/
`pdi_ic_swing` above, which are already baseline-subtracted. Self-review flagged the
inconsistency: a resting Poes baseline is typically −5 to −8 cmH₂O (balloon
zero-offset, not physiological zero), so the absolute-peak form silently inflated the
reference by that offset, and a later normalisation ratio ("this breath's swing
as a fraction of the max reference") would have divided a baseline-subtracted number
by one that still carried it — never a physiologically meaningful ratio. Both are now
swings from `inspiration[...][0]`, matching `poes_ic_swing`/`pdi_ic_swing` exactly.

**27-09-2026 — `BOUNDARY` is judged directly from the argument breath NUMBERS
`extract()` was given (first/last among the file's tidal breaths plus itself), not
from a separate flag threaded in from the pipeline (author's decision, 27-09-2026).**
Keeps `extract()` a pure function of exactly its own five arguments; the pipeline
never needs to compute or pass "is this the file's first/last breath" itself.

---

**27-09-2026 — `use_expiration`/`reference_intervals` remain the two saved noise-
reference fields for a flow-bearing analysis; `reference_mode` only names the EMG-only
alternatives (author's decision, 27-09-2026).** An EMG-only signal set has no
inspiration/expiration phases for `use_expiration` to mean anything about, so it needed
its own reference-resolution rule rather than reusing or repurposing that flag. Rather
than overload `use_expiration`/`reference_intervals` with a third, EMG-only meaning, a
new `reference_mode` field (default `'auto'`) was added purely to NAME the two EMG-only
sources (`'rest_segments'`, a segment typed `'rest'` in the reference file; `'interburst'`,
not yet implemented) — `'auto'` with a flow channel declared reproduces the existing
`use_expiration`/`reference_intervals` rule exactly, unchanged. `resolve_noise_reference_
mode(settings)` is the one function every consumer (the pipeline, the run report, the
Provenance sheet) reads this from, so the resolution logic cannot drift between them. A
typed breath (any kind) is also now excluded from the flow-bearing reference file's own
expiration mask when that mode is `'expiration'` — a manoeuvre's expiration is not the
diaphragm-quiet period the profile is trusted for — with no effect on any analysis that
has no typed breaths at all (every existing scenario). See `docs/NOISE_ECG_OPTIMIZATION.
md` §6 for the full resolution chain.

**27-09-2026 — A typed breath counts toward `bcnt`/`vefactor` exactly like a manually
excluded one: neither is scaled down when a breath is retyped as a manoeuvre (author's
decision, 27-09-2026).** `bcnt` (the breath count used to scale breathing frequency and
minute ventilation) is `len(breaths)` — every breath the segmenter DETECTED, whether or
not it is later excluded or typed — and `vefactor` is a pure function of the recording's
own duration, unrelated to which breaths are counted. Typing a breath as `ic`/`fvc`/
`max_insp`/`sniff`/`rest`/`other` folds it into the same exclusion machinery a manual
click already used (see `core._legacy_ns.to_legacy_ns`'s union of `exclude_breaths` and
`breath_types`), so it is left out of the tidal average the same way, but the file's
own breath count is not renormalised to pretend the manoeuvre breath was never there.
This is not a new rule invented for breath typing: the codebase has never scaled the
breath count by exclusion, and a typed breath is deliberately not made a special case
of that existing behaviour.

**27-09-2026 — Relevance-driven visibility (R7) is a permitted refinement of
inverted gating, not an exception, and Sample entropy is never hidden by the
signal set (R8) (author's decision, 27-09-2026).** A surface — a sub-tab, an
Advanced… settings group, a result column, a figure — that CANNOT apply to the
declared signal set is hidden as a pure function of `analysis.signals`, never
as a function of how far the user has progressed through the workflow.
Everything hidden this way comes back the moment the signal set changes
(Setup ▸ Signals): there is no separate "unlock" step and no progressive
disclosure. The commitment sheet stays the only gate that can actually block a
run. Sample entropy is the one deliberate exception to the exception: it stays
conditioned on whether a column is assigned to it (`bool(channels.entropy)`)
alone, regardless of the declared signal set, because entropy is an
opportunistic measurement any recording can carry, not a consequence of which
pressure/flow signals the analysis otherwise declares.

**26-09-2026 — Changing an analysis's declared signal set clears the breath-keyed
state built for the previous segmentation, after one confirmation (author's
decision, 26-09-2026).** Removing or adding Flow changes which segmenter produces
every breath number and kind, so a breath exclusion, breath-count override, or
(once they exist) a breath type, reference or manual separator made
under the old segmentation no longer describes anything real under the new one.
The signal-set picker's one funnel (`apply_signal_set`) asks once, covering all of
those lists together, and only when flow's membership of the set actually changes
AND there is something to lose — a freshly reset analysis is never asked, since
nothing has been built yet to lose. Precedent for the general principle (dropping
a role clears the derived state that depended on it) is the existing EMG-role
removal: clearing a now-invalid `ecg_auto_detect` the moment the EMG role leaves
the channel set, silently, no confirmation, because a stuck-invalid `Settings()`
with no visible way out was worse than asking. This decision keeps that same
principle but adds the confirmation specifically for flow, because losing flow
invalidates every existing breath number, not one boolean flag — a change large
enough that silently discarding it would be a surprise, not a repair.

**11-09-2026 — Three number-changing findings from the 04-08-2026 mechanism review
stay unimplemented (author's decision, 04-08-2026).** A review of the analysis
mechanisms surfaced three ideas that would change computed numbers: scaling the
breath count on exclusion, unit declaration with a plausibility check, and
filtering phantom breaths detected from flow noise. All three were deliberately
rejected — do not re-propose them without a new, separate decision.

**11-09-2026 — The resistive-work crossing in `compute.py` (recoil line crosses at
V = 0, not at the inspiration's starting volume) stays as-is (author's decision,
05-09-2026).** Inherited verbatim from the v1 code and locked by golden. The
behaviour is kept and documented as known v1 heritage, with a recommendation to
use `correct_trend` when EELV drifts — not fixed in place. This is a fourth entry
alongside the three findings above; don't reopen without a new, separate decision.

**11-09-2026 — Plain semver only, no `-rc`/`-beta` suffixes (established practice,
`docs/RELEASING.md`).** A PyPI version is immutable once published: a broken
release is fixed with a patch bump and a brand-new tag, never a re-push of the
same tag or a pre-release suffix.

**11-09-2026 — The release e-mail is never sent automatically by any app-side
workflow (author's decision, carried from the website's own decision of
03-08-2026).** Only the website's own announce workflow mails subscribers, and
only when triggered separately from a deploy. Framed here from the app side: the
release workflow's `repository_dispatch` to the website repo updates the site
and its changelog page, nothing more — announcing is a deliberate, separate act.

**11-09-2026 — Releases use GitHub's Trusted Publishing (OIDC) for PyPI, no
long-lived tokens stored anywhere (established practice, `docs/RELEASING.md`).**
PyPI trusts exactly `publish-pypi.yml` running in the `pypi` environment of this
repository; there is no API token to leak or rotate.

**11-09-2026 — Windows installers are signed locally, after release, never in
CI (established practice, `docs/SIGNING.md`).** The chosen certificate class is
hardware-backed and cannot sign on a CI runner, so CI always ships an unsigned
MSI and a local, manual step (`scripts/sign-msi-certum.sh`) replaces the release
asset afterwards. macOS signs and notarises inside CI instead, since its
Developer ID key is not hardware-bound.

**11-09-2026 — Changes with no changelog-worthy user effect are marked
`<!-- changelog-skip <sha> <reason> -->` in `CHANGELOG.md`, never simply
omitted (established practice, see the entries already in `CHANGELOG.md`).** The
comment names which entry already covers the change's effect, or states plainly
that there is none, so a change is never silently unaccounted for.

**11-09-2026 — The v2 UI overhaul (two-tab workspace, `FileRail`, the Run &
results drawer, inverted gating) is the current, settled shape of the app,
shipped as v2.4.0 (21-08-2026).** It replaced an earlier multi-screen wizard
flow entirely; do not reintroduce tab-gated or progressive-disclosure UI
patterns without a fresh decision — the inverted-gating principle (every
surface always reachable, only the ACTION disabled) was a deliberate, reviewed
design change, not an incremental one.
