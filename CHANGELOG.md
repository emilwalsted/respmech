# Changelog

All notable changes to RespMech are documented here, most recent release first.
Dates are the release (tag) date. See [Releasing](docs/RELEASING.md) for how a
release is cut, and the [GitHub releases](https://github.com/emilwalsted/respmech/releases)
for the installers themselves.

## Unreleased

**A modular analysis: signal sets, typed breaths, reference manoeuvres, EMG-only
recordings and operating lung volumes.** This release stops assuming that every
recording has flow, all three pressures and EMG, and that every breath is tidal. A new
analysis starts with a choice of **signal set** (Flow only, Flow + Poes, the complete
set, each with or without EMG, or EMG only), and Setup, Preview & QC, the columns a run
writes and the figures it draws follow it. Any breath can be typed as a manoeuvre (IC,
FVC, maximal inspiratory effort, sniff), and a file's IC, FVC, baseline and maximal-effort
references can come from a typed breath in the same or another file, with per-participant
spirometry (TLC, VC, RV, FEV1, MVV) applied to every file of that participant. On top of
that come operating lung volumes (EELV, EILV, IRV), FVC/FEV1/PEF extraction with the
placement of tidal loops in the maximal flow-volume curve (expiratory flow limitation and
ventilatory capacity), opt-in PEEPi with a modified Campbell diagram, normalisation of
pressure and EMG to a maximal effort with tension-time indices, automatic EMG-only
segmentation (fixed windows and EMG bursts), manual repair of the breath detector, and
the `respmech init` and `respmech breaths` commands. Everything new is off, empty or
absent until it is asked for, an analysis that does not use it is read, run and written
as before. Earlier results change only where an entry below says so (sample entropy on a
volume column, the default entropy template length, the noise-reduction reconstruction and
the ECG R-wave detector). README has a section on each part, and
`examples/settings.toml` a commented example of every new settings table.
**An analysis that uses the new settings tables must not be opened and saved by RespMech
older than 2.5**, which drops the tables it does not know (see `docs/INSTALL.md`).

**The Mechanics test run in Preview & QC works on a complete signal set that names EMG
explicitly.** The per-breath table and the Campbell diagram showed "Test run failed:
input.channels.emg must name at least one column" for an analysis with an explicit
`analysis.signals` that lists `emg`, such as the built-in sample, because the test run drops
the EMG channels (the EMG tabs show that work) but kept naming EMG in the signal set.

**Smaller changes since 2.4.0.**

- The Provenance sheet and `run-report.txt` now record the Python, numpy, scipy, pandas
  and librosa versions next to the RespMech version, so a numerical difference against an
  older result can be checked against a library upgrade before it is treated as a
  regression (a SciPy point release alone has moved the Simpson-integrated columns).
- Units read the same everywhere: the Units sheet and the output writers used `L/s` and
  `L/min` where the documentation used `L·s⁻¹` and `L·min⁻¹`; both now use the dot form.
- Preview's shaded area on the Campbell diagram is labelled "inspiratory resistive work",
  which is what it shows (the read-out above it is the total work of breathing).
- The citation example (README, About dialog) points at the concept DOI
  10.5281/zenodo.3270825, which always resolves to the latest archived version, instead of
  the version DOI of a 2019 test release; the About dialog also shows the citation year.
- The "Run & results" drawer button reads "Run & results" again; it had shipped in 2.4.0
  as "Run _results" (a lone ampersand is a mnemonic marker in Qt).
- Axis labels in the mechanics and ECG channel stacks no longer run into each other on a
  short panel: a label is written in full, then as the name alone, then left out, and comes
  back when the panel grows again.
- A breath that has been typed (IC, FVC, sniff and so on) or excluded now says so in text
  under its number in Preview & QC, besides its colour: "#3" over "FVC", or over
  "(Excluded)". When the view is zoomed out so far that the text would run into the next
  breath's, the text gives way to a one- or two-character marker in the same colour (the
  initial, "×" for an excluded breath), and returns when you zoom in.
- `.xls` (the old binary Excel format) is no longer offered as an input format: it could
  not be read without a dependency RespMech does not have and failed with an import error;
  `.xlsx` is unchanged. The unused `seaborn` dependency was removed.

**Sample entropy on a volume column now uses the conditioned volume (numbers change for
analyses with entropy on the volume column).** When the volume column is also ticked as an
entropy column, its sample entropy is now computed on the volume RespMech itself analyses,
after zeroing and drift/trend correction as configured, instead of on the raw file column,
in the same way an EMG column that is also an entropy column already used the processed
EMG. The sample-entropy values of an analysis with entropy on its volume column therefore
change without anything else having been changed; a saved older analysis shows a notice
once when it is opened, and the Provenance sheet names the rule. Flow and pressure columns
are unchanged, and so is every analysis without entropy on the volume column. New: 'Entropy
on derived volume' (Setup ▸ Sample entropy, or `input.channels.entropy_derived = ["volume"]`)
computes entropy on the integrated volume when the volume has no column of its own, for the
whole breath, inspiration and expiration (`sample_entropy_col_volume`,
`sample_entropy_insp_col_volume`, `sample_entropy_exp_col_volume`), without changing
`sample_entropy_max`, `_min` or `_mean`.

**Tidal EMG-only recordings without separators.** The third question in the EMG-only signal
set, 'Tidal breathing — detect bursts automatically', now works: each file is cut at the
onset of every burst of inspiratory EMG activity, found on the envelope of the
ECG-removed signal (the raw signal when ECG removal is off; a threshold with hysteresis, a minimum duration for a burst and for a
gap, after Hodges & Bui 1996). Every segment reports the neural timing of its own cycle
under `ti_emg`, `te_emg`, `ttot_emg`, `ti_ttot_emg` and `bf_emg` (the last burst of a
recording has no following onset, so only its `ti_emg` is filled in), and three per-file
quality columns, `emg_seg_n_bursts`, `emg_seg_burst_frac` and `emg_seg_contrast`. A
recording where no channel's envelope rises clearly above its resting level fails that
file with a named error instead of being cut into noise. The noise reference can be the
periods between the bursts (Preview & QC ▸ EMG – noise reduction, 'Use the periods
between this file's bursts'), and the automatic noise-reduction strength is chosen from
bursts against those periods. `method = "fixed_windows"` (settings file) cuts equal
windows instead, every `hop_s` seconds of `window_s` (5 s and 5 s by default). The window
length, the step and the four burst thresholds are in Preview & QC ▸ EMG – segments ▸
Advanced…, and the run report's Segmentation line names the method and the number of
bursts found (the Provenance sheet has a row for the thresholds). Choosing the burst door
also sets the noise reference to the periods between the bursts, unless a reference was
already picked. The burst thresholds are starting values that have been checked on
synthetic recordings only; look at the shaded segments before trusting them on real ones.

**Pressure and EMG as a percentage of a maximal effort, and tension-time indices.** A new
`processing.pressure.normalization` setting (off by default) expresses every tidal
breath's inspiratory oesophageal and transdiaphragmatic pressure against the same
person's own maximal inspiratory effort: the breath typed `max_insp` or `sniff` (an
explicit link, then the group default, then the file's own typed breath, exactly like the
other manoeuvre references; for a nasal sniff, type the breath that contains it). Each
file's workbook gains a "Pressure normalised" sheet with the swing and the mean
inspiratory pressure (`poes_insp_swing`, `poes_mean_insp`, and the `pdi_` pair) in cmH₂O
and as a percentage of the maximum, the tension-time indices `tt_es` (oesophageal) and
`tt_di` (diaphragmatic; the two have different thresholds and are never mixed), and, with
EMG, `rms_insp_max_pct` and the neural respiratory drive index `nrdi` (EMG percentage
of maximum times breathing rate). A file with no reference reads blank, with one note
saying why; the Provenance sheet and the run report name the reference used, including
whether it was a sniff or a maximal inspiration, since those give different
diaphragm pressures. Nothing else changes with it on or off. Independently of that setting, the Manoeuvres sheet of a
`max_insp`/`sniff` breath gains each EMG channel's own peak (`rms_max_ref_col_<channel>`), a
sniff's pressure swings are now taken over the whole typed breath, and the
shared EMG normalisation reference (`processing.emg.normalization_reference_file`, with
`normalization = "per_file_max"`) is read at the file's typed `max_insp`/`sniff` breath when it has one (each channel against
its own peak in that breath), which also lets a file containing nothing but the
maximal manoeuvre serve as that reference.

**Opt-in VT/Ti breathing-pattern columns and variability.** Two new switches,
`processing.breathing_pattern.extended` and `.variability` (both off by default, in
Preview & QC ▸ Mechanics ▸ Advanced… ▸ Breathing pattern), add columns that need nothing
but flow and volume, so they work for a Flow-only analysis too. With `extended`, every
breath gains its mean inspiratory and expiratory flow (`mean_in_flow`, `mean_ex_flow`, the
VT/Ti drive component and its expiratory counterpart), the volume moved in each phase
(`vol_insp`, `vol_exp`), the rate that single breath would give (`bf_inst`) and the time to
peak inspiratory and expiratory flow (`t_peak_in_flow`, `t_peak_ex_flow`, and the
inspiratory one as a fraction of Ti, `t_peak_in_flow_frac`). With `variability`, each
file's row in the averages gains the coefficient of variation of VT, Ti, Te, Ttot and
Ti/Ttot over its included breaths (`vt_cv`, `ti_cv`, `te_cv`, `ttot_cv`, `ti_ttot_cv`) and
the number of breaths they rest on (`n_breaths`); the CVs read blank below three breaths. The
Provenance sheet names what is on. Nothing else changes with them on or off.

**Opt-in PEEPi-corrected work of breathing (modified Campbell diagram).** A new
`processing.pressure.peepi` setting (off by default) measures intrinsic PEEP for every
breath from the oesophageal-pressure deflection that precedes inspiratory flow — found
from the true start of flow, not from the breath boundary, so the result does not depend
on the breath-separation buffer — and reports the extra work that threshold adds to the
Campbell diagram in new columns beside the existing ones: the deflection (`peepi_dyn`,
and with a gastric-pressure channel the expiratory-muscle-corrected `peepi_corr`), how
long it lasted, its pressure–time area, the threshold work of breathing and the
totals with it (`wob_in_thr`, `wob_in_total_thr`, `wobtotal_thr`), and the
inspiratory pressure–time products including it. Existing columns, the Campbell
polygon and every earlier result are unchanged, with the setting on or off. The first
breath of a file, which has no preceding breath to search in, reads blank.
The four detection thresholds are starting values that have not yet been measured on
real recordings; on the built-in sample, which has no intrinsic PEEP, the default
minimum deflection is below that recording's own pressure wander, so treat small
values with caution until the thresholds are tuned.

**FVC/FEV1/PEF from a typed forced-vital-capacity breath, and how well a tidal
breath's own flow keeps up with it.** A `fvc`/`ic_fvc`-typed breath's Manoeuvres row
now carries the numeric extraction the previous release deferred: FVC, FEV1 and its
%FVC, PEF, the back-extrapolated volume (flagged when it is unusually large), forced
expiratory time and whether the manoeuvre reached a genuine end (ATS/ERS 2019), a peak
inspiratory-flow reference from a near-maximal breath right after it (some protocols
record one), and — when the same file also has an IC manoeuvre — whether the two
independently imply the same total lung capacity. Every tidal breath in that same
file then gets a set of new columns placing its own tidal flow-volume loop against
that curve: how much of its own exhaled breath runs at or above the ceiling the FVC
curve set (expiratory flow limitation, as a percentage and as a plain yes/no), how its
own peak flow compares to that ceiling, the fastest that breath COULD have been
exhaled given the ceiling, and the ventilatory capacity and breathing reserve that
implies (against either a spirometry-supplied maximum voluntary ventilation or the
classic FEV1×40 estimate). Without an inspiratory-capacity reference in the same
file, only the two flow-vs-peak columns are filled; the rest read blank, with a note
explaining why. A new figure, `flow-volume (tidal in MFVL).pdf`, draws those tidal
loops inside the file's own maximal flow-volume curve (grey loops, a bold average
loop, EELV and EILV marked), and for a Flow-only analysis the Preview panel shows the
same picture whenever the previewed file has a breath typed as a forced vital
capacity. The forced vital capacity manoeuvre is drawn as a whole loop: the inhalation
to total lung capacity is shown beside the forced expiration, both in the figure and in
the Preview panel. Mechanics ▸ Advanced… ▸ Lung volumes gained the curve source (largest
attempt or envelope), the two flow-limitation tolerances and the FEV1-to-MVV
multiplier; the figure can be switched off with `output.diagnostics.save_flow_volume`.

**New `respmech breaths` command; `respmech validate` reports reference-link
cautions; `respmech run --dry-run` and `respmech migrate` name more of what an
analysis will actually do.** `respmech breaths settings.toml [FILE]` lists every
detected breath (or, for an EMG-only signal set, segment) for one file or all matched
files, with its onset, duration, kind and whether it is excluded — the same breath
numbers a real run's output uses — and suggests which untyped breath looks like a
forced vital capacity manoeuvre, with a ready-to-paste settings snippet for it.
`respmech validate` now also reports any reference/subject link that will not resolve
once the batch's real files are known (advisory, except a genuinely missing reference
source when `require_references` is set, which now fails validation). `respmech run
--dry-run` prints each file's resolved inspiratory-capacity/forced-vital-capacity
reference. `respmech migrate`'s report now has a "Defaulted" section (an absent
Pgas/Pdi channel, or the derived signal set — a legacy file never named either).

**Repair a mis-detected breath boundary by hand (cut/join).** A flow wobble that
splits one real breath into two, or a flat/leaky expiration that merges two real
breaths into one, can now be fixed directly in Preview & QC ▸ Mechanics: click a
channel trace to cut a new boundary there, or click an existing one to join it away.
This repairs the automatic segmentation, it does not replace it — an analysis with no
manual repair is unaffected.

**The noise profile's expiration mask now follows the analysis's own segmentation
method and volume trend correction.** When shared-profile EMG noise reduction builds
its reference clip from a rest/expiration window (or samples active/quiet EMG across
the batch to auto-select the reduction strength), it used to always segment breaths by
flow with no trend correction, regardless of what the analysis itself was configured
to do. It now segments through the same trim/zero/drift/trend/segment sequence every
other stage uses, so the mask is built from the same breaths the analysis actually
runs on. This can change reported numbers for an analysis that both has EMG noise
reduction enabled AND either segments breaths by volume or has volume trend correction
on; every flow-method, no-trend analysis (the common case) is unaffected.

**New `respmech init` command; `respmech validate` and the run report now name the
signal set.** `respmech init new_settings.toml --signals flow,poes,emg [--folder DIR
--files MASK --fs 1000]` writes a commented starting TOML file with only the channel
entries the chosen signal set needs. `respmech validate` prints a `Signals: ... ·
Entropy: N columns · Analyses: ... · off: ...` summary line after "Settings valid",
the same `Analyses:` line is now part of `respmech run --dry-run`'s output and the
Run screen's commitment sheet, and run-report.txt and each workbook's Provenance
sheet both gain "Signals"/"Analyses" rows.

**One IC file per participant can now reference all of that participant's other
files.** A recording's inspiratory-capacity reference no longer has to be a breath
in the SAME file — a dedicated IC recording (or a shared, per-participant default)
can supply the value for every one of that participant's exercise files, and a file
excluded from a single-file test run still resolves the same reference a full batch
would. Reported as new `vol_ic_ref`/`ic_ref_n`/`ic_ref_source` columns, a
"Reference manoeuvres" line and block in the run report, and an "IC reference" row
in a resolved file's own Provenance sheet.

**Choose which IC/FVC/baseline/maximal-effort manoeuvre a file references — in the
same file or a different one.** The reference-picker menu entries mentioned above are
live: right-click an already-typed IC manoeuvre in Preview & QC ▸ Mechanics for "Use
as IC reference for ▸ this file / all files of its group / all files", or open
"Reference manoeuvres…" (also on a file's row in the file rail) for the full picker
across all four reference kinds and any file's typed breaths. A resolved IC reference
now shows next to the analysis window; the Run screen's commitment sheet names how
many files have one linked; and Setup gets a read-only "Subjects && lung volumes"
card listing the per-participant spirometry (`[[input.subjects]]`) these calculations
use.

**Operating lung volumes derived from a resolved IC reference.** Once a file has a
resolved inspiratory-capacity reference (see above), every tidal breath now also
reports its operating IC, EELV, EILV and inspiratory reserve volume (IRV) — as a
volume above residual volume (from a spirometry-derived vital capacity entered in
`[[input.subjects]]`) and, alongside it whenever a total lung capacity is entered,
the same set as an absolute value above RV. Per-file TLC/VC and the change in
IC/EELV against a baseline recording are reported too. A new "LUNG VOLUMES" block in
the run report and two new Provenance rows name the tracking mode and datum used.
Settings-file only for now, same as the reference feature it builds on.

**EMG-only analyses of maximal manoeuvres, with no flow channel at all.** Choosing
'EMG only' in the signal-set picker (New analysis, or Setup ▸ Signals ▸ Change… on an
existing analysis) now asks how each recording should be split into the segments
RMS/integral-EMG/sample-entropy are computed on — 'One maximal manoeuvre' treats the
entire file as one segment, or 'Several efforts or breaths — I will place separators'
lets you mark the boundaries yourself in Preview & QC's 'EMG – segments' tab (click to
place, click near an existing one to remove; any exclusion or type on a segment follows
it through a renumbering). 'Try it on sample data' opens a ready-made EMG-only demo,
already split at its own natural breath boundaries. Tidal breathing, with nothing to
place separators on, has its own door in the picker: see the next entry.
<!-- changelog-skip 6f2151f developer project-memory files only (the three CLAUDE.md files);
     nothing in the app or its user documentation changes. -->
<!-- changelog-skip f4e19c8 documentation-only decision record (docs/beslutninger.md); no
     user-visible behaviour changes, the feature it describes lands in a later change. -->
<!-- changelog-skip 72afcff the 'EMG – segments' tab landed in Preview & QC in this
     commit; folded into the rewritten bullet above once the signal-set picker's EMG-only
     preset itself was activated (see the 4ea83a7 marker below and the bullet's own
     history). -->
<!-- changelog-skip 4ea83a7 manual separator placement/removal landed in that same
     'EMG – segments' tab in this commit; folded into the rewritten bullet above once the
     signal-set picker's EMG-only preset itself was activated, making the whole feature
     reachable end to end for the first time. -->

**Right-click a breath to mark it as an IC manoeuvre and get its volume and Poes
without hand extraction.** Preview & QC ▸ Mechanics's breath overlays now take a
right-click (or Ctrl+left-click), offering Tidal, Excluded, IC manoeuvre, FVC
manoeuvre, IC + FVC, Maximal inspiratory effort, Sniff and Other… (the EMG-only 'EMG –
segments' tab offers Tidal, Excluded, Rest and Other… instead — the flow-only kinds
need a channel that tab does not have). A breath typed `ic`/`ic_fvc` reports its own
inspiratory-capacity volume, timing and pressure swings — plus quality flags for a low
effort, an unstable pre-manoeuvre baseline, no held plateau, an unrepeatable
measurement against a repeat attempt in the same file, or a manoeuvre right at the edge
of the recording — in a new "Manoeuvres" table shown right under the ordinary
breath-by-breath one, and in its own sheet alongside that file's workbook. A `fvc`/
`ic_fvc`-typed breath's expiration is checked for a plausible forced-manoeuvre
duration (the numeric extraction is described in the FVC/FEV1/PEF entry below); a
`max_insp`/`sniff`-typed breath reports its own peak effort (Poes/Pdi/EMG) as a
reference value for the normalisation described below. A disabled "Suggested: FVC" hint
points at the untyped breath with the longest expiration, the most plausible untyped
candidate. A dedicated recording where every breath is typed (a separate IC or FVC
file, say) used to fail the whole file with "no breaths detected"; it now runs and
writes its own workbook (the Manoeuvres table stands in for the usual breath-by-breath
data, with a note explaining why) instead. The reference-picker menu entries ("Use as IC
reference for…", "Reference manoeuvres…") are described above.
<!-- changelog-skip f380c02 the right-click/Ctrl+left-click primitive itself landed in
     this commit (BreathSpansItem.typeRequested + a minimal Tidal/Excluded/Rest menu);
     folded into the rewritten bullet above once the fuller manoeuvre menu made typing
     reachable end to end. -->
<!-- changelog-skip 5b0282d self-review fixes on top of f380c02 (a real-dispatch crash
     in the menu popup, a stale-scene click guard, the t_onset_s clock, distinct
     breath-kind colours, hardened tests); folded, same as f380c02 above. -->
<!-- changelog-skip 1bde775 IC/FVC/max-effort manoeuvre extraction and the Manoeuvres
     workbook sheet landed here, settings-model-only (no menu yet); folded into the
     rewritten bullet above. -->
<!-- changelog-skip 4d820a3 merge commit for 1bde775; same fold as 1bde775 above. -->
<!-- changelog-skip 9329db1 reference-only files (every breath typed, no tidal
     breathing at all) landed here; folded into the rewritten bullet above. -->
<!-- changelog-skip 950447d self-review fixes on top of 9329db1 (noise-report/QC-chip
     for the reference-only preview, a spurious cardiac-gate notice, a negative
     typed-breath count on EMG-only); folded, same as 9329db1 above. -->
<!-- changelog-skip a1c0245 merge commit for 9329db1/950447d; same fold as 9329db1
     above. -->
<!-- changelog-skip 578c691 test-infrastructure only: a new golden/characterisation
     scenario (typed_ic_fvc_same_file) locks the IC/FVC manoeuvre extraction feature
     already described in the bullet above against a dedicated synthetic recording,
     with its own analytical vol_ic == 3.0 L check -- no user-visible behaviour change,
     the feature itself already shipped via 1bde775/9329db1/f380c02 above. Also fixes a
     latent int-vs-string JSON key mismatch in the test harness (golden_newcore.py),
     never reachable before this ticket since no prior scenario emitted a "manoeuvres"
     key at all. -->
<!-- changelog-skip cd740fe test-infrastructure only: a new golden/characterisation
     scenario (typed_ic_crossfile) locks the CROSS-file counterpart of
     typed_ic_fvc_same_file's own self-reference against a dedicated,
     reference-only synthetic recording and an explicit processing.references
     link -- no user-visible behaviour change, the cross-file reference feature
     itself already shipped via the reference-manoeuvres/lung-volume bullets
     above. Also adds an end-to-end workflow test covering the realistic
     multi-file-per-participant lab setup (group-default resolution across two
     participants, run_batch + write_batch, group_readout, check_links, the
     cohort summary) that no earlier ticket in this programme exercised
     together. -->
<!-- changelog-skip d073670 test-infrastructure only: PREPARES (does not bake --
     no production data exists in the build environment) a new production-golden
     scenario, typed_ic_h5, that will exercise the same typed-IC/cross-file-
     reference feature already described above against a real recording -- no
     user-visible behaviour change, and no scenario is actually runnable yet
     without a locally-authored settings file the maintainer still has to create.
     Also adds a skip-marked test template for two of IcSettings' measured
     threshold flags (EELV_UNSTABLE, NOT_REPEATABLE), with placeholder numbers,
     pending calibration against that same real recording. -->

**Explore with sample data follows the signal set.** The startup chooser's own
'Explore with sample data' door still always opens the complete demo recording, but
'File/Analysis > Explore with sample data' now opens a sample matching whichever
signal set the current analysis declares: Flow only or Flow + Poes opens a matching
reduced sample (no pressure/EMG channels it would not otherwise have), and every other
signal set (including the full one) still opens the complete recording with ECG
removal and noise reduction demonstrated. `examples/settings.toml`'s Poes/Pgas/Pdi
channels are documented as optional now, matching what `Settings.validate()` has
allowed since the signal-set model was introduced.

**A new analysis now starts with a choice of signal set, and two reduced presets are
usable end to end: Flow only, and Flow + Poes.** 'New analysis' (from the startup
chooser, 'Get started…' and File/Analysis > New) opens a picker naming which signals
the analysis will use before anything else is set up; 'New from last rig' carries its
previous analysis's signal set forward automatically and skips the picker. Three
presets are reachable now — Flow only, Flow + Poes, and the complete set (Flow + Poes
+ Pgas + Pdi, still the default) — each with or without an 'Also EMG' toggle; 'EMG only'
is described in its own entry below, and 'Custom…' is shown but stays disabled until a
checkbox picker exists. Preview & QC follows the chosen set: with Poes absent, the
EMG sub-tabs still track EMG channels alone, the Mechanics channel stack draws only
the present channels, the Campbell panel draws a flow-volume loop instead of a
Campbell diagram (nothing to plot work of breathing against without Poes), and the
Advanced… dialog drops the Work-of-breathing and Pressure–time-product groups — all of
it purely a function of the declared signal set, never of how far a run has
progressed, and all of it returns the moment Poes is added back in Setup. Sample
entropy is never affected by any of this; it is shown whenever a column is assigned to
it, in every signal set. Changing the set after that point always goes through one
funnel that keeps `input.channels` consistent with it: a channel whose role leaves the
set is cleared, never silently left assigned to a role the analysis no longer
declares.

**Poes, Pgas and Pdi are no longer mandatory in `Settings.validate()`.** A new,
optional `[analysis] signals` table names which of Flow/Poes/Pgas/Pdi/EMG an analysis
actually uses; left out (the default), the signal set is derived from whichever
channels are assigned, so nothing changes for an existing analysis. An explicit set is
chosen in the signal-set picker (see the entry above) or written in the settings file;
the settings model and its validation understand the concept and keep an explicit set
reconciled with the channels actually assigned.

- `Settings.validate()` now requires only 'Flow' or 'EMG' to be present, not all four
  legacy roles: an analysis with no Poes/Pgas/Pdi channel assigned (and none named in
  `analysis.signals`) validates fine, and simply computes fewer columns. Requesting a
  pressure signal (Poes/Pgas/Pdi) without Flow, or an unrecognised signal name, is
  still rejected — breath segmentation needs Flow either way.
- A hand-edited `settings.toml` that assigns a channel without adding it to an
  explicit `analysis.signals` list is reconciled automatically on load (never on any
  other write path): the channel's role is added to the list, and a plain-English
  notice says so (visible in the desktop app and in `run-report.txt`, the same way
  every other schema-upgrade notice already is).
- Saving an analysis omits the `[analysis]` table entirely while its signal set still
  matches the assigned channels (the ordinary case) — an explicit table is only
  written once it genuinely diverges. The run manifest (`analysis-used.toml`) always
  records the actual, effective set, explicit or not, so a run's own provenance is
  never ambiguous about which signals it used.

**The CLI now validates several things only the desktop app used to catch, closing
gaps between what the two run.**

- **`processing.emg.noise.enabled` now requires `processing.emg.remove_ecg` to also be
  on — enforced in `Settings.validate()`, not just as a GUI activation gate.** A
  hand-written or migrated `settings.toml` that combined the two used to run the noise
  profile against a signal that still contained heartbeats, modelling the cardiac
  artefact as steady background noise. **This is a breaking change**: an existing
  analysis file with `noise.enabled = true` and `remove_ecg = false` now fails
  `respmech validate`/`respmech run` with a `SettingsError` naming both keys. Turn ECG
  removal on first to keep running it.
- `processing.emg.detect_channel` (a 0-based index into `input.channels.emg`) is now
  rejected by `Settings.validate()` when it is out of range, instead of crashing the
  batch mid-run with a raw `IndexError`.
- A misspelled or renamed TOML key is no longer silently ignored: `respmech validate`
  and the new `UNKNOWN SETTINGS KEYS` section of `run-report.txt` both name it,
  alongside the default that was used instead.
- `respmech validate` now also probes that the output folder is actually writable (the
  same real write-and-remove check the desktop app's Dry run performs — never
  `os.access`, which is unreliable against Windows ACLs), so a read-only or missing
  output folder is caught before an entire batch has been computed rather than after.
- `respmech run --dry-run` now prints the same output plan (`core.io.plan.plan_outputs`)
  the desktop app's Dry run shows — every output group and its file count under
  `data/`, `diagnostics/` and the provenance files — instead of only per-file breath
  counts.
- Running without the `plots` extra (or any other diagnostic-figure failure) now also
  prints a `WARNING: N diagnostic figure(s) skipped …` line to stderr (and to the
  desktop app's Run log) as it happens, in addition to the existing `FIGURES SKIPPED`
  section of `run-report.txt`. The run still completes; this is a warning, never a
  hard failure. Fixed a false positive from this same warning: figures drawn
  in-process because the isolated subprocess was unavailable or disabled
  (`RESPMECH_NO_FIGURE_SUBPROCESS=1`, or a packaged build) are not skipped — every
  figure is still written — so that case no longer triggers the warning.
- `respmech run`'s final line now reports the file count against the output folder
  itself (`Wrote N file(s) to <output.folder>`) rather than `<output.folder>/data`,
  which undercounted everything written outside `data/` (diagnostics, WAV exports, the
  two provenance files).
- `run-report.txt`'s `PROCESSING` block now also names any breath-count overrides,
  excluded breaths, the PTP baseline window, the work-of-breathing source and the
  cohort grouping regex — settings that previously only lived in `analysis-used.toml`,
  even though they can change the reported numbers the most directly.
- `run-report.txt`'s `DIAGNOSTICS` block now names every EMG channel by BOTH its
  0-based index into `input.channels.emg` and its 1-based data-column number (e.g.
  `channel index 1 (column 4)`) — previously only the index was shown, which could be
  misread against the workbook's `rms_col_<n>` columns and `EMG col <n>` WAV filenames,
  both 1-based.
- Per-file quality notices (an `ecg_auto_detect` quality-check mismatch, or the reason
  a cardiac-gated peak column came out NaN) are now also recorded in `run-report.txt`'s
  `DIAGNOSTICS` block — previously visible only as a Python warning on stderr, which a
  packaged desktop app never shows.
- Fixed a crash when an assigned pressure or flow channel is an all-integer column in
  the source file (for example an unused "dummy" port wired to a constant `0`): the
  loader now casts every channel to float64 on load, so arithmetic further down the
  pipeline (the VMR ratio, in particular) no longer depends on the input file's own
  dtype. Previously this could fail with a raw `UFuncTypeError` instead of a result.
- Volume-based breath segmentation (`processing.segmentation.method = "volume"`) no
  longer crashes with a bare `IndexError` when it cannot pair every inspiratory peak
  with an expiratory one (which can happen with slow breathing and pauses at zero flow
  between phases) — it now raises a named, explanatory error instead.
- A batch where at least one file failed now marks `Average breathdata.xlsx` and
  `Cohort summary.xlsx` as incomplete (an `INCOMPLETE` row in each workbook's
  Provenance sheet, and a `COHORT FILES INCOMPLETE` line in `run-report.txt`, both
  naming which files failed) — previously both were written from the successful files
  alone with nothing inside either file to say one was missing.
- `outlier_rms_sd_limit` (EMG RMS outlier filtering) no longer raises a `KeyError` on
  an analysis with no EMG channels configured; it now has nothing to filter and does
  nothing, matching every other EMG-only setting.
- The desktop app's File menu (New analysis, Open analysis…, Save, Save as…, Get
  started…, Explore with sample data, Duplicate…, and Open Recent) is now locked for
  the duration of a run — previously only the header's Analysis button was, while the
  File menu's identical actions (and their keyboard shortcuts) still worked and could
  swap the running settings out from under the batch.
- **A single click on a breath now marks it instead of toggling its exclusion.** In
  Preview & QC, clicking a breath (on the Mechanics stack, the EMG views or the segments
  tab) paints it green on every plot, in its number label and in its row of the
  per-breath or Manoeuvres table (scrolled into view if it is off screen), and draws its
  loop in green, with a legend entry, in the Campbell diagram or flow-volume loop. Click
  the marked breath again to unmark it, or another breath to move the mark. To exclude or
  include a breath, right-click it and choose *Excluded* or *Tidal*; the mark is a view
  aid only, it never changes the analysis and is left out of exported figures.
- **A recording cut mid-breath at either end is now flagged instead of analysed
  silently.** Trimming only ever discarded a leading partial expiration and a trailing
  partial inspiration — it never verified that the breath it *keeps* at either
  boundary is itself complete. A recording that begins already in inspiration, or ends
  still in expiration, kept that truncated boundary breath and analysed it as if
  whole, which (with drift correction on, the default) tilted the volume baseline of
  every OTHER breath in the file too, with no error and no warning anywhere. RespMech
  now raises a quality notice whenever a boundary breath is much shorter than that
  file's own typical breath — in Preview & QC while you tune, live in the Run log and
  at the terminal while a batch runs, and in `run-report.txt`'s `DIAGNOSTICS` block
  afterwards — naming which end looks truncated and how to fix it (re-export the
  epoch, or exclude the affected breath). This is a heuristic, not a hard boundary: a
  boundary breath only modestly shorter than usual is not flagged. The file still
  completes either way; nothing about the computed numbers changes.
- The boundary-truncation notice's threshold (how much shorter than the file's own
  typical breath counts as truncated) and minimum-comparison-breaths floor are now a
  per-analysis setting (`processing.segmentation.boundary_notice_min_relative_duration`
  / `boundary_notice_min_other_breaths` in `settings.toml`), defaulting to the same
  0.8/3 as before. A study whose recordings have unusually high natural
  breath-to-breath variability can raise the threshold itself instead of living with
  more false notices; nothing changes for an analysis that does not set these fields.

**Sample entropy's default `Template length (m + 1)` has changed from 2 to 3.** The
previous default of 2 reported a sample entropy computed at m = 1, not the m = 2 that is
the near-universal convention in the sample-entropy literature (Richman & Moorman 2000;
Yentes et al. 2013) — an established discrepancy between what the field calls "the
default" and what RespMech actually computed. Any analysis that leaves this field
unset now reports a different sample entropy value than before. Set the field to 2 if
you need to reproduce entropy values from an analysis run before this change; every
other Setup field, and every other measurement, is unaffected. The Tolerance (r), × SD
field's description was also expanded to state the literature's published interval
(0.1-0.25 × SD, 0.2 × SD the most common) and its source, and both the manual and the
app's own tooltips now agree on what m the current default corresponds to. The bundled
pyEntropy routine's licence is correctly stated as Apache-2.0 (it was mislabelled MIT
in a code comment); README's two entropy references have been corrected to their
verified PubMed listings (author names were previously garbled).

**The spectral noise-reduction reconstruction is now amplitude-neutral.** The STFT
round-trip used to reconstruct a bin's magnitude and phase additively instead of
scaling one complex number (masked magnitude × sign of the real part, plus the
unmasked imaginary part), so the "fidelity" metric (fraction of in-band EMG power
retained) routinely drifted above its documented ceiling of 1.0 — by roughly 30-40%
even at `prop_decrease = 0`, i.e. noise reduction inflated the signal it was meant to
leave untouched. The reconstruction is corrected to scale magnitude while keeping the
original phase exactly; at `prop_decrease = 0` the result is now numerically
unchanged, and fidelity can no longer exceed 1 (aside from float round-off). Every
noise-reduced EMG number changes as a result — re-run any analysis that used noise
reduction if you need numbers comparable with an analysis from before this change.

**`respmech.core.emg.remove_ecg`'s R-wave detector now matches Auto-suggest and can
detect an inverted R-wave.** The detector used to run on the raw EMG channel while
Auto-suggest derives `ecg_min_height` from a median-subtracted copy, so a channel
with any DC offset made a suggested height wrong for what the detector actually saw;
and only positive-going peaks were ever found, so a channel whose R-wave is inverted
could not be used for ECG removal at all. Both are fixed: detection now runs on the
same median-removed signal Auto-suggest uses, and both polarities are searched (a
biphasic complex is still counted once — the taller lobe wins). This can move the
detected peak set, and therefore the ECG-removed EMG, for an existing analysis.

**A shared cross-file EMG-amplitude reference (`processing.emg.normalization_reference_file`).**
The existing "% of peak/mean breath" normalisation (`processing.emg.normalization`)
reports each file's RMS relative to that SAME file's own maximum or mean, which makes
every file's own peak reach 100% by construction and does not, on its own, make
amplitudes comparable across files or subjects — the manual's wording has been
corrected to say so plainly. Setting `normalization_reference_file` to another file
already in the batch (typically a maximal inspiratory/expiratory manoeuvre recorded
once per subject) normalises every file's RMS against THAT file's own max/mean
instead, so a percentage now means the same thing across the whole study. Off by
default; existing analyses are unaffected until the field is set.

**Two new pre-run quality checks catch problems that used to surface only as a
confusing, misleading downstream error (or, for a flow channel flat at zero, an
outright hang).** Both are checked by the desktop app's Setup screen (a caution on the
QC strip) and by `respmech validate`, from the same shared logic:

- A recording whose time column looks like a regular sample clock but has duplicated
  or decreasing timestamps — the signature of more than one recording (e.g. several
  LabChart blocks) merged row-by-row by timestamp, OR simply concatenated one after
  another (each restarting its own time base, seen as a single large backward jump) —
  is now flagged by name in either shape, instead of silently letting segmentation
  jump between different recordings from sample to sample. A time column that is
  merely printed at coarser precision than its true sampling interval (giving a
  uniform rate of identical-looking timestamps end to end, tail included) is not
  mistaken for this: a genuine merge's own duplicate run ends once only one recording
  remains, which this check now requires.
- An assigned channel (flow, Poes, Pgas, Pdi, an EMG channel, or an entropy channel)
  that never varies — across the whole recording, or across a stretch reached partway
  through — is now flagged by name and column number; for anything other than flow
  this is advisory only (`respmech validate` reports it but does not fail on it — a
  permanently unused pressure port is a legitimate setup). **A flow channel flat at
  exactly zero specifically now raises a named error (`ConstantFlowError`) instead of
  running at all**, whether that is the whole recording or only a stretch of it (e.g.
  a genuine pause with no flow): breath segmentation cannot make progress past a point
  where flow is neither negative nor positive. This was traced, not assumed: with no
  EMG/entropy channels configured the previous behaviour was an unrecoverable hang;
  with at least one configured, it was instead a bare `ValueError: negative dimensions
  are not allowed` naming neither the file nor the cause. This is technically a
  breaking change for an analysis that happens to have an intentionally flat-zero flow
  channel or pause, which is not a realistic configuration to analyse as breaths
  either way.
- `DegenerateBreathError`'s message ("an incomplete breath right at the start or end
  of a recording") is now only shown for an actual first-or-last breath. A degenerate
  breath in the middle of a recording gets its own message instead, naming the
  breath's own timestamp and the more likely mid-recording causes (noise around zero
  flow, a mis-assigned flow channel, or merged data, per the check above) — reported
  by a user whose file's real problem was the merged-timestamps case above, but whose
  only symptom was a misleading "start or end" error for one breath in the middle of
  the recording.
- Settings keys this version does not recognise (a whole unknown table, a whole
  unknown list of tables, or an unrecognised field on one entry of an otherwise-known
  list such as `processing.exclude_breaths`) now survive a save instead of being
  silently dropped. Previously, opening a `settings.toml` written by a newer RespMech
  and saving it again from this version quietly deleted every table it didn't
  understand — **an analysis carrying settings this version doesn't recognise must
  not be saved by RespMech older than 2.5**, or that content is lost for good. Fixed
  in the same pass: a settings field declared with Python's `X | None` union syntax
  (rather than `typing.Optional[X]`) that points at another dataclass was built from
  the raw dict instead of an `X` instance — not yet reachable from any existing
  setting, but load-bearing groundwork for fields landing in upcoming releases.
- `core/quantities.py`'s unit lookup gained generic naming-convention rules — a
  result column named with a `_pct`/`_frac`/`_cv`/`_db` suffix or a `t_`/`peepi`/`tt_`
  prefix now resolves to the right unit (%, dimensionless, %, dB, s, cmH₂O)
  automatically, instead of needing a column-by-column entry. Latent for now: no
  column emitted by the pipeline today follows one of these conventions through the
  Units sheet or the on-screen result table, so nothing currently visible changes —
  this is groundwork for upcoming result columns that will.
- The ECG and EMG-normalisation references are now folder-tracked, like the noise
  reference: picking either against one recordings folder and later opening the
  analysis against a different one now shows the Setup Keep/Clear banner, instead of
  the reference either applying invisibly or (for the ECG reference specifically)
  being silently cleared the moment "Duplicate for another recordings folder…" pointed
  the analysis somewhere new.

<!-- changelog-skip be4ee99 internal, behaviour-neutral groundwork for a future modular
     analysis pipeline: a Qt-free capabilities/signal-set skeleton (its own new module and
     tests only), nothing in compute/pipeline calls into it yet, no user-visible change -->

<!-- changelog-skip 604d7f7 internal, behaviour-neutral groundwork for the same future
     modular analysis pipeline: loaders.py/settings adapter can now treat an absent
     poes/pgas/pdi column as an empty array instead of always raising, but the public
     settings API still requires all four columns today, so no analysis can actually
     reach this path yet; only a hand-built settings adapter in a new test exercises it -->

<!-- changelog-skip 85b0f24 internal, behaviour-neutral groundwork for the same future
     modular analysis pipeline: a per-breath EMG/entropy computation block was extracted
     into its own function with a not-yet-used phases= flag, and several places that read
     breath data gained forward-compatible presence guards (a channel or a phase split
     that can become absent in a future signal set) -- golden 5/5 byte-identical, no
     analysis can reach any of the new guarded branches yet -->

<!-- changelog-skip 9408f72 internal, behaviour-neutral groundwork for the same future
     modular analysis pipeline: run_batch's per-file trim/EMG-condition/volume-correct/
     segment sequence was extracted into its own segment_file() function, called from
     the exact same place -- golden 5/5 byte-identical, no user-visible change. Also
     splits the golden test harness's scenario table into legacy- and v2-core-bagt
     halves (the latter still empty) and removes a handful of internal ticket-id
     references from source comments/docstrings (public-repo hygiene only) -->

<!-- changelog-skip e3a13d2 test- and docs-only: generalized the existing lone-ampersand
     mnemonic guard (test_no_button_caption_turns_an_ampersand_into_a_mnemonic) from push
     buttons alone to group-box titles, menu/menu-bar actions, tab captions and buddy
     labels, as a shared _lone_ampersands(root) helper other screens' tests can reuse,
     plus a matching ui/CLAUDE.md note -- no runtime app behaviour differs -->

<!-- changelog-skip 3142f25 groundwork for the same future modular analysis pipeline:
     calculateaveragebreaths/calculatemechanics now skip Poes/Pgas/Pdi-derived values
     when settings.capabilities says the channel is absent, and the mechanics
     OrderedDict is built from the existing pinned key order instead of a hardcoded
     literal -- golden 5/5 byte-identical (every guard is True and unchanged on the
     full-channel path). This is the change that makes a flow-only/poes-only analysis
     run end to end through run_batch() without crashing (pipeline/results already
     handled a reduced column set generically); still not written up as a real
     changelog entry here because the wider modular-analysis feature this belongs to
     is still mid-flight on an integration branch, several tickets from done -->

<!-- changelog-skip ce1fe70 groundwork for the same future modular analysis pipeline:
     a reduced signal set (flow only, or flow+Poes without Pgas/Pdi) now reaches
     core/pipeline.py, core/results.py, ui/workers.py's preview staging and
     core/io/loaders.py end to end -- pipeline/results needed no change of their own
     (already generic over whatever columns a breath's mechanics dict contains), so
     this mostly makes an existing "None means absent" contract explicit (a raw
     diagnostic array is None, not empty, when its channel is absent; a preview
     series dict omits the key entirely) and fixes a real KeyError this exposed in
     Preview & QC's Mechanics channel stack for an already-reachable reduced
     mapping. Golden 5/5 byte-identical; still not written up as a real changelog
     entry here for the same reason as 3142f25 above -->

<!-- changelog-skip c085574 groundwork for the same future modular analysis pipeline: the
     diagnostic-figure writer and the pre-flight plan now both skip the Campbell/PV-loop
     figures (per-file and cross-file) for a signal set with no Poes, and the
     volume-correction/trend/drift figures for one with no volume trace, instead of
     attempting (and failing) to draw a pressure trace that is not there -- the reduced
     signal sets this enables (flow only, flow+Poes) are already reachable today via a
     hand-edited settings file, though the picker that will make them a normal, supported
     choice is still a later step of the same in-flight feature (see the earlier
     changelog-skip entries above). Golden untouched (this is plotting/reporting only, not
     compute). Still not written up as a real changelog entry here for the same reason as
     3142f25/ce1fe70 above -->

<!-- changelog-skip 41fbb6b internal fix: MainWindow's construction crashed on a
     malformed, hand-edited analysis.signals (a bare string instead of a list) because
     PreviewScreen's own build path called Capabilities.from_settings unprotected;
     Settings.validate() already reports this case cleanly, so the fix is purely
     defensive (degrade to the safest render, never crash) and never shipped in any
     release -- nothing for a reader of this entry to be told -->

<!-- changelog-skip 400884a same internal fix as 41fbb6b, extended to two more
     Capabilities.from_settings call sites on the same before-validate() real-startup
     path (self-review finding): the Setup 'You will get' preview and the Mechanics
     stack's floor sizing. Same reasoning: purely defensive, never shipped -->

<!--
"Unreleased" above is a hand-maintained draft of the next release's entry. It is
updated ONLY when explicitly asked to (not automatically on every commit), and it
describes everything since the last tag. When a version is tagged, fold these
bullets into that release's own dated entry as step 1 of docs/RELEASING.md, then
collapse this section back to the empty placeholder below and wrap it in an HTML
comment (like this one) so it stays invisible until asked for again:

## Unreleased

(nothing pending — ask for an update to populate this section)
-->

## v2.4.0 — 2026-08-21

The desktop app's workspace has been rebuilt around two tabs instead of three: Setup and
Preview & QC, with Run & results folded into a drawer under a shared, searchable file list
instead of living as a separate tab. Every screen and action stays reachable at all times —
only the action itself is ever blocked, and always with a full-sentence reason next to it.
Nothing in the analysis changed: a settings file run before and after this release produces
identical output.

<!-- changelog-skip 163a542 D17 self-review follow-up: honest button state and partial-clear reporting in Run & results. Its user-visible effect IS in this entry, under "nothing overstates what happened" — this release's entry is deliberately high-level for a UI overhaul, so the individual review passes behind a bullet are not itemised -->
<!-- changelog-skip be8ae93 D19 self-review follow-up (and a D18 regression it surfaced) in the run/cohort output path. Same reason as 163a542: covered by the "nothing overstates what happened" bullet, not itemised in a deliberately high-level entry -->
<!-- changelog-skip ca14414 maintainer-only doc/screenshot regeneration script, no app behaviour changed -->
<!-- changelog-skip 7da2f1f internal refactor: preview_screen.py split into a package, no behaviour change -->
<!-- changelog-skip 39c8e1f review-pass hardening of the durability fix already described in the memory bullet; no separate user-visible behaviour -->
<!-- changelog-skip 8dce581 self-review polish of D25's error handling; the user-visible wording behaviour is already described in the errors bullet -->

- **Two tabs, not three.** *Setup* and *Preview & QC*, with *Run & results* folded into a
  drawer under a shared, searchable file list that opens itself when a run starts. Every screen
  stays reachable at all times — only the action itself is ever disabled, and always with a
  plain-language reason beside it
- **Setup redesigned:** two columns instead of one long stack, a real File/View/Help menu bar
  with an offline About box, a startup chooser (New analysis, Open, Explore with sample data),
  and "Duplicate for another recordings folder…" for the next subject in a multi-subject study
- **A batch is read as a batch.** Column count, sampling frequency and LabChart header warnings
  are checked across the whole folder instead of assumed from the first file, and a file that
  disagrees is named rather than silently dropped. The channel dialog suggests roles from the
  file's own column headers and handles European comma decimals on its own
- **Your choices follow the recordings folder, not the filename.** Breath exclusions,
  breath-count overrides and the EMG rest reference are remembered per folder, so pointing the
  same analysis at a new subject who reuses a LabChart export name no longer inherits the
  previous subject's decisions — and anything carried over is flagged rather than applied quietly
- **Errors explain themselves, and nothing overstates what happened.** Plain-language diagnoses
  with the technical detail behind a Details toggle, an unreadable file named and skipped
  instead of aborting the batch, and progress, the output-folder confirmation and the pre-run
  plan all reflecting exactly what was and will be written
- **Preview & QC keeps up.** The mechanics stack no longer freezes the interface while
  redrawing, fits a short screen, and labels its channels, units, crosshair and analysis
  window. The Campbell preview matches the orientation of the figure RespMech actually writes.
  The EMG tabs show the live signal instead of a stale one and say plainly whether a panel is
  raw or processed
- **Smaller things that add up:** drag-and-drop of recordings folders and analysis files,
  double-clicking a .toml analysis in packaged builds, file-navigation shortcuts that no longer
  fight a focused table, visible focus rings and higher-contrast disabled fields, a round of
  layout fixes for Windows' wider fonts, and long sessions that no longer build up memory
- **Fixes found while testing this release:** the noise-reduction tab's row of controls no
  longer resizes and reshuffles itself without settling, and a test run with both Remove ECG and
  auto-detect switched on no longer stops with a settings error
- **A few labels changed to match what they actually do:** the Setup tab's blocking message is
  now "Setup incomplete" (was "Settings incomplete"), the noise tab's bare "Auto" checkbox is
  "Auto strength", the ECG tab's "Auto (whole batch)" is "Auto-detect for the batch", and the
  mechanics "Breath-separation buffer" is "Breath-separation debounce" — same field, same
  behaviour, just a name that no longer misdescribes it. An older screenshot or note naming the
  old label is talking about the same control

## v2.3.4 — 2026-08-03

An interface release: every window now fits every screen, both themes paint correctly on
Windows, and the Preview screens give their space back to the graphs. Nothing in the
analysis changed — a settings file run before and after this release produces identical
output.

<!-- changelog-skip c08a2b6 regression introduced and fixed inside this same release: the fidelity panel briefly drew a small plot over the larger one; the v2.3.3 -> v2.3.4 diff never contains it, so there is nothing for a reader of this entry to be told -->
<!-- changelog-skip 58adf8b maintainer tooling: scripts/sign-msi-certum.sh guards the release signing step, nothing an installed copy contains or an analysis can reach -->

- Fixed "Advanced…" dialogs opening taller than the screen, with OK/Cancel out of reach
  and the window impossible to shrink. Reported on a 1080p Windows laptop at 150%
  scaling, where the Mechanics dialog asked for 904 px against the 650 px available and
  put its buttons in the missing 254 px. The settings are now grouped into titled cards
  laid out in one to three columns depending on the width available, inside a scrollable
  body, with the intro, the live hint and the buttons pinned outside it so they are
  always reachable. Every window is also sized to the screen it opens on
- Fixed windows and dialogs rendering **black** in a light-mode session on Windows. Two
  style rules of equal weight set the window background in the wrong order, so top-level
  windows painted no background at all; macOS happened to hide it and Windows showed the
  black backing store
- Fixed **Enter** discarding your edits on any dialog: Cancel was the default button, so
  the one keyboard action anyone tries on a stuck modal threw the changes away. Enter now
  commits and Esc cancels, everywhere
- Preview & QC pages now scroll on a short screen instead of compressing the graphs into
  unreadable slivers, and each stacked channel keeps a minimum height of its own — five
  channels sharing one panel used to leave about 10 px of trace each. On a screen with
  room to spare, nothing scrolls
- Fixed several controls forcing the whole app to be wider than a laptop screen: the Run
  screen's button row now wraps, the header subtitle shortens, the noise-profile window
  scrolls per channel (with twelve channels each trace had no height at all), and the
  channel setup window opens sized to its content
- Fixed the colours that only worked in one theme: the Campbell diagram accent and the
  EMG RMS envelope stayed light-mode red on the dark background, the tick inside a
  checkbox could sit at barely visible contrast, the error card's icon and hint were dark
  on a light card, the "busy" overlay dimmed a dark panel by almost nothing, and the
  resize grip was invisible. A Campbell diagram **exported** from dark mode is now
  rendered light, so a figure for print does not come out near-black
- Preview & QC's EMG screens give the space back to the graphs: each panel's title and
  its channel picker now float in the top edge of the graph instead of taking a row of
  their own, the gated-peak settings moved into "Advanced…" (they change no panel — they
  only add columns to the saved data), and the Campbell diagram keeps a margin instead of
  running into the panel edge
- The "EMG – noise reduction" tab now divides into roughly equal thirds — controls, the
  two working views, and the three reference panels — instead of giving the two working
  views half the window and leaving the raw channels, the fidelity frontier and the
  detail PSD a sliver each
- Fixed the Campbell diagram panel cutting off its own labels. At the height the panel gets
  on a laptop the heading was drawn with its top outside the figure, so it read as a
  half-cut "Campbell diagram", and the rotated axis caption came out as "olume above
  end-ex". The heading now sits in the panel's own title bar, which cannot clip; the axis
  caption shortens to fit and returns in full on a taller panel; and the key no longer
  covers the loops it names when there is no room for it. Exporting the diagram is
  unchanged — a figure saved for a report still carries its title
- Fixed the graphs becoming unreadable at those compact sizes: y-axis tick labels printed
  through one another, axis captions were clipped to their last few characters, and the
  in-plot legends covered the very traces they named. Tick labels now thin out and return
  as a panel grows, axis captions shorten to the longest wording that fits, the trace
  names moved out of the plot into the panel's top edge, and the two small diagnostic
  figures drop the furniture they have no room for and get it back when dragged larger

## v2.3.3 — 2026-07-30

Batch-mode ECG auto-detection, a volume-trend correction that scales to the recording,
and a set of interface fixes, since v2.3.2.

- Added `processing.emg.ecg_auto_detect` so ECG auto-detection (previously reachable
  only via the GUI's "Auto-suggest" button) can now drive CLI/batch runs: analysed
  once on a reference file and applied to every file in the batch. Off by default,
  so existing `settings.toml` behaviour is unchanged. A per-file `detection_quality`
  warning (plus the auto-detected settings and confidence) is surfaced in CLI stdout
  and `run-report.txt`, so an unsupervised batch still leaves a visible trail of any
  file whose beats the shared parameters seem to be missing
- Added GUI parity for the above: an "Auto (whole batch)" checkbox on the Preview
  screen's ECG tab, mirroring the existing noise-reduction "Auto" checkbox, plus the
  same ECG/EMG-channel validation the CLI already enforced
- Mechanics "Advanced…" dialog: the "Resample to" and "Trend interpolation" detail
  fields now grey out when their own checkbox is unticked, instead of staying
  enabled-looking with no effect
- Volume trend correction now scales to the recording. The end-expiratory troughs
  that anchor the trend envelope were selected with an absolute depth threshold
  inherited from v1 (`processing.volume.trend_peak_min_height`, default 0.8), which
  matches nothing on ordinary tidal breathing — so on many recordings the correction
  either did nothing or failed inside numpy. The new
  `processing.volume.trend_peak_min_prominence_frac` is a fraction of the recording's
  own volume range, so it works at any tidal volume and in any volume unit, and it is
  editable in Mechanics ▸ "Advanced…". An analysis that set the old threshold
  deliberately keeps it and reproduces bit-identically; only the retired 0.8 default
  is upgraded, and that upgrade is reported in the GUI and in `run-report.txt` rather
  than applied silently (settings schema 1 → 2)
- A recording that cannot support a trend fit (no detectable trough, a single anchor,
  a flat or non-finite volume signal) or in which breath separation finds no breaths
  is now refused by name and with the reason, instead of surfacing as an opaque numpy
  or pandas error — and the rest of the batch still runs
- Fixed the main window running wider than the screen on Preview & QC: the EMG
  noise and ECG control strips no longer force an oversized minimum window width.
  On Windows this was still happening after the first attempt, because each control
  chip was itself a single unbreakable row: the strip could wrap around a chip but
  never inside it, so the window demanded 1516 px where a 13" laptop offers 1280.
  The chips now wrap cluster by cluster, which brings the minimum to 1228 px on
  Windows and about 700 px on macOS
- Fixed an analysis file that stored ECG auto-detect switched ON with Remove ECG
  switched OFF becoming unusable: the tickbox came up ticked and greyed at once, every
  preview was refused as "Settings incomplete", a test run stopped on a settings error,
  and there was no way to untick it without turning Remove ECG back on first. Loading
  such a file now corrects the pair and says so, and the correction is saved with the
  analysis. The command-line tool still reports the combination as an error, because in
  a hand-written file it is one
- The control strips on Preview & QC now line their captions, tickboxes, fields and
  buttons up on a shared centre line, instead of hanging from a shared top edge
- Added this CHANGELOG.md, covering the full v2.0.0–v2.3.2 release history
  <!-- site: 0 docs "Added this CHANGELOG.md" respmech.dk's changelog page already IS this list -->

## v2.3.2 — 2026-07-24

Performance pass across the analysis pipeline and the UI, plus better progress
feedback during batch runs.

- Cached file loading and ECG removal per run instead of repeating them
- Vectorised the sliding-window RMS calculation and the sample-entropy candidate counting
- Narrowed the ECG template-fit search before evaluating candidates
  <!-- site: 0 merged "Narrowed the ECG template-fit" the page says it as one faster-EMG-processing bullet -->
- Removed an unnecessary import of the compute core just to open a window
- Preview plots now use pyqtgraph decimation and view clipping
- The Run screen shows progress during the write phase, and the overall progress bar now reaches 100%

## v2.3.1 — 2026-07-22

Fixes a packaging issue where the installed app could relaunch itself while
writing batch figures.

- Fixed the packaged app re-launching itself when writing batch figures
- Regenerated the README screenshots for the v2.3.0 UI
  <!-- site: 0 docs "Regenerated the README screenshots" repository images only; nothing changes in the app -->
- More realistic sample EMG, with a recruited motor-unit interference pattern

## v2.3.0 — 2026-07-22

A UI reorganisation: a leaner Setup screen, "Advanced…" modals for less-common
settings on Preview/Mechanics/EMG, and refined plot layouts.

- Setup screen slimmed down; less-common settings moved into "Advanced…" modals on the Preview, Mechanics and EMG screens
- Channels list is now a read-only readout; the channel picker is the only place that assigns roles
- A column can now be flagged for sample entropy independently of its other role
- Cardiac-gated peak EMG exposed as an opt-in per-breath statistic, on the EMG tab of Preview beside the EMG it changes
- Flow silhouette is now drawn around zero rather than around the EMG midpoint
- Noise reduction now says in the interface that it waits for ECG removal first, instead of appearing inert
- Diagnostic figures are now written in a separate process
  <!-- site: 0 internal "Diagnostic figures are now written" figures are byte-identical, and it falls back in process -->
- Fixed three silent-mutation bugs: an unassigned channel could silently analyse the wrong (last) column instead of raising an error, opening the channel-assignment dialog could silently drop the EMG role from analysis, and scrolling the mouse wheel over a form could silently change the setting under the cursor — including the EMG RMS window, which changes every reported EMG value. If you ran analyses on an affected build, it is worth re-checking the results
- Numerous plot-theming, spin-box and dialog fixes for Windows/macOS parity
- Releases are now published as full GitHub releases rather than pre-releases

## v2.2.2 — 2026-07-18

A packaging-only fix.

- Fixed the PyPI publish workflow to reference the publish action by tag instead of commit SHA

## v2.2.1 — 2026-07-18

First PyPI publish.

- RespMech is now published to PyPI (`pip install respmech`) via Trusted Publishing — no tokens involved
- README refreshed with new screenshots and feature graphics for the 2.2 UI, including a live before/after EMG noise-reduction example
- Sample data made more realistic: EMG bursts confined to inspiration, ECG artefact scaled to dwarf the EMG as in real recordings
- A new release now automatically notifies respmech.dk to re-publish

## v2.2.0 — 2026-07-17

A large audit pass: trust and settings hygiene, restored diagnostics, and a
full UI polish round.

- Pre-analysis resampling, previously inert, now actually runs
- Restored and upgraded the diagnostic PDF outputs
- Trend correction applied in the mechanics preview
- TOML-only settings surfaced in an Advanced panel
- XLSX output polish; removed dead entropy code
- New EMG–ECG reduction tab, with capture markers and a reactive workflow
- A discrete flow silhouette is now superimposed behind every EMG time-domain graph
- UI polish across all screens (labels, alignment, spacing, button rendering)
- The v1 monolith moved into `legacy/`, and the README rewritten for the 2.x app

## v2.1.1 — 2026-07-14

A noise-fidelity fix, plus signed installers.

- Fixed a noise-reduction precondition that previously surfaced a raw traceback instead of a clear message
- macOS installer is now notarised and signed; the Windows MSI is signed locally via Certum

## v2.1.0 — 2026-07-12

Reactive recompute redesign: the Preview screen now updates incrementally
instead of recomputing everything on every change.

- Preview recomputation is now debounced and scoped to only the parts of the pipeline a change actually affects
- Cooperative cancellation, so switching files or settings no longer waits for a stale computation to finish
- Several Windows CI timeout fixes uncovered during the redesign

## v2.0.0 — 2026-07-12

First v2 release: the PySide6 desktop app, published with installers for
Windows and macOS.

- Guided Setup → Preview & QC → Run workflow
- Windows MSI and macOS dmg installers via briefcase
- CI: Windows + macOS GUI smoke tests, Linux numerical golden tests, protecting the physiology port from v1
- Various cross-platform fixes surfaced by CI (Windows/headless failures, dark-mode rendering, checkbox visibility)
