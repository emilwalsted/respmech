# Changelog

All notable changes to RespMech are documented here, most recent release first.
Dates are the release (tag) date. See [Releasing](docs/RELEASING.md) for how a
release is cut, and the [GitHub releases](https://github.com/emilwalsted/respmech/releases)
for the installers themselves.

## Unreleased

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
