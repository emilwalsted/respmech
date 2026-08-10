# Changelog

All notable changes to RespMech are documented here, most recent release first.
Dates are the release (tag) date. See [Releasing](docs/RELEASING.md) for how a
release is cut, and the [GitHub releases](https://github.com/emilwalsted/respmech/releases)
for the installers themselves.

<!--
"Unreleased" below is a hand-maintained draft of the next release's entry. It is
updated ONLY when explicitly asked to (not automatically on every commit), and it
describes everything since the last tag. When a version is tagged, fold these
bullets into that release's own dated entry as step 1 of docs/RELEASING.md, then
collapse this section back to the empty placeholder below and wrap it in an HTML
comment (like this one) so it stays invisible until asked for again:

## Unreleased

(nothing pending — ask for an update to populate this section)
-->

## Unreleased

The desktop app's workspace has been rebuilt around two tabs instead of three: Setup and
Preview & QC, with Run & results folded into a drawer under a shared, searchable file list
instead of living as a separate tab. Every screen and action stays reachable at all times —
only the action itself is ever blocked, and always with a full-sentence reason next to it.
Nothing in the analysis changed: a settings file run before and after this release produces
identical output.

<!-- changelog-skip bfe68a8 maintainer-only doc/screenshot regeneration script, no app behaviour changed -->
<!-- changelog-skip f2c0a46 internal refactor: preview_screen.py split into a package, no behaviour change -->

- The app now has two tabs, not three. Run & results lives in a collapsible drawer under the
  shared file list ("FileRail") inside Preview & QC, and expands itself automatically whenever
  a run starts. Setup and Preview & QC share the same searchable, filterable file list,
  replacing the old dropdown on one screen and the separate results table on the other
- Every screen and action is reachable at all times now — the old "reveal one Setup card at a
  time" guided flow and the tab-locking that kept Preview & QC and Run out of reach until Setup
  validated are both gone. Only the action itself (Run, "Process & write this file", clicking a
  breath) is ever disabled, and the reason is always spelled out as a full sentence next to the
  control it blocks — on Setup's status strip and on Run's "commitment sheet" above the Run
  button, both reading from the same underlying check so they can never disagree
- Setup gained a real menu bar (File, View, Help) alongside its existing "Analysis" button —
  New/Open/Save/Save as/Get started/Explore with sample data/Duplicate for another recordings
  folder are available from both, and an offline About box documents the version, licence and
  citation without needing a browser
- Setup is now two columns — recording setup (input, channels) on the left, output and cohort
  summary on the right — instead of every card stacked full-width
- A batch of recordings now gets one consistent read: RespMech inspects the matched files up
  front and shows a single verdict for column count, sampling frequency and (for LabChart
  exports) header-preamble warnings, instead of Setup silently assuming the first file speaks
  for the whole folder while a separate, stricter check could quietly exclude other files
  without ever saying so. A file whose column count or detected sampling frequency disagrees
  with the rest of the batch is now named, not dropped without comment
- The channel setup dialog now suggests roles from a file's own column headers (flow, volume,
  oesophageal/gastric/transdiaphragmatic pressure, EMG) when nothing is saved yet, clearly
  marked "suggested" until you touch that row. European CSVs using a comma decimal separator
  are now detected and applied automatically before the channel picker ever reads the file,
  instead of showing scrambled values. And a file that turns out to be missing a required
  channel — a flow-only rig with no volume trace — no longer blocks the channel dialog from
  being confirmed at all; it explains what is optional and lets you proceed
- Breath exclusions, breath-count overrides and the EMG noise reference are now remembered per
  recordings folder, not just per filename — pointing the same analysis at a new subject's
  folder that happens to reuse a LabChart export name no longer silently carries the previous
  subject's choices over. Setup shows a keep-or-clear banner for carried-over settings, Preview
  shades a carried-over exclusion, and the file list marks it
- Drag-and-drop: the Recordings folder and Output folder fields on Setup accept a file or
  folder dragged from Finder/Explorer (with native path separators handled either way), the
  main window accepts a dropped `.toml`/`.py` analysis (behind the same unsaved-changes prompt
  as every other way of opening one), and a packaged build now opens RespMech directly when you
  double-click a `.toml` analysis file
- Opening a fresh analysis is friendlier: a startup chooser offers New analysis, Open, Explore
  with sample data and Get started, all available again later from the menu;
  "Duplicate for another recordings folder…" clones the current settings for a new subject in
  the same multi-subject study and offers to keep or clear the carried-over settings described
  above; and a spurious "folder does not exist yet" warning when writing into a fresh
  temp/sample output folder is gone
- The splash screen and the built-in sample analysis no longer make you wait for nothing —
  RespMech now warms its compute core in the background while the splash is still showing
  instead of after, and the sample analysis loads without an artificial delay
- Errors explain themselves. A LabChart export whose header preamble confuses the column count
  now says so in plain language instead of a raw traceback; an unreadable file in a batch no
  longer aborts the entire run — it is named and skipped, and the run continues with the rest;
  and error dialogs across Setup and Run lead with a plain-language diagnosis, when RespMech has
  one, with the full traceback tucked behind a "Details" toggle instead of shown by default.
  Validation messages throughout Setup and Run now name the actual on-screen control to fix,
  never an internal settings key
- Run & results no longer overstates what happened. The output folder button and its
  confirmation no longer claim files were written when a run partly or fully failed; a
  cancelled run's progress reflects the real state of the write phase instead of finishing as
  if nothing was interrupted; and the same output plan that Run's dry-run/commitment sheet
  shows before you click Run is now the exact plan the writer executes afterwards, so the
  preview can no longer promise something the run doesn't deliver. Re-running a subset of files
  (after fixing one) never rebuilds or overwrites the whole study's cohort summary output — only
  a full run does
- Preview & QC and Run now share one table implementation for both the per-breath table and the
  averaged results table, gained a single consistent vocabulary for the app's own parts across
  every panel and tooltip, and the status bar now has one clear owner per tab instead of the
  last-touched panel silently overwriting another panel's message; Run's own status line no
  longer sits blank at the very start or the very end of a run
- The Mechanics preview stack no longer freezes the interface while it redraws — rendering now
  happens in stages, so the busy indicator keeps animating and clicks are not queued up behind a
  frozen window — and on a short screen its channel stack now gives way instead of forcing a
  scroll, with the QC verdict band pinned outside the scrolled area so it always stays visible.
  The stack now labels its own channels, units and crosshair readout, and keeps a persistent
  label naming the current analysis window (start time, duration, trimmed length) on its time
  axis
- During a run, Preview & QC's own write actions ("Process & write this file", clicking a breath
  to exclude it) are now locked too, not just Setup and the Analysis menu — with a status-bar
  message confirming why, that reaches the visible bar even while a run's own progress message
  would otherwise take priority
- The "Advanced…" settings dialogs gained a third button, Apply, that commits your edits without
  closing the dialog; the Mechanics tab's Advanced dialog can now stay open (non-modal) while
  you watch its live breath count update against the plot behind it, and it no longer risks
  hanging open forever if the window it belongs to is closed or destroyed without going through
  its own OK/Cancel/✕
- On the EMG tabs: the noise-profile picker and the Detail band now show the actual signal a
  noise profile is built from instead of a stale one; the fidelity-frontier chart draws real
  data and explains what its metric means; the "Advanced…" button and dialog for EMG
  conditioning now lead with the RMS window, the setting that matters most; ECG Auto (whole
  batch) preview no longer shows numbers left over from a previous file; the ECG tab shows a
  suppression number and every panel title says plainly whether it is showing raw or processed
  data; and the two rest-span guards that used to disagree are now a single check, with the
  core's own warning surfaced instead of silently swallowed
- Small naming and labelling fixes: the work-of-breathing table names its source explicitly
  when every row repeats the same value, instead of leaving it to be inferred; the
  sample-entropy fields are labelled for what they measure; Setup now shows exactly where a
  sampling frequency came from (the file itself, or a value you typed) instead of a bare number;
  and breath-separation debounce is expressed in seconds, with a live readout, instead of an
  opaque buffer count
- "Group files by" (the cohort-summary grouping pattern) can now be checked on Setup before you
  run — it shows how a batch will actually be grouped, using the exact same logic and the exact
  same set of matched files the run itself will use, so the preview can no longer disagree with
  what actually gets written
- A further round of Windows-metrics layout fixes beyond v2.3.4: Setup's card rows no longer let
  a long label squeeze the field next to it, the collapsed Run & results summary row keeps its
  own compact margins instead of a full card's padding, the EMG rest-reference badge no longer
  overflows its panel under Windows' wider fonts, and several panels now reserve just enough
  width or height for Windows' font metrics instead of the narrower ones macOS happens to use

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
