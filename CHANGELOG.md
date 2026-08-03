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

## v2.3.4 — 2026-08-03

An interface release: every window now fits every screen, both themes paint correctly on
Windows, and the Preview screens give their space back to the graphs. Nothing in the
analysis changed — a settings file run before and after this release produces identical
output.

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
