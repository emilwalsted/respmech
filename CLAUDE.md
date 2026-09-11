# CLAUDE.md — RespMech

Project memory for **RespMech** — respiratory mechanics, work of breathing and
diaphragm-EMG analysis (respiratory mechanics, Campbell-diagram work of breathing,
diaphragm EMG with ECG removal + spectral noise reduction, sample entropy). Public
repo `emilwalsted/respmech` (GPL-3.0-or-later); the companion marketing site
`respmech.dk` lives in the separate, private `respmech-website` repo. Author: Emil
Ingerslev Walsted (ORCID [0000-0002-6640-7175](https://orcid.org/0000-0002-6640-7175));
each release gets its own Zenodo DOI under the concept DOI
10.5281/zenodo.3270825 (see `README.md`'s badge). See `README.md` and `docs/` for the
full picture; this file is the quick orientation and the rules that apply everywhere.

**Version status:** `src/respmech/__init__.py`'s `__version__` and `CHANGELOG.md` are
the source of truth for what is released vs. still unreleased on `master` — read those
directly rather than trusting a specific commit count or date written here, which goes
stale within days.

**The rest of this project's memory lives beside the code it applies to:**
`tests/CLAUDE.md` (test-writing hazards), `src/respmech/ui/CLAUDE.md` (Qt/GUI
gotchas) — both load automatically when you work with files there — and
`.claude/skills/release/SKILL.md` (releasing, signing, PyPI, respmech.dk), which
sits alongside `docs/RELEASING.md` and `docs/SIGNING.md`.

## Sources of truth

- **This file** for orientation and cross-cutting rules; `tests/CLAUDE.md` and
  `src/respmech/ui/CLAUDE.md` for test- and Qt-specific hazards.
- **`docs/`**: `REVERSE_ENGINEERING.md` (the formulas/units v2 must match),
  `PLAN.md` (target architecture), `RELEASING.md` + `SIGNING.md` (the release
  runbook), `DISTRIBUTION.md`, `INSTALL.md`, `NOISE_ECG_OPTIMIZATION.md`,
  `CARDIAC_GATED_PEAK_EMG.md`, `PTP_INVESTIGATION.md`, and `beslutninger.md`
  (dated decisions and "settled, don't reopen" points — read it before reopening
  anything that sounds already decided).
- **`CHANGELOG.md`** is the canonical release log.
- This file is a distillate, not a replacement for the above; file:line references in
  older notes go stale fast — verify against the actual code.

## Non-negotiable rules

- **Ask before any upload to production/a live site.** Deploys and release tags are
  only run with explicit authorization for that specific change.
- **Physiological correctness is sacrosanct.** Refactoring must never change a number
  silently. `tests/golden/` pins the engine's output byte-for-byte against the v1
  reference (`rtol=1e-9`); a golden diff is a bug in the change, not the reference,
  unless the change is a deliberate, explained numerical change with the golden
  re-baked in the same commit.
- **`legacy/` is the frozen v1 oracle — never touch it.** It exists solely to
  re-bake the golden reference; `src/respmech/` is a faithful port of it.
- **Three clean layers**: the computation core (`core/`), the CLI and the Qt GUI
  share one core and never import each other's concerns; plotting only consumes
  results.
- **A task is not done while its own CI is red.** A green local Linux run is
  necessary, not sufficient — the Windows/macOS jobs see real portability
  differences a Linux run structurally cannot.
- Two decision records worth reading before you touch numbers or exclusions:
  `docs/beslutninger.md` lists several number-changing ideas that were deliberately
  rejected — don't re-propose them without new evidence.

## Layout / correctness

- `legacy/` — the **frozen v1 monolith**; the v2 engine is a faithful port of it.
  It is the oracle the golden tests compare against — never delete it or clean it up.
- `tests/golden/` — characterisation tests that pin v2 output **byte-for-byte**
  against v1 references. `docs/REVERSE_ENGINEERING.md` = the formulas/units.

### A boundary sample's sign is not evidence of truncation — compare durations instead (K-035, 06-09-2026)

`core.compute.trim_boundary_notices` (the K-035 fix: warns when the boundary breath
`trim()` keeps looks truncated, instead of silently analysing it as whole) went
through two designs. The first compared `trim()`'s own `startix`/`endix` against the
raw array's edges (`startix == 0` / `endix == n - 1`) — mechanically correct as a
description of when `trim()` keeps nothing beyond a boundary, but it false-flagged
**both** the built-in sample recording and the committed golden synthetic inputs
(`tests/golden/input/synth_case_*.csv`): none of those are truncated, they simply end
(or begin) a hair's-breadth short of a full extra phase, which is indistinguishable
from real truncation at the single-sample level. The general lesson: a synthetically
generated or idealised recording routinely ends *exactly* at (or a discretisation
step short of) a phase boundary — real truncation and a clean, minimal-margin cut are
the same shape at the boundary sample alone. The fix compares the boundary breath's
own phase duration against the file's own median duration for that phase across the
OTHER detected breaths — self-calibrating, no assumption about breathing rate, and
insensitive to where exactly the file happens to end.

The chosen threshold (`min_relative_duration`, currently 0.8) is itself an empirical,
not a guessed, number — verified by replaying K-035's own reported reproduction
(0.5 s cut into the built-in sample's first inspiration / last expiration) through
the shipped function: ratios of ~0.72 and ~0.30 against the file's own median. A
threshold below ~0.75 does NOT catch the inspiratory case at all. The tightest known
*non-truncated* ratio measured (the built-in sample's own last breath, whose synthetic
generator varies each breath's period by design) is ~0.88. Any change to this
threshold, or to which recordings feed `separateintobreaths`, should re-measure both
ends of that range rather than adjust the number on feel — `tests/unit/
test_trim_boundary_notices.py::TestTrimBoundaryNoticesPure::
test_reproduces_k035_own_measured_case` pins the lower end.

**A self-calibrating statistic is not automatically "more robust" than a fixed ratio —
measure the ACTUAL trade-off before switching (ticket 20260906-1307, 06-09-2026).** A
follow-up review raised a literature-backed concern (published breath-timing CVs of
~18-25% in real resting breathing) that the 0.8 ratio above over-flags ordinary
high-variability recordings. The seemingly obvious fix — a MAD-based robust z-score
that adapts to each file's own measured spread instead of a fixed 80% — was built and
Monte-Carlo-compared against the ratio check at matching CVs, using this function's
own two known reference cases re-measured with REAL median/MAD (not assumed values):
it does cut false positives substantially, but because a real truncation (K-035's
reproduction: a fixed 0.5 s cut) shrinks relative to a file's OWN spread as that
spread grows, the MAD check's sensitivity to that exact same truncation falls even
faster than its false-positive rate does (~83% detection at CV 10% down to ~6-47% at
CV 25-30%, depending on the z-threshold) — i.e. it gets weakest exactly on the more
variable recordings where a human is least likely to catch a bad cut by eye. Rejected:
not a proven improvement, a different (and for an advisory, never-fails notice, worse)
trade-off. The full numbers and reasoning are pinned in `trim_boundary_notices`'s own
docstring. Decision instead: keep 0.8/3 as a documented, deliberate trade-off, and
expose both as `Settings.processing.segmentation.boundary_notice_min_relative_duration`
/ `boundary_notice_min_other_breaths` (wired through `_legacy_ns.py` as
`boundarynoticeminrelativeduration`/`boundarynoticeminotherbreaths`) so a study that
knows its own recordings are unusually variable can retune it deliberately, instead of
the whole install trading detection power away on one unverified system-wide guess.
General lesson: when a review proposes replacing a measured threshold with a
"smarter" adaptive statistic, simulate the ACTUAL detection-power trade-off (not just
the false-positive side) before adopting it — an adaptive statistic can easily trade
a nuisance failure mode (false positives) for a worse, silent one (missed detections)
precisely where the original problem mattered most.

### Two different screenshot tools — do not confuse them (found 10-08-2026)

`scripts/gen_readme_figures.py` is the **canonical generator for the 7 images in
`docs/img/`** that `README.md` embeds by name. Run it locally (`python scripts/gen_readme_figures.py`,
offscreen Qt, deterministic) whenever a UI or figure change should be reflected in the
README.

`tools/capture_screens.py` is a separate, broader tool (added for
`.github/workflows/screenshots.yml`, on-demand, real Windows/macOS runners).

Both scripts drive the same `MainWindow`, so both break the same way when the UI's shape
changes underneath them — fix BOTH if you change how a screen is reached, selected, or driven headlessly.

### CI showing red does not always mean a test failed (found 07-08-2026)

Two independent, unrelated defects made CI look permanently red on a branch fed by
rapid successive pushes (`ui-overhaul` under chained ticket dispatch), while every
individual test passed. Both are fixed, but the diagnostic habit is the lasting lesson:
if local reproduction of every CI step is 100% green, look at the *workflow
infrastructure* next, not just the tests.

If you add another tool that shells out to `git` and reads its output as text, give
the `subprocess.run` call an explicit `encoding='utf-8'` — never rely on the platform
default.

Regression tests: `tests/unit/test_check_changelog.py::test_a_non_utf8_default_locale_does_not_crash_the_tool`,
`tests/unit/test_ci_workflow_concurrency.py`.

### A ticket is not done while its own CI run is red (added 10-08-2026)

Every ticket session runs on **Linux**, and a green local `pytest tests/unit
tests/golden` there is necessary but **not sufficient**: the win/mac smoke fails on
real portability differences a Linux run structurally cannot see.

The protocol, after **every** push:

1. `gh run list --branch <branch> --limit 3` — the run for your HEAD sha appears
   within seconds of the push.
2. Watch it to a verdict: `gh run watch <run-id> --exit-status` (or poll
   `gh run view <run-id>`). **`GUI smoke · ubuntu-latest` (~15 min) is the same claim
   as your local suite and must be green. The Windows jobs (~35 min) must be green
   before the ticket reports success.** Read failures with
   `gh run view <run-id> --log-failed`.
3. macOS can queue for hours behind earlier runs — do not block the hand-off on it,
   but check the latest *completed* macOS smoke on the branch before starting new
   work, and treat an inherited red as yours to clear before building on top of it.
4. Layout or wording changes: model the Windows runner **before** pushing —
   `windows_metrics` fixture / `QFont.setStretch(145)` (see `tests/CLAUDE.md`). A
   pixel-marginal row that fits your DejaVu does not fit Segoe, and macOS adds
   button chrome DejaVu maths won't predict.
5. If `gh` is unavailable in the session, say so in the hand-off instead of implying
   green: "suite green locally; CI not checked" is honest and lets the next session
   check. Never report a ticket done while its run shows a failed job.

Related, and the reason a red run is worth re-running rather than shrugging off:
**A test that passes alone but fails in a big suite run is not automatically
flaky — run the same sequence against the OLD code first before writing it off.**

### Point 6 (suite scaling) — RESOLVED 11-08-2026: don't re-litigate

The pyqtgraph `QMenu` accumulation behind macOS CI's multi-hour wall is fixed
(`ui/plot_perf.py::close_plots`, `ColumnStack.close_plots()`, and a `closeEvent` on
each plot-owning widget), pinned by `tests/unit/test_plot_cleanup_contract.py`, and the
load-bearing close ordering is documented in `ui/plot_perf.py`'s own docstring.

See the project's internal change history for 11-08-2026 for the full investigation, every
measured number, and the review passes for both fixes.

**Investigated and RULED OUT (11-08-2026): mid-session `<container>.clear()` re-renders.**
**But it made ZERO measured difference on the representative metric**, confirmed on TWO
independent full-file `RESPMECH_NET_CENSUS` runs of `test_gui_interactive.py` (57 tests):
3,797 both with and without the fix, exact integer match, twice.

**Re-verified 11-08-2026, don't re-litigate a third time.** Re-ran the exact same
measurement against the current baseline: **276 both with and without, exact match,
57/57 tests unchanged.** If this is proposed again, point back here rather than
repeating the experiment.

## Dev environment — check which interpreter you are actually running

`respmech-gui` is a console script, and on a machine with more than one environment it may
**not** be the repo's `.venv`. On the maintainer's Mac it resolves to
`/opt/anaconda3/bin/respmech-gui`; `.venv/bin/respmech-gui` exists alongside it. Both are
*editable* installs of the same `src/respmech`, so the **code is identical** — but the Qt
version underneath is not, and GUI behaviour follows Qt.

```bash
which respmech-gui
/opt/anaconda3/bin/python3.13 -c "from PySide6.QtCore import qVersion; print(qVersion())"
.venv/bin/python              -c "from PySide6.QtCore import qVersion; print(qVersion())"
```

Before reproducing any GUI report, confirm you are on the interpreter the reporter used. A
repro in the wrong environment yields confident false negatives that look like eliminations.

### Known non-issue: `modalSession has been exited prematurely`

macOS/AppKit prints this on stderr under **Qt 6.11.0**; it is **silent on 6.11.1**. Verified by
a controlled A/B — same code, same flow, same session, only the interpreter swapped. It is an
upstream Qt bug fixed in the patch release, with no functional consequence, and **not** a
RespMech defect. Fix by running `.venv/bin/respmech-gui` or upgrading PySide6 in the other env.
(Packaged builds pin their own PySide6, so end users are unaffected.)

There is a separate, genuine instance of this pattern that *was* ours and is fixed:
`StartupDialog._choose_open` called `accept()` from inside the stack the native macOS open
panel returned into. Opening a native panel from within a Qt modal dialog nests two AppKit
modal sessions; ending them out of order is what produces the message. If you add a native
panel inside a modal, defer the `accept()`/`reject()` by one event-loop turn.

### Filename-keyed batch state needs a folder tag, or a re-pointed analysis silently reapplies stale decisions

Ticket B06 found and fixed a real data-integrity bug, not just a UI one: `exclude_breaths`,
`breath_counts` and the EMG noise reference (`core/settings.py`) all key on a bare
**filename**, with no idea which recordings folder that filename was chosen in.

**The fix, and the pattern to reuse for anything ELSE that keys on a bare filename in
future work:** give the entry an optional `folder` field (`ExcludeEntry.folder`,
`BreathCountEntry.folder`, `NoiseSettings.reference_folder`), stamped with the live
`settings.input.folder` wherever the entry is *created* (never on a mere edit of an
existing entry — see below), rebased/relativized in `settingsio/toml_io.py` exactly like
`input.folder`/`output.folder` already are (so a portable, relative-path analysis doesn't
falsely read as carried-over the moment it's reopened somewhere else). `core.settings`
gets ONE pure, Qt-free source of truth for "does this still match" —
`is_carried_folder(entry_folder, current_folder)` (an unrecorded/`None` folder on EITHER
side always counts as unproven, never guessed at; `os.path.normcase` + `os.path.normpath`,
matching `ui.prefs`'s existing recent-analyses dedup, so a same-folder path differing only
in case on Windows doesn't false-flag) — reused identically by `carried_over_state()`/
`clear_carried_over()`, the Setup banner, Preview's overlay hatching, the QC line and the
file rail's badge, so none of them can disagree about what counts as carried.

**The calculation core is deliberately blind to all of this.** `core.compute`/
`core.pipeline`/`core._legacy_ns` still key purely on filename — `folder` is dropped at
the `to_legacy_ns()` boundary (`excludebreaths=[[e.file, list(e.breaths)] for e in ...]`)
and never reaches a run. The UI/settings layer alone decides which entries are even IN the
list by the time a run starts (mutating `settings.processing.exclude_breaths` directly);
compute's own numeric behaviour for a given list is byte-identical to before this ticket
— any golden-test change here would be a bug in the change, not the reference.

**A subtlety worth knowing before touching this again:** the folder tag is ONE per
file-entry, not one per breath, because `ExcludeEntry.breaths` is a flat `list[int]`. Two
self-review rounds converged on the same rule from opposite directions: `_toggle_breath`
(`ui/screens/preview/_mechanics.py`) stamps `folder` **only when creating a brand-new
entry** — an existing entry's `folder` is never rewritten by a plain click, even one that
un-excludes one of ITS OWN breaths, because the entry can hold a MIX of a breath the user
just decided on and others still carried from a different folder that this click never
looked at. An earlier version restamped on every touch; that silently "confirmed" the
untouched breaths too, exactly the invisible application this ticket exists to stop, one
click later. The accepted, documented imprecision this leaves: a genuinely NEW breath
added to an already-carried entry still reads as carried until the whole entry is cleared.

**Wherever a write path can resolve carried-over state, refresh whatever is SHOWING it.**
Any FUTURE state this pattern is extended to needs the same treatment: know every path that can create OR resolve it, not
just the one this ticket happened to add a banner for.

## The app's shape (v2.x) — supersedes any older "three screens" description

- **Two tabs: Setup and Preview & QC** (sub-tabs: Mechanics, EMG – ECG reduction,
  EMG – noise reduction). **Run & results is not a tab**: it's a collapsed drawer
  ("Run & results ▸") under Preview & QC's file list, visible across sub-tabs, that
  expands when a run starts. **A single `&` in a Qt button label is a mnemonic
  marker**: always write `&&` ("Preview && QC", "Run && results"), including in
  handlers that toggle ▸/▾ — a guard test rejects a lone `&` with no alphanumeric
  follower.
- **File selection via a searchable `FileRail`** (one shared instance between
  Preview and Run). Each row shows a ✓/✗/• verdict, `[N excl]`, manifest ⚠
  caveats and a ↺ for inherited folder settings; filterable and sortable
  failed-first. There is no per-file results table separate from this.
- **A File/View/Help menu bar** with keyboard shortcuts and an offline About box.
  File shares New/Open/Save/Save as as the SAME `QAction` objects as the header's
  Analysis button, plus Get started…, Explore with sample data and **Duplicate for
  another recordings folder…** (the multi-cohort pattern). View is built
  dynamically, one entry per tab. The startup dialog offers New/Open/Explore; the
  recent-analyses list shows the folder next to the name — so **take screenshots
  of it with an empty recent-analyses list**, or a local dataset folder name ends
  up in a published image.
- **Gating is inverted: every surface is always reachable.** It is the ACTION that
  gets disabled (the Run button, "Process && write this file", clicking a breath),
  never the tab, with the reason spelled out as a full sentence at the action
  itself. The Run drawer's **commitment sheet** (files read, what will be written,
  blockers in fix-order) is the ONLY gating surface. During a run, Preview & QC's
  write actions lock too, while graphs/zoom/file navigation stay live. No tab
  locking, no progressive-disclosure flow.
- Setup is two columns (Input+Channels left, Output+Cohort summary right). The
  channel dialog suggests roles from column names ("suggested"), detects
  comma-decimals, and never blocks on a missing optional channel; removing the
  EMG role clears whatever auto-detection depended on it, without asking.
- **Four ways to open an analysis:** the Open dialog, drag-and-drop onto the path
  fields, dropping a `.toml`/`.py` onto the window, double-clicking a `.toml` in a
  packaged build. `open_analysis(path)` does NOT itself guard against discarding
  unsaved changes — the caller runs `confirm_discard_changes` first.
  `QUrl.toLocalFile()` paths are normalised with `os.path.normpath` at the drop
  boundary (Windows separators).
- **`exclude_breaths`/`breath_counts`/the noise reference are folder-tracked** (a
  `folder` field; a Keep/Clear banner; hatched inherited exclusions). The
  calculation core stays deliberately blind to this (keyed on filename;
  `to_legacy_ns()` drops `folder`). The folder is stamped ONLY when a new entry is
  created, never on a mere edit; any write path that can resolve carried-over
  state must also refresh whatever is showing it (see above).
- **A subset re-run NEVER rebuilds the cohort summary** — computing and committing
  a cohort summary are deliberately split operations.

## The single source of truth for each concern

- `ui/manifest.py`: what a folder/batch contains (column counts, sampling rate,
  LabChart warnings, `is_clean`). `group_readout()` predicts the grouping from
  `core.summary.group_key` — always call it with the FULL matched file set.
- `ui/file_rail.py`: which file is selected, and what the app knows about it.
- `ui/validation.py::blockers()`: why a run can't start (collision → validation →
  path); drives BOTH Setup's QC strip and the commitment sheet.
  `friendly_settings_error()` turns a `Settings.validate()` exception into a
  sentence naming the UI control — any new `validate()` check needs an entry in
  `_FRIENDLY_SETTINGS_ERRORS`/`_FRIENDLY_PREFIXES`.
- `core/io/plan.py::plan_outputs()`: the one place that knows what a run WILL
  write (`is_cap=True` = a ceiling, never a promise). `write_planned()` rewrites a
  computed result to another folder without recomputing.
- `core/settings.py::carried_over_state()`/`is_carried_folder()`: folder-tracked
  settings (`None` on either side always means unproven, never guessed;
  `normcase`+`normpath`).
- Errors with a known diagnosis go through `TextViewerDialog`'s `collapsed_detail`
  plus a DEDICATED exception type, never a bare `ValueError`; in
  `_build_noise_set`/`_reference_noise_clip` the exception TYPE is preserved on
  re-raise (`stage_noise_fidelity` catches by type).
- Point 6 (suite scaling) is closed — see "Point 6 (suite scaling)" above; don't
  reopen the `<container>.clear()` hypothesis without a fresh `RESPMECH_NET_CENSUS`
  measurement.

## App layout, tests and CI

- `src/respmech/`: `core/` (computation + IO), `ui/` (GUI), `cli/`, `settingsio/`
  (TOML + v1-migration). Settings are declarative **TOML**; `respmech migrate
  old.py -o new.toml` converts an old-style settings file with a migration report.
- **Tests are the gate:** `pytest tests/unit` + `tests/golden` green. CI
  (`ci.yml`) runs **smoke-linux** on every push (~15 min; installs Qt libs from
  the runner image's own index and only refreshes when needed, each apt call
  wrapped in a `timeout`); the full 2×2 (OS × Python) matrix runs only on
  master/PR/dispatch. Every job has `timeout-minutes`; concurrency is keyed per
  sha (push) or per PR number.

## Release (one tag triggers everything)

Pushing `vX.Y.Z` (must point at `master`) triggers both `release.yml` (signed dmg +
MSI, GitHub release marked Latest) and `publish-pypi.yml` (PyPI via Trusted
Publishing/OIDC, no tokens, behind a test gate that runs the whole suite on a
release runner with a wide, unmeasured timeout margin — treat it as noticeably
slower than `ci.yml`, and watch for race-sensitive asserts failing there first).
Full runbook: `docs/RELEASING.md`
and `docs/SIGNING.md`; the release skill (`.claude/skills/release/SKILL.md`) walks
through it step by step. In short: add the `CHANGELOG.md` entry (folding in
`## Unreleased` if one exists), bump `__version__` in one place
(`src/respmech/__init__.py`, keep `pyproject.toml`'s briefcase version in sync),
verify coverage with `tools/check_changelog.py`, and only ever use plain semver
(`vX.Y.Z`, never `-rc`/`-beta` — a PyPI version is immutable, so a broken release
is fixed with a patch bump and a new tag, never a re-push of the same tag).
Windows MSI signing is a manual, local, post-release step
(`scripts/sign-msi-certum.sh <tag>`) — see `docs/SIGNING.md`.

## App ↔ website coupling

- **A release notifies the website.** The last step of the release workflow sends
  a `repository_dispatch` to the website repo, which bumps its version and
  redeploys; if that hook is ever unavailable, the website's own daily poll picks
  up the release within a day regardless. Keep the notify step non-fatal and
  strictly after `gh release create`.
- **The changelog is mirrored, not duplicated by hand:** the website's changelog
  page renders `CHANGELOG.md` trimmed to what a user of the app cares about, and
  its own hand-written "Coming next" section is promoted to `vX.Y.Z` automatically
  at release time — keep it current, in the site's own voice, while merging.
- **The release e-mail is never sent automatically by anything on the app side.**
  Only the website's own announce workflow mails subscribers, and only when
  triggered separately.

## Physiology / algorithm anchors (from `docs/`)

- **EMG is quantified relative to a patient+test-specific maximum**, so every file
  in one test undergoes the IDENTICAL transformation. The optimisation target is
  the largest SNR improvement with fidelity ≥ 0.8 (retain ≥ 80% of inspiratory EMG
  power). The noise profile is built once per test from an expiration-based rest
  reference.
- **Cardiac-gated peak EMG** (`processing.emg.robust_peak`, opt-in, default off):
  blanks the RMS envelope around R-peaks. Adds columns only; the golden passes
  with the feature both on and off; gated columns go NaN on unreliable R-peak
  detection.
- **PTP baseline: the principle from `PTP_INVESTIGATION.md` stands** — one
  end-expiratory baseline, never a double subtraction. v2 uses the **mean over a
  50 ms window** (`processing.ptp.baseline_window_s`, default 0.05), not the
  noise-sensitive single-sample `pressure[0]`. Don't reopen this.
- **v1↔v2 deviations are catalogued** in the reverse-engineering documentation:
  the PTP window (deliberate), a legacy entropy-trim bug (v1 was wrong), and a
  scipy-Simpson drift.
- Sample-entropy fields are named "Template length (m + 1)" and "Tolerance (r), ×
  SD", with provenance recorded. Ground truth is the input run through the
  newest algorithm, never an old spreadsheet.
