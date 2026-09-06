# CLAUDE.md — RespMech

Project memory for **RespMech** — respiratory mechanics, work of breathing and
diaphragm-EMG analysis. Public repo `emilwalsted/respmech` (GPL-3.0-or-later).
Author: Emil Ingerslev Walsted. See `README.md` and `docs/` for the full picture;
this file is the quick orientation and the rules that apply everywhere.

**The rest of this project's memory lives beside the code it applies to:**
`tests/CLAUDE.md` (test-writing hazards), `src/respmech/ui/CLAUDE.md` (Qt/GUI
gotchas) — both load automatically when you work with files there — and
`.claude/skills/release/SKILL.md` (releasing, signing, PyPI, respmech.dk), which
sits alongside `docs/RELEASING.md` and `docs/SIGNING.md`.

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

See ticket `20260811-0910-ci-tests.md` (claude-ops) for the full investigation, every
measured number, and the review passes for both fixes.

**Investigated and RULED OUT (11-08-2026): mid-session `<container>.clear()` re-renders.**
**But it made ZERO measured difference on the representative metric**, confirmed on TWO
independent full-file `RESPMECH_NET_CENSUS` runs of `test_gui_interactive.py` (57 tests):
3,797 both with and without the fix, exact integer match, twice.

**Re-verified 11-08-2026 (ticket `20260811-1232-flere-ci-fixes.md`, claude-ops), don't
re-litigate a third time.** Re-ran the exact same measurement against the current
baseline: **276 both with and without, exact match, 57/57 tests unchanged.** If a future
ticket proposes this again, point it here rather than repeating the experiment.

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
