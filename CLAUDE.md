# CLAUDE.md — RespMech

Project memory for **RespMech** — respiratory mechanics, work of breathing and
diaphragm-EMG analysis. Public repo `emilwalsted/respmech` (GPL-3.0-or-later).
Author: Emil Ingerslev Walsted. See `README.md` and `docs/` for the full picture;
this file is the quick orientation plus the release/website setup.

## What it is

Analyses time-series respiratory recordings (LabChart/CSV/Excel/MATLAB) **breath
by breath** and computes: respiratory mechanics (timing, VT, VE, oesophageal/
gastric/transdiaphragmatic pressures, PTP), **work of breathing** (Campbell
diagram, J and J·min⁻¹), **diaphragm EMG** (RMS + integrated, optional ECG
removal + spectral noise reduction) and **sample entropy**. v2 is a **PySide6
desktop app** (`respmech-gui`) with two tabs, **Setup** and **Preview & QC**
(a `File`/`View`/`Help` menu bar sits above them), plus a **Run & results**
drawer folded under Preview & QC's file rail (no longer a third tab), and a
CLI (`respmech run/validate/migrate`). Settings are declarative **TOML**.

## Layout / correctness

- `src/respmech/` — the v2 package (`core/` compute+IO, `ui/` GUI, `cli/`,
  `settingsio/` TOML + v1 migration). `pyproject.toml`: version, extras
  (`gui`/`emg`/`plots`/`dev`/`packaging`), briefcase config.
- `legacy/` — the **frozen v1 monolith**; the v2 engine is a faithful port of it.
- `tests/golden/` — characterisation tests that pin v2 output **byte-for-byte**
  against v1 references. `docs/REVERSE_ENGINEERING.md` = the formulas/units.
- CI: `.github/workflows/ci.yml` — GUI smoke on win/mac (full 2×2 matrix on
  master/PRs/dispatch, one Python per OS on feature-branch pushes, to keep the
  macOS runner queue drainable), a fast `smoke-linux` job on ubuntu (the exact
  claim a Linux ticket sandbox can make locally, ~15 min), and the numerical
  golden on ubuntu. Every job carries `timeout-minutes` — a hung Qt test used to
  burn a runner for the 6 h default. Runs on every branch.

### Two different screenshot tools — do not confuse them (found 10-08-2026)

`scripts/gen_readme_figures.py` is the **canonical generator for the 7 images in
`docs/img/`** that `README.md` embeds by name (`setup.png`, `preview-mechanics.png`,
`run.png`, `campbell.png`, `drift.png`, `emg-stages.png`, `breath-exclusion.png`): the
three UI screenshots plus four matplotlib feature figures drawn straight from the core
diagnostic plot writers against the app's own onboarding sample, so they can never drift
from what the app actually computes. Run it locally (`python scripts/gen_readme_figures.py`,
offscreen Qt, deterministic) whenever a UI or figure change should be reflected in the
README.

`tools/capture_screens.py` is a separate, broader tool (added for
`.github/workflows/screenshots.yml`, on-demand, real Windows/macOS runners): it captures
the startup dialog, both tabs (Preview & QC's 3 subtabs each get their own shot), the Run
& results drawer and 5 more dialogs — 11 shots in both themes — meant for a wider
documentation/QA sweep, not for producing the README's own curated seven. It has no
matplotlib feature figures and writes different filenames (`01_setup.png` etc.).

Both scripts drive the same `MainWindow`, so both break the same way when the UI's shape
changes underneath them (found via `gen_readme_figures.py`'s `_screenshots()` crashing on
a `file_combo` attribute removed by ticket B02, and a `win.tabs.setCurrentIndex(2)` that
silently no-ops since ticket B03 folded Run & results into a drawer instead of a third
tab) — fix BOTH if you change how a screen is reached, selected, or driven headlessly.

### CI showing red does not always mean a test failed (found 07-08-2026)

Two independent, unrelated defects made CI look permanently red on a branch fed by
rapid successive pushes (`ui-overhaul` under chained ticket dispatch), while every
individual test passed. Both are fixed, but the diagnostic habit is the lasting lesson:
if local reproduction of every CI step is 100% green, look at the *workflow
infrastructure* next, not just the tests.

1. **`tools/check_changelog.py` crashed on Windows' cp1252 default locale.** Its
   `kør()` helper called `subprocess.run(..., text=True)` with no explicit `encoding=`,
   so it decoded git's (always-UTF-8) output using the process's locale-preferred
   encoding — UTF-8 on Linux/macOS, but **cp1252 on Windows**. Any commit whose diff or
   subject contained one of the typographic characters this project's history is full
   of (`·`, `–`, `—`, `›`, `→`) crashed with an uncaught `UnicodeDecodeError`, exit code
   1 — despite the CI step being commented "informational... must not fail a branch"
   with no `continue-on-error`. Measured: 4 of 35 commits in one range already crashed
   cp1252 decoding of their own diff. Fixed by decoding with explicit
   `encoding='utf-8', errors='replace'`; also added `continue-on-error: true` to the
   step as a second layer, matching what its own comment already promised. If you add
   another tool that shells out to `git` and reads its output as text, give the
   `subprocess.run` call an explicit `encoding='utf-8'` — never rely on the platform
   default.
2. **The concurrency group was keyed by `github.ref` alone**, shared by every push to
   the same branch, so a later push could cancel an earlier, unrelated commit's
   still-running check (4-way Windows/macOS matrix + full unit suite) before it
   reported anything — a real but secondary contributor, since ticket-driven commits
   have landed as close as 20-44 minutes apart. Fixed by keying push-triggered runs on
   `github.sha` (unique per commit) instead, while PR-triggered runs still key by PR
   number (where cancelling a stale review of an old head is the intended behaviour):
   `group: ci-${{ github.event.pull_request.number || github.sha }}`.

Both fixes are on `master` and were merged forward into `ui-overhaul`. Regression
tests: `tests/unit/test_check_changelog.py::test_a_non_utf8_default_locale_does_not_crash_the_tool`,
`tests/unit/test_ci_workflow_concurrency.py`.

### A ticket is not done while its own CI run is red (added 10-08-2026)

Every ticket session runs on **Linux**, and a green local `pytest tests/unit
tests/golden` there is necessary but **not sufficient**: the win/mac smoke fails on
real portability differences a Linux run structurally cannot see. Measured
05→10-08-2026 on `ui-overhaul`: ~45 consecutive pushes failed the win/mac GUI smoke,
every pushing ticket believed it was green because its local suite was, nobody looked
back at CI after pushing, and the maintainer's inbox took a failure mail per push for
five days. The reds were all of exactly the classes this file already warns about:
seven `QUrl.toLocalFile()` paths compared against native separators (`'C:/…' !=
'C:\\…'` — fixed 10-08 with `os.path.normpath` at the drop boundary in
`ui/path_drop.py` / `MainWindow._dropped_analysis_path`; any future URL→path
conversion needs the same), one assertion on an ElidingLabel's rendered `text()`
(the elide rule above, again — assert `fullText()`/`toolTip()`), and six
Windows-metrics layout budgets (five still open when this was written).

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
   `windows_metrics` fixture / `QFont.setStretch(145)` (metrics section above). A
   pixel-marginal row that fits your DejaVu does not fit Segoe, and macOS adds
   button chrome DejaVu maths won't predict.
5. If `gh` is unavailable in the session, say so in the hand-off instead of implying
   green: "suite green locally; CI not checked" is honest and lets the next session
   check. Never report a ticket done while its run shows a failed job.

### Point 6 (suite scaling) — RESOLVED 11-08-2026: 6,137 → 276 surviving QMenus (95.5%), and the 276 are legitimate

`tests/unit/conftest.py` never deletes a closed `MainWindow` — deleting one segfaults
nondeterministically on Python 3.11 (see the reaper comment in
`_close_top_level_windows`), so every pyqtgraph `PlotItem`/`ViewBox` a window ever built
kept its own auto-constructed context menu alive for the life of the test session:
`PlotItem.__init__` unconditionally builds one `ctrlMenu` + six submenus, `ViewBox.__init__`
one `ViewBoxMenu`, both EAGERLY at construction — measured (`RESPMECH_NET_CENSUS`) at
6,137 surviving `QMenu`s after just 57 GUI-heavy tests (`test_gui_interactive.py`), the
root of macOS CI's multi-hour wall and the sandbox's mid-suite OOM.

**Fix 1 (10-08/11-08-2026, `plot_perf.close_plots`):** `PreviewScreen.shutdown()` (called
from `MainWindow.closeEvent`) closes every `PlotItem` its SIX known containers (`plots`,
`ecg_capture_plot`, `ecg_processed_plots`, `emg_raw_plots`, `emg_result_plots`,
`emg_plots`) CURRENTLY hold, via pyqtgraph's own documented cleanup —
`ViewBox.setMenuEnabled(False)` then `PlotItem.close()` per item, then the container's own
`.close()` — see `ui/plot_perf.py::close_plots()`. A fundamentally different, much
lower-risk shape than an earlier "reap already-closed windows from outside" attempt that
segfaulted (`deleteLater()` + `sendPostedEvents(DeferredDelete)` swept over
`app.topLevelWidgets()`, never committed, only its diagnostics did — commit 9f2339c):
closing each plot's OWN internals at the moment ITS OWNER shuts down touches only live,
fully-valid objects. Reduced the measured count 6,137 → 3,797 (~38%).

**Order matters and is load-bearing, not stylistic:** `setMenuEnabled(False)` must run
BEFORE `PlotItem.close()` (which drops the item's reference to its view box), and both
before the container's `.close()` — `GraphicsView.close()` calls `scene().clear()`,
which invalidates any `PlotItem`/`ViewBox` still attached; closing the container first
raises `RuntimeError: Signal source has been deleted` on the now-dead C++ objects. The
SAME ordering constraint means a closed `PlotWidget`'s own `getPlotItem()`/`.plotItem`
reads back `None` afterwards (`scene().clear()` deletes the PlotItem's C++ object outright
— it was still IN the scene, only its view box had been explicitly removed) — tests that
assert on a plot's state after closing it must capture the `PlotItem` reference BEFORE
calling `close_plots()`, never re-fetch it after (see the `close_plots()`-adjacent tests
in `test_column_stack.py`/`test_channel_summary.py`/`test_channel_setup.py`/
`test_preview_screen.py` for the pattern).

**Investigated and RULED OUT (11-08-2026): mid-session `<container>.clear()` re-renders.**
The obvious next hypothesis — roughly a dozen call sites re-render a panel mid-session by
calling `<container>.clear()` then rebuilding with fresh `addPlot()` calls (`screen.py`
×2 blocks, `_mechanics.py` ×2, `_ecg.py` ×1 on the three `GraphicsLayoutWidget`
containers; the matching calls on the three bare-`PlotWidget` containers do NOT apply —
`PlotWidget.clear()` reaches `PlotItem.clear()`, which empties the SAME still-alive
PlotItem's curves and never orphans its menus at all) — was built, and a targeted fix
(`plot_perf.close_layout_items()`, closing each item WITHOUT closing the container, called
immediately before each `.clear()`) was verified to work in an isolated single-window
measurement (204 → 149 QMenus immediately, 138 → 83 after the eventual window close).
**But it made ZERO measured difference on the representative metric**, confirmed on TWO
independent full-file `RESPMECH_NET_CENSUS` runs of `test_gui_interactive.py` (57 tests):
3,797 both with and without the fix, exact integer match, twice. Root cause: unlike the
ColumnStack case below, a `.clear()`-discarded `PlotItem` really does become ordinary
Python garbage (nothing else references it once dropped from `container.ci.items`) —
`gc.collect()`, called explicitly both by `RESPMECH_NET_CENSUS`'s own census function AND
by `tests/unit/conftest.py`'s `_close_top_level_windows` fixture after EVERY test, already
reclaims it, making an additional explicit close redundant on this metric. The code was
written, isolated-verified, then REVERTED once the full-suite measurement showed no
effect — see ticket `20260811-0910-ci-tests.md` (claude-ops) for the four-experiment
trail (a naive isolated loop that showed no leak at all, a `close_layout_items()`-only
test that DID show -55, a combined `close+clear` test matching it, then the full-file
test showing the "fix" made no difference once `run_batch()`/realistic session length was
in the mix) that took the hypothesis from "obviously the dominant source" to "measurably
irrelevant." A genuinely unbounded PRODUCTION concern this rules out only for the TEST
metric, not necessarily for an hours-long real session with no `gc.collect()` ever called
explicitly — Python's own generational collector still runs on allocation thresholds, so
no permanent leak is expected there either, but this was not separately measured.

**Fix 2 (11-08-2026, the actual dominant source): `ColumnStack.close_plots()`.**
`column_stack.py`'s `ColumnStack` widget (used by `channel_setup_dialog.py`'s modal AND
`channel_summary.py`'s Setup-screen widget) builds its own `PlotWidget`s and already
called `plot.setMenuEnabled(False)` — which drops the `ViewBoxMenu` but NOT the
7-`QMenu` `ctrlMenu` `PlotItem.__init__` builds unconditionally regardless. Unlike the
ruled-out hypothesis above, THIS one is not ordinary garbage: `SettingsScreen`'s
`ChannelSummary` builds a REAL `ColumnStack` eagerly at `MainWindow` construction (Setup
renders the initial channel mapping on load) and never rebuilds it unless the mapping
changes, so the instance alive when the window closes stays REACHABLE the whole time via
`settings_screen.channel_summary.stack.plots` — never garbage at all, so no amount of
`gc.collect()` was ever going to free it. Measured directly: a bare, freshly constructed
`MainWindow` that renders nothing else (no Preview interaction at all) still left 83
surviving `QMenu`s after `close()` before this fix, 6 after — and those 6 are the
window's OWN File/Edit/Help menu-bar menus (`46 windows × 6 = 276`, the exact number the
full `test_gui_interactive.py` census confirms), unrelated to pyqtgraph and not a leak.
The fix: `ColumnStack.close_plots()` (closes every embedded `PlotWidget` via
`plot_perf.close_plots()`, empties `self.plots`), called from three places —
`ChannelSummary._clear()` (via its own `ChannelSummary.close_plots()`, so a mapping
rebuild releases the OUTGOING stack before replacing it), `MainWindow.closeEvent` (the
LAST-built stack, alongside the existing `preview_screen`/`run_screen` shutdown loop —
this is the one that mattered for the measured count), and `ChannelSetupDialog`'s
`finished` signal (covers accept/reject/window-close alike, for a caller that keeps a
dialog reference alive rather than discarding it after `exec()`).

**Combined result, full-file measurement (`RESPMECH_NET_CENSUS`,
`test_gui_interactive.py`, 57 tests, unchanged pass count):** 6,137 → 3,797 (fix 1) →
**276** (fix 2) — a 95.5% reduction from the original baseline, and the remaining 276 are
the 46 closed-but-undeleted windows' own legitimate menu bars, not pyqtgraph accumulation.
Point 6 is considered CLOSED: no further `QMenu` leak source is known. If the suite's
macOS wall time or sandbox OOM recur, re-measure with `RESPMECH_NET_CENSUS`/
`RESPMECH_NET_PROFILE` before assuming this is the same class of bug — the population
this ticket targeted is gone.

See ticket `20260811-0910-ci-tests.md` (claude-ops) for the full investigation, every
measured number, and the review passes for both fixes.

**Re-verified 11-08-2026 (ticket `20260811-1232-flere-ci-fixes.md`, claude-ops), don't
re-litigate a third time.** A follow-up ticket asked for exactly the ruled-out
mid-session-`.clear()` fix above, unaware Point 6 had already been closed by a
different mechanism. Rebuilt it (a `plot_perf.release_plot_items()` helper, called
immediately before all 9 relevant `GraphicsLayoutWidget.clear()` sites — the same set
listed in the "RULED OUT" paragraph above) and re-ran the exact same
`RESPMECH_NET_CENSUS` measurement against the CURRENT 276 baseline (not the old 3,797
one the original ruling-out used): **276 both with and without, exact match, 57/57
tests unchanged.** Confirms the ruling-out was not an artefact of the pre-`ColumnStack`
-fix baseline. The experimental code was discarded (not committed) for the same reason
the original attempt was: no measured benefit, and keeping it would be complexity
without justification. If a future ticket proposes this again, point it here rather
than repeating the experiment.

**Made durable 11-08-2026 (ticket `20260811-1735-test-durability-og-cleanup.md`,
claude-ops): cleanup moved from ONE orchestrator to each plot-owning widget itself.**
Both fixes above only ran when `MainWindow.closeEvent` orchestrated them — `PreviewScreen`,
`RunScreen` and `ColumnStack` had no `closeEvent` of their own, so any composition without
a `MainWindow` bypassed cleanup entirely. Real, not hypothetical: 15 tests construct
`PreviewScreen` standalone, 6 `RunScreen`, 17 `ColumnStack`. Qt never delivers `closeEvent`
to a child widget when its PARENT window closes (that is exactly why the orchestration
exists — verified empirically, not just from general Qt knowledge, in the review below), so
each now has its own `closeEvent` that calls its EXISTING cleanup function —
`PreviewScreen.closeEvent` → `self.shutdown()`, `RunScreen.closeEvent` → `self.shutdown()`,
`ColumnStack.closeEvent` → `self.close_plots()` — then `super().closeEvent(ev)`. This only
changes behaviour for widgets closed WITHOUT a `MainWindow` around them: an embedded child
still gets cleaned up exclusively via `MainWindow.closeEvent`'s explicit orchestration, so
the `test_gui_interactive.py` 276-QMenu baseline is unaffected (re-measured: 276, 57/57
unchanged).

**Idempotence needed one small, targeted fix to be genuinely true, not just masked.**
`RunScreen.shutdown()` and `ColumnStack.close_plots()` really do empty their own
bookkeeping on first call, so a second call is a true no-op. `PreviewScreen.shutdown()`
does NOT — its six plot containers (`self.plots`, `ecg_capture_plot`,
`ecg_processed_plots`, `emg_raw_plots`, `emg_result_plots`, `emg_plots`) stay set to the
same already-closed widget forever. Independent review found that a second call
therefore re-entered `plot_perf.close_plots()` on three of those (the bare
`pg.PlotWidget()`-typed ones — `ecg_capture_plot`/`emg_result_plots`/`emg_plots`;
`GraphicsLayoutWidget`-typed ones are unaffected) and raised `AttributeError` on
`None.close()` three times, since pyqtgraph 0.14's own `PlotWidget.close()` unconditionally
runs `self.plotItem.close(); self.plotItem = None` with no re-entry guard. It never
crashed — `plot_perf.close_plots()`'s own blanket `except Exception: pass` around
`container.close()` already caught it — but the mask was doing load-bearing work the
docstrings didn't admit to. Fixed at the source: `plot_perf.close_plots()` now checks
`getattr(container, "plotItem", "sentinel") is None` and skips the redundant `close()`
call outright for an already-closed bare `PlotWidget`, verified directly (instrumented a
double `PreviewScreen.shutdown()` to intercept every `container.close()` call — zero
exceptions reach it now, versus three before). This also reduces (does not eliminate —
see below) the wasted-work cost `conftest.py`'s per-test sweep pays for these.

**A closed bare `PlotWidget` becomes its own new top-level widget — pre-existing, not
introduced here, but now reachable through more call sites.** pyqtgraph's
`PlotWidget.close()` calls `self.setParent(None)`, so it detaches from whatever container
held it. This is NOT new: the `test_gui_interactive.py` baseline above already carries 138
such orphaned `PlotWidget`s (visible in a `RESPMECH_NET_CENSUS` type breakdown) from the
MainWindow-orchestrated path, and Point 6 was closed on the `QMenu` metric, which these do
not affect (confirmed: 0 surviving `QMenu`s from any of the standalone paths this ticket
covers). What durability adds is a modest number more of the same already-accepted kind,
now also from the 15+6+17 previously-untouched standalone construction sites — not a new
class of leak, and out of scope for this ticket to redesign (`plot_perf.close_plots()`'s
use of `container.close()` is the existing, shipped Fix 1/Fix 2 behaviour).

**`RunScreen.shutdown()` now also stops its own write-phase heartbeat FIRST**, the same
defensive pattern `PreviewScreen.shutdown()` already used for its debounce timer — found in
review: closing a `RunScreen` without a `MainWindow` around it (newly possible via
`closeEvent`) could otherwise leave a live 1 s `QTimer` ticking against an instance that is
closed but, per this suite's own policy, never deleted. Left deliberately UNFIXED (documented,
not silently accepted): the `WriteWorker.finished`/`failed` `Qt.QueuedConnection`s
(`run_screen.py`, `_write_elsewhere`) stay wired after `shutdown()` — a genuinely
still-running orphaned write thread could still deliver a queued signal to a closed
`RunScreen` afterwards. Bounded risk (no crash — the closed screen is still a valid Python/Qt
object, so at worst a pointless UI update on a widget nobody can see), not reachable in
production today (RunScreen is only ever closed via `MainWindow` there), and the fix would
need to touch signal-disconnection logic broader than this ticket's scope to verify safely.

The invariant is now also a NAMED regression test, not just a memory:
`tests/unit/test_plot_cleanup_contract.py` — one guard per type
(`PreviewScreen`/`RunScreen`/`ColumnStack`/`MainWindow`), each constructing the widget
standalone with a representative small render (a real mechanics render for `PreviewScreen`,
a real small synthetic write for `RunScreen`, a real `build()` for `ColumnStack`), closing
it, and asserting the top-level `QMenu` census does not GROW past its pre-construction
baseline (`<=`, not `==` — `gc.collect()` reclaiming an unrelated earlier test's menus could
legitimately push the count below `before`, and exact equality would make that an
order-dependent flake) — except `MainWindow`'s guard, which asserts a small ceiling (15)
above the legitimate 6-menu baseline (its own File/Edit/Help bar, independently re-measured
here at delta 6, stable across three cycles) rather than 0, since a `MainWindow` is closed
but deliberately never deleted in this suite. `RunScreen`'s guard checks bookkeeping reset
(`_write_thread` back to `None`) after EXPLICITLY pumping the write to completion first
(`_pump_until_write_done`, adapted from `test_run_screen.py`'s own
`_pump_until_thread_done`), not a plain blocking `wait()` and not racing `shutdown()`'s own
5000 ms budget against it. Two things were measured directly building this test: a plain
`QThread.wait()` on the calling thread never observed the write finish AT ALL — the
worker's cross-thread queued `finished` signal needs the caller's own Qt event loop
actually spinning to be delivered, which is exactly why that pump helper exists in the
first place — and closing immediately after starting the write let `shutdown()`'s own
5000 ms budget time out and park a REAL, still-running background thread (and its spawned
child process) in `_ORPHANED_THREADS` for the rest of the session. That parking is
CORRECT, INTENTIONAL production behaviour (a slow write must never block the app from
closing) — a unit test asserting on it must simply not gamble on winning that race or
leave a live thread running behind it. `RunScreen` owns no `PlotWidget`/`GraphicsLayoutWidget` at all (`grep`
confirms), so its point 6 exposure was always this thread hazard, never `QMenu`
accumulation.

**Two independent reviews (self-review, point 6 durability, 11-08-2026), both empirically
verifying rather than assuming Qt behaviour:** confirmed `closeEvent` is genuinely never
delivered to a child on a parent's close (tested `QStackedWidget`, `QDialog`
`close()`/`done()`, `QApplication.closeAllWindows()` — all zero deliveries) and that
stubbing out any one of the three new overrides makes 3 of the 4 new regression tests fail
(not vacuous). py3.11 stress-verified beyond the named suite (offscreen, this sandbox): 300
`ColumnStack` and 60 `PreviewScreen` construct/render/close/close cycles from one review
pass (zero errors), independently repeated by the other with 100 `PreviewScreen` cycles
including mid-session FILE SWITCHES (the historically segfault-prone pattern from the
mid-session-`.clear()` investigation above) and 60 cycles with genuinely in-flight worker
`QThread`s at close time — zero crashes, zero orphaned threads, zero leaked `QMenu`s across
both. Full unit suite on the branch: 1360 passed, 1 skipped, 1 known sandbox-only failure
(`test_gui.py::test_splash_resolves_fonts_to_installed_families`, already documented
elsewhere in this file, unrelated to this change).

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

### What the test suite structurally cannot see

Tests run with `QT_QPA_PLATFORM=offscreen` **and** set `AA_DontUseNativeDialogs`
(`tests/unit/conftest.py`). So no native macOS panel is ever opened and no AppKit modal session
is ever created — neither locally nor in CI. Bugs in that class are invisible to all 554 unit
tests by construction; they surface only in a real, native, interactive run.

### macOS has the narrowest font metrics we ship to — layout limits must be checked wider

A pixel budget verified only here is verified on the friendliest platform there is. The same
Preview chips measure ~1.5x wider on the Windows runner, and that is a routine Windows-only
red: the `test_window_fits_screen.py` ceiling read 1005 px on macOS and 1516 px on Windows
from the same code, so adding one checkbox to the ECG strip broke CI while every local run
stayed green. Two habits follow.

- Assert **ratios, not pixel figures**, and state every precondition relative to something
  measured in the same run (`win.minimumSizeHint().width()`, not `860`). A pixel literal in a
  layout test is a measurement of the developer's fonts: the first cut of the guards in
  `test_window_fits_screen.py` hard-coded numbers read off macOS and went red on Windows for
  precisely the reason the guards exist.
- To reproduce the Windows runner locally, widen the horizontal advance and leave the height
  alone: `QFont.setStretch(145)` on the application font lands within a few percent of it
  (modelled 1506-1538 px against CI's measured 1516 px). Scaling the **point size** is the
  wrong instrument -- it inflates row heights too, so it understates the width problem and
  invents height failures Windows does not have.
- Never assert on a QLabel's rendered `text()` when `flow_layout.elide` set it — how much
  survives is a font measurement. Assert on `toolTip()`, which holds the full string by
  contract. This is what made two unrelated EMG tests fail on Windows only.

A row of controls only wraps where a layout can break it. `flow_layout.FlowLayout` makes the
minimum width the widest single ITEM, so a chip built on a plain `QHBoxLayout` is one
unbreakable item whose minimum is still the sum of its contents. Build chips with
`install_flow` + `cluster` so each caption+field pair is its own item. `install_flow` also
sets `QSizePolicy.Preferred` + `setHeightForWidth(True)`: under `Maximum` Qt caps the widget
at its one-line `sizeHint` height and paints the wrapped row outside it.

### Three more Windows-metrics fixes worth reusing (ticket 20260810-1059, ui-overhaul)

- **A `QHeaderView.maximumSectionSize()` cap and a header's own legibility are different
  budgets, and Qt's `resizeColumnsToContents()` conflates them.** `result_table.py`'s
  `_MAX_SECTION_PX` exists to stop one pathologically wide CELL VALUE (a long file path)
  eating the viewport, but the cap also clamps the HEADER text while sizing it, so a font
  wider than the one the cap was tuned against (Segoe UI vs. this app's reference font) can
  clip an ordinary column identifier like `poes_tidal_swing` right along with a genuinely
  oversized value. `QHeaderView.sectionSizeHint(col)` returns a section's width from the
  HEADER content alone — independent of both the current cap and of any cell content, measured
  directly (`header.setMaximumSectionSize(-1)` to read it uncapped, `header.resizeSection()` to
  apply it — `resizeSection` itself still respects whatever cap is currently set, so the cap has
  to be widened to admit the floor BEFORE calling it). Use that as the floor no column may be
  resized below; a cell value stays free to be squeezed under the ordinary cap. `resize_result_table()` is the worked example.
- **`ElidingLabel`'s general-purpose floor (24 px) is too small for a label that is the ONLY
  thing naming something** (a panel header with no other visible caption). `panel.py`'s
  `titled_panel()` grew an optional `title_floor_chars` — a floor derived from the title's OWN
  length (`fontMetrics().averageCharWidth() * min(len(title), title_floor_chars)`), so a short
  title never elides at all and a long one shortens to a readable abbreviation instead of a
  bare "…". Left as `None` (the small default) everywhere else on purpose: `titled_panel`'s own
  fidelity-panel caller relies on the SMALL floor to let that whole panel shrink regardless of
  how long its tooltip explanation is (`test_fidelity_panel_tooltip_...` asserts
  `minimumSizeHint().width() <= 24`) — raising the floor globally broke that test. Pass the
  parameter at the specific call site that needs it, never in the shared default.
- **A "refit on resize" that re-derives its answer from a PURE function of (content, size) can
  be made idempotent by skipping the redo, not by re-deriving more carefully.**
  `_figure_fit.py`'s `refit_compact_figure()` re-measures matplotlib's tight-layout margins on
  every call, and `TightLayoutEngine.execute()` measures text extents against the axes' CURRENT
  position — which on this sandbox's Agg renderer reproduces bit-for-bit over 20 repeated calls
  at a fixed size, but on a real Windows runner is not bit-for-bit repeatable call to call
  (measured: 0.0131 figure-fraction drift over 20 refits at an unchanged size, against a 0.01
  budget). Since `_fit_compact_figure`'s decision is provably a pure function of
  `(ax._rm_full, canvas.width(), canvas.height())` once the stash is set, a refit at a size
  already fitted can only reproduce the same decision — so `refit_compact_figure` now caches
  `ax._rm_last_fit_size` and skips the redo when the size matches, rather than trying to out-round
  Windows' measurement jitter. Cache the FULL size tuple, not just height: `_pick_xlabel`'s room
  measurement depends on width too, and a first cut of this fix that cached height alone silently
  skipped legitimate re-fits when only the width changed (caught by
  `test_the_fidelity_x_label_never_runs_off_its_panel`, which resizes a HORIZONTAL splitter).

### `_pump_until` in the reactive Preview tests needs two calls, not one

`tests/unit/test_gui_reactive.py`'s `_pump_until(qapp, predicate, timeout)` returns
immediately if `predicate()` is already true — including on the very first check, before the
debounced `QTimer.singleShot` autorun has had a chance to fire. A single
`_pump_until(qapp, lambda: not pv._jobs and not pv._draining, 60)` right after selecting a
file (or switching it back) is therefore not "wait for the jobs to finish", it is "wait for
the jobs to finish IF they have already started" — and if nothing has started yet, `pv._jobs`
is already empty, so it returns `True` on the spot with **no job having run at all**. The
existing `test_selecting_a_file_autoruns_all_panels` gets this right with two calls back to
back: first `_pump_until(qapp, lambda: bool(pv._jobs) or bool(pv.busy_panels()), 10)` to wait
for something to actually start, then the "wait for it to finish" call. Skip the first call
and the test can pass while asserting on a UI that never recomputed — three separate
assertions did exactly this in the same PR (04-08-2026) before being caught, one of them
because the field it checked happened to already contain a matching substring in its
pre-recompute (blank/default) state, which made the false pass silent rather than an
obvious `AssertionError` on an empty string.

### Known non-issue: a font-resolution test fails on a minimal Linux sandbox

`tests/unit/test_gui.py::test_splash_resolves_fonts_to_installed_families` fails
reproducibly (not flaky) on a barebones Linux container with only DejaVu/Bitstream/FreeMono
installed: `respmech.ui.splash._resolve_svg_fonts()` falls back to
`QFontDatabase.systemFont(FixedFont).family()`, which on such a container returns the
generic string `"monospace"` — a name `QFontDatabase.families()` itself never lists (it
only enumerates concrete family names). The real CI (`ci.yml`) only runs `tests/unit` on
`windows-latest`/`macos-latest`, both of which have a real `Consolas`/`Menlo` install and
never hit this fallback path, so this is a sandbox-only artefact, not a product defect.
Confirmed 29-07-2026 while baselining a documentation-only change (no Python touched):
553 passed / 1 failed, before and after.

### A `QShortcut`'s context only matches on a VISIBLE widget chain — `hasFocus()` alone lies about this offscreen

Testing a `Qt.ShortcutContext.WindowShortcut`/`WidgetWithChildrenShortcut` fix (ticket
C02, the PageUp/PageDown-steals-the-focused-table bug) against the real app initially
looked like it didn't reproduce at all — a shortcut registered on the not-yet-selected
Preview & QC tab never fired, whether or not the fix was applied, so the "before" and
"after" test runs were indistinguishable. Cause: `pv`/`pv.table` reported
`isVisible() == False` (Setup was still the current tab in `MainWindow.tabs`) even
though `pv.table.setFocus()` still succeeded and `pv.table.hasFocus()` still returned
`True` — the offscreen QPA platform is lenient about focus on a hidden widget in a way a
real windowing system is not. Qt's shortcut matcher, unlike `hasFocus()`, DOES check
that the whole widget chain up to the shortcut's context widget is actually visible, so
a shortcut on an unshown tab page never gets a chance to fire regardless of which
widget "has focus". A GUI test exercising a shortcut on any widget that lives inside a
`QTabWidget` page (or another widget hidden until switched to) must call
`win.tabs.setCurrentWidget(...)` (and, if the target is on a `PreviewScreen`
sub-tab, `pv.subtabs.setCurrentWidget(...)` too) BEFORE simulating the key — confirmed
by making the C02 regression test fail against the unfixed source only once this call
was added, and pass against the fix.

### A per-widget `setStyleSheet()` shadows the app-wide QSS for pseudo-states it never mentions

`_emg_noise.py`'s per-channel result-picker `QCheckBox`es each carry their own
`setStyleSheet()` (to draw the indicator in that channel's plot colour). Adding a new
`QCheckBox::indicator:focus` rule to the GLOBAL theme QSS (`theme.py`, ticket C02) had
no effect on these checkboxes at all — a keyboard user tabbing to one showed zero
visual change, the exact defect the ticket existed to fix, but only for widgets with
their own local stylesheet. A widget-level `setStyleSheet()` does not layer additively
on top of the app-wide one for a selector it also declares: it wins outright for that
selector, including pseudo-states the LOCAL sheet never wrote a rule for. Any future
per-instance-styled control (colour-coded checkboxes, channel-tinted buttons, etc.)
needs its OWN copy of every shared interaction-state rule (`:focus`, `:hover`,
`:disabled`) it wants to keep — grep for `setStyleSheet(` on the widget TYPE you are
adding a global rule for before assuming the app-wide QSS reaches every instance.

### A `dtype=object` DataFrame column can still silently turn `None` into NaN

Building a test fixture for a "missing value" cell — a real, plausible state in the
result tables (e.g. an unreliable EMG detector writing NaN, or a genuinely absent
field) — is not as simple as `pd.array([None, "a string"], dtype=object)` or
`np.array([None, "a string"], dtype=object)` fed straight into `pd.DataFrame`: on
this project's pinned pandas (3.0.5), mixing `None` with a string in either of
those makes pandas infer its own `str` extension dtype for the column, and as
part of that inference it normalises the `None` to a float NaN before any
application code ever sees it — so a test meant to cover "a real `None` survived
into the model" silently covers NaN instead, twice over, without an error.
`pd.Series([None, "a string"], dtype=object)` does keep the `None`. If a test
needs to distinguish `None` from NaN (they can render differently — see
`ui/result_table.py`'s `_format_display`), build the column that way, not via
`pd.array(...)`/`np.array(...)`.

### A worker signal connected to a lambda across a `Qt.QueuedConnection` can segfault

A second-thread `Signal` (`BatchWorker`/`WriteWorker` in `ui/workers.py`) must be
connected to a **bound method**, never a bare `lambda`, whenever the connection is
explicit `Qt.QueuedConnection` (the pattern this app always uses for a worker-thread
signal — see the comment at every such `.connect(...)` call in `ui/screens/run_screen.py`).
A lambda has no `QObject` identity of its own, so PySide6 cannot resolve which thread's
event loop the queued call should be delivered on. Found independently twice while
building ticket A06's "Write results to another folder…" feature (`ui/screens/
run_screen.py`'s `WriteWorker` wiring): the symptom was not a clean exception but
`QThread::wait: Thread tried to wait on itself` and a reproducible segfault inside a
pytest run. Store the target as a bound method (`self._on_write_elsewhere_finished`, not
`lambda r: self._on_write_elsewhere_finished(r)`) and connect that.

### A GUI-thread flag driven by a `Qt.QueuedConnection` signal lags the worker thread's own state

A worker-thread transition (e.g. `BatchWorker` entering its uninterruptible write phase)
is real the instant the worker thread makes it — but anything the GUI derives from a
*signal* announcing that transition (a heartbeat timer started in the signal's handler,
a flag set there) only becomes true once Qt's event loop has actually delivered that
queued signal, which is unavoidably asynchronous (a worker thread must never touch
widgets directly, so `Qt.QueuedConnection` is correct and not the bug). Code that reacts
to a user action in that gap — e.g. `RunScreen._cancel()` deciding which message to show
based on `self._heartbeat.isActive()` — can act on stale information for however long
that one event-loop tick takes. Found by three independent review agents on the same
diff (ticket "Cancel and progress become honest during the write phase"): a Cancel click
landing in that gap logged the pre-transition message even though the worker had already
committed to the phase where cancelling does nothing.

Fix: have the worker thread set a plain attribute on itself (`self._writing = True` in
`ui/workers.py`, as literally its first action on entering the phase) and have the GUI
read that attribute directly (`getattr(self._worker, "_writing", False)`) instead of
inferring the transition from a Qt-delivered side effect. A simple attribute read/write
is atomic under the GIL, so this closes the race to bytecode width instead of one event-
loop tick. Applies to any future GUI code deciding "has the worker done X yet" — read the
worker's own state directly when it is a plain value, don't infer it from a queued
signal's side effects.

### A widget populated for the first time must not fire the signal that starts background work

`PreviewScreen`'s file selector used to be a `QComboBox` that auto-selected index 0 on
its very first populate as a bare Qt side effect: the populate itself ran under
`blockSignals(True)`, so that auto-select never fired `currentTextChanged`, and nothing
downstream ever reacted to a freshly built screen's first file. Ticket B02
(`ui/file_rail.py`'s `FileRail`) replaced the combo with a list-backed rail, and its
first cut got this wrong: `FileRail.set_manifest()` explicitly called `select_filename()`
on the first file, which — unlike the combo's silent auto-select — DOES emit
`selectionChanged`, which routes into `_on_file_selected` → `_begin_file_switch()` →
arms the 300 ms reactive debounce. The result: constructing a bare `PreviewScreen`
(hundreds of times across the test suite) now started real background analysis work it
never used to, given enough incidental event-loop turns for the timer to fire.

This surfaced as a **reproducible, order-dependent failure in a completely unrelated
layout test** (`test_window_fits_screen.py::test_the_preview_pages_scroll_instead_of_
compressing_their_graphs`) — a real layout race between a late-arriving async render and
the page-fit mechanism, only exposed because analysis now started when it never had
before. It did NOT reproduce in isolation (not enough incidental event-loop turns for the
newly-armed timer to fire within the test's fixed `processEvents()` budget), only inside
a ~150-test run — which is what made a controlled before/after diff (same test sequence,
old vs. new source) the only way to actually prove causation rather than guess "probably
flaky." **A test that passes alone but fails in a big suite run is not automatically
flaky — run the same sequence against the OLD code first before writing it off.**

Fix: `FileRail.set_manifest()` now *quietly* adopts the first file as `current_filename()`
when the rail has no identity at all yet (mirrors the old combo's silent auto-select
exactly), and only a REAL subsequent switch (a caller explicitly calling
`select_filename`/`select_index`/`step`, or the previous file vanishing from a rebuilt
list) emits `selectionChanged`. Applies generally: a widget's initial-populate state
must reproduce a REPLACED widget's exact side-effect profile, not just its visible value —
"looks selected" and "IS selected enough to react to" are different guarantees, and only
the second one starts work a test (or a user) may not be expecting yet.

### Monkeypatching a staging function must target where it was IMPORTED TO, not FROM

`preview/_mechanics.py` does `from respmech.ui.workers import (..., stage_mechanics_preview,
...)` — a direct name binding. `monkeypatch.setattr(workers, "stage_mechanics_preview",
fake)` (patching the SOURCE module `ui.workers`) has no effect on `_mechanics.py`'s own
call to `stage_mechanics_preview(...)`, since that name was already bound into
`_mechanics.py`'s namespace at import time and never looks the attribute up on `workers`
again. Patch it where it is actually CALLED FROM: `monkeypatch.setattr(
respmech.ui.screens.preview._mechanics, "stage_mechanics_preview", fake)` (or import the
`_mechanics` module and patch the name on it directly). Same applies to the `_ecg`/
`_emg_noise` mixins, which import their own staging functions the same way. Found while
writing a B02 test that meant to simulate "a file that cannot be loaded" and silently
patched nothing.

### Embedding one screen permanently under another's content can blow the combined minimum-size budget — even collapsed

Ticket B03 folded `RunScreen` into `PreviewScreen`'s own layout (a drawer under the file
rail, replacing the old third tab) instead of a separate `QTabWidget` page. Each screen's
own `minimumSizeHint()` was independently within the `test_window_fits_screen.py` budget
(1280×547 usable) — but STACKING one under the other in one tab measured 611 px against
that 547 px ceiling, because a tab-page's minimum used to be measured on its own; two
screens sharing one tab sum their minimums instead of each being measured in isolation.

Fixed with a genuinely collapsed-by-default drawer (`RunScreen._results_section`, hidden
until a "Run & results ▸" toggle — or a run itself — expands it), so only a single compact
summary row (~44-56 px) is ever a permanent tax on the tab. Two further findings from
getting there:

- **`FitScrollArea`/`_PageScroll`'s "fill the viewport, else use minimumSizeHint" fit
  logic is for STATIC pages, not ones that change size at runtime.** Wrapping the whole
  drawer in it (reusing Preview's own subtab-page scrolling, lifted out to
  `ui/panel.py::FitScrollArea`/`scrollable()`) made the collapsed drawer WORSE (362 px,
  not smaller) — an early fit pass against the page's larger pre-collapse size stuck in
  the scroll area's own reported size and fed back into the outer layout as if that were
  still the page's natural height. Don't reach for it when a page's own content toggles
  visibility; it is the right tool only for a page whose natural height is fixed once
  built (Preview & QC's own EMG/mechanics/noise subtabs are the case it was built for).
- **Even a minimal, collapsed addition can still tip an unrelated, previously-passing
  layout test that was calibrated close to a threshold.** `test_the_noise_tab_divides_
  into_thirds` (an unrelated Preview subtab test) went from ~33% to ~44-50% "chrome" share
  of an 800 px test window purely from the drawer's ~56-90 px collapsed footprint, and a
  DIFFERENT test's "tall panel" splitter scenario (`test_the_diagnostic_figures_follow_
  their_panel_in_both_directions`) stopped delivering enough absolute height for a
  matplotlib legend to render at all, at the SAME window height that used to work. Fixed
  by giving that one test more window height (with the reasoning written into the test
  itself), not by chasing the drawer's footprint down further — there is a floor below
  which "collapsed" can't go and still be an affordance a user can find. **Any future
  permanent addition to Preview & QC's tab must be checked against the WHOLE of
  `test_window_fits_screen.py` and `test_compact_plots.py`, not just the tests that
  exercise the new code directly** — both are tuned against absolute window/splitter
  pixels, and neither failure looked anything like the change that caused it.

### A construction-time `QTimer.singleShot(0, ...)` can segfault a COMPLETELY unrelated test, hundreds of tests later

Deferring `RunScreen.__init__`'s one real disk glob (`_update_plan_summary`, which
lazy-imports pandas/scipy via `core.io.plan.plan_outputs`) past construction looked like
the obviously-correct fix for "don't do this synchronously in every window's `__init__``"
— schedule it with `QTimer.singleShot(0, self._update_plan_summary)` instead. It produced
a **reproducible segfault in `test_section_flow.py`**, a file with no relationship to
`RunScreen` at all, always at the same ~71% point in a full `pytest tests/unit` run,
confirmed 3 times in a row before the cause was found and once more (clean) after
reverting to a plain synchronous call.

The mechanism: almost no unit test spins a real Qt event loop before it constructs a
`RunScreen`/`MainWindow` and closes it — `qapp.processEvents()` is called explicitly only
where a test actually needs it. A zero-delay `singleShot` scheduled in `__init__` and
never fired before the widget is closed does not fire NEVER; it sits queued on the
(session-scoped) `QApplication` forever, still holding a bound-method reference to the
widget. Across a ~900-test suite, hundreds of these accumulate. The first test anywhere
in the suite that happens to call `processEvents()` — in this case `test_section_flow.py`,
which pumps events for an unrelated reason — flushes the ENTIRE backlog in one burst,
invoking `_update_plan_summary()` on widgets that were destroyed dozens or hundreds of
tests earlier. That is a use-after-free at the C++ level (a shiboken-wrapped deleted
`QObject`), hence a segfault rather than a clean Python exception — and it always lands
in whatever test happens to be first to call `processEvents()` after the backlog has
grown large enough, which looks completely unrelated to the actual defect.

**Never schedule a construction-time `QTimer.singleShot` (any delay, but especially 0) on
`self` inside a widget's `__init__` unless something in the SAME constructor guarantees
the event loop will run before the widget can be destroyed** (a real GUI session does;
nearly all unit tests do not). If deferred, one-shot work is genuinely needed, tie its
lifetime to something that gets cancelled on teardown, or — the simpler, correct choice
made here — just call it synchronously and accept the (real, but far smaller and already
precedented — `_start()`'s `_append_plan` pays the identical import cost on every click)
construction-time cost instead.

### A `FlowLayout`-holding card inside `section_flow.SectionColumns` inflates the column width unless the card's own `sizeHint` is made honest

Ticket B05 split the Setup screen into two columns (`section_flow.install_sections`,
`max_columns=2`: a "rig" column of Input+Channels, a "leverance" column of Output+Sample
entropy) and, separately, ran the Output card's two checkbox groups (Tables, Diagnostic
figures) through `flow_layout.install_flow` so they wrap instead of stacking one checkbox
per line. Each change alone tested fine. Combined, the two-column split silently almost
never happened at any realistic window size — an offscreen test resized to a hand-picked
1700 px "wide enough" width and passed, which is exactly what hid it.

The mechanism: `SectionColumns.column_target()` decides how wide a column should be from
the upper quartile of its items' `sizeHint().width()` (`_comfort_width`, `section_flow.py`).
`FlowLayout.sizeHint()` is *deliberately* "everything on one line" (its own docstring) — the
right contract for a chip strip placed directly in a plain `QVBoxLayout`, where the caller
just wants the natural, unwrapped width when there's room. But that same "natural" width
also propagates straight up through Qt's default `QGroupBox`→`QFormLayout` spanning-row
sizing and the plain `QVBoxLayout` wrapper stacking the cards, into the ONE number
`SectionColumns` uses to decide how wide a "comfortable" column is. Measured: the Output
card's checkbox rows inflated its `sizeHint().width()` to ~1260 px (the sum of every chip on
one line), which pushed the real two-column threshold to ~1550 px — past the app's own
default window width, so the split existed in the code but not in practice, and a test
resized to 1700 px (comfortably past that inflated threshold) could not tell the difference.

**Fix and the rule going forward:** any container that both (a) holds a `FlowLayout`
somewhere inside it (directly or nested in a card) and (b) is fed as an item to a
width-deciding column balancer (`SectionColumns` or anything using the same
quartile-of-sizeHint pattern) needs an HONEST `sizeHint()` — one that does not let the
FlowLayout's one-line width vote. `settings_screen.py`'s `SettingsScreen._FlowGroup` is the
fix here: a tiny `QWidget` subclass whose `sizeHint()` returns its own layout's
`minimumSize()` (the width the row can already be safely squeezed to) instead of Qt's
default delegation. This changes nothing about the row's actual runtime wrapping — that is
governed by `heightForWidth` at whatever width the row is actually GIVEN, independent of
`sizeHint()` — only how wide a column the row is allowed to ask for. `SectionCard.sizeHint()`
(`section_flow.py`) already solves the identical problem for prose notes ("excluded on
purpose: prose wraps to whatever it is given... letting one paragraph vote here would have
it decide the column width of the entire dialog") — the same reasoning applies to any
FlowLayout row, and any future card combining `install_sections` with `install_flow` should
reach for the same honest-`sizeHint` pattern from the start, not discover it by measuring a
threshold that never fires.

### Filename-keyed batch state needs a folder tag, or a re-pointed analysis silently reapplies stale decisions

Ticket B06 found and fixed a real data-integrity bug, not just a UI one: `exclude_breaths`,
`breath_counts` and the EMG noise reference (`core/settings.py`) all key on a bare
**filename**, with no idea which recordings folder that filename was chosen in. Point the
same loaded analysis at a DIFFERENT folder that happens to contain a file of the same
name — routine in a multi-subject study where every export shares a fixed LabChart
filename — and the old exclusions/overrides/reference kept silently applying to the new
folder's file. Nothing said so: the Setup QC strip stayed green, and only the excluded-
breath count in Preview's QC line hinted anything was off, with no way to tell it apart
from a genuine decision.

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
Building true per-breath provenance would fix that but is a bigger change than this
ticket's own field, singular, on the entry — out of scope unless a future ticket
specifically asks for it.

**Wherever a write path can resolve carried-over state, refresh whatever is SHOWING it.**
The Setup banner only rechecks on an actual `input.folder` value change
(`_on_inputs_changed`) or on `from_state()` (opening/loading an analysis) — by itself that
leaves it blind to a Preview-side edit (a breath toggle, committing the breath-count-
overrides dialog, picking a noise reference) that resolves the very state it's warning
about, so a dismissed-but-then-fixed banner could sit there indefinitely as a stale, already
wrong warning. Fixed by wiring `PreviewScreen.settings_edited` to
`SettingsScreen._update_carried_banner` in `main_window.py`, plus a direct call in
`SettingsScreen.set_noise_reference` for the one write path (the noise-reference picker)
that goes through its own dedicated signal instead. Any FUTURE state this pattern is
extended to needs the same treatment: know every path that can create OR resolve it, not
just the one this ticket happened to add a banner for.

### Sending a `QDropEvent` with no preceding accepted `QDragEnterEvent` segfaults this sandbox's offscreen Qt

Found writing tests for ticket C04 (drag-and-drop). `QApplication.sendEvent(widget,
QDropEvent(...))` sent on its own — without first sending a `QDragEnterEvent` for the same
gesture and having it accepted — does not raise or return `False`: it **segfaults the whole
pytest process**, mid-test, with the crash traceback pointing at pytest's own assertion
`saferepr()` machinery rather than at anything applicaton-related, which makes it look like
an unrelated pytest/repr bug rather than what it is. A real drag session (mouse or a native
Finder/Explorer drop) can never produce a bare drop — the platform always sends
DragEnter→[DragMove…]→Drop — so this is purely a test-construction hazard, not a
production one, but it costs real time to diagnose blind: **any test that synthesizes a
`QDropEvent` must send a matching, accepted `QDragEnterEvent` to the same widget first**,
even when the assertion under test only cares about the drop. See `install_path_drop`'s
tests (`tests/unit/test_path_drop.py`, `test_file_association.py`) for the pattern.

### A test helper that monkeypatches `QDialog.exec` misses a subclass that overrides `exec()` itself — and hangs, not fails

Found writing/verifying ticket D13 (`AdvancedDialog` gained a `modal` parameter and its own
`exec()` override — see `ui/advanced_dialog.py`). Two existing test helpers,
`test_dialog_fits_screen.py::_advanced_dialogs` and
`test_theme_paints_both_modes.py::_all_windows`, build all three Advanced modals by
monkeypatching `QDialog.exec = _capture` (auto-reject, no blocking) around the three
`_open_*_advanced()` openers — a standard, previously-safe Qt testing pattern, since every
dialog used to be a bare `QDialog.exec()` call.

`AdvancedDialog.exec()` now branches on `modal`: for `modal=True` (ECG/EMG, and the default)
it still calls `super().exec()`, so the `QDialog.exec` patch keeps working for those. But for
`modal=False` (the mechanics dialog only) it **never calls `QDialog.exec`/`super().exec()` at
all** — it calls `self.show()` and runs its own `QEventLoop`, because `QDialog.exec()` sets
`Qt.WA_ShowModal` unconditionally regardless of `setModal(False)` (see the class's own
docstring). A patch on the base class therefore silently never intercepts this one subclass
in this one mode: the dialog really shows and really blocks, forever, since nothing in either
helper ever accepts/rejects/closes it.

**The failure mode is a hang, not a red test or an exception** — measured directly: the
process burned ~2s of CPU time over 3 minutes of wall clock (an idle wait, not a slow
computation), and it looked exactly like the sandbox's known unrelated OOM/resource-pressure
symptoms until `time` on the isolated command made the near-zero CPU usage obvious. Both
helpers were fixed by patching `AdvancedDialog.exec` itself instead of `QDialog.exec` —
that intercepts before the modal/non-modal branch is even evaluated, so it correctly covers
both cases regardless of which one a given caller uses.

**Rule for any future dialog subclass that overrides `exec()`** (not just `AdvancedDialog`):
grep for `\.exec\s*=` across `tests/unit/` and check whether an existing "build without
blocking" helper targets the BASE class rather than the actual subclass being constructed.
Patching the most-derived class an opener actually builds is the only version of this pattern
that survives a future override.

### One aggregate `pg.GraphicsObject` beats N `pg.LinearRegionItem`s for "many same-shaped shaded spans" — and `pg.LinearRegionItem`'s own trick for full-height-regardless-of-zoom is reusable

Found fixing ticket D15 (the mechanics stack froze the GUI thread for up to ~12s stepping
through files with real recordings — up to 1210 `QGraphicsItem`s for 110 breaths across 5
channel plots, 11 per breath per plot: a region, a now-redundant boundary line, and a label).
`ui/screens/preview/_plot_helpers.py::BreathSpansItem` replaces the per-breath
`pg.LinearRegionItem` with ONE `pg.GraphicsObject` subclass instance per plot that paints every
breath span itself in one `paint()` call, holding `[(t0, t1, brush), ...]` and a
`set_brush(index, brush)` for the include/exclude toggle repaint.

The part worth knowing for any future "many shaded regions on one plot" need: `LinearRegionItem`
gets its "spans the full plot height regardless of y-zoom" behaviour from
`self.viewRect()` (`GraphicsItem.viewRect()`, cached and auto-invalidated by the base class's
`viewTransformChanged` slot on every pan/zoom/resize) — **not** from tracking the ViewBox's
y-range itself. A hand-rolled aggregate item can reuse this directly: `boundingRect()`/`paint()`
both call `self.viewRect()` for the y-extent and only override left/right with the item's own
x-extent. And `dataBounds(axis, ...)` must mirror `LinearRegionItem`'s own axis restriction —
return `None` for the y-axis — or the item's self-derived y-extent feeds back into the very
y-autorange computation it derives from, which is nonsensical and, depending on call order, can
produce a runaway range.

Applied in two places (`_draw_breath_overlays` for the mechanics stack, `_paint_breaths` for the
shared raw/detail/result EMG views) — same anti-pattern, same fix, generalized because both were
part of the same measured freeze. Click-to-toggle needed no changes: it already resolved a click
to a breath number from scene coordinates against a plain dict (`_breath_spans`), never from the
graphics item the mouse actually hit, so swapping what PAINTS the region is invisible to it. One
real loss: a per-region hover tooltip on the old `LinearRegionItem` (explaining a "carried over"
exclusion) cannot be reproduced on one shared item without new hover-tracking — dropped, since the
same fact is already shown by the hatched brush and the QC line.

**Rule:** the next time a screen needs to draw "N same-shaped shaded regions along a plot's time
axis" (event markers, segment colouring, more breath-like overlays), reach for this
`BreathSpansItem` pattern — one item per plot with an internal list — not a `pg.LinearRegionItem`
per element. It is the difference between an O(1)-per-plot item count and an O(N) one.

### A "clear the stale error card" call must not also silence a currently-busy overlay — `BusyOverlay.stop()` conflates the two

Found in the same D15 pass, as a second, independent bug behind the same symptom (the busy
spinner over Preview & QC's Mechanics/raw panels appeared "frozen" during the freeze above,
rather than obviously hidden or obviously animating). `_render_preview` calls
`_clear_panel_overlays("channels", "raw")` as its first statement, so a stale error card from a
previous failed render never sits over a fresh one — but `BusyOverlay.stop()` (what that call
uses) unconditionally does `busy = False; error = None; hide()`, whatever state the overlay was
actually in. The REACTIVE job dispatch (`screen.py::_launch`) shows the busy overlay BEFORE the
worker thread even starts, so by the time the worker finishes and `_render_preview` runs, the
overlay is legitimately busy=True — and that first-line `stop()` call hides it immediately,
before a single expensive draw call has run. Because `QWidget.hide()`'s visual effect is not
painted until the event loop next turns, and the very next thing that happens is the GUI-thread
block this ticket exists to fix, the LAST rendered frame of the spinner just sits on screen,
unchanged, for the whole freeze — visually indistinguishable from "frozen", because it effectively
is: a real widget's disappearance that Qt was never given a chance to actually paint.

Fixed with `BusyOverlay.clear_error()`: only `stop()`s (hides) the overlay if `self.error is not
None`; otherwise a no-op, so a legitimately busy overlay is left exactly as it was.
`_render_preview_stage1` (see the next entry) uses `clear_error()`, not
`_clear_panel_overlays`/`.stop()`.

**Rule:** anywhere in this screen (or a future one with the same busy/error overlay pattern) that
needs to "dismiss a stale error before drawing", use `clear_error()`. Reach for
`_clear_panel_overlays`/`.stop()` only where a WHOLE panel set is being reset from scratch (a file
switch, an invalid-settings blank) and nothing — busy or not — should survive that reset.

### Splitting a synchronous render across `QTimer.singleShot(0, ...)` needs its OWN staleness guard, distinct from the job-token check that already exists

Also from D15. The reactive job dispatch already guards against a superseded WORKER result:
`_on_job_done` checks `job.token != self._tokens[job.kind]` before calling the render function at
all. That guard is not enough once the render function ITSELF is split into deferred pieces: once
`_render_preview_async`'s stage1 (the cheap part — clear stale state, draw the 5 channel curves)
returns and hands control back to the event loop, the user can switch files before stage2/stage3
(the two expensive, now-deferred pieces — breath overlays, raw EMG stack) get their
`QTimer.singleShot(0, ...)` turn. `_begin_file_switch` clears `self._channel_plots`/calls
`self.plots.clear()` synchronously and immediately (no async gap of its own), so a stale stage2
firing afterwards would either draw nothing useful (the plot list is now empty) or — worse —
repopulate `self._breath_spans`/`self._breath_regions` with the OLD file's data right after
`_reset_breath_state()` had deliberately emptied them for the new one, silently resurrecting stale
click-to-toggle state for a file no longer on screen.

Fixed with a plain integer generation counter, `self._mech_render_gen`: bumped once at the top of
`_render_preview_stage1` (covers every new render, sync or async) AND once in
`_reset_breath_state()` (covers a file switch that hasn't yet triggered a new render). Each
deferred continuation (`_render_preview_async_stage2/3`) captures the counter's value at schedule
time and checks it still matches before touching anything; a mismatch means silently abandoning —
a newer render or reset already owns the panels. `QTimer.singleShot(0, ...)`'s lambda captures
`self`, but nothing explicitly cancels a pending one on window close; a stale callback firing after
teardown is caught the same way (the generation will not match, since nothing else bumps it after
close — verify this holds if `MainWindow.closeEvent`/`shutdown()` is ever changed to reset state
during teardown, which would need its own bump too).

**Rule:** a job-token check at DISPATCH time (`_schedule`/`_on_job_done`'s existing pattern) and a
generation counter at RENDER time (this ticket's pattern) answer two different questions —
"is this still the current worker result?" vs. "does this still-executing, already-dispatched
render still own the widgets it's about to touch?" — and splitting any other reactive render across
`QTimer.singleShot(0, ...)` needs BOTH, not just the one that already existed.

## Releases (`.github/workflows/release.yml` = "Build installers")

- Trigger: push a `v*` tag (or manual dispatch). Builds a Windows **MSI** and a
  macOS **dmg** with briefcase, then (on a tag) the `publish-release` job creates
  a GitHub **release** (marked **Latest**) with the installers attached.
- macOS signing is **secret-gated** (Developer ID + notarisation when the Apple
  secrets are present, else ad-hoc). The Windows MSI is built unsigned and
  **Certum-signed locally** after release (`scripts/sign-msi-certum.sh`); see
  `docs/SIGNING.md`.
- Releases are full releases (the newest is marked **Latest**; `release.yml` passes
  `--latest`). Keep tags clean semver `vX.Y.Z` (no `-rc/-beta`) — the website picks the
  version that way.
- **`CHANGELOG.md`** (repo root, added 29-07-2026) is the canonical, complete release
  log — one section per release, newest first. Add its entry as **step 1** of every
  release (see `docs/RELEASING.md`), before bumping the version. `respmech-website`'s
  `changelog.html` mirrors it in a version trimmed to what an app user cares about
  (no CI/packaging-only notes); update both together. Since 30-07-2026 the website
  side is automatic: on a release its workflow *promotes* the hand-written "Coming
  next" section into `vX.Y.Z` and takes only the lead sentence from the entry here.
  So keep "Coming next" on respmech.dk current as you merge, and the release
  announces itself; the mailing-list e-mail is built from that very section, and a
  missing one used to mean subscribers silently got nothing.
- **`tools/check_changelog.py`** (added 30-07-2026) answers "is the entry
  exhaustive?" with evidence instead of memory. It walks the commits in the range,
  sets aside the ones touching only tests/docs/CI/tooling, and prints every
  user-visible change beside the bullet that best matches it, weakest first. It
  fails on the one thing a word comparison can be certain of: a change with **no**
  trace in the entry. Weaker matches are a worksheet, not a verdict — three
  successively cleverer rules were tried and each was measurably foolable on the
  same data, which is documented in the tool and pinned by
  `tests/unit/test_check_changelog.py`. A deliberate omission is recorded with
  `<!-- changelog-skip <sha7> <reason> -->`, never merely silenced. Hard gate on the
  tag in `publish-pypi.yml`; informational worksheet on every push in `ci.yml`.
- **`## Unreleased`** (added 29-07-2026) is a hand-maintained draft sitting above the
  latest dated release, describing everything since the last tag. It is updated only
  when explicitly asked to, never automatically per commit. At release time (step 1
  above), fold it into the new dated entry and collapse it back to an HTML-commented,
  empty placeholder — see the mechanism documented directly in `CHANGELOG.md` and in
  `docs/RELEASING.md`.

## Website (respmech.dk)

The marketing/info site lives in the **private** repo
`emilwalsted/respmech-website` and deploys to https://www.respmech.dk. It names
the current version on its download button / labels by resolving the **highest
clean-semver, non-draft release** of this repo (it also counts any pre-releases, so it
keeps working regardless of the release/pre-release flag).

**Release → website hook.** The last step of `publish-release`
("Notify respmech.dk…") sends a `repository_dispatch` (`event_type:
respmech-release`) to the website repo, which then refreshes its version and
redeploys — so the site updates within seconds of a release.

- Requires the secret **`WEBSITE_DISPATCH_TOKEN`** here: a fine-grained PAT with
  **Contents: write** on `emilwalsted/respmech-website`. If it is absent the step
  is skipped and the release/build is unaffected.
- Without the hook, the website's daily poll still catches a new release within a
  day. Setup helper + full docs: `deploy/setup.sh` and `deploy/README.md` in the
  website repo.

> When changing `release.yml`, keep the notify step **after** `gh release create`
> and non-fatal (guarded on the token) so it can never break an installer build.
