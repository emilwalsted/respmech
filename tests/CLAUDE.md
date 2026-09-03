# CLAUDE.md — `tests/`

Test-writing hazards for this suite: what offscreen Qt lies about, what the suite
structurally cannot see, and the fixture/pump/patching patterns that were each found
the hard way. The project-wide rules stay in the repo-root `CLAUDE.md`.

### What the test suite structurally cannot see

Tests run with `QT_QPA_PLATFORM=offscreen` **and** set `AA_DontUseNativeDialogs`
(`tests/unit/conftest.py`). So no native macOS panel is ever opened and no AppKit modal session
is ever created — neither locally nor in CI. Bugs in that class are invisible to all unit
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

A third variant of the same lesson (run #245, master's first full-matrix run,
12-08-2026): since D15 split the mechanics render into deferred stages, the
'channels'/'raw' busy overlays are stopped by the async chain's own final
`QTimer.singleShot(0)` stage — one or more event-loop turns AFTER `_jobs`/`_draining`
empty. A bare `assert pv.busy_panels() == set()` immediately after the drain pump is
therefore a race against the deferred stages' loop turn: it passed everywhere except
macOS×py3.11, the fleet's slowest combination, where the drain flag fell inside the
window before the stages fired. Four such asserts existed; all now go through
`_assert_panels_idle()` (a third, pumped wait). The general rule: any state the async
render chain itself finalizes (spinners, stage-3-drawn content) must be awaited with
its own `_pump_until`, never read bare off the back of the drain predicate —
"all jobs drained" is not "all panels idle".

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

### Closing plots in a test (point 6)

`tests/unit/conftest.py` never deletes a closed `MainWindow` — deleting one segfaults
nondeterministically on Python 3.11 (see the reaper comment in
`_close_top_level_windows`).

Qt never delivers `closeEvent` to a child widget when its PARENT window closes — which is
why `PreviewScreen`/`RunScreen`/`ColumnStack` each carry their own, and why
`tests/unit/test_plot_cleanup_contract.py` guards them per type.

Tests that assert on a plot's state after closing it must capture the `PlotItem` reference BEFORE
calling `close_plots()`, never re-fetch it after (see the `close_plots()`-adjacent tests
in `test_column_stack.py`/`test_channel_summary.py`/`test_channel_setup.py`/
`test_preview_screen.py` for the pattern).

If the suite's macOS wall time or sandbox OOM recur, re-measure with `RESPMECH_NET_CENSUS`/
`RESPMECH_NET_PROFILE` before assuming this is the same class of bug — the population
this ticket targeted is gone.
