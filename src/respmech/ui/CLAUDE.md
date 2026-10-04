# CLAUDE.md — `src/respmech/ui/`

Qt/GUI gotchas for the PySide6 app: layout and font-metric budgets, styling, worker
threads and queued signals, deferred rendering, and pyqtgraph. The project-wide rules
stay in the repo-root `CLAUDE.md`; test-side hazards are in `tests/CLAUDE.md`.

### A literal `&` in any button, group-box, menu/action, tab or buddy-label caption must be doubled

Qt treats a single `&` as a mnemonic marker: it is swallowed and the following character
underlined, which silently eats a whole word when that character is a space ("Run &
results" rendered as "Run _results" in a shipped release). Every caption in this app that
wants a literal ampersand doubles it ("Preview && QC", "Process && write this file"), and
`tests/unit/test_ui_wording.py::test_no_caption_anywhere_turns_an_ampersand_into_a_mnemonic`
enforces it across the whole window — not just push buttons, but `QGroupBox` titles,
every menu/menu-bar `QAction`, `QTabBar` captions and buddy `QLabel`s too, via the shared
`_lone_ampersands(root)` helper in `tests/unit/_helpers.py`. A new dialog or window does
not need its own copy of this guard: construct it once inside the existing test (or add
it to whichever window the test already builds) so the scan reaches it.

### A `QDialog` without `Qt.WA_DeleteOnClose` is never destroyed by its own `accept()`/`close()`

Set the attribute explicitly on any one-shot dialog, or use the `prior=` pattern for
a dialog a screen keeps its own reference to. Separately: `QDialog.exec()`
unconditionally sets `Qt.WA_ShowModal`, regardless of what `setModal()` was told —
a non-modal dialog that still wants the `exec()` calling convention has to override
`exec()` itself rather than rely on `setModal(False)` alone (`AdvancedDialog`'s
`modal=`/`on_apply`/`derived_debounce_ms` is the worked example). Any test helper
that stubs `exec` to avoid blocking must patch the MOST-DERIVED class actually
constructed — patching the base `QDialog.exec` silently misses a subclass override
and hangs instead of failing (see `tests/CLAUDE.md`).

### Status messages from other screens are suppressed during a run

While a batch run is in progress, status/toast messages from screens other than the
active one are held back; an action that gets rejected because a run is in progress
uses the dedicated, forced signal `write_action_blocked` instead of the normal
status-message path, so it is never accidentally swallowed by that suppression.
Related state plumbing: `theme.set_stack_floor(..., viewport_height=...)` for the
minimum-height floor of the screen stack, and splitter-position persistence via
`ui/prefs` — note that `setSizes()` does not itself emit `splitterMoved`, so code
persisting a splitter's position must not rely on that signal firing from a
programmatic resize.

### Chips and `FlowLayout`: where a row of controls can actually wrap

A row of controls only wraps where a layout can break it. `flow_layout.FlowLayout` makes the
minimum width the widest single ITEM, so a chip built on a plain `QHBoxLayout` is one
unbreakable item whose minimum is still the sum of its contents. Build chips with
`install_flow` + `cluster` so each caption+field pair is its own item. `install_flow` also
sets `QSizePolicy.Preferred` + `setHeightForWidth(True)`: under `Maximum` Qt caps the widget
at its one-line `sizeHint` height and paints the wrapped row outside it.

### Three more Windows-metrics fixes worth reusing (found 10-08-2026, ui-overhaul)

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

### A worker signal connected to a lambda across a `Qt.QueuedConnection` can segfault

A second-thread `Signal` (`BatchWorker`/`WriteWorker` in `ui/workers.py`) must be
connected to a **bound method**, never a bare `lambda`, whenever the connection is
explicit `Qt.QueuedConnection` (the pattern this app always uses for a worker-thread
signal — see the comment at every such `.connect(...)` call in `ui/screens/run_screen.py`).
A lambda has no `QObject` identity of its own, so PySide6 cannot resolve which thread's
event loop the queued call should be delivered on.
Store the target as a bound method (`self._on_write_elsewhere_finished`, not
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
that one event-loop tick takes.

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

### Embedding one screen permanently under another's content can blow the combined minimum-size budget — even collapsed

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

**Rule:** the next time a screen needs to draw "N same-shaped shaded regions along a plot's time
axis" (event markers, segment colouring, more breath-like overlays), reach for this
`BreathSpansItem` pattern — one item per plot with an internal list — not a `pg.LinearRegionItem`
per element. It is the difference between an O(1)-per-plot item count and an O(N) one.

### A "clear the stale error card" call must not also silence a currently-busy overlay — `BusyOverlay.stop()` conflates the two

Fixed with `BusyOverlay.clear_error()`: only `stop()`s (hides) the overlay if `self.error is not
None`; otherwise a no-op, so a legitimately busy overlay is left exactly as it was.
`_render_preview_stage1` (see the next entry) uses `clear_error()`, not
`_clear_panel_overlays`/`.stop()`.

**Rule:** anywhere in this screen (or a future one with the same busy/error overlay pattern) that
needs to "dismiss a stale error before drawing", use `clear_error()`. Reach for
`_clear_panel_overlays`/`.stop()` only where a WHOLE panel set is being reset from scratch (a file
switch, an invalid-settings blank) and nothing — busy or not — should survive that reset.

### Splitting a synchronous render across `QTimer.singleShot(0, ...)` needs its OWN staleness guard, distinct from the job-token check that already exists

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

### A pyqtgraph axis label that picks its own wording must make `labelString()` return the pick, and "hide when it does not fit" is not a fallback on a screen whose job is to name the channel (06-09-2026)

`AxisItem._updateLabel()` re-renders the label from `labelString()` on every range
change (`setRange` → `updateAutoSIPrefix`), inside `showLabel(True)`, and from
`setLabel`/`enableAutoSIPrefix`. A fit-picker that swaps the label's HTML directly
(`label.setHtml(...)`) while `labelString()` still returns the full wording is undone
the moment any of those run: 14.3's `SciAxis` picked "name alone" and had it overwritten
inside the very `showLabel(True)` that confirmed the pick (measured: a 94 px
"Poes (cmH₂O)" back on a 76 px axis, the overrun the picker existed to stop). `_FitAxis`
never had the problem because it picks via `setLabel(text)`, so `labelText` IS the pick.
`SciAxis` now keeps the pick in state (`_include_unit`/`_label_size`) that
`labelString()` reads, and overrides `_updateLabel` to re-pick, since the SI scale it
may just have changed is part of the wording's width.

Second lesson, the one that turned Windows CI red: "name + unit, then name, then
nothing" is platform-dependent at the stack's 96 px row floor, because the Windows
runner's font is ~1.5x wider than macOS's (the axis is 76 px there and "Volume" alone
measures ~92 px on Windows, ~60 px on Linux). On the mechanics stack a blank axis is a
defect (the screen exists to confirm the channel assignment), so the picker now shrinks
the name's font towards `_MIN_LABEL_SCALE` of the base before it hides anything, and a
shortened wording keeps the `·10ⁿ` annotation while the ticks are scaled (dropping it
would leave "500" meaning 0.5 L with nothing on screen saying so; pyqtgraph itself
only pins the scale at 1.0 while the label is fully hidden).

Third lesson, from the follow-up that went red on macOS: **`windows_metrics` stacks its
1.45x on top of whatever the runner's own font already measures**, so under it "Volume"
is ~108 px on the macOS runner and ~130 px on the Windows runner, both below any
legible floor for a 76 px axis, while no shipped platform is that wide (Windows itself:
~92 px). A pixel-tight geometry therefore cannot demand *visibility* under the fixture.
The split that holds on all three runners: the unmodelled per-platform test
(`test_mechanics_channel_stack_is_x_aligned`) asserts every channel names itself, and the
modelled one (`..._labels_fit_or_hide_for_cause_in_windows_metrics`) asserts the
mechanism: a shown label is never wider than its axis, and a hidden one is hidden only
because the name at the smallest allowed font (`SciAxis._label_sizes()`) is wider still.
Never a pixel literal in either.

### Item-level click vs scene-signal click: two different pyqtgraph mechanisms for two different buttons (M-20)

`BreathSpansItem`'s right-click/Ctrl+left-click "request a breath-type menu" primitive
and the plain-left-click primitive (`_on_plot_clicked`/`_toggle_from_emg_click`, wired to
`scene().sigMouseClicked`; since the single-click-selects change it MARKS a breath via
`_select_breath`, exclusion lives in the right-click menu) look
like the same kind of thing but are resolved through genuinely different pyqtgraph
machinery, and mixing them up produces a menu that never opens or a ViewBox context
menu that never goes away.

**The plain left-click toggle is scene-level and always fires.** `GraphicsScene.
sendClickEvent` calls `self.sigMouseClicked.emit(ev)` unconditionally at the end,
regardless of which item's (if any) `mouseClickEvent` accepted the event first — so a
handler connected to that signal (as this app's toggle handlers are) sees EVERY click,
and must check `ev.isAccepted()`/`ev.button()` itself to ignore what it doesn't want.

**The right-click-for-a-menu primitive is item-level, and item-level resolution order
is NOT what it looks like.** `GraphicsScene.itemsNearEvent` sorts candidate items by
their absolute z-value (each item's own `zValue()` summed up its `parentItem()` chain),
descending. Measured directly (`pyqtgraph.graphicsItems.ViewBox.ViewBox` itself has
`zValue() == -100`): a `BreathSpansItem` painted at its usual `zValue(-10)` (so its
translucent breath fill stays visually BEHIND the channel traces) has an absolute z of
`-110` — BELOW the ViewBox it sits inside, which is itself an eligible click candidate
with `mouseClickEvent` (it accepts a right-click to raise its own context menu,
`menuEnabled()` permitting). The naive fix — raise the item's zValue so it is checked
before ViewBox — collides with the paint requirement, since raising it to 0 (ViewBox's
threshold) makes it paint on top of same-z-value trace curves instead of behind them.

**The fix pyqtgraph itself provides for exactly this ambiguity is `HoverEvent.
acceptClicks(button)`**, documented on `HoverEvent` in `pyqtgraph/GraphicsScene/
mouseEvents.py`: an item's `hoverEvent()` can claim a SPECIFIC button ahead of the
actual click, and `GraphicsScene.sendClickEvent` checks that claim FIRST — if claimed,
the item's own `mouseClickEvent` is called directly, and the whole z-ordered
`itemsNearEvent` loop (where ViewBox would otherwise win) never runs at all for that
button. `BreathSpansItem.hoverEvent` claims `Qt.RightButton` only when the hover
position is over an actual breath span (never a gap, so a right-click that misses every
span still reaches ViewBox's own menu unclaimed, unmodified zValue and all). Ctrl+left-
click needs no such claim: `ViewBox.mouseClickEvent` never accepts the left button
regardless of modifiers, so the ordinary z-ordered fallback already reaches
`BreathSpansItem` for that button without any hover trick — verified empirically (a
small offscreen `pg.PlotWidget` + simulated hover/press/release), not assumed from
reading pyqtgraph's source alone; the class docstring has the exact measured numbers.

**Rule for any future "claim a button ahead of a competing item" need on a pyqtgraph
item:** reach for `HoverEvent.acceptClicks`, not for a zValue fight — it is the
documented mechanism for precisely this, and it decouples click-priority from paint
order, which a zValue change never can.

### A transient popup `QMenu` needs `Qt.WA_DeleteOnClose` or it never gets cleaned up

`_MechanicsMixin._build_type_menu` (M-20) constructs a brand-new `QMenu` on every
right-click/Ctrl+left-click (parented to `self.plots`, so `_lone_ampersands`'s
`findChildren(QMenu)` scan reaches it — an unparented menu is invisible to that scan).
Popped up with `.popup()`, not `.exec()`, matching pyqtgraph's own `ViewBox.
raiseContextMenu` convention (non-blocking; the choice is handled via each action's
`triggered` signal instead of `.exec()`'s blocking return value). Without
`setAttribute(Qt.WA_DeleteOnClose)` the closed menu is never destroyed — Qt does not
garbage-collect a widget just because it lost focus or hid — so a long interactive
session accumulates one dead `QMenu` QObject per right-click, forever. The same
`WA_DeleteOnClose` gotcha this file already documents for a one-shot `QDialog` applies
identically here.

### One click-menu-building code path serves two mutually-exclusive tabs — state it caches must be reset in BOTH renderers (M-31)

`_build_type_menu`/`_handle_type_requested` are shared between the flow-bearing
Mechanics stack and the EMG-only 'EMG – segments' tab (M-26's own docstring on
`_draw_breath_overlays` explains why: the two never render breaths at once, so sharing
the overlay/click machinery is safe). M-31 added `self._suggested_fvc` — the breath
number `core.analysis.manoeuvres.suggest_fvc()` points at, read by the menu's disabled
"Suggested: FVC" hint — as more such shared, cached state, and it has the SAME trap
`_breath_spans`/`_breath_regions`/etc. already had before this ticket touched them: it
is only ever WRITTEN by `_render_preview_stage1` (Mechanics) and `_render_segments_
preview` (Segments), never by a shared reset routine both paths funnel through. Setting
it in one and forgetting the other silently lets a suggestion computed against one
file/tab's breath numbering survive into a DIFFERENT file/tab's menu, where the number
means something else entirely — the failure is not a crash, just a hint pointing at the
wrong breath, which is easy to miss in review since neither render path errors. Any
FUTURE per-file/per-tab cache added to this shared machinery needs to be written (or
explicitly cleared) in every renderer that can leave it stale, not just the one whose
ticket happens to introduce it — grep both `_render_preview_stage1` (`_mechanics.py`)
and `_render_segments_preview` (`_segments.py`) before assuming one reset site is
enough.

### Preview's tabs, jobs and menus are driven by the signal-set shape, not by the channels alone

Everything that asks "does this analysis have flow / Poes / EMG, and is it EMG-only?" reads a
`core.analysis.signals.Capabilities` (built with `Capabilities.from_settings_or_none` on any
UI path that runs before validation, so a hand-edited bare-string `analysis.signals` degrades
the render instead of crashing the window). Three places consume it and must agree:

- **The sub-tab bar**: `subtab_plan(caps)` in `screens/preview/_emg_noise.py` returns the
  ordered `(widget, title)` pairs: Mechanics or, for an EMG-only set, EMG – segments first,
  then the ECG and noise tabs whenever `caps.emg`. `_update_subtabs` inserts and removes the
  existing widgets; never recreate them (cleanup-contract tests hold their identity).
- **Which preview jobs a settings edit re-runs**: `_kinds_for_settings_path(path, caps=None)` in
  `screens/preview/_jobs.py`. `caps=None` keeps the flow-bearing rule; under an EMG-only set an
  EMG channel or `processing.emg.*` edit also re-dispatches `batch` and `segments`, because
  the test run's mechanics are built from the EMG channels there. A path it does not classify
  falls through to ALL kinds: erring wide only costs a recompute, erring narrow leaves a stale
  panel. `_schedule` gates the `segments` job to `caps.mode == 'emg_only'`, mirroring the
  tab plan, so the two can never disagree about which shape gets which preview.
- **The breath menu** (built in `_mechanics.py`, reused by the segments tab): typing a breath and the "Use as IC
  reference for" submenu are item-level clicks on `BreathSpansItem` (see the item-level vs
  scene-signal section above); the reference picker dialog (`reference_picker_dialog.py`) is
  settings-only and reads `processing.breath_types`, never a recording.

Relevance decides whether a card or tab EXISTS for this shape (`_cond_cards` /
`_apply_card_visibility` in `screens/settings_screen.py`); the gating rule still decides
whether an ACTION is enabled. Keep the two apart: hiding a surface because it does not apply is
fine, hiding one because a precondition is unmet is not.

Every new caption, including a new group box like *Subjects && lung volumes*, is covered by the
lone-ampersand guard only if its dialog or window is built inside the existing scan in
`tests/unit/test_ui_wording.py` (or its own test calling `_lone_ampersands(dlg)`, as
`test_reference_picker.py` does).
