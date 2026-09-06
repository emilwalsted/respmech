# CLAUDE.md — `src/respmech/ui/`

Qt/GUI gotchas for the PySide6 app: layout and font-metric budgets, styling, worker
threads and queued signals, deferred rendering, and pyqtgraph. The project-wide rules
stay in the repo-root `CLAUDE.md`; test-side hazards are in `tests/CLAUDE.md`.

### Chips and `FlowLayout`: where a row of controls can actually wrap

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
