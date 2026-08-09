"""D14 (UI-overhaul): the Mechanics sub-tab must be usable on a 1280x720 screen.

Two related fixes, both layout/persistence only — no computed value changes:

1. ``theme.set_stack_floor`` used to floor the five-channel stack at a flat
   ``rows * PLOT_ROW_MIN_HEIGHT`` (480 px) regardless of the window. On a 1280x720 screen
   that pinned the vertical splitter's lower half at ITS OWN floor with zero pixels of
   travel, and the QC verdict + its two actions sat at the very bottom of the scrolled
   page — below the fold on exactly this screen. See ``theme.set_stack_floor`` and
   ``_MechanicsMixin._build_mech_action_band`` for the full story.
2. The vertical splitter's chosen size now persists across sessions (``ui.prefs``).

The QC verdict + its two actions ARE fully freed from scrolling (they moved out of the
scrolled page entirely). The channel stack + Campbell/table half are NOT fully freed from
scrolling at the exact 1280x720 bug-report size, measured in this harness — the table/
Campbell panel's own pre-existing floor plus fixed window chrome already exceed that
viewport on their own, independently of the stack floor this ticket targets. What IS proven
here is that the stack-floor fix meaningfully reduces the scroll requirement and gives the
splitter real, demonstrable travel once the viewport has any slack to redistribute — see
each test's own docstring for exactly which claim it makes and why.

Every assertion here is a RELATIONSHIP (fits within the viewport, smaller than the full
floor, grew when asked, less scroll than the old floor needed), never a pixel literal —
macOS has the narrowest font metrics this app ships to, and a pixel literal read off one
platform is a measurement of its fonts, not of the layout (see RespMech's CLAUDE.md and
test_window_fits_screen.py)."""
import os

from respmech.ui.state import AppState

from _helpers import INPUT, requires_synth, synth_settings

pytestmark = requires_synth()

# The tightest configuration this app is expected to fit, per test_window_fits_screen.py.
_TARGET_W, _TARGET_H = 1280, 720


def _window(qapp, tmp_path, size=(_TARGET_W, _TARGET_H)):
    from respmech.ui.main_window import MainWindow
    s = synth_settings(str(tmp_path))
    win = MainWindow(AppState(s))
    win.resize(*size)
    win.show()
    for _ in range(8):
        qapp.processEvents()
    win.tabs.setCurrentIndex(1)
    pv = win.preview_screen
    pv.subtabs.setCurrentIndex(pv.subtabs.indexOf(pv._mech_tab))
    for _ in range(8):
        qapp.processEvents()
    return win, pv, s


def _render_mech(pv, s):
    from respmech.ui.workers import stage_mechanics_preview
    pv._refresh_files(); pv.file_rail.select_filename("synth_case_A.csv")
    pv._render_preview(stage_mechanics_preview(s, os.path.join(INPUT, "synth_case_A.csv")))


def test_mech_action_band_fits_without_scrolling_at_1280x720(qapp, tmp_path):
    """The QC verdict + 'Process & write' + 'Export Campbell…' must be visible without
    scrolling, on the screen the original bug report was measured on."""
    win, pv, s = _window(qapp, tmp_path)
    _render_mech(pv, s)
    for _ in range(8):
        qapp.processEvents()

    assert pv._mech_action_band.isVisible(), "the action band is hidden on the Mechanics tab"

    band_bottom = pv._mech_action_band.mapToGlobal(
        pv._mech_action_band.rect().bottomLeft()).y()
    window_bottom = win.mapToGlobal(win.rect().bottomLeft()).y()
    assert band_bottom <= window_bottom, (
        f"the action band's bottom edge ({band_bottom}) sits below the window's own bottom "
        f"edge ({window_bottom}) at a {_TARGET_W}x{_TARGET_H} window — it needs scrolling "
        f"to reach")
    win.close()


def test_mech_action_band_hides_on_other_subtabs_and_returns_on_mechanics(qapp, tmp_path):
    """The band is Mechanics-only (its own docstring: 'the ECG/EMG sub-tabs have no
    equivalent band') — wired via ``subtabs.currentChanged``, not asserted anywhere else.
    The canonical synth settings wire EMG channels by default, so the ECG/noise sub-tabs
    exist to switch to."""
    win, pv, s = _window(qapp, tmp_path)
    _render_mech(pv, s)
    for _ in range(8):
        qapp.processEvents()
    assert pv.subtabs.count() > 1, "no second sub-tab to switch to — EMG channels missing?"
    assert pv._mech_action_band.isVisible()

    other = next(i for i in range(pv.subtabs.count()) if pv.subtabs.widget(i) is not pv._mech_tab)
    pv.subtabs.setCurrentIndex(other)
    for _ in range(6):
        qapp.processEvents()
    assert not pv._mech_action_band.isVisible(), (
        "the action band stayed visible on a non-Mechanics sub-tab")

    pv.subtabs.setCurrentIndex(pv.subtabs.indexOf(pv._mech_tab))
    for _ in range(6):
        qapp.processEvents()
    assert pv._mech_action_band.isVisible(), (
        "the action band did not return when switching back to Mechanics")
    win.close()


def test_stack_floor_recomputes_on_a_live_resize_of_an_already_shown_window(qapp, tmp_path):
    """``_MechStackFloorFitter`` exists for a user resizing an ALREADY-open window — every
    other test here only exercises the initial layout (built/resized before ``show()``,
    so the event filter only ever sees the construction-time ``QEvent.Show``). This
    resizes a live, already-rendered window and checks the floor actually reacts."""
    from respmech.ui.screens.preview_screen import _CHANNELS
    from respmech.ui import theme

    win, pv, s = _window(qapp, tmp_path, size=(1280, 1400))
    _render_mech(pv, s)
    for _ in range(8):
        qapp.processEvents()
    full_floor = len(_CHANNELS) * theme.PLOT_ROW_MIN_HEIGHT
    assert pv.plots.minimumHeight() == full_floor, (
        "precondition failed: the floor is not the full, unshrunk value before the resize")

    win.resize(_TARGET_W, _TARGET_H)             # live resize of the SHOWN window
    for _ in range(10):
        qapp.processEvents()

    assert pv.plots.minimumHeight() < full_floor, (
        f"resizing an already-shown window down to {_TARGET_W}x{_TARGET_H} did not shrink "
        f"the stack floor — the live-resize reactive path is not working")
    win.close()


def test_stack_floor_gives_way_at_the_exact_bug_report_size(qapp, tmp_path):
    """The bug this ticket fixes: a flat 480 px floor left the vertical splitter's lower
    half pinned at its own floor with zero pixels of travel — checked at the exact window
    size the bug was reported on. See test_splitter_obeys_a_request_to_give_the_lower_
    half_more_room for the drag itself, which needs a viewport with some slack to show."""
    from respmech.ui.screens.preview_screen import _CHANNELS
    from respmech.ui import theme

    win, pv, s = _window(qapp, tmp_path)
    _render_mech(pv, s)
    for _ in range(8):
        qapp.processEvents()

    full_floor = len(_CHANNELS) * theme.PLOT_ROW_MIN_HEIGHT
    assert pv.plots.minimumHeight() < full_floor, (
        f"the stack floor is still the full {full_floor} px at a "
        f"{_TARGET_W}x{_TARGET_H} window — the splitter has no travel to give")
    # never below the hard per-row floor either — this is "gives way", not "collapses"
    hard_floor = len(_CHANNELS) * theme.PLOT_ROW_HARD_MIN
    assert pv.plots.minimumHeight() >= hard_floor, (
        f"the stack floor ({pv.plots.minimumHeight()} px) fell below the hard per-row "
        f"floor ({hard_floor} px) — a row is no longer readable")
    win.close()


def test_splitter_obeys_a_request_to_give_the_lower_half_more_room(qapp, tmp_path):
    """The other half of the same claim, demonstrated where it is actually observable.

    At the exact 1280x720 bug-report height the page is still taller than the viewport
    (residual scrolling — see test_stack_floor_relief_meaningfully_reduces_scrolling_at_
    1280x720 for why: the table/Campbell panel's OWN pre-existing floor plus fixed chrome
    already exceed that viewport on their own, D14 or no D14), and a scrolled page's
    ``FitScrollArea`` sizes it to EXACTLY its minimum — every splitter child sits at its
    own bare minimum with zero slack, so no drag can move anything, with or without this
    fix. A moderately taller window has genuine slack to redistribute and is where "the
    splitter regained travel" is actually visible; the floor is still proven to have given
    way (below the full, pre-fix floor) at that height too.

    Measured at this height: the DEFAULT allocation already gives the channel stack
    exactly its (now-shrunk) floor and hands the surplus to the table/Campbell half — so
    the drag direction that actually has room to give is GROWING the stack (shrinking the
    table/Campbell half down toward ITS OWN floor), not the other way around. That surplus
    itself is worth asserting: it is proof, without any drag at all, that the shrunk floor
    freed real room downstream."""
    from respmech.ui.screens.preview_screen import _CHANNELS
    from respmech.ui import theme

    win, pv, s = _window(qapp, tmp_path, size=(1280, 900))
    _render_mech(pv, s)
    for _ in range(8):
        qapp.processEvents()

    full_floor = len(_CHANNELS) * theme.PLOT_ROW_MIN_HEIGHT
    assert pv.plots.minimumHeight() < full_floor, (
        f"the stack floor is still the full {full_floor} px at a 1280x900 window")

    split = pv._mech_vsplit
    lower_widget = split.widget(1)
    before = split.sizes()
    # the default allocation already gives the table/Campbell half more than its own bare
    # minimum — direct proof the shrunk stack floor freed real room, no drag required
    assert before[1] > lower_widget.minimumSizeHint().height(), (
        f"the table/Campbell half got exactly its bare minimum "
        f"({lower_widget.minimumSizeHint().height()} px) by default — the shrunk stack "
        f"floor freed no room downstream")

    # and the room really is reachable by a drag: ask for more room for the STACK, taking
    # it from the table/Campbell half's own surplus — equivalent to a user dragging the
    # handle down
    total = sum(before)
    split.setSizes([before[0] + 60, total - (before[0] + 60)])
    for _ in range(6):
        qapp.processEvents()
    after = split.sizes()
    assert after[0] > before[0], (
        f"asking the splitter to give the channel stack more room did not move it: "
        f"{before} -> {after} — it is still pinned at its floor")
    win.close()


def test_stack_floor_unchanged_at_a_tall_viewport(qapp, tmp_path):
    """The floor must stay exactly ``rows * PLOT_ROW_MIN_HEIGHT`` — unchanged from before
    this ticket — once the viewport has room to spare."""
    from respmech.ui.screens.preview_screen import _CHANNELS
    from respmech.ui import theme

    win, pv, s = _window(qapp, tmp_path, size=(1280, 1400))
    _render_mech(pv, s)
    for _ in range(8):
        qapp.processEvents()

    full_floor = len(_CHANNELS) * theme.PLOT_ROW_MIN_HEIGHT
    assert pv.plots.minimumHeight() == full_floor, (
        f"the stack floor changed on a tall screen ({pv.plots.minimumHeight()} px against "
        f"an unchanged {full_floor} px) — large screens must be unaffected by D14")
    win.close()


def test_stack_floor_relief_meaningfully_reduces_scrolling_at_1280x720(qapp, tmp_path):
    """D14 does not claim to eliminate scrolling on the Mechanics tab at 1280x720 outright.
    Measured in this harness: the table/Campbell panel's OWN pre-existing floor (out of
    this ticket's scope — the vertical splitter's lower half; see D12 for that panel's
    width/axis/colour) plus fixed chrome (the crosshair row, tab bar, window frame) already
    exceed a 1280x720 viewport on their own, with or without this fix. What D14 DOES fix is
    the channel stack's floor, previously the single largest — and entirely avoidable —
    contributor. Proven here as a RATIO against the OLD, viewport-blind floor on the exact
    same built window, not a pixel literal: the fix must reduce the scroll requirement, not
    a specific number of pixels of it."""
    from respmech.core.pipeline import run_batch

    win, pv, s = _window(qapp, tmp_path)
    _render_mech(pv, s)
    result = run_batch(s, only_files=["synth_case_A.csv"])
    pv._on_batch_result(result)
    for _ in range(8):
        qapp.processEvents()

    assert pv.campbell.isVisible() and pv.table.isVisible() and pv.plots.isVisible()
    new_scroll = pv._mech_tab.verticalScrollBar().maximum()

    # simulate the OLD, viewport-blind floor directly, on the SAME already-built window
    from respmech.ui import theme
    from respmech.ui.screens.preview_screen import _CHANNELS
    theme.set_stack_floor(pv.plots, len(_CHANNELS))       # no viewport_height -> old behaviour
    for _ in range(8):
        qapp.processEvents()
    old_scroll = pv._mech_tab.verticalScrollBar().maximum()

    assert new_scroll < old_scroll, (
        f"the viewport-aware floor did not reduce the Mechanics tab's scroll requirement "
        f"at all at a {_TARGET_W}x{_TARGET_H} window: old {old_scroll} px, new {new_scroll} px")

    pv._update_mech_stack_floor()   # restore the real, viewport-aware floor before continuing
    for _ in range(6):
        qapp.processEvents()

    # exercise the actual interaction the layout exists to support — must not regress
    a_breath = next(iter(pv._breath_spans))
    pv._toggle_breath(a_breath)
    for _ in range(6):
        qapp.processEvents()
    assert pv._mech_tab.verticalScrollBar().maximum() <= new_scroll, (
        "excluding a breath made the tab need MORE scrolling than before the click")
    # _toggle_breath marks the analysis edited, and MainWindow.closeEvent's unsaved-changes
    # guard only fires on a VISIBLE window — hide() first, exactly like conftest's own
    # stray-window net, or close() blocks forever on a real confirmation dialog nobody is
    # there to answer.
    win.hide()
    win.close()


def test_breath_numbers_are_the_tables_first_column(qapp, tmp_path):
    """D14's acceptance criterion ('vejrtrækningsnumrene kan læses sammen med tabellens
    rækker') is already satisfied by an existing design decision, not new code: the core
    inserts ``breath_no`` at column 0 (``results.py``), and the table's own row header is
    hidden precisely because this column carries the numbering instead (see
    ``_build_mech_tab``). Locked in here so a future change on either side cannot silently
    drop it without a test noticing."""
    from respmech.core.pipeline import run_batch

    win, pv, s = _window(qapp, tmp_path)
    _render_mech(pv, s)
    result = run_batch(s, only_files=["synth_case_A.csv"])
    pv._on_batch_result(result)

    assert pv.table.verticalHeader().isVisible() is False
    # column_id() is the raw identifier (no unit/newline) — headerData()'s DisplayRole
    # deliberately renders "breath_no\n<unit>" for a fixed two-line header height.
    assert pv._table_model.column_id(0) == "breath_no"
    win.close()


def test_mech_vsplit_size_persists_across_sessions(qapp, tmp_path, isolated_prefs):
    """A user who finds a split that works keeps it — ``ui.prefs`` had no generic
    splitter-state accessor before this ticket.

    ``QSplitter.setSizes`` clamps a request against widget minimums/stretch factors, so
    what actually gets applied is not necessarily the literal numbers asked for — the
    round trip is checked against split1's own POST-clamp sizes, not the naive request."""
    win1, pv1, s = _window(qapp, tmp_path, size=(1280, 900))
    split1 = pv1._mech_vsplit
    total = sum(split1.sizes())
    # a ratio well away from the 3:2 stretch-factor default, so a pass cannot be an accident
    split1.setSizes([total // 3, total - total // 3])
    for _ in range(6):
        qapp.processEvents()
    applied = split1.sizes()
    pv1._flush_mech_split_save()          # force the debounced write instead of waiting
    win1.close()

    win2, pv2, _ = _window(qapp, tmp_path, size=(1280, 900))
    got = pv2._mech_vsplit.sizes()
    assert abs(got[0] - applied[0]) <= 4, (
        f"the vertical splitter did not restore the previously chosen split across "
        f"sessions: wanted ~{applied}, got {got}")
    win2.close()


def test_a_real_splitter_drag_is_auto_saved_via_the_debounced_signal(qapp, tmp_path, isolated_prefs):
    """The persistence test above only proves the ``prefs.save/restore_splitter_state``
    round trip — it calls ``_flush_mech_split_save()`` directly and drives the splitter
    with ``setSizes()``, which does NOT emit ``splitterMoved`` (verified: only
    ``moveSplitter()`` and a real handle drag do). So it would still pass even if
    ``split.splitterMoved.connect(self._schedule_mech_split_save)`` were deleted. This
    drives the splitter the way a real drag does and lets the actual 400 ms debounce
    timer fire (``QTest.qWait``, which spins the event loop), proving the SIGNAL path — not
    just the accessor pair — actually saves."""
    from PySide6.QtTest import QTest
    from respmech.ui.screens.preview._mechanics import _SPLIT_SAVE_DEBOUNCE_MS

    win1, pv1, s = _window(qapp, tmp_path, size=(1280, 900))
    split1 = pv1._mech_vsplit
    before = split1.sizes()
    assert not pv1._mech_split_save_pending, "a save is already pending before any drag"

    split1.moveSplitter(before[0] + 40, 1)    # the real signal-emitting API, unlike setSizes()
    for _ in range(4):
        qapp.processEvents()
    assert pv1._mech_split_save_pending, (
        "moveSplitter() did not schedule a debounced save — splitterMoved is not wired, "
        "or _schedule_mech_split_save regressed")

    QTest.qWait(_SPLIT_SAVE_DEBOUNCE_MS + 200)   # let the real debounce timer fire
    assert not pv1._mech_split_save_pending, "the debounced save never actually ran"
    dragged = split1.sizes()
    assert dragged[0] != before[0], "the drag did not actually change the split"
    win1.close()

    win2, pv2, _ = _window(qapp, tmp_path, size=(1280, 900))
    got = pv2._mech_vsplit.sizes()
    assert abs(got[0] - dragged[0]) <= 4, (
        f"the real-drag signal path did not persist the split: dragged to {dragged}, "
        f"reopened as {got}")
    win2.close()
