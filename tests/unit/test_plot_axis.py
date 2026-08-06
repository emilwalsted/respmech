"""``ui/plot_axis.py`` — extracted (ticket B05) from Preview & QC's ``_FitAxis`` so
``ColumnStack``'s own short rows get the same tick-thinning behaviour. The climb-terminates
guard already lived in ``test_compact_plots.py`` against ``_FitAxis``/``_next_nice`` and is
NOT duplicated here — this file covers what is NEW: the base class itself, and that
``_FitAxis`` still behaves identically now that it subclasses it.
"""
import math

from respmech.ui.plot_axis import MinPitchAxis, _next_nice


def test_fit_axis_is_a_min_pitch_axis():
    """Preview's own axis must come out the other side of the extraction unchanged in
    kind, not just in behaviour — a future caller relying on isinstance() should see this."""
    from respmech.ui.screens.preview_screen import _FitAxis
    assert issubclass(_FitAxis, MinPitchAxis)


def test_next_nice_is_the_same_function_both_modules_see():
    """``ui/screens/preview/_plot_helpers.py`` re-exports ``_next_nice`` for
    ``preview_screen.py`` and ``preview/__init__.py``, which still do
    ``from ._plot_helpers import _next_nice`` — it must be the SAME function object, not a
    second copy that could drift from this one."""
    from respmech.ui.screens.preview_screen import _next_nice as reexported
    assert reexported is _next_nice


def test_a_short_axis_thins_its_ticks(qapp):
    """The behaviour itself, directly on the base class rather than through ``_FitAxis``,
    on the exact span/size already proven in test_compact_plots.py's
    test_the_tick_ladder_always_terminates (major spacing 0.5 at height 63 px) — a real,
    previously-measured number rather than a plausible-looking guess. Self-review fix
    (06-08-2026): an earlier version compared two THINNED sizes to each other and passed
    even when both happened to floor at the same spacing (this exact span does, at 43 vs
    400 px) — a vacuous, non-discriminating assertion. Comparing against a PLAIN,
    unmodified pyqtgraph axis at the same short height is what actually tells a working
    MinPitchAxis apart from a no-op: the plain ladder must pick something FINER."""
    import pyqtgraph as pg

    ax = MinPitchAxis(orientation="left")
    ax.setHeight(63)
    assert ax.tickSpacing(-0.429, 0.638, 63.0)[0][0] == 0.5

    plain = pg.AxisItem(orientation="left")
    plain.setHeight(63)
    plain_major = plain.tickSpacing(-0.429, 0.638, 63.0)[0][0]
    assert plain_major < 0.5, (
        f"the unmodified ladder picked {plain_major}, not finer than the thinned 0.5 — "
        f"this span/size no longer discriminates a working MinPitchAxis from a no-op")


def test_a_tall_axis_is_left_alone(qapp):
    """With room to spare, thinning must not kick in at all — the base pyqtgraph ladder
    already fits, and MinPitchAxis's job is to intervene only when it does not."""
    ax = MinPitchAxis(orientation="left")
    ax.setHeight(400)
    import pyqtgraph as pg
    plain = pg.AxisItem(orientation="left")
    plain.setHeight(400)
    assert ax.tickSpacing(0, 100, 400.0) == plain.tickSpacing(0, 100, 400.0)


def test_next_nice_never_hangs_near_the_subnormal_floor():
    """Mirrors the guard in test_compact_plots.py's test_the_tick_ladder_always_terminates,
    against the function's new home directly."""
    for e in range(-320, 300):
        for m in (1.0, 1.5, 2.0, 3.0, 4.999, 5.0, 7.0, 9.99):
            v = m * 10.0 ** e
            if v <= 0 or math.isinf(v):
                continue
            nxt = _next_nice(v)
            if v > 3.3e-311:
                assert nxt > v, f"_next_nice({v!r}) returned {nxt!r} — the ladder cannot climb"
    assert _next_nice(5e-324) == 5e-324, "a subnormal input must be returned, not raise"
