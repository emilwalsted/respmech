"""PEEPi in the UI and the figures: the Setup card, the hatched rectangle on the Campbell
diagram (written figure and Preview panel), the Preview re-dispatch rule and the run report.

The rectangle spans the tidal volume (EELV to EILV) and rises above the end-expiratory Poes by
the amount of the pre-flow deflection the elastic-recoil polygon does not already hold, so its
area is the threshold work ``wob_in_thr``. A figure without the feature must stay exactly what
it always was: no patch, no legend entry.
"""
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PySide6.QtWidgets import QApplication

from respmech.core import plots
from respmech.core.analysis import pressure
from respmech.ui.state import AppState

from _helpers import requires_synth, synth_settings  # noqa: F401

_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


def _rects(ax):
    return [p for p in ax.patches if isinstance(p, Rectangle)]


# -- the height a breath contributes ------------------------------------------
def test_height_is_none_without_the_feature_or_a_positive_finite_value():
    assert pressure.peepi_rectangle_height({}) is None
    assert pressure.peepi_rectangle_height({"peepi_added": None}) is None
    assert pressure.peepi_rectangle_height({"peepi_added": float("nan")}) is None
    assert pressure.peepi_rectangle_height({"peepi_added": 0.0}) is None
    assert pressure.peepi_rectangle_height({"peepi_added": "x"}) is None
    assert pressure.peepi_rectangle_height({"peepi_added": 1.25}) == 1.25


def test_mean_height_ignores_breaths_without_one():
    bs = [{"peepi_added": 1.0}, {}, {"peepi_added": 3.0}, {"peepi_added": float("nan")}]
    assert plots.mean_peepi_rectangle_height(bs) == pytest.approx(2.0)
    assert plots.mean_peepi_rectangle_height([{}, {}]) is None


def test_mean_height_counts_a_breath_with_nothing_to_add_as_zero():
    """Two of ten breaths with 4 cmH2O and eight with none draw 0.8, matching the mean of
    ``wob_in_thr``, not 4 (the mean over only the breaths that happen to have PEEPi)."""
    bs = [{"peepi_added": 4.0}] * 2 + [{"peepi_added": 0.0}] * 8
    assert plots.mean_peepi_rectangle_height(bs) == pytest.approx(0.8)
    assert plots.mean_peepi_rectangle_height([{"peepi_added": 0.0}]) == 0.0
    assert pressure.peepi_rectangle_height({"peepi_added": 0.0}) is None, "nothing to draw at 0"


# -- drawing --------------------------------------------------------------------
def test_rectangle_spans_the_tidal_volume_above_end_expiratory_poes():
    ax = Figure().add_subplot(111)
    plots.draw_peepi_rectangle(ax, eilv=[0.9, -6.0], eelv=[0.1, -1.0], height=2.0)
    (r,) = _rects(ax)
    assert r.get_x() == pytest.approx(0.1) and r.get_width() == pytest.approx(0.8)
    assert r.get_y() == pytest.approx(-1.0) and r.get_height() == pytest.approx(2.0)
    assert r.get_hatch()


@pytest.mark.parametrize("height", [None, 0.0, -1.0, float("nan")])
def test_no_rectangle_without_a_positive_height(height):
    ax = Figure().add_subplot(111)
    plots.draw_peepi_rectangle(ax, [0.9, -6.0], [0.1, -1.0], height)
    assert not ax.patches


def test_no_rectangle_for_an_unusable_endpoint_pair():
    ax = Figure().add_subplot(111)
    plots.draw_peepi_rectangle(ax, None, [0.1, -1.0], 1.0)
    plots.draw_peepi_rectangle(ax, [0.9, -6.0], [0.1, float("nan")], 1.0)
    assert not ax.patches


def test_recoil_and_polygon_is_unchanged_without_a_height_and_adds_the_rectangle_first():
    ref = Figure().add_subplot(111)
    plots._recoil_and_polygon(ref, [0.9, -6.0], [0.1, -1.0])
    same = Figure().add_subplot(111)
    plots._recoil_and_polygon(same, [0.9, -6.0], [0.1, -1.0], peepi_height=None)
    assert len(same.patches) == len(ref.patches) == 1 and not _rects(same)

    withp = Figure().add_subplot(111)
    plots._recoil_and_polygon(withp, [0.9, -6.0], [0.1, -1.0], peepi_height=1.5)
    assert len(withp.patches) == 2
    assert isinstance(withp.patches[0], Rectangle), "the rectangle is drawn BEFORE the polygon"


# -- the written figure, end to end ----------------------------------------------
@requires_synth()
def test_figure_run_is_unaffected_with_the_feature_off_and_carries_the_rectangle_when_on(tmp_path, monkeypatch):
    from respmech.core.pipeline import run_batch
    s = synth_settings(tmp_path)
    off = run_batch(s)
    for fr in off.ok_files.values():
        assert all("peepi_added" not in b for b in fr.breaths.values())
    s.processing.pressure.peepi.enabled = True
    on = run_batch(s)
    any_height = False
    for fr in on.ok_files.values():
        kept = [b for b in fr.breaths.values() if not b.get("ignored")]
        # every breath that got PEEPi columns also carries a rectangle height (or none)
        for b in kept:
            if "pressure_ext" in b and np.isfinite(b["pressure_ext"]["wob_in_thr"]):
                h = b.get("peepi_added")
                assert h is not None and h >= 0
                any_height = any_height or h > 0
    fname, fr = next(iter(on.ok_files.items()))
    drawn = []
    real = plots.draw_peepi_rectangle
    monkeypatch.setattr(plots, "draw_peepi_rectangle",
                        lambda ax, eilv, eelv, height, **kw: (drawn.append(height),
                                                              real(ax, eilv, eelv, height, **kw)))
    p = plots._pv_average(fr, fname, str(tmp_path / "avg.pdf"))
    assert p and (tmp_path / "avg.pdf").stat().st_size > 0
    # the figure asked for exactly the mean height of the breaths it draws
    mean_h = plots.mean_peepi_rectangle_height(plots._breaths(fr))
    assert drawn == [mean_h]
    # the individual grid asks once per drawn breath, each with that breath's own height
    drawn.clear()
    plots._pv_individual(fr, fname, str(tmp_path / "ind.pdf"), 2, 2)
    kept = [b for b in plots._ordered(fr) if not b.get("ignored")]
    assert drawn == [pressure.peepi_rectangle_height(b) for b in kept]


# -- Preview re-dispatch ---------------------------------------------------------
def test_a_pressure_setting_only_redispatches_the_batch_test_run():
    from respmech.ui.screens.preview import _kinds_for_settings_path
    for path in ("processing.pressure.peepi.enabled", "processing.pressure.peepi.min_deflection",
                 "processing.pressure.peepi.search_window_s"):
        assert _kinds_for_settings_path(path) == frozenset({"batch"}), path


@requires_synth()
def test_preview_overlay_draws_the_rectangle_only_with_a_height(tmp_path):
    from respmech.ui import theme
    from respmech.ui.screens.preview import PreviewScreen
    pal = theme._PLOT_LIGHT
    base = {"ignored": False, "volumeavg": [0.1, 0.9], "poesavg": [-1.0, -6.0],
            "eelvavg": [0.1, -1.0], "eilvavg": [0.9, -6.0], "inspiration": {}, "wob": {}}
    ax = Figure().add_subplot(111)
    PreviewScreen._overlay_campbell_work(None, ax, [dict(base)], pal)
    assert not _rects(ax)
    ax = Figure().add_subplot(111)
    PreviewScreen._overlay_campbell_work(None, ax, [dict(base, peepi_added=2.0)], pal)
    (r,) = _rects(ax)
    assert r.get_height() == pytest.approx(2.0) and r.get_x() == pytest.approx(0.1)


# -- run report ---------------------------------------------------------------------
@requires_synth()
def test_run_report_names_the_peepi_source_only_when_the_feature_is_on(tmp_path):
    from respmech.core.pipeline import run_batch
    from respmech.core.io import writers

    def _report(enabled):
        s = synth_settings(tmp_path)
        s.processing.pressure.peepi.enabled = enabled
        result = run_batch(s)
        path = writers._write_run_report(result, s, str(tmp_path), [], None)
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    assert "PEEPi source" not in _report(False)
    on = _report(True)
    assert "PEEPi source:" in on
    # the synthetic input carries Pgas, so the gastric-corrected value feeds the threshold work
    assert "PEEPi source:            corrected" in on


# -- the Setup card ---------------------------------------------------------------------
def _screen(qapp, tmp_path, *, poes=True):
    from respmech.ui.screens.settings_screen import SettingsScreen
    s = synth_settings(str(tmp_path), data_out=_OUT)
    if not poes:
        s.analysis.signals = ["flow"]
        s.input.channels.poes = None
        s.input.channels.pgas = None
        s.input.channels.pdi = None
    sc = SettingsScreen(AppState(s))
    sc.show()
    qapp.processEvents()
    return sc


def _card(sc):
    return next(c for c, _pred in sc._cond_cards if c.title() == "Intrinsic PEEP (PEEPi)")


@requires_synth()
def test_card_is_hidden_in_flow_only_and_shown_with_poes(qapp, tmp_path):
    sc = _screen(qapp, tmp_path, poes=False)
    card = _card(sc)
    assert not card.isVisible()
    s = sc.state.settings
    s.analysis.signals = ["flow", "poes"]
    s.input.channels.poes = 2
    sc._update_disclosure()
    qapp.processEvents()
    assert card.isVisible()
    s.analysis.signals = ["flow"]
    s.input.channels.poes = None
    sc._update_disclosure()
    qapp.processEvents()
    assert not card.isVisible()


@requires_synth()
def test_card_is_visible_from_the_start_with_poes(qapp, tmp_path):
    assert _card(_screen(qapp, tmp_path)).isVisible()


@requires_synth()
def test_fields_round_trip_and_are_greyed_while_off(qapp, tmp_path):
    sc = _screen(qapp, tmp_path)
    assert not sc.peepi_enabled.isChecked()
    assert not sc.peepi_window.isEnabled() and not sc.peepi_min.isEnabled()
    sc.peepi_enabled.setChecked(True)
    assert sc.peepi_window.isEnabled() and sc.peepi_slope.isEnabled()
    sc.peepi_window.setValue(0.8)
    sc.peepi_smooth.setValue(0.04)
    sc.peepi_slope.setValue(0.2)
    sc.peepi_min.setValue(0.7)
    out = sc.to_state().processing.pressure.peepi
    assert out.enabled is True
    assert (out.search_window_s, out.smooth_s, out.onset_slope_frac, out.min_deflection) == \
        pytest.approx((0.8, 0.04, 0.2, 0.7))
    sc.state.settings.validate()      # the ranges the spin boxes allow are all valid


@requires_synth()
def test_loading_an_analysis_fills_the_card(qapp, tmp_path):
    from respmech.ui.screens.settings_screen import SettingsScreen
    s = synth_settings(str(tmp_path), data_out=_OUT)
    pp = s.processing.pressure.peepi
    pp.enabled, pp.search_window_s, pp.min_deflection = True, 0.6, 1.25
    sc = SettingsScreen(AppState(s))
    assert sc.peepi_enabled.isChecked()
    assert sc.peepi_window.value() == pytest.approx(0.6)
    assert sc.peepi_min.value() == pytest.approx(1.25)


@requires_synth()
@pytest.mark.parametrize("name,value", [("peepi_window", 0.7), ("peepi_smooth", 0.08),
                                         ("peepi_slope", 0.3), ("peepi_min", 1.0)])
def test_editing_a_threshold_marks_the_analysis_modified(qapp, tmp_path, name, value):
    sc = _screen(qapp, tmp_path)
    sc._mark_clean()
    getattr(sc, name).setValue(value)
    assert sc.is_dirty(), name


@requires_synth()
def test_toggling_the_switch_marks_the_analysis_modified(qapp, tmp_path):
    sc = _screen(qapp, tmp_path)
    sc._mark_clean()
    sc.peepi_enabled.setChecked(True)
    assert sc.is_dirty()


@requires_synth()
def test_the_card_holds_no_lone_ampersand_and_carries_tooltips(qapp, tmp_path):
    sc = _screen(qapp, tmp_path)
    card = _card(sc)
    assert "&" not in card.title().replace("&&", "")
    for w in (sc.peepi_enabled, sc.peepi_window, sc.peepi_smooth, sc.peepi_slope, sc.peepi_min):
        assert "processing.pressure.peepi." in w.toolTip(), w


@requires_synth()
def test_card_fits_under_windows_font_metrics(qapp, tmp_path, windows_metrics):
    """Acceptance criterion: no card text clipped at the wider Windows advance."""
    sc = _screen(qapp, tmp_path)
    sc.resize(1000, 900)
    for _ in range(6):
        qapp.processEvents()
    card = _card(sc)
    assert card.isVisible()
    for w in (sc.peepi_window, sc.peepi_smooth, sc.peepi_slope, sc.peepi_min):
        # the suffix (" cmH₂O"/" s") and the value both fit: the spin box asks for at least
        # what its own content needs
        assert w.width() >= w.minimumSizeHint().width(), w.toolTip()[:60]
    assert card.width() >= card.minimumSizeHint().width()
    assert card.height() >= card.minimumSizeHint().height()


@requires_synth()
def test_a_malformed_signal_set_hides_the_card_instead_of_crashing(qapp, tmp_path):
    sc = _screen(qapp, tmp_path)
    sc.state.settings.analysis.signals = "flow"          # a hand-edited bare string
    sc._update_disclosure()
    qapp.processEvents()
    assert not _card(sc).isVisible()


# -- Preview Mechanics ▸ Advanced… ▸ PTP note ------------------------------------------
@requires_synth()
@pytest.mark.parametrize("on", [True, False])
def test_advanced_ptp_names_the_preflow_column_only_while_peepi_is_on(qapp, tmp_path, monkeypatch, on):
    from PySide6.QtWidgets import QDialog
    from respmech.ui import advanced_dialog as ad
    from respmech.ui.screens.preview_screen import PreviewScreen
    s = synth_settings(str(tmp_path), data_out=_OUT)
    s.processing.pressure.peepi.enabled = on
    pv = PreviewScreen(AppState(s))
    pv._refresh_files()
    seen = {}

    class _Stub(ad.AdvancedDialog):
        def exec(self):
            note = self.note("baseline_window_s")
            seen["text"] = None if note is None else note.text()
            return QDialog.Rejected
    monkeypatch.setattr(ad, "AdvancedDialog", _Stub)
    pv._open_mech_advanced()
    if on:
        assert "int_oes_preflow" in seen["text"] and "never added to PTP" in seen["text"]
    else:
        assert seen["text"] is None
    pv.shutdown()


@requires_synth()
def test_an_untouched_hand_edited_value_is_never_rewritten_by_the_card(qapp, tmp_path):
    """A value outside a box's range or precision (hand-edited TOML) is clipped/rounded on
    DISPLAY. Editing some other field must not write that display back into the analysis."""
    from respmech.ui.screens.settings_screen import SettingsScreen
    s = synth_settings(str(tmp_path), data_out=_OUT)
    pp = s.processing.pressure.peepi
    pp.search_window_s, pp.onset_slope_frac, pp.min_deflection = 0.025, 0.005, 60.0
    sc = SettingsScreen(AppState(s))
    sc.group_regex.setText("case_(.)")
    out = sc.to_state().processing.pressure.peepi
    assert (out.search_window_s, out.onset_slope_frac, out.min_deflection) == (0.025, 0.005, 60.0)
    sc.peepi_min.setValue(1.5)                     # an actual edit IS written
    out = sc.to_state().processing.pressure.peepi
    assert out.min_deflection == 1.5 and out.search_window_s == 0.025


def test_cohort_grid_draws_each_files_own_mean_height(tmp_path, monkeypatch):
    def _breath(name, h):
        return {"ignored": False, "filename": name, "volume": np.linspace(0, 1, 5),
                "poes": np.linspace(0, -5, 5), "volumeavg": np.linspace(0, 1, 5),
                "poesavg": np.linspace(0, -5, 5), "eilvavg": [1.0, -5.0], "eelvavg": [0.0, 0.0],
                "peepi_added": h}
    frs = {n: type("FR", (), {"breaths": {1: _breath(n, h)}})() for n, h in (("a", 1.0), ("b", 3.0))}
    result = type("R", (), {"ok_files": frs})()
    drawn = []
    real = plots.draw_peepi_rectangle
    monkeypatch.setattr(plots, "draw_peepi_rectangle",
                        lambda ax, eilv, eelv, height, **kw: (drawn.append(height),
                                                              real(ax, eilv, eelv, height, **kw)))
    assert plots._pv_cohort(result, str(tmp_path / "c.pdf"), 2, 1)
    assert sorted(drawn) == [1.0, 3.0]
