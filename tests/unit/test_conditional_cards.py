"""Cards whose relevance depends on the analysis itself.

Sample entropy has two parameters (embedding m, tolerance r) that mean nothing unless a
column is actually assigned to entropy. They used to sit in the "Advanced (rarely changed)"
grab-bag, which hid them from the users who DO compute entropy while still showing them to
everyone who does not. They now have their own card, shown only when it applies.

Ticket B04 retired progressive disclosure (every OTHER card is visible unconditionally, in
every mode), but this one is a different mechanic: it stays a separate registry
(_cond_cards), ANDed in its own pass by _apply_card_visibility, because "shown iff it
applies" is not the same rule as "always shown" — an entropy-less default AppState must
still hide it, in every mode, including 'open'.
"""
from PySide6.QtWidgets import QApplication

from respmech.ui.state import AppState

from _helpers import requires_synth, synth_settings  # noqa: F401

pytestmark = requires_synth()

_OUT = {"saveaveragedata": True, "savebreathbybreathdata": True}


def _screen(qapp, tmp_path, entropy=(10, 11, 12)):
    from respmech.ui.screens.settings_screen import SettingsScreen
    s = synth_settings(str(tmp_path), data_out=_OUT)
    s.input.channels.entropy = list(entropy)
    sc = SettingsScreen(AppState(s))
    sc.show()
    qapp.processEvents()
    return sc


def _card(sc, title):
    """By title, not by index — there is more than one conditional card now."""
    return next(c for c, _pred in sc._cond_cards if c.title() == title)


def _entropy_card(sc):
    return _card(sc, "Sample entropy")


def test_the_card_follows_the_channel_assignment(qapp, tmp_path):
    sc = _screen(qapp, tmp_path)
    card = _entropy_card(sc)
    assert card.isVisible()
    sc.state.settings.input.channels.entropy = []
    sc._update_disclosure()
    qapp.processEvents()
    assert not card.isVisible()
    sc.state.settings.input.channels.entropy = [10]
    sc._update_disclosure()
    qapp.processEvents()
    assert card.isVisible()


def test_it_is_hidden_from_the_start_when_no_entropy_is_assigned(qapp, tmp_path):
    sc = _screen(qapp, tmp_path, entropy=())
    assert not _entropy_card(sc).isVisible()


def test_an_unrelated_edit_does_not_un_hide_it(qapp, tmp_path):
    """The failure mode the separate registry exists to prevent: were this card's
    predicate folded into the SAME unconditional pass B04 uses for every other card, a
    predicate registered there would be overwritten by the next field change instead of
    staying gated on its own relevance."""
    sc = _screen(qapp, tmp_path, entropy=())
    card = _entropy_card(sc)
    assert not card.isVisible()
    sc.group_regex.setText("case_(.)")
    sc._on_field_changed()
    qapp.processEvents()
    assert not card.isVisible(), "an unrelated keystroke revealed an irrelevant card"

def test_open_mode_does_not_force_it_visible(qapp, tmp_path):
    sc = _screen(qapp, tmp_path, entropy=())
    sc.enter_open_mode()
    qapp.processEvents()
    assert not _entropy_card(sc).isVisible()


def test_a_card_holding_the_focused_widget_is_not_yanked_away(qapp, tmp_path):
    """Conditional cards break the "a card never retracts once shown" promise, so they get
    the one exemption that keeps it honest where it matters: the widget you are typing in
    does not vanish mid-edit."""
    sc = _screen(qapp, tmp_path)
    card = _entropy_card(sc)
    sc.ent_tol.setFocus()
    qapp.processEvents()
    if QApplication.focusWidget() is not sc.ent_tol:
        import pytest
        pytest.skip("the offscreen platform did not grant focus")
    sc.state.settings.input.channels.entropy = []
    sc._update_disclosure()
    qapp.processEvents()
    assert card.isVisible(), "the card vanished while the user was editing it"


def test_the_declared_signal_set_never_hides_the_entropy_card(qapp, tmp_path):
    """R8, restated as a regression: entropy is bool(ch.entropy) alone, never a function of
    ``analysis.signals`` — unlike the Work-of-breathing/PTP cards M-17 makes conditional on
    Poes, the entropy card's predicate is untouched by that ticket and must stay that way.
    A Flow-only (no Poes/Pgas/Pdi/EMG) and an EMG-only-SHAPED (no flow at all) declared set
    both still show it, as long as a column is assigned to entropy."""
    sc = _screen(qapp, tmp_path)
    card = _entropy_card(sc)
    assert card.isVisible()
    sc.state.settings.analysis.signals = ["flow"]
    sc.state.settings.input.channels.poes = None
    sc.state.settings.input.channels.pgas = None
    sc.state.settings.input.channels.pdi = None
    sc._update_disclosure()
    qapp.processEvents()
    assert card.isVisible(), "a Flow-only signal set hid the entropy card"
    sc.state.settings.analysis.signals = ["emg"]
    sc.state.settings.input.channels.flow = None
    sc._update_disclosure()
    qapp.processEvents()
    assert card.isVisible(), "an EMG-only-shaped signal set hid the entropy card"


# -- the parameters still round-trip ------------------------------------------
def test_the_moved_parameters_still_load_and_save(qapp, tmp_path):
    sc = _screen(qapp, tmp_path)
    assert sc.ent_epochs.value() == sc.state.settings.processing.entropy.epochs
    sc.ent_epochs.setValue(4)
    sc.ent_tol.setValue(0.25)
    out = sc.to_state().processing.entropy
    assert out.epochs == 4 and out.tolerance == 0.25


def test_editing_them_still_marks_the_analysis_modified(qapp, tmp_path):
    """They moved card, so their entries in _wire_reactivity had to survive the move."""
    for name, value in (("ent_epochs", 5), ("ent_tol", 0.3)):
        sc = _screen(qapp, tmp_path)
        sc._mark_clean()
        getattr(sc, name).setValue(value)
        assert sc.is_dirty(), f"{name} no longer marks the analysis modified"


# -- Subjects && lung volumes (M-37) ------------------------------------------
def _subjects_card(sc):
    return _card(sc, "Subjects && lung volumes")


def test_subjects_card_is_visible_even_when_no_subjects_are_declared(qapp, tmp_path):
    """The card is the only place a subject can be added, so it is never conditional
    (it used to be hidden while input.subjects was empty)."""
    sc = _screen(qapp, tmp_path, entropy=())
    assert not sc.state.settings.input.subjects
    assert _subjects_card(sc).isVisible()


def test_subjects_card_stays_visible_once_a_subject_is_added(qapp, tmp_path):
    from respmech.core.settings import SubjectEntry
    sc = _screen(qapp, tmp_path, entropy=())
    card = _subjects_card(sc)
    sc.state.settings.input.subjects.append(SubjectEntry(key="synth_case", tlc_l=6.0))
    sc._update_disclosure()
    qapp.processEvents()
    assert card.isVisible()


def test_entropy_on_derived_volume_box_follows_flow_and_integrated_volume(qapp, tmp_path):
    """The 'Entropy on derived volume' box lives in the Sample entropy card and is offered only
    when Flow is declared and the volume is integrated from it; ticking it writes
    input.channels.entropy_derived, and the card shows even with no entropy column assigned."""
    sc = _screen(qapp, tmp_path, entropy=())
    card, box = _entropy_card(sc), sc.ent_derived_volume
    assert not card.isVisible()
    sc.state.settings.processing.volume.integrate_from_flow = True
    sc._update_disclosure()
    qapp.processEvents()
    assert card.isVisible() and box.isVisible()
    box.setChecked(True)
    sc._on_field_changed()
    assert sc.state.settings.input.channels.entropy_derived == ["volume"]
    box.setChecked(False)
    sc._on_field_changed()
    assert sc.state.settings.input.channels.entropy_derived == []
