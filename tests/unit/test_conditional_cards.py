"""Cards whose relevance depends on the analysis itself.

Sample entropy has two parameters (embedding m, tolerance r) that mean nothing unless a
column is actually assigned to entropy. They used to sit in the "Advanced (rarely changed)"
grab-bag, which hid them from the users who DO compute entropy while still showing them to
everyone who does not. They now have their own card, shown only when it applies.

The mechanism is the fussy part. _apply_card_visibility forces every card in _stage_cards
visible outside "new" mode, so a conditional card registered there would be un-hidden on the
next keystroke — and test_startup_flow's "open mode reveals everything" walk would fail on a
default AppState, which has no entropy channels. Hence a separate registry, ANDed in its own
pass.
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


def _entropy_card(sc):
    return sc._cond_cards[0][0]


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
    """The failure mode the separate registry exists to prevent: _apply_card_visibility
    forces every _stage_cards entry visible, so a predicate registered there would be
    overwritten by the next field change."""
    sc = _screen(qapp, tmp_path, entropy=())
    card = _entropy_card(sc)
    assert not card.isVisible()
    sc.group_regex.setText("case_(.)")
    sc._on_field_changed()
    qapp.processEvents()
    assert not card.isVisible(), "an unrelated keystroke revealed an irrelevant card"


def test_the_conditional_card_is_not_in_the_staged_registry(qapp, tmp_path):
    """Structural: test_startup_flow walks every staged card and asserts it is visible after
    enter_open_mode. A conditional card there fails that on a default AppState."""
    sc = _screen(qapp, tmp_path)
    staged = [c for cards in sc._stage_cards for c in cards]
    assert _entropy_card(sc) not in staged


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
