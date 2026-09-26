"""Signal-set derivation and :class:`Capabilities`.

Qt-free and numeric-stack-free by design: this module and ``registry.py`` sit on
the GUI startup path (``Settings.validate()``, ``ui/validation.py``, both of
which a later ticket wires to consult them) and must stay importable without
pulling in numpy, scipy, pandas or ``respmech.core.compute`` at module level —
see ``tests/unit/test_startup_imports.py``. Nothing here calls into compute or
pipeline; it only reasons about *which* signals/analyses are in play, never
about the numbers themselves.
"""
from __future__ import annotations

from dataclasses import dataclass

#: The four single-role pressure/flow signals a channel can be assigned to and
#: that make up an explicit ``analysis.signals`` list. ``volume`` is not one of
#: these: it is never chosen directly, only implied by ``flow`` plus either an
#: assigned Volume channel or ``processing.volume.integrate_from_flow``. ``emg``
#: and ``entropy`` are handled separately below for the same reason.
SINGLE_SIGNALS = ("flow", "poes", "pgas", "pdi")

_ALL_CORE = frozenset(SINGLE_SIGNALS)


def derived_signals(ch) -> frozenset:
    """The signal set implied by which channels are currently assigned.

    A pure function of a ``Channels``-shaped object: duck-typed (only attribute
    access, no isinstance check), so a bare namespace with the same attribute
    names works as well as a real
    :class:`respmech.core.settings.Channels`.
    """
    signals = {role for role in SINGLE_SIGNALS if getattr(ch, role, None) is not None}
    if getattr(ch, "emg", None):
        signals.add("emg")
    return frozenset(signals)


def effective_signals(settings) -> frozenset:
    """The signal set that actually governs this analysis.

    An explicit, non-empty ``settings.analysis.signals`` wins; otherwise the set
    is derived from the assigned channels (:func:`derived_signals`).
    ``settings.analysis`` does not exist on :class:`respmech.core.settings.Settings`
    as of this ticket — the ``AnalysisSettings`` dataclass lands with a later
    ticket — so the lookup below is defensive on purpose: this function already
    works against today's ``Settings`` and needs no change when that dataclass
    is added (a save file/settings object with no ``analysis`` table simply
    behaves as "derive it").
    """
    analysis = getattr(settings, "analysis", None)
    raw = getattr(analysis, "signals", None) or ()
    # A bare string is iterable too, so `frozenset("flow")` would silently produce
    # {'f','l','o','w'} instead of {'flow'} for a `signals = "flow"` typo (an easy
    # one once a real, hand-editable `AnalysisSettings.signals` field exists) --
    # guard it explicitly rather than let that corrupt the signal set in silence.
    if isinstance(raw, str):
        raise TypeError(
            f"analysis.signals must be a list of signal names, not a bare string: {raw!r}"
        )
    explicit = frozenset(raw)
    if explicit:
        return explicit
    return derived_signals(settings.input.channels)


def _mode_for(declared: frozenset) -> str:
    """Classify a declared signal set into one of the five named shapes.

    ``emg`` is orthogonal to the flow/pressure shapes (a preset's "Also EMG"
    checkbox adds it without changing which of the four flow-family shapes the
    rest of the set is), so it is stripped before classifying. ``entropy`` never
    enters here at all: R8 keeps it independent of the signal set entirely.
    """
    core = declared - {"emg"}
    if core == _ALL_CORE:
        return "full"
    if core == frozenset({"flow", "poes"}):
        return "poes_only"
    if core == frozenset({"flow"}):
        return "flow_only"
    if not core and "emg" in declared:
        return "emg_only"
    return "custom"


@dataclass(frozen=True)
class Capabilities:
    """What a single analysis can do, derived from its signal set and channels.

    Frozen and built only from bool/str/frozenset fields, so it is hashable,
    equality-comparable and picklable without any extra work — ``BatchWorker``
    deepcopies ``Settings`` for its background thread, and this needs to survive
    that unchanged. Once a later ticket wires it into ``compute``, that code
    reads it as ``getattr(settings, "capabilities", Capabilities.FULL)`` (the
    same defensive idiom ``processing.segmentation.boundary_notice_*`` already
    uses), because hand-built ``SimpleNamespace`` settings reach the
    segmenterers from several call sites; nothing in compute/pipeline calls into
    this yet.
    """

    flow: bool
    volume: bool
    poes: bool
    pgas: bool
    pdi: bool
    emg: bool
    entropy: bool
    declared: frozenset
    mode: str

    @classmethod
    def from_settings(cls, settings) -> "Capabilities":
        declared = effective_signals(settings)
        ch = settings.input.channels
        flow = "flow" in declared
        # Defensive for the same reason as `effective_signals`' own `settings.analysis`
        # lookup: every real Settings always has processing.volume.integrate_from_flow
        # (settings.py's VolumeSettings default), but a lightweight SimpleNamespace test
        # double built for something else entirely (several exist in tests/unit/) need
        # not carry it, and should read as "off" rather than raise.
        processing = getattr(settings, "processing", None)
        volume_settings = getattr(processing, "volume", None)
        integrate_from_flow = bool(getattr(volume_settings, "integrate_from_flow", False))
        return cls(
            flow=flow,
            volume=flow and (getattr(ch, "volume", None) is not None or integrate_from_flow),
            poes="poes" in declared,
            pgas="pgas" in declared,
            pdi="pdi" in declared,
            emg="emg" in declared,
            # R8: sample entropy is not an EMG (or anything else) companion —
            # bool(ch.entropy) alone, never a function of `declared`.
            entropy=bool(getattr(ch, "entropy", None)),
            declared=declared,
            mode=_mode_for(declared),
        )

    def required_roles(self) -> frozenset:
        """Channel roles a valid analysis with this shape must have assigned.

        Meant to be consulted by ``ui/validation.py::channel_collision`` once a
        later ticket wires it in. Entropy is deliberately excluded: it is
        opportunistic (any channel, any signal set may carry it), never a
        blocker on any shape.
        """
        roles = {role for role in SINGLE_SIGNALS if getattr(self, role)}
        if self.volume:
            roles.add("volume")
        if self.emg:
            roles.add("emg")
        return frozenset(roles)

    def analyses(self) -> tuple[str, ...]:
        """Human-readable labels for the run-report / commitment sheet.

        Every pressure-family label requires ``flow`` too, matching
        ``registry.py``'s own capability requirements for those columns
        (breath segmentation itself needs flow — a lone ``pgas``/``pdi``
        channel with no flow computes nothing).
        """
        labels: list[str] = []
        if self.flow:
            labels.append("Breath timing")
        if self.flow and self.poes:
            labels.append("Work of breathing")
        if self.flow and self.pgas:
            labels.append("Gastric pressure")
        if self.flow and self.pdi:
            labels.append("Transdiaphragmatic pressure")
        if self.flow and self.poes and self.pgas:
            labels.append("Ventilatory muscle ratio")
        if self.emg:
            labels.append("EMG")
        if self.entropy:
            labels.append("Sample entropy")
        return tuple(labels)


Capabilities.FULL = Capabilities(
    flow=True, volume=True, poes=True, pgas=True, pdi=True, emg=True, entropy=True,
    declared=frozenset({"flow", "poes", "pgas", "pdi", "emg"}), mode="full",
)
