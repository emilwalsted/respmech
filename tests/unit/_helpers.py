"""Pure test helpers shared across the unit suite (no fixtures — import these directly).

The one canonical synthetic-input settings builder lives here so the channel mapping
(poes 7 · pgas 8 · pdi 9 · volume 6 · flow 5 · EMG 2,3,4 · entropy 10,11,12) is defined
once instead of copy-pasted — previously under six names across a dozen files, free to
drift. ``synth_legacy_dict`` returns the pre-migration dict; ``synth_settings`` returns a
migrated ``Settings`` with folders resolved (and optional shared-profile noise wired).
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT = os.path.join(ROOT, "tests", "golden", "input")


def requires_synth():
    """A ``pytest.mark.skipif`` for tests that need the committed synthetic recordings."""
    import pytest
    return pytest.mark.skipif(
        not os.path.exists(os.path.join(INPUT, "synth_case_A.csv")),
        reason="synthetic input absent")


def synth_legacy_dict(out="", *, samplingfrequency=1000, remove_ecg=False, remove_noise=False,
                      avgresamplingobs=300, calcwobfromaverage=None, data_out=None):
    """The canonical synthetic-input legacy settings dict (for ``migrate_dict``)."""
    mech = {"breathseparationbuffer": 200, "separateby": "flow", "avgresamplingobs": avgresamplingobs}
    if calcwobfromaverage is not None:
        mech["calcwobfromaverage"] = calcwobfromaverage
    return {
        "input": {"inputfolder": INPUT, "files": "synth_case_*.csv",
                  "format": {"samplingfrequency": samplingfrequency},
                  "data": {"column_poes": 7, "column_pgas": 8, "column_pdi": 9,
                           "column_volume": 6, "column_flow": 5,
                           "columns_emg": [2, 3, 4], "columns_entropy": [10, 11, 12]}},
        "processing": {"mechanics": mech,
                       "emg": {"remove_ecg": remove_ecg, "remove_noise": remove_noise}},
        "output": {"outputfolder": str(out), "data": data_out if data_out is not None else {}},
    }


def synth_settings(out="", *, noise=False, channels=None, **kw):
    """Migrated ``Settings`` over the canonical synthetic input, folders resolved.
    ``noise=True`` wires the shared-profile EMG noise reduction (reference synth_case_A,
    expiration window) exactly as the reactive-preview tests expect.

    ``channels``: optional ``{role: value}`` applied to ``input.channels`` AFTER
    migration, overriding/removing individual roles — e.g.
    ``channels={'pgas': None, 'pdi': None}`` drops those two pressure roles, leaving
    flow/poes/volume/emg at their canonical synthetic-input assignment. Entropy is
    sent separately (it is its own kwarg on the returned Settings, not a channels=
    key): it is an independent capability (R8), not a channel role fase 0 varies.
    ``None`` (the default) changes nothing — every existing call site is unaffected."""
    from respmech.settingsio.migrate import migrate_dict
    if noise:
        kw.setdefault("remove_ecg", True)
    s, _ = migrate_dict(synth_legacy_dict(out, remove_noise=noise, **kw))
    s.input.folder = INPUT
    s.output.folder = str(out)
    if noise:
        n = s.processing.emg.noise
        n.enabled = True
        n.reference_file = "synth_case_A.csv"
        n.use_expiration = False
        n.reference_intervals = [[1.0, 5.0]]
        n.auto_prop = True
    if channels:
        for role, value in channels.items():
            setattr(s.input.channels, role, value)
    return s


def segment_synth_case(settings, filename="synth_case_A.csv"):
    """Load, trim, condition and segment ``filename`` exactly as ``core.pipeline.run_batch``
    does up to (but not including) ``calculateaveragebreaths``/``calculatemechanics`` —
    for tests that need to call those two directly, bypassing ``run_batch``/``core.results``
    entirely — narrower and faster than a full batch run, and it isolates a failure to
    these two functions specifically. ``run_batch`` itself already succeeds end to end for
    a flow-only/poes-only settings object once ``compute.py``'s guards are in place
    (``core/pipeline.py``/``core/results.py`` are written generically over whatever the
    mechanics table ends up containing, rather than hardcoding pressure columns).

    Returns ``(legacy_ns, breaths, n_trimmed)`` — ``legacy_ns`` is the same
    ``to_legacy_ns(settings)`` ``run_batch`` itself would build (carries
    ``.capabilities``), ``breaths`` is the ``separateintobreaths()`` dict, ready for
    ``compute.check_breaths``/``compute.calculateaveragebreaths``/
    ``compute.calculatemechanics``, and ``n_trimmed`` is ``len(flow)`` AFTER trimming
    (``run_batch``'s own ``vefactor = 60 / (len(flow) / fs)`` needs this exact count,
    which is not simply the sum of every breath's own sample count back up: trimming
    keeps only whole breaths, so a partial sample or two at either edge of the
    zero-crossing search can fall outside every breath's own slice)."""
    import numpy as np

    from respmech.core import compute
    from respmech.core._legacy_ns import to_legacy_ns
    from respmech.core.io.loaders import load

    s = to_legacy_ns(settings)
    path = os.path.join(s.input.inputfolder, filename)
    flowraw, volumeraw, poesraw, pgasraw, pdiraw, entraw, emgraw = load(path, s)
    timecolraw = np.arange(0, len(flowraw), dtype=int) / s.input.format.samplingfrequency
    timecol, flow, volume, poes, pgas, pdi, _emgtrim, startix, endix = compute.trim(
        timecolraw, flowraw, volumeraw, poesraw, pgasraw, pdiraw,
        np.array(emgraw) if len(emgraw) else np.array([]), s)
    entcols = entraw[startix:endix] if len(entraw) else entraw
    emgcols = emgraw[startix:endix] if len(emgraw) else emgraw
    zerovol = compute.zero(volume)
    driftvol = compute.correctdrift(zerovol, s) if s.processing.mechanics.correctvolumedrift else zerovol
    volume = compute.correcttrend(driftvol, s) if s.processing.mechanics.correctvolumetrend else driftvol
    breaths = compute.separateintobreaths(
        s.processing.mechanics.separateby, filename, timecol, flow, volume,
        poes, pgas, pdi, entcols, emgcols, s)
    return s, breaths, len(flow)


def compute_all_breaths(s, breaths, n_trimmed):
    """Run ``calculateaveragebreaths`` + ``calculatemechanics`` over every non-ignored
    breath from :func:`segment_synth_case`, mutating and returning ``breaths`` in place —
    the same ``vefactor``/``bcnt`` derivation ``core.pipeline.run_batch`` uses.
    ``n_trimmed`` is the third element :func:`segment_synth_case` returns."""
    from respmech.core import compute

    compute.check_breaths(breaths, "synth_case_A.csv", s)
    vefactor = 60 / (n_trimmed / s.input.format.samplingfrequency)
    bcnt = len(breaths)
    avgvolumein, avgvolumeex, avgpoesin, avgpoesex = compute.calculateaveragebreaths(breaths, s)
    for b in breaths.values():
        if b["ignored"]:
            continue
        compute.calculatemechanics(b, bcnt, vefactor, avgvolumein, avgvolumeex, avgpoesin, avgpoesex, s)
    return breaths


def _lone_ampersands(root):
    """Scan ``root`` (typically a ``QMainWindow``) for every caption this app's own
    convention treats as button-like text — ``QAbstractButton.text()`` (including a
    ``QCommandLinkButton``'s own second-line ``description()``, which renders through
    the SAME mnemonic-eating path as its title but is a separate string Qt never
    exposes via ``text()``), ``QGroupBox.title()``, every ``QAction.text()`` reachable
    from ``root.menuBar()`` (when it has one) and from ``root.findChildren(QMenu)``,
    ``QTabBar.tabText(i)``, and the text of any ``QLabel`` with a ``buddy()`` set.
    Returns a list of
    ``(widget-type-name, offending-text)`` pairs for every caption carrying a LONE ``&``
    that Qt would silently swallow as a mnemonic marker instead of rendering as a
    literal ampersand — the same defect class as the v2.4.0 "Run & results" ->
    "Run _results" regression, generalised beyond just push buttons.

    A caption in this app that wants a LITERAL ``&`` always doubles it ("Preview && QC",
    "Process && write this file"), so a lone ``&`` is either that bug or a deliberate
    mnemonic — and a deliberate mnemonic is never on a space (it is always immediately
    followed by an alphanumeric character, e.g. menu-bar titles like "&File" or a
    recent-file entry's "&3  filename.toml").

    QAction objects can legitimately live in two containers at once (the same action
    added to both a toolbar-style menu and the File menu, "one enable-state behind two
    doors" — see ``main_window.py``), so actions are de-duplicated by identity before
    their text is checked.
    """
    import re

    from PySide6.QtWidgets import (QAbstractButton, QCommandLinkButton, QGroupBox,
                                   QLabel, QMenu, QTabBar)

    def _has_lone_ampersand(text):
        # Scan whole RUNS of consecutive '&', not one character at a time: Qt itself
        # resolves a run left to right, greedily pairing "&&" into one literal '&' and
        # leaving at most one unpaired '&' (only when the run's length is odd) to act as
        # a mnemonic marker on whatever comes right after the run. A naive per-character
        # check that only looks at immediate neighbours misclassifies an odd run of 3+
        # ('&&&', '&&&&&', ...): each '&' sees a '&' neighbour on one side or the other
        # and is waved through as "part of a pair", so a genuinely broken trailing '&'
        # (e.g. '&&&' at the end of a caption) is silently missed.
        for m in re.finditer(r"&+", text):
            if (m.end() - m.start()) % 2 == 0:
                continue                                      # fully paired, e.g. "&&"
            next_ch = text[m.end():m.end() + 1]
            if next_ch.isalnum():                              # a deliberate mnemonic
                continue
            return True
        return False

    offenders = []

    def _check(kind, text):
        if _has_lone_ampersand(text):
            offenders.append((kind, text))

    for b in root.findChildren(QAbstractButton):
        _check(type(b).__name__, b.text())
        # QCommandLinkButton's second line (its own caption, not a tooltip) goes
        # through the same mnemonic-eating text-drawing path as .text() -- found by
        # self-review missing a lone '&' here that .text() alone could never catch.
        if isinstance(b, QCommandLinkButton):
            _check(f"{type(b).__name__}.description()", b.description())
    for g in root.findChildren(QGroupBox):
        _check(type(g).__name__, g.title())
    for tb in root.findChildren(QTabBar):
        for i in range(tb.count()):
            _check("QTabBar", tb.tabText(i))
    for lbl in root.findChildren(QLabel):
        if lbl.buddy() is not None:
            _check("QLabel", lbl.text())

    seen_action_ids = set()

    def _check_action(act):
        if id(act) in seen_action_ids:
            return
        seen_action_ids.add(id(act))
        _check("QAction", act.text())

    menu_bar = getattr(root, "menuBar", None)
    if callable(menu_bar):
        bar = menu_bar()
        if bar is not None:
            for act in bar.actions():
                _check_action(act)
    for menu in root.findChildren(QMenu):
        for act in menu.actions():
            _check_action(act)

    return offenders


def assert_units(mapping):
    """Assert ``quantities.unit_for(column) == unit`` for every ``{column: unit}``
    pair in ``mapping`` — a small shared helper for "every column a feature emits
    resolves to the unit it was designed for" tests. Reports every mismatch at
    once, (got, expected) per column, rather than stopping at the first."""
    from respmech.core import quantities

    got = {c: quantities.unit_for(c) for c in mapping}
    bad = {c: (got[c], u) for c, u in mapping.items() if got[c] != u}
    assert not bad, f"unit_for mismatch, column: (got, expected) = {bad}"


def is_dark_hex(h):
    """True when a #RRGGBB colour reads as a dark surface (perceived luminance)."""
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (0.299 * r + 0.587 * g + 0.114 * b) < 110


def write_delim(path, ncols, sep=",", nrows=30, fs=1000.0, header=True):
    """A synthetic delimited recording (write ``path`` with a ``.csv``/``.txt`` suffix):
    column 0 is a real, regularly-sampled time-in-seconds axis, so
    ``workers.detect_sampling_frequency`` recognises it (used by the manifest scanner's
    ``probe_sampling_frequency`` and by tests exercising it directly). The rest are
    arbitrary numbers — this is for column-count/frequency detection, not physiology."""
    dt = 1.0 / fs
    lines = []
    if header:
        lines.append(sep.join(["time"] + [f"c{i}" for i in range(2, ncols + 1)]))
    for i in range(nrows):
        row = [f"{i * dt:.6f}"] + [str(i + j) for j in range(2, ncols + 1)]
        lines.append(sep.join(row))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_xlsx(path, ncols, nrows=30, fs=1000.0):
    """A synthetic .xlsx recording, same shape convention as :func:`write_delim`."""
    import pandas as pd
    dt = 1.0 / fs
    cols = {"time": [i * dt for i in range(nrows)]}
    for i in range(2, ncols + 1):
        cols[f"c{i}"] = [i + j for j in range(nrows)]
    pd.DataFrame(cols).to_excel(path, index=False)
