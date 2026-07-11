"""Screen 1 — settings / batch setup.

A form bound to the typed settings model. Load/save TOML; import a legacy .py
settings file (runs the migrator and shows its report); inline validation reuses the
core ``Settings.validate``. Editing any field syncs the shared state live and emits
signals so the Preview/Run screens stay in step (e.g. the file list refreshes when
the input folder/mask changes). Only the most-used fields are surfaced; the full
model is preserved on load/save round-trips.
"""
from __future__ import annotations

import glob
import os
import traceback

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea,
                               QSpinBox, QVBoxLayout, QWidget)
from PySide6.QtCore import Signal, QTimer

from respmech.core.settings import Settings, SettingsError
from respmech.ui.dialogs import open_error_dialog, short_error
from respmech.ui.help_text import tooltip as _tip
from respmech.ui.startup_dialog import OPEN_FILTER
from respmech.ui.validation import matching_files

# the guided-flow default file mask (multi-pattern; narrowed to the found extension on the
# channel-setup OK so the single-pattern core batch runner still finds the files)
_DEFAULT_MASK = "*.csv; *.txt"


class SettingsScreen(QWidget):
    settings_changed = Signal()     # any field edited -> shared state is current
    inputs_changed = Signal()       # input folder/mask edited -> file list is stale
    status_changed = Signal(str)
    flow_ready_changed = Signal(bool)   # downstream (Preview/Run) tabs should be shown
    analysis_state_changed = Signal()   # loaded file / unsaved-edits state changed (title)

    def __init__(self, state, on_settings_changed=None):
        super().__init__()
        self.state = state
        self.on_settings_changed = on_settings_changed
        self._loading = True          # suppress reactions during programmatic fills
        self._dirty = False           # unsaved edits since the last save/open/new
        self._err_dialog = None       # copyable error dialog (replace, don't accumulate)
        self._report_dialog = None    # migration report dialog
        self._build()
        self.from_state()
        self._wire_reactivity()
        self._loading = False

    # -- UI construction ----------------------------------------------------
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Action bar stays pinned above the scroll area so it is always reachable.
        # "Analysis" is the user-facing name for a settings file (never "TOML"); "Open
        # analysis…" accepts both a saved .toml analysis and a legacy .py setup, routed
        # by file extension.
        bar = QHBoxLayout()
        for label, slot in (("New analysis", self._new_analysis),
                            ("Open analysis…", self._open_analysis_dialog),
                            ("Save analysis…", self._save),
                            ("Validate", self._validate)):
            b = QPushButton(label)
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addStretch(1)
        outer.addLayout(bar)

        # The long vertical form lives inside a scroll area so every section is
        # reachable on small screens. setWidgetResizable(True) keeps the content
        # at the viewport width (no horizontal scrollbar). All self.* widgets
        # below are still built exactly as before — only their container changed.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        root = QVBoxLayout(content)

        # Input ------------------------------------------------------------
        gin = QGroupBox("Input")
        f = QFormLayout(gin)
        self.in_folder = QLineEdit()
        self.in_files = QLineEdit()
        self.samp_freq = QSpinBox(); self.samp_freq.setRange(1, 1_000_000); self.samp_freq.setSuffix(" Hz")
        self._browse_row(f, "Recordings folder", self.in_folder, "input.folder",
                         "Folder containing the recording files to analyse; defaults to 'input'.", folder=True)
        self._row(f, "Files to analyse", self.in_files, "input.files",
                  "Filename or wildcard mask picking which recordings to load, e.g. *.txt; defaults to *.* (all files).")
        self._row(f, "Sampling frequency", self.samp_freq, "input.format.sampling_frequency",
                  "Samples recorded per second, in hertz (Hz); required, and must match the acquisition system (e.g. 2000).")
        root.addWidget(gin)

        # Output (second in the flow: an analysis reads as Input -> Output -> rest) --
        gout = QGroupBox("Output")
        fo = QFormLayout(gout)
        self.out_folder = QLineEdit()
        self._browse_row(fo, "Output folder", self.out_folder, "output.folder",
                         "Where results are saved; files are written to a 'data' subfolder inside it; defaults to 'output'.", folder=True)
        root.addWidget(gout)

        # Channels ---------------------------------------------------------
        gch = QGroupBox("Channels — 1-based column numbers in the data file")
        fc = QFormLayout(gch)
        self.btn_assign_channels = QPushButton("Assign channels from data…")
        self.btn_assign_channels.setToolTip(
            "Open a visual picker: plot every column of your data and pick what each one is.")
        self.btn_assign_channels.clicked.connect(lambda: self._open_channel_setup())
        fc.addRow("", self.btn_assign_channels)
        self.col_flow = self._spin(); self.col_volume = self._spin(allow_zero=True)
        self.col_poes = self._spin(); self.col_pgas = self._spin(); self.col_pdi = self._spin()
        self.cols_emg = QLineEdit(); self.cols_entropy = QLineEdit()
        self._row(fc, "Flow signal column", self.col_flow, "input.channels.flow",
                  "Column number (counting from 1) holding the airflow signal, which reads negative during inspiration.")
        self._row(fc, "Volume signal column", self.col_volume, "input.channels.volume",
                  "Column number (counting from 1) holding the volume signal; or tick 'Calculate volume "
                  "from flow' in Processing to derive it from flow instead (this column is then ignored).")
        self._row(fc, "Oesophageal pressure (Poes) column", self.col_poes, "input.channels.poes",
                  "Column number (counting from 1) holding the oesophageal pressure signal, in cmH2O.")
        self._row(fc, "Gastric pressure (Pgas) column", self.col_pgas, "input.channels.pgas",
                  "Column number (counting from 1) holding the gastric pressure signal, in cmH2O.")
        self._row(fc, "Transdiaphragmatic pressure (Pdi) column", self.col_pdi, "input.channels.pdi",
                  "Column number (counting from 1) holding the transdiaphragmatic pressure signal, in cmH2O.")
        self._row(fc, "EMG channel columns", self.cols_emg, "input.channels.emg",
                  "Column numbers (counting from 1) of the diaphragm EMG channels to analyse; enter one or more.")
        self._row(fc, "Sample entropy columns", self.cols_entropy, "input.channels.entropy",
                  "Column numbers (counting from 1) of the channels used for sample-entropy analysis; leave empty to skip.")
        root.addWidget(gch)

        # Processing — breath mechanics ------------------------------------
        gpr = QGroupBox("Processing")
        fp = QFormLayout(gpr)
        self.seg_method = QComboBox(); self.seg_method.addItems(["flow", "volume"])
        self.wob_from = QComboBox(); self.wob_from.addItems(["average", "individual"])
        self.integrate = QCheckBox("Calculate volume from flow")
        self._row(fp, "Signal used to split breaths", self.seg_method, "processing.segmentation.method",
                  "Which signal marks where each breath begins and ends — flow (the default) or volume.")
        self._row(fp, "Work of breathing calculation", self.wob_from, "processing.wob.calc_from",
                  "Compute work of breathing from one averaged breath (the default), or from each breath separately and then average.")
        self._check_row(fp, self.integrate, "processing.volume.integrate_from_flow",
                        "Derive volume by integrating the flow signal instead of reading a separate volume channel; off by default.")
        root.addWidget(gpr)

        # EMG — RMS envelope -----------------------------------------------
        gemg = QGroupBox("EMG — RMS envelope")
        gemg.setToolTip("How the diaphragm-EMG amplitude is quantified for each breath.")
        fe = QFormLayout(gemg)
        self.emg_rms_window = QDoubleSpinBox()
        self.emg_rms_window.setRange(0.01, 0.5); self.emg_rms_window.setSingleStep(0.01)
        self.emg_rms_window.setDecimals(3); self.emg_rms_window.setSuffix(" s")
        self.emg_outlier_sd = QDoubleSpinBox()
        self.emg_outlier_sd.setRange(0.0, 10.0); self.emg_outlier_sd.setSingleStep(0.5)
        self.emg_outlier_sd.setDecimals(1); self.emg_outlier_sd.setSuffix(" SD")
        self.emg_outlier_sd.setSpecialValueText("off (0)")   # minimum (0.0) reads as disabled
        self._row(fe, "EMG RMS window length", self.emg_rms_window, "processing.emg.rms_window_s",
                  "Sliding-window length for the EMG RMS envelope, in seconds; each breath takes its largest windowed value (default 0.05).")
        self._row(fe, "EMG RMS outlier limit", self.emg_outlier_sd, "processing.emg.outlier_rms_sd_limit",
                  "Replace any breath's EMG RMS value lying more than this many standard deviations from the across-breath mean; 0 = off (default).")
        root.addWidget(gemg)

        # EMG — ECG removal ------------------------------------------------
        gecg = QGroupBox("EMG — ECG removal")
        gecg.setToolTip("Detect R-waves on one EMG channel and subtract an averaged "
                        "ECG template from every EMG channel.")
        fg = QFormLayout(gecg)
        self.remove_ecg = QCheckBox("Remove ECG artefact from EMG")
        self.ecg_detect = QComboBox()
        self.ecg_min_height = QDoubleSpinBox()
        self.ecg_min_height.setRange(0.0, 1_000_000.0); self.ecg_min_height.setDecimals(6)
        self.ecg_min_height.setSingleStep(0.0001)
        self.ecg_min_distance = QDoubleSpinBox()
        self.ecg_min_distance.setRange(0.05, 2.0); self.ecg_min_distance.setSingleStep(0.05)
        self.ecg_min_distance.setDecimals(3); self.ecg_min_distance.setSuffix(" s")
        self.ecg_min_width = QDoubleSpinBox()
        self.ecg_min_width.setRange(0.0, 0.1); self.ecg_min_width.setSingleStep(0.001)
        self.ecg_min_width.setDecimals(4); self.ecg_min_width.setSuffix(" s")
        self.ecg_window = QDoubleSpinBox()
        self.ecg_window.setRange(0.05, 1.0); self.ecg_window.setSingleStep(0.05)
        self.ecg_window.setDecimals(3); self.ecg_window.setSuffix(" s")
        self._check_row(fg, self.remove_ecg, "processing.emg.remove_ecg",
                        "Subtracts an averaged ECG template from each EMG channel to cancel cardiac contamination; off by default.")
        self._row(fg, "EMG channel for heartbeat detection", self.ecg_detect, "processing.emg.detect_channel",
                  "Index of the EMG channel used to detect R-wave (heartbeat) peaks when building the ECG template; defaults to the first channel (0).")
        self._row(fg, "Minimum heartbeat peak height", self.ecg_min_height, "processing.emg.ecg_min_height",
                  "Smallest peak amplitude, in raw EMG units, accepted as a heartbeat during R-wave detection; default 0.0005.")
        self._row(fg, "Minimum time between heartbeats", self.ecg_min_distance, "processing.emg.ecg_min_distance_s",
                  "Shortest interval allowed between detected heartbeats, in seconds; a refractory period that caps the maximum heart rate (default 0.5).")
        self._row(fg, "Minimum heartbeat peak width", self.ecg_min_width, "processing.emg.ecg_min_width_s",
                  "Narrowest peak accepted as a heartbeat, in seconds; rejects thin spikes that aren't true R-waves (default 0.001).")
        self._row(fg, "ECG template window length", self.ecg_window, "processing.emg.ecg_window_s",
                  "Total width, in seconds, of the ECG template averaged and subtracted around each beat, spanning the QRS-T complex (default 0.4).")
        root.addWidget(gecg)

        # EMG — noise reduction (shared profile) ---------------------------
        gns = QGroupBox("EMG — noise reduction (shared profile)")
        gns.setToolTip("One shared noise profile + one parameter set are built from the "
                       "reference and applied IDENTICALLY to every file in the test. "
                       "Tune the noise window visually in Preview ▸ EMG processing.")
        fn = QFormLayout(gns)
        self.remove_noise = QCheckBox("Reduce EMG background noise")
        self.noise_ref = QLineEdit()
        self.noise_use_exp = QCheckBox("Build noise profile from expiration (recommended)")
        self._check_row(fn, self.remove_noise, "processing.emg.noise.enabled",
                        "Subtracts a shared spectral noise profile, built from a rest reference, from every EMG channel; off by default.")
        self._browse_row(fn, "Noise reference recording", self.noise_ref, "processing.emg.noise.reference_file",
                         "Rest or baseline recording whose EMG-free portion defines the noise profile applied to every file in the test.",
                         folder=False)
        self._check_row(fn, self.noise_use_exp, "processing.emg.noise.use_expiration",
                        "Samples the noise profile from the reference's expiratory phase, which is EMG-free and more stable; on by default.")
        hint = QLabel("Noise-window options (suppression, fidelity target) are in "
                      "Preview ▸ EMG processing, active once a reference file is set.")
        hint.setWordWrap(True); hint.setProperty("status", "muted")
        fn.addRow("", hint)
        root.addWidget(gns)

        # Status text is shown in the window's bottom status bar (see main_window). This
        # label is kept only as a hidden text holder mirroring that message — showing it
        # inline too would duplicate the same sentence on screen.
        self.status = QLabel("")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.status.hide()
        root.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # Live QC strip, pinned below the (scrolling) form: every current caution at a
        # glance, so a first-timer sees them without ever clicking Validate.
        self.qc = QLabel("")
        self.qc.setWordWrap(True)
        self.qc.setContentsMargins(2, 2, 2, 2)
        outer.addWidget(self.qc)

        # --- progressive-disclosure stages (used only in guided "new analysis" mode) -
        # Ordered reveal groups. Placement-agnostic: reordering or moving a card only
        # changes this registry, not the reveal logic. Stage 0 shows immediately; each
        # later stage's cards appear once the PRECEDING stage's semantic gate passes,
        # and the Preview/Run tabs unlock once everything validates (_all_ok).
        self._stage_cards = [
            [gin],                                   # 0: Input (always shown)
            [gout],                                  # 1: Output (after Input is valid)
            [gch, gpr, gemg, gecg, gns],             # 2: the rest (after Output is valid)
        ]
        self._stage_gate = [self._input_stage_ok, self._output_stage_ok]
        self._mode = "full"          # "full" = every card+tab visible (default/open); "new" = guided
        self._revealed = len(self._stage_cards)
        self._flow_ready = True
        # the visual channel-assignment modal sits between Output and the rest of the
        # cards in the guided flow (see _update_disclosure); these track its one-shot
        # auto-open so it fires once per new-analysis session.
        self._channel_modal_done = False
        self._channel_modal_pending = False

    def _row(self, form, label, widget, var, desc):
        """A labelled form row whose LABEL and FIELD both carry the same tooltip:
        the settings variable path (bold) + a one-line description."""
        tip = _tip(var, desc)
        lab = QLabel(label); lab.setToolTip(tip)
        widget.setToolTip(tip)
        form.addRow(lab, widget)

    def _check_row(self, form, checkbox, var, desc):
        """A checkbox row (the checkbox text is the label); the tooltip carries the
        variable path + description."""
        checkbox.setToolTip(_tip(var, desc))
        form.addRow("", checkbox)

    def _browse_row(self, form, label, line, var, desc, folder):
        """A labelled 'line edit + Browse…' row; the label, the field and its inner
        line edit all carry the variable path + description tooltip."""
        tip = _tip(var, desc)
        lab = QLabel(label); lab.setToolTip(tip)
        line.setToolTip(tip)
        wrapper = self._with_browse(line, folder=folder)
        wrapper.setToolTip(tip)
        form.addRow(lab, wrapper)

    def _spin(self, allow_zero=False):
        s = QSpinBox(); s.setRange(0 if allow_zero else 1, 9999); return s

    def _with_browse(self, line: QLineEdit, folder=False):
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
        h.addWidget(line, 1)
        b = QPushButton("Browse…")
        b.setProperty("compact", True)
        b.setToolTip("Choose a folder…" if folder else "Choose a file…")
        b.clicked.connect(lambda: self._browse(line, folder))
        h.addWidget(b)
        return w

    def _browse(self, line, folder):
        if folder:
            d = QFileDialog.getExistingDirectory(self, "Select folder", line.text() or ".")
        else:
            d, _ = QFileDialog.getOpenFileName(self, "Select file", line.text() or ".")
        if d:
            line.setText(d)
            if self._loading:
                return
            (self._on_inputs_changed if line is self.in_folder else self._on_field_changed)()

    # -- state <-> widgets --------------------------------------------------
    def from_state(self):
        prev, self._loading = self._loading, True
        try:
            s = self.state.settings
            self.in_folder.setText(s.input.folder)
            self.in_files.setText(s.input.files)
            self.samp_freq.setValue(s.input.format.sampling_frequency or 2000)
            ch = s.input.channels
            self.col_flow.setValue(ch.flow or 1)
            self.col_volume.setValue(ch.volume or 0)
            self.col_poes.setValue(ch.poes or 1)
            self.col_pgas.setValue(ch.pgas or 1)
            self.col_pdi.setValue(ch.pdi or 1)
            self.cols_emg.setText(",".join(map(str, ch.emg)))
            self.cols_entropy.setText(",".join(map(str, ch.entropy)))
            self.seg_method.setCurrentText(s.processing.segmentation.method)
            self.wob_from.setCurrentText(s.processing.wob.calc_from)
            self.integrate.setChecked(s.processing.volume.integrate_from_flow)

            emg, n = s.processing.emg, s.processing.emg.noise
            # EMG RMS envelope
            self.emg_rms_window.setValue(emg.rms_window_s)
            self.emg_outlier_sd.setValue(emg.outlier_rms_sd_limit)
            # ECG removal
            self.remove_ecg.setChecked(emg.remove_ecg)
            self.ecg_min_height.setValue(emg.ecg_min_height)
            self.ecg_min_distance.setValue(emg.ecg_min_distance_s)
            self.ecg_min_width.setValue(emg.ecg_min_width_s)
            self.ecg_window.setValue(emg.ecg_window_s)
            # Noise reduction (enable + reference file; tuning lives in the EMG tab)
            self.remove_noise.setChecked(n.enabled)               # rebind to the real gate
            self.noise_ref.setText(n.reference_file or "")
            self.noise_use_exp.setChecked(n.use_expiration)
            self._repopulate_ecg_detect()
            if 0 <= emg.detect_channel < self.ecg_detect.count():
                self.ecg_detect.setCurrentIndex(emg.detect_channel)
            self.out_folder.setText(s.output.folder)
            self._sync_widgets()
        finally:
            self._loading = prev
        self._update_qc()

    def to_state(self):
        s = self.state.settings
        s.input.folder = self.in_folder.text()
        s.input.files = self.in_files.text()
        s.input.format.sampling_frequency = self.samp_freq.value()
        ch = s.input.channels
        ch.flow = self.col_flow.value()
        ch.volume = self.col_volume.value() or None
        ch.poes = self.col_poes.value()
        ch.pgas = self.col_pgas.value()
        ch.pdi = self.col_pdi.value()
        ch.emg = _parse_ints(self.cols_emg.text())
        ch.entropy = _parse_ints(self.cols_entropy.text())
        s.processing.segmentation.method = self.seg_method.currentText()
        s.processing.wob.calc_from = self.wob_from.currentText()
        s.processing.volume.integrate_from_flow = self.integrate.isChecked()

        emg, n = s.processing.emg, s.processing.emg.noise
        # EMG RMS envelope
        emg.rms_window_s = self.emg_rms_window.value()
        emg.outlier_rms_sd_limit = self.emg_outlier_sd.value()
        # ECG removal
        emg.remove_ecg = self.remove_ecg.isChecked()
        emg.ecg_min_height = self.ecg_min_height.value()
        emg.ecg_min_distance_s = self.ecg_min_distance.value()
        emg.ecg_min_width_s = self.ecg_min_width.value()
        emg.ecg_window_s = self.ecg_window.value()
        # Noise reduction (enable + reference file; tuning owned by the EMG tab)
        n.enabled = self.remove_noise.isChecked()
        emg.remove_noise = self.remove_noise.isChecked()      # keep legacy mirror so TOML round-trips
        n.reference_file = self.noise_ref.text().strip() or None
        n.use_expiration = self.noise_use_exp.isChecked()
        emg.detect_channel = max(0, self.ecg_detect.currentIndex())
        s.output.folder = self.out_folder.text()
        if self.on_settings_changed:
            self.on_settings_changed()
        return s

    def set_noise_reference(self, file, intervals, use_expiration):
        """Apply a noise reference chosen on the Preview graph (feature B) so the
        Settings form and shared state stay coherent. ``reference_intervals`` has no
        widget, so it is written straight to state (and survives ``to_state``, which
        does not touch it); the file + use_expiration mirror into the widgets and are
        committed via ``to_state``. Guarded against reactive feedback with ``_loading``."""
        prev, self._loading = self._loading, True
        try:
            self.noise_ref.setText(file or "")
            self.noise_use_exp.setChecked(bool(use_expiration))
            self.state.settings.processing.emg.noise.reference_intervals = list(intervals or [])
        finally:
            self._loading = prev
        self.to_state()
        self._sync_widgets()
        self._set_status(f"Noise reference set from graph selection: {file}")

    # -- helpers ------------------------------------------------------------
    def _repopulate_ecg_detect(self):
        prev, self._loading = getattr(self, "_loading", True), True
        try:
            cols = _parse_ints(self.cols_emg.text())
            cur = self.ecg_detect.currentIndex()
            self.ecg_detect.clear()
            for i, c in enumerate(cols):
                self.ecg_detect.addItem(f"EMGdi col {c} (ECG ref)" if i == 0 else f"EMGdi col {c}")
            if 0 <= cur < self.ecg_detect.count():
                self.ecg_detect.setCurrentIndex(cur)
        finally:
            self._loading = prev

    def _sync_widgets(self):
        noise_on = self.remove_noise.isChecked()
        self.noise_ref.setEnabled(noise_on)
        # "use expiration" only makes sense once a reference file is chosen
        self.noise_use_exp.setEnabled(noise_on and bool(self.noise_ref.text().strip()))
        ecg_on = self.remove_ecg.isChecked()
        for w in (self.ecg_detect, self.ecg_min_height, self.ecg_min_distance,
                  self.ecg_min_width, self.ecg_window):
            w.setEnabled(ecg_on)
        self.col_volume.setEnabled(not self.integrate.isChecked())

    # -- reactivity ---------------------------------------------------------
    def _wire_reactivity(self):
        self.in_folder.editingFinished.connect(self._on_inputs_changed)
        self.in_files.editingFinished.connect(self._on_inputs_changed)
        self.cols_emg.editingFinished.connect(self._on_emg_cols_changed)
        for le in (self.cols_entropy, self.out_folder, self.noise_ref):
            le.editingFinished.connect(self._on_field_changed)
        for sb in (self.samp_freq, self.col_flow, self.col_volume, self.col_poes,
                   self.col_pgas, self.col_pdi):
            sb.valueChanged.connect(self._on_field_changed)
        for dsb in (self.emg_rms_window, self.emg_outlier_sd, self.ecg_min_height,
                    self.ecg_min_distance, self.ecg_min_width, self.ecg_window):
            dsb.valueChanged.connect(self._on_field_changed)
        for cb in (self.seg_method, self.wob_from):
            cb.currentTextChanged.connect(self._on_field_changed)
        self.ecg_detect.currentIndexChanged.connect(self._on_field_changed)
        for chk in (self.integrate, self.remove_ecg, self.remove_noise, self.noise_use_exp):
            chk.toggled.connect(self._on_field_changed)

    def _on_field_changed(self, *_):
        if self._loading:
            return
        self._sync_widgets()
        self.to_state()
        self._mark_dirty()
        self.settings_changed.emit()
        self._update_disclosure()   # last, so guided-flow status wins over downstream
                                    # screens' technical validation messages

    def _on_inputs_changed(self, *_):
        if self._loading:
            return
        self.to_state()
        self._normalize_mask()      # keep the mask a single pattern the core runner can glob
        self._mark_dirty()
        self.inputs_changed.emit()
        self.settings_changed.emit()
        self._update_disclosure()   # last, so guided-flow status wins (see above)

    def _on_emg_cols_changed(self, *_):
        if self._loading:
            return
        # warn (non-blocking) if a typo silently dropped a non-numeric column token
        raw = [p.strip() for p in self.cols_emg.text().replace(";", ",").split(",") if p.strip()]
        dropped = [p for p in raw if not _is_int(p)]
        self._repopulate_ecg_detect()
        self._on_field_changed()
        if dropped:
            self._set_status(f"Ignored non-numeric EMG column(s): {', '.join(dropped)}")

    def _set_status(self, text):
        self.status.setText(text)
        self.status_changed.emit(text)

    def _mark_dirty(self):
        if not self._dirty:
            self._dirty = True
            self.analysis_state_changed.emit()

    def _mark_clean(self):
        """Record a saved / freshly-opened / new analysis as having no unsaved edits."""
        self._dirty = False
        self.analysis_state_changed.emit()

    def is_dirty(self):
        return self._dirty

    # -- guided flow / progressive disclosure -------------------------------
    def enter_new_mode(self):
        """Enter the guided 'new analysis' flow: collapse the reveal back to just the
        Input card and keep the downstream (Preview/Run) tabs hidden until every setting
        validates. Cards then appear stage by stage as each is completed."""
        self._mode = "new"
        self._revealed = 1
        self._flow_ready = None          # force the next _set_flow_ready() to emit
        self._channel_modal_done = False     # re-arm the channel-assignment modal
        self._channel_modal_pending = False
        # A fresh analysis starts with the gating folders empty, so each stage is a real
        # step: the relative "input"/"output" defaults would otherwise auto-satisfy their
        # gate (an "output" folder's parent — the cwd — always exists) and reveal the
        # later cards before the user has actually chosen anything.
        prev, self._loading = self._loading, True
        try:
            self.state.settings.input.folder = ""
            self.state.settings.output.folder = ""
            self.state.settings.input.files = _DEFAULT_MASK
            self.in_folder.setText("")
            self.out_folder.setText("")
            self.in_files.setText(_DEFAULT_MASK)
        finally:
            self._loading = prev
        # let the downstream screens react to the blanked settings FIRST, so the guided
        # status set by _update_disclosure below lands last and is not clobbered by their
        # technical validation messages in the shared status bar
        self.inputs_changed.emit()
        self.settings_changed.emit()
        self._update_disclosure()   # sets the stage-appropriate guidance status (last word)

    def enter_open_mode(self):
        """Enter 'open' mode: reveal every card and every downstream tab, then validate
        so any problem in the opened analysis is surfaced right away."""
        self._mode = "full"
        self._revealed = len(self._stage_cards)
        self._normalize_mask()      # a saved analysis may carry a multi-pattern mask
        self._apply_card_visibility()
        self._flow_ready = None
        self._set_flow_ready(True)
        self._show_validation_status()

    def open_analysis(self, path):
        """Open an existing analysis from a saved .toml or a legacy .py (routed by file
        extension). On success reveal everything and validate; on a failed/rolled-back
        open leave the current mode untouched (the error is already surfaced by the
        Open/Import action). Returns True iff the open succeeded."""
        ok = self._import(path) if str(path).lower().endswith(".py") else self._load(path)
        if ok:
            self.enter_open_mode()
        return ok

    def _new_analysis(self):
        """Action-bar 'New analysis': discard the current settings for a fresh set and
        re-enter the guided flow. Only confirm when there are REAL unsaved edits, so the
        warning means something instead of training the user to click through it."""
        if self._dirty and QMessageBox.question(
                self, "New analysis",
                "You have unsaved changes. Start a new analysis and discard them?"
                ) != QMessageBox.Yes:
            return
        self.state.settings = Settings()
        self.state.settings_path = None
        self.from_state()
        self.enter_new_mode()       # emits inputs/settings_changed then sets guided status
        self._mark_clean()          # a fresh analysis has no unsaved edits yet

    def _open_analysis_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open analysis", ".", OPEN_FILTER)
        if p:
            self.open_analysis(p)

    # -- visual channel assignment ------------------------------------------
    def _open_channel_setup_for_flow(self):
        """Deferred one-shot auto-open of the channel modal in the guided flow, once
        Output is valid. Marks the step done first (so it never re-opens), then reveals
        the remaining cards whether the user assigns channels or cancels."""
        self._channel_modal_pending = False
        if self._channel_modal_done:
            return
        self._channel_modal_done = True
        self._open_channel_setup(initial={})     # fresh analysis -> no pre-selection
        self._update_disclosure()                # reveal the rest (OK applied, or cancelled)

    def _open_channel_setup(self, initial=None):
        """Show the visual channel-assignment modal over the valid data files matching the
        input mask, and on OK write the channel columns into the form. Returns True iff a
        mapping was applied. ``initial`` pre-selects the dropdowns (None -> current settings)."""
        files = self._valid_input_files()
        if not files:
            self._set_status("No valid data files found to assign channels — set them in the Channels card.")
            return False
        from respmech.ui.channel_setup_dialog import ChannelSetupDialog
        from respmech.ui.workers import load_raw_matrix
        s = self.state.settings
        fs = s.input.format.sampling_frequency or 1000
        if initial is None:
            initial = self._current_channel_mapping()
        try:
            dlg = ChannelSetupDialog(files, fs, initial,
                                     loader=lambda p: load_raw_matrix(s, p), parent=self)
        except Exception:                        # noqa: BLE001 — copyable error, don't trap the flow
            self._report_error("Channel setup", traceback.format_exc())
            return False
        if dlg.exec() != QDialog.Accepted:
            return False
        fmt_note = self._probe_and_apply_file_settings(files)
        self._apply_channel_mapping(dlg.selected_mapping(), fmt_note=fmt_note)
        return True

    def _normalize_mask(self):
        """Keep ``input.files`` a SINGLE glob pattern, because the core batch runner globs one
        pattern: a multi-pattern mask (the '*.csv; *.txt' default, or any ';'/','-separated
        mask) is narrowed to the dominant extension of the files it currently matches. A
        specific single mask the user chose is left untouched; a no-op when nothing matches
        yet (or no folder). Never raises."""
        s = self.state.settings
        mask = s.input.files or ""
        if ";" not in mask and "," not in mask:
            return
        folder = (s.input.folder or "").strip()
        if not folder or not os.path.isdir(folder):
            return
        from collections import Counter
        exts = [os.path.splitext(f)[1].lower() for f in matching_files(folder, mask)]
        if not exts:
            return
        counts = Counter(exts)
        top = max(counts.values())
        ext = sorted(e for e, c in counts.items() if c == top)[0]   # deterministic on a tie
        narrowed = f"*{ext}"
        if narrowed != mask:
            prev, self._loading = self._loading, True
            try:
                s.input.files = narrowed
                self.in_files.setText(narrowed)
            finally:
                self._loading = prev

    def _probe_and_apply_file_settings(self, files):
        """On channel-setup OK, intelligently read the actual data to fill format fields the
        user would otherwise set by hand: the decimal separator (for .txt) and the sampling
        frequency (from the time column). The file mask is kept single-pattern separately by
        _normalize_mask. Returns a short human summary of what was set."""
        import os as _os
        from respmech.ui.workers import (detect_decimal, detect_sampling_frequency,
                                         load_raw_matrix)
        if not files:
            return ""
        ref = files[0]
        ext = _os.path.splitext(ref)[1].lower()
        parts = []
        prev, self._loading = self._loading, True
        try:
            if ext == ".txt":                        # decimal is the only format field the loader honours
                dec = detect_decimal(ref, self.state.settings.input.format.decimal)
                self.state.settings.input.format.decimal = dec
                parts.append(f"decimal '{dec}'")
            try:                                     # sampling frequency from the (col 0) time axis
                matrix, _names = load_raw_matrix(self.state.settings, ref)
                fs = detect_sampling_frequency(matrix[:, 0]) if matrix.shape[1] else None
            except Exception:                        # noqa: BLE001 — detection is best-effort
                fs = None
            if fs:
                self.state.settings.input.format.sampling_frequency = fs
                self.samp_freq.setValue(fs)
                parts.append(f"{fs} Hz")
        finally:
            self._loading = prev
        return ", ".join(parts)

    def _valid_input_files(self):
        """Sorted paths of files matching the input mask that are loadable data files with a
        consistent column count (the most common one) — so a stray non-data file that
        matches the mask is excluded and channel assignments carry across every listed file."""
        s = self.state.settings
        folder = (s.input.folder or "").strip()
        if not folder or not os.path.isdir(folder):
            return []
        from collections import Counter
        from respmech.ui.workers import probe_data_columns
        matches = matching_files(folder, s.input.files)
        probed = [(f, probe_data_columns(s, f)) for f in matches]
        probed = [(f, n) for f, n in probed if n and n >= 2]
        if not probed:
            return []
        counts = Counter(n for _f, n in probed)
        top = max(counts.values())
        ref = max(n for n, c in counts.items() if c == top)   # tie -> the widest layout
        return [f for f, n in probed if n == ref]

    def _current_channel_mapping(self):
        ch = self.state.settings.input.channels
        return {"flow": ch.flow, "volume": ch.volume, "poes": ch.poes, "pgas": ch.pgas,
                "pdi": ch.pdi, "emg": list(ch.emg), "entropy": list(ch.entropy)}

    def _apply_channel_mapping(self, m, fmt_note=""):
        """Write a role->column mapping from the modal into the channel widgets, then run
        the normal reactive commit (which advances the guided disclosure). ``fmt_note`` is
        an optional summary of any auto-detected file settings to mention in the status."""
        prev, self._loading = self._loading, True
        try:
            if m.get("flow"):
                self.col_flow.setValue(int(m["flow"]))
            self.col_volume.setValue(int(m.get("volume") or 0))
            if m.get("poes"):
                self.col_poes.setValue(int(m["poes"]))
            if m.get("pgas"):
                self.col_pgas.setValue(int(m["pgas"]))
            if m.get("pdi"):
                self.col_pdi.setValue(int(m["pdi"]))
            self.cols_emg.setText(",".join(str(c) for c in m.get("emg", [])))
            self.cols_entropy.setText(",".join(str(c) for c in m.get("entropy", [])))
            self._repopulate_ecg_detect()
        finally:
            self._loading = prev
        self._sync_widgets()
        self._on_inputs_changed()   # a narrowed mask means the file list may have changed
        n_emg = len(m.get("emg", []))
        msg = (f"Channels assigned from data ({n_emg} EMG channel"
               f"{'s' if n_emg != 1 else ''}).")
        if fmt_note:
            msg += f" Detected file settings: {fmt_note}."
        self._set_status(msg)

    def _update_disclosure(self):
        """Recompute which cards are revealed and whether the downstream tabs are ready.
        In 'new' mode the reveal is a monotonic high-water mark (a card never retracts
        once shown, so editing one can't yank another away), while the downstream gate is
        live (it re-locks if the settings become invalid). In any other mode everything
        is visible and the downstream tabs are always available."""
        self._update_qc()               # keep the live caution strip current in every mode
        if self._mode != "new":
            self._apply_card_visibility()
            self._set_flow_ready(True)
            return
        n_visible = 1
        for gate in self._stage_gate:
            if gate():
                n_visible += 1
            else:
                break
        # the visual channel-assignment modal sits between Output (stage 1) and the rest
        # (stage 2): once Output is valid, hold the remaining cards until the modal has
        # been dealt with, auto-opening it once
        if n_visible >= 3 and not self._channel_modal_done:
            n_visible = 2
            if not self._channel_modal_pending:
                self._channel_modal_pending = True
                QTimer.singleShot(0, self._open_channel_setup_for_flow)
        self._revealed = max(self._revealed, n_visible)      # monotonic
        self._apply_card_visibility()
        ready = self._all_ok()
        self._set_flow_ready(ready)
        self._set_status(self._new_mode_status(ready))       # friendly stage guidance

    def _apply_card_visibility(self):
        for i, cards in enumerate(self._stage_cards):
            visible = (self._mode != "new") or (i < self._revealed)
            for c in cards:
                c.setVisible(visible)

    def _new_mode_status(self, ready):
        """The guided-flow status line for the stage the user is on. The Input/Output
        stages stay card-friendly (their gate fields are the only thing that matters);
        once both pass, every card is revealed, so it is safe — and far more helpful —
        to name the actual remaining blocker rather than a vague 'almost done'."""
        if ready:
            base = "Settings complete — Preview and Run are now available."
            note = self._science_note()
            return f"{base} Note: {note}." if note else base
        if not self._input_stage_ok():
            # if the folder + frequency are fine, the only thing wrong is that the mask
            # matches nothing (easy to hit given the *.txt default) — say so instead of the
            # generic prompt, so the user knows to edit 'Files to analyse'
            s = self.state.settings
            folder = (s.input.folder or "").strip()
            fq = s.input.format.sampling_frequency
            if folder and os.path.isdir(folder) and isinstance(fq, int) and fq >= 1:
                if not matching_files(folder, s.input.files):
                    return (f"No files match '{s.input.files}' in this folder — "
                            "edit 'Files to analyse'.")
            return "New analysis — fill in the Input card to begin."
        if not self._output_stage_ok():
            return "Input set — now choose an Output folder."
        if not self._channel_modal_done:
            return "Output set — assign your data channels to continue."
        blocker = self._first_blocker()
        return f"Almost done — {blocker}." if blocker else \
            "Almost done — complete the remaining settings to unlock Preview."

    def _channel_collision(self):
        """A HARD channel-mapping error (message, else '') that the core's null-only
        validate() would miss but which gives a silently wrong analysis: a required channel
        (flow/poes/pgas/pdi) still pointing at column 1 — the time axis, the value they
        default to when the mapping is skipped/cancelled — or two of them sharing a column."""
        ch = self.state.settings.input.channels
        req = [("flow", ch.flow), ("poes", ch.poes), ("pgas", ch.pgas), ("pdi", ch.pdi)]
        on_time = [n for n, c in req if c == 1]
        if on_time:
            return (f"{', '.join(on_time)} still point at column 1 (the time axis) — "
                    "click 'Assign channels from data…'")
        cols = [c for _n, c in req if c]
        dup = sorted({c for c in cols if cols.count(c) > 1})
        if dup:
            names = [n for n, c in req if c in dup]
            return f"{', '.join(names)} are mapped to the same column"
        return ""

    def _first_blocker(self):
        """A short, human reason the analysis is not yet valid, for the final guided
        stage (channel collision, else core validation message, else the first path problem)."""
        collision = self._channel_collision()
        if collision:
            return collision
        try:
            self.state.settings.validate()
        except SettingsError as e:
            return str(e)
        except Exception:                           # noqa: BLE001 — unexpected -> no hint
            return None
        return self._path_problem()

    def _set_flow_ready(self, ready):
        ready = bool(ready)
        if ready != self._flow_ready:
            self._flow_ready = ready
            self.flow_ready_changed.emit(ready)

    # -- stage validity (semantic, so it survives future card re-placement) --
    def _input_stage_ok(self):
        """Input card complete: a real recordings folder with matching files, and a
        sampling frequency."""
        s = self.state.settings
        folder = (s.input.folder or "").strip()
        if not folder or not os.path.isdir(folder):
            return False
        if not matching_files(folder, s.input.files):
            return False
        fq = s.input.format.sampling_frequency
        return isinstance(fq, int) and fq >= 1

    def _output_stage_ok(self):
        """Output card complete: an output folder whose location exists on disk."""
        out = (self.state.settings.output.folder or "").strip()
        if not out:
            return False
        parent = out if os.path.isdir(out) else os.path.dirname(os.path.abspath(out))
        return os.path.isdir(parent)

    def _all_ok(self):
        """Every setting is valid (core validation + filesystem paths) — the gate that
        reveals the Preview/Run tabs."""
        try:
            self.state.settings.validate()
        except Exception:                           # noqa: BLE001 — any invalidity -> not ready
            return False
        if self._channel_collision():               # required channel on time / colliding
            return False                            # (validate() only null-checks) -> not ready
        mask = self.state.settings.input.files or ""
        if ";" in mask or "," in mask:              # a multi-pattern mask the core runner
            return False                            # can't glob -> not runnable (defensive)
        return self._path_problem() is None

    def _science_note(self):
        """A non-fatal science caution about the current settings, or '' — surfaced both by
        Validate and at the end of the guided flow so a first-timer sees it without clicking
        Validate. Covers EMG columns overlapping a pressure channel and a low EMG sample rate."""
        s = self.state.settings
        ch = s.input.channels
        clash = [n for n, c in (("flow", ch.flow), ("poes", ch.poes),
                                ("pgas", ch.pgas), ("pdi", ch.pdi)) if c and c in ch.emg]
        if clash:
            return f"EMG columns overlap {', '.join(clash)}"
        if ch.emg and (s.input.format.sampling_frequency or 0) < 1000:
            return "sampling frequency < 1000 Hz is low for EMG"
        return ""

    def _all_cautions(self):
        """Every current caution (hard channel collision first, then soft science notes) —
        for the live QC strip."""
        out = []
        c = self._channel_collision()
        if c:
            out.append(c)
        n = self._science_note()
        if n:
            out.append(n)
        return out

    def _update_qc(self):
        """Refresh the pinned QC strip with the live cautions (styled warn), or an all-clear."""
        if getattr(self, "qc", None) is None:
            return
        cautions = self._all_cautions()
        if cautions:
            self.qc.setText("⚠  " + "   ·   ".join(cautions))
            self.qc.setProperty("status", "warn")
        else:
            self.qc.setText("✓  No warnings — channels, columns and sampling look consistent.")
            self.qc.setProperty("status", "ok")
        st = self.qc.style()
        if st is not None:
            st.unpolish(self.qc); st.polish(self.qc)

    def _show_validation_status(self):
        """Run validation for its status line (used on open); never raises."""
        try:
            self._validate()
        except Exception:                           # noqa: BLE001 — status is cosmetic
            pass

    # -- actions ------------------------------------------------------------
    def _resync_form(self):
        """Best-effort refill of the form from the (restored) settings after a
        rolled-back load/import."""
        try:
            self.from_state()
        except Exception:                       # noqa: BLE001
            pass

    def _report_error(self, operation, detail):
        """Consistent failure surface: a short status line + a copyable full trace
        (matches the Preview/Run screens)."""
        self._set_status(f"{operation} failed — {short_error(detail)}")
        self._err_dialog = open_error_dialog(
            self, f"{operation} — error", detail,
            intro=f"{operation} failed. The full detail below is copyable.",
            prior=self._err_dialog)

    def _load(self, path=None):
        """Load a saved .toml analysis. Returns True on success, False if cancelled or
        rolled back, so the caller (open_analysis) can gate the mode transition."""
        p = path or QFileDialog.getOpenFileName(self, "Open analysis", ".", "Saved analysis (*.toml)")[0]
        if not p:
            return False
        prior, prior_path = self.state.settings, self.state.settings_path
        try:
            self.state.load_toml(p)
            self.from_state()
        except Exception:                       # noqa: BLE001 — roll back so nothing partially applies
            detail = traceback.format_exc()
            self.state.settings, self.state.settings_path = prior, prior_path
            self._resync_form()
            self._report_error("Open analysis", detail)
            return False
        self._set_status(f"Opened {p}")
        self._mark_clean()
        self.inputs_changed.emit()
        self.settings_changed.emit()
        return True

    def _save(self):
        self.to_state()
        p, _ = QFileDialog.getSaveFileName(self, "Save analysis", "analysis.toml", "Saved analysis (*.toml)")
        if not p:
            return
        try:
            self.state.save_toml(p)
        except Exception:                       # noqa: BLE001
            self._report_error("Save analysis", traceback.format_exc())
            return
        self._set_status(f"Saved {p}")
        self._mark_clean()

    def _import(self, path=None):
        """Import a legacy .py setup (runs the migrator). Returns True on success, False
        if cancelled or rolled back, so open_analysis can gate the mode transition."""
        p = path or QFileDialog.getOpenFileName(self, "Open legacy analysis (.py)", ".", "Legacy setup (*.py)")[0]
        if not p:
            return False
        prior, prior_path = self.state.settings, self.state.settings_path
        try:
            report = self.state.import_legacy(p)
            self.from_state()
        except Exception:                       # noqa: BLE001 — roll back so nothing partially applies
            detail = traceback.format_exc()
            self.state.settings, self.state.settings_path = prior, prior_path
            self._resync_form()
            self._report_error("Open legacy analysis", detail)
            return False
        self._report_dialog = open_error_dialog(
            self, "Migration report", report,
            intro="Legacy analysis imported and converted.", prior=self._report_dialog)
        self._set_status(f"Imported {p}")
        self._mark_clean()
        self.inputs_changed.emit()
        self.settings_changed.emit()
        return True

    def _validate(self):
        self.to_state()
        try:
            self.state.settings.validate()
        except SettingsError as e:
            self._set_status(f"Invalid: {e}")
            return
        except Exception:                       # noqa: BLE001 — unexpected: show a copyable trace
            self._report_error("Validate settings", traceback.format_exc())
            return
        # path checks (the core validate() is filesystem-agnostic)
        problem = self._path_problem()
        if problem:
            self._set_status(f"Invalid: {problem}")
            return
        # non-fatal science guardrails (shared with the guided-flow completion status)
        note = self._science_note()
        self._set_status(f"Valid, but check: {note}." if note else "Settings valid ✓")

    def _path_problem(self):
        """Return a human message for the first invalid path, or None if all OK.
        Shared with the Run screen so both surface the same filesystem checks."""
        from respmech.ui.validation import path_problem
        return path_problem(self.state.settings)


def _is_int(token):
    try:
        int(token)
        return True
    except ValueError:
        return False


def _parse_ints(text):
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if part and _is_int(part):
            out.append(int(part))
    return out
