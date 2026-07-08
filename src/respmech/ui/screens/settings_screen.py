"""Screen 1 — settings / batch setup.

A form bound to the typed settings model. Load/save TOML; import a legacy .py
settings file (runs the migrator and shows its report); inline validation reuses the
core ``Settings.validate``. Editing any field syncs the shared state live and emits
signals so the Preview/Run screens stay in step (e.g. the file list refreshes when
the input folder/mask changes). Only the most-used fields are surfaced; the full
model is preserved on load/save round-trips.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QScrollArea,
                               QSpinBox, QVBoxLayout, QWidget)
from PySide6.QtCore import Signal

from respmech.core.settings import SettingsError


class SettingsScreen(QWidget):
    settings_changed = Signal()     # any field edited -> shared state is current
    inputs_changed = Signal()       # input folder/mask edited -> file list is stale
    status_changed = Signal(str)

    def __init__(self, state, on_settings_changed=None):
        super().__init__()
        self.state = state
        self.on_settings_changed = on_settings_changed
        self._loading = True          # suppress reactions during programmatic fills
        self._build()
        self.from_state()
        self._wire_reactivity()
        self._loading = False

    # -- UI construction ----------------------------------------------------
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Action bar stays pinned above the scroll area so it is always reachable.
        bar = QHBoxLayout()
        for label, slot in (("Load TOML…", self._load), ("Save TOML…", self._save),
                            ("Import legacy .py…", self._import), ("Validate", self._validate)):
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
        self.in_folder.setToolTip("Folder containing the recordings to analyse.")
        self.in_files = QLineEdit()
        self.in_files.setToolTip("Filename or glob mask, e.g. *.txt, RIU_H5_*.txt")
        self.samp_freq = QSpinBox(); self.samp_freq.setRange(1, 1_000_000); self.samp_freq.setSuffix(" Hz")
        self.samp_freq.setToolTip("Samples per second of the recording.")
        f.addRow("Input folder", self._with_browse(self.in_folder, folder=True))
        f.addRow("File mask", self.in_files)
        f.addRow("Sampling frequency", self.samp_freq)
        root.addWidget(gin)

        # Channels ---------------------------------------------------------
        gch = QGroupBox("Channels — 1-based column numbers in the data file")
        fc = QFormLayout(gch)
        self.col_flow = self._spin(); self.col_volume = self._spin(allow_zero=True)
        self.col_poes = self._spin(); self.col_pgas = self._spin(); self.col_pdi = self._spin()
        self.cols_emg = QLineEdit(); self.cols_entropy = QLineEdit()
        self.col_flow.setToolTip("Flow (L/s); negative on inspiration.")
        self.col_volume.setToolTip("Inspired volume (L). 0 = integrate volume from flow.")
        self.col_poes.setToolTip("Oesophageal pressure Poes (cmH₂O).")
        self.col_pgas.setToolTip("Gastric pressure Pgas (cmH₂O).")
        self.col_pdi.setToolTip("Transdiaphragmatic pressure Pdi (cmH₂O).")
        self.cols_emg.setToolTip("Comma-separated 1-based columns of the EMG channels, e.g. 2,3,4,5,6")
        self.cols_entropy.setToolTip("Comma-separated columns for sample-entropy (may be empty).")
        for label, w in (("Flow", self.col_flow), ("Volume (0 = integrate from flow)", self.col_volume),
                        ("Oesophageal pressure (Poes)", self.col_poes),
                        ("Gastric pressure (Pgas)", self.col_pgas),
                        ("Transdiaphragmatic (Pdi)", self.col_pdi),
                        ("EMG columns (comma list)", self.cols_emg),
                        ("Entropy columns (comma list)", self.cols_entropy)):
            fc.addRow(label, w)
        root.addWidget(gch)

        # Processing — breath mechanics ------------------------------------
        gpr = QGroupBox("Processing")
        fp = QFormLayout(gpr)
        self.seg_method = QComboBox(); self.seg_method.addItems(["flow", "volume"])
        self.seg_method.setToolTip("Signal used to split the recording into breaths.")
        self.wob_from = QComboBox(); self.wob_from.addItems(["average", "individual"])
        self.wob_from.setToolTip("Compute work of breathing from an averaged breath, or per breath then average.")
        self.integrate = QCheckBox("Integrate volume from flow")
        fp.addRow("Breath separation", self.seg_method)
        fp.addRow("WOB from", self.wob_from)
        fp.addRow("", self.integrate)
        root.addWidget(gpr)

        # EMG — RMS envelope -----------------------------------------------
        gemg = QGroupBox("EMG — RMS envelope")
        gemg.setToolTip("How the diaphragm-EMG amplitude is quantified for each breath.")
        fe = QFormLayout(gemg)
        self.emg_rms_window = QDoubleSpinBox()
        self.emg_rms_window.setRange(0.01, 0.5); self.emg_rms_window.setSingleStep(0.01)
        self.emg_rms_window.setDecimals(3); self.emg_rms_window.setSuffix(" s")
        self.emg_rms_window.setToolTip(
            "Sliding-window length for the EMG root-mean-square (RMS) envelope; the "
            "breath value is the maximum windowed RMS. Longer = smoother but less "
            "temporal detail. Typical 0.05 s.")
        self.emg_outlier_sd = QDoubleSpinBox()
        self.emg_outlier_sd.setRange(0.0, 10.0); self.emg_outlier_sd.setSingleStep(0.5)
        self.emg_outlier_sd.setDecimals(1); self.emg_outlier_sd.setSuffix(" SD")
        self.emg_outlier_sd.setSpecialValueText("off (0)")   # minimum (0.0) reads as disabled
        self.emg_outlier_sd.setToolTip(
            "Replace per-breath RMS values lying more than this many standard "
            "deviations from the across-breath mean (0 = disabled). Guards against "
            "motion/artefact spikes distorting the average.")
        fe.addRow("RMS window", self.emg_rms_window)
        fe.addRow("RMS outlier limit", self.emg_outlier_sd)
        root.addWidget(gemg)

        # EMG — ECG removal ------------------------------------------------
        gecg = QGroupBox("EMG — ECG removal")
        gecg.setToolTip("Detect R-waves on one EMG channel and subtract an averaged "
                        "ECG template from every EMG channel.")
        fg = QFormLayout(gecg)
        self.remove_ecg = QCheckBox("Remove ECG from EMG")
        self.remove_ecg.setToolTip("Enable averaged-template ECG subtraction on the EMG channels.")
        self.ecg_detect = QComboBox()
        self.ecg_detect.setToolTip("Which EMG channel to detect R-waves on (use where the ECG is most prominent).")
        self.ecg_min_height = QDoubleSpinBox()
        self.ecg_min_height.setRange(0.0, 1_000_000.0); self.ecg_min_height.setDecimals(6)
        self.ecg_min_height.setSingleStep(0.0001)
        self.ecg_min_height.setToolTip(
            "Minimum R-wave peak amplitude for detection, in the SAME units as the "
            "raw EMG signal. Raise to ignore small peaks; lower if beats are missed.")
        self.ecg_min_distance = QDoubleSpinBox()
        self.ecg_min_distance.setRange(0.05, 2.0); self.ecg_min_distance.setSingleStep(0.05)
        self.ecg_min_distance.setDecimals(3); self.ecg_min_distance.setSuffix(" s")
        self.ecg_min_distance.setToolTip(
            "Minimum time between successive detected R-waves (refractory). Caps the "
            "detectable heart rate: 0.5 s ≈ 120 bpm ceiling — lower for tachycardia.")
        self.ecg_min_width = QDoubleSpinBox()
        self.ecg_min_width.setRange(0.0, 0.1); self.ecg_min_width.setSingleStep(0.001)
        self.ecg_min_width.setDecimals(4); self.ecg_min_width.setSuffix(" s")
        self.ecg_min_width.setToolTip(
            "Minimum width of an accepted R-wave peak. Rejects narrow "
            "spikes/artefacts. Typical ~0.001 s.")
        self.ecg_window = QDoubleSpinBox()
        self.ecg_window.setRange(0.05, 1.0); self.ecg_window.setSingleStep(0.05)
        self.ecg_window.setDecimals(3); self.ecg_window.setSuffix(" s")
        self.ecg_window.setToolTip(
            "Total width of the template window centred on each R-wave (half each "
            "side) that is averaged and subtracted from the EMG. Should span the "
            "QRS-T complex. Typical 0.4 s.")
        fg.addRow("", self.remove_ecg)
        fg.addRow("Detection channel", self.ecg_detect)
        fg.addRow("R-wave min height", self.ecg_min_height)
        fg.addRow("Min beat spacing", self.ecg_min_distance)
        fg.addRow("Min R-wave width", self.ecg_min_width)
        fg.addRow("Template window", self.ecg_window)
        root.addWidget(gecg)

        # EMG — noise reduction (shared profile) ---------------------------
        gns = QGroupBox("EMG — noise reduction (shared profile)")
        gns.setToolTip("One shared noise profile + one parameter set are built from the "
                       "reference and applied IDENTICALLY to every file in the test. "
                       "Tune the noise window visually in Preview ▸ EMG processing.")
        fn = QFormLayout(gns)
        self.remove_noise = QCheckBox("Reduce EMG noise (shared profile)")
        self.remove_noise.setToolTip("Enable spectral noise subtraction using the shared reference profile.")
        self.noise_ref = QLineEdit()
        self.noise_ref.setToolTip("A rest/baseline recording whose EMG-free portion defines the noise profile. "
                                  "You can also select a rest region on the EMG graph (Preview ▸ EMG processing).")
        self.noise_use_exp = QCheckBox("Use expiration of the reference (recommended)")
        self.noise_use_exp.setToolTip("Build the noise sample from the reference's expiration (many STFT frames = stable).")
        fn.addRow("", self.remove_noise)
        fn.addRow("Reference file", self._with_browse(self.noise_ref, folder=False))
        fn.addRow("", self.noise_use_exp)
        hint = QLabel("Noise-window options (suppression, fidelity target) are in "
                      "Preview ▸ EMG processing, active once a reference file is set.")
        hint.setWordWrap(True); hint.setProperty("status", "muted")
        fn.addRow("", hint)
        root.addWidget(gns)

        # Output -----------------------------------------------------------
        gout = QGroupBox("Output")
        fo = QFormLayout(gout)
        self.out_folder = QLineEdit()
        self.out_folder.setToolTip("Results are written to <output>/data/.")
        fo.addRow("Output folder", self._with_browse(self.out_folder, folder=True))
        root.addWidget(gout)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        root.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

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
        self.settings_changed.emit()

    def _on_inputs_changed(self, *_):
        if self._loading:
            return
        self.to_state()
        self.inputs_changed.emit()
        self.settings_changed.emit()

    def _on_emg_cols_changed(self, *_):
        if self._loading:
            return
        self._repopulate_ecg_detect()
        self._on_field_changed()

    def _set_status(self, text):
        self.status.setText(text)
        self.status_changed.emit(text)

    # -- actions ------------------------------------------------------------
    def _load(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load settings TOML", ".", "TOML (*.toml)")
        if p:
            self.state.load_toml(p)
            self.from_state()
            self._set_status(f"Loaded {p}")
            self.inputs_changed.emit()
            self.settings_changed.emit()

    def _save(self):
        self.to_state()
        p, _ = QFileDialog.getSaveFileName(self, "Save settings TOML", "settings.toml", "TOML (*.toml)")
        if p:
            self.state.save_toml(p)
            self._set_status(f"Saved {p}")

    def _import(self):
        p, _ = QFileDialog.getOpenFileName(self, "Import legacy settings .py", ".", "Python (*.py)")
        if p:
            report = self.state.import_legacy(p)
            self.from_state()
            dlg = QMessageBox(self); dlg.setWindowTitle("Migration report")
            dlg.setText("Legacy settings imported and converted.")
            dlg.setDetailedText(report)
            dlg.exec()
            self._set_status(f"Imported {p}")
            self.inputs_changed.emit()
            self.settings_changed.emit()

    def _validate(self):
        self.to_state()
        try:
            self.state.settings.validate()
        except SettingsError as e:
            self._set_status(f"Invalid: {e}")
            return
        # path checks (the core validate() is filesystem-agnostic)
        problem = self._path_problem()
        if problem:
            self._set_status(f"Invalid: {problem}")
            return
        # non-fatal science guardrails
        s = self.state.settings
        ch = s.input.channels
        mono = [("flow", ch.flow), ("poes", ch.poes), ("pgas", ch.pgas), ("pdi", ch.pdi)]
        clash = [name for name, col in mono if col and col in ch.emg]
        if clash:
            self._set_status(f"Valid, but check: EMG columns overlap {', '.join(clash)}.")
        elif ch.emg and (s.input.format.sampling_frequency or 0) < 1000:
            self._set_status("Valid, but sampling frequency < 1000 Hz is low for EMG.")
        else:
            self._set_status("Settings valid ✓")

    def _path_problem(self):
        """Return a human message for the first invalid path, or None if all OK."""
        import glob
        import os
        s = self.state.settings
        folder = (s.input.folder or "").strip()
        if not folder or not os.path.isdir(folder):
            return f"input folder does not exist: {folder or '(unset)'}"
        matches = [f for f in glob.glob(os.path.join(folder, s.input.files or "*.*"))
                   if os.path.isfile(f)]
        if not matches:
            return f"no files match '{s.input.files}' in the input folder"
        out = (s.output.folder or "").strip()
        if not out:
            return "output folder is not set"
        parent = out if os.path.isdir(out) else os.path.dirname(os.path.abspath(out))
        if not os.path.isdir(parent):
            return f"output folder's location does not exist: {out}"
        n = s.processing.emg.noise
        if n.enabled and n.reference_file:
            ref = n.reference_file
            if not os.path.isabs(ref):
                ref = os.path.join(folder, ref)
            if not os.path.isfile(ref):
                return f"noise reference file not found: {n.reference_file}"
        return None


def _parse_ints(text):
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if part:
            try:
                out.append(int(part))
            except ValueError:
                pass
    return out
