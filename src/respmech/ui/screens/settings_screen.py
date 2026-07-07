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
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget)
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
        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        for label, slot in (("Load TOML…", self._load), ("Save TOML…", self._save),
                            ("Import legacy .py…", self._import), ("Validate", self._validate)):
            b = QPushButton(label)
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addStretch(1)
        root.addLayout(bar)

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

        # Processing -------------------------------------------------------
        gpr = QGroupBox("Processing")
        fp = QFormLayout(gpr)
        self.seg_method = QComboBox(); self.seg_method.addItems(["flow", "volume"])
        self.seg_method.setToolTip("Signal used to split the recording into breaths.")
        self.wob_from = QComboBox(); self.wob_from.addItems(["average", "individual"])
        self.wob_from.setToolTip("Compute work of breathing from an averaged breath, or per breath then average.")
        self.integrate = QCheckBox("Integrate volume from flow")
        self.remove_ecg = QCheckBox("Remove ECG from EMG")
        self.remove_noise = QCheckBox("Reduce EMG noise (shared profile)")
        fp.addRow("Breath separation", self.seg_method)
        fp.addRow("WOB from", self.wob_from)
        fp.addRow("", self.integrate)
        fp.addRow("", self.remove_ecg)
        fp.addRow("", self.remove_noise)
        root.addWidget(gpr)

        # EMG noise reduction & ECG detection ------------------------------
        gns = QGroupBox("EMG noise reduction & ECG detection")
        gns.setToolTip("One shared noise profile + one parameter set are built from the "
                       "rest reference and applied IDENTICALLY to every file in the test.")
        fn = QFormLayout(gns)
        self.noise_ref = QLineEdit()
        self.noise_ref.setToolTip("A rest/baseline recording whose EMG-free portion defines the noise profile.")
        self.noise_use_exp = QCheckBox("Use expiration of the reference (recommended)")
        self.noise_use_exp.setToolTip("Build the noise sample from the reference's expiration (many STFT frames = stable).")
        self.noise_auto = QCheckBox("Auto-select suppression (keep worst channel ≥ fidelity target)")
        self.noise_target = QDoubleSpinBox(); self.noise_target.setRange(0.50, 0.99); self.noise_target.setSingleStep(0.05)
        self.noise_target.setToolTip("Minimum fraction of inspiratory EMG power to retain (over-subtraction guard).")
        self.noise_prop = QDoubleSpinBox(); self.noise_prop.setRange(0.0, 1.0); self.noise_prop.setSingleStep(0.05)
        self.noise_prop.setToolTip("Manual suppression strength (0 = none, 1 = full) when auto-select is off.")
        self.noise_nstd = QDoubleSpinBox(); self.noise_nstd.setRange(0.0, 10.0); self.noise_nstd.setSingleStep(0.1)
        self.noise_nstd.setToolTip("Gate threshold: mean + n_std·SD of the noise spectrum. Lower = gentler.")
        self.ecg_detect = QComboBox()
        self.ecg_detect.setToolTip("Which EMG channel to detect R-waves on (use where the ECG is most prominent).")
        fn.addRow("Rest reference file", self._with_browse(self.noise_ref, folder=False))
        fn.addRow("", self.noise_use_exp)
        fn.addRow("", self.noise_auto)
        fn.addRow("Fidelity target (0–1)", self.noise_target)
        fn.addRow("Manual suppression (prop_decrease)", self.noise_prop)
        fn.addRow("n_std_thresh", self.noise_nstd)
        fn.addRow("ECG detection channel", self.ecg_detect)
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

    def _spin(self, allow_zero=False):
        s = QSpinBox(); s.setRange(0 if allow_zero else 1, 9999); return s

    def _with_browse(self, line: QLineEdit, folder=False):
        w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(line)
        b = QPushButton("…"); b.setFixedWidth(30)
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
            self.remove_ecg.setChecked(s.processing.emg.remove_ecg)

            emg, n = s.processing.emg, s.processing.emg.noise
            self.remove_noise.setChecked(n.enabled)               # rebind to the real gate
            self.noise_ref.setText(n.reference_file or "")
            self.noise_use_exp.setChecked(n.use_expiration)
            self.noise_auto.setChecked(n.auto_prop)
            self.noise_target.setValue(n.fidelity_target)
            self.noise_prop.setValue(n.prop_decrease)
            self.noise_nstd.setValue(n.n_std_thresh)
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
        s.processing.emg.remove_ecg = self.remove_ecg.isChecked()

        emg, n = s.processing.emg, s.processing.emg.noise
        n.enabled = self.remove_noise.isChecked()
        emg.remove_noise = self.remove_noise.isChecked()      # keep legacy mirror so TOML round-trips
        n.reference_file = self.noise_ref.text().strip() or None
        n.use_expiration = self.noise_use_exp.isChecked()
        n.auto_prop = self.noise_auto.isChecked()
        n.fidelity_target = self.noise_target.value()
        n.prop_decrease = self.noise_prop.value()
        n.n_std_thresh = self.noise_nstd.value()
        emg.detect_channel = max(0, self.ecg_detect.currentIndex())
        s.output.folder = self.out_folder.text()
        if self.on_settings_changed:
            self.on_settings_changed()
        return s

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
        for w in (self.noise_ref, self.noise_use_exp, self.noise_auto,
                  self.noise_target, self.noise_prop, self.noise_nstd):
            w.setEnabled(noise_on)
        if noise_on:
            self.noise_prop.setEnabled(not self.noise_auto.isChecked())
        self.ecg_detect.setEnabled(self.remove_ecg.isChecked())
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
        for dsb in (self.noise_target, self.noise_prop, self.noise_nstd):
            dsb.valueChanged.connect(self._on_field_changed)
        for cb in (self.seg_method, self.wob_from):
            cb.currentTextChanged.connect(self._on_field_changed)
        self.ecg_detect.currentIndexChanged.connect(self._on_field_changed)
        for chk in (self.integrate, self.remove_ecg, self.remove_noise,
                    self.noise_use_exp, self.noise_auto):
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
