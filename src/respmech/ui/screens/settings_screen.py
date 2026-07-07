"""Screen 1 — settings / batch setup.

A form bound to the settings model. Load/save TOML; import a legacy .py settings
file (runs the migrator and shows its report). Inline validation reuses the core
``Settings.validate``. Only the most-used fields are surfaced as widgets; the full
model is preserved on load/save round-trips.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QSpinBox, QTextEdit,
                               QVBoxLayout, QWidget)

from respmech.core.settings import SettingsError


class SettingsScreen(QWidget):
    def __init__(self, state, on_settings_changed=None):
        super().__init__()
        self.state = state
        self.on_settings_changed = on_settings_changed
        self._build()
        self.from_state()

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

        # Input
        gin = QGroupBox("Input")
        f = QFormLayout(gin)
        self.in_folder = QLineEdit()
        self.in_files = QLineEdit()
        self.samp_freq = QSpinBox(); self.samp_freq.setRange(1, 1_000_000)
        f.addRow("Input folder", self._with_browse(self.in_folder, folder=True))
        f.addRow("File mask", self.in_files)
        f.addRow("Sampling frequency (Hz)", self.samp_freq)
        root.addWidget(gin)

        # Channels
        gch = QGroupBox("Channels (1-based column numbers)")
        fc = QFormLayout(gch)
        self.col_flow = self._spin(); self.col_volume = self._spin(allow_zero=True)
        self.col_poes = self._spin(); self.col_pgas = self._spin(); self.col_pdi = self._spin()
        self.cols_emg = QLineEdit(); self.cols_entropy = QLineEdit()
        for label, w in (("Flow", self.col_flow), ("Volume (0 = integrate from flow)", self.col_volume),
                        ("Poes", self.col_poes), ("Pgas", self.col_pgas), ("Pdi", self.col_pdi),
                        ("EMG columns (comma list)", self.cols_emg),
                        ("Entropy columns (comma list)", self.cols_entropy)):
            fc.addRow(label, w)
        root.addWidget(gch)

        # Processing
        gpr = QGroupBox("Processing")
        fp = QFormLayout(gpr)
        self.seg_method = QComboBox(); self.seg_method.addItems(["flow", "volume"])
        self.wob_from = QComboBox(); self.wob_from.addItems(["average", "individual"])
        self.integrate = QCheckBox("Integrate volume from flow")
        self.remove_ecg = QCheckBox("Remove ECG")
        self.remove_noise = QCheckBox("Reduce EMG noise")
        fp.addRow("Breath separation", self.seg_method)
        fp.addRow("WOB from", self.wob_from)
        fp.addRow("", self.integrate)
        fp.addRow("", self.remove_ecg)
        fp.addRow("", self.remove_noise)
        root.addWidget(gpr)

        # Output
        gout = QGroupBox("Output")
        fo = QFormLayout(gout)
        self.out_folder = QLineEdit()
        fo.addRow("Output folder", self._with_browse(self.out_folder, folder=True))
        root.addWidget(gout)

        self.status = QLabel("")
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

    # -- state <-> widgets --------------------------------------------------
    def from_state(self):
        s = self.state.settings
        self.in_folder.setText(s.input.folder)
        self.in_files.setText(s.input.files)
        self.samp_freq.setValue(s.input.format.sampling_frequency or 1)
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
        self.remove_noise.setChecked(s.processing.emg.remove_noise)
        self.out_folder.setText(s.output.folder)

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
        s.processing.emg.remove_noise = self.remove_noise.isChecked()
        s.output.folder = self.out_folder.text()
        if self.on_settings_changed:
            self.on_settings_changed()
        return s

    # -- actions ------------------------------------------------------------
    def _load(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load settings TOML", ".", "TOML (*.toml)")
        if p:
            self.state.load_toml(p)
            self.from_state()
            self.status.setText(f"Loaded {p}")

    def _save(self):
        self.to_state()
        p, _ = QFileDialog.getSaveFileName(self, "Save settings TOML", "settings.toml", "TOML (*.toml)")
        if p:
            self.state.save_toml(p)
            self.status.setText(f"Saved {p}")

    def _import(self):
        p, _ = QFileDialog.getOpenFileName(self, "Import legacy settings .py", ".", "Python (*.py)")
        if p:
            report = self.state.import_legacy(p)
            self.from_state()
            dlg = QMessageBox(self); dlg.setWindowTitle("Migration report")
            dlg.setText("Legacy settings imported and converted.")
            dlg.setDetailedText(report)
            dlg.exec()

    def _validate(self):
        self.to_state()
        try:
            self.state.settings.validate()
            self.status.setText("Settings valid ✓")
        except SettingsError as e:
            self.status.setText(f"Invalid: {e}")


def _parse_ints(text):
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out
