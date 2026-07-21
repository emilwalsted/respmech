"""Screen 1 — settings / batch setup.

A form bound to the typed settings model. Load/save TOML; import a legacy .py
settings file (runs the migrator and shows its report); inline validation reuses the
core ``Settings.validate``. Editing any field syncs the shared state live and emits
signals so the Preview/Run screens stay in step (e.g. the file list refreshes when
the input folder/mask changes). Only the most-used fields are surfaced; the full
model is preserved on load/save round-trips.
"""
from __future__ import annotations

import os
import traceback

from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
                               QScrollArea, QSpinBox, QVBoxLayout, QWidget)
from PySide6.QtCore import Signal, QTimer

from respmech.core.settings import BreathCountEntry, Settings, SettingsError
from respmech.ui.dialogs import open_error_dialog, short_error
from respmech.ui.help_text import tooltip as _tip
from respmech.ui.startup_dialog import LEGACY_FILTER, OPEN_FILTER, TOML_FILTER
from respmech.ui.validation import matching_files
from respmech.ui import wheel as _wheel
from respmech.ui.channel_summary import ChannelSummary

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
        # conditional cards default to visible as built, so settle them once against the
        # analysis we just loaded rather than waiting for the first edit
        self._apply_card_visibility()
        self._loading = False

    # -- UI construction ----------------------------------------------------
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(11, 11, 11, 11)   # one shared inset for the form + QC strip

        # New/Open/Save are NOT here: opening and saving an analysis are window-level
        # actions rather than part of the Setup step, so they live in the header's Analysis
        # menu and stay reachable from every tab (see main_window._make_header). There is no
        # Validate button either — every edit re-validates (see _update_disclosure) and
        # reports into the window's status bar.

        # The long vertical form lives inside a scroll area so every section is
        # reachable on small screens. setWidgetResizable(True) keeps the content
        # at the viewport width (no horizontal scrollbar). All self.* widgets
        # below are still built exactly as before — only their container changed.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        root = QVBoxLayout(content); root.setContentsMargins(0, 0, 0, 0)   # align groups to the 11px inset

        # Input ------------------------------------------------------------
        gin = QGroupBox("Input")
        f = QFormLayout(gin)
        f.setRowWrapPolicy(QFormLayout.WrapLongRows)   # long labels wrap the field below instead of clipping
        self.in_folder = QLineEdit()   # absolute path + Browse button: full width is the point
        self.in_files = QLineEdit()
        self.in_files.setProperty("formField", "compact")   # a short glob mask, not a path (theme.py)
        self.samp_freq = QSpinBox(); self.samp_freq.setRange(1, 1_000_000); self.samp_freq.setSuffix(" Hz")
        self._browse_row(f, "Recordings folder", self.in_folder, "input.folder",
                         "Folder containing the recording files to analyse; defaults to 'input'.", folder=True)
        self._row(f, "Files to analyse", self.in_files, "input.files",
                  "Filename or wildcard mask picking which recordings to load, e.g. *.txt; defaults to *.* (all files).")
        self._row(f, "Sampling frequency", self.samp_freq, "input.format.sampling_frequency",
                  "Samples recorded per second, in hertz (Hz); required, and must match the acquisition system (e.g. 2000).")
        # P28: a live read-out of what was actually detected in the chosen recordings.
        self.format_readout = QLabel("")
        self.format_readout.setWordWrap(True)
        self.format_readout.setProperty("banner", True)   # box baked at first polish (theme.py)
        self.format_readout.setProperty("status", "muted")
        f.addRow("", self.format_readout)
        self.matlab_variant = QComboBox()
        # 'wide', not 'compact': "MATLAB (Unix/Mac)" plus the drop-down arrow outgrows the
        # 150px column on wider fonts (measured 131px of text against the column's 126px
        # text box on the CI runners' fonts) — the cap must never clip its own content.
        self.matlab_variant.setProperty("formField", "wide")
        self.matlab_variant.addItem("MATLAB (Windows)", "windows")
        self.matlab_variant.addItem("MATLAB (Unix/Mac)", "mac")
        self._row(f, "MATLAB file variant", self.matlab_variant, "input.format.matlab_variant",
                  "Variant/byte-order for .mat input files (ignored for CSV/Excel/text).")
        root.addWidget(gin)

        # Output (second in the flow: an analysis reads as Input -> Output -> rest) --
        gout = QGroupBox("Output")
        fo = QFormLayout(gout)
        fo.setRowWrapPolicy(QFormLayout.WrapLongRows)   # long labels wrap the field below instead of clipping
        self.out_folder = QLineEdit()
        self._browse_row(fo, "Output folder", self.out_folder, "output.folder",
                         "Where results are saved; files are written to a 'data' subfolder inside it; defaults to 'output'.", folder=True)
        root.addWidget(gout)

        # Channels ---------------------------------------------------------
        gch = QGroupBox("Channels")
        fc = QVBoxLayout(gch)
        fc.setSpacing(8)
        self.btn_assign_channels = QPushButton("Assign channels from data…")
        self.btn_assign_channels.setProperty("compact", True)   # a button, not a banner
        self.btn_assign_channels.setToolTip(
            "Open a visual picker: plot every column of your data and pick what each one is.")
        self.btn_assign_channels.clicked.connect(lambda: self._open_channel_setup())
        _brow = QWidget(); _bl = QHBoxLayout(_brow)
        _bl.setContentsMargins(0, 0, 0, 0)
        _bl.addWidget(self.btn_assign_channels); _bl.addStretch(1)
        fc.addWidget(_brow)
        # Read-only: the dialog is the only way to assign a channel, so this says what was
        # chosen rather than asking for it. Typing a column number against a column count you
        # cannot see is what the picker replaced.
        self.channel_summary = ChannelSummary()
        fc.addWidget(self.channel_summary)
        root.addWidget(gch)

        # Processing — breath mechanics ------------------------------------
        gpr = QGroupBox("Mechanics")
        fp = QFormLayout(gpr)
        fp.setRowWrapPolicy(QFormLayout.WrapLongRows)   # long labels wrap the field below instead of clipping
        self.seg_method = QComboBox()
        self.seg_method.setProperty("formField", "compact")   # (theme.py)
        self.seg_method.addItem("Flow", "flow"); self.seg_method.addItem("Volume", "volume")
        self.wob_from = QComboBox()
        self.wob_from.setProperty("formField", "compact")
        self.wob_from.addItem("Average", "average"); self.wob_from.addItem("Individual", "individual")
        self.integrate = QCheckBox("Calculate volume from flow")
        self._row(fp, "Signal used to split breaths", self.seg_method, "processing.segmentation.method",
                  "Which signal marks where each breath begins and ends — flow (the default) or volume.")
        self._row(fp, "Work of breathing calculation", self.wob_from, "processing.wob.calc_from",
                  "Compute work of breathing from one averaged breath (the default), or from each breath separately and then average.")
        self._check_row(fp, self.integrate, "processing.volume.integrate_from_flow",
                        "Derive volume by integrating the flow signal instead of reading a separate volume channel; off by default.")

        # -- Volume & drift transforms. These CHANGE the volume trace, the Campbell loop
        #    area and therefore WOB, so they are surfaced (not left to hidden model defaults);
        #    correct_drift in particular defaults ON. --
        _vsub = QLabel("Volume and drift corrections"); _vsub.setProperty("status", "muted")
        _vsub.setContentsMargins(0, 8, 0, 0)
        fp.addRow(_vsub)
        self.correct_drift = QCheckBox("Correct volume drift")
        self.correct_trend = QCheckBox("Correct end-expiratory trend")
        self.trend_method = QComboBox()
        self.trend_method.setProperty("formField", "compact")
        for _lbl, _tok in (("Linear", "linear"), ("Nearest", "nearest"), ("Cubic", "cubic"),
                           ("Quadratic", "quadratic"), ("Previous", "previous"), ("Next", "next")):
            self.trend_method.addItem(_lbl, _tok)
        self.inverse_flow = QCheckBox("Invert the flow signal")
        self.inverse_volume = QCheckBox("Invert the volume signal")
        self.resample = QCheckBox("Resample before analysis")
        self.resample_hz = QSpinBox(); self.resample_hz.setRange(1, 1_000_000); self.resample_hz.setSuffix(" Hz")
        self._check_row(fp, self.correct_drift, "processing.volume.correct_drift",
                        "Remove slow baseline drift from the volume trace before analysis; ON by default.")
        self._check_row(fp, self.correct_trend, "processing.volume.correct_trend",
                        "Remove a between-breath trend in end-expiratory lung volume; off by default.")
        self._row(fp, "Trend interpolation", self.trend_method, "processing.volume.trend_method",
                  "How the end-expiratory trend is interpolated between breaths (a scipy interp1d kind).")
        self._check_row(fp, self.inverse_flow, "processing.volume.inverse_flow",
                        "Flip the sign of the flow signal, if inspiration reads positive in your recordings.")
        self._check_row(fp, self.inverse_volume, "processing.volume.inverse_volume",
                        "Flip the sign of the volume signal.")
        self._check_row(fp, self.resample, "processing.sampling.resample",
                        "Resample every recording to a common frequency before analysis; off by default.")
        self._row(fp, "Resample to", self.resample_hz, "processing.sampling.resample_to_frequency",
                  "Target sampling frequency (Hz) used when 'Resample before analysis' is on.")
        _rshint = QLabel("Keep ≥ ~1000 Hz when EMG channels are present — a low rate discards EMG "
                         "energy and degrades RMS / entropy / noise reduction.")
        _rshint.setWordWrap(True); _rshint.setProperty("status", "muted")
        fp.addRow("", _rshint)
        self.btn_mech_advanced = QPushButton("Advanced…")
        self.btn_mech_advanced.setProperty("compact", True)
        self.btn_mech_advanced.setToolTip("Breath-detection tuning, the averaging grid, the "
                                          "PTP baseline and per-file breath-count overrides.")
        self.btn_mech_advanced.clicked.connect(self._open_mech_advanced)
        _mrow = QWidget(); _ml = QHBoxLayout(_mrow)
        _ml.setContentsMargins(0, 0, 0, 0)
        _ml.addWidget(self.btn_mech_advanced); _ml.addStretch(1)
        fp.addRow("", _mrow)
        root.addWidget(gpr)

        # EMG — RMS envelope -----------------------------------------------
        gemg = QGroupBox("EMG — RMS envelope")
        gemg.setToolTip("How the diaphragm-EMG amplitude is quantified for each breath.")
        fe = QFormLayout(gemg)
        fe.setRowWrapPolicy(QFormLayout.WrapLongRows)   # long labels wrap the field below instead of clipping
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
        self.emg_norm = QComboBox()
        self.emg_norm.setProperty("formField", "wide")   # its choices are sentences, not words
        for _label, _token in (("None — raw amplitude", "none"),
                               ("Per-file maximum (% of peak breath)", "per_file_max"),
                               ("Per-file mean (% of mean breath)", "per_file_mean")):
            self.emg_norm.addItem(_label, _token)
        self._row(fe, "EMG amplitude normalisation", self.emg_norm, "processing.emg.normalization",
                  "Also report each file's EMG RMS as a percentage of a per-file reference so amplitudes "
                  "compare across subjects/electrodes. Adds a normalised sheet to the output; never changes "
                  "the raw RMS (default: per-file maximum).")
        root.addWidget(gemg)

        # EMG — ECG removal is tuned LIVE on the Preview screen now (the "› EMG – ECG
        # reduction" tab shows the detected R-peak capture on every EMG channel), so it no
        # longer lives here — only a pointer remains.
        gecg = QLabel("EMG — ECG removal is now tuned live on the Preview screen, on the "
                      "“› EMG – ECG reduction” tab (pick the capture channel + parameters, "
                      "or use Auto-suggest, and watch the detected heartbeats).")
        gecg.setProperty("status", "muted"); gecg.setWordWrap(True)
        root.addWidget(gecg)

        # EMG — noise reduction (shared profile) ---------------------------
        gns = QGroupBox("EMG — noise reduction (shared profile)")
        gns.setToolTip("One shared noise profile + one parameter set are built from the "
                       "reference and applied IDENTICALLY to every file in the test. "
                       "Tune the noise window visually on the Preview screen's "
                       "› EMG – noise reduction tab.")
        fn = QFormLayout(gns)
        fn.setRowWrapPolicy(QFormLayout.WrapLongRows)   # long labels wrap the field below instead of clipping
        self.remove_noise = QCheckBox("Reduce EMG background noise")
        self._check_row(fn, self.remove_noise, "processing.emg.noise.enabled",
                        "Subtracts a shared spectral noise profile, built from a rest reference, from every EMG channel; off by default.")
        # READ-ONLY. The reference used to be settable here as well as in the picker, and the
        # two disagreed in both directions: Browse stored an ABSOLUTE path (so the analysis
        # stopped being portable, and only worked at all because os.path.join ignores its base
        # when the second argument is absolute) while the picker stores a bare file name; and
        # the "from expiration" checkbox here was silently unticked by the picker whenever a
        # span was marked, then — re-ticked — silently made that span inert. The picker owns
        # both halves now, and this says what it chose.
        self.noise_summary = QLabel("")
        self.noise_summary.setWordWrap(True)
        self.noise_summary.setToolTip(_tip(
            "processing.emg.noise.reference_file",
            "The rest recording, and which part of it, that defines the noise profile applied "
            "to every file in the test. Chosen with 'Set noise profile' on the Preview "
            "screen's › EMG – noise reduction tab."))
        fn.addRow("Reference", self.noise_summary)
        hint = QLabel("Pick the reference — and the suppression and fidelity target — with "
                      "'Set noise profile' on the Preview screen's › EMG – noise reduction tab.")
        hint.setWordWrap(True); hint.setProperty("status", "muted")
        fn.addRow("", hint)
        root.addWidget(gns)

        # Output — what to save. The deliverable set was previously governed by hidden model
        # defaults (only the folder was surfaced); this makes every workbook/CSV/figure an
        # explicit, per-run choice. Lives in the 'rest' stage so the guided Output step stays
        # to just the folder.
        # Entropy ----------------------------------------------------------
        # Shown only when a column is actually assigned to entropy — see _cond_cards. Its two
        # parameters are meaningless otherwise, and burying them in "Advanced (rarely
        # changed)" hid them from the users who DO compute entropy.
        gent = QGroupBox("Sample entropy")
        fent = QFormLayout(gent)
        fent.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.ent_epochs = QSpinBox(); self.ent_epochs.setRange(1, 100)
        self._row(fent, "Embedding (m)", self.ent_epochs, "processing.entropy.epochs",
                  "Embedding dimension (m) for sample entropy; 2 by convention.")
        self.ent_tol = QDoubleSpinBox(); self.ent_tol.setRange(0.0, 10.0)
        self.ent_tol.setDecimals(4); self.ent_tol.setSingleStep(0.05)
        self._row(fent, "Tolerance (r)", self.ent_tol, "processing.entropy.tolerance",
                  "Matching tolerance (r) for sample entropy; 0.1 by convention.")
        _enthint = QLabel("Computed on the columns ticked as Entropy in the channel picker.")
        _enthint.setWordWrap(True); _enthint.setProperty("status", "muted")
        fent.addRow("", _enthint)
        root.addWidget(gent)

        gsave = QGroupBox("Output — what to save")
        fsv = QFormLayout(gsave)
        fsv.setRowWrapPolicy(QFormLayout.WrapLongRows)   # long labels wrap the field below instead of clipping
        self.save_average = QCheckBox("Average breath-data workbook")
        self.save_bbb = QCheckBox("Breath-by-breath workbook (per file)")
        self.save_processed = QCheckBox("Processed-signal CSV (per file)")
        self.include_ignored = QCheckBox("Include excluded breaths in the processed CSV")
        self.save_pv_avg = QCheckBox("Campbell / PV diagram — averaged")
        self.save_pv_ind = QCheckBox("Campbell / PV diagram — individual breaths")
        self.save_raw_fig = QCheckBox("Raw-signal figures")
        self.save_trimmed_fig = QCheckBox("Trimmed-signal figures")
        self.save_drift_fig = QCheckBox("Drift-correction figures")
        self.save_emg_fig = QCheckBox("EMG channel overviews")
        _lt = QLabel("Tables"); _lt.setProperty("status", "muted"); _lt.setContentsMargins(0, 8, 0, 0); fsv.addRow(_lt)
        for cb, var, tip in (
                (self.save_average, "output.data.save_average", "The across-breath average workbook."),
                (self.save_bbb, "output.data.save_breath_by_breath", "A per-file breath-by-breath workbook."),
                (self.save_processed, "output.data.save_processed", "The processed signal as a CSV per file."),
                (self.include_ignored, "output.data.include_ignored_breaths",
                 "Include the breaths you've excluded in the per-file processed-signal CSV only. "
                 "Excluded breaths are always left out of the averages and workbooks; this does not change them.")):
            self._check_row(fsv, cb, var, tip)
        _lf = QLabel("Diagnostic figures"); _lf.setProperty("status", "muted"); _lf.setContentsMargins(0, 8, 0, 0); fsv.addRow(_lf)
        for cb, var, tip in (
                (self.save_pv_avg, "output.diagnostics.save_pv_average", "The averaged Campbell (pressure–volume) diagram."),
                (self.save_pv_ind, "output.diagnostics.save_pv_individual", "A Campbell diagram per individual breath."),
                (self.save_raw_fig, "output.diagnostics.save_raw", "Raw-signal diagnostic figures."),
                (self.save_trimmed_fig, "output.diagnostics.save_trimmed", "Trimmed-signal diagnostic figures."),
                (self.save_drift_fig, "output.diagnostics.save_drift", "Drift-correction diagnostic figures."),
                (self.save_emg_fig, "output.diagnostics.save_emg",
                 "Per-channel EMG overview figures (raw / ECG-removed / noise-reduced) with the flow "
                 "reference and R-peak capture markers.")):
            self._check_row(fsv, cb, var, tip)
        _lg = QLabel("Cohort summary"); _lg.setProperty("status", "muted"); _lg.setContentsMargins(0, 8, 0, 0); fsv.addRow(_lg)
        self.group_regex = QLineEdit()
        # 'wide', not 'compact': the placeholder below measures 281px at the 13pt macOS
        # font — it fits the 320px 'wide' contents box but would elide in the spin column,
        # and a QLineEdit's sizeHint ignores its placeholder, so no sizeHint check catches
        # a clipped one.
        self.group_regex.setProperty("formField", "wide")
        self.group_regex.setPlaceholderText("leading filename token (e.g. P03_120W → P03)")
        self._row(fsv, "Group files by", self.group_regex, "output.group_regex",
                  "How files are grouped (subject / condition) for the by-group summary. Leave blank to use "
                  "the leading filename token; or enter a regular expression whose first capture group is the key.")
        self.save_preview = QLabel(""); self.save_preview.setWordWrap(True)
        self.save_preview.setProperty("banner", True)
        self.save_preview.setProperty("status", "info")
        fsv.addRow("You will get", self.save_preview)
        root.addWidget(gsave)

        # --- Advanced (rarely changed): knobs that were previously TOML-only (audit #16/17/22-26)
        # The Mechanics "Advanced…" settings keep their widgets — from_state/to_state still
        # round-trip through them, which is what keeps the analysis file honest — but they are
        # no longer laid out anywhere. The modal edits its own copies and commits into these,
        # so Cancel costs nothing and the commit funnel is unchanged.
        self.seg_buffer = QSpinBox(); self.seg_buffer.setRange(0, 100_000)
        self.peak_height = QDoubleSpinBox(); self.peak_height.setRange(0.0, 1_000_000.0)
        self.peak_height.setDecimals(4); self.peak_height.setSingleStep(0.01)
        self.peak_distance = QDoubleSpinBox(); self.peak_distance.setRange(0.0, 60.0)
        self.peak_distance.setDecimals(4); self.peak_distance.setSingleStep(0.05); self.peak_distance.setSuffix(" s")
        self.peak_width = QDoubleSpinBox(); self.peak_width.setRange(0.0, 60.0)
        self.peak_width.setDecimals(4); self.peak_width.setSingleStep(0.05); self.peak_width.setSuffix(" s")
        self.avg_resamp = QSpinBox(); self.avg_resamp.setRange(10, 100_000)
        self.ptp_baseline = QDoubleSpinBox(); self.ptp_baseline.setRange(0.0, 1.0)
        self.ptp_baseline.setDecimals(4); self.ptp_baseline.setSingleStep(0.01); self.ptp_baseline.setSuffix(" s")
        self.breath_counts_edit = QPlainTextEdit()
        self.breath_counts_edit.setProperty("formField", "wide")   # holds 'filename = count' lines (theme.py)
        self.breath_counts_edit.setPlaceholderText("one 'filename = count' per line")
        self.breath_counts_edit.setFixedHeight(64)

        # Status text is shown in the window's bottom status bar (see main_window). This
        # label is kept only as a hidden text holder mirroring that message — showing it
        # inline too would duplicate the same sentence on screen.
        self.status = QLabel("")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        self.status.hide()
        root.addStretch(1)

        scroll.setWidget(content)
        # Scrolling the form must not edit it. Every spin box and combo here accepts the
        # wheel and steps its own value, so scrolling past one silently changed a setting,
        # marked the analysis modified and scheduled a recompute — a wheel over "EMG RMS
        # window" moved it from 0.05 s to 0.02 s, which changes every reported EMG number.
        self._wheel_guard = _wheel.guard_scroll_area(scroll)
        # The breath-count box has its own scrollbar, so it was a dead patch in the middle of
        # the form: it ate the wheel even when already at its own end. Let it scroll itself
        # first, then hand the wheel out.
        self._wheel_chain = _wheel.chain_nested(scroll, self.breath_counts_edit)
        outer.addWidget(scroll, 1)

        # Live QC strip, pinned below the (scrolling) form: every current caution at a
        # glance, so a first-timer sees them the moment they appear.
        self.qc = QLabel("")
        self.qc.setWordWrap(True)
        self.qc.setProperty("banner", True)   # the box comes from the QSS, not extra margins
        outer.addWidget(self.qc)

        # --- progressive-disclosure stages (used only in guided "new analysis" mode) -
        # Ordered reveal groups. Placement-agnostic: reordering or moving a card only
        # changes this registry, not the reveal logic. Stage 0 shows immediately; each
        # later stage's cards appear once the PRECEDING stage's semantic gate passes,
        # and the Preview/Run tabs unlock once everything validates (_all_ok).
        self._stage_cards = [
            [gin],                                   # 0: Input (always shown)
            [gout],                                  # 1: Output (after Input is valid)
            [gch, gpr, gemg, gecg, gns, gsave],  # 2: the rest (after Output is valid)
        ]
        # Cards whose relevance depends on the analysis itself. Deliberately NOT in
        # _stage_cards: that loop forces every registered card visible outside "new" mode, so
        # a predicate there would be overwritten on the next keystroke — and
        # test_startup_flow's "open mode reveals everything" walk would fail on a default
        # AppState. Kept separate, ANDed in its own pass.
        self._cond_cards = [
            (gent, lambda: bool(self.state.settings.input.channels.entropy)),
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
        from respmech.ui import prefs  # noqa: PLC0415
        start = line.text() or prefs.last_folder("browse", ".")   # P26 sticky folder
        if folder:
            d = QFileDialog.getExistingDirectory(self, "Select folder", start)
        else:
            d, _ = QFileDialog.getOpenFileName(self, "Select file", start)
        if d:
            prefs.set_last_folder("browse", d)
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
            self._refresh_channel_view()
            _si = self.seg_method.findData(s.processing.segmentation.method)
            self.seg_method.setCurrentIndex(_si if _si >= 0 else 0)
            _wi = self.wob_from.findData(s.processing.wob.calc_from)
            self.wob_from.setCurrentIndex(_wi if _wi >= 0 else 0)
            self.integrate.setChecked(s.processing.volume.integrate_from_flow)
            self.correct_drift.setChecked(s.processing.volume.correct_drift)
            self.correct_trend.setChecked(s.processing.volume.correct_trend)
            _ti = self.trend_method.findData(s.processing.volume.trend_method)
            self.trend_method.setCurrentIndex(_ti if _ti >= 0 else 0)
            self.inverse_flow.setChecked(s.processing.volume.inverse_flow)
            self.inverse_volume.setChecked(s.processing.volume.inverse_volume)
            self.resample.setChecked(s.processing.sampling.resample)
            self.resample_hz.setValue(s.processing.sampling.resample_to_frequency or 200)
            d, dg = s.output.data, s.output.diagnostics
            self.save_average.setChecked(d.save_average)
            self.save_bbb.setChecked(d.save_breath_by_breath)
            self.save_processed.setChecked(d.save_processed)
            self.include_ignored.setChecked(d.include_ignored_breaths)
            self.save_pv_avg.setChecked(dg.save_pv_average)
            self.save_pv_ind.setChecked(dg.save_pv_individual)
            self.save_raw_fig.setChecked(dg.save_raw)
            self.save_trimmed_fig.setChecked(dg.save_trimmed)
            self.save_drift_fig.setChecked(dg.save_drift)
            self.save_emg_fig.setChecked(getattr(dg, "save_emg", True))
            self.group_regex.setText(s.output.group_regex or "")

            emg, n = s.processing.emg, s.processing.emg.noise
            # EMG RMS envelope
            self.emg_rms_window.setValue(emg.rms_window_s)
            self.emg_outlier_sd.setValue(emg.outlier_rms_sd_limit)
            _ni = self.emg_norm.findData(emg.normalization)
            self.emg_norm.setCurrentIndex(_ni if _ni >= 0 else 0)
            # Noise reduction (enable + reference file; tuning lives in the EMG tab). ECG
            # removal is tuned on the Preview "› EMG – ECG reduction" tab, not here.
            self.remove_noise.setChecked(n.enabled)               # rebind to the real gate
            self._refresh_noise_summary()
            self.out_folder.setText(s.output.folder)
            # Advanced (rarely changed)
            seg, peak = s.processing.segmentation, s.processing.segmentation.peak
            self.seg_buffer.setValue(seg.buffer)
            self.peak_height.setValue(peak.height)
            self.peak_distance.setValue(peak.distance_s)
            self.peak_width.setValue(peak.width_s)
            self.avg_resamp.setValue(s.processing.wob.avg_resampling_obs)
            self.ptp_baseline.setValue(s.processing.ptp.baseline_window_s)
            self.ent_epochs.setValue(s.processing.entropy.epochs)
            self.ent_tol.setValue(s.processing.entropy.tolerance)
            _mi = self.matlab_variant.findData(s.input.format.matlab_variant)
            self.matlab_variant.setCurrentIndex(_mi if _mi >= 0 else 0)
            self.breath_counts_edit.setPlainText(
                "\n".join(f"{e.file} = {e.count}" for e in s.processing.breath_counts))
            self._sync_widgets()
        finally:
            self._loading = prev
        self._update_qc()
        self._update_format_readout()       # P28 detected-format read-out

    def to_state(self):
        s = self.state.settings
        s.input.folder = self.in_folder.text()
        s.input.files = self.in_files.text()
        s.input.format.sampling_frequency = self.samp_freq.value()
        ch = s.input.channels
        # input.channels is written ONLY by _apply_channel_mapping, from the picker. It has
        # no widget here, so rewriting it from one would mean rewriting it from nothing —
        # and to_state runs on every tab change, which would erase the mapping on the first
        # switch away from Setup.
        s.processing.segmentation.method = self.seg_method.currentData()
        s.processing.wob.calc_from = self.wob_from.currentData()
        s.processing.volume.integrate_from_flow = self.integrate.isChecked()
        s.processing.volume.correct_drift = self.correct_drift.isChecked()
        s.processing.volume.correct_trend = self.correct_trend.isChecked()
        s.processing.volume.trend_method = self.trend_method.currentData()
        s.processing.volume.inverse_flow = self.inverse_flow.isChecked()
        s.processing.volume.inverse_volume = self.inverse_volume.isChecked()
        s.processing.sampling.resample = self.resample.isChecked()
        s.processing.sampling.resample_to_frequency = self.resample_hz.value()
        d, dg = s.output.data, s.output.diagnostics
        d.save_average = self.save_average.isChecked()
        d.save_breath_by_breath = self.save_bbb.isChecked()
        d.save_processed = self.save_processed.isChecked()
        d.include_ignored_breaths = self.include_ignored.isChecked()
        dg.save_pv_average = self.save_pv_avg.isChecked()
        dg.save_pv_individual = self.save_pv_ind.isChecked()
        dg.save_raw = self.save_raw_fig.isChecked()
        dg.save_trimmed = self.save_trimmed_fig.isChecked()
        dg.save_drift = self.save_drift_fig.isChecked()
        dg.save_emg = self.save_emg_fig.isChecked()
        s.output.group_regex = self.group_regex.text().strip() or None

        emg, n = s.processing.emg, s.processing.emg.noise
        # EMG RMS envelope
        emg.rms_window_s = self.emg_rms_window.value()
        emg.outlier_rms_sd_limit = self.emg_outlier_sd.value()
        emg.normalization = self.emg_norm.currentData()
        # ECG removal (remove_ecg / detect_channel / ecg_*) is written by the Preview
        # "› EMG – ECG reduction" tab, not here — leave those model fields untouched.
        # Noise reduction (enable + reference file; tuning owned by the EMG tab)
        n.enabled = self.remove_noise.isChecked()
        # reference_file / use_expiration / reference_intervals are written ONLY by the
        # noise picker (preview_screen._apply_noise_reference / _apply_noise_expiration).
        # to_state runs on every tab change, so rewriting them from here — where they have no
        # widget — would erase the choice on the first switch away from Setup.
        s.output.folder = self.out_folder.text()
        # Advanced (rarely changed)
        s.processing.segmentation.buffer = self.seg_buffer.value()
        s.processing.segmentation.peak.height = self.peak_height.value()
        s.processing.segmentation.peak.distance_s = self.peak_distance.value()
        s.processing.segmentation.peak.width_s = self.peak_width.value()
        s.processing.wob.avg_resampling_obs = self.avg_resamp.value()
        s.processing.ptp.baseline_window_s = self.ptp_baseline.value()
        s.processing.entropy.epochs = self.ent_epochs.value()
        s.processing.entropy.tolerance = self.ent_tol.value()
        # noise.n_fft is written by the Preview EMG tab's Advanced modal, alongside
        # win_length and hop_length. Keeping a second writer here silently reverted it
        # on the next tab change — measured.
        # processing.emg.robust_peak is written ONLY by the Preview EMG tab, which is where
        # its controls now live — beside the EMG they change, and one tab from the ECG removal
        # they depend on. to_state runs on every tab change, so rewriting it from here would
        # erase a gated-peak edit on the first switch to Setup.
        s.input.format.matlab_variant = self.matlab_variant.currentData()
        bcs = []
        for line in self.breath_counts_edit.toPlainText().splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            fpart, _, cpart = line.rpartition("=")      # split on the LAST '=' (filenames may contain '=')
            try:
                cnt = int(cpart.strip())
            except ValueError:
                continue
            if fpart.strip():
                bcs.append(BreathCountEntry(fpart.strip(), cnt))
        s.processing.breath_counts = bcs
        if self.on_settings_changed:
            self.on_settings_changed()
        return s

    def set_noise_reference(self, file, intervals, use_expiration):
        """Reflect a noise reference chosen in the picker. The picker has already written all
        three fields into state — it is their only writer — so this just re-renders the
        read-only summary and records that the analysis changed."""
        self._refresh_noise_summary()
        self._mark_dirty()   # a picked noise reference is a user edit that lands in the .toml
        where = "every expiration" if use_expiration else "a marked rest span"
        self._set_status(f"Noise reference set from {file}: {where}.")

    def _refresh_noise_summary(self):
        """What the picker chose, in words. Empty is a real state and says so, because the
        alternative — a blank line beside a ticked 'Reduce EMG background noise' — reads as
        configured when nothing will run."""
        n = self.state.settings.processing.emg.noise
        if not n.reference_file:
            self.noise_summary.setText("Not set — noise reduction will not run.")
            self.noise_summary.setProperty("status", "muted")
        elif n.use_expiration or not n.reference_intervals:
            self.noise_summary.setText(f"{n.reference_file} · every expiration")
            self.noise_summary.setProperty("status", None)
        else:
            spans = ", ".join(f"{a:.2f}–{b:.2f} s" for a, b in n.reference_intervals)
            self.noise_summary.setText(f"{n.reference_file} · marked span {spans}")
            self.noise_summary.setProperty("status", None)
        self.noise_summary.style().unpolish(self.noise_summary)
        self.noise_summary.style().polish(self.noise_summary)

    # -- helpers ------------------------------------------------------------
    def _sync_widgets(self):
        self._refresh_noise_summary()
        self._refresh_channel_view()   # 'Volume: derived from flow' follows the checkbox
        self.trend_method.setEnabled(self.correct_trend.isChecked())
        self.resample_hz.setEnabled(self.resample.isChecked())
        self._update_save_preview()

    def _update_save_preview(self):
        """The 'You will get' line under the output checklist — the deliverables the current
        ticks will actually write, so the output is explicit before Run."""
        if getattr(self, "save_preview", None) is None:
            return
        got = []
        if self.save_average.isChecked():
            got.append("average workbook")
            got.append("cohort summary (mean ± SD, CV%, by group)")   # always paired with the average
        if self.save_bbb.isChecked():
            got.append("breath-by-breath workbook")
            if self.emg_norm.currentData() != "none" and self.state.settings.input.channels.emg:
                got.append("normalised-EMG sheet")     # only when EMG channels are configured
        if self.save_processed.isChecked():
            got.append("processed CSV")
        figs = sum(cb.isChecked() for cb in (self.save_pv_avg, self.save_pv_ind,
                   self.save_raw_fig, self.save_trimmed_fig, self.save_drift_fig, self.save_emg_fig))
        if figs:
            got.append(f"{figs} diagnostic-figure set{'s' if figs != 1 else ''}")
        got.append("run report + settings snapshot")                  # P7, always written
        self.save_preview.setText(", ".join(got) if got else "run report + settings snapshot only.")

    # -- reactivity ---------------------------------------------------------
    def _wire_reactivity(self):
        self.in_folder.editingFinished.connect(self._on_inputs_changed)
        self.in_files.editingFinished.connect(self._on_inputs_changed)
        for le in (self.out_folder, self.group_regex):
            le.editingFinished.connect(self._on_field_changed)
        self.breath_counts_edit.textChanged.connect(self._on_field_changed)   # QPlainTextEdit: no editingFinished
        for sb in (self.samp_freq, self.resample_hz,
                   self.seg_buffer, self.avg_resamp, self.ent_epochs):
            sb.valueChanged.connect(self._on_field_changed)
        for dsb in (self.emg_rms_window, self.emg_outlier_sd, self.peak_height,
                    self.peak_distance, self.peak_width, self.ptp_baseline, self.ent_tol):
            dsb.valueChanged.connect(self._on_field_changed)
        for cb in (self.seg_method, self.wob_from, self.trend_method):
            cb.currentTextChanged.connect(self._on_field_changed)
        self.emg_norm.currentIndexChanged.connect(self._on_field_changed)
        self.matlab_variant.currentIndexChanged.connect(self._on_field_changed)
        for chk in (self.integrate, self.remove_noise,
                    self.correct_drift, self.correct_trend, self.inverse_flow, self.inverse_volume,
                    self.resample, self.save_average, self.save_bbb, self.save_processed,
                    self.include_ignored, self.save_pv_avg, self.save_pv_ind, self.save_raw_fig,
                    self.save_trimmed_fig, self.save_drift_fig, self.save_emg_fig):
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
        self._update_format_readout()
        self.inputs_changed.emit()
        self.settings_changed.emit()
        self._update_disclosure()   # last, so guided-flow status wins (see above)

    def _update_format_readout(self):
        """P28: peek the first matching recording and report what was detected —
        column count + delimiter — so a mis-picked folder/mask is obvious at once."""
        lab = getattr(self, "format_readout", None)
        if lab is None:
            return
        folder = self.in_folder.text().strip()
        mask = self.in_files.text().strip()
        status, text = "muted", ""
        if folder and os.path.isdir(folder):
            files = matching_files(folder, mask)
            if not files:
                status, text = "warn", f"No files match '{mask or '*.*'}' in this folder."
            else:
                first = files[0]
                line = ""
                try:
                    # Sniff the raw bytes so the column-count hint is right across encodings:
                    # UTF-16 (Excel "Unicode Text"), UTF-8(-BOM) and cp1252 all decode here
                    # rather than mojibake under the Windows locale default. Matches the
                    # tolerant reader core.load actually uses; latin-1 never raises.
                    with open(first, "rb") as fh:
                        head = fh.read(8192)
                    if head[:2] in (b"\xff\xfe", b"\xfe\xff"):
                        text0 = head.decode("utf-16", errors="replace")
                    else:
                        for enc in ("utf-8-sig", "cp1252", "latin-1"):
                            try:
                                text0 = head.decode(enc)
                                break
                            except UnicodeDecodeError:
                                continue
                    line = text0.splitlines()[0].strip() if text0 else ""
                except Exception:               # pragma: no cover - unreadable file
                    line = ""
                if "\t" in line:
                    delim, dname = "\t", "tab"
                elif "," in line:
                    delim, dname = ",", "comma"
                else:
                    delim, dname = None, "whitespace"
                ncol = len(line.split(delim)) if line else 0
                status = "info"
                text = (f"{len(files)} file(s) matched · first: {os.path.basename(first)} — "
                        f"{ncol} columns, {dname}-separated")
        lab.setText(text)
        lab.setProperty("status", status)
        lab.style().unpolish(lab); lab.style().polish(lab)

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
    def enter_new_mode(self, use_last_rig: bool = False):
        """Enter the guided 'new analysis' flow: collapse the reveal back to just the
        Input card and keep the downstream (Preview/Run) tabs locked until every setting
        validates. Cards then appear stage by stage as each is completed.

        ``use_last_rig`` (P25) pre-fills the channel mapping + sampling from the last
        analysis, so a returning user keeps their hardware layout and the channel-
        assignment modal doesn't force-open."""
        self._mode = "new"
        self._revealed = 1
        self._flow_ready = None          # force the next _set_flow_ready() to emit
        self._channel_modal_done = bool(use_last_rig)   # rig inherited → don't force the modal
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
            if use_last_rig:                          # P25: inherit the last channel rig
                from respmech.ui import prefs  # noqa: PLC0415
                rig = prefs.last_rig()
                if rig:
                    prefs.apply_rig(self.state.settings, rig)
                    self.from_state()                # reflect the rig in the widgets
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

    def open_sample_analysis(self):
        """P23: generate a small synthetic recording, wire a ready analysis around it,
        and open it in full mode — a no-setup door for first-time users. Returns True on
        success. The sample lives in a temp folder (throwaway)."""
        import tempfile  # noqa: PLC0415
        from respmech.core.sample import write_sample_recording, build_sample_settings  # noqa: PLC0415
        try:
            base = os.path.join(tempfile.gettempdir(), "respmech_sample")
            desc = write_sample_recording(os.path.join(base, "input"))
            # the sample carries an ECG artefact and EMG noise, so the ready analysis
            # switches on ECG removal + noise reduction to demonstrate the full pipeline
            s = build_sample_settings(desc, os.path.join(base, "output"))
            self.state.settings, self.state.settings_path = s, None
            self.from_state()
            self.enter_open_mode()
            self._mark_clean()
            self.inputs_changed.emit()
            self.settings_changed.emit()
            self._set_status("Sample analysis loaded — try Preview & QC, then Run.")
            return True
        except Exception:                       # noqa: BLE001
            self._report_error("Explore sample data", traceback.format_exc())
            return False

    def confirm_discard_changes(self, title="RespMech", question="Save them before closing?"):
        """Offer to save before an action that would drop unsaved edits. Returns True to
        proceed, False to abort the action. A clean analysis never asks — a warning that
        fires when nothing is at stake trains the user to click through it. ``question``
        names the pending action, so an open-flow never asks about "closing".

        Re-entrant calls return False (the NEW action is aborted, the first prompt stays):
        a second window-modal box over the first — e.g. Cmd+Q arriving while an open-flow
        prompt is up — is cocoa's "modalSession exited prematurely" recipe. Save is only
        offered while the settings CAN be saved; otherwise the choice is Discard/Cancel,
        matching the menu, and no _refuse_save warning can stack onto this prompt."""
        if not self._dirty:
            return True
        if getattr(self, "_discard_prompt_up", False):
            return False
        savable = self.can_save()
        buttons = (QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel) if savable \
            else (QMessageBox.Discard | QMessageBox.Cancel)
        self._discard_prompt_up = True
        try:
            ans = QMessageBox.question(
                self, title,
                f"This analysis has unsaved changes.\n\n{question}",
                buttons, QMessageBox.Save if savable else QMessageBox.Cancel)
        finally:
            self._discard_prompt_up = False
        if ans == QMessageBox.Cancel:
            return False
        if ans == QMessageBox.Discard:
            return True
        # The user just picked "Save" in THIS dialog — that is the overwrite consent, so
        # save_analysis's own confirm is skipped: it would be modal-after-modal in one
        # call stack, which cocoa's session cleanup can end out of order.
        return self.save_analysis(confirm_overwrite=False)   # a refused save aborts the action too

    def new_analysis(self):
        """Analysis > 'New analysis': discard the current settings for a fresh set and
        re-enter the guided flow. Guarded like every other action that would drop unsaved
        edits (open, recents, close) — the guard only asks when there are REAL edits, and
        offers Save rather than a bare discard."""
        if not self.confirm_discard_changes(
                "New analysis", question="Save them before starting a new analysis?"):
            return
        self.state.settings = Settings()
        self.state.settings_path = None
        self.from_state()
        self.enter_new_mode()       # emits inputs/settings_changed then sets guided status
        self._mark_clean()          # a fresh analysis has no unsaved edits yet

    def open_analysis_dialog(self):
        """Analysis > 'Open analysis…': pick a saved .toml analysis or a legacy .py setup
        (routed by extension). "Analysis" is the user-facing name — never "TOML". Guarded
        like every other way of opening over unsaved edits (recents, close): opening
        replaces state.settings and marks it clean, which silently destroys the edits."""
        if not self.confirm_discard_changes(
                "Open analysis", question="Save them before opening another analysis?"):
            return
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

    def _open_mech_advanced(self):
        """Breath detection, the averaging grid, the PTP baseline and the per-file breath
        counts. Real settings, but not ones anyone sets while getting a result out.

        The screen's own widgets still exist and still round-trip through from_state/to_state
        — they are simply not laid out any more. The dialog edits its own copies and commits
        into them, so Cancel costs nothing and the commit funnel below is the same one an
        ordinary field edit uses, which is what keeps the dirty flag and the recompute scope
        right without repeating either here."""
        from respmech.ui.advanced_dialog import AdvancedDialog, Field
        fields = [
            Field("seg_buffer", "Breath-separation buffer", "int",
                  "processing.segmentation.buffer",
                  "Guard samples added around each detected breath boundary.",
                  lo=0, hi=100_000, step=10),
            Field("peak_height", "Breath peak — minimum height", "float",
                  "processing.segmentation.peak.height",
                  "Minimum peak height for breath detection (volume-based separation).",
                  lo=0.0, hi=1_000_000.0, step=0.01, decimals=4),
            Field("peak_distance", "Breath peak — minimum distance", "float",
                  "processing.segmentation.peak.distance_s",
                  "Minimum time between detected breath peaks.",
                  lo=0.0, hi=60.0, step=0.05, decimals=4, suffix=" s"),
            Field("peak_width", "Breath peak — minimum width", "float",
                  "processing.segmentation.peak.width_s",
                  "Minimum width of a detected breath peak.",
                  lo=0.0, hi=60.0, step=0.05, decimals=4, suffix=" s"),
            Field("avg_resamp", "Average-breath resampling points", "int",
                  "processing.wob.avg_resampling_obs",
                  "Points each breath is resampled to when building the average breath / WOB.",
                  lo=10, hi=100_000, step=10),
            Field("ptp_baseline", "PTP baseline window", "float",
                  "processing.ptp.baseline_window_s",
                  "End-expiratory window whose mean is the pressure-time-product baseline.",
                  lo=0.0, hi=1.0, step=0.01, decimals=4, suffix=" s"),
            Field("breath_counts", "Breath-count overrides", "text",
                  "processing.breath_counts",
                  "Per-file override of the breath count used for per-minute rate scaling — "
                  "one 'filename = count' per line, e.g. 'RIU_H5_40W.txt = 12'. Blank means "
                  "each file's detected count.",
                  placeholder="one 'filename = count' per line"),
        ]
        current = {
            "seg_buffer": self.seg_buffer.value(),
            "peak_height": self.peak_height.value(),
            "peak_distance": self.peak_distance.value(),
            "peak_width": self.peak_width.value(),
            "avg_resamp": self.avg_resamp.value(),
            "ptp_baseline": self.ptp_baseline.value(),
            "breath_counts": self.breath_counts_edit.toPlainText(),
        }
        dlg = AdvancedDialog("Mechanics — advanced", fields, current, parent=self,
                             intro="Breath detection and the averaging grid. The defaults "
                                   "suit ordinary recordings; reach for these when a "
                                   "recording is segmenting badly.")
        if dlg.exec() != QDialog.Accepted:
            return
        staged = dlg.values()
        if staged == current:
            return                      # OK without an edit: no dirty flag, no recompute
        prev, self._loading = self._loading, True
        try:
            self.seg_buffer.setValue(int(staged["seg_buffer"]))
            self.peak_height.setValue(float(staged["peak_height"]))
            self.peak_distance.setValue(float(staged["peak_distance"]))
            self.peak_width.setValue(float(staged["peak_width"]))
            self.avg_resamp.setValue(int(staged["avg_resamp"]))
            self.ptp_baseline.setValue(float(staged["ptp_baseline"]))
            self.breath_counts_edit.setPlainText(staged["breath_counts"])
        finally:
            self._loading = prev
        self._on_field_changed()        # the ordinary commit: dirty + sync + recompute scope

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

    def _channel_view_signature(self):
        """What the summary actually depends on. Rebuilding a stack of pyqtgraph plots on
        every keystroke would be unusable, so the render is skipped unless one of these
        moved."""
        ch = self.state.settings.input.channels
        f = self.state.settings.input.format
        return (ch.flow, ch.volume, ch.poes, ch.pgas, ch.pdi, tuple(ch.emg), tuple(ch.entropy),
                self.integrate.isChecked(), f.sampling_frequency, f.decimal,
                self.in_folder.text(), self.in_files.text())

    def _refresh_channel_view(self, force=False):
        """Re-render the read-only channel summary. The traces need a readable data file; the
        rows do not, so a mapping with no loadable file still shows which column is what."""
        sig = self._channel_view_signature()
        if not force and sig == getattr(self, "_channel_view_sig", None):
            return
        self._channel_view_sig = sig
        s = self.state.settings
        matrix = names = None
        files = self._valid_input_files()
        if files:
            key = (files[0], s.input.format.sampling_frequency, s.input.format.decimal)
            if getattr(self, "_raw_cache_key", None) == key:
                matrix, names = self._raw_cache
            else:
                try:
                    from respmech.ui.workers import load_raw_matrix
                    matrix, names = load_raw_matrix(s, files[0])
                    self._raw_cache_key, self._raw_cache = key, (matrix, names)
                except Exception:      # noqa: BLE001 — the rows are still worth showing
                    matrix = names = None
        self.channel_summary.show_mapping(
            s.input.channels, matrix=matrix, names=names,
            fs=s.input.format.sampling_frequency or 1000,
            integrate_from_flow=s.processing.volume.integrate_from_flow)

    def _current_channel_mapping(self):
        ch = self.state.settings.input.channels
        return {"flow": ch.flow, "volume": ch.volume, "poes": ch.poes, "pgas": ch.pgas,
                "pdi": ch.pdi, "emg": list(ch.emg), "entropy": list(ch.entropy)}

    def _apply_channel_mapping(self, m, fmt_note=""):
        """Write a role->column mapping from the picker STRAIGHT INTO the model, then run the
        normal reactive commit (which advances the guided disclosure). ``fmt_note`` is an
        optional summary of any auto-detected file settings to mention in the status.

        The model is written here rather than through widgets because there are none: this is
        the only writer of input.channels, and to_state deliberately leaves it alone."""
        ch = self.state.settings.input.channels
        ch.flow = int(m["flow"]) if m.get("flow") else None
        ch.volume = int(m["volume"]) if m.get("volume") else None
        ch.poes = int(m["poes"]) if m.get("poes") else None
        ch.pgas = int(m["pgas"]) if m.get("pgas") else None
        ch.pdi = int(m["pdi"]) if m.get("pdi") else None
        ch.emg = [int(c) for c in m.get("emg", [])]
        ch.entropy = [int(c) for c in m.get("entropy", [])]
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
            self._set_status(self._validation_status())   # no Validate button: every edit re-checks
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
        # Conditional cards break this function's monotonicity promise ("a card never
        # retracts once shown"), so they get the one exemption that keeps that promise
        # honest where it matters: a card holding the widget the user is typing in is never
        # yanked out from under them.
        staged = (self._mode != "new") or (len(self._stage_cards) - 1 < self._revealed)
        for card, relevant in self._cond_cards:
            if card.isVisible() and card.isAncestorOf(QApplication.focusWidget()):
                continue
            card.setVisible(staged and relevant())

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
        """A HARD channel-mapping error (message, else ''): a required channel
        (flow/poes/pgas/pdi) not assigned at all, one pointing at column 1 — the time axis —
        or two of them sharing a column.

        Unassigned used to be impossible to express: the spin boxes had a minimum of 1, so a
        skipped mapping read as column 1 and that was the sentinel this checked. With the
        picker as the only writer the value is genuinely None, so both cases are named — and
        they are named separately, because "you have not chosen yet" and "you chose the time
        axis" call for different advice."""
        ch = self.state.settings.input.channels
        req = [("flow", ch.flow), ("poes", ch.poes), ("pgas", ch.pgas), ("pdi", ch.pdi)]
        unset = [n for n, c in req if c is None]
        if unset:
            return (f"{', '.join(unset)} not assigned — "
                    "click 'Assign channels from data…'")
        on_time = [n for n, c in req if c == 1]
        if on_time:
            return (f"{', '.join(on_time)} point at column 1 (the time axis) — "
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

    def _science_notes(self):
        """Every non-fatal science caution about the current settings, worst first.

        A list, not a single string: these are independent conditions, and a first-match-wins
        chain silently hides one behind another — a sub-1000 Hz recording would have masked
        the gated-peak prerequisite below, which is the one caution the user cannot diagnose
        from this screen."""
        s = self.state.settings
        ch = s.input.channels
        out = []
        clash = [n for n, c in (("flow", ch.flow), ("poes", ch.poes),
                                ("pgas", ch.pgas), ("pdi", ch.pdi)) if c and c in ch.emg]
        if clash:
            out.append(f"EMG columns overlap {', '.join(clash)}")
        # A missing prerequisite outranks advice about signal quality: this one makes a
        # requested output come back empty, rather than merely making it noisier. The gated
        # peak reuses the heartbeats the ECG stage detects, and with ECG removal off there are
        # none. Caution rather than block — remove_ecg lives on the Preview screen, so a hard
        # failure here would strand the user with no control to change.
        rp = getattr(s.processing.emg, "robust_peak", None)
        if rp is not None and rp.enabled and not s.processing.emg.remove_ecg:
            out.append("cardiac-gated peak EMG needs ECG removal on (Preview & QC ▸ "
                       "› EMG – ECG reduction), or its columns will be blank")
        if ch.emg and (s.input.format.sampling_frequency or 0) < 1000:
            out.append("sampling frequency < 1000 Hz is low for EMG")
        return out

    def _science_note(self):
        """The most important caution, for the one-line live verdict. '' when there is none."""
        notes = self._science_notes()
        return notes[0] if notes else ""

    def _all_cautions(self):
        """Every current caution (hard channel collision first, then soft science notes) —
        for the live QC strip."""
        out = []
        c = self._channel_collision()
        if c:
            out.append(c)
        out.extend(self._science_notes())   # every note, not just the first
        return out

    def refresh_qc(self):
        """Recompute the caution strip without touching any widget value.

        Public because a caution can now depend on a field this screen does not own: the
        gated-peak prerequisite reads processing.emg.remove_ecg, which the Preview ECG tab
        writes. Without this the strip goes stale in both directions — most awkwardly, the
        warning tells the user to go and fix remove_ecg on Preview and then stays pinned here
        after they have done exactly that."""
        self._update_qc()

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
        """Reconcile the form back into state, then report the validation verdict (used on
        open). from_state() CLAMPS values the widgets cannot represent (e.g. a missing
        sampling frequency shows as 2000, an unknown interpolation token shows as the first
        entry); to_state() persists those so what the batch worker runs matches what the form
        shows. Only called on open, so this is not on the per-keystroke path."""
        self.to_state()
        self._set_status(self._validation_status())

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
        from respmech.ui import prefs  # noqa: PLC0415
        p = path or QFileDialog.getOpenFileName(
            self, "Open analysis", prefs.last_folder("analysis", "."), TOML_FILTER)[0]
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
        prefs.add_recent_analysis(p)                 # P26 recent analyses
        prefs.set_last_folder("analysis", p)
        prefs.save_rig(self.state.settings)          # P25 remember the rig
        self.inputs_changed.emit()
        self.settings_changed.emit()
        return True

    def can_save(self):
        """Whether the current settings may be written to an analysis file."""
        return self._save_blocker() is None

    def _save_blocker(self):
        """A short, human reason the analysis may not be saved, or None.

        The SETTINGS must be internally valid, but the filesystem is deliberately NOT
        consulted: an analysis file is a portable document, and one whose input folder is
        momentarily unmounted (or belongs to a colleague's machine) is still a legitimate
        thing to save. Runnability is the stricter, path-aware _all_ok(), which already
        gates Preview/Run and is reported in the status bar.
        """
        collision = self._channel_collision()
        if collision:
            return collision
        try:
            self.state.settings.validate()
        except SettingsError as e:
            return str(e)
        except Exception:                       # noqa: BLE001 — an unexpected fault is not savable
            return short_error(traceback.format_exc())
        return None

    def _refuse_save(self, blocker):
        self._set_status(f"Cannot save — {blocker}")
        QMessageBox.warning(self, "Save analysis",
                            f"This analysis cannot be saved yet:\n\n{blocker}")
        return False

    def _write_analysis(self, p):
        """Write the settings to `p` and record it as the saved analysis. A failed write
        leaves the analysis dirty, so the edits are never silently lost."""
        from respmech.ui import prefs  # noqa: PLC0415
        try:
            self.state.save_toml(p)
        except Exception:                       # noqa: BLE001
            self._report_error("Save analysis", traceback.format_exc())
            return False
        self._set_status(f"Saved {p}")
        self._mark_clean()
        prefs.add_recent_analysis(p)                 # P26 recent analyses
        prefs.set_last_folder("analysis", p)
        prefs.save_rig(self.state.settings)          # P25 remember the rig
        return True

    def save_analysis(self, confirm_overwrite=True):
        """Analysis > 'Save': overwrite the analysis file that was opened. Confirms first —
        this replaces a file the user may share with collaborators, and the menu item sits
        one slot from 'Save as…'. The unsaved-changes guard passes confirm_overwrite=False:
        there the user has JUST answered "Save" in a modal, so a second "overwrite?" box
        would be redundant — and opening it in the same call stack the first modal returned
        from is the one place cocoa can end the two NSModalSessions out of order (the
        one-off "modalSession has been exited prematurely" stderr noise). With no file
        associated yet there is nothing to overwrite, so a new analysis falls through to
        Save as…. Returns True iff saved."""
        self.to_state()
        p = self.state.settings_path
        if not p:
            return self.save_analysis_as()
        blocker = self._save_blocker()
        if blocker:
            return self._refuse_save(blocker)
        if confirm_overwrite and QMessageBox.question(
                self, "Save analysis",
                f"Save the changes to\n\n{p}\n\noverwriting the file?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return False
        return self._write_analysis(p)

    def save_analysis_as(self):
        """Analysis > 'Save as…': write the current settings to a chosen analysis file. The
        chooser asks before overwriting, so unlike Save it needs no extra confirmation.
        Returns True iff saved."""
        from respmech.ui import prefs  # noqa: PLC0415
        self.to_state()
        blocker = self._save_blocker()
        if blocker:
            return self._refuse_save(blocker)
        start = self.state.settings_path or os.path.join(
            prefs.last_folder("analysis", "."), "analysis.toml")
        p, _ = QFileDialog.getSaveFileName(self, "Save analysis as", start, TOML_FILTER)
        if not p:
            return False
        return self._write_analysis(p)

    def _import(self, path=None):
        """Import a legacy .py setup (runs the migrator). Returns True on success, False
        if cancelled or rolled back, so open_analysis can gate the mode transition."""
        p = path or QFileDialog.getOpenFileName(self, "Open legacy analysis (.py)", ".", LEGACY_FILTER)[0]
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

    def _validation_status(self):
        """The current settings' validation verdict, as one status-bar sentence.

        Runs on every edit (there is no Validate button), so it must never raise and never
        pop a dialog — an unexpected fault degrades to a message like any other verdict.
        """
        try:
            self.state.settings.validate()
        except SettingsError as e:
            return f"Invalid: {e}"
        except Exception:                       # noqa: BLE001 — never let a fault break typing
            return f"Could not validate: {short_error(traceback.format_exc())}"
        # path checks (the core validate() is filesystem-agnostic)
        problem = self._path_problem()
        if problem:
            return f"Invalid: {problem}"
        # non-fatal science guardrails (shared with the guided-flow completion status)
        note = self._science_note()
        return f"Valid, but check: {note}." if note else "Settings valid ✓"

    def _path_problem(self):
        """Return a human message for the first invalid path, or None if all OK.
        Shared with the Run screen so both surface the same filesystem checks."""
        from respmech.ui.validation import path_problem
        return path_problem(self.state.settings)


