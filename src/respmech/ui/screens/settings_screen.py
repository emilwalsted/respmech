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
import tomllib
import traceback

from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
                               QScrollArea, QSpinBox, QVBoxLayout, QWidget)
from PySide6.QtCore import Signal, QTimer, Qt

from respmech.core.settings import BreathCountEntry, Settings, SettingsError
from respmech.ui.dialogs import open_error_dialog, short_error
from respmech.ui.migration_report_dialog import open_migration_report
from respmech.ui.flow_layout import FormLabel, install_flow
from respmech.ui.help_text import tooltip as _tip
from respmech.ui.manifest import build_manifest, group_readout, narrow_mask
from respmech.ui.path_drop import install_path_drop
from respmech.ui.section_flow import install_sections
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
        self._save_preview_warm = False   # see _update_save_preview
        self._manifest = None         # last-built ui.manifest.Manifest for the current folder/mask
        self._manifest_cache = {}     # (path, mtime, size)-keyed probe memo, reused across builds
        # D26: whether the sampling-frequency FIELD currently holds a value written by
        # _probe_and_apply_file_settings (drives the "detected from the time column" marker),
        # and, when that detection overwrote a DIFFERENT value the field already had, the
        # (previous, detected) pair the persistent read-out warns about. Cleared the moment
        # the user edits the field themselves (_on_sampling_frequency_changed) or a fresh
        # analysis loads (from_state). ``_samp_freq_detection_folder`` records which folder
        # the detection was actually made against; self-review found the two flags above
        # otherwise survive UNCHANGED across a folder/mask edit or "Duplicate for another
        # recordings folder…" (neither routes through from_state or a manual samp_freq
        # edit), so a probe result from folder A kept re-appearing as a warning against
        # folder B's completely different, unprobed data. _update_format_readout is the
        # one place that rebuilds for every folder-changing path (including duplicate's own
        # direct call), so it self-heals this by clearing all three the moment the folder it
        # reads no longer matches the one recorded here — see there.
        self._samp_freq_from_detection = False
        self._samp_freq_detection_note = None
        self._samp_freq_detection_folder = None
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
        # D26: a persistent, dimmed provenance marker for the field just above — set by
        # _probe_and_apply_file_settings whenever it writes the field from the time column,
        # cleared the moment the user edits the field by hand. setRowVisible keeps it from
        # leaving an empty gap the rest of the time (same idiom as format_readout below).
        self.samp_freq_detected_label = QLabel("Detected from the time column.")
        self.samp_freq_detected_label.setProperty("status", "muted")
        f.addRow("", self.samp_freq_detected_label)
        f.setRowVisible(self.samp_freq_detected_label, False)
        # P28: a live read-out of what was actually detected in the chosen recordings.
        self.format_readout = QLabel("")
        self.format_readout.setWordWrap(True)
        self.format_readout.setProperty("banner", True)   # box baked at first polish (theme.py)
        self.format_readout.setProperty("status", "muted")
        f.addRow("", self.format_readout)
        # ticket C03 point 6: its OWN row (not nested inside format_readout's), shown only
        # when the recordings folder is SET but no longer exists (renamed/moved/unmounted
        # since the analysis was saved) — see _update_format_readout's missing_folder
        # branch. Kept as a SEPARATE addRow rather than wrapping the two in one container
        # widget, so setRowVisible(self.format_readout, ...) keeps meaning exactly what it
        # already did everywhere else in this class.
        self.btn_locate_folder = QPushButton("Locate folder…")
        self.btn_locate_folder.setProperty("compact", True)
        self.btn_locate_folder.setVisible(False)
        self.btn_locate_folder.setToolTip(
            "The recordings folder saved with this analysis no longer exists at this "
            "location. Pick its new location.")
        self.btn_locate_folder.clicked.connect(self._locate_missing_folder)
        _locate_wrap = QWidget(); _lw = QHBoxLayout(_locate_wrap)
        _lw.setContentsMargins(0, 0, 0, 0)
        _lw.addWidget(self.btn_locate_folder); _lw.addStretch(1)
        f.addRow("", _locate_wrap)
        self._locate_folder_row = _locate_wrap
        self.matlab_variant = QComboBox()
        # 'wide', not 'compact': "MATLAB (Unix/Mac)" plus the drop-down arrow outgrows the
        # 150px column on wider fonts (measured 131px of text against the column's 126px
        # text box on the CI runners' fonts) — the cap must never clip its own content.
        self.matlab_variant.setProperty("formField", "wide")
        self.matlab_variant.addItem("MATLAB (Windows)", "windows")
        self.matlab_variant.addItem("MATLAB (Unix/Mac)", "mac")
        self._row(f, "MATLAB file variant", self.matlab_variant, "input.format.matlab_variant",
                  "Variant/byte-order for .mat input files (ignored for CSV/Excel/text).")
        # Ticket D03: the manual override for CSV/text decimal separator. _detect_decimal
        # seeds this from the data when the channel picker opens; this is here for the
        # cases the guess gets wrong (or is ambiguous and keeps whatever was already set).
        self.decimal_sep = QComboBox()
        self.decimal_sep.setProperty("formField", "wide")
        self.decimal_sep.addItem("Point (.)", ".")
        self.decimal_sep.addItem("Comma (,)", ",")
        self._row(f, "Decimal separator", self.decimal_sep, "input.format.decimal",
                  "Decimal separator used in your CSV/text files (e.g. 0.5 vs 0,5). European "
                  "exports are often comma-decimal with columns separated by a semicolon "
                  "instead of a comma. Auto-detected when you assign channels from data; "
                  "override here if the guess is wrong. Ignored for Excel/MATLAB input.")
        # Kept so _update_format_readout can hide/show the read-out ROW (not just clear its
        # text) when there is nothing to say yet — see there.
        self._input_form = f

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

        gout = QGroupBox("Output")
        fo = QFormLayout(gout)
        fo.setRowWrapPolicy(QFormLayout.WrapLongRows)   # long labels wrap the field below instead of clipping
        self.out_folder = QLineEdit()
        self._browse_row(fo, "Output folder", self.out_folder, "output.folder",
                         "Where results are saved; files are written to a 'data' subfolder inside it; defaults to 'output'.", folder=True)

        # Processing — breath mechanics ------------------------------------
        # Entropy ----------------------------------------------------------
        # Shown only when a column is actually assigned to entropy — see _cond_cards. Its two
        # parameters are meaningless otherwise, and burying them in "Advanced (rarely
        # changed)" hid them from the users who DO compute entropy.
        gent = QGroupBox("Sample entropy")
        fent = QFormLayout(gent)
        fent.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.ent_epochs = QSpinBox(); self.ent_epochs.setRange(1, 100)
        # D11 (UI-overhaul): labelled/explained as what the FIELD actually is, not what the
        # literature calls the neighbouring parameter it is one more than. core/entropy.py's
        # own docstring says "sample_length is equal to m + 1" and sets M = sample_length - 1.
        # Content review, 05-09-2026: the default itself changed 2 -> 3 (core/settings.py's
        # EntropySettings.epochs), because the old default of 2 reported m = 1, not the m = 2
        # that is the near-universal literature convention. The tooltip below now names the
        # new default and tells a user how to recover the old value.
        self._row(fent, "Template length (m + 1)", self.ent_epochs, "processing.entropy.epochs",
                  "Template length for sample entropy, one more than the embedding "
                  "dimension. The default 3 gives m = 2, which is conventional in the "
                  "literature; set 2 for the m = 1 RespMech used before this change.")
        self.ent_tol = QDoubleSpinBox(); self.ent_tol.setRange(0.0, 10.0)
        self.ent_tol.setDecimals(4); self.ent_tol.setSingleStep(0.05)
        # Same fix: core/compute.py multiplies this by each column's own np.std, so the
        # actual tolerance is value x SD, which neither the old label nor tooltip said.
        # Content review, 05-09-2026: names the published interval, not just a single
        # "common" value, so this agrees with README/the manual rather than being a
        # narrower, uncited claim next to them.
        self._row(fent, "Tolerance (r), × SD", self.ent_tol, "processing.entropy.tolerance",
                  "Matching tolerance r, as a multiple of the per-column standard "
                  "deviation. Published values are typically 0.1-0.25 × SD, with 0.2 × "
                  "SD the most common; RespMech defaults to 0.1 × SD.")
        _enthint = QLabel("Computed on the columns ticked as Entropy in the channel picker.")
        _enthint.setWordWrap(True); _enthint.setProperty("status", "muted")
        fent.addRow("", _enthint)
        # Live read-out in the app's own vocabulary (m, not "template length"), so a user can
        # write their methods section straight off this line without opening the source.
        self.ent_caption = QLabel(""); self.ent_caption.setWordWrap(True)
        self.ent_caption.setProperty("status", "muted")
        fent.addRow("", self.ent_caption)
        self.ent_epochs.valueChanged.connect(self._update_entropy_caption)
        self.ent_tol.valueChanged.connect(self._update_entropy_caption)

        # 'What to save' lives inside the Output card now (one place for everything the run
        # produces and where it goes), so these rows attach to the Output form (fo). The two
        # checkbox groups are each run through a FlowLayout (B05): ten checkboxes stacked in
        # one column ran to ~430 px tall in a card measuring ~1700 px wide — a third of a
        # screen's height for something that would fit two or three columns wide. Wrapping
        # keeps the grouping (Tables / Diagnostic figures stay their own visual blocks) while
        # letting each reflow with the card's width, same mechanism as the EMG control strips.
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
        _lt = QLabel("Tables"); _lt.setProperty("status", "muted"); _lt.setContentsMargins(0, 8, 0, 0); fo.addRow(_lt)
        _tables = self._FlowGroup(); self._tables_flow = install_flow(_tables, h=18, v=6)
        for cb, var, tip in (
                (self.save_average, "output.data.save_average", "The across-breath average workbook."),
                (self.save_bbb, "output.data.save_breath_by_breath", "A per-file breath-by-breath workbook."),
                (self.save_processed, "output.data.save_processed", "The processed signal as a CSV per file."),
                (self.include_ignored, "output.data.include_ignored_breaths",
                 "Include the breaths you've excluded in the per-file processed-signal CSV only. "
                 "Excluded breaths are always left out of the averages and workbooks; this does not change them.")):
            cb.setToolTip(_tip(var, tip))
            self._tables_flow.addWidget(cb)
        fo.addRow(_tables)
        _lf = QLabel("Diagnostic figures"); _lf.setProperty("status", "muted"); _lf.setContentsMargins(0, 8, 0, 0); fo.addRow(_lf)
        _diag = self._FlowGroup(); self._diagnostics_flow = install_flow(_diag, h=18, v=6)
        for cb, var, tip in (
                (self.save_pv_avg, "output.diagnostics.save_pv_average", "The averaged Campbell (pressure–volume) diagram."),
                (self.save_pv_ind, "output.diagnostics.save_pv_individual", "A Campbell diagram per individual breath."),
                (self.save_raw_fig, "output.diagnostics.save_raw", "Raw-signal diagnostic figures."),
                (self.save_trimmed_fig, "output.diagnostics.save_trimmed", "Trimmed-signal diagnostic figures."),
                (self.save_drift_fig, "output.diagnostics.save_drift", "Drift-correction diagnostic figures."),
                (self.save_emg_fig, "output.diagnostics.save_emg",
                 "Per-channel EMG overview figures (raw / ECG-removed / noise-reduced) with the flow "
                 "reference and R-peak capture markers.")):
            cb.setToolTip(_tip(var, tip))
            self._diagnostics_flow.addWidget(cb)
        fo.addRow(_diag)
        _lg = QLabel("Cohort summary"); _lg.setProperty("status", "muted"); _lg.setContentsMargins(0, 8, 0, 0); fo.addRow(_lg)
        self.group_regex = QLineEdit()
        # 'wide', not 'compact': the placeholder below measures 281px at the 13pt macOS
        # font — it fits the 320px 'wide' contents box but would elide in the spin column,
        # and a QLineEdit's sizeHint ignores its placeholder, so no sizeHint check catches
        # a clipped one.
        self.group_regex.setProperty("formField", "wide")
        self.group_regex.setPlaceholderText("leading filename token (e.g. P03_120W → P03)")
        self._row(fo, "Group files by", self.group_regex, "output.group_regex",
                  "How files are grouped (subject / condition) for the by-group summary. Leave blank to use "
                  "the leading filename token; or enter a regular expression whose first capture group is the key.")
        # D19: a live read-out of what the pattern above ACTUALLY does, computed from the
        # matched files' basenames alone (no data read, instant regardless of folder
        # size) — before this, the only way to see whether a group pattern worked was to
        # run the whole batch and open the written "By group" sheet in Excel, and a
        # PARTIALLY matching pattern silently pooled the non-matching files into "(all)"
        # with nothing on screen to distinguish that from a clean grouping. Same banner/
        # status-property convention as ``format_readout`` above; see
        # ``_update_group_readout``.
        self.group_readout = QLabel(""); self.group_readout.setWordWrap(True)
        self.group_readout.setProperty("banner", True)
        self.group_readout.setProperty("status", "muted")
        fo.addRow("", self.group_readout)
        self._output_form = fo   # so _update_group_readout can hide the row when empty
        self.save_preview = QLabel(""); self.save_preview.setWordWrap(True)
        self.save_preview.setProperty("banner", True)
        self.save_preview.setProperty("status", "info")
        fo.addRow("You will get", self.save_preview)

        # Two columns — the rig (Input + Channels) on the left, the leverance (Output +
        # Sample entropy) on the right — collapsing to one column under a threshold width
        # (ticket B05). Built with the house mechanic (section_flow.install_sections)
        # rather than a bespoke layout: it already solves "report the widest single card as
        # the minimum, zero minimum height, heightForWidth for the scroll area" for the
        # Advanced modals, and the same three problems apply here. No max-width cap is set —
        # the folder paths and the channel rows are exactly the content on this screen that
        # benefits from the width, unlike the Advanced modals' short captions.
        self._card_input, self._card_channels = gin, gch
        self._card_output, self._card_entropy = gout, gent
        rig = QWidget(); rig_col = QVBoxLayout(rig)
        rig_col.setContentsMargins(0, 0, 0, 0); rig_col.setSpacing(11)
        rig_col.addWidget(gin); rig_col.addWidget(gch)
        _rig_sp = rig.sizePolicy(); _rig_sp.setHeightForWidth(True); rig.setSizePolicy(_rig_sp)
        leverance = QWidget(); lev_col = QVBoxLayout(leverance)
        lev_col.setContentsMargins(0, 0, 0, 0); lev_col.setSpacing(11)
        lev_col.addWidget(gout); lev_col.addWidget(gent)
        _lev_sp = leverance.sizePolicy(); _lev_sp.setHeightForWidth(True); leverance.setSizePolicy(_lev_sp)
        self._rig, self._leverance = rig, leverance
        columns = QWidget()
        self._columns_layout = install_sections(columns, max_columns=2, hgap=16, vgap=11)
        self._columns_layout.addWidget(rig)
        self._columns_layout.addWidget(leverance)
        root.addWidget(columns)

        # Reading order now runs left-column-then-right (Input, Channels, Output, Sample
        # entropy) rather than one long vertical stack, so it can no longer be inferred from
        # widget-creation order alone (the old single-column layout's implicit tab order,
        # which a stale comment above claimed was "Input -> Output -> rest" while the code
        # actually built Input -> Channels -> Output — see ticket B05). Bridge the seams
        # between cards explicitly; each card's OWN internal order is still the natural one
        # Qt derives from construction order.
        self.setTabOrder(self.matlab_variant, self.btn_assign_channels)
        self.setTabOrder(self.btn_assign_channels, self.out_folder)
        self.setTabOrder(self.group_regex, self.ent_epochs)

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

        # ticket C03 point 8: a persisting notice while the built-in sample is loaded,
        # pinned ABOVE the scrolling form (mirroring how the QC strip is pinned below it)
        # so it is seen immediately regardless of scroll position — nothing on screen used
        # to say "sample" at all, and it writes into a temp folder the OS may clean up.
        self.sample_banner = QLabel(
            "Built-in sample recording — results go to a temporary folder. Pick your own "
            "recordings and output folder to start a real analysis.")
        self.sample_banner.setWordWrap(True)
        self.sample_banner.setProperty("banner", True)
        self.sample_banner.setProperty("status", "info")
        self.sample_banner.setVisible(False)
        outer.addWidget(self.sample_banner)
        outer.addWidget(scroll, 1)

        # Live QC strip, pinned below the (scrolling) form: every current caution at a
        # glance, so a first-timer sees them the moment they appear.
        self.qc = QLabel("")
        self.qc.setWordWrap(True)
        self.qc.setProperty("banner", True)   # the box comes from the QSS, not extra margins
        outer.addWidget(self.qc)

        # Carried-over exclusions/breath-counts/noise-reference banner: shown only when the
        # input folder just changed AND state named against a DIFFERENT (or unrecorded)
        # folder is still sitting in the analysis — see core.settings.carried_over_state.
        # Two explicit choices, no default: "Keep" just dismisses (the state was never
        # touched, so it still applies exactly as it did before — an inherited exclusion is
        # only ever hatched/named differently in Preview, never silently dropped); "Clear"
        # actually removes it. Pinned beside the QC strip, not inside the scrolling form, so
        # it is seen without hunting for it.
        self.carried_banner = QWidget()
        self.carried_banner.setVisible(False)
        cb = QHBoxLayout(self.carried_banner)
        cb.setContentsMargins(0, 0, 0, 0)
        self.carried_label = QLabel("")
        self.carried_label.setWordWrap(True)
        self.carried_label.setProperty("banner", True)
        self.carried_label.setProperty("status", "warn")
        cb.addWidget(self.carried_label, 1)
        self.btn_carried_keep = QPushButton("Keep")
        self.btn_carried_keep.setProperty("compact", True)
        self.btn_carried_keep.setToolTip(
            "Dismiss this notice. Nothing is changed — the carried-over state keeps "
            "applying exactly as before, shown hatched in Preview & QC.")
        self.btn_carried_keep.clicked.connect(self._dismiss_carried_banner)
        self.btn_carried_clear = QPushButton("Clear")
        self.btn_carried_clear.setProperty("compact", True)
        self.btn_carried_clear.setToolTip(
            "Remove the exclusions/breath-count overrides/rest reference that belong to "
            "the previous recordings folder.")
        self.btn_carried_clear.clicked.connect(self._clear_carried_banner)
        cb.addWidget(self.btn_carried_keep)
        cb.addWidget(self.btn_carried_clear)
        outer.addWidget(self.carried_banner)

        # B04: progressive disclosure is retired — every card (Input/Channels/Output) is
        # visible from the first frame in every mode. Only cards whose RELEVANCE depends on
        # the analysis itself still hide/show (_cond_cards, ANDed in its own pass by
        # _apply_card_visibility) — Sample entropy's two parameters are meaningless unless a
        # column is actually assigned to entropy, in either mode.
        self._cond_cards = [
            (gent, lambda: bool(self.state.settings.input.channels.entropy)),
        ]
        self._mode = "full"          # "full" = an opened/default analysis; "new" = guided
        self._flow_ready = True
        # the visual channel-assignment modal sits between Output and the rest of the
        # cards in the guided flow (see _update_disclosure); these track its one-shot
        # auto-open so it fires once per new-analysis session.
        self._channel_modal_done = False
        self._channel_modal_pending = False

    class _FlowGroup(QWidget):
        """A ``FlowLayout``-wrapped checkbox row, sized HONESTLY for a column-balancing
        caller (self-review finding, 06-08-2026).

        ``FlowLayout.sizeHint()`` is deliberately "everything on one line" (see its own
        docstring) — right for a strip placed directly in a plain ``QVBoxLayout``, wrong once
        it feeds ``SectionColumns``' comfort-width quantile: the Output card's two checkbox
        rows inflated the card's measured ``sizeHint`` to roughly 1260 px (the sum of every
        chip on one line), which pushed the two-column split's real threshold to roughly
        1550 px wide — past the app's own default window width, so the split this ticket
        exists for silently never happened, even though an offscreen test at a hand-picked
        1700 px width passed. ``sizeHint`` here reports the layout's own MINIMUM instead: the
        widest SINGLE chip, which is the width the row can already be safely squeezed to and
        is what a column-balancer should be comfort-measuring against. This changes nothing
        about the row's actual runtime wrapping (that is decided by ``heightForWidth`` at
        whatever width the row is actually given, unaffected by this override), only how wide
        a column this row is allowed to make its Output card ask for.
        """
        def sizeHint(self):                     # noqa: N802 - Qt API
            lay = self.layout()
            return lay.minimumSize() if lay is not None else super().sizeHint()

    def _row(self, form, label, widget, var, desc):
        """A labelled form row whose LABEL and FIELD both carry the same tooltip:
        the settings variable path (bold) + a one-line description.

        ``FormLabel``, not a plain ``QLabel`` — and not a bare ``ElidingLabel`` either:
        QFormLayout allocates the label COLUMN from the widest label's *sizeHint*
        whenever there is room, and ElidingLabel only lowers ``minimumSizeHint`` (its
        sizeHint is still the full text — the right contract for a FlowLayout, the wrong
        one here). Measured on the Windows runner: swapping QLabel→ElidingLabel changed
        nothing — the identical 371 px "Recordings folder" field on an 825 px card,
        below the half-card floor the browse row needs, in runs #192 and #215/#216
        (``test_form_fields_are_bounded_not_full_width``). FormLabel caps the sizeHint
        itself (metric-derived, never eliding any current label in practice), so the
        label column is bounded and the field keeps its half of the card; the full label
        text stays one hover away in the tooltip either way."""
        tip = _tip(var, desc)
        lab = FormLabel(label); lab.setToolTip(tip)
        widget.setToolTip(tip)
        form.addRow(lab, widget)

    def _browse_row(self, form, label, line, var, desc, folder):
        """A labelled 'line edit + Browse…' row: label ABOVE, field + Browse below,
        spanning the card's full width.

        A SPANNING row, not a (label, field) form row — decided on run #217's numbers.
        In the form's shared label column, the path field gets what the widest label
        leaves it, and on the Windows runner's substituted font the labels measured
        228-270 px sizeHint (≈14 px per character — far past every metric-derived cap
        tried: the hint cap at 24 average chars computed 276 there and never bit), which
        squeezed the field to 367 px on an 825 px card, below the half-card floor
        test_form_fields_are_bounded_not_full_width holds it to. A path is the one field
        on this form whose useful width IS the card's width (the test's own comment:
        'the browse-row paths legitimately stay full width — of their own CARD'), so the
        row now takes the card's width by construction: no label column arithmetic on any
        font can squeeze it again, and the label keeps its full wording instead of
        gambling on elision thresholds. The label, the field and its inner line edit all
        carry the variable path + description tooltip, same as every ``_row()``."""
        tip = _tip(var, desc)
        lab = FormLabel(label); lab.setToolTip(tip)
        line.setToolTip(tip)
        wrapper = self._with_browse(line, folder=folder)
        wrapper.setToolTip(tip)
        stack = QWidget()
        col = QVBoxLayout(stack)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)
        col.addWidget(lab)
        col.addWidget(wrapper)
        form.addRow(stack)

    def _spin(self, allow_zero=False):
        s = QSpinBox(); s.setRange(0 if allow_zero else 1, 9999); return s

    def _with_browse(self, line: QLineEdit, folder=False):
        # C04: dragging a single local path onto the field; folder= matches the Browse…
        # button below (getExistingDirectory vs. getOpenFileName) so the two stay consistent.
        install_path_drop(line, folder=folder)
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
        # B05: the output picker had been sharing the INPUT folder's sticky-folder key —
        # "browse" — so once a recordings folder was chosen, "Browse…" on Output opened
        # inside it. The two pickers are for different things and now remember separately.
        key = "browse_output" if line is self.out_folder else "browse"
        start = line.text() or prefs.last_folder(key, ".")   # P26 sticky folder
        if folder:
            d = QFileDialog.getExistingDirectory(self, "Select folder", start)
        else:
            d, _ = QFileDialog.getOpenFileName(self, "Select file", start)
        if d:
            prefs.set_last_folder(key, d)
            line.setText(d)
            if line is self.in_folder and not self.out_folder.text().strip():
                # B05: suggest a sibling folder rather than leaving Output blank (which is
                # what used to steer a first-time user into picking the SAME folder for
                # both, per the caution below). Safe by construction: both
                # validation.path_problem and writers.py only require the PARENT to exist
                # (writers.py creates the 'data' subfolder, with parents, at write time), and
                # a not-yet-existing sibling folder satisfies exactly that.
                sibling = os.path.join(os.path.dirname(os.path.abspath(d)), "respmech-output")
                self.out_folder.setText(sibling)
            if line is self.out_folder:
                # A real write probe here (ticket A06 point 6), not just at Run time — the
                # picker is exactly the moment a user is choosing where results go, so a
                # folder that turns out to be read-only should say so immediately rather
                # than surface deep inside write_batch after a whole batch has computed.
                from respmech.core.io.plan import probe_write_folder  # noqa: PLC0415
                probe = probe_write_folder(d)
                if not probe.ok:
                    self._set_status(f"Warning: {probe.message}")
            if self._loading:
                return
            (self._on_inputs_changed if line is self.in_folder else self._on_field_changed)()

    @staticmethod
    def _deepest_existing_ancestor(path):
        """The deepest directory in ``path``'s chain that still exists on disk — where the
        'Locate folder…' picker (ticket C03 point 6) should start, rather than a bare '.'
        (the process cwd, unrelated to the missing recordings) or the dead path itself
        (which most native file dialogs refuse to start in at all)."""
        p = os.path.abspath(path) if path else ""
        while p and not os.path.isdir(p):
            parent = os.path.dirname(p)
            if parent == p:                     # reached the filesystem root
                break
            p = parent
        return p if p and os.path.isdir(p) else os.path.expanduser("~")

    def _locate_missing_folder(self):
        """'Locate folder…' (ticket C03 point 6): point the analysis at the recordings
        folder's new location, reusing the ordinary input-folder-edit pipeline
        (_on_inputs_changed) so this behaves exactly like editing the field by hand."""
        folder = self.in_folder.text().strip()
        start = self._deepest_existing_ancestor(folder)
        d = QFileDialog.getExistingDirectory(self, "Locate recordings folder", start)
        if d:
            self.in_folder.setText(d)
            self._on_inputs_changed()

    # -- state <-> widgets --------------------------------------------------
    def from_state(self):
        prev, self._loading = self._loading, True
        try:
            s = self.state.settings
            # D26: a value loaded from a saved/new analysis was not detected in THIS
            # session, so any leftover marker/note from a previous analysis must not
            # survive onto it.
            self._samp_freq_from_detection = False
            self._samp_freq_detection_note = None
            self._samp_freq_detection_folder = None
            self._update_samp_freq_marker()
            self.in_folder.setText(s.input.folder)
            self.in_files.setText(s.input.files)
            self.samp_freq.setValue(s.input.format.sampling_frequency or 2000)
            self._refresh_channel_view()
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
            self.out_folder.setText(s.output.folder)
            self.ent_epochs.setValue(s.processing.entropy.epochs)
            self.ent_tol.setValue(s.processing.entropy.tolerance)
            _mi = self.matlab_variant.findData(s.input.format.matlab_variant)
            self.matlab_variant.setCurrentIndex(_mi if _mi >= 0 else 0)
            _di = self.decimal_sep.findData(s.input.format.decimal or ".")
            self.decimal_sep.setCurrentIndex(_di if _di >= 0 else 0)
            # Mechanics, EMG conditioning, noise reduction and the gated peak are all
            # PREVIEW-OWNED now (their controls live on the Preview sub-tabs and their
            # Advanced… modals, writing the model directly); Setup neither fills nor reads
            # their widgets — it has none.
            self._sync_widgets()
        finally:
            self._loading = prev
        # _update_format_readout rebuilds self._manifest AND refreshes the QC strip from it
        # (self-review finding, 05-08-2026) — a separate _update_qc() call here would read
        # the manifest from BEFORE this rebuild (stale, or None on the very first call).
        self._update_format_readout()       # P28 detected-format read-out
        # a freshly opened analysis can ALREADY carry state from a folder other than the
        # one it currently names — most commonly one written before this field existed
        # (folder is unrecorded, which counts as carried too, see is_carried_folder) — so
        # this has to be checked here, not only after a live folder edit.
        self._update_carried_banner()
        self._update_sample_banner()   # ticket C03 point 8 — every load/import/new path routes through here

    def to_state(self):
        s = self.state.settings
        s.input.folder = self.in_folder.text()
        s.input.files = self.in_files.text()
        s.input.format.sampling_frequency = self.samp_freq.value()
        # input.channels is written ONLY by _apply_channel_mapping (the picker). Everything
        # under Mechanics, EMG conditioning, noise reduction and the gated peak is
        # Preview-owned and model-direct, so to_state must NOT rewrite any of it — this runs
        # on every tab change, and a Setup write would revert the Preview edit on the first
        # switch away.
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
        s.output.folder = self.out_folder.text()
        s.processing.entropy.epochs = self.ent_epochs.value()
        s.processing.entropy.tolerance = self.ent_tol.value()
        s.input.format.matlab_variant = self.matlab_variant.currentData()
        s.input.format.decimal = self.decimal_sep.currentData()
        if self.on_settings_changed:
            self.on_settings_changed()
        return s

    def set_output_folder(self, folder: str):
        """Point the analysis's output folder at ``folder`` — the Run drawer's temporary-
        output-folder confirmation (ticket C03 point 8) calls this via
        ``RunScreen.output_folder_change_requested`` rather than writing
        ``self.state.settings.output.folder`` directly: this widget is what
        ``to_state()`` reads on the next Setup edit, and a model-only write behind its
        back would be silently reverted by that."""
        self.out_folder.setText(folder)
        self._on_field_changed()

    def set_noise_reference(self, file, intervals, use_expiration):
        """Record that the picker chose a noise reference. The picker (Preview) is the only
        writer of the four fields (Preview also stamps reference_folder — see
        preview/_emg_noise.py) and shows the read-out itself now; Setup just marks the
        analysis modified and, since picking a reference can resolve carried-over state the
        banner is showing, refreshes it (this signal is Preview's only path to Setup for a
        noise-reference edit — the generic pv.settings_edited that other Preview edits use
        for this same refresh is not emitted here)."""
        self._mark_dirty()   # a picked noise reference is a user edit that lands in the .toml
        self._update_carried_banner()
        where = "every expiration" if use_expiration else "a marked rest span"
        self._set_status(f"Rest reference set from {file}: {where}.")

    def sync_from_preview(self):
        """A Preview-owned edit (mechanics / EMG conditioning / noise) may change what the
        Setup channel summary and the 'You will get' line say — integrate_from_flow drives
        the 'Volume: derived from flow' row, normalisation drives the normalised-EMG-sheet
        deliverable — so refresh both when Preview signals an edit.

        No ``force``: every Preview-owned input this depends on already sits in
        ``_channel_view_signature`` (``integrate_from_flow`` explicitly, since D28), so the
        normal signature check already rebuilds when something relevant changed and skips
        the ~47 ms re-render (measured) on every other keystroke that doesn't touch it.
        Should a future Preview-owned field feed the summary, add it to the signature
        instead of reintroducing the bypass."""
        self._refresh_channel_view()
        self._update_save_preview()

    # -- helpers ------------------------------------------------------------
    def _sync_widgets(self):
        self._refresh_channel_view()   # 'Volume: derived from flow' follows the model
        self._update_save_preview()
        self._update_entropy_caption()   # a loaded analysis may set m/r without a valueChanged

    def _update_save_preview(self):
        """The 'You will get' line under the output checklist — the deliverables the current
        ticks will actually write, so the output is explicit before Run. Shares its wording
        and its diagnostic-figure count with the Run screen's pre-flight plan (ticket A06),
        so the two screens can never again tell a different story about the same settings —
        measured before: a single 'save_drift' tick was counted here as 1 deliverable while
        actually producing 3 distinct figures (volume correction, trend, volume endpoints)."""
        if getattr(self, "save_preview", None) is None:
            return
        if not self._save_preview_warm:
            # The real count needs core.plots (numpy) — cheap once imported, but a fresh
            # import here cost ~180ms measured on this exact change, and this is the FIRST
            # call, made synchronously from __init__ (from_state -> _sync_widgets), still
            # inside MainWindow's own construction and therefore on ui/app.py's synchronous
            # startup path — well before showMaximized() reveals the window (a self-review
            # finding: an earlier version of this comment claimed "once shown", which is
            # not accurate — ui/app.py pumps one app.processEvents() right after
            # MainWindow(...) returns, still behind the splash, and that is what actually
            # resolves this deferred call). ui/validation.py and ui/workers.py already keep
            # the heavier compute-core imports off this exact path for the same reason
            # (tests/unit/test_startup_imports.py); do the same here — show a light
            # placeholder now and let the real text resolve whenever the event loop next
            # turns, off the synchronous construction call stack. Every later call (a real
            # edit) runs synchronously.
            self.save_preview.setText("…")
            self._save_preview_warm = True
            QTimer.singleShot(0, self._update_save_preview)
            return
        from respmech.core.plots import diagnostic_figure_type_count
        got = []
        if self.save_average.isChecked():
            got.append("Average breath-data workbook")
            got.append("cohort summary (mean ± SD, CV%, by group)")   # always paired with the average
        if self.save_bbb.isChecked():
            got.append("Breath-by-breath workbook, one per recording")
            emg = self.state.settings.processing.emg
            if emg.normalization != "none" and self.state.settings.input.channels.emg:
                got.append("normalised-EMG sheet")     # only when EMG channels are configured
        if self.save_processed.isChecked():
            got.append("Processed-signal CSV, one per recording")
        figs = diagnostic_figure_type_count(self.state.settings)
        if figs:
            got.append(f"{figs} diagnostic-figure type{'s' if figs != 1 else ''} per recording")
        emg = self.state.settings.processing.emg
        if emg.save_sound:
            got.append("EMG audio export (WAV), one per channel per conditioning stage")
        if emg.robust_peak.enabled:
            got.append("cardiac-gated peak EMG columns")
        got.append("run report + analysis snapshot")                  # P7, always written
        self.save_preview.setText(", ".join(got) if got else "run report + analysis snapshot only.")

    def _update_entropy_caption(self, *_):
        """D11 (UI-overhaul): 'Template length (m + 1)' names the FIELD, but a user still
        needs the number the paper calls m — this line does that arithmetic for them, in the
        app's own vocabulary, so the methods section can be written straight off it."""
        if getattr(self, "ent_caption", None) is None:
            return
        m = self.ent_epochs.value() - 1
        r = self.ent_tol.value()
        self.ent_caption.setText(f"Computing SampEn with m = {m}, r = {r:g} × SD.")

    # -- reactivity ---------------------------------------------------------
    def _wire_reactivity(self):
        self.in_folder.editingFinished.connect(self._on_inputs_changed)
        self.in_files.editingFinished.connect(self._on_inputs_changed)
        for le in (self.out_folder, self.group_regex):
            le.editingFinished.connect(self._on_field_changed)
        for sb in (self.samp_freq, self.ent_epochs):
            sb.valueChanged.connect(self._on_field_changed)
        # B01 self-review fix (05-08-2026): the manifest's frequency-mismatch caution
        # freezes settings_fs at scan time (Manifest.settings_fs) and _on_field_changed
        # alone never rebuilds it, so correcting samp_freq to what the caution itself told
        # you to set still quoted the OLD value back at you until the next folder/mask
        # edit. Connected AFTER _on_field_changed (same signal, Qt fires in connection
        # order) so state is already synced when this reads it. Cache-backed — every file
        # here was already probed, so this costs nothing beyond the majority/outlier
        # recompute.
        self.samp_freq.valueChanged.connect(self._on_sampling_frequency_changed)
        self.ent_tol.valueChanged.connect(self._on_field_changed)
        self.matlab_variant.currentIndexChanged.connect(self._on_field_changed)
        self.decimal_sep.currentIndexChanged.connect(self._on_field_changed)
        for chk in (self.save_average, self.save_bbb, self.save_processed,
                    self.include_ignored, self.save_pv_avg, self.save_pv_ind, self.save_raw_fig,
                    self.save_trimmed_fig, self.save_drift_fig, self.save_emg_fig):
            chk.toggled.connect(self._on_field_changed)

    def _on_field_changed(self, *_):
        if self._loading:
            return
        self._sync_widgets()
        self.to_state()
        self._mark_dirty()
        # D19: covers group_regex specifically -- the matched file list is unchanged (no
        # need to rebuild self._manifest, unlike _on_inputs_changed), but the PATTERN
        # just did, and the read-out must reflect it the moment the field is committed.
        self._update_group_readout()
        self.settings_changed.emit()
        self._update_disclosure()   # last, so this screen's own validation status wins

    def _on_sampling_frequency_changed(self, *_):
        """Rebuild the manifest so ``Manifest.settings_fs`` (and thus the frequency-
        mismatch caution) reflects the value just set — see the wiring comment in
        ``_wire_reactivity``. Runs AFTER ``_on_field_changed`` has already synced
        ``to_state()``, so this reads the NEW value; the fields/folder/mask themselves are
        unchanged, so the current mask is exactly what was already scanned.

        D26: this only ever fires on a REAL edit — ``_probe_and_apply_file_settings``'s own
        ``setValue()`` runs with ``self._loading`` True, so the guard above already screens
        it out. A real edit here means the user has just acted on (or overridden) whatever
        the field showed, so the "detected from the time column" marker and the persistent
        mismatch note both go stale and are cleared before the read-out is rebuilt."""
        if self._loading:
            return
        self._samp_freq_from_detection = False
        self._samp_freq_detection_note = None
        self._samp_freq_detection_folder = None
        self._update_samp_freq_marker()
        self._update_format_readout()

    def _on_inputs_changed(self, *_):
        if self._loading:
            return
        prev_folder = self.state.settings.input.folder
        self.to_state()
        raw_mask = self.state.settings.input.files   # captured BEFORE narrowing, for the manifest
        self._normalize_mask()      # keep the mask a single pattern the core runner can glob
        self._mark_dirty()
        self._update_format_readout(raw_mask=raw_mask)
        # only a REAL folder change (not a files-mask-only edit, which fires this same
        # handler, and not a field re-committed unchanged) can make previously-fine state
        # start naming the wrong folder — see core.settings.carried_over_state.
        if self.state.settings.input.folder != prev_folder:
            self._update_carried_banner()
        self.inputs_changed.emit()
        self.settings_changed.emit()
        self._update_disclosure()   # last, so this screen's own validation status wins (see above)

    def _update_carried_banner(self):
        """Show/hide the carried-over exclusions/breath-counts/noise-reference notice —
        see core.settings.carried_over_state, and the banner built in _build()."""
        from respmech.core.settings import carried_over_state
        state = carried_over_state(self.state.settings)
        if not state:
            self.carried_banner.setVisible(False)
            return
        parts = []
        if state.exclude_files:
            parts.append(f"breath exclusions for {self._named_by_filename(state.exclude_files)}")
        if state.breath_count_files:
            parts.append("breath-count overrides for "
                         f"{self._named_by_filename(state.breath_count_files)}")
        if state.noise_reference:
            parts.append("the EMG rest reference")
        self.carried_label.setText(
            "This analysis still has " + "; ".join(parts) + " set against a DIFFERENT "
            "recordings folder than the one now loaded. Keep them if you want the same "
            "choices reapplied here; clear them if they belong to the other folder.")
        self.carried_banner.setVisible(True)

    @staticmethod
    def _named_by_filename(names, limit=3):
        """'a, b, c' for up to ``limit`` plain filenames, then '+N more' — same shape as
        _named_list, which takes manifest File objects rather than bare strings."""
        text = ", ".join(names[:limit])
        extra = len(names) - limit
        return text + (f", +{extra} more" if extra > 0 else "")

    def _update_sample_banner(self):
        """Show/hide the built-in-sample notice (ticket C03 point 8) — reads
        ``AppState.is_sample``, set True only by ``open_sample_analysis`` and cleared by
        every other load/import/new/save path (see their own docstrings)."""
        self.sample_banner.setVisible(bool(getattr(self.state, "is_sample", False)))

    def _dismiss_carried_banner(self):
        """"Keep": nothing is mutated — the carried-over state keeps applying exactly as it
        already did (an inherited exclusion is only ever drawn differently, in Preview &
        QC's overlay), this only stops the notice asking again this session."""
        self.carried_banner.setVisible(False)

    def _clear_carried_banner(self):
        from respmech.core.settings import clear_carried_over
        clear_carried_over(self.state.settings)
        self._mark_dirty()
        self.carried_banner.setVisible(False)
        # exclude_breaths/breath_counts/the noise reference all just changed — the same
        # signal a live edit of any of them would emit, so Preview & QC's rail badges and
        # overlay pick the clear up the same way they would any other settings edit.
        self.settings_changed.emit()

    def _delimiter_label(self, manifest):
        """A deterministic delimiter label from the format actually being parsed — never
        sniffed from raw bytes, which misreported a binary .xlsx as 'whitespace-separated'
        (an .xlsx is a zip archive of XML, not delimited text at all)."""
        ext = manifest.files[0].ext if manifest.files else ""
        if ext == ".txt":
            return "tab-separated"
        if ext == ".csv":
            dec = getattr(self.state.settings.input.format, "decimal", ".") or "."
            return "semicolon-separated" if dec == "," else "comma-separated"
        if ext == ".xlsx":
            return "Excel"
        if ext == ".mat":
            return "MATLAB"
        return "unknown format"

    @staticmethod
    def _narrowed_note(m):
        """The mask-narrowing phrase, shared by the read-out and the QC strip so the two
        can never drift apart (self-review fix, 05-08-2026: they used to duplicate this
        computation, and each dropped extension needs ITS OWN leading dot — joining
        stripped extensions with ', ' and prepending a single dot only decorates the
        FIRST one, e.g. '.txt, xlsx' silently loses xlsx's dot for a 3+-pattern mask)."""
        exts = ", ".join(f".{e.lstrip('.')}" for e in sorted(m.narrowed_out_exts))
        return (f"only one file pattern can be analysed at a time — narrowed to "
               f"'{m.mask}' ({m.narrowed_out_count} matching {exts} file"
               f"{'s' if m.narrowed_out_count != 1 else ''} excluded)")

    @staticmethod
    def _header_warning_note(m):
        """Ticket D01: the read-out's phrase for files whose sniffed head looks like an
        instrument export's preamble rather than channel data — a majority-consistent
        caveat ``outliers`` cannot see (see ``Manifest.header_warnings``). Deliberately
        names BOTH possible reasons rather than asserting "header block": the underlying
        probe (``workers.peek_header_warning``) also fires on a first line with under 3
        fields, which is not itself proof of a preamble — self-review finding, to avoid
        mis-diagnosing a merely too-narrow file as one with a header it does not have."""
        n = len(m.header_warnings)
        return (f"{n} file{'s' if n != 1 else ''} may not be real channel data (a header "
               f"block, or too few columns): {SettingsScreen._named_list(m.header_warnings)}")

    @staticmethod
    def _named_list(entries, limit=3):
        """'#.filename' for up to ``limit`` entries, then '+N more' — keeps a caution from
        growing unboundedly long over a batch of a hundred files."""
        names = ", ".join(e.filename for e in entries[:limit])
        extra = len(entries) - limit
        if extra > 0:
            names += f", +{extra} more"
        return names

    def _update_format_readout(self, raw_mask=None):
        """P28 / B01: build the batch MANIFEST (ui/manifest.py) for the current folder/mask
        and report what it found — not just the first file, as before, but the majority
        column layout and every file that disagrees with it — so a mis-picked folder/mask,
        or a single outlier file buried in a large batch, is obvious here rather than only
        surfacing as a run-time DataValidationError. ``raw_mask`` is the mask BEFORE
        ``_normalize_mask`` narrows it (the caller captures it first) — build_manifest
        narrows it again itself, so the two computations can never disagree, and the
        narrowing note survives being told AFTER the settings model already collapsed to a
        single pattern.

        Refreshes the QC strip itself at the end (self-review finding, 05-08-2026): this is
        the ONLY place ``self._manifest`` is rebuilt, and ``_qc_verdict``/``_update_qc``
        now read it — a caller that rebuilt the manifest without also refreshing QC could
        leave the strip showing a stale (or, on the very first call, entirely absent)
        verdict. That happened concretely on the open-analysis path: from_state() called
        _update_qc() BEFORE this method (so it read the OLD manifest), and
        enter_open_mode()'s own call to this method afterward never re-ran QC at all —
        together, opening a saved analysis whose folder had a genuine caveat still showed
        'No warnings' until the user made an edit or switched tabs. Centralising the refresh
        here means no future call site can reintroduce that gap by forgetting the follow-up
        call."""
        lab = getattr(self, "format_readout", None)
        if lab is None:
            return
        folder = self.in_folder.text().strip()
        # D26 self-review fix: this is the ONE place that rebuilds for every folder-changing
        # path (see the __init__ comment on _samp_freq_detection_folder) — a probe result
        # recorded against a DIFFERENT folder than the one now being read is stale and must
        # not be shown as if it were about this folder's (unprobed) data.
        if (self._samp_freq_detection_folder is not None
                and self._samp_freq_detection_folder != folder):
            self._samp_freq_from_detection = False
            self._samp_freq_detection_note = None
            self._samp_freq_detection_folder = None
            self._update_samp_freq_marker()
        mask = raw_mask if raw_mask is not None else self.state.settings.input.files
        status, text = "muted", ""
        self._manifest = None
        # ticket C03 point 6: a folder that is SET but no longer resolves (renamed, moved,
        # or an unmounted network share since the analysis was saved) used to leave this
        # read-out silently empty and the QC strip's own 'muted — nothing scanned yet'
        # message, which reads as "no folder chosen", not "this one is gone". Checked
        # before the isdir branch below, which only ever runs for a folder that DOES exist.
        # ``isabs`` matters: a brand-new (or guided-flow) analysis's DEFAULT placeholder
        # (input.folder == "input", a bare relative string nobody chose) also fails
        # isdir() in almost any cwd, and must keep reading as "nothing scanned yet" —
        # only an ABSOLUTE path (what a folder picker or a saved .toml always produces)
        # can be a real folder the user actually pointed at that then went missing.
        missing_folder = bool(folder) and os.path.isabs(folder) and not os.path.isdir(folder)
        if missing_folder:
            status, text = "warn", f"This folder no longer exists: {folder}"
        elif folder and os.path.isdir(folder):
            m = build_manifest(folder, mask, self.state.settings, cache=self._manifest_cache)
            self._manifest = m
            if not m.files:
                status, text = "warn", f"No files match '{m.mask or '*.*'}' in this folder."
            else:
                n = len(m.files)
                parts = [f"{n} file{'s' if n != 1 else ''} matched"]
                if m.majority_columns is None:
                    # unreadable AND "read fine but under 2 columns" (no signal beyond a
                    # time axis) both land here — neither can win a majority, see
                    # build_manifest's own >=2 floor — so the message covers both rather
                    # than naming only the first and misdescribing the second
                    parts.append("none of these files have enough columns to analyse "
                                 "(unreadable, or too few columns)")
                elif m.outliers:
                    delim = self._delimiter_label(m)
                    n_maj = len(m.included_files)
                    parts.append(f"{n_maj} with {m.majority_columns} columns, {delim}")
                    n_out = len(m.outliers)
                    parts.append(f"{n_out} file{'s' if n_out != 1 else ''} "
                                 f"{'differs' if n_out == 1 else 'differ'}: "
                                 f"{self._named_list(m.outliers)}")
                else:
                    delim = self._delimiter_label(m)
                    parts.append(f"{m.majority_columns} columns, {delim}")
                if m.header_warnings:
                    parts.append(self._header_warning_note(m))
                if m.mask_narrowed_from:
                    parts.append(self._narrowed_note(m))
                # D26: the field just got overwritten by _probe_and_apply_file_settings
                # with a DIFFERENT value than it held before — by the time this method
                # runs, to_state() has already synced the new value into settings, so
                # Manifest.settings_fs matches the just-detected rate and freq_mismatches
                # (below) can no longer see the change that just happened, only future
                # disagreements. This note is the one place that change itself is told,
                # and it stays until the user edits the field (_samp_freq_detection_note
                # is cleared there, not here) — see _probe_and_apply_file_settings.
                if self._samp_freq_detection_note is not None:
                    prev_fs, new_fs = self._samp_freq_detection_note
                    parts.append(f"time column implies {new_fs} Hz "
                                 f"(the field said {prev_fs} Hz)")
                # B01 self-review fix (05-08-2026): status now follows Manifest.is_clean
                # exactly, so it can never disagree with the QC strip's own gate on the
                # same manifest — the original condition listed outliers/narrowing/no-
                # majority explicitly but forgot freq_mismatches, so a frequency-only
                # mismatch (same column count, wrong rate) showed this label as "info"
                # while the QC strip correctly said "warn" for the identical scan.
                status = "info" if m.is_clean else "warn"
                if self._samp_freq_detection_note is not None:
                    status = "warn"
                text = " · ".join(parts)
        lab.setText(text)
        lab.setProperty("status", status)
        lab.style().unpolish(lab); lab.style().polish(lab)
        # B05: a fresh Setup (no folder chosen yet) used to leave the label empty but its
        # FORM ROW still there — a 28 px gap between "Sampling frequency" and "MATLAB file
        # variant" with nothing in it. setRowVisible hides label and field together, so an
        # empty read-out takes no space at all rather than an empty banner-shaped hole.
        form = getattr(self, "_input_form", None)
        if form is not None:
            form.setRowVisible(lab, bool(text))
        btn = getattr(self, "btn_locate_folder", None)
        locate_row = getattr(self, "_locate_folder_row", None)
        if btn is not None:
            btn.setVisible(missing_folder)
        if form is not None and locate_row is not None:
            form.setRowVisible(locate_row, missing_folder)
        if missing_folder:
            # ticket C03 point 6, the "warm" case: a folder that vanishes MID-session must
            # not go on showing the previous load's channel traces beside a dead path.
            # force=True bypasses _channel_view_signature's cache, which would otherwise
            # see an unchanged folder STRING (only the filesystem changed) and skip the
            # rebuild entirely.
            self._refresh_channel_view(force=True)
        self._update_qc()   # self._manifest just (re)built — the QC strip must never lag it
        self._update_group_readout()   # D19: same reason — the matched file list may have changed

    def _update_group_readout(self):
        """D19: refresh the live "Group files by" read-out from the currently matched
        batch. Called both here (piggy-backing on ``_update_format_readout``, which just
        rebuilt ``self._manifest`` — the matched file list may have changed) and from
        ``_on_field_changed`` (the pattern field itself was edited; no manifest rebuild
        needed, the file list is unchanged).

        Uses ``self._manifest.files`` -- EVERY matched file, not ``included_files`` (the
        UI's own majority-column-count subset). Self-review fix (10-08-2026): the first
        version of this method used ``included_files``, reasoning that a column-count
        outlier "is already excluded from the batch before grouping ever runs" -- that is
        false. ``core.pipeline.run_batch`` iterates the FULL matched file list with no
        column-count pre-filter at all (``included_files``/``outliers`` is a manifest
        concept the UI computes for its OWN cautions; the core run path has never heard
        of it). Verified end to end: a batch with one column-count outlier showed a clean
        "info" read-out on Setup while the real run pooled that same file into "(all)" in
        the written "By group" sheet -- a false all-clear in exactly the scenario this
        feature exists to catch. Using every matched file instead means this can show a
        FALSE WARNING for a file that would separately fail to load entirely (and so
        never reach the written summary either way) -- an accepted, conservative
        trade-off: overwarning about a file that will fail anyway is far cheaper than
        silently mis-reporting one that will actually be grouped as "(all)"."""
        m = getattr(self, "_manifest", None)
        filenames = [f.filename for f in m.files] if m is not None else []
        status, text = group_readout(filenames, self.state.settings)
        lab = getattr(self, "group_readout", None)
        if lab is None:
            return
        lab.setText(text)
        lab.setProperty("status", status)
        lab.style().unpolish(lab); lab.style().polish(lab)
        # Same convention as format_readout above: hide the row entirely rather than
        # leave an empty banner-shaped gap when there is nothing to say yet.
        form = getattr(self, "_output_form", None)
        if form is not None:
            form.setRowVisible(lab, bool(text))

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

    # -- guided ('new analysis') flow ----------------------------------------
    def enter_new_mode(self, use_last_rig: bool = False):
        """Enter the guided 'new analysis' flow: blank the input/output folders and mask,
        and re-arm the one-shot channel-assignment modal (B04: cards no longer hide/reveal
        stage by stage — every card is already visible — but a fresh analysis still gets
        one auto-offered trip through the channel picker once its input folder has
        matching files, see _update_disclosure).

        ``use_last_rig`` (P25) pre-fills the channel mapping + sampling from the last
        analysis, so a returning user keeps their hardware layout and the channel-
        assignment modal doesn't force-open."""
        self._mode = "new"
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
        # let the downstream screens react to the blanked settings FIRST, so the validation
        # status set by _update_disclosure below lands last and is not clobbered by their
        # own technical messages on the shared status bar
        self.inputs_changed.emit()
        self.settings_changed.emit()
        self._update_disclosure()   # sets this screen's own validation status (last word)

    def enter_open_mode(self):
        """Enter 'open' mode: normalise the mask and validate, so any problem in the
        opened analysis is surfaced right away. Every card is already visible in every
        mode since B04 — the only thing this stops is treating the session as a guided
        new-analysis flow, so the channel modal never force-opens over an
        already-configured analysis."""
        self._mode = "full"
        raw_mask = self.state.settings.input.files   # captured BEFORE narrowing, for the manifest
        self._normalize_mask()      # a saved analysis may carry a multi-pattern mask
        self._update_format_readout(raw_mask=raw_mask)   # reflect the (possibly just-narrowed) mask
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
        success. The sample lives in a temp folder (throwaway).

        The first call in a process is a synchronous ~1 s stall (measured: mostly the
        lazy import of the compute/reader stack, not the small CSV write itself, and a
        one-off cost — later calls are two orders of magnitude faster) with nothing
        threaded, so a wait cursor plus a status line is enough to say something is
        happening instead of the window appearing to freeze."""
        import tempfile  # noqa: PLC0415
        from respmech.core.sample import write_sample_recording, build_sample_settings  # noqa: PLC0415
        self._set_status("Building the sample recording…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # processEvents() lives INSIDE the try (not between it and setOverrideCursor
            # above): pumping the event loop can in principle surface an exception from an
            # unrelated queued callback, and that must still hit the finally below —
            # nothing between a successful setOverrideCursor and the finally is unguarded.
            QApplication.processEvents()   # paint the status text and cursor before the stall
            base = os.path.join(tempfile.gettempdir(), "respmech_sample")
            desc = write_sample_recording(os.path.join(base, "input"))
            # the sample carries an ECG artefact and EMG noise, so the ready analysis
            # switches on ECG removal + noise reduction to demonstrate the full pipeline
            s = build_sample_settings(desc, os.path.join(base, "output"))
            self.state.settings, self.state.settings_path = s, None
            self.state.display_name = None
            self.state.legacy_source_path = None
            # ticket C03 point 8: say so — before this nothing on screen distinguished the
            # sample from a real analysis, and it writes into a temp folder the OS may
            # clean up. _update_sample_banner (called from from_state below) reads this.
            self.state.is_sample = True
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
        finally:
            # the except branch above opens a non-modal error dialog (_report_error) — a
            # wait cursor still hanging over the window behind it would be worse than the
            # freeze this exists to fix.
            QApplication.restoreOverrideCursor()

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
        self.state.display_name = None
        self.state.legacy_source_path = None
        self.state.is_sample = False
        self.from_state()
        self.enter_new_mode()       # emits inputs/settings_changed then sets guided status
        self._mark_clean()          # a fresh analysis has no unsaved edits yet

    def _analysis_dialog_start(self):
        """Where an Open-analysis-style file dialog should start (ticket C03 point 4):
        the CURRENT analysis's own folder if one is open, so importing/opening the next
        file for the same study starts right there — else the sticky 'analysis' folder
        from the last one opened, mirroring ``_browse``'s existing
        ``line.text() or prefs.last_folder`` precedence. ``open_analysis_dialog`` and
        ``_import`` used to start at a bare '.', the PROCESS's cwd — in a packaged .app or
        .msi that is nowhere near the user's data, so the action a returning user repeats
        most (open the next subject's file from the same study folder) opened somewhere
        unrelated every time. ``_load`` and ``save_analysis_as`` already got this right."""
        from respmech.ui import prefs  # noqa: PLC0415
        p = getattr(self.state, "settings_path", None)
        if p:
            return os.path.dirname(p)
        return prefs.last_folder("analysis", ".")

    def open_analysis_dialog(self):
        """Analysis > 'Open analysis…': pick a saved .toml analysis or a legacy .py setup
        (routed by extension). "Analysis" is the user-facing name — never "TOML". Guarded
        like every other way of opening over unsaved edits (recents, close): opening
        replaces state.settings and marks it clean, which silently destroys the edits."""
        if not self.confirm_discard_changes(
                "Open analysis", question="Save them before opening another analysis?"):
            return
        p, _ = QFileDialog.getOpenFileName(self, "Open analysis", self._analysis_dialog_start(), OPEN_FILTER)
        if p:
            self.open_analysis(p)

    # -- visual channel assignment ------------------------------------------
    def _open_channel_setup_for_flow(self):
        """Deferred one-shot auto-open of the channel modal in the guided flow, once the
        input folder has matching files (B04 — no longer gated on the Output card, since
        there is no longer a disclosure stage to gate). Marks the step done first (so it
        never re-opens), whether the user assigns channels or cancels.

        Only re-runs ``_update_disclosure`` itself on a CANCEL. On accept,
        ``_open_channel_setup`` -> ``_apply_channel_mapping`` already ran it once (via
        ``_on_inputs_changed``) before setting its own, more specific "Channels
        assigned…" status — since B04 made ``_update_disclosure`` end by writing the
        live validation status (there is no longer a separate guidance label for it to
        land on instead), an unconditional second call here would immediately overwrite
        that message with the generic "Setup valid ✓" the instant the modal closed.
        Cancelling never reaches ``_apply_channel_mapping``, so nothing else refreshes
        the QC strip/flow_ready/status for the still-unassigned channels without this."""
        self._channel_modal_pending = False
        if self._channel_modal_done:
            return
        self._channel_modal_done = True
        applied = self._open_channel_setup(initial={})     # fresh analysis -> no pre-selection
        if not applied:
            self._update_disclosure()

    def _open_channel_setup(self, initial=None):
        """Show the visual channel-assignment modal over the valid data files matching the
        input mask, and on OK write the channel columns into the form. Returns True iff a
        mapping was applied. ``initial`` pre-selects the dropdowns (None -> current settings)."""
        files = self._valid_input_files()
        if not files:
            self._set_status("No valid data files found to assign channels — set them in the Channels card.")
            return False
        # Ticket D03: detect the decimal separator BEFORE the picker reads any data, not
        # only after OK as before — a semicolon-delimited, comma-decimal CSV read every
        # column as NaN under the wrong guess, which meant OK could never be reached and
        # the detection that used to live there was unreachable for exactly the file it
        # was meant to fix. Re-probe the valid-file set afterwards: a decimal change can
        # change which files even parse as data (see _valid_input_files/probe_data_columns).
        # ``decimal_detected`` is remembered so a successful OK can still tell the user what
        # was auto-detected (self-review finding: silently changing the setting with no
        # confirmation anywhere was a real loss of transparency the old .txt-only message
        # used to provide).
        decimal_detected = self._detect_decimal(files)
        if decimal_detected:
            files = self._valid_input_files()
            if not files:
                dec = self.state.settings.input.format.decimal
                self._set_status(
                    "No valid data files found to assign channels — the decimal separator "
                    f"was just auto-detected as '{dec}'; override it on the Input card if "
                    "that guess is wrong.")
                return False
        from respmech.ui.channel_setup_dialog import ChannelSetupDialog, NoReadableFileError
        from respmech.ui.workers import load_raw_matrix
        s = self.state.settings
        fs = s.input.format.sampling_frequency or 1000
        if initial is None:
            initial = self._current_channel_mapping()
        # B01: name the files the manifest excluded (a different column count) so the
        # dialog's "this mapping applies to all N files" banner can reconcile itself with
        # the batch's true size, instead of only ever describing the majority subset.
        excluded = ([(f.filename, f.columns) for f in self._manifest.outliers]
                   if self._manifest is not None else [])
        try:
            dlg = ChannelSetupDialog(files, fs, initial, loader=lambda p: load_raw_matrix(s, p),
                                     parent=self, excluded=excluded,
                                     integrate_from_flow=s.processing.volume.integrate_from_flow)
        except NoReadableFileError as exc:
            # ticket D01: the dialog already diagnosed WHY none of its files could be read
            # (see _no_files_readable_message) — show that diagnosis as the message, not a
            # traceback, with the copyable trace one click away behind Details instead of
            # dumped straight on screen. A dedicated exception type, not a bare ValueError:
            # self-review found that catching ValueError broadly would also swallow an
            # UNRELATED ValueError from elsewhere in the dialog's constructor (e.g. a numpy
            # reshape error) and misrepresent its raw message as if it were this diagnosis.
            self._set_status(f"Channel setup failed — {exc}")
            self._err_dialog = open_error_dialog(
                self, "Channel setup — error", traceback.format_exc(), intro=str(exc),
                prior=self._err_dialog, collapsed_detail=True)
            return False
        except Exception:                        # noqa: BLE001 — copyable error, don't trap the flow
            self._report_error("Channel setup", traceback.format_exc())
            return False
        if dlg.exec() != QDialog.Accepted:
            return False
        # Written BEFORE _apply_channel_mapping (ticket D02) so the reactive re-validation
        # that mapping triggers already sees the up-to-date value — otherwise a flow-only
        # rig's OK would apply the channel columns first and transiently re-validate
        # against the OLD (unset) integrate_from_flow, showing a blocker for the one
        # instant it takes to reach the next line.
        s.processing.volume.integrate_from_flow = dlg.integrate_from_flow()
        fmt_note = self._probe_and_apply_file_settings(files)
        if decimal_detected:                    # surface the auto-detected decimal too (D03)
            dec_note = f"decimal '{s.input.format.decimal}'"
            fmt_note = f"{dec_note}, {fmt_note}" if fmt_note else dec_note
        self._apply_channel_mapping(dlg.selected_mapping(), fmt_note=fmt_note)
        return True

    def _normalize_mask(self):
        """Keep ``input.files`` a SINGLE glob pattern, because the core batch runner globs one
        pattern: a multi-pattern mask (the '*.csv; *.txt' default, or any ';'/','-separated
        mask) is narrowed to the dominant extension of the files it currently matches. A
        specific single mask the user chose is left untouched; a no-op when nothing matches
        yet (or no folder). Never raises.

        Delegates the actual narrowing decision to ``ui.manifest.narrow_mask`` (B01) — the
        same function the format read-out's Manifest uses to decide (and report) what it
        narrowed FROM, so this mutation and that report can never compute a different
        answer for the same folder/mask."""
        s = self.state.settings
        narrowed, narrowed_from, _dropped = narrow_mask(s.input.folder, s.input.files)
        if narrowed_from is None:
            return
        prev, self._loading = self._loading, True
        try:
            s.input.files = narrowed
            self.in_files.setText(narrowed)
        finally:
            self._loading = prev

    def _detect_decimal(self, files):
        """Detect the decimal separator from the reference file — called from
        _open_channel_setup BEFORE the picker builds/reads anything (ticket D03). The
        previous call site (inside _probe_and_apply_file_settings, on channel-setup OK)
        was unreachable for exactly the file it needed to fix: a semicolon-delimited,
        comma-decimal CSV parses as one mostly-NaN column under the wrong guess, so the
        picker's OK button never became available and the flow could never reach the
        detection. Only .csv/.txt carry a meaningful decimal setting (see
        core.io.loaders — .xlsx/.mat are parsed natively, without one).

        Keeps detect_decimal's own fallback contract: passing the CURRENT setting means a
        doubtful file leaves it untouched, so moving the call earlier cannot make a guess
        overwrite an explicit choice (this picker or a loaded analysis) that the data
        itself does not clearly contradict. Returns True iff the setting changed, so the
        caller knows whether the just-computed valid-file set needs recomputing (a decimal
        change can change which files parse as data at all)."""
        import os as _os
        if not files or _os.path.splitext(files[0])[1].lower() not in (".csv", ".txt"):
            return False
        from respmech.ui.workers import detect_decimal
        current = self.state.settings.input.format.decimal
        dec = detect_decimal(files[0], current)
        if dec == current:
            return False
        self.state.settings.input.format.decimal = dec
        prev, self._loading = self._loading, True
        try:
            idx = self.decimal_sep.findData(dec)
            if idx >= 0:
                self.decimal_sep.setCurrentIndex(idx)
            self._sync_widgets()
        finally:
            self._loading = prev
        self._update_format_readout()      # the read-out's delimiter label depends on this too
        return True

    def _probe_and_apply_file_settings(self, files):
        """On channel-setup OK, read the actual data to fill the sampling frequency (from
        the time column) — the user would otherwise set it by hand. The decimal separator
        is detected earlier now, in _open_channel_setup via _detect_decimal, before the
        picker reads anything (ticket D03) — see that method's docstring for why it moved.
        The file mask is kept single-pattern separately by _normalize_mask. Returns a short
        human summary of what was set.

        D26: this used to be the ONLY trace of the detection — a status-line word that a
        busy Preview refresh (~460 ms later, measured) routinely overwrote before anyone
        read it, leaving no record on screen of where the field's value came from, or that
        it had just changed under the user's hand. Records that provenance on ``self`` so
        the persistent read-out (_update_format_readout) and the "detected from the time
        column" marker beside the field (_update_samp_freq_marker) can both show it."""
        from respmech.ui.workers import detect_sampling_frequency, load_raw_matrix
        if not files:
            return ""
        ref = files[0]
        parts = []
        previous_fs = self.samp_freq.value()
        prev, self._loading = self._loading, True
        try:
            try:                                     # sampling frequency from the (col 0) time axis
                matrix, _names = load_raw_matrix(self.state.settings, ref)
                fs = detect_sampling_frequency(matrix[:, 0]) if matrix.shape[1] else None
            except Exception:                        # noqa: BLE001 — detection is best-effort
                fs = None
            if fs:
                self.state.settings.input.format.sampling_frequency = fs
                self.samp_freq.setValue(fs)
                parts.append(f"{fs} Hz")
                self._samp_freq_from_detection = True
                self._samp_freq_detection_note = (previous_fs, fs) if fs != previous_fs else None
                self._samp_freq_detection_folder = self.in_folder.text().strip()
                self._update_samp_freq_marker()
        finally:
            self._loading = prev
        return ", ".join(parts)

    def _update_samp_freq_marker(self):
        """Show/hide the dimmed "Detected from the time column." row beside the sampling-
        frequency field, purely from ``self._samp_freq_from_detection`` — see its own
        comment in __init__ for when that is set/cleared."""
        lab = getattr(self, "samp_freq_detected_label", None)
        form = getattr(self, "_input_form", None)
        if lab is None:
            return
        lab.setVisible(self._samp_freq_from_detection)
        if form is not None:
            form.setRowVisible(lab, self._samp_freq_from_detection)

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
                self.state.settings.processing.volume.integrate_from_flow,  # Preview-owned now
                f.sampling_frequency, f.decimal, self.in_folder.text(), self.in_files.text())

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
        normal reactive commit (which re-validates and, in the guided flow, marks the
        channel step done). ``fmt_note`` is an optional summary of any auto-detected file
        settings to mention in the status.

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
        """Recompute the live QC strip, the one-shot channel-modal auto-open (guided 'new
        analysis' flow only), and the live validation status. B04 retired progressive
        disclosure: every card is visible from the first frame in every mode (only
        _apply_card_visibility's relevance-based _cond_cards still hide/show), so there is
        no longer a separate 'full' vs 'new' path here except for the channel-modal
        one-shot, which stays guided-flow-only — auto-opening it over an already-open
        analysis would be a surprise, not a courtesy."""
        self._update_qc()               # keep the live caution strip current in every mode
        self._apply_card_visibility()
        if (self._mode == "new" and not self._channel_modal_done
                and not self._channel_modal_pending and self._input_stage_ok()):
            self._channel_modal_pending = True
            # Bound to self (same idiom as dialogs.py's copy-button reset): if this screen
            # is ever destroyed before the deferred open runs, Qt drops the call instead of
            # firing it against a stale object. Found in self-review after a related bug
            # (a test that outlived this timer and left it pending — see
            # test_reentry_rearms_the_channel_gate) that this alone does not fix, since
            # closing a window without WA_DeleteOnClose does not destroy the C++ object.
            QTimer.singleShot(0, self, self._open_channel_setup_for_flow)
        ready = self._all_ok()
        self._set_flow_ready(ready)
        self._set_status(self._validation_status())   # no Validate button: every edit re-checks

    def _apply_card_visibility(self):
        """Conditional cards only (B04 retired the staged reveal): Sample entropy stays
        hidden unless a column is actually assigned to it, in every mode. The one exemption
        that survives is the focus guard — a card holding the widget the user is typing in
        is never yanked out from under them."""
        for card, relevant in self._cond_cards:
            if card.isVisible() and card.isAncestorOf(QApplication.focusWidget()):
                continue
            card.setVisible(relevant())

    def _channel_collision(self):
        """A HARD channel-mapping error (message, else ''), delegating to the Qt-free
        ``ui.validation.channel_collision`` (moved there in B04 so the Run screen's
        commitment sheet can name exactly the same blocker this screen's QC strip does,
        for the identical mapping)."""
        from respmech.ui.validation import channel_collision
        return channel_collision(self.state.settings) or ""

    def _set_flow_ready(self, ready):
        ready = bool(ready)
        if ready != self._flow_ready:
            self._flow_ready = ready
            self.flow_ready_changed.emit(ready)

    def _input_stage_ok(self):
        """Input card complete: a real recordings folder with matching files, and a
        sampling frequency. B04's sole remaining use: the guided flow's one-shot
        channel-modal auto-open hangs on this (an input folder that has matching files),
        not on a disclosure stage — the modal needs actual data to preview."""
        s = self.state.settings
        folder = (s.input.folder or "").strip()
        if not folder or not os.path.isdir(folder):
            return False
        if not matching_files(folder, s.input.files):
            return False
        fq = s.input.format.sampling_frequency
        return isinstance(fq, int) and fq >= 1

    def _all_ok(self):
        """Every setting is valid (core validation + filesystem paths) — drives
        flow_ready_changed and this screen's own 'Setup valid ✓' status. The Run
        drawer's own primary-action gate (B04) is computed independently, over the same
        shared checks (``ui.validation.channel_collision``/``path_problem``), so the two
        can never disagree about a setting without also disagreeing about a run."""
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
        # A per-file sampling-frequency mismatch invalidates essentially every derived
        # number for the affected files (every time-derived quantity is wrong by exactly
        # the frequency ratio, and the values stay plausible rather than absurd — see the
        # B01 manifest scanner), so it outranks the two notes below it, which only degrade
        # signal quality rather than silently misreporting it.
        m = self._manifest
        if m is not None and m.freq_mismatches:
            mism = m.freq_mismatches
            rates = sorted({f.detected_fs for f in mism})
            rate_txt = " or ".join(f"{r} Hz" for r in rates)
            out.append(f"{len(mism)} of {len(m.included_files)} files look like {rate_txt}, "
                       f"not the {m.settings_fs} Hz set here: {self._named_list(mism)}")
        # A missing prerequisite outranks advice about signal quality: this one makes a
        # requested output come back empty, rather than merely making it noisier. The gated
        # peak reuses the heartbeats the ECG stage detects, and with ECG removal off there are
        # none. Caution rather than block — remove_ecg lives on the Preview screen, so a hard
        # failure here would strand the user with no control to change.
        rp = getattr(s.processing.emg, "robust_peak", None)
        if rp is not None and rp.enabled and not s.processing.emg.remove_ecg:
            out.append("cardiac-gated peak EMG needs ECG removal on (Preview & QC ▸ "
                       "EMG – ECG reduction), or its columns will be blank")
        # Tests the rate the batch will actually ANALYSE at, not the raw input rate: with
        # 'Resample before analysis' on (Preview & QC ▸ Mechanics ▸ Advanced…), every EMG
        # channel is normalised against the resampled envelope, never the recorded one (see
        # workers.py's own note that Preview never resamples) — same fs_eff formula
        # core.settings.Settings.validate() already uses for trend_peak_min_distance_s, so
        # the two can never disagree about what "the analysis rate" means.
        samp = s.processing.sampling
        resampling = bool(samp.resample and samp.resample_to_frequency and samp.resample_to_frequency > 0)
        fs_input = s.input.format.sampling_frequency or 0
        fs_eff = samp.resample_to_frequency if resampling else fs_input
        if ch.emg and fs_eff and fs_eff < 1000:
            if resampling and fs_eff != fs_input:
                out.append(f"analysis rate {fs_eff} Hz (resampled from {fs_input} Hz) is low "
                           "for EMG — Preview & QC ▸ Mechanics ▸ Advanced… ▸ Sampling")
            else:
                out.append(f"sampling frequency {fs_eff} Hz is low for EMG")
        return out

    def _science_note(self):
        """The most important caution, for the one-line live verdict. '' when there is none."""
        notes = self._science_notes()
        return notes[0] if notes else ""

    def _manifest_cautions(self):
        """B01: caveats about the batch ITSELF (mask narrowing, a column-count outlier,
        and — ticket D01 — a header-block file that is majority-consistent and so invisible
        to the outlier check) — as distinct from ``_science_notes``, which is about the
        channel/settings configuration. Kept separate so ``_science_notes`` (used verbatim
        for the guided-flow one-line verdict) is not forced to explain a fact about the
        file set."""
        m = self._manifest
        if m is None:
            return []
        out = []
        if m.mask_narrowed_from:
            out.append(self._narrowed_note(m))
        if m.outliers:
            n_out = len(m.outliers)
            out.append(f"{n_out} of {len(m.files)} files have a different column count "
                       f"and will fail: {self._named_list(m.outliers)}")
        if m.header_warnings:
            n_hdr = len(m.header_warnings)
            out.append(f"{n_hdr} of {len(m.files)} files may not be real channel data (a "
                       f"header block, or too few columns): "
                       f"{self._named_list(m.header_warnings)}")
        return out

    def _output_is_input_folder(self):
        """True when Output points at the same folder as the recordings (ticket B05).

        Not itself a save/run blocker — ``validation.path_problem`` only requires the
        output's PARENT to exist — but a run then writes ``analysis-used.toml``,
        ``run-report.txt`` and a ``data/`` subfolder straight into the raw-data folder,
        often a synced or read-only patient drive, with nothing on screen to say so until
        it happens. Compares absolute paths so 'input' and './input' still match."""
        a = (self.state.settings.input.folder or "").strip()
        b = (self.state.settings.output.folder or "").strip()
        if not a or not b:
            return False
        try:
            return os.path.abspath(a) == os.path.abspath(b)
        except Exception:                       # noqa: BLE001 — a caution is cosmetic
            return False

    def _qc_verdict(self):
        """The Setup QC strip's single verdict, as ``(status, text)``.

        Ticket B07: the strip used to reach its own conclusion from ``_channel_collision()``
        plus ``_science_notes()`` alone — never calling ``Settings.validate()`` or
        ``_path_problem()``, the exact checks ``_all_ok()`` gates a run on — so it could
        show a green checkmark in states that cannot actually run (an unset volume channel,
        a broken output path, a folder no file of which could be read as data). The strip no
        longer judges anything itself: it renders the SAME verdict the rest of the app
        already reached, worst first.

        Four tiers, in priority order:
        - ``'muted'``: nothing has been scanned yet (no input folder, or the mask matches no
          files) — not an error, just nothing to report. Checked FIRST, ahead of the shared
          hard-blocker list, because ``ui.validation.path_problem``'s own "no files match…"
          message is exactly this same fact said in alarming style; a folder nobody has
          pointed the app at yet is not a broken one.
        - ``'error'``: a hard blocker — channel collision, a ``SettingsError`` from
          ``validate()``, or a path problem (``ui.validation.blockers``, ticket B07: the ONE
          shared list the Run screen's commitment sheet also reads, so the two can never
          name a different top blocker for the same settings) — or the manifest found files
          but could not read ANY of them as data (B01's ``majority_columns is None``, e.g. a
          decimal-separator mismatch: the mask matched real files, so ``path_problem`` alone
          cannot see the problem).
        - ``'warn'``: a soft caution — a column-count outlier, a header-block file that
          slipped past that outlier check (ticket D01), a narrowed mask, output
          pointing at the input folder, or a science note (frequency mismatch, ECG
          prerequisite, low EMG rate) — nothing here blocks a run, but every one of them
          changes what the batch actually does or writes.
        - ``'ok'``: only once ``_all_ok()``'s own checks are clean AND the manifest has
          actually read at least one file — never for a caveat-free batch nobody has looked
          inside yet.

        Ticket C03 point 6: a folder that is SET but no longer exists is not the same fact
        as "nobody has pointed the app at one yet" — checked before the muted branch below,
        which would otherwise catch it too (``_update_format_readout`` never builds a
        manifest for a missing folder, so ``self._manifest`` is ``None`` either way)."""
        folder = self.in_folder.text().strip()
        # isabs, matching _update_format_readout's own guard: a bare relative default
        # ("input", nobody's chosen it) must not read as a real, now-missing folder.
        if folder and os.path.isabs(folder) and not os.path.isdir(folder):
            return "warn", f"Recordings folder not found: {folder}"
        m = self._manifest
        if m is None or not m.files:
            return "muted", "Nothing scanned yet — set an input folder with matching files."
        from respmech.ui.validation import blockers as _shared_blockers
        # matches=... straight from the manifest we already built (self-review finding):
        # path_problem only ever asks "is this non-empty?", and the manifest can never be
        # non-empty here without a real folder edit having rebuilt it first — so this is the
        # exact same answer a fresh glob would give, without repeating the directory scan
        # _update_format_readout() already just did.
        hard = _shared_blockers(self.state.settings, matches=[f.path for f in m.files])
        if hard:
            return "error", hard[0]
        if m.majority_columns is None:
            return "error", ("no file in this folder could be read as data — check "
                             "'Files to analyse' and the decimal separator")
        cautions = self._manifest_cautions()
        if self._output_is_input_folder():
            cautions.append("Results will be written into your recordings folder.")
        cautions.extend(self._science_notes())   # every note, not just the first
        if cautions:
            return "warn", "   ·   ".join(cautions)
        return "ok", "Ready to run — no warnings."

    def refresh_qc(self):
        """Recompute the caution strip without touching any widget value.

        Public because a caution can now depend on a field this screen does not own: the
        gated-peak prerequisite reads processing.emg.remove_ecg, which the Preview ECG tab
        writes. Without this the strip goes stale in both directions — most awkwardly, the
        warning tells the user to go and fix remove_ecg on Preview and then stays pinned here
        after they have done exactly that."""
        self._update_qc()

    def _update_qc(self):
        """Refresh the pinned QC strip with the live verdict (``_qc_verdict``): a hard error,
        a soft caution, a muted "nothing scanned" note, or an all-clear."""
        if getattr(self, "qc", None) is None:
            return
        status, text = self._qc_verdict()
        icon = {"error": "✗  ", "warn": "⚠  ", "muted": "", "ok": "✓  "}[status]
        self.qc.setText(icon + text)
        self.qc.setProperty("status", status)
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

    def _report_error(self, operation, detail, intro=None, collapsed_detail=False):
        """Consistent failure surface: a short status line + a copyable full trace
        (matches the Preview/Run screens). ``intro``/``collapsed_detail`` (ticket D16,
        reusing D01's pattern) let a caller that already has a plain-language diagnosis
        lead with it — the trace stays one click away behind a "Details" toggle instead
        of the generic "<operation> failed." caption that explains nothing."""
        self._set_status(f"{operation} failed — {short_error(detail)}")
        self._err_dialog = open_error_dialog(
            self, f"{operation} — error", detail,
            intro=intro or f"{operation} failed. The full detail below is copyable.",
            collapsed_detail=collapsed_detail,
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
        except Exception as e:                  # noqa: BLE001 — roll back so nothing partially applies
            detail = traceback.format_exc()
            self.state.settings, self.state.settings_path = prior, prior_path
            self._resync_form()
            if isinstance(e, tomllib.TOMLDecodeError):
                # ticket D16: a malformed .toml used to surface as a bare "Open analysis
                # failed" caption over a 15-line traceback ending inside tomllib itself —
                # the user's mental model is "my analysis file", and the app deliberately
                # never says TOML anywhere else in the interface. str(e) already carries
                # tomllib's own "(at line N, column M)" location, so it is reused verbatim
                # rather than re-parsed.
                self._report_error(
                    "Open analysis", detail,
                    intro=f"'{os.path.basename(p)}' could not be read — it is not a "
                          f"valid analysis file ({e}). Your current analysis is unchanged.",
                    collapsed_detail=True)
            else:
                self._report_error("Open analysis", detail)
            return False
        self._set_status(f"Opened {p}")
        self._mark_clean()
        self.surface_notices()
        prefs.add_recent_analysis(p)                 # P26 recent analyses
        prefs.set_last_folder("analysis", p)
        prefs.save_rig(self.state.settings)          # P25 remember the rig
        self.inputs_changed.emit()
        self.settings_changed.emit()
        return True

    def surface_notices(self):
        """Tell the user about any schema upgrade applied while loading this analysis.

        A setting this version reads differently from the version that saved the file
        changes the results, so it is said on screen at open time — not only in the run
        report, which is written after the numbers already exist."""
        for note in self.state.settings.notices:
            QMessageBox.information(self, "Analysis updated for this version", note)

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
            from respmech.ui.validation import friendly_settings_error
            return friendly_settings_error(e)
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

    def _detach_sample_from_temp(self, dest_path):
        """K-047 (indholdsgennemgang respmech.dk, ticket 7.3): the built-in sample's
        recording, output and EMG noise-reference folders all live under the OS temp
        directory (``open_sample_analysis``/``core.sample``), which the OS may clear at
        any time. A plain 'Save as…' used to copy those absolute temp paths verbatim into
        the saved file, so the saved analysis stopped working the moment the temp folder
        was cleared: ``validate`` reported 'matches 0 files' and a run failed with 'No
        input files found'.

        Copies the recording out of the temp input folder into an ``input`` subfolder next
        to the destination file, and repoints ``input.folder``/``output.folder``, the EMG
        noise reference folder, and any exclude-breaths/breath-count entry that named the
        same temp folder — the carried-folder tags this screen elsewhere compares against
        the live ``input.folder`` (``core.settings.is_carried_folder``) would otherwise
        still point at the OLD temp folder while ``input.folder`` itself moved, making a
        breath excluded moments earlier during this same sample session falsely read as
        "carried over from a different folder" the instant the analysis is reopened.
        ``AppState.save_toml`` -> ``settingsio.toml_io`` already relativizes any
        input/output folder that ends up living at/under the file's own directory, so
        nothing here needs to compute a relative path itself — it only needs to move the
        folders somewhere ``save_toml`` will recognise as portable. Runs before the model
        is written, so the very first save already lands with a working, portable
        analysis rather than needing a second pass.

        If the temp recording is already gone (the OS cleared it before this save), the
        copy is a no-op and the user is warned explicitly — silently writing an empty
        ``input`` folder would reproduce the exact "matches 0 files" failure this method
        exists to prevent, just one save later and with no clue why."""
        import shutil  # noqa: PLC0415
        s = self.state.settings
        old_input = s.input.folder
        dest_dir = os.path.dirname(os.path.abspath(dest_path)) or "."
        new_input = os.path.join(dest_dir, "input")
        os.makedirs(new_input, exist_ok=True)
        copied_any = False
        if old_input and os.path.isdir(old_input):
            for name in os.listdir(old_input):
                src = os.path.join(old_input, name)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(new_input, name))
                    copied_any = True
        if not copied_any:
            QMessageBox.warning(
                self, "Save analysis",
                "The sample's temporary recording could not be found — the operating "
                "system may already have cleared it. The analysis will be saved, but "
                "it has no matching input file yet; point Setup at a real recordings "
                "folder before running it.")
        noise = s.processing.emg.noise
        if noise.reference_folder == old_input:
            noise.reference_folder = new_input
        for entry in (*s.processing.exclude_breaths, *s.processing.breath_counts):
            if entry.folder == old_input:
                entry.folder = new_input
        s.input.folder = new_input
        s.output.folder = os.path.join(dest_dir, "output")

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

    def save_analysis_as(self, suggested_path=None):
        """Analysis > 'Save as…': write the current settings to a chosen analysis file. The
        chooser asks before overwriting, so unlike Save it needs no extra confirmation.
        ``suggested_path`` (ticket C03 point 5) lets a caller pre-fill somewhere other
        than the currently-open file — 'Duplicate for another recordings folder…' uses it
        to suggest the SAME analysis filename inside the NEW recordings folder, since the
        currently-open path is the template being duplicated, not where the duplicate
        belongs. Saving the built-in sample (K-047) first detaches it from the OS temp
        folder — see ``_detach_sample_from_temp``. Returns True iff saved."""
        from respmech.ui import prefs  # noqa: PLC0415
        self.to_state()
        blocker = self._save_blocker()
        if blocker:
            return self._refuse_save(blocker)
        start = suggested_path or self.state.settings_path or os.path.join(
            prefs.last_folder("analysis", "."), "analysis.toml")
        p, _ = QFileDialog.getSaveFileName(self, "Save analysis as", start, TOML_FILTER)
        if not p:
            return False
        if getattr(self.state, "is_sample", False):
            self._detach_sample_from_temp(p)
        return self._write_analysis(p)

    def duplicate_for_another_folder(self):
        """Analysis > 'Duplicate for another recordings folder…' (ticket C03 point 5): keep
        every setting, point the analysis at a NEW batch of recordings, and open Save
        as… so the duplicate is written to a NEW file — never overwriting the template it
        came from.

        The output folder is only ever a SUGGESTION (``ui.duplicate.derive_sibling_output``),
        shown in ``DuplicateFolderDialog`` for confirmation/editing, never applied silently.

        The file-keyed state (exclude_breaths/breath_counts/the EMG noise reference) is
        deliberately NOT force-cleared here: switching ``input.folder`` makes every entry
        recorded against the OLD folder "carried-over" by B06's own definition
        (``core.settings.carried_over_state``), so the Setup Behold/Ryd banner this screen
        already shows will ask about exactly that state right after — reusing B06's
        existing ask, not a second copy of it, per this ticket's own instruction to prefer
        the already-built helper. ``processing.emg.ecg_reference_file`` has no such
        folder-tracked ask mechanism (see core/settings.py), so it is cleared directly —
        it can only ever have named a file in the OLD folder."""
        if not self.confirm_discard_changes(
                "Duplicate for another recordings folder",
                question="Save them before duplicating this analysis?"):
            return
        from respmech.ui import prefs  # noqa: PLC0415
        from respmech.ui.duplicate import derive_sibling_output
        from respmech.ui.duplicate_dialog import DuplicateFolderDialog
        old_input = self.state.settings.input.folder
        old_output = self.state.settings.output.folder
        old_settings_path = self.state.settings_path
        start = old_input or prefs.last_folder("browse", ".")
        new_input = QFileDialog.getExistingDirectory(
            self, "New recordings folder for the duplicate", start)
        if not new_input:
            return
        suggested_output = derive_sibling_output(old_input, old_output, new_input)
        dlg = DuplicateFolderDialog(new_input, suggested_output or "", parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        new_output = dlg.output_folder()
        if not new_output:
            QMessageBox.warning(self, "Duplicate for another recordings folder",
                                "An output folder is required.")
            return
        self.in_folder.setText(new_input)
        self.out_folder.setText(new_output)
        self.to_state()
        self.state.settings.processing.emg.ecg_reference_file = None
        # Duplicating a sample-derived analysis onto real folders means it is no longer
        # the built-in sample — see AppState.is_sample's own docstring.
        self.state.is_sample = False
        self._normalize_mask()
        self._mark_dirty()
        self.inputs_changed.emit()
        self.settings_changed.emit()
        self._update_format_readout()
        self._update_carried_banner()      # B06's own Behold/Ryd ask — see docstring above
        self._update_sample_banner()
        self._set_status(f"Duplicated for {new_input}. Choose where to save this as a new analysis.")
        suggested_name = os.path.basename(old_settings_path) if old_settings_path else "analysis.toml"
        self.save_analysis_as(suggested_path=os.path.join(new_input, suggested_name))

    def _import(self, path=None):
        """Import a legacy .py setup (runs the migrator). Returns True on success, False
        if cancelled or rolled back, so open_analysis can gate the mode transition."""
        p = path or QFileDialog.getOpenFileName(
            self, "Open legacy analysis (.py)", self._analysis_dialog_start(), LEGACY_FILTER)[0]
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
        self._report_dialog = open_migration_report(
            self, report, source_path=p, prior=self._report_dialog)
        self._set_status(f"Imported {p}")
        # ticket C03 point 7: a converted 1.x analysis has no .toml of its own yet — it was
        # marked CLEAN before, which is what let closeEvent's confirm_discard_changes and
        # the Analysis menu's Save-enable check both treat a just-imported migration as
        # nothing-to-lose. The conversion itself is deterministic and repeatable from the
        # untouched .py (state.import_legacy can be re-run any time), but the analysis now
        # sitting in memory is genuinely unsaved work.
        self._mark_dirty()
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
            from respmech.ui.validation import friendly_settings_error
            return f"Invalid: {friendly_settings_error(e)}"
        except Exception:                       # noqa: BLE001 — never let a fault break typing
            return f"Could not validate: {short_error(traceback.format_exc())}"
        # path checks (the core validate() is filesystem-agnostic)
        problem = self._path_problem()
        if problem:
            return f"Invalid: {problem}"
        # non-fatal science guardrails (shared with the guided-flow completion status)
        note = self._science_note()
        return f"Valid, but check: {note}." if note else "Setup valid ✓"

    def _path_problem(self):
        """Return a human message for the first invalid path, or None if all OK.
        Shared with the Run screen so both surface the same filesystem checks."""
        from respmech.ui.validation import path_problem
        return path_problem(self.state.settings)


