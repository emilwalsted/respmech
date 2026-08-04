"""BusyOverlay: the per-panel spinner/error card shown while a preview job is
in flight. Split out of preview_screen.py (ticket A02); moved verbatim."""

from __future__ import annotations

import copy
import math
import os
import traceback
from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton,
                               QScrollArea, QSplitter, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget)
from PySide6.QtCore import Qt, QEvent, QObject, QSize, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QFontMetrics

import pyqtgraph as pg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from respmech.core.settings import ExcludeEntry
from respmech.ui.dialogs import TextViewerDialog, short_error
from respmech.ui.help_text import tooltip as _help_tip
from respmech.ui import plot_perf
from respmech.ui.plot_overlays import add_flow_background, add_ecg_capture_markers
from respmech.ui import wheel as _wheel
from respmech.ui.flow_layout import (FlowLayout, cluster as _cluster,
                                     elide as _elide, install_flow as _install_flow)
from respmech.ui.workers import (BatchWorker, EmgAllChannelsWorker,
                                  EmgConditioningWorker, FnWorker,
                                  stage_ecg_reduction, stage_mechanics_preview,
                                  stage_noise_fidelity)

try:
    from respmech.ui import theme as _theme
except Exception:  # pragma: no cover
    _theme = None



class BusyOverlay(QWidget):
    """A translucent overlay that covers one panel — either a spinner while the
    panel's job runs, or an error card (short message + a round info button that
    opens the full, copyable trace) if the job failed.

    State lives in plain attributes (``busy``/``error``/``message``), so it is
    inspectable under a headless test without any pixels. It re-covers its parent
    on resize/move (splitter drags) and is only ever shown/hidden from the GUI
    thread."""

    def __init__(self, panel: QWidget):
        super().__init__(panel)
        # Qt only auto-sets WA_StyledBackground when the widget's metaObject IS
        # QWidget's — a Python SUBCLASS like this one is excluded, so the
        # `#busyOverlay { background-color: rgba(...) }` rule below was parsed, matched and
        # then never painted: the panel was not dimmed at all while a job ran, in either
        # theme. Setting it explicitly is the documented way to opt a custom widget in.
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.busy = False
        self.error = None
        self.message = ""
        self._detail_dialog = None
        self.setObjectName("busyOverlay")

        # -- busy card: caption + indeterminate bar --
        self._busy_box = QWidget(self)
        self._busy_box.setObjectName("busyBox")
        bl = QVBoxLayout(self._busy_box)
        bl.setContentsMargins(18, 14, 18, 14)
        bl.setSpacing(8)
        self._label = QLabel("", self._busy_box)
        self._label.setObjectName("busyLabel")
        self._label.setAlignment(Qt.AlignCenter)
        self._bar = QProgressBar(self._busy_box)
        self._bar.setObjectName("busyBar")
        self._bar.setRange(0, 0)          # indeterminate = animated spinner
        self._bar.setTextVisible(False)
        self._bar.setFixedWidth(150)
        bl.addWidget(self._label)
        bl.addWidget(self._bar, 0, Qt.AlignCenter)

        # -- error card: ⚠ + summary + round info button --
        self._error_box = QWidget(self)
        self._error_box.setObjectName("errorBox")
        self._error_box.setMaximumWidth(360)
        el = QVBoxLayout(self._error_box)
        el.setContentsMargins(16, 14, 16, 14)
        el.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(9)
        self._err_icon = QLabel("⚠")
        self._err_icon.setObjectName("errIcon")
        self._err_label = QLabel("")
        self._err_label.setObjectName("errLabel")
        self._err_label.setWordWrap(True)
        self._info_btn = QPushButton("i")
        self._info_btn.setObjectName("infoBtn")
        self._info_btn.setFixedSize(24, 24)
        self._info_btn.setToolTip("Show the full error detail (copyable)")
        self._info_btn.setCursor(Qt.PointingHandCursor)
        self._info_btn.clicked.connect(self._open_detail)
        head.addWidget(self._err_icon, 0, Qt.AlignTop)
        head.addWidget(self._err_label, 1)
        head.addWidget(self._info_btn, 0, Qt.AlignTop)
        el.addLayout(head)
        hint = QLabel("Click the info button for the full trace.")
        hint.setObjectName("errHint")
        hint.setWordWrap(True)
        el.addWidget(hint)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._busy_box, 0, Qt.AlignCenter)
        lay.addWidget(self._error_box, 0, Qt.AlignCenter)
        self._error_box.hide()

        # the round info button tracks the active theme's accent (so it is the dark
        # accent in dark mode, not the light-mode steel-blue) — everything else on the
        # overlay already sits on a translucent dark scrim and works in both themes.
        t = _theme.active_theme() if _theme is not None else {}
        acc = t.get("accent", "#2C6E9B")
        acc_fg = t.get("accent_fg", "white")
        acc_hover = t.get("accent_hover", "#3E8AC0")
        # The error CARD is not on the scrim — it paints palette(window), so in light mode it
        # is #EEF1F5 and the two dark-tuned literals that used to sit here (#E08A4F icon,
        # rgba(147,164,183) hint) rendered at 2.3:1 against it. Both meanings already have a
        # theme token, so use them and let each mode pick its own value.
        warn_fg = t.get("st_warn_fg", "#8A5A12")
        hint_fg = t.get("text_muted", "#5C6B7A")
        self.setStyleSheet(
            "#busyOverlay { background-color: " + t.get("scrim", "rgba(18,22,28,0.42)") + "; }"
            "#busyBox, #errorBox { background-color: palette(window);"
            " border: 1px solid rgba(128,128,128,0.35); border-radius: 10px; }"
            "#errorBox { border-color: rgba(180,50,42,0.55); }"
            "#busyLabel, #errLabel { color: palette(window-text); font-weight: 600; }"
            "#errIcon { color: " + warn_fg + "; font-size: 17px; font-weight: 700; }"
            "#errHint { color: " + hint_fg + "; font-size: 11px; }"
            # padding/min-* MUST be reset: the global QPushButton padding (7px 16px) is
            # wider than this 24x24 circle and would clip the 'i' away entirely.
            "#infoBtn { background-color: " + acc + "; color: " + acc_fg + "; border: none;"
            " border-radius: 12px; font-weight: 700; font-style: italic;"
            " padding: 0; min-width: 0; min-height: 0; }"
            "#infoBtn:hover { background-color: " + acc_hover + "; }")
        panel.installEventFilter(self)
        self.setGeometry(panel.rect())
        self.hide()

    def centre_on_visible(self):
        """Keep the spinner/error card in the part of the panel the user can actually SEE.

        The card is centred in the overlay, and the overlay covers the whole panel — which
        was the same thing until the Preview pages started scrolling. A panel that is half
        below the fold then centres its card below the fold too: measured 6 of 10 panels with
        the card outside the viewport at 1097x547, so a failed job reported itself somewhere
        the user had no reason to scroll to.

        Padding the layout by exactly the clipped-away strips re-centres the card on the
        visible band without moving or resizing the overlay itself.
        """
        try:
            lay = self.layout()
            if lay is None:
                return
            vis = self.visibleRegion().boundingRect()
            if vis.isEmpty():
                return
            top = max(0, vis.top())
            bottom = max(0, self.height() - vis.bottom() - 1)
            if (lay.contentsMargins().top(), lay.contentsMargins().bottom()) != (top, bottom):
                lay.setContentsMargins(0, top, 0, bottom)
        except Exception:                      # pragma: no cover - positioning is never fatal
            pass

    def eventFilter(self, obj, ev):
        if obj is self.parent() and ev.type() in (QEvent.Resize, QEvent.Move):
            self.setGeometry(self.parent().rect())
        return False

    def start(self, message="Working…"):
        self.busy = True
        self.error = None
        self.message = message
        self._label.setText(message)
        self._error_box.hide()
        self._busy_box.show()
        self.setGeometry(self.parent().rect())
        self.raise_()
        self.show()

    def stop(self):
        self.busy = False
        self.error = None
        self.message = ""
        self.hide()

    def show_error(self, summary, detail):
        """Switch to the error card: a short summary + a round info button that
        opens the full ``detail`` (traceback) in a copyable dialog."""
        self.busy = False
        self.error = detail or ""
        self._err_label.setText(summary or "Something went wrong")
        self._busy_box.hide()
        self._error_box.show()
        self.setGeometry(self.parent().rect())
        self.raise_()
        self.show()

    def _open_detail(self):
        if not self.error:
            return
        if self._detail_dialog is not None:   # replace, don't accumulate windows
            self._detail_dialog.close()
            self._detail_dialog.deleteLater()
            self._detail_dialog = None
        dlg = TextViewerDialog(
            "Error detail", self.error, self,
            intro="Full error trace — select the text or use Copy to clipboard.")
        self._detail_dialog = dlg          # keep a reference so it isn't GC'd
        dlg.show()
        dlg.raise_()

    def is_busy(self):
        return self.busy
