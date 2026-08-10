"""Capture every RespMech screen and dialog with the bundled synthetic sample loaded.

One process captures one theme:

    RESPMECH_THEME=light|dark  RM_OUT=<dir>  python tools/capture_screens.py

Written for the screenshots CI workflow (.github/workflows/screenshots.yml), which runs it
on a real Windows and a real macOS runner so the shots carry each platform's true font
metrics — the difference that decides what wraps and what fits. It runs fine locally too.
``RM_STRETCH=145`` additionally applies the repo's Windows font MODEL (see CLAUDE.md); on a
real Windows machine leave it unset.

The reactive pipeline runs for real (QThread workers + a pumped event loop), so every panel
shows computed results, and the Run screen a completed dry run — nothing is written outside
the sample's own temp folder. Renders via win.render() into a QImage: grab() does not paint
a top-level's background offscreen, which hides exactly the theming defects worth seeing.
"""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.path.isdir("src"):
    sys.path.insert(0, "src")   # repo-root invocation; CI uses the installed package

from PySide6.QtCore import QRect, QPoint
from PySide6.QtGui import QImage, QColor
from PySide6.QtWidgets import QApplication, QDialog

app = QApplication(sys.argv)
from respmech.ui import theme

MODE = theme.apply_theme(app)
STRETCH = int(os.environ.get("RM_STRETCH", "100"))
if STRETCH != 100:
    f = app.font()
    f.setStretch(STRETCH)
    app.setFont(f)
OUT = os.environ.get("RM_OUT") or sys.exit("set RM_OUT to the output directory")
os.makedirs(OUT, exist_ok=True)
W, H = 1280, 800
AVAIL = QRect(0, 0, W, H)


def pump(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.02)


def settle(pv, timeout=180):
    """Pump until no panel overlay is busy (the reactive jobs have delivered)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.05)
        busy = [n for n, ov in pv._overlays.items() if ov.isVisible() and getattr(ov, "busy", False)]
        if not busy and not pv._jobs and not pv._launch_queue:
            pump(0.5)          # one more beat for paints scheduled by the delivery
            if not pv._jobs and not pv._launch_queue:
                return True
    return False


def shoot(widget, name):
    img = QImage(widget.size(), QImage.Format_ARGB32)
    img.fill(QColor(theme.active_theme()["window"]))
    widget.render(img)
    path = os.path.join(OUT, f"{name}.png")
    img.save(path)
    print(f"  {name}: {widget.width()}x{widget.height()}")


from respmech.ui.state import AppState
from respmech.ui.main_window import MainWindow
from respmech.ui import screen_fit

win = MainWindow(AppState())
win.resize(W, H)
win.show()
pump(0.5)
ok = win.settings_screen.open_sample_analysis()
print(f"theme={MODE} stretch={STRETCH} sample loaded: {ok}")
pump(1.0)

# --- Setup ------------------------------------------------------------------
win.tabs.setCurrentIndex(0)
pump(1.0)
shoot(win, "01_setup")

# --- Preview subtabs --------------------------------------------------------
win.tabs.setCurrentIndex(1)
pv = win.preview_screen
pump(1.0)
names = {0: "02_preview_mechanics", 1: "03_preview_ecg", 2: "04_preview_noise"}
for i in range(pv.subtabs.count()):
    pv.subtabs.setCurrentIndex(i)
    pump(0.5)
    done = settle(pv)
    print(f"  subtab {i} settled: {done}")
    shoot(win, names.get(i, f"0x_preview_{i}"))

# --- Run & results (dry run: computes everything, writes nothing) -----------
# Run & results is a drawer under Preview & QC (ticket B03), not a third tab — there is
# no tabs index 2 to select. Land explicitly on Preview & QC's Mechanics subtab so the
# shot is deterministic regardless of which subtab the loop above left current on.
win.tabs.setCurrentIndex(1)
pv.subtabs.setCurrentIndex(0)
pump(0.3)
rn = win.run_screen
pump(0.5)
finished = []
rn.run_finished.connect(lambda: finished.append(1))
try:
    rn._start(write=False)
    end = time.monotonic() + 600
    while not finished and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.05)
    print(f"  dry run finished: {bool(finished)}")
    pump(1.0)
except Exception as e:  # noqa: BLE001
    print(f"  dry run failed: {e}")
shoot(win, "05_run_results")

# --- Dialogs ----------------------------------------------------------------
# Patch BOTH QDialog.exec and AdvancedDialog.exec, not just the base class: the
# Mechanics Advanced dialog is opened with modal=False, and AdvancedDialog.exec()
# (ticket D13) never calls super().exec() in that case — it shows itself and runs its
# own QEventLoop instead (see its own docstring for why: QDialog.exec() forces
# WA_ShowModal unconditionally, which a genuinely non-modal dialog must not accept). A
# patch that only replaces QDialog.exec therefore never intercepts THAT dialog: it pops
# up for real and blocks forever, since nothing here simulates closing it. Same gotcha,
# same fix, as CLAUDE.md documents for the app's own test suite (test_dialog_fits_screen.py
# / test_theme_paints_both_modes.py) — patch the MOST DERIVED class an opener actually
# builds. Confirmed by reproducing the hang locally before this fix (offscreen, >15 min,
# 99% CPU, stuck immediately after "05_run_results").
from respmech.ui.advanced_dialog import AdvancedDialog

captured = {}
orig_exec = QDialog.exec
orig_advanced_exec = AdvancedDialog.exec


def fake_exec(self):
    captured[self.windowTitle() or type(self).__name__] = self
    return QDialog.Rejected


def snap_dialog(key, name):
    dlg = captured.pop(key, None)
    if dlg is None:
        print(f"  !! no dialog captured for {name}")
        return
    dlg.show()
    screen_fit.clamp_to_screen(dlg, avail=AVAIL)
    pump(0.6)
    shoot(dlg, name)
    dlg.close()


win.tabs.setCurrentIndex(1)
pump(0.5)
QDialog.exec = fake_exec
AdvancedDialog.exec = fake_exec
try:
    pv._open_mech_advanced()
    snap_dialog("Mechanics — advanced", "06_dlg_mech_advanced")
    pv._open_emg_advanced()
    snap_dialog("EMG — advanced", "07_dlg_emg_advanced")
    pv._open_ecg_advanced()
    snap_dialog("ECG removal — advanced", "08_dlg_ecg_advanced")
    pv._open_noise_profile_dialog()
    key = next((k for k in captured if k.startswith("Set noise profile")), None)
    snap_dialog(key, "09_dlg_noise_profile")
    win.tabs.setCurrentIndex(0)
    pump(0.3)
    win.settings_screen._open_channel_setup()
    key = next(iter(captured), None)
    snap_dialog(key, "10_dlg_channel_setup")
finally:
    QDialog.exec = orig_exec
    AdvancedDialog.exec = orig_advanced_exec

# --- Startup chooser (its own window; shown last so it cannot block) --------
from respmech.ui.startup_dialog import StartupDialog

dlg = StartupDialog()
dlg.show()
screen_fit.clamp_to_screen(dlg, avail=AVAIL)
pump(0.5)
shoot(dlg, "00_startup")
dlg.close()

win.close()
pump(0.3)
print("done")
