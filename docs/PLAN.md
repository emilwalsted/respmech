# RespMech — Production refactor plan (Phase 1 proposal)

_Status: **awaiting Emil's approval**. No production refactor or UI code is written
yet. This document proposes the target architecture, a redesigned settings model
with a migration path, preserved CLI/batch use, a Qt UI design, and a homebrew
distribution strategy. Decisions marked **[DECIDE]** need Emil's sign-off._

Companion: [`REVERSE_ENGINEERING.md`](REVERSE_ENGINEERING.md) (current behaviour +
physiological formulas) and [`tests/golden/`](../tests/golden/) (behaviour lock).

---

## 1. Goals & guard-rails

1. **Physiological correctness is non-negotiable.** The refactor is a
   restructuring, not a recalculation. Every change is gated by the golden tests
   (§6) — the calculation core must reproduce the frozen numbers within
   `rtol=1e-9` until a change is *deliberately* made and justified.
2. **Three clean layers**: a pure **calculation core**, a **CLI/batch** front end,
   and a **Qt GUI** — all three sharing one core, none importing the others'
   concerns.
3. **Distributable** (homebrew) and still usable as a batch/command-line tool.
4. **Backwards compatible inputs**: old settings files can be imported/converted.

---

## 2. Target architecture

Move from one 1660-line script to an installable package `respmech/`:

```
respmech/
  core/                     # PURE computation — no I/O side effects, no plotting, no Qt
    model.py                # dataclasses: Recording, Breath, BreathMechanics, WobResult, ...
    io/
      loaders.py            # mat/csv/xls/txt -> Recording (typed), format-agnostic
      writers.py            # Recording/results -> xlsx/csv (engine-agnostic)
    signal/
      trim.py, drift.py, segment.py     # trimming, drift/trend, flow & volume segmentation
    mechanics.py            # timing, volumes, pressures, PTP, VMR, resistance
    wob.py                  # Campbell-diagram WOB (elastic/resistive/expiratory)
    emg.py                  # RMS, integrated EMG, ECG removal, noise reduction
    entropy.py              # sample entropy (vendored, unchanged math)
    pipeline.py             # orchestrates one file / one batch; emits progress events
    settings.py             # typed settings model + validation + defaults
  settingsio/
    schema.py               # versioned settings schema (JSON/TOML)
    migrate.py              # v0 (legacy .py dict) -> v1 converter
  cli/
    __main__.py             # `respmech` command (argparse/typer)
  ui/                       # Qt app (Phase 2)
    app.py, windows/, widgets/, workers.py
  plots/
    diagnostics.py          # matplotlib PDF plots, decoupled from computation
tests/
  golden/                   # existing behaviour lock (Phase 1)
  unit/                     # per-formula unit tests (Phase 2)
pyproject.toml              # packaging, entry points, pinned deps
```

**Key principles**

- **Core emits data, not files.** `pipeline.run()` returns typed results and emits
  **progress events** via a callback/observer. The CLI prints them; the GUI turns
  them into signals. This is what makes "live output/status" and "test runs"
  possible without duplicating logic.
- **Plotting is a consumer of results**, not interleaved with calculation (today a
  WOB calc and a PDF render happen in the same function). This removes the matplotlib
  fragility from the compute path (and fixes latent bug (1)).
- **No import-by-path, no global `sys.excepthook`, no per-breath module reloads.**
- **Deterministic & typed**: `numpy`/`scipy` kept, but wrapped so the SciPy API
  breakage (§4 of the RE doc) is fixed in one place (e.g. `cumulative_trapezoid`,
  keyword `x=` in `simpson`).

---

## 3. Settings redesign

### 3.1 Problems with the current model
Executable-Python settings (a security and reproducibility hazard), a nested dict
that has silently **drifted from the code** (`calcwobfrom` vs `calcwobfromaverage`),
fields the code reads but that are **missing from defaults** (`separateby`,
`column_detect`), machine-specific absolute paths, and 1-based column numbers.

### 3.2 Proposed format — versioned **TOML** (human-editable, safe, diffable)
Rationale: TOML is declarative (no code execution), comment-friendly, git-diffable,
and natural for a GUI to read/write. JSON is the fallback if Emil prefers it.
**[DECIDE] TOML vs JSON** (recommendation: TOML).

```toml
schema_version = 1

[input]
folder = "./input"
files  = "*.mat"
[input.format]
sampling_frequency = 4000
matlab_variant = "mac"            # was matlabfileformat 1/2 -> "windows"/"mac"
decimal = "."
[input.channels]                  # names, not bare numbers; 1-based kept for familiarity
poes = 8; pgas = 9; pdi = 6; volume = 23; flow = 20
emg = [1,2,3,4,5]; entropy = [1,2,3,4,5]

[processing.segmentation]
method = "flow"                   # "flow" | "volume"  (was the undocumented `separateby`)
buffer = 800
# volume-method peak gates only used when method = "volume"
peak = { height = 0.1, distance_s = 0.1, width_s = 0.5 }

[processing.volume]
inverse_flow = false; integrate_from_flow = false; inverse_volume = false
correct_drift = true; correct_trend = false; trend_method = "linear"

[processing.wob]
calc_from = "average"             # "average" | "individual"
avg_resampling_obs = 500

[processing.emg]
rms_window_s = 0.05
remove_ecg = false; detect_channel = 4
remove_noise = false
outlier_rms_sd_limit = 0

[processing.entropy]
epochs = 2; tolerance = 0.1

[[processing.exclude_breaths]]     # filename-keyed, but relative & explicit
file = "Testsubject - Rest.mat"; breaths = [5, 7]

[output]
folder = "./output"
save_average = true; save_breath_by_breath = true; save_processed = false
# diagnostics.* ...
```

### 3.3 Migration path (old → new) — **required, and testable**
`settingsio/migrate.py` provides `respmech migrate old_settings.py -o new.toml`:
1. Load the legacy settings **without executing arbitrary code** — parse the
   `settings = { ... }` literal with `ast.literal_eval` (reject anything that isn't
   a plain literal). This safely handles the existing dict style.
2. Map keys to the new schema, **normalising the known drift**: accept
   `mechanics.calcwobfromaverage` (bool) → `wob.calc_from`; accept
   `mechanics.separateby` → `segmentation.method`; drop dead keys
   (`volumetrendpeakminwidth`) with a warning; translate `matlabfileformat` 1/2 →
   `"windows"/"mac"`.
3. Emit the new TOML plus a **migration report** listing every field moved,
   renamed, defaulted or dropped, so nothing changes silently.
4. **Equivalence test**: run the core with the legacy dict (via a thin compat
   adapter) and with the migrated TOML and assert identical golden output.

---

## 4. Preserved CLI / batch usage

A real console entry point (`respmech`) installed via `pyproject.toml`:

```bash
respmech run       settings.toml                 # batch a folder, as today
respmech run       settings.toml --dry-run       # test run: compute, print summary, no writes
respmech migrate   old_settings.py -o new.toml   # convert legacy settings
respmech validate  settings.toml                 # check settings + input without running
respmech --version
```

- `run` reproduces today's behaviour (glob a folder, process each file, write
  Excel/plots), but with a **non-zero exit code on failure** and structured
  progress to stdout (and `--json` for machine consumption).
- Batching stays first-class: the same core `pipeline.run_batch(settings)` backs
  both CLI and GUI. Scriptability (cron, Make, shell loops) is preserved and
  improved (no editing a Python file to launch a run).

---

## 5. UI plan — **Qt for Python** (decided by Emil)

### 5.1 [DECIDE] PyQt6 vs PySide6 — recommendation: **PySide6**
Same Qt API, so application code is portable between them. The deciding factor is
**licensing for a distributable package**:

| | PySide6 (Qt for Python, official) | PyQt6 (Riverbank) |
|---|---|---|
| License | **LGPL v3** (+ commercial) | **GPL v3** or paid commercial |
| Effect on distribution | App can stay under its own license; LGPL is friendly to a pip/homebrew bundle | GPL is **viral** — shipping under GPL, or buy a commercial license |
| Homebrew / PyPI | `pyside6` wheels published by the Qt Company | `PyQt6` wheels by Riverbank |
| API | `PySide6` (Signal/Slot) | `PyQt6` (pyqtSignal/pyqtSlot) |

RespMech is **already GPL-3.0**, so PyQt6's GPL is *legally* compatible. But
**PySide6 (LGPL) keeps future licensing options open and is the smoother path for
homebrew/PyPI distribution**, which is an explicit goal. Recommendation: **PySide6**,
written against a thin `qtcompat` shim so a later switch is trivial. Final call is
Emil's.

### 5.2 Screens
1. **Settings / batch setup** — a form bound to the settings model: input folder &
   file mask, channel mapping, segmentation, WOB, EMG, entropy, output options.
   Load/save `.toml`; **import legacy `.py`** (runs the migrator, shows the
   migration report). Inline validation (reuses `core.settings` validation).
2. **Source-data preview & tuning** — pick one input file; render its channels
   (flow/volume/pressures/EMG) so the user can **fine-tune settings before the full
   batch**: overlay detected breath boundaries and trim points, the drift/trend
   correction, and a **test-run Campbell diagram**. This is the "see the source
   data + do test runs" requirement.
3. **Run** — live console/log pane + per-file progress bar + status; start/cancel;
   on completion, links to the output folder and a results summary table.

### 5.3 Responsiveness — worker threads (the core requirement)
Computation runs off the GUI thread so the UI never freezes and can stream output:

```
QThread + a QObject worker (moveToThread pattern)
  worker.run()  -> calls core.pipeline.run_batch(settings, progress=emit)
  signals:  progress(file, breath, message)   -> append to log pane / progress bar
            file_done(file, summary)           -> update results table
            failed(file, error)                -> show error, keep going
            finished(summary)                  -> re-enable UI
  cancellation: cooperative flag checked between files/breaths
```

The core's **progress-event callback** (§2) is the seam that makes this clean: the
worker simply forwards core events as Qt signals. **Test runs** reuse the same path
with a "compute-only, don't write" flag and render results in-app instead of to
disk.

### 5.4 Plotting in Qt
- **Interactive preview** (pan/zoom over long recordings): **`pyqtgraph`**
  (fast, Qt-native, handles millions of samples) — best for the source-data preview
  and live signal display.
- **Publication/diagnostic figures** (Campbell diagrams, the existing PDF outputs):
  reuse **matplotlib** via its Qt backend (`FigureCanvasQTAgg`), sharing the exact
  plotting code that also produces the batch PDFs.
- **[DECIDE]** confirm pyqtgraph as the interactive-plot dependency (recommended).

---

## 6. Correctness strategy (how the refactor stays honest)

1. **Golden lock (done, Phase 1)** — `tests/golden/` freezes current numeric output
   across the scenario matrix; the refactored core must reproduce it within
   tolerance. Volume-separation and ECG/noise paths will be locked against **Emil's
   real production dataset + expected results** (offered — see §8).
2. **Unit tests per formula (Phase 2)** — small analytic inputs with hand-derived
   expected values for WOB (triangle areas, known integrals), PTP, VE/VT/timing,
   sample entropy (reference vectors), RMS — so each formula is independently
   pinned, not just the end-to-end numbers.
3. **Legacy-equivalence test (Phase 2)** — migrated TOML ≡ legacy dict output.
4. **Deliberate-fix protocol** — when a latent bug (RE doc §6) is fixed, the golden
   reference is regenerated **in the same commit** with the numeric diff explained.

---

## 7. Distribution to homebrew

**Recommended: PyPI package + a Homebrew tap (formula), with the Qt GUI as an
optional extra.**

1. **Package** with `pyproject.toml` (build backend `hatchling`/`setuptools`),
   console entry point `respmech`, GUI extra `respmech[gui]` (pulls PySide6 +
   pyqtgraph). Vendored `entropy.py` retained with its license.
2. **Publish to PyPI** (`respmech`), versioned (keep Zenodo DOI workflow for
   citation).
3. **Homebrew tap** `emilwalsted/homebrew-respmech` with a formula that installs
   into a `libexec` virtualenv (the standard `Homebrew::Language::Python`/
   `virtualenv_install_with_resources` pattern) and links the `respmech` CLI.
   - CLI works out of the box. For the GUI, the formula can `depends_on` Qt or ship
     the PySide6 wheel as a resource; **[DECIDE]** whether the homebrew artifact is
     CLI-only (GUI via `pip install respmech[gui]`) or bundles the GUI.
4. **Alternative considered — BeeWare Briefcase**: produces a signed native `.app`/
   `.dmg` (nicer for non-technical clinicians), but is a heavier toolchain and less
   natural for a CLI. Recommendation: **PyPI + tap now**; optionally add a Briefcase
   `.dmg` later for GUI-only clinical users. **[DECIDE]**.
5. macOS specifics: the GUI needs a code-signed/notarized bundle for smooth install
   outside Homebrew; the CLI does not. SciPy/librosa wheels cover Apple Silicon.

---

## 8. Open decisions & what's needed from Emil

**Decisions [DECIDE]:** (1) PySide6 vs PyQt6 — _rec. PySide6_; (2) settings format
TOML vs JSON — _rec. TOML_; (3) pyqtgraph for interactive plots — _rec. yes_;
(4) homebrew artifact CLI-only vs GUI-bundled; (5) add a Briefcase `.dmg` later?

**Inputs needed:**
- ✅ **UI framework** — decided: Qt (PySide6/PyQt6 to finalise above).
- ⏳ **Real production dataset + expected results** — Emil has offered this; it is
  what lets us lock correctness of the **volume-separation** and **ECG/noise-removal**
  paths (not coverable by synthetic data) and validate against clinically
  meaningful numbers. Please share the input files + their settings + the expected
  output spreadsheets; they will be added as a second golden reference.

---

## 9. Phased delivery

- **Phase 1 (this branch, for approval)** — reverse-engineering doc, golden tests,
  this plan. *No production code changes.*
- **Phase 2** — extract the pure calculation core behind the golden tests; add the
  SciPy-compat fixes and latent-bug fixes (each with a justified golden update);
  new settings model + migrator; `respmech` CLI. Core reaches parity.
- **Phase 3** — PySide6 GUI (settings, preview/tuning, live run) on the Phase 2
  core; packaging + homebrew tap; docs refresh (the README is currently stale).
