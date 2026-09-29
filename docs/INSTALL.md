# Installing RespMech (v2)

RespMech v2 ships as a **desktop installer** (which bundles its own Python) and as an
installable Python package with a command-line tool (`respmech`) and a PySide6 desktop
GUI (`respmech-gui`).

## Desktop app (recommended)

Download the installer for your platform from the
[latest release](https://github.com/emilwalsted/respmech/releases):

| Platform | File | Notes |
|---|---|---|
| macOS | `RespMech-<version>.dmg` | signed + notarised; drag to Applications. Apple silicon, macOS 12+ |
| Windows | `RespMech-<version>.msi` | signed |

Nothing else to install — the bundle carries its own Python and dependencies.

## Python package (PyPI)

Requires Python **3.11+**. The desktop installers above are built only for macOS and
Windows; on Linux, install the Python package below.

```bash
pip install respmech               # CLI + core analysis engine
pip install "respmech[gui]"        # + the PySide6 desktop app (respmech-gui)
pip install "respmech[gui,emg]"    # + spectral EMG noise reduction (librosa)
```

The base install gives you the `respmech` command-line tool and the analysis core (no Qt,
no heavyweight audio stack). Add extras for more:

| Extra | Adds |
|---|---|
| `gui` | desktop app — `PySide6`, `pyqtgraph`, `matplotlib` |
| `emg` | EMG spectral noise reduction — `librosa` |
| `plots` | diagnostic PDF figures — `matplotlib` |

```bash
respmech --version
respmech --help
respmech-gui                       # needs the [gui] extra
```

## From source

Requires Python **3.11+**.

```bash
git clone https://github.com/emilwalsted/respmech
cd respmech
pip install -e ".[dev,gui]"     # gui = desktop app; dev = the test stack

respmech-gui                    # launch the app
respmech-gui my.toml            # launch with an analysis preloaded
respmech --help                 # the command-line tool
pytest tests/unit               # run the tests
```

## Optional features

Install extras with `pip install -e ".[<extra>]"` from a source checkout:

| Extra | Enables |
|---|---|
| `emg` | EMG spectral noise reduction (`librosa`) |
| `plots` | diagnostic PDF figures (`matplotlib`) |
| `gui` | desktop app (`PySide6`, `pyqtgraph`, `matplotlib`) |
| `dev` | the test stack |

## Publishing

- **PyPI** — automated. Every `vX.Y.Z` tag publishes `respmech` to PyPI via GitHub Actions
  **Trusted Publishing** (OIDC; no stored tokens) — see
  [`.github/workflows/publish-pypi.yml`](../.github/workflows/publish-pypi.yml) and
  [RELEASING.md](RELEASING.md). `pip install respmech` works from the first published
  release onward.
- **Homebrew** (`brew install emilwalsted/respmech/respmech`): the formula exists in
  `packaging/homebrew/respmech.rb`, but the tap `emilwalsted/homebrew-respmech` is **not
  yet published** — its url/sha256 are placeholders pending a release. Use PyPI, the
  installer, or a source checkout for now.

## Analysis files and older versions

An analysis (`.toml`) that uses typed breaths (`[[processing.breath_types]]`), reference
manoeuvres (`[[processing.references]]`, `[[processing.reference_defaults]]`), subjects
(`[[input.subjects]]`), separators (`[[processing.segmentation.separators]]` or
`[[processing.segmentation.overrides]]`) or any of the newer `processing.*` tables (lung
volumes, MFVL, PEEPi, pressure normalisation, EMG segmentation) **must not be opened and
saved by RespMech older than 2.5.** A version that does not know a table cannot use it, and
before 2.5 it also dropped the table when the file was saved, so those settings are lost for
good. From 2.5 on, RespMech keeps every setting it does not recognise when it saves, so a
file written by a newer version survives being opened and saved by 2.5 or later; the older
versions cannot be fixed after the fact. If several people or installations share analysis
files, update them together, and keep a copy of the file before opening it in an older
version. The `analysis-used.toml` written next to every run's results records the exact
settings used, and is safe to keep as that copy.

## Usage

```bash
# Convert an old (v1) settings .py to the new TOML format (prints a migration report)
respmech migrate old_settings.py -o settings.toml

# Check settings + that input files are found
respmech validate settings.toml

# Process the batch (writes Excel/CSV to the output folder's data/ subfolder)
respmech run settings.toml

# Compute without writing anything (quick check / tuning)
respmech run settings.toml --dry-run
```

The GUI offers the same workflow interactively, across two tabs: **Setup** (recordings,
channel mapping, what to save) and **Preview & QC** (breath segmentation, Campbell loops,
and dedicated subtabs to tune EMG ECG-reduction and noise-reduction against the live
signal). **Run & results** (batch progress and per-file status) is a drawer under Preview
& QC's file list, not a separate screen — it opens itself the moment a run starts.
