# Installing RespMech (v2)

RespMech v2 ships as a **desktop installer** (which bundles its own Python) and as an
installable Python package with a command-line tool (`respmech`) and a PySide6 desktop
GUI (`respmech-gui`).

## Desktop app (recommended)

Download the installer for your platform from the
[latest release](https://github.com/emilwalsted/respmech/releases):

| Platform | File | Notes |
|---|---|---|
| macOS | `RespMech-<version>.dmg` | signed + notarised; drag to Applications |
| Windows | `RespMech-<version>.msi` | signed |

Nothing else to install — the bundle carries its own Python and dependencies.

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
| `plots` | diagnostic PDF figures (`matplotlib`, `seaborn`) |
| `gui` | desktop app (`PySide6`, `pyqtgraph`, `matplotlib`) |
| `dev` | the test stack |

## Not yet published

These distribution channels are prepared but **not live** — the commands below do not
work yet:

- **PyPI** (`pipx install respmech` / `pip install respmech`): the package is not
  published; there is no publish step in the release workflow.
- **Homebrew** (`brew install emilwalsted/respmech/respmech`): the formula exists in
  `packaging/homebrew/respmech.rb`, but the tap `emilwalsted/homebrew-respmech` is not
  published. The formula's url/sha256 are placeholders pending a release.

Until then, use the installer or a source checkout.

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

The GUI offers the same workflow interactively: **Setup** (recordings, channel mapping,
what to save) → **Preview & QC** (breath segmentation, Campbell loops, and dedicated tabs
to tune EMG ECG-reduction and noise-reduction against the live signal) → **Run & results**
(batch progress and per-file status).
