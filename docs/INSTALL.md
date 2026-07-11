# Installing RespMech (v2)

RespMech v2 is an installable Python package with a command-line tool (`respmech`)
and an optional PySide6 desktop GUI.

## Requirements
- Python **3.11+** (the GUI and EMG noise reduction pull extra dependencies).

## Command-line tool

### With pipx (recommended for a CLI)
```bash
pipx install respmech
respmech --version
```

### With pip
```bash
pip install respmech
```

### With Homebrew (macOS/Linux)
```bash
brew install emilwalsted/respmech/respmech
```
(The tap `emilwalsted/homebrew-respmech` ships the formula in
`packaging/homebrew/respmech.rb`. The Homebrew build installs the **CLI only**;
see the GUI note below.)

## Desktop GUI (PySide6)
The GUI depends on Qt (PySide6) and pyqtgraph, installed via the `gui` extra:
```bash
pip install "respmech[gui]"
respmech-gui              # launch the app
respmech-gui my.toml      # launch with a settings file preloaded
```

## Optional features
| Extra | Enables | Install |
|---|---|---|
| `emg` | EMG spectral noise reduction (`librosa`) | `pip install "respmech[emg]"` |
| `plots` | diagnostic PDF plots (`matplotlib`, `seaborn`) | `pip install "respmech[plots]"` |
| `gui` | desktop app (`PySide6`, `pyqtgraph`, `matplotlib`) | `pip install "respmech[gui]"` |

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

The GUI offers the same workflow interactively: **Settings → Preview & tuning
(inspect channels, breath boundaries and a test-run Campbell diagram) → Run**
(live output and status).

## From source (development)
```bash
git clone https://github.com/emilwalsted/respmech
cd respmech
pip install -e ".[dev,gui]"
pytest tests/unit
```
