# Release & distribution (maintainer)

Strategy: **PyPI package + a Homebrew tap** (CLI works out of the box; GUI is an
optional extra). This keeps the CLI trivially installable everywhere and avoids
bundling Qt into the Homebrew build.

## 1. Build
```bash
python -m build            # -> dist/respmech-<ver>.tar.gz + .whl
```
Verify the wheel installs and runs in a clean environment:
```bash
python -m venv /tmp/rmtest && /tmp/rmtest/bin/pip install dist/respmech-*.whl
/tmp/rmtest/bin/respmech --version
```

## 2. Publish to PyPI
```bash
python -m twine upload dist/*
```
Keep the Zenodo DOI workflow for citation (each release gets a DOI).

## 3. Homebrew tap
The tap is a separate repo `emilwalsted/homebrew-respmech` with
`Formula/respmech.rb` (template in this repo at `packaging/homebrew/respmech.rb`).

Per release:
1. Update `url` to the new PyPI sdist and set `sha256` to its checksum.
2. Regenerate the Python dependency `resource` stanzas (numpy/scipy/pandas/…):
   ```bash
   brew update-python-resources Formula/respmech.rb
   ```
3. Test locally:
   ```bash
   brew install --build-from-source Formula/respmech.rb
   brew test respmech
   brew audit --new respmech
   ```
Users then install with `brew install emilwalsted/respmech/respmech`.

### CLI-only vs GUI
The formula installs the **CLI only** (numpy/scipy/pandas as binary wheels — no
compiler needed). Qt/PySide6 is large and not bundled; GUI users run
`pip install "respmech[gui]"`. A signed/notarized `.app` bundle (via BeeWare
Briefcase) can be added later for non-technical clinical users — see PLAN.md §7.

## 4. Versioning
Bump `version` in `pyproject.toml` (single source of truth; `respmech.__version__`
should track it). Tag the release `vX.Y.Z`.
