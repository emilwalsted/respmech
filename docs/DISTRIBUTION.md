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

## 2. Publish to PyPI — automated, no tokens

Publishing is done by GitHub Actions via **Trusted Publishing (OIDC)**, not by hand.
Pushing a `vX.Y.Z` tag runs [`.github/workflows/publish-pypi.yml`](../.github/workflows/publish-pypi.yml):
test gate → build sdist+wheel → publish to PyPI. There is **no API token** stored anywhere;
PyPI mints a short-lived credential only for a run whose identity matches the pending
publisher you configure once (owner `emilwalsted`, repo `respmech`, workflow
`publish-pypi.yml`, environment `pypi`). The full runbook is in [RELEASING.md](RELEASING.md).

`python -m twine upload dist/*` is **not** the path anymore — do not upload by hand (it
would need a long-lived token, which we deliberately avoid on a public repo). For a safe
dry-run, trigger the workflow manually (Actions ▸ Publish to PyPI ▸ Run workflow) to publish
to **TestPyPI**.

Keep the Zenodo DOI workflow for citation (each GitHub release gets a DOI).

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
`src/respmech/__init__.py`'s `__version__` is the **single source of truth** — the PyPI
package version is derived from it (hatchling dynamic version). Bump it, and keep the
manual `[tool.briefcase] version` in `pyproject.toml` in sync (briefcase can't read the
dynamic version). Then tag `vX.Y.Z` matching `__version__`. The publish workflow fails the
build if the tag and the package version disagree. See [RELEASING.md](RELEASING.md).
