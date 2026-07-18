# Releasing RespMech (maintainer)

One tag cuts everything: pushing `vX.Y.Z` triggers **both** workflows —
[`release.yml`](../.github/workflows/release.yml) builds the signed macOS dmg + Windows MSI
and attaches them to a GitHub pre-release, and
[`publish-pypi.yml`](../.github/workflows/publish-pypi.yml) publishes the `respmech` package
to PyPI via **Trusted Publishing (OIDC)** — no tokens anywhere.

## One-time setup (once per project, no secrets)

On **PyPI** ▸ your account ▸ **Publishing** ▸ *Add a pending publisher*:

| Field | Value |
|---|---|
| PyPI Project Name | `respmech` |
| Owner | `emilwalsted` |
| Repository name | `respmech` |
| Workflow name | `publish-pypi.yml` |
| Environment name | `pypi` |

That is the entire credential setup — PyPI will trust exactly this repo's `publish-pypi.yml`
running in the `pypi` environment, and nothing else. (Optional: repeat on
**test.pypi.org** with environment `testpypi` to enable the dry-run path.)

Optionally, in GitHub ▸ Settings ▸ Environments, add protection rules to the `pypi`
environment (e.g. required reviewer, or restrict to tags) for a manual gate before publish.

## Per release

1. **Bump the version** — `src/respmech/__init__.py`'s `__version__` is the single source
   of truth (the PyPI version derives from it). Keep `[tool.briefcase] version` in
   `pyproject.toml` in sync (briefcase can't read the dynamic version):

   ```bash
   # e.g. 2.2.0 -> 2.3.0  (edit both, they must match the tag)
   #   src/respmech/__init__.py :  __version__ = "2.3.0"
   #   pyproject.toml           :  [tool.briefcase] version = "2.3.0"
   ```

2. **Verify locally** (optional but recommended):

   ```bash
   python -m pytest tests/unit tests/golden/test_golden.py -q
   python -m build && python -m twine check dist/*
   ```

3. **Commit and land on `master`** — the workflows must exist at the tagged commit, so the
   tag has to point at `master` (as the signing pipeline already requires).

4. **Tag and push:**

   ```bash
   git tag v2.3.0            # must equal __version__
   git push origin v2.3.0
   ```

That is it. CI then: runs the test gate → builds sdist+wheel → checks the built version
equals the tag → publishes to PyPI; and separately builds and attaches the installers to a
GitHub pre-release. (The Windows MSI is Authenticode-signed locally afterwards — see
[SIGNING.md](SIGNING.md).)

## Dry-run to TestPyPI

Before a real release, GitHub ▸ Actions ▸ **Publish to PyPI** ▸ *Run workflow* publishes the
current build to **TestPyPI** (environment `testpypi`), never to real PyPI. Verify with:

```bash
pip install -i https://test.pypi.org/simple/ respmech
```

## Notes

- **A PyPI version is immutable** — you cannot re-upload `X.Y.Z`. If a release is broken,
  bump the patch (e.g. `2.3.1`) and tag again; never try to overwrite.
- **No tokens.** Trusted Publishing means there is nothing to rotate or leak. Do not add a
  `PYPI_API_TOKEN` secret or a `twine upload` step.
- **Zenodo DOI** — each GitHub release still gets its own citation DOI; keep referencing the
  latest in the README/citation.
