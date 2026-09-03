# Releasing RespMech (maintainer)

One tag cuts everything: pushing `vX.Y.Z` triggers **both** workflows —
[`release.yml`](../.github/workflows/release.yml) builds the signed macOS dmg + Windows MSI
and attaches them to a GitHub release (marked Latest), and
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

1. **Add a [CHANGELOG.md](../CHANGELOG.md) entry** — a new `## vX.Y.Z — YYYY-MM-DD` section
   at the top, above the previous release, with a short lead sentence and bullet points for
   the user-visible changes (`git log <previous-tag>..HEAD --oneline` is the source list; write
   it for a reader, not a commit dump). This is also what respmech.dk's changelog page
   reproduces, so keep it accurate and free of internal-only detail.
   - If an **Unreleased** section already sits above the previous release, it is a
     hand-maintained draft of exactly this list (updated only when explicitly asked, not on
     every commit). Fold its bullets into the new dated entry — double-check nothing landed
     since its last update and nothing in it has gone stale — rather than re-deriving the
     list from scratch. Afterwards, collapse the section back to its empty placeholder and
     wrap it in an HTML comment (see the comment already in CHANGELOG.md) so it stays
     invisible until asked for again.
2. **Bump the version** — `src/respmech/__init__.py`'s `__version__` is the single source
   of truth (the PyPI version derives from it). Keep `[tool.briefcase] version` in
   `pyproject.toml` in sync (briefcase can't read the dynamic version):

   ```bash
   # e.g. 2.2.0 -> 2.3.0  (edit both, they must match the tag)
   #   src/respmech/__init__.py :  __version__ = "2.3.0"
   #   pyproject.toml           :  [tool.briefcase] version = "2.3.0"
   ```

3. **Check the entry actually covers the release:**

   ```bash
   python3 tools/check_changelog.py --version vX.Y.Z   # after tagging
   python3 tools/check_changelog.py                    # before, against Unreleased
   ```

   It walks the commits in the range, sets aside the ones that only touch tests, docs,
   CI or tooling, and reports every user-visible change together with the bullet that
   best matches it, weakest match first. It **fails** on the one thing a word
   comparison can be certain of: a change with no trace in the entry at all. Weaker
   matches are a worksheet for you to read, not a verdict — the tool says so itself,
   and the reasoning is in its docstring. A deliberate omission is recorded rather
   than silenced, with `<!-- changelog-skip <sha7> <reason> -->` in CHANGELOG.md.

   `publish-pypi.yml` runs the same check on the tag, so an incomplete entry stops the
   PyPI publish instead of shipping quietly. `ci.yml` prints the worksheet on every
   push, which is the cheap moment to keep the entry honest.

4. **The website's changelog page updates itself.** Nothing to do here, but worth
   knowing why it matters: respmech.dk's changelog page carries a hand-written
   "Coming next" section, and on a release the website's workflow *promotes* that
   section into `vX.Y.Z` (`tools/promote-changelog.py` over there), leaving a fresh
   empty one behind. It promotes rather than generates because that page is the
   reader-facing rewrite of these bullets, in the site's own voice, and no script
   should try to reproduce that from Markdown. It takes only the lead sentence from
   the entry you wrote in step 1.
   The reason it is automated at all: the site's `api/notify.php` builds the
   release-notification e-mail from that section, matched by `id="vX-Y-Z"`, and
   answers 404 without it. So a missing section used to mean the mailing list
   silently got nothing. Keep "Coming next" on respmech.dk up to date as you merge;
   the promotion is automatic, the announcement is not: only **Actions ▸ Announce a
   release** in the website repo mails the list, and you run it by hand (see step 5).

5. **Check the page says all of it** — from the website clone, beside this one:

   ```bash
   cd ../respmech-website
   python3 tools/check-coverage.py --version vX.Y.Z --changelog ../respmech/CHANGELOG.md
   ```

   Step 3 asks whether the entry covers the commits. This asks whether the *page*
   covers the entry, which is a different question with the same failure mode: v2.3.3
   went out with nine entries in CHANGELOG.md and six bullets on the page, because two
   fixes were written here while the release was being cut and never over there. The
   e-mail is built from the page, so those two were never announced.

   Before the tag it reads "Coming next", which is the moment when fixing it costs
   nothing. The verdict is a count, not a text comparison: matching the two by wording
   was measured and does not work, because the page deliberately says the same thing
   in different words. An entry that does not belong on the public page says so next
   to itself, and the reason becomes part of the record:

   ```markdown
   - Removed an unnecessary import of the compute core just to open a window
     <!-- site: 0 internal "Removed an unnecessary import" no visible difference -->
   ```

   The website's deploy workflow runs the same check after promoting. It cannot fail
   the deploy, and it sends nothing: since 03-08-2026 the deploy does not call
   `notify.php` at all, and the only thing that mails the subscribers is
   **Actions ▸ Announce a release** in the website repo, run by hand after ticking
   its confirmation box (that you have previewed the mail). On a shortfall the deploy
   writes a **Not ready to announce** verdict into its run summary with the recipe,
   and the Announce workflow runs the same check as a hard gate and refuses to send
   while it finds one (an inconclusive check — API down, no `CHANGELOG.md` — does not
   block), because `notify.php` sends once per version and a thin mail cannot be
   taken back.

   To read the announcement itself before it goes out, `php tools/preview-mail.php`
   in the website repo renders it to a file, and **Actions ▸ Send a test release
   e-mail** sends the real thing to one allowlisted address without touching the
   subscriber list or spending the release's one announcement.

6. **Verify locally** (optional but recommended):

   ```bash
   python -m pytest tests/unit tests/golden/test_golden.py -q
   python -m build && python -m twine check dist/*
   ```

7. **Commit and land on `master`** — the workflows must exist at the tagged commit, so the
   tag has to point at `master` (as the signing pipeline already requires).

8. **Tag and push:**

   ```bash
   git tag v2.3.0            # must equal __version__
   git push origin v2.3.0
   ```

That is it. CI then: runs the test gate → builds sdist+wheel → checks the built version
equals the tag → publishes to PyPI; and separately builds and attaches the installers to a
GitHub release. (The Windows MSI is Authenticode-signed locally afterwards — see
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
