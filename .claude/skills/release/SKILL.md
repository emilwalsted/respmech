---
name: release
description: Releasing RespMech — CHANGELOG entry, version bump, v* tag, the installer/PyPI workflows, macOS notarisation and Certum MSI signing, and the respmech.dk version/changelog sync. Use for any release, changelog, signing or website-sync task.
---

# Releasing RespMech

Step-by-step runbook: `docs/RELEASING.md`. Signing detail: `docs/SIGNING.md`.
What follows is the project memory that sits above both.

## Releases (`.github/workflows/release.yml` = "Build installers")

- Trigger: push a `v*` tag (or manual dispatch). Builds a Windows **MSI** and a
  macOS **dmg** with briefcase, then (on a tag) the `publish-release` job creates
  a GitHub **release** (marked **Latest**) with the installers attached.
- macOS signing is **secret-gated** (Developer ID + notarisation when the Apple
  secrets are present, else ad-hoc). The Windows MSI is built unsigned and
  **Certum-signed locally** after release (`scripts/sign-msi-certum.sh`); see
  `docs/SIGNING.md`.
- Releases are full releases (the newest is marked **Latest**; `release.yml` passes
  `--latest`). Keep tags clean semver `vX.Y.Z` (no `-rc/-beta`) — the website picks the
  version that way.
- **`CHANGELOG.md`** (repo root, added 29-07-2026) is the canonical, complete release
  log — one section per release, newest first. Add its entry as **step 1** of every
  release (see `docs/RELEASING.md`), before bumping the version. `respmech-website`'s
  `changelog.html` mirrors it in a version trimmed to what an app user cares about
  (no CI/packaging-only notes); update both together. Since 30-07-2026 the website
  side is automatic: on a release its workflow *promotes* the hand-written "Coming
  next" section into `vX.Y.Z` and takes only the lead sentence from the entry here.
  So keep "Coming next" on respmech.dk current as you merge; since 03-08-2026 the
  deploy publishes the page but never mails anyone, and the subscriber list is
  mailed only by the website repo's manual **Announce a release** workflow
  (`announce-release.yml`); the mailing-list e-mail is built from that very
  section, and a missing one used to mean subscribers silently got nothing.
- **`tools/check_changelog.py`** (added 30-07-2026) answers "is the entry
  exhaustive?" with evidence instead of memory. It walks the commits in the range,
  sets aside the ones touching only tests/docs/CI/tooling, and prints every
  user-visible change beside the bullet that best matches it, weakest first. It
  fails on the one thing a word comparison can be certain of: a change with **no**
  trace in the entry. Weaker matches are a worksheet, not a verdict — three
  successively cleverer rules were tried and each was measurably foolable on the
  same data, which is documented in the tool and pinned by
  `tests/unit/test_check_changelog.py`. A deliberate omission is recorded with
  `<!-- changelog-skip <sha7> <reason> -->`, never merely silenced. Hard gate on the
  tag in `publish-pypi.yml`; informational worksheet on every push in `ci.yml`.
- **`## Unreleased`** (added 29-07-2026) is a hand-maintained draft sitting above the
  latest dated release, describing everything since the last tag. It is updated only
  when explicitly asked to, never automatically per commit. At release time (step 1
  above), fold it into the new dated entry and collapse it back to an HTML-commented,
  empty placeholder — see the mechanism documented directly in `CHANGELOG.md` and in
  `docs/RELEASING.md`.

## Website (respmech.dk)

The marketing/info site lives in the **private** repo
`emilwalsted/respmech-website` and deploys to https://www.respmech.dk. It names
the current version on its download button / labels by resolving the **highest
clean-semver, non-draft release** of this repo (it also counts any pre-releases, so it
keeps working regardless of the release/pre-release flag).

**Release → website hook.** The last step of `publish-release`
("Notify respmech.dk…") sends a `repository_dispatch` (`event_type:
respmech-release`) to the website repo, which then refreshes its version and
redeploys — so the site updates within seconds of a release.

- Requires the secret **`WEBSITE_DISPATCH_TOKEN`** here: a fine-grained PAT with
  **Contents: write** on `emilwalsted/respmech-website`. If it is absent the step
  is skipped and the release/build is unaffected.
- Without the hook, the website's daily poll still catches a new release within a
  day. Setup helper + full docs: `deploy/setup.sh` and `deploy/README.md` in the
  website repo.

> When changing `release.yml`, keep the notify step **after** `gh release create`
> and non-fatal (guarded on the token) so it can never break an installer build.
