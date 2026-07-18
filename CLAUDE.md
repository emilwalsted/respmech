# CLAUDE.md — RespMech

Project memory for **RespMech** — respiratory mechanics, work of breathing and
diaphragm-EMG analysis. Public repo `emilwalsted/respmech` (GPL-3.0-or-later).
Author: Emil Ingerslev Walsted. See `README.md` and `docs/` for the full picture;
this file is the quick orientation plus the release/website setup.

## What it is

Analyses time-series respiratory recordings (LabChart/CSV/Excel/MATLAB) **breath
by breath** and computes: respiratory mechanics (timing, VT, VE, oesophageal/
gastric/transdiaphragmatic pressures, PTP), **work of breathing** (Campbell
diagram, J and J·min⁻¹), **diaphragm EMG** (RMS + integrated, optional ECG
removal + spectral noise reduction) and **sample entropy**. v2 is a **PySide6
desktop app** (`respmech-gui`) with Setup → Preview & QC → Run screens, plus a
CLI (`respmech run/validate/migrate`). Settings are declarative **TOML**.

## Layout / correctness

- `src/respmech/` — the v2 package (`core/` compute+IO, `ui/` GUI, `cli/`,
  `settingsio/` TOML + v1 migration). `pyproject.toml`: version, extras
  (`gui`/`emg`/`plots`/`dev`/`packaging`), briefcase config.
- `legacy/` — the **frozen v1 monolith**; the v2 engine is a faithful port of it.
- `tests/golden/` — characterisation tests that pin v2 output **byte-for-byte**
  against v1 references. `docs/REVERSE_ENGINEERING.md` = the formulas/units.
- CI: `.github/workflows/ci.yml` (GUI smoke on win/mac + numerical golden on
  ubuntu), runs on every branch.

## Releases (`.github/workflows/release.yml` = "Build installers")

- Trigger: push a `v*` tag (or manual dispatch). Builds a Windows **MSI** and a
  macOS **dmg** with briefcase, then (on a tag) the `publish-release` job creates
  a GitHub **pre-release** with the installers attached.
- macOS signing is **secret-gated** (Developer ID + notarisation when the Apple
  secrets are present, else ad-hoc). The Windows MSI is built unsigned and
  **Certum-signed locally** after release (`scripts/sign-msi-certum.sh`); see
  `docs/SIGNING.md`.
- Releases are marked **pre-release** ("installers for testers"). Keep tags clean
  semver `vX.Y.Z` (no `-rc/-beta`) — the website picks the version that way.

## Website (respmech.dk)

The marketing/info site lives in the **private** repo
`emilwalsted/respmech-website` and deploys to https://www.respmech.dk. It names
the current version on its download button / labels by resolving the **highest
clean-semver, non-draft release** of this repo (so the pre-release flag is fine).

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
