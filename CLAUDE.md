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

## Dev environment — check which interpreter you are actually running

`respmech-gui` is a console script, and on a machine with more than one environment it may
**not** be the repo's `.venv`. On the maintainer's Mac it resolves to
`/opt/anaconda3/bin/respmech-gui`; `.venv/bin/respmech-gui` exists alongside it. Both are
*editable* installs of the same `src/respmech`, so the **code is identical** — but the Qt
version underneath is not, and GUI behaviour follows Qt.

```bash
which respmech-gui
/opt/anaconda3/bin/python3.13 -c "from PySide6.QtCore import qVersion; print(qVersion())"
.venv/bin/python              -c "from PySide6.QtCore import qVersion; print(qVersion())"
```

Before reproducing any GUI report, confirm you are on the interpreter the reporter used. A
repro in the wrong environment yields confident false negatives that look like eliminations.

### Known non-issue: `modalSession has been exited prematurely`

macOS/AppKit prints this on stderr under **Qt 6.11.0**; it is **silent on 6.11.1**. Verified by
a controlled A/B — same code, same flow, same session, only the interpreter swapped. It is an
upstream Qt bug fixed in the patch release, with no functional consequence, and **not** a
RespMech defect. Fix by running `.venv/bin/respmech-gui` or upgrading PySide6 in the other env.
(Packaged builds pin their own PySide6, so end users are unaffected.)

There is a separate, genuine instance of this pattern that *was* ours and is fixed:
`StartupDialog._choose_open` called `accept()` from inside the stack the native macOS open
panel returned into. Opening a native panel from within a Qt modal dialog nests two AppKit
modal sessions; ending them out of order is what produces the message. If you add a native
panel inside a modal, defer the `accept()`/`reject()` by one event-loop turn.

### What the test suite structurally cannot see

Tests run with `QT_QPA_PLATFORM=offscreen` **and** set `AA_DontUseNativeDialogs`
(`tests/unit/conftest.py`). So no native macOS panel is ever opened and no AppKit modal session
is ever created — neither locally nor in CI. Bugs in that class are invisible to all 554 unit
tests by construction; they surface only in a real, native, interactive run.

### macOS has the narrowest font metrics we ship to — layout limits must be checked wider

A pixel budget verified only here is verified on the friendliest platform there is. The same
Preview chips measure ~1.5x wider on the Windows runner, and that is a routine Windows-only
red: the `test_window_fits_screen.py` ceiling read 1005 px on macOS and 1516 px on Windows
from the same code, so adding one checkbox to the ECG strip broke CI while every local run
stayed green. Two habits follow.

- Assert **ratios, not pixel figures**, and state every precondition relative to something
  measured in the same run (`win.minimumSizeHint().width()`, not `860`). A pixel literal in a
  layout test is a measurement of the developer's fonts: the first cut of the guards in
  `test_window_fits_screen.py` hard-coded numbers read off macOS and went red on Windows for
  precisely the reason the guards exist.
- To reproduce the Windows runner locally, widen the horizontal advance and leave the height
  alone: `QFont.setStretch(145)` on the application font lands within a few percent of it
  (modelled 1506-1538 px against CI's measured 1516 px). Scaling the **point size** is the
  wrong instrument -- it inflates row heights too, so it understates the width problem and
  invents height failures Windows does not have.
- Never assert on a QLabel's rendered `text()` when `flow_layout.elide` set it — how much
  survives is a font measurement. Assert on `toolTip()`, which holds the full string by
  contract. This is what made two unrelated EMG tests fail on Windows only.

A row of controls only wraps where a layout can break it. `flow_layout.FlowLayout` makes the
minimum width the widest single ITEM, so a chip built on a plain `QHBoxLayout` is one
unbreakable item whose minimum is still the sum of its contents. Build chips with
`install_flow` + `cluster` so each caption+field pair is its own item. `install_flow` also
sets `QSizePolicy.Preferred` + `setHeightForWidth(True)`: under `Maximum` Qt caps the widget
at its one-line `sizeHint` height and paints the wrapped row outside it.

### Known non-issue: a font-resolution test fails on a minimal Linux sandbox

`tests/unit/test_gui.py::test_splash_resolves_fonts_to_installed_families` fails
reproducibly (not flaky) on a barebones Linux container with only DejaVu/Bitstream/FreeMono
installed: `respmech.ui.splash._resolve_svg_fonts()` falls back to
`QFontDatabase.systemFont(FixedFont).family()`, which on such a container returns the
generic string `"monospace"` — a name `QFontDatabase.families()` itself never lists (it
only enumerates concrete family names). The real CI (`ci.yml`) only runs `tests/unit` on
`windows-latest`/`macos-latest`, both of which have a real `Consolas`/`Menlo` install and
never hit this fallback path, so this is a sandbox-only artefact, not a product defect.
Confirmed 29-07-2026 while baselining a documentation-only change (no Python touched):
553 passed / 1 failed, before and after.

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
  (no CI/packaging-only notes); update both together.
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
