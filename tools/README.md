# tools/

## Refreshing the manual's screenshots

`capture_screens.py` renders every screen and dialog with the bundled sample
analysis loaded and writes one PNG per screen/dialog to `RM_OUT` (see the
script's own docstring for the mechanism). This is the fixed recipe for
turning that output into the eleven screenshots the website actually uses,
so a future refresh does not have to be reverse-engineered from the images
themselves. It replaces no record of how the *currently published* images
were made — that was never written down and is not being reconstructed;
this is the recipe for every refresh from here on.

**Runner:** `windows-latest`, via the `Screenshots` GitHub Actions workflow
(`.github/workflows/screenshots.yml`, `workflow_dispatch`). Windows renders
on the native platform plugin (`QT_QPA_PLATFORM=windows` in that workflow),
which gives real DirectWrite font metrics — the platform the rest of this
app's layout is already modelled against (see this file's own font-stretch
notes in the repo-root `CLAUDE.md`). Don't substitute the macOS leg for the
site's screenshots; it exists for spot-checking layout on that platform, not
for producing published images.

**Theme:** light only. The website shows exclusively light-theme screenshots
(measured: all published images sit at mid-luminance 237-242, consistent
with the light palette); ignore the workflow's dark-theme output for this
purpose.

**Neutral environment, set before capture:**
- `TMPDIR` (Linux/macOS runners) / `TEMP` and `TMP` (Windows): point at a
  short, fixed, non-machine-specific path (e.g. `C:\rmshots` on Windows)
  before running the sample analysis, so a path the UI happens to display
  (the sample's working folder, a "Save as..." suggestion) never bakes a
  runner-specific absolute path into a published image.
- Locale: force English (`LC_ALL=en_US.UTF-8`/`LANG=en_US.UTF-8` on
  Linux/macOS; on Windows the workflow already sets `PYTHONIOENCODING=utf-8`
  for the UI's `·`/`–`/`›` glyphs — leave the runner's own locale, which
  `actions/setup-python` on `windows-latest` provides as English, at its
  default rather than overriding it to something else).

**Cropping and renaming.** `capture_screens.py` writes eleven raw files
(`00_startup` … `10_dlg_channel_setup`); the site uses ten of them, renamed,
plus one manual crop:

| Raw file | Site file | Note |
|---|---|---|
| `01_setup.png` | `setup.png` | |
| `01_setup.png` | `setup-output.png` | manual crop to the Output card, 615x563 |
| `02_preview_mechanics.png` | `preview-mechanics.png` | |
| `03_preview_ecg.png` | `preview-ecg.png` | |
| `04_preview_noise.png` | `preview-noise.png` | |
| `05_run_results.png` | `run.png` | |
| `06_dlg_mech_advanced.png` | `advanced-mech.png` | |
| `07_dlg_emg_advanced.png` | `advanced-emg.png` | |
| `00_startup.png` | `startup.png` | |
| `10_dlg_channel_setup.png` | `channels.png` | |

`08_dlg_ecg_advanced.png` and `09_dlg_noise_profile.png` are captured but not
used anywhere on the site; ignore them. Save the renamed files to
`respmech-website`'s `assets/img/docs/`.

The site's three `index.html` screenshots (`setup.png`, `preview-mechanics.png`,
`run.png`, at 1400x880 rather than this recipe's 1280x800) are a separate
family: they come from the app repo's own `scripts/gen_readme_figures.py`
(the README figures), copied from `docs/img/`, not from this script — do not
substitute a `capture_screens.py` shot for them.

After copying the renamed files into `respmech-website`: run
`python3 tools/generate-image-variants.py` (AVIF/WebP) and
`python3 tools/version-assets.py` (cache-bust hashes) there, and update any
`width`/`height` attribute that changed.
