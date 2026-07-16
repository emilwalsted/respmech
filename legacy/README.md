# Legacy v1 — frozen reference implementation

**Do not modify, refactor or "clean up" the files in this folder.** They are a frozen
snapshot of RespMech v1.0.0, kept deliberately. This is not dead code.

| File | What it is |
|---|---|
| `respmech.py` | The original single-file analysis monolith (load → correct → segment → mechanics/WOB/entropy → plots/output). |
| `emg.py` | v1 EMG processing (RMS, ECG removal, spectral noise reduction). Loaded by `respmech.py` by path, relative to its own directory. |
| `entropy.py` | v1 sample entropy (vendored pyEntropy — see `../LICENSE pyentrp`). |
| `example.py` | The v1 usage template (a flat settings dict passed to `analyse()`). Inert: its hard-coded paths point at the original author's machine. |

## Why it is still here

It is the **pre-refactor oracle** for the production golden tests. The 2.x package in
`src/respmech` is a port of this code, and the production goldens exist to prove the port
is numerically faithful. The chain is:

```
tests/golden/build_production_golden.py   (documented in tests/golden/README.md)
        └─> tests/golden/prod_runner.py   -> loads legacy/respmech.py -> rm.analyse(settings)
                └─> production_golden.json / production_manifest.json   (the baked reference)
```

The golden **tests** (`test_production_golden.py`, `test_production_emg_golden.py`) do *not*
need these files — they compare the current engine against the already-baked JSON. The oracle
is only needed to **re-bake** that reference from the original implementation, e.g. when adding
a scenario. Running it needs the old-SciPy environment described in `tests/golden/README.md`.

## Relationship to the 2.x package

Nothing in `src/respmech`, the unit tests, CI or the installers imports this folder — the
shipped package is `src/respmech` only (`pyproject.toml`: `packages = ["src/respmech"]`).

These files previously sat in the repository root, where `respmech.py` **shadowed** the
`respmech` package for any Python process started with the repo root as its working directory
(`import respmech` picked up the monolith instead of the package). Moving them here removes
that trap while keeping the oracle.

The port itself — every formula, unit and deliberate deviation — is documented in
[`../docs/REVERSE_ENGINEERING.md`](../docs/REVERSE_ENGINEERING.md), which is written against
exactly this code and is what `src/respmech/settingsio/migrate.py` maps onto the 2.x settings.
