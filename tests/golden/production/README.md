# Production golden data (drop zone)

This is where a **real** RespMech dataset goes so we can lock correctness against
clinically meaningful numbers — and cover the paths the synthetic golden cannot:
**volume-based breath separation** and **ECG removal / EMG noise reduction**.

Nothing here is used until the data is present (the production golden is built in a
follow-up once Emil provides a representative set).

## What to drop here

```
tests/golden/production/
  input/            # the raw recording(s): .mat / .csv / .xlsx / .txt
  settings.py       # the settings file actually used for this dataset
                    #   (or settings.json — the existing dict, as-is)
  expected/         # the output RespMech produced for this run:
                    #   Average breathdata.xlsx, <file>.breathdata.xlsx,
                    #   and any "<file> – Processed data.csv"
```

Alternatively, point me at a location (e.g. a Dropbox link) and I'll place it.

## Notes

- A small but representative set is ideal: enough breaths to be meaningful, ideally
  one file that exercises **volume separation** and one that uses **ECG/noise
  removal**, so those paths get locked.
- If any of the raw recordings are large or sensitive, say so — we can keep them out
  of git (store hashes + expected outputs) or use a redacted/sample subset.
- Once the data is here, a `production` golden will be added alongside the synthetic
  one: it re-runs the current code, compares against your `expected/` spreadsheets
  (documented tolerance), and then guards the refactor.
