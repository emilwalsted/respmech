"""Profile a full RespMech batch on a large synthetic recording.

    python tools/profile_large_recording.py <minutes> <fs> <workdir> [noplots]

Scales the bundled sample generator (core/sample.py) to ``minutes`` of breathing at
``fs`` Hz with every signal (flow, volume, Poes, Pgas, Pdi and three EMG channels),
runs ``run_batch`` and ``write_batch`` with the sample's own settings (ECG removal and
noise reduction on, all diagnostic figures unless ``noplots``) under cProfile, and
writes to ``workdir``:

* ``profile.prof`` / ``write.prof`` -- the two profiles (load with ``pstats``);
* ``top.txt`` -- the hottest functions of the compute phase;
* ``hashes.txt`` -- a SHA-256 per output table (CSV/TXT/JSON files, every xlsx sheet
  read back at full float precision, and each in-memory result table), so two runs
  can be compared for bit-identity with ``diff``.

The input CSV is reused if it already exists in ``workdir/input``. Figures are drawn
in a separate process (core/io/_figure_process.py), which cProfile does not see; their
cost shows up only as wall time (the output files' timestamps give it per figure).
The main guard below is required: that process is spawned and re-imports this module.
See docs/PROFILING.md for the measured numbers.
"""
import cProfile
import hashlib
import io
import os
import pstats
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")


def main():
    minutes = float(sys.argv[1])
    fs = int(sys.argv[2])
    work = sys.argv[3]
    noplots = len(sys.argv) > 4 and sys.argv[4] == "noplots"

    from respmech.core import sample

    sample.FS = fs
    sample.N_BREATHS = max(9, int(minutes * 60 / sample.PERIOD_S))

    inp = os.path.join(work, "input")
    out = os.path.join(work, "output")
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()
    if not os.path.exists(os.path.join(inp, sample.FILENAME)):
        desc = sample.write_sample_recording(inp, "full")
    else:
        desc = {"path": os.path.join(inp, sample.FILENAME), "folder": inp,
                "filename": sample.FILENAME, "variant": "full", "sampling_frequency": fs,
                "detect_channel": sample.DETECT_CHANNEL,
                "reference_interval": [0.1, sample.LEAD_S - 0.1],
                "breath_onsets_s": [],
                "mapping": {"flow": 2 + sample.N_EMG, "volume": 3 + sample.N_EMG,
                            "poes": 4 + sample.N_EMG, "pgas": 5 + sample.N_EMG,
                            "pdi": 6 + sample.N_EMG,
                            "emg": list(range(2, 2 + sample.N_EMG)), "entropy": []}}
    print(f"recording ready in {time.perf_counter() - t0:.1f} s, "
          f"{os.path.getsize(desc['path']) / 1e6:.0f} MB, {sample.N_BREATHS} breaths", flush=True)

    s = sample.build_sample_settings(desc, out, "full")
    if noplots:
        d = s.output.diagnostics
        d.save_pv_average = d.save_pv_individual = False
        d.save_raw = d.save_trimmed = d.save_drift = d.save_emg = d.save_flow_volume = False

    from respmech.core.pipeline import run_batch

    prof = cProfile.Profile()
    t1 = time.perf_counter()
    prof.enable()
    res = run_batch(s)
    prof.disable()
    wall = time.perf_counter() - t1
    print(f"run_batch wall {wall:.1f} s", flush=True)
    prof.dump_stats(os.path.join(work, "profile.prof"))

    import datetime

    from respmech.core.io.writers import write_batch

    wprof = cProfile.Profile()
    t2 = time.perf_counter()
    wprof.enable()
    written = write_batch(res, s, out, when=datetime.datetime(2026, 1, 1, 12, 0, 0))
    wprof.disable()
    wwall = time.perf_counter() - t2
    print(f"write_batch wall {wwall:.1f} s, {len(written)} files", flush=True)
    wprof.dump_stats(os.path.join(work, "write.prof"))

    buf = io.StringIO()
    st = pstats.Stats(prof, stream=buf)
    st.sort_stats("tottime").print_stats(30)
    st.sort_stats("cumulative").print_stats(40)
    with open(os.path.join(work, "top.txt"), "w") as fh:
        fh.write(f"wall {wall:.1f} s\n")
        fh.write(buf.getvalue())

    lines = []
    for root, _dirs, files in os.walk(out):
        for f in sorted(files):
            p = os.path.join(root, f)
            if f.lower().endswith((".csv", ".txt", ".json")):
                h = hashlib.sha256(open(p, "rb").read()).hexdigest()
                lines.append(f"{h}  {os.path.relpath(p, out)}")
            elif f.lower().endswith(".xlsx"):
                import pandas as pd
                for name, df in pd.read_excel(p, sheet_name=None).items():
                    blob = df.to_csv(float_format="%.17g").encode()
                    h = hashlib.sha256(blob).hexdigest()
                    lines.append(f"{h}  {os.path.relpath(p, out)}[{name}]")
    # the in-memory result too: every per-file breath table, at full float precision
    import pickle
    for fname, fr in sorted(res.files.items()):
        for attr in sorted(vars(fr)):
            val = getattr(fr, attr)
            if hasattr(val, "to_csv"):
                h = hashlib.sha256(val.to_csv(float_format="%.17g").encode()).hexdigest()
                lines.append(f"{h}  result:{fname}.{attr}")
    if res.average_table is not None and hasattr(res.average_table, "to_csv"):
        h = hashlib.sha256(res.average_table.to_csv(float_format="%.17g").encode()).hexdigest()
        lines.append(f"{h}  result:average_table")
    lines.append(f"{hashlib.sha256(pickle.dumps(res.noise_report)).hexdigest()}  result:noise_report")
    with open(os.path.join(work, "hashes.txt"), "w") as fh:
        fh.write("\n".join(sorted(lines, key=lambda x: x.split("  ", 1)[1])) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
