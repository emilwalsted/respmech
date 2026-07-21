"""RespMech command-line interface.

    respmech run       settings.toml [--dry-run]
    respmech migrate   old_settings.py -o new_settings.toml
    respmech validate  settings.toml
    respmech --version

Batch processing is first-class and scriptable (no editing a Python file to launch
a run). ``run`` returns a non-zero exit code if any file fails.
"""
from __future__ import annotations

import argparse
import sys

from respmech import __version__


def _progress_printer():
    def cb(ev):
        if ev.kind == "file_start":
            print(f"\n{ev.file}: {ev.message}")
        elif ev.kind == "stage":
            print(f"  {ev.message}...")
        elif ev.kind == "breath":
            print(f"\r  breath {ev.breath}/{ev.total_breaths}", end="", flush=True)
        elif ev.kind == "file_done":
            print(f"\r  done ({ev.message})")
        elif ev.kind == "file_error":
            print(f"\r  ERROR: {ev.message}")
        elif ev.kind == "finished":
            print(f"\n{ev.message}")
    return cb


def cmd_run(args) -> int:
    from respmech.settingsio.toml_io import load_toml
    from respmech.core.pipeline import run_batch
    from respmech.core.io.writers import write_batch

    settings = load_toml(args.settings)
    settings.validate()
    result = run_batch(settings, progress=_progress_printer())

    if not args.dry_run:
        written = write_batch(result, settings, settings.output.folder)
        print(f"\nWrote {len(written)} file(s) to {settings.output.folder}/data")
    else:
        print("\n[dry-run] computation complete; no files written.")
        for fname, fr in result.ok_files.items():
            n = 0 if fr.breaths_table is None else len(fr.breaths_table)
            print(f"  {fname}: {n} breaths")

    if result.failed_files:
        print(f"\n{len(result.failed_files)} file(s) FAILED:", file=sys.stderr)
        for fname, fr in result.failed_files.items():
            print(f"  {fname}: {fr.error}", file=sys.stderr)
        return 1
    return 0


def cmd_migrate(args) -> int:
    from respmech.settingsio.migrate import migrate_file
    from respmech.settingsio.toml_io import save_toml

    settings, report = migrate_file(args.legacy)
    settings.validate()
    save_toml(settings, args.output)
    print(f"Wrote {args.output}")
    print()
    print(report.text())
    return 0


def cmd_validate(args) -> int:
    import os
    from respmech.settingsio.toml_io import load_toml
    from respmech.core.pipeline import match_input_files

    settings = load_toml(args.settings)
    settings.validate()
    pattern = os.path.join(settings.input.folder, settings.input.files)
    # match_input_files: the SAME matcher run_batch uses, so the reported count is exactly
    # what `respmech run` will process (case-insensitive; safe against folder metacharacters).
    files = match_input_files(settings.input.folder, settings.input.files)
    print(f"Settings valid. Input pattern '{pattern}' matches {len(files)} file(s).")
    if not files:
        print("WARNING: no input files match.", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="respmech", description="Respiratory mechanics, WOB and EMG analysis.")
    p.add_argument("--version", action="version", version=f"respmech {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="process a batch defined by a TOML settings file")
    pr.add_argument("settings")
    pr.add_argument("--dry-run", action="store_true", help="compute but do not write output files")
    pr.set_defaults(func=cmd_run)

    pm = sub.add_parser("migrate", help="convert a legacy .py settings file to TOML")
    pm.add_argument("legacy")
    pm.add_argument("-o", "--output", required=True, help="output .toml path")
    pm.set_defaults(func=cmd_migrate)

    pv = sub.add_parser("validate", help="validate a TOML settings file and its inputs")
    pv.add_argument("settings")
    pv.set_defaults(func=cmd_validate)
    return p


def main(argv=None) -> int:
    # See ui/app.main: figures are written in a spawned child, and a packaged binary must not
    # re-run main() when it is started as one. No-op when not frozen.
    import multiprocessing
    multiprocessing.freeze_support()

    # On Windows the console is often cp1252; a file name or message carrying a
    # non-cp1252 character (Excel exports, é/ø/…) would otherwise abort a whole run
    # with UnicodeEncodeError. Degrade unencodable characters instead of crashing.
    for _stream in (sys.stdout, sys.stderr):
        if _stream is not None and hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(errors="replace")
            except Exception:
                pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
