"""RespMech command-line interface.

    respmech run       settings.toml [--dry-run]
    respmech migrate   old_settings.py -o new_settings.toml
    respmech validate  settings.toml
    respmech init      new_settings.toml --signals flow,poes,emg [--folder DIR --files MASK --fs HZ]
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
        elif ev.kind == "warning":
            # Without respmech[plots], write_batch's figure step degrades to a
            # silent skip (exit 0, every workbook still written) — this is the only
            # place a terminal user sees it as it happens, alongside the FIGURES
            # SKIPPED section write_batch always leaves in run-report.txt.
            print(f"\nWARNING: {ev.message}", file=sys.stderr)
        elif ev.kind == "finished":
            print(f"\n{ev.message}")
    return cb


def cmd_run(args) -> int:
    from respmech.settingsio.toml_io import load_toml
    from respmech.core.pipeline import run_batch
    from respmech.core.io.writers import ecg_auto_detect_summary, write_batch

    settings = load_toml(args.settings)
    settings.validate()
    result = run_batch(settings, progress=_progress_printer())
    emg_cols = list(settings.input.channels.emg or [])

    if result.ecg_auto_report:
        # Shared with run-report.txt's DIAGNOSTICS block (writers.ecg_auto_detect_summary)
        # so the two can never again describe the same run in different words.
        print(f"\nECG auto-detect: {ecg_auto_detect_summary(result.ecg_auto_report, emg_cols)}")

    if not args.dry_run:
        written = write_batch(result, settings, settings.output.folder,
                              progress=_progress_printer())
        # K-278: the old "... to <output>/data" undercounted what a run actually
        # writes — diagnostics/ figures, WAV exports and the two root-level provenance
        # files are all part of `written` too (a 65-file run measured only 8 of them
        # under data/). Name the output root a real run writes into instead.
        print(f"\nWrote {len(written)} file(s) to {settings.output.folder}")
    else:
        # The same ceiling `core.io.plan.plan_outputs` builds for the GUI's
        # Dry run, over `result.files` (ok AND failed — a plan never depends on which
        # files happened to succeed, see the module docstring), so a CLI dry run stops
        # promising a different set of outputs than the app does.
        from respmech.core.io.plan import plan_outputs
        plan = plan_outputs(settings, list(result.files), cohort_outputs=True)
        print("\n[dry-run] computation complete; no files written. Output plan:")
        for g in plan.groups:
            cap = "up to " if g.is_cap else ""
            target = g.target or "(folder root)"
            print(f"  {g.category}: {cap}{g.count} file(s) in {target}")
        total_cap = "up to " if plan.is_cap else ""
        print(f"  Total: {total_cap}{plan.total_count} file(s) in {settings.output.folder}")
        from respmech.core.analysis.signals import Capabilities
        caps = Capabilities.from_settings(settings)
        # M-13: same line the GUI's commitment sheet prints -- what this signal set
        # actually computes, right where the output plan already answers "what/where".
        print(f"  Analyses: {', '.join(caps.analyses())}")
        print()
        unit_word = "segments" if caps.mode == "emg_only" else "breaths"
        for fname, fr in result.ok_files.items():
            if getattr(fr, "role", "tidal") == "reference":
                # M-30: no tidal table for this file at all — say so plainly instead
                # of a bare, misleading "0 breaths".
                n_ref = len(fr.manoeuvres or {})
                print(f"  {fname}: 0 tidal breaths, {n_ref} reference manoeuvre"
                     f"{'s' if n_ref != 1 else ''}")
            else:
                n = 0 if fr.breaths_table is None else len(fr.breaths_table)
                print(f"  {fname}: {n} {unit_word}")

    if result.failed_files:
        print(f"\n{len(result.failed_files)} file(s) FAILED:", file=sys.stderr)
        for fname, fr in result.failed_files.items():
            print(f"  {fname}: {fr.error}", file=sys.stderr)
        return 1
    return 0


def _toml_string(value: str) -> str:
    """A syntactically valid TOML string literal for an arbitrary ``value`` (M-13's
    ``_init_template`` uses this for ``folder``/``files``, hand-built text rather than
    routed through ``tomli_w`` -- see that function's own docstring for why).

    Prefers a TOML *literal* (single-quoted) string, which needs NO escaping at all --
    the standard TOML idiom for a filesystem path, so a Windows ``--folder
    C:\\Users\\...`` is written back out unchanged rather than needing every backslash
    escaped. Falls back to a properly escaped *basic* (double-quoted) string only when
    ``value`` itself contains something a literal string cannot hold (a literal ``'``,
    which a single-quoted string has no escape for at all, or a raw control character).
    """
    if "'" not in value and not any(ord(c) < 0x20 for c in value):
        return f"'{value}'"
    escaped = []
    for ch in value:
        if ch == "\\":
            escaped.append("\\\\")
        elif ch == '"':
            escaped.append('\\"')
        elif ch == "\n":
            escaped.append("\\n")
        elif ch == "\t":
            escaped.append("\\t")
        elif ch == "\r":
            escaped.append("\\r")
        elif ord(ch) < 0x20:
            escaped.append(f"\\u{ord(ch):04x}")
        else:
            escaped.append(ch)
    return '"' + "".join(escaped) + '"'


def _init_template(signals: list, *, folder, files, fs: "int | None") -> str:
    """The commented, signal-set-filtered starting point ``respmech init`` writes (M-13).

    Only the ``[input.channels]`` entries the CHOSEN signals actually need are written
    (no Pgas/Pdi keys for a poes-only set); ``folder``/``files``/``sampling_frequency``
    are written as real values when the caller supplied them, or as a commented
    placeholder line otherwise. ``Settings.validate()`` requires
    ``sampling_frequency``, so the file only becomes runnable once that (and
    folder/files, for anything to actually match) is filled in -- every OTHER value
    here (the channel column numbers, guessed sequentially from column 2; the
    segmentation method for an EMG-only set) is already a valid placeholder, so
    ``load_toml(path).validate()`` succeeds on the channels/analysis side immediately,
    unedited.
    """
    from respmech.core.analysis.signals import SINGLE_SIGNALS

    declared = set(signals)
    order = [s for s in SINGLE_SIGNALS if s in declared]
    if "emg" in declared:
        order.append("emg")
    has_flow = "flow" in declared

    lines: list[str] = [
        f"# RespMech analysis settings — generated by `respmech init --signals "
        f"{','.join(order)}`.",
        "#",
        "# Fill in the placeholders below (folder, files, sampling frequency, and the",
        "# channel column numbers, which are only a starting guess), then check it",
        "# with:",
        "#",
        "#     respmech validate <this file>",
        "",
        "[analysis]",
        "signals = [" + ", ".join(f'"{s}"' for s in order) + "]",
        "",
        "[input]",
    ]
    if folder:
        lines.append(f"folder = {_toml_string(folder)}")
    else:
        lines.append('# folder = "<input.folder>"   # REQUIRED — point this at your recordings')
    if files:
        lines.append(f"files = {_toml_string(files)}")
    else:
        lines.append('# files  = "*.csv"            # case-insensitive glob')
    lines += ["", "[input.format]"]
    if fs:
        lines.append(f"sampling_frequency = {fs}   # Hz")
    else:
        lines.append("# sampling_frequency = 1000   # Hz — REQUIRED, integer")
    lines += ["", "[input.channels]            # 1-based column numbers; column 1 is usually time"]
    col = 2
    for role in order:
        lines.append(f"emg  = [{col}]" if role == "emg" else f"{role:<5}= {col}")
        col += 1
    lines.append("")
    if has_flow:
        lines += ["[processing.volume]",
                  "integrate_from_flow = true   # derive volume from flow — no volume channel "
                  "needed above",
                  ""]
    else:
        lines += ["[processing.segmentation]",
                  'method = "whole_file"        # no flow channel to detect breaths from — see '
                  'also "separators"/"fixed_windows"/"emg_burst"',
                  ""]
    lines += [
        "# Example: manually exclude one breath from a file's tidal average. An entry",
        "# WITHOUT `folder =` is read as carried over from a DIFFERENT recordings folder",
        "# — tag a fresh entry you add here with the input folder above so it is never",
        "# mistaken for one:",
        "#",
        "# [[processing.exclude_breaths]]",
        '# file = "P01.csv"',
        "# breaths = [3]",
        '# folder = "<input.folder>"',
        "",
    ]
    return "\n".join(lines) + "\n"


def cmd_init(args) -> int:
    from respmech.core.analysis.signals import SINGLE_SIGNALS

    raw = [s.strip() for s in args.signals.split(",") if s.strip()]
    known = frozenset(SINGLE_SIGNALS) | {"emg"}
    unknown = [s for s in raw if s not in known]
    if unknown:
        print(f"error: unknown signal(s) in --signals: {', '.join(unknown)}", file=sys.stderr)
        return 2
    declared = set(raw)
    # Mirrors Settings.validate()'s own two analysis.signals rules EXACTLY (same
    # condition, same order, same wording -- core/settings.py), so a --signals value
    # rejected here and the equivalent hand-written TOML fail with the identical
    # message. The first rule fires only for a genuinely empty set (--signals ",", or
    # all-unknown entries already caught above) -- NOT "does it contain flow/emg",
    # which would wrongly pre-empt the second, more specific rule for e.g. --signals
    # poes (a non-empty set that still needs flow).
    if not declared:
        print("error: --signals must name at least one of 'flow' or 'emg'", file=sys.stderr)
        return 2
    if (declared & {"poes", "pgas", "pdi"}) and "flow" not in declared:
        print("error: --signals: 'poes', 'pgas' and 'pdi' require 'flow'", file=sys.stderr)
        return 2

    text = _init_template(raw, folder=args.folder, files=args.files, fs=args.fs)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {args.output}")
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
    from respmech.core.io.plan import probe_write_folder
    from respmech.core.io.loaders import probe_constant_channels, probe_merged_time_blocks
    from respmech.core.analysis.signals import (
        Capabilities, effective_signals, off_signals_text, signals_text)

    settings = load_toml(args.settings)
    settings.validate()
    # An EMG-only signal set has no flow channel for a constant one to ever be
    # assigned against, so the flow-only hard-failure rule below is meaningless there —
    # a constant EMG channel is EMG-only's equivalent hard failure instead (RMS/entropy
    # on a dead channel is a meaningless number, not merely an unused pressure port).
    declared = effective_signals(settings)
    emg_only = "flow" not in declared and "emg" in declared
    pattern = os.path.join(settings.input.folder, settings.input.files)
    # match_input_files: the SAME matcher run_batch uses, so the reported count is exactly
    # what `respmech run` will process (case-insensitive; safe against folder metacharacters).
    files = match_input_files(settings.input.folder, settings.input.files)
    print(f"Settings valid. Input pattern '{pattern}' matches {len(files)} file(s).")
    # M-13: which signals govern this analysis and what they compute, so `validate`
    # alone answers "what will this run actually produce" without a dry run.
    caps = Capabilities.from_settings(settings)
    n_entropy = len(settings.input.channels.entropy or [])
    summary = (f"Signals: {signals_text(settings)} · Entropy: {n_entropy} columns · "
              f"Analyses: {', '.join(caps.analyses())}")
    off = off_signals_text(caps.declared)
    if off:
        summary += f" · off: {off} (not in signal set)"
    print(summary)
    ok = True
    if not files:
        print("WARNING: no input files match.", file=sys.stderr)
        ok = False
    # Merged-block / constant-channel probes (the same core.quality-backed checks
    # ui.manifest.build_manifest runs for the GUI's Setup QC strip): a headless `validate`
    # never builds a Manifest (that would pull ui.manifest -> ui.workers -> PySide6 into a
    # CLI-only install), so these are called directly on core.io.loaders instead.
    for f in files:
        merged = probe_merged_time_blocks(settings, f)
        if merged:
            print(f"WARNING: {os.path.basename(f)}: {merged}", file=sys.stderr)
            ok = False
        constant = probe_constant_channels(settings, f)
        if constant:
            print(f"WARNING: {os.path.basename(f)}: constant channel(s) that never "
                  f"vary: {', '.join(constant)}", file=sys.stderr)
            # A constant FLOW channel really will fail the run (ConstantFlowError --
            # segmentation cannot proceed) and is worth failing validate over too; any
            # OTHER constant channel is advisory only, same as Manifest.
            # constant_channel_files' own docstring promises the GUI's QC strip -- a
            # permanently unused pressure port (e.g. no Pdi balloon) is a legitimate
            # real setup, and validate should not fail every time on it. An EMG-only
            # set has no flow channel at all -- a constant EMG channel there is the
            # equivalent hard failure instead.
            if emg_only:
                if any(name.startswith("EMG #") for name in constant):
                    ok = False
            elif any(name.startswith("Flow ") for name in constant):
                ok = False
    # Settings.unknown is collected by from_dict but was never read anywhere —
    # a misspelled key silently ran on the default it was meant to override, with no
    # warning from validate, the run, or run-report.txt. Report it here so the site's
    # promise ("skim the validate output ... to catch this") is actually true.
    if settings.unknown:
        print(f"WARNING: {len(settings.unknown)} unrecognised setting(s) — the default "
              "was used instead of the value below:", file=sys.stderr)
        for key, val in settings.unknown.items():
            print(f"  {key} = {val!r}", file=sys.stderr)
        ok = False
    # K-098: a real write probe (never os.access, unreliable against Windows ACLs — see
    # core.io.plan.probe_write_folder's own docstring), the same one the GUI's Dry run
    # already performs, so a read-only or missing output folder is caught here instead
    # of after an entire batch has been computed. This gives `validate` a side effect
    # (it creates, and then removes, the output folder if it does not exist yet) —
    # accepted deliberately (Emil, 05-09-2026) for the earlier, cheaper failure.
    probe = probe_write_folder(settings.output.folder)
    if not probe.ok:
        print(f"WARNING: output folder is not writable: {probe.message}", file=sys.stderr)
        ok = False
    return 0 if ok else 1


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

    pi = sub.add_parser("init", help="write a commented, signal-set-filtered starting TOML file")
    pi.add_argument("output", help="path to write, e.g. new_settings.toml")
    pi.add_argument("--signals", required=True,
                    help="comma-separated signal set, e.g. flow,poes,emg")
    pi.add_argument("--folder", help="recordings folder (left as a placeholder if omitted)")
    pi.add_argument("--files", help="input file mask (left as a placeholder if omitted)")
    pi.add_argument("--fs", type=int, help="sampling frequency in Hz (left as a placeholder if omitted)")
    pi.set_defaults(func=cmd_init)
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
