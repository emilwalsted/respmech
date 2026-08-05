"""What a batch actually IS, as one object — instead of the single first-file peek the
rest of the UI used to reason from.

Before this module, ``SettingsScreen._update_format_readout`` looked at ``files[0]`` and
reported that as if it described the whole folder, while ``_valid_input_files`` (used to
build the channel-assignment mapping) silently probed every file and excluded any that
did not share the majority column count. The two numbers could disagree, and neither
mentioned a per-file sampling-frequency mismatch or a narrowed multi-pattern mask at all.
``Manifest``/``FileEntry`` give every screen that cares about the batch (Setup's read-out
and QC strip, the channel-assignment dialog, and — from ticket B02 onward — the file rail
and Run) ONE shared, pre-computed answer to "what files are in this batch, and what is
wrong with any of them".

Qt-free by design: only the SCANNER that fills a ``Manifest`` (``build_manifest`` here,
and any QThread wrapper a later ticket puts around it) may know Qt. The dataclasses and
``build_manifest`` itself import nothing from PySide6 and can be built and asserted on in
a plain unit test, with no ``QApplication``. The default probers reach lazily into
``respmech.ui.workers`` (which DOES import Qt for its worker classes, alongside these
plain functions) only when a caller does not inject its own — importing that module does
not require a live ``QApplication``, but a test wanting total isolation can pass its own
``columns_prober``/``freq_prober`` instead and never touch it.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass

from respmech.ui.validation import matching_files


def _file_token(path):
    """(path, mtime_ns, size) freshness token for the per-file probe cache, or ``None`` if
    the file cannot be stat'd (then the caller never caches — always re-probes)."""
    try:
        st = os.stat(path)
        return (path, st.st_mtime_ns, st.st_size)
    except OSError:
        return None


@dataclass
class FileEntry:
    """One file the batch's mask matched, with what the scanner learned about it."""

    path: str
    filename: str
    ext: str
    columns: int | None = None          # None: unreadable / could not be probed
    detected_fs: int | None = None      # None: not probed (excluded, or .mat, or inconclusive)
    included: bool = True               # part of the majority-column-count batch
    exclude_reason: str | None = None   # human reason `included` is False; None when included


@dataclass
class Manifest:
    """A whole matched batch: the effective mask, the majority column layout, and every
    file that disagrees with it — either in column count (``outliers``) or in detected
    sampling frequency (``freq_mismatches``, a subset of ``included_files``)."""

    folder: str
    mask: str                                  # effective, single-pattern mask actually used
    settings_fs: int | None = None             # the sampling frequency configured in Settings
    mask_narrowed_from: str | None = None       # the original multi-pattern mask, or None
    narrowed_out_exts: tuple = ()               # extensions dropped by the narrowing
    narrowed_out_count: int = 0                 # how many matched files that narrowing excluded
    files: tuple = ()                           # every file the (narrowed) mask matched
    majority_columns: int | None = None

    @property
    def included_files(self):
        return tuple(f for f in self.files if f.included)

    @property
    def outliers(self):
        """Matched files excluded from the batch because their column count is not the
        majority (includes unreadable files, which are never counted as included)."""
        return tuple(f for f in self.files if not f.included)

    @property
    def freq_mismatches(self):
        """Included files whose detected sampling frequency disagrees with the Settings
        value. Empty whenever ``settings_fs`` is unset — nothing to compare against."""
        if not self.settings_fs:
            return ()
        return tuple(f for f in self.included_files
                    if f.detected_fs is not None and f.detected_fs != self.settings_fs)

    @property
    def is_clean(self):
        """No caveats at all: nothing was narrowed, nothing disagrees on columns or
        frequency. The condition ``_update_qc`` must require before it may say so."""
        return not self.mask_narrowed_from and not self.outliers and not self.freq_mismatches


def narrow_mask(folder, mask):
    """If ``mask`` is a multi-pattern mask (``'*.csv; *.txt'``), narrow it to the SINGLE
    pattern matching the extension with the most currently-matching files — the core batch
    runner globs one pattern only (ties broken alphabetically, matching the pre-existing
    behaviour this replaces). Returns ``(effective_mask, narrowed_from, dropped)`` where
    ``narrowed_from`` is the original ``mask`` (``None`` if nothing needed narrowing) and
    ``dropped`` is ``{ext: count}`` for every extension NOT chosen.

    A no-op (``mask`` unchanged, ``narrowed_from=None``) when the mask is already a single
    pattern, or when there is no folder/files yet to decide from — mirrors
    ``SettingsScreen._normalize_mask``, which now delegates its own narrowing decision to
    this function so the two can never compute a different answer for the same inputs."""
    mask = mask or ""
    if ";" not in mask and "," not in mask:
        return mask, None, {}
    folder = (folder or "").strip()
    if not folder or not os.path.isdir(folder):
        return mask, None, {}
    exts = [os.path.splitext(f)[1].lower() for f in matching_files(folder, mask)]
    if not exts:
        return mask, None, {}
    counts = Counter(exts)
    top = max(counts.values())
    ext = sorted(e for e, c in counts.items() if c == top)[0]      # deterministic on a tie
    narrowed = f"*{ext}"
    if narrowed == mask:
        return mask, None, {}
    dropped = {e: c for e, c in counts.items() if e != ext}
    return narrowed, mask, dropped


def build_manifest(folder, mask, settings, *, columns_prober=None, freq_prober=None, cache=None):
    """Build a :class:`Manifest` describing every file ``mask`` matches in ``folder``.

    Column counts and detected sampling frequencies are probed at most ONCE per
    ``(path, mtime, size)`` — ``cache`` is a plain dict the CALLER owns and reuses across
    builds (a memo, not part of the Manifest), so rebuilding over an unchanged folder never
    re-probes a single file. Pass a fresh ``{}`` (the default) for a one-off build, or the
    same dict across repeated calls (as ``SettingsScreen`` does) to make that memoisation
    actually pay off.

    ``columns_prober(settings, path) -> int|None`` and ``freq_prober(settings, path) ->
    int|None`` default to :func:`respmech.ui.workers.peek_columns` (the cheap 8 KB byte-peek
    for .csv/.txt, falling back to ``probe_data_columns`` for .xlsx/.mat) and
    :func:`respmech.ui.workers.probe_sampling_frequency` (a capped column-0 read). Inject
    your own to test this function without importing ``workers`` at all, or to count/assert
    probe calls (see the ticket's "zero calls to probe_data_columns for a .csv folder"
    requirement).

    Frequency is probed only for INCLUDED files (the majority column-count group) — an
    outlier is already excluded from the batch on column count alone, so its frequency
    would never be acted on."""
    if columns_prober is None or freq_prober is None:
        from respmech.ui import workers as _workers   # lazy: keeps this module importable
                                                        # without pulling PySide6 in for a
                                                        # caller that injects its own probers
        if columns_prober is None:
            columns_prober = _workers.peek_columns
        if freq_prober is None:
            freq_prober = _workers.probe_sampling_frequency
    if cache is None:
        cache = {}

    settings_fs = getattr(settings.input.format, "sampling_frequency", None)
    # the decimal separator changes how peek_columns/probe_sampling_frequency PARSE a .csv
    # (and how probe_sampling_frequency parses numbers for .txt too) — folded into the cache
    # key so a decimal-separator edit (auto-detected by _probe_and_apply_file_settings, or
    # set by hand) can never serve a value probed under the OLD separator back out of a
    # cache that is otherwise keyed purely on file identity.
    decimal = getattr(settings.input.format, "decimal", ".") or "."

    effective_mask, narrowed_from, dropped = narrow_mask(folder, mask)
    narrowed_out_count = sum(dropped.values())

    matched = matching_files(folder, effective_mask) if folder and os.path.isdir(folder) else []

    probed = []                      # (path, filename, ext, columns, token)
    counts = Counter()
    for path in matched:
        filename = os.path.basename(path)
        ext = os.path.splitext(filename)[1].lower()
        token = _file_token(path)
        key = ("cols", token, decimal) if token is not None else None
        if key is not None and key in cache:
            columns = cache[key]
        else:
            columns = columns_prober(settings, path)
            if key is not None:
                cache[key] = columns
        # A file with fewer than 2 columns (no signal beyond the time axis, or a stray
        # non-data file the mask happened to match) can never WIN the majority vote —
        # mirrors _valid_input_files's own `n >= 2` filter exactly, so the two votes can
        # never disagree about which layout is "the batch" (a folder dominated by
        # degenerate 1-column files used to let build_manifest crown THEM the majority
        # and mark the real recordings as outliers, while _valid_input_files — which
        # excludes them before voting — still correctly picked the real recordings).
        if columns is not None and columns >= 2:
            counts[columns] += 1
        probed.append((path, filename, ext, columns, token))

    majority = None
    if counts:
        top = max(counts.values())
        # a tie gives the WIDEST layout — matches _valid_input_files's existing rule
        majority = max(n for n, c in counts.items() if c == top)

    entries = []
    for path, filename, ext, columns, token in probed:
        if columns is None:
            entries.append(FileEntry(path=path, filename=filename, ext=ext, columns=None,
                                     included=False, exclude_reason="could not be read"))
            continue
        included = columns == majority
        detected_fs = None
        if included:
            fkey = ("fs", token, decimal) if token is not None else None
            if fkey is not None and fkey in cache:
                detected_fs = cache[fkey]
            else:
                detected_fs = freq_prober(settings, path)
                if fkey is not None:
                    cache[fkey] = detected_fs
        reason = None if included else f"{columns} column{'s' if columns != 1 else ''} (majority is {majority})"
        entries.append(FileEntry(path=path, filename=filename, ext=ext, columns=columns,
                                 detected_fs=detected_fs, included=included, exclude_reason=reason))

    return Manifest(folder=folder, mask=effective_mask, settings_fs=settings_fs,
                    mask_narrowed_from=narrowed_from,
                    narrowed_out_exts=tuple(sorted(dropped)), narrowed_out_count=narrowed_out_count,
                    files=tuple(entries), majority_columns=majority)
