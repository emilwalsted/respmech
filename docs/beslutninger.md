# Decisions — settled, don't reopen without new evidence

Dated project decisions and "considered and rejected" points, newest first. Each
entry: date, decision, reasoning, source (a conversation, a review, or "author's
decision <date>" — never an internal ticket reference; this repo is public).

---

**27-09-2026 — `use_expiration`/`reference_intervals` remain the two saved noise-
reference fields for a flow-bearing analysis; `reference_mode` only names the EMG-only
alternatives (author's decision, 27-09-2026).** An EMG-only signal set has no
inspiration/expiration phases for `use_expiration` to mean anything about, so it needed
its own reference-resolution rule rather than reusing or repurposing that flag. Rather
than overload `use_expiration`/`reference_intervals` with a third, EMG-only meaning, a
new `reference_mode` field (default `'auto'`) was added purely to NAME the two EMG-only
sources (`'rest_segments'`, a segment typed `'rest'` in the reference file; `'interburst'`,
not yet implemented) — `'auto'` with a flow channel declared reproduces the existing
`use_expiration`/`reference_intervals` rule exactly, unchanged. `resolve_noise_reference_
mode(settings)` is the one function every consumer (the pipeline, the run report, the
Provenance sheet) reads this from, so the resolution logic cannot drift between them. A
typed breath (any kind) is also now excluded from the flow-bearing reference file's own
expiration mask when that mode is `'expiration'` — a manoeuvre's expiration is not the
diaphragm-quiet period the profile is trusted for — with no effect on any analysis that
has no typed breaths at all (every existing scenario). See `docs/NOISE_ECG_OPTIMIZATION.
md` §6 for the full resolution chain.

**27-09-2026 — A typed breath counts toward `bcnt`/`vefactor` exactly like a manually
excluded one: neither is scaled down when a breath is retyped as a manoeuvre (author's
decision, 27-09-2026).** `bcnt` (the breath count used to scale breathing frequency and
minute ventilation) is `len(breaths)` — every breath the segmenter DETECTED, whether or
not it is later excluded or typed — and `vefactor` is a pure function of the recording's
own duration, unrelated to which breaths are counted. Typing a breath as `ic`/`fvc`/
`max_insp`/`sniff`/`rest`/`other` folds it into the same exclusion machinery a manual
click already used (see `core._legacy_ns.to_legacy_ns`'s union of `exclude_breaths` and
`breath_types`), so it is left out of the tidal average the same way, but the file's
own breath count is not renormalised to pretend the manoeuvre breath was never there.
This is not a new rule invented for breath typing: the codebase has never scaled the
breath count by exclusion, and a typed breath is deliberately not made a special case
of that existing behaviour.

**27-09-2026 — Relevance-driven visibility (R7) is a permitted refinement of
inverted gating, not an exception, and Sample entropy is never hidden by the
signal set (R8) (author's decision, 27-09-2026).** A surface — a sub-tab, an
Advanced… settings group, a result column, a figure — that CANNOT apply to the
declared signal set is hidden as a pure function of `analysis.signals`, never
as a function of how far the user has progressed through the workflow.
Everything hidden this way comes back the moment the signal set changes
(Setup ▸ Signals): there is no separate "unlock" step and no progressive
disclosure. The commitment sheet stays the only gate that can actually block a
run. Sample entropy is the one deliberate exception to the exception: it stays
conditioned on whether a column is assigned to it (`bool(channels.entropy)`)
alone, regardless of the declared signal set, because entropy is an
opportunistic measurement any recording can carry, not a consequence of which
pressure/flow signals the analysis otherwise declares.

**26-09-2026 — Changing an analysis's declared signal set clears the breath-keyed
state built for the previous segmentation, after one confirmation (author's
decision, 26-09-2026).** Removing or adding Flow changes which segmenter produces
every breath number and kind, so a breath exclusion, breath-count override, or
(once later tickets add them) a breath type, reference or manual separator made
under the old segmentation no longer describes anything real under the new one.
The signal-set picker's one funnel (`apply_signal_set`) asks once, covering all of
those lists together, and only when flow's membership of the set actually changes
AND there is something to lose — a freshly reset analysis is never asked, since
nothing has been built yet to lose. Precedent for the general principle (dropping
a role clears the derived state that depended on it) is the existing EMG-role
removal: clearing a now-invalid `ecg_auto_detect` the moment the EMG role leaves
the channel set, silently, no confirmation, because a stuck-invalid `Settings()`
with no visible way out was worse than asking. This decision keeps that same
principle but adds the confirmation specifically for flow, because losing flow
invalidates every existing breath number, not one boolean flag — a change large
enough that silently discarding it would be a surprise, not a repair.

**11-09-2026 — Three number-changing findings from the 04-08-2026 mechanism review
stay unimplemented (author's decision, 04-08-2026).** A review of the analysis
mechanisms surfaced three ideas that would change computed numbers: scaling the
breath count on exclusion, unit declaration with a plausibility check, and
filtering phantom breaths detected from flow noise. All three were deliberately
rejected — do not re-propose them without a new, separate decision.

**11-09-2026 — The resistive-work crossing in `compute.py` (recoil line crosses at
V = 0, not at the inspiration's starting volume) stays as-is (author's decision,
05-09-2026).** Inherited verbatim from the v1 code and locked by golden. The
behaviour is kept and documented as known v1 heritage, with a recommendation to
use `correct_trend` when EELV drifts — not fixed in place. This is a fourth entry
alongside the three findings above; don't reopen without a new, separate decision.

**11-09-2026 — Plain semver only, no `-rc`/`-beta` suffixes (established practice,
`docs/RELEASING.md`).** A PyPI version is immutable once published: a broken
release is fixed with a patch bump and a brand-new tag, never a re-push of the
same tag or a pre-release suffix.

**11-09-2026 — The release e-mail is never sent automatically by any app-side
workflow (author's decision, carried from the website's own decision of
03-08-2026).** Only the website's own announce workflow mails subscribers, and
only when triggered separately from a deploy. Framed here from the app side: the
release workflow's `repository_dispatch` to the website repo updates the site
and its changelog page, nothing more — announcing is a deliberate, separate act.

**11-09-2026 — Releases use GitHub's Trusted Publishing (OIDC) for PyPI, no
long-lived tokens stored anywhere (established practice, `docs/RELEASING.md`).**
PyPI trusts exactly `publish-pypi.yml` running in the `pypi` environment of this
repository; there is no API token to leak or rotate.

**11-09-2026 — Windows installers are signed locally, after release, never in
CI (established practice, `docs/SIGNING.md`).** The chosen certificate class is
hardware-backed and cannot sign on a CI runner, so CI always ships an unsigned
MSI and a local, manual step (`scripts/sign-msi-certum.sh`) replaces the release
asset afterwards. macOS signs and notarises inside CI instead, since its
Developer ID key is not hardware-bound.

**11-09-2026 — Changes with no changelog-worthy user effect are marked
`<!-- changelog-skip <sha> <reason> -->` in `CHANGELOG.md`, never simply
omitted (established practice, see the entries already in `CHANGELOG.md`).** The
comment names which entry already covers the change's effect, or states plainly
that there is none, so a change is never silently unaccounted for.

**11-09-2026 — The v2 UI overhaul (two-tab workspace, `FileRail`, the Run &
results drawer, inverted gating) is the current, settled shape of the app,
shipped as v2.4.0 (21-08-2026).** It replaced an earlier multi-screen wizard
flow entirely; do not reintroduce tab-gated or progressive-disclosure UI
patterns without a fresh decision — the inverted-gating principle (every
surface always reachable, only the ACTION disabled) was a deliberate, reviewed
design change, not an incremental one.
