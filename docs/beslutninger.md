# Decisions — settled, don't reopen without new evidence

Dated project decisions and "considered and rejected" points, newest first. Each
entry: date, decision, reasoning, source (a conversation, a review, or "author's
decision <date>" — never an internal ticket reference; this repo is public).

---

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
