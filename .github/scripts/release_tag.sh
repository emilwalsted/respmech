#!/usr/bin/env bash
# .github/scripts/release_tag.sh — create and push the annotated release tag
# SERVER-SIDE, on behalf of a "release request". Called only by
# .github/workflows/release-request.yml; runnable standalone for testing:
#
#   DEFAULT_BRANCH=master REQUEST_SHA=<sha> DRY_RUN=true bash .github/scripts/release_tag.sh
#
# WHY THIS EXISTS. A RespMech release is cut by one thing only: a vX.Y.Z tag on
# master (docs/RELEASING.md). Some maintainer environments can push branches but
# are not allowed to push tags. For those, the request is expressed as a branch
# instead: an EMPTY commit whose first line is exactly "Release vX.Y.Z", sitting
# directly on top of master's current tip, pushed to the branch
# "release-request". This script reads that commit, checks it, and creates the
# tag ON MASTER'S TIP (the request commit's parent), never on the request commit
# itself, so "a release tag must point at master" stays literally true and the
# tag-triggered release.yml / publish-pypi.yml take over completely unchanged.
#
# WHY THE PARENT IS TAGGED, NOT THE REQUEST COMMIT. A tag on the request commit
# would build the same tree, but it would not point at master, and it would leave
# an extra commit outside master permanently visible through `git describe` and
# the tag's ancestry. Tagging the parent keeps history identical to the manual
# `git tag vX.Y.Z master` practice.
#
# A PyPI version is immutable, and by the same discipline a pushed tag is never
# moved. The script is therefore deliberately IDEMPOTENT (an existing tag that
# already points at exactly the requested commit is success, so a re-run after a
# network failure is harmless) but REFUSES outright to move or overwrite a tag
# that points anywhere else.
#
# Requires a clone with FULL history (fetch-depth 0), so the request commit's
# parent and all tags are available.
set -euo pipefail

log() { printf '%s\n' "$*" >&2; }

DEFAULT_BRANCH="${DEFAULT_BRANCH:?DEFAULT_BRANCH must be set}"
# REQUEST_SHA: the commit that triggered the workflow (github.sha of the push to
# release-request). Falls back to HEAD for a standalone run.
REQUEST_SHA="${REQUEST_SHA:-$(git rev-parse HEAD)}"
DRY_RUN="${DRY_RUN:-false}"
# VERSION_FILE: where the package version lives. The tag must equal it (see 5).
VERSION_FILE="${VERSION_FILE:-src/respmech/__init__.py}"

# Strict boolean: a plain `[ "$DRY_RUN" = "true" ]` would silently read "True",
# "1" or " true" as "run for real", the least safe interpretation. Fail loudly.
case "$DRY_RUN" in
  true|false) : ;;
  *) log "release_tag: invalid DRY_RUN='$DRY_RUN' (must be 'true' or 'false')"; exit 2 ;;
esac

if ! git rev-parse --verify --quiet "${REQUEST_SHA}^{commit}" >/dev/null; then
  log "release_tag: REQUEST_SHA='$REQUEST_SHA' is not a known commit in this clone (fetch-depth 0 required)"
  exit 2
fi

if ! git rev-parse --verify --quiet "refs/remotes/origin/${DEFAULT_BRANCH}" >/dev/null; then
  log "release_tag: origin/${DEFAULT_BRANCH} does not exist, nothing to compare against"
  exit 1
fi

# ---------- 1. derive the version from the FIRST LINE of the commit message ----------
# The whole first line must be exactly "Release vX.Y.Z": no loose grep for a
# version somewhere in the text. That rules out both false positives (a version
# mentioned in a longer sentence) and suffix ambiguity ("Release v2.5.1-rc1":
# only plain semver is ever released, see docs/RELEASING.md).
# Deliberately `%B` + `head -1`, NOT `%s`: git's %s ("subject") is the first
# PARAGRAPH, so a blank-line-free message "Release\nv1.2.3" folds to
# "Release v1.2.3" and would match by mistake.
FIRST_LINE="$(git show -s --format=%B "$REQUEST_SHA" | head -1 | tr -d '\r')"
# Strict semver: no component with a leading zero ("v1.02.3" is not valid semver,
# and PyPI would normalise the "02" away, so the git tag and the immutable PyPI
# version would differ forever).
if [[ ! "$FIRST_LINE" =~ ^Release\ (v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*))$ ]]; then
  log "release_tag: the commit message's first line is not of the form 'Release vX.Y.Z' (strict semver, no leading zero): '$FIRST_LINE'"
  exit 1
fi
VERSION="${BASH_REMATCH[1]}"

# The fully qualified ref, never the bare name "origin/master": git resolves a bare
# name to refs/tags/ FIRST, so a tag that happens to be called "origin/master" would
# silently stand in for the branch and put the release tag on whatever it points at.
DEFAULT_SHA="$(git rev-parse --verify "refs/remotes/origin/${DEFAULT_BRANCH}^{commit}")"

# ---------- 2. the request commit itself: one parent, and EMPTY ----------
# These two depend only on the request commit, not on where master is now, so they
# run BEFORE the idempotent exit below: a request that carries a change must be
# refused even when its tag already exists, or the run would end green as "already
# present" while the change the requester believes is released is in no release.
# `--verify --quiet` (NOT a bare `git rev-parse X^`): without --verify git prints
# the UNRESOLVED string "<sha>^" on stdout and exits 128 for a root commit, so
# PARENT_SHA would be non-empty text and never reach the emptiness check below.
PARENT_SHA="$(git rev-parse --verify --quiet "${REQUEST_SHA}^" 2>/dev/null || true)"
if [ -z "$PARENT_SHA" ]; then
  log "release_tag: REQUEST_SHA='$REQUEST_SHA' has no parent (root commit?), so it cannot be a release-request commit"
  exit 1
fi
if git rev-parse --verify --quiet "${REQUEST_SHA}^2" >/dev/null 2>&1; then
  log "release_tag: the request commit is a merge commit. A release request is one plain, empty commit on top of ${DEFAULT_BRANCH}'s tip."
  exit 1
fi
# EMPTY means the same tree as its parent. The tag goes on the parent, so any change
# carried by the request commit would silently NOT be part of the release.
if [ "$(git rev-parse "${REQUEST_SHA}^{tree}")" != "$(git rev-parse "${PARENT_SHA}^{tree}")" ]; then
  log "release_tag: the request commit is not empty: it changes files relative to its parent. Those changes would NOT be part of the tagged commit. Merge them to ${DEFAULT_BRANCH} first, then request again with an empty commit."
  exit 1
fi

# ---------- 3. idempotency: an existing, correct tag is success ----------
# Checked BEFORE the staleness check below: re-running the workflow for a version
# that already succeeded must stay green even if master has moved on since (an
# unrelated later commit on master must not turn a harmless re-run red).
git fetch origin '+refs/tags/*:refs/tags/*' --quiet 2>/dev/null || true
EXISTING="$(git rev-parse --verify --quiet "refs/tags/${VERSION}^{commit}" 2>/dev/null || true)"
if [ -n "$EXISTING" ]; then
  # Compared with the request commit's OWN parent, not with master's current tip:
  # a tag on the commit THIS request pointed at is still correct after master moved.
  if [ "$EXISTING" = "$PARENT_SHA" ]; then
    # "Exists" has to mean "exists on origin". In a clone where an earlier push of
    # this tag failed, the tag is local only, and calling that success would report
    # a release that never started.
    if ! git ls-remote --exit-code --tags origin "refs/tags/${VERSION}" >/dev/null 2>&1; then
      log "release_tag: tag ${VERSION} exists in this clone but NOT on origin (an earlier push failed?). Delete the local tag (git tag -d ${VERSION}) and run again."
      exit 1
    fi
    log "release_tag: tag ${VERSION} already exists and points at exactly the commit this request pointed at ($EXISTING): nothing to do (idempotent)."
    echo "TAG: ${VERSION} (${EXISTING:0:12}, already present)"
    exit 0
  fi
  log "release_tag: tag ${VERSION} ALREADY exists but points at ${EXISTING}, not at the commit this request points at. REFUSING to move an existing tag (a PyPI version is immutable; fix a broken release with a patch bump and a new tag)."
  exit 1
fi

# ---------- 4. the request must sit on master's CURRENT tip ----------
if [ "$PARENT_SHA" != "$DEFAULT_SHA" ]; then
  log "release_tag: the request commit's parent ($PARENT_SHA) is NOT origin/${DEFAULT_BRANCH}'s current tip ($DEFAULT_SHA)."
  log "release_tag: the branch is either stale (${DEFAULT_BRANCH} moved since the request) or carries more than one commit. Rebuild release-request on top of a fresh ${DEFAULT_BRANCH}."
  exit 1
fi

# ---------- 5. the tag must equal the package version at the commit being tagged ----------
# publish-pypi.yml refuses to publish when the built version differs from the tag,
# but by then the tag already exists, and a pushed tag is never moved: the version
# number would be burned. Refuse BEFORE creating the tag instead.
# `|| true` on git show: under pipefail a missing file would otherwise abort the
# script right here with git's raw exit 128, before the explanation below.
PACKAGE_VERSION="$({ git show "${DEFAULT_SHA}:${VERSION_FILE}" 2>/dev/null || true; } \
  | sed -n -E 's/^__version__[[:space:]]*=[[:space:]]*["'"'"']([^"'"'"']+)["'"'"'].*$/\1/p' | head -1)"
if [ -z "$PACKAGE_VERSION" ]; then
  log "release_tag: could not read __version__ from ${VERSION_FILE} at ${DEFAULT_SHA}"
  exit 1
fi
if [ "v${PACKAGE_VERSION}" != "$VERSION" ]; then
  log "release_tag: requested ${VERSION}, but ${VERSION_FILE} at origin/${DEFAULT_BRANCH}'s tip says __version__ = \"${PACKAGE_VERSION}\". Bump the version (and CHANGELOG.md) on ${DEFAULT_BRANCH} first: see docs/RELEASING.md."
  exit 1
fi

if [ "$DRY_RUN" = "true" ]; then
  log "release_tag: dry run: would create annotated tag ${VERSION} on ${DEFAULT_SHA} (origin/${DEFAULT_BRANCH}'s tip) and push it."
  echo "TAG: (dry run) ${VERSION} would point at ${DEFAULT_SHA:0:12}"
  exit 0
fi

# ---------- 6. create and push the annotated tag ON MASTER'S TIP ----------
# An ANNOTATED tag is a real git object with a tagger, and the tagger is git's
# COMMITTER identity. `git tag -a` fails hard (exit 128, "unable to auto-detect email
# address") without one, and an Actions runner's auto-detected identity (a
# domainless hostname) is exactly that situation. Set through the environment, so
# nothing depends on the clone's git config.
export GIT_COMMITTER_NAME="github-actions[bot]"
export GIT_COMMITTER_EMAIL="41898282+github-actions[bot]@users.noreply.github.com"
git tag -a "$VERSION" -m "Release ${VERSION}" "$DEFAULT_SHA"
git push origin "refs/tags/${VERSION}"
log "release_tag: tag ${VERSION} created on ${DEFAULT_SHA} (origin/${DEFAULT_BRANCH}'s tip) and pushed."
echo "TAG: ${VERSION} (${DEFAULT_SHA:0:12})"
