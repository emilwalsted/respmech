#!/usr/bin/env python3
"""Give every GitHub release of RespMech its own Zenodo DOI, including the releases the
GitHub↔Zenodo integration missed.

    python3 tools/zenodo_backfill.py plan                  # GitHub releases vs. Zenodo versions; what is missing
    python3 tools/zenodo_backfill.py backfill --sandbox    # rehearse on sandbox.zenodo.org (fake DOIs, own token)
    python3 tools/zenodo_backfill.py backfill              # archive the missing releases on zenodo.org, oldest first
    python3 tools/zenodo_backfill.py fix-metadata          # licence, creator, title on the records that already exist

Exit 0 = done (or nothing to do). Exit 1 = a step failed; what was published before the
failure stays published, and a rerun picks up where it stopped. Exit 2 = bad arguments,
missing token, or a resource that could not be read.

WHY
---
"Every released version has its own DOI" (README) is only true while the GitHub↔Zenodo
integration is awake. It archived v0.9a (2019), the 0.9x betas (2021) and v1.0.x (2023),
then went quiet: Zenodo's stored GitHub authorisation expired, every release event it
received after that failed, and no v2.x release was archived. A DOI cannot be minted
retroactively by the integration (it only sees new release events), but Zenodo's REST API
can add versions to the existing concept record with the ORIGINAL publication dates, which
is what ``backfill`` does. ``fix-metadata`` brings the seven records the integration made
in line with ``.zenodo.json`` (GPL-3.0-or-later instead of CC-BY/"other", the creator's
full name, ORCID and affiliation, the current release title), which the integration never
revisits on its own.

HOW
---
* Reads nothing secret from the repository. The token comes from the environment
  (``ZENODO_TOKEN`` for zenodo.org, ``ZENODO_SANDBOX_TOKEN`` for the sandbox) or from
  ``~/.config/zenodo.env`` (``KEY=value`` lines, mode 600). A token needs the scopes
  ``deposit:write`` and ``deposit:actions``; its value is never printed.
* Uses Zenodo's documented deposit API (developers.zenodo.org): ``newversion`` on the
  latest record of the concept, replace the inherited file with GitHub's own source
  archive of the tag, set the metadata, publish. One unpublished version draft may exist
  per concept, so the versions go one at a time, oldest first — Zenodo's concept DOI
  resolves to the most recently PUBLISHED version, not to the newest date, so the order
  is what keeps the badge pointing at the newest release.
* Metadata mirrors what the integration itself writes (title ``owner/repo: <release
  name>``, version = tag, publication date = the GitHub release's date, a related
  identifier to the tag's tree), with creators, licence, keywords and description from
  ``.zenodo.json``, plus a note saying the record was archived after the fact.
* Idempotent: a tag whose version already exists under the concept is skipped, and a
  draft left behind by an interrupted run is discarded before a new one is made.

Zenodo records are permanent (an owner can delete one within 30 days; later only
support can). Rehearse on the sandbox first: ``backfill --sandbox`` creates a sandbox
record on its first run and chains the rest onto it, so the whole sequence is exercised
with fake 10.5072 DOIs before anything touches zenodo.org.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "emilwalsted/respmech"
CONCEPT_RECID = 3270825
HOSTS = {"production": "https://zenodo.org", "sandbox": "https://sandbox.zenodo.org"}
TOKEN_VARS = {"production": "ZENODO_TOKEN", "sandbox": "ZENODO_SANDBOX_TOKEN"}
ENV_FILE = os.path.join(os.path.expanduser("~"), ".config", "zenodo.env")
GITHUB_API = "https://api.github.com"
SITE_URL = "https://www.respmech.dk/"
USER_AGENT = "respmech-zenodo-backfill (https://github.com/emilwalsted/respmech)"
# Tags that deliberately get no record of their own, with the reason shown in `plan`.
SKIP = {
    "v0.9": "same source tree as v0.9a, which the integration archived two minutes later",
}
# The legacy deposit API's GET returns these alongside the writable fields; they must not
# be sent back in a PUT.
READ_ONLY_METADATA = {"doi", "prereserve_doi"}


class BackfillError(Exception):
    """A step failed in a way the maintainer has to look at."""


# --------------------------------------------------------------------------- token

def load_env_file(path: str) -> dict[str, str]:
    """``KEY=value`` lines; blank lines and ``#`` comments ignored; quotes stripped."""
    values: dict[str, str] = {}
    if not os.path.isfile(path):
        return values
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key:
                values[key] = value
    return values


def token_for(host_key: str, environ=None, env_file: str = ENV_FILE) -> str:
    var = TOKEN_VARS[host_key]
    environ = os.environ if environ is None else environ
    token = environ.get(var) or load_env_file(env_file).get(var)
    if not token:
        raise BackfillError(
            f"no {var} in the environment or in {env_file}; create a personal access token "
            f"with the scopes deposit:write and deposit:actions on {HOSTS[host_key]}"
        )
    return token


# ---------------------------------------------------------------------------- http

class Http:
    """Thin urllib wrapper: JSON in/out, bearer token, retries on the transient codes."""

    RETRY_STATUS = {409, 429, 500, 502, 503, 504}

    def __init__(self, token: str | None = None, *, sleep=time.sleep, log=print):
        self.token = token
        self.sleep = sleep
        self.log = log

    def request(self, method: str, url: str, *, json_body=None, data: bytes | None = None,
                headers: dict | None = None, retries: int = 8):
        hdrs = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if self.token:
            hdrs["Authorization"] = f"Bearer {self.token}"
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        elif data is not None:
            hdrs["Content-Type"] = "application/octet-stream"
        hdrs.update(headers or {})
        delay = 5.0
        for attempt in range(retries + 1):
            req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    return resp.status, _decode(resp.read(), resp.headers.get("Content-Type", ""))
            except urllib.error.HTTPError as err:
                body = _decode(err.read(), err.headers.get("Content-Type", ""))
                if err.code in self.RETRY_STATUS and attempt < retries:
                    wait = _retry_after(err.headers, delay)
                    self.log(f"    {method} {_short(url)} -> {err.code}; retrying in {wait:.0f} s")
                    self.sleep(wait)
                    delay = min(delay * 2, 120.0)
                    continue
                raise BackfillError(f"{method} {_short(url)} -> HTTP {err.code}: {_message(body)}") from None
            except urllib.error.URLError as err:
                if attempt < retries:
                    self.log(f"    {method} {_short(url)} -> {err.reason}; retrying in {delay:.0f} s")
                    self.sleep(delay)
                    delay = min(delay * 2, 120.0)
                    continue
                raise BackfillError(f"{method} {_short(url)} -> {err.reason}") from None
        raise AssertionError("unreachable")

    def get_json(self, url: str):
        return self.request("GET", url)[1]


def _decode(raw: bytes, content_type: str):
    if "json" in content_type:
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            return raw
    return raw


def _message(body) -> str:
    if isinstance(body, dict):
        msg = body.get("message", "")
        errors = body.get("errors")
        if errors:
            msg += " " + json.dumps(errors)[:500]
        return msg or json.dumps(body)[:500]
    if isinstance(body, bytes):
        return body[:300].decode("utf-8", "replace")
    return str(body)[:500]


def _retry_after(headers, fallback: float) -> float:
    try:
        return float(headers.get("Retry-After", fallback))
    except (TypeError, ValueError):
        return fallback


def _short(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return f"{parts.netloc}{parts.path}"


def _dep_id(url: str) -> str:
    """The deposition id at the end of a deposit-API URL."""
    return urllib.parse.urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]


# -------------------------------------------------------------------------- github

def github_releases(http: Http, repo: str = REPO) -> list[dict]:
    """Every non-draft GitHub release of ``repo``, oldest first."""
    out: list[dict] = []
    page = 1
    while True:
        chunk = http.get_json(f"{GITHUB_API}/repos/{repo}/releases?per_page=100&page={page}")
        if not chunk:
            break
        out.extend(r for r in chunk if not r.get("draft"))
        if len(chunk) < 100:
            break
        page += 1
    return sorted(out, key=lambda r: r["published_at"])


def download_zipball(http: Http, repo: str, tag: str, dest_dir: str) -> str:
    """GitHub's own source archive of the tag, the same bytes the integration archives."""
    path = os.path.join(dest_dir, zip_name(repo, tag))
    # The API answers 302 to codeload.github.com, which urllib follows; it insists on a
    # JSON Accept header even though the redirect target serves a zip.
    _, body = http.request("GET", f"{GITHUB_API}/repos/{repo}/zipball/{urllib.parse.quote(tag)}",
                           headers={"Accept": "application/vnd.github+json"})
    if not isinstance(body, bytes) or len(body) < 1000:
        raise BackfillError(f"zipball for {tag} came back empty")
    with open(path, "wb") as fh:
        fh.write(body)
    return path


def zip_name(repo: str, tag: str) -> str:
    # The integration names its file "owner/repo-<tag>.zip"; a slash is not a usable key in
    # the deposit files API, so the owner is dropped.
    return f"{repo.split('/', 1)[1]}-{tag}.zip"


# -------------------------------------------------------------------------- zenodo

def zenodo_versions(http: Http, base: str, concept_recid: int) -> list[dict]:
    """Every published version under the concept, oldest index first.

    Each entry: id, doi, version, publication_date, title, tree (the GitHub tree URL the
    record is a supplement to, or None).
    """
    _, latest = http.request("GET", f"{base}/api/records/{concept_recid}")
    versions_url = latest.get("links", {}).get("versions")
    if not versions_url:
        raise BackfillError(f"record {concept_recid} has no versions link; is it a concept record?")
    found: list[dict] = []
    page = 1
    while True:
        chunk = http.get_json(f"{versions_url}?size=25&page={page}")
        hits = chunk.get("hits", {}).get("hits", [])
        for hit in hits:
            found.append(_summarise(hit))
        if len(hits) < 25:
            break
        page += 1
    return list(reversed(found))  # the API lists newest first


def _summarise(hit: dict) -> dict:
    meta = hit.get("metadata", {})
    tree = None
    for rel in meta.get("related_identifiers", []) or []:
        ident = rel.get("identifier", "")
        if "/tree/" in ident:
            tree = ident
    return {
        "id": hit.get("id"),
        "doi": hit.get("doi") or meta.get("doi"),
        "version": meta.get("version"),
        "publication_date": meta.get("publication_date"),
        "title": meta.get("title"),
        "tree": tree,
    }


def existing_tags(versions: list[dict], repo: str = REPO) -> set[str]:
    tags = set()
    for v in versions:
        if v.get("version"):
            tags.add(v["version"])
        tree = v.get("tree") or ""
        prefix = f"https://github.com/{repo}/tree/"
        if tree.startswith(prefix):
            tags.add(tree[len(prefix):])
    return tags


def plan_tags(releases: list[dict], existing: set[str], *, skip: dict | None = None,
              only: set[str] | None = None) -> tuple[list[dict], list[tuple[str, str]]]:
    """The releases still to archive, oldest first, and the (tag, reason) pairs left out."""
    skip = SKIP if skip is None else skip
    todo, left_out = [], []
    for rel in sorted(releases, key=lambda r: r["published_at"]):
        tag = rel["tag_name"]
        if only and tag not in only:
            continue
        if tag in existing:
            left_out.append((tag, "already on Zenodo"))
        elif tag in skip:
            left_out.append((tag, skip[tag]))
        else:
            todo.append(rel)
    return todo, left_out


def load_zenodo_json(root: str = ROOT) -> dict:
    with open(os.path.join(root, ".zenodo.json"), encoding="utf-8") as fh:
        return json.load(fh)


def build_metadata(release: dict, zenodo_json: dict, *, repo: str = REPO,
                   today: str | None = None) -> dict:
    """Deposit metadata for one release, in the shape the integration itself writes."""
    tag = release["tag_name"]
    name = release.get("name") or tag
    published = release["published_at"][:10]
    today = today or _dt.date.today().isoformat()
    licence = zenodo_json.get("license", {})
    licence_id = licence.get("id") if isinstance(licence, dict) else licence
    abstract = zenodo_json.get("description", "").strip()
    if not abstract.startswith("<"):
        abstract = f"<p>{abstract}</p>"
    html_url = release.get("html_url") or f"https://github.com/{repo}/releases/tag/{tag}"
    description = (
        f"{abstract}"
        f"<p>Source archive of the tag <code>{tag}</code>, released on {published}. "
        f"Installers and release page: <a href=\"{html_url}\">{html_url}</a>.</p>"
    )
    related = [
        {
            "identifier": f"https://github.com/{repo}/tree/{tag}",
            "relation": "isSupplementTo",
            "resource_type": "software",
            "scheme": "url",
        },
        {
            "identifier": SITE_URL,
            "relation": "isDocumentedBy",
            "resource_type": "other",
            "scheme": "url",
        },
    ]
    for extra in zenodo_json.get("related_identifiers", []) or []:
        if extra.get("identifier") not in {r["identifier"] for r in related}:
            related.append(dict(extra))
    return {
        "upload_type": "software",
        "title": f"{repo}: {name}",
        "version": tag,
        "publication_date": published,
        "description": description,
        "creators": [dict(c) for c in zenodo_json["creators"]],
        "access_right": "open",
        "license": licence_id,
        "keywords": list(zenodo_json.get("keywords", [])),
        "related_identifiers": related,
        "notes": (
            f"Archived on {today} from the GitHub release of {published} with Zenodo's REST API; "
            f"the GitHub–Zenodo integration was not active when this version was released, "
            f"so the record was created after the fact. The archive is GitHub's own source "
            f"archive of the tag."
        ),
    }


def corrected_metadata(current: dict, release: dict | None, zenodo_json: dict, *,
                       repo: str = REPO) -> dict | None:
    """What ``fix-metadata`` would send for an existing record, or None if nothing changes."""
    target = {k: v for k, v in current.items() if k not in READ_ONLY_METADATA}
    licence = zenodo_json.get("license", {})
    target["license"] = licence.get("id") if isinstance(licence, dict) else licence
    target["creators"] = [dict(c) for c in zenodo_json["creators"]]
    if not target.get("keywords"):
        target["keywords"] = list(zenodo_json.get("keywords", []))
    if release is not None:
        target["title"] = f"{repo}: {release.get('name') or release['tag_name']}"
    current_licence = current.get("license")
    if isinstance(current_licence, dict):
        current_licence = current_licence.get("id")
    unchanged = (
        current_licence == target["license"]
        and current.get("creators") == target["creators"]
        and (current.get("keywords") or []) == target["keywords"]
        and current.get("title") == target["title"]
    )
    return None if unchanged else target


class Deposit:
    """The handful of deposit-API calls the backfill needs."""

    def __init__(self, http: Http, base: str, *, log=print):
        self.http = http
        self.base = base
        self.log = log

    def _dep(self, dep_id) -> str:
        return f"{self.base}/api/deposit/depositions/{dep_id}"

    def get(self, dep_id) -> dict:
        return self.http.get_json(self._dep(dep_id))

    def create_fresh(self) -> dict:
        return self.http.request("POST", f"{self.base}/api/deposit/depositions", json_body={})[1]

    def new_version(self, latest_id) -> dict:
        """A draft for the next version, after discarding one an earlier run left behind.

        A published deposition's ``latest_draft`` link points at ITSELF when no draft is
        pending (measured on the sandbox), so a pending draft is one with another id that
        is still unsubmitted; nothing else is ever deleted.
        """
        latest = self.get(latest_id)
        pending = latest.get("links", {}).get("latest_draft")
        if pending and _dep_id(pending) != str(latest_id):
            draft = self.http.get_json(pending)
            if draft.get("state") == "unsubmitted" or draft.get("submitted") is False:
                self.log(f"    discarding the unpublished draft {_short(pending)} from an earlier run")
                self.http.request("DELETE", pending)
        _, original = self.http.request("POST", f"{self._dep(latest_id)}/actions/newversion")
        draft_url = (original or {}).get("links", {}).get("latest_draft")
        if not draft_url or _dep_id(draft_url) == str(latest_id):
            draft_url = self.get(latest_id).get("links", {}).get("latest_draft")
        if not draft_url or _dep_id(draft_url) == str(latest_id):
            raise BackfillError(f"newversion on {latest_id} produced no new draft")
        return self.http.get_json(draft_url)

    def clear_files(self, draft: dict) -> None:
        for f in draft.get("files", []) or []:
            self.http.request("DELETE", f["links"]["self"])

    def upload(self, draft: dict, path: str, filename: str) -> dict:
        bucket = draft.get("links", {}).get("bucket")
        if not bucket:
            raise BackfillError("draft has no bucket link")
        with open(path, "rb") as fh:
            payload = fh.read()
        _, info = self.http.request("PUT", f"{bucket}/{urllib.parse.quote(filename)}", data=payload)
        local = "md5:" + hashlib.md5(payload).hexdigest()
        remote = info.get("checksum") if isinstance(info, dict) else None
        if remote and remote != local:
            raise BackfillError(f"checksum mismatch after upload: {remote} != {local}")
        return info

    def set_metadata(self, dep_id, metadata: dict) -> dict:
        return self.http.request("PUT", self._dep(dep_id), json_body={"metadata": metadata})[1]

    def publish(self, dep_id) -> dict:
        return self.http.request("POST", f"{self._dep(dep_id)}/actions/publish")[1]

    def edit(self, dep_id) -> dict:
        return self.http.request("POST", f"{self._dep(dep_id)}/actions/edit")[1]

    def discard(self, dep_id) -> None:
        try:
            self.http.request("POST", f"{self._dep(dep_id)}/actions/discard", retries=1)
        except BackfillError as err:
            self.log(f"    could not discard the editing session on {dep_id}: {err}")


# ------------------------------------------------------------------------ commands

def _hdr(text: str, log=print) -> None:
    log(f"\n== {text}")


def cmd_plan(args, log=print) -> int:
    base = HOSTS[args.host]
    anon = Http(log=log)
    releases = github_releases(anon, args.repo)
    versions = zenodo_versions(anon, base, args.concept) if args.concept else []
    todo, left_out = plan_tags(releases, existing_tags(versions, args.repo), only=args.only)
    _hdr(f"Zenodo versions under {base}/records/{args.concept}", log)
    for v in versions:
        log(f"  {v['version'] or '?':<12} {v['publication_date']}  {v['doi']}  {v['title']}")
    _hdr("GitHub releases left out", log)
    for tag, reason in left_out:
        log(f"  {tag:<12} {reason}")
    _hdr(f"To archive, in this order ({len(todo)})", log)
    for rel in todo:
        log(f"  {rel['tag_name']:<12} {rel['published_at'][:10]}  {rel.get('name') or ''}")
    return 0


def cmd_backfill(args, log=print) -> int:
    base = HOSTS[args.host]
    token = token_for(args.host)
    anon = Http(log=log)
    auth = Http(token, log=log)
    zenodo_json = load_zenodo_json()
    releases = github_releases(anon, args.repo)
    versions = zenodo_versions(anon, base, args.concept) if args.concept else []
    todo, left_out = plan_tags(releases, existing_tags(versions, args.repo), only=args.only)
    for tag, reason in left_out:
        log(f"  skip {tag}: {reason}")
    if not todo:
        log("Nothing to archive.")
        return 0
    log(f"{len(todo)} release(s) to archive on {base}, oldest first: "
        + ", ".join(r["tag_name"] for r in todo))
    if args.host == "production" and not args.yes:
        answer = input("Publish them on zenodo.org? Records are permanent. [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            log("Not published.")
            return 2
    deposit = Deposit(auth, base, log=log)
    latest_id = versions[-1]["id"] if versions else None
    with tempfile.TemporaryDirectory(prefix="zenodo-backfill-") as tmp:
        for rel in todo:
            tag = rel["tag_name"]
            _hdr(f"{tag} ({rel['published_at'][:10]})", log)
            path = download_zipball(anon, args.repo, tag, tmp)
            log(f"    source archive {os.path.basename(path)}: {os.path.getsize(path):,} bytes")
            draft = deposit.new_version(latest_id) if latest_id else deposit.create_fresh()
            draft_id = draft["id"]
            deposit.clear_files(draft)
            deposit.upload(draft, path, zip_name(args.repo, tag))
            deposit.set_metadata(draft_id, build_metadata(rel, zenodo_json, repo=args.repo))
            published = deposit.publish(draft_id)
            doi = published.get("doi") or published.get("metadata", {}).get("doi")
            latest_id = published.get("id", draft_id)
            log(f"    published: {doi}  ({base}/records/{latest_id})")
            if args.pause:
                time.sleep(args.pause)
    _hdr("Result", log)
    concept = args.concept or deposit.get(latest_id).get("conceptrecid") or latest_id
    for v in zenodo_versions(anon, base, concept):
        log(f"  {v['version'] or '?':<12} {v['publication_date']}  {v['doi']}")
    log(f"\nConcept record: {base}/records/{concept} (resolves to the version published last)")
    return 0


def cmd_fix_metadata(args, log=print) -> int:
    base = HOSTS[args.host]
    token = token_for(args.host)
    anon = Http(log=log)
    auth = Http(token, log=log)
    zenodo_json = load_zenodo_json()
    by_tag = {r["tag_name"]: r for r in github_releases(anon, args.repo)}
    deposit = Deposit(auth, base, log=log)
    changed = 0
    for v in zenodo_versions(anon, base, args.concept):
        if args.only and v["version"] not in args.only:
            continue
        current = deposit.get(v["id"]).get("metadata", {})
        target = corrected_metadata(current, by_tag.get(v["version"]), zenodo_json, repo=args.repo)
        if target is None:
            log(f"  {v['version'] or '?':<12} {v['doi']}  unchanged")
            continue
        log(f"  {v['version'] or '?':<12} {v['doi']}  updating licence/creators/title")
        if args.host == "production" and not args.yes:
            answer = input("    Publish this metadata change? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                log("    skipped")
                continue
        deposit.edit(v["id"])
        try:
            deposit.set_metadata(v["id"], target)
            deposit.publish(v["id"])
        except BackfillError:
            # Leave the published record as it was instead of in an open editing session.
            deposit.discard(v["id"])
            raise
        changed += 1
        if args.pause:
            time.sleep(args.pause)
    log(f"\n{changed} record(s) updated.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn in (("plan", cmd_plan), ("backfill", cmd_backfill), ("fix-metadata", cmd_fix_metadata)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--sandbox", dest="host", action="store_const", const="sandbox",
                       default="production", help="use sandbox.zenodo.org and ZENODO_SANDBOX_TOKEN")
        s.add_argument("--repo", default=REPO)
        s.add_argument("--concept", type=int, default=None,
                       help=f"concept record id (default {CONCEPT_RECID} on zenodo.org; none on the "
                            "sandbox, where the first release creates a record)")
        s.add_argument("--only", nargs="*", default=None, metavar="TAG",
                       help="restrict to these tags")
        s.add_argument("--pause", type=float, default=5.0,
                       help="seconds between published versions (default 5)")
        s.add_argument("--yes", action="store_true", help="do not ask before publishing on zenodo.org")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.concept is None and args.host == "production":
        args.concept = CONCEPT_RECID
    if args.only is not None:
        args.only = set(args.only)
    try:
        return args.fn(args)
    except BackfillError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
