"""``tools/zenodo_backfill.py`` must add the missing releases to the Zenodo concept record in
the right order, with the integration's own metadata shape, and never touch what is there.

The tool talks to two live services, so these tests pin the parts that can go wrong
without a network: which tags are chosen and in what order (the concept DOI resolves to
the version published LAST, so order is the whole point), what each record's metadata
looks like, that an existing record's read-only fields are not sent back on edit, and
the exact call sequence against a fake deposit API, including discarding a draft left
behind by an interrupted run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "zenodo_backfill.py"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("zenodo_backfill", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ZENODO_JSON = {
    "description": "Respiratory mechanics analysis.",
    "license": {"id": "gpl-3.0-or-later"},
    "upload_type": "software",
    "creators": [{"name": "Walsted, Emil Ingerslev", "orcid": "0000-0002-6640-7175",
                  "affiliation": "Bispebjerg Hospital, Copenhagen, Denmark"}],
    "keywords": ["respiratory mechanics"],
}


def _release(tag, published, name=None, prerelease=False, draft=False):
    return {"tag_name": tag, "name": name or f"RespMech {tag}", "published_at": published,
            "prerelease": prerelease, "draft": draft,
            "html_url": f"https://github.com/emilwalsted/respmech/releases/tag/{tag}"}


RELEASES = [
    _release("v2.5.0", "2026-10-16T15:40:04Z"),
    _release("v1.0.1", "2023-11-08T20:03:47Z", name="Version 1.0.1"),
    _release("v2.0.0", "2026-07-11T22:25:14Z", prerelease=True),
    _release("v0.9", "2019-07-07T14:20:06Z", name="Pre-release beta"),
    _release("v0.9a", "2019-07-07T14:22:56Z", name="Testing"),
    _release("v2.3.0", "2026-07-22T06:44:32Z"),
]


# ------------------------------------------------------------------ choosing the tags

def test_missing_releases_come_out_oldest_first_and_skip_what_zenodo_already_has(tool):
    existing = tool.existing_tags([
        {"version": "v1.0.1", "tree": "https://github.com/emilwalsted/respmech/tree/v1.0.1"},
        {"version": None, "tree": "https://github.com/emilwalsted/respmech/tree/v0.9a"},
    ])
    todo, left_out = tool.plan_tags(RELEASES, existing)
    assert [r["tag_name"] for r in todo] == ["v2.0.0", "v2.3.0", "v2.5.0"]
    assert dict(left_out) == {
        "v0.9": tool.SKIP["v0.9"],
        "v0.9a": "already on Zenodo",
        "v1.0.1": "already on Zenodo",
    }


def test_a_tag_is_recognised_from_the_tree_link_when_the_record_has_no_version_field(tool):
    assert tool.existing_tags([{"version": None, "tree": "https://github.com/emilwalsted/respmech/tree/v0.95-beta"}]) == {"v0.95-beta"}


def test_only_restricts_without_reordering(tool):
    todo, _ = tool.plan_tags(RELEASES, set(), only={"v2.5.0", "v2.0.0"})
    assert [r["tag_name"] for r in todo] == ["v2.0.0", "v2.5.0"]


# ------------------------------------------------------------------------- metadata

def test_metadata_mirrors_the_integration_and_zenodo_json(tool):
    meta = tool.build_metadata(_release("v2.0.0", "2026-07-11T22:25:14Z"), ZENODO_JSON, today="2026-10-26")
    assert meta["title"] == "emilwalsted/respmech: RespMech v2.0.0"
    assert meta["version"] == "v2.0.0"
    assert meta["publication_date"] == "2026-07-11"
    assert meta["upload_type"] == "software"
    assert meta["access_right"] == "open"
    assert meta["license"] == "gpl-3.0-or-later"
    assert meta["creators"] == ZENODO_JSON["creators"]
    assert meta["keywords"] == ["respiratory mechanics"]
    assert meta["related_identifiers"][0] == {
        "identifier": "https://github.com/emilwalsted/respmech/tree/v2.0.0",
        "relation": "isSupplementTo", "resource_type": "software", "scheme": "url",
    }
    assert "Respiratory mechanics analysis." in meta["description"]
    assert "releases/tag/v2.0.0" in meta["description"]
    assert "2026-10-26" in meta["notes"] and "2026-07-11" in meta["notes"]


def test_corrected_metadata_drops_read_only_fields_and_is_none_when_nothing_changes(tool):
    current = {
        "doi": "10.5281/zenodo.10085498", "prereserve_doi": {"doi": "x"},
        "title": "emilwalsted/respmech: Version 1.0.1", "version": "v1.0.1",
        "license": {"id": "cc-by-4.0"}, "creators": [{"name": "Emil S. Walsted"}],
        "publication_date": "2023-11-08", "description": "<p>Fixed PTP.</p>",
        "communities": [{"identifier": "x"}],
    }
    target = tool.corrected_metadata(current, _release("v1.0.1", "2023-11-08T20:03:47Z", name="Version 1.0.1"), ZENODO_JSON)
    assert "doi" not in target and "prereserve_doi" not in target
    assert target["license"] == "gpl-3.0-or-later"
    assert target["creators"] == ZENODO_JSON["creators"]
    assert target["keywords"] == ["respiratory mechanics"]
    assert target["communities"] == [{"identifier": "x"}]        # untouched fields survive
    assert target["description"] == "<p>Fixed PTP.</p>"
    assert tool.corrected_metadata(dict(current, **target), _release("v1.0.1", "2023-11-08T20:03:47Z", name="Version 1.0.1"), ZENODO_JSON) is None


def test_the_release_name_on_github_wins_over_a_stale_zenodo_title(tool):
    current = {"title": "emilwalsted/respmech: Includes savinf processed data", "version": "v0.96-beta",
               "license": "gpl-3.0-or-later", "creators": ZENODO_JSON["creators"], "keywords": ["k"]}
    target = tool.corrected_metadata(current, _release("v0.96-beta", "2021-07-03T19:59:37Z", name="Includes saving processed data"), ZENODO_JSON)
    assert target["title"] == "emilwalsted/respmech: Includes saving processed data"


# ---------------------------------------------------------------------------- token

def test_env_file_parsing_strips_quotes_and_ignores_comments(tool, tmp_path):
    env = tmp_path / "zenodo.env"
    env.write_text("# tokens\nZENODO_TOKEN='abc'\n\nZENODO_SANDBOX_TOKEN = \"def\"\nNOEQUALS\n", encoding="utf-8")
    assert tool.load_env_file(str(env)) == {"ZENODO_TOKEN": "abc", "ZENODO_SANDBOX_TOKEN": "def"}
    assert tool.token_for("sandbox", environ={}, env_file=str(env)) == "def"
    assert tool.token_for("sandbox", environ={"ZENODO_SANDBOX_TOKEN": "env-wins"}, env_file=str(env)) == "env-wins"
    with pytest.raises(tool.BackfillError, match="ZENODO_TOKEN"):
        tool.token_for("production", environ={}, env_file=str(tmp_path / "missing.env"))


# ----------------------------------------------------------------- the deposit flow

class FakeDepositApi:
    """Just enough of Zenodo's deposit API to record what the tool does, in order."""

    def __init__(self, base, *, latest_id=100, pending_draft=None):
        self.base = base
        self.calls = []
        self.latest_id = latest_id
        self.pending_draft = pending_draft
        self.drafts = {}
        self.published = []
        self._next = 200

    def request(self, method, url, *, json_body=None, data=None, headers=None, retries=8):
        self.calls.append((method, url.replace(self.base, ""), json_body if json_body is not None else (len(data) if data else None)))
        path = url.replace(self.base, "")
        dep = "/api/deposit/depositions/"
        if method == "POST" and path == dep.rstrip("/"):
            return self._new_draft(fresh=True)
        if method == "GET" and path.startswith(dep) and path.endswith(f"{self.latest_id}"):
            # Measured on the sandbox: a published deposition's latest_draft points at ITSELF
            # unless a newer draft exists.
            draft = self.pending_draft if self.pending_draft is not None else self.latest_id
            return 200, {"id": self.latest_id, "state": "done", "submitted": True,
                         "links": {"latest_draft": f"{self.base}{dep}{draft}"}}
        if method == "DELETE" and "/files/" in path:
            draft_id = int(path.split("/")[4])
            self.drafts[draft_id]["files"] = []
            return 204, b""
        if method == "DELETE" and path.startswith(dep) and path.rstrip("/").split("/")[-1] == str(self.pending_draft):
            self.pending_draft = None
            return 204, b""
        if method == "DELETE" and path.startswith(dep):
            raise AssertionError(f"tried to delete a published deposition: {path}")
        if method == "POST" and path.endswith("/actions/newversion"):
            status, draft = self._new_draft(fresh=False)
            self.pending_draft = draft["id"]
            return status, {"id": self.latest_id, "links": {"latest_draft": f"{self.base}{dep}{draft['id']}"}}
        if method == "GET" and path.startswith(dep):
            draft_id = int(path.rstrip("/").split("/")[-1])
            if draft_id not in self.drafts and draft_id == self.pending_draft:
                return 200, {"id": draft_id, "state": "unsubmitted", "submitted": False, "links": {}, "files": []}
            return 200, self.drafts[draft_id]
        if method == "DELETE" and "/files/" in path:
            draft_id = int(path.split("/")[4])
            self.drafts[draft_id]["files"] = []
            return 204, b""
        if method == "PUT" and "/api/files/" in path:
            return 201, {"checksum": "md5:" + hashlib.md5(data).hexdigest(), "key": path.split("/")[-1]}
        if method == "PUT" and path.startswith(dep):
            draft_id = int(path.rstrip("/").split("/")[-1])
            self.drafts[draft_id]["metadata"] = json_body["metadata"]
            return 200, self.drafts[draft_id]
        if method == "POST" and path.endswith("/actions/publish"):
            draft_id = int(path.split("/")[4])
            self.published.append(draft_id)
            self.latest_id = draft_id
            self.pending_draft = None
            return 202, {"id": draft_id, "doi": f"10.5072/zenodo.{draft_id}", "conceptrecid": 99,
                         "metadata": self.drafts[draft_id]["metadata"]}
        raise AssertionError(f"unexpected call {method} {path}")

    def get_json(self, url):
        return self.request("GET", url)[1]

    def _new_draft(self, *, fresh):
        draft_id = self._next
        self._next += 1
        self.drafts[draft_id] = {
            "id": draft_id,
            "links": {"bucket": f"{self.base}/api/files/bucket-{draft_id}"},
            "files": [] if fresh else [{"id": "inherited", "links": {"self": f"{self.base}/api/deposit/depositions/{draft_id}/files/inherited"}}],
            "metadata": {},
        }
        return 201, self.drafts[draft_id]


@pytest.fixture
def wired(tool, monkeypatch, tmp_path):
    """The tool with every network edge replaced; returns the fake deposit API."""
    base = tool.HOSTS["sandbox"]
    api = FakeDepositApi(base)
    existing = [{"id": 50, "doi": "10.5072/zenodo.50", "version": None, "publication_date": "2019-07-07",
                 "title": "emilwalsted/respmech: Testing", "tree": "https://github.com/emilwalsted/respmech/tree/v0.9a"},
                {"id": 100, "doi": "10.5072/zenodo.100", "version": "v1.0.1", "publication_date": "2023-11-08",
                 "title": "emilwalsted/respmech: Version 1.0.1", "tree": "https://github.com/emilwalsted/respmech/tree/v1.0.1"}]

    monkeypatch.setattr(tool, "Http", lambda token=None, **kw: api)
    monkeypatch.setattr(tool, "token_for", lambda host, **kw: "fake-token")
    monkeypatch.setattr(tool, "load_zenodo_json", lambda root=None: ZENODO_JSON)
    monkeypatch.setattr(tool, "github_releases", lambda http, repo=None: list(RELEASES))
    monkeypatch.setattr(tool, "zenodo_versions", lambda http, b, concept: list(existing))

    def fake_zipball(http, repo, tag, dest_dir):
        path = Path(dest_dir) / tool.zip_name(repo, tag)
        path.write_bytes(b"PK\x03\x04" + tag.encode() * 400)
        return str(path)
    monkeypatch.setattr(tool, "download_zipball", fake_zipball)
    monkeypatch.setattr(tool.time, "sleep", lambda s: None)
    return api


def test_backfill_publishes_the_missing_releases_oldest_first_with_one_draft_at_a_time(tool, wired):
    rc = tool.main(["backfill", "--sandbox", "--concept", "99", "--pause", "0"])
    assert rc == 0
    api = wired
    assert len(api.published) == 3
    titles = [api.drafts[d]["metadata"]["title"] for d in api.published]
    assert titles == ["emilwalsted/respmech: RespMech v2.0.0",
                      "emilwalsted/respmech: RespMech v2.3.0",
                      "emilwalsted/respmech: RespMech v2.5.0"]
    versions = [api.drafts[d]["metadata"]["version"] for d in api.published]
    assert versions == ["v2.0.0", "v2.3.0", "v2.5.0"]          # the last published is the newest
    # Per version: newversion on the then-latest id, inherited file dropped, upload,
    # metadata, publish — and the next version chains on the record just published.
    seq = [(m, p) for m, p, _ in api.calls]
    first = seq.index(("POST", "/api/deposit/depositions/100/actions/newversion"))
    assert ("DELETE", "/api/deposit/depositions/200/files/inherited") in seq[first:]
    assert ("PUT", "/api/files/bucket-200/respmech-v2.0.0.zip") in seq[first:]
    assert seq.index(("PUT", "/api/deposit/depositions/200")) < seq.index(("POST", "/api/deposit/depositions/200/actions/publish"))
    assert ("POST", "/api/deposit/depositions/200/actions/newversion") in seq   # v2.3.0 chains on v2.0.0's record
    assert ("POST", "/api/deposit/depositions/201/actions/newversion") in seq   # v2.5.0 chains on v2.3.0's record
    assert api.drafts[202]["files"] == []


def test_a_draft_left_by_an_interrupted_run_is_discarded_first(tool, wired):
    wired.pending_draft = 150
    rc = tool.main(["backfill", "--sandbox", "--concept", "99", "--only", "v2.5.0", "--pause", "0"])
    assert rc == 0
    seq = [(m, p) for m, p, _ in wired.calls]
    assert seq.index(("DELETE", "/api/deposit/depositions/150")) < seq.index(("POST", "/api/deposit/depositions/100/actions/newversion"))
    assert wired.published == [200]


def test_nothing_to_do_is_not_an_error(tool, wired):
    rc = tool.main(["backfill", "--sandbox", "--concept", "99", "--only", "v1.0.1", "--pause", "0"])
    assert rc == 0
    assert wired.published == []


def test_production_refuses_to_publish_without_a_yes(tool, wired, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    rc = tool.main(["backfill", "--pause", "0"])
    assert rc == 2
    assert wired.published == []


# ------------------------------------------------------------------------------ http

def test_request_retries_a_429_then_succeeds(tool, monkeypatch):
    attempts = []

    class Resp(io.BytesIO):
        status = 200
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=0):
        attempts.append(req.get_method())
        if len(attempts) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests",
                                         {"Retry-After": "1", "Content-Type": "application/json"}, io.BytesIO(b"{}"))
        return Resp(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(tool.urllib.request, "urlopen", fake_urlopen)
    slept = []
    http = tool.Http("t", sleep=slept.append, log=lambda *a: None)
    status, body = http.request("GET", "https://zenodo.org/api/records/1")
    assert (status, body) == (200, {"ok": True})
    assert attempts == ["GET", "GET"] and slept == [1.0]


def test_a_non_retryable_error_carries_zenodo_s_message(tool, monkeypatch):
    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {"Content-Type": "application/json"},
                                     io.BytesIO(b'{"message": "Validation error.", "errors": [{"field": "metadata.license"}]}'))
    monkeypatch.setattr(tool.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(tool.BackfillError, match="HTTP 400: Validation error.*metadata.license"):
        tool.Http("t", log=lambda *a: None).request("PUT", "https://zenodo.org/api/deposit/depositions/1", json_body={})
