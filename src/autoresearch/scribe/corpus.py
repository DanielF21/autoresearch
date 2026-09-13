"""The style corpus: a target repo's recent merged PRs by people, frozen to disk.

Fetched once through ``gh`` and never refetched into the same directory, so every
body the writer imitates and every body a judge compares against is the same
bytes on every dev run. The manifest carries a sha256 per file and loading
verifies it.

Split, seeded, into exemplars and holdout. The writer sees exemplars only. The
discrimination judge and the blind rank packets use holdout only. A body the
writer was shown is never the body its draft is judged against.
"""

from __future__ import annotations

import json
import random
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from autoresearch.scribe.layout import (
    read_json,
    sha256_bytes,
    stamp,
    write_json_once,
    write_text_once,
)

PR_FIELDS = "number,title,body,author,labels,additions,deletions,files,mergedAt,url"
MANIFEST = "manifest.json"
TEMPLATE_FILE = "template.md"


class GhError(RuntimeError):
    pass


class GhRunner(Protocol):
    def run(self, args: list[str]) -> str: ...


class SubprocessGh:
    def __init__(self, timeout: float = 120.0) -> None:
        self._timeout = timeout

    def run(self, args: list[str]) -> str:
        r = subprocess.run(
            ["gh", *args], capture_output=True, text=True, timeout=self._timeout, check=False
        )
        if r.returncode != 0:
            raise GhError(f"gh {' '.join(args[:3])} failed: {r.stderr.strip()[-500:]}")
        return r.stdout


@dataclass(frozen=True)
class CorpusPR:
    number: int
    title: str
    body: str
    author: str
    labels: tuple[str, ...]
    additions: int
    deletions: int
    files: tuple[str, ...]
    merged_at: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "body": self.body,
            "author": self.author,
            "labels": list(self.labels),
            "additions": self.additions,
            "deletions": self.deletions,
            "files": list(self.files),
            "merged_at": self.merged_at,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CorpusPR:
        return cls(
            number=int(d["number"]),
            title=str(d["title"]),
            body=str(d.get("body") or ""),
            author=str(d.get("author", "")),
            labels=tuple(str(x) for x in d.get("labels", [])),
            additions=int(d.get("additions", 0)),
            deletions=int(d.get("deletions", 0)),
            files=tuple(str(x) for x in d.get("files", [])),
            merged_at=str(d.get("merged_at", "")),
            url=str(d.get("url", "")),
        )


def is_bot(author: dict[str, Any], exclude: tuple[str, ...]) -> bool:
    login = str(author.get("login", ""))
    return (
        bool(author.get("is_bot"))
        or login.endswith("[bot]")
        or login.startswith("app/")
        or login in exclude
    )


def select_prs(
    raw: list[dict[str, Any]], n: int, exclude: tuple[str, ...]
) -> tuple[list[CorpusPR], list[dict[str, Any]]]:
    """Newest ``n`` merged PRs by people. ``gh`` orders by creation, so sort by merge time."""
    excluded: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for item in raw:
        author = item.get("author") or {}
        if is_bot(author, exclude):
            excluded.append({"number": item.get("number"), "author": author.get("login", "")})
        elif item.get("mergedAt"):
            kept.append(item)
    kept.sort(key=lambda item: str(item["mergedAt"]), reverse=True)
    prs = [
        CorpusPR(
            number=int(item["number"]),
            title=str(item.get("title") or ""),
            body=str(item.get("body") or ""),
            author=str((item.get("author") or {}).get("login", "")),
            labels=tuple(str(lbl.get("name", "")) for lbl in item.get("labels") or []),
            additions=int(item.get("additions") or 0),
            deletions=int(item.get("deletions") or 0),
            files=tuple(str(f.get("path", "")) for f in item.get("files") or []),
            merged_at=str(item["mergedAt"]),
            url=str(item.get("url") or ""),
        )
        for item in kept[:n]
    ]
    return prs, excluded


def split(numbers: list[int], holdout_n: int, seed: int) -> tuple[list[int], list[int]]:
    """Seeded split into (exemplar, holdout), each in the input order."""
    shuffled = list(numbers)
    random.Random(seed).shuffle(shuffled)
    holdout = set(shuffled[:holdout_n])
    return [x for x in numbers if x not in holdout], [x for x in numbers if x in holdout]


def fetch_corpus(
    gh: GhRunner,
    *,
    repo: str,
    n: int,
    pool: int,
    exclude_authors: tuple[str, ...],
    holdout_n: int,
    seed: int,
    template_path: str,
    out_root: Path,
    now: datetime | None = None,
    stats: Callable[[list[CorpusPR]], dict[str, Any]] | None = None,
) -> Path:
    args = [
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "merged",
        "--limit",
        str(pool),
        "--json",
        PR_FIELDS,
    ]
    raw_text = gh.run(args)
    raw = json.loads(raw_text)
    prs, excluded = select_prs(raw, n, exclude_authors)
    if len(prs) < n:
        raise GhError(f"only {len(prs)} merged PRs by people in a pool of {pool}; raise the pool")
    try:
        template = gh.run(
            [
                "api",
                "-H",
                "Accept: application/vnd.github.raw",
                f"repos/{repo}/contents/{template_path}",
            ]
        )
    except GhError:
        template = ""

    raw_hash = sha256_bytes(raw_text.encode())
    out = out_root / repo.replace("/", "__") / f"{stamp(now)}-{raw_hash[:8]}"
    if out.exists():
        raise FileExistsError(f"corpus already exists: {out}")
    files: dict[str, bytes] = {
        "raw/pr_list.json": raw_text.encode(),
        TEMPLATE_FILE: template.encode(),
    }
    for pr in prs:
        files[f"prs/{pr.number}.json"] = (json.dumps(pr.to_dict(), indent=2) + "\n").encode()
    exemplar, holdout = split([p.number for p in prs], holdout_n, seed)
    files["split.json"] = (
        json.dumps({"seed": seed, "exemplar": exemplar, "holdout": holdout}, indent=2) + "\n"
    ).encode()
    if stats is not None:
        files["stats.json"] = (json.dumps(stats(prs), indent=2, sort_keys=True) + "\n").encode()
    for rel, data in files.items():
        write_text_once(out / rel, data.decode())
    write_json_once(
        out / MANIFEST,
        {
            "repo": repo,
            "command": ["gh", *args],
            "fetched_at": stamp(now),
            "pool": pool,
            "n": n,
            "excluded_bots": excluded,
            "template_path": template_path,
            "template_found": bool(template),
            "sha256": {rel: sha256_bytes(data) for rel, data in sorted(files.items())},
        },
    )
    return out


@dataclass(frozen=True)
class Corpus:
    dir: Path
    repo: str
    prs: tuple[CorpusPR, ...]
    exemplars: tuple[CorpusPR, ...]
    holdout: tuple[CorpusPR, ...]
    template: str
    stats: dict[str, Any]
    hash: str

    @property
    def id(self) -> str:
        return self.dir.name


class CorpusError(RuntimeError):
    pass


def load_corpus(corpus_dir: Path) -> Corpus:
    manifest = read_json(corpus_dir / MANIFEST)
    for rel, digest in manifest["sha256"].items():
        path = corpus_dir / rel
        if not path.exists() or sha256_bytes(path.read_bytes()) != digest:
            raise CorpusError(f"corpus file changed or missing since fetch: {path}")
    by_number = {
        pr.number: pr
        for pr in (
            CorpusPR.from_dict(read_json(p)) for p in sorted((corpus_dir / "prs").glob("*.json"))
        )
    }
    order = sorted(by_number.values(), key=lambda pr: pr.merged_at, reverse=True)
    sp = read_json(corpus_dir / "split.json")
    stats_path = corpus_dir / "stats.json"
    return Corpus(
        dir=corpus_dir,
        repo=str(manifest["repo"]),
        prs=tuple(order),
        exemplars=tuple(by_number[n] for n in sp["exemplar"]),
        holdout=tuple(by_number[n] for n in sp["holdout"]),
        template=(corpus_dir / TEMPLATE_FILE).read_text(),
        stats=read_json(stats_path) if stats_path.exists() else {},
        hash=sha256_bytes(json.dumps(manifest["sha256"], sort_keys=True).encode()),
    )


def render_pr(pr: CorpusPR) -> str:
    return f"# {pr.title}\n\n{pr.body.strip()}\n"
