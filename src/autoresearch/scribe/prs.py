"""The last pull requests merged in a GitHub repository by people, fetched with gh."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

PR_FIELDS = "number,title,body,author,mergedAt,url"
# gh lists merged pull requests by creation, not by merge time, so it fetches more
# than it keeps and sorts locally.
POOL_FACTOR = 3


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
class MergedPR:
    number: int
    title: str
    body: str
    author: str
    merged_at: str
    url: str


def is_bot(author: dict[str, Any]) -> bool:
    login = str(author.get("login", ""))
    return bool(author.get("is_bot")) or login.endswith("[bot]") or login.startswith("app/")


def newest_by_people(raw: list[dict[str, Any]], n: int) -> list[MergedPR]:
    kept = [item for item in raw if item.get("mergedAt") and not is_bot(item.get("author") or {})]
    kept.sort(key=lambda item: str(item["mergedAt"]), reverse=True)
    return [
        MergedPR(
            number=int(item["number"]),
            title=str(item.get("title") or ""),
            body=str(item.get("body") or ""),
            author=str((item.get("author") or {}).get("login", "")),
            merged_at=str(item["mergedAt"]),
            url=str(item.get("url") or ""),
        )
        for item in kept[:n]
    ]


def fetch_merged_prs(gh: GhRunner, repo: str, n: int) -> list[MergedPR]:
    pool = POOL_FACTOR * n
    args = ["pr", "list", "--repo", repo, "--state", "merged", "--limit", str(pool)]
    raw = json.loads(gh.run([*args, "--json", PR_FIELDS]))
    prs = newest_by_people(raw, n)
    if len(prs) < n:
        raise GhError(f"only {len(prs)} merged pull requests by people among the last {pool}")
    return prs


def github_repo(url: str) -> str:
    """``owner/name`` from a GitHub URL."""
    m = re.match(r"^(?:https?://github\.com/|git@github\.com:)([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if not m:
        raise ValueError(f"not a GitHub URL: {url}; pass --repo owner/name")
    return f"{m.group(1)}/{m.group(2)}"


def save_prs(path: Path, prs: list[MergedPR]) -> None:
    path.write_text(json.dumps([asdict(p) for p in prs], indent=2) + "\n")


def load_prs(path: Path) -> list[MergedPR]:
    return [MergedPR(**d) for d in json.loads(path.read_text())]
