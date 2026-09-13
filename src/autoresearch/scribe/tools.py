"""Read only tools for Scribe models, confined to named roots.

A path is ``root:relative/path``. A directory root is a real tree on disk, and a
path resolves inside it or is refused, symlinks included. A virtual root is an
explicit allowlist of files: the run directory is exposed this way, so a model can
read a patch or a measurement but can never ask for ``config.toml``.

No tool runs a shell. The two git tools build their argument lists here, take no
flags from the model, refuse anything that starts with ``-``, and only show
history reachable from the base commit.
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoresearch.scribe.repo import GitError, GitRunner

MAX_OUTPUT = 12_000
READ_LINES = 400
MAX_FILE_BYTES = 2_000_000
GREP_MAX_MATCHES = 200
LIST_MAX_ENTRIES = 500
HEX_REV = re.compile(r"^[0-9a-f]{7,40}$")


class ToolError(Exception):
    """A bad request. Returned to the model as text, never raised out of ``execute``."""


def clip(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} characters clipped] ...\n{text[-half:]}"


@dataclass(frozen=True)
class ToolResult:
    text: str
    submitted: dict[str, Any] | None = None


@dataclass(frozen=True)
class Roots:
    dirs: dict[str, Path] = field(default_factory=dict)
    virtual: dict[str, dict[str, Path]] = field(default_factory=dict)

    def names(self) -> list[str]:
        return sorted([*self.dirs, *self.virtual])

    def split(self, spec: object) -> tuple[str, str]:
        if not isinstance(spec, str) or "\x00" in spec:
            raise ToolError("a path must be a string without NUL bytes")
        if ":" not in spec:
            raise ToolError(f"paths look like root:relative/path; roots are {self.names()}")
        root, rel = spec.split(":", 1)
        if root not in self.dirs and root not in self.virtual:
            raise ToolError(f"unknown root {root!r}; roots are {self.names()}")
        rel = rel.strip()
        if rel.startswith("-"):
            raise ToolError("a path may not start with '-'")
        if rel.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", rel):
            raise ToolError("absolute paths are not allowed")
        norm = posixpath.normpath(rel) if rel else "."
        if norm == ".." or norm.startswith("../"):
            raise ToolError("path escapes its root")
        return root, norm

    def real(self, root: str, rel: str) -> Path:
        """The file or directory on disk for a directory root, confined to it."""
        base = self.dirs[root].resolve()
        path = (base / rel).resolve()
        if not path.is_relative_to(base):
            raise ToolError("path escapes its root")
        return path

    def virtual_file(self, root: str, rel: str) -> Path:
        files = self.virtual[root]
        if rel not in files:
            raise ToolError(f"{root}:{rel} is not available")
        return files[rel]

    def display(self, root: str, rel: str) -> str:
        return f"{root}:{'' if rel == '.' else rel}"


@dataclass(frozen=True)
class ToolContext:
    roots: Roots
    git: GitRunner | None = None
    clone: Path | None = None
    base_sha: str = ""


Runner = Callable[[ToolContext, dict[str, Any]], ToolResult]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Runner

    @property
    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _int(args: dict[str, Any], key: str, default: int, lo: int, hi: int) -> int:
    value = args.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ToolError(f"{key} must be an integer")
    try:
        n = int(value)
    except ValueError as e:
        raise ToolError(f"{key} must be an integer") from e
    return max(lo, min(hi, n))


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise ToolError("not a file")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ToolError(f"file is larger than {MAX_FILE_BYTES} bytes")
    data = path.read_bytes()
    if b"\x00" in data[:8000]:
        raise ToolError("binary file")
    return data.decode(errors="replace")


# ----- list_dir ------------------------------------------------------------------------


def _list_dir(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    root, rel = ctx.roots.split(args.get("path", ""))
    depth = _int(args, "depth", 1, 1, 2)
    lines: list[str] = []
    if root in ctx.roots.virtual:
        # An allowlist is flat and short, so every file under the prefix is listed.
        prefix = "" if rel == "." else rel.rstrip("/") + "/"
        for key in sorted(ctx.roots.virtual[root]):
            if key.startswith(prefix):
                lines.append(f"{root}:{key}")
        if not lines:
            raise ToolError(f"nothing under {ctx.roots.display(root, rel)}")
        return ToolResult("\n".join(lines[:LIST_MAX_ENTRIES]))

    base = ctx.roots.real(root, rel)
    if not base.is_dir():
        raise ToolError("not a directory")
    top = ctx.roots.dirs[root].resolve()

    def walk(d: Path, level: int) -> None:
        for p in sorted(d.iterdir()):
            if p.name == ".git" or len(lines) >= LIST_MAX_ENTRIES:
                continue
            shown = f"{root}:{p.relative_to(top).as_posix()}"
            if p.is_dir():
                lines.append(shown + "/")
                if level < depth:
                    walk(p, level + 1)
            else:
                lines.append(f"{shown}  {p.stat().st_size}")

    walk(base, 1)
    return ToolResult("\n".join(lines) if lines else "(empty)")


# ----- read_file -----------------------------------------------------------------------


def _read_file(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    root, rel = ctx.roots.split(args.get("path", ""))
    path = (
        ctx.roots.virtual_file(root, rel)
        if root in ctx.roots.virtual
        else ctx.roots.real(root, rel)
    )
    lines = _read_text(path).splitlines()
    start = _int(args, "start_line", 1, 1, max(1, len(lines)))
    end = _int(args, "end_line", start + READ_LINES - 1, start, len(lines) or start)
    end = min(end, start + READ_LINES - 1)
    body = "\n".join(f"{i:6d}  {lines[i - 1]}" for i in range(start, end + 1) if i <= len(lines))
    header = (
        f"{ctx.roots.display(root, rel)} lines {start} to {min(end, len(lines))} of {len(lines)}"
    )
    return ToolResult(clip(f"{header}\n{body}"))


# ----- grep ----------------------------------------------------------------------------


def _grep(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise ToolError("pattern must be a non empty string")
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise ToolError(f"invalid regular expression: {e}") from e
    root, rel = ctx.roots.split(args.get("path", ""))
    glob = args.get("glob", "")
    if not isinstance(glob, str) or glob.startswith("-"):
        raise ToolError("glob must be a string that does not start with '-'")

    files: list[tuple[str, Path]] = []
    if root in ctx.roots.virtual:
        prefix = "" if rel == "." else rel
        files = [(k, p) for k, p in sorted(ctx.roots.virtual[root].items()) if k.startswith(prefix)]
    else:
        top = ctx.roots.dirs[root].resolve()
        base = ctx.roots.real(root, rel)
        if base.is_file():
            files = [(base.relative_to(top).as_posix(), base)]
        else:
            for p in sorted(base.rglob("*")):
                if ".git" in p.relative_to(top).parts or not p.is_file():
                    continue
                if p.resolve().is_relative_to(top):
                    files.append((p.relative_to(top).as_posix(), p))

    matches: list[str] = []
    for key, path in files:
        if glob and not fnmatch.fnmatch(key, glob):
            continue
        try:
            text = _read_text(path)
        except ToolError:
            continue
        for n, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                matches.append(f"{root}:{key}:{n}: {line[:300]}")
                if len(matches) >= GREP_MAX_MATCHES:
                    matches.append(f"[stopped at {GREP_MAX_MATCHES} matches]")
                    return ToolResult(clip("\n".join(matches)))
    return ToolResult(clip("\n".join(matches)) if matches else "no matches")


# ----- git -----------------------------------------------------------------------------


def _git(ctx: ToolContext) -> tuple[GitRunner, Path]:
    if ctx.git is None or ctx.clone is None or not ctx.base_sha:
        raise ToolError("git history is not available here")
    return ctx.git, ctx.clone


def _repo_path(value: object) -> str:
    if value in (None, ""):
        return ""
    if not isinstance(value, str) or "\x00" in value:
        raise ToolError("path must be a string")
    rel = value.split(":", 1)[1] if ":" in value else value
    rel = rel.strip()
    if rel.startswith(("-", "/")):
        raise ToolError("path may not start with '-' or '/'")
    norm = posixpath.normpath(rel)
    if norm == ".." or norm.startswith("../"):
        raise ToolError("path escapes the repository")
    return norm


def _git_log(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    git, clone = _git(ctx)
    rel = _repo_path(args.get("path"))
    n = _int(args, "max_count", 20, 1, 50)
    argv = ["log", f"--max-count={n}", "--date=short", "--format=%h %ad %an  %s", ctx.base_sha]
    argv += ["--", rel] if rel else []
    try:
        return ToolResult(clip(git.run(argv, cwd=clone)) or "no commits")
    except GitError as e:
        raise ToolError(str(e)) from e


def _git_show(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    git, clone = _git(ctx)
    rev = args.get("rev")
    if not isinstance(rev, str) or not HEX_REV.match(rev):
        raise ToolError("rev must be a commit hash of 7 to 40 lowercase hex characters")
    try:
        git.run(["merge-base", "--is-ancestor", rev, ctx.base_sha], cwd=clone)
    except GitError as e:
        raise ToolError(f"{rev} is not an ancestor of the base commit") from e
    rel = _repo_path(args.get("path"))
    argv = ["show", "--stat", "--patch", "--format=%H %an %ad%n%s%n%n%b", rev]
    argv += ["--", rel] if rel else []
    try:
        return ToolResult(clip(git.run(argv, cwd=clone)))
    except GitError as e:
        raise ToolError(str(e)) from e


_PATH = {"type": "string", "description": "root:relative/path"}

LIST_DIR = Tool(
    "list_dir",
    "List a directory, or the files of a virtual root. Depth 1 or 2.",
    {
        "type": "object",
        "properties": {"path": _PATH, "depth": {"type": "integer"}},
        "required": ["path"],
    },
    _list_dir,
)
READ_FILE = Tool(
    "read_file",
    f"Read a text file with line numbers, at most {READ_LINES} lines per call.",
    {
        "type": "object",
        "properties": {
            "path": _PATH,
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        },
        "required": ["path"],
    },
    _read_file,
)
GREP = Tool(
    "grep",
    "Search files under a path for a Python regular expression. Optional glob on the relative path.",
    {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": _PATH,
            "glob": {"type": "string"},
        },
        "required": ["pattern", "path"],
    },
    _grep,
)
GIT_LOG = Tool(
    "git_log",
    "Commits reachable from the base commit, newest first, optionally for one repository path.",
    {
        "type": "object",
        "properties": {"path": {"type": "string"}, "max_count": {"type": "integer"}},
    },
    _git_log,
)
GIT_SHOW = Tool(
    "git_show",
    "One commit that is an ancestor of the base: message, stat and patch, optionally for one path.",
    {
        "type": "object",
        "properties": {"rev": {"type": "string"}, "path": {"type": "string"}},
        "required": ["rev"],
    },
    _git_show,
)

READ_TOOLS: tuple[Tool, ...] = (LIST_DIR, READ_FILE, GREP, GIT_LOG, GIT_SHOW)


def submit_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    validate: Callable[[dict[str, Any]], str | None],
) -> Tool:
    """A terminal tool. A rejected submission goes back to the model as text."""

    def run(_ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        problem = validate(args)
        if problem:
            return ToolResult(f"not accepted: {problem}")
        return ToolResult("accepted", submitted=args)

    return Tool(name, description, parameters, run)


def execute(tools: Sequence[Tool], ctx: ToolContext, name: str, args: dict[str, Any]) -> ToolResult:
    by_name = {t.name: t for t in tools}
    tool = by_name.get(name)
    if tool is None:
        return ToolResult(f"error: unknown tool {name!r}; tools are {sorted(by_name)}")
    try:
        return tool.run(ctx, args)
    except ToolError as e:
        return ToolResult(f"error: {e}")
    except OSError as e:
        return ToolResult(f"error: {e.strerror or e}")
