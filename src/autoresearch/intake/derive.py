"""What can be read off a clone: the target's metadata, and a brief for the proposer.

The draft is JSON, not a config. A config needs a call and inputs, and those are
the proposal's to choose, so a config file exists only once one is accepted.
"""

from __future__ import annotations

import configparser
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from autoresearch.intake import scope

TEST_GROUPS = ("test", "tests", "testing", "dev")
HARNESS_PIP = {"pytest", "pytest-xdist"}
DENY_DIRS = ("tests", "test", "docs", "doc", "examples", "benchmarks", "bench")
BRIEF_FILES = 30
BENCH_FILES = 20
DRAFT_FILE = "draft.json"
BRIEF_FILE = "brief.md"
REPO_DIR = "repo"
TOX_FACTOR = re.compile(r"^([A-Za-z0-9_,{}.-]+):\s+(.+)$")
TOX_TEST_FACTORS = {"unit", "test", "tests", "testing"}


@dataclass(frozen=True)
class Draft:
    name: str
    repo: str
    sha: str
    package: str
    package_root: str
    pip: tuple[str, ...]
    tests_full: str
    allow: tuple[str, ...]
    deny: tuple[str, ...]
    run_id: str
    template: str
    benchmarks: tuple[str, ...] = ()
    findings: tuple[scope.Finding, ...] = field(default_factory=tuple)

    @property
    def package_dir(self) -> str:
        """The package directory relative to the repo, with no leading ``./``."""
        return self.package if self.package_root == "." else f"{self.package_root}/{self.package}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Draft:
        return cls(
            name=d["name"],
            repo=d["repo"],
            sha=d["sha"],
            package=d["package"],
            package_root=d["package_root"],
            pip=tuple(d["pip"]),
            tests_full=d["tests_full"],
            allow=tuple(d["allow"]),
            deny=tuple(d["deny"]),
            run_id=d["run_id"],
            template=d["template"],
            benchmarks=tuple(d.get("benchmarks", ())),
            findings=tuple(scope.Finding(**f) for f in d.get("findings", ())),
        )


def _tox_entries(value: str) -> list[str]:
    """Entries of a tox ``[testenv]`` list that a test environment gets.

    An entry may carry a factor, ``unit: pytest``; only factors naming tests or a
    Python version are kept, so a docs or lint environment's packages are not.
    """
    out: list[str] = []
    for line in value.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")) or "://" in line:
            continue
        m = TOX_FACTOR.match(line)
        if m:
            words = re.split(r"[^A-Za-z0-9]+", m.group(1))
            if not any(w in TOX_TEST_FACTORS or re.fullmatch(r"py\d*", w) for w in words):
                continue
            line = m.group(2)
        out += [part.strip() for part in line.split(",") if part.strip()]
    return out


def tox_test_requirements(repo: Path) -> tuple[list[str], list[str]]:
    """``deps`` and ``extras`` of tox's ``[testenv]``, for its test factors."""
    path = repo / "tox.ini"
    cp = configparser.ConfigParser(interpolation=None)
    if not path.is_file():
        return [], []
    try:
        cp.read_string(path.read_text(errors="replace"))
    except configparser.Error:
        return [], []
    if not cp.has_section("testenv"):
        return [], []
    deps = _tox_entries(cp.get("testenv", "deps", fallback=""))
    extras = _tox_entries(cp.get("testenv", "extras", fallback=""))
    return deps, extras


def dependencies(pyproject: dict[str, Any], repo: Path) -> tuple[str, ...]:
    """Runtime dependencies and the ones tests get, names only, in file order.

    Tests get the extras and groups named ``test`` and the like, and whatever
    tox's test environment installs: its deps and the extras it asks for.
    """
    project = pyproject.get("project", {})
    raw: list[str] = [str(r) for r in project.get("dependencies", [])]
    extras = project.get("optional-dependencies", {})
    groups = pyproject.get("dependency-groups", {})
    tox_deps, tox_extras = tox_test_requirements(repo)
    for key in (*TEST_GROUPS, *tox_extras):
        for source in (extras, groups):
            for r in source.get(key, []):
                if isinstance(r, str):  # a dependency group may include another by table
                    raw.append(r)
    raw += tox_deps
    names: list[str] = []
    for r in raw:
        name = scope.requirement_name(r)
        if name and name not in HARNESS_PIP and name not in names:
            names.append(name)
    return tuple(names)


def benchmark_sources(repo: Path) -> tuple[str, ...]:
    found: list[str] = []
    if (repo / "asv.conf.json").is_file():
        found.append("asv.conf.json")
    for p in sorted(repo.rglob("*.py")):
        rel = p.relative_to(repo)
        if ".git" in rel.parts:
            continue
        lowered = str(rel).lower()
        if "bench" in lowered or "perf" in p.name.lower():
            found.append(str(rel))
        if len(found) >= BENCH_FILES:
            break
    return tuple(found)


def derive(
    url: str, repo: Path, sha: str, template: str, template_width: int, package_override: str = ""
) -> tuple[Draft | None, list[scope.Finding]]:
    """The draft, or None with the refusals that stopped it. Findings come back either way."""
    pyproject = scope.read_pyproject(repo)
    name = scope.repo_name(url)
    chosen, refusal = scope.choose_package(
        scope.find_packages(repo), (scope.project_name(repo, pyproject), name), package_override
    )
    findings = scope.judge(repo, pyproject, chosen)
    if refusal is not None:
        findings.insert(0, refusal)
    if chosen is None or any(f.refused for f in findings):
        return None, findings

    root, package = chosen
    pip = dependencies(pyproject, repo)
    if "numpy" in pip:
        findings.append(
            scope.Finding(
                "hot path",
                "warn",
                "numpy is a dependency: time spent inside it is C, and agents tend to trade "
                "Python loops for array code that is fast on one input shape only",
            )
        )
    if not pyproject.get("project", {}).get("dependencies") and (repo / "setup.py").is_file():
        findings.append(
            scope.Finding(
                "dependencies",
                "warn",
                "no [project].dependencies to read; setup.py is not executed, so check is "
                "where a missing dependency shows",
            )
        )
    prefix = "" if root == "." else f"{root}/"
    full = scope.full_suite(repo, root, package, scope.pytest_testpaths(repo, pyproject))
    deny = [f"{d}/**" for d in DENY_DIRS if (repo / d).is_dir()]
    if full == f"{prefix}{package}":
        deny.insert(0, f"{prefix}{package}/**/tests/**")
    draft = Draft(
        name=name,
        repo=url,
        sha=sha,
        package=package,
        package_root=root,
        pip=pip,
        tests_full=full,
        allow=(f"{prefix}{package}/**",),
        deny=tuple(deny),
        run_id=f"{name}_w{template_width}",
        template=template,
        benchmarks=benchmark_sources(repo),
        findings=tuple(findings),
    )
    return draft, findings


def _largest_modules(repo: Path, draft: Draft) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for p in (repo / draft.package_dir).rglob("*.py"):
        try:
            rows.append((str(p.relative_to(repo)), len(p.read_text(errors="replace").splitlines())))
        except OSError:
            continue
    rows.sort(key=lambda r: -r[1])
    return rows[:BRIEF_FILES]


def brief(repo: Path, draft: Draft) -> str:
    """What the proposer is told first, and what a person reads before its go."""
    lines = [
        f"# {draft.name}",
        "",
        f"Repository {draft.repo} at {draft.sha}. The checkout is the root of every tool path.",
        "",
        "## Derived from the tree",
        "",
        f"- package `{draft.package}`, imported by path from `{draft.package_root}`",
        f"- dependencies installed in the box: {', '.join(draft.pip) or 'none found'}",
        f"- whole test suite: `{draft.tests_full}`",
        f"- a patch may change: {', '.join(draft.allow)}; never: {', '.join(draft.deny) or 'n/a'}",
        "",
        "## Warnings",
        "",
    ]
    warns = [f for f in draft.findings if not f.refused]
    lines += [f"- {f.rule}: {f.detail}" for f in warns] or ["- none"]
    extras = scope.read_pyproject(repo).get("project", {}).get("optional-dependencies", {})
    installed = set(draft.pip) | HARNESS_PIP
    missing = {
        group: names
        for group, reqs in extras.items()
        if (names := [n for r in reqs if (n := scope.requirement_name(str(r))) not in installed])
    }
    lines += ["", "## Optional dependencies not installed", ""]
    lines += [f"- `{group}`: {', '.join(names)}" for group, names in missing.items()] or ["- none"]
    lines += [
        "",
        "If the hot module's tests or the whole suite import one of these without skipping, "
        "name it in extra_pip.",
    ]
    lines += ["", "## Benchmark sources in the repository", ""]
    lines += [f"- `{b}`" for b in draft.benchmarks] or ["- none found"]
    lines += ["", "## Largest modules in the package, by lines", ""]
    lines += [f"- `{path}` {n}" for path, n in _largest_modules(repo, draft)]
    lines += [
        "",
        "## What to decide",
        "",
        "The hot file, the call, the module's own tests, and the inputs, with the axis they "
        "span and why a patch could win on one end of it and lose on the other. Read the code, "
        "then call submit_proposal.",
    ]
    return "\n".join(lines) + "\n"


def write_draft(out: Path, repo: Path, draft: Draft) -> None:
    (out / DRAFT_FILE).write_text(json.dumps(draft.to_dict(), indent=2) + "\n")
    (out / BRIEF_FILE).write_text(brief(repo, draft))


def load_draft(out: Path) -> Draft:
    return Draft.from_dict(json.loads((out / DRAFT_FILE).read_text()))
