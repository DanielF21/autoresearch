"""Whether a repository is one the harness can measure, judged by reading it.

The rules are the README's "What the harness can measure", as far as a file
listing can see them: a Python project, no compiled build, a package that
imports by path from ``.`` or ``src``, and tests pytest can find. What only
running it can show (the call's length, determinism, a passing suite) is
``check``'s to judge, on a box.

Nothing here imports, installs or executes anything from the clone. setup.py is
read with ``ast``.
"""

from __future__ import annotations

import ast
import configparser
import contextlib
import os
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COMPILED_BACKENDS = ("scikit_build_core", "mesonpy", "maturin", "setuptools_rust")
COMPILED_REQUIRES = (
    "cython",
    "scikit-build",
    "scikit-build-core",
    "meson-python",
    "maturin",
    "setuptools-rust",
    "pybind11",
    "nanobind",
)
COMPILED_SUFFIXES = (".pyx", ".pxd", ".c", ".cc", ".cpp", ".rs")
NOT_PACKAGES = {
    "tests",
    "test",
    "docs",
    "doc",
    "examples",
    "benchmarks",
    "bench",
    "build",
    "dist",
    "scripts",
    "tools",
}
THREAD_IMPORT = re.compile(r"^\s*(?:import|from)\s+(threading|multiprocessing|concurrent)\b", re.M)
CLONE_TIMEOUT_S = 600


class IntakeError(RuntimeError):
    """Intake could not read the repository at all."""


@dataclass(frozen=True)
class Finding:
    rule: str
    level: str  # "refuse" or "warn"
    detail: str

    @property
    def refused(self) -> bool:
        return self.level == "refuse"


def repo_name(url: str) -> str:
    """The last path segment of the URL, lowercased, with anything else made ``_``."""
    last = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
    name = re.sub(r"[^a-z0-9]+", "_", last.lower()).strip("_")
    if not name:
        raise IntakeError(f"no repository name in {url!r}")
    return name


def git_env() -> dict[str, str]:
    """The environment with every ``GIT_`` variable removed, so a hook's cannot leak in."""
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def clone(url: str, dest: Path) -> str:
    """A shallow clone of the default branch at ``dest``. Returns its HEAD sha."""
    if url.startswith("-"):
        raise IntakeError(f"not a repository URL: {url!r}")
    if dest.exists():
        raise IntakeError(f"{dest} already exists; remove it to take this repository in again")
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["git", "clone", "--depth", "1", "-q", "--", url, str(dest)],
        capture_output=True,
        text=True,
        timeout=CLONE_TIMEOUT_S,
        check=False,
        env=git_env(),
    )
    if r.returncode != 0:
        raise IntakeError(f"git clone {url} failed: {r.stderr.strip()[-500:]}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=dest,
        capture_output=True,
        text=True,
        check=False,
        env=git_env(),
    )
    sha = head.stdout.strip()
    if head.returncode != 0 or len(sha) != 40:
        raise IntakeError(f"no HEAD in the clone of {url}: {head.stderr.strip()[-300:]}")
    return sha


def read_pyproject(repo: Path) -> dict[str, Any]:
    path = repo / "pyproject.toml"
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text())
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise IntakeError(f"pyproject.toml does not parse: {e}") from e


def _setup_cfg(repo: Path) -> configparser.ConfigParser:
    cp = configparser.ConfigParser(interpolation=None)
    path = repo / "setup.cfg"
    if path.is_file():
        with contextlib.suppress(configparser.Error):
            cp.read_string(path.read_text(errors="replace"))
    return cp


def project_name(repo: Path, pyproject: dict[str, Any]) -> str:
    name = pyproject.get("project", {}).get("name")
    if isinstance(name, str) and name:
        return name
    cp = _setup_cfg(repo)
    if cp.has_option("metadata", "name"):
        return cp.get("metadata", "name")
    return ""


def _normal(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def find_packages(repo: Path, names: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    """``(package_root, package)`` for every package directory at ``.`` or ``src``.

    A single module ``<root>/<stem>.py`` counts too, but only when its stem is
    one of ``names`` (the project's and the repository's), so setup.py,
    conftest.py and noxfile.py are never candidates. A directory of the same
    name takes precedence over such a module.
    """
    found: list[tuple[str, str]] = []
    modules: list[tuple[str, str]] = []
    wanted = {_normal(n) for n in names if n}
    for root in (".", "src"):
        base = repo if root == "." else repo / root
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if child.name in NOT_PACKAGES:
                continue
            if child.is_dir() and child.name.isidentifier() and (child / "__init__.py").is_file():
                found.append((root, child.name))
            elif (
                child.is_file()
                and child.suffix == ".py"
                and child.stem.isidentifier()
                and child.stem not in NOT_PACKAGES
                and _normal(child.stem) in wanted
            ):
                modules.append((root, child.stem))
    taken = {p for _, p in found}
    return found + [m for m in modules if m[1] not in taken]


def choose_package(
    candidates: list[tuple[str, str]], names: tuple[str, ...], override: str = ""
) -> tuple[tuple[str, str] | None, Finding | None]:
    """The package the target is, or the refusal that says why none could be picked."""
    listed = ", ".join(f"{r}/{p}" for r, p in candidates) or "none"
    if override:
        match = [c for c in candidates if c[1] == override]
        if not match:
            return None, Finding("package", "refuse", f"--package {override} not among: {listed}")
        return match[0], None
    if not candidates:
        return None, Finding(
            "package",
            "refuse",
            "no package directory with an __init__.py at the root or under src/, "
            "or a single module named after the project; the harness imports by path "
            "from one of those",
        )
    if len(candidates) == 1:
        return candidates[0], None
    wanted = {_normal(n) for n in names if n}
    match = [c for c in candidates if _normal(c[1]) in wanted]
    if len(match) == 1:
        return match[0], None
    return None, Finding(
        "package",
        "refuse",
        f"several packages and none matches the project name: {listed}; name one with --package",
    )


def pytest_testpaths(repo: Path, pyproject: dict[str, Any]) -> list[str]:
    raw = pyproject.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("testpaths")
    if isinstance(raw, list):
        return [str(p) for p in raw]
    if isinstance(raw, str):
        return raw.split()
    for fname, section in (
        ("pytest.ini", "pytest"),
        ("tox.ini", "pytest"),
        ("setup.cfg", "tool:pytest"),
    ):
        path = repo / fname
        if not path.is_file():
            continue
        cp = configparser.ConfigParser(interpolation=None)
        try:
            cp.read_string(path.read_text(errors="replace"))
        except configparser.Error:
            continue
        if cp.has_option(section, "testpaths"):
            return cp.get(section, "testpaths").split()
    return []


def full_suite(repo: Path, package_root: str, package: str, testpaths: list[str]) -> str:
    """The path the whole suite runs from, relative to the repo, or "" when there is none."""
    for t in testpaths:
        if (repo / t).exists():
            return t
    for d in ("tests", "test"):
        if (repo / d).is_dir():
            return d
    pkg = repo / package_root / package
    if any(p.is_dir() and p.name in ("tests", "test") for p in pkg.rglob("test*")):
        return str((pkg).relative_to(repo))
    return ""


def _setup_py_builds_extensions(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "ext_modules":
            return True
        if isinstance(node, ast.Name) and node.id in (
            "Extension",
            "cythonize",
            "Pybind11Extension",
        ):
            return True
        if isinstance(node, ast.Attribute) and node.attr in ("Extension", "cythonize"):
            return True
    return False


def requirement_name(req: str) -> str:
    m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req)
    return m.group(1).lower() if m else ""


def judge(repo: Path, pyproject: dict[str, Any], package: tuple[str, str] | None) -> list[Finding]:
    """Every refusal and warning a reading of the tree supports."""
    out: list[Finding] = []
    if not any((repo / f).is_file() for f in ("pyproject.toml", "setup.py", "setup.cfg")):
        out.append(Finding("python project", "refuse", "no pyproject.toml, setup.py or setup.cfg"))
        return out

    build = pyproject.get("build-system", {})
    backend = str(build.get("build-backend", ""))
    if any(b in backend for b in COMPILED_BACKENDS):
        out.append(Finding("compiled build", "refuse", f"build backend {backend} compiles code"))
    requires = {requirement_name(str(r)) for r in build.get("requires", [])}
    compiled_requires = sorted(requires & set(COMPILED_REQUIRES))
    if compiled_requires:
        out.append(
            Finding(
                "compiled build",
                "refuse",
                f"the build requires {', '.join(compiled_requires)}, so what users run is compiled",
            )
        )
    if "mypyc" in requires or "hatch-mypyc" in requires:
        out.append(
            Finding(
                "compiled build",
                "warn",
                "mypyc is in the build: released wheels may run compiled code, not the Python "
                "a patch changes",
            )
        )
    if (repo / "Cargo.toml").is_file():
        out.append(Finding("compiled build", "refuse", "Cargo.toml at the root: a Rust build"))
    if (repo / "setup.py").is_file() and _setup_py_builds_extensions(repo / "setup.py"):
        out.append(Finding("compiled build", "refuse", "setup.py builds extension modules"))

    if package is None:
        return out
    root, name = package
    pkg = repo / root / name
    if not pkg.is_dir():
        pkg = pkg.with_suffix(".py")  # a single module; rglob over a file yields nothing
    compiled = sorted(
        str(p.relative_to(repo)) for p in pkg.rglob("*") if p.suffix in COMPILED_SUFFIXES
    )
    if compiled:
        shown = ", ".join(compiled[:5]) + (" and more" if len(compiled) > 5 else "")
        out.append(
            Finding(
                "compiled build",
                "refuse",
                f"compiled sources inside the package ({shown}); the harness imports by path "
                "and never builds",
            )
        )

    if not full_suite(repo, root, name, pytest_testpaths(repo, pyproject)):
        out.append(
            Finding(
                "tests",
                "refuse",
                "no tests directory, no pytest testpaths and no tests inside the package",
            )
        )

    threaded: list[str] = []
    for p in [pkg] if pkg.is_file() else sorted(pkg.rglob("*.py")):
        try:
            if THREAD_IMPORT.search(p.read_text(errors="replace")):
                threaded.append(str(p.relative_to(repo)))
        except OSError:
            continue
    if threaded:
        out.append(
            Finding(
                "one core",
                "warn",
                f"threads or processes imported in {', '.join(threaded[:5])}: every timing is "
                "pinned to one core, so parallel work measures serialised",
            )
        )
    if (repo / "tox.ini").is_file() or (repo / "noxfile.py").is_file():
        out.append(
            Finding(
                "test runner",
                "warn",
                "tox or nox is configured; check runs plain pytest, so a suite that needs "
                "their setup fails there",
            )
        )
    requires_python = pyproject.get("project", {}).get("requires-python")
    if requires_python:
        out.append(
            Finding(
                "python version",
                "warn",
                f"requires-python {requires_python}; check reports the box's Python",
            )
        )
    return out
