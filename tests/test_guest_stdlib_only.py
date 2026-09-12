"""Guest programs are uploaded into boxes that have only the standard library
and the target's own dependencies. This test makes that boundary a failure, not
a convention."""

import ast
import sys
from pathlib import Path

GUEST = Path(__file__).parent.parent / "src" / "autoresearch" / "guest"

# Modules guest programs may import from each other, by bare name, because they
# sit side by side after upload and put their own directory on sys.path.
SIBLINGS = {p.stem for p in GUEST.glob("*.py")}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_guest_modules_import_only_the_standard_library() -> None:
    stdlib = sys.stdlib_module_names
    offenders = {}
    for path in GUEST.glob("*.py"):
        bad = {n for n in _imports(path) if n not in stdlib and n not in SIBLINGS}
        if bad:
            offenders[path.name] = bad
    assert offenders == {}, f"non stdlib imports in guest code: {offenders}"


def test_guest_modules_never_import_the_host_package() -> None:
    for path in GUEST.glob("*.py"):
        assert "autoresearch" not in _imports(path), path.name
