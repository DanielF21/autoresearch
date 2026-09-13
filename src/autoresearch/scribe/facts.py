"""Facts about a diff that code can establish, so no model has to be trusted for them.

Pure. Takes the diff text and, when available, the full base and patched sources
of the files it touches. Python files get AST facts; any other file gets the diff
level ones only. Nothing here knows what the target is.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

from autoresearch.patch import changed_files

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
_FILE = re.compile(r"^diff --git a/(\S+) b/(\S+)$")
_IMPORT_LINE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))")
_COMPARE_NUMBER = re.compile(r"(?:<=|>=|==|!=|<|>)\s*(?!(?:0|1|2)\b)\d")
TRIVIAL_NUMBERS = {0, 1, -1, 2}


@dataclass
class FileDiff:
    path: str
    added: dict[int, str] = field(default_factory=dict)  # new line number -> text
    removed: dict[int, str] = field(default_factory=dict)  # old line number -> text
    hunks: int = 0
    contexts: list[str] = field(default_factory=list)


def parse_diff(diff: str) -> list[FileDiff]:
    files: list[FileDiff] = []
    current: FileDiff | None = None
    old = new = 0
    for line in diff.splitlines():
        m = _FILE.match(line)
        if m:
            current = FileDiff(path=m.group(2))
            files.append(current)
            continue
        if current is None:
            continue
        h = _HUNK.match(line)
        if h:
            old, new = int(h.group(1)), int(h.group(3))
            current.hunks += 1
            if h.group(5).strip():
                current.contexts.append(h.group(5).strip())
            continue
        if line.startswith(("+++", "---", "index ", "new file", "deleted file", "similarity")):
            continue
        if line.startswith("+"):
            current.added[new] = line[1:]
            new += 1
        elif line.startswith("-"):
            current.removed[old] = line[1:]
            old += 1
        elif line.startswith(" ") or line == "":
            old += 1
            new += 1
    return files


@dataclass(frozen=True)
class FunctionChange:
    path: str
    name: str
    added: int
    removed: int

    @property
    def old_path_kept(self) -> bool:
        return self.added > 0 and self.removed == 0


@dataclass(frozen=True)
class Facts:
    files: tuple[str, ...]
    added: int
    removed: int
    hunks: int
    functions: tuple[FunctionChange, ...]
    new_imports: tuple[str, ...]
    public_api_changes: tuple[str, ...]
    new_condition_numbers: tuple[str, ...]
    private_attribute_reads: tuple[str, ...]
    numeric_literals: tuple[str, ...]
    parse_errors: tuple[str, ...]

    @property
    def changed_lines(self) -> int:
        return self.added + self.removed

    @property
    def old_path_kept(self) -> bool:
        return bool(self.functions) and all(f.old_path_kept for f in self.functions)

    def function_names(self) -> set[str]:
        return {f"{f.path}::{f.name}" for f in self.functions}

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": list(self.files),
            "added": self.added,
            "removed": self.removed,
            "changed_lines": self.changed_lines,
            "hunks": self.hunks,
            "functions": [
                {
                    "path": f.path,
                    "name": f.name,
                    "added": f.added,
                    "removed": f.removed,
                    "old_path_kept": f.old_path_kept,
                }
                for f in self.functions
            ],
            "old_path_kept": self.old_path_kept,
            "new_imports": list(self.new_imports),
            "public_api_changes": list(self.public_api_changes),
            "new_condition_numbers": list(self.new_condition_numbers),
            "private_attribute_reads": list(self.private_attribute_reads),
            "numeric_literals": list(self.numeric_literals),
            "parse_errors": list(self.parse_errors),
        }


def _spans(tree: ast.AST) -> list[tuple[int, int, str]]:
    """(start, end, qualified name) for every def and class, innermost last."""
    out: list[tuple[int, int, str]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f"{prefix}{child.name}"
                start = min([child.lineno, *(d.lineno for d in child.decorator_list)])
                out.append((start, child.end_lineno or child.lineno, name))
                walk(child, name + ".")
            else:
                walk(child, prefix)

    walk(tree, "")
    return out


def _enclosing(spans: list[tuple[int, int, str]], line: int) -> str:
    best = "<module>"
    best_size = None
    for start, end, name in spans:
        if start <= line <= end and (best_size is None or end - start < best_size):
            best, best_size = name, end - start
    return best


def _imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _public_signatures(tree: ast.Module) -> dict[str, str]:
    sigs: dict[str, str] = {}

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if not node.name.startswith("_"):
                    sigs[prefix + node.name] = ast.unparse(node.args)
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                visit(node.body, prefix + node.name + ".")
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "__all__":
                        sigs["__all__"] = ast.unparse(node.value)

    visit(tree.body, "")
    return sigs


def _parse(source: str | None) -> ast.Module | None:
    if source is None:
        return None
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def compute_facts(
    diff: str,
    base_sources: dict[str, str] | None = None,
    patched_sources: dict[str, str] | None = None,
) -> Facts:
    base_sources = base_sources or {}
    patched_sources = patched_sources or {}
    parsed = parse_diff(diff)
    functions: dict[tuple[str, str], list[int]] = {}
    new_imports: list[str] = []
    api: list[str] = []
    cond_numbers: list[str] = []
    private: list[str] = []
    literals: list[str] = []
    errors: list[str] = []

    for fd in parsed:
        for text in fd.added.values():
            literals.extend(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?:e[+-]?\d+)?", text))
        if not fd.path.endswith(".py"):
            for ctx in fd.contexts or ["<file>"]:
                functions.setdefault((fd.path, ctx), [len(fd.added), len(fd.removed)])
            continue
        base_tree = _parse(base_sources.get(fd.path))
        new_tree = _parse(patched_sources.get(fd.path))
        if fd.path in patched_sources and new_tree is None:
            errors.append(f"{fd.path}: patched source does not parse")
        if new_tree is None or (fd.path in base_sources and base_tree is None):
            # Without both trees, fall back to what the diff text says.
            for line, text in fd.added.items():
                m = _IMPORT_LINE.match(text)
                if m:
                    new_imports.append(m.group(1) or m.group(2))
                if _COMPARE_NUMBER.search(text):
                    cond_numbers.append(f"{fd.path}:{line}: {text.strip()}")
            name = fd.contexts[0] if fd.contexts else "<unknown>"
            functions[(fd.path, name)] = [len(fd.added), len(fd.removed)]
            continue

        new_spans = _spans(new_tree)
        base_spans = _spans(base_tree) if base_tree is not None else []
        for line in fd.added:
            key = (fd.path, _enclosing(new_spans, line))
            functions.setdefault(key, [0, 0])[0] += 1
        for line in fd.removed:
            key = (fd.path, _enclosing(base_spans, line))
            functions.setdefault(key, [0, 0])[1] += 1

        before = _imports(base_tree) if base_tree is not None else set()
        new_imports.extend(sorted(_imports(new_tree) - before))

        old_sigs = _public_signatures(base_tree) if base_tree is not None else {}
        new_sigs = _public_signatures(new_tree)
        for name in sorted(set(old_sigs) | set(new_sigs)):
            if name not in new_sigs:
                api.append(f"{fd.path}: removed {name}")
            elif name not in old_sigs:
                api.append(f"{fd.path}: added {name}")
            elif old_sigs[name] != new_sigs[name]:
                api.append(f"{fd.path}: signature of {name} changed")

        added_lines = set(fd.added)
        for node in ast.walk(new_tree):
            lineno = getattr(node, "lineno", None)
            if lineno not in added_lines:
                continue
            if isinstance(node, ast.Compare):
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Constant)
                        and isinstance(sub.value, int | float)
                        and not isinstance(sub.value, bool)
                        and sub.value not in TRIVIAL_NUMBERS
                    ):
                        cond_numbers.append(f"{fd.path}:{lineno}: {ast.unparse(node)}")
                        break
            elif (
                isinstance(node, ast.Attribute)
                and node.attr.startswith("_")
                and not node.attr.startswith("__")
                and not (isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"})
            ):
                private.append(f"{fd.path}:{lineno}: {ast.unparse(node)}")

    added = sum(len(f.added) for f in parsed)
    removed = sum(len(f.removed) for f in parsed)
    return Facts(
        files=changed_files(diff),
        added=added,
        removed=removed,
        hunks=sum(f.hunks for f in parsed),
        functions=tuple(FunctionChange(p, n, a, r) for (p, n), (a, r) in functions.items()),
        new_imports=tuple(dict.fromkeys(new_imports)),
        public_api_changes=tuple(api),
        new_condition_numbers=tuple(dict.fromkeys(cond_numbers)),
        private_attribute_reads=tuple(dict.fromkeys(private)),
        numeric_literals=tuple(dict.fromkeys(literals)),
        parse_errors=tuple(errors),
    )
