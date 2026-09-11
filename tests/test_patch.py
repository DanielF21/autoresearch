from autoresearch.patch import (
    changed_files,
    changed_line_count,
    matches_any,
    normalised_hash,
    scope_violations,
)

DIFF = """\
diff --git a/networkx/algorithms/cluster.py b/networkx/algorithms/cluster.py
index 1111111..2222222 100644
--- a/networkx/algorithms/cluster.py
+++ b/networkx/algorithms/cluster.py
@@ -160,7 +160,8 @@ def _directed_triangles_and_degree_iter(G, nodes=None):
     for i, preds, succs in nodes_nbrs:
-        ipreds = set(preds) - {i}
+        ipreds = set(preds)
+        ipreds.discard(i)
         isuccs = set(succs) - {i}
diff --git a/networkx/algorithms/tests/test_cluster.py b/networkx/algorithms/tests/test_cluster.py
--- a/networkx/algorithms/tests/test_cluster.py
+++ b/networkx/algorithms/tests/test_cluster.py
@@ -1,3 +1,3 @@
-assert True
+assert 1
"""

ALLOW = ("networkx/**",)
DENY = ("networkx/**/tests/**", "benchmarks/**")


def test_changed_files_from_git_headers() -> None:
    assert changed_files(DIFF) == (
        "networkx/algorithms/cluster.py",
        "networkx/algorithms/tests/test_cluster.py",
    )


def test_changed_files_without_git_headers_uses_plus_lines() -> None:
    plain = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n--- a/y.py\n+++ /dev/null\n"
    assert changed_files(plain) == ("x.py",)


def test_glob_double_star_crosses_directories_and_single_star_does_not() -> None:
    assert matches_any("networkx/algorithms/cluster.py", ("networkx/**",))
    assert matches_any("networkx/algorithms/tests/test_cluster.py", ("networkx/**/tests/**",))
    assert not matches_any("networkx/algorithms/cluster.py", ("networkx/*",))
    assert matches_any("networkx/cluster.py", ("networkx/*",))
    assert not matches_any("benchmarks/x.py", ("networkx/**",))


def test_scope_violations_name_the_test_file() -> None:
    v = scope_violations(changed_files(DIFF), ALLOW, DENY)
    assert len(v) == 1
    assert v[0].path == "networkx/algorithms/tests/test_cluster.py"
    assert "denied" in v[0].reason


def test_scope_violation_outside_allow() -> None:
    v = scope_violations(("setup.py",), ALLOW, DENY)
    assert v[0].reason == "matches no allowed pattern"


def test_changed_line_count_excludes_headers() -> None:
    assert changed_line_count(DIFF) == 5


def test_normalised_hash_ignores_index_line_numbers_and_trailing_space() -> None:
    a = DIFF
    b = (
        DIFF.replace("index 1111111..2222222 100644", "index 3333333..4444444 100644")
        .replace("@@ -160,7 +160,8 @@", "@@ -161,7 +161,8 @@")
        .replace("ipreds.discard(i)", "ipreds.discard(i)   ")
    )
    assert normalised_hash(a) == normalised_hash(b)
    assert normalised_hash(a) != normalised_hash(a.replace("discard", "remove"))
