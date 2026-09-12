"""The pure parts of the box shell: how ``cd`` is made to stick, and the guard."""

import inspect_cli


def test_the_command_is_run_in_the_tracked_directory() -> None:
    cmd = inspect_cli.wrap("ls", "/mnt/autoresearch/runs")
    assert cmd.startswith("cd /mnt/autoresearch/runs")
    assert "\nls\n" in cmd
    # The exit code of the command, not of the marker printf that follows it.
    assert cmd.index("__rc=$?") < cmd.index("printf")
    assert cmd.rstrip().endswith("exit $__rc")


def test_a_directory_with_spaces_is_quoted() -> None:
    assert "'/a b'" in inspect_cli.wrap("ls", "/a b")


def test_the_new_directory_is_taken_from_the_marker_and_hidden() -> None:
    out = f"a\nb\n\n{inspect_cli.MARKER}/workspace/repo\n"
    body, cwd = inspect_cli.split_pwd(out, "/")
    assert cwd == "/workspace/repo"
    assert inspect_cli.MARKER not in body
    assert body.splitlines()[:2] == ["a", "b"]


def test_output_without_a_marker_keeps_the_old_directory() -> None:
    # A command that kills the shell never prints the marker. Losing the
    # directory would be worse than keeping a stale one.
    body, cwd = inspect_cli.split_pwd("partial output", "/workspace")
    assert body == "partial output" and cwd == "/workspace"


def test_the_last_marker_wins() -> None:
    out = f"{inspect_cli.MARKER}/one\ntext\n{inspect_cli.MARKER}/two\n"
    body, cwd = inspect_cli.split_pwd(out, "/")
    # Only the real trailing marker is consumed; anything the command itself
    # printed stays visible rather than being silently eaten.
    assert cwd == "/two"
    assert f"{inspect_cli.MARKER}/one" in body


def test_the_destructive_guard_catches_what_would_lose_a_run() -> None:
    for cmd in (
        "rm -rf /mnt/autoresearch/runs/t1_w4",
        "rm -r attempts",
        "dd if=/dev/zero of=/dev/sda",
        "echo x > /mnt/autoresearch/runs/t1_w1/run.log",
        "git push origin main",
    ):
        assert inspect_cli.DESTRUCTIVE.search(cmd), cmd


def test_the_guard_leaves_ordinary_inspection_alone() -> None:
    for cmd in (
        "ls -la /mnt/autoresearch/runs",
        "cat run.log",
        "grep -rn rm src",
        "python3 -c 'print(1)'",
        "git log --oneline -5",
        "tail -f run.log",
    ):
        assert not inspect_cli.DESTRUCTIVE.search(cmd), cmd
