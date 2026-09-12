"""The referee against a scripted FakeBox that answers each guest program."""

from pathlib import Path

import pytest

from autoresearch.boxes.fake_box import FakeBox, ok
from autoresearch.boxes.protocol import BoxError
from autoresearch.config import RunConfig, load_config
from autoresearch.referee.referee import (
    INCUMBENT_TREE,
    PATCHED_TREE,
    Referee,
    guest_command,
)
from autoresearch.types import Verdict
from tests.helpers import referee_box as make_box

ROOT = Path(__file__).parent.parent
PATCHES = ROOT / "tests" / "fixtures" / "patches"


@pytest.fixture
def config() -> RunConfig:
    return load_config(ROOT / "configs" / "t1_w1.toml")


def _judge(box: FakeBox, config: RunConfig, patch_name: str = "precompute") -> Referee:
    ref = Referee(box, config)
    ref.setup()
    return ref


def test_accepts_a_real_speedup_and_records_everything(config: RunConfig) -> None:
    box = make_box(speedup=1.05)
    ref = _judge(box, config)
    result = ref.judge("inc_sha", (PATCHES / "precompute.diff").read_text())
    assert result.verdict == Verdict.ACCEPTED
    assert result.median_ratio == pytest.approx(1.05)
    assert result.threshold == 1.0106
    assert len(result.pairs) == 6
    assert [p.order for p in result.pairs] == ["incumbent_first", "patched_first"] * 3
    assert [p.hash_seed for p in result.pairs] == [0, 1, 2, 3, 4, 0]
    assert result.ir is not None and result.ir.incumbent == 1000 and result.ir.patched == 800
    assert result.canary_s == 0.5
    assert [t.scope for t in result.tests] == ["module", "full"]
    assert result.provenance.box_id == "sb_fake"
    assert ref.broken == ""


def test_rejects_below_threshold(config: RunConfig) -> None:
    result = _judge(make_box(speedup=1.005), config).judge(
        "s", (PATCHES / "whitespace.diff").read_text()
    )
    assert result.verdict == Verdict.REJECTED_BELOW_THRESHOLD
    assert result.median_ratio == pytest.approx(1.005)
    assert result.ir is not None


def test_rejects_a_slowdown(config: RunConfig) -> None:
    result = _judge(make_box(speedup=0.7), config).judge(
        "s", (PATCHES / "slowdown.diff").read_text()
    )
    assert result.verdict == Verdict.REJECTED_BELOW_THRESHOLD


def test_rejects_out_of_scope_before_touching_the_box(config: RunConfig) -> None:
    box = make_box()
    ref = _judge(box, config)
    before = len(box.commands)
    diff = "diff --git a/networkx/algorithms/tests/test_cluster.py b/networkx/algorithms/tests/test_cluster.py\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    result = ref.judge("s", diff)
    assert result.verdict == Verdict.REJECTED_SCOPE
    assert "denied" in result.reason
    assert len(box.commands) == before


def test_rejects_a_patch_that_does_not_apply(config: RunConfig) -> None:
    result = _judge(make_box(apply_ok=False), config).judge(
        "s", (PATCHES / "precompute.diff").read_text()
    )
    assert result.verdict == Verdict.REJECTED_APPLY
    assert "does not apply" in result.reason


def test_module_test_failure_skips_the_full_suite(config: RunConfig) -> None:
    box = make_box(module_ok=False)
    result = _judge(box, config).judge("s", (PATCHES / "precompute.diff").read_text())
    assert result.verdict == Verdict.REJECTED_TESTS_MODULE
    assert [t.scope for t in result.tests] == ["module"]
    assert not any("--scope full" in c for c in box.commands)
    assert not any("time_target.py" in c for c in box.commands)


def test_full_test_failure(config: RunConfig) -> None:
    result = _judge(make_box(full_ok=False), config).judge(
        "s", (PATCHES / "precompute.diff").read_text()
    )
    assert result.verdict == Verdict.REJECTED_TESTS_FULL
    assert [t.scope for t in result.tests] == ["module", "full"]


def test_different_benchmark_result_is_a_failure_not_a_speedup(config: RunConfig) -> None:
    result = _judge(make_box(speedup=2.0, fp_match=False), config).judge(
        "s", (PATCHES / "precompute.diff").read_text()
    )
    assert result.verdict == Verdict.FAILED
    assert "differs" in result.reason


def test_contaminated_pairs_retry_once_then_unmeasurable(config: RunConfig) -> None:
    box = make_box(contaminate_pairs=True)
    result = _judge(box, config).judge("s", (PATCHES / "precompute.diff").read_text())
    assert result.verdict == Verdict.UNMEASURABLE
    launches = [c for c in box.commands if "time_target.py" in c and "--repeats" in c]
    assert len(launches) == 2 * 6 * 2  # two passes of six pairs, two launches each
    assert all(p.contaminated for p in result.pairs)
    assert result.pairs[0].reasons == ("steal",)


def test_ir_failure_does_not_change_the_verdict(config: RunConfig) -> None:
    result = _judge(make_box(speedup=1.05, ir_ok=False), config).judge(
        "s", (PATCHES / "precompute.diff").read_text()
    )
    assert result.verdict == Verdict.ACCEPTED
    assert result.ir is None


def test_timing_launches_are_pinned_and_seeded(config: RunConfig) -> None:
    box = make_box()
    _judge(box, config).judge("s", (PATCHES / "precompute.diff").read_text())
    launches = [c for c in box.commands if "time_target.py" in c and "--repeats" in c]
    assert launches and all("taskset -c 2" in c for c in launches)
    first_pair = launches[:2]
    assert INCUMBENT_TREE in first_pair[0] and PATCHED_TREE in first_pair[1]
    second_pair = launches[2:4]
    assert PATCHED_TREE in second_pair[0] and INCUMBENT_TREE in second_pair[1]


def test_cleanup_runs_after_every_verdict_and_marks_broken_on_failure(config: RunConfig) -> None:
    box = make_box(cleanup_ok=False, module_ok=False)
    ref = _judge(box, config)
    ref.judge("s", (PATCHES / "precompute.diff").read_text())
    removes = [c for c in box.commands if "--remove" in c]
    assert len(removes) == 2
    assert "cleanup" in ref.broken


def test_box_error_propagates_to_the_orchestrator(config: RunConfig) -> None:
    box = make_box()
    ref = _judge(box, config)
    box.terminate()
    with pytest.raises(BoxError):
        ref.judge("s", (PATCHES / "precompute.diff").read_text())


def test_sync_incumbent_checks_the_tree_hash(config: RunConfig) -> None:
    box = make_box()
    box.on("git checkout -q --detach", ok("newhead\ntree123\n"), first=True)
    ref = _judge(box, config)
    assert ref.sync_incumbent("base", "diff --git a/x b/x\n", "tree123") == "newhead"
    assert box.files["/workspace/work/incumbent.diff"] == b"diff --git a/x b/x\n"
    with pytest.raises(BoxError, match="tree mismatch"):
        ref.sync_incumbent("base", "", "othertree")


def test_guest_command_quotes_and_pins() -> None:
    cmd = guest_command("time_target.py", "--graph", "nx.g(1, 0.5)", pin=2)
    assert (
        cmd == "cd /workspace/guest && taskset -c 2 python3 time_target.py --graph 'nx.g(1, 0.5)'"
    )
