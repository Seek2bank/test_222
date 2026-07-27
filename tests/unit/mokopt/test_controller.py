from __future__ import annotations

from pathlib import Path

from mokopt.backends.mock_backend import MockBackend
from mokopt.executor.local_executor import LocalExecutor
from mokopt.runner import MoKOptRunner
from mokopt.state import BetaConfig, Budget, TaskSpec


def test_mock_run_promotes_faster_correct_patch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 10\n")
    output = tmp_path / "output"

    result = MoKOptRunner(
        executor=LocalExecutor(repo),
        backend=MockBackend(
            solutions=["sed -i 's/VALUE = 10/VALUE = 5/' app.py"]
        ),
        task=TaskSpec(
            task_id="example",
            repo_path=str(repo),
            instruction="Reduce VALUE while keeping it positive.",
        ),
        benchmark_cmd=(
            "python -c 'from app import VALUE; print(f\"Metric: {VALUE}\")'"
        ),
        metric_regex=r"Metric: ([0-9.]+)",
        test_cmd="python -c 'from app import VALUE; assert VALUE > 0'",
        output_dir=output,
        beta=BetaConfig(
            branch_k=1,
            max_rounds=1,
            readonly_guard=False,
            sig_threshold=1.01,
            full_promotion_threshold=1.01,
        ),
        budget=Budget(
            rounds=1,
            fast_evals=1,
            full_evals=1,
            profile_runs=1,
        ),
    ).run()

    assert result.speedup == 2.0
    assert result.tests_pass
    assert result.best_branch == "r0-b0"
    assert "-VALUE = 10" in Path(result.patch_path).read_text()
    assert "+VALUE = 5" in Path(result.patch_path).read_text()


def test_local_output_cannot_be_inside_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 1\n")

    runner = MoKOptRunner(
        executor=LocalExecutor(repo),
        backend=MockBackend(),
        task=TaskSpec(task_id="example", repo_path=str(repo)),
        benchmark_cmd="echo 'Metric: 1'",
        metric_regex=r"Metric: ([0-9.]+)",
        output_dir=repo / "output",
        beta=BetaConfig(readonly_guard=False),
    )

    try:
        runner.run()
    except ValueError as exc:
        assert "output must be outside" in str(exc)
    else:
        raise AssertionError("expected an unsafe output-path error")

