from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mokopt.adapters import (
    load_swefficiency_instance,
    prepare_fc_eval,
    prepare_gso,
    prepare_swefficiency,
)


def _task_dir(tmp_path: Path) -> Path:
    task = tmp_path / "sample"
    task.mkdir()
    (task / "task.yaml").write_text(
        "instruction: |\n  Optimize this repository.\n"
    )
    (task / "Dockerfile").write_text("FROM scratch\n")
    return task


def test_gso_adapter_uses_prob_runner_contract(tmp_path: Path) -> None:
    task = prepare_gso(_task_dir(tmp_path))

    assert task.repo_path == "/testbed"
    assert "/perf/prob_runner.py" in task.benchmark_cmd
    assert "--eqcheck" in task.test_cmd
    assert "cat /tmp/mokopt_gso_eval.txt" in task.benchmark_cmd
    assert task.metric_regex.startswith("Execution time:")
    assert "/perf/install_cmds.sh" in task.prepare_cmd


def test_fc_eval_adapter_injects_setup_script(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path)
    (task_dir / "run-setup.sh").write_text("#!/bin/sh\necho setup\n")

    task = prepare_fc_eval(
        task_dir,
        benchmark_cmd="python benchmark.py",
        test_cmd="pytest -q",
    )

    assert task.repo_path == "/workspace/repo"
    assert task.task.instruction == "Optimize this repository.\n"
    assert "/tmp/mokopt-run-setup.sh" in task.injected_files
    assert task.build_context == task_dir


def test_swefficiency_adapter_preserves_workload_and_baseline_failures() -> None:
    task = prepare_swefficiency(
        {
            "instance_id": "owner__repo-1",
            "workload": "print('Mean: 1.25')",
            "test_cmd": "pytest -q",
            "rebuild_cmd": "pip install -e .",
            "covering_tests": ["tests/test_hot.py"],
            "problem_statement": "Make the hot path faster.",
        }
    )

    assert task.repo_path == "/testbed"
    assert task.metric_regex.startswith("Mean:")
    assert task.injected_files["/tmp/mokopt_workload.py"].startswith("print")
    assert "mokopt_baseline_failures" in task.setup_cmd
    assert "grep -Fxq" in task.test_cmd
    assert float(re.search(task.metric_regex, "Mean: 1.25e-05").group(1)) == 1.25e-5


def test_load_swefficiency_jsonl_selects_instance(tmp_path: Path) -> None:
    source = tmp_path / "instances.jsonl"
    source.write_text(
        json.dumps({"instance_id": "one", "workload": "print('Mean: 1')"}) + "\n"
    )

    row = load_swefficiency_instance(str(source), instance_id="one")

    assert row["instance_id"] == "one"


def test_swefficiency_requires_workload() -> None:
    with pytest.raises(ValueError, match="workload"):
        prepare_swefficiency({"instance_id": "broken"})
