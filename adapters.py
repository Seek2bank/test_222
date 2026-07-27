"""Dataset adapters for FC-Eval, GSO, and SWEfficiency."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mokopt.state import TaskSpec


@dataclass
class PreparedTask:
    task: TaskSpec
    image: str
    repo_path: str
    benchmark_cmd: str
    test_cmd: str | None
    metric_regex: str | None = None
    prepare_cmd: str | None = None
    setup_cmd: str | None = None
    injected_files: dict[str, str] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    build_context: Path | None = None


def _load_task_instruction(task_dir: Path) -> str:
    task_yaml = task_dir / "task.yaml"
    if not task_yaml.is_file():
        raise FileNotFoundError(f"task file does not exist: {task_yaml}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read FC-Eval tasks") from exc
    data = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
    instruction = data.get("instruction") if isinstance(data, dict) else None
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{task_yaml} has no non-empty instruction")
    return instruction


def prepare_fc_eval(
    task_dir: Path,
    *,
    benchmark_cmd: str,
    test_cmd: str | None,
    metric_regex: str | None = None,
    image: str | None = None,
) -> PreparedTask:
    """Prepare a FormulaCode/FC-Eval task.

    FormulaCode benchmark suites differ across repositories, so callers provide
    the scalar benchmark command. It may either be timed by MoKOpt or print a
    lower-is-better number selected by ``metric_regex``.
    """
    task_dir = task_dir.resolve()
    setup_script = task_dir / "run-setup.sh"
    injected: dict[str, str] = {}
    setup_cmd = None
    if setup_script.is_file():
        path = "/tmp/mokopt-run-setup.sh"
        injected[path] = setup_script.read_text(encoding="utf-8")
        setup_cmd = f"chmod +x {path} && {path}"
    return PreparedTask(
        task=TaskSpec(
            task_id=task_dir.name,
            repo_path="/workspace/repo",
            instruction=_load_task_instruction(task_dir),
        ),
        image=image or f"mokopt-fceval-{task_dir.name}:latest",
        repo_path="/workspace/repo",
        benchmark_cmd=benchmark_cmd,
        test_cmd=test_cmd,
        metric_regex=metric_regex,
        setup_cmd=setup_cmd,
        injected_files=injected,
        build_context=None if image else task_dir,
    )


def prepare_gso(
    task_dir: Path,
    *,
    image: str | None = None,
) -> PreparedTask:
    task_dir = task_dir.resolve()
    activate = (
        "set +u; source /testbed/.venv/bin/activate 2>/dev/null || true; "
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0; set -u; "
    )
    reference = (
        "cd /tmp && rm -f mokopt_gso_result.json; "
        "python /perf/prob_runner.py /tmp/mokopt_gso_ref.txt "
        "--reference --file_prefix mokopt_gso"
    )
    eqcheck = (
        "cd /tmp && rm -f /tmp/mokopt_gso_eval.txt && "
        "python /perf/prob_runner.py /tmp/mokopt_gso_eval.txt "
        "--eqcheck --file_prefix mokopt_gso && "
        "cat /tmp/mokopt_gso_eval.txt"
    )
    correctness = (
        "cd /tmp && python /perf/prob_runner.py /tmp/mokopt_gso_check.txt "
        "--eqcheck --file_prefix mokopt_gso"
    )
    reinstall = (
        f"{activate} if [ -x /perf/install_cmds.sh ]; then "
        "bash /perf/install_cmds.sh; fi"
    )
    return PreparedTask(
        task=TaskSpec(
            task_id=task_dir.name,
            repo_path="/testbed",
            instruction=_load_task_instruction(task_dir),
        ),
        image=image or f"mokopt-gso-{task_dir.name}:latest",
        repo_path="/testbed",
        benchmark_cmd=activate + eqcheck,
        test_cmd=activate + correctness,
        metric_regex=r"Execution time:\s*([0-9]+(?:\.[0-9]+)?)s",
        prepare_cmd=reinstall,
        setup_cmd=activate + reinstall + "; " + reference,
        environment={
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "TZ": "UTC",
        },
        build_context=None if image else task_dir,
    )


def load_swefficiency_instance(
    source: str,
    *,
    instance_id: str,
    split: str = "test",
) -> dict[str, Any]:
    path = Path(source)
    rows: list[dict[str, Any]]
    if path.is_file() and path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.is_file() and path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else [payload]
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "loading Hugging Face or parquet SWEfficiency data requires "
                "the 'datasets' optional dependency"
            ) from exc
        if path.is_dir():
            files = sorted(str(item) for item in path.rglob("*.parquet"))
            dataset = load_dataset("parquet", data_files=files, split="train")
        else:
            dataset = load_dataset(source, split=split)
        rows = [dict(row) for row in dataset]
    for row in rows:
        if row.get("instance_id") == instance_id:
            return row
    raise KeyError(f"instance {instance_id!r} was not found in {source!r}")


def prepare_swefficiency(
    instance: dict[str, Any],
    *,
    image: str | None = None,
) -> PreparedTask:
    instance_id = str(instance["instance_id"])
    workload = instance.get("workload")
    if not isinstance(workload, str) or not workload.strip():
        raise ValueError("SWEfficiency instance has no workload script")
    covering_tests = instance.get("covering_tests") or []
    if isinstance(covering_tests, str):
        covering_tests = [covering_tests]
    covering_text = "\n".join(str(item) for item in covering_tests) + "\n"
    test_prefix = str(instance.get("test_cmd") or "python -m pytest -q")
    rebuild = str(
        instance.get("rebuild_cmd") or "pip install --no-build-isolation -e ."
    )
    test_file = "/tmp/mokopt_covering_tests.txt"
    failures = "/tmp/mokopt_baseline_failures.txt"
    baseline_tests = (
        f": > {failures}; while IFS= read -r t; do "
        '[ -z "$t" ] && continue; '
        f'{test_prefix} "$t" >/tmp/mokopt-test.log 2>&1 || '
        f'echo "$t" >> {failures}; done < {test_file}'
    )
    candidate_tests = (
        "status=0; while IFS= read -r t; do "
        '[ -z "$t" ] && continue; '
        f'if ! {test_prefix} "$t" >/tmp/mokopt-test.log 2>&1; then '
        f'grep -Fxq "$t" {failures} || status=1; fi; '
        f"done < {test_file}; exit $status"
    )
    instruction = str(
        instance.get("problem_statement")
        or instance.get("instruction")
        or "Optimize the supplied workload without changing its behavior."
    )
    return PreparedTask(
        task=TaskSpec(
            task_id=instance_id,
            repo_path="/testbed",
            instruction=instruction,
        ),
        image=image or f"ghcr.io/swefficiency/swefficiency-images:{instance_id}",
        repo_path="/testbed",
        benchmark_cmd="python /tmp/mokopt_workload.py",
        test_cmd=candidate_tests if covering_tests else None,
        metric_regex=(
            r"Mean:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
        ),
        prepare_cmd=rebuild,
        setup_cmd=baseline_tests if covering_tests else None,
        injected_files={
            "/tmp/mokopt_workload.py": workload,
            test_file: covering_text,
        },
    )


def build_task_image(task: PreparedTask, *, pull: bool = False) -> str:
    if task.build_context is None:
        return task.image
    try:
        import docker
    except ImportError as exc:
        raise RuntimeError("Docker support requires the 'docker' dependency") from exc
    client = docker.from_env()
    image, _ = client.images.build(
        path=str(task.build_context),
        tag=task.image,
        pull=pull,
        rm=True,
    )
    return image.tags[0] if image.tags else task.image


def shell_join(values: list[str]) -> str:
    """Return a safely quoted command fragment, exposed for adapter tests."""
    return " ".join(shlex.quote(value) for value in values)
