"""MoKOpt command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mokopt.adapters import (
    PreparedTask,
    build_task_image,
    load_swefficiency_instance,
    prepare_fc_eval,
    prepare_gso,
    prepare_swefficiency,
)
from mokopt.backends import build_backend
from mokopt.executor import DockerExecutor, LocalExecutor
from mokopt.runner import MoKOptRunner
from mokopt.state import BetaConfig, Budget, TaskSpec


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=("mock", "copilot", "codex", "claude", "command"),
        default="copilot",
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--agent-command",
        help=(
            "Shell command template for the command backend. Available fields: "
            "{prompt_file}, {model}, {model_flag}, {workdir}."
        ),
    )
    parser.add_argument("--agent-setup-command")
    parser.add_argument("--agent-timeout", type=int, default=3600)
    parser.add_argument(
        "--mock-solution",
        action="append",
        default=[],
        help="Shell edit command consumed by one mock branch; repeatable.",
    )
    parser.add_argument("--output", type=Path, default=Path("mokopt-runs"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--branch-k", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--isolation", choices=("single", "phased"))
    parser.add_argument("--sig-threshold", type=float)
    parser.add_argument("--promotion-threshold", type=float)
    parser.add_argument("--stop-no-improvement", type=int)
    parser.add_argument("--wall-time", type=float, default=7200.0)
    parser.add_argument("--dynamic-routing", action="store_true")
    parser.add_argument("--use-bandit", action="store_true")
    parser.add_argument("--disable-skills", action="store_true")
    parser.add_argument("--inject-all-skills", action="store_true")
    parser.add_argument("--ralph-history", action="store_true")
    parser.add_argument("--no-readonly-guard", action="store_true")
    parser.add_argument("--env-check-cmd")
    parser.add_argument("--pull", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mokopt",
        description="Multi-branch LLM search for repository performance optimization.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generic = subparsers.add_parser("run", help="Optimize a generic repository.")
    generic.add_argument("--repo", type=Path, required=True)
    generic.add_argument("--task-id", default="local")
    generic.add_argument("--instruction", required=True)
    generic.add_argument("--benchmark-cmd", required=True)
    generic.add_argument("--test-cmd")
    generic.add_argument("--metric-regex")
    generic.add_argument("--prepare-cmd")
    generic.add_argument("--setup-cmd")
    generic.add_argument("--image")
    generic.add_argument("--repo-path", default="/workspace/repo")
    _common_parser(generic)

    fc_eval = subparsers.add_parser(
        "fc-eval", help="Optimize a generated FormulaCode/FC-Eval task."
    )
    fc_eval.add_argument("--task-dir", type=Path, required=True)
    fc_eval.add_argument("--benchmark-cmd", required=True)
    fc_eval.add_argument("--test-cmd")
    fc_eval.add_argument("--metric-regex")
    fc_eval.add_argument("--image")
    _common_parser(fc_eval)

    gso = subparsers.add_parser("gso", help="Optimize a generated GSO task.")
    gso.add_argument("--task-dir", type=Path, required=True)
    gso.add_argument("--image")
    _common_parser(gso)

    sweff = subparsers.add_parser(
        "swefficiency", help="Optimize one SWEfficiency instance."
    )
    sweff.add_argument("--dataset", required=True)
    sweff.add_argument("--instance-id", required=True)
    sweff.add_argument("--split", default="test")
    sweff.add_argument("--image")
    _common_parser(sweff)
    return parser


def _beta_from_args(args: argparse.Namespace) -> BetaConfig:
    beta = (
        BetaConfig.from_path(args.config)
        if args.config is not None
        else BetaConfig.from_path(Path(__file__).parent / "default_beta.json")
    )
    overrides = {
        "branch_k": args.branch_k,
        "max_rounds": args.rounds,
        "isolation": args.isolation,
        "sig_threshold": args.sig_threshold,
        "full_promotion_threshold": args.promotion_threshold,
        "stop_no_improvement": args.stop_no_improvement,
        "dynamic_routing": True if args.dynamic_routing else None,
        "use_bandit": True if args.use_bandit else None,
        "disable_skills": True if args.disable_skills else None,
        "inject_all_skills": True if args.inject_all_skills else None,
        "ralph_history": True if args.ralph_history else None,
        "readonly_guard": False if args.no_readonly_guard else None,
        "env_check_cmd": args.env_check_cmd,
    }
    return beta.overlay(overrides)


def _prepared_from_args(args: argparse.Namespace) -> tuple[PreparedTask, Path | None]:
    if args.command == "run":
        task = PreparedTask(
            task=TaskSpec(
                task_id=args.task_id,
                repo_path=args.repo_path,
                instruction=args.instruction,
            ),
            image=args.image or "",
            repo_path=args.repo_path,
            benchmark_cmd=args.benchmark_cmd,
            test_cmd=args.test_cmd,
            metric_regex=args.metric_regex,
            prepare_cmd=args.prepare_cmd,
            setup_cmd=args.setup_cmd,
        )
        return task, args.repo.resolve()
    if args.command == "fc-eval":
        return (
            prepare_fc_eval(
                args.task_dir,
                benchmark_cmd=args.benchmark_cmd,
                test_cmd=args.test_cmd,
                metric_regex=args.metric_regex,
                image=args.image,
            ),
            None,
        )
    if args.command == "gso":
        return prepare_gso(args.task_dir, image=args.image), None
    instance = load_swefficiency_instance(
        args.dataset,
        instance_id=args.instance_id,
        split=args.split,
    )
    return prepare_swefficiency(instance, image=args.image), None


def _backend_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.backend == "mock":
        return {"solutions": args.mock_solution}
    kwargs: dict[str, Any] = {"timeout_s": args.agent_timeout}
    if args.agent_command:
        kwargs["command"] = args.agent_command
    if args.agent_setup_command:
        kwargs["setup_command"] = args.agent_setup_command
    return kwargs


def _container_environment(task: PreparedTask) -> dict[str, str]:
    environment = dict(task.environment)
    for key in (
        "COPILOT_GITHUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "CODEX_WIRE_API",
    ):
        if value := os.environ.get(key):
            if key in {"ANTHROPIC_BASE_URL", "OPENAI_BASE_URL"}:
                value = re.sub(
                    r"(https?://)(localhost|127\.0\.0\.1|0\.0\.0\.0)(:|\b)",
                    r"\1host.docker.internal\3",
                    value,
                )
            environment[key] = value
    environment.setdefault("IS_SANDBOX", "1")
    return environment


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    beta = _beta_from_args(args)
    task, repo_source = _prepared_from_args(args)
    backend = build_backend(
        args.backend,
        model=args.model,
        **_backend_kwargs(args),
    )
    output_dir = args.output / task.task.task_id

    executor = None
    local_workspace: Path | None = None
    try:
        if args.command == "run" and not task.image:
            if repo_source is None:
                raise RuntimeError("generic local execution requires --repo")
            local_workspace = Path(
                tempfile.mkdtemp(prefix=f"mokopt-{task.task.task_id}-")
            )
            workspace = local_workspace / "repo"
            shutil.copytree(repo_source, workspace, symlinks=True)
            executor = LocalExecutor(workspace)
            task.task.repo_path = executor.repo_path
            task.repo_path = executor.repo_path
        else:
            image = build_task_image(task, pull=args.pull)
            executor = DockerExecutor.create(
                image=image,
                repo_host_path=repo_source,
                repo_path=task.repo_path,
                env=_container_environment(task),
                pull=args.pull and task.build_context is None,
            )
        budget = Budget(
            rounds=beta.max_rounds,
            fast_evals=max(1, beta.branch_k * beta.max_rounds),
            full_evals=max(1, beta.full_top_n * beta.max_rounds),
            profile_runs=1,
            wall_time_s=args.wall_time,
        )
        result = MoKOptRunner(
            executor=executor,
            backend=backend,
            task=task.task,
            benchmark_cmd=task.benchmark_cmd,
            test_cmd=task.test_cmd,
            metric_regex=task.metric_regex,
            prepare_cmd=task.prepare_cmd,
            setup_cmd=task.setup_cmd,
            injected_files=task.injected_files,
            output_dir=output_dir,
            model=args.model,
            beta=beta,
            budget=budget,
        ).run()
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"mokopt: error: {exc}", file=sys.stderr)
        return 1
    finally:
        if executor is not None:
            executor.close()
        if local_workspace is not None:
            shutil.rmtree(local_workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
